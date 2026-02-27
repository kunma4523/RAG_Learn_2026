#!/usr/bin/env python3
"""
测试: 图RAG (09_graph_rag.py)
==============================

使用知识图谱进行实体关系检索的RAG系统。

运行: python tests/rag_architectures/09_graph_rag.py
"""

import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

from tests.rag_architectures import (
    TestConfig,
    run_test,
    print_test_results,
    create_vector_store,
    create_llm,
)


# ============================================
# 知识图谱数据
# ============================================

KNOWLEDGE_GRAPH = {
    "entities": [
        {"id": "rag", "name": "RAG", "type": "概念", "description": "检索增强生成"},
        {
            "id": "self_rag",
            "name": "Self-RAG",
            "type": "架构",
            "description": "自反思RAG框架",
        },
        {
            "id": "agentic_rag",
            "name": "Agentic RAG",
            "type": "架构",
            "description": "智能体驱动的RAG",
        },
        {
            "id": "graphrag",
            "name": "GraphRAG",
            "type": "架构",
            "description": "基于图谱的RAG",
        },
        {
            "id": "hyde",
            "name": "HyDE",
            "type": "技术",
            "description": "假设文档Embedding",
        },
        {
            "id": "bm25",
            "name": "BM25",
            "type": "方法",
            "description": "基于关键词的检索",
        },
        {
            "id": "dense_retrieval",
            "name": "密集检索",
            "type": "方法",
            "description": "基于embedding的检索",
        },
        {
            "id": "transformer",
            "name": "Transformer",
            "type": "架构",
            "description": "神经网络架构",
        },
        {"id": "llm", "name": "LLM", "type": "概念", "description": "大语言模型"},
    ],
    "relationships": [
        {"source": "rag", "target": "self_rag", "type": "has_architecture"},
        {"source": "rag", "target": "agentic_rag", "type": "has_architecture"},
        {"source": "rag", "target": "graphrag", "type": "has_architecture"},
        {"source": "rag", "target": "hyde", "type": "uses_technique"},
        {"source": "rag", "target": "transformer", "type": "based_on"},
        {"source": "rag", "target": "llm", "type": "enhances"},
        {"source": "bm25", "target": "rag", "type": "used_by"},
        {"source": "dense_retrieval", "target": "rag", "type": "used_by"},
        {"source": "self_rag", "target": "rag", "type": "is_type_of"},
        {"source": "agentic_rag", "target": "rag", "type": "is_type_of"},
        {"source": "graphrag", "target": "rag", "type": "is_type_of"},
    ],
}


# ============================================
# 测试文档
# ============================================

TEST_DOCUMENTS = [
    "RAG结合检索与生成，减少LLM的幻觉。",
    "Self-RAG是自反思框架，LLM决定何时检索。",
    "Agentic RAG使用AI智能体规划检索策略。",
    "GraphRAG使用知识图谱进行实体关系检索。",
    "HyDE生成假设文档改进检索。",
    "BM25是基于关键词的稀疏检索方法。",
    "密集检索使用神经网络embedding进行语义搜索。",
    "Transformer注意力机制让模型关注相关上下文。",
]


# ============================================
# 图RAG实现
# ============================================


class GraphRAG:
    """使用知识图谱的RAG"""

    def __init__(self, vectorstore, llm, knowledge_graph: dict):
        self.vectorstore = vectorstore
        self.llm = llm
        self.kg = knowledge_graph
        self.entity_cache = {}

    def _extract_entities(self, query: str) -> list:
        """从查询中提取相关实体"""

        entity_names = [e["name"] for e in self.kg["entities"]]

        prompt = f"""查询: "{query}"

可用实体: {", ".join(entity_names)}

哪些实体与这个查询相关?
请回复实体名称，用逗号分隔:"""

        response = self.llm.invoke(prompt)

        # 匹配已知实体
        found = []
        response_lower = response.content.lower()

        for entity in self.kg["entities"]:
            if entity["name"].lower() in response_lower:
                found.append(entity["id"])

        return found if found else []

    def _get_graph_context(self, entity_ids: list) -> str:
        """从知识图谱获取上下文"""

        context_parts = []

        for eid in entity_ids:
            entity = next((e for e in self.kg["entities"] if e["id"] == eid), None)
            if entity:
                context_parts.append(f"实体: {entity['name']} ({entity['type']})")
                context_parts.append(f"  描述: {entity['description']}")

                # 查找关系
                for rel in self.kg["relationships"]:
                    if rel["source"] == eid:
                        target = next(
                            (
                                e
                                for e in self.kg["entities"]
                                if e["id"] == rel["target"]
                            ),
                            None,
                        )
                        if target:
                            context_parts.append(
                                f"  -> {rel['type']} -> {target['name']}"
                            )

        return "\n".join(context_parts)

    def query(self, query: str) -> dict:
        """执行图RAG"""

        # 从查询中提取实体
        entity_ids = self._extract_entities(query)

        # 获取图谱上下文
        graph_context = self._get_graph_context(entity_ids)

        # 同时从向量存储检索
        retriever = self.vectorstore.as_retriever(search_kwargs={"k": 3})
        vector_docs = retriever.invoke(query)

        # 组合上下文
        combined_context = f"""知识图谱上下文:
{graph_context}

文档上下文:
{" ".join([doc.page_content for doc in vector_docs])}"""

        # 生成答案
        prompt = f"""上下文:
{combined_context}

问题: {query}

请综合知识图谱和文档上下文给出全面答案:"""

        response = self.llm.invoke(prompt)

        return {
            "answer": response.content,
            "entities_found": entity_ids,
            "graph_context": graph_context[:300] if graph_context else "",
            "num_vector_docs": len(vector_docs),
        }


# ============================================
# 测试函数
# ============================================


def test_01_graph_rag_basic(config: TestConfig):
    """测试: 基本图RAG"""
    print("\n[测试 01] 基本图RAG")

    vectorstore = create_vector_store(TEST_DOCUMENTS, config)
    if vectorstore is None:
        return {"passed": False, "message": "创建向量存储失败"}

    llm = create_llm(config, temperature=0)
    if llm is None:
        return {"passed": False, "message": "创建LLM失败"}

    graph_rag = GraphRAG(vectorstore, llm, KNOWLEDGE_GRAPH)
    result = graph_rag.query("什么是RAG?")

    return {
        "passed": "answer" in result,
        "message": f"图RAG找到 {len(result.get('entities_found', []))} 个实体",
        "details": {
            "entities_found": result.get("entities_found", []),
            "num_vector_docs": result.get("num_vector_docs", 0),
        },
    }


def test_02_entity_extraction(config: TestConfig):
    """测试: 实体提取"""
    print("\n[测试 02] 实体提取")

    vectorstore = create_vector_store(TEST_DOCUMENTS, config)
    if vectorstore is None:
        return {"passed": False, "message": "创建向量存储失败"}

    llm = create_llm(config, temperature=0)
    if llm is None:
        return {"passed": False, "message": "创建LLM失败"}

    graph_rag = GraphRAG(vectorstore, llm, KNOWLEDGE_GRAPH)
    result = graph_rag.query("Self-RAG和RAG是什么关系?")

    has_entities = len(result.get("entities_found", [])) > 0

    return {
        "passed": has_entities,
        "message": f"提取的实体: {result.get('entities_found', [])}",
        "details": {"entities": result.get("entities_found", [])},
    }


def test_03_graph_traversal(config: TestConfig):
    """测试: 图关系遍历"""
    print("\n[测试 03] 图关系遍历")

    vectorstore = create_vector_store(TEST_DOCUMENTS, config)
    if vectorstore is None:
        return {"passed": False, "message": "创建向量存储失败"}

    llm = create_llm(config, temperature=0)
    if llm is None:
        return {"passed": False, "message": "创建LLM失败"}

    graph_rag = GraphRAG(vectorstore, llm, KNOWLEDGE_GRAPH)
    result = graph_rag.query("有哪些RAG架构?")

    has_graph_context = bool(result.get("graph_context"))

    return {
        "passed": has_graph_context,
        "message": f"图谱上下文已获取: {has_graph_context}",
        "details": {"graph_context_length": len(result.get("graph_context", ""))},
    }


# ============================================
# 主函数
# ============================================


def main():
    print("=" * 60)
    print("图RAG架构测试")
    print("=" * 60)

    config = TestConfig()

    if not config.is_configured:
        print("\n⚠️  警告: API未配置!")
        return

    print(f"\n[配置] LLM: {config.llm_provider}/{config.openai_model}")

    results = []

    results.append(run_test(lambda: test_01_graph_rag_basic(config), "基本图RAG"))
    results.append(run_test(lambda: test_02_entity_extraction(config), "实体提取"))
    results.append(run_test(lambda: test_03_graph_traversal(config), "图遍历"))

    print_test_results(results)

    failed = sum(1 for r in results if not r.passed)
    return 0 if failed == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
