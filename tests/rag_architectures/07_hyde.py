#!/usr/bin/env python3
"""
测试: HyDE (07_hyde.py)
========================

假设文档Embedding，先生成假设文档再用它检索真实文档。

运行: python tests/rag_architectures/07_hyde.py

架构:
    ┌──────────┐    ┌──────────────┐    ┌─────────────┐    ┌────────────┐
    │   查询    │───▶│   生成       │───▶│   检索      │───▶│   生成    │
    │          │    │ 假设文档     │    │  使用HyDE   │    │           │
    │          │    │              │    │   文档      │    │           │
    └──────────┘    └──────────────┘    └─────────────┘    └────────────┘
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
# 测试文档
# ============================================

TEST_DOCUMENTS = [
    "RAG结合检索与生成，检索器从知识库中查找相关文档。",
    "Self-RAG是自反思框架，LLM决定何时检索。",
    "Agentic RAG使用AI智能体规划检索策略。",
    "GraphRAG使用知识图谱进行实体关系检索。",
    "HyDE生成假设回答文档来改进检索质量。",
    "密集检索使用神经网络embedding进行语义相似性搜索。",
    "BM25是基于关键词的检索方法，适合精确匹配。",
    "融合RAG结合多种检索方法并重排序融合结果。",
    "Transformer模型使用注意力机制处理序列。",
    "向量数据库存储embeddings支持高效相似性搜索。",
]


# ============================================
# HyDE实现
# ============================================


class HyDE:
    """HyDE(假设文档Embedding)实现"""

    def __init__(self, vectorstore, llm):
        self.vectorstore = vectorstore
        self.llm = llm
        self.hypothetical_doc = None

    def _generate_hypothetical_doc(self, query: str) -> str:
        """生成假设回答文档"""

        prompt = f"""生成一个假设的文档来回答这个问题:

问题: {query}

请像一个专家一样写出详细的假设回答:"""

        response = self.llm.invoke(prompt)
        self.hypothetical_doc = response.content

        return self.hypothetical_doc

    def query(self, query: str) -> dict:
        """执行HyDE流程"""

        # 步骤1: 生成假设文档
        hypo_doc = self._generate_hypothetical_doc(query)

        # 步骤2: 使用假设文档检索
        retriever = self.vectorstore.as_retriever(search_kwargs={"k": 3})

        # 用假设文档内容指导检索
        combined_query = f"{query} {hypo_doc[:200]}"
        retrieved_docs = retriever.invoke(combined_query)

        # 步骤3: 用真实上下文生成最终答案
        context = "\n\n".join([doc.page_content for doc in retrieved_docs])

        prompt = f"""根据检索到的上下文，提供准确的答案:

上下文:
{context}

问题: {query}

答案:"""

        response = self.llm.invoke(prompt)

        return {
            "answer": response.content,
            "hypothetical_doc": hypo_doc[:300],
            "retrieved_docs": [doc.page_content for doc in retrieved_docs],
            "num_docs": len(retrieved_docs),
        }


# ============================================
# 测试函数
# ============================================


def test_01_hyde_basic(config: TestConfig):
    """测试: 基本HyDE"""
    print("\n[测试 01] 基本HyDE")

    vectorstore = create_vector_store(TEST_DOCUMENTS, config)
    if vectorstore is None:
        return {"passed": False, "message": "创建向量存储失败"}

    llm = create_llm(config, temperature=0.7)  # 较高温度生成创意假设
    if llm is None:
        return {"passed": False, "message": "创建LLM失败"}

    hyde = HyDE(vectorstore, llm)
    result = hyde.query("什么是HyDE?")

    if not result.get("hypothetical_doc"):
        return {"passed": False, "message": "未生成假设文档"}

    return {
        "passed": True,
        "message": f"生成了假设文档，检索了 {result.get('num_docs', 0)} 个真实文档",
        "details": {
            "hypo_doc_length": len(result.get("hypothetical_doc", "")),
            "num_retrieved": result.get("num_docs", 0),
            "answer_preview": result.get("answer", "")[:100],
        },
    }


def test_02_hypothetical_doc_quality(config: TestConfig):
    """测试: 假设文档质量"""
    print("\n[测试 02] 假设文档质量")

    vectorstore = create_vector_store(TEST_DOCUMENTS, config)
    if vectorstore is None:
        return {"passed": False, "message": "创建向量存储失败"}

    llm = create_llm(config, temperature=0.7)
    if llm is None:
        return {"passed": False, "message": "创建LLM失败"}

    hyde = HyDE(vectorstore, llm)
    result = hyde.query("RAG是如何工作的?")

    hypo_len = len(result.get("hypothetical_doc", ""))

    return {
        "passed": hypo_len > 50,
        "message": f"假设文档长度: {hypo_len} 字符",
        "details": {"hypo_doc_preview": result.get("hypothetical_doc", "")[:150]},
    }


def test_03_retrieval_improvement(config: TestConfig):
    """测试: 检索改进"""
    print("\n[测试 03] 检索改进")

    vectorstore = create_vector_store(TEST_DOCUMENTS, config)
    if vectorstore is None:
        return {"passed": False, "message": "创建向量存储失败"}

    llm = create_llm(config, temperature=0.7)
    if llm is None:
        return {"passed": False, "message": "创建LLM失败"}

    hyde = HyDE(vectorstore, llm)
    result = hyde.query("比较不同的RAG架构")

    has_docs = result.get("num_docs", 0) > 0

    return {
        "passed": has_docs,
        "message": f"使用HyDE检索了 {result.get('num_docs', 0)} 个文档",
        "details": {"num_docs": result.get("num_docs", 0)},
    }


# ============================================
# 主函数
# ============================================


def main():
    print("=" * 60)
    print("HyDE 架构测试")
    print("=" * 60)

    config = TestConfig()

    if not config.is_configured:
        print("\n⚠️  警告: API未配置!")
        return

    print(f"\n[配置] LLM: {config.llm_provider}/{config.openai_model}")

    results = []

    results.append(run_test(lambda: test_01_hyde_basic(config), "基本HyDE"))
    results.append(
        run_test(lambda: test_02_hypothetical_doc_quality(config), "假设文档质量")
    )
    results.append(run_test(lambda: test_03_retrieval_improvement(config), "检索改进"))

    print_test_results(results)

    failed = sum(1 for r in results if not r.passed)
    return 0 if failed == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
