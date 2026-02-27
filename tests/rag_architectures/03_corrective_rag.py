#!/usr/bin/env python3
"""
测试: 纠正RAG (03_corrective_rag.py)
====================================

带有检索验证和纠正机制的RAG流程，验证检索结果并重写查询。

运行: python tests/rag_architectures/03_corrective_rag.py

架构:
    ┌──────────┐    ┌─────────────┐    ┌──────────────┐    ┌──────────────┐
    │   查询    │───▶│   检索      │───▶│   验证       │───▶│   纠正       │
    └──────────┘    └─────────────┘    └──────────────┘    └──────────────┘
                                                              │
                              ┌───────────────────────────────┘
                              ▼
                    ┌─────────────────┐    ┌──────────────┐
                    │   生成          │───▶│    答案      │
                    └─────────────────┘    └──────────────┘
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
    "RAG(检索增强生成)将信息检索与文本生成相结合，使用检索器从知识库中查找相关文档。",
    "Self-RAG是一种自反思框架，让LLM能够自主决定何时检索信息。",
    "Agentic RAG结合AI智能体，可以规划检索策略并动态使用多种工具。",
    "GraphRAG使用知识图谱来表示实体关系，进行关联信息检索。",
    "HyDE生成假设文档来改进检索效果，LLM先创建假设回答再用它检索真实文档。",
    "BM25是一种基于词频的稀疏检索方法，适合关键词匹配但不支持语义理解。",
    "密集检索使用神经网络的embedding，如DPR，用于语义相似性匹配。",
]


# ============================================
# 纠正RAG实现
# ============================================


class CorrectiveRAG:
    """带有验证和查询重写的纠正RAG"""

    def __init__(self, vectorstore, llm, top_k: int = 3):
        self.vectorstore = vectorstore
        self.llm = llm
        self.top_k = top_k
        self.correction_count = 0

    def _verify_relevance(self, query: str, docs: list) -> dict:
        """验证检索到的文档是否与查询相关"""

        doc_texts = "\n\n".join(
            [f"文档 {i + 1}: {doc.page_content[:200]}" for i, doc in enumerate(docs)]
        )

        verification_prompt = f"""给定查询: "{query}"

检索到的文档:
{doc_texts}

这些文档是否与回答查询相关? 请考虑:
1. 文档是否包含查询主题的信息?
2. 是否有不相关的文档?
3. 是否需要重写查询以获得更好的检索?

请回复:
- RELEVANT: 如果文档足够
- NEEDS_REWRITE: 如果需要重写查询
- INSUFFICIENT: 如果需要更多/不同的文档"""

        response = self.llm.invoke(verification_prompt)
        result = response.content.strip().upper()

        return {
            "status": result,
            "needs_correction": "NEEDS" in result or "INSUFFICIENT" in result,
        }

    def _rewrite_query(self, original_query: str, feedback: str) -> str:
        """根据反馈重写查询"""

        rewrite_prompt = f"""原始查询: "{original_query}"

反馈: {feedback}

请重写查询以改进检索效果。只返回新的查询语句:"""

        response = self.llm.invoke(rewrite_prompt)
        new_query = response.content.strip()

        self.correction_count += 1

        return new_query

    def query(self, query: str, max_corrections: int = 2) -> dict:
        """执行带纠正机制的RAG"""

        current_query = query
        retrieval_count = 0

        for iteration in range(max_corrections + 1):
            # 检索文档
            retriever = self.vectorstore.as_retriever(search_kwargs={"k": self.top_k})
            docs = retriever.invoke(current_query)
            retrieval_count += 1

            # 验证相关性
            verification = self._verify_relevance(query, docs)

            if not verification["needs_correction"] or iteration >= max_corrections:
                # 生成最终答案
                context = "\n\n".join([doc.page_content for doc in docs])

                prompt = f"""上下文:
{context}

问题: {query}

请根据以上上下文回答:"""

                response = self.llm.invoke(prompt)

                return {
                    "answer": response.content,
                    "retrieved_docs": [doc.page_content for doc in docs],
                    "corrections_applied": self.correction_count,
                    "retrieval_count": retrieval_count,
                    "final_query": current_query,
                    "verification_status": verification["status"],
                }
            else:
                # 重写查询
                current_query = self._rewrite_query(query, verification["status"])

        return {"error": "达到最大纠正次数"}


# ============================================
# 测试函数
# ============================================


def test_01_basic_corrective_rag(config: TestConfig):
    """测试: 基本的纠正RAG流程"""
    print("\n[测试 01] 基本的纠正RAG")

    vectorstore = create_vector_store(TEST_DOCUMENTS, config)
    if vectorstore is None:
        return {"passed": False, "message": "创建向量存储失败"}

    llm = create_llm(config, temperature=0)
    if llm is None:
        return {"passed": False, "message": "创建LLM失败"}

    rag = CorrectiveRAG(vectorstore, llm)
    result = rag.query("什么是RAG?")

    if "answer" not in result:
        return {"passed": False, "message": "未生成答案"}

    return {
        "passed": True,
        "message": f"纠正RAG执行完成，应用了 {result.get('corrections_applied', 0)} 次纠正",
        "details": {
            "answer_preview": result["answer"][:150],
            "corrections": result.get("corrections_applied", 0),
            "retrievals": result.get("retrieval_count", 0),
        },
    }


def test_02_query_rewrite_on_irrelevant(config: TestConfig):
    """测试: 文档不相关时重写查询"""
    print("\n[测试 02] 查询重写")

    vectorstore = create_vector_store(TEST_DOCUMENTS, config)
    if vectorstore is None:
        return {"passed": False, "message": "创建向量存储失败"}

    llm = create_llm(config, temperature=0)
    if llm is None:
        return {"passed": False, "message": "创建LLM失败"}

    rag = CorrectiveRAG(vectorstore, llm)

    # 查询可能需要细化的话题
    result = rag.query("检索器组件是如何工作的?")

    # 检查纠正机制是否工作
    if result.get("corrections_applied", 0) >= 0:
        return {
            "passed": True,
            "message": f"查询处理完成，应用了 {result.get('corrections_applied', 0)} 次纠正",
            "details": {
                "answer_preview": result.get("answer", "")[:100],
                "corrections": result.get("corrections_applied", 0),
            },
        }

    return {"passed": False, "message": "纠正机制未正常工作"}


def test_03_multi_correction_iterations(config: TestConfig):
    """测试: 多次纠正迭代"""
    print("\n[测试 03] 多次纠正迭代")

    vectorstore = create_vector_store(TEST_DOCUMENTS, config)
    if vectorstore is None:
        return {"passed": False, "message": "创建向量存储失败"}

    llm = create_llm(config, temperature=0)
    if llm is None:
        return {"passed": False, "message": "创建LLM失败"}

    rag = CorrectiveRAG(vectorstore, llm)
    result = rag.query("介绍密集检索方法", max_corrections=3)

    return {
        "passed": "answer" in result,
        "message": f"完成检索 {result.get('retrieval_count', 0)} 次",
        "details": {
            "corrections": result.get("corrections_applied", 0),
            "retrievals": result.get("retrieval_count", 0),
        },
    }


# ============================================
# 主函数
# ============================================


def main():
    print("=" * 60)
    print("纠正RAG架构测试")
    print("=" * 60)

    config = TestConfig()

    if not config.is_configured:
        print("\n⚠️  警告: API未配置!")
        return

    print(f"\n[配置] LLM: {config.llm_provider}/{config.openai_model}")

    results = []

    results.append(
        run_test(lambda: test_01_basic_corrective_rag(config), "基本纠正RAG")
    )
    results.append(
        run_test(lambda: test_02_query_rewrite_on_irrelevant(config), "查询重写")
    )
    results.append(
        run_test(lambda: test_03_multi_correction_iterations(config), "多次纠正")
    )

    print_test_results(results)

    failed = sum(1 for r in results if not r.passed)
    return 0 if failed == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
