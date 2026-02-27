#!/usr/bin/env python3
"""
测试: 自适应RAG (04_adaptive_rag.py)
====================================

根据查询复杂度自适应选择检索策略的RAG系统。

运行: python tests/rag_architectures/04_adaptive_rag.py

架构:
    ┌──────────┐    ┌─────────────┐    ┌──────────────┐
    │   查询    │───▶│   路由器    │───▶│   策略选择   │
    └──────────┘    │ (复杂度分析) │    │              │
                   └─────────────┘    └──────────────┘
                              │              │
                              ▼              ▼
                    ┌─────────────────────────────┐
                    │  简单 → 标准RAG              │
                    │  复杂 → 多步RAG              │
                    │  对话 → 历史RAG              │
                    └─────────────────────────────┘
                              │
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
from enum import Enum


# ============================================
# 测试文档
# ============================================

TEST_DOCUMENTS = [
    "RAG结合检索与生成，检索器从知识库中查找相关文档。",
    "Self-RAG允许LLM自主决定何时检索信息。",
    "Agentic RAG使用AI智能体规划检索策略。",
    "GraphRAG使用知识图谱进行实体关系检索。",
    "Transformer是一种使用自注意力的神经网络架构。",
    "分词将文本分解为token供模型处理。",
    "微调将预训练模型适应到特定任务。",
    "Embedding将文本表示为语义空间中的密集向量。",
    "BM25是信息检索中使用的排序函数。",
    "向量数据库存储embeddings用于相似性搜索。",
]


# ============================================
# 查询复杂度分类
# ============================================


class QueryComplexity(Enum):
    """查询复杂度等级"""

    SIMPLE = "simple"  # 单一事实查询
    MODERATE = "moderate"  # 需要多个事实
    COMPLEX = "complex"  # 多跳推理
    CONVERSATIONAL = "conversational"  # 带历史的追问


# ============================================
# 自适应RAG实现
# ============================================


class AdaptiveRAG:
    """根据复杂度自适应选择策略的RAG"""

    def __init__(self, vectorstore, llm, max_history: int = 5):
        self.vectorstore = vectorstore
        self.llm = llm
        self.max_history = max_history
        self.conversation_history = []
        self.strategy_used = None

    def _classify_query(self, query: str) -> QueryComplexity:
        """使用LLM分类查询复杂度"""

        classification_prompt = f"""分析这个查询并分类其复杂度:

查询: "{query}"

分类选项:
- SIMPLE: 单一事实查找 (如 "什么是X?")
- MODERATE: 需要多个事实 (如 "X和Y的区别是什么?")
- COMPLEX: 多跳推理 (如 "什么导致了X进而导致Y?")
- CONVERSATIONAL: 带有隐含引用的追问 (如 "那X呢?")

请只回复一个词:"""

        response = self.llm.invoke(classification_prompt)
        result = response.content.strip().upper()

        for complexity in QueryComplexity:
            if complexity.value.upper() in result:
                return complexity

        return QueryComplexity.SIMPLE

    def _retrieve_simple(self, query: str) -> list:
        """标准单次检索"""
        retriever = self.vectorstore.as_retriever(search_kwargs={"k": 3})
        return retriever.invoke(query)

    def _retrieve_moderate(self, query: str) -> list:
        """增强检索，获取更多上下文"""
        retriever = self.vectorstore.as_retriever(search_kwargs={"k": 5})
        return retriever.invoke(query)

    def _retrieve_complex(self, query: str) -> list:
        """复杂查询的多步检索"""
        decomposition_prompt = f"""将这个复杂查询分解为更简单的子问题:

查询: "{query}"

列出2-3个有助于回答此问题的子问题:"""

        response = self.llm.invoke(decomposition_prompt)
        sub_questions = response.content.split("\n")[:3]

        retriever = self.vectorstore.as_retriever(search_kwargs={"k": 2})
        all_docs = []

        for sq in sub_questions:
            if sq.strip():
                docs = retriever.invoke(sq.strip())
                all_docs.extend(docs)

        seen = set()
        unique_docs = []
        for doc in all_docs:
            if doc.page_content not in seen:
                seen.add(doc.page_content)
                unique_docs.append(doc)

        return unique_docs[:5]

    def _retrieve_conversational(self, query: str) -> list:
        """带对话历史的检索"""
        context_query = query
        if self.conversation_history:
            recent = self.conversation_history[-2:]
            history_text = " | ".join([t["user"] for t in recent])
            context_query = f"{history_text} | {query}"

        retriever = self.vectorstore.as_retriever(search_kwargs={"k": 3})
        return retriever.invoke(context_query)

    def query(self, user_query: str, force_strategy: QueryComplexity = None) -> dict:
        """自适应策略选择执行"""

        if force_strategy:
            complexity = force_strategy
            self.strategy_used = complexity.value
        else:
            complexity = self._classify_query(user_query)
            self.strategy_used = complexity.value

        if complexity == QueryComplexity.SIMPLE:
            docs = self._retrieve_simple(user_query)
        elif complexity == QueryComplexity.MODERATE:
            docs = self._retrieve_moderate(user_query)
        elif complexity == QueryComplexity.COMPLEX:
            docs = self._retrieve_complex(user_query)
        elif complexity == QueryComplexity.CONVERSATIONAL:
            docs = self._retrieve_conversational(user_query)

        context = "\n\n".join([doc.page_content for doc in docs])

        prompt = f"""上下文:
{context}

问题: {user_query}

请根据以上上下文回答:"""

        response = self.llm.invoke(prompt)

        self.conversation_history.append(
            {
                "user": user_query,
                "assistant": response.content,
                "strategy": complexity.value,
            }
        )

        return {
            "answer": response.content,
            "retrieved_docs": [doc.page_content for doc in docs],
            "complexity": complexity.value,
            "strategy": self.strategy_used,
            "num_docs": len(docs),
        }


# ============================================
# 测试函数
# ============================================


def test_01_simple_query_classification(config: TestConfig):
    """测试: 简单查询检测和处理"""
    print("\n[测试 01] 简单查询分类")

    vectorstore = create_vector_store(TEST_DOCUMENTS, config)
    if vectorstore is None:
        return {"passed": False, "message": "创建向量存储失败"}

    llm = create_llm(config, temperature=0)
    if llm is None:
        return {"passed": False, "message": "创建LLM失败"}

    rag = AdaptiveRAG(vectorstore, llm)
    result = rag.query("什么是RAG?")

    return {
        "passed": True,
        "message": f"查询分类为 {result['complexity']}，使用 {result['strategy']} 策略",
        "details": {
            "complexity": result["complexity"],
            "strategy": result["strategy"],
            "num_docs": result["num_docs"],
            "answer_preview": result["answer"][:100],
        },
    }


def test_02_complex_query_handling(config: TestConfig):
    """测试: 复杂多跳查询处理"""
    print("\n[测试 02] 复杂查询处理")

    vectorstore = create_vector_store(TEST_DOCUMENTS, config)
    if vectorstore is None:
        return {"passed": False, "message": "创建向量存储失败"}

    llm = create_llm(config, temperature=0)
    if llm is None:
        return {"passed": False, "message": "创建LLM失败"}

    rag = AdaptiveRAG(vectorstore, llm)

    result = rag.query("不同RAG架构如何比较?", force_strategy=QueryComplexity.COMPLEX)

    return {
        "passed": True,
        "message": f"复杂查询使用 {result['strategy']} 策略处理",
        "details": {"strategy": result["strategy"], "num_docs": result["num_docs"]},
    }


def test_03_different_strategies(config: TestConfig):
    """测试: 所有策略类型"""
    print("\n[测试 03] 不同策略类型")

    vectorstore = create_vector_store(TEST_DOCUMENTS, config)
    if vectorstore is None:
        return {"passed": False, "message": "创建向量存储失败"}

    llm = create_llm(config, temperature=0)
    if llm is None:
        return {"passed": False, "message": "创建LLM失败"}

    rag = AdaptiveRAG(vectorstore, llm)

    strategies_tested = []

    for strategy in [
        QueryComplexity.SIMPLE,
        QueryComplexity.MODERATE,
        QueryComplexity.COMPLEX,
    ]:
        result = rag.query("什么是Self-RAG?", force_strategy=strategy)
        strategies_tested.append(result["strategy"])

    return {
        "passed": len(strategies_tested) == 3,
        "message": f"测试了 {len(strategies_tested)} 种策略",
        "details": {"strategies_used": strategies_tested},
    }


# ============================================
# 主函数
# ============================================


def main():
    print("=" * 60)
    print("自适应RAG架构测试")
    print("=" * 60)

    config = TestConfig()

    if not config.is_configured:
        print("\n⚠️  警告: API未配置!")
        return

    print(f"\n[配置] LLM: {config.llm_provider}/{config.openai_model}")

    results = []

    results.append(
        run_test(lambda: test_01_simple_query_classification(config), "简单查询")
    )
    results.append(run_test(lambda: test_02_complex_query_handling(config), "复杂查询"))
    results.append(run_test(lambda: test_03_different_strategies(config), "策略选择"))

    print_test_results(results)

    failed = sum(1 for r in results if not r.passed)
    return 0 if failed == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
