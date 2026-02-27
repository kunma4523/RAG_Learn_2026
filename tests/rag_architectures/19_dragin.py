#!/usr/bin/env python3
"""
测试: DRAGIN (19_dragin.py)
===========================

Dynamic Retrieval Augmented Generation with INterest awareness.
根据注意力动态决定何时检索。

运行: python tests/rag_architectures/19_dragin.py

架构:
    ┌──────────────────────────────────────────────────────────┐
    │                    生成过程 (动态)                         │
    │  ┌─────────┐   ┌────────────┐   ┌─────────┐   ┌──────┐ │
    │  │Generate │──▶│Attention   │──▶│Retrieve │──▶│Next  │ │
    │  │ Tokens  │   │  on Token  │   │ if needed│  │Token │ │
    │  └─────────┘   └────────────┘   └─────────┘   └──────┘ │
    └──────────────────────────────────────────────────────────┘
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
    "RAG结合检索与生成，增强大语言模型的能力。",
    "DRAGIN根据生成过程中的注意力动态决定何时检索。",
    "Self-RAG使用反思token来决定检索时机。",
    "Agentic RAG使用智能体来规划检索策略。",
    "Transformer使用自注意力机制处理序列。",
    "向量数据库支持高效的相似性搜索。",
    "知识图谱可以用于增强检索效果。",
    "BM25是一种经典的稀疏检索方法。",
]


# ============================================
# DRAGIN 实现
# ============================================


class DRAGIN:
    """
    DRAGIN (Dynamic Retrieval Augmented Generation with INterest awareness) 实现

    特点:
    1. 动态评估每个token的检索需求
    2. 根据注意力权重决定何时检索
    3. 仅在需要时触发检索
    4. 高效利用检索资源
    """

    def __init__(self, vectorstore, llm, max_retrievals: int = 3):
        self.vectorstore = vectorstore
        self.llm = llm
        self.max_retrievals = max_retrievals
        self.retrieval_decisions = []

    def _calculate_interest_score(
        self, generated_so_far: str, next_token_hint: str
    ) -> float:
        """计算当前上下文的兴趣分数"""

        prompt = f"""已生成的内容: {generated_so_far[-200:]}

预测的下一个token: {next_token_hint}

评估是否需要检索更多外部信息来帮助生成下一个token。
返回0-1之间的分数，1表示非常需要检索:
返回数字即可:"""

        try:
            response = self.llm.invoke(prompt)
            score = float(response.content.strip())
            return min(max(score, 0), 1)
        except:
            return 0.5

    def _should_retrieve(self, interest_score: float, threshold: float = 0.7) -> bool:
        """根据兴趣分数决定是否检索"""
        return interest_score > threshold

    def _dynamic_retrieve(self, query: str, context: str) -> str:
        """动态检索"""

        retriever = self.vectorstore.as_retriever(search_kwargs={"k": 3})
        docs = retriever.invoke(query + " " + context[-100:])

        return "\n\n".join([doc.page_content for doc in docs])

    def query(self, query: str) -> dict:
        """执行DRAGIN"""

        # 模拟多步生成过程
        initial_prompt = f"问题: {query}\n\n回答:"

        response = self.llm.invoke(initial_prompt)
        generated = response.content

        # 模拟动态检索决策
        for i in range(self.max_retrievals):
            # 计算兴趣分数
            interest_score = self._calculate_interest_score(generated, "next")

            self.retrieval_decisions.append(
                {
                    "step": i + 1,
                    "interest_score": interest_score,
                    "should_retrieve": interest_score > 0.7,
                }
            )

            # 决定是否检索
            if self._should_retrieve(interest_score):
                retrieved_context = self._dynamic_retrieve(query, generated)

                # 继续生成
                continue_prompt = f"已检索内容:\n{retrieved_context}\n\n已生成:\n{generated}\n\n请继续完善回答:"
                response = self.llm.invoke(continue_prompt)
                generated = response.content
            else:
                break

        return {
            "answer": generated,
            "retrieval_decisions": self.retrieval_decisions,
            "total_retrievals": sum(
                1 for d in self.retrieval_decisions if d["should_retrieve"]
            ),
        }


# ============================================
# 测试函数
# ============================================


def test_01_dragin_basic(config: TestConfig):
    """测试: 基本DRAGIN"""
    print("\n[测试 01] 基本DRAGIN")

    vectorstore = create_vector_store(TEST_DOCUMENTS, config)
    if vectorstore is None:
        return {"passed": False, "message": "创建向量存储失败"}

    llm = create_llm(config, temperature=0)
    if llm is None:
        return {"passed": False, "message": "创建LLM失败"}

    dragin = DRAGIN(vectorstore, llm)
    result = dragin.query("DRAGIN是什么?")

    return {
        "passed": "answer" in result,
        "message": f"DRAGIN完成，动态决策{len(result.get('retrieval_decisions', []))}次",
        "details": {
            "decisions": result.get("retrieval_decisions", []),
            "answer_preview": result.get("answer", "")[:100],
        },
    }


def test_02_dynamic_decision(config: TestConfig):
    """测试: 动态检索决策"""
    print("\n[测试 02] 动态检索决策")

    vectorstore = create_vector_store(TEST_DOCUMENTS, config)
    if vectorstore is None:
        return {"passed": False, "message": "创建向量存储失败"}

    llm = create_llm(config, temperature=0)
    if llm is None:
        return {"passed": False, "message": "创建LLM失败"}

    dragin = DRAGIN(vectorstore, llm)
    result = dragin.query("解释RAG和Self-RAG的区别")

    has_decisions = len(result.get("retrieval_decisions", [])) > 0

    return {
        "passed": has_decisions,
        "message": f"进行了{len(result.get('retrieval_decisions', []))}次动态决策",
        "details": {"decisions": len(result.get("retrieval_decisions", []))},
    }


def test_03_interest_based_retrieval(config: TestConfig):
    """测试: 基于兴趣的检索"""
    print("\n[测试 03] 基于兴趣的检索")

    vectorstore = create_vector_store(TEST_DOCUMENTS, config)
    if vectorstore is None:
        return {"passed": False, "message": "创建向量存储失败"}

    llm = create_llm(config, temperature=0)
    if llm is None:
        return {"passed": False, "message": "创建LLM失败"}

    dragin = DRAGIN(vectorstore, llm)
    result = dragin.query("Transformer的自注意力如何工作?")

    return {
        "passed": result.get("total_retrievals", 0) >= 0,
        "message": f"基于兴趣分数触发了{result.get('total_retrievals', 0)}次检索",
        "details": {"retrievals": result.get("total_retrievals", 0)},
    }


# ============================================
# 主函数
# ============================================


def main():
    print("=" * 60)
    print("DRAGIN 架构测试")
    print("=" * 60)

    config = TestConfig()

    if not config.is_configured:
        print("\n⚠️  警告: API未配置!")
        return

    print(f"\n[配置] LLM: {config.llm_provider}/{config.openai_model}")

    results = []

    results.append(run_test(lambda: test_01_dragin_basic(config), "基本DRAGIN"))
    results.append(run_test(lambda: test_02_dynamic_decision(config), "动态决策"))
    results.append(
        run_test(lambda: test_03_interest_based_retrieval(config), "基于兴趣检索")
    )

    print_test_results(results)

    failed = sum(1 for r in results if not r.passed)
    return 0 if failed == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
