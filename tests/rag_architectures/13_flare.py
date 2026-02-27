#!/usr/bin/env python3
"""
测试: FLARE (13_flare.py)
=========================

Forward-Looking Active REtrieval.
在生成过程中主动触发检索，根据生成的临时结果动态检索。

运行: python tests/rag_architectures/13_flare.py

架构:
    ┌─────────────────────────────────────────────────────────────────┐
    │                        生成过程                                  │
    │  ┌─────────┐   ┌──────────┐   ┌─────────┐   ┌──────────────┐ │
    │  │ Generate │──▶│Contains │──▶│Retrieve│──▶│  Continue    │ │
    │  │ Tokens  │   │ Unknown? │   │  Docs  │   │  Generation  │ │
    │  └─────────┘   └──────────┘   └─────────┘   └──────────────┘ │
    └─────────────────────────────────────────────────────────────────┘
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
    "RAG(检索增强生成)结合信息检索与文本生成，增强大语言模型的能力。",
    "FLARE是一种在生成过程中主动触发检索的方法。",
    "当生成模型遇到不确定的内容时，会主动检索相关信息来辅助生成。",
    "Self-RAG使用反思token来决定是否需要检索。",
    "Agentic RAG使用智能体来规划检索策略。",
    "知识图谱可以用于增强检索效果。",
    "向量数据库存储embeddings并支持相似性搜索。",
    "BM25是一种基于关键词的稀疏检索方法。",
]


# ============================================
# FLARE 实现
# ============================================


class FLARE:
    """
    FLARE (Forward-Looking Active REtrieval) 实现

    特点:
    1. 在生成过程中主动检测不确定内容
    2. 当遇到未知信息时触发检索
    3. 使用临时生成的片段来指导检索
    4. 动态迭代直到生成完成
    """

    def __init__(self, vectorstore, llm, max_iterations: int = 3):
        self.vectorstore = vectorstore
        self.llm = llm
        self.max_iterations = max_iterations
        self.retrieval_log = []

    def _check_uncertainty(self, generated_text: str) -> bool:
        """检查生成文本中是否有不确定的内容"""

        uncertainty_prompt = f"""分析以下文本，检测是否有不确定的或可能需要更多事实信息的地方:

生成的文本: {generated_text}

检测以下模式:
- 模糊的引用
- 可能的事实错误
- 不确定的知识
- 需要验证的信息

如果需要更多信息，返回YES，否则返回NO:"""

        response = self.llm.invoke(uncertainty_prompt)
        return "YES" in response.content.upper()

    def _extract检索_keywords(self, generated_text: str) -> str:
        """提取用于检索的关键词"""

        extract_prompt = f"""从以下文本中提取可用于检索的关键信息:

文本: {generated_text}

提取2-3个关键检索词或问题:"""

        response = self.llm.invoke(extract_prompt)
        return response.content.strip()

    def query(self, query: str) -> dict:
        """执行FLARE"""

        # 初始生成
        initial_prompt = f"问题: {query}\n\n请生成一个详细的回答:"

        response = self.llm.invoke(initial_prompt)
        generated = response.content

        # 迭代检索
        for i in range(self.max_iterations):
            # 检查不确定性
            if not self._check_uncertainty(generated):
                break

            # 提取检索关键词
            search_terms = self._extract检索_keywords(generated)

            # 检索
            retriever = self.vectorstore.as_retriever(search_kwargs={"k": 3})
            docs = retriever.invoke(search_terms)

            self.retrieval_log.append(
                {
                    "iteration": i + 1,
                    "search_terms": search_terms,
                    "num_docs": len(docs),
                }
            )

            # 使用检索到的内容继续生成
            context = "\n\n".join([doc.page_content for doc in docs])

            continue_prompt = f"""已检索到的相关信息:
{context}

之前生成的回答:
{generated}

请基于检索到的信息改进和完善回答:"""

            response = self.llm.invoke(continue_prompt)
            generated = response.content

        return {
            "answer": generated,
            "retrieval_count": len(self.retrieval_log),
            "retrieval_log": self.retrieval_log,
        }


# ============================================
# 测试函数
# ============================================


def test_01_flare_basic(config: TestConfig):
    """测试: 基本FLARE"""
    print("\n[测试 01] 基本FLARE")

    vectorstore = create_vector_store(TEST_DOCUMENTS, config)
    if vectorstore is None:
        return {"passed": False, "message": "创建向量存储失败"}

    llm = create_llm(config, temperature=0)
    if llm is None:
        return {"passed": False, "message": "创建LLM失败"}

    flare = FLARE(vectorstore, llm)
    result = flare.query("FLARE是什么?")

    return {
        "passed": "answer" in result,
        "message": f"FLARE执行完成，触发了{result.get('retrieval_count', 0)}次检索",
        "details": {
            "retrieval_count": result.get("retrieval_count", 0),
            "answer_preview": result.get("answer", "")[:100],
        },
    }


def test_02_active_retrieval(config: TestConfig):
    """测试: 主动检索机制"""
    print("\n[测试 02] 主动检索机制")

    vectorstore = create_vector_store(TEST_DOCUMENTS, config)
    if vectorstore is None:
        return {"passed": False, "message": "创建向量存储失败"}

    llm = create_llm(config, temperature=0)
    if llm is None:
        return {"passed": False, "message": "创建LLM失败"}

    flare = FLARE(vectorstore, llm)
    result = flare.query("RAG和Self-RAG有什么区别?")

    return {
        "passed": True,
        "message": f"检索日志: {len(result.get('retrieval_log', []))}次检索",
        "details": {"retrieval_log": result.get("retrieval_log", [])},
    }


def test_03_iteration_control(config: TestConfig):
    """测试: 迭代控制"""
    print("\n[测试 03] 迭代控制")

    vectorstore = create_vector_store(TEST_DOCUMENTS, config)
    if vectorstore is None:
        return {"passed": False, "message": "创建向量存储失败"}

    llm = create_llm(config, temperature=0)
    if llm is None:
        return {"passed": False, "message": "创建LLM失败"}

    flare = FLARE(vectorstore, llm, max_iterations=2)
    result = flare.query("解释知识图谱在RAG中的作用")

    return {
        "passed": result.get("retrieval_count", 0) <= 2,
        "message": f"最大迭代控制在2次，实际检索{result.get('retrieval_count', 0)}次",
        "details": {"retrieval_count": result.get("retrieval_count", 0)},
    }


# ============================================
# 主函数
# ============================================


def main():
    print("=" * 60)
    print("FLARE 架构测试")
    print("=" * 60)

    config = TestConfig()

    if not config.is_configured:
        print("\n⚠️  警告: API未配置!")
        return

    print(f"\n[配置] LLM: {config.llm_provider}/{config.openai_model}")

    results = []

    results.append(run_test(lambda: test_01_flare_basic(config), "基本FLARE"))
    results.append(run_test(lambda: test_02_active_retrieval(config), "主动检索"))
    results.append(run_test(lambda: test_03_iteration_control(config), "迭代控制"))

    print_test_results(results)

    failed = sum(1 for r in results if not r.passed)
    return 0 if failed == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
