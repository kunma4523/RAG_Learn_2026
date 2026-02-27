#!/usr/bin/env python3
"""
测试: 自反思RAG (05_self_rag.py)
================================

自反思RAG，LLM在生成过程中决定何时检索以及如何使用检索内容。

运行: python tests/rag_architectures/05_self_rag.py

架构:
    ┌──────────────────────────────────────────────────────────────┐
    │                      生成过程                                   │
    │  ┌──────────┐    ┌───────────┐    ┌──────────┐    ┌────────┐ │
    │  │  生成    │───▶│   需要    │───▶│  检索   │───▶│验证使用│ │
    │  │  Token  │    │  更多信息?│    │   文档   │    │        │ │
    │  └──────────┘    └───────────┘    └──────────┘    └────────┘ │
    │                         │ No                                  │
    │                         ▼                                     │
    │                  ┌─────────────┐                            │
    │                  │  继续生成   │                            │
    │                  └─────────────┘                            │
    └──────────────────────────────────────────────────────────────┘
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
    "RAG(检索增强生成)将检索与文本生成相结合，帮助减少LLM的幻觉。",
    "Self-RAG是一个让LLM主动决定何时检索信息的自反思框架。",
    "RAG系统中的检索器从知识库中查找相关文档。",
    "Agentic RAG使用AI智能体来规划和执行检索策略。",
    "GraphRAG使用知识图谱来捕获实体关系，实现更复杂的多跳推理。",
    "HyDE先生成假设回答，然后用它来检索相似的真实文档。",
]


# ============================================
# 自反思RAG实现
# ============================================


class SelfRAG:
    """简化的Self-RAG实现"""

    def __init__(self, vectorstore, llm):
        self.vectorstore = vectorstore
        self.llm = llm
        self.retrieval_count = 0
        self.reflection_log = []

    def _should_retrieve(self, generated_so_far: str, question: str) -> bool:
        """根据已生成内容决定是否需要检索"""

        decision_prompt = f"""问题: {question}

目前已生成: "{generated_so_far}"

生成的文本是否需要更多事实信息来准确回答问题?
考虑:
- 我们是否完全回答了问题?
- 信息是否有空白?
- 更多上下文是否有帮助?

请只回复 YES 或 NO:"""

        response = self.llm.invoke(decision_prompt)
        decision = response.content.strip().upper()

        self.reflection_log.append(
            {"generated": generated_so_far[:100], "decision": decision}
        )

        return "YES" in decision

    def _verify_retrieved_docs(self, query: str, docs: list) -> list:
        """验证并过滤相关检索文档"""

        if not docs:
            return []

        doc_texts = "\n".join(
            [f"{i + 1}. {doc.page_content[:150]}..." for i, doc in enumerate(docs)]
        )

        verification_prompt = f"""问题: {query}

检索到的文档:
{doc_texts}

请评估每个文档与问题的相关性(1-5):
请只回复数字(如 "1, 3, 4"):"""

        response = self.llm.invoke(verification_prompt)

        try:
            ratings = [int(x.strip()) for x in response.content.split(",")]
            relevant_docs = [
                doc
                for i, doc in enumerate(docs)
                if i < len(ratings) and ratings[i] >= 3
            ]
            return relevant_docs
        except:
            return docs[:3]

    def query(self, question: str, max_retrievals: int = 3) -> dict:
        """执行自反思RAG"""

        self.retrieval_count = 0
        self.reflection_log = []

        # 初始生成
        initial_prompt = f"""问题: {question}

请根据你的知识生成一个简洁的回答:"""

        initial_response = self.llm.invoke(initial_prompt)
        generated = initial_response.content

        # 检查是否需要检索
        if not self._should_retrieve(generated, question):
            return {
                "answer": generated,
                "retrieval_count": 0,
                "reflection_log": self.reflection_log,
                "retrieved_needed": False,
            }

        # 检索循环
        for i in range(max_retrievals):
            self.retrieval_count += 1

            # 检索文档
            retriever = self.vectorstore.as_retriever(search_kwargs={"k": 3})
            docs = retriever.invoke(question)

            # 验证相关性
            relevant_docs = self._verify_retrieved_docs(question, docs)

            if not relevant_docs:
                break

            # 使用检索到的上下文
            context = "\n\n".join([doc.page_content for doc in relevant_docs])

            refinement_prompt = f"""问题: {question}

之前的回答: {generated}

检索到的上下文:
{context}

请使用检索到的信息改进和完善回答:"""

            refined_response = self.llm.invoke(refinement_prompt)
            generated = refined_response.content

            # 检查是否还需要更多检索
            if i < max_retrievals - 1:
                if not self._should_retrieve(generated, question):
                    break

        return {
            "answer": generated,
            "retrieval_count": self.retrieval_count,
            "reflection_log": self.reflection_log,
            "retrieved_needed": self.retrieval_count > 0,
        }


# ============================================
# 测试函数
# ============================================


def test_01_self_rag_basic(config: TestConfig):
    """测试: 基本的Self-RAG功能"""
    print("\n[测试 01] 基本Self-RAG")

    vectorstore = create_vector_store(TEST_DOCUMENTS, config)
    if vectorstore is None:
        return {"passed": False, "message": "创建向量存储失败"}

    llm = create_llm(config, temperature=0)
    if llm is None:
        return {"passed": False, "message": "创建LLM失败"}

    rag = SelfRAG(vectorstore, llm)
    result = rag.query("什么是Self-RAG?")

    return {
        "passed": "answer" in result,
        "message": f"Self-RAG完成，检索 {result.get('retrieval_count', 0)} 次",
        "details": {
            "retrieval_count": result.get("retrieval_count", 0),
            "retrieved_needed": result.get("retrieved_needed", False),
            "answer_preview": result.get("answer", "")[:150],
        },
    }


def test_02_reflection_decisions(config: TestConfig):
    """测试: 反思和决策"""
    print("\n[测试 02] 反思决策")

    vectorstore = create_vector_store(TEST_DOCUMENTS, config)
    if vectorstore is None:
        return {"passed": False, "message": "创建向量存储失败"}

    llm = create_llm(config, temperature=0)
    if llm is None:
        return {"passed": False, "message": "创建LLM失败"}

    rag = SelfRAG(vectorstore, llm)
    result = rag.query("什么是RAG，它是如何工作的?")

    has_reflection = len(result.get("reflection_log", [])) > 0

    return {
        "passed": has_reflection,
        "message": f"反思日志有 {len(result.get('reflection_log', []))} 条记录",
        "details": {
            "reflection_count": len(result.get("reflection_log", [])),
            "retrieval_count": result.get("retrieval_count", 0),
        },
    }


def test_03_multiple_retrievals(config: TestConfig):
    """测试: 多次检索迭代"""
    print("\n[测试 03] 多次检索迭代")

    vectorstore = create_vector_store(TEST_DOCUMENTS, config)
    if vectorstore is None:
        return {"passed": False, "message": "创建向量存储失败"}

    llm = create_llm(config, temperature=0)
    if llm is None:
        return {"passed": False, "message": "创建LLM失败"}

    rag = SelfRAG(vectorstore, llm)
    result = rag.query("比较不同的RAG架构及其使用场景", max_retrievals=3)

    return {
        "passed": "answer" in result,
        "message": f"完成最多3次检索，实际: {result.get('retrieval_count', 0)}次",
        "details": {"retrieval_count": result.get("retrieval_count", 0)},
    }


# ============================================
# 主函数
# ============================================


def main():
    print("=" * 60)
    print("自反思RAG架构测试")
    print("=" * 60)

    config = TestConfig()

    if not config.is_configured:
        print("\n⚠️  警告: API未配置!")
        return

    print(f"\n[配置] LLM: {config.llm_provider}/{config.openai_model}")

    results = []

    results.append(run_test(lambda: test_01_self_rag_basic(config), "基本Self-RAG"))
    results.append(run_test(lambda: test_02_reflection_decisions(config), "反思决策"))
    results.append(run_test(lambda: test_03_multiple_retrievals(config), "多次检索"))

    print_test_results(results)

    failed = sum(1 for r in results if not r.passed)
    return 0 if failed == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
