#!/usr/bin/env python3
"""
测试: Self-Ask + RAG (15_self_ask_rag.py)
==========================================

自问自答型RAG，通过逐步提问来引导检索和生成。

运行: python tests/rag_architectures/15_self_ask_rag.py

架构:
    ┌──────────┐    ┌─────────────┐    ┌──────────────┐
    │   Query  │───▶│  Self-Ask   │───▶│   Follow-up  │
    │          │    │ Decompose   │    │   Questions  │
    └──────────┘    └─────────────┘    └──────────────┘
                                             │
                    ┌─────────────────────────┘
                    ▼
            ┌───────────────┐    ┌──────────────┐
            │   Retrieve    │───▶│    Answer    │
            │   & Answer    │    │   Sub-Quest  │
            └───────────────┘    └──────────────┘
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
    "RAG(检索增强生成)是一种结合信息检索和文本生成的技术。",
    "Transformer是一种使用自注意力机制的神经网络架构。",
    "BERT是基于Transformer的预训练语言模型。",
    "GPT是OpenAI开发的大语言模型系列。",
    "大语言模型通过海量文本数据进行预训练。",
    "微调是在特定任务上进一步训练预训练模型。",
    "知识图谱使用图结构表示实体和关系。",
    "向量数据库存储embeddings用于相似性搜索。",
]


# ============================================
# Self-Ask RAG 实现
# ============================================


class SelfAskRAG:
    """
    Self-Ask + RAG 实现

    特点:
    1. 将复杂问题分解为子问题
    2. 自问自答逐步推理
    3. 每个子问题触发检索
    4. 组合所有子答案形成最终答案
    """

    def __init__(self, vectorstore, llm):
        self.vectorstore = vectorstore
        self.llm = llm
        self.question_log = []

    def _decompose_question(self, query: str) -> list:
        """将问题分解为自问自答的子问题"""

        prompt = f"""将这个问题分解为2-3个简单的自问自答子问题:

问题: {query}

格式示例:
Q1: [子问题1]
A1: [简短回答]
Q2: [子问题2]
A2: [简短回答]"""

        response = self.llm.invoke(prompt)

        # 解析子问题
        lines = response.content.split("\n")
        questions = []
        for line in lines:
            if line.strip().startswith("Q") and ":" in line:
                questions.append(line.split(":", 1)[1].strip())

        return questions[:3] if questions else [query]

    def _answer_sub_question(self, question: str) -> str:
        """回答子问题"""

        # 检索相关文档
        retriever = self.vectorstore.as_retriever(search_kwargs={"k": 3})
        docs = retriever.invoke(question)

        context = "\n\n".join([doc.page_content for doc in docs])

        prompt = f"相关上下文:\n{context}\n\n问题: {question}\n\n简短回答:"

        response = self.llm.invoke(prompt)

        return response.content

    def query(self, query: str) -> dict:
        """执行Self-Ask RAG"""

        # 分解问题
        sub_questions = self._decompose_question(query)

        self.question_log.append({"main_query": query, "sub_questions": sub_questions})

        # 回答每个子问题
        answers = []
        for sq in sub_questions:
            answer = self._answer_sub_question(sq)
            answers.append({"question": sq, "answer": answer})

            self.question_log.append(
                {"type": "sub_answer", "question": sq, "answer": answer}
            )

        # 组合最终答案
        combined_context = "\n".join(
            [f"Q: {a['question']}\nA: {a['answer']}" for a in answers]
        )

        final_prompt = f"""基于以下自问自答的解答:

{combined_context}

原始问题: {query}

请综合以上解答，给出最终答案:"""

        final_response = self.llm.invoke(final_prompt)

        return {
            "answer": final_response.content,
            "sub_questions": sub_questions,
            "answers": answers,
            "num_sub_questions": len(sub_questions),
        }


# ============================================
# 测试函数
# ============================================


def test_01_self_ask_basic(config: TestConfig):
    """测试: 基本Self-Ask"""
    print("\n[测试 01] 基本Self-Ask RAG")

    vectorstore = create_vector_store(TEST_DOCUMENTS, config)
    if vectorstore is None:
        return {"passed": False, "message": "创建向量存储失败"}

    llm = create_llm(config, temperature=0)
    if llm is None:
        return {"passed": False, "message": "创建LLM失败"}

    selfask = SelfAskRAG(vectorstore, llm)
    result = selfask.query("解释Transformer和BERT的关系")

    return {
        "passed": "answer" in result,
        "message": f"Self-Ask分解为{result.get('num_sub_questions', 0)}个子问题",
        "details": {
            "num_sub": result.get("num_sub_questions", 0),
            "answer_preview": result.get("answer", "")[:100],
        },
    }


def test_02_question_decomposition(config: TestConfig):
    """测试: 问题分解"""
    print("\n[测试 02] 问题分解")

    vectorstore = create_vector_store(TEST_DOCUMENTS, config)
    if vectorstore is None:
        return {"passed": False, "message": "创建向量存储失败"}

    llm = create_llm(config, temperature=0)
    if llm is None:
        return {"passed": False, "message": "创建LLM失败"}

    selfask = SelfAskRAG(vectorstore, llm)
    result = selfask.query("大语言模型如何通过RAG减少幻觉?")

    has_sub_questions = len(result.get("sub_questions", [])) > 0

    return {
        "passed": has_sub_questions,
        "message": f"分解为{len(result.get('sub_questions', []))}个子问题",
        "details": {"sub_questions": result.get("sub_questions", [])},
    }


def test_03_step_by_step_reasoning(config: TestConfig):
    """测试: 逐步推理"""
    print("\n[测试 03] 逐步推理")

    vectorstore = create_vector_store(TEST_DOCUMENTS, config)
    if vectorstore is None:
        return {"passed": False, "message": "创建向量存储失败"}

    llm = create_llm(config, temperature=0)
    if llm is None:
        return {"passed": False, "message": "创建LLM失败"}

    selfask = SelfAskRAG(vectorstore, llm)
    result = selfask.query("比较GPT系列模型和BERT模型的区别")

    return {
        "passed": len(result.get("answers", [])) > 0,
        "message": f"逐步推理完成，共{len(result.get('answers', []))}步",
        "details": {"steps": len(result.get("answers", []))},
    }


# ============================================
# 主函数
# ============================================


def main():
    print("=" * 60)
    print("SELF-ASK + RAG 架构测试")
    print("=" * 60)

    config = TestConfig()

    if not config.is_configured:
        print("\n⚠️  警告: API未配置!")
        return

    print(f"\n[配置] LLM: {config.llm_provider}/{config.openai_model}")

    results = []

    results.append(run_test(lambda: test_01_self_ask_basic(config), "基本Self-Ask"))
    results.append(run_test(lambda: test_02_question_decomposition(config), "问题分解"))
    results.append(run_test(lambda: test_03_step_by_step_reasoning(config), "逐步推理"))

    print_test_results(results)

    failed = sum(1 for r in results if not r.passed)
    return 0 if failed == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
