#!/usr/bin/env python3
"""
测试: In-Context RALM (12_in_context_ralm.py)
=============================================

不进行微调，仅依靠上下文学习进行检索增强生成。
In-Context RALM: Retrieval-Augmented Language Model with in-context learning.

运行: python tests/rag_architectures/12_in_context_ralm.py

架构:
    ┌──────────┐    ┌─────────────┐    ┌──────────────┐    ┌──────────────┐
    │   Query  │───▶│  Retrieve   │───▶│  Add to      │───▶│  Generate    │
    │          │    │  Docs       │    │  Context     │    │  (no finetune)│
    └──────────┘    └─────────────┘    └──────────────┘    └──────────────┘
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
    "RAG结合检索与生成，通过从外部知识库检索相关信息来增强大语言模型。",
    "In-Context RALM不需要对模型进行微调，仅通过在提示中添加检索到的文档来增强生成。",
    "Transformer架构使用自注意力机制来处理序列中的依赖关系。",
    "大语言模型在大量文本数据上进行预训练，学习语言的通用表示。",
    "检索增强可以帮助减少LLM的幻觉问题。",
    "上下文学习允许模型通过少量示例来学习新任务。",
    "Embedding将文本表示为密集向量，相似文本具有相似的向量表示。",
    "向量数据库支持高效的相似性搜索。",
]


# ============================================
# In-Context RALM 实现
# ============================================


class InContextRALM:
    """
    In-Context RALM 实现

    特点:
    1. 不对LLM进行微调
    2. 将检索到的文档直接添加到上下文
    3. 利用LLM的上下文学习能力
    4. 通过示例演示检索增强的效果
    """

    def __init__(self, vectorstore, llm):
        self.vectorstore = vectorstore
        self.llm = llm

    def _create_in_context_examples(self) -> list:
        """创建上下文学习示例"""
        return [
            {
                "question": "什么是RAG?",
                "context": "RAG结合检索与生成，通过从外部知识库检索相关信息来增强大语言模型。",
                "answer": "RAG(检索增强生成)是一种结合信息检索和文本生成的技术...",
            },
            {
                "question": "Transformer是什么?",
                "context": "Transformer架构使用自注意力机制来处理序列中的依赖关系。",
                "answer": "Transformer是一种基于注意力机制的神经网络架构...",
            },
        ]

    def query(self, query: str) -> dict:
        """执行In-Context RALM"""

        # 检索相关文档
        retriever = self.vectorstore.as_retriever(search_kwargs={"k": 3})
        docs = retriever.invoke(query)

        # 获取上下文示例
        examples = self._create_in_context_examples()

        # 构建上下文学习提示
        example_text = "\n\n".join(
            [
                f"示例问题: {ex['question']}\n相关上下文: {ex['context']}\n答案: {ex['answer']}"
                for ex in examples
            ]
        )

        retrieved_context = "\n\n".join([doc.page_content for doc in docs])

        prompt = f"""以下是一些检索增强的回答示例:

{example_text}

现在请回答新的问题:

相关上下文:
{retrieved_context}

问题: {query}

答案:"""

        response = self.llm.invoke(prompt)

        return {
            "answer": response.content,
            "retrieved_docs": [doc.page_content for doc in docs],
            "num_docs": len(docs),
            "examples_used": len(examples),
        }


# ============================================
# 测试函数
# ============================================


def test_01_in_context_ralm_basic(config: TestConfig):
    """测试: 基本的In-Context RALM"""
    print("\n[测试 01] 基本的In-Context RALM")

    vectorstore = create_vector_store(TEST_DOCUMENTS, config)
    if vectorstore is None:
        return {"passed": False, "message": "创建向量存储失败"}

    llm = create_llm(config, temperature=0)
    if llm is None:
        return {"passed": False, "message": "创建LLM失败"}

    icalm = InContextRALM(vectorstore, llm)
    result = icalm.query("什么是RAG?")

    return {
        "passed": "answer" in result,
        "message": f"In-Context RALM完成，使用了{result.get('examples_used', 0)}个示例",
        "details": {
            "num_examples": result.get("examples_used", 0),
            "num_retrieved": result.get("num_docs", 0),
            "answer_preview": result.get("answer", "")[:100],
        },
    }


def test_02_context_learning_examples(config: TestConfig):
    """测试: 上下文学习示例的使用"""
    print("\n[测试 02] 上下文学习示例")

    vectorstore = create_vector_store(TEST_DOCUMENTS, config)
    if vectorstore is None:
        return {"passed": False, "message": "创建向量存储失败"}

    llm = create_llm(config, temperature=0)
    if llm is None:
        return {"passed": False, "message": "创建LLM失败"}

    icalm = InContextRALM(vectorstore, llm)
    result = icalm.query("Transformer是如何工作的?")

    has_examples = result.get("examples_used", 0) > 0

    return {
        "passed": has_examples,
        "message": f"使用了{result.get('examples_used', 0)}个上下文示例",
        "details": {"examples": result.get("examples_used", 0)},
    }


def test_03_retrieval_in_context(config: TestConfig):
    """测试: 检索增强的上下文"""
    print("\n[测试 03] 检索增强的上下文")

    vectorstore = create_vector_store(TEST_DOCUMENTS, config)
    if vectorstore is None:
        return {"passed": False, "message": "创建向量存储失败"}

    llm = create_llm(config, temperature=0)
    if llm is None:
        return {"passed": False, "message": "创建LLM失败"}

    icalm = InContextRALM(vectorstore, llm)
    result = icalm.query("大语言模型如何减少幻觉?")

    return {
        "passed": result.get("num_docs", 0) > 0,
        "message": f"检索到{result.get('num_docs', 0)}个文档",
        "details": {"num_docs": result.get("num_docs", 0)},
    }


# ============================================
# 主函数
# ============================================


def main():
    print("=" * 60)
    print("IN-CONTEXT RALM 架构测试")
    print("=" * 60)

    config = TestConfig()

    if not config.is_configured:
        print("\n⚠️  警告: API未配置!")
        print("请在.env文件中设置环境变量")
        return

    print(f"\n[配置] LLM: {config.llm_provider}/{config.openai_model}")

    results = []

    results.append(
        run_test(lambda: test_01_in_context_ralm_basic(config), "基本In-Context RALM")
    )
    results.append(
        run_test(lambda: test_02_context_learning_examples(config), "上下文学习示例")
    )
    results.append(
        run_test(lambda: test_03_retrieval_in_context(config), "检索增强上下文")
    )

    print_test_results(results)

    failed = sum(1 for r in results if not r.passed)
    return 0 if failed == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
