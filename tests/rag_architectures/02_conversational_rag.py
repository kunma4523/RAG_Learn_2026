#!/usr/bin/env python3
"""
测试: 对话式RAG (02_conversational_rag.py)
==========================================

支持对话历史维护的RAG流程，可以在多轮对话中保持上下文。

运行: python tests/rag_architectures/02_conversational_rag.py

架构:
    ┌─────────────┐    ┌─────────────┐    ┌──────────────┐    ┌──────────────┐
    │ Query+历史   │───▶│  检索       │───▶│  增强        │───▶│   生成       │
    └─────────────┘    └─────────────┘    └──────────────┘    └──────────────┘
                                │                    │
                                │                    ▼
                                │            ┌──────────────┐
                                │            │   历史       │
                                │            │   记忆        │
                                │            └──────────────┘
                                │                    │
                                └────────────────────┘
"""

import sys
import os

# 添加项目根目录到路径
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
    "RAG(检索增强生成)将信息检索与文本生成相结合，增强大语言模型的能力。",
    "RAG系统包括三个主要组件：检索器从知识库中查找相关文档。",
    "Self-RAG是一种自反思的检索增强生成框架，让LLM能够自主决定何时检索信息。",
    "Agentic RAG结合AI智能体，可以规划检索策略并动态使用多种工具。",
    "GraphRAG使用知识图谱来表示实体关系，实现关联信息检索。",
]


# ============================================
# 对话式RAG实现
# ============================================


class ConversationalRAG:
    """支持对话历史的RAG"""

    def __init__(self, vectorstore, llm, max_history: int = 5):
        self.vectorstore = vectorstore
        self.llm = llm
        self.max_history = max_history
        self.conversation_history = []

    def query(self, user_query: str, include_history: bool = True) -> dict:
        """带历史记录的查询"""

        # 构建带历史上下文的搜索查询
        if include_history and self.conversation_history:
            search_query = self._build_context_query(user_query)
        else:
            search_query = user_query

        # 检索文档
        retriever = self.vectorstore.as_retriever(search_kwargs={"k": 3})
        retrieved_docs = retriever.invoke(search_query)

        # 构建上下文
        context = "\n\n".join([doc.page_content for doc in retrieved_docs])

        # 带历史的提示词
        if include_history and self.conversation_history:
            history_text = self._format_history()
            prompt = f"""之前的对话:
{history_text}

当前上下文:
{context}

当前问题: {user_query}

请根据以上上下文回答当前问题:"""
        else:
            prompt = f"""上下文:
{context}

问题: {user_query}

回答:"""

        # 生成回答
        response = self.llm.invoke(prompt)

        # 存入历史记录
        self.conversation_history.append(
            {"user": user_query, "assistant": response.content}
        )

        # 修剪过长的历史
        if len(self.conversation_history) > self.max_history:
            self.conversation_history = self.conversation_history[-self.max_history :]

        return {
            "answer": response.content,
            "retrieved_docs": [doc.page_content for doc in retrieved_docs],
            "search_query": search_query,
            "history_length": len(self.conversation_history),
        }

    def _build_context_query(self, current_query: str) -> str:
        """构建带历史的查询"""
        history_queries = [turn["user"] for turn in self.conversation_history]
        return " | ".join(history_queries[-3:] + [current_query])

    def _format_history(self) -> str:
        """格式化对话历史"""
        lines = []
        for i, turn in enumerate(self.conversation_history[-self.max_history :], 1):
            lines.append(f"轮次 {i}:")
            lines.append(f"  用户: {turn['user']}")
            lines.append(f"  助手: {turn['assistant']}")
        return "\n".join(lines)

    def clear_history(self):
        """清空对话历史"""
        self.conversation_history = []


# ============================================
# 测试函数
# ============================================


def test_01_single_turn_query(config: TestConfig):
    """测试: 单轮查询(基本检索)"""
    print("\n[测试 01] 单轮查询")

    vectorstore = create_vector_store(TEST_DOCUMENTS, config)
    if vectorstore is None:
        return {"passed": False, "message": "创建向量存储失败"}

    llm = create_llm(config, temperature=0)
    if llm is None:
        return {"passed": False, "message": "创建LLM失败"}

    rag = ConversationalRAG(vectorstore, llm)

    result = rag.query("什么是RAG?")

    if not result["answer"]:
        return {"passed": False, "message": "未生成回答"}

    return {
        "passed": True,
        "message": f"查询成功，历史长度: {result['history_length']}",
        "details": {
            "answer": result["answer"][:150],
            "num_docs": len(result["retrieved_docs"]),
        },
    }


def test_02_multi_turn_conversation(config: TestConfig):
    """测试: 多轮对话与历史"""
    print("\n[测试 02] 多轮对话")

    vectorstore = create_vector_store(TEST_DOCUMENTS, config)
    if vectorstore is None:
        return {"passed": False, "message": "创建向量存储失败"}

    llm = create_llm(config, temperature=0)
    if llm is None:
        return {"passed": False, "message": "创建LLM失败"}

    rag = ConversationalRAG(vectorstore, llm, max_history=5)

    # 第一轮
    result1 = rag.query("什么是RAG?")

    # 第二轮 - 应包含历史上下文
    result2 = rag.query("它的主要组成部分是什么?")

    # 第三轮 - 追问
    result3 = rag.query("详细介绍一下检索器")

    if len(rag.conversation_history) != 3:
        return {
            "passed": False,
            "message": f"期望3条历史记录，实际{len(rag.conversation_history)}条",
        }

    # 检查搜索查询是否包含历史
    if "什么是RAG?" not in result2["search_query"]:
        return {"passed": False, "message": "历史未包含在搜索查询中"}

    return {
        "passed": True,
        "message": f"多轮对话成功，共 {len(rag.conversation_history)} 轮",
        "details": {
            "history_length": len(rag.conversation_history),
            "turn_1_query": result1["search_query"],
            "turn_2_query": result2["search_query"],
            "turn_3_query": result3["search_query"],
        },
    }


def test_03_history_context_awareness(config: TestConfig):
    """测试: 历史上下文感知"""
    print("\n[测试 03] 历史上下文感知")

    vectorstore = create_vector_store(TEST_DOCUMENTS, config)
    if vectorstore is None:
        return {"passed": False, "message": "创建向量存储失败"}

    llm = create_llm(config, temperature=0)
    if llm is None:
        return {"passed": False, "message": "创建LLM失败"}

    rag = ConversationalRAG(vectorstore, llm)

    # 询问特定主题
    rag.query("什么是Self-RAG?")

    # 后续使用代词的问题
    result = rag.query("它什么时候应该检索信息?")

    # 搜索查询应包含历史
    history_included = (
        "Self-RAG" in result["search_query"] or len(result["search_query"]) > 20
    )

    return {
        "passed": history_included,
        "message": "后续查询中包含历史上下文"
        if history_included
        else "历史上下文可能未正确包含",
        "details": {
            "search_query": result["search_query"],
            "answer_preview": result["answer"][:100],
        },
    }


def test_04_clear_history(config: TestConfig):
    """测试: 清空历史功能"""
    print("\n[测试 04] 清空历史")

    vectorstore = create_vector_store(TEST_DOCUMENTS, config)
    if vectorstore is None:
        return {"passed": False, "message": "创建向量存储失败"}

    llm = create_llm(config, temperature=0)
    if llm is None:
        return {"passed": False, "message": "创建LLM失败"}

    rag = ConversationalRAG(vectorstore, llm)

    # 添加一些历史
    rag.query("什么是RAG?")
    rag.query("什么是Agentic RAG?")

    if len(rag.conversation_history) != 2:
        return {"passed": False, "message": "历史未正确添加"}

    # 清空历史
    rag.clear_history()

    if len(rag.conversation_history) != 0:
        return {"passed": False, "message": "历史未清空"}

    return {
        "passed": True,
        "message": "历史清空成功",
        "details": {"history_after_clear": len(rag.conversation_history)},
    }


# ============================================
# 主函数
# ============================================


def main():
    """运行所有对话式RAG测试"""
    print("=" * 60)
    print("对话式RAG架构测试")
    print("=" * 60)

    config = TestConfig()

    if not config.is_configured:
        print("\n⚠️  警告: API未配置!")
        print("请在.env文件中设置环境变量。")
        return

    print(f"\n[配置] LLM: {config.llm_provider}/{config.openai_model}")
    print(f"[配置] Embedding: {config.embedding_provider}")

    results = []

    results.append(run_test(lambda: test_01_single_turn_query(config), "单轮查询"))
    results.append(
        run_test(lambda: test_02_multi_turn_conversation(config), "多轮对话")
    )
    results.append(
        run_test(lambda: test_03_history_context_awareness(config), "历史上下文感知")
    )
    results.append(run_test(lambda: test_04_clear_history(config), "清空历史"))

    print_test_results(results)

    failed = sum(1 for r in results if not r.passed)
    return 0 if failed == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
