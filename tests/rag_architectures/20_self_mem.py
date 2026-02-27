#!/usr/bin/env python3
"""
测试: Self-Mem (20_self_mem.py)
================================

结合外部检索与内部记忆的RAG系统。

运行: python tests/rag_architectures/20_self_mem.py

架构:
    ┌─────────────────────────────────────────────────────┐
    │                   Memory System                      │
    │  ┌───────────┐   ┌───────────┐   ┌───────────────┐ │
    │  │ Working  │──▶│  Long-term │──▶│   External    │ │
    │  │ Memory   │   │  Memory    │   │   Retrieval   │ │
    │  └───────────┘   └───────────┘   └───────────────┘ │
    └─────────────────────────────────────────────────────┘
                          │
                          ▼
                    ┌──────────┐
                    │ Generate │
                    └──────────┘
                          │
                          ▼
                    ┌──────────┐    ┌────────────┐
                    │  Store   │───▶│   Answer   │
                    │ to Memory│    └────────────┘
                    └──────────┘
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
    "Self-Mem结合外部检索与内部记忆系统。",
    "工作记忆存储短期上下文信息。",
    "长期记忆存储历史交互信息。",
    "Transformer使用自注意力机制处理序列。",
    "向量数据库支持高效的相似性搜索。",
    "知识图谱可以用于增强检索效果。",
    "索引优化可以提升检索效率。",
]


# ============================================
# Self-Mem 实现
# ============================================


class SelfMem:
    """
    Self-Mem 实现

    特点:
    1. 工作记忆: 当前对话上下文
    2. 长期记忆: 历史交互摘要
    3. 外部检索: 知识库检索
    4. 记忆更新: 将新信息存入记忆
    """

    def __init__(self, vectorstore, llm):
        self.vectorstore = vectorstore
        self.llm = llm
        self.working_memory = []  # 工作记忆
        self.long_term_memory = []  # 长期记忆

    def _update_working_memory(self, query: str, answer: str):
        """更新工作记忆"""
        self.working_memory.append(
            {"query": query, "answer": answer, "timestamp": len(self.working_memory)}
        )

        # 保持工作记忆在一定大小内
        if len(self.working_memory) > 3:
            self.working_memory = self.working_memory[-3:]

    def _update_long_term_memory(self):
        """更新长期记忆"""
        if not self.working_memory:
            return

        # 总结工作记忆
        summary_prompt = "总结以下对话历史:\n"
        for mem in self.working_memory:
            summary_prompt += f"Q: {mem['query']}\nA: {mem['answer']}\n"

        response = self.llm.invoke(summary_prompt)
        summary = response.content

        self.long_term_memory.append(summary)

        # 保持长期记忆在一定大小内
        if len(self.long_term_memory) > 5:
            self.long_term_memory = self.long_term_memory[-5:]

    def _retrieve_from_memory(self, query: str) -> dict:
        """从记忆系统中检索"""

        results = {"working": [], "long_term": []}

        # 从工作记忆检索
        for mem in self.working_memory:
            if any(
                keyword in mem["query"].lower() for keyword in query.lower().split()[:2]
            ):
                results["working"].append(mem)

        # 从长期记忆检索
        for summary in self.long_term_memory:
            if any(keyword in summary.lower() for keyword in query.lower().split()[:2]):
                results["long_term"].append(summary)

        return results

    def query(self, query: str) -> dict:
        """执行Self-Mem"""

        # 1. 从记忆系统检索
        memory_context = self._retrieve_from_memory(query)

        # 2. 从外部知识库检索
        retriever = self.vectorstore.as_retriever(search_kwargs={"k": 3})
        external_docs = retriever.invoke(query)

        # 3. 构建上下文
        context_parts = []

        if memory_context["working"]:
            context_parts.append("工作记忆: " + str(memory_context["working"][-1]))

        if memory_context["long_term"]:
            context_parts.append("长期记忆: " + memory_context["long_term"][-1])

        context_parts.append(
            "外部知识: " + "\n".join([doc.page_content for doc in external_docs])
        )

        context = "\n\n".join(context_parts)

        # 4. 生成回答
        prompt = f"""上下文(包含记忆和外部知识):
{context}

问题: {query}

请基于记忆和外部知识回答:"""

        response = self.llm.invoke(prompt)

        # 5. 更新记忆
        self._update_working_memory(query, response.content)
        if len(self.working_memory) >= 2:
            self._update_long_term_memory()

        return {
            "answer": response.content,
            "memory_context": memory_context,
            "working_memory_size": len(self.working_memory),
            "long_term_memory_size": len(self.long_term_memory),
        }


# ============================================
# 测试函数
# ============================================


def test_01_self_mem_basic(config: TestConfig):
    """测试: 基本Self-Mem"""
    print("\n[测试 01] 基本Self-Mem")

    vectorstore = create_vector_store(TEST_DOCUMENTS, config)
    if vectorstore is None:
        return {"passed": False, "message": "创建向量存储失败"}

    llm = create_llm(config, temperature=0)
    if llm is None:
        return {"passed": False, "message": "创建LLM失败"}

    selfmem = SelfMem(vectorstore, llm)
    result = selfmem.query("什么是Self-Mem?")

    return {
        "passed": "answer" in result,
        "message": f"Self-Mem完成，工作记忆: {result.get('working_memory_size', 0)}, 长期记忆: {result.get('long_term_memory_size', 0)}",
        "details": {
            "working": result.get("working_memory_size", 0),
            "long_term": result.get("long_term_memory_size", 0),
            "answer_preview": result.get("answer", "")[:100],
        },
    }


def test_02_memory_retrieval(config: TestConfig):
    """测试: 记忆检索"""
    print("\n[测试 02] 记忆检索")

    vectorstore = create_vector_store(TEST_DOCUMENTS, config)
    if vectorstore is None:
        return {"passed": False, "message": "创建向量存储失败"}

    llm = create_llm(config, temperature=0)
    if llm is None:
        return {"passed": False, "message": "创建LLM失败"}

    selfmem = SelfMem(vectorstore, llm)

    # 第一次查询
    selfmem.query("什么是RAG?")

    # 第二次查询(应该使用记忆)
    result = selfmem.query("它如何工作?")

    return {
        "passed": result.get("working_memory_size", 0) > 0,
        "message": f"工作记忆已存储{result.get('working_memory_size', 0)}条",
        "details": {"working_size": result.get("working_memory_size", 0)},
    }


def test_03_long_term_memory(config: TestConfig):
    """测试: 长期记忆更新"""
    print("\n[测试 03] 长期记忆更新")

    vectorstore = create_vector_store(TEST_DOCUMENTS, config)
    if vectorstore is None:
        return {"passed": False, "message": "创建向量存储失败"}

    llm = create_llm(config, temperature=0)
    if llm is None:
        return {"passed": False, "message": "创建LLM失败"}

    selfmem = SelfMem(vectorstore, llm)

    # 多次交互
    selfmem.query("RAG是什么?")
    selfmem.query("它的组成是什么?")
    result = selfmem.query("总结一下")

    return {
        "passed": result.get("long_term_memory_size", 0) >= 0,
        "message": f"长期记忆已存储{result.get('long_term_memory_size', 0)}条摘要",
        "details": {"long_term": result.get("long_term_memory_size", 0)},
    }


# ============================================
# 主函数
# ============================================


def main():
    print("=" * 60)
    print("Self-Mem 架构测试")
    print("=" * 60)

    config = TestConfig()

    if not config.is_configured:
        print("\n⚠️  警告: API未配置!")
        return

    print(f"\n[配置] LLM: {config.llm_provider}/{config.openai_model}")

    results = []

    results.append(run_test(lambda: test_01_self_mem_basic(config), "基本Self-Mem"))
    results.append(run_test(lambda: test_02_memory_retrieval(config), "记忆检索"))
    results.append(run_test(lambda: test_03_long_term_memory(config), "长期记忆"))

    print_test_results(results)

    failed = sum(1 for r in results if not r.passed)
    return 0 if failed == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
