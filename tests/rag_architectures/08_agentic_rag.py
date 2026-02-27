#!/usr/bin/env python3
"""
测试: 智能体RAG (08_agentic_rag.py)
====================================

结合AI智能体的RAG，可以规划、工具调用和动态适应检索策略。

运行: python tests/rag_architectures/08_agentic_rag.py
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
    "RAG结合检索与生成，减少幻觉。",
    "Self-RAG让LLM决定何时检索。",
    "Agentic RAG使用AI智能体规划检索。",
    "GraphRAG使用知识图谱检索。",
    "HyDE生成假设文档改进检索。",
    "融合RAG结合多种检索方法。",
    "BM25是关键词检索方法。",
    "密集检索使用embedding。",
    "向量数据库支持相似性搜索。",
    "Transformer使用注意力机制。",
]


# ============================================
# 智能体RAG实现
# ============================================


class AgenticRAG:
    """带工具使用和规划的智能体RAG"""

    def __init__(self, vectorstore, llm):
        self.vectorstore = vectorstore
        self.llm = llm
        self.action_log = []

    def _plan(self, query: str) -> list:
        """规划检索策略"""

        prompt = f"""分解这个查询为步骤:

查询: "{query}"

你需要哪些信息来回答? 列出2-3个具体搜索方面:"""

        response = self.llm.invoke(prompt)
        steps = [s.strip() for s in response.content.split("\n") if s.strip()]

        self.action_log.append({"action": "plan", "steps": steps})

        return steps[:3]

    def _search(self, search_query: str) -> list:
        """执行搜索"""

        retriever = self.vectorstore.as_retriever(search_kwargs={"k": 3})
        docs = retriever.invoke(search_query)

        self.action_log.append(
            {"action": "search", "query": search_query, "num_docs": len(docs)}
        )

        return docs

    def _reflect(self, query: str, docs: list) -> dict:
        """评估检索结果"""

        doc_texts = "\n".join([f"- {doc.page_content[:100]}..." for doc in docs])

        prompt = f"""查询: "{query}"

检索到的文档:
{doc_texts}

这些信息是否足以回答查询?
请回复 YES 或 NO 并简要说明:"""

        response = self.llm.invoke(prompt)

        sufficient = "YES" in response.content.upper()

        self.action_log.append({"action": "reflect", "sufficient": sufficient})

        return {"sufficient": sufficient, "feedback": response.content}

    def _rewrite(self, query: str, feedback: str) -> str:
        """根据反馈重写查询"""

        prompt = f"""原始查询: "{query}"

反馈: {feedback}

重写查询以解决问题:"""

        response = self.llm.invoke(prompt)
        new_query = response.content.strip()

        self.action_log.append({"action": "rewrite", "new_query": new_query})

        return new_query

    def query(self, query: str, max_iterations: int = 3) -> dict:
        """执行智能体RAG"""

        self.action_log = []

        # 规划
        steps = self._plan(query)

        all_docs = []

        # 执行计划步骤
        for i, step in enumerate(steps):
            if i >= max_iterations:
                break

            # 搜索
            docs = self._search(step)
            all_docs.extend(docs)

            # 反思
            evaluation = self._reflect(query, docs)

            if not evaluation["sufficient"] and i < len(steps) - 1:
                # 重写并继续
                next_step = self._rewrite(
                    steps[i + 1] if i + 1 < len(steps) else query,
                    evaluation["feedback"],
                )

        # 去重
        seen = set()
        unique_docs = []
        for doc in all_docs:
            if doc.page_content not in seen:
                seen.add(doc.page_content)
                unique_docs.append(doc)

        # 生成最终答案
        context = "\n\n".join([doc.page_content for doc in unique_docs[:5]])

        prompt = f"""基于检索到的信息:

{context}

问题: {query}

请给出全面的答案:"""

        response = self.llm.invoke(prompt)

        return {
            "answer": response.content,
            "action_log": self.action_log,
            "num_docs_retrieved": len(unique_docs),
            "steps_executed": len(steps),
        }


# ============================================
# 测试函数
# ============================================


def test_01_agentic_planning(config: TestConfig):
    """测试: 智能体规划能力"""
    print("\n[测试 01] 智能体规划")

    vectorstore = create_vector_store(TEST_DOCUMENTS, config)
    if vectorstore is None:
        return {"passed": False, "message": "创建向量存储失败"}

    llm = create_llm(config, temperature=0)
    if llm is None:
        return {"passed": False, "message": "创建LLM失败"}

    agent = AgenticRAG(vectorstore, llm)
    result = agent.query("什么是RAG，它是如何工作的?")

    has_planning = any(
        log.get("action") == "plan" for log in result.get("action_log", [])
    )

    return {
        "passed": has_planning,
        "message": f"智能体规划了 {result.get('steps_executed', 0)} 个步骤",
        "details": {
            "action_log": [log.get("action") for log in result.get("action_log", [])],
            "num_docs": result.get("num_docs_retrieved", 0),
        },
    }


def test_02_tool_execution(config: TestConfig):
    """测试: 多个工具执行"""
    print("\n[测试 02] 工具执行")

    vectorstore = create_vector_store(TEST_DOCUMENTS, config)
    if vectorstore is None:
        return {"passed": False, "message": "创建向量存储失败"}

    llm = create_llm(config, temperature=0)
    if llm is None:
        return {"passed": False, "message": "创建LLM失败"}

    agent = AgenticRAG(vectorstore, llm)
    result = agent.query("比较RAG架构")

    actions = [log.get("action") for log in result.get("action_log", [])]

    return {
        "passed": len(actions) >= 2,
        "message": f"执行了 {len(actions)} 个动作: {actions}",
        "details": {"actions": actions},
    }


def test_03_reflection_loop(config: TestConfig):
    """测试: 反思循环"""
    print("\n[测试 03] 反思循环")

    vectorstore = create_vector_store(TEST_DOCUMENTS, config)
    if vectorstore is None:
        return {"passed": False, "message": "创建向量存储失败"}

    llm = create_llm(config, temperature=0)
    if llm is None:
        return {"passed": False, "message": "创建LLM失败"}

    agent = AgenticRAG(vectorstore, llm)
    result = agent.query("RAG有哪些类型?")

    has_reflect = any(
        log.get("action") == "reflect" for log in result.get("action_log", [])
    )

    return {
        "passed": has_reflect,
        "message": f"反思执行: {has_reflect}",
        "details": {
            "reflection_count": sum(
                1
                for log in result.get("action_log", [])
                if log.get("action") == "reflect"
            )
        },
    }


# ============================================
# 主函数
# ============================================


def main():
    print("=" * 60)
    print("智能体RAG架构测试")
    print("=" * 60)

    config = TestConfig()

    if not config.is_configured:
        print("\n⚠️  警告: API未配置!")
        return

    print(f"\n[配置] LLM: {config.llm_provider}/{config.openai_model}")

    results = []

    results.append(run_test(lambda: test_01_agentic_planning(config), "智能体规划"))
    results.append(run_test(lambda: test_02_tool_execution(config), "工具执行"))
    results.append(run_test(lambda: test_03_reflection_loop(config), "反思循环"))

    print_test_results(results)

    failed = sum(1 for r in results if not r.passed)
    return 0 if failed == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
