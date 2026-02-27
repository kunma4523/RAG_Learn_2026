#!/usr/bin/env python3
"""
测试: 迭代RAG (11_iterative_rag.py)
====================================

多跳检索与生成交替执行的RAG架构。

运行: python tests/rag_architectures/11_iterative_rag.py
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
    "RAG结合检索与生成，检索器查找相关文档。",
    "Self-RAG让LLM决定何时检索，使用反思token。",
    "Agentic RAG使用AI智能体规划和使用工具。",
    "GraphRAG使用知识图谱表示实体关系。",
    "HyDE生成假设文档来改进检索。",
    "Transformer使用自注意力机制。",
    "LLM是大语言模型，在海量文本上训练。",
    "微调将预训练模型适应特定任务。",
    "Embedding将文本表示为向量。",
    "向量数据库存储embedding进行相似性搜索。",
]


# ============================================
# 迭代RAG实现
# ============================================


class IterativeRAG:
    """迭代RAG，多跳检索"""

    def __init__(self, vectorstore, llm, max_iterations: int = 3):
        self.vectorstore = vectorstore
        self.llm = llm
        self.max_iterations = max_iterations
        self.iteration_log = []

    def _decompose_query(self, query: str, context: str = "") -> str:
        """分解复杂查询"""
        prompt = f"""{"之前上下文: " + context if context else ""}

问题: {query}

如需要，重写为更具体的子问题来检索更多信息:"""
        response = self.llm.invoke(prompt)
        return response.content.strip()

    def _generate_partial_answer(self, query: str, docs: list) -> str:
        """从检索文档生成部分答案"""
        context = "\n".join([d.page_content for d in docs])
        prompt = f"上下文:\n{context}\n\n根据此回答: {query}\n\n部分答案:"
        response = self.llm.invoke(prompt)
        return response.content

    def _should_continue(self, query: str, answer: str) -> bool:
        """判断是否继续迭代"""
        prompt = f"问题: {query}\n\n当前答案: {answer}\n\n这是否完整? YES 或 NO:"
        response = self.llm.invoke(prompt)
        return "NO" in response.content.upper()

    def query(self, query: str) -> dict:
        """执行迭代RAG"""

        current_context = ""

        for i in range(self.max_iterations):
            # 分解/子问题
            sub_query = self._decompose_query(query, current_context)

            # 检索
            retriever = self.vectorstore.as_retriever(search_kwargs={"k": 3})
            docs = retriever.invoke(sub_query)

            # 生成部分答案
            partial = self._generate_partial_answer(query, docs)
            current_context += f"\n\n迭代 {i + 1}: {partial}"

            self.iteration_log.append(
                {
                    "iteration": i + 1,
                    "sub_query": sub_query,
                    "num_docs": len(docs),
                    "partial_answer": partial[:100],
                }
            )

            # 判断是否继续
            if (
                not self._should_continue(query, partial)
                or i == self.max_iterations - 1
            ):
                break

        # 最终答案
        final_prompt = f"完整答案:\n{current_context}\n\n原始问题: {query}\n\n最终答案:"
        final_response = self.llm.invoke(final_prompt)

        return {
            "answer": final_response.content,
            "iterations": self.iteration_log,
            "num_iterations": len(self.iteration_log),
        }


# ============================================
# 测试函数
# ============================================


def test_01_iterative_basic(config):
    print("\n[测试 01] 基本迭代RAG")
    vectorstore = create_vector_store(TEST_DOCUMENTS, config)
    llm = create_llm(config, temperature=0)

    if not vectorstore or not llm:
        return {"passed": False, "message": "组件初始化失败"}

    rag = IterativeRAG(vectorstore, llm)
    result = rag.query("什么是RAG?")

    return {
        "passed": "answer" in result,
        "message": f"完成 {result.get('num_iterations', 0)} 次迭代",
        "details": {"iterations": result.get("num_iterations", 0)},
    }


def test_02_multi_hop(config):
    print("\n[测试 02] 多跳检索")
    vectorstore = create_vector_store(TEST_DOCUMENTS, config)
    llm = create_llm(config, temperature=0)

    if not vectorstore or not llm:
        return {"passed": False, "message": "组件初始化失败"}

    rag = IterativeRAG(vectorstore, llm, max_iterations=2)
    result = rag.query("Self-RAG和Transformer什么关系?")

    return {
        "passed": result.get("num_iterations", 0) > 0,
        "message": f"多跳: {result.get('num_iterations', 0)} 跳",
        "details": {"hops": result.get("num_iterations", 0)},
    }


def test_03_iteration_log(config):
    print("\n[测试 03] 迭代日志")
    vectorstore = create_vector_store(TEST_DOCUMENTS, config)
    llm = create_llm(config, temperature=0)

    if not vectorstore or not llm:
        return {"passed": False, "message": "组件初始化失败"}

    rag = IterativeRAG(vectorstore, llm)
    result = rag.query("比较RAG架构")

    return {
        "passed": len(result.get("iterations", [])) > 0,
        "message": f"记录 {len(result.get('iterations', []))} 次迭代",
        "details": {"log": result.get("iterations", [])},
    }


def main():
    print("=" * 60)
    print("迭代RAG 架构测试")
    print("=" * 60)

    config = TestConfig()
    if not config.is_configured:
        print("\n⚠️  警告: API未配置!")
        return

    results = []
    results.append(run_test(lambda: test_01_iterative_basic(config), "基本迭代"))
    results.append(run_test(lambda: test_02_multi_hop(config), "多跳"))
    results.append(run_test(lambda: test_03_iteration_log(config), "迭代日志"))

    print_test_results(results)
    return 0 if sum(1 for r in results if not r.passed) == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
