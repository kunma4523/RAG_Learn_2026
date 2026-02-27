#!/usr/bin/env python3
"""
测试: REPLUG (10_replug.py)
===========================

检索插件架构，检索器与LLM解耦，使用加权输入。

运行: python tests/rag_architectures/10_replug.py
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
    "Agentic RAG使用智能体规划。",
    "GraphRAG使用图谱检索。",
    "HyDE生成假设文档。",
    "融合RAG组合多种方法。",
    "BM25是关键词检索。",
    "密集检索用embedding。",
    "Transformer注意力机制。",
    "向量数据库支持搜索。",
]


# ============================================
# REPLUG实现
# ============================================


class REPLUG:
    """检索插件，解耦检索器和LLM"""

    def __init__(self, vectorstore, llm):
        self.vectorstore = vectorstore
        self.llm = llm
        self.retriever_weights = {}

    def _retrieve_with_strategy(self, query: str, strategy: str, top_k: int = 3):
        """不同策略检索"""

        if strategy == "similarity":
            retriever = self.vectorstore.as_retriever(
                search_type="similarity", search_kwargs={"k": top_k}
            )
        elif strategy == "mmr":  # 最大边际相关性
            retriever = self.vectorstore.as_retriever(
                search_type="mmr", search_kwargs={"k": top_k, "fetch_k": 10}
            )
        else:
            retriever = self.vectorstore.as_retriever(search_kwargs={"k": top_k})

        return retriever.invoke(query)

    def _weight_documents(self, query: str, docs_per_strategy: dict) -> list:
        """加权合并文档"""

        all_docs = {}

        for strategy, docs in docs_per_strategy.items():
            for i, doc in enumerate(docs):
                key = doc.page_content[:50]
                if key not in all_docs:
                    all_docs[key] = {"doc": doc, "strategies": [], "total_score": 0}
                all_docs[key]["strategies"].append(strategy)
                all_docs[key]["total_score"] += (top_k - i) / top_k

        # LLM重排序
        doc_texts = "\n".join(
            [
                f"{i + 1}. {d['doc'].page_content[:100]}... (来自 {', '.join(d['strategies'])})"
                for i, d in enumerate(list(all_docs.values())[:5])
            ]
        )

        prompt = f"""查询: "{query}"

按相关性排序这些文档(1 = 最相关):
{doc_texts}

请按顺序回复编号:"""

        try:
            response = self.llm.invoke(prompt)
            ranking = [int(x.strip()) for x in response.content.split(",")]

            weighted_docs = []
            for idx in ranking:
                if idx - 1 < len(list(all_docs.values())):
                    weighted_docs.append(list(all_docs.values())[idx - 1]["doc"])

            return weighted_docs[:5]
        except:
            return list(all_docs.values())[:5]

    def query(self, query: str, strategies: list = None) -> dict:
        """执行REPLUG"""

        if strategies is None:
            strategies = ["similarity", "mmr"]

        # 每个策略检索
        docs_per_strategy = {}
        for strategy in strategies:
            docs = self._retrieve_with_strategy(query, strategy)
            docs_per_strategy[strategy] = docs

        # 加权合并
        weighted_docs = self._weight_documents(query, docs_per_strategy)

        # 生成
        context = "\n\n".join([doc.page_content for doc in weighted_docs])

        prompt = f"""上下文(多策略检索):
{context}

问题: {query}

答案:"""

        response = self.llm.invoke(prompt)

        return {
            "answer": response.content,
            "docs_per_strategy": {k: len(v) for k, v in docs_per_strategy.items()},
            "weighted_docs": [doc.page_content for doc in weighted_docs],
            "num_final_docs": len(weighted_docs),
        }


# ============================================
# 测试函数
# ============================================


def test_01_replug_basic(config):
    print("\n[测试 01] 基本REPLUG")
    vectorstore = create_vector_store(TEST_DOCUMENTS, config)
    llm = create_llm(config, temperature=0)

    if not vectorstore or not llm:
        return {"passed": False, "message": "组件初始化失败"}

    replug = REPLUG(vectorstore, llm)
    result = replug.query("什么是RAG?")

    return {
        "passed": "answer" in result,
        "message": f"使用 {len(result.get('docs_per_strategy', {}))} 种策略",
        "details": {"strategies": list(result.get("docs_per_strategy", {}).keys())},
    }


def test_02_multiple_retrievers(config):
    print("\n[测试 02] 多检索器")
    vectorstore = create_vector_store(TEST_DOCUMENTS, config)
    llm = create_llm(config, temperature=0)

    if not vectorstore or not llm:
        return {"passed": False, "message": "组件初始化失败"}

    replug = REPLUG(vectorstore, llm)
    result = replug.query("检索如何工作?")

    has_multiple = len(result.get("docs_per_strategy", {})) >= 2

    return {
        "passed": has_multiple,
        "message": f"使用策略: {result.get('docs_per_strategy', {})}",
        "details": result.get("docs_per_strategy", {}),
    }


def test_03_weighting(config):
    print("\n[测试 03] 文档加权")
    vectorstore = create_vector_store(TEST_DOCUMENTS, config)
    llm = create_llm(config, temperature=0)

    if not vectorstore or not llm:
        return {"passed": False, "message": "组件初始化失败"}

    replug = REPLUG(vectorstore, llm)
    result = replug.query("比较RAG架构")

    return {
        "passed": len(result.get("weighted_docs", [])) > 0,
        "message": f"加权选择 {len(result.get('weighted_docs', []))} 个文档",
        "details": {"num_weighted": len(result.get("weighted_docs", []))},
    }


def main():
    print("=" * 60)
    print("REPLUG 架构测试")
    print("=" * 60)

    config = TestConfig()
    if not config.is_configured:
        print("\n⚠️  警告: API未配置!")
        return

    print(f"\n[配置] LLM: {config.llm_provider}/{config.openai_model}")

    results = []
    results.append(run_test(lambda: test_01_replug_basic(config), "基本REPLUG"))
    results.append(run_test(lambda: test_02_multiple_retrievers(config), "多检索器"))
    results.append(run_test(lambda: test_03_weighting(config), "文档加权"))

    print_test_results(results)
    return 0 if sum(1 for r in results if not r.passed) == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
