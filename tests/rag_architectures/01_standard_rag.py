#!/usr/bin/env python3
"""
测试: 标准RAG (01_standard_rag.py)
====================================

基础RAG流程测试: 查询 → 检索 → 增强 → 生成

运行: python tests/rag_architectures/01_standard_rag.py

架构:
    ┌──────────┐    ┌─────────────┐    ┌──────────────┐    ┌──────────────┐
    │   Query  │───▶│  Retrieval  │───▶│ Augmentation │───▶│  Generation  │
    └──────────┘    └─────────────┘    └──────────────┘    └──────────────┘
                                                              │
                                                              ▼
                                                     ┌──────────────┐
                                                     │    Answer    │
                                                     └──────────────┘
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
    "RAG(检索增强生成)将信息检索与文本生成相结合，通过从外部知识库检索相关信息来增强大语言模型的能力。",
    "RAG系统主要包括三个组件：检索器负责从知识库中查找相关文档，增强器对检索结果进行排序优化，生成器基于增强后的上下文生成最终回答。",
    "稀疏检索方法如BM25使用关键词匹配，适合精确查找。密集检索方法如DPR使用神经网络的embedding来理解语义相似性。",
    "混合检索结合稀疏检索和密集检索的优点，提供更好的检索效果。",
    "Self-RAG是一种自反思的检索增强生成框架，让LLM能够自主决定何时检索以及如何使用检索到的内容。",
    "Agentic RAG结合AI智能体，可以规划检索策略、决定搜索时机、动态使用多种工具。",
    "GraphRAG使用知识图谱来表示实体关系，实现更复杂的关联信息检索。",
    "RAG的评估指标包括：检索指标(Recall@K, Precision@K, MRR, NDCG)和生成指标(BLEU, ROUGE, BERTScore)。",
    "HyDE(假设文档Embedding)先生成假设回答文档，然后用它来检索真实的相关文档。",
    "Fusion RAG结合多种检索方法(稀疏、密集、图谱)，通过重排序来融合结果，提升召回率和精确率。",
    "大语言模型可能产生幻觉或不准确的回答，RAG通过检索事实依据来减少这一问题。"
]


# ============================================
# 测试函数
# ============================================

def test_01_basic_rag_pipeline(config: TestConfig):
    """测试: 基本的RAG流程，包含检索和生成"""
    print("\n[测试 01] 基本的RAG流程")

    # 创建向量存储
    vectorstore = create_vector_store(TEST_DOCUMENTS, config)
    if vectorstore is None:
        return {"passed": False, "message": "创建向量存储失败"}

    # 创建检索器
    retriever = vectorstore.as_retriever(search_kwargs={"k": 3})

    # 测试检索
    query = "什么是RAG?"
    retrieved_docs = retriever.invoke(query)

    if not retrieved_docs:
        return {"passed": False, "message": "未检索到文档"}

    # 创建LLM
    llm = create_llm(config, temperature=0)
    if llm is None:
        return {"passed": False, "message": "创建LLM失败"}

    # 创建提示词并生成
    from langchain_core.prompts import ChatPromptTemplate

    context = "\n\n".join([doc.page_content for doc in retrieved_docs])

    prompt = ChatPromptTemplate.from_template(
        "上下文:\n{context}\n\n问题: {question}\n\n请根据以上上下文回答:"
    )

    chain = prompt | llm
    response = chain.invoke({"context": context, "question": query})

    return {
        "passed": True,
        "message": f"检索到 {len(retrieved_docs)} 个文档，生成回答成功",
        "details": {
            "query": query,
            "num_retrieved": len(retrieved_docs),
            "response": response.content[:200],
        },
    }


def test_02_different_queries(config: TestConfig):
    """测试: 多个不同的查询"""
    print("\n[测试 02] 不同类型的查询")

    vectorstore = create_vector_store(TEST_DOCUMENTS, config)
    if vectorstore is None:
        return {"passed": False, "message": "创建向量存储失败"}

    retriever = vectorstore.as_retriever(search_kwargs={"k": 3})
    llm = create_llm(config, temperature=0)

    if llm is None:
        return {"passed": False, "message": "创建LLM失败"}

    queries = [
        "RAG的主要组成部分是什么?",
        "Self-RAG是如何工作的?",
        "什么是Agentic RAG?"
    ]

    results = []
    from langchain_core.prompts import ChatPromptTemplate

    for query in queries:
        retrieved_docs = retriever.invoke(query)
        context = "\n\n".join([doc.page_content for doc in retrieved_docs])

        prompt = ChatPromptTemplate.from_template(
            "上下文:\n{context}\n\n问题: {question}\n\n回答:"
        )

        chain = prompt | llm
        response = chain.invoke({"context": context, "question": query})
        results.append({"query": query, "response": response.content[:100]})

    return {
        "passed": len(results) == len(queries),
        "message": f"成功处理 {len(results)} 个查询",
        "details": {"results": results},
    }


def test_03_top_k_variation(config: TestConfig):
    """测试: 不同的top_k值"""
    print("\n[测试 03] Top-K 参数变化")

    vectorstore = create_vector_store(TEST_DOCUMENTS, config)
    if vectorstore is None:
        return {"passed": False, "message": "创建向量存储失败"}

    query = "什么是RAG?"

    for k in [1, 3, 5]:
        retriever = vectorstore.as_retriever(search_kwargs={"k": k})
        docs = retriever.invoke(query)

        if len(docs) != k:
            return {"passed": False, "message": f"期望 {k} 个文档，实际获取 {len(docs)}"}

    return {
        "passed": True,
        "message": "Top-k 检索功能正常": {"tested",
        "details_k": [1, 3, 5]},
    }


# ============================================
# 主测试运行器
# ============================================

def main():
    """运行所有标准RAG测试"""
    print("=" * 60)
    print("标准RAG架构测试")
    print("=" * 60)

    # 加载配置
    config = TestConfig()

    if not config.is_configured:
        print("\n⚠️  警告: API未配置!")
        print("请在.env文件中设置环境变量:")
        print("  - LLM_PROVIDER=openai")
        print("  - OPENAI_API_KEY=your-key-here")
        print("\n或使用mock数据进行开发调试。")
        return

    print(f"\n[配置] LLM: {config.llm_provider}/{config.openai_model}")
    print(f"[配置] Embedding: {config.embedding_provider}")

    # 运行测试
    results = []

    results.append(
        run_test(lambda: test_01_basic_rag_pipeline(config), "基本RAG流程")
    )

    results.append(
        run_test(lambda: test_02_different_queries(config), "不同查询类型")
    )

    results.append(run_test(lambda: test_03_top_k_variation(config), "Top-K参数变化"))

    # 打印结果
    print_test_results(results)

    # 返回退出码
    failed = sum(1 for r in results if not r.passed)
    return 0 if failed == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
