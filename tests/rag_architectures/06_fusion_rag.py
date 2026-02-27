#!/usr/bin/env python3
"""
测试: 融合RAG (06_fusion_rag.py)
================================

结合多种检索方法(稀疏、密集、图谱)并使用重排序融合结果。

运行: python tests/rag_architectures/06_fusion_rag.py

架构:
    ┌────────────┐
    │   查询      │
    └────────────┘
          │
    ┌─────┼─────┬──────────┐
    ▼     ▼     ▼          ▼
 ┌────┐ ┌────┐ ┌────┐  ┌─────┐
 │BM25│ │密集│ │图谱│  │...  │
 └────┘ └────┘ └────┘  └─────┘
    │     │     │        │
    └─────┼─────┼────────┘
          ▼
   ┌───────────┐
   │  重排序   │
   │  (融合)   │
   └───────────┘
          │
          ▼
   ┌───────────┐    ┌──────────────┐
   │  生成     │───▶│    答案      │
   └───────────┘    └──────────────┘
"""

import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

from tests.rag_architectures import TestConfig, run_test, print_test_results


# ============================================
# 测试文档
# ============================================

TEST_DOCUMENTS = [
    {"id": "doc1", "text": "RAG结合检索与生成，检索器从知识库中查找相关文档。"},
    {"id": "doc2", "text": "Self-RAG是自反思框架，LLM决定何时检索。"},
    {"id": "doc3", "text": "Agentic RAG使用AI智能体规划检索策略。"},
    {"id": "doc4", "text": "GraphRAG使用知识图谱进行实体关系检索。"},
    {"id": "doc5", "text": "HyDE生成假设文档来改进检索。"},
    {"id": "doc6", "text": "BM25是基于词频的稀疏检索方法。"},
    {"id": "doc7", "text": "密集检索使用神经网络的embedding进行语义相似性搜索。"},
    {"id": "doc8", "text": "向量数据库存储embeddings用于相似性搜索。"},
    {"id": "doc9", "text": "融合RAG结合多种检索方法并重排序融合结果。"},
    {"id": "doc10", "text": "Transformer的注意力机制让模型关注相关上下文。"},
]


# ============================================
# 融合RAG实现
# ============================================


class FusionRAG:
    """融合RAG，组合多种检索方法"""

    def __init__(self, config: TestConfig):
        self.config = config
        self.vectorstore = None
        self.llm = None
        self._init_components()

    def _init_components(self):
        """初始化组件"""
        from tests.rag_architectures import create_vector_store, create_llm

        texts = [d["text"] for d in TEST_DOCUMENTS]
        self.vectorstore = create_vector_store(texts, self.config)
        self.llm = create_llm(self.config, temperature=0)

    def _sparse_retrieve(self, query: str, top_k: int = 3) -> list:
        """模拟稀疏检索(基于关键词)"""
        from langchain_community.retrievers import BM25Retriever
        from langchain.schema import Document

        docs = [
            Document(page_content=d["text"], metadata={"id": d["id"]})
            for d in TEST_DOCUMENTS
        ]

        try:
            retriever = BM25Retriever.from_documents(docs)
            retriever.k = top_k
            return retriever.invoke(query)
        except:
            return self.vectorstore.as_retriever(search_kwargs={"k": top_k}).invoke(
                query
            )

    def _dense_retrieve(self, query: str, top_k: int = 3) -> list:
        """密集语义检索"""
        return self.vectorstore.as_retriever(search_kwargs={"k": top_k}).invoke(query)

    def _rerank_fuse(self, query: str, sparse_docs: list, dense_docs: list) -> list:
        """重排序融合结果"""

        all_docs = {}
        for doc in sparse_docs + dense_docs:
            content = doc.page_content
            if content not in all_docs:
                all_docs[content] = {"doc": doc, "sparse_score": 0, "dense_score": 0}

        for doc in sparse_docs:
            all_docs[doc.page_content]["sparse_score"] += 1

        for doc in dense_docs:
            all_docs[doc.page_content]["dense_score"] += 1

        doc_list = "\n".join(
            [
                f"{i + 1}. {d['doc'].page_content}"
                for i, d in enumerate(all_docs.values())
            ]
        )

        rerank_prompt = f"""给定查询: "{query}"

按相关性排序这些文档(1 = 最相关):
{doc_list}

请按顺序回复文档编号(如 "3, 1, 5"):"""

        response = self.llm.invoke(rerank_prompt)

        try:
            ranking = [int(x.strip()) for x in response.content.split(",")]
            ranked_docs = []
            for idx in ranking[:5]:
                if idx - 1 < len(list(all_docs.values())):
                    ranked_docs.append(list(all_docs.values())[idx - 1]["doc"])
            return ranked_docs
        except:
            return dense_docs[:3]

    def query(self, query: str) -> dict:
        """执行融合RAG"""

        # 并行检索
        sparse_docs = self._sparse_retrieve(query)
        dense_docs = self._dense_retrieve(query)

        # 融合重排序
        fused_docs = self._rerank_fuse(query, sparse_docs, dense_docs)

        # 生成答案
        context = "\n\n".join([doc.page_content for doc in fused_docs])

        prompt = f"""上下文:
{context}

问题: {query}

回答:"""

        response = self.llm.invoke(prompt)

        return {
            "answer": response.content,
            "fused_docs": [doc.page_content for doc in fused_docs],
            "num_sparse": len(sparse_docs),
            "num_dense": len(dense_docs),
            "num_fused": len(fused_docs),
        }


# ============================================
# 测试函数
# ============================================


def test_01_fusion_rag_basic(config: TestConfig):
    """测试: 基本融合RAG"""
    print("\n[测试 01] 基本融合RAG")

    try:
        rag = FusionRAG(config)
        if rag.vectorstore is None or rag.llm is None:
            return {"passed": False, "message": "组件初始化失败"}

        result = rag.query("什么是RAG?")

        return {
            "passed": "answer" in result,
            "message": f"融合了 {result.get('num_fused', 0)} 个文档(稀疏+密集)",
            "details": {
                "num_sparse": result.get("num_sparse", 0),
                "num_dense": result.get("num_dense", 0),
                "num_fused": result.get("num_fused", 0),
            },
        }
    except Exception as e:
        return {"passed": False, "message": f"错误: {str(e)[:50]}"}


def test_02_sparse_dense_combination(config: TestConfig):
    """测试: 稀疏+密集组合"""
    print("\n[测试 02] 稀疏+密集组合")

    try:
        rag = FusionRAG(config)
        result = rag.query("RAG中的检索是如何工作的?")

        has_combination = (
            result.get("num_sparse", 0) > 0 and result.get("num_dense", 0) > 0
        )

        return {
            "passed": has_combination,
            "message": f"检索 {result.get('num_sparse', 0)} 个稀疏 + {result.get('num_dense', 0)} 个密集",
            "details": result,
        }
    except Exception as e:
        return {"passed": False, "message": f"错误: {str(e)[:50]}"}


def test_03_reranking(config: TestConfig):
    """测试: 重排序功能"""
    print("\n[测试 03] 重排序")

    try:
        rag = FusionRAG(config)
        result = rag.query("比较不同的RAG架构")

        return {
            "passed": len(result.get("fused_docs", [])) > 0,
            "message": f"重排序并融合了 {len(result.get('fused_docs', []))} 个文档",
            "details": {"num_fused": len(result.get("fused_docs", []))},
        }
    except Exception as e:
        return {"passed": False, "message": f"错误: {str(e)[:50]}"}


# ============================================
# 主函数
# ============================================


def main():
    print("=" * 60)
    print("融合RAG架构测试")
    print("=" * 60)

    config = TestConfig()

    if not config.is_configured:
        print("\n⚠️  警告: API未配置!")
        return

    print(f"\n[配置] LLM: {config.llm_provider}/{config.openai_model}")

    results = []

    results.append(run_test(lambda: test_01_fusion_rag_basic(config), "基本融合RAG"))
    results.append(
        run_test(lambda: test_02_sparse_dense_combination(config), "稀疏+密集")
    )
    results.append(run_test(lambda: test_03_reranking(config), "重排序"))

    print_test_results(results)

    failed = sum(1 for r in results if not r.passed)
    return 0 if failed == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
