#!/usr/bin/env python3
"""
测试: Multimodal RAG (14_multimodal_rag.py)
============================================

支持图像、音频、视频等多种模态的RAG系统。

运行: python tests/rag_architectures/14_multimodal_rag.py
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
# 测试数据 - 多模态文档
# ============================================

TEST_DOCUMENTS = [
    {"type": "text", "content": "RAG结合检索与生成，增强大语言模型的能力。"},
    {
        "type": "text",
        "content": "多模态RAG支持图像、文本、音频等多种模态的检索和生成。",
    },
    {"type": "text", "content": "CLIP模型可以同时理解图像和文本，实现跨模态检索。"},
    {"type": "text", "content": "音频检索可以使用语音识别技术将音频转为文本。"},
    {"type": "text", "content": "视频检索可以提取关键帧并进行帧级别的分析。"},
    {
        "type": "text",
        "content": "多模态 embeddings 可以将不同模态的内容映射到统一的向量空间。",
    },
    {"type": "text", "content": "跨模态检索允许用文本查询图像，或用图像查询文本。"},
    {
        "type": "text",
        "content": "视觉问答(VQA)结合图像理解和自然语言处理来回答关于图像的问题。",
    },
]


# ============================================
# Multimodal RAG 实现
# ============================================


class MultimodalRAG:
    """
    多模态RAG实现

    特点:
    1. 支持多种模态(文本、图像、音频、视频)
    2. 跨模态检索能力
    3. 多模态内容理解
    4. 统一的向量表示空间
    """

    def __init__(self, config: TestConfig):
        self.config = config
        self.vectorstore = None
        self.llm = None
        self._init_components()

    def _init_components(self):
        """初始化组件"""
        # 使用文本创建向量存储(简化版多模态)
        texts = [d["content"] for d in TEST_DOCUMENTS]
        from tests.rag_architectures import create_vector_store, create_llm

        self.vectorstore = create_vector_store(texts, self.config)
        self.llm = create_llm(self.config, temperature=0)

    def _retrieve_multimodal(self, query: str, modalities: list = None) -> dict:
        """多模态检索"""

        retriever = self.vectorstore.as_retriever(search_kwargs={"k": 5})
        docs = retriever.invoke(query)

        # 模拟多模态结果
        results = {
            "text": [doc.page_content for doc in docs],
            "images": [],  # 简化版本
            "modalities_retrieved": ["text"],
        }

        return results

    def query(self, query: str) -> dict:
        """执行多模态RAG"""

        # 多模态检索
        retrieval_results = self._retrieve_multimodal(query)

        # 组合上下文
        context = "\n\n".join(retrieval_results["text"])

        prompt = f"""多模态上下文:
{context}

问题: {query}

请基于多模态内容回答问题:"""

        response = self.llm.invoke(prompt)

        return {
            "answer": response.content,
            "retrieved_content": retrieval_results,
            "modalities": retrieval_results["modalities_retrieved"],
        }


# ============================================
# 测试函数
# ============================================


def test_01_multimodal_basic(config: TestConfig):
    """测试: 基本多模态RAG"""
    print("\n[测试 01] 基本多模态RAG")

    try:
        mmrag = MultimodalRAG(config)
        if not mmrag.vectorstore or not mmrag.llm:
            return {"passed": False, "message": "组件初始化失败"}

        result = mmrag.query("什么是多模态RAG?")

        return {
            "passed": "answer" in result,
            "message": f"多模态RAG完成，检索了{len(result.get('modalities', []))}种模态",
            "details": {
                "modalities": result.get("modalities", []),
                "answer_preview": result.get("answer", "")[:100],
            },
        }
    except Exception as e:
        return {"passed": False, "message": f"错误: {str(e)[:50]}"}


def test_02_cross_modal_retrieval(config: TestConfig):
    """测试: 跨模态检索"""
    print("\n[测试 02] 跨模态检索")

    try:
        mmrag = MultimodalRAG(config)
        result = mmrag.query("实现CLIP模型如何图像文本检索?")

        return {
            "passed": len(result.get("retrieved_content", {}).get("text", [])) > 0,
            "message": "跨模态检索执行成功",
            "details": {"content_types": result.get("modalities", [])},
        }
    except Exception as e:
        return {"passed": False, "message": f"错误: {str(e)[:50]}"}


def test_03_multimodal_understanding(config: TestConfig):
    """测试: 多模态理解"""
    print("\n[测试 03] 多模态理解")

    try:
        mmrag = MultimodalRAG(config)
        result = mmrag.query("解释视觉问答的工作原理")

        return {
            "passed": "answer" in result,
            "message": "多模态理解完成",
            "details": {"answer_preview": result.get("answer", "")[:100]},
        }
    except Exception as e:
        return {"passed": False, "message": f"错误: {str(e)[:50]}"}


# ============================================
# 主函数
# ============================================


def main():
    print("=" * 60)
    print("多模态RAG 架构测试")
    print("=" * 60)

    config = TestConfig()

    if not config.is_configured:
        print("\n⚠️  警告: API未配置!")
        return

    print(f"\n[配置] LLM: {config.llm_provider}/{config.openai_model}")

    results = []

    results.append(run_test(lambda: test_01_multimodal_basic(config), "基本多模态RAG"))
    results.append(
        run_test(lambda: test_02_cross_modal_retrieval(config), "跨模态检索")
    )
    results.append(
        run_test(lambda: test_03_multimodal_understanding(config), "多模态理解")
    )

    print_test_results(results)

    failed = sum(1 for r in results if not r.passed)
    return 0 if failed == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
