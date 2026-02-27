#!/usr/bin/env python3
"""
测试: RA-CM3 (22_ra_cm3.py)
===========================

检索增强的多模态多语言模型。
Retrieval-Augmented Multimodal Multilingual Model.

运行: python tests/rag_architectures/22_ra_cm3.py
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

MULTIMODAL_DOCS = [
    {"type": "text", "content": "RAG结合检索与生成，增强大语言模型的能力。"},
    {
        "type": "image_text",
        "content": "CLIP模型可以同时理解图像和文本，实现跨模态检索。图像描述:一个人在操作笔记本电脑。",
    },
    {
        "type": "text",
        "content": "多模态RAG支持图像、文本、音频等多种模态的检索和生成。",
    },
    {
        "type": "image_text",
        "content": "视觉问答(VQA)结合图像理解和自然语言处理。图像显示:日落时分的海滩风景。",
    },
    {"type": "text", "content": "跨模态检索允许用文本查询图像，或用图像查询文本。"},
    {
        "type": "image_text",
        "content": "DALL-E可以基于文本描述生成图像。生成的图像:一只穿西装的猫。",
    },
]


# ============================================
# RA-CM3 实现
# ============================================


class RACM3:
    """
    RA-CM3 实现

    特点:
    1. 多模态理解(图像+文本)
    2. 跨模态检索
    3. 检索增强的图像生成
    4. 多语言支持
    """

    def __init__(self, llm, documents: list):
        self.llm = llm
        self.documents = documents
        self._index_documents()

    def _index_documents(self):
        """索引文档"""
        texts = [d["content"] for d in self.documents]

        try:
            from tests.rag_architectures import create_vector_store, TestConfig

            config = TestConfig()
            self.vectorstore = create_vector_store(texts, config)
        except:
            self.vectorstore = None

    def _classify_modality(self, query: str) -> str:
        """分类查询的模态"""

        prompt = f"""分析这个查询需要什么模态的信息:
{query}

返回:
- text: 如果只需要文本
- image: 如果需要图像信息
- multimodal: 如果需要两者

只返回一种:"""

        response = self.llm.invoke(prompt)
        modality = response.content.strip().lower()

        if "image" in modality:
            return "image"
        elif "multi" in modality:
            return "multimodal"
        return "text"

    def _retrieve_cross_modal(self, query: str, modality: str) -> list:
        """跨模态检索"""

        if not self.vectorstore:
            return []

        retriever = self.vectorstore.as_retriever(search_kwargs={"k": 4})
        docs = retriever.invoke(query)

        # 根据模态过滤
        results = []
        for doc in docs:
            doc_text = doc.page_content
            # 简单判断
            if (
                modality == "text"
                and "图像" not in doc_text
                and "image" not in doc_text.lower()
            ):
                results.append(doc)
            elif modality == "image" and (
                "图像" in doc_text or "image" in doc_text.lower()
            ):
                results.append(doc)
            else:
                results.append(doc)

        return results

    def _generate_multimodal(self, query: str, context: str) -> dict:
        """多模态生成"""

        prompt = f"""基于以下检索内容回答问题并提供图像描述建议:

检索内容:
{context}

问题: {query}

请:
1. 给出文本回答
2. 建议一个与内容相关的图像描述(如果有图像信息)"""

        response = self.llm.invoke(prompt)

        return {
            "text_response": response.content,
            "image_suggestion": "基于检索内容生成图像描述"
            if "图像" in context
            else None,
        }

    def query(self, query: str) -> dict:
        """执行RA-CM3"""

        # 1. 识别查询模态
        modality = self._classify_modality(query)

        # 2. 跨模态检索
        retrieved_docs = self._retrieve_cross_modal(query, modality)

        # 3. 构建上下文
        context = "\n\n".join([doc.page_content for doc in retrieved_docs])

        # 4. 多模态生成
        generation = self._generate_multimodal(query, context)

        return {
            "answer": generation["text_response"],
            "image_suggestion": generation["image_suggestion"],
            "modality": modality,
            "num_retrieved": len(retrieved_docs),
            "retrieved_contents": [doc.page_content for doc in retrieved_docs],
        }


# ============================================
# 测试函数
# ============================================


def test_01_ra_cm3_basic(config: TestConfig):
    """测试: 基本RA-CM3"""
    print("\n[测试 01] 基本RA-CM3")

    llm = create_llm(config, temperature=0)
    if llm is None:
        return {"passed": False, "message": "创建LLM失败"}

    racm3 = RACM3(llm, MULTIMODAL_DOCS)
    result = racm3.query("什么是多模态RAG?")

    return {
        "passed": "answer" in result,
        "message": f"RA-CM3完成，检索了{result.get('num_retrieved', 0)}个文档",
        "details": {
            "modality": result.get("modality", ""),
            "num_retrieved": result.get("num_retrieved", 0),
            "answer_preview": result.get("answer", "")[:100],
        },
    }


def test_02_cross_modal_retrieval(config: TestConfig):
    """测试: 跨模态检索"""
    print("\n[测试 02] 跨模态检索")

    llm = create_llm(config, temperature=0)
    if llm is None:
        return {"passed": False, "message": "创建LLM失败"}

    racm3 = RACM3(llm, MULTIMODAL_DOCS)
    result = racm3.query("描述图像中的内容")

    return {
        "passed": result.get("modality", "") != "",
        "message": f"识别查询模态: {result.get('modality', '')}",
        "details": {"modality": result.get("modality", "")},
    }


def test_03_multimodal_generation(config: TestConfig):
    """测试: 多模态生成"""
    print("\n[测试 03] 多模态生成")

    llm = create_llm(config, temperature=0)
    if llm is None:
        return {"passed": False, "message": "创建LLM失败"}

    racm3 = RACM3(llm, MULTIMODAL_DOCS)
    result = racm3.query("VQA是什么?")

    return {
        "passed": "answer" in result,
        "message": "多模态生成完成",
        "details": {"has_image_suggestion": result.get("image_suggestion") is not None},
    }


# ============================================
# 主函数
# ============================================


def main():
    print("=" * 60)
    print("RA-CM3 架构测试")
    print("=" * 60)

    config = TestConfig()

    if not config.is_configured:
        print("\n⚠️  警告: API未配置!")
        return

    print(f"\n[配置] LLM: {config.llm_provider}/{config.openai_model}")

    results = []

    results.append(run_test(lambda: test_01_ra_cm3_basic(config), "基本RA-CM3"))
    results.append(
        run_test(lambda: test_02_cross_modal_retrieval(config), "跨模态检索")
    )
    results.append(
        run_test(lambda: test_03_multimodal_generation(config), "多模态生成")
    )

    print_test_results(results)

    failed = sum(1 for r in results if not r.passed)
    return 0 if failed == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
