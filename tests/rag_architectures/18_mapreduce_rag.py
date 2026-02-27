#!/usr/bin/env python3
"""
测试: MapReduce RAG (18_mapreduce_rag.py)
==========================================

分而治之的RAG架构，将大文档分块处理后合并摘要。

运行: python tests/rag_architectures/18_mapreduce_rag.py

架构:
    ┌────────────┐
    │ Large Doc  │
    └────────────┘
          │
    ┌─────┴─────┬────────┐
    ▼           ▼        ▼
  Chunk 1   Chunk 2   Chunk 3
    │           │        │
    ▼           ▼        ▼
  Map()     Map()     Map()
    │           │        │
    ▼           ▼        ▼
 Sum1       Sum2      Sum3
    └─────┬─────┴────────┘
          ▼
    ┌──────────┐
    │ Reduce() │
    └──────────┘
          │
          ▼
    ┌──────────┐    ┌────────────┐
    │ Generate │───▶│   Answer   │
    └──────────┘    └────────────┘
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
# 测试文档 - 长文档
# ============================================

LONG_DOCUMENT = """
第一章: RAG概述
检索增强生成(Retrieval-Augmented Generation, RAG)是一种将信息检索与文本生成相结合的技术。
它通过从外部知识库中检索相关信息来增强大语言模型的生成能力。

第二章: RAG的组成
RAG系统主要包括三个组件:检索器(Retriever)、增强器(Augmenter)和生成器(Generator)。
检索器负责从知识库中找到相关文档，增强器将检索结果进行预处理，生成器基于增强后的上下文生成回答。

第三章: 检索技术
常见的检索技术包括:
1. 稀疏检索:如BM25，基于关键词匹配
2. 密集检索:如DPR，使用神经网络编码
3. 混合检索:结合稀疏和密集方法

第四章: 应用场景
RAG技术广泛应用于:
- 问答系统
- 对话系统
- 文本摘要
- 信息抽取

第五章: 挑战与未来
RAG面临的主要挑战包括:
- 检索质量与效率的平衡
- 多模态信息处理
- 实时更新知识库
"""


# ============================================
# MapReduce RAG 实现
# ============================================


class MapReduceRAG:
    """
    MapReduce RAG 实现

    特点:
    1. 将大文档分块(Map)
    2. 对每个块进行处理
    3. 合并所有块的结果(Reduce)
    4. 基于合并结果生成最终答案
    """

    def __init__(self, llm):
        self.llm = llm

    def _split_into_chunks(self, text: str, chunk_size: int = 500) -> list:
        """将文档分块"""
        words = text.split()
        chunks = []

        for i in range(0, len(words), chunk_size):
            chunk = " ".join(words[i : i + chunk_size])
            chunks.append(chunk)

        return chunks

    def _map_chunk(self, chunk: str, query: str) -> str:
        """Map阶段:处理单个块"""

        prompt = f"""文档片段:
{chunk}

问题: {query}

如果这个片段与问题相关，请提取相关信息并总结:"""

        response = self.llm.invoke(prompt)
        return response.content

    def _reduce_summaries(self, summaries: list, query: str) -> str:
        """Reduce阶段:合并所有摘要"""

        combined = "\n\n".join(
            [f"摘要{i + 1}:\n{s}" for i, s in enumerate(summaries) if s.strip()]
        )

        prompt = f"""以下是多个文档片段的摘要:

{combined}

问题: {query}

请整合所有摘要，给出最终答案:"""

        response = self.llm.invoke(prompt)
        return response.content

    def query(self, query: str, chunk_size: int = 200) -> dict:
        """执行MapReduce RAG"""

        # 1. 分块
        chunks = self._split_into_chunks(LONG_DOCUMENT, chunk_size)

        # 2. Map: 处理每个块
        summaries = []
        for chunk in chunks:
            summary = self._map_chunk(chunk, query)
            if summary.strip():
                summaries.append(summary)

        # 3. Reduce: 合并摘要
        final_answer = self._reduce_summaries(summaries, query)

        return {
            "answer": final_answer,
            "num_chunks": len(chunks),
            "num_summaries": len(summaries),
            "chunk_summaries": summaries,
        }


# ============================================
# 测试函数
# ============================================


def test_01_mapreduce_basic(config: TestConfig):
    """测试: 基本MapReduce"""
    print("\n[测试 01] 基本MapReduce RAG")

    llm = create_llm(config, temperature=0)
    if llm is None:
        return {"passed": False, "message": "创建LLM失败"}

    mrrag = MapReduceRAG(llm)
    result = mrrag.query("RAG的组成组件是什么?")

    return {
        "passed": "answer" in result,
        "message": f"MapReduce完成，处理了{result.get('num_chunks', 0)}个块",
        "details": {
            "num_chunks": result.get("num_chunks", 0),
            "num_summaries": result.get("num_summaries", 0),
            "answer_preview": result.get("answer", "")[:100],
        },
    }


def test_02_chunk_processing(config: TestConfig):
    """测试: 分块处理"""
    print("\n[测试 02] 分块处理")

    llm = create_llm(config, temperature=0)
    if llm is None:
        return {"passed": False, "message": "创建LLM失败"}

    mrrag = MapReduceRAG(llm)
    result = mrrag.query("RAG的检索技术有哪些?")

    return {
        "passed": result.get("num_chunks", 0) > 1,
        "message": f"文档被分成{result.get('num_chunks', 0)}个块处理",
        "details": {"chunks": result.get("num_chunks", 0)},
    }


def test_03_reduce_phase(config: TestConfig):
    """测试: Reduce合并阶段"""
    print("\n[测试 03] Reduce合并阶段")

    llm = create_llm(config, temperature=0)
    if llm is None:
        return {"passed": False, "message": "创建LLM失败"}

    mrrag = MapReduceRAG(llm)
    result = mrrag.query("RAG的未来挑战是什么?")

    has_reduce = result.get("num_summaries", 0) > 0

    return {
        "passed": has_reduce,
        "message": f"合并了{result.get('num_summaries', 0)}个摘要",
        "details": {"summaries": result.get("num_summaries", 0)},
    }


# ============================================
# 主函数
# ============================================


def main():
    print("=" * 60)
    print("MapReduce RAG 架构测试")
    print("=" * 60)

    config = TestConfig()

    if not config.is_configured:
        print("\n⚠️  警告: API未配置!")
        return

    print(f"\n[配置] LLM: {config.llm_provider}/{config.openai_model}")

    results = []

    results.append(run_test(lambda: test_01_mapreduce_basic(config), "基本MapReduce"))
    results.append(run_test(lambda: test_02_chunk_processing(config), "分块处理"))
    results.append(run_test(lambda: test_03_reduce_phase(config), "Reduce合并"))

    print_test_results(results)

    failed = sum(1 for r in results if not r.passed)
    return 0 if failed == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
