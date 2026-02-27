#!/usr/bin/env python3
"""
测试: RAPTOR (21_raptor.py)
===========================

Recursive Abstractive Processing for Tree-Organized Retrieval.
构建文档摘要树，实现分层检索。

运行: python tests/rag_architectures/21_raptor.py

架构:
                    ┌─────────────┐
                    │   Query     │
                    └─────────────┘
                          │
           ┌──────────────┼──────────────┐
           ▼              ▼              ▼
        Level 3       Level 2        Level 1
       (摘要)         (摘要)       (原文块)
           │              │              │
           └──────────────┼──────────────┘
                          │
                    ┌─────┴─────┐
                    │   融合    │
                    └───────────┘
                          │
                    ┌─────┴─────┐
                    │ Generate  │
                    └───────────┘
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

DOCUMENT_CHUNKS = [
    "RAG概述: 检索增强生成是一种结合信息检索与文本生成的技术。它通过从外部知识库中检索相关信息来增强大语言模型的输出质量。",
    "RAG组成: RAG系统包括三个主要组件: 检索器从知识库中查找相关文档, 增强器对检索结果进行处理和排序, 生成器基于增强后的上下文生成最终回答。",
    "检索技术: 常见的检索技术包括稀疏检索(如BM25)和密集检索(如DPR)。稀疏检索基于关键词匹配, 密集检索使用神经网络编码来理解语义。",
    "应用场景: RAG技术广泛应用于问答系统、对话系统、文本摘要和信息抽取等场景。它可以帮助减少大语言模型的幻觉问题。",
    "评估指标: RAG系统的评估指标包括检索指标(如Recall@K、MRR)和生成指标(如BLEU、ROUGE)。端到端评估也很重要。",
    "挑战: RAG面临的主要挑战包括检索质量与效率的平衡、多模态信息处理、实时更新知识库以及如何更好地融合检索结果。",
]


# ============================================
# RAPTOR 实现
# ============================================


class RAPTOR:
    """
    RAPTOR 实现

    特点:
    1. 构建多层次文档摘要树
    2. 底层是原始文档块
    3. 更高层是摘要
    4. 分层检索
    """

    def __init__(self, llm, documents: list):
        self.llm = llm
        self.documents = documents
        self.tree_levels = {}
        self._build_tree()

    def _build_tree(self):
        """构建摘要树"""

        # Level 1: 原始文档
        self.tree_levels[1] = self.documents

        # Level 2+: 递归生成摘要
        current_level = self.documents
        level = 2

        while len(current_level) > 1:
            # 将当前层分成小组
            group_size = max(2, len(current_level) // 2)
            groups = [
                current_level[i : i + group_size]
                for i in range(0, len(current_level), group_size)
            ]

            # 每组生成摘要
            summaries = []
            for group in groups:
                if len(group) > 1:
                    combined = "\n\n".join(group)
                    prompt = f"总结以下文档要点:\n\n{combined}\n\n简洁总结:"
                    response = self.llm.invoke(prompt)
                    summaries.append(response.content)
                else:
                    summaries.extend(group)

            self.tree_levels[level] = summaries
            current_level = summaries
            level += 1

            if level > 3:  # 限制层数
                break

    def _retrieve_from_level(self, query: str, level: int) -> list:
        """从指定层检索"""

        if level not in self.tree_levels:
            return []

        # 创建临时向量存储
        try:
            from tests.rag_architectures import create_vector_store, TestConfig

            config = TestConfig()
            vs = create_vector_store(self.tree_levels[level], config)

            if vs:
                retriever = vs.as_retriever(search_kwargs={"k": 2})
                return retriever.invoke(query)
        except:
            pass

        return []

    def query(self, query: str) -> dict:
        """执行RAPTOR检索"""

        results_from_levels = {}

        # 从高层到低层检索
        for level in sorted(self.tree_levels.keys(), reverse=True):
            docs = self._retrieve_from_level(query, level)
            if docs:
                results_from_levels[level] = [doc.page_content for doc in docs]

        # 融合所有层的结果
        all_context = []
        for level in sorted(results_from_levels.keys(), reverse=True):
            all_context.extend(results_from_levels[level])

        context = "\n\n".join(all_context[:5])

        # 生成回答
        prompt = f"""基于以下分层检索结果:

{context}

问题: {query}

请给出综合回答:"""

        response = self.llm.invoke(prompt)

        return {
            "answer": response.content,
            "levels_retrieved": list(results_from_levels.keys()),
            "results_per_level": {k: len(v) for k, v in results_from_levels.items()},
            "tree_structure": {k: len(v) for k, v in self.tree_levels.items()},
        }


# ============================================
# 测试函数
# ============================================


def test_01_raptor_basic(config: TestConfig):
    """测试: 基本RAPTOR"""
    print("\n[测试 01] 基本RAPTOR")

    llm = create_llm(config, temperature=0)
    if llm is None:
        return {"passed": False, "message": "创建LLM失败"}

    raptor = RAPTOR(llm, DOCUMENT_CHUNKS)
    result = raptor.query("RAG的组成是什么?")

    return {
        "passed": "answer" in result,
        "message": f"RAPTOR构建了{len(result.get('tree_structure', {}))}层摘要树",
        "details": {
            "tree_levels": result.get("tree_structure", {}),
            "answer_preview": result.get("answer", "")[:100],
        },
    }


def test_02_tree_structure(config: TestConfig):
    """测试: 树结构构建"""
    print("\n[测试 02] 树结构构建")

    llm = create_llm(config, temperature=0)
    if llm is None:
        return {"passed": False, "message": "创建LLM失败"}

    raptor = RAPTOR(llm, DOCUMENT_CHUNKS)

    has_levels = len(raptor.tree_levels) > 1

    return {
        "passed": has_levels,
        "message": f"摘要树有{len(raptor.tree_levels)}层",
        "details": {"levels": list(raptor.tree_levels.keys())},
    }


def test_03_layered_retrieval(config: TestConfig):
    """测试: 分层检索"""
    print("\n[测试 03] 分层检索")

    llm = create_llm(config, temperature=0)
    if llm is None:
        return {"passed": False, "message": "创建LLM失败"}

    raptor = RAPTOR(llm, DOCUMENT_CHUNKS)
    result = raptor.query("RAG面临哪些挑战?")

    return {
        "passed": len(result.get("levels_retrieved", [])) > 0,
        "message": f"从{len(result.get('levels_retrieved', []))}个层级检索",
        "details": {"retrieved_levels": result.get("levels_retrieved", [])},
    }


# ============================================
# 主函数
# ============================================


def main():
    print("=" * 60)
    print("RAPTOR 架构测试")
    print("=" * 60)

    config = TestConfig()

    if not config.is_configured:
        print("\n⚠️  警告: API未配置!")
        return

    print(f"\n[配置] LLM: {config.llm_provider}/{config.openai_model}")

    results = []

    results.append(run_test(lambda: test_01_raptor_basic(config), "基本RAPTOR"))
    results.append(run_test(lambda: test_02_tree_structure(config), "树结构"))
    results.append(run_test(lambda: test_03_layered_retrieval(config), "分层检索"))

    print_test_results(results)

    failed = sum(1 for r in results if not r.passed)
    return 0 if failed == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
