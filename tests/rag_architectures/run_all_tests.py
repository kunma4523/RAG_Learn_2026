#!/usr/bin/env python3
"""
RAG架构测试运行器
=================

运行所有22个RAG架构测试。

运行:
    python tests/rag_architectures/run_all_tests.py

或单独运行某个测试:
    python tests/rag_architectures/01_standard_rag.py
"""

import sys
import os

# 添加项目路径
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

from tests.rag_architectures import TestConfig, print_test_results


# ============================================
# 测试文件列表
# ============================================

TEST_FILES = [
    ("01_standard_rag.py", "标准RAG (Standard RAG)"),
    ("02_conversational_rag.py", "对话式RAG (Conversational RAG)"),
    ("03_corrective_rag.py", "纠正RAG (Corrective RAG)"),
    ("04_adaptive_rag.py", "自适应RAG (Adaptive RAG)"),
    ("05_self_rag.py", "自反思RAG (Self-RAG)"),
    ("06_fusion_rag.py", "融合RAG (Fusion RAG)"),
    ("07_hyde.py", "HyDE (Hypothetical Document Embeddings)"),
    ("08_agentic_rag.py", "智能体RAG (Agentic RAG)"),
    ("09_graph_rag.py", "图RAG (GraphRAG)"),
    ("10_replug.py", "REPLUG (Retrieval Plugin)"),
    ("11_iterative_rag.py", "迭代RAG (Iterative RAG)"),
    ("12_in_context_ralm.py", "In-Context RALM"),
    ("13_flare.py", "FLARE (Forward-Looking Active Retrieval)"),
    ("14_multimodal_rag.py", "多模态RAG (Multimodal RAG)"),
    ("15_self_ask_rag.py", "Self-Ask + RAG"),
    ("16_sql_rag.py", "SQL-RAG"),
    ("17_table_rag.py", "Table RAG"),
    ("18_mapreduce_rag.py", "MapReduce RAG"),
    ("19_dragin.py", "DRAGIN (Dynamic Retrieval)"),
    ("20_self_mem.py", "Self-Mem (Memory-augmented RAG)"),
    ("21_raptor.py", "RAPTOR (Tree-organized Retrieval)"),
    ("22_ra_cm3.py", "RA-CM3 (Multimodal Multilingual)"),
]


# ============================================
# 主函数
# ============================================


def main():
    """运行所有RAG架构测试"""

    print("=" * 70)
    print("RAG架构测试套件 - 共22个架构测试")
    print("=" * 70)

    # 检查配置
    config = TestConfig()

    if not config.is_configured:
        print("\n⚠️  警告: API未配置!")
        print("\n请在.env文件中设置以下环境变量:")
        print("  LLM_PROVIDER=openai")
        print("  OPENAI_API_KEY=your-api-key-here")
        print("\n或运行单个测试文件进行开发调试。")
        print("\n可用的测试文件:")
        for idx, (filename, name) in enumerate(TEST_FILES, 1):
            print(f"  {idx}. {name}")
        return 1

    print(f"\n[配置] LLM: {config.llm_provider}/{config.openai_model}")
    print(f"[配置] Embedding: {config.embedding_provider}")
    print()

    # 运行每个测试
    all_results = []

    for idx, (filename, name) in enumerate(TEST_FILES, 1):
        print(f"\n{'=' * 70}")
        print(f"[{idx:02d}/22] 运行测试: {name}")
        print("=" * 70)

        try:
            # 动态导入并运行测试
            test_module = __import__(
                f"tests.rag_architectures.{filename[:-3]}", fromlist=["main"]
            )

            result = test_module.main()

            if result == 0:
                print(f"\n✓ {name} - 通过")
            else:
                print(f"\n✗ {name} - 失败")

        except Exception as e:
            print(f"\n✗ {name} - 错误: {str(e)[:100]}")

    print("\n" + "=" * 70)
    print("测试运行完成!")
    print("=" * 70)

    return 0


def list_tests():
    """列出所有可用测试"""
    print("\n可用的RAG架构测试:")
    print("-" * 50)

    for idx, (filename, name) in enumerate(TEST_FILES, 1):
        print(f"  {idx:2d}. {name}")
        print(f"      运行: python tests/rag_architectures/{filename}")

    print("\n" + "-" * 50)
    print(f"总计: {len(TEST_FILES)} 个架构测试")
    print("\n运行所有测试: python tests/rag_architectures/run_all_tests.py")


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "--list":
        list_tests()
    else:
        main()
