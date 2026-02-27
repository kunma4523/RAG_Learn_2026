#!/usr/bin/env python
"""
嵌入检索器测试
==============

测试嵌入检索器是否正常工作。
直接运行: python test_embedding.py

注意: 运行前请确保已安装依赖并在 .env 文件中配置好 API Key
"""

import sys
import os

# 将项目根目录添加到 Python 路径
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.retrievers.dense import (
    get_embedding_retriever,
    SentenceTransformerRetriever,
    OpenAIEmbeddingRetriever,
    DashScopeEmbeddingRetriever,
)
from src.utils.config import get_embedding_provider, get_embedding_config


def test_embedding_retriever():
    """
    测试根据环境配置自动选择嵌入检索器

    Returns:
        bool: 测试是否成功
    """
    print("=" * 60)
    print("测试嵌入检索器自动选择")
    print("=" * 60)

    # 打印当前配置
    provider = get_embedding_provider()
    config = get_embedding_config()

    print(f"\n当前 Embedding 提供商: {provider}")
    print(f"模型: {config.get('model', 'N/A')}")
    print()

    # 创建检索器
    try:
        retriever = get_embedding_retriever()
        print(f"✓ 成功创建检索器: {retriever}")
    except Exception as e:
        print(f"✗ 创建检索器失败: {e}")
        return False

    # 测试数据
    documents = [
        "人工智能是计算机科学的一个分支，致力于开发能够执行通常需要人类智能的任务的系统。",
        "机器学习是人工智能的一个子领域，专注于开发能够从数据中学习和改进的算法。",
        "深度学习是机器学习的一个分支，使用多层神经网络来学习数据的分层表示。",
        "自然语言处理是人工智能的一个领域，专注于使计算机能够理解和生成人类语言。",
        "计算机视觉是人工智能的一个领域，致力于使计算机能够理解和处理图像和视频。",
    ]

    # 索引文档
    print("\n正在索引文档...")
    try:
        retriever.index(documents)
        print(f"✓ 成功索引 {len(documents)} 个文档")
    except Exception as e:
        print(f"✗ 索引失败: {e}")
        return False

    # 测试检索
    test_query = "什么是机器学习?"

    print(f"\n测试查询: {test_query}")
    print("-" * 40)

    try:
        results = retriever.retrieve(test_query)

        if results:
            print(f"\n✓ 检索成功! 找到 {len(results)} 个结果")

            for i, result in enumerate(results):
                print(f"\n  [{i + 1}] 相似度: {result.score:.4f}")
                print(f"      内容: {result.text[:80]}...")
                print(f"      元数据: {result.metadata}")

            return True
        else:
            print(f"\n✗ 检索失败: 未找到结果")
            return False

    except Exception as e:
        print(f"\n✗ 检索异常: {e}")
        return False


def test_specific_retriever():
    """
    测试特定的嵌入检索器（如果配置了相应的 API Key）

    Returns:
        bool: 测试是否成功
    """
    print("\n" + "=" * 60)
    print("测试特定嵌入检索器")
    print("=" * 60)

    config = get_embedding_config()
    api_key = config.get("api_key")

    if not api_key or api_key in ["your-api-key-here", "sk-your-key-here"]:
        print("\n⚠ 未配置 API Key，跳过特定检索器测试")
        print("  请在 .env 文件中配置相应的 API Key")
        return True

    # 测试 OpenAI Embedding
    if config.get("provider") == "openai":
        print("\n测试 OpenAI Embedding 检索器...")
        try:
            retriever = OpenAIEmbeddingRetriever()
            retriever.index(["测试文档"])
            results = retriever.retrieve("测试查询")
            print(f"  ✓ OpenAI 检索成功: {len(results)} 个结果")
        except Exception as e:
            print(f"  ✗ OpenAI 测试失败: {e}")

    # 测试 DashScope Embedding
    if config.get("provider") == "dashscope":
        print("\n测试 DashScope Embedding 检索器...")
        try:
            retriever = DashScopeEmbeddingRetriever()
            retriever.index(["测试文档"])
            results = retriever.retrieve("测试查询")
            print(f"  ✓ DashScope 检索成功: {len(results)} 个结果")
        except Exception as e:
            print(f"  ✗ DashScope 测试失败: {e}")

    return True


def test_batch_retrieve():
    """
    测试批量检索功能

    Returns:
        bool: 测试是否成功
    """
    print("\n" + "=" * 60)
    print("测试批量检索")
    print("=" * 60)

    try:
        retriever = get_embedding_retriever()

        # 测试数据
        documents = [
            "人工智能是计算机科学的一个分支。",
            "机器学习是人工智能的一个子领域。",
            "深度学习使用多层神经网络。",
            "自然语言处理处理文本数据。",
            "计算机视觉处理图像和视频。",
        ]

        queries = [
            "什么是人工智能?",
            "什么是机器学习?",
            "深度学习是什么?",
        ]

        # 索引文档
        retriever.index(documents)

        print(f"\n索引 {len(documents)} 个文档")
        print(f"批量检索 {len(queries)} 个查询...")

        results = retriever.retrieve_batch(queries)

        print(f"\n✓ 批量检索完成")

        for i, query_results in enumerate(results):
            print(f"\n  查询: {queries[i]}")
            for j, result in enumerate(query_results[:2]):
                print(
                    f"    [{j + 1}] {result.text[:40]}... (score: {result.score:.4f})"
                )

        return True

    except Exception as e:
        print(f"\n✗ 批量检索测试失败: {e}")
        return False


def test_similarity_computation():
    """
    测试相似度计算功能

    Returns:
        bool: 测试是否成功
    """
    print("\n" + "=" * 60)
    print("测试相似度计算")
    print("=" * 60)

    try:
        retriever = get_embedding_retriever()

        # 测试数据
        documents = [
            "人工智能是计算机科学的一个分支。",
            "机器学习是人工智能的一个子领域。",
        ]

        # 索引
        retriever.index(documents)

        # 计算相似度
        import numpy as np

        # 创建两个简单的向量
        vec1 = np.array([1.0, 0.0])
        vec2 = np.array([1.0, 0.0])

        similarity = retriever.compute_similarity(
            vec1.reshape(1, -1), vec2.reshape(1, -1)
        )

        print(f"\n  向量1: {vec1}")
        print(f"  向量2: {vec2}")
        print(f"  相似度: {similarity[0]:.4f}")

        # 测试正交向量
        vec3 = np.array([0.0, 1.0])
        similarity2 = retriever.compute_similarity(
            vec1.reshape(1, -1), vec3.reshape(1, -1)
        )

        print(f"\n  向量3: {vec3}")
        print(f"  与向量1的相似度: {similarity2[0]:.4f}")

        print(f"\n✓ 相似度计算功能正常")

        return True

    except Exception as e:
        print(f"\n✗ 相似度计算测试失败: {e}")
        return False


def test_top_k():
    """
    测试 top_k 参数

    Returns:
        bool: 测试是否成功
    """
    print("\n" + "=" * 60)
    print("测试 top_k 参数")
    print("=" * 60)

    try:
        # 创建 top_k=3 的检索器
        retriever = get_embedding_retriever(top_k=3)

        # 测试数据
        documents = [
            "这是第1个文档，关于人工智能。",
            "这是第2个文档，关于机器学习。",
            "这是第3个文档，关于深度学习。",
            "这是第4个文档，关于自然语言处理。",
            "这是第5个文档，关于计算机视觉。",
        ]

        # 索引
        retriever.index(documents)

        # 检索
        query = "关于 AI 和机器学习的内容"
        results = retriever.retrieve(query)

        print(f"\n  请求 top_k: 3")
        print(f"  实际返回: {len(results)} 个结果")

        if len(results) == 3:
            print(f"\n✓ top_k 参数正常工作")
            return True
        else:
            print(f"\n✗ top_k 参数异常")
            return False

    except Exception as e:
        print(f"\n✗ top_k 测试失败: {e}")
        return False


def main():
    """主测试函数"""
    print("\n" + "=" * 60)
    print("       RAG-Learning 嵌入检索器测试")
    print("=" * 60)
    print("\n本测试将验证嵌入检索器配置是否正确")
    print("请确保已在 .env 文件中配置好 API Key\n")

    # 检查环境配置
    print("检查环境配置...")
    provider = get_embedding_provider()
    config = get_embedding_config()

    print(f"  Embedding 提供商: {provider}")
    print(f"  模型: {config.get('model', 'N/A')}")

    # 检查 API Key
    api_key = config.get("api_key")
    if not api_key or api_key in ["your-api-key-here", "sk-your-key-here"]:
        if provider in ["openai", "dashscope"]:
            print("\n⚠ 警告: 未在 .env 中配置有效的 API Key")
            print("  测试可能会失败，请配置后再试")

    print()

    # 运行测试
    all_passed = True

    # 测试 1: 自动选择检索器
    all_passed &= test_embedding_retriever()

    # 测试 2: 批量检索
    all_passed &= test_batch_retrieve()

    # 测试 3: 相似度计算
    all_passed &= test_similarity_computation()

    # 测试 4: top_k 参数
    all_passed &= test_top_k()

    # 测试 5: 特定检索器
    all_passed &= test_specific_retriever()

    # 总结
    print("\n" + "=" * 60)
    if all_passed:
        print("✓ 所有测试通过!")
    else:
        print("✗ 部分测试失败，请检查配置")
    print("=" * 60)

    return 0 if all_passed else 1


if __name__ == "__main__":
    sys.exit(main())
