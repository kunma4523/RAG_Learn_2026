#!/usr/bin/env python
"""
向量存储测试
============

测试基于 NumPy + Pickle 的向量存储是否正常工作。
直接运行: python test_vector_store.py
"""

import sys
import os

# 将项目根目录添加到 Python 路径
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.utils.vector_store import VectorStore, FileVectorStore
import numpy as np


def test_basic_operations():
    """
    测试基本操作

    Returns:
        bool: 测试是否成功
    """
    print("=" * 60)
    print("测试向量存储基本操作")
    print("=" * 60)

    # 创建向量存储
    store = VectorStore(embedding_dim=4, top_k=3)

    # 测试数据
    documents = [
        "人工智能是计算机科学的一个分支。",
        "机器学习是人工智能的一个子领域。",
        "深度学习使用多层神经网络。",
        "自然语言处理处理文本数据。",
        "计算机视觉处理图像和视频。",
    ]

    # 生成随机嵌入向量（实际使用时应由 embedding 模型生成）
    embeddings = [
        [0.1, 0.2, 0.3, 0.4],
        [0.15, 0.25, 0.35, 0.45],
        [0.5, 0.5, 0.5, 0.5],
        [0.8, 0.1, 0.1, 0.1],
        [0.9, 0.05, 0.05, 0.05],
    ]

    # 添加文档
    print("\n添加文档...")
    store.add(documents, embeddings)
    print(f"✓ 成功添加 {len(documents)} 个文档")
    print(f"  当前存储: {store}")

    # 测试搜索
    query = [0.15, 0.25, 0.35, 0.45]  # 接近第二个文档
    print(f"\n查询向量: {query}")

    docs, scores, metadata = store.search(query, top_k=3)

    print(f"\n✓ 搜索成功! 返回 {len(docs)} 个结果")
    for i, (doc, score) in enumerate(zip(docs, scores)):
        print(f"  [{i + 1}] 相似度: {score:.4f}")
        print(f"      内容: {doc}")

    return True


def test_persistence():
    """
    测试持久化功能

    Returns:
        bool: 测试是否成功
    """
    print("\n" + "=" * 60)
    print("测试持久化功能")
    print("=" * 60)

    test_file = "test_vector_store.pkl"

    # 创建并保存
    store1 = VectorStore(embedding_dim=3)
    store1.add(["文档1", "文档2", "文档3"], [[1, 0, 0], [0, 1, 0], [0, 0, 1]])

    print(f"\n保存到 {test_file}...")
    store1.save(test_file)
    print(f"✓ 保存成功")

    # 加载
    store2 = VectorStore(embedding_dim=3)
    print(f"从 {test_file} 加载...")
    store2.load(test_file)

    print(f"✓ 加载成功")
    print(f"  文档数量: {len(store2.documents)}")
    print(f"  向量形状: {store2.vectors.shape}")

    # 搜索测试
    docs, scores, _ = store2.search([1, 0, 0])
    print(f"\n搜索 [1, 0, 0]:")
    print(f"  结果: {docs[0]} (相似度: {scores[0]:.4f})")

    # 清理
    if os.path.exists(test_file):
        os.remove(test_file)
        print(f"\n✓ 清理测试文件")

    return True


def test_file_vector_store():
    """
    测试文件向量存储（自动持久化）

    Returns:
        bool: 测试是否成功
    """
    print("\n" + "=" * 60)
    print("测试文件向量存储（自动持久化）")
    print("=" * 60)

    test_file = "test_fvs.pkl"

    # 清理可能存在的旧文件
    if os.path.exists(test_file):
        os.remove(test_file)

    # 创建（会自动加载已存在的）
    print(f"\n创建 FileVectorStore: {test_file}")
    store = FileVectorStore(filepath=test_file, embedding_dim=2, auto_save=True)

    # 添加文档（会自动保存）
    print("添加文档...")
    store.add(["苹果", "香蕉", "橙子"], [[1.0, 0.0], [0.0, 1.0], [0.5, 0.5]])
    print(f"✓ 添加成功，当前文档数: {len(store)}")

    # 搜索
    print("\n搜索 [0.9, 0.1]...")
    docs, scores, _ = store.search([0.9, 0.1])
    print(f"  结果: {docs[0]} (相似度: {scores[0]:.4f})")

    # 重新创建（验证自动加载）
    print("\n重新创建 store 验证自动加载...")
    store2 = FileVectorStore(filepath=test_file, embedding_dim=2)
    print(f"✓ 自动加载成功，文档数: {len(store2)}")

    # 清理
    if os.path.exists(test_file):
        os.remove(test_file)
        print(f"\n✓ 清理测试文件")

    return True


def test_similarity_calculation():
    """
    测试相似度计算

    Returns:
        bool: 测试是否成功
    """
    print("\n" + "=" * 60)
    print("测试相似度计算")
    print("=" * 60)

    store = VectorStore(embedding_dim=2, normalize=True)

    # 添加测试向量
    store.add(
        ["向量A", "向量B", "向量C"],
        [
            [1.0, 0.0],  # 方向 A
            [0.0, 1.0],  # 方向 B (与 A 正交)
            [0.707, 0.707],  # 对角线方向
        ],
    )

    # 测试相同方向的相似度
    docs, scores, _ = store.search([1.0, 0.0])
    print(f"\n搜索 [1.0, 0.0] (与向量A相同方向):")
    print(f"  最佳匹配: {docs[0]} (相似度: {scores[0]:.4f})")

    # 测试正交向量的相似度
    docs, scores, _ = store.search([0.0, 1.0])
    print(f"\n搜索 [0.0, 1.0] (与向量B相同方向):")
    print(f"  最佳匹配: {docs[0]} (相似度: {scores[0]:.4f})")

    # 测试对角线方向
    docs, scores, _ = store.search([0.707, 0.707])
    print(f"\n搜索 [0.707, 0.707] (对角线方向):")
    print(f"  最佳匹配: {docs[0]} (相似度: {scores[0]:.4f})")

    return True


def test_delete():
    """
    测试删除功能

    Returns:
        bool: 测试是否成功
    """
    print("\n" + "=" * 60)
    print("测试删除功能")
    print("=" * 60)

    store = VectorStore(embedding_dim=2)

    store.add(
        ["文档1", "文档2", "文档3", "文档4", "文档5"],
        [[1, 0], [2, 0], [3, 0], [4, 0], [5, 0]],
    )

    print(f"\n原始文档数: {len(store)}")
    print(f"文档: {store.documents}")

    # 删除索引 1 和 3（文档2和文档4）
    store.delete([1, 3])

    print(f"\n删除后文档数: {len(store)}")
    print(f"文档: {store.documents}")

    # 验证
    if len(store) == 3 and "文档2" not in store.documents:
        print("\n✓ 删除功能正常")
        return True
    else:
        print("\n✗ 删除功能异常")
        return False


def test_metadata():
    """
    测试元数据功能

    Returns:
        bool: 测试是否成功
    """
    print("\n" + "=" * 60)
    print("测试元数据功能")
    print("=" * 60)

    store = VectorStore(embedding_dim=2)

    documents = ["Python教程", "Java教程", "JavaScript教程", "Go教程"]
    embeddings = [[1, 0], [1, 1], [0, 1], [1, 0.5]]
    metadata = [
        {"category": "后端", "rating": 5},
        {"category": "后端", "rating": 4},
        {"category": "前端", "rating": 5},
        {"category": "后端", "rating": 3},
    ]

    store.add(documents, embeddings, metadata)

    # 无过滤搜索
    print("\n无过滤搜索:")
    docs, scores, meta = store.search([1, 0], top_k=4)
    for doc, score, m in zip(docs, scores, meta):
        print(f"  {doc} (score: {score:.4f}, category: {m.get('category')})")

    # 按 category 过滤
    print("\n按 category='后端' 过滤:")
    docs, scores, meta = store.search(
        [1, 0], top_k=4, filter_metadata={"category": "后端"}
    )
    for doc, score, m in zip(docs, scores, meta):
        print(f"  {doc} (score: {score:.4f}, rating: {m.get('rating')})")

    return True


def main():
    """主测试函数"""
    print("\n" + "=" * 60)
    print("       RAG-Learning 向量存储测试")
    print("=" * 60)
    print("\n本测试验证基于 NumPy + Pickle 的向量存储功能\n")

    all_passed = True

    all_passed &= test_basic_operations()
    all_passed &= test_persistence()
    all_passed &= test_file_vector_store()
    all_passed &= test_similarity_calculation()
    all_passed &= test_delete()
    all_passed &= test_metadata()

    print("\n" + "=" * 60)
    if all_passed:
        print("✓ 所有测试通过!")
    else:
        print("✗ 部分测试失败")
    print("=" * 60)

    return 0 if all_passed else 1


if __name__ == "__main__":
    sys.exit(main())
