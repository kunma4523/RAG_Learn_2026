"""
向量存储
========

基于 NumPy 和 Pickle 的简单向量存储实现。
用于存储文档向量并支持高效的相似度检索。
"""

import os
import pickle
from typing import List, Optional, Tuple
import numpy as np


class VectorStore:
    """
    基于 NumPy + Pickle 的向量存储类

    使用余弦相似度进行向量检索，支持持久化存储。
    """

    def __init__(
        self, embedding_dim: int = 768, top_k: int = 5, normalize: bool = True
    ):
        """
        初始化向量存储

        Args:
            embedding_dim: 嵌入向量的维度
            top_k: 默认返回的最近邻数量
            normalize: 是否对向量进行归一化
        """
        self.embedding_dim = embedding_dim
        self.top_k = top_k
        self.normalize = normalize

        # 存储数据
        self.vectors: Optional[np.ndarray] = None
        self.documents: List[str] = []
        self.metadata: List[dict] = []

        # 索引
        self._indexed = False

    def add(
        self,
        documents: List[str],
        embeddings: List[List[float]],
        metadata: Optional[List[dict]] = None,
    ) -> None:
        """
        添加文档和对应的嵌入向量

        Args:
            documents: 文档文本列表
            embeddings: 嵌入向量列表
            metadata: 可选的元数据列表
        """
        if len(documents) != len(embeddings):
            raise ValueError("Documents and embeddings must have the same length")

        # 转换为 numpy 数组
        embeddings_array = np.array(embeddings, dtype=np.float32)

        # 归一化（如果需要）
        if self.normalize:
            embeddings_array = self._normalize(embeddings_array)

        # 添加到存储
        if self.vectors is None:
            self.vectors = embeddings_array
        else:
            self.vectors = np.vstack([self.vectors, embeddings_array])

        self.documents.extend(documents)

        if metadata is None:
            self.metadata.extend([{}] * len(documents))
        else:
            self.metadata.extend(metadata)

        self._indexed = False

    def search(
        self,
        query_embedding: List[float],
        top_k: Optional[int] = None,
        filter_metadata: Optional[dict] = None,
    ) -> Tuple[List[str], List[float], List[dict]]:
        """
        搜索最相似的文档

        Args:
            query_embedding: 查询向量
            top_k: 返回结果数量，None 则使用默认值
            filter_metadata: 可选的元数据过滤条件

        Returns:
            (documents, scores, metadata) 元组
        """
        if self.vectors is None or len(self.vectors) == 0:
            return [], [], []

        top_k = top_k or self.top_k

        # 转换查询向量
        query_vec = np.array(query_embedding, dtype=np.float32).reshape(1, -1)

        if self.normalize:
            query_vec = self._normalize(query_vec)

        # 计算余弦相似度
        scores = np.dot(self.vectors, query_vec.T).flatten()

        # 获取 top-k 索引
        if filter_metadata:
            # 应用元数据过滤
            valid_indices = self._filter_by_metadata(filter_metadata)
            if not valid_indices:
                return [], [], []

            filtered_scores = np.full(len(scores), -np.inf)
            filtered_scores[valid_indices] = scores[valid_indices]

            top_indices = np.argsort(filtered_scores)[::-1][:top_k]
            top_indices = [i for i in top_indices if i in valid_indices]
        else:
            top_indices = np.argsort(scores)[::-1][:top_k]

        # 构建结果
        results_docs = [self.documents[i] for i in top_indices]
        results_scores = [float(scores[i]) for i in top_indices]
        results_metadata = [self.metadata[i] for i in top_indices]

        return results_docs, results_scores, results_metadata

    def save(self, filepath: str) -> None:
        """
        保存向量存储到文件

        Args:
            filepath: 保存路径
        """
        data = {
            "vectors": self.vectors,
            "documents": self.documents,
            "metadata": self.metadata,
            "embedding_dim": self.embedding_dim,
            "normalize": self.normalize,
        }

        # 确保目录存在
        dir_path = os.path.dirname(filepath)
        if dir_path:
            os.makedirs(dir_path, exist_ok=True)

        with open(filepath, "wb") as f:
            pickle.dump(data, f)

    def load(self, filepath: str) -> None:
        """
        从文件加载向量存储

        Args:
            filepath: 加载路径
        """
        with open(filepath, "rb") as f:
            data = pickle.load(f)

        self.vectors = data["vectors"]
        self.documents = data["documents"]
        self.metadata = data["metadata"]
        self.embedding_dim = data["embedding_dim"]
        self.normalize = data["normalize"]
        self._indexed = False

    def delete(self, indices: List[int]) -> None:
        """
        删除指定索引的文档

        Args:
            indices: 要删除的索引列表
        """
        if self.vectors is None:
            return

        indices_set = set(indices)

        # 保留未被删除的
        keep_mask = np.ones(len(self.documents), dtype=bool)
        keep_mask[list(indices_set)] = False

        self.vectors = self.vectors[keep_mask]
        self.documents = [
            d for i, d in enumerate(self.documents) if i not in indices_set
        ]
        self.metadata = [m for i, m in enumerate(self.metadata) if i not in indices_set]

        self._indexed = False

    def clear(self) -> None:
        """清空所有存储的数据"""
        self.vectors = None
        self.documents = []
        self.metadata = []
        self._indexed = False

    def __len__(self) -> int:
        """返回存储的文档数量"""
        return len(self.documents)

    def __repr__(self) -> str:
        return f"VectorStore(documents={len(self.documents)}, dim={self.embedding_dim})"

    def _normalize(self, vectors: np.ndarray) -> np.ndarray:
        """归一化向量"""
        norms = np.linalg.norm(vectors, axis=1, keepdims=True)
        norms = np.where(norms == 0, 1, norms)  # 避免除零
        return vectors / norms

    def _filter_by_metadata(self, filter_dict: dict) -> List[int]:
        """根据元数据过滤返回有效索引"""
        valid_indices = []
        for i, meta in enumerate(self.metadata):
            match = all(meta.get(k) == v for k, v in filter_dict.items())
            if match:
                valid_indices.append(i)
        return valid_indices


class FileVectorStore(VectorStore):
    """
    支持文件自动持久化的向量存储

    每次添加文档后自动保存到文件。
    """

    def __init__(
        self,
        filepath: str,
        embedding_dim: int = 768,
        top_k: int = 5,
        normalize: bool = True,
        auto_save: bool = True,
    ):
        """
        初始化文件向量存储

        Args:
            filepath: 存储文件路径
            embedding_dim: 嵌入维度
            top_k: 默认返回数量
            normalize: 是否归一化
            auto_save: 是否在添加后自动保存
        """
        super().__init__(embedding_dim, top_k, normalize)

        self.filepath = filepath
        self.auto_save = auto_save

        # 尝试加载已存在的文件
        if os.path.exists(filepath):
            try:
                self.load(filepath)
                print(f"✓ 已从 {filepath} 加载向量存储")
            except Exception as e:
                print(f"⚠ 加载失败: {e}, 创建新的存储")

    def add(
        self,
        documents: List[str],
        embeddings: List[List[float]],
        metadata: Optional[List[dict]] = None,
    ) -> None:
        """添加文档（带自动保存）"""
        super().add(documents, embeddings, metadata)

        if self.auto_save:
            self.save(self.filepath)

    def delete(self, indices: List[int]) -> None:
        """删除文档（带自动保存）"""
        super().delete(indices)

        if self.auto_save:
            self.save(self.filepath)

    def clear(self) -> None:
        """清空存储（带自动保存）"""
        super().clear()

        if self.auto_save:
            if os.path.exists(self.filepath):
                os.remove(self.filepath)
