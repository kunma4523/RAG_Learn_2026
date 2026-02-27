"""
基础检索器接口
==============

所有检索器实现的抽象基类。
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import List, Dict, Any, Optional
import numpy as np


@dataclass
class RetrievalResult:
    """表示单个检索结果"""

    text: str
    score: float
    metadata: Dict[str, Any]

    def __repr__(self) -> str:
        return f"RetrievalResult(text={self.text[:50]}..., score={self.score:.4f})"


class BaseRetriever(ABC):
    """所有检索器的抽象基类"""

    def __init__(self, top_k: int = 5):
        """
        初始化检索器

        Args:
            top_k: 检索的文档数量
        """
        self.top_k = top_k

    @abstractmethod
    def index(self, documents: List[str], **kwargs) -> None:
        """
        为检索索引文档

        Args:
            documents: 要索引的文档文本列表
            **kwargs: 额外索引参数
        """
        pass

    @abstractmethod
    def retrieve(self, query: str, **kwargs) -> List[RetrievalResult]:
        """
        为查询检索相关文档

        Args:
            query: 查询字符串
            **kwargs: 额外检索参数

        Returns:
            按相关性排序的 RetrievalResult 对象列表
        """
        pass

    def retrieve_batch(
        self, queries: List[str], **kwargs
    ) -> List[List[RetrievalResult]]:
        """
        为多个查询检索文档

        Args:
            queries: 查询字符串列表

        Returns:
            每个查询的检索结果列表
        """
        return [self.retrieve(q, **kwargs) for q in queries]

    def compute_similarity(
        self, query_embedding: np.ndarray, doc_embeddings: np.ndarray
    ) -> np.ndarray:
        """
        计算查询和文档嵌入之间的余弦相似度

        Args:
            query_embedding: 查询嵌入向量
            doc_embeddings: 文档嵌入矩阵

        Returns:
            相似度分数
        """
        # 归一化嵌入向量
        query_norm = query_embedding / (np.linalg.norm(query_embedding) + 1e-8)
        docs_norm = doc_embeddings / (
            np.linalg.norm(doc_embeddings, axis=1, keepdims=True) + 1e-8
        )

        # 计算余弦相似度
        return np.dot(docs_norm, query_norm)

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(top_k={self.top_k})"
