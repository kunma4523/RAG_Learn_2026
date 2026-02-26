"""
Hybrid Retriever
================

Combines sparse and dense retrieval methods.
"""

from typing import List, Dict, Any, Optional, Callable
import numpy as np

from src.retrievers.base import BaseRetriever, RetrievalResult


class HybridRetriever(BaseRetriever):
    """Hybrid retriever combining sparse and dense methods."""
    
    def __init__(
        self,
        sparse_retriever: BaseRetriever,
        dense_retriever: BaseRetriever,
        top_k: int = 5,
        alpha: float = 0.5,
        fusion_method: str = "rrf"
    ):
        """
        Initialize hybrid retriever.
        
        Args:
            sparse_retriever: Sparse retrieval method (e.g., BM25)
            dense_retriever: Dense retrieval method (e.g., DenseRetriever)
            top_k: Number of documents to retrieve
            alpha: Weight for dense retriever (1-alpha for sparse)
            fusion_method: Fusion method - 'rrf' (reciprocal rank fusion) or 'weighted'
        """
        super().__init__(top_k)
        self.sparse_retriever = sparse_retriever
        self.dense_retriever = dense_retriever
        self.alpha = alpha
        self.fusion_method = fusion_method
        self.documents: List[str] = []
    
    def index(self, documents: List[str], **kwargs) -> None:
        """Index documents using both retrievers."""
        self.documents = documents
        
        # Index with both retrievers
        self.sparse_retriever.index(documents, **kwargs)
        self.dense_retriever.index(documents, **kwargs)
    
    def retrieve(self, query: str, **kwargs) -> List[RetrievalResult]:
        """Retrieve documents using hybrid approach."""
        # Get results from both retrievers
        sparse_results = self.sparse_retriever.retrieve(query, **kwargs)
        dense_results = self.dense_retriever.retrieve(query, **kwargs)
        
        if self.fusion_method == "rrf":
            return self._reciprocal_rank_fusion(sparse_results, dense_results)
        elif self.fusion_method == "weighted":
            return self._weighted_fusion(sparse_results, dense_results)
        else:
            raise ValueError(f"Unknown fusion method: {self.fusion_method}")
    
    def _reciprocal_rank_fusion(
        self,
        sparse_results: List[RetrievalResult],
        dense_results: List[RetrievalResult]
    ) -> List[RetrievalResult]:
        """Reciprocal Rank Fusion algorithm."""
        # Create score maps
        k = 60  # RRF parameter
        
        sparse_scores = {}
        for i, result in enumerate(sparse_results):
            sparse_scores[result.metadata["index"]] = 1.0 / (k + i + 1)
        
        dense_scores = {}
        for i, result in enumerate(dense_results):
            dense_scores[result.metadata["index"]] = 1.0 / (k + i + 1)
        
        # Combine scores
        all_indices = set(sparse_scores.keys()) | set(dense_scores.keys())
        combined_scores = []
        
        for idx in all_indices:
            score = sparse_scores.get(idx, 0) + dense_scores.get(idx, 0)
            combined_scores.append((idx, score))
        
        # Sort by combined score
        combined_scores.sort(key=lambda x: x[1], reverse=True)
        
        # Get top-k results
        results = []
        for idx, score in combined_scores[:self.top_k]:
            results.append(RetrievalResult(
                text=self.documents[idx],
                score=score,
                metadata={"index": idx}
            ))
        
        return results
    
    def _weighted_fusion(
        self,
        sparse_results: List[RetrievalResult],
        dense_results: List[RetrievalResult]
    ) -> List[RetrievalResult]:
        """Weighted score fusion."""
        # Create normalized score maps
        max_sparse = max(r.score for r in sparse_results) if sparse_results else 1
        max_dense = max(r.score for r in dense_results) if dense_results else 1
        
        sparse_scores = {}
        for result in sparse_results:
            idx = result.metadata["index"]
            sparse_scores[idx] = result.score / max_sparse if max_sparse > 0 else 0
        
        dense_scores = {}
        for result in dense_results:
            idx = result.metadata["index"]
            dense_scores[idx] = result.score / max_dense if max_dense > 0 else 0
        
        # Combine scores
        all_indices = set(sparse_scores.keys()) | set(dense_scores.keys())
        combined_scores = []
        
        for idx in all_indices:
            sparse_s = sparse_scores.get(idx, 0)
            dense_s = dense_scores.get(idx, 0)
            score = (1 - self.alpha) * sparse_s + self.alpha * dense_s
            combined_scores.append((idx, score))
        
        # Sort by combined score
        combined_scores.sort(key=lambda x: x[1], reverse=True)
        
        # Get top-k results
        results = []
        for idx, score in combined_scores[:self.top_k]:
            results.append(RetrievalResult(
                text=self.documents[idx],
                score=score,
                metadata={"index": idx}
            ))
        
        return results
