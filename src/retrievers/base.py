"""
Base Retriever Interface
=========================

Abstract base class for all retriever implementations.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import List, Dict, Any, Optional
import numpy as np


@dataclass
class RetrievalResult:
    """Represents a single retrieval result."""
    
    text: str
    score: float
    metadata: Dict[str, Any]
    
    def __repr__(self) -> str:
        return f"RetrievalResult(text={self.text[:50]}..., score={self.score:.4f})"


class BaseRetriever(ABC):
    """Abstract base class for all retrievers."""
    
    def __init__(self, top_k: int = 5):
        """
        Initialize the retriever.
        
        Args:
            top_k: Number of documents to retrieve
        """
        self.top_k = top_k
    
    @abstractmethod
    def index(self, documents: List[str], **kwargs) -> None:
        """
        Index documents for retrieval.
        
        Args:
            documents: List of document texts to index
            **kwargs: Additional indexing parameters
        """
        pass
    
    @abstractmethod
    def retrieve(self, query: str, **kwargs) -> List[RetrievalResult]:
        """
        Retrieve relevant documents for a query.
        
        Args:
            query: Query string
            **kwargs: Additional retrieval parameters
            
        Returns:
            List of RetrievalResult objects sorted by relevance
        """
        pass
    
    def retrieve_batch(self, queries: List[str], **kwargs) -> List[List[RetrievalResult]]:
        """
        Retrieve documents for multiple queries.
        
        Args:
            queries: List of query strings
            
        Returns:
            List of retrieval results for each query
        """
        return [self.retrieve(q, **kwargs) for q in queries]
    
    def compute_similarity(self, query_embedding: np.ndarray, 
                          doc_embeddings: np.ndarray) -> np.ndarray:
        """
        Compute cosine similarity between query and document embeddings.
        
        Args:
            query_embedding: Query embedding vector
            doc_embeddings: Document embedding matrix
            
        Returns:
            Similarity scores
        """
        # Normalize embeddings
        query_norm = query_embedding / (np.linalg.norm(query_embedding) + 1e-8)
        docs_norm = doc_embeddings / (np.linalg.norm(doc_embeddings, axis=1, keepdims=True) + 1e-8)
        
        # Compute cosine similarity
        return np.dot(docs_norm, query_norm)
    
    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(top_k={self.top_k})"
