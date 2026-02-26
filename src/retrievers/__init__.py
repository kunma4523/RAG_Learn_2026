"""
Retrievers Module
=================

This module provides various retriever implementations for RAG systems.

Available Retrievers:
- BaseRetriever: Abstract base class for all retrievers
- SparseRetriever: TF-IDF, BM25 based sparse retrieval
- DenseRetriever: Dense vector based retrieval (DPR, Contriever)
- HybridRetriever: Combines sparse and dense retrieval
- GraphRetriever: Knowledge graph based retrieval
"""

from src.retrievers.base import BaseRetriever
from src.retrievers.sparse import BM25Retriever, TFIDFRetriever
from src.retrievers.dense import DenseRetriever, SentenceTransformerRetriever
from src.retrievers.hybrid import HybridRetriever
from src.retrievers.graph import GraphRetriever

__all__ = [
    "BaseRetriever",
    "BM25Retriever",
    "TFIDFRetriever", 
    "DenseRetriever",
    "SentenceTransformerRetriever",
    "HybridRetriever",
    "GraphRetriever",
]
