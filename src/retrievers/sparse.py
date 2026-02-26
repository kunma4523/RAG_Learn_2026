"""
Sparse Retrievers
=================

Implementation of sparse retrieval methods (TF-IDF, BM25).
"""

from typing import List, Dict, Any, Optional
import numpy as np
from collections import Counter, defaultdict
import re

from src.retrievers.base import BaseRetriever, RetrievalResult


class TFIDFRetriever(BaseRetriever):
    """TF-IDF based sparse retriever."""
    
    def __init__(self, top_k: int = 5, min_df: int = 1):
        super().__init__(top_k)
        self.min_df = min_df
        self.vocab: Dict[str, int] = {}
        self.idf: np.ndarray = None
        self.doc_term_matrix: np.ndarray = None
        self.documents: List[str] = []
    
    def _tokenize(self, text: str) -> List[str]:
        """Simple tokenization."""
        text = text.lower()
        tokens = re.findall(r'\w+', text)
        return tokens
    
    def _compute_tf(self, tokens: List[str]) -> Counter:
        """Compute term frequencies."""
        return Counter(tokens)
    
    def _compute_idf(self, doc_count: int, doc_freqs: Dict[str, int]) -> np.ndarray:
        """Compute IDF values."""
        idf = np.zeros(len(self.vocab))
        for word, idx in self.vocab.items():
            df = doc_freqs.get(word, 0)
            idf[idx] = np.log((doc_count - df + 0.5) / (df + 0.5) + 1)
        return idf
    
    def index(self, documents: List[str], **kwargs) -> None:
        """Build TF-IDF index."""
        self.documents = documents
        
        # Build vocabulary
        doc_freqs = defaultdict(int)
        for doc in documents:
            tokens = self._tokenize(doc)
            unique_tokens = set(tokens)
            for token in unique_tokens:
                if token not in self.vocab:
                    self.vocab[token] = len(self.vocab)
                doc_freqs[token] += 1
        
        # Filter by min_df
        self.vocab = {w: i for w, i in self.vocab.items() 
                     if doc_freqs[w] >= self.min_df}
        
        # Build document-term matrix
        n_docs = len(documents)
        n_terms = len(self.vocab)
        self.doc_term_matrix = np.zeros((n_docs, n_terms))
        
        for i, doc in enumerate(documents):
            tokens = self._tokenize(doc)
            tf = self._compute_tf(tokens)
            for word, count in tf.items():
                if word in self.vocab:
                    self.doc_term_matrix[i, self.vocab[word]] = count
        
        # Compute IDF
        self.idf = self._compute_idf(n_docs, doc_freqs)
        
        # Normalize
        doc_lengths = np.linalg.norm(self.doc_term_matrix, axis=1, keepdims=True)
        doc_lengths[doc_lengths == 0] = 1
        self.doc_term_matrix = self.doc_term_matrix / doc_lengths
    
    def retrieve(self, query: str, **kwargs) -> List[RetrievalResult]:
        """Retrieve documents using TF-IDF."""
        # Compute query vector
        tokens = self._tokenize(query)
        query_vec = np.zeros(len(self.vocab))
        for token in tokens:
            if token in self.vocab:
                query_vec[self.vocab[token]] += 1
        
        # Normalize
        if np.linalg.norm(query_vec) > 0:
            query_vec = query_vec / np.linalg.norm(query_vec)
        
        # Compute similarities
        scores = np.dot(self.doc_term_matrix, query_vec)
        
        # Get top-k
        top_indices = np.argsort(scores)[::-1][:self.top_k]
        
        results = []
        for idx in top_indices:
            if scores[idx] > 0:
                results.append(RetrievalResult(
                    text=self.documents[idx],
                    score=float(scores[idx]),
                    metadata={"index": int(idx)}
                ))
        
        return results


class BM25Retriever(BaseRetriever):
    """BM25 based sparse retriever."""
    
    def __init__(self, top_k: int = 5, k1: float = 1.5, b: float = 0.75):
        super().__init__(top_k)
        self.k1 = k1
        self.b = b
        self.vocab: Dict[str, int] = {}
        self.avgdl: float = 0
        self.doc_freqs: Dict[int, Dict[str, int]] = {}
        self.doc_lengths: List[int] = []
        self.N: int = 0
        self.doc_term_freqs: List[Counter] = []
        self.documents: List[str] = []
    
    def _tokenize(self, text: str) -> List[str]:
        """Simple tokenization."""
        text = text.lower()
        return re.findall(r'\w+', text)
    
    def index(self, documents: List[str], **kwargs) -> None:
        """Build BM25 index."""
        self.documents = documents
        self.N = len(documents)
        
        for doc in documents:
            tokens = self._tokenize(doc)
            self.doc_lengths.append(len(tokens))
            self.doc_term_freqs.append(Counter(tokens))
        
        self.avgdl = sum(self.doc_lengths) / self.N if self.N > 0 else 0
    
    def _score_bm25(self, query: str, doc_idx: int) -> float:
        """Compute BM25 score for a single document."""
        query_tokens = self._tokenize(query)
        doc_tf = self.doc_term_freqs[doc_idx]
        doc_len = self.doc_lengths[doc_idx]
        
        score = 0.0
        for token in query_tokens:
            if token in doc_tf:
                tf = doc_tf[token]
                # IDF approximation
                df = sum(1 for dtf in self.doc_term_freqs if token in dtf)
                idf = np.log((self.N - df + 0.5) / (df + 0.5) + 1)
                
                # BM25 formula
                numerator = tf * (self.k1 + 1)
                denominator = tf + self.k1 * (1 - self.b + self.b * doc_len / self.avgdl)
                score += idf * numerator / denominator
        
        return score
    
    def retrieve(self, query: str, **kwargs) -> List[RetrievalResult]:
        """Retrieve documents using BM25."""
        scores = [self._score_bm25(query, i) for i in range(self.N)]
        
        # Get top-k
        top_indices = np.argsort(scores)[::-1][:self.top_k]
        
        results = []
        for idx in top_indices:
            if scores[idx] > 0:
                results.append(RetrievalResult(
                    text=self.documents[idx],
                    score=float(scores[idx]),
                    metadata={"index": int(idx)}
                ))
        
        return results
