"""
Hybrid RAG Pipeline
=================

Combines sparse and dense retrieval with reranking.
"""

from typing import List, Dict, Any, Optional
import time

from src.pipelines.base import BasePipeline, PipelineResult
from src.retrievers.hybrid import HybridRetriever
from src.retrievers.base import BaseRetriever


class HybridRAGPipeline(BasePipeline):
    """
    Hybrid RAG Pipeline.
    
    Combines:
    1. Sparse retrieval (BM25/TF-IDF)
    2. Dense retrieval (Embedding-based)
    3. Reranking (optional)
    
    Flow: Query → Sparse Ret → Dense Ret → Fusion → Rerank → Generate
    """
    
    def __init__(
        self,
        retriever: HybridRetriever,
        generator: Any,
        top_k: int = 10,
        final_k: int = 5,
        use_reranker: bool = True,
        **kwargs
    ):
        super().__init__(retriever, generator, top_k)
        
        self.final_k = final_k
        self.use_reranker = use_reranker
        self._reranker = None
    
    def _create_reranker(self):
        """Create a simple reranker."""
        try:
            from sentence_transformers import CrossEncoder
            self._reranker = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")
        except ImportError:
            self._reranker = None
    
    def _rerank(self, query: str, documents: List[str]) -> List[str]:
        """Rerank documents using cross-encoder."""
        if not self.use_reranker or self._reranker is None:
            return documents[:self.final_k]
        
        # Create pairs
        pairs = [(query, doc) for doc in documents]
        
        # Get scores
        scores = self._reranker.predict(pairs)
        
        # Sort by score
        sorted_indices = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)
        
        return [documents[i] for i in sorted_indices[:self.final_k]]
    
    def index_documents(
        self,
        documents: List[str],
        **kwargs
    ) -> None:
        """Index documents for retrieval."""
        if self.use_reranker and self._reranker is None:
            self._create_reranker()
        
        self.retriever.index(documents, **kwargs)
        self.indexed = True
    
    def query(
        self,
        query: str,
        top_k: Optional[int] = None,
        **kwargs
    ) -> PipelineResult:
        """
        Process query through hybrid pipeline.
        
        Args:
            query: User query
            top_k: Number of docs to retrieve before reranking
            
        Returns:
            PipelineResult with answer
        """
        k = top_k or self.top_k
        
        # Step 1: Hybrid Retrieval
        retrieval_start = time.time()
        retrieved_docs = self.retriever.retrieve(query, top_k=k)
        
        # Get all candidate documents
        candidate_docs = [doc.text for doc in retrieved_docs]
        retrieval_time = time.time() - retrieval_start
        
        # Step 2: Reranking
        rerank_start = time.time()
        final_docs = self._rerank(query, candidate_docs)
        rerank_time = time.time() - rerank_start
        
        # Step 3: Generation
        generation_start = time.time()
        prompt = self.generator.create_prompt(query, final_docs)
        generation_result = self.generator.generate(prompt, **kwargs)
        generation_time = time.time() - generation_start
        
        return PipelineResult(
            query=query,
            answer=generation_result.text,
            retrieved_documents=retrieved_docs[:self.final_k],
            generation_result=generation_result,
            metadata={
                "retrieval_time": retrieval_time,
                "rerank_time": rerank_time,
                "generation_time": generation_time,
                "total_time": retrieval_time + rerank_time + generation_time,
                "method": "hybrid"
            }
        )
    
    def __repr__(self) -> str:
        return f"HybridRAGPipeline(use_reranker={self.use_reranker}, top_k={self.top_k})"
