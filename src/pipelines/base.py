"""
Base Pipeline Interface
=======================

Abstract base class for all RAG pipelines.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional
import time

from src.retrievers.base import BaseRetriever, RetrievalResult
from src.generators.base import BaseGenerator, GenerationResult


@dataclass
class PipelineResult:
    """Represents a complete RAG pipeline result."""
    
    query: str
    answer: str
    retrieved_documents: List[RetrievalResult] = field(default_factory=list)
    generation_result: Optional[GenerationResult] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def __repr__(self) -> str:
        return f"PipelineResult(query={self.query[:30]}..., answer={self.answer[:50]}...)"


class BasePipeline(ABC):
    """Abstract base class for all RAG pipelines."""
    
    def __init__(
        self,
        retriever: BaseRetriever,
        generator: BaseGenerator,
        top_k: int = 5
    ):
        """
        Initialize the pipeline.
        
        Args:
            retriever: Retriever instance
            generator: Generator instance
            top_k: Number of documents to retrieve
        """
        self.retriever = retriever
        self.generator = generator
        self.top_k = top_k
        self.indexed = False
    
    @abstractmethod
    def index_documents(
        self,
        documents: List[str],
        **kwargs
    ) -> None:
        """
        Index documents for retrieval.
        
        Args:
            documents: List of document texts
            **kwargs: Additional parameters
        """
        pass
    
    @abstractmethod
    def query(self, query: str, **kwargs) -> PipelineResult:
        """
        Process a query through the pipeline.
        
        Args:
            query: User query
            **kwargs: Additional parameters
            
        Returns:
            PipelineResult with answer and metadata
        """
        pass
    
    def retrieve(self, query: str) -> List[RetrievalResult]:
        """
        Retrieve documents for a query.
        
        Args:
            query: User query
            
        Returns:
            List of retrieved documents
        """
        return self.retriever.retrieve(query, top_k=self.top_k)
    
    def generate(
        self,
        query: str,
        context: List[str],
        **kwargs
    ) -> GenerationResult:
        """
        Generate answer from query and context.
        
        Args:
            query: User query
            context: Retrieved context documents
            
        Returns:
            GenerationResult
        """
        prompt = self.generator.create_prompt(query, context)
        return self.generator.generate(prompt, **kwargs)
    
    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(retriever={self.retriever}, generator={self.generator})"
