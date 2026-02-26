"""
Standard RAG Pipeline
====================

Basic RAG pipeline implementation.
"""

from typing import List, Dict, Any, Optional, Union
import time

from src.pipelines.base import BasePipeline, PipelineResult
from src.retrievers.base import BaseRetriever, RetrievalResult
from src.generators.base import BaseGenerator, GenerationResult
from src.retrievers.dense import SentenceTransformerRetriever
from src.generators.llm import OpenAIGenerator, HuggingFaceGenerator


class StandardRAGPipeline(BasePipeline):
    """
    Standard RAG Pipeline.
    
    Flow: Query → Retrieval → Augmentation → Generation
    
    This is the basic RAG implementation where:
    1. User query is used to retrieve relevant documents
    2. Retrieved documents are concatenated as context
    3. LLM generates answer based on query + context
    """
    
    def __init__(
        self,
        retriever: Optional[BaseRetriever] = None,
        generator: Optional[BaseGenerator] = None,
        top_k: int = 5,
        embedding_model: str = "BAAI/bge-base-zh-v1.5",
        llm_model: str = "gpt-4",
        **kwargs
    ):
        """
        Initialize Standard RAG Pipeline.
        
        Args:
            retriever: Optional retriever (auto-created if not provided)
            generator: Optional generator (auto-created if not provided)
            top_k: Number of documents to retrieve
            embedding_model: Name of embedding model
            llm_model: Name of LLM model
            **kwargs: Additional parameters
        """
        # Auto-create retriever if not provided
        if retriever is None:
            retriever = SentenceTransformerRetriever(
                model_name=embedding_model,
                top_k=top_k
            )
        
        # Auto-create generator if not provided
        if generator is None:
            # Try to detect best generator
            model_lower = llm_model.lower()
            if "gpt" in model_lower or "claude" in model_lower:
                generator = OpenAIGenerator(model_name=llm_model)
            else:
                generator = HuggingFaceGenerator(model_name=llm_model)
        
        super().__init__(retriever, generator, top_k)
        
        self.embedding_model = embedding_model
        self.llm_model = llm_model
    
    def index_documents(
        self,
        documents: List[str],
        batch_size: int = 32,
        show_progress: bool = True,
        **kwargs
    ) -> None:
        """
        Index documents for retrieval.
        
        Args:
            documents: List of document texts
            batch_size: Batch size for encoding
            show_progress: Whether to show progress
        """
        if show_progress:
            print(f"Indexing {len(documents)} documents...")
        
        start_time = time.time()
        
        self.retriever.index(documents, batch_size=batch_size, **kwargs)
        self.indexed = True
        
        if show_progress:
            print(f"Indexed in {time.time() - start_time:.2f}s")
    
    def index_from_files(
        self,
        file_paths: List[str],
        show_progress: bool = True,
        **kwargs
    ) -> None:
        """
        Index documents from files.
        
        Args:
            file_paths: List of file paths
            show_progress: Whether to show progress
        """
        from src.utils.document_loader import load_documents
        
        if show_progress:
            print(f"Loading {len(file_paths)} files...")
        
        documents = load_documents(file_paths)
        
        if show_progress:
            print(f"Loaded {len(documents)} documents")
        
        self.index_documents(documents, show_progress=show_progress, **kwargs)
    
    def query(
        self,
        query: str,
        top_k: Optional[int] = None,
        return_context: bool = True,
        **kwargs
    ) -> PipelineResult:
        """
        Process a query through the RAG pipeline.
        
        Args:
            query: User query
            top_k: Override number of documents to retrieve
            return_context: Whether to include context in result
            
        Returns:
            PipelineResult with answer and metadata
        """
        k = top_k or self.top_k
        
        # Step 1: Retrieval
        retrieval_start = time.time()
        retrieved_docs = self.retriever.retrieve(query, top_k=k)
        
        # Extract context
        context = [doc.text for doc in retrieved_docs]
        
        retrieval_time = time.time() - retrieval_start
        
        # Step 2: Generation
        generation_start = time.time()
        prompt = self.generator.create_prompt(query, context)
        generation_result = self.generator.generate(prompt, **kwargs)
        
        generation_time = time.time() - generation_start
        
        return PipelineResult(
            query=query,
            answer=generation_result.text,
            retrieved_documents=retrieved_docs,
            generation_result=generation_result,
            metadata={
                "retrieval_time": retrieval_time,
                "generation_time": generation_time,
                "total_time": retrieval_time + generation_time,
                "num_retrieved": len(retrieved_docs),
                "model": self.llm_model
            }
        )
    
    def query_batch(
        self,
        queries: List[str],
        **kwargs
    ) -> List[PipelineResult]:
        """
        Process multiple queries.
        
        Args:
            queries: List of user queries
            
        Returns:
            List of PipelineResults
        """
        return [self.query(q, **kwargs) for q in queries]
    
    def get_relevant_documents(
        self,
        query: str,
        top_k: Optional[int] = None
    ) -> List[RetrievalResult]:
        """
        Get relevant documents without generating answer.
        
        Args:
            query: User query
            top_k: Number of documents
            
        Returns:
            List of retrieved documents
        """
        k = top_k or self.top_k
        return self.retriever.retrieve(query, top_k=k)
    
    def __repr__(self) -> str:
        return f"StandardRAGPipeline(embedding={self.embedding_model}, llm={self.llm_model}, top_k={self.top_k})"
