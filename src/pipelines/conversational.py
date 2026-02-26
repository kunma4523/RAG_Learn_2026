"""
Conversational RAG Pipeline
==========================

RAG pipeline with conversation history support.
"""

from typing import List, Dict, Any, Optional
import time

from src.pipelines.base import BasePipeline, PipelineResult
from src.generators.chat import ChatGenerator
from src.retrievers.base import BaseRetriever


class ConversationalRAGPipeline(BasePipeline):
    """
    Conversational RAG Pipeline.
    
    Extends Standard RAG with conversation history support.
    Maintains dialogue context across multiple turns.
    
    Flow: Query + History → Retrieval → Augmentation → Generation
    """
    
    def __init__(
        self,
        retriever: BaseRetriever,
        generator: ChatGenerator,
        top_k: int = 5,
        max_history: int = 10,
        **kwargs
    ):
        super().__init__(retriever, generator, top_k)
        
        self.max_history = max_history
        self.conversation_history: List[Dict[str, str]] = []
    
    def index_documents(
        self,
        documents: List[str],
        **kwargs
    ) -> None:
        """Index documents for retrieval."""
        self.retriever.index(documents, **kwargs)
        self.indexed = True
    
    def _build_context_query(self, query: str) -> str:
        """Build query string with conversation history."""
        # Extract key information from history
        if not self.conversation_history:
            return query
        
        # Use recent history to enrich query
        recent_turns = self.conversation_history[-self.max_history:]
        
        context_parts = [query]
        for turn in recent_turns:
            if turn.get("role") == "user":
                context_parts.append(f"Previous question: {turn.get('content', '')}")
        
        return " | ".join(context_parts)
    
    def query(
        self,
        query: str,
        top_k: Optional[int] = None,
        include_history: bool = True,
        **kwargs
    ) -> PipelineResult:
        """
        Process a query with conversation history.
        
        Args:
            query: User query
            top_k: Override number of documents to retrieve
            include_history: Whether to include history in retrieval
            
        Returns:
            PipelineResult with answer
        """
        k = top_k or self.top_k
        
        # Build query with history context
        if include_history:
            search_query = self._build_context_query(query)
        else:
            search_query = query
        
        # Step 1: Retrieval
        retrieval_start = time.time()
        retrieved_docs = self.retriever.retrieve(search_query, top_k=k)
        context = [doc.text for doc in retrieved_docs]
        retrieval_time = time.time() - retrieval_start
        
        # Step 2: Generation with chat history
        generation_start = time.time()
        
        # Add user message to chat history
        self.generator.add_message("user", query)
        
        # Generate response with context
        generation_result = self.generator.generate(
            prompt=query,
            context=context,
            **kwargs
        )
        
        generation_time = time.time() - generation_start
        
        # Add assistant response to history
        if generation_result.text:
            self.generator.add_message("assistant", generation_result.text)
        
        return PipelineResult(
            query=query,
            answer=generation_result.text,
            retrieved_documents=retrieved_docs,
            generation_result=generation_result,
            metadata={
                "retrieval_time": retrieval_time,
                "generation_time": generation_time,
                "total_time": retrieval_time + generation_time,
                "history_length": len(self.conversation_history)
            }
        )
    
    def clear_history(self) -> None:
        """Clear conversation history."""
        self.conversation_history = []
        self.generator.clear_history()
    
    def __repr__(self) -> str:
        return f"ConversationalRAGPipeline(history_len={len(self.conversation_history)})"
