"""
Base Generator Interface
=========================

Abstract base class for all LLM generators.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import List, Dict, Any, Optional, Union
import time


@dataclass
class GenerationResult:
    """Represents a generation result."""
    
    text: str
    prompt: str
    metadata: Dict[str, Any]
    
    def __repr__(self) -> str:
        return f"GenerationResult(text={self.text[:50]}..., metadata={self.metadata})"


class BaseGenerator(ABC):
    """Abstract base class for all generators."""
    
    def __init__(
        self,
        model_name: str,
        temperature: float = 0.7,
        max_tokens: int = 2048,
        top_p: float = 0.9,
        **kwargs
    ):
        """
        Initialize the generator.
        
        Args:
            model_name: Name of the model to use
            temperature: Sampling temperature
            max_tokens: Maximum tokens to generate
            top_p: Nucleus sampling parameter
            **kwargs: Additional parameters
        """
        self.model_name = model_name
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.top_p = top_p
        self.extra_params = kwargs
    
    @abstractmethod
    def generate(self, prompt: str, **kwargs) -> GenerationResult:
        """
        Generate text from prompt.
        
        Args:
            prompt: Input prompt
            **kwargs: Additional generation parameters
            
        Returns:
            GenerationResult object
        """
        pass
    
    @abstractmethod
    def batch_generate(self, prompts: List[str], **kwargs) -> List[GenerationResult]:
        """
        Generate text for multiple prompts.
        
        Args:
            prompts: List of input prompts
            **kwargs: Additional generation parameters
            
        Returns:
            List of GenerationResult objects
        """
        pass
    
    def create_prompt(
        self,
        query: str,
        context: List[str],
        system_prompt: Optional[str] = None,
        template: Optional[str] = None
    ) -> str:
        """
        Create a formatted prompt with context.
        
        Args:
            query: User query
            context: Retrieved context documents
            system_prompt: Optional system prompt
            template: Optional custom template
            
        Returns:
            Formatted prompt string
        """
        if template:
            return template.format(query=query, context="\n\n".join(context))
        
        # Default template
        context_str = "\n\n".join([
            f"[Document {i+1}]\n{doc}"
            for i, doc in enumerate(context)
        ])
        
        if system_prompt:
            prompt = f"{system_prompt}\n\nContext:\n{context_str}\n\nQuestion: {query}\n\nAnswer:"
        else:
            prompt = f"Context:\n{context_str}\n\nQuestion: {query}\n\nAnswer:"
        
        return prompt
    
    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(model={self.model_name})"
