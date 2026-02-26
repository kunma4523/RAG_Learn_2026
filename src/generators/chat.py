"""
Chat Generator
=============

Generator with conversation history support.
"""

from typing import List, Dict, Any, Optional
from dataclasses import dataclass, field

from src.generators.base import BaseGenerator, GenerationResult


@dataclass
class Message:
    """Represents a chat message."""
    
    role: str  # "user", "assistant", "system"
    content: str
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, str]:
        return {"role": self.role, "content": self.content}


class ChatGenerator(BaseGenerator):
    """Generator with conversation history support."""
    
    def __init__(
        self,
        model_name: str,
        temperature: float = 0.7,
        max_tokens: int = 2048,
        top_p: float = 0.9,
        system_prompt: Optional[str] = None,
        max_history: int = 10,
        **kwargs
    ):
        super().__init__(model_name, temperature, max_tokens, top_p, **kwargs)
        
        self.system_prompt = system_prompt or "You are a helpful assistant."
        self.max_history = max_history
        self.conversation_history: List[Message] = []
        self._client = None
    
    @property
    def client(self):
        """Get or create the underlying client."""
        if self._client is None:
            # Try to detect the appropriate client
            try:
                from openai import OpenAI
                self._client = OpenAI()
            except ImportError:
                pass
        return self._client
    
    @client.setter
    def client(self, value):
        self._client = value
    
    def add_message(self, role: str, content: str, metadata: Optional[Dict] = None) -> None:
        """Add a message to conversation history."""
        message = Message(role=role, content=content, metadata=metadata or {})
        self.conversation_history.append(message)
        
        # Trim history if needed
        if len(self.conversation_history) > self.max_history:
            # Keep system prompt and recent messages
            system_msg = self.conversation_history[0] if self.conversation_history[0].role == "system" else None
            self.conversation_history = (
                [system_msg] if system_msg else []
            ) + self.conversation_history[-(self.max_history):]
    
    def clear_history(self) -> None:
        """Clear conversation history."""
        # Keep system prompt if exists
        if self.conversation_history and self.conversation_history[0].role == "system":
            self.conversation_history = [self.conversation_history[0]]
        else:
            self.conversation_history = []
    
    def _build_messages(
        self,
        context: Optional[List[str]] = None,
        user_query: Optional[str] = None
    ) -> List[Dict[str, str]]:
        """Build message list for API call."""
        messages = []
        
        # Add system prompt
        messages.append({"role": "system", "content": self.system_prompt})
        
        # Add conversation history
        for msg in self.conversation_history:
            if msg.role != "system":
                messages.append(msg.to_dict())
        
        # If there's context and a query, add them
        if context and user_query:
            context_str = "\n\n".join([
                f"[Document {i+1}]\n{doc}"
                for i, doc in enumerate(context)
            ])
            messages.append({
                "role": "user",
                "content": f"Context:\n{context_str}\n\nQuestion: {user_query}\n\nAnswer:"
            })
        elif user_query:
            messages.append({"role": "user", "content": user_query})
        
        return messages
    
    def generate(
        self,
        prompt: str,
        context: Optional[List[str]] = None,
        **kwargs
    ) -> GenerationResult:
        """Generate response with conversation context."""
        import time
        start_time = time.time()
        
        messages = self._build_messages(context=context, user_query=prompt)
        
        try:
            if self.client:
                # Use OpenAI API
                response = self.client.chat.completions.create(
                    model=self.model_name,
                    messages=messages,
                    temperature=kwargs.get("temperature", self.temperature),
                    max_tokens=kwargs.get("max_tokens", self.max_tokens),
                    top_p=kwargs.get("top_p", self.top_p),
                )
                
                text = response.choices[0].message.content or ""
                
                # Add to history
                self.add_message("user", prompt)
                self.add_message("assistant", text)
                
                return GenerationResult(
                    text=text,
                    prompt=prompt,
                    metadata={
                        "model": self.model_name,
                        "latency": time.time() - start_time,
                        "tokens": response.usage.total_tokens if response.usage else 0
                    }
                )
            else:
                # Fallback - return prompt as response
                return GenerationResult(
                    text="No LLM client configured.",
                    prompt=prompt,
                    metadata={"error": "No client", "latency": time.time() - start_time}
                )
                
        except Exception as e:
            return GenerationResult(
                text="",
                prompt=prompt,
                metadata={"error": str(e), "latency": time.time() - start_time}
            )
    
    def chat(
        self,
        user_message: str,
        context: Optional[List[str]] = None,
        **kwargs
    ) -> GenerationResult:
        """
        Convenience method for chat interactions.
        
        Args:
            user_message: User's message
            context: Optional context (retrieved documents)
            
        Returns:
            GenerationResult with response
        """
        result = self.generate(
            prompt=user_message,
            context=context,
            **kwargs
        )
        
        return result
    
    def batch_generate(self, prompts: List[str], **kwargs) -> List[GenerationResult]:
        """Generate responses for multiple prompts."""
        return [self.generate(p, **kwargs) for p in prompts]
    
    def __repr__(self) -> str:
        return f"ChatGenerator(model={self.model_name}, history_len={len(self.conversation_history)})"
