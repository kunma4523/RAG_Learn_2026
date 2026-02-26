"""
LLM Generator Implementations
==============================

Various LLM generator implementations.
"""

from typing import List, Dict, Any, Optional
import time

from src.generators.base import BaseGenerator, GenerationResult


class LLMGenerator(BaseGenerator):
    """Generic LLM generator wrapper."""
    
    def __init__(
        self,
        model_name: str,
        client: Any,
        temperature: float = 0.7,
        max_tokens: int = 2048,
        top_p: float = 0.9,
        **kwargs
    ):
        super().__init__(model_name, temperature, max_tokens, top_p, **kwargs)
        self.client = client
    
    def generate(self, prompt: str, **kwargs) -> GenerationResult:
        """Generate text from prompt."""
        start_time = time.time()
        
        try:
            response = self.client.generate(
                prompt=prompt,
                model=self.model_name,
                temperature=kwargs.get("temperature", self.temperature),
                max_tokens=kwargs.get("max_tokens", self.max_tokens),
                top_p=kwargs.get("top_p", self.top_p),
            )
            
            text = response.get("text", "") or response.get("choices", [{}])[0].get("text", "")
            
            return GenerationResult(
                text=text,
                prompt=prompt,
                metadata={
                    "model": self.model_name,
                    "latency": time.time() - start_time,
                    "tokens": response.get("usage", {}).get("total_tokens", 0)
                }
            )
        except Exception as e:
            return GenerationResult(
                text="",
                prompt=prompt,
                metadata={"error": str(e), "latency": time.time() - start_time}
            )
    
    def batch_generate(self, prompts: List[str], **kwargs) -> List[GenerationResult]:
        """Generate text for multiple prompts."""
        return [self.generate(p, **kwargs) for p in prompts]


class OpenAIGenerator(BaseGenerator):
    """OpenAI API generator."""
    
    def __init__(
        self,
        model_name: str = "gpt-4",
        api_key: Optional[str] = None,
        temperature: float = 0.7,
        max_tokens: int = 2048,
        top_p: float = 0.9,
        **kwargs
    ):
        super().__init__(model_name, temperature, max_tokens, top_p, **kwargs)
        
        try:
            from openai import OpenAI
        except ImportError:
            raise ImportError("openai is required. Install with: pip install openai")
        
        self.client = OpenAI(api_key=api_key)
    
    def generate(self, prompt: str, **kwargs) -> GenerationResult:
        """Generate text using OpenAI API."""
        start_time = time.time()
        
        try:
            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=[{"role": "user", "content": prompt}],
                temperature=kwargs.get("temperature", self.temperature),
                max_tokens=kwargs.get("max_tokens", self.max_tokens),
                top_p=kwargs.get("top_p", self.top_p),
            )
            
            text = response.choices[0].message.content or ""
            
            return GenerationResult(
                text=text,
                prompt=prompt,
                metadata={
                    "model": self.model_name,
                    "latency": time.time() - start_time,
                    "tokens": response.usage.total_tokens if response.usage else 0
                }
            )
        except Exception as e:
            return GenerationResult(
                text="",
                prompt=prompt,
                metadata={"error": str(e), "latency": time.time() - start_time}
            )
    
    def batch_generate(self, prompts: List[str], **kwargs) -> List[GenerationResult]:
        """Generate text for multiple prompts."""
        return [self.generate(p, **kwargs) for p in prompts]


class HuggingFaceGenerator(BaseGenerator):
    """Hugging Face local/generative model generator."""
    
    def __init__(
        self,
        model_name: str = "Qwen/Qwen2-7B-Instruct",
        temperature: float = 0.7,
        max_tokens: int = 2048,
        top_p: float = 0.9,
        device: Optional[str] = None,
        **kwargs
    ):
        super().__init__(model_name, temperature, max_tokens, top_p, **kwargs)
        
        self.device = device or "cuda" if __import__("torch").cuda.is_available() else "cpu"
        self._model = None
        self._tokenizer = None
    
    def _load_model(self):
        """Lazy load the model and tokenizer."""
        if self._model is None:
            try:
                import torch
                from transformers import AutoModelForCausalLM, AutoTokenizer
                
                self._tokenizer = AutoTokenizer.from_pretrained(
                    self.model_name,
                    trust_remote_code=True
                )
                
                self._model = AutoModelForCausalLM.from_pretrained(
                    self.model_name,
                    torch_dtype=torch.float16 if self.device == "cuda" else torch.float32,
                    device_map="auto" if self.device == "cuda" else None,
                    trust_remote_code=True
                )
                
                if self.device == "cpu":
                    self._model = self._model.to(self.device)
                    
            except ImportError:
                raise ImportError("transformers and torch are required. Install with: pip install transformers torch")
    
    def generate(self, prompt: str, **kwargs) -> GenerationResult:
        """Generate text using Hugging Face model."""
        start_time = time.time()
        
        try:
            self._load_model()
            
            # Format prompt for chat models
            if self._tokenizer.chat_template:
                messages = [{"role": "user", "content": prompt}]
                formatted_prompt = self._tokenizer.apply_chat_template(
                    messages,
                    tokenize=False,
                    add_generation_prompt=True
                )
            else:
                formatted_prompt = prompt
            
            inputs = self._tokenizer(
                formatted_prompt,
                return_tensors="pt"
            ).to(self.device)
            
            with __import__("torch").no_grad():
                outputs = self._model.generate(
                    **inputs,
                    temperature=kwargs.get("temperature", self.temperature),
                    max_new_tokens=kwargs.get("max_tokens", self.max_tokens),
                    top_p=kwargs.get("top_p", self.top_p),
                    do_sample=True,
                )
            
            # Decode output
            generated_text = self._tokenizer.decode(
                outputs[0][inputs.input_ids.shape[1]:],
                skip_special_tokens=True
            )
            
            return GenerationResult(
                text=generated_text,
                prompt=prompt,
                metadata={
                    "model": self.model_name,
                    "latency": time.time() - start_time,
                    "device": self.device
                }
            )
        except Exception as e:
            return GenerationResult(
                text="",
                prompt=prompt,
                metadata={"error": str(e), "latency": time.time() - start_time}
            )
    
    def batch_generate(self, prompts: List[str], **kwargs) -> List[GenerationResult]:
        """Generate text for multiple prompts."""
        # For efficiency, could implement batch generation
        return [self.generate(p, **kwargs) for p in prompts]
