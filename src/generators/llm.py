"""
LLM 生成器实现
================

各种大语言模型(LLM)生成器的实现。
支持 OpenAI、Anthropic、Google、阿里云 Qwen 等远端 API，
以及本地 HuggingFace 模型。
"""

from typing import List, Dict, Any, Optional
import time

from src.generators.base import BaseGenerator, GenerationResult
from src.utils.config import get_llm_config, get_llm_provider


class LLMGenerator(BaseGenerator):
    """通用 LLM 生成器封装"""

    def __init__(
        self,
        model_name: str,
        client: Any,
        temperature: float = 0.7,
        max_tokens: int = 2048,
        top_p: float = 0.9,
        **kwargs,
    ):
        super().__init__(model_name, temperature, max_tokens, top_p, **kwargs)
        self.client = client

    def generate(self, prompt: str, **kwargs) -> GenerationResult:
        """根据提示生成文本"""
        start_time = time.time()

        try:
            response = self.client.generate(
                prompt=prompt,
                model=self.model_name,
                temperature=kwargs.get("temperature", self.temperature),
                max_tokens=kwargs.get("max_tokens", self.max_tokens),
                top_p=kwargs.get("top_p", self.top_p),
            )

            text = response.get("text", "") or response.get("choices", [{}])[0].get(
                "text", ""
            )

            return GenerationResult(
                text=text,
                prompt=prompt,
                metadata={
                    "model": self.model_name,
                    "latency": time.time() - start_time,
                    "tokens": response.get("usage", {}).get("total_tokens", 0),
                },
            )
        except Exception as e:
            return GenerationResult(
                text="",
                prompt=prompt,
                metadata={"error": str(e), "latency": time.time() - start_time},
            )

    def batch_generate(self, prompts: List[str], **kwargs) -> List[GenerationResult]:
        """为多个提示生成文本"""
        return [self.generate(p, **kwargs) for p in prompts]


class OpenAIGenerator(BaseGenerator):
    """OpenAI API 生成器"""

    def __init__(
        self,
        model_name: Optional[str] = None,
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        temperature: float = 0.7,
        max_tokens: int = 2048,
        top_p: float = 0.9,
        **kwargs,
    ):
        # 如果没有提供参数，从环境变量读取
        config = get_llm_config()

        model_name = model_name or config.get("model", "gpt-4o-mini")
        api_key = api_key or config.get("api_key")
        base_url = base_url or config.get("base_url", "https://api.openai.com/v1")

        super().__init__(model_name, temperature, max_tokens, top_p, **kwargs)

        try:
            from openai import OpenAI
        except ImportError:
            raise ImportError("openai is required. Install with: pip install openai")

        if not api_key:
            raise ValueError("OpenAI API key is required. Set OPENAI_API_KEY in .env")

        self.client = OpenAI(api_key=api_key, base_url=base_url)

    def generate(self, prompt: str, **kwargs) -> GenerationResult:
        """使用 OpenAI API 生成文本"""
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
                    "tokens": response.usage.total_tokens if response.usage else 0,
                },
            )
        except Exception as e:
            return GenerationResult(
                text="",
                prompt=prompt,
                metadata={"error": str(e), "latency": time.time() - start_time},
            )

    def batch_generate(self, prompts: List[str], **kwargs) -> List[GenerationResult]:
        """为多个提示生成文本"""
        return [self.generate(p, **kwargs) for p in prompts]


class AnthropicGenerator(BaseGenerator):
    """Anthropic Claude API 生成器"""

    def __init__(
        self,
        model_name: Optional[str] = None,
        api_key: Optional[str] = None,
        temperature: float = 0.7,
        max_tokens: int = 2048,
        top_p: float = 0.9,
        **kwargs,
    ):
        # 从环境变量读取配置
        config = get_llm_config()

        model_name = model_name or config.get("model", "claude-3-haiku-20240307")
        api_key = api_key or config.get("api_key")

        super().__init__(model_name, temperature, max_tokens, top_p, **kwargs)

        try:
            from anthropic import Anthropic
        except ImportError:
            raise ImportError(
                "anthropic is required. Install with: pip install anthropic"
            )

        if not api_key:
            raise ValueError(
                "Anthropic API key is required. Set ANTHROPIC_API_KEY in .env"
            )

        self.client = Anthropic(api_key=api_key)

    def generate(self, prompt: str, **kwargs) -> GenerationResult:
        """使用 Anthropic API 生成文本"""
        start_time = time.time()

        try:
            response = self.client.messages.create(
                model=self.model_name,
                messages=[{"role": "user", "content": prompt}],
                temperature=kwargs.get("temperature", self.temperature),
                max_tokens=kwargs.get("max_tokens", self.max_tokens),
                top_p=kwargs.get("top_p", self.top_p),
            )

            text = response.content[0].text if response.content else ""

            return GenerationResult(
                text=text,
                prompt=prompt,
                metadata={
                    "model": self.model_name,
                    "latency": time.time() - start_time,
                    "tokens": response.usage.input_tokens + response.usage.output_tokens
                    if response.usage
                    else 0,
                },
            )
        except Exception as e:
            return GenerationResult(
                text="",
                prompt=prompt,
                metadata={"error": str(e), "latency": time.time() - start_time},
            )

    def batch_generate(self, prompts: List[str], **kwargs) -> List[GenerationResult]:
        """为多个提示生成文本"""
        return [self.generate(p, **kwargs) for p in prompts]


class GoogleGenerator(BaseGenerator):
    """Google Gemini API 生成器"""

    def __init__(
        self,
        model_name: Optional[str] = None,
        api_key: Optional[str] = None,
        temperature: float = 0.7,
        max_tokens: int = 2048,
        top_p: float = 0.9,
        **kwargs,
    ):
        # 从环境变量读取配置
        config = get_llm_config()

        model_name = model_name or config.get("model", "gemini-1.5-flash")
        api_key = api_key or config.get("api_key")

        super().__init__(model_name, temperature, max_tokens, top_p, **kwargs)

        try:
            import google.genai as genai
        except ImportError:
            raise ImportError(
                "google-genai is required. Install with: pip install google-genai"
            )

        if not api_key:
            raise ValueError("Google API key is required. Set GOOGLE_API_KEY in .env")

        self.client = genai.Client(api_key=api_key)

    def generate(self, prompt: str, **kwargs) -> GenerationResult:
        """使用 Google Gemini API 生成文本"""
        start_time = time.time()

        try:
            response = self.client.models.generate_content(
                model=self.model_name,
                contents=prompt,
                config={
                    "temperature": kwargs.get("temperature", self.temperature),
                    "max_output_tokens": kwargs.get("max_tokens", self.max_tokens),
                    "top_p": kwargs.get("top_p", self.top_p),
                },
            )

            text = response.text or ""

            return GenerationResult(
                text=text,
                prompt=prompt,
                metadata={
                    "model": self.model_name,
                    "latency": time.time() - start_time,
                },
            )
        except Exception as e:
            return GenerationResult(
                text="",
                prompt=prompt,
                metadata={"error": str(e), "latency": time.time() - start_time},
            )

    def batch_generate(self, prompts: List[str], **kwargs) -> List[GenerationResult]:
        """为多个提示生成文本"""
        return [self.generate(p, **kwargs) for p in prompts]


class QwenGenerator(BaseGenerator):
    """阿里云 Qwen API 生成器"""

    def __init__(
        self,
        model_name: Optional[str] = None,
        api_key: Optional[str] = None,
        temperature: float = 0.7,
        max_tokens: int = 2048,
        top_p: float = 0.9,
        **kwargs,
    ):
        # 从环境变量读取配置
        config = get_llm_config()

        model_name = model_name or config.get("model", "qwen-turbo")
        api_key = api_key or config.get("api_key")

        super().__init__(model_name, temperature, max_tokens, top_p, **kwargs)

        try:
            import dashscope
        except ImportError:
            raise ImportError(
                "dashscope is required. Install with: pip install dashscope"
            )

        if not api_key:
            raise ValueError(
                "DashScope API key is required. Set DASHSCOPE_API_KEY in .env"
            )

        dashscope.api_key = api_key
        self.client = dashscope

    def generate(self, prompt: str, **kwargs) -> GenerationResult:
        """使用阿里云 Qwen API 生成文本"""
        start_time = time.time()

        try:
            response = self.client.Generation.call(
                model=self.model_name,
                prompt=prompt,
                temperature=kwargs.get("temperature", self.temperature),
                max_tokens=kwargs.get("max_tokens", self.max_tokens),
                top_p=kwargs.get("top_p", self.top_p),
            )

            if response.status_code == 200:
                text = response.output.text or ""
                return GenerationResult(
                    text=text,
                    prompt=prompt,
                    metadata={
                        "model": self.model_name,
                        "latency": time.time() - start_time,
                        "tokens": response.usage.total_tokens if response.usage else 0,
                    },
                )
            else:
                return GenerationResult(
                    text="",
                    prompt=prompt,
                    metadata={
                        "error": response.message,
                        "latency": time.time() - start_time,
                    },
                )
        except Exception as e:
            return GenerationResult(
                text="",
                prompt=prompt,
                metadata={"error": str(e), "latency": time.time() - start_time},
            )

    def batch_generate(self, prompts: List[str], **kwargs) -> List[GenerationResult]:
        """为多个提示生成文本"""
        return [self.generate(p, **kwargs) for p in prompts]


class HuggingFaceGenerator(BaseGenerator):
    """HuggingFace 本地/生成模型生成器"""

    def __init__(
        self,
        model_name: str = "Qwen/Qwen2-7B-Instruct",
        temperature: float = 0.7,
        max_tokens: int = 2048,
        top_p: float = 0.9,
        device: Optional[str] = None,
        **kwargs,
    ):
        # 从环境变量读取配置
        config = get_llm_config()

        model_name = config.get("model", model_name)
        device = device or config.get(
            "device", "cuda" if __import__("torch").cuda.is_available() else "cpu"
        )

        super().__init__(model_name, temperature, max_tokens, top_p, **kwargs)

        self.device = device
        self._model = None
        self._tokenizer = None

    def _load_model(self):
        """懒加载模型和分词器"""
        if self._model is None:
            try:
                import torch
                from transformers import AutoModelForCausalLM, AutoTokenizer

                self._tokenizer = AutoTokenizer.from_pretrained(
                    self.model_name, trust_remote_code=True
                )

                self._model = AutoModelForCausalLM.from_pretrained(
                    self.model_name,
                    torch_dtype=torch.float16
                    if self.device == "cuda"
                    else torch.float32,
                    device_map="auto" if self.device == "cuda" else None,
                    trust_remote_code=True,
                )

                if self.device == "cpu":
                    self._model = self._model.to(self.device)

            except ImportError:
                raise ImportError(
                    "transformers and torch are required. Install with: pip install transformers torch"
                )

    def generate(self, prompt: str, **kwargs) -> GenerationResult:
        """使用 HuggingFace 模型生成文本"""
        start_time = time.time()

        try:
            self._load_model()

            # 为聊天模型格式化提示
            if self._tokenizer.chat_template:
                messages = [{"role": "user", "content": prompt}]
                formatted_prompt = self._tokenizer.apply_chat_template(
                    messages, tokenize=False, add_generation_prompt=True
                )
            else:
                formatted_prompt = prompt

            inputs = self._tokenizer(formatted_prompt, return_tensors="pt").to(
                self.device
            )

            with __import__("torch").no_grad():
                outputs = self._model.generate(
                    **inputs,
                    temperature=kwargs.get("temperature", self.temperature),
                    max_new_tokens=kwargs.get("max_tokens", self.max_tokens),
                    top_p=kwargs.get("top_p", self.top_p),
                    do_sample=True,
                )

            # 解码输出
            generated_text = self._tokenizer.decode(
                outputs[0][inputs.input_ids.shape[1] :], skip_special_tokens=True
            )

            return GenerationResult(
                text=generated_text,
                prompt=prompt,
                metadata={
                    "model": self.model_name,
                    "latency": time.time() - start_time,
                    "device": self.device,
                },
            )
        except Exception as e:
            return GenerationResult(
                text="",
                prompt=prompt,
                metadata={"error": str(e), "latency": time.time() - start_time},
            )

    def batch_generate(self, prompts: List[str], **kwargs) -> List[GenerationResult]:
        """为多个提示生成文本"""
        return [self.generate(p, **kwargs) for p in prompts]


def get_llm_generator(**kwargs) -> BaseGenerator:
    """
    根据环境配置获取 LLM 生成器

    Args:
        **kwargs: 覆盖默认配置的参数

    Returns:
        LLM 生成器实例
    """
    provider = get_llm_provider()

    if provider == "openai":
        return OpenAIGenerator(**kwargs)
    elif provider == "anthropic":
        return AnthropicGenerator(**kwargs)
    elif provider == "google":
        return GoogleGenerator(**kwargs)
    elif provider == "qwen":
        return QwenGenerator(**kwargs)
    elif provider == "local":
        return HuggingFaceGenerator(**kwargs)
    else:
        raise ValueError(f"Unknown LLM provider: {provider}")
