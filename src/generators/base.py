"""
基础生成器接口
==============

所有 LLM 生成器的抽象基类。
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import List, Dict, Any, Optional, Union
import time


@dataclass
class GenerationResult:
    """表示生成结果"""

    text: str
    prompt: str
    metadata: Dict[str, Any]

    def __repr__(self) -> str:
        return f"GenerationResult(text={self.text[:50]}..., metadata={self.metadata})"


class BaseGenerator(ABC):
    """所有生成器的抽象基类"""

    def __init__(
        self,
        model_name: str,
        temperature: float = 0.7,
        max_tokens: int = 2048,
        top_p: float = 0.9,
        **kwargs,
    ):
        """
        初始化生成器

        Args:
            model_name: 模型名称
            temperature: 采样温度
            max_tokens: 最大生成 token 数
            top_p: 核采样参数
            **kwargs: 额外参数
        """
        self.model_name = model_name
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.top_p = top_p
        self.extra_params = kwargs

    @abstractmethod
    def generate(self, prompt: str, **kwargs) -> GenerationResult:
        """
        根据提示生成文本

        Args:
            prompt: 输入提示
            **kwargs: 额外生成参数

        Returns:
            GenerationResult 对象
        """
        pass

    @abstractmethod
    def batch_generate(self, prompts: List[str], **kwargs) -> List[GenerationResult]:
        """
        为多个提示生成文本

        Args:
            prompts: 输入提示列表
            **kwargs: 额外生成参数

        Returns:
            GenerationResult 对象列表
        """
        pass

    def create_prompt(
        self,
        query: str,
        context: List[str],
        system_prompt: Optional[str] = None,
        template: Optional[str] = None,
    ) -> str:
        """
        创建包含上下文的格式化提示

        Args:
            query: 用户查询
            context: 检索到的上下文文档
            system_prompt: 可选的系统提示
            template: 可选的自定义模板

        Returns:
            格式化的提示字符串
        """
        if template:
            return template.format(query=query, context="\n\n".join(context))

        # 默认模板
        context_str = "\n\n".join(
            [f"[Document {i + 1}]\n{doc}" for i, doc in enumerate(context)]
        )

        if system_prompt:
            prompt = f"{system_prompt}\n\nContext:\n{context_str}\n\nQuestion: {query}\n\nAnswer:"
        else:
            prompt = f"Context:\n{context_str}\n\nQuestion: {query}\n\nAnswer:"

        return prompt

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(model={self.model_name})"
