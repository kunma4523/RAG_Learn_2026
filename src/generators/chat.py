"""
聊天生成器
==========

支持对话历史记录的生成器。
"""

from typing import List, Dict, Any, Optional
from dataclasses import dataclass, field

from src.generators.base import BaseGenerator, GenerationResult


@dataclass
class Message:
    """表示聊天消息"""

    role: str  # "user", "assistant", "system"
    content: str
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, str]:
        return {"role": self.role, "content": self.content}


class ChatGenerator(BaseGenerator):
    """支持对话历史记录的生成器"""

    def __init__(
        self,
        model_name: str,
        temperature: float = 0.7,
        max_tokens: int = 2048,
        top_p: float = 0.9,
        system_prompt: Optional[str] = None,
        max_history: int = 10,
        **kwargs,
    ):
        super().__init__(model_name, temperature, max_tokens, top_p, **kwargs)

        self.system_prompt = system_prompt or "You are a helpful assistant."
        self.max_history = max_history
        self.conversation_history: List[Message] = []
        self._client = None

    @property
    def client(self):
        """获取或创建底层客户端"""
        if self._client is None:
            # 尝试检测合适的客户端
            try:
                from openai import OpenAI

                self._client = OpenAI()
            except ImportError:
                pass
        return self._client

    @client.setter
    def client(self, value):
        self._client = value

    def add_message(
        self, role: str, content: str, metadata: Optional[Dict] = None
    ) -> None:
        """添加消息到对话历史"""
        message = Message(role=role, content=content, metadata=metadata or {})
        self.conversation_history.append(message)

        # 根据需要修剪历史
        if len(self.conversation_history) > self.max_history:
            # 保留系统提示和最近的消息
            system_msg = (
                self.conversation_history[0]
                if self.conversation_history[0].role == "system"
                else None
            )
            self.conversation_history = (
                [system_msg] if system_msg else []
            ) + self.conversation_history[-(self.max_history) :]

    def clear_history(self) -> None:
        """清除对话历史"""
        # 如果存在则保留系统提示
        if self.conversation_history and self.conversation_history[0].role == "system":
            self.conversation_history = [self.conversation_history[0]]
        else:
            self.conversation_history = []

    def _build_messages(
        self, context: Optional[List[str]] = None, user_query: Optional[str] = None
    ) -> List[Dict[str, str]]:
        """为 API 调用构建消息列表"""
        messages = []

        # 添加系统提示
        messages.append({"role": "system", "content": self.system_prompt})

        # 添加对话历史
        for msg in self.conversation_history:
            if msg.role != "system":
                messages.append(msg.to_dict())

        # 如果有上下文和查询，添加它们
        if context and user_query:
            context_str = "\n\n".join(
                [f"[Document {i + 1}]\n{doc}" for i, doc in enumerate(context)]
            )
            messages.append(
                {
                    "role": "user",
                    "content": f"Context:\n{context_str}\n\nQuestion: {user_query}\n\nAnswer:",
                }
            )
        elif user_query:
            messages.append({"role": "user", "content": user_query})

        return messages

    def generate(
        self, prompt: str, context: Optional[List[str]] = None, **kwargs
    ) -> GenerationResult:
        """使用对话上下文生成响应"""
        import time

        start_time = time.time()

        messages = self._build_messages(context=context, user_query=prompt)

        try:
            if self.client:
                # 使用 OpenAI API
                response = self.client.chat.completions.create(
                    model=self.model_name,
                    messages=messages,
                    temperature=kwargs.get("temperature", self.temperature),
                    max_tokens=kwargs.get("max_tokens", self.max_tokens),
                    top_p=kwargs.get("top_p", self.top_p),
                )

                text = response.choices[0].message.content or ""

                # 添加到历史
                self.add_message("user", prompt)
                self.add_message("assistant", text)

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
                # 后备方案 - 返回提示作为响应
                return GenerationResult(
                    text="No LLM client configured.",
                    prompt=prompt,
                    metadata={
                        "error": "No client",
                        "latency": time.time() - start_time,
                    },
                )

        except Exception as e:
            return GenerationResult(
                text="",
                prompt=prompt,
                metadata={"error": str(e), "latency": time.time() - start_time},
            )

    def chat(
        self, user_message: str, context: Optional[List[str]] = None, **kwargs
    ) -> GenerationResult:
        """
        方便的方法，用于聊天交互

        Args:
            user_message: 用户消息
            context: 可选上下文（检索到的文档）

        Returns:
            包含响应的 GenerationResult
        """
        result = self.generate(prompt=user_message, context=context, **kwargs)

        return result

    def batch_generate(self, prompts: List[str], **kwargs) -> List[GenerationResult]:
        """为多个提示生成响应"""
        return [self.generate(p, **kwargs) for p in prompts]

    def __repr__(self) -> str:
        return f"ChatGenerator(model={self.model_name}, history_len={len(self.conversation_history)})"
