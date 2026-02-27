"""
密集检索器
==========

使用 Transformer 嵌入的密集检索方法实现。
支持 OpenAI、阿里云 DashScope 等远端嵌入 API，
以及本地 Sentence Transformers 模型。
"""

from typing import List, Dict, Any, Optional, Union
import numpy as np

from src.retrievers.base import BaseRetriever, RetrievalResult
from src.utils.config import get_embedding_config, get_embedding_provider


class SentenceTransformerRetriever(BaseRetriever):
    """使用 Sentence Transformers 的密集检索器"""

    def __init__(
        self,
        model_name: str = "BAAI/bge-base-zh-v1.5",
        top_k: int = 5,
        device: Optional[str] = None,
        normalize_embeddings: bool = True,
    ):
        # 从环境变量读取配置
        config = get_embedding_config()

        model_name = config.get("model", model_name)
        device = device or config.get(
            "device", "cuda" if __import__("torch").cuda.is_available() else "cpu"
        )
        top_k = config.get("top_k", top_k)

        super().__init__(top_k)
        self.model_name = model_name
        self.device = device
        self.normalize_embeddings = normalize_embeddings
        self.embeddings: Optional[np.ndarray] = None
        self.documents: List[str] = []
        self._model = None

    def _load_model(self):
        """懒加载 Sentence Transformer 模型"""
        if self._model is None:
            try:
                from sentence_transformers import SentenceTransformer

                self._model = SentenceTransformer(self.model_name, device=self.device)
            except ImportError:
                raise ImportError(
                    "sentence-transformers is required. Install with: pip install sentence-transformers"
                )

    def _encode(self, texts: Union[str, List[str]]) -> np.ndarray:
        """将文本编码为嵌入向量"""
        self._load_model()

        if isinstance(texts, str):
            texts = [texts]

        embeddings = self._model.encode(
            texts,
            normalize_embeddings=self.normalize_embeddings,
            show_progress_bar=False,
        )

        return embeddings

    def index(self, documents: List[str], batch_size: int = 32, **kwargs) -> None:
        """使用密集嵌入对文档进行索引"""
        self.documents = documents

        # 编码所有文档
        self.embeddings = self._encode(documents)

    def retrieve(self, query: str, **kwargs) -> List[RetrievalResult]:
        """使用密集检索检索文档"""
        # 编码查询
        query_embedding = self._encode(query)

        # 计算相似度
        scores = np.dot(self.embeddings, query_embedding.T).flatten()

        # 获取 top-k
        top_indices = np.argsort(scores)[::-1][: self.top_k]

        results = []
        for idx in top_indices:
            results.append(
                RetrievalResult(
                    text=self.documents[idx],
                    score=float(scores[idx]),
                    metadata={"index": int(idx)},
                )
            )

        return results


class OpenAIEmbeddingRetriever(BaseRetriever):
    """使用 OpenAI API 的嵌入检索器"""

    def __init__(
        self, model_name: str = "text-embedding-3-small", top_k: int = 5, **kwargs
    ):
        # 从环境变量读取配置
        config = get_embedding_config()

        model_name = config.get("model", model_name)
        top_k = config.get("top_k", top_k)

        super().__init__(top_k)
        self.model_name = model_name

        try:
            from openai import OpenAI
        except ImportError:
            raise ImportError("openai is required. Install with: pip install openai")

        # 从环境变量获取 API key
        import os

        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise ValueError("OpenAI API key is required. Set OPENAI_API_KEY in .env")

        self.client = OpenAI()
        self.embeddings: Optional[np.ndarray] = None
        self.documents: List[str] = []

    def _encode(self, texts: Union[str, List[str]]) -> np.ndarray:
        """使用 OpenAI API 将文本编码为嵌入向量"""
        if isinstance(texts, str):
            texts = [texts]

        response = self.client.embeddings.create(model=self.model_name, input=texts)

        # 提取嵌入向量
        embeddings = np.array([item.embedding for item in response.data])
        return embeddings

    def index(self, documents: List[str], batch_size: int = 32, **kwargs) -> None:
        """对文档进行索引"""
        self.documents = documents

        # 分批编码文档
        all_embeddings = []
        for i in range(0, len(documents), batch_size):
            batch = documents[i : i + batch_size]
            embeddings = self._encode(batch)
            all_embeddings.append(embeddings)

        self.embeddings = np.vstack(all_embeddings)

    def retrieve(self, query: str, **kwargs) -> List[RetrievalResult]:
        """检索文档"""
        # 编码查询
        query_embedding = self._encode(query)

        # 计算余弦相似度
        scores = self.compute_similarity(query_embedding, self.embeddings)

        # 获取 top-k
        top_indices = np.argsort(scores)[::-1][: self.top_k]

        results = []
        for idx in top_indices:
            results.append(
                RetrievalResult(
                    text=self.documents[idx],
                    score=float(scores[idx]),
                    metadata={"index": int(idx)},
                )
            )

        return results


class DashScopeEmbeddingRetriever(BaseRetriever):
    """使用阿里云 DashScope API 的嵌入检索器"""

    def __init__(self, model_name: str = "text-embedding-v3", top_k: int = 5, **kwargs):
        # 从环境变量读取配置
        config = get_embedding_config()

        model_name = config.get("model", model_name)
        top_k = config.get("top_k", top_k)

        super().__init__(top_k)
        self.model_name = model_name

        try:
            import dashscope
        except ImportError:
            raise ImportError(
                "dashscope is required. Install with: pip install dashscope"
            )

        # 从环境变量获取 API key
        import os

        api_key = os.getenv("DASHSCOPE_API_KEY")
        if not api_key:
            raise ValueError(
                "DashScope API key is required. Set DASHSCOPE_API_KEY in .env"
            )

        dashscope.api_key = api_key
        self.embeddings: Optional[np.ndarray] = None
        self.documents: List[str] = []

    def _encode(self, texts: Union[str, List[str]]) -> np.ndarray:
        """使用 DashScope API 将文本编码为嵌入向量"""
        if isinstance(texts, str):
            texts = [texts]

        from dashscope import TextEmbedding

        # 分批处理（DashScope 每次最多 20 条）
        all_embeddings = []
        batch_size = 20

        for i in range(0, len(texts), batch_size):
            batch = texts[i : i + batch_size]
            response = TextEmbedding.call(model=self.model_name, input=batch)

            if response.status_code == 200:
                embeddings = np.array(
                    [item["embedding"] for item in response.output["embeddings"]]
                )
                all_embeddings.append(embeddings)
            else:
                raise ValueError(f"DashScope API error: {response.message}")

        return np.vstack(all_embeddings) if all_embeddings else np.array([])

    def index(self, documents: List[str], batch_size: int = 32, **kwargs) -> None:
        """对文档进行索引"""
        self.documents = documents

        # 编码所有文档
        self.embeddings = self._encode(documents)

    def retrieve(self, query: str, **kwargs) -> List[RetrievalResult]:
        """检索文档"""
        # 编码查询
        query_embedding = self._encode(query)

        # 计算余弦相似度
        scores = self.compute_similarity(query_embedding, self.embeddings)

        # 获取 top-k
        top_indices = np.argsort(scores)[::-1][: self.top_k]

        results = []
        for idx in top_indices:
            results.append(
                RetrievalResult(
                    text=self.documents[idx],
                    score=float(scores[idx]),
                    metadata={"index": int(idx)},
                )
            )

        return results


class DenseRetriever(BaseRetriever):
    """可自定义编码器的通用密集检索器"""

    def __init__(self, encoder: Any, top_k: int = 5, batch_size: int = 32):
        """
        初始化密集检索器

        Args:
            encoder: 编码器对象，需要有 encode(texts) 方法
            top_k: 检索的文档数量
            batch_size: 编码的批处理大小
        """
        super().__init__(top_k)
        self.encoder = encoder
        self.batch_size = batch_size
        self.embeddings: Optional[np.ndarray] = None
        self.documents: List[str] = []

    def index(self, documents: List[str], **kwargs) -> None:
        """对文档进行索引"""
        self.documents = documents

        # 分批编码
        all_embeddings = []
        for i in range(0, len(documents), self.batch_size):
            batch = documents[i : i + self.batch_size]
            embeddings = self.encoder.encode(batch)
            all_embeddings.append(embeddings)

        self.embeddings = np.vstack(all_embeddings)

    def retrieve(self, query: str, **kwargs) -> List[RetrievalResult]:
        """检索文档"""
        query_embedding = self.encoder.encode([query])

        # 计算余弦相似度
        scores = self.compute_similarity(query_embedding, self.embeddings)

        # 获取 top-k
        top_indices = np.argsort(scores)[::-1][: self.top_k]

        results = []
        for idx in top_indices:
            results.append(
                RetrievalResult(
                    text=self.documents[idx],
                    score=float(scores[idx]),
                    metadata={"index": int(idx)},
                )
            )

        return results


class DPRRetriever(DenseRetriever):
    """密集段落检索器 (Dense Passage Retriever) 实现"""

    def __init__(
        self,
        passage_encoder: str = "facebook/dpr-ctx_encoder-single-nq-base",
        question_encoder: str = "facebook/dpr-question_encoder-single-nq-base",
        top_k: int = 5,
        device: Optional[str] = None,
    ):
        """
        初始化 DPR 检索器

        Args:
            passage_encoder: 段落编码器的名称或路径
            question_encoder: 问题编码器的名称或路径
            top_k: 检索的文档数量
            device: 运行模型的设备
        """
        self.passage_encoder_name = passage_encoder
        self.question_encoder_name = question_encoder
        self.device = (
            device or "cuda" if __import__("torch").cuda.is_available() else "cpu"
        )

        # 懒加载
        self._passage_encoder = None
        self._question_encoder = None

        super().__init__(encoder=None, top_k=top_k)

    def _load_models(self):
        """懒加载 DPR 模型"""
        if self._passage_encoder is None:
            try:
                from transformers import DPRContextEncoder, DPRQuestionEncoder
                import torch

                self._passage_encoder = DPRContextEncoder.from_pretrained(
                    self.passage_encoder_name
                ).to(self.device)

                self._question_encoder = DPRQuestionEncoder.from_pretrained(
                    self.question_encoder_name
                ).to(self.device)

                self.encoder = self
            except ImportError:
                raise ImportError(
                    "transformers is required. Install with: pip install transformers"
                )

    def encode(self, texts: List[str], is_query: bool = False) -> np.ndarray:
        """使用 DPR 编码器编码文本"""
        self._load_models()

        import torch

        with torch.no_grad():
            if is_query:
                inputs = self._question_encoder(text["input_ids"] for text in texts)
            else:
                inputs = self._passage_encoder(text["input_ids"] for text in texts)

        return inputs.pooler_output.cpu().numpy()

    def index(self, documents: List[str], **kwargs) -> None:
        """使用段落编码器对文档进行索引"""
        self.documents = documents
        self._load_models()

        # 使用段落编码器编码
        self.embeddings = self.encode(documents, is_query=False)

    def retrieve(self, query: str, **kwargs) -> List[RetrievalResult]:
        """使用问题编码器检索文档"""
        # 使用问题编码器编码查询
        query_embedding = self.encode([query], is_query=True)

        # 计算相似度
        scores = self.compute_similarity(query_embedding, self.embeddings)

        # 获取 top-k
        top_indices = np.argsort(scores)[::-1][: self.top_k]

        results = []
        for idx in top_indices:
            results.append(
                RetrievalResult(
                    text=self.documents[idx],
                    score=float(scores[idx]),
                    metadata={"index": int(idx)},
                )
            )

        return results


def get_embedding_retriever(**kwargs) -> BaseRetriever:
    """
    根据环境配置获取嵌入检索器

    Args:
        **kwargs: 覆盖默认配置的参数

    Returns:
        嵌入检索器实例
    """
    provider = get_embedding_provider()

    if provider == "openai":
        return OpenAIEmbeddingRetriever(**kwargs)
    elif provider == "dashscope":
        return DashScopeEmbeddingRetriever(**kwargs)
    elif provider == "local":
        return SentenceTransformerRetriever(**kwargs)
    else:
        raise ValueError(f"Unknown embedding provider: {provider}")
