"""
Dense Retrievers
================

Implementation of dense retrieval methods using transformer embeddings.
"""

from typing import List, Dict, Any, Optional, Union
import numpy as np

from src.retrievers.base import BaseRetriever, RetrievalResult


class SentenceTransformerRetriever(BaseRetriever):
    """Dense retriever using sentence transformers."""
    
    def __init__(
        self,
        model_name: str = "BAAI/bge-base-zh-v1.5",
        top_k: int = 5,
        device: Optional[str] = None,
        normalize_embeddings: bool = True
    ):
        super().__init__(top_k)
        self.model_name = model_name
        self.device = device or "cuda" if __import__("torch").cuda.is_available() else "cpu"
        self.normalize_embeddings = normalize_embeddings
        self.embeddings: Optional[np.ndarray] = None
        self.documents: List[str] = []
        self._model = None
    
    def _load_model(self):
        """Lazy load the sentence transformer model."""
        if self._model is None:
            try:
                from sentence_transformers import SentenceTransformer
                self._model = SentenceTransformer(self.model_name, device=self.device)
            except ImportError:
                raise ImportError("sentence-transformers is required. Install with: pip install sentence-transformers")
    
    def _encode(self, texts: Union[str, List[str]]) -> np.ndarray:
        """Encode texts to embeddings."""
        self._load_model()
        
        if isinstance(texts, str):
            texts = [texts]
        
        embeddings = self._model.encode(
            texts,
            normalize_embeddings=self.normalize_embeddings,
            show_progress_bar=False
        )
        
        return embeddings
    
    def index(self, documents: List[str], batch_size: int = 32, **kwargs) -> None:
        """Index documents using dense embeddings."""
        self.documents = documents
        
        # Encode all documents
        self.embeddings = self._encode(documents)
    
    def retrieve(self, query: str, **kwargs) -> List[RetrievalResult]:
        """Retrieve documents using dense retrieval."""
        # Encode query
        query_embedding = self._encode(query)
        
        # Compute similarities
        scores = np.dot(self.embeddings, query_embedding.T).flatten()
        
        # Get top-k
        top_indices = np.argsort(scores)[::-1][:self.top_k]
        
        results = []
        for idx in top_indices:
            results.append(RetrievalResult(
                text=self.documents[idx],
                score=float(scores[idx]),
                metadata={"index": int(idx)}
            ))
        
        return results


class DenseRetriever(BaseRetriever):
    """Generic dense retriever with customizable encoder."""
    
    def __init__(
        self,
        encoder: Any,
        top_k: int = 5,
        batch_size: int = 32
    ):
        """
        Initialize dense retriever.
        
        Args:
            encoder: Encoder object with encode(texts) method
            top_k: Number of documents to retrieve
            batch_size: Batch size for encoding
        """
        super().__init__(top_k)
        self.encoder = encoder
        self.batch_size = batch_size
        self.embeddings: Optional[np.ndarray] = None
        self.documents: List[str] = []
    
    def index(self, documents: List[str], **kwargs) -> None:
        """Index documents."""
        self.documents = documents
        
        # Encode in batches
        all_embeddings = []
        for i in range(0, len(documents), self.batch_size):
            batch = documents[i:i + self.batch_size]
            embeddings = self.encoder.encode(batch)
            all_embeddings.append(embeddings)
        
        self.embeddings = np.vstack(all_embeddings)
    
    def retrieve(self, query: str, **kwargs) -> List[RetrievalResult]:
        """Retrieve documents."""
        query_embedding = self.encoder.encode([query])
        
        # Compute cosine similarity
        scores = self.compute_similarity(query_embedding, self.embeddings)
        
        # Get top-k
        top_indices = np.argsort(scores)[::-1][:self.top_k]
        
        results = []
        for idx in top_indices:
            results.append(RetrievalResult(
                text=self.documents[idx],
                score=float(scores[idx]),
                metadata={"index": int(idx)}
            ))
        
        return results


class DPRRetriever(DenseRetriever):
    """Dense Passage Retriever implementation."""
    
    def __init__(
        self,
        passage_encoder: str = "facebook/dpr-ctx_encoder-single-nq-base",
        question_encoder: str = "facebook/dpr-question_encoder-single-nq-base",
        top_k: int = 5,
        device: Optional[str] = None
    ):
        """
        Initialize DPR retriever.
        
        Args:
            passage_encoder: Name or path of passage encoder
            question_encoder: Name or path of question encoder
            top_k: Number of documents to retrieve
            device: Device to run models on
        """
        self.passage_encoder_name = passage_encoder
        self.question_encoder_name = question_encoder
        self.device = device or "cuda" if __import__("torch").cuda.is_available() else "cpu"
        
        # Will be loaded lazily
        self._passage_encoder = None
        self._question_encoder = None
        
        super().__init__(encoder=None, top_k=top_k)
    
    def _load_models(self):
        """Load DPR models lazily."""
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
                raise ImportError("transformers is required. Install with: pip install transformers")
    
    def encode(self, texts: List[str], is_query: bool = False) -> np.ndarray:
        """Encode texts using DPR encoder."""
        self._load_models()
        
        import torch
        
        with torch.no_grad():
            if is_query:
                inputs = self._question_encoder(
                    text["input_ids"] for text in texts
                )
            else:
                inputs = self._passage_encoder(
                    text["input_ids"] for text in texts
                )
        
        return inputs.pooler_output.cpu().numpy()
    
    def index(self, documents: List[str], **kwargs) -> None:
        """Index documents using passage encoder."""
        self.documents = documents
        self._load_models()
        
        # Encode using passage encoder
        self.embeddings = self.encode(documents, is_query=False)
    
    def retrieve(self, query: str, **kwargs) -> List[RetrievalResult]:
        """Retrieve documents using question encoder."""
        # Encode query using question encoder
        query_embedding = self.encode([query], is_query=True)
        
        # Compute similarities
        scores = self.compute_similarity(query_embedding, self.embeddings)
        
        # Get top-k
        top_indices = np.argsort(scores)[::-1][:self.top_k]
        
        results = []
        for idx in top_indices:
            results.append(RetrievalResult(
                text=self.documents[idx],
                score=float(scores[idx]),
                metadata={"index": int(idx)}
            ))
        
        return results
