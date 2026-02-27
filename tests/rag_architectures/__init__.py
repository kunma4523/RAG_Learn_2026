"""
Test Utilities for RAG Architecture Tests
=========================================

Shared fixtures, helpers, and test data for all RAG architecture tests.
"""

import os
import json
import time
from typing import List, Dict, Any, Optional
from dataclasses import dataclass, field

# Load environment variables
from dotenv import load_dotenv

load_dotenv()


# ============================================
# Configuration
# ============================================


@dataclass
class TestConfig:
    """Configuration for RAG tests."""

    # LLM settings
    llm_provider: str = os.getenv("LLM_PROVIDER", "openai")
    openai_api_key: str = os.getenv("OPENAI_API_KEY", "")
    openai_model: str = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
    anthropic_api_key: str = os.getenv("ANTHROPIC_API_KEY", "")
    anthropic_model: str = os.getenv("ANTHROPIC_MODEL", "claude-3-haiku-20240307")
    google_api_key: str = os.getenv("GOOGLE_API_KEY", "")
    google_model: str = os.getenv("GOOGLE_MODEL", "gemini-1.5-flash")
    dashscope_api_key: str = os.getenv("DASHSCOPE_API_KEY", "")
    qwen_model: str = os.getenv("QWEN_MODEL", "qwen-turbo")

    # Embedding settings
    embedding_provider: str = os.getenv("EMBEDDING_PROVIDER", "openai")
    openai_embedding_model: str = os.getenv(
        "OPENAI_EMBEDDING_MODEL", "text-embedding-3-small"
    )
    local_embedding_model: str = os.getenv(
        "LOCAL_EMBEDDING_MODEL", "BAAI/bge-base-zh-v1.5"
    )
    dashscope_embedding_model: str = os.getenv(
        "DASHSCOPE_EMBEDDING_MODEL", "text-embedding-v3"
    )

    # Retrieval settings
    top_k: int = int(os.getenv("EMBEDDING_TOP_K", "5"))

    # Test data
    test_documents: List[Dict[str, Any]] = field(default_factory=_load_test_documents)

    @property
    def is_configured(self) -> bool:
        """Check if API keys are configured."""
        if self.llm_provider == "openai":
            return bool(self.openai_api_key)
        elif self.llm_provider == "anthropic":
            return bool(self.anthropic_api_key)
        elif self.llm_provider == "google":
            return bool(self.google_api_key)
        elif self.llm_provider == "qwen" or self.llm_provider == "dashscope":
            return bool(self.dashscope_api_key)
        return False


def _load_test_documents() -> List[Dict[str, Any]]:
    """Load test documents from sample data."""
    # Try to load from project data
    data_path = os.path.join(
        os.path.dirname(os.path.dirname(os.path.dirname(__file__))),
        "data",
        "sample",
        "documents.json",
    )

    if os.path.exists(data_path):
        with open(data_path, "r") as f:
            return json.load(f)

    # Fallback to embedded test data
    return [
        {
            "id": "doc_001",
            "text": "Retrieval-Augmented Generation (RAG) combines information retrieval with text generation. It enhances Large Language Models (LLMs) by retrieving relevant information from external knowledge bases.",
            "category": "fundamentals",
        },
        {
            "id": "doc_002",
            "text": "A RAG system consists of three main components: retriever, reranker, and generator. The retriever finds relevant documents, the reranker improves relevance ordering, and the generator produces the final answer.",
            "category": "fundamentals",
        },
        {
            "id": "doc_003",
            "text": "Sparse retrieval (BM25, TF-IDF) uses keywords and works well for exact matching. Dense retrieval (DPR, Contriever) uses neural embeddings for semantic similarity. Hybrid retrieval combines both approaches.",
            "category": "retrieval",
        },
        {
            "id": "doc_004",
            "text": "BM25 is a probabilistic ranking function that scores documents based on term frequency and inverse document frequency with length normalization.",
            "category": "retrieval",
        },
        {
            "id": "doc_005",
            "text": "Self-RAG is a self-reflective retrieval-augmented generation framework. It enables the LLM to decide when to retrieve, when to use retrieved content, and to critically evaluate the generated output.",
            "category": "architectures",
        },
        {
            "id": "doc_006",
            "text": "Agentic RAG combines RAG with AI agents. The agent can plan retrieval strategies, decide when to search, rewrite queries, and use multiple tools.",
            "category": "architectures",
        },
        {
            "id": "doc_007",
            "text": "GraphRAG combines knowledge graphs with RAG. It uses graph structures to represent entity relationships for sophisticated retrieval considering relational information.",
            "category": "architectures",
        },
        {
            "id": "doc_008",
            "text": "HyDE (Hypothetical Document Embeddings) generates a hypothetical answer document first, then uses it to retrieve real documents for better matching.",
            "category": "architectures",
        },
        {
            "id": "doc_009",
            "text": "Fusion RAG combines multiple retrieval methods (sparse, dense, graph) and uses reranking to fuse results for improved recall and precision.",
            "category": "architectures",
        },
        {
            "id": "doc_010",
            "text": "Evaluation metrics for RAG include: Retrieval metrics (Recall@K, Precision@K, MRR, NDCG), Generation metrics (BLEU, ROUGE, BERTScore), and End-to-end metrics (RAGAS).",
            "category": "evaluation",
        },
    ]


# ============================================
# Test Data for Different RAG Types
# ============================================

MULTIMODAL_TEST_DATA = {
    "images": [
        {
            "id": "img_001",
            "text": "A photo of a sunset over the ocean with orange and purple colors",
            "image_path": "https://example.com/sunset.jpg",
        },
        {
            "id": "img_002",
            "text": "A cat sitting on a windowsill looking outside",
            "image_path": "https://example.com/cat.jpg",
        },
    ],
    "tables": [
        {
            "id": "table_001",
            "headers": ["Product", "Price", "Stock"],
            "rows": [
                ["Laptop", "$999", "50"],
                ["Phone", "$599", "100"],
                ["Tablet", "$399", "75"],
            ],
        }
    ],
    "sql_schema": {
        "tables": {
            "users": ["id", "name", "email", "created_at"],
            "orders": ["id", "user_id", "product_id", "quantity", "total_price"],
        }
    },
}


GRAPH_TEST_DATA = {
    "entities": [
        {"id": "person_1", "name": "Alice", "type": "Person"},
        {"id": "person_2", "name": "Bob", "type": "Person"},
        {"id": "company_1", "name": "TechCorp", "type": "Company"},
    ],
    "relationships": [
        {"source": "person_1", "target": "company_1", "type": "works_at"},
        {"source": "person_2", "target": "company_1", "type": "works_at"},
        {"source": "person_1", "target": "person_2", "type": "friends_with"},
    ],
}


# ============================================
# Helper Functions
# ============================================


def get_test_questions() -> Dict[str, List[str]]:
    """Get test questions for different RAG types."""
    return {
        "general": [
            "What is RAG?",
            "What are the main components of a RAG system?",
            "How does retrieval work in RAG?",
        ],
        "conversational": [
            "What is RAG?",
            "What are the main components?",
            "How does the retriever work?",
        ],
        "multi-hop": [
            "Who works at TechCorp?",
            "What is the relationship between Alice and Bob?",
        ],
        "multimodal": ["Describe the sunset image", "What products are available?"],
        "sql": ["How many users are there?", "What are the total orders?"],
    }


def create_vector_store(documents: List[str], config: TestConfig):
    """Create a vector store with test documents."""
    try:
        from langchain_community.vectorstores import Chroma
        from langchain_openai import OpenAIEmbeddings
        from langchain_community.embeddings import HuggingFaceEmbeddings

        if config.embedding_provider == "openai":
            embeddings = OpenAIEmbeddings(
                model=config.openai_embedding_model, api_key=config.openai_api_key
            )
        else:
            embeddings = HuggingFaceEmbeddings(model_name=config.local_embedding_model)

        vectorstore = Chroma.from_texts(
            texts=documents, embedding=embeddings, collection_name="test_rag"
        )
        return vectorstore
    except Exception as e:
        print(f"Error creating vector store: {e}")
        return None


def create_llm(config: TestConfig, **kwargs):
    """Create an LLM instance based on config."""
    try:
        if config.llm_provider == "openai":
            from langchain_openai import ChatOpenAI

            return ChatOpenAI(
                model=config.openai_model, api_key=config.openai_api_key, **kwargs
            )
        elif config.llm_provider == "anthropic":
            from langchain_anthropic import ChatAnthropic

            return ChatAnthropic(
                model=config.anthropic_model,
                anthropic_api_key=config.anthropic_api_key,
                **kwargs,
            )
        elif config.llm_provider == "google":
            from langchain_google_genai import ChatGoogleGenerativeAI

            return ChatGoogleGenerativeAI(
                model=config.google_model,
                google_api_key=config.google_api_key,
                **kwargs,
            )
        elif config.llm_provider == "qwen":
            from langchain_dashscope import ChatQWen

            return ChatQWen(
                model=config.qwen_model, api_key=config.dashscope_api_key, **kwargs
            )
    except Exception as e:
        print(f"Error creating LLM: {e}")
        return None


def create_retriever(config: TestConfig, vectorstore, top_k: int = 5):
    """Create a retriever from vector store."""
    try:
        return vectorstore.as_retriever(
            search_type="similarity", search_kwargs={"k": top_k}
        )
    except Exception as e:
        print(f"Error creating retriever: {e}")
        return None


# ============================================
# Test Runner Utilities
# ============================================


class TestResult:
    """Represents a test result."""

    def __init__(
        self,
        name: str,
        passed: bool,
        message: str = "",
        duration: float = 0,
        details: Dict = None,
    ):
        self.name = name
        self.passed = passed
        self.message = message
        self.duration = duration
        self.details = details or {}

    def __repr__(self):
        status = "✓ PASS" if self.passed else "✗ FAIL"
        return f"[{status}] {self.name} ({self.duration:.2f}s) - {self.message}"


def run_test(test_func, test_name: str) -> TestResult:
    """Run a single test and capture results."""
    start_time = time.time()
    try:
        result = test_func()
        duration = time.time() - start_time

        if result is None:
            return TestResult(test_name, True, "Test completed", duration)
        elif isinstance(result, bool):
            return TestResult(
                test_name, result, "Passed" if result else "Failed", duration
            )
        elif isinstance(result, dict):
            return TestResult(
                test_name,
                result.get("passed", True),
                result.get("message", ""),
                duration,
                result.get("details", {}),
            )
        else:
            return TestResult(test_name, True, str(result), duration)

    except Exception as e:
        duration = time.time() - start_time
        return TestResult(test_name, False, f"Error: {str(e)}", duration)


def print_test_results(results: List[TestResult]):
    """Print test results in a formatted way."""
    print("\n" + "=" * 60)
    print("TEST RESULTS")
    print("=" * 60)

    passed = sum(1 for r in results if r.passed)
    failed = sum(1 for r in results if not r.passed)

    for result in results:
        status = "✓" if result.passed else "✗"
        print(f"{status} {result.name}: {result.message} ({result.duration:.2f}s)")

    print("-" * 60)
    print(f"Total: {len(results)} | Passed: {passed} | Failed: {failed}")
    print("=" * 60)


# ============================================
# Export
# ============================================

__all__ = [
    "TestConfig",
    "TestResult",
    "get_test_questions",
    "create_vector_store",
    "create_llm",
    "create_retriever",
    "run_test",
    "print_test_results",
    "MULTIMODAL_TEST_DATA",
    "GRAPH_TEST_DATA",
]
