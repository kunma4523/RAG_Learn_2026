"""
RAG Evaluation Metrics
======================

Comprehensive evaluation metrics for RAG systems.
"""

from typing import List, Dict, Any, Optional, Set
import numpy as np
from collections import Counter


class RetrievalMetrics:
    """Metrics for retrieval component."""
    
    @staticmethod
    def recall_at_k(
        retrieved_docs: List[str],
        relevant_docs: List[str],
        k: int = 10
    ) -> float:
        """
        Calculate Recall@K.
        
        Args:
            retrieved_docs: List of retrieved document texts
            relevant_docs: List of relevant document texts
            k: Consider top-k retrieved documents
            
        Returns:
            Recall@K score
        """
        if not relevant_docs:
            return 0.0
        
        retrieved_set = set(retrieved_docs[:k])
        relevant_set = set(relevant_docs)
        
        return len(retrieved_set & relevant_set) / len(relevant_set)
    
    @staticmethod
    def precision_at_k(
        retrieved_docs: List[str],
        relevant_docs: List[str],
        k: int = 10
    ) -> float:
        """
        Calculate Precision@K.
        
        Args:
            retrieved_docs: List of retrieved document texts
            relevant_docs: List of relevant document texts
            k: Consider top-k retrieved documents
            
        Returns:
            Precision@K score
        """
        if k == 0:
            return 0.0
        
        retrieved_set = set(retrieved_docs[:k])
        relevant_set = set(relevant_docs)
        
        return len(retrieved_set & relevant_set) / k
    
    @staticmethod
    def mrr(retrieved_docs: List[str], relevant_docs: List[str]) -> float:
        """
        Calculate Mean Reciprocal Rank (MRR).
        
        Args:
            retrieved_docs: List of retrieved document texts
            relevant_docs: List of relevant document texts
            
        Returns:
            MRR score
        """
        relevant_set = set(relevant_docs)
        
        for i, doc in enumerate(retrieved_docs):
            if doc in relevant_set:
                return 1.0 / (i + 1)
        
        return 0.0
    
    @staticmethod
    def ndcg_at_k(
        retrieved_docs: List[str],
        relevant_docs: List[str],
        k: int = 10,
        relevance_scores: Optional[Dict[str, float]] = None
    ) -> float:
        """
        Calculate Normalized Discounted Cumulative Gain (NDCG@K).
        
        Args:
            retrieved_docs: List of retrieved document texts
            relevant_docs: List of relevant document texts
            k: Consider top-k retrieved documents
            relevance_scores: Optional relevance scores for each doc
            
        Returns:
            NDCG@K score
        """
        if not relevant_docs:
            return 0.0
        
        # Default relevance scores
        if relevance_scores is None:
            relevance_scores = {doc: 1.0 for doc in relevant_docs}
        
        # Calculate DCG
        dcg = 0.0
        for i, doc in enumerate(retrieved_docs[:k]):
            rel = relevance_scores.get(doc, 0.0)
            dcg += rel / np.log2(i + 2)  # i+2 because i is 0-indexed
        
        # Calculate IDCG
        ideal_scores = sorted(
            [relevance_scores.get(doc, 0.0) for doc in relevant_docs],
            reverse=True
        )
        idcg = sum(
            rel / np.log2(i + 2)
            for i, rel in enumerate(ideal_scores[:k])
        )
        
        if idcg == 0:
            return 0.0
        
        return dcg / idcg
    
    @staticmethod
    def average_precision(
        retrieved_docs: List[str],
        relevant_docs: List[str]
    ) -> float:
        """
        Calculate Average Precision (AP).
        
        Args:
            retrieved_docs: List of retrieved document texts
            relevant_docs: List of relevant document texts
            
        Returns:
            Average Precision score
        """
        if not relevant_docs:
            return 0.0
        
        relevant_set = set(relevant_docs)
        num_relevant = 0
        sum_precision = 0.0
        
        for i, doc in enumerate(retrieved_docs):
            if doc in relevant_set:
                num_relevant += 1
                precision = num_relevant / (i + 1)
                sum_precision += precision
        
        return sum_precision / len(relevant_set) if relevant_set else 0.0


class GenerationMetrics:
    """Metrics for generation component."""
    
    @staticmethod
    def bleu(
        reference: str,
        candidate: str,
        n: int = 4
    ) -> float:
        """
        Calculate BLEU score.
        
        Args:
            reference: Reference text
            candidate: Generated text
            n: Maximum n-gram order
            
        Returns:
            BLEU score
        """
        from collections import Counter
        
        def get_ngrams(text: str, n: int) -> Counter:
            words = text.split()
            return Counter(tuple(words[i:i+n]) for i in range(len(words)-n+1))
        
        # Calculate precision for each n
        precisions = []
        ref_ngrams = get_ngrams(reference.lower(), 1)
        
        for n_gram in range(1, n+1):
            ref_ngrams = get_ngrams(reference.lower(), n_gram)
            cand_ngrams = get_ngrams(candidate.lower(), n_gram)
            
            if not cand_ngrams:
                precisions.append(0.0)
                continue
            
            matches = sum((ref_ngrams & cand_ngrams).values())
            total = sum(cand_ngrams.values())
            
            precisions.append(matches / total if total > 0 else 0.0)
        
        # Handle zero precision
        if all(p == 0 for p in precisions):
            return 0.0
        
        # Calculate geometric mean
        import math
        log_precisions = [math.log(p) if p > 0 else float('-inf') for p in precisions]
        avg_log_precision = sum(log_precisions) / n
        
        # Brevity penalty
        ref_len = len(reference.split())
        cand_len = len(candidate.split())
        
        if cand_len >= ref_len:
            bp = 1.0
        else:
            bp = math.exp(1 - ref_len / cand_len) if cand_len > 0 else 0.0
        
        return bp * math.exp(avg_log_precision)
    
    @staticmethod
    def rouge_l(
        reference: str,
        candidate: str
    ) -> float:
        """
        Calculate ROUGE-L (Longest Common Subsequence).
        
        Args:
            reference: Reference text
            candidate: Generated text
            
        Returns:
            ROUGE-L score
        """
        ref_words = reference.lower().split()
        cand_words = candidate.lower().split()
        
        # Find LCS length
        m, n = len(ref_words), len(cand_words)
        
        # Dynamic programming
        dp = [[0] * (n + 1) for _ in range(m + 1)]
        
        for i in range(1, m + 1):
            for j in range(1, n + 1):
                if ref_words[i-1] == cand_words[j-1]:
                    dp[i][j] = dp[i-1][j-1] + 1
                else:
                    dp[i][j] = max(dp[i-1][j], dp[i][j-1])
        
        lcs_length = dp[m][n]
        
        if lcs_length == 0:
            return 0.0
        
        # Calculate precision and recall
        precision = lcs_length / n if n > 0 else 0
        recall = lcs_length / m if m > 0 else 0
        
        if precision + recall == 0:
            return 0.0
        
        return 2 * precision * recall / (precision + recall)


class RAGEvaluation:
    """Comprehensive RAG evaluation."""
    
    def __init__(
        self,
        retrieval_metrics: Optional[RetrievalMetrics] = None,
        generation_metrics: Optional[GenerationMetrics] = None
    ):
        self.retrieval = retrieval_metrics or RetrievalMetrics()
        self.generation = generation_metrics or GenerationMetrics()
    
    def evaluate_retrieval(
        self,
        retrieved_docs: List[str],
        relevant_docs: List[str],
        k_values: List[int] = [1, 3, 5, 10]
    ) -> Dict[str, float]:
        """
        Evaluate retrieval performance.
        
        Args:
            retrieved_docs: Retrieved documents
            relevant_docs: Relevant documents
            k_values: K values to evaluate
            
        Returns:
            Dictionary of metrics
        """
        metrics = {}
        
        for k in k_values:
            metrics[f"recall@{k}"] = self.retrieval.recall_at_k(
                retrieved_docs, relevant_docs, k
            )
            metrics[f"precision@{k}"] = self.retrieval.precision_at_k(
                retrieved_docs, relevant_docs, k
            )
            metrics[f"ndcg@{k}"] = self.retrieval.ndcg_at_k(
                retrieved_docs, relevant_docs, k
            )
        
        metrics["mrr"] = self.retrieval.mrr(retrieved_docs, relevant_docs)
        metrics["map"] = self.retrieval.average_precision(
            retrieved_docs, relevant_docs
        )
        
        return metrics
    
    def evaluate_generation(
        self,
        reference: str,
        candidate: str
    ) -> Dict[str, float]:
        """
        Evaluate generation performance.
        
        Args:
            reference: Reference answer
            candidate: Generated answer
            
        Returns:
            Dictionary of metrics
        """
        return {
            "bleu": self.generation.bleu(reference, candidate),
            "rouge_l": self.generation.rouge_l(reference, candidate)
        }
    
    def evaluate(
        self,
        query: str,
        retrieved_docs: List[str],
        relevant_docs: List[str],
        reference_answer: str,
        generated_answer: str,
        k_values: List[int] = [1, 3, 5, 10]
    ) -> Dict[str, Any]:
        """
        Complete RAG evaluation.
        
        Args:
            query: User query
            retrieved_docs: Retrieved documents
            relevant_docs: Relevant documents
            reference_answer: Reference answer
            generated_answer: Generated answer
            k_values: K values for evaluation
            
        Returns:
            Complete evaluation results
        """
        retrieval_metrics = self.evaluate_retrieval(
            retrieved_docs, relevant_docs, k_values
        )
        
        generation_metrics = self.evaluate_generation(
            reference_answer, generated_answer
        )
        
        return {
            "query": query,
            "retrieval": retrieval_metrics,
            "generation": generation_metrics,
            "summary": {
                "retrieval_score": np.mean(list({
                    k: v for k, v in retrieval_metrics.items()
                    if "recall" in k or "mrr" in k or "map" in k
                }.values())),
                "generation_score": generation_metrics.get("rouge_l", 0.0)
            }
        }
