"""
Graph Retriever
===============

Knowledge graph based retrieval for GraphRAG.
"""

from typing import List, Dict, Any, Optional, Set, Tuple
import numpy as np
from collections import defaultdict

from src.retrievers.base import BaseRetriever, RetrievalResult


class GraphRetriever(BaseRetriever):
    """Knowledge graph based retriever."""
    
    def __init__(
        self,
        embedding_retriever: BaseRetriever,
        top_k: int = 5,
        max_hops: int = 2
    ):
        """
        Initialize graph retriever.
        
        Args:
            embedding_retriever: Dense retriever for initial retrieval
            top_k: Number of documents to retrieve
            max_hops: Maximum graph traversal depth
        """
        super().__init__(top_k)
        self.embedding_retriever = embedding_retriever
        self.max_hops = max_hops
        
        # Graph structure
        self.graph: Dict[str, Set[str]] = defaultdict(set)
        self.entity_to_docs: Dict[str, List[int]] = defaultdict(list)
        self.documents: List[str] = []
        self.entity_embeddings: Dict[str, np.ndarray] = {}
    
    def _extract_entities(self, text: str) -> List[str]:
        """Extract entities from text (simple version)."""
        # Simple entity extraction - in production use NER
        words = text.lower().split()
        # Return bigrams as simple "entities"
        entities = []
        for i in range(len(words) - 1):
            entities.append(f"{words[i]}_{words[i+1]}")
        return entities
    
    def _build_graph(self, documents: List[str]) -> None:
        """Build knowledge graph from documents."""
        self.documents = documents
        
        # Extract entities from each document
        for doc_idx, doc in enumerate(documents):
            entities = self._extract_entities(doc)
            for entity in entities:
                self.entity_to_docs[entity].append(doc_idx)
                # Create edges between entities in same document
                for other_entity in entities:
                    if entity != other_entity:
                        self.graph[entity].add(other_entity)
    
    def index(self, documents: List[str], **kwargs) -> None:
        """Index documents and build graph."""
        self._build_graph(documents)
        self.embedding_retriever.index(documents, **kwargs)
    
    def _get_related_entities(
        self,
        seed_entities: Set[str],
        max_hops: int
    ) -> Set[str]:
        """Get entities within k hops of seed entities."""
        if max_hops == 0:
            return seed_entities
        
        visited = set(seed_entities)
        frontier = set(seed_entities)
        
        for _ in range(max_hops):
            new_frontier = set()
            for entity in frontier:
                # Add directly connected entities
                for neighbor in self.graph.get(entity, []):
                    if neighbor not in visited:
                        visited.add(neighbor)
                        new_frontier.add(neighbor)
            
            frontier = new_frontier
            if not frontier:
                break
        
        return visited
    
    def _get_relevant_documents(
        self,
        entities: Set[str]
    ) -> Dict[int, Set[str]]:
        """Get documents associated with entities."""
        doc_entities = defaultdict(set)
        
        for entity in entities:
            for doc_idx in self.entity_to_docs.get(entity, []):
                doc_entities[doc_idx].add(entity)
        
        return doc_entities
    
    def retrieve(self, query: str, **kwargs) -> List[RetrievalResult]:
        """Retrieve documents using graph-based approach."""
        # Step 1: Initial retrieval using embeddings
        initial_results = self.embedding_retriever.retrieve(query, top_k=self.top_k * 3)
        
        # Extract entities from initial retrieved documents
        seed_entities = set()
        for result in initial_results:
            entities = self._extract_entities(result.text)
            seed_entities.update(entities)
        
        # Step 2: Expand entities through graph traversal
        expanded_entities = self._get_related_entities(seed_entities, self.max_hops)
        
        # Step 3: Get documents associated with expanded entities
        doc_entities = self._get_relevant_documents(expanded_entities)
        
        # Step 4: Score documents by entity overlap
        doc_scores = []
        for doc_idx, entities in doc_entities.items():
            # Score based on number of matching entities
            score = len(entities) / len(self.documents[doc_idx].split())
            doc_scores.append((doc_idx, score))
        
        # Sort by score
        doc_scores.sort(key=lambda x: x[1], reverse=True)
        
        # Step 5: Combine with embedding similarity
        embedding_scores = {r.metadata["index"]: r.score for r in initial_results}
        
        final_scores = []
        seen_docs = set()
        
        # Add initial results with high embedding scores
        for r in initial_results[:self.top_k]:
            idx = r.metadata["index"]
            seen_docs.add(idx)
            # Combine graph and embedding scores
            graph_score = doc_entities.get(idx, set())
            combined_score = 0.5 * r.score + 0.5 * (len(graph_score) / 10)
            final_scores.append((idx, combined_score))
        
        # Add graph-based results
        for doc_idx, graph_score in doc_scores:
            if doc_idx not in seen_docs:
                embedding_score = embedding_scores.get(doc_idx, 0)
                combined_score = 0.3 * embedding_score + 0.7 * graph_score
                final_scores.append((doc_idx, combined_score))
                seen_docs.add(doc_idx)
        
        # Sort by final score
        final_scores.sort(key=lambda x: x[1], reverse=True)
        
        # Get top-k results
        results = []
        for idx, score in final_scores[:self.top_k]:
            results.append(RetrievalResult(
                text=self.documents[idx],
                score=score,
                metadata={"index": idx, "entities": list(doc_entities.get(idx, set()))}
            ))
        
        return results


class GraphBuilder:
    """Build knowledge graphs from documents."""
    
    def __init__(self):
        self.graph: Dict[str, Set[str]] = defaultdict(set)
        self.entity_counts: Dict[str, int] = defaultdict(int)
    
    def add_document(self, text: str, doc_id: str) -> None:
        """Add a document to the graph."""
        # Extract entities (simplified)
        words = text.lower().split()
        
        # Simple entity extraction
        for i in range(len(words)):
            entity = words[i]
            self.entity_counts[entity] += 1
            
            # Connect to nearby entities
            for j in range(max(0, i-3), min(len(words), i+4)):
                if i != j:
                    self.graph[words[i]].add(words[j])
    
    def get_subgraph(self, entity: str, depth: int = 1) -> Dict[str, Set[str]]:
        """Get subgraph around an entity."""
        visited = {entity}
        frontier = {entity}
        
        for _ in range(depth):
            new_frontier = set()
            for e in frontier:
                for neighbor in self.graph.get(e, []):
                    if neighbor not in visited:
                        visited.add(neighbor)
                        new_frontier.add(neighbor)
            frontier = new_frontier
        
        return {e: self.graph.get(e, set()) & visited for e in visited}
    
    def get_important_entities(self, top_k: int = 100) -> List[str]:
        """Get most important entities by frequency."""
        return sorted(
            self.entity_counts.keys(),
            key=lambda x: self.entity_counts[x],
            reverse=True
        )[:top_k]
