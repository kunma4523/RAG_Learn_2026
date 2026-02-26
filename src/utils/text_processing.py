"""
Text Processing Utilities
==========================

Utilities for processing text and creating chunks.
"""

from typing import List, Callable, Optional, Dict, Any
import re


class TextChunker:
    """Split text into chunks."""
    
    def __init__(
        self,
        chunk_size: int = 512,
        chunk_overlap: int = 50,
        separator: str = "\n"
    ):
        """
        Initialize text chunker.
        
        Args:
            chunk_size: Maximum chunk size in characters
            chunk_overlap: Overlap between chunks
            separator: Separator for splitting
        """
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.separator = separator
    
    def chunk_text(self, text: str) -> List[str]:
        """Split text into chunks."""
        if not text:
            return []
        
        # Split by separator
        parts = text.split(self.separator)
        
        chunks = []
        current_chunk = ""
        current_size = 0
        
        for part in parts:
            part = part.strip()
            if not part:
                continue
            
            part_size = len(part)
            
            # If single part is too large, split it further
            if part_size > self.chunk_size:
                if current_chunk:
                    chunks.append(current_chunk.strip())
                    current_chunk = ""
                    current_size = 0
                
                # Split large part
                sub_chunks = self._split_large_part(part)
                chunks.extend(sub_chunks)
                continue
            
            # Check if adding this part would exceed limit
            if current_size + part_size + len(self.separator) > self.chunk_size:
                if current_chunk:
                    chunks.append(current_chunk.strip())
                
                # Start new chunk with overlap
                if self.chunk_overlap > 0 and chunks:
                    # Keep the end of previous chunk
                    overlap_text = chunks[-1][-self.chunk_overlap:] if len(chunks[-1]) > self.chunk_overlap else chunks[-1]
                    current_chunk = overlap_text + self.separator + part
                    current_size = len(overlap_text) + len(self.separator) + part_size
                else:
                    current_chunk = part
                    current_size = part_size
            else:
                if current_chunk:
                    current_chunk += self.separator + part
                else:
                    current_chunk = part
                current_size += len(self.separator) + part_size
        
        # Add last chunk
        if current_chunk:
            chunks.append(current_chunk.strip())
        
        return [c for c in chunks if c]
    
    def _split_large_part(self, text: str) -> List[str]:
        """Split a large text part into smaller chunks."""
        chunks = []
        
        # Try to split by sentences first
        sentences = re.split(r'(?<=[。！？])\s*', text)
        
        current_chunk = ""
        current_size = 0
        
        for sentence in sentences:
            sentence = sentence.strip()
            if not sentence:
                continue
            
            sentence_size = len(sentence)
            
            if sentence_size > self.chunk_size:
                # Add current chunk
                if current_chunk:
                    chunks.append(current_chunk.strip())
                    current_chunk = ""
                    current_size = 0
                
                # Split sentence by characters
                for i in range(0, sentence_size, self.chunk_size - self.chunk_overlap):
                    chunk = sentence[i:i + self.chunk_size]
                    chunks.append(chunk)
                continue
            
            if current_size + sentence_size > self.chunk_size:
                chunks.append(current_chunk.strip())
                current_chunk = sentence
                current_size = sentence_size
            else:
                if current_chunk:
                    current_chunk += " " + sentence
                else:
                    current_chunk = sentence
                current_size += sentence_size
        
        if current_chunk:
            chunks.append(current_chunk.strip())
        
        return chunks


def create_chunks(
    documents: List[str],
    chunk_size: int = 512,
    chunk_overlap: int = 50,
    show_progress: bool = True
) -> List[str]:
    """
    Create chunks from documents.
    
    Args:
        documents: List of document texts
        chunk_size: Maximum chunk size
        chunk_overlap: Overlap between chunks
        show_progress: Whether to show progress
        
    Returns:
        List of text chunks
    """
    chunker = TextChunker(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap
    )
    
    all_chunks = []
    
    for i, doc in enumerate(documents):
        chunks = chunker.chunk_text(doc)
        all_chunks.extend(chunks)
        
        if show_progress and (i + 1) % 10 == 0:
            print(f"Processed {i + 1}/{len(documents)} documents")
    
    if show_progress:
        print(f"Created {len(all_chunks)} chunks from {len(documents)} documents")
    
    return all_chunks


def split_documents(
    documents: List[str],
    method: str = "simple",
    **kwargs
) -> List[str]:
    """
    Split documents into chunks using different methods.
    
    Args:
        documents: List of document texts
        method: Splitting method ("simple", "recursive", "markdown")
        **kwargs: Additional parameters for the method
        
    Returns:
        List of text chunks
    """
    if method == "simple":
        return create_chunks(documents, **kwargs)
    elif method == "recursive":
        try:
            from langchain_text_splitters import RecursiveCharacterTextSplitter
            splitter = RecursiveCharacterTextSplitter(**kwargs)
            return splitter.split_texts(documents)
        except ImportError:
            print("langchain-text-splitters not installed, using simple method")
            return create_chunks(documents, **kwargs)
    else:
        raise ValueError(f"Unknown method: {method}")
