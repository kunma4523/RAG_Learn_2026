"""
Document Loader
==============

Utilities for loading documents from various sources.
"""

from typing import List, Union
from pathlib import Path
import os


def load_txt(file_path: str) -> str:
    """Load text from a .txt file."""
    with open(file_path, 'r', encoding='utf-8') as f:
        return f.read()


def load_pdf(file_path: str) -> List[str]:
    """Load text from a PDF file."""
    try:
        from pypdf import PdfReader
    except ImportError:
        raise ImportError("pypdf is required. Install with: pip install pypdf")
    
    reader = PdfReader(file_path)
    pages = []
    
    for page in reader.pages:
        text = page.extract_text()
        if text:
            pages.append(text)
    
    return pages


def load_docx(file_path: str) -> List[str]:
    """Load text from a Word document."""
    try:
        from docx import Document
    except ImportError:
        raise ImportError("python-docx is required. Install with: pip install python-docx")
    
    doc = Document(file_path)
    paragraphs = [p.text for p in doc.paragraphs if p.text.strip()]
    
    return paragraphs


def load_markdown(file_path: str) -> str:
    """Load text from a Markdown file."""
    with open(file_path, 'r', encoding='utf-8') as f:
        return f.read()


def load_documents(
    file_paths: List[str],
    show_progress: bool = True
) -> List[str]:
    """
    Load documents from multiple files.
    
    Args:
        file_paths: List of file paths
        show_progress: Whether to show progress
        
    Returns:
        List of document texts
    """
    documents = []
    
    for i, path in enumerate(file_paths):
        path = Path(path)
        ext = path.suffix.lower()
        
        if show_progress:
            print(f"Loading {path.name}...")
        
        if ext == '.txt':
            docs = [load_txt(str(path))]
        elif ext == '.pdf':
            docs = load_pdf(str(path))
        elif ext in ['.docx', '.doc']:
            docs = load_docx(str(path))
        elif ext == '.md':
            docs = [load_markdown(str(path))]
        else:
            print(f"Unsupported file type: {ext}")
            continue
        
        documents.extend(docs)
    
    if show_progress:
        print(f"Loaded {len(documents)} documents")
    
    return documents


def load_directory(
    directory: str,
    extensions: List[str] = ['.txt', '.pdf', '.docx', '.md'],
    recursive: bool = True,
    show_progress: bool = True
) -> List[str]:
    """
    Load all documents from a directory.
    
    Args:
        directory: Directory path
        extensions: File extensions to load
        recursive: Whether to search recursively
        show_progress: Whether to show progress
        
    Returns:
        List of document texts
    """
    path = Path(directory)
    
    # Find all files
    if recursive:
        files = []
        for ext in extensions:
            files.extend(path.rglob(f'*{ext}'))
    else:
        files = []
        for ext in extensions:
            files.extend(path.glob(f'*{ext}'))
    
    # Convert to strings
    file_paths = [str(f) for f in files]
    
    return load_documents(file_paths, show_progress=show_progress)
