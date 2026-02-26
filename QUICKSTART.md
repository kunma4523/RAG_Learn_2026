# Quick Start Guide

This guide will help you get started with RAG-Learning-2026.

## Installation

### 1. Clone the Repository

```bash
git clone https://github.com/your-username/RAG-Learning-2026.git
cd RAG-Learning-2026
```

### 2. Create Virtual Environment

```bash
# Using uv (recommended)
uv venv
source .venv/bin/activate  # Linux/Mac
# or
.venv\Scripts\activate     # Windows

# Using conda
conda create -n rag-learn python=3.10
conda activate rag-learn
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

## Your First RAG Application

### Using the Library

```python
from src.pipelines.standard import StandardRAGPipeline

# Create pipeline
pipeline = StandardRAGPipeline(
    embedding_model="BAAI/bge-base-zh-v1.5",
    llm_model="Qwen/Qwen2-7B-Instruct"
)

# Index documents
documents = [
    "RAG combines retrieval with generation...",
    "BM25 is a ranking function...",
]
pipeline.index_documents(documents)

# Query
result = pipeline.query("What is RAG?")
print(result.answer)
```

### Using the Notebooks

```bash
# Start Jupyter
jupyter notebook
```

Open `notebooks/1_basic_rag.ipynb` to get started.

## Project Structure

```
RAG-Learning-2026/
├── docs/           # Documentation
├── notebooks/      # Jupyter notebooks
├── src/            # Source code
│   ├── retrievers/ # Retrieval implementations
│   ├── generators/ # LLM generators
│   ├── pipelines/  # RAG pipelines
│   └── evaluation/ # Evaluation metrics
├── experiments/    # Experiment configs
├── data/           # Sample data
└── references/     # Papers and resources
```

## Learning Path

1. **Start with basics**: Read `docs/1-foundations/`
2. **Try notebooks**: Start with `notebooks/1_basic_rag.ipynb`
3. **Learn architectures**: Explore `docs/2-architectures/`
4. **Experiment**: Run experiments in `experiments/`
5. **Contribute**: See `CONTRIBUTING.md`

## Quick Examples

### Standard RAG

```python
from src.pipelines import StandardRAGPipeline
pipeline = StandardRAGPipeline(...)
```

### Conversational RAG

```python
from src.pipelines import ConversationalRAGPipeline
pipeline = ConversationalRAGPipeline(...)
```

### Hybrid RAG

```python
from src.pipelines import HybridRAGPipeline
pipeline = HybridRAGPipeline(...)
```

### Agentic RAG

```python
from src.pipelines import AgenticRAGPipeline
pipeline = AgenticRAGPipeline(...)
```

## Common Issues

### ImportError

Make sure you've installed all dependencies:
```bash
pip install -r requirements.txt
```

### CUDA Out of Memory

Try using a smaller model or CPU:
```python
pipeline = StandardRAGPipeline(
    embedding_model="sentence-transformers/all-MiniLM-L6-v2",
    llm_model="gpt-3.5-turbo"  # or use CPU-based model
)
```

## Next Steps

- Read the [documentation](docs/)
- Try the [notebooks](notebooks/)
- Explore [architectures](docs/2-architectures/)
- Run [experiments](experiments/)
- Join the community!
