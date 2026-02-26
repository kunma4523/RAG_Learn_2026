# Architecture Documentation Template
# ==================================

Use this template when adding a new RAG architecture.

## 1. Basic Information

**Architecture Name**: [Name]
**Type**: [Basic/Interactive/Self-reflective/etc.]
**Difficulty**: ⭐ to ⭐⭐⭐⭐⭐

## 2. Paper/Source

- **Paper Title**: [Full title]
- **Authors**: [Author list]
- **Year**: [Year]
- **arXiv/Link**: [URL]
- **Venue**: [Conference/Journal]

## 3. Core Concept

[Explain the main idea in 2-3 sentences]

## 4. How It Works

### Architecture Diagram

```
[Insert diagram showing the flow]
```

### Key Components

1. **Component 1**: Description
2. **Component 2**: Description
3. **Component N**: Description

### Algorithm/Pseudocode

```python
def architecture_name(query, documents):
    # Step 1
    result = step_one(query)
    # Step 2
    result = step_two(result)
    # Step N
    return final_result
```

## 5. Advantages

- [Advantage 1]
- [Advantage 2]

## 6. Limitations

- [Limitation 1]
- [Limitation 2]

## 7. Use Cases

- [Use Case 1]
- [Use Case 2]

## 8. Code Example

```python
from src.pipelines import CustomPipeline

pipeline = CustomPipeline(...)
result = pipeline.query("Your question here")
print(result.answer)
```

## 9. Further Reading

- [Related Paper 1]
- [Related Paper 2]

## 10. Implementation Notes

[Any specific implementation details or tips]
