# 服务化部署

将RAG部署为API服务。

## 1. FastAPI示例

```python
from fastapi import FastAPI
from pydantic import BaseModel

app = FastAPI()

class QueryRequest(BaseModel):
    question: str

@app.post("/query")
def query(req: QueryRequest):
    result = rag.query(req.question)
    return {"answer": result["answer"]}
```

## 2. 部署选项

- Docker
- Kubernetes
- 云服务 (AWS, GCP, Azure)
