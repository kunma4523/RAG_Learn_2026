# RAG评估工具

评估RAG系统的工具。

## 1. RAGAS

```python
from ragas import evaluate
from ragas.metrics import faithfulness, answer_relevancy, context_precision

results = evaluate(
    dataset=eval_dataset,
    metrics=[faithfulness, answer_relevancy, context_precision]
)
```

## 2. DeepEval

```python
from deepeval import evaluate
from deepeval.metrics import AnswerCorrectnessMetric

metric = AnswerCorrectnessMetric()
evaluate(test_cases, [metric])
```
