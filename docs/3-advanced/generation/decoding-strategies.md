# 解码策略

控制LLM输出的策略。

## 1. 参数

- temperature
- top_p (nucleus sampling)
- top_k
- presence/frequency penalty

## 2. 策略

- Beam Search
- Greedy
- Sampling

## 3. RAG场景建议

- factual: temperature=0
- creative: temperature=0.7-1.0
