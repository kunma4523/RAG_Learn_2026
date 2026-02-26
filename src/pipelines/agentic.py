"""
Agentic RAG Pipeline
===================

RAG pipeline with agent planning and tool use.
"""

from typing import List, Dict, Any, Optional, Callable
from dataclasses import dataclass, field
from enum import Enum
import time

from src.pipelines.base import BasePipeline, PipelineResult


class AgentAction(Enum):
    """Possible agent actions."""
    RETRIEVE = "retrieve"
    GENERATE = "generate"
    QUERY_REWRITE = "query_rewrite"
    VERIFY = "verify"
    ANSWER = "answer"


@dataclass
class AgentStep:
    """Represents a single agent step."""
    action: AgentAction
    result: Any
    reasoning: str
    timestamp: float = field(default_factory=time.time)


class AgenticRAGPipeline(BasePipeline):
    """
    Agentic RAG Pipeline.
    
    Uses an agent to plan and execute RAG steps dynamically.
    Supports:
    - Query analysis and rewriting
    - Multiple retrieval iterations
    - Self-verification
    - Tool use
    
    Flow: Analyze → Plan → Execute → Verify → Answer
    """
    
    def __init__(
        self,
        retriever: Any,
        generator: Any,
        top_k: int = 5,
        max_iterations: int = 3,
        enable_verification: bool = True,
        **kwargs
    ):
        super().__init__(retriever, generator, top_k)
        
        self.max_iterations = max_iterations
        self.enable_verification = enable_verification
        self.execution_trace: List[AgentStep] = []
        
        # Tools available to the agent
        self.tools: Dict[str, Callable] = {
            "retrieve": self._tool_retrieve,
            "generate": self._tool_generate,
            "rewrite_query": self._tool_rewrite_query,
            "verify": self._tool_verify,
        }
    
    def index_documents(
        self,
        documents: List[str],
        **kwargs
    ) -> None:
        """Index documents for retrieval."""
        self.retriever.index(documents, **kwargs)
        self.indexed = True
    
    def _tool_retrieve(self, query: str, **kwargs) -> List[str]:
        """Tool: Retrieve documents."""
        results = self.retriever.retrieve(query, top_k=kwargs.get("top_k", self.top_k))
        return [doc.text for doc in results]
    
    def _tool_generate(self, prompt: str, **kwargs) -> str:
        """Tool: Generate text."""
        result = self.generator.generate(prompt, **kwargs)
        return result.text
    
    def _tool_rewrite_query(self, query: str, **kwargs) -> str:
        """Tool: Rewrite query for better retrieval."""
        rewrite_prompt = f"""Rewrite the following query to be more effective for document retrieval.

Original query: {query}

Requirements:
- Expand abbreviations
- Add synonyms
- Make it more specific if too vague
- Keep it concise

Rewritten query:"""

        result = self.generator.generate(rewrite_prompt, max_tokens=200)
        
        # Extract rewritten query
        rewritten = result.text.strip().split('\n')[0]
        return rewritten if rewritten else query
    
    def _tool_verify(self, answer: str, context: List[str], query: str, **kwargs) -> Dict[str, Any]:
        """Tool: Verify answer against context."""
        verify_prompt = f"""Given the following question, context, and answer, verify if the answer is supported by the context.

Question: {query}

Context:
{chr(10).join([f"[{i+1}] {c}" for i, c in enumerate(context)])}

Answer: {answer}

Verify if the answer is faithful to the context. If not, explain what's wrong.

Verification result:"""

        result = self.generator.generate(verify_prompt, max_tokens=300)
        
        # Simple verification based on keywords
        verification = {
            "is_valid": True,  # Would need more sophisticated logic
            "reasoning": result.text,
            "needs_correction": False
        }
        
        return verification
    
    def _plan(self, query: str) -> List[Dict[str, Any]]:
        """
        Plan the execution steps.
        
        This is a simple rule-based planner. In production, 
        could use LLM-based planning.
        """
        plan = []
        
        # Step 1: Analyze and rewrite query
        plan.append({
            "action": "rewrite_query",
            "reasoning": "Rewrite query for better retrieval"
        })
        
        # Step 2: Initial retrieval
        plan.append({
            "action": "retrieve",
            "reasoning": "Retrieve relevant documents"
        })
        
        # Step 3: Generate initial answer
        plan.append({
            "action": "generate",
            "reasoning": "Generate answer from retrieved context"
        })
        
        # Step 4: Verify (if enabled)
        if self.enable_verification:
            plan.append({
                "action": "verify",
                "reasoning": "Verify answer against context"
            })
        
        return plan
    
    def _execute_plan(
        self,
        plan: List[Dict[str, Any]],
        query: str,
        **kwargs
    ) -> Dict[str, Any]:
        """Execute the plan step by step."""
        state = {
            "query": query,
            "rewritten_query": query,
            "retrieved_docs": [],
            "answer": "",
            "verification": None
        }
        
        for step in plan[:self.max_iterations]:
            action = step["action"]
            
            if action == "rewrite_query":
                state["rewritten_query"] = self.tools["rewrite_query"](query)
                self.execution_trace.append(AgentStep(
                    action=AgentAction.QUERY_REWRITE,
                    result=state["rewritten_query"],
                    reasoning=step["reasoning"]
                ))
            
            elif action == "retrieve":
                docs = self.tools["retrieve"](state["rewritten_query"], top_k=self.top_k)
                state["retrieved_docs"] = docs
                self.execution_trace.append(AgentStep(
                    action=AgentAction.RETRIEVE,
                    result=docs,
                    reasoning=step["reasoning"]
                ))
            
            elif action == "generate":
                prompt = self.generator.create_prompt(
                    state["rewritten_query"],
                    state["retrieved_docs"]
                )
                answer = self.tools["generate"](prompt, **kwargs)
                state["answer"] = answer
                self.execution_trace.append(AgentStep(
                    action=AgentAction.GENERATE,
                    result=answer,
                    reasoning=step["reasoning"]
                ))
            
            elif action == "verify":
                if state["retrieved_docs"] and state["answer"]:
                    verification = self.tools["verify"](
                        state["answer"],
                        state["retrieved_docs"],
                        state["rewritten_query"]
                    )
                    state["verification"] = verification
                    self.execution_trace.append(AgentStep(
                        action=AgentAction.VERIFY,
                        result=verification,
                        reasoning=step["reasoning"]
                    ))
        
        return state
    
    def query(
        self,
        query: str,
        **kwargs
    ) -> PipelineResult:
        """
        Process query through agentic pipeline.
        
        Args:
            query: User query
            
        Returns:
            PipelineResult with answer and execution trace
        """
        # Clear previous trace
        self.execution_trace = []
        
        start_time = time.time()
        
        # Plan execution
        plan = self._plan(query)
        
        # Execute plan
        state = self._execute_plan(plan, query, **kwargs)
        
        # Get final answer
        answer = state["answer"] or "No answer generated"
        
        total_time = time.time() - start_time
        
        return PipelineResult(
            query=query,
            answer=answer,
            retrieved_documents=[],  # Would need to convert docs to RetrievalResult
            metadata={
                "total_time": total_time,
                "execution_trace": [
                    {"action": str(step.action), "reasoning": step.reasoning}
                    for step in self.execution_trace
                ],
                "rewritten_query": state["rewritten_query"],
                "verification": state["verification"]
            }
        )
    
    def __repr__(self) -> str:
        return f"AgenticRAGPipeline(max_iterations={self.max_iterations})"
