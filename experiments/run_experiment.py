"""
Run RAG Experiments
====================

Script to run RAG experiments with configuration.
"""

import argparse
import yaml
import json
import os
import sys
from pathlib import Path
from typing import Dict, Any, List
import time

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.pipelines.standard import StandardRAGPipeline
from src.evaluation.metrics import RAGEvaluation


def load_config(config_path: str) -> Dict[str, Any]:
    """Load configuration from YAML file."""
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


def load_dataset(dataset_path: str) -> List[Dict[str, Any]]:
    """Load dataset from JSON file."""
    with open(dataset_path, 'r') as f:
        return json.load(f)


def run_experiment(config: Dict[str, Any], dataset: List[Dict[str, Any]]):
    """Run RAG experiment."""
    print(f"Running experiment: {config['experiment']['name']}")
    print(f"Dataset size: {len(dataset)}")
    
    # Initialize pipeline
    pipeline = StandardRAGPipeline(
        embedding_model=config['retrieval']['embedding_model'],
        llm_model=config['generation']['model_name'],
        top_k=config['retrieval']['top_k']
    )
    
    # Index documents
    print("Indexing documents...")
    # In practice, load from actual documents
    documents = ["Sample document " + str(i) for i in range(1000)]
    pipeline.index_documents(documents)
    
    # Evaluate
    evaluator = RAGEvaluation()
    
    results = []
    for i, item in enumerate(dataset):
        query = item['question']
        
        start_time = time.time()
        result = pipeline.query(query)
        total_time = time.time() - start_time
        
        # Compute metrics if we have ground truth
        metrics = {}
        if 'answer' in item:
            gen_metrics = evaluator.evaluate_generation(
                item['answer'],
                result.answer
            )
            metrics.update(gen_metrics)
        
        metrics['latency'] = total_time
        
        results.append({
            'query': query,
            'answer': result.answer,
            'metrics': metrics
        })
        
        if (i + 1) % 10 == 0:
            print(f"Processed {i + 1}/{len(dataset)}")
    
    # Save results
    output_dir = config['evaluation']['results_dir']
    os.makedirs(output_dir, exist_ok=True)
    
    output_path = os.path.join(output_dir, f"{config['experiment']['name']}_results.json")
    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"Results saved to {output_path}")
    
    # Print summary
    avg_latency = sum(r['metrics']['latency'] for r in results) / len(results)
    print(f"\nAverage latency: {avg_latency:.3f}s")


def main():
    parser = argparse.ArgumentParser(description="Run RAG experiments")
    parser.add_argument('--config', type=str, required=True, help='Path to config file')
    parser.add_argument('--dataset', type=str, help='Path to dataset (overrides config)')
    
    args = parser.parse_args()
    
    # Load config
    config = load_config(args.config)
    
    # Load dataset
    dataset_path = args.dataset or config['data']['dev_path']
    dataset = load_dataset(dataset_path)
    
    # Run experiment
    run_experiment(config, dataset)


if __name__ == "__main__":
    main()
