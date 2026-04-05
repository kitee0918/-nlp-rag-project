"""Main RAG Grid Experiment Script."""

import os
import sys
import json
import argparse
from datetime import datetime
from itertools import product
from typing import Dict, List
import torch

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.config import (
    DATA_DIR, RESULTS_DIR, RETRIEVAL_K, SEED,
    ExperimentConfig
)
from src.retrievers import get_retriever
from src.generators import get_generator
from src.evaluation import compute_metrics, aggregate_metrics, FActScorer
from src.data_utils import load_all_datasets, create_corpus_from_dataset


def set_seed(seed: int):
    """Set random seeds for reproducibility."""
    import random
    import numpy as np
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def run_single_experiment(
    config: ExperimentConfig,
    dataset: List[Dict],
    corpus: List[Dict],
    device: str = "cuda"
) -> Dict:
    """
    Run a single experiment configuration.
    
    Args:
        config: Experiment configuration
        dataset: QA dataset
        corpus: Document corpus
        device: Device to run on
        
    Returns:
        Results dictionary
    """
    print(f"\n{'='*60}")
    print(f"Running: {config.retriever_type} + {config.generator_model} on {config.dataset}")
    print(f"{'='*60}")
    
    # Initialize retriever
    index_path = os.path.join(
        RESULTS_DIR, 
        f"indices/{config.retriever_type}_{config.dataset}"
    )
    retriever = get_retriever(
        config.retriever_type, 
        corpus,
        index_path=index_path if config.retriever_type != "hybrid" else None,
        bm25_index_path=f"{index_path}_bm25",
        contriever_index_path=f"{index_path}_contriever",
        device=device
    )
    
    # Initialize generator
    generator = get_generator(
        config.generator_model,
        device=device,
        load_in_8bit=True  # Use 8-bit for memory efficiency
    )
    
    # Initialize fact scorer
    fact_scorer = FActScorer()
    
    # Run evaluation
    results = []
    all_metrics = []
    
    for i, item in enumerate(dataset):
        print(f"\rProcessing example {i+1}/{len(dataset)}", end="", flush=True)
        
        # Retrieve documents
        retrieved = retriever.retrieve(item["question"], top_k=config.retrieval_k)
        retrieved_docs = [doc for doc, score in retrieved]
        retrieved_ids = [doc["id"] for doc in retrieved_docs]
        
        # Generate answer
        gen_result = generator.generate(item["question"], retrieved_docs)
        answer = gen_result["answer"]
        
        # Compute metrics
        gold_ids = [item.get("doc_id", "")]
        metrics = compute_metrics(
            prediction=answer,
            ground_truth=item["answer"],
            retrieved_ids=retrieved_ids,
            gold_ids=gold_ids,
            k=config.retrieval_k
        )
        
        # Compute FActScore
        context = "\n".join([doc["text"] for doc in retrieved_docs])
        fact_metrics = fact_scorer.compute_factscore(answer, context)
        metrics.update(fact_metrics)
        
        # Store result
        result = {
            "id": item["id"],
            "question": item["question"],
            "ground_truth": item["answer"],
            "prediction": answer,
            "retrieved_ids": retrieved_ids,
            "metrics": metrics
        }
        results.append(result)
        all_metrics.append(metrics)
    
    print()  # New line after progress
    
    # Aggregate metrics
    aggregated = aggregate_metrics(all_metrics)
    
    return {
        "config": {
            "retriever": config.retriever_type,
            "generator": config.generator_model,
            "dataset": config.dataset,
            "retrieval_k": config.retrieval_k
        },
        "aggregated_metrics": aggregated,
        "num_examples": len(dataset),
        "results": results,
        "timestamp": datetime.now().isoformat()
    }


def run_grid_experiment(
    retrievers: List[str] = ["bm25", "contriever", "hybrid"],
    generators: List[str] = ["llama", "qwen"],
    datasets: List[str] = ["financebench", "cuad", "legalbench_rag"],
    subset_size: int = None,
    device: str = "cuda"
) -> Dict:
    """
    Run the full 3x2x2 grid experiment.
    
    Args:
        retrievers: List of retriever types
        generators: List of generator types
        datasets: List of dataset names
        subset_size: Subset size for testing (None for full)
        device: Device to run on
        
    Returns:
        All experiment results
    """
    print("=" * 60)
    print("CS6493 RAG Grid Experiment")
    print(f"Retrievers: {retrievers}")
    print(f"Generators: {generators}")
    print(f"Datasets: {datasets}")
    print(f"Subset size: {subset_size or 'Full'}")
    print("=" * 60)
    
    # Load all datasets
    print("\nLoading datasets...")
    all_datasets = load_all_datasets(DATA_DIR, subset_size)
    
    all_results = {}
    
    # Run grid
    for dataset_name, retriever_type, generator_type in product(
        datasets, retrievers, generators
    ):
        config = ExperimentConfig(
            retriever_type=retriever_type,
            generator_model=generator_type,
            domain="finance" if dataset_name == "financebench" else "law",
            dataset=dataset_name
        )
        
        # Get dataset and corpus
        dataset = all_datasets[dataset_name]
        corpus = create_corpus_from_dataset(dataset)
        
        # Run experiment
        result = run_single_experiment(config, dataset, corpus, device)
        
        # Save individual result
        result_path = config.get_result_path()
        os.makedirs(os.path.dirname(result_path), exist_ok=True)
        with open(result_path, "w") as f:
            json.dump(result, f, indent=2)
        
        all_results[f"{retriever_type}_{generator_type}_{dataset_name}"] = result
        
        # Print summary
        metrics = result["aggregated_metrics"]
        print(f"\nResults for {retriever_type} + {generator_type} on {dataset_name}:")
        print(f"  EM: {metrics.get('exact_match', 0):.4f}")
        print(f"  F1: {metrics.get('f1', 0):.4f}")
        print(f"  Recall@{config.retrieval_k}: {metrics.get('recall_at_k', 0):.4f}")
        print(f"  FActScore: {metrics.get('factscore', 0):.4f}")
    
    return all_results


def main():
    parser = argparse.ArgumentParser(description="Run RAG Grid Experiment")
    parser.add_argument("--retrievers", nargs="+", 
                       default=["bm25", "contriever", "hybrid"],
                       help="Retriever types to test")
    parser.add_argument("--generators", nargs="+",
                       default=["llama", "qwen"],
                       help="Generator models to test")
    parser.add_argument("--datasets", nargs="+",
                       default=["financebench", "cuad", "legalbench_rag"],
                       help="Datasets to evaluate on")
    parser.add_argument("--subset-size", type=int, default=10,
                       help="Subset size for testing (None for full)")
    parser.add_argument("--device", default="cuda",
                       help="Device to run on")
    parser.add_argument("--seed", type=int, default=SEED,
                       help="Random seed")
    
    args = parser.parse_args()
    
    # Set seed
    set_seed(args.seed)
    
    # Create directories
    os.makedirs(DATA_DIR, exist_ok=True)
    os.makedirs(RESULTS_DIR, exist_ok=True)
    
    # Check GPU
    if args.device == "cuda" and not torch.cuda.is_available():
        print("CUDA not available, falling back to CPU")
        args.device = "cpu"
    
    # Run experiment
    results = run_grid_experiment(
        retrievers=args.retrievers,
        generators=args.generators,
        datasets=args.datasets,
        subset_size=args.subset_size,
        device=args.device
    )
    
    # Save combined results
    combined_path = os.path.join(RESULTS_DIR, "grid_experiment_results.json")
    with open(combined_path, "w") as f:
        json.dump(results, f, indent=2)
    
    print(f"\n{'='*60}")
    print("Experiment completed!")
    print(f"Results saved to: {combined_path}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
