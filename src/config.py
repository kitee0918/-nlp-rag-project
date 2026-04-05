"""RAG Grid Experiment Configuration"""

import os
from dataclasses import dataclass, field
from typing import List, Optional

# Base paths
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(PROJECT_ROOT, "data")
RESULTS_DIR = os.path.join(PROJECT_ROOT, "results")

# Model paths (HuggingFace model IDs or local paths)
MODELS = {
    "llama": "meta-llama/Llama-3.1-8B-Instruct",
    "qwen": "Qwen/Qwen2.5-7B-Instruct",
    "contriever": "facebook/contriever-msmarco",
}

# Retrieval settings
RETRIEVAL_K = 3  # Top-k documents to retrieve
RRF_K = 60  # Reciprocal Rank Fusion constant

# Experiment settings
SEED = 42


@dataclass
class ExperimentConfig:
    """Configuration for a single experiment run."""
    retriever_type: str  # "bm25", "contriever", "hybrid"
    generator_model: str  # "llama", "qwen"
    domain: str  # "finance", "law"
    dataset: str  # "financebench", "cuad", "legalbench_rag"
    retrieval_k: int = RETRIEVAL_K
    
    def get_result_path(self) -> str:
        return os.path.join(
            RESULTS_DIR,
            f"{self.retriever_type}_{self.generator_model}_{self.dataset}.json"
        )
