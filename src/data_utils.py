"""Utility functions for data loading and processing."""

import json
import os
from typing import List, Dict, Optional
from datasets import load_dataset


def load_financebench(data_dir: str, subset_size: Optional[int] = None) -> List[Dict]:
    """
    Load FinanceBench dataset.
    
    Args:
        data_dir: Directory to store/load data
        subset_size: Optional subset size for testing
        
    Returns:
        List of QA examples
    """
    try:
        # Try loading from HuggingFace
        dataset = load_dataset("PatronusAI/financebench", split="test")
        examples = []
        for item in dataset:
            examples.append({
                "id": item.get("id", item.get("question_id", str(len(examples)))),
                "question": item["question"],
                "answer": item["answer"],
                "evidence": item.get("evidence", item.get("context", "")),
                "doc_id": item.get("doc_id", ""),
                "domain": "finance"
            })
        if subset_size:
            examples = examples[:subset_size]
        return examples
    except Exception as e:
        print(f"Could not load FinanceBench from HuggingFace: {e}")
        return _load_sample_financebench()


def _load_sample_financebench() -> List[Dict]:
    """Load sample FinanceBench data for testing."""
    return [
        {
            "id": "fb_001",
            "question": "What was Apple's total revenue in fiscal year 2023?",
            "answer": "Apple's total revenue in fiscal year 2023 was $383.3 billion.",
            "evidence": "Apple Inc. reported total net sales of $383.3 billion for the fiscal year ended September 30, 2023.",
            "doc_id": "apple_10k_2023",
            "domain": "finance"
        },
        {
            "id": "fb_002",
            "question": "What is the current ratio of Microsoft as of 2023?",
            "answer": "Microsoft's current ratio was approximately 1.77 as of June 30, 2023.",
            "evidence": "Microsoft Corporation had total current assets of $184.3 billion and total current liabilities of $103.9 billion as of June 30, 2023.",
            "doc_id": "msft_10k_2023",
            "domain": "finance"
        },
        {
            "id": "fb_003",
            "question": "What was NVIDIA's revenue growth rate in fiscal 2024?",
            "answer": "NVIDIA's revenue grew 126% in fiscal year 2024.",
            "evidence": "NVIDIA reported revenue of $60.9 billion for fiscal year 2024, up 126% from $26.9 billion in fiscal year 2023.",
            "doc_id": "nvda_10k_2024",
            "domain": "finance"
        }
    ]


def load_cuad(data_dir: str, subset_size: Optional[int] = None) -> List[Dict]:
    """
    Load CUAD (Contract Understanding Atticus Dataset).
    
    Args:
        data_dir: Directory to store/load data
        subset_size: Optional subset size for testing
        
    Returns:
        List of QA examples
    """
    try:
        dataset = load_dataset("cuad", split="test")
        examples = []
        for item in dataset:
            examples.append({
                "id": item.get("id", str(len(examples))),
                "question": item["question"],
                "answer": item["answers"]["text"][0] if item["answers"]["text"] else "",
                "context": item["context"],
                "doc_id": item.get("document_id", ""),
                "domain": "law"
            })
        if subset_size:
            examples = examples[:subset_size]
        return examples
    except Exception as e:
        print(f"Could not load CUAD from HuggingFace: {e}")
        return _load_sample_cuad()


def _load_sample_cuad() -> List[Dict]:
    """Load sample CUAD data for testing."""
    return [
        {
            "id": "cuad_001",
            "question": "What is the termination date of the agreement?",
            "answer": "December 31, 2025",
            "context": "This Agreement shall commence on January 1, 2024 and shall continue until December 31, 2025, unless earlier terminated in accordance with the provisions hereof.",
            "doc_id": "contract_001",
            "domain": "law"
        },
        {
            "id": "cuad_002",
            "question": "What is the governing law of this contract?",
            "answer": "State of Delaware",
            "context": "This Agreement shall be governed by and construed in accordance with the laws of the State of Delaware, without regard to its conflict of laws principles.",
            "doc_id": "contract_002",
            "domain": "law"
        },
        {
            "id": "cuad_003",
            "question": "What is the confidentiality period?",
            "answer": "5 years",
            "context": "Each party agrees to maintain the confidentiality of any proprietary information disclosed by the other party for a period of five (5) years from the date of disclosure.",
            "doc_id": "contract_003",
            "domain": "law"
        }
    ]


def load_legalbench_rag(data_dir: str, subset_size: Optional[int] = None) -> List[Dict]:
    """
    Load LegalBench-RAG-mini dataset.
    
    Args:
        data_dir: Directory to store/load data
        subset_size: Optional subset size for testing
        
    Returns:
        List of QA examples
    """
    try:
        dataset = load_dataset("CoHesIVAI/LegalBench-RAG-mini", split="test")
        examples = []
        for item in dataset:
            examples.append({
                "id": item.get("id", str(len(examples))),
                "question": item["question"],
                "answer": item.get("answer", ""),
                "context": item["context"],
                "doc_id": item.get("doc_id", ""),
                "domain": "law"
            })
        if subset_size:
            examples = examples[:subset_size]
        return examples
    except Exception as e:
        print(f"Could not load LegalBench-RAG from HuggingFace: {e}")
        return _load_sample_legalbench()


def _load_sample_legalbench() -> List[Dict]:
    """Load sample LegalBench-RAG data for testing."""
    return [
        {
            "id": "lbr_001",
            "question": "Under what circumstances can a party assign this agreement?",
            "answer": "A party may assign this agreement with prior written consent of the other party.",
            "context": "Neither party may assign or transfer this Agreement, in whole or in part, without the prior written consent of the other party, except that either party may assign this Agreement to an affiliate or in connection with a merger or acquisition.",
            "doc_id": "legal_doc_001",
            "domain": "law"
        },
        {
            "id": "lbr_002",
            "question": "What are the indemnification obligations?",
            "answer": "Each party must indemnify the other against claims arising from breach of the agreement.",
            "context": "Each party shall indemnify, defend and hold harmless the other party from and against any and all claims, damages, losses, costs and expenses arising out of or relating to any breach of this Agreement.",
            "doc_id": "legal_doc_002",
            "domain": "law"
        }
    ]


def load_all_datasets(data_dir: str, subset_size: Optional[int] = None) -> Dict[str, List[Dict]]:
    """Load all datasets for the experiment."""
    return {
        "financebench": load_financebench(data_dir, subset_size),
        "cuad": load_cuad(data_dir, subset_size),
        "legalbench_rag": load_legalbench_rag(data_dir, subset_size)
    }


def create_corpus_from_dataset(dataset: List[Dict]) -> List[Dict]:
    """Create a document corpus from a dataset."""
    corpus = []
    seen_docs = set()
    
    for item in dataset:
        doc_id = item.get("doc_id", item["id"])
        if doc_id not in seen_docs:
            corpus.append({
                "id": doc_id,
                "text": item.get("context", item.get("evidence", "")),
                "metadata": {"domain": item.get("domain", "general")}
            })
            seen_docs.add(doc_id)
    
    return corpus
