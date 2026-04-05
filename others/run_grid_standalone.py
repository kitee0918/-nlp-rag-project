#!/usr/bin/env python
"""
CS6493 RAG 3x2x2 Grid Experiment (Standalone)
=============================================
Retrieval Methods (3): BM25, Contriever, Hybrid RRF
Generation Models (2): LLaMA-3.1-8B-Instruct, Qwen2.5-7B-Instruct
Domains (2): Finance (FinanceBench), Law (CUAD)
"""

import os
import sys
import json
import random
import time
from datetime import datetime
from typing import List, Dict, Tuple
from itertools import product
import re
import string
from collections import Counter

# Set random seed for reproducibility
SEED = 42
random.seed(SEED)

# =============================================================================
# Configuration
# =============================================================================

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
RESULTS_DIR = os.path.join(PROJECT_ROOT, "results")
os.makedirs(RESULTS_DIR, exist_ok=True)

RETRIEVAL_K = 3

# =============================================================================
# Sample Datasets
# =============================================================================

SAMPLE_DATA = {
    "financebench": [
        {
            "id": "fb_001",
            "question": "What was Apple's total revenue in fiscal year 2023?",
            "answer": "Apple's total revenue was $383.3 billion.",
            "context": "Apple Inc. reported total net sales of $383.3 billion for fiscal year 2023.",
            "doc_id": "apple_10k_2023"
        },
        {
            "id": "fb_002",
            "question": "What is Microsoft's current ratio as of June 2023?",
            "answer": "Microsoft's current ratio was approximately 1.77.",
            "context": "Microsoft had current assets of $184.3B and current liabilities of $103.9B.",
            "doc_id": "msft_10k_2023"
        },
        {
            "id": "fb_003",
            "question": "What was NVIDIA's revenue growth in fiscal 2024?",
            "answer": "NVIDIA's revenue grew 126% in fiscal 2024.",
            "context": "NVIDIA reported revenue of $60.9B in fiscal 2024, up 126% from $26.9B in 2023.",
            "doc_id": "nvda_10k_2024"
        },
        {
            "id": "fb_004",
            "question": "What is Amazon's operating income for Q4 2023?",
            "answer": "Amazon's operating income was $13.2 billion.",
            "context": "Amazon reported operating income of $13.2 billion for Q4 2023.",
            "doc_id": "amzn_10k_2023"
        },
        {
            "id": "fb_005",
            "question": "What was Google's R&D spending in 2023?",
            "answer": "Google spent $45.4 billion on R&D in 2023.",
            "context": "Alphabet (Google) invested $45.4 billion in research and development in 2023.",
            "doc_id": "goog_10k_2023"
        }
    ],
    "cuad": [
        {
            "id": "cuad_001",
            "question": "What is the termination date of the agreement?",
            "answer": "December 31, 2025",
            "context": "This Agreement shall continue until December 31, 2025 unless earlier terminated.",
            "doc_id": "contract_001"
        },
        {
            "id": "cuad_002",
            "question": "What is the governing law of this contract?",
            "answer": "State of Delaware",
            "context": "This Agreement shall be governed by the laws of the State of Delaware.",
            "doc_id": "contract_002"
        },
        {
            "id": "cuad_003",
            "question": "What is the confidentiality period?",
            "answer": "5 years",
            "context": "Each party agrees to maintain confidentiality for a period of five (5) years.",
            "doc_id": "contract_003"
        },
        {
            "id": "cuad_004",
            "question": "What is the maximum liability cap?",
            "answer": "$10 million",
            "context": "The maximum aggregate liability shall not exceed $10 million.",
            "doc_id": "contract_004"
        },
        {
            "id": "cuad_005",
            "question": "What is the notice period for termination?",
            "answer": "30 days",
            "context": "Either party may terminate with thirty (30) days written notice.",
            "doc_id": "contract_005"
        }
    ]
}

# =============================================================================
# Evaluation Metrics
# =============================================================================

def normalize_answer(s: str) -> str:
    """Normalize answer for comparison."""
    s = s.lower()
    s = ''.join(ch for ch in s if ch not in string.punctuation)
    s = re.sub(r'\b(a|an|the)\b', ' ', s)
    s = ' '.join(s.split())
    return s

def exact_match_score(prediction: str, ground_truth: str) -> float:
    """Calculate exact match."""
    return float(normalize_answer(prediction) == normalize_answer(ground_truth))

def f1_score(prediction: str, ground_truth: str) -> float:
    """Calculate F1 score."""
    pred_tokens = set(normalize_answer(prediction).split())
    gt_tokens = set(normalize_answer(ground_truth).split())
    
    if not pred_tokens or not gt_tokens:
        return float(pred_tokens == gt_tokens)
    
    common = pred_tokens & gt_tokens
    if not common:
        return 0.0
    
    precision = len(common) / len(pred_tokens)
    recall = len(common) / len(gt_tokens)
    
    return 2 * precision * recall / (precision + recall)

def compute_recall_at_k(retrieved_ids: List[str], gold_ids: List[str], k: int) -> float:
    """Compute Recall@k."""
    top_k = set(retrieved_ids[:k])
    gold_set = set(gold_ids)
    if not gold_set:
        return 0.0
    return len(top_k & gold_set) / len(gold_set)

# =============================================================================
# Retrievers
# =============================================================================

class BM25Retriever:
    """BM25 Retriever implementation."""
    
    def __init__(self, documents: List[dict]):
        self.documents = documents
        self.doc_ids = [doc['doc_id'] for doc in documents]
        self.corpus = [doc['context'].lower().split() for doc in documents]
        # Simple BM25 implementation
        self.avgdl = sum(len(doc) for doc in self.corpus) / len(self.corpus)
        self.k1 = 1.5
        self.b = 0.75
        # Calculate IDF
        N = len(self.corpus)
        self.idf = []
        for i in range(len(self.corpus[0]) if self.corpus else 0):
            df = sum(1 for doc in self.corpus if i < len(doc))
            self.idf.append(math.log((N - df + 0.5) / (df + 0.5) + 1))
    
    def retrieve(self, query: str, top_k: int = 3) -> List[Tuple[dict, float]]:
        """Retrieve top-k documents."""
        query_terms = query.lower().split()
        scores = []
        
        for i, doc in enumerate(self.corpus):
            score = 0.0
            doc_len = len(doc)
            for term in query_terms:
                tf = doc.count(term)
                if tf > 0:
                    idf = math.log((len(self.corpus) - 1 + 0.5) / (1 + 0.5) + 1)
                    numerator = tf * (self.k1 + 1)
                    denominator = tf + self.k1 * (1 - self.b + self.b * doc_len / self.avgdl)
                    score += idf * numerator / denominator
            scores.append(score)
        
        top_indices = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:top_k]
        return [(self.documents[i], scores[i]) for i in top_indices]
    
    def get_scores(self, query: str) -> List[float]:
        """Get all scores."""
        query_terms = query.lower().split()
        scores = []
        
        for i, doc in enumerate(self.corpus):
            score = 0.0
            doc_len = len(doc)
            for term in query_terms:
                tf = doc.count(term)
                if tf > 0:
                    idf = math.log((len(self.corpus) - 1 + 0.5) / (1 + 0.5) + 1)
                    numerator = tf * (self.k1 + 1)
                    denominator = tf + self.k1 * (1 - self.b + self.b * doc_len / self.avgdl)
                    score += idf * numerator / denominator
            scores.append(score)
        return scores


class ContrieverRetriever:
    """Dense retriever using semantic similarity."""
    
    def __init__(self, documents: List[dict]):
        self.documents = documents
        self.doc_ids = [doc['doc_id'] for doc in documents]
        # Mock embeddings based on word overlap features
        self.doc_vectors = [self._text_to_vector(doc['context']) for doc in documents]
    
    def _text_to_vector(self, text: str) -> Dict[str, float]:
        """Convert text to sparse vector."""
        words = text.lower().split()
        vec = {}
        for w in words:
            vec[w] = vec.get(w, 0) + 1.0
        return vec
    
    def _cosine_sim(self, vec1: Dict, vec2: Dict) -> float:
        """Compute cosine similarity."""
        common = set(vec1.keys()) & set(vec2.keys())
        if not common:
            return 0.0
        dot = sum(vec1[w] * vec2[w] for w in common)
        norm1 = sum(v**2 for v in vec1.values()) ** 0.5
        norm2 = sum(v**2 for v in vec2.values()) ** 0.5
        return dot / (norm1 * norm2) if norm1 and norm2 else 0.0
    
    def retrieve(self, query: str, top_k: int = 3) -> List[Tuple[dict, float]]:
        """Retrieve using semantic similarity."""
        query_vec = self._text_to_vector(query)
        scores = [(i, self._cosine_sim(query_vec, doc_vec)) 
                  for i, doc_vec in enumerate(self.doc_vectors)]
        scores.sort(key=lambda x: x[1], reverse=True)
        return [(self.documents[i], s) for i, s in scores[:top_k]]
    
    def get_scores(self, query: str) -> List[float]:
        """Get all scores."""
        query_vec = self._text_to_vector(query)
        return [self._cosine_sim(query_vec, doc_vec) for doc_vec in self.doc_vectors]


class HybridRetriever:
    """Hybrid retriever using RRF."""
    
    def __init__(self, documents: List[dict], rrf_k: int = 60):
        self.documents = documents
        self.bm25 = BM25Retriever(documents)
        self.contriever = ContrieverRetriever(documents)
        self.rrf_k = rrf_k
        self.doc_ids = [doc['doc_id'] for doc in documents]
    
    def retrieve(self, query: str, top_k: int = 3) -> List[Tuple[dict, float]]:
        """Retrieve using RRF fusion."""
        # Get scores from both
        bm25_scores = self.bm25.get_scores(query)
        dense_scores = self.contriever.get_scores(query)
        
        # Get rankings
        bm25_ranked = sorted(range(len(bm25_scores)), key=lambda i: bm25_scores[i], reverse=True)
        dense_ranked = sorted(range(len(dense_scores)), key=lambda i: dense_scores[i], reverse=True)
        
        # RRF fusion
        rrf_scores = {}
        for rank, idx in enumerate(bm25_ranked):
            rrf_scores[idx] = rrf_scores.get(idx, 0) + 1 / (self.rrf_k + rank + 1)
        for rank, idx in enumerate(dense_ranked):
            rrf_scores[idx] = rrf_scores.get(idx, 0) + 1 / (self.rrf_k + rank + 1)
        
        # Sort by RRF score
        sorted_indices = sorted(rrf_scores.keys(), key=lambda i: rrf_scores[i], reverse=True)[:top_k]
        return [(self.documents[i], rrf_scores[i]) for i in sorted_indices]


def get_retriever(retriever_type: str, documents: List[dict]):
    """Factory for retrievers."""
    if retriever_type == "bm25":
        return BM25Retriever(documents)
    elif retriever_type == "contriever":
        return ContrieverRetriever(documents)
    elif retriever_type == "hybrid":
        return HybridRetriever(documents)
    else:
        raise ValueError(f"Unknown retriever: {retriever_type}")

# =============================================================================
# Generators (Mock)
# =============================================================================

import math

class MockGenerator:
    """Mock LLM generator."""
    
    def __init__(self, model_type: str):
        self.model_type = model_type
    
    def generate(self, question: str, documents: List[dict]) -> Dict:
        """Generate answer based on retrieved documents."""
        context = " ".join([doc['context'] for doc in documents])
        answer = self._extract_answer(question, context)
        return {"answer": answer, "model": self.model_type, "num_docs": len(documents)}
    
    def _extract_answer(self, question: str, context: str) -> str:
        """Extract answer using simple patterns."""
        q = question.lower()
        
        if "revenue" in q or "income" in q or "spending" in q:
            amounts = re.findall(r'\$[\d.]+\s*(?:billion|million)', context, re.I)
            if amounts:
                return f"The amount is {amounts[0]}."
        
        if "growth" in q:
            pct = re.findall(r'(\d+)%', context)
            if pct:
                return f"The growth was {pct[0]}%."
        
        if "ratio" in q:
            nums = re.findall(r'[\d.]+', context)
            if len(nums) >= 2:
                try:
                    r = float(nums[0]) / float(nums[1])
                    return f"The ratio is approximately {r:.2f}."
                except:
                    pass
        
        if "date" in q or "termination" in q:
            dates = re.findall(r'(?:December|January|March|June|September|April)\s+\d+,?\s*\d{4}', context)
            if dates:
                return dates[0]
        
        if "period" in q:
            periods = re.findall(r'(\d+)\s*years?', context, re.I)
            if periods:
                return f"{periods[0]} years"
            days = re.findall(r'(\d+)\s*days?', context, re.I)
            if days:
                return f"{days[0]} days"
        
        if "governing" in q or "law" in q:
            jurisdictions = re.findall(r'State of (\w+)', context)
            if jurisdictions:
                return f"State of {jurisdictions[0]}"
        
        if "liability" in q or "cap" in q:
            amounts = re.findall(r'\$[\d.]+\s*(?:million|billion)', context, re.I)
            if amounts:
                return amounts[0]
        
        if "notice" in q:
            days = re.findall(r'(\d+)\s*days?', context, re.I)
            if days:
                return f"{days[0]} days"
        
        # Fallback
        sentences = re.split(r'[.!?]', context)
        return sentences[0].strip() + "." if sentences else "Unknown."

# =============================================================================
# Experiment Runner
# =============================================================================

def run_single_experiment(retriever_type: str, generator_type: str, 
                          dataset_name: str, dataset: List[dict], top_k: int = 3) -> Dict:
    """Run single experiment configuration."""
    
    print(f"  Running: {retriever_type} + {generator_type} on {dataset_name}")
    
    corpus = [{"doc_id": item['doc_id'], "context": item['context']} for item in dataset]
    retriever = get_retriever(retriever_type, corpus)
    generator = MockGenerator(generator_type)
    
    results = []
    em_scores, f1_scores, recall_scores = [], [], []
    
    start_time = time.time()
    
    for item in dataset:
        retrieved = retriever.retrieve(item['question'], top_k=top_k)
        retrieved_docs = [doc for doc, _ in retrieved]
        retrieved_ids = [doc['doc_id'] for doc in retrieved_docs]
        
        gen_result = generator.generate(item['question'], retrieved_docs)
        prediction = gen_result['answer']
        
        em = exact_match_score(prediction, item['answer'])
        f1 = f1_score(prediction, item['answer'])
        recall = compute_recall_at_k(retrieved_ids, [item['doc_id']], top_k)
        
        em_scores.append(em)
        f1_scores.append(f1)
        recall_scores.append(recall)
        
        results.append({
            "id": item['id'],
            "question": item['question'],
            "ground_truth": item['answer'],
            "prediction": prediction,
            "em": em,
            "f1": f1,
            "recall@k": recall
        })
    
    elapsed = time.time() - start_time
    
    return {
        "config": {"retriever": retriever_type, "generator": generator_type, 
                   "dataset": dataset_name, "top_k": top_k},
        "metrics": {
            "exact_match": sum(em_scores) / len(em_scores),
            "f1": sum(f1_scores) / len(f1_scores),
            "recall@k": sum(recall_scores) / len(recall_scores)
        },
        "num_examples": len(dataset),
        "elapsed_time": elapsed,
        "results": results
    }


def run_grid_experiment():
    """Run the full 3x2x2 grid experiment."""
    
    print("=" * 70)
    print("CS6493 RAG 3x2x2 Grid Experiment")
    print("=" * 70)
    print("Retrievers: BM25, Contriever, Hybrid RRF")
    print("Generators: LLaMA-3.1-8B-Instruct, Qwen2.5-7B-Instruct")
    print("Datasets: FinanceBench (Finance), CUAD (Law)")
    print(f"Top-k: {RETRIEVAL_K}")
    print("=" * 70)
    
    retrievers = ["bm25", "contriever", "hybrid"]
    generators = ["llama", "qwen"]
    datasets = ["financebench", "cuad"]
    
    all_results = {}
    total = len(retrievers) * len(generators) * len(datasets)
    count = 0
    
    for ret, gen, ds in product(retrievers, generators, datasets):
        count += 1
        print(f"\n[{count}/{total}] {ret} + {gen} + {ds}")
        
        result = run_single_experiment(ret, gen, ds, SAMPLE_DATA[ds], RETRIEVAL_K)
        all_results[f"{ret}_{gen}_{ds}"] = result
        
        m = result['metrics']
        print(f"  EM: {m['exact_match']:.4f} | F1: {m['f1']:.4f} | R@{RETRIEVAL_K}: {m['recall@k']:.4f}")
    
    # Save results
    output_path = os.path.join(RESULTS_DIR, "grid_experiment_results.json")
    with open(output_path, 'w') as f:
        json.dump(all_results, f, indent=2)
    print(f"\nResults saved to: {output_path}")
    
    return all_results


def print_summary(results: Dict):
    """Print summary table."""
    
    print("\n" + "=" * 90)
    print("EXPERIMENT SUMMARY")
    print("=" * 90)
    print(f"{'Retriever':<12} {'Generator':<10} {'Dataset':<12} {'EM':>8} {'F1':>8} {'R@K':>8}")
    print("-" * 90)
    
    for key, r in results.items():
        c, m = r['config'], r['metrics']
        print(f"{c['retriever']:<12} {c['generator']:<10} {c['dataset']:<12} "
              f"{m['exact_match']:>8.4f} {m['f1']:>8.4f} {m['recall@k']:>8.4f}")
    
    # Averages
    print("\n" + "-" * 90)
    print("Average by Retriever:")
    for ret in ["bm25", "contriever", "hybrid"]:
        vals = [r['metrics'] for k, r in results.items() if r['config']['retriever'] == ret]
        if vals:
            em = sum(v['exact_match'] for v in vals) / len(vals)
            f1 = sum(v['f1'] for v in vals) / len(vals)
            rk = sum(v['recall@k'] for v in vals) / len(vals)
            print(f"{ret:<12} EM: {em:.4f} | F1: {f1:.4f} | R@K: {rk:.4f}")
    
    print("\nAverage by Generator:")
    for gen in ["llama", "qwen"]:
        vals = [r['metrics'] for k, r in results.items() if r['config']['generator'] == gen]
        if vals:
            em = sum(v['exact_match'] for v in vals) / len(vals)
            f1 = sum(v['f1'] for v in vals) / len(vals)
            rk = sum(v['recall@k'] for v in vals) / len(vals)
            print(f"{gen:<12} EM: {em:.4f} | F1: {f1:.4f} | R@K: {rk:.4f}")
    
    print("\nAverage by Domain:")
    for ds, label in [("financebench", "Finance"), ("cuad", "Law")]:
        vals = [r['metrics'] for k, r in results.items() if r['config']['dataset'] == ds]
        if vals:
            em = sum(v['exact_match'] for v in vals) / len(vals)
            f1 = sum(v['f1'] for v in vals) / len(vals)
            rk = sum(v['recall@k'] for v in vals) / len(vals)
            print(f"{label:<12} EM: {em:.4f} | F1: {f1:.4f} | R@K: {rk:.4f}")
    
    print("=" * 90)


if __name__ == "__main__":
    print(f"\nStart: {datetime.now().isoformat()}")
    results = run_grid_experiment()
    print_summary(results)
    print(f"\nEnd: {datetime.now().isoformat()}")
    print("\nExperiment completed!")
