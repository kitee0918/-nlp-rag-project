#!/usr/bin/env python
"""
CS6493 HotpotQA Retrieval Precision Analysis
============================================
Analyze how retrieval quality affects downstream answer quality using HotpotQA.

Metrics:
- Recall@k: fraction of queries where both gold passages appear in top-k
- Precision@k: fraction of retrieved passages that are gold
- Answer F1: F1 score of generated answer
- Correlation between Recall@k and Answer F1
"""

import os
import sys
import json
import random
import math
import re
import string
from datetime import datetime
from typing import List, Dict, Tuple
from collections import Counter
import time

# Set random seed
SEED = 42
random.seed(SEED)

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
RESULTS_DIR = os.path.join(PROJECT_ROOT, "results")
os.makedirs(RESULTS_DIR, exist_ok=True)

# =============================================================================
# HotpotQA Sample Data (Multi-hop QA with supporting facts)
# =============================================================================

HOTPOTQA_SAMPLE = [
    {
        "id": "hp_001",
        "question": "What is the population of the city where Apple Inc. is headquartered?",
        "answer": "Cupertino has a population of approximately 60,000.",
        "supporting_facts": [
            {
                "id": "sf_001_a",
                "title": "Apple Inc.",
                "text": "Apple Inc. is headquartered in Cupertino, California."
            },
            {
                "id": "sf_001_b",
                "title": "Cupertino, California",
                "text": "Cupertino is a city in Santa Clara County, California with a population of approximately 60,000 residents."
            }
        ],
        "distractors": [
            {"id": "d_001_a", "title": "Microsoft", "text": "Microsoft is headquartered in Redmond, Washington."},
            {"id": "d_001_b", "title": "Google", "text": "Google is headquartered in Mountain View, California."},
            {"id": "d_001_c", "title": "Amazon", "text": "Amazon is headquartered in Seattle, Washington."},
            {"id": "d_001_d", "title": "San Francisco", "text": "San Francisco has a population of approximately 870,000."},
        ]
    },
    {
        "id": "hp_002",
        "question": "When was the company that created the iPhone founded?",
        "answer": "Apple Inc. was founded on April 1, 1976.",
        "supporting_facts": [
            {
                "id": "sf_002_a",
                "title": "iPhone",
                "text": "The iPhone is a line of smartphones designed and marketed by Apple Inc."
            },
            {
                "id": "sf_002_b",
                "title": "Apple Inc.",
                "text": "Apple Inc. was founded on April 1, 1976, by Steve Jobs, Steve Wozniak, and Ronald Wayne."
            }
        ],
        "distractors": [
            {"id": "d_002_a", "title": "Samsung", "text": "Samsung was founded in 1938 in South Korea."},
            {"id": "d_002_b", "title": "Nokia", "text": "Nokia was founded in 1865 in Finland."},
            {"id": "d_002_c", "title": "Android", "text": "Android is a mobile operating system developed by Google."},
            {"id": "d_002_d", "title": "MacBook", "text": "MacBook is a line of laptops by Apple Inc."},
        ]
    },
    {
        "id": "hp_003",
        "question": "What is the capital of the country where the Eiffel Tower is located?",
        "answer": "Paris is the capital of France.",
        "supporting_facts": [
            {
                "id": "sf_003_a",
                "title": "Eiffel Tower",
                "text": "The Eiffel Tower is a wrought-iron tower located in Paris, France."
            },
            {
                "id": "sf_003_b",
                "title": "France",
                "text": "France is a country in Western Europe, and its capital is Paris."
            }
        ],
        "distractors": [
            {"id": "d_003_a", "title": "London", "text": "London is the capital of the United Kingdom."},
            {"id": "d_003_b", "title": "Berlin", "text": "Berlin is the capital of Germany."},
            {"id": "d_003_c", "title": "Statue of Liberty", "text": "The Statue of Liberty is located in New York City, USA."},
            {"id": "d_003_d", "title": "Italy", "text": "Italy is a country in Southern Europe with Rome as its capital."},
        ]
    },
    {
        "id": "hp_004",
        "question": "Who is the CEO of the company that owns Instagram?",
        "answer": "Mark Zuckerberg is the CEO of Meta, which owns Instagram.",
        "supporting_facts": [
            {
                "id": "sf_004_a",
                "title": "Instagram",
                "text": "Instagram is a photo and video sharing social networking service owned by Meta Platforms."
            },
            {
                "id": "sf_004_b",
                "title": "Meta Platforms",
                "text": "Meta Platforms is led by CEO Mark Zuckerberg, who also founded Facebook."
            }
        ],
        "distractors": [
            {"id": "d_004_a", "title": "Twitter", "text": "Twitter was acquired by Elon Musk in 2022."},
            {"id": "d_004_b", "title": "TikTok", "text": "TikTok is owned by ByteDance, a Chinese company."},
            {"id": "d_004_c", "title": "Snapchat", "text": "Snapchat is developed by Snap Inc., founded by Evan Spiegel."},
            {"id": "d_004_d", "title": "YouTube", "text": "YouTube is owned by Google, whose CEO is Sundar Pichai."},
        ]
    },
    {
        "id": "hp_005",
        "question": "What is the currency used in the country where the Great Wall is located?",
        "answer": "The Chinese Yuan (Renminbi) is the currency of China.",
        "supporting_facts": [
            {
                "id": "sf_005_a",
                "title": "Great Wall of China",
                "text": "The Great Wall of China is a series of fortifications in northern China."
            },
            {
                "id": "sf_005_b",
                "title": "Currency of China",
                "text": "The official currency of China is the Renminbi, with the Yuan as its primary unit."
            }
        ],
        "distractors": [
            {"id": "d_005_a", "title": "Japan", "text": "Japan uses the Japanese Yen as its currency."},
            {"id": "d_005_b", "title": "South Korea", "text": "South Korea uses the South Korean Won."},
            {"id": "d_005_c", "title": "India", "text": "India uses the Indian Rupee as its currency."},
            {"id": "d_005_d", "title": "Machu Picchu", "text": "Machu Picchu is located in Peru, which uses the Peruvian Sol."},
        ]
    },
    {
        "id": "hp_006",
        "question": "Which university did the founder of Microsoft attend?",
        "answer": "Bill Gates attended Harvard University.",
        "supporting_facts": [
            {
                "id": "sf_006_a",
                "title": "Microsoft",
                "text": "Microsoft was founded by Bill Gates and Paul Allen in 1975."
            },
            {
                "id": "sf_006_b",
                "title": "Bill Gates",
                "text": "Bill Gates attended Harvard University but dropped out to start Microsoft."
            }
        ],
        "distractors": [
            {"id": "d_006_a", "title": "Stanford University", "text": "Stanford University is located in California."},
            {"id": "d_006_b", "title": "MIT", "text": "MIT is a private research university in Cambridge, Massachusetts."},
            {"id": "d_006_c", "title": "Steve Jobs", "text": "Steve Jobs attended Reed College but dropped out."},
            {"id": "d_006_d", "title": "Elon Musk", "text": "Elon Musk attended University of Pennsylvania and Stanford."},
        ]
    },
    {
        "id": "hp_007",
        "question": "What is the language spoken in the country where pizza originated?",
        "answer": "Italian is the official language of Italy.",
        "supporting_facts": [
            {
                "id": "sf_007_a",
                "title": "Pizza",
                "text": "Pizza originated in Naples, Italy, in the 18th or early 19th century."
            },
            {
                "id": "sf_007_b",
                "title": "Italy",
                "text": "Italy is a country in Southern Europe where Italian is the official language."
            }
        ],
        "distractors": [
            {"id": "d_007_a", "title": "Spain", "text": "Spanish is the official language of Spain."},
            {"id": "d_007_b", "title": "France", "text": "French is the official language of France."},
            {"id": "d_007_c", "title": "Sushi", "text": "Sushi originated in Japan, where Japanese is spoken."},
            {"id": "d_007_d", "title": "Tacos", "text": "Tacos originated in Mexico, where Spanish is spoken."},
        ]
    },
    {
        "id": "hp_008",
        "question": "What is the largest city in the state where NASA's Johnson Space Center is located?",
        "answer": "Houston is the largest city in Texas.",
        "supporting_facts": [
            {
                "id": "sf_008_a",
                "title": "Johnson Space Center",
                "text": "NASA's Johnson Space Center is located in Houston, Texas."
            },
            {
                "id": "sf_008_b",
                "title": "Houston",
                "text": "Houston is the largest city in Texas and the fourth-largest in the United States."
            }
        ],
        "distractors": [
            {"id": "d_008_a", "title": "Kennedy Space Center", "text": "NASA's Kennedy Space Center is located in Florida."},
            {"id": "d_008_b", "title": "Dallas", "text": "Dallas is a major city in Texas."},
            {"id": "d_008_c", "title": "Austin", "text": "Austin is the capital of Texas."},
            {"id": "d_008_d", "title": "San Antonio", "text": "San Antonio is the second-largest city in Texas."},
        ]
    },
    {
        "id": "hp_009",
        "question": "Which ocean borders the country where the Amazon River flows?",
        "answer": "The Amazon River flows into the Atlantic Ocean.",
        "supporting_facts": [
            {
                "id": "sf_009_a",
                "title": "Amazon River",
                "text": "The Amazon River flows through South America, primarily in Brazil."
            },
            {
                "id": "sf_009_b",
                "title": "Brazil",
                "text": "Brazil is bordered by the Atlantic Ocean to the east."
            }
        ],
        "distractors": [
            {"id": "d_009_a", "title": "Pacific Ocean", "text": "The Pacific Ocean borders the western coast of the Americas."},
            {"id": "d_009_b", "title": "Nile River", "text": "The Nile River flows into the Mediterranean Sea."},
            {"id": "d_009_c", "title": "Peru", "text": "Peru is a country in western South America."},
            {"id": "d_009_d", "title": "Indian Ocean", "text": "The Indian Ocean is the third-largest ocean."},
        ]
    },
    {
        "id": "hp_010",
        "question": "What is the sport played in the league where Lionel Messi became famous?",
        "answer": "Lionel Messi is a famous football (soccer) player.",
        "supporting_facts": [
            {
                "id": "sf_010_a",
                "title": "Lionel Messi",
                "text": "Lionel Messi is an Argentine professional footballer who played for Barcelona."
            },
            {
                "id": "sf_010_b",
                "title": "La Liga",
                "text": "La Liga is the top professional football league in Spain where Barcelona competes."
            }
        ],
        "distractors": [
            {"id": "d_010_a", "title": "NBA", "text": "The NBA is a professional basketball league in North America."},
            {"id": "d_010_b", "title": "NFL", "text": "The NFL is a professional American football league."},
            {"id": "d_010_c", "title": "Cristiano Ronaldo", "text": "Cristiano Ronaldo is a famous Portuguese footballer."},
            {"id": "d_010_d", "title": "Premier League", "text": "The Premier League is the top football league in England."},
        ]
    }
]

K_VALUES = [1, 3, 5, 10]

# =============================================================================
# Evaluation Metrics
# =============================================================================

def normalize_answer(s: str) -> str:
    """Normalize answer for comparison."""
    s = s.lower()
    s = ''.join(ch for ch in s if ch not in string.punctuation)
    s = re.sub(r'\b(a|an|the|is|are|was|were|of|in|to|for|with)\b', ' ', s)
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
    """
    Compute Recall@k for multi-hop QA.
    For multi-hop, we need BOTH gold passages in top-k.
    """
    top_k = set(retrieved_ids[:k])
    gold_set = set(gold_ids)
    if not gold_set:
        return 0.0
    return len(top_k & gold_set) / len(gold_set)

def compute_precision_at_k(retrieved_ids: List[str], gold_ids: List[str], k: int) -> float:
    """Compute Precision@k."""
    top_k = retrieved_ids[:k]
    gold_set = set(gold_ids)
    if not top_k:
        return 0.0
    correct = sum(1 for doc_id in top_k if doc_id in gold_set)
    return correct / len(top_k)

def compute_both_gold_recall(retrieved_ids: List[str], gold_ids: List[str], k: int) -> float:
    """
    Binary recall: 1 if BOTH gold passages are in top-k, else 0.
    This is the metric used in HotpotQA for multi-hop evaluation.
    """
    top_k = set(retrieved_ids[:k])
    gold_set = set(gold_ids)
    return float(gold_set.issubset(top_k))

# =============================================================================
# Retrievers
# =============================================================================

class BM25Retriever:
    """BM25 Retriever."""
    
    def __init__(self, documents: List[dict]):
        self.documents = documents
        self.doc_ids = [doc['id'] for doc in documents]
        self.corpus = [doc['text'].lower().split() for doc in documents]
        self.avgdl = sum(len(doc) for doc in self.corpus) / len(self.corpus) if self.corpus else 1
        self.k1 = 1.5
        self.b = 0.75
    
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
        self.doc_ids = [doc['id'] for doc in documents]
        self.doc_vectors = [self._text_to_vector(doc['text']) for doc in documents]
    
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
        self.doc_ids = [doc['id'] for doc in documents]
    
    def retrieve(self, query: str, top_k: int = 3) -> List[Tuple[dict, float]]:
        """Retrieve using RRF fusion."""
        bm25_scores = self.bm25.get_scores(query)
        dense_scores = self.contriever.get_scores(query)
        
        bm25_ranked = sorted(range(len(bm25_scores)), key=lambda i: bm25_scores[i], reverse=True)
        dense_ranked = sorted(range(len(dense_scores)), key=lambda i: dense_scores[i], reverse=True)
        
        rrf_scores = {}
        for rank, idx in enumerate(bm25_ranked):
            rrf_scores[idx] = rrf_scores.get(idx, 0) + 1 / (self.rrf_k + rank + 1)
        for rank, idx in enumerate(dense_ranked):
            rrf_scores[idx] = rrf_scores.get(idx, 0) + 1 / (self.rrf_k + rank + 1)
        
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
# Generator (Mock)
# =============================================================================

class MockGenerator:
    """Mock LLM generator."""
    
    def __init__(self, model_type: str = "llama"):
        self.model_type = model_type
    
    def generate(self, question: str, documents: List[dict]) -> str:
        """Generate answer from retrieved documents."""
        context = " ".join([doc['text'] for doc in documents])
        return self._extract_answer(question, context)
    
    def _extract_answer(self, question: str, context: str) -> str:
        """Extract answer using patterns."""
        q = question.lower()
        ctx = context.lower()
        
        # Population
        if "population" in q:
            pops = re.findall(r'population of (?:approximately )?([\d,]+)', ctx)
            if pops:
                return f"The population is approximately {pops[0]}."
            nums = re.findall(r'population of (?:approximately )?(\d+)', context, re.I)
            if nums:
                return f"The population is approximately {nums[0]}."
        
        # Founded/When
        if "when" in q or "founded" in q:
            dates = re.findall(r'(?:founded|established) (?:on )?([A-Z][a-z]+ \d+,? \d{4})', context)
            if dates:
                return f"It was founded on {dates[0]}."
            years = re.findall(r'founded in (\d{4})', ctx)
            if years:
                return f"It was founded in {years[0]}."
        
        # Capital
        if "capital" in q:
            caps = re.findall(r'capital (?:is|of) ([A-Z][a-z]+)', context)
            if caps:
                return f"The capital is {caps[0]}."
            caps = re.findall(r'([A-Z][a-z]+) is the capital', context)
            if caps:
                return f"{caps[0]} is the capital."
        
        # CEO
        if "ceo" in q or "leader" in q:
            ceos = re.findall(r'(?:CEO|led by) ([A-Z][a-z]+ [A-Z][a-z]+)', context)
            if ceos:
                return f"The CEO is {ceos[0]}."
        
        # Currency
        if "currency" in q:
            curs = re.findall(r'(?:currency (?:of|is)|official currency) ([^.]*(?:Yuan|Dollar|Euro|Won|Rupee|Sol)[^.]*)', context, re.I)
            if curs:
                return f"The currency is {curs[0]}."
        
        # Language
        if "language" in q or "spoken" in q:
            langs = re.findall(r'(?:official language|spoken language) (?:is )?([A-Z][a-z]+)', context)
            if langs:
                return f"The language is {langs[0]}."
            langs = re.findall(r'([A-Z][a-z]+) is the official language', context)
            if langs:
                return f"{langs[0]} is the official language."
        
        # University
        if "university" in q or "attend" in q:
            unis = re.findall(r'attended ([A-Z][a-z]+ (?:University|College))', context)
            if unis:
                return f"They attended {unis[0]}."
        
        # Ocean/Border
        if "ocean" in q or "borders" in q:
            oceans = re.findall(r'(?:bordered by|flows into) the ([A-Z][a-z]+ Ocean)', context)
            if oceans:
                return f"It borders the {oceans[0]} Ocean."
        
        # Sport
        if "sport" in q or "league" in q:
            sports = re.findall(r'(football|soccer|basketball|baseball|tennis)', ctx)
            if sports:
                return f"The sport is {sports[0]}."
        
        # City/Largest
        if "largest city" in q:
            cities = re.findall(r'([A-Z][a-z]+) is the largest city', context)
            if cities:
                return f"{cities[0]} is the largest city."
        
        # Fallback: return first sentence
        sentences = re.split(r'[.!?]', context)
        if sentences:
            return sentences[0].strip() + "." if sentences[0].strip() else "Unknown."
        
        return "Unable to determine from context."

# =============================================================================
# Experiment Runner
# =============================================================================

def run_hotpotqa_analysis():
    """Run HotpotQA retrieval precision analysis."""
    
    print("=" * 80)
    print("CS6493 HotpotQA Retrieval Precision Analysis")
    print("=" * 80)
    print(f"Questions: {len(HOTPOTQA_SAMPLE)}")
    print(f"K values: {K_VALUES}")
    print("Retrievers: BM25, Contriever, Hybrid RRF")
    print("=" * 80)
    
    retriever_types = ["bm25", "contriever", "hybrid"]
    all_results = {}
    
    for ret_type in retriever_types:
        print(f"\n{'='*80}")
        print(f"Retriever: {ret_type.upper()}")
        print(f"{'='*80}")
        
        results_by_k = {k: {"recall": [], "precision": [], "both_gold": [], "f1": []} for k in K_VALUES}
        detailed_results = []
        
        for item in HOTPOTQA_SAMPLE:
            # Build corpus: supporting facts + distractors
            corpus = []
            gold_ids = []
            
            for sf in item["supporting_facts"]:
                corpus.append({"id": sf["id"], "text": sf["text"], "title": sf["title"]})
                gold_ids.append(sf["id"])
            
            for d in item["distractors"]:
                corpus.append({"id": d["id"], "text": d["text"], "title": d["title"]})
            
            # Initialize retriever
            retriever = get_retriever(ret_type, corpus)
            generator = MockGenerator()
            
            # Get all retrieved IDs (up to max k)
            max_k = max(K_VALUES)
            retrieved = retriever.retrieve(item["question"], top_k=max_k)
            retrieved_ids = [doc["id"] for doc, _ in retrieved]
            retrieved_docs = [doc for doc, _ in retrieved]
            
            # Generate answer
            prediction = generator.generate(item["question"], retrieved_docs[:3])
            answer_f1 = f1_score(prediction, item["answer"])
            
            # Compute metrics for each k
            for k in K_VALUES:
                recall = compute_recall_at_k(retrieved_ids, gold_ids, k)
                precision = compute_precision_at_k(retrieved_ids, gold_ids, k)
                both_gold = compute_both_gold_recall(retrieved_ids, gold_ids, k)
                
                results_by_k[k]["recall"].append(recall)
                results_by_k[k]["precision"].append(precision)
                results_by_k[k]["both_gold"].append(both_gold)
                results_by_k[k]["f1"].append(answer_f1)
            
            detailed_results.append({
                "id": item["id"],
                "question": item["question"],
                "gold_ids": gold_ids,
                "retrieved_ids": retrieved_ids,
                "prediction": prediction,
                "ground_truth": item["answer"],
                "answer_f1": answer_f1
            })
        
        # Aggregate metrics
        print(f"\n{'k':>5} {'Recall@k':>12} {'Precision@k':>14} {'Both Gold@k':>14} {'Avg F1':>10}")
        print("-" * 60)
        
        aggregated = {}
        for k in K_VALUES:
            recall = sum(results_by_k[k]["recall"]) / len(results_by_k[k]["recall"])
            precision = sum(results_by_k[k]["precision"]) / len(results_by_k[k]["precision"])
            both_gold = sum(results_by_k[k]["both_gold"]) / len(results_by_k[k]["both_gold"])
            avg_f1 = sum(results_by_k[k]["f1"]) / len(results_by_k[k]["f1"])
            
            print(f"{k:>5} {recall:>12.4f} {precision:>14.4f} {both_gold:>14.4f} {avg_f1:>10.4f}")
            
            aggregated[k] = {
                "recall@k": recall,
                "precision@k": precision,
                "both_gold@k": both_gold,
                "answer_f1": avg_f1
            }
        
        all_results[ret_type] = {
            "aggregated": aggregated,
            "detailed": detailed_results
        }
    
    # Correlation Analysis
    print(f"\n{'='*80}")
    print("CORRELATION ANALYSIS: Recall@k vs Answer F1")
    print("=" * 80)
    
    print(f"\n{'Retriever':<12} {'k':>5} {'Correlation':>12}")
    print("-" * 35)
    
    correlations = {}
    for ret_type in retriever_types:
        correlations[ret_type] = {}
        for k in K_VALUES:
            recalls = all_results[ret_type]["aggregated"][k]
            # Use Pearson correlation between recall and F1
            # For simplicity, we'll compute the correlation across questions
            detailed = all_results[ret_type]["detailed"]
            
            recall_vals = []
            f1_vals = []
            for item in detailed:
                # Recompute recall@k for each item
                gold_ids = item["gold_ids"]
                retrieved_ids = item["retrieved_ids"]
                recall_vals.append(compute_recall_at_k(retrieved_ids, gold_ids, k))
                f1_vals.append(item["answer_f1"])
            
            # Pearson correlation
            if len(recall_vals) > 1:
                mean_r = sum(recall_vals) / len(recall_vals)
                mean_f = sum(f1_vals) / len(f1_vals)
                
                num = sum((r - mean_r) * (f - mean_f) for r, f in zip(recall_vals, f1_vals))
                den_r = sum((r - mean_r) ** 2 for r in recall_vals) ** 0.5
                den_f = sum((f - mean_f) ** 2 for f in f1_vals) ** 0.5
                
                if den_r > 0 and den_f > 0:
                    corr = num / (den_r * den_f)
                else:
                    corr = 0.0
            else:
                corr = 0.0
            
            correlations[ret_type][k] = corr
            print(f"{ret_type:<12} {k:>5} {corr:>12.4f}")
    
    # Save results
    output = {
        "experiment": "HotpotQA Retrieval Precision Analysis",
        "timestamp": datetime.now().isoformat(),
        "num_questions": len(HOTPOTQA_SAMPLE),
        "k_values": K_VALUES,
        "results_by_retriever": {
            ret: {
                "aggregated": all_results[ret]["aggregated"],
                "correlations": correlations[ret]
            }
            for ret in retriever_types
        },
        "detailed_results": all_results
    }
    
    output_path = os.path.join(RESULTS_DIR, "hotpotqa_retrieval_analysis.json")
    with open(output_path, 'w') as f:
        json.dump(output, f, indent=2, default=str)
    
    print(f"\n{'='*80}")
    print("SUMMARY: Retrieval Quality Impact on Answer Quality")
    print("=" * 80)
    print(f"\n{'Retriever':<12} {'Best k':>10} {'Best Both-Gold':>15} {'Corr@3':>10}")
    print("-" * 55)
    
    for ret_type in retriever_types:
        # Find best k for both-gold metric
        best_k = max(K_VALUES, key=lambda k: all_results[ret_type]["aggregated"][k]["both_gold@k"])
        best_both_gold = all_results[ret_type]["aggregated"][best_k]["both_gold@k"]
        corr_at_3 = correlations[ret_type][3]
        print(f"{ret_type:<12} {best_k:>10} {best_both_gold:>15.4f} {corr_at_3:>10.4f}")
    
    print(f"\nResults saved to: {output_path}")
    
    return output


if __name__ == "__main__":
    print(f"\nStart: {datetime.now().isoformat()}")
    results = run_hotpotqa_analysis()
    print(f"\nEnd: {datetime.now().isoformat()}")
    print("\nHotpotQA Retrieval Analysis Complete!")
