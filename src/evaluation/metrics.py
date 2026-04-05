"""Evaluation Metrics Implementation."""

import re
import string
from typing import List, Dict, Tuple, Optional
from collections import Counter
import numpy as np


def normalize_answer(s: str) -> str:
    """Normalize answer string for comparison."""
    # Lower case
    s = s.lower()
    # Remove punctuation
    s = ''.join(ch for ch in s if ch not in string.punctuation)
    # Remove articles
    s = re.sub(r'\b(a|an|the)\b', ' ', s)
    # Remove extra whitespace
    s = ' '.join(s.split())
    return s


def exact_match_score(prediction: str, ground_truth: str) -> float:
    """Calculate exact match score."""
    return float(normalize_answer(prediction) == normalize_answer(ground_truth))


def f1_score(prediction: str, ground_truth: str) -> float:
    """Calculate F1 score between prediction and ground truth."""
    pred_tokens = normalize_answer(prediction).split()
    gt_tokens = normalize_answer(ground_truth).split()
    
    if len(pred_tokens) == 0 or len(gt_tokens) == 0:
        return float(pred_tokens == gt_tokens)
    
    common = Counter(pred_tokens) & Counter(gt_tokens)
    num_common = sum(common.values())
    
    if num_common == 0:
        return 0.0
    
    precision = num_common / len(pred_tokens)
    recall = num_common / len(gt_tokens)
    
    return 2 * precision * recall / (precision + recall)


def compute_recall_at_k(
    retrieved_ids: List[str],
    gold_ids: List[str],
    k: int
) -> float:
    """
    Compute Recall@k.
    
    Args:
        retrieved_ids: List of retrieved document IDs (ordered)
        gold_ids: List of gold document IDs
        k: Number of top documents to consider
        
    Returns:
        Recall@k score (1.0 if all gold docs are in top-k, else partial)
    """
    top_k = set(retrieved_ids[:k])
    gold_set = set(gold_ids)
    
    if not gold_set:
        return 0.0
    
    return len(top_k & gold_set) / len(gold_set)


def compute_precision_at_k(
    retrieved_ids: List[str],
    gold_ids: List[str],
    k: int
) -> float:
    """
    Compute Precision@k.
    
    Args:
        retrieved_ids: List of retrieved document IDs (ordered)
        gold_ids: List of gold document IDs
        k: Number of top documents to consider
        
    Returns:
        Precision@k score
    """
    top_k = retrieved_ids[:k]
    gold_set = set(gold_ids)
    
    if not top_k:
        return 0.0
    
    correct = sum(1 for doc_id in top_k if doc_id in gold_set)
    return correct / len(top_k)


def compute_metrics(
    prediction: str,
    ground_truth: str,
    retrieved_ids: Optional[List[str]] = None,
    gold_ids: Optional[List[str]] = None,
    k: int = 3
) -> Dict[str, float]:
    """
    Compute all metrics for a single prediction.
    
    Args:
        prediction: Generated answer
        ground_truth: Ground truth answer
        retrieved_ids: Retrieved document IDs
        gold_ids: Gold document IDs
        k: Top-k for retrieval metrics
        
    Returns:
        Dict of metric names to values
    """
    metrics = {
        "exact_match": exact_match_score(prediction, ground_truth),
        "f1": f1_score(prediction, ground_truth)
    }
    
    if retrieved_ids and gold_ids:
        metrics["recall_at_k"] = compute_recall_at_k(retrieved_ids, gold_ids, k)
        metrics["precision_at_k"] = compute_precision_at_k(retrieved_ids, gold_ids, k)
    
    return metrics


def aggregate_metrics(metrics_list: List[Dict[str, float]]) -> Dict[str, float]:
    """Aggregate metrics across multiple examples."""
    if not metrics_list:
        return {}
    
    aggregated = {}
    keys = metrics_list[0].keys()
    
    for key in keys:
        values = [m[key] for m in metrics_list if key in m]
        aggregated[key] = np.mean(values)
    
    return aggregated


class FActScorer:
    """
    FActScore implementation for factual precision evaluation.
    
    Simplified version that checks factual claims against retrieved context.
    """
    
    def __init__(self, llm_check: bool = False):
        """
        Initialize FActScorer.
        
        Args:
            llm_check: Whether to use LLM for fact verification
        """
        self.llm_check = llm_check
    
    def extract_claims(self, text: str) -> List[str]:
        """Extract atomic factual claims from text."""
        # Simple sentence-based claim extraction
        sentences = re.split(r'[.!?]+', text)
        claims = [s.strip() for s in sentences if len(s.strip()) > 10]
        return claims
    
    def check_claim_supported(
        self,
        claim: str,
        context: str
    ) -> bool:
        """Check if a claim is supported by context."""
        # Simple overlap-based check (can be replaced with LLM)
        claim_words = set(normalize_answer(claim).split())
        context_words = set(normalize_answer(context).split())
        
        # Remove common words
        stopwords = {'a', 'an', 'the', 'is', 'are', 'was', 'were', 'be', 'been',
                     'being', 'have', 'has', 'had', 'do', 'does', 'did', 'will',
                     'would', 'could', 'should', 'may', 'might', 'must', 'shall',
                     'can', 'need', 'dare', 'ought', 'used', 'to', 'of', 'in',
                     'for', 'on', 'with', 'at', 'by', 'from', 'as', 'into',
                     'through', 'during', 'before', 'after', 'above', 'below',
                     'between', 'under', 'again', 'further', 'then', 'once'}
        
        claim_keywords = claim_words - stopwords
        context_keywords = context_words - stopwords
        
        if not claim_keywords:
            return True
        
        overlap = claim_keywords & context_keywords
        return len(overlap) / len(claim_keywords) >= 0.5
    
    def compute_factscore(
        self,
        generated_answer: str,
        context: str
    ) -> Dict[str, float]:
        """
        Compute FActScore for generated answer.
        
        Args:
            generated_answer: The generated answer text
            context: The retrieved context
            
        Returns:
            Dict with FActScore and related metrics
        """
        claims = self.extract_claims(generated_answer)
        
        if not claims:
            return {"factscore": 1.0, "num_claims": 0, "supported_claims": 0}
        
        supported = 0
        for claim in claims:
            if self.check_claim_supported(claim, context):
                supported += 1
        
        return {
            "factscore": supported / len(claims),
            "num_claims": len(claims),
            "supported_claims": supported
        }
