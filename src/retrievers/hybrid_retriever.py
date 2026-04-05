"""Hybrid RRF Retriever Implementation."""

import os
import pickle
from typing import List, Tuple, Optional, Dict
from .bm25_retriever import BM25Retriever
from .contriever_retriever import ContrieverRetriever


class HybridRetriever:
    """Hybrid retrieval combining BM25 and Contriever using Reciprocal Rank Fusion."""
    
    def __init__(
        self,
        documents: List[dict],
        bm25_index_path: Optional[str] = None,
        contriever_index_path: Optional[str] = None,
        rrf_k: int = 60,
        device: str = "cuda"
    ):
        """
        Initialize Hybrid retriever.
        
        Args:
            documents: List of document dicts
            bm25_index_path: Path for BM25 index
            contriever_index_path: Path for Contriever index
            rrf_k: RRF constant (default 60)
            device: Device for dense retrieval
        """
        self.documents = documents
        self.rrf_k = rrf_k
        
        # Initialize both retrievers
        self.bm25 = BM25Retriever(documents, index_path=bm25_index_path)
        self.contriever = ContrieverRetriever(
            documents, 
            index_path=contriever_index_path,
            device=device
        )
    
    def _reciprocal_rank_fusion(
        self, 
        rankings: Dict[str, Dict[str, int]]
    ) -> List[Tuple[str, float]]:
        """
        Apply Reciprocal Rank Fusion to multiple rankings.
        
        RRF(d) = sum(1 / (k + rank(d)) for each ranking)
        
        Args:
            rankings: Dict mapping retriever name -> {doc_id: rank}
            
        Returns:
            Sorted list of (doc_id, fused_score)
        """
        fused_scores = {}
        
        for doc_id in self.bm25.doc_ids:
            score = 0.0
            for retriever_name, ranking in rankings.items():
                if doc_id in ranking:
                    rank = ranking[doc_id]
                    score += 1.0 / (self.rrf_k + rank)
            fused_scores[doc_id] = score
        
        # Sort by fused score
        sorted_docs = sorted(fused_scores.items(), key=lambda x: x[1], reverse=True)
        return sorted_docs
    
    def retrieve(self, query: str, top_k: int = 3) -> List[Tuple[dict, float]]:
        """
        Retrieve top-k documents using hybrid RRF.
        
        Args:
            query: Query string
            top_k: Number of documents to retrieve
            
        Returns:
            List of (document, score) tuples
        """
        # Get rankings from both retrievers
        bm25_ranking = self.bm25.get_ranking(query)
        contriever_ranking = self.contriever.get_ranking(query)
        
        # Convert to dict format
        rankings = {
            'bm25': {doc_id: rank for doc_id, rank in bm25_ranking},
            'contriever': {doc_id: rank for doc_id, rank in contriever_ranking}
        }
        
        # Apply RRF
        fused = self._reciprocal_rank_fusion(rankings)
        
        # Get top-k documents
        doc_id_to_doc = {doc['id']: doc for doc in self.documents}
        results = []
        
        for doc_id, score in fused[:top_k]:
            results.append((doc_id_to_doc[doc_id], score))
        
        return results
    
    def get_individual_scores(self, query: str) -> Dict[str, Dict[str, float]]:
        """Get scores from individual retrievers."""
        bm25_scores = self.bm25.get_scores(query)
        contriever_scores = self.contriever.get_scores(query)
        
        return {
            'bm25': {self.bm25.doc_ids[i]: float(s) for i, s in enumerate(bm25_scores)},
            'contriever': {self.contriever.doc_ids[i]: float(s) for i, s in enumerate(contriever_scores)}
        }
