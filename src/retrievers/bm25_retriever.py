"""BM25 Sparse Retriever Implementation."""

import pickle
from typing import List, Tuple, Optional
from rank_bm25 import BM25Okapi
import json


class BM25Retriever:
    """BM25 sparse retrieval using rank_bm25."""
    
    def __init__(self, documents: List[dict], index_path: Optional[str] = None):
        """
        Initialize BM25 retriever.
        
        Args:
            documents: List of document dicts with 'id', 'text', 'metadata' keys
            index_path: Path to load/save index (optional)
        """
        self.documents = documents
        self.doc_ids = [doc['id'] for doc in documents]
        self.corpus = [doc['text'] for doc in documents]
        
        # Tokenize
        self.tokenized_corpus = [doc.lower().split() for doc in self.corpus]
        
        # Build or load index
        if index_path and self._load_index(index_path):
            pass
        else:
            self.bm25 = BM25Okapi(self.tokenized_corpus)
            if index_path:
                self._save_index(index_path)
    
    def _save_index(self, path: str):
        """Save BM25 index to disk."""
        with open(path, 'wb') as f:
            pickle.dump({
                'bm25': self.bm25,
                'doc_ids': self.doc_ids,
                'documents': self.documents
            }, f)
    
    def _load_index(self, path: str) -> bool:
        """Load BM25 index from disk."""
        try:
            with open(path, 'rb') as f:
                data = pickle.load(f)
                self.bm25 = data['bm25']
                self.doc_ids = data['doc_ids']
                self.documents = data['documents']
            return True
        except FileNotFoundError:
            return False
    
    def retrieve(self, query: str, top_k: int = 3) -> List[Tuple[dict, float]]:
        """
        Retrieve top-k documents for a query.
        
        Args:
            query: Query string
            top_k: Number of documents to retrieve
            
        Returns:
            List of (document, score) tuples
        """
        tokenized_query = query.lower().split()
        scores = self.bm25.get_scores(tokenized_query)
        
        # Get top-k indices
        top_indices = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:top_k]
        
        results = []
        for idx in top_indices:
            results.append((self.documents[idx], scores[idx]))
        
        return results
    
    def get_scores(self, query: str) -> List[float]:
        """Get BM25 scores for all documents."""
        tokenized_query = query.lower().split()
        return self.bm25.get_scores(tokenized_query)
    
    def get_ranking(self, query: str) -> List[Tuple[str, int]]:
        """Get document IDs with their ranks."""
        scores = self.get_scores(query)
        ranked = sorted(enumerate(scores), key=lambda x: x[1], reverse=True)
        return [(self.doc_ids[i], rank + 1) for rank, (i, _) in enumerate(ranked)]
