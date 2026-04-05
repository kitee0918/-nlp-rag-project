"""Contriever Dense Retriever Implementation."""

import os
import pickle
from typing import List, Tuple, Optional
import torch
import numpy as np
from transformers import AutoTokenizer, AutoModel


class ContrieverRetriever:
    """Dense retrieval using Facebook Contriever model."""
    
    def __init__(
        self, 
        documents: List[dict],
        model_name: str = "facebook/contriever-msmarco",
        index_path: Optional[str] = None,
        device: str = "cuda"
    ):
        """
        Initialize Contriever retriever.
        
        Args:
            documents: List of document dicts with 'id', 'text', 'metadata' keys
            model_name: HuggingFace model ID
            index_path: Path to load/save FAISS index
            device: Device to run model on
        """
        self.documents = documents
        self.doc_ids = [doc['id'] for doc in documents]
        self.device = device
        self.model_name = model_name
        
        # Load model and tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name)
        self.model.to(device)
        self.model.eval()
        
        # Build or load embeddings
        if index_path and self._load_index(index_path):
            pass
        else:
            self.embeddings = self._encode_documents()
            if index_path:
                self._save_index(index_path)
    
    def _mean_pooling(self, token_embeddings, attention_mask):
        """Apply mean pooling to get sentence embeddings."""
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
        return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(
            input_mask_expanded.sum(1), min=1e-9
        )
    
    @torch.no_grad()
    def _encode_texts(self, texts: List[str], batch_size: int = 32) -> np.ndarray:
        """Encode texts to embeddings."""
        all_embeddings = []
        
        for i in range(0, len(texts), batch_size):
            batch_texts = texts[i:i + batch_size]
            
            # Tokenize
            encoded = self.tokenizer(
                batch_texts,
                padding=True,
                truncation=True,
                max_length=512,
                return_tensors="pt"
            ).to(self.device)
            
            # Get embeddings
            outputs = self.model(**encoded)
            embeddings = self._mean_pooling(outputs.last_hidden_state, encoded['attention_mask'])
            
            # Normalize
            embeddings = torch.nn.functional.normalize(embeddings, p=2, dim=1)
            
            all_embeddings.append(embeddings.cpu().numpy())
        
        return np.vstack(all_embeddings)
    
    def _encode_documents(self) -> np.ndarray:
        """Encode all documents to embeddings."""
        texts = [doc['text'] for doc in self.documents]
        return self._encode_texts(texts)
    
    def _save_index(self, path: str):
        """Save embeddings to disk."""
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, 'wb') as f:
            pickle.dump({
                'embeddings': self.embeddings,
                'doc_ids': self.doc_ids,
                'documents': self.documents
            }, f)
    
    def _load_index(self, path: str) -> bool:
        """Load embeddings from disk."""
        try:
            with open(path, 'rb') as f:
                data = pickle.load(f)
                self.embeddings = data['embeddings']
                self.doc_ids = data['doc_ids']
                self.documents = data['documents']
            return True
        except FileNotFoundError:
            return False
    
    @torch.no_grad()
    def retrieve(self, query: str, top_k: int = 3) -> List[Tuple[dict, float]]:
        """
        Retrieve top-k documents for a query.
        
        Args:
            query: Query string
            top_k: Number of documents to retrieve
            
        Returns:
            List of (document, score) tuples
        """
        # Encode query
        query_embedding = self._encode_texts([query])
        
        # Compute similarities (dot product for normalized vectors = cosine similarity)
        similarities = np.dot(self.embeddings, query_embedding.T).flatten()
        
        # Get top-k indices
        top_indices = np.argsort(similarities)[::-1][:top_k]
        
        results = []
        for idx in top_indices:
            results.append((self.documents[idx], float(similarities[idx])))
        
        return results
    
    def get_scores(self, query: str) -> np.ndarray:
        """Get similarity scores for all documents."""
        query_embedding = self._encode_texts([query])
        return np.dot(self.embeddings, query_embedding.T).flatten()
    
    def get_ranking(self, query: str) -> List[Tuple[str, int]]:
        """Get document IDs with their ranks."""
        scores = self.get_scores(query)
        ranked = np.argsort(scores)[::-1]
        return [(self.doc_ids[i], rank + 1) for rank, i in enumerate(ranked)]
