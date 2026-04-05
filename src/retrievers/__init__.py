"""Retrievers module for RAG system."""

from .bm25_retriever import BM25Retriever
from .contriever_retriever import ContrieverRetriever
from .hybrid_retriever import HybridRetriever

__all__ = ["BM25Retriever", "ContrieverRetriever", "HybridRetriever"]


def get_retriever(retriever_type: str, documents: list, **kwargs):
    """Factory function to create retrievers."""
    retrievers = {
        "bm25": BM25Retriever,
        "contriever": ContrieverRetriever,
        "hybrid": HybridRetriever,
    }
    if retriever_type not in retrievers:
        raise ValueError(f"Unknown retriever type: {retriever_type}")
    return retrievers[retriever_type](documents, **kwargs)
