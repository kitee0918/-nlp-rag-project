"""Generators module for RAG system."""

from .llm_generator import LLMGenerator

__all__ = ["LLMGenerator"]


def get_generator(model_type: str, **kwargs):
    """Factory function to create generators."""
    generators = {
        "llama": LLMGenerator,
        "qwen": LLMGenerator,
    }
    if model_type not in generators:
        raise ValueError(f"Unknown generator type: {model_type}")
    return generators[model_type](model_type=model_type, **kwargs)
