"""LLM Generator Implementation."""

import torch
from typing import List, Dict, Optional
from transformers import AutoTokenizer, AutoModelForCausalLM
from src.config import MODELS


class LLMGenerator:
    """Generator using instruction-tuned LLMs."""
    
    # Model-specific prompt templates
    PROMPT_TEMPLATES = {
        "llama": """<|begin_of_text|><|start_header_id|>system<|end_header_id|>

You are a helpful assistant. Answer the question based on the provided context. If the answer cannot be found in the context, say "I cannot answer this question based on the provided context."

<|eot_id|><|start_header_id|>user<|end_header_id|>

Context:
{context}

Question: {question}

<|eot_id|><|start_header_id|>assistant<|end_header_id|>
""",
        "qwen": """<|im_start|>system
You are a helpful assistant. Answer the question based on the provided context. If the answer cannot be found in the context, say "I cannot answer this question based on the provided context."<|im_end|>
<|im_start|>user
Context:
{context}

Question: {question}<|im_end|>
<|im_start|>assistant
"""
    }
    
    def __init__(
        self,
        model_type: str,
        device: str = "cuda",
        load_in_8bit: bool = False,
        **kwargs
    ):
        """
        Initialize LLM generator.
        
        Args:
            model_type: "llama" or "qwen"
            device: Device to run model on
            load_in_8bit: Whether to use 8-bit quantization
        """
        self.model_type = model_type
        self.device = device
        self.model_id = MODELS[model_type]
        
        # Load tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_id,
            trust_remote_code=True
        )
        
        # Set padding token
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        
        # Load model
        model_kwargs = {
            "trust_remote_code": True,
            "torch_dtype": torch.float16,
        }
        if load_in_8bit:
            model_kwargs["load_in_8bit"] = True
            model_kwargs["device_map"] = "auto"
        else:
            model_kwargs["device_map"] = device
        
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_id,
            **model_kwargs
        )
        self.model.eval()
    
    def format_context(self, documents: List[Dict]) -> str:
        """Format retrieved documents into context string."""
        context_parts = []
        for i, doc in enumerate(documents, 1):
            context_parts.append(f"[Document {i}]\n{doc['text']}\n")
        return "\n".join(context_parts)
    
    def format_prompt(self, question: str, documents: List[Dict]) -> str:
        """Format the full prompt with context and question."""
        context = self.format_context(documents)
        template = self.PROMPT_TEMPLATES[self.model_type]
        return template.format(context=context, question=question)
    
    @torch.no_grad()
    def generate(
        self,
        question: str,
        documents: List[Dict],
        max_new_tokens: int = 512,
        temperature: float = 0.7,
        top_p: float = 0.9,
        do_sample: bool = True
    ) -> Dict:
        """
        Generate answer given question and retrieved documents.
        
        Args:
            question: The question to answer
            documents: List of retrieved documents
            max_new_tokens: Maximum tokens to generate
            temperature: Sampling temperature
            top_p: Top-p sampling parameter
            do_sample: Whether to sample
            
        Returns:
            Dict with 'answer', 'prompt', and metadata
        """
        # Format prompt
        prompt = self.format_prompt(question, documents)
        
        # Tokenize
        inputs = self.tokenizer(
            prompt,
            return_tensors="pt",
            truncation=True,
            max_length=4096
        ).to(self.device)
        
        # Generate
        outputs = self.model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_p=top_p,
            do_sample=do_sample,
            pad_token_id=self.tokenizer.pad_token_id,
            eos_token_id=self.tokenizer.eos_token_id
        )
        
        # Decode
        generated_text = self.tokenizer.decode(
            outputs[0],
            skip_special_tokens=True
        )
        
        # Extract only the generated answer (remove prompt)
        answer = generated_text[len(prompt):].strip()
        
        return {
            "answer": answer,
            "prompt": prompt,
            "num_docs": len(documents),
            "input_tokens": inputs['input_ids'].shape[1],
            "output_tokens": outputs.shape[1] - inputs['input_ids'].shape[1]
        }
    
    def generate_batch(
        self,
        questions: List[str],
        documents_list: List[List[Dict]],
        **generation_kwargs
    ) -> List[Dict]:
        """Generate answers for multiple questions."""
        results = []
        for question, documents in zip(questions, documents_list):
            result = self.generate(question, documents, **generation_kwargs)
            results.append(result)
        return results
