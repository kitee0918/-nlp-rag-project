"""提示词模板集中定义。"""

QWEN_RAG_PROMPT = """你是一个智能问答助手。请根据以下提供的文档信息回答问题。
如果文档中没有足够的信息回答问题，请说“根据提供的文档，我无法回答这个问题”。

文档信息:
{context}

问题: {question}

请给出简洁准确的回答:"""

LLAMA_RAG_PROMPT = """You are a helpful AI assistant. Answer the question based on the provided documents.
If the documents don't contain enough information, say \"I cannot answer this question based on the provided documents.\"

Documents:
{context}

Question: {question}

Please provide a concise and accurate answer:"""

SELF_RAG_NEED_RETRIEVE_PROMPT = """判断回答以下问题是否需要检索外部知识。

问题：{question}

请只回答“需要检索”或“不需要检索”。"""

__all__ = [
    "QWEN_RAG_PROMPT",
    "LLAMA_RAG_PROMPT",
    "SELF_RAG_NEED_RETRIEVE_PROMPT",
]
