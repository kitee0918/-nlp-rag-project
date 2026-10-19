# Retrieval-Augmented Generation Research Project

This project studies how **retrieval strategy** and **retrieved context** affect answer quality in knowledge-intensive question answering. It provides an experimental framework for comparing **sparse**, **dense**, and **hybrid retrieval** with **LLaMA**- and **Qwen**-based generation across general, financial, and legal QA tasks.

## Features

- Implements **BM25** sparse retrieval, **Contriever** dense retrieval, and **hybrid retrieval** using **Reciprocal Rank Fusion (RRF)**.
- Supports **LLaMA** and **Qwen** generators, with experiment configurations for **HotpotQA**, **FinanceBench**, **CUAD**, and **LegalBench-RAG**.
- Evaluates answer quality with **Exact Match (EM)** and **F1**, retrieval quality with **Recall@k** and **Precision@k**, and includes a lightweight context-support **FActScore** implementation.
- Includes HotpotQA retrieval-impact analysis to examine the relationship between **retrieval quality**, **context quality**, and downstream **generation quality**.

## Key Python Files

- `rag_hotpotqa.py` implements a HotpotQA RAG pipeline with BM25 and Contriever retrieval, Qwen generation through the DashScope-compatible API, and answer and retrieval evaluation. It also provides a comparison experiment for studying how retrieval choices affect QA results.
- `rag_qa_temp.py` provides a FastAPI question-answering service over a Markdown knowledge document. It splits the document into chunks, indexes them with FAISS and OpenAI embeddings, and uses a retrieval-augmented prompt with GPT-4o mini to produce grounded answers.

The current evaluation code does not implement **Citation Quality** scoring, **Hallucination Rate** measurement, **Structured RAG**, or **Self-RAG-inspired Reflection**; these methods are not reported here as completed system features.
