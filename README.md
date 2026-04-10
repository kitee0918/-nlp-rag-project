# NLP RAG Project (Refactored)

本项目已按精简结构重整为：

- `dataset/`
- `.gitignore`
- `README.md`
- `eval.py`
- `llm.py`
- `main.py`
- `methods.py`
- `prompt.py`
- `retriever.py`

## 功能

- FinanceBench RAG 实验（BM25 / Contriever）
- LLM Only / Vanilla RAG / Self-RAG 幻觉评测对比
- FActScore、Hit@K、F1、Gen EM、Ret EM、citation / hallucination 等指标

## 环境

- Python 3.10+
- 如使用 Qwen API，请配置环境变量：`DASHSCOPE_API_KEY`

## 运行

默认主入口：

```bash
python main.py --subset 20 --top_k 3 --retriever bm25 --model qwen-plus
```

参数示例：

```bash
python main.py --subset 10 --top_k 3 --retriever contriever --verbose
```

## 说明

- `methods.py`：主流程与实验方法
- `retriever.py`：检索类导出
- `llm.py`：生成器类导出
- `eval.py`：幻觉与事实性评估
- `prompt.py`：集中管理提示词模板
