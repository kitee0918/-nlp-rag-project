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
- LLM Only / Vanilla RAG 检索-生成交互对比
- Hit@K、F1、Gen EM、Ret EM 等评估指标

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

## Colab 拉取最新代码（推荐）

```bash
git clone -b merged_finance https://github.com/kitee0918/-nlp-rag-project.git /content/-nlp-rag-project
cd /content/-nlp-rag-project
```

如目录已存在，使用：

```bash
git -C /content/-nlp-rag-project fetch origin
git -C /content/-nlp-rag-project checkout merged_finance
git -C /content/-nlp-rag-project pull origin merged_finance
```

## 说明

- `methods.py`：主流程与实验方法（含 `release_contriever_gpu`）
- `retriever.py`：检索类导出
- `llm.py`：生成器类导出
- `eval.py`：评估模块（含 `evaluate_all_metrics`）
- `prompt.py`：集中管理提示词模板

