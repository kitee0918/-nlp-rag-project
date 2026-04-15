# NLP RAG 金融问答项目总结文档

> 文档生成时间：2026年4月15日
> 项目路径：`d:/myEnv/CodeBuddy_AgentWorkSpace/nlp/proj_finance/-nlp-rag-project`

---

## 一、项目概述

本项目是一个基于 **FinanceBench** 数据集的 RAG（检索增强生成）系统，专注于金融领域的问答任务。项目支持多种检索方法和生成模型，实现了检索-生成交互的对比实验。

### 核心功能

- 金融问答（FinanceBench 数据集，150 条金融问答）
- 多检索器支持（BM25 稀疏检索、Contriever 稠密检索）
- 多生成器支持（阿里百炼 API、本地 Llama 模型）
- 检索-生成对比实验

---

## 二、模块架构

```
┌─────────────────────────────────────────────────────────────────┐
│                         main.py (主执行入口)                      │
│              检索-生成交互实验：LLM Only / Vanilla RAG            │
└───────────────────────────┬─────────────────────────────────────┘
                            │
         ┌──────────────────┼──────────────────┐
         │                  │                  │
         ▼                  ▼                  ▼
┌─────────────┐    ┌─────────────────┐    ┌─────────────┐
│   eval.py   │    │   methods.py    │    │  prompt.py  │
│  评估模块    │◄──│  核心RAG实现     │───►│ 提示词模板   │
└─────────────┘    └────────┬────────┘    └─────────────┘
         │                   │
         │         ┌────────┴────────┐
         │         │                 │
         ▼         ▼                 ▼
┌─────────────┐  ┌──────────┐  ┌───────────┐
│  llm.py     │  │retriever.py│ │ (配置)     │
│ LLM导出模块  │  │ 检索导出模块│ │ Config类  │
└─────────────┘  └───────────┘  └───────────┘
```

---

## 三、模块详细说明

### 3.1 methods.py - 核心 RAG 系统实现

**职责**：提供 RAG 系统的完整实现，是所有其他模块的基础依赖。

| 组件 | 类/函数 | 功能描述 |
|------|---------|----------|
| 配置 | `Config` | 全局配置类（数据集、模型路径、API Key等） |
| 数据加载 | `load_financebench()` | 加载 FinanceBench 数据集 |
| 稀疏检索 | `BM25Retriever` | BM25 检索器，支持缓存 |
| 稠密检索 | `ContrieverRetriever` | Contriever 向量检索器 |
| API生成 | `QwenGenerator` | 阿里百炼 API 调用 Qwen 模型 |
| 本地生成 | `LLMGenerator` | 本地 Llama 模型（支持 4bit/8bit 量化） |
| 工厂函数 | `create_generator()` | 统一生成器创建接口 |
| RAG系统 | `RAGSystem` | 整合检索器 + 生成器 |
| 评估指标 | `compute_f1()`, `compute_retrieval_metrics()` | F1 分数、Hit@K 等 |

**统一接口设计**：
```python
# 检索器接口
class Retriever:
    def retrieve(query: str, top_k: int) -> List[Tuple[Dict, float]]
    def get_name() -> str

# 生成器接口
class Generator:
    def generate(question: str, documents: List[Dict]) -> Dict
    def get_name() -> str
```

### 3.2 llm.py - LLM 生成模块

**职责**：轻量级导出模块，提供清晰的模块边界。

```python
# 导出内容
- QwenGenerator      # API 生成器
- LLMGenerator       # 本地生成器
- create_generator() # 工厂函数
```

### 3.3 retriever.py - 检索模块

**职责**：轻量级导出模块。

```python
# 导出内容
- BM25Retriever       # 稀疏检索
- ContrieverRetriever # 稠密检索
```

### 3.4 prompt.py - 提示词模板

**职责**：集中管理所有 Prompt 模板。

| 模板 | 用途 |
|------|------|
| `LEGAL_QA_PROMPT` | 法律文档抽取式 QA |
| `FINANCEBENCH_QA_PROMPT` | 金融问答（严格 JSON 输出） |

### 3.5 eval.py - 评估模块

**职责**：提供评估能力，包括检索评估、生成评估、综合评估指标。

| 组件 | 功能 |
|------|------|
| `normalise()`, `split_claims()` | 文本标准化、声明分割 |
| `retrieval_em()`, `recall_at_k()` | 检索精确匹配、召回率 |
| `generation_em()`, `generation_f1()` | 生成精确匹配、F1 |
| `evaluate_all_metrics()` | 一次调用计算所有指标 |

### 3.6 main.py - 主执行脚本

**职责**：检索-生成交互实验的入口，协调多种方法的评估。

| 函数 | 功能 |
|------|------|
| `evaluate_llm_only()` | 评估纯 LLM（无检索） |
| `evaluate_vanilla_rag()` | 评估标准 RAG |
| `run_comparison()` | **核心入口**：对比多种方法 |
| `run_single_evaluation()` | 单一评估（支持惰性加载） |

**输出指标矩阵**：

| 方法 | Hit@K | Gen EM | Ret EM | F1 |
|------|-------|--------|--------|-----|
| LLM Only | 0.0 | - | 0.0 | - |
| VanillaRAG | - | - | - | - |

---

## 四、当前实现状态

### 4.1 已实现功能 ✅

| 功能 | 实现位置 | 状态 |
|------|----------|------|
| BM25 稀疏检索 | `methods.py` → `BM25Retriever` | ✅ 完成 |
| Contriever 稠密检索 | `methods.py` → `ContrieverRetriever` | ✅ 完成 |
| 阿里百炼 API 生成 | `methods.py` → `QwenGenerator` | ✅ 完成 |
| 本地 Llama 生成 | `methods.py` → `LLMGenerator` | ✅ 完成 |
| 多方法对比实验 | `main.py` → `run_comparison()` | ✅ 完成 |
| 检索缓存机制 | `methods.py` 各检索器 | ✅ 完成 |
| 4bit/8bit 量化 | `methods.py` → `LLMGenerator` | ✅ 完成 |
| 惰性加载 | 各组件 `lazy_load` 参数 | ✅ 完成 |

### 4.2 待实现功能 ❌

| 功能 | 描述 | 优先级 |
|------|------|--------|
| **Hybrid RRF 混合检索** | Reciprocal Rank Fusion 融合 BM25 + Contriever | 高 |
| RRF 算法实现 | 公式：`score(d) = Σ 1/(k + rank(d))`，k=60 | 高 |
| 混合检索器类 | `HybridRRFRetriever` 整合两种检索器 | 高 |
| 混合检索评估流程 | 对比混合 vs 单一检索器性能 | 中 |

---

## 五、Hybrid RRF 功能分析

### 5.1 功能描述

> Hybrid RRF (BM25 + Contriever): Reciprocal Rank Fusion (RRF, k=60) combining BM25 and Contriever score lists. Top-k=3 documents per query. Expected to outperform either retriever in isolation by covering both lexical and semantic dimensions.

### 5.2 当前实现 vs 期望功能

```
当前实现（独立对比模式）：

   Query ──► BM25 Retriever ──► Top-3 Docs ──► 评估
      │
      └──► Contriever ──► Top-3 Docs ──► 评估

   结果：BM25 vs Contriever 对比（各自独立运行）


期望实现（混合融合模式）：

   Query ──► BM25 Retriever ──► Ranked List A
      │               │
      │               ▼
      └──► Contriever ──► Ranked List B
                      │
                      ▼
              ┌─────────────────┐
              │ RRF Fusion (k=60)│
              └────────┬────────┘
                       ▼
              Fused Top-3 Docs ──► 评估

   结果：混合检索 vs 单一检索器对比
```

### 5.3 差异检查表

| 检查项 | 期望功能 | 当前代码 | 状态 |
|--------|----------|----------|------|
| RRF 算法 | Reciprocal Rank Fusion | 未实现 | ❌ |
| k=60 参数 | RRF 公式参数 | 无此参数 | ❌ |
| 融合逻辑 | BM25 + Contriever 结果合并 | 无融合 | ❌ |
| Top-k=3 | 最终返回文档数 | `RETRIEVAL_K = 3` | ✅ |
| 混合检索验证 | 混合 vs 单一检索器对比 | 仅独立对比 | ❌ |

### 5.4 RRF 算法参考实现

```python
def reciprocal_rank_fusion(
    rankings_list: List[List[Tuple[Dict, float]]], 
    k: int = 60
) -> List[Tuple[Dict, float]]:
    """
    Reciprocal Rank Fusion 算法
    
    公式: RRF_score(d) = Σ_{r in R} 1/(k + rank_r(d))
    
    Args:
        rankings_list: 多个检索器的排序列表
        k: 平滑参数，默认 60
    
    Returns:
        融合后的排序列表
    """
    doc_scores = {}
    doc_info = {}
    
    for ranking in rankings_list:
        for rank, (doc, score) in enumerate(ranking, start=1):
            doc_id = doc['id']
            rrf_score = 1.0 / (k + rank)
            doc_scores[doc_id] = doc_scores.get(doc_id, 0) + rrf_score
            if doc_id not in doc_info:
                doc_info[doc_id] = doc
    
    sorted_docs = sorted(doc_scores.items(), key=lambda x: x[1], reverse=True)
    return [(doc_info[doc_id], score) for doc_id, score in sorted_docs]
```

---

## 六、设计模式总结

| 模式 | 应用场景 |
|------|----------|
| **工厂模式** | `create_generator()` 统一创建生成器 |
| **策略模式** | 不同检索器/生成器实现统一接口，可互换 |
| **门面模式** | `RAGSystem` 封装复杂流程 |
| **懒加载模式** | `LLMGenerator` 支持 `lazy_load=True` |
| **模板方法模式** | `evaluate_vanilla_rag()` 遵循统一评估流程 |

---

## 七、模块依赖关系

```
main.py
  ├── 导入 methods.py: load_financebench, BM25Retriever, ContrieverRetriever, 
  │                   QwenGenerator, LLMGenerator, RAGSystem, compute_f1...
  │
  └── 导入 eval.py: generation_em, retrieval_em, evaluate_all_metrics...

eval.py
  ├── 导入 transformers: AutoTokenizer, AutoModelForCausalLM
  ├── 导入 openai: OpenAI (用于 API 调用)
  └── 独立实现评估逻辑

llm.py       → 导出 methods.py 中的生成器类
retriever.py → 导出 methods.py 中的检索器类
prompt.py    → 独立模块，被其他模块按需导入
```

---

## 八、后续工作建议

### 8.1 高优先级

1. **实现 HybridRRFRetriever 类**
   - 位置：`methods.py` 或新建 `hybrid_retriever.py`
   - 需实现 RRF 融合算法
   - 支持 k 参数配置（默认 60）

2. **添加混合检索评估流程**
   - 修改 `main.py` 的 `run_comparison()`
   - 新增混合检索 vs 单一检索器的对比

### 8.2 中优先级

3. **参数配置优化**
   - 将 RRF 的 k 参数加入 `Config` 类
   - 支持命令行参数 `--hybrid-k`

---

## 九、运行命令参考

```bash
# 使用 BM25 检索器运行实验
python main.py --retriever bm25 --subset 20 --top_k 3

# 使用 Contriever 检索器运行实验
python main.py --retriever contriever --subset 20 --top_k 3

# 对比两种检索器
python main.py --retriever both --subset 20 --top_k 3

# 使用 API 生成器
python main.py --generator-type api --model qwen-plus

# 使用本地模型（Colab 环境）
python main.py --generator-type local --model-path /content/models/llama-3.1-8b
```

---

## 十、关键配置

| 配置项 | 默认值 | 说明 |
|--------|--------|------|
| `DATASET_NAME` | `PatronusAI/financebench` | 数据集名称 |
| `SUBSET_SIZE` | 50 | 测试样本数量 |
| `RETRIEVAL_K` | 3 | 检索文档数量 |
| `QWEN_MODEL` | `qwen-plus` | API 模型名称 |
| `DASHSCOPE_BASE_URL` | `https://dashscope.aliyuncs.com/compatible-mode/v1` | API 地址 |
| `CACHE_DIR` | `./cache` | 缓存目录 |
| `RESULTS_DIR` | `./results` | 结果目录 |

---

**文档结束**

*如有问题或建议，请联系项目维护人员。*
