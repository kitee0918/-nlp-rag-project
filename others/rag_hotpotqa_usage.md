# RAG 系统使用指南

基于 HotpotQA 数据集的原生 RAG 系统实现，支持 BM25 和 Contriever 两种检索方法，支持 Qwen API 和本地 Llama 模型生成。

---

## 目录

1. [快速开始](#快速开始)
2. [环境配置](#环境配置)
3. [核心组件](#核心组件)
4. [使用方式](#使用方式)
   - [方式一：命令行运行](#方式一命令行运行)
   - [方式二：Colab/Jupyter 函数调用](#方式二colabjupyter-函数调用)
   - [方式三：交互式会话（推荐）](#方式三交互式会话推荐)
5. [参数说明](#参数说明)
6. [缓存机制](#缓存机制)
7. [评估指标](#评估指标)
8. [常见问题](#常见问题)

---

## 快速开始

### Colab 环境一键运行

```python
# 1. 下载代码
!curl -H "Authorization: token $GITHUB_TOKEN" \
  -L "https://raw.githubusercontent.com/kitee0918/-nlp-rag-project/llama-bm25-contriever_REPL/rag_hotpotqa.py" \
  -o rag_hotpotqa.py

# 2. 导入并创建会话
from rag_hotpotqa import InteractiveRAGSession
session = InteractiveRAGSession()

# 3. 加载数据和模型
session.load_data(100)  # 加载100个样本
session.load_generator("llama", "/content/models/llama-3.1-8b")

# 4. 运行实验
results = session.run_experiment(["bm25", "contriever"], top_k=3)
```

---

## 环境配置

### 依赖安装

```bash
pip install torch transformers datasets rank-bm25 numpy tqdm openai httpx python-dotenv
```

### 可选：8bit/4bit 量化

```bash
pip install bitsandbytes accelerate
```

### 环境变量

```bash
# 阿里百炼 API Key（使用 Qwen 时需要）
export DASHSCOPE_API_KEY="sk-xxx"

# HuggingFace 镜像（国内加速）
export HF_ENDPOINT="https://hf-mirror.com"
```

---

## 核心组件

### 1. 检索器

| 检索器 | 类名 | 特点 | 适用场景 |
|--------|------|------|----------|
| **BM25** | `BM25Retriever` | 稀疏检索，基于词频 | 快速、无需 GPU、关键词匹配 |
| **Contriever** | `ContrieverRetriever` | 稠密检索，语义向量 | 语义理解、跨语言、需要 GPU |

### 2. 生成器

| 生成器 | 类名 | 特点 | 适用场景 |
|--------|------|------|----------|
| **Qwen** | `QwenGenerator` | API 调用，阿里百炼 | 无需本地 GPU，快速验证 |
| **Llama** | `LlamaGenerator` | 本地模型，8bit 量化 | 隐私保护，无网络限制 |

### 3. 评估指标

| 指标 | 说明 |
|------|------|
| **F1 Score** | 生成答案与标准答案的词级别 F1 |
| **Recall@K** | 检索到的支持文档比例 |
| **Precision@K** | 检索文档中正确文档的比例 |
| **F1@K** | Recall@K 和 Precision@K 的调和平均 |

---

## 使用方式

### 方式一：命令行运行

```bash
# BM25 + Qwen（默认）
python rag_hotpotqa.py --retriever bm25 --subset 100 --top_k 3

# Contriever + Llama
python rag_hotpotqa.py --retriever contriever --generator-type llama \
  --llama-model-path /content/models/llama-3.1-8b --subset 100

# 两个检索器对比
python rag_hotpotqa.py --retriever both --generator-type llama

# 交互式会话
python rag_hotpotqa.py --interactive
# 或
python rag_hotpotqa.py -i
```

### 方式二：Colab/Jupyter 函数调用

#### 快速测试（单检索器）

```python
from rag_hotpotqa import run_llama_bm25_quick

# 最小配置，快速验证
results = run_llama_bm25_quick(
    subset_size=10,  # 仅10个样本
    top_k=3,
    llama_model_path="/content/models/llama-3.1-8b"
)
```

#### 完整对比实验

```python
from rag_hotpotqa import run_comparison_experiment

# Qwen API
results = run_comparison_experiment(
    subset_size=100,
    top_k=3,
    generator_type="qwen",
    api_key="sk-xxx",
    use_retrievers=["bm25", "contriever"]
)

# 本地 Llama
results = run_comparison_experiment(
    subset_size=100,
    top_k=3,
    generator_type="llama",
    llama_model_path="/content/models/llama-3.1-8b",
    use_retrievers=["bm25", "contriever"]
)

# 仅测试单个检索器
results = run_comparison_experiment(
    subset_size=50,
    generator_type="llama",
    use_retrievers=["bm25"]  # 跳过 Contriever 模型下载
)
```

### 方式三：交互式会话（推荐）

**最佳实践：资源只加载一次，多次运行实验**

```python
from rag_hotpotqa import InteractiveRAGSession

session = InteractiveRAGSession()
```

#### 命令行交互模式

```python
session.start()
```

```
🎮 RAG 交互式实验环境
============================================================
输入 'help' 或 '?' 查看命令列表
============================================================

RAG> load 100           # 加载数据集
RAG> gen llama /content/models/llama-3.1-8b  # 加载生成器
RAG> ret all            # 加载所有检索器
RAG> run                # 运行实验
RAG> run bm25           # 仅运行 BM25
RAG> run -k 5           # 指定 top_k=5
RAG> status             # 查看状态
RAG> quit               # 退出
```

#### 编程调用模式

```python
# 1. 加载数据（缓存）
session.load_data(100)

# 2. 加载生成器（缓存）
session.load_generator("llama", model_path="/content/models/llama-3.1-8b")

# 3. 多次实验（复用缓存）
session.run_experiment(["bm25"], top_k=3)
session.run_experiment(["contriever"], top_k=3)
session.run_experiment(["bm25", "contriever"], top_k=5)  # 对比

# 4. 切换生成器
session.load_generator("qwen", model_name="qwen-plus", force_reload=True)
session.run_experiment(["bm25"])  # 用 Qwen 再跑一次

# 5. 查看状态
print(session.status())

# 6. 清除缓存
session.clear_cache()
```

---

## 参数说明

### 命令行参数

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--retriever` | `bm25` | 检索器类型：`bm25`, `contriever`, `both` |
| `--subset` | `100` | 测试样本数量 |
| `--top_k` | `3` | 检索文档数量 |
| `--generator-type` | `qwen` | 生成器类型：`qwen`, `llama` |
| `--model` | `qwen-plus` | Qwen 模型名称 |
| `--llama-model-path` | `/content/models/llama-3.1-8b` | Llama 模型路径 |
| `--device` | `cuda` | 运行设备 |
| `--interactive` / `-i` | - | 启动交互式会话 |

### 函数参数

#### `run_comparison_experiment()`

```python
run_comparison_experiment(
    subset_size: int = 100,      # 样本数量
    top_k: int = 3,              # 检索文档数
    model: str = "qwen-plus",    # Qwen 模型名
    api_key: str = None,         # API Key
    device: str = None,          # 设备（自动检测）
    use_retrievers: List = None, # ["bm25", "contriever"]
    generator_type: str = "qwen", # "qwen" 或 "llama"
    llama_model_path: str = None # Llama 路径
) -> Dict
```

#### `InteractiveRAGSession.run_experiment()`

```python
session.run_experiment(
    retrievers: List[str] = None,  # ["bm25", "contriever"]
    top_k: int = None,             # 默认 3
    subset_size: int = None        # 可动态调整
) -> Dict
```

---

## 缓存机制

### 文件缓存

| 缓存文件 | 内容 | 路径 |
|----------|------|------|
| BM25 索引 | 倒排索引 | `./cache/bm25_index.pkl` |
| Contriever 嵌入 | 文档向量矩阵 | `./cache/contriever_embeddings.pkl` |

### 内存缓存（InteractiveRAGSession）

```python
session = InteractiveRAGSession()

# 首次加载：耗时
session.load_data(100)           # 加载数据集
session.load_generator("llama")  # 加载模型（~3分钟）

# 后续调用：秒级
session.run_experiment(["bm25"])         # 复用缓存
session.run_experiment(["contriever"])   # 复用缓存
session.run_experiment(["bm25", "contriever"])  # 复用缓存
```

### Google Drive 持久化（Colab）

```python
from google.colab import drive
drive.mount('/content/drive')

# 修改缓存目录
import os
os.makedirs('/content/drive/MyDrive/rag_cache', exist_ok=True)

# 在代码中设置
from rag_hotpotqa import Config
Config.CACHE_DIR = '/content/drive/MyDrive/rag_cache'
```

---

## 评估指标

### 检索指标

```
Recall@K = 命中的支持文档数 / 总支持文档数
Precision@K = 命中的文档数 / 检索的文档数
F1@K = 2 * Recall * Precision / (Recall + Precision)
```

### 生成指标

```
F1 Score = 词级别 F1（预测答案 vs 标准答案）
```

### 结果示例

```
📈 最终对比结果
======================================================================
检索器          F1 Score    Recall@K    Precision@K  F1@K       
----------------------------------------------------------------------
BM25            0.3521      0.4521      0.3014       0.3612     
Contriever      0.4123      0.5892      0.3928       0.4715     
======================================================================

🏆 最佳 Recall@K: Contriever (0.5892)
🏆 最佳 F1@K:     Contriever (0.4715)
🏆 最佳 F1 Score: Contriever (0.4123)

📊 Contriever vs BM25 提升:
   Recall@K: +30.31%
   F1@K:     +30.54%
```

---

## 常见问题

### Q1: 模型加载报错 `torch_dtype is deprecated`

**原因**：transformers 版本更新，参数名称变更。

**解决**：代码已修复，使用 `dtype=torch.float16` 替代 `torch_dtype`。

### Q2: `AttributeError: model.device`

**原因**：`device_map="auto"` 时模型分布在多 GPU，`model.device` 属性不存在。

**解决**：代码已修复，使用 `next(self.model.parameters()).device`。

### Q3: Contriever 加载 TypeError

**原因**：`resume_download` 和 `force_download` 参数已废弃。

**解决**：代码已修复，移除废弃参数。

### Q4: `BatchEncoding` 类型错误

**原因**：`apply_chat_template` 返回 `BatchEncoding` 类型，不是普通 dict。

**解决**：代码已修复，使用 `hasattr(x, 'input_ids')` 检测。

### Q5: Colab 中 ImportError

**原因**：旧模块缓存未清除。

**解决**：
```python
import sys
if 'rag_hotpotqa' in sys.modules:
    del sys.modules['rag_hotpotqa']
from rag_hotpotqa import InteractiveRAGSession
```

### Q6: Llama 模型在哪里下载？

```python
from huggingface_hub import snapshot_download, login

# 登录 HuggingFace（需要访问权限）
login(token="your_hf_token")

# 下载模型
snapshot_download(
    repo_id="meta-llama/Meta-Llama-3.1-8B-Instruct",
    local_dir="/content/models/llama-3.1-8b"
)
```

### Q7: 显存不足怎么办？

**方案 1**：使用 4bit 量化
```python
generator = LlamaGenerator(
    model_path="...",
    load_in_4bit=True  # 更省显存
)
```

**方案 2**：减少 batch size（Contriever）
```python
# 在 ContrieverRetriever 中修改
# 默认 batch_size=32，可改为 16 或 8
```

---

## 完整示例：Colab 工作流

```python
# ============================================
# Cell 1: 安装依赖
# ============================================
!pip install torch transformers datasets rank-bm25 numpy tqdm openai httpx python-dotenv bitsandbytes accelerate

# ============================================
# Cell 2: 设置环境变量
# ============================================
import os
os.environ['GITHUB_TOKEN'] = 'your_github_token'
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'

# ============================================
# Cell 3: 下载代码
# ============================================
!curl -H "Authorization: token $GITHUB_TOKEN" \
  -L "https://raw.githubusercontent.com/kitee0918/-nlp-rag-project/llama-bm25-contriever_REPL/rag_hotpotqa.py" \
  -o rag_hotpotqa.py

# ============================================
# Cell 4: 下载 Llama 模型（首次运行）
# ============================================
from huggingface_hub import snapshot_download, login
login(token="your_hf_token")
snapshot_download(
    repo_id="meta-llama/Meta-Llama-3.1-8B-Instruct",
    local_dir="/content/models/llama-3.1-8b"
)

# ============================================
# Cell 5: 创建会话并加载资源
# ============================================
from rag_hotpotqa import InteractiveRAGSession

session = InteractiveRAGSession()
session.load_data(100)
session.load_generator("llama", model_path="/content/models/llama-3.1-8b")

# ============================================
# Cell 6: 运行实验
# ============================================
# 实验 1: BM25
results_bm25 = session.run_experiment(["bm25"], top_k=3)

# 实验 2: Contriever
results_contriever = session.run_experiment(["contriever"], top_k=3)

# 实验 3: 对比
results_both = session.run_experiment(["bm25", "contriever"], top_k=5)

# ============================================
# Cell 7: 查看状态
# ============================================
print(session.status())

# ============================================
# Cell 8: 清理
# ============================================
session.clear_cache()
```

---

## 版本历史

| 版本 | 分支 | 主要更新 |
|------|------|----------|
| v1.0 | `main` | 初始版本，BM25 + Qwen |
| v1.1 | `llama-bm25-contriever` | 添加 Llama 支持，修复 torch_dtype |
| v2.0 | `llama-bm25-contriever_REPL` | 交互式会话，资源缓存，BatchEncoding 修复 |
