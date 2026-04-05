# RAG项目任务清单

> 基于 task4.pdf 与 CS6493 RAG Progress Report 整理

---

## 一、数据准备

| 数据集 | 领域 | 规模 | 用途 |
|--------|------|------|------|
| FinanceBench | 金融 | ~150 QA pairs | SEC 10-K/10-Q 文件问答，测试数值推理 |
| HotpotQA (dev) | 通用多跳 | ~7.4k pairs | 多跳问答，含干扰段落，评估检索精度 |
| CUAD | 法律 | 510 contracts, ~13k QA pairs | 法律合同条款抽取 |
| LegalBench-RAG-mini | 法律 | 776 QA pairs, 79M字符语料 | 法律领域检索专项评估 |

---

## 二、检索模块实现 (至少2种)

| 方法 | 类型 | 实现细节 |
|------|------|----------|
| **BM25** | 稀疏检索 | 使用 rank_bm25 / Pyserini 实现，TF-IDF关键词匹配，无需GPU |
| **Contriever** | 稠密检索 | facebook/contriever-msmarco，768维向量，FAISS flat索引，float16运行 |
| **Hybrid RRF** | 混合检索 | RRF (k=60) 融合 BM25 + Contriever，top-k=3文档/查询 |

---

## 三、生成模型部署 (2种)

| 模型 | 说明 |
|------|------|
| **LLaMA-3.1-8B-Instruct** | Meta指令微调模型 |
| **Qwen2.5-7B-Instruct** | 阿里指令微调模型 |

**额外对比实验：**
- Qwen2.5-7B (base, zero-shot) vs Qwen2.5-7B-Instruct

---

## 四、核心实验

### 实验1：检索-生成网格 (3×2×2)

| 维度 | 选项 |
|------|------|
| 检索方法 | BM25, Contriever, Hybrid RRF |
| 生成模型 | LLaMA-3.1-8B-Instruct, Qwen2.5-7B-Instruct |
| 领域 | Finance, Law |

**评估指标：** EM, F1, FActScore, Recall@3

---

### 实验2：HotpotQA 检索精度分析

| 内容 | 说明 |
|------|------|
| 索引范围 | 7.4k dev-set 支撑事实 + 干扰段落 |
| 检索数量 | k ∈ {1, 3, 5, 10} |
| 检索指标 | Recall@k, Precision@k |
| 生成指标 | F1 score (LLaMA-3.1-8B-Instruct生成) |
| 分析目标 | 检索质量与答案质量的关联性 |

---

### 实验3：Zero-shot vs Instruction-tuned 对比

| 配置 | 说明 |
|------|------|
| 检索器 | Contriever, k=3 |
| 数据集 | 全部4个 |
| 对比模型 | Qwen2.5-7B (base) vs Qwen2.5-7B-Instruct vs LLaMA-3.1-8B-Instruct |
| 指标 | EM, F1, FActScore, Hallucination Rate, Response Length, Refusal Rate |

---

### 实验4：幻觉消减策略对比

| 条件 | 组件 | 机制 |
|------|------|------|
| **LLM Only** | 无 | 纯参数化生成，无上下文，最高幻觉基线 |
| **Vanilla RAG** | Retriever + Generator | 标准检索-生成，top-k=3文档拼接到prompt |
| **LangChain Structured Chain** | RetrievalQA + PromptTemplate + LLaMA-3.1-8B | 强制引用 [Source N]，链式思考，512-token分块+10%重叠 |
| **Self-RAG** | Retriever + LLaMA-3.1-8B | 生成反思token判断检索必要性和答案可信度 |

**评估细节：**
- 每领域100个响应
- 生成器：LLaMA-3.1-8B-Instruct
- 检索：Hybrid RRF
- 主指标：FActScore
- 人工评估：50个样本的 Citation Precision/Recall

---

## 五、评估指标

| 指标 | 说明 |
|------|------|
| **EM / F1** | 答案准确率 |
| **Recall@k / Precision@k** | 检索精度 |
| **FActScore** | 细粒度原子事实评估 |
| **Citation Precision/Recall** | 引用准确性（人工评估） |
| **Hallucination Rate** | 包含事实错误陈述的比例 |
| **Fairness** | 公平性指标（需补充实现） |
| **Accuracy** | 整体准确率 |

---

## 六、跨领域性能分析

| 对比维度 | 说明 |
|----------|------|
| Finance vs Law | 金融领域 vs 法律领域性能差异 |
| 检索方法差异 | 不同检索方法在各领域的适用性 |
| 生成模型差异 | 不同生成模型在各领域的表现 |

---

## 七、系统架构

```
三阶段架构：
1. 离线语料索引
2. 在线查询时检索
3. Prompt增强生成

技术栈：Python + LangChain编排 + 模块化检索器/生成器
硬件：NVIDIA A100 40GB
```

---

## 八、task4.pdf 要求但报告未详细覆盖

| 要求 | 当前状态 | 建议 |
|------|----------|------|
| 公平性 (Fairness) 评估 | 未详细规划 | 需补充实现 |
| 偏见评估（性别、年龄等人口群体） | 未详细规划 | 需补充实现 |
| 多样化数据集减少偏见 | 未详细规划 | 需补充实现 |

---

## 九、时间规划

| 阶段 | 时间 | 任务 |
|------|------|------|
| Week 5 | 4月1-7日 | 连接生成模型、3×2×2实验、Zero-shot消融实验 |
| Week 6 | 4月8-12日 | 幻觉评估、跨领域对比、报告撰写 |
| 最终 | 4月13-14日 | 演示PPT、代码整理、端到端Demo |

---

## 十、小组成员

| 姓名 | Student EID |
|------|-------------|
| Huibo LI | huiboli2 |
| Lingke MENG | lkmeng4 |
| Shibo WANG | shibowang3 |
| Xinrui SHAN | xinruis2 |
| Yue ZHANG | yzhang8384 |

**Group Name:** RAGdoll
