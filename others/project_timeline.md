# CS6493 RAG项目时间线

## Retrieval-Augmented Generation for Knowledge-Intensive Tasks

**课程**: CS6493 Natural Language Processing  
**学期**: Q2 2025-2026  
**学校**: City University of Hong Kong  
**小组**: RAGdoll  
**成员**: Huibo LI, Lingke MENG, Shibo WANG, Xinrui SHAN, Yue ZHANG  
**GPU**: NVIDIA A100 40GB

---

## 📅 项目时间线总览

```
Phase 1 ████████████████████████ 完成 (3/1 - 3/14)
Phase 2 ████████████████████████ 完成 (3/15 - 3/21)
Phase 3 ████████████████████████ 进行中 (3/22 - 3/28)
Phase 4 ░░░░░░░░░░░░░░░░░░░░░░░░ 计划中 (4/1 - 4/7)
Phase 5 ░░░░░░░░░░░░░░░░░░░░░░░░ 计划中 (4/8 - 4/12)
Phase 6 ░░░░░░░░░░░░░░░░░░░░░░░░ 计划中 (4/13 - 4/14)
```


```
Week5 Part 1:完成3x2x2检索-生成网格实验
Week 5 Part 2:HotpotQA检索精度分析实验
Week 5 Part 3: Zero-shot vs Instruction-tuned模型
Week6 Part1:幻觉缓解策略对比实验
Week6 Part 2:人工评估与数据集质量分析
Week6 Part 3:跨领域性能对比与报告撰写
```
---

## Phase 1: 项目启动与环境搭建

| 属性 | 详情 |
|------|------|
| **时间** | 3月1日 - 3月14日 |
| **状态** | ✅ 已完成 |
| **周次** | Week 1-2 |

### 主要任务
- [x] 确定研究主题：Finance + Law 双领域对比
- [x] 完成文献综述 (RAG, FiD, HotpotQA, CUAD, Self-RAG)
- [x] NVIDIA A100 40GB GPU 环境配置
- [x] 技术栈选型：LangChain, FAISS, Pyserini

### 交付成果
- 项目方向确定文档
- 文献综述报告
- 开发环境搭建完成

---

## Phase 2: 数据集收集与基础索引

| 属性 | 详情 |
|------|------|
| **时间** | 3月15日 - 3月21日 |
| **状态** | ✅ 已完成 |
| **周次** | Week 3 |

### 主要任务
- [x] 收集 FinanceBench 数据集 (~150 QA pairs)
- [x] 收集 HotpotQA 开发集 (~7.4k pairs)
- [x] 收集 CUAD 数据集 (510 contracts, ~13k QA pairs)
- [x] 收集 LegalBench-RAG-mini (776 QA pairs)
- [x] 完成 BM25 索引构建
- [x] 建立 Recall@3 基线

### 数据集概览

| 数据集 | 领域 | 规模 | 用途 |
|--------|------|------|------|
| FinanceBench | 金融 | ~150 pairs | SEC 10-K/10-Q 财务问答 |
| HotpotQA | 通用多跳 | ~7.4k pairs | 多跳推理 + 检索精度测试 |
| CUAD | 法律 | ~13k pairs | 合同条款提取 |
| LegalBench-RAG-mini | 法律 | 776 pairs | 法律检索基准 |

---

## Phase 3: 检索系统实现

| 属性 | 详情 |
|------|------|
| **时间** | 3月22日 - 3月28日 |
| **状态** | 🔄 进行中 |
| **周次** | Week 4 |

### 主要任务
- [x] Contriever 集成 + FAISS 向量索引
- [ ] Hybrid RRF (Reciprocal Rank Fusion) 实现
- [ ] LangChain RetrievalQA 框架搭建
- [ ] HotpotQA 检索精度分析 (BM25 vs Contriever vs Hybrid)

### 检索方法对比

| 方法 | 类型 | 特点 | 适用场景 |
|------|------|------|----------|
| BM25 | 稀疏检索 | 关键词匹配，无需GPU | 法律条款精确匹配 |
| Contriever | 稠密检索 | 语义相似度，768维向量 | 跨语言/同义表达 |
| Hybrid RRF | 混合检索 | RRF融合(k=60)，Top-k=3 | 综合最优 |

---

## Phase 4: 生成模型集成与网格实验

| 属性 | 详情 |
|------|------|
| **时间** | 4月1日 - 4月7日 |
| **状态** | 📋 计划中 |
| **周次** | Week 5 |

### 主要任务
- [ ] 连接 LLaMA-3.1-8B-Instruct
- [ ] 连接 Qwen2.5-7B-Instruct
- [ ] 执行 3×2×2 RAG 网格实验
- [ ] Zero-shot vs Instruction-tuned 消融实验
- [ ] 分析检索精度对答案质量的影响

### 3×2×2 实验设计

```
┌─────────────────────────────────────────────────────────┐
│                    实验变量矩阵                           │
├─────────────────────────────────────────────────────────┤
│  检索方法 (3): BM25, Contriever, Hybrid RRF             │
│  生成模型 (2): LLaMA-3.1-8B-Instruct, Qwen2.5-7B-Instruct│
│  领域数据 (2): Finance (FinanceBench), Law (CUAD)       │
├─────────────────────────────────────────────────────────┤
│  总实验数: 3 × 2 × 2 = 12 组                            │
│  评估指标: EM, F1, FActScore, Recall@k                 │
└─────────────────────────────────────────────────────────┘
```

### Zero-shot vs Instruction-tuned 对比

| 模型 | 类型 | 提示方式 |
|------|------|----------|
| Qwen2.5-7B (base) | 基础模型 | Zero-shot，无系统指令 |
| Qwen2.5-7B-Instruct | 指令微调 | 标准RAG系统提示 |
| LLaMA-3.1-8B-Instruct | 指令微调 | 标准RAG系统提示 |

---

## Phase 5: 幻觉缓解与评估

| 属性 | 详情 |
|------|------|
| **时间** | 4月8日 - 4月12日 |
| **状态** | 📋 计划中 |
| **周次** | Week 6 |

### 主要任务
- [ ] FActScore 幻觉率评估
- [ ] LangChain Structured Chain 实现
- [ ] Self-RAG 集成对比
- [ ] 跨领域性能分析
- [ ] 人工评估 (50-response stratified subsample)

### 幻觉缓解策略对比

| 方法 | 组件 | 机制 |
|------|------|------|
| LLM Only | 无 | 纯参数化，最高幻觉基线 |
| Vanilla RAG | Retriever + Generator | 标准检索-生成，无约束 |
| LangChain Structured | RetrievalQA + PromptTemplate | 强制引用 [Source N]，CoT推理 |
| Self-RAG | 检索器 + 反思token | 自我评判检索必要性和答案可信度 |

### 评估指标

| 指标 | 说明 |
|------|------|
| EM (Exact Match) | 精确匹配率 |
| F1 Score | Token级别F1 |
| Recall@k | 检索召回率 |
| Precision@k | 检索精确率 |
| FActScore | 事实精度评分 |
| Citation Precision/Recall | 引用质量 (人工评估) |
| Hallucination Rate | 幻觉率 |

---

## Phase 6: 最终交付

| 属性 | 详情 |
|------|------|
| **时间** | 4月13日 - 4月14日 |
| **状态** | 📋 计划中 |
| **周次** | 最终周 |

### 主要任务
- [ ] 最终报告撰写与分析
- [ ] 演示幻灯片制作
- [ ] 代码清理与文档化
- [ ] 端到端 Demo 展示

---

## 📋 任务4要求对应表

### 任务1: Hybrid Generation Interaction

| 要求 | 对应实验 | 执行阶段 | 状态 |
|------|----------|----------|------|
| 实现2种检索方法 | BM25 + Contriever + Hybrid | Phase 3-4 | ✅ |
| 实现2种生成模型 | LLaMA-3.1-8B + Qwen2.5-7B | Phase 4 | 📋 |
| HotpotQA检索精度分析 | Recall@k vs Answer F1 | Phase 3-4 | 🔄 |
| Zero-shot vs Instruction-tuned | Qwen base vs Instruct | Phase 4 | 📋 |
| 人工评估数据集质量 | Crowd-sourcing | Phase 5 | 📋 |

### 任务2: Multifaceted Mitigation

| 要求 | 对应实验 | 执行阶段 | 状态 |
|------|----------|----------|------|
| 量化幻觉率 | FActScore | Phase 5 | 📋 |
| 跨领域引用质量 | Finance vs Law对比 | Phase 5 | 📋 |
| 人工评估 | 50样本分层抽样 | Phase 5 | 📋 |
| 4种缓解策略对比 | LLM-only/Vanilla/LangChain/Self-RAG | Phase 5 | 📋 |

---

## 💡 实验建议

1. **检索文档数量**: 建议限制在 **5-9篇/查询**，平衡性能与计算成本
2. **当前设置**: k=3，可根据 HotpotQA 实验结果调整
3. **评估框架**: 详见附录 (Appendix)
4. **跨领域泛化**: 关注 Finance 和 Law 领域的性能差异

---

## 📚 参考文献

1. Lewis P. et al. *Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks*. NeurIPS 2020.
2. Izacard G, Grave E. *Leveraging Passage Retrieval with Generative Models for Open Domain QA*. EACL 2021.
3. Yang Z. et al. *HotpotQA: A Dataset for Diverse, Explainable Multi-hop Question Answering*. EMNLP 2018.
4. Asai A. et al. *Self-RAG: Learning to Retrieve, Generate, and Critique through Self-Reflection*. ICLR 2024.
5. Min S. et al. *FActScore: Fine-Grained Atomic Evaluation of Factual Precision in Long Form Text Generation*. EMNLP 2023.
6. Islam K. et al. *FinanceBench: A New Benchmark for Financial Question Answering*. NeurIPS Workshop 2023.
7. Hendrycks D. et al. *CUAD: An Expert-Annotated NLP Dataset for Legal Contract Review*. NeurIPS 2021.
8. Pipitone N, Houir Alami G. *LegalBench-RAG: A Benchmark for Retrieval-Augmented Generation in the Legal Domain*. arXiv 2024.

---

*文档生成日期: 2026年3月31日*
