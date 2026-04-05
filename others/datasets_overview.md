# 数据集介绍

---

## FinanceBench 数据集

### 概述

**FinanceBench** 是由 Patronus AI 发布的首个专门用于评估大语言模型（LLM）在"开卷"财务问答任务上表现的基准测试数据集。该数据集发表于 2023 年，论文发布在 arXiv（编号：2311.11944）。

---

### 核心特点

| 特性 | 说明 |
|------|------|
| **目的** | 评估 LLM 在金融领域的问答能力，检测幻觉问题 |
| **规模** | 完整基准包含 **10,231** 个问题；开源版本包含 **150** 个带注释示例 |
| **数据来源** | 公开上市公司的财务报告（10-K、10-Q、8-K 等） |
| **许可证** | CC-BY-NC-4.0（非商业使用） |

---

### 数据集结构

每个问题包含以下主要字段：

| 字段 | 说明 |
|------|------|
| `question` | 具体的财务问题 |
| `answer` | 标准答案 |
| `justification` | 答案的理由/计算过程 |
| `evidence` | 证据片段（原文摘录、页码等） |
| `company` | 公司名称（如 Amazon、Adobe） |
| `doc_type` | 文档类型（10k/10q/8k） |
| `question_reasoning` | 推理类型（信息抽取、数值推理、逻辑推理） |
| `gics_sector` | 行业分类 |

---

### 主要贡献

1. **生态效度高**：问题基于真实的财务报告场景，具有实际应用价值
2. **暴露模型局限**：测试发现即使是 GPT-4 等先进模型也存在幻觉问题
3. **支持 RAG 评估**：提供证据链，可用于评估检索增强生成系统
4. **多样化推理**：涵盖信息抽取、数值计算、逻辑推理等多种推理类型

---

### 使用方法

```python
from datasets import load_dataset

# 加载数据集
dataset = load_dataset("PatronusAI/financebench")

# 访问数据
question = dataset["train"][0]["question"]
answer = dataset["train"][0]["answer"]
```

---

### 应用场景

- **LLM 评估**：测试模型在金融领域的准确性和可靠性
- **RAG 系统开发**：验证模型能否基于财报片段正确回答问题
- **金融 NLP 研究**：分析模型在不同推理类型上的表现差异

---

### 相关资源

- 论文：https://arxiv.org/abs/2311.11944
- HuggingFace：https://huggingface.co/datasets/PatronusAI/financebench
- GitHub：https://github.com/patronus-ai/financebench

---

## HotpotQA (dev) 数据集

### 概述

**HotpotQA**（Hotpot Question Answering）是由卡内基梅隆大学、蒙特利尔大学和斯坦福大学联合发布的大规模多跳问答数据集，发表于 EMNLP 2018。该数据集旨在推动**多跳推理问答**的研究。

---

### 核心特点

| 特性 | 说明 |
|------|------|
| **任务类型** | 多跳问答 |
| **数据规模** | 约 **113,000** 个问答对（完整数据集） |
| **数据来源** | Wikipedia 文档 |
| **核心要求** | 问题需要跨多个文档进行推理才能回答 |
| **独特之处** | 需要标注**支撑事实**，具有可解释性 |

---

### Dev 集特点

HotpotQA 的开发集有两种评估设置：

| 设置 | 说明 |
|------|------|
| **Distractor 设置** | 每个问题提供 **10 个段落**（包含正确答案和干扰项），模型需从中找出答案和支撑事实 |
| **Full-wiki 设置** | 在整个 Wikipedia 中检索，更接近真实开放域问答场景 |

**Dev 集规模**（BEIR 版本）：
- 查询数量：**5,447** 个问题
- 相关性标注：**10,894** 条

---

### 数据结构

每条数据包含以下主要字段：

| 字段 | 类型 | 说明 |
|------|------|------|
| `id` | string | 问题唯一标识符 |
| `question` | string | 问题文本 |
| `answer` | string | 答案（短答案） |
| `context` | list | 相关文档段落列表 |
| `supporting_facts` | list | 支撑事实（包含文档标题和句子索引） |
| `type` | string | 问题类型（bridge / comparator） |
| `level` | string | 难度级别（easy / medium / hard） |

---

### 多跳推理类型

HotpotQA 定义了两种主要的多跳推理模式：

1. **Bridge（桥接型）**
   - 需要先找到"桥接实体"才能回答问题
   - 例：*《乱世佳人》的原著作者是哪个国家的？*
     - 第一步：找到《乱世佳人》的原著作者是玛格丽特·米切尔
     - 第二步：找到玛格丽特·米切尔的国籍是美国

2. **Comparator（比较型）**
   - 需要比较两个实体的属性
   - 例：*迈克尔·杰克逊和麦当娜谁更早出生？*
     - 需要查找两人的出生日期并进行比较

---

### 评估指标

HotpotQA 使用多个指标进行评估：

| 指标 | 说明 |
|------|------|
| **EM (Exact Match)** | 答案完全匹配准确率 |
| **F1** | 答案的词级别 F1 分数 |
| **Sup EM** | 支撑事实完全匹配准确率 |
| **Sup F1** | 支撑事实的 F1 分数 |
| **Joint EM/F1** | 答案和支撑事实的联合评估 |

---

### 使用方法

```python
from datasets import load_dataset

# 加载 HotpotQA 数据集
dataset = load_dataset("hotpot_qa", "distractor")

# 访问 dev 集
dev_data = dataset['validation']

# 查看数据示例
example = dev_data[0]
print(f"问题: {example['question']}")
print(f"答案: {example['answer']}")
print(f"支撑事实: {example['supporting_facts']}")
print(f"推理类型: {example['type']}")
```

---

### 应用场景

- **多跳问答系统**：测试模型跨文档推理能力
- **检索增强生成（RAG）**：评估文档检索和答案生成
- **可解释性研究**：通过支撑事实标注分析模型推理过程
- **信息检索基准**：作为 BEIR 基准的一部分，评估零样本检索能力

---

### 相关资源

- 官网：https://hotpotqa.github.io/
- 论文：https://arxiv.org/abs/1809.09600
- HuggingFace：https://huggingface.co/datasets/hotpot_qa
- BEIR 版本：https://huggingface.co/datasets/irds/beir_hotpotqa_dev

---

## CUAD 数据集

### 概述

**CUAD**（Contract Understanding Atticus Dataset，合同理解 Atticus 数据集）是由 The Atticus Project 发布的专业法律合同审查数据集，发表于 NeurIPS 2021。这是首个由法律专家大规模标注的合同理解基准数据集。

---

### 核心特点

| 特性 | 说明 |
|------|------|
| **任务类型** | 合同条款抽取与问答 |
| **数据规模** | **510** 份商业法律合同，**13,000+** 人工标注 |
| **标注类别** | **41** 类重要法律条款 |
| **标注者** | 数十位法律专家 |
| **许可证** | CC-BY-4.0 |

---

### 数据集背景

在商业交易（如并购、投资、IPO）中，律师需要仔细审查合同中的关键条款。CUAD 旨在解决法律领域 AI 应用的数据瓶颈，帮助自动化识别律师审查合同时最关注的条款。

---

### 标注类型（41 类条款）

CUAD 定义了 41 种重要的合同条款类型，答案格式分为三类：

#### 1. 存在性判断（33 类）
- 如果找到对应条款，答案为 **"Yes"**
- 如果未找到，答案为 **"No"**
- 例：是否存在"排他性条款"、是否存在"竞业禁止条款"

#### 2. 实体/数值提取（8 类）
- 提取具体信息：日期、数字、实体名、地点等
- 例：合同签署日期、到期日、续约期限

#### 3. 依赖推导
- 某些答案需结合其他字段计算
- 例：若条款写"自签署日起生效"，则生效日期 = 签署日期

#### 主要条款类别示例

| 类别 | 描述 | 答案格式 |
|------|------|----------|
| Document Name | 合同名称 | 文本 |
| Parties | 签约方 | 实体名称 |
| Agreement Date | 签署日期 | 日期 |
| Effective Date | 生效日期 | 日期 |
| Expiration Date | 到期日期 | 日期 |
| Governing Law | 适用法律 | 州/国家名 |
| License Grant | 许可授予 | Yes/No |
| Non-Compete | 竞业禁止 | Yes/No |
| Confidentiality | 保密条款 | Yes/No |

---

### 数据集结构

| 文件类型 | 描述 |
|----------|------|
| **Master Clauses CSV** | 核心文件，510 份合同 × 83 列 |
| **SQuAD-style JSON** | 转换为 SQuAD 2.0 问答格式 |
| **完整合同 PDF/TXT** | 原始合同文件（510 份） |
| **Excel 报告** | 按类别分组的条款报告 |

---

### 评估指标

CUAD 使用类似 SQuAD 的评估方式：

| 指标 | 说明 |
|------|------|
| **Exact Match (EM)** | 预测答案与标准答案完全匹配的比例 |
| **F1 Score** | 词级别的精确率和召回率的调和平均 |
| **Precision@K** | 在合同检索场景中的准确率 |

论文显示，即使是最先进的 Transformer 模型在 CUAD 上的表现仍远低于人类专家，说明法律合同理解具有较高难度。

---

### 使用方法

```python
from datasets import load_dataset

# 加载 CUAD 数据集
dataset = load_dataset("theatticusproject/cuad")

# 查看数据
print(dataset["train"][0])

# SQuAD 格式版本（用于问答任务）
dataset_qa = load_dataset("theatticusproject/cuad-qa")
```

---

### 应用场景

1. **合同审查自动化**：自动识别关键条款，辅助律师工作
2. **法律问答系统**：基于合同文本回答特定问题
3. **条款分类**：判断合同是否包含特定类型条款
4. **信息抽取**：从非结构化合同文本中提取结构化信息

---

### 模型表现

论文测试了多种模型：
- **BERT-base** 在 CUAD 上的 F1 约为 60-70%
- **Human Expert** 的 F1 超过 90%
- 这表明法律领域仍有很大提升空间

---

### 相关资源

- 论文：https://arxiv.org/abs/2103.06268
- HuggingFace：https://huggingface.co/datasets/theatticusproject/cuad
- GitHub：https://github.com/TheAtticusProject/cuad
- 官网：https://www.atticusprojectai.org/cuad

---

## LegalBench-RAG-mini 数据集

### 概述

**LegalBench-RAG-mini** 是 LegalBench-RAG 的轻量级版本，由 ZeroEntropy AI 团队发布于 2024 年 8 月。它是一个专门用于评估法律领域检索增强生成（RAG）系统中**检索组件**性能的基准数据集。

---

### 核心特点

| 特性 | 说明 |
|------|------|
| **任务类型** | 信息检索 |
| **查询数量** | **776** 个查询-答案对 |
| **文档数量** | **72** 份法律文档 |
| **文档规模** | 约 **868 万**字符 |
| **标注方式** | 完全人工标注 |
| **用途** | 快速迭代和实验验证 |

---

### 数据来源

LegalBench-RAG-mini 由以下四个数据集的子集构成，每个数据集选取 **194** 个查询：

| 来源数据集 | 描述 |
|------------|------|
| **PrivacyQA** | 隐私政策相关问答 |
| **CUAD** | 合同理解数据集 |
| **MAUD** | 并购协议数据集 |
| **ContractNLI** | 合同自然语言推理数据集 |

---

### 与完整版的对比

| 特性 | LegalBench-RAG (完整版) | LegalBench-RAG-mini |
|------|-------------------------|---------------------|
| 查询数量 | 6,858 | 776 |
| 文档规模 | 7,900万+ 字符 | 868万 字符 |
| 用途 | 全面评估 | 快速迭代实验 |
| 来源数据集 | 4 个 | 4 个（各取 194 条） |

---

### 数据结构

每个测试用例包含：

```
{
  "query": "具体的问题文本",
  "ground_truth": [
    {
      "file_path": "文档路径",
      "snippet": "相关文本片段",
      "char_range": [起始位置, 结束位置]
    }
  ]
}
```

**核心特点**：
- 标注了**精确的字符范围**，而非整段或整篇文档
- 强调检索的**精确性**，要求模型找到最小化且高度相关的文本片段

---

### 评估指标

LegalBench-RAG-mini 使用以下指标评估检索性能：

| 指标 | 说明 |
|------|------|
| **Recall@K** | 前 K 个检索结果中包含正确答案的比例 |
| **Precision@K** | 前 K 个检索结果的精确率 |
| **MRR** | 平均倒数排名 |

---

### 使用方法

```python
# 从 GitHub 克隆仓库
git clone https://github.com/zeroentropy-ai/legalbenchrag.git
cd legalbenchrag

# 安装依赖
pip install -r requirements.txt

# 运行基准测试
python ./legalbenchrag/benchmark.py
```

数据集结构：
```
legalbenchrag/
├── corpus/          # 法律文档语料库
│   └── *.txt
└── benchmarks/      # 基准测试数据
    └── *.json       # 包含 query 和 ground truth
```

---

### 设计动机

1. **填补空白**：首个专注于法律 RAG 检索组件评估的基准
2. **精确检索**：要求找到最小化文本片段，而非粗粒度文档
3. **快速迭代**：mini 版本便于开发者在有限资源下快速验证方法

---

### 应用场景

- **RAG 系统评估**：评估法律领域 RAG 的检索质量
- **分块策略研究**：测试不同文本分块方法的效果
- **重排序模型**：评估检索结果重排序算法
- **法律 AI 开发**：快速验证法律文档检索方案

---

### 相关资源

- 论文：https://arxiv.org/abs/2408.10343
- GitHub：https://github.com/zeroentropy-ai/legalbenchrag
- 开发团队：ZeroEntropy AI
