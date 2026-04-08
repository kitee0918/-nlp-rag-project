# 任务解读：Zero-shot vs. Instruction-tuned Models 对比实验

## 核心要求

**对比「零样本模型」与「指令微调模型」在 RAG 系统中的表现差异**

---

## 两个关键概念

### 1. Zero-shot Model（零样本模型）

- **定义**：未经特定任务微调的原始预训练模型
- **特点**：直接使用预训练权重，没有额外的指令跟随训练
- **例子**：原始的 LLaMA、Mistral、Qwen 基座模型等

### 2. Instruction-tuned Model（指令微调模型）

- **定义**：在指令数据集上微调过的模型，专门学习"听从指令回答问题"
- **特点**：更善于理解用户意图、格式化输出、拒绝不当请求
- **例子**：
  - [Intel/neural-chat-7b-v3-3](https://huggingface.co/Intel/neural-chat-7b-v3-3)（任务中提到的）
  - Qwen-Chat、Llama-2-Chat、Mistral-Instruct 等

---

## 在当前项目中的具体含义

结合 `rag_hotpotqa.py` 代码，这个任务意味着：

```
对比实验：
┌─────────────────────────────────────────────────────────────┐
│  实验组 A: Zero-shot 模型（如 Qwen 基座）                    │
│  实验组 B: Instruction-tuned 模型（如 Qwen-Chat / neural-chat）│
│                                                             │
│  控制变量：相同的检索器（BM25/Contriever）                   │
│  观察指标：F1 Score、Recall@K、答案质量                      │
└─────────────────────────────────────────────────────────────┘
```

---

## 研究问题

这个任务旨在回答：

1. **指令微调是否提升 RAG 答案质量？**
2. **指令微调模型是否更善于利用检索到的上下文？**
3. **两种模型在生成风格上有何差异？**

---

## 可能的实现方式

| 方案 | Zero-shot 模型 | Instruction-tuned 模型 |
|------|---------------|----------------------|
| 方案1 | Qwen（基座） | Qwen-Chat |
| 方案2 | Mistral | Mistral-Instruct |
| 方案3 | Llama-2 | Llama-2-Chat |

> **注意**：当前项目使用的是 **Qwen-Plus（API）**，这实际上是一个指令微调模型。要做这个对比实验，需要引入一个 zero-shot 基座模型进行对照。

---

## 参考资料

- [Intel/neural-chat-7b-v3-3](https://huggingface.co/Intel/neural-chat-7b-v3-3) - 任务中提到的指令微调模型示例
