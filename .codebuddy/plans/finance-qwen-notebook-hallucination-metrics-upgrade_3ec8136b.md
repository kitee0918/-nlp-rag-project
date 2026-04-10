---
name: finance-qwen-notebook-hallucination-metrics-upgrade
overview: 检查并补齐 finance_qwen.ipynb 中缺失的幻觉评测指标，使其与 run_hallucination_experiment.py 和 hallucination_eval.py 的评估口径一致。
todos:
  - id: audit-metric-gap
    content: 使用 [subagent:code-explorer] 复核脚本与notebook指标差异
    status: completed
  - id: wire-tri-comparison-cell
    content: 在finance_qwen.ipynb接入run_comparison三路评估单元
    status: completed
    dependencies:
      - audit-metric-gap
  - id: add-all-metrics-cell
    content: 新增evaluate_all_metrics样本级综合指标展示单元
    status: completed
    dependencies:
      - wire-tri-comparison-cell
  - id: unify-metric-summary
    content: 汇总打印缺失指标并统一命名口径
    status: completed
    dependencies:
      - wire-tri-comparison-cell
      - add-all-metrics-cell
  - id: verify-notebook-output
    content: 校验notebook单元顺序与输出完整性
    status: completed
    dependencies:
      - unify-metric-summary
---

## User Requirements

- 阅读并对照以下两个脚本的评测口径：
- `d:/myEnv/CodeBuddy_AgentWorkSpace/nlp/proj_finance/-nlp-rag-project/run_hallucination_experiment.py`
- `d:/myEnv/CodeBuddy_AgentWorkSpace/nlp/proj_finance/-nlp-rag-project/hallucination_eval.py`
- 检查 `d:/myEnv/CodeBuddy_AgentWorkSpace/nlp/proj_finance/-nlp-rag-project/others/finance_qwen.ipynb` 当前实际输出中缺失了哪些幻觉评测指标。
- 将缺失指标补充进 notebook 的可执行评估单元中，并形成可直接查看的结果输出。

## Product Overview

- 在现有 notebook 中补齐“幻觉评估标准”所需指标输出，确保不仅有 `F1/Hit@K`，还能看到幻觉相关核心指标与三路对比结果。
- 输出应与现有项目评估口径保持一致，避免新定义指标或改变原始实验逻辑。

## Core Features

- 对齐并补齐指标：`FActScore / Gen EM / Ret EM / citation_precision / citation_recall / hallucination_rate / factscore_simple / hit_rate / support_rate`（按适用场景输出）。
- 在 notebook 增加三路对比调用：`LLM Only / Vanilla RAG / Self-RAG` 的统一摘要输出。
- 增加单样本或小样本综合评估展示（基于 `evaluate_all_metrics`）以验证指标链路。

## Tech Stack Selection

- 现有 Python + Jupyter Notebook（`.ipynb`）流程。
- 复用现有模块：
- `run_hallucination_experiment.py`（`run_comparison` 三路对比）
- `hallucination_eval.py`（`evaluate_all_metrics` 与基础指标函数）
- 不引入新依赖或新评估框架。

## Implementation Approach

- 采用“复用现有评测管线 + notebook 补齐展示层”的策略：  
先以 `run_comparison()` 输出主实验指标（FActScore/Hit@K/F1/GenEM/RetEM/support_rate），再用 `evaluate_all_metrics()` 对代表样本补充 citation 与 hallucination 细粒度指标。
- 关键决策：

1. 不改评估核心脚本口径，仅在 notebook 接入与展示，降低回归风险。  
2. 保留当前主实验单元，新增/替换空白单元承载“标准幻觉评估输出”。  
3. 指标分层展示：总体对比（宏观）+ 样本级综合指标（微观）。

## Implementation Notes

- 以 notebook 当前已存在但注释/空白的评估单元为落点，减少结构变更。
- 主流程优先调用 `run_comparison(subset_size, top_k, retriever_type, generator_type, model/model_path)`，避免手写重复评估逻辑。
- `evaluate_all_metrics` 需传入：
- `prediction`（生成答案）
- `pred_docs`（检索文档列表，含 `text`）
- `gold_answers`（标准答案列表）
- 仅添加评估输出，不改数据集加载和模型加载路径，控制影响面。

## Architecture Design

- 现有结构保持不变：  
数据加载/检索/生成 → 评估函数调用 → notebook 控制台汇总输出。
- 新增的是 notebook 中“评估编排层”，不改变 `rag_finance.py`、`run_hallucination_experiment.py`、`hallucination_eval.py` 的核心实现。

## Directory Structure

## Directory Structure Summary

本次仅修改 notebook，接入已存在评估能力并补齐幻觉指标展示。

d:/myEnv/CodeBuddy_AgentWorkSpace/nlp/proj_finance/-nlp-rag-project/
└── others/
└── finance_qwen.ipynb  # [MODIFY] 补齐幻觉评测输出单元。新增三路对比调用与综合指标展示；输出 FActScore/Gen EM/Ret EM/citation_precision/citation_recall/hallucination_rate/factscore_simple/hit_rate/support_rate（按方法适用）。

## Agent Extensions

### SubAgent

- **code-explorer**
- Purpose: 快速复核 notebook 中可插入单元位置与现有评估调用链一致性。
- Expected outcome: 精确定位应替换/新增的单元，确保指标接入不遗漏且不破坏原执行顺序。��原执行顺序。