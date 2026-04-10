---
name: Notebook显存释放与LLM-Only基线
overview: 清空旧计划后，仅实现两个任务：在notebook中先释放Contriever显存再加载生成器，并补齐LLM Only无检索基线；其余已明确排除。
todos:
  - id: rescope-plan
    content: 按新范围重置任务并移除已取消项影响
    status: completed
  - id: analyze-call-chain
    content: 使用 [subagent:code-explorer] 复核生成器与评估调用链
    status: completed
    dependencies:
      - rescope-plan
  - id: optimize-notebook-vram
    content: 修改 finance_bm25_qwen(3).ipynb 实现先释放显存再加载模型
    status: completed
    dependencies:
      - analyze-call-chain
  - id: implement-llm-only-eval
    content: 在 run_hallucination_experiment.py 新增 LLM Only 评估与指标统计
    status: completed
    dependencies:
      - analyze-call-chain
  - id: integrate-comparison-output
    content: 扩展 run_comparison 汇总、打印与 JSON 保存包含 llm_only
    status: completed
    dependencies:
      - implement-llm-only-eval
  - id: notebook-llm-only-demo
    content: 在 notebook 增加 LLM Only 调用单元与三路对比展示
    status: completed
    dependencies:
      - optimize-notebook-vram
      - integrate-comparison-output
---

## User Requirements

- 清除此前不匹配的新计划内容，仅保留当前两项实施范围。
- 在 notebook 中完成“先释放 Contriever 模型显存，再加载生成器”，解决当前显存不足导致的模型加载失败。
- 实现 LLM Only 基线（不经过检索，直接生成答案）并纳入现有实验对比流程。
- 明确忽略已取消或不在范围内的内容：LangChain Structured Chain、CUAD、LegalBench-RAG-mini。

## Product Overview

当前项目将保留“Vanilla RAG / Self-RAG”主流程，并新增“LLM Only”对照能力；同时优化 notebook 运行顺序，使本地模型加载更稳定。最终表现为：notebook 可直接执行显存释放与基线对比，实验脚本输出包含 LLM Only 结果。

## Core Features

- notebook 显存释放与加载顺序优化（先清理 GPU 占用，再加载生成器）。
- LLM Only 无检索生成与指标评估（与现有指标口径一致）。
- 实验汇总结果扩展（新增 LLM Only 指标与明细保存）。

## Tech Stack Selection

- 语言与运行：Python（现有脚本与 notebook）
- 核心模块复用：`rag_finance.py`（生成器/检索器）、`run_hallucination_experiment.py`（实验编排）、`hallucination_eval.py`（指标）

## Implementation Approach

采用“最小侵入扩展”策略：
1) 在 notebook 先增加显存回收与对象释放步骤，再执行本地模型加载；
2) 在实验脚本新增独立 `LLM Only` 评估函数，复用现有指标函数；
3) 将 `run_comparison` 扩展为三路对比（LLM Only / Vanilla RAG / Self-RAG），并统一输出结构。
时间复杂度与现有评估同量级，新增开销主要是多一条基线路径（线性随样本数增长）。

## Implementation Notes

- 显存释放需覆盖：删除 Contriever/相关大对象、`gc.collect()`、`torch.cuda.empty_cache()`、必要时 `torch.cuda.ipc_collect()`。
- LLM Only 不调用 `RAGSystem.answer()`，直接走生成器 `generate(question, [])`，避免检索依赖。
- 保持现有结果字段向后兼容，新增字段放在并列层级，避免破坏旧分析脚本。
- 不改动 Structured Chain 与数据集扩展代码，控制影响面。

## Architecture Design

- 数据流新增分支：`examples -> LLM Only generator -> 指标计算 -> 汇总`
- 原有分支保留：`Vanilla RAG`、`Self-RAG`
- 三分支统一进入结果聚合与 JSON 保存层。

## Directory Structure

## Directory Structure Summary

本次仅修改现有 notebook 与实验脚本，不新增无关模块。

d:/myEnv/CodeBuddy_AgentWorkSpace/nlp/proj_finance/-nlp-rag-project/

- others/finance_bm25_qwen (3).ipynb  # [MODIFY] 增加显存释放单元，调整执行顺序，新增 LLM Only 运行与对比展示单元。
- run_hallucination_experiment.py  # [MODIFY] 新增 LLM Only 评估函数，扩展 run_comparison 输出与结果保存结构。

## Agent Extensions

- **SubAgent: code-explorer**
- Purpose: 复核 `rag_finance.py` 与 `run_hallucination_experiment.py` 的调用链和字段兼容性。
- Expected outcome: 确认 LLM Only 新增路径与现有指标、结果结构一致且无断链。