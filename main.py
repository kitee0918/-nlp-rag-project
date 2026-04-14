"""
幻觉缓解对比实验脚本
对比 Vanilla RAG vs Self-RAG 在 FinanceBench 上的幻觉缓解效果

运行方式:
    python run_hallucination_experiment.py --subset 20 --top_k 3
    
在 Colab/Jupyter 中:
    from run_hallucination_experiment import run_comparison
    results = run_comparison(subset_size=20)
"""

import os
import sys

# ============================================
# 重要：在导入任何其他模块之前设置镜像站点
# HuggingFace 镜像需要在 datasets 模块导入前设置才能生效
# ============================================
if 'HF_ENDPOINT' not in os.environ:
    os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'
    print(f"使用 HuggingFace 镜像站点: {os.environ['HF_ENDPOINT']}")

import json
import argparse
from datetime import datetime
from typing import List, Dict

# 设置 UTF-8 编码
os.environ['PYTHONIOENCODING'] = 'utf-8'
os.environ['PYTHONUTF8'] = '1'

# 尝试加载环境变量
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

import torch

# 导入现有模块
from methods import (
    load_financebench,
    BM25Retriever,
    ContrieverRetriever,
    QwenGenerator,
    LlamaGenerator,
    RAGSystem,
    compute_f1,
    compute_retrieval_metrics,
    Config
)

from eval import (
    FActScoreEvaluator,
    FActScoreLocalEvaluator,
    SelfRAGGenerator,
    HallucinationEvaluator,
    HallucinationConfig,
    # 新增评估指标（复用自 helpers.py）
    normalise,
    retrieval_em,
    generation_em,
    generation_f1,
    evaluate_all_metrics
)


# 向后兼容映射
GENERATOR_TYPE_MAPPING = {
    "qwen": "api",
    "llama": "local"
}


# ============================================
# Vanilla RAG 幻觉评估
# ============================================

def evaluate_vanilla_rag(
    rag_system: RAGSystem,
    examples: List[Dict],
    factscore_evaluator,
    top_k: int = 3,
    verbose: bool = False
) -> Dict:
    """
    评估 Vanilla RAG 的幻觉情况
    
    Args:
        rag_system: Vanilla RAG 系统实例
        examples: 测试示例
        factscore_evaluator: FActScore 评估器（API 或本地版本）
        top_k: 检索文档数量
        verbose: 是否打印详细信息
        
    Returns:
        评估结果
    """
    factscores = []
    hit_at_ks = []
    f1_scores = []
    gen_em_scores = []  # 新增：生成精确匹配
    ret_em_scores = []  # 新增：检索精确匹配
    all_results = []
    
    for i, example in enumerate(examples):
        # 生成答案
        result = rag_system.answer(example['question'])
        
        # 获取检索文档
        retrieved_docs = [doc for doc in result['retrieved_documents']]
        
        # 计算 FActScore
        factscore_result = factscore_evaluator.compute_factscore(
            answer=result['answer'],
            evidence=example.get('evidence_text', ''),
            verbose=verbose and i < 3
        )
        
        # 计算 Hit@K（引用质量）
        retrieved_ids = [doc['id'] for doc in retrieved_docs]
        hit_at_k = 1.0 if example.get('evidence_doc') in retrieved_ids else 0.0
        
        # 计算 F1（使用改进的 generation_f1）
        f1 = generation_f1(result['answer'], [example['answer']])
        
        # 新增：计算生成精确匹配
        gen_em = generation_em(result['answer'], [example['answer']])
        
        # 新增：计算检索精确匹配
        # 需要将检索文档转换为带 text 字段的字典
        retrieved_docs_with_text = [{'text': doc.get('text', '')} for doc in retrieved_docs]
        ret_em = retrieval_em(retrieved_docs_with_text, [example['answer']])
        
        factscores.append(factscore_result['factscore'])
        hit_at_ks.append(hit_at_k)
        f1_scores.append(f1)
        gen_em_scores.append(gen_em)
        ret_em_scores.append(ret_em)
        
        all_results.append({
            "question": example['question'],
            "answer": result['answer'],
            "ground_truth": example['answer'],
            "factscore": factscore_result['factscore'],
            "hit_at_k": hit_at_k,
            "f1": f1,
            "gen_em": gen_em,
            "ret_em": ret_em,
            "num_atoms": factscore_result['num_atoms'],
            "num_supported_atoms": factscore_result['num_supported']  # 注意: hallucination_eval 返回的是 num_supported
        })
    
    return {
        "avg_factscore": sum(factscores) / len(factscores) if factscores else 0.0,
        "avg_hit_at_k": sum(hit_at_ks) / len(hit_at_ks) if hit_at_ks else 0.0,
        "avg_f1": sum(f1_scores) / len(f1_scores) if f1_scores else 0.0,
        "avg_gen_em": sum(gen_em_scores) / len(gen_em_scores) if gen_em_scores else 0.0,
        "avg_ret_em": sum(ret_em_scores) / len(ret_em_scores) if ret_em_scores else 0.0,
        "num_examples": len(examples),
        "individual_results": all_results,
        "retriever": rag_system.retriever.get_name(),
        "generator": rag_system.generator.get_name(),
        "method": "Vanilla RAG"
    }


# ============================================
# LLM Only 幻觉评估（无检索）
# ============================================

def evaluate_llm_only(
    examples: List[Dict],
    generator,
    factscore_evaluator,
    verbose: bool = False
) -> Dict:
    """
    评估 LLM Only（不检索，直接生成）
    """
    factscores = []
    f1_scores = []
    gen_em_scores = []
    all_results = []

    for i, example in enumerate(examples):
        result = generator.generate(example['question'], [])

        factscore_result = factscore_evaluator.compute_factscore(
            answer=result['answer'],
            evidence=example.get('evidence_text', ''),
            verbose=verbose and i < 3
        )

        f1 = generation_f1(result['answer'], [example['answer']])
        gen_em = generation_em(result['answer'], [example['answer']])

        factscores.append(factscore_result['factscore'])
        f1_scores.append(f1)
        gen_em_scores.append(gen_em)

        all_results.append({
            "question": example['question'],
            "answer": result['answer'],
            "ground_truth": example['answer'],
            "factscore": factscore_result['factscore'],
            "hit_at_k": 0.0,
            "f1": f1,
            "gen_em": gen_em,
            "ret_em": 0.0,
            "num_atoms": factscore_result['num_atoms'],
            "num_supported_atoms": factscore_result['num_supported'],
            "retrieved_documents": []
        })

    return {
        "avg_factscore": sum(factscores) / len(factscores) if factscores else 0.0,
        "avg_hit_at_k": 0.0,
        "avg_f1": sum(f1_scores) / len(f1_scores) if f1_scores else 0.0,
        "avg_gen_em": sum(gen_em_scores) / len(gen_em_scores) if gen_em_scores else 0.0,
        "avg_ret_em": 0.0,
        "num_examples": len(examples),
        "individual_results": all_results,
        "retriever": "LLM_ONLY",
        "generator": generator.get_name(),
        "method": "LLM Only"
    }


# ============================================
# Self-RAG 幻觉评估
# ============================================

def evaluate_self_rag(

    examples: List[Dict],
    hallucination_evaluator: HallucinationEvaluator,
    top_k: int = 3,
    verbose: bool = False
) -> Dict:
    """
    评估 Self-RAG 的幻觉情况
    
    Args:
        examples: 测试示例
        hallucination_evaluator: 幻觉评估器
        top_k: 检索文档数量
        verbose: 是否打印详细信息
        
    Returns:
        评估结果
    """
    results = hallucination_evaluator.batch_evaluate(
        examples,
        top_k=top_k,
        verbose=verbose
    )
    
    # 计算 F1 分数和其他指标
    f1_scores = []
    gen_em_scores = []
    ret_em_scores = []
    
    for i, result in enumerate(results['individual_results']):
        # 使用改进的 generation_f1
        f1 = generation_f1(result['self_rag_answer'], [examples[i]['answer']])
        f1_scores.append(f1)
        result['f1'] = f1
        
        # 新增：生成精确匹配
        gen_em = generation_em(result['self_rag_answer'], [examples[i]['answer']])
        gen_em_scores.append(gen_em)
        result['gen_em'] = gen_em
        
        # 新增：检索精确匹配
        retrieved_docs = [{'text': doc.get('text', '')} for doc in result['retrieved_documents']]
        ret_em = retrieval_em(retrieved_docs, [examples[i]['answer']])
        ret_em_scores.append(ret_em)
        result['ret_em'] = ret_em
    
    
    results['avg_f1'] = sum(f1_scores) / len(f1_scores) if f1_scores else 0.0
    results['avg_gen_em'] = sum(gen_em_scores) / len(gen_em_scores) if gen_em_scores else 0.0
    results['avg_ret_em'] = sum(ret_em_scores) / len(ret_em_scores) if ret_em_scores else 0.0
    results['method'] = "Self-RAG"
    
    return results


def aggregate_all_metrics(rows: List[Dict], answer_key: str, gold_key: str) -> Dict:
    """对 evaluate_all_metrics 做样本均值聚合。"""
    if not rows:
        return {}

    metrics_sum = {}
    count = 0
    for row in rows:
        prediction = row.get(answer_key, "")
        gold_answer = row.get(gold_key, "")
        pred_docs = row.get("retrieved_documents", []) or []
        doc_texts = [d.get("text", "") for d in pred_docs if d.get("text")]

        metric_row = evaluate_all_metrics(
            prediction=prediction,
            pred_docs=pred_docs,
            gold_answers=[gold_answer] if gold_answer else [],
            documents=doc_texts if doc_texts else None,
        )

        for k, v in metric_row.items():
            metrics_sum[k] = metrics_sum.get(k, 0.0) + float(v)
        count += 1

    return {k: (v / count if count > 0 else 0.0) for k, v in metrics_sum.items()}


# ============================================
# 单一评估函数（支持惰性加载和内存释放）
# ============================================

def run_single_evaluation(
    examples: List[Dict],
    corpus: List[Dict],
    evaluation_type: str,  # "llm_only", "vanilla_rag", "self_rag"
    top_k: int = 3,
    retriever_type: str = "bm25",
    generator_type: str = "api",
    model: str = "qwen-plus",
    model_path: str = None,
    api_key: str = None,
    device: str = None,
    verbose: bool = False
) -> Dict:
    """
    运行单一评估（惰性加载，用完即释放）

    Args:
        examples: 测试示例
        corpus: 文档语料库
        evaluation_type: 评估类型，三选一
            - "llm_only": 仅 LLM
            - "vanilla_rag": Vanilla RAG
            - "self_rag": Self-RAG
        top_k: 检索文档数量
        retriever_type: 检索器类型 ("bm25" 或 "contriever")
        generator_type: 生成器类型，"api" 或 "local"
        model: API 模型名称 (仅 generator_type="api" 时有效)
        model_path: 本地模型路径 (仅 generator_type="local" 时有效)
        api_key: DashScope API Key
        device: 运行设备
        verbose: 是否打印详细信息

    Returns:
        评估结果字典（与 run_comparison 返回的对应部分格式相同）
    """
    # 向后兼容映射
    generator_type = GENERATOR_TYPE_MAPPING.get(generator_type.lower(), generator_type.lower())

    # 设置 API Key
    if api_key:
        os.environ['DASHSCOPE_API_KEY'] = api_key

    # 检测设备
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    print("=" * 70)
    print(f"📊 单评估：{evaluation_type.upper()}")
    print("=" * 70)
    print(f"样本数量：{len(examples)}")
    print(f"Top-K: {top_k}")
    print(f"检索器：{retriever_type.upper()}")
    print(f"生成器类型：{generator_type.upper()}")
    print(f"设备：{device}")
    print("=" * 70)

    # 初始化检索器
    os.makedirs(Config.CACHE_DIR, exist_ok=True)
    if retriever_type.lower() == "bm25":
        retriever = BM25Retriever(
            corpus,
            cache_path=os.path.join(Config.CACHE_DIR, "bm25_index_finance.pkl")
        )
    else:
        retriever = ContrieverRetriever(
            corpus,
            cache_path=os.path.join(Config.CACHE_DIR, "contriever_embeddings_finance.pkl"),
            device=device
        )

    results = None
    generator = None
    factscore_evaluator = None
    hallucination_evaluator = None

    try:
        if evaluation_type == "llm_only":
            print("\n🔹 运行 LLM Only 评估...")
            if generator_type == "local":
                generator = LlamaGenerator(
                    model_path=model_path or Config.LLAMA_MODEL_PATH,
                    device=device
                )
                # 使用懒加载
                factscore_evaluator = FActScoreLocalEvaluator(
                    model_path=model_path or Config.LLAMA_MODEL_PATH,
                    device=device,
                    lazy_load=True  # 启用懒加载
                )
            else:
                generator = QwenGenerator(model=model)
                factscore_evaluator = FActScoreEvaluator(model=model)

            # 确保在评估前初始化（如果支持懒加载）
            if hasattr(factscore_evaluator, 'ensure_initialized'):
                factscore_evaluator.ensure_initialized()

            results = evaluate_llm_only(examples, generator, factscore_evaluator, verbose=verbose)

        elif evaluation_type == "vanilla_rag":
            print("\n🔹 运行 Vanilla RAG 评估...")
            if generator_type == "local":
                generator = LlamaGenerator(
                    model_path=model_path or Config.LLAMA_MODEL_PATH,
                    device=device
                )
                # 使用懒加载
                factscore_evaluator = FActScoreLocalEvaluator(
                    model_path=model_path or Config.LLAMA_MODEL_PATH,
                    device=device,
                    lazy_load=True  # 启用懒加载
                )
            else:
                generator = QwenGenerator(model=model)
                factscore_evaluator = FActScoreEvaluator(model=model)

            # 确保在评估前初始化（如果支持懒加载）
            if hasattr(factscore_evaluator, 'ensure_initialized'):
                factscore_evaluator.ensure_initialized()

            rag_system = RAGSystem(retriever, generator, top_k=top_k)
            results = evaluate_vanilla_rag(rag_system, examples, factscore_evaluator, top_k=top_k, verbose=verbose)

        elif evaluation_type == "self_rag":
            print("\n🔹 运行 Self-RAG 评估...")
            # 使用惰性加载模式
            hallucination_evaluator = HallucinationEvaluator(
                retriever=retriever,
                generator_type=generator_type,
                model=model,
                model_path=model_path,
                device=device,
                lazy_load=True  # 惰性加载
            )
            results = evaluate_self_rag(examples, hallucination_evaluator, top_k=top_k, verbose=verbose)
        else:
            raise ValueError(f"未知的评估类型：{evaluation_type}")

        # 打印结果摘要
        print(f"\n{evaluation_type.upper()} 结果:")
        print(f"  FActScore:    {results['avg_factscore']:.4f}")
        print(f"  Hit@K:        {results['avg_hit_at_k']:.4f}")
        print(f"  F1 Score:     {results['avg_f1']:.4f}")
        print(f"  Gen EM:       {results['avg_gen_em']:.4f}")
        print(f"  Ret EM:       {results['avg_ret_em']:.4f}")

        return results

    finally:
        # 释放资源
        print("\n🧹 清理资源...")
        if hallucination_evaluator is not None:
            hallucination_evaluator.cleanup()
        if generator is not None and hasattr(generator, 'model') and generator.model is not None:
            del generator.model
            del generator.tokenizer
        if factscore_evaluator is not None:
            # 如果支持 cleanup 方法，使用它
            if hasattr(factscore_evaluator, 'cleanup'):
                factscore_evaluator.cleanup()
            # 否则手动清理
            elif hasattr(factscore_evaluator, 'model') and factscore_evaluator.model is not None:
                del factscore_evaluator.model
                del factscore_evaluator.tokenizer
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            print(f"GPU 显存已清空，当前分配：{torch.cuda.memory_allocated() / 1024**2:.2f} MB")
        print("✅ 资源清理完成")


# ============================================
# 对比实验
# ============================================

def run_comparison(

    subset_size: int = 20,
    top_k: int = 3,
    retriever_type: str = "bm25",
    generator_type: str = "api",
    model: str = "qwen-plus",
    model_path: str = None,
    api_key: str = None,
    device: str = None,
    verbose: bool = False
) -> Dict:
    """
    运行 Vanilla RAG vs Self-RAG 对比实验
    
    Args:
        subset_size: 测试样本数量
        top_k: 检索文档数量
        retriever_type: 检索器类型 ("bm25" 或 "contriever")
        generator_type: 生成器类型，"api" 或 "local"（向后兼容: "qwen"=api, "llama"=local）
        model: API 模型名称，默认 "qwen-plus" (仅 generator_type="api" 时有效)
        model_path: 本地模型路径 (仅 generator_type="local" 时有效)
        api_key: DashScope API Key
        device: 运行设备
        verbose: 是否打印详细信息
        
    Returns:
        对比结果
    """
    # 向后兼容映射
    generator_type = GENERATOR_TYPE_MAPPING.get(generator_type.lower(), generator_type.lower())
    
    # 设置 API Key
    if api_key:
        os.environ['DASHSCOPE_API_KEY'] = api_key
    
    # 检测设备
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    
    print("=" * 70)
    print("🔬 幻觉缓解对比实验: Vanilla RAG vs Self-RAG")
    print("=" * 70)
    print(f"测试样本: {subset_size}")
    print(f"Top-K: {top_k}")
    print(f"检索器: {retriever_type.upper()}")
    print(f"生成器类型: {generator_type.upper()}")
    if generator_type == "api":
        print(f"生成模型: {model}")
    else:
        print(f"模型路径: {model_path or Config.LLAMA_MODEL_PATH}")
    print(f"设备: {device}")
    print("=" * 70)
    
    # 1. 加载数据
    print("\n📂 加载 FinanceBench 数据...")
    examples, corpus = load_financebench(subset_size=subset_size)
    
    # 2. 初始化检索器
    print(f"\n🔧 初始化 {retriever_type.upper()} 检索器...")
    os.makedirs(Config.CACHE_DIR, exist_ok=True)
    
    if retriever_type.lower() == "bm25":
        retriever = BM25Retriever(
            corpus,
            cache_path=os.path.join(Config.CACHE_DIR, "bm25_index_finance.pkl")
        )
    else:
        retriever = ContrieverRetriever(
            corpus,
            cache_path=os.path.join(Config.CACHE_DIR, "contriever_embeddings_finance.pkl"),
            device=device
        )
    
    # 3. 初始化评估组件
    print("\n🤖 初始化评估组件...")
    
    # Vanilla RAG 生成器
    if generator_type == "local":
        vanilla_generator = LlamaGenerator(
            model_path=model_path or Config.LLAMA_MODEL_PATH,
            device=device
        )
        # 本地模型使用 FActScoreLocalEvaluator
        factscore_evaluator = FActScoreLocalEvaluator(
            model_path=model_path or Config.LLAMA_MODEL_PATH,
            device=device
        )
        # 加载本地模型
        factscore_evaluator._load_model(model_path or Config.LLAMA_MODEL_PATH)
    else:
        vanilla_generator = QwenGenerator(model=model)
        factscore_evaluator = FActScoreEvaluator(model=model)
    
    # Self-RAG 幻觉评估器
    hallucination_evaluator = HallucinationEvaluator(
        retriever=retriever,
        generator_type=generator_type,
        model=model,
        model_path=model_path,
        device=device
    )
    
    # 4. 评估 LLM Only
    print("\n" + "=" * 70)
    print("📊 评估 LLM Only（无检索）...")
    print("=" * 70)

    llm_only_results = evaluate_llm_only(
        examples,
        vanilla_generator,
        factscore_evaluator,
        verbose=verbose
    )

    print(f"\nLLM Only 结果:")
    print(f"  FActScore:    {llm_only_results['avg_factscore']:.4f}")
    print(f"  Hit@K:        {llm_only_results['avg_hit_at_k']:.4f}")
    print(f"  F1 Score:     {llm_only_results['avg_f1']:.4f}")
    print(f"  Gen EM:       {llm_only_results['avg_gen_em']:.4f}")
    print(f"  Ret EM:       {llm_only_results['avg_ret_em']:.4f}")

    # 5. 评估 Vanilla RAG
    print("\n" + "=" * 70)
    print("📊 评估 Vanilla RAG...")
    print("=" * 70)

    vanilla_rag = RAGSystem(retriever, vanilla_generator, top_k=top_k)
    vanilla_results = evaluate_vanilla_rag(
        vanilla_rag,
        examples,
        factscore_evaluator,
        top_k=top_k,
        verbose=verbose
    )

    print(f"\nVanilla RAG 结果:")
    print(f"  FActScore:    {vanilla_results['avg_factscore']:.4f}")
    print(f"  Hit@K:        {vanilla_results['avg_hit_at_k']:.4f}")
    print(f"  F1 Score:     {vanilla_results['avg_f1']:.4f}")
    print(f"  Gen EM:       {vanilla_results['avg_gen_em']:.4f}")
    print(f"  Ret EM:       {vanilla_results['avg_ret_em']:.4f}")

    # 6. 评估 Self-RAG
    print("\n" + "=" * 70)
    print("📊 评估 Self-RAG...")
    print("=" * 70)

    self_rag_results = evaluate_self_rag(
        examples,
        hallucination_evaluator,
        top_k=top_k,
        verbose=verbose
    )

    print(f"\nSelf-RAG 结果:")
    print(f"  FActScore:    {self_rag_results['avg_factscore']:.4f}")
    print(f"  Hit@K:        {self_rag_results['avg_hit_at_k']:.4f}")
    print(f"  F1 Score:     {self_rag_results['avg_f1']:.4f}")
    print(f"  Gen EM:       {self_rag_results['avg_gen_em']:.4f}")
    print(f"  Ret EM:       {self_rag_results['avg_ret_em']:.4f}")
    print(f"  支持率:       {self_rag_results['support_rate']:.4f}")

    # 7. evaluate_all_metrics 全量指标（按方法聚合均值）
    llm_all_metrics = aggregate_all_metrics(
        llm_only_results["individual_results"], answer_key="answer", gold_key="ground_truth"
    )
    vanilla_all_metrics = aggregate_all_metrics(
        vanilla_results["individual_results"], answer_key="answer", gold_key="ground_truth"
    )
    self_all_metrics = aggregate_all_metrics(
        self_rag_results["individual_results"], answer_key="self_rag_answer", gold_key="ground_truth_answer"
    )

    print("\n=== evaluate_all_metrics 均值 ===")
    for name, metrics in [
        ("LLM Only", llm_all_metrics),
        ("Vanilla RAG", vanilla_all_metrics),
        ("Self-RAG", self_all_metrics),
    ]:
        print(f"\n{name}:")
        print(
            "  "
            f"GenEM={metrics.get('generation_em', 0.0):.4f}, "
            f"GenF1={metrics.get('generation_f1', 0.0):.4f}, "
            f"HitRate={metrics.get('hit_rate', 0.0):.4f}, "
            f"RetEM={metrics.get('retrieval_em', 0.0):.4f}, "
            f"CitP={metrics.get('citation_precision', 0.0):.4f}, "
            f"CitR={metrics.get('citation_recall', 0.0):.4f}, "
            f"Hall={metrics.get('hallucination_rate', 0.0):.4f}, "
            f"FactSimple={metrics.get('factscore_simple', 0.0):.4f}"
        )

    # 8. 对比分析

    print("\n" + "=" * 80)
    print("📈 对比结果")
    print("=" * 80)
    print(f"{'方法':<12} {'FActScore':<10} {'Hit@K':<10} {'F1':<10} {'Gen EM':<10} {'Ret EM':<10}")
    print("-" * 80)
    print(f"{'LLM Only':<12} {llm_only_results['avg_factscore']:<10.4f} {llm_only_results['avg_hit_at_k']:<10.4f} {llm_only_results['avg_f1']:<10.4f} {llm_only_results['avg_gen_em']:<10.4f} {llm_only_results['avg_ret_em']:<10.4f}")
    print(f"{'Vanilla RAG':<12} {vanilla_results['avg_factscore']:<10.4f} {vanilla_results['avg_hit_at_k']:<10.4f} {vanilla_results['avg_f1']:<10.4f} {vanilla_results['avg_gen_em']:<10.4f} {vanilla_results['avg_ret_em']:<10.4f}")
    print(f"{'Self-RAG':<12} {self_rag_results['avg_factscore']:<10.4f} {self_rag_results['avg_hit_at_k']:<10.4f} {self_rag_results['avg_f1']:<10.4f} {self_rag_results['avg_gen_em']:<10.4f} {self_rag_results['avg_ret_em']:<10.4f}")
    print("=" * 80)

    # 计算提升（保留原有 Self-RAG vs Vanilla 口径，另增对 LLM-Only 的提升）
    def _calc_improve(new_val, old_val):
        return (new_val - old_val) / old_val * 100 if old_val > 0 else 0

    factscore_improve = _calc_improve(self_rag_results['avg_factscore'], vanilla_results['avg_factscore'])
    hit_improve = _calc_improve(self_rag_results['avg_hit_at_k'], vanilla_results['avg_hit_at_k'])
    f1_improve = _calc_improve(self_rag_results['avg_f1'], vanilla_results['avg_f1'])
    gen_em_improve = _calc_improve(self_rag_results['avg_gen_em'], vanilla_results['avg_gen_em'])
    ret_em_improve = _calc_improve(self_rag_results['avg_ret_em'], vanilla_results['avg_ret_em'])

    self_vs_llm_factscore = _calc_improve(self_rag_results['avg_factscore'], llm_only_results['avg_factscore'])
    self_vs_llm_f1 = _calc_improve(self_rag_results['avg_f1'], llm_only_results['avg_f1'])

    print(f"\n📊 Self-RAG vs Vanilla RAG 提升:")
    print(f"   FActScore: {'+' if factscore_improve >= 0 else ''}{factscore_improve:.2f}%")
    print(f"   Hit@K:     {'+' if hit_improve >= 0 else ''}{hit_improve:.2f}%")
    print(f"   F1 Score:  {'+' if f1_improve >= 0 else ''}{f1_improve:.2f}%")
    print(f"   Gen EM:    {'+' if gen_em_improve >= 0 else ''}{gen_em_improve:.2f}%")
    print(f"   Ret EM:    {'+' if ret_em_improve >= 0 else ''}{ret_em_improve:.2f}%")

    print(f"\n📊 Self-RAG vs LLM Only 提升:")
    print(f"   FActScore: {'+' if self_vs_llm_factscore >= 0 else ''}{self_vs_llm_factscore:.2f}%")
    print(f"   F1 Score:  {'+' if self_vs_llm_f1 >= 0 else ''}{self_vs_llm_f1:.2f}%")

    
    # 7. 保存结果
    os.makedirs(Config.RESULTS_DIR, exist_ok=True)
    result_file = os.path.join(
        Config.RESULTS_DIR,
        f"hallucination_comparison_{retriever_type}_{subset_size}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    )
    
    comparison_results = {
        "config": {
            "subset_size": subset_size,
            "top_k": top_k,
            "retriever": retriever_type,
            "model": model,
            "generator_type": generator_type
        },
        "llm_only": {
            "avg_factscore": llm_only_results['avg_factscore'],
            "avg_hit_at_k": llm_only_results['avg_hit_at_k'],
            "avg_f1": llm_only_results['avg_f1'],
            "avg_gen_em": llm_only_results['avg_gen_em'],
            "avg_ret_em": llm_only_results['avg_ret_em']
        },
        "vanilla_rag": {
            "avg_factscore": vanilla_results['avg_factscore'],
            "avg_hit_at_k": vanilla_results['avg_hit_at_k'],
            "avg_f1": vanilla_results['avg_f1'],
            "avg_gen_em": vanilla_results['avg_gen_em'],
            "avg_ret_em": vanilla_results['avg_ret_em']
        },
        "self_rag": {
            "avg_factscore": self_rag_results['avg_factscore'],
            "avg_hit_at_k": self_rag_results['avg_hit_at_k'],
            "avg_f1": self_rag_results['avg_f1'],
            "avg_gen_em": self_rag_results['avg_gen_em'],
            "avg_ret_em": self_rag_results['avg_ret_em'],
            "support_rate": self_rag_results['support_rate']
        },
        "all_metrics": {
            "llm_only": llm_all_metrics,
            "vanilla_rag": vanilla_all_metrics,
            "self_rag": self_all_metrics
        },
        "improvement": {

            "factscore": factscore_improve,
            "hit_at_k": hit_improve,
            "f1": f1_improve,
            "gen_em": gen_em_improve,
            "ret_em": ret_em_improve,
            "self_vs_llm_factscore": self_vs_llm_factscore,
            "self_vs_llm_f1": self_vs_llm_f1
        },
        "llm_only_details": llm_only_results['individual_results'][:10],
        "vanilla_rag_details": vanilla_results['individual_results'][:10],  # 只保存前10个详细结果
        "self_rag_details": self_rag_results['individual_results'][:10]
    }

    
    with open(result_file, 'w', encoding='utf-8') as f:
        json.dump(comparison_results, f, ensure_ascii=False, indent=2)
    
    print(f"\n💾 结果已保存至: {result_file}")
    
    return comparison_results


# ============================================
# 主函数
# ============================================

def main():
    parser = argparse.ArgumentParser(description="幻觉缓解对比实验")
    parser.add_argument("--subset", type=int, default=20, help="测试样本数量")
    parser.add_argument("--top_k", type=int, default=3, help="检索文档数量")
    parser.add_argument("--retriever", type=str, default="bm25", choices=["bm25", "contriever"],
                        help="检索器类型")
    parser.add_argument("--model", type=str, default="qwen-plus", help="Qwen 模型名称")
    parser.add_argument("--device", type=str, default=None, help="运行设备")
    parser.add_argument("--verbose", action="store_true", help="打印详细信息")
    
    args, _ = parser.parse_known_args()
    
    run_comparison(
        subset_size=args.subset,
        top_k=args.top_k,
        retriever_type=args.retriever,
        model=args.model,
        device=args.device,
        verbose=args.verbose
    )


if __name__ == "__main__":
    main()
