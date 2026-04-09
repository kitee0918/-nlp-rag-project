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
from rag_finance import (
    load_financebench,
    BM25Retriever,
    ContrieverRetriever,
    QwenGenerator,
    RAGSystem,
    compute_f1,
    compute_retrieval_metrics,
    Config
)

from hallucination_eval import (
    FActScoreEvaluator,
    FActScoreLocalEvaluator,
    SelfRAGGenerator,
    HallucinationEvaluator,
    HallucinationConfig,
    # 新增评估指标（复用自 helpers.py）
    normalise,
    retrieval_em,
    generation_em,
    generation_f1
)


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
            "num_supported_atoms": factscore_result['num_supported_atoms']
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


# ============================================
# 对比实验
# ============================================

def run_comparison(
    subset_size: int = 20,
    top_k: int = 3,
    retriever_type: str = "bm25",
    model: str = "qwen-plus",
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
        model: Qwen 模型名称
        api_key: DashScope API Key
        device: 运行设备
        verbose: 是否打印详细信息
        
    Returns:
        对比结果
    """
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
    print(f"生成模型: {model}")
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
    vanilla_generator = QwenGenerator(model=model)
    
    # FActScore 评估器
    factscore_evaluator = FActScoreEvaluator(model=model)
    
    # Self-RAG 幻觉评估器
    hallucination_evaluator = HallucinationEvaluator(
        retriever=retriever,
        model=model
    )
    
    # 4. 评估 Vanilla RAG
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
    
    # 5. 评估 Self-RAG
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
    
    # 6. 对比分析
    print("\n" + "=" * 80)
    print("📈 对比结果")
    print("=" * 80)
    print(f"{'方法':<12} {'FActScore':<10} {'Hit@K':<10} {'F1':<10} {'Gen EM':<10} {'Ret EM':<10}")
    print("-" * 80)
    print(f"{'Vanilla RAG':<12} {vanilla_results['avg_factscore']:<10.4f} {vanilla_results['avg_hit_at_k']:<10.4f} {vanilla_results['avg_f1']:<10.4f} {vanilla_results['avg_gen_em']:<10.4f} {vanilla_results['avg_ret_em']:<10.4f}")
    print(f"{'Self-RAG':<12} {self_rag_results['avg_factscore']:<10.4f} {self_rag_results['avg_hit_at_k']:<10.4f} {self_rag_results['avg_f1']:<10.4f} {self_rag_results['avg_gen_em']:<10.4f} {self_rag_results['avg_ret_em']:<10.4f}")
    print("=" * 80)
    
    # 计算提升
    def _calc_improve(new_val, old_val):
        return (new_val - old_val) / old_val * 100 if old_val > 0 else 0
    
    factscore_improve = _calc_improve(self_rag_results['avg_factscore'], vanilla_results['avg_factscore'])
    hit_improve = _calc_improve(self_rag_results['avg_hit_at_k'], vanilla_results['avg_hit_at_k'])
    f1_improve = _calc_improve(self_rag_results['avg_f1'], vanilla_results['avg_f1'])
    gen_em_improve = _calc_improve(self_rag_results['avg_gen_em'], vanilla_results['avg_gen_em'])
    ret_em_improve = _calc_improve(self_rag_results['avg_ret_em'], vanilla_results['avg_ret_em'])
    
    print(f"\n📊 Self-RAG vs Vanilla RAG 提升:")
    print(f"   FActScore: {'+' if factscore_improve >= 0 else ''}{factscore_improve:.2f}%")
    print(f"   Hit@K:     {'+' if hit_improve >= 0 else ''}{hit_improve:.2f}%")
    print(f"   F1 Score:  {'+' if f1_improve >= 0 else ''}{f1_improve:.2f}%")
    print(f"   Gen EM:    {'+' if gen_em_improve >= 0 else ''}{gen_em_improve:.2f}%")
    print(f"   Ret EM:    {'+' if ret_em_improve >= 0 else ''}{ret_em_improve:.2f}%")
    
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
            "model": model
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
        "improvement": {
            "factscore": factscore_improve,
            "hit_at_k": hit_improve,
            "f1": f1_improve,
            "gen_em": gen_em_improve,
            "ret_em": ret_em_improve
        },
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
