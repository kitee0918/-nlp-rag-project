"""
检索精度对答案质量影响分析

基于 rag_hotpotqa.py 扩展，分析不同检索精度对最终答案质量的影响
"""

import os
import sys
import json
import numpy as np
from typing import List, Dict, Tuple
from collections import defaultdict

# 导入 rag_hotpotqa 的模块
from rag_hotpotqa import (
    load_hotpotqa,
    BM25Retriever,
    ContrieverRetriever,
    QwenGenerator,
    RAGSystem,
    compute_f1,
    compute_retrieval_metrics,
    Config,
)

import torch


# ============================================
# 配置
# ============================================

class AnalysisConfig:
    """分析实验配置"""
    # 测试样本数量（调试用）
    SUBSET_SIZE = 10
    
    # Top-K 值列表（自变量）
    TOP_K_LIST = [1, 3, 5]
    
    # 检索器列表
    RETRIEVERS = ["bm25", "contriever"]  # BM25 和 Contriever 对比
    
    # 生成模型
    MODEL = "qwen-plus"
    
    # 输出目录
    OUTPUT_DIR = "./results"


# ============================================
# 数据收集函数
# ============================================

def collect_detailed_results(
    retriever,
    generator,
    examples: List[Dict],
    top_k: int
) -> List[Dict]:
    """
    收集每个样本的详细结果（包含检索指标和答案质量）
    
    Returns:
        每个样本的详细结果列表
    """
    results = []
    
    for example in examples:
        question = example['question']
        ground_truth = example['answer']
        supporting_facts = example['supporting_facts']
        
        # 1. 检索
        retrieved_docs = retriever.retrieve(question, top_k=top_k)
        
        # 2. 生成答案
        gen_result = generator.generate(question, [doc for doc, _ in retrieved_docs])
        predicted_answer = gen_result['answer']
        
        # 3. 计算检索指标
        retrieval_metrics = compute_retrieval_metrics(
            [{'id': doc['id'], 'title': doc.get('title', '')} for doc, _ in retrieved_docs],
            supporting_facts
        )
        
        # 4. 计算答案质量
        answer_f1 = compute_f1(predicted_answer, ground_truth)
        
        # 5. 组装结果
        result = {
            'question_id': example['id'],
            'question': question,
            'ground_truth': ground_truth,
            'predicted_answer': predicted_answer,
            'answer_f1': answer_f1,
            'recall@k': retrieval_metrics['recall@k'],
            'precision@k': retrieval_metrics['precision@k'],
            'f1@k': retrieval_metrics['f1@k'],
            'top_k': top_k,
            'retriever': retriever.get_name(),
            'retrieved_docs': [
                {'id': doc['id'], 'title': doc.get('title', ''), 'score': score}
                for doc, score in retrieved_docs
            ]
        }
        results.append(result)
    
    return results


# ============================================
# 分析函数
# ============================================

def compute_correlation(data: List[Dict]) -> Dict:
    """
    计算检索指标与答案质量的相关性
    
    Args:
        data: 样本结果列表
        
    Returns:
        相关性分析结果
    """
    # 提取数组
    recall_arr = np.array([d['recall@k'] for d in data])
    precision_arr = np.array([d['precision@k'] for d in data])
    f1k_arr = np.array([d['f1@k'] for d in data])
    answer_f1_arr = np.array([d['answer_f1'] for d in data])
    
    # 计算 Pearson 相关系数
    def pearson_correlation(x, y):
        """计算 Pearson 相关系数"""
        if np.std(x) == 0 or np.std(y) == 0:
            return 0.0
        return np.corrcoef(x, y)[0, 1]
    
    correlations = {
        'recall_vs_answer_f1': pearson_correlation(recall_arr, answer_f1_arr),
        'precision_vs_answer_f1': pearson_correlation(precision_arr, answer_f1_arr),
        'f1k_vs_answer_f1': pearson_correlation(f1k_arr, answer_f1_arr),
    }
    
    return correlations


def analyze_by_recall_group(data: List[Dict]) -> Dict:
    """
    按 Recall@K 分组分析答案质量
    
    分组：0-0.3 (低), 0.3-0.6 (中), 0.6-1.0 (高)
    """
    groups = {
        'low (0-0.3)': [],
        'medium (0.3-0.6)': [],
        'high (0.6-1.0)': []
    }
    
    for d in data:
        recall = d['recall@k']
        if recall < 0.3:
            groups['low (0-0.3)'].append(d['answer_f1'])
        elif recall < 0.6:
            groups['medium (0.3-0.6)'].append(d['answer_f1'])
        else:
            groups['high (0.6-1.0)'].append(d['answer_f1'])
    
    # 计算每组的平均 F1
    group_stats = {}
    for group_name, f1_list in groups.items():
        if f1_list:
            group_stats[group_name] = {
                'count': len(f1_list),
                'avg_answer_f1': np.mean(f1_list),
                'std_answer_f1': np.std(f1_list)
            }
        else:
            group_stats[group_name] = {
                'count': 0,
                'avg_answer_f1': 0.0,
                'std_answer_f1': 0.0
            }
    
    return group_stats


# ============================================
# 主实验函数
# ============================================

def run_analysis_experiment(
    subset_size: int = None,
    top_k_list: List[int] = None,
    use_retrievers: List[str] = None,
    model: str = None
) -> Dict:
    """
    运行检索精度影响分析实验
    
    Args:
        subset_size: 测试样本数量
        top_k_list: Top-K 值列表
        use_retrievers: 使用的检索器列表
        model: 生成模型名称
        
    Returns:
        实验结果字典
    """
    # 使用默认配置
    subset_size = subset_size or AnalysisConfig.SUBSET_SIZE
    top_k_list = top_k_list or AnalysisConfig.TOP_K_LIST
    use_retrievers = use_retrievers or AnalysisConfig.RETRIEVERS
    model = model or AnalysisConfig.MODEL
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    print("=" * 70)
    print("📊 检索精度对答案质量影响分析")
    print("=" * 70)
    print(f"测试样本: {subset_size}")
    print(f"Top-K 值: {top_k_list}")
    print(f"检索器: {use_retrievers}")
    print(f"生成模型: {model}")
    print(f"设备: {device}")
    print("=" * 70)
    
    # 1. 加载数据
    print("\n📂 加载数据...")
    examples, corpus = load_hotpotqa(subset_size=subset_size)
    
    # 确保目录存在
    os.makedirs(Config.CACHE_DIR, exist_ok=True)
    os.makedirs(AnalysisConfig.OUTPUT_DIR, exist_ok=True)
    
    # 2. 初始化检索器
    print("\n📦 初始化检索器...")
    retrievers = {}
    
    if "bm25" in use_retrievers:
        bm25_cache = os.path.join(Config.CACHE_DIR, "bm25_index.pkl")
        retrievers["bm25"] = BM25Retriever(corpus, cache_path=bm25_cache)
    
    if "contriever" in use_retrievers:
        contriever_cache = os.path.join(Config.CACHE_DIR, "contriever_embeddings.pkl")
        retrievers["contriever"] = ContrieverRetriever(
            corpus, 
            cache_path=contriever_cache,
            device=device
        )
    
    # 3. 初始化生成器
    print(f"\n🤖 初始化生成器: {model}")
    generator = QwenGenerator(model=model)
    
    # 4. 运行实验
    all_results = {}
    
    for retriever_name, retriever in retrievers.items():
        retriever_results = []
        
        for top_k in top_k_list:
            print(f"\n{'='*70}")
            print(f"🔬 实验配置: {retriever_name.upper()} + Top-{top_k}")
            print(f"{'='*70}")
            
            # 收集详细结果
            detailed_results = collect_detailed_results(
                retriever, generator, examples, top_k
            )
            
            # 汇总统计
            avg_recall = np.mean([d['recall@k'] for d in detailed_results])
            avg_precision = np.mean([d['precision@k'] for d in detailed_results])
            avg_f1k = np.mean([d['f1@k'] for d in detailed_results])
            avg_answer_f1 = np.mean([d['answer_f1'] for d in detailed_results])
            
            print(f"\n📈 汇总统计:")
            print(f"   平均 Recall@{top_k}:    {avg_recall:.4f}")
            print(f"   平均 Precision@{top_k}: {avg_precision:.4f}")
            print(f"   平均 F1@{top_k}:        {avg_f1k:.4f}")
            print(f"   平均 Answer F1:        {avg_answer_f1:.4f}")
            
            # 相关性分析
            correlations = compute_correlation(detailed_results)
            print(f"\n📊 相关性分析:")
            print(f"   Recall vs Answer F1:    {correlations['recall_vs_answer_f1']:.4f}")
            print(f"   Precision vs Answer F1: {correlations['precision_vs_answer_f1']:.4f}")
            print(f"   F1@K vs Answer F1:      {correlations['f1k_vs_answer_f1']:.4f}")
            
            # 分组分析
            group_stats = analyze_by_recall_group(detailed_results)
            print(f"\n📊 分组分析 (按 Recall@K):")
            for group_name, stats in group_stats.items():
                if stats['count'] > 0:
                    print(f"   {group_name}: n={stats['count']}, avg_F1={stats['avg_answer_f1']:.4f} ± {stats['std_answer_f1']:.4f}")
                else:
                    print(f"   {group_name}: n=0")
            
            retriever_results.append({
                'top_k': top_k,
                'detailed_results': detailed_results,
                'summary': {
                    'avg_recall': avg_recall,
                    'avg_precision': avg_precision,
                    'avg_f1k': avg_f1k,
                    'avg_answer_f1': avg_answer_f1
                },
                'correlations': correlations,
                'group_stats': group_stats
            })
        
        all_results[retriever_name] = retriever_results
    
    # 5. 保存结果
    output_file = os.path.join(
        AnalysisConfig.OUTPUT_DIR, 
        f"retrieval_impact_analysis_{subset_size}.json"
    )
    
    # 转换为可序列化格式
    output_data = {}
    for retriever_name, retriever_results in all_results.items():
        output_data[retriever_name] = []
        for result in retriever_results:
            output_data[retriever_name].append({
                'top_k': result['top_k'],
                'summary': result['summary'],
                'correlations': result['correlations'],
                'group_stats': result['group_stats'],
                'detailed_results': result['detailed_results']
            })
    
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(output_data, f, ensure_ascii=False, indent=2)
    
    print(f"\n💾 结果已保存至: {output_file}")
    
    return all_results


# ============================================
# 入口
# ============================================

if __name__ == "__main__":
    results = run_analysis_experiment()
