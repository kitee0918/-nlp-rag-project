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

# 可视化
import matplotlib.pyplot as plt
import matplotlib
matplotlib.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans', 'Arial Unicode MS']
matplotlib.rcParams['axes.unicode_minus'] = False


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
    RETRIEVERS = ["bm25"]  # 当前环境 Contriever 有兼容问题，先用 BM25
    
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
            
            # 可视化
            print(f"\n📊 生成可视化图表...")
            plot_recall_vs_answer_f1(detailed_results, retriever_name, top_k, AnalysisConfig.OUTPUT_DIR)
            plot_group_analysis(group_stats, retriever_name, top_k, AnalysisConfig.OUTPUT_DIR)
            
            # 案例分析
            cases = analyze_cases(detailed_results, retriever_name, top_k)
            print_case_analysis(cases, retriever_name, top_k)
            
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
                'group_stats': group_stats,
                'cases_summary': {k: len(v) for k, v in cases.items()}
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
    
    # 6. 生成对比可视化
    print(f"\n📊 生成检索器对比图表...")
    plot_retriever_comparison(all_results, AnalysisConfig.OUTPUT_DIR)
    
    # 7. 总结报告
    print("\n" + "="*70)
    print("📋 分析总结")
    print("="*70)
    for retriever_name, retriever_results in all_results.items():
        print(f"\n【{retriever_name.upper()}】")
        for result in retriever_results:
            top_k = result['top_k']
            summary = result['summary']
            corr = result['correlations']
            print(f"  Top-{top_k}: Recall={summary['avg_recall']:.2f}, "
                  f"Precision={summary['avg_precision']:.2f}, "
                  f"Answer F1={summary['avg_answer_f1']:.2f}, "
                  f"相关性={corr['recall_vs_answer_f1']:.3f}")
    
    return all_results


# ============================================
# 可视化函数
# ============================================

def plot_recall_vs_answer_f1(data: List[Dict], retriever_name: str, top_k: int, output_dir: str):
    """
    绘制 Recall@K vs Answer F1 散点图
    """
    recall_arr = [d['recall@k'] for d in data]
    answer_f1_arr = [d['answer_f1'] for d in data]
    
    plt.figure(figsize=(8, 6))
    plt.scatter(recall_arr, answer_f1_arr, alpha=0.6, s=100, c='steelblue', edgecolors='white')
    
    # 添加趋势线
    if len(recall_arr) > 1:
        z = np.polyfit(recall_arr, answer_f1_arr, 1)
        p = np.poly1d(z)
        x_line = np.linspace(0, 1, 100)
        plt.plot(x_line, p(x_line), 'r--', alpha=0.8, label=f'趋势线 (斜率={z[0]:.2f})')
    
    plt.xlabel('Recall@K', fontsize=12)
    plt.ylabel('Answer F1', fontsize=12)
    plt.title(f'{retriever_name} - Top-{top_k}: Recall vs Answer Quality', fontsize=14)
    plt.xlim(-0.05, 1.05)
    plt.ylim(-0.05, 1.05)
    plt.grid(True, alpha=0.3)
    plt.legend()
    
    # 计算相关系数
    corr = np.corrcoef(recall_arr, answer_f1_arr)[0, 1] if len(recall_arr) > 1 else 0
    plt.text(0.05, 0.95, f'Pearson r = {corr:.3f}', transform=plt.gca().transAxes, fontsize=11)
    
    output_path = os.path.join(output_dir, f'scatter_{retriever_name}_top{top_k}.png')
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"   散点图已保存: {output_path}")


def plot_retriever_comparison(results: Dict, output_dir: str):
    """
    绘制不同检索器的对比柱状图
    """
    retrievers = list(results.keys())
    metrics = ['avg_recall', 'avg_precision', 'avg_answer_f1']
    metric_labels = ['Recall@K', 'Precision@K', 'Answer F1']
    
    # 收集数据
    all_top_k = set()
    for r in retrievers:
        for item in results[r]:
            all_top_k.add(item['top_k'])
    top_k_list = sorted(all_top_k)
    
    x = np.arange(len(top_k_list))
    width = 0.35
    
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    for idx, (metric, label) in enumerate(zip(metrics, metric_labels)):
        ax = axes[idx]
        
        for i, retriever in enumerate(retrievers):
            values = []
            for tk in top_k_list:
                found = False
                for item in results[retriever]:
                    if item['top_k'] == tk:
                        if metric == 'avg_answer_f1':
                            values.append(item['summary']['avg_answer_f1'])
                        else:
                            values.append(item['summary'][metric])
                        found = True
                        break
                if not found:
                    values.append(0)
            
            bars = ax.bar(x + i * width, values, width, label=retriever.upper(), alpha=0.8)
        
        ax.set_xlabel('Top-K', fontsize=11)
        ax.set_ylabel(label, fontsize=11)
        ax.set_title(f'{label} 对比', fontsize=12)
        ax.set_xticks(x + width / 2)
        ax.set_xticklabels([f'Top-{k}' for k in top_k_list])
        ax.legend()
        ax.set_ylim(0, 1.1)
        ax.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    output_path = os.path.join(output_dir, 'retriever_comparison.png')
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"\n📊 对比柱状图已保存: {output_path}")


def plot_group_analysis(group_stats: Dict, retriever_name: str, top_k: int, output_dir: str):
    """
    绘制按 Recall 分组的答案质量柱状图
    """
    groups = list(group_stats.keys())
    avg_f1 = [group_stats[g]['avg_answer_f1'] for g in groups]
    counts = [group_stats[g]['count'] for g in groups]
    
    fig, ax1 = plt.subplots(figsize=(10, 5))
    
    x = np.arange(len(groups))
    bars = ax1.bar(x, avg_f1, alpha=0.7, color='steelblue', label='Avg Answer F1')
    ax1.set_xlabel('Recall Group', fontsize=11)
    ax1.set_ylabel('Average Answer F1', fontsize=11, color='steelblue')
    ax1.set_ylim(0, 1.0)
    ax1.tick_params(axis='y', labelcolor='steelblue')
    
    # 添加样本数标签
    for bar, count in zip(bars, counts):
        height = bar.get_height()
        ax1.annotate(f'n={count}',
                    xy=(bar.get_x() + bar.get_width() / 2, height),
                    xytext=(0, 3),
                    textcoords="offset points",
                    ha='center', va='bottom', fontsize=10)
    
    ax1.set_xticks(x)
    ax1.set_xticklabels(groups, rotation=15)
    ax1.set_title(f'{retriever_name} - Top-{top_k}: 分组分析', fontsize=13)
    ax1.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    output_path = os.path.join(output_dir, f'group_{retriever_name}_top{top_k}.png')
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"   分组柱状图已保存: {output_path}")


# ============================================
# 案例分析函数
# ============================================

def analyze_cases(data: List[Dict], retriever_name: str, top_k: int) -> Dict:
    """
    分析典型案例：
    1. 检索好但答案差（应该答对但没答对）
    2. 检索差但答案好（侥幸答对）
    """
    cases = {
        'good_retrieval_bad_answer': [],
        'bad_retrieval_good_answer': [],
        'good_both': [],
        'bad_both': []
    }
    
    for d in data:
        recall = d['recall@k']
        answer_f1 = d['answer_f1']
        
        # 定义阈值
        good_retrieval = recall >= 0.5
        good_answer = answer_f1 >= 0.3
        
        if good_retrieval and not good_answer:
            cases['good_retrieval_bad_answer'].append(d)
        elif not good_retrieval and good_answer:
            cases['bad_retrieval_good_answer'].append(d)
        elif good_retrieval and good_answer:
            cases['good_both'].append(d)
        else:
            cases['bad_both'].append(d)
    
    return cases


def print_case_analysis(cases: Dict, retriever_name: str, top_k: int):
    """
    打印案例分析结果
    """
    print(f"\n{'='*70}")
    print(f"📋 案例分析: {retriever_name} - Top-{top_k}")
    print(f"{'='*70}")
    
    print(f"\n{'类别':<30} {'数量':<10}")
    print("-" * 40)
    for case_type, case_list in cases.items():
        print(f"{case_type:<30} {len(case_list):<10}")
    
    # 打印典型案例详情
    def print_case_detail(case, index):
        print(f"\n  案例 {index + 1}:")
        print(f"    问题: {case['question'][:60]}...")
        print(f"    标准答案: {case['ground_truth']}")
        print(f"    预测答案: {case['predicted_answer'][:80]}...")
        print(f"    Recall@K: {case['recall@k']:.2f}, Answer F1: {case['answer_f1']:.2f}")
        print(f"    检索文档: {[doc['id'] for doc in case['retrieved_docs'][:3]]}")
    
    # 检索好但答案差
    if cases['good_retrieval_bad_answer']:
        print(f"\n🔍 '检索好但答案差' 典型案例 (前2个):")
        for i, case in enumerate(cases['good_retrieval_bad_answer'][:2]):
            print_case_detail(case, i)
    
    # 检索差但答案好
    if cases['bad_retrieval_good_answer']:
        print(f"\n🔍 '检索差但答案好' 典型案例 (前2个):")
        for i, case in enumerate(cases['bad_retrieval_good_answer'][:2]):
            print_case_detail(case, i)


# ============================================
# 入口
# ============================================

if __name__ == "__main__":
    results = run_analysis_experiment()
