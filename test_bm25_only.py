"""仅测试 BM25 检索器"""
import os
os.environ['PYTHONIOENCODING'] = 'utf-8'
os.environ['PYTHONUTF8'] = '1'
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'

from rag_hotpotqa import load_hotpotqa, BM25Retriever, QwenGenerator, RAGSystem, evaluate

# 只测试 BM25
print('加载数据...')
examples, corpus = load_hotpotqa(subset_size=5)
print(f'加载完成: {len(examples)} 个问题')

print('初始化 BM25...')
os.makedirs('./cache', exist_ok=True)
bm25 = BM25Retriever(corpus, cache_path='./cache/bm25_index.pkl')

print('初始化生成器...')
generator = QwenGenerator(model='qwen-plus')

print('创建 RAG 系统...')
rag = RAGSystem(bm25, generator, top_k=3)

print('评估中...')
result = evaluate(rag, examples)
print(f"\nBM25 结果:")
print(f"  F1 Score:  {result['f1_score']:.4f}")
print(f"  Recall@K:  {result['recall@k']:.4f}")
print(f"  Precision@K: {result['precision@k']:.4f}")
print(f"  F1@K:      {result['f1@k']:.4f}")
