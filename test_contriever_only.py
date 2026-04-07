"""测试 Contriever 检索器"""
import os
os.environ['PYTHONIOENCODING'] = 'utf-8'
os.environ['PYTHONUTF8'] = '1'
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'

from rag_hotpotqa import load_hotpotqa, ContrieverRetriever, QwenGenerator, RAGSystem, evaluate

print('加载数据...')
examples, corpus = load_hotpotqa(subset_size=5)
print(f'加载完成: {len(examples)} 个问题')

print('初始化 Contriever...')
os.makedirs('./cache', exist_ok=True)
contriever = ContrieverRetriever(
    corpus, 
    cache_path='./cache/contriever_embeddings.pkl',
    device='cpu'
)

print('初始化生成器...')
generator = QwenGenerator(model='qwen-plus')

print('创建 RAG 系统...')
rag = RAGSystem(contriever, generator, top_k=3)

print('评估中...')
result = evaluate(rag, examples)
print(f"\nContriever 结果:")
print(f"  F1 Score:    {result['f1_score']:.4f}")
print(f"  Recall@K:    {result['recall@k']:.4f}")
print(f"  Precision@K: {result['precision@k']:.4f}")
print(f"  F1@K:        {result['f1@k']:.4f}")
