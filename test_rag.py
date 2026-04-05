"""测试 RAG 系统"""

import os
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'

# 加载 .env 文件
from dotenv import load_dotenv
load_dotenv()
print(f"API Key 已加载: {'DASHSCOPE_API_KEY' in os.environ}")

from rag_hotpotqa import load_hotpotqa, BM25Retriever, Config, RAGSystem, QwenGenerator

# 测试数据加载
print('测试数据加载...')
examples, corpus = load_hotpotqa(subset_size=10)
print()

# 测试 BM25 检索 - 使用第一个问题的候选文档
print('测试 BM25 检索 (使用问题自带的10个候选文档)...')
test_example = examples[0]
retriever = BM25Retriever(test_example['candidate_docs'])  # 使用该问题的10个候选文档

# 测试检索
query = test_example['question']
print(f'查询: {query}')
print(f'候选文档: {[d["title"] for d in test_example["candidate_docs"]]}')
results = retriever.retrieve(query, top_k=3)
print('检索结果:')
for doc, score in results:
    text_preview = doc['text'][:80] if len(doc['text']) > 80 else doc['text']
    print(f'  - [{score:.4f}] {doc["title"]}: {text_preview}...')

print()
print('[SUCCESS] BM25 检索测试成功!')

# 测试完整 RAG 流程
print()
print('='*60)
print('测试完整 RAG 流程 (BM25 + Qwen)...')
print('='*60)

# 初始化生成器
try:
    generator = QwenGenerator(model='qwen-plus')
    print('[OK] Qwen 生成器初始化成功')
except ValueError as e:
    print(f'[FAIL] 请先配置 DASHSCOPE_API_KEY 环境变量: {e}')
    exit(1)

# 创建 RAG 系统
rag = RAGSystem(retriever, generator, top_k=3)

# 测试单个问题
print(f'测试问题: {test_example["question"]}')
print(f'真实答案: {test_example["answer"]}')
print(f'支持事实: {test_example["supporting_facts"]}')

result = rag.answer(test_example['question'])
print(f'生成答案: {result["answer"]}')

print()
print('='*60)
print('测试完成!')
