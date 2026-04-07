"""
原生 RAG 系统实现 - 基于 HotpotQA 数据集
支持 BM25 和 Contriever 两种检索方法
支持通过阿里百炼 API 调用 Qwen 模型
"""

import os
import sys

# 设置环境变量确保 UTF-8 编码
os.environ['PYTHONIOENCODING'] = 'utf-8'
os.environ['PYTHONUTF8'] = '1'

# 强制设置 stdout/stderr 编码为 UTF-8（适用于所有平台）
def _setup_utf8_output():
    """设置 UTF-8 输出编码"""
    if hasattr(sys.stdout, 'reconfigure'):
        try:
            sys.stdout.reconfigure(encoding='utf-8', errors='replace')
        except Exception:
            pass
    if hasattr(sys.stderr, 'reconfigure'):
        try:
            sys.stderr.reconfigure(encoding='utf-8', errors='replace')
        except Exception:
            pass

_setup_utf8_output()

import json
import pickle
import argparse
import traceback
from typing import List, Dict, Tuple, Optional
from tqdm import tqdm

# 数据加载
import os

# 尝试从 .env 文件加载环境变量
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass  # python-dotenv 未安装，跳过

# 设置 HuggingFace 镜像（如果需要）
# 取消下面这行的注释以使用镜像
# os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'

from datasets import load_dataset

# 检索相关
import numpy as np
import torch
from rank_bm25 import BM25Okapi
from transformers import AutoTokenizer, AutoModel

# 生成相关
from openai import OpenAI
import httpx

# ============================================
# 配置部分
# ============================================

class Config:
    """全局配置"""
    # 数据集
    DATASET_NAME = "hotpotqa/hotpot_qa"
    DATASET_SPLIT = "validation"  # 使用 validation 分割（dev 集）
    SUBSET_SIZE = 100  # 测试样本数量
    
    # 检索配置
    RETRIEVAL_K = 3  # 每次检索返回 top-k 文档
    
    # 模型配置
    CONTRIEVER_MODEL = "facebook/contriever-msmarco"
    
    # 阿里百炼 API 配置
    # 请在此处填写你的 API Key，或通过环境变量设置
    DASHSCOPE_API_KEY = os.environ.get("DASHSCOPE_API_KEY", "")
    DASHSCOPE_BASE_URL = "https://dashscope.aliyuncs.com/compatible-mode/v1"
    
    # 模型名称
    QWEN_MODEL = "qwen-plus"  # 或 qwen-turbo, qwen-max 等
    
    # 缓存路径
    CACHE_DIR = "./cache"
    RESULTS_DIR = "./results"


# ============================================
# 数据加载模块
# ============================================

def load_hotpotqa(subset_size: Optional[int] = None) -> Tuple[List[Dict], List[Dict]]:
    """
    加载 HotpotQA dev 数据集
    
    Returns:
        examples: QA 示例列表
        corpus: 文档语料库
    """
    print(f"正在加载 HotpotQA 数据集...")
    
    # 设置镜像站点（如果需要）
    if 'HF_ENDPOINT' not in os.environ:
        os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'
        print(f"使用镜像站点: {os.environ['HF_ENDPOINT']}")
    
    # 从 HuggingFace 加载
    dataset = load_dataset(Config.DATASET_NAME, "distractor", split=Config.DATASET_SPLIT)
    
    examples = []
    corpus = []
    seen_docs = set()
    
    for item in dataset:
        # 构建示例
        # HuggingFace 版本的字段格式
        # context: {'title': [...], 'sentences': [[...], ...]}
        # supporting_facts: {'title': [...], 'sent_id': [...]}
        
        # 提取该问题的候选文档（HotpotQA distractor 设置：每个问题10个候选文档）
        titles = item["context"]["title"]
        sentences_list = item["context"]["sentences"]
        candidate_docs = []
        for title, sentences in zip(titles, sentences_list):
            doc_text = " ".join(sentences)
            candidate_docs.append({
                "id": title,
                "title": title,
                "text": doc_text,
            })
        
        example = {
            "id": item["id"],
            "question": item["question"],
            "answer": item["answer"],
            "type": item["type"],
            "level": item["level"],
            "supporting_facts": list(zip(
                item["supporting_facts"]["title"],
                item["supporting_facts"]["sent_id"]
            )),  # 转换为 [(title, sent_id), ...] 格式
            "candidate_docs": candidate_docs,  # 每个问题的10个候选文档
        }
        examples.append(example)
        
        # 构建语料库（从 context 中提取文档）
        # context 格式: {'title': [title1, title2, ...], 'sentences': [[sent1, sent2, ...], ...]}
        titles = item["context"]["title"]
        sentences_list = item["context"]["sentences"]
        
        for title, sentences in zip(titles, sentences_list):
            doc_id = title
            if doc_id not in seen_docs:
                doc_text = " ".join(sentences)
                corpus.append({
                    "id": doc_id,
                    "title": title,
                    "text": doc_text,
                    "metadata": {
                        "level": item["level"],
                        "type": item["type"]
                    }
                })
                seen_docs.add(doc_id)
    
    # 限制样本数量
    if subset_size and subset_size < len(examples):
        examples = examples[:subset_size]
    
    print(f"加载完成: {len(examples)} 个问题, {len(corpus)} 篇文档")
    return examples, corpus


# ============================================
# BM25 检索器
# ============================================

class BM25Retriever:
    """BM25 稀疏检索器"""
    
    def __init__(self, documents: List[Dict], cache_path: Optional[str] = None):
        """
        初始化 BM25 检索器
        
        Args:
            documents: 文档列表，每个文档包含 'id', 'text' 等字段
            cache_path: 索引缓存路径
        """
        self.documents = documents
        self.doc_ids = [doc['id'] for doc in documents]
        self.corpus = [doc['text'] for doc in documents]
        
        # 尝试加载缓存
        if cache_path and os.path.exists(cache_path):
            print(f"从缓存加载 BM25 索引: {cache_path}")
            with open(cache_path, 'rb') as f:
                data = pickle.load(f)
                self.bm25 = data['bm25']
        else:
            print("构建 BM25 索引...")
            # 简单分词
            self.tokenized_corpus = [doc.lower().split() for doc in tqdm(self.corpus, ascii=False)]
            self.bm25 = BM25Okapi(self.tokenized_corpus)
            
            # 保存缓存
            if cache_path:
                os.makedirs(os.path.dirname(cache_path), exist_ok=True)
                with open(cache_path, 'wb') as f:
                    pickle.dump({'bm25': self.bm25}, f)
    
    def retrieve(self, query: str, top_k: int = 3) -> List[Tuple[Dict, float]]:
        """
        检索 top-k 相关文档
        
        Args:
            query: 查询字符串
            top_k: 返回文档数量
            
        Returns:
            List of (document, score) tuples
        """
        tokenized_query = query.lower().split()
        scores = self.bm25.get_scores(tokenized_query)
        
        # 获取 top-k 索引
        top_indices = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:top_k]
        
        results = []
        for idx in top_indices:
            results.append((self.documents[idx], float(scores[idx])))
        
        return results
    
    def get_name(self) -> str:
        return "BM25"


# ============================================
# Contriever 检索器
# ============================================

class ContrieverRetriever:
    """Contriever 稠密检索器"""
    
    def __init__(
        self, 
        documents: List[Dict], 
        model_name: str = "facebook/contriever-msmarco",
        cache_path: Optional[str] = None,
        device: str = "cuda"
    ):
        """
        初始化 Contriever 检索器
        
        Args:
            documents: 文档列表
            model_name: HuggingFace 模型名称
            cache_path: 嵌入缓存路径
            device: 运行设备
        """
        self.documents = documents
        self.doc_ids = [doc['id'] for doc in documents]
        self.device = device
        self.model_name = model_name
        
        # 加载模型
        print(f"加载 Contriever 模型: {model_name}")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name)
        self.model.to(device)
        self.model.eval()
        
        # 尝试加载缓存
        if cache_path and os.path.exists(cache_path):
            print(f"从缓存加载嵌入: {cache_path}")
            with open(cache_path, 'rb') as f:
                data = pickle.load(f)
                self.embeddings = data['embeddings']
        else:
            # 构建嵌入
            self.embeddings = self._encode_documents()
            
            # 保存缓存
            if cache_path:
                os.makedirs(os.path.dirname(cache_path), exist_ok=True)
                with open(cache_path, 'wb') as f:
                    pickle.dump({'embeddings': self.embeddings}, f)
    
    def _mean_pooling(self, token_embeddings, attention_mask):
        """平均池化获取句子嵌入"""
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
        return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(
            input_mask_expanded.sum(1), min=1e-9
        )
    
    @torch.no_grad()
    def _encode_texts(self, texts: List[str], batch_size: int = 32) -> np.ndarray:
        """编码文本为嵌入向量"""
        all_embeddings = []
        
        for i in tqdm(range(0, len(texts), batch_size), desc="编码文档", ascii=False):
            batch_texts = texts[i:i + batch_size]
            
            # Tokenize
            encoded = self.tokenizer(
                batch_texts,
                padding=True,
                truncation=True,
                max_length=512,
                return_tensors="pt"
            ).to(self.device)
            
            # 获取嵌入
            outputs = self.model(**encoded)
            embeddings = self._mean_pooling(outputs.last_hidden_state, encoded['attention_mask'])
            
            # 归一化
            embeddings = torch.nn.functional.normalize(embeddings, p=2, dim=1)
            
            all_embeddings.append(embeddings.cpu().numpy())
        
        return np.vstack(all_embeddings)
    
    def _encode_documents(self) -> np.ndarray:
        """编码所有文档"""
        texts = [doc['text'] for doc in self.documents]
        return self._encode_texts(texts)
    
    @torch.no_grad()
    def retrieve(self, query: str, top_k: int = 3) -> List[Tuple[Dict, float]]:
        """
        检索 top-k 相关文档
        
        Args:
            query: 查询字符串
            top_k: 返回文档数量
            
        Returns:
            List of (document, score) tuples
        """
        # 编码查询
        query_embedding = self._encode_texts([query])
        
        # 计算相似度
        similarities = np.dot(self.embeddings, query_embedding.T).flatten()
        
        # 获取 top-k 索引
        top_indices = np.argsort(similarities)[::-1][:top_k]
        
        results = []
        for idx in top_indices:
            results.append((self.documents[idx], float(similarities[idx])))
        
        return results
    
    def get_name(self) -> str:
        return "Contriever"


# ============================================
# LLM 生成器（阿里百炼 API）
# ============================================

class QwenGenerator:
    """通过阿里百炼 API 调用 Qwen 模型"""
    
    def __init__(self, api_key: str = None, model: str = "qwen-plus"):
        """
        初始化生成器
        
        Args:
            api_key: 阿里百炼 API Key
            model: 模型名称
        """
        self.api_key = api_key or Config.DASHSCOPE_API_KEY
        self.model = model
        
        if not self.api_key:
            raise ValueError("请设置 DASHSCOPE_API_KEY 环境变量或在 Config 中配置")
        
        # 配置 httpx 客户端以支持 UTF-8 编码
        http_client = httpx.Client(
            timeout=httpx.Timeout(60.0, connect=10.0),
        )
        
        # 初始化 OpenAI 客户端（兼容阿里百炼 API）
        self.client = OpenAI(
            api_key=self.api_key,
            base_url=Config.DASHSCOPE_BASE_URL,
            http_client=http_client
        )
        
        print(f"初始化 Qwen 生成器: {model}")
    
    def format_context(self, documents: List[Dict]) -> str:
        """格式化检索到的文档"""
        context_parts = []
        for i, doc in enumerate(documents, 1):
            title = doc.get('title', 'Unknown')
            context_parts.append(f"[文档 {i}] {title}\n{doc['text']}\n")
        return "\n".join(context_parts)
    
    def generate(self, question: str, documents: List[Dict]) -> Dict:
        """
        根据问题和检索文档生成答案
        
        Args:
            question: 问题
            documents: 检索到的文档列表
            
        Returns:
            生成结果字典
        """
        context = self.format_context(documents)
        
        prompt = f"""你是一个智能问答助手。请根据以下提供的文档信息回答问题。
如果文档中没有足够的信息回答问题，请说"根据提供的文档，我无法回答这个问题"。

文档信息:
{context}

问题: {question}

请给出简洁准确的回答:"""
        
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": "你是一个智能问答助手，擅长根据提供的文档信息回答问题。"},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.7,
                max_tokens=512
            )
            
            answer = response.choices[0].message.content
            
            return {
                "answer": answer,
                "prompt": prompt,
                "model": self.model
            }
            
        except Exception as e:
            # 安全打印错误信息（处理 Windows 控制台编码问题）
            error_msg = str(e)
            error_type = type(e).__name__
            
            # 打印详细错误信息和堆栈（仅第一次错误时打印完整堆栈）
            try:
                print(f"\n生成错误 [{error_type}]: {error_msg}")
                # 打印完整堆栈以定位问题
                tb_lines = traceback.format_exc().split('\n')
                for line in tb_lines[-5:]:  # 只打印最后5行
                    if line.strip():
                        print(f"  {line}")
            except UnicodeEncodeError:
                safe_msg = error_msg.encode('ascii', errors='replace').decode('ascii')
                print(f"\n生成错误 [{error_type}]: {safe_msg}")
            
            return {
                "answer": "",  # 返回空字符串而非错误信息，避免影响 F1 计算
                "prompt": prompt,
                "model": self.model,
                "error": error_msg,
                "error_type": error_type
            }
    
    def get_name(self) -> str:
        return f"Qwen-{self.model}"


# ============================================
# RAG 系统
# ============================================

class RAGSystem:
    """完整的 RAG 系统"""
    
    def __init__(
        self,
        retriever,
        generator,
        top_k: int = 3
    ):
        """
        初始化 RAG 系统
        
        Args:
            retriever: 检索器实例
            generator: 生成器实例
            top_k: 检索文档数量
        """
        self.retriever = retriever
        self.generator = generator
        self.top_k = top_k
    
    def answer(self, question: str) -> Dict:
        """
        回答单个问题
        
        Args:
            question: 问题字符串
            
        Returns:
            包含答案、检索文档等信息的字典
        """
        # 1. 检索
        retrieved_docs = self.retriever.retrieve(question, top_k=self.top_k)
        
        # 2. 生成
        result = self.generator.generate(question, [doc for doc, _ in retrieved_docs])
        
        # 3. 组装结果
        result.update({
            "question": question,
            "retrieved_documents": [
                {"id": doc['id'], "title": doc.get('title', ''), "score": score}
                for doc, score in retrieved_docs
            ],
            "retriever": self.retriever.get_name(),
            "generator": self.generator.get_name()
        })
        
        return result
    
    def batch_answer(self, questions: List[str]) -> List[Dict]:
        """批量回答问题"""
        results = []
        for question in tqdm(questions, desc="处理问题", ascii=False):
            result = self.answer(question)
            results.append(result)
        return results


# ============================================
# 评估指标
# ============================================

# def compute_exact_match(prediction: str, ground_truth: str) -> bool:
#     """计算精确匹配 - 对于 finance/law 领域过于严苛，已弃用"""
#     pred_normalized = prediction.lower().strip()
#     truth_normalized = ground_truth.lower().strip()
#     return pred_normalized == truth_normalized


def compute_f1(prediction: str, ground_truth: str) -> float:
    """计算 F1 分数（词级别）- 用于生成答案评估"""
    pred_tokens = set(prediction.lower().split())
    truth_tokens = set(ground_truth.lower().split())
    
    if not pred_tokens or not truth_tokens:
        return 0.0
    
    common = pred_tokens & truth_tokens
    if not common:
        return 0.0
    
    precision = len(common) / len(pred_tokens)
    recall = len(common) / len(truth_tokens)
    
    f1 = 2 * precision * recall / (precision + recall)
    return f1


def compute_retrieval_metrics(retrieved_docs: List[Dict], supporting_facts: List[List]) -> Dict[str, float]:
    """
    计算检索指标：Recall@K, Precision@K, F1@K
    
    适用于 finance/law 等领域，这些领域对检索质量更关注
    
    Args:
        retrieved_docs: 检索到的文档列表
        supporting_facts: 支持事实列表 [[title, sent_id], ...]
        
    Returns:
        包含 recall@k, precision@k, f1@k 的字典
    """
    # 提取支持文档的标题
    supporting_titles = set(fact[0] for fact in supporting_facts)
    retrieved_titles = set(doc['id'] for doc in retrieved_docs)
    
    # 计算交集
    hit_titles = supporting_titles & retrieved_titles
    
    # Recall@K: 命中的支持文档数 / 总支持文档数
    recall = len(hit_titles) / len(supporting_titles) if supporting_titles else 0.0
    
    # Precision@K: 命中的文档数 / 检索的文档数
    precision = len(hit_titles) / len(retrieved_titles) if retrieved_titles else 0.0
    
    # F1@K: Recall 和 Precision 的调和平均
    if recall + precision > 0:
        f1 = 2 * recall * precision / (recall + precision)
    else:
        f1 = 0.0
    
    return {
        'recall@k': recall,
        'precision@k': precision,
        'f1@k': f1
    }


def evaluate(rag_system: RAGSystem, examples: List[Dict]) -> Dict:
    """
    评估 RAG 系统
    
    使用检索指标（Recall@K, Precision@K, F1@K）评估，
    更适合 finance/law 等领域
    
    Args:
        rag_system: RAG 系统实例
        examples: 测试示例列表
        
    Returns:
        评估结果字典
    """
    print(f"\n开始评估 {rag_system.retriever.get_name()} + {rag_system.generator.get_name()}...")
    
    f1_scores = []
    recall_scores = []
    precision_scores = []
    f1_at_k_scores = []
    
    for example in tqdm(examples, desc="评估中", ascii=False):
        result = rag_system.answer(example['question'])
        
        # 计算生成答案的 F1 分数
        f1 = compute_f1(result['answer'], example['answer'])
        f1_scores.append(f1)
        
        # 计算检索指标
        retrieval_metrics = compute_retrieval_metrics(
            result['retrieved_documents'], 
            example['supporting_facts']
        )
        recall_scores.append(retrieval_metrics['recall@k'])
        precision_scores.append(retrieval_metrics['precision@k'])
        f1_at_k_scores.append(retrieval_metrics['f1@k'])
    
    return {
        "f1_score": sum(f1_scores) / len(f1_scores),
        "recall@k": sum(recall_scores) / len(recall_scores),
        "precision@k": sum(precision_scores) / len(precision_scores),
        "f1@k": sum(f1_at_k_scores) / len(f1_at_k_scores),
        "num_examples": len(examples),
        "retriever": rag_system.retriever.get_name(),
        "generator": rag_system.generator.get_name()
    }


# ============================================
# 主函数
# ============================================

def main():
    parser = argparse.ArgumentParser(description="RAG 系统实验")
    parser.add_argument("--retriever", type=str, default="bm25", choices=["bm25", "contriever", "both"],
                        help="检索方法选择")
    parser.add_argument("--subset", type=int, default=Config.SUBSET_SIZE,
                        help="测试样本数量")
    parser.add_argument("--top_k", type=int, default=Config.RETRIEVAL_K,
                        help="检索文档数量")
    parser.add_argument("--model", type=str, default="qwen-plus",
                        help="Qwen 模型名称")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu",
                        help="运行设备")
    
    # 使用 parse_known_args() 以兼容 Jupyter/Colab 环境
    args, _ = parser.parse_known_args()
    
    print("=" * 60)
    print("RAG 系统实验")
    print("=" * 60)
    print(f"检索方法: {args.retriever}")
    print(f"测试样本: {args.subset}")
    print(f"Top-K: {args.top_k}")
    print(f"生成模型: {args.model}")
    print(f"设备: {args.device}")
    print("=" * 60)
    
    # 1. 加载数据
    examples, corpus = load_hotpotqa(subset_size=args.subset)
    
    # 确保缓存目录存在
    os.makedirs(Config.CACHE_DIR, exist_ok=True)
    os.makedirs(Config.RESULTS_DIR, exist_ok=True)
    
    # 2. 初始化检索器
    retrievers = {}
    
    if args.retriever in ["bm25", "both"]:
        bm25_cache = os.path.join(Config.CACHE_DIR, "bm25_index.pkl")
        retrievers["bm25"] = BM25Retriever(corpus, cache_path=bm25_cache)
    
    if args.retriever in ["contriever", "both"]:
        contriever_cache = os.path.join(Config.CACHE_DIR, "contriever_embeddings.pkl")
        retrievers["contriever"] = ContrieverRetriever(
            corpus, 
            cache_path=contriever_cache,
            device=args.device
        )
    
    # 3. 初始化生成器
    generator = QwenGenerator(model=args.model)
    
    # 4. 运行实验
    results = []
    for retriever_name, retriever in retrievers.items():
        print(f"\n{'='*60}")
        print(f"测试配置: {retriever_name} + {args.model}")
        print(f"{'='*60}")
        
        # 创建 RAG 系统
        rag = RAGSystem(retriever, generator, top_k=args.top_k)
        
        # 评估
        eval_result = evaluate(rag, examples)
        results.append(eval_result)
        
        # 打印结果
        print(f"\n评估结果:")
        print(f"  F1 Score:    {eval_result['f1_score']:.4f}")
        print(f"  Recall@K:    {eval_result['recall@k']:.4f}")
        print(f"  Precision@K: {eval_result['precision@k']:.4f}")
        print(f"  F1@K:        {eval_result['f1@k']:.4f}")
    
    # 5. 保存结果
    result_file = os.path.join(
        Config.RESULTS_DIR, 
        f"results_{args.retriever}_{args.model}_{args.subset}.json"
    )
    with open(result_file, 'w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    print(f"\n结果已保存至: {result_file}")
    
    # 6. 对比展示
    if len(results) > 1:
        print("\n" + "=" * 60)
        print("检索方法对比:")
        print("=" * 80)
        print(f"{'检索器':<12} {'F1 Score':<12} {'Recall@K':<12} {'Precision@K':<12} {'F1@K':<12}")
        print("-" * 80)
        for r in results:
            print(f"{r['retriever']:<12} {r['f1_score']:<12.4f} {r['recall@k']:<12.4f} {r['precision@k']:<12.4f} {r['f1@k']:<12.4f}")


if __name__ == "__main__":
    main()
