"""
原生 RAG 系统实现 - 基于 FinanceBench 数据集
支持 BM25 和 Contriever 两种检索方法
支持通过阿里百炼 API 调用 Qwen 模型

FinanceBench 数据格式说明:
- 数据集: PatronusAI/financebench (150 条金融问答)
- 字段: financebench_id, company, doc_name, question, answer, evidence, doc_type, doc_link
- 特点: 单文档检索（无多跳结构），证据来源为财务报告（10-K/10-Q）
- 语料库: 从 evidence 字段构建，每个问题的证据文本作为文档
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

import gc
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

# Colab 环境检测
IN_COLAB = False
try:
    import google.colab
    IN_COLAB = True
except ImportError:
    pass

# ============================================
# 配置部分
# ============================================

class Config:
    """全局配置"""
    # 数据集
    DATASET_NAME = "PatronusAI/financebench"  # FinanceBench 数据集
    DATASET_SPLIT = "train"  # FinanceBench 只有 train split (150条)
    SUBSET_SIZE = 50  # 测试样本数量（默认50条）
    
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
    
    # Llama 模型配置 (Colab 本地运行)
    LLAMA_MODEL_PATH = "/content/models/llama-3.1-8b"  # Colab 默认路径
    # 本地环境路径（如果本地运行）
    # LLAMA_MODEL_PATH = "./models/llama-3.1-8b"
    
    # 缓存路径
    CACHE_DIR = "./cache"
    RESULTS_DIR = "./results"


# ============================================
# GPU 显存清理工具
# ============================================

def release_contriever_gpu(retriever=None):
    """释放 Contriever 相关对象并清理 CUDA 显存（便于后续加载 Qwen 本地模型）。"""
    if retriever is not None:
        for name in ["model", "tokenizer", "embeddings"]:
            if hasattr(retriever, name):
                try:
                    delattr(retriever, name)
                except Exception:
                    pass

    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        if hasattr(torch.cuda, "ipc_collect"):
            torch.cuda.ipc_collect()


# ============================================
# 数据加载模块
# ============================================


def load_financebench(subset_size: Optional[int] = None) -> Tuple[List[Dict], List[Dict]]:
    """
    加载 FinanceBench 数据集
    
    FinanceBench 数据格式:
    - financebench_id: 唯一标识符
    - company: 公司名称
    - doc_name: 文档名称 (如 "3M_2018_10K")
    - question: 问题文本
    - answer: 答案
    - evidence: 证据列表，包含 evidence_text, doc_name, evidence_page_num 等
    - doc_type: 文档类型 (10k/10q)
    - doc_link: 文档链接
    
    与 HotpotQA 的差异:
    - 无多跳结构（每个问题只有一个证据文档）
    - 无干扰项（开放域检索场景）
    - 语料库从 evidence 字段构建
    
    Returns:
        examples: QA 示例列表
        corpus: 文档语料库
    """
    print(f"正在加载 FinanceBench 数据集...")
    
    # 设置镜像站点（如果需要）
    if 'HF_ENDPOINT' not in os.environ:
        os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'
        print(f"使用镜像站点: {os.environ['HF_ENDPOINT']}")
    
    try:
        # 从 HuggingFace 加载 FinanceBench
        dataset = load_dataset(Config.DATASET_NAME, split=Config.DATASET_SPLIT)
        print(f"成功加载 {len(dataset)} 条数据")
    except Exception as e:
        print(f"从 HuggingFace 加载失败: {e}")
        print("使用本地示例数据进行测试...")
        return _load_sample_financebench(subset_size)
    
    examples = []
    corpus = []
        
    for idx, item in enumerate(dataset):
        # 提取证据文档
        # evidence 是一个列表，通常包含一个或多个证据片段
        evidence_list = item.get("evidence", [])
        if not evidence_list:
            continue
            
        # 构建证据文档文本（合并所有证据片段）
        evidence_texts = []
        for ev in evidence_list:
            if isinstance(ev, dict):
                # 优先使用 evidence_text_full_page（更完整）
                ev_text = ev.get("evidence_text_full_page", "") or ev.get("evidence_text", "")
                if ev_text:
                    evidence_texts.append(ev_text)
            
        # 主要文档名
        doc_name = item.get("doc_name", "unknown")
        financebench_id = item.get("financebench_id", str(idx))
            
        # 证据文档 ID（使用问题ID作为唯一标识）
        evidence_doc_id = f"{financebench_id}"
            
        # 构建示例
        example = {
            "id": financebench_id,
            "question": item["question"],
            "answer": item["answer"],
            "company": item.get("company", ""),
            "doc_name": doc_name,
            "doc_type": item.get("doc_type", ""),
            "question_type": item.get("question_type", ""),
            "evidence_doc": evidence_doc_id,  # 用于检索评估：每个问题对应唯一的证据片段
            "evidence_text": " ".join(evidence_texts) if evidence_texts else "",
        }
        examples.append(example)
            
        # 构建语料库：每个问题的证据作为单独的文档
        # 这样检索就变成了：在所有证据片段中找到正确的那个
        doc_text = " ".join(evidence_texts) if evidence_texts else ""
        if doc_text:
            corpus.append({
                "id": evidence_doc_id,  # 使用问题ID作为文档ID
                "title": f"{doc_name} - Q{idx+1}",
                "text": doc_text,
                "metadata": {
                    "company": item.get("company", ""),
                    "doc_type": item.get("doc_type", ""),
                    "doc_link": item.get("doc_link", ""),
                    "doc_name": doc_name,
                    "question": item["question"][:100]  # 保存问题前100字符作为参考
                }
            })
        
    # 限制样本数量
    if subset_size and subset_size < len(examples):
        examples = examples[:subset_size]
        # 同时限制语料库
        example_ids = set(ex["id"] for ex in examples)
        corpus = [doc for doc in corpus if doc["id"] in example_ids]
        
    print(f"加载完成: {len(examples)} 个问题, {len(corpus)} 篇文档 (证据片段级别)")
    return examples, corpus


def _load_sample_financebench(subset_size: Optional[int] = None) -> Tuple[List[Dict], List[Dict]]:
    """
    加载示例 FinanceBench 数据（离线模式）
    当无法从 HuggingFace 加载时使用
    """
    print("加载本地示例数据...")
    
    sample_data = [
        {
            "id": "fb_001",
            "question": "What was Apple's total revenue in fiscal year 2023?",
            "answer": "$383.3 billion",
            "company": "Apple",
            "doc_name": "Apple_2023_10K",
            "doc_type": "10k",
            "question_type": "metrics-generated",
            "evidence_doc": "fb_001",  # 使用问题ID
            "evidence_text": "Apple Inc. reported total net sales of $383.3 billion for the fiscal year ended September 30, 2023. iPhone accounted for $200.6 billion, Services for $85.2 billion.",
        },
        {
            "id": "fb_002",
            "question": "What is Microsoft's current ratio as of June 30, 2023?",
            "answer": "1.77",
            "company": "Microsoft",
            "doc_name": "Microsoft_2023_10K",
            "doc_type": "10k",
            "question_type": "metrics-generated",
            "evidence_doc": "fb_002",  # 使用问题ID
            "evidence_text": "Microsoft Corporation had total current assets of $184.3 billion and total current liabilities of $103.9 billion as of June 30, 2023.",
        },
        {
            "id": "fb_003",
            "question": "What was NVIDIA's revenue growth rate in fiscal 2024?",
            "answer": "126%",
            "company": "NVIDIA",
            "doc_name": "NVIDIA_2024_10K",
            "doc_type": "10k",
            "question_type": "metrics-generated",
            "evidence_doc": "fb_003",  # 使用问题ID
            "evidence_text": "NVIDIA reported revenue of $60.9 billion for fiscal year 2024, up 126% from $26.9 billion in fiscal year 2023.",
        },
        {
            "id": "fb_004",
            "question": "What is Amazon's operating income for FY2023?",
            "answer": "$36.9 billion",
            "company": "Amazon",
            "doc_name": "Amazon_2023_10K",
            "doc_type": "10k",
            "question_type": "metrics-generated",
            "evidence_doc": "fb_004",  # 使用问题ID
            "evidence_text": "Amazon.com reported operating income of $36.9 billion for the fiscal year ended December 31, 2023, compared to $12.2 billion in 2022.",
        },
        {
            "id": "fb_005",
            "question": "What is Google's (Alphabet) total revenue for FY2023?",
            "answer": "$307.4 billion",
            "company": "Alphabet",
            "doc_name": "Alphabet_2023_10K",
            "doc_type": "10k",
            "question_type": "metrics-generated",
            "evidence_doc": "fb_005",  # 使用问题ID
            "evidence_text": "Alphabet Inc. reported total revenues of $307.4 billion for the fiscal year ended December 31, 2023, an increase of 9% compared to 2022.",
        },
    ]
        
    examples = sample_data[:subset_size] if subset_size else sample_data
        
    # 构建语料库：每个问题的证据作为单独的文档
    corpus = []
    for ex in sample_data:
        corpus.append({
            "id": ex["evidence_doc"],  # 使用问题ID
            "title": f"{ex['doc_name']} - {ex['id']}",
            "text": ex["evidence_text"],
            "metadata": {
                "company": ex["company"],
                "doc_type": ex["doc_type"],
                "doc_name": ex["doc_name"],
            }
        })
        
    # 限制语料库
    if subset_size:
        example_ids = set(ex["id"] for ex in examples)
        corpus = [doc for doc in corpus if doc["id"] in example_ids]
        
    print(f"加载完成: {len(examples)} 个问题, {len(corpus)} 篇文档 (证据片段级别)")
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
            
        # 尝试加载缓存（需要验证缓存有效性）
        cache_valid = False
        if cache_path and os.path.exists(cache_path):
            try:
                print(f"从缓存加载 BM25 索引: {cache_path}")
                with open(cache_path, 'rb') as f:
                    data = pickle.load(f)
                    self.bm25 = data['bm25']
                # 验证缓存：检查文档数量是否匹配
                cached_corpus_size = len(self.bm25.doc_len) if hasattr(self.bm25, 'doc_len') else 0
                if cached_corpus_size == len(self.corpus):
                    print(f"缓存验证通过: {cached_corpus_size} 个文档")
                    cache_valid = True
                else:
                    print(f"缓存失效: 缓存有 {cached_corpus_size} 个文档, 当前有 {len(self.corpus)} 个文档，重新构建...")
            except Exception as e:
                print(f"缓存加载失败: {e}，重新构建...")
            
        if not cache_valid:
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
        
        # 加载模型（兼容新版 transformers/huggingface_hub）
        print(f"加载 Contriever 模型: {model_name}")
        max_retries = 3
        import time

        last_error = None
        for attempt in range(max_retries):
            try:
                print(f"  尝试加载 ({attempt + 1}/{max_retries})...")
                # 注意：不要再传 resume_download/force_download，
                # 在部分新版依赖组合下会触发 TypeError。
                self.tokenizer = AutoTokenizer.from_pretrained(
                    model_name,
                    use_fast=True,
                )
                self.model = AutoModel.from_pretrained(model_name)
                print("  模型加载成功！")
                last_error = None
                break
            except TypeError as e:
                last_error = e
                print(f"  加载失败: {type(e).__name__}: {e}")
                if attempt < max_retries - 1:
                    wait_time = 5 + attempt * 3
                    print(f"  等待 {wait_time} 秒后重试...")
                    time.sleep(wait_time)
                else:
                    print("  最终失败：检测到依赖参数兼容问题。")
            except Exception as e:
                last_error = e
                print(f"  加载失败: {type(e).__name__}: {e}")
                if attempt < max_retries - 1:
                    wait_time = 5 + attempt * 3
                    print(f"  等待 {wait_time} 秒后重试...")
                    time.sleep(wait_time)
                else:
                    print("  最终失败：请检查网络或 HuggingFace 访问。")

        if last_error is not None:
            raise RuntimeError(
                "Contriever 加载失败。建议先运行 BM25（use_retrievers=['bm25']）验证主流程，"
                "随后再排查 HF 依赖/网络。"
            ) from last_error

        
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
            api_key: 阿里百炼 API Key（如未提供，将从环境变量 DASHSCOPE_API_KEY 读取）
            model: 模型名称
        """
        # 获取 API key
        self.api_key = api_key or Config.DASHSCOPE_API_KEY
        self.model = model
        
        # 调试：显示 API key 来源和内容预览
        if api_key:
            print(f"API key 来源: 参数传入")
        else:
            print(f"API key 来源: 环境变量 DASHSCOPE_API_KEY")
        
        if not self.api_key:
            raise ValueError(
                "未找到 API key！请通过以下方式之一设置：\n"
                "1. 在 .env 文件中设置 DASHSCOPE_API_KEY=sk-xxx\n"
                "2. 设置环境变量: export DASHSCOPE_API_KEY=sk-xxx\n"
                "3. 直接传入参数: QwenGenerator(api_key='sk-xxx')"
            )
        
        # 清理 API key（移除可能的空白字符和不可见字符）
        self.api_key = self.api_key.strip()
        
        # 验证 API key 格式
        # 1. 检查是否包含非 ASCII 字符
        try:
            self.api_key.encode('ascii')
        except UnicodeEncodeError as e:
            # 检查是否是中文占位符
            has_chinese = any('\u4e00' <= c <= '\u9fff' for c in self.api_key)
            if has_chinese:
                print(f"\n{'='*60}")
                print("错误: API key 包含中文字符！")
                print(f"API key 内容: {repr(self.api_key[:20])}...")
                print(f"{'='*60}")
                print("可能的原因:")
                print("  1. 环境变量 DASHSCOPE_API_KEY 被设置成了中文占位符")
                print("  2. .env 文件内容不正确")
                print("\n解决方法:")
                print("  1. 检查远程服务器上的环境变量:")
                print("     !echo $DASHSCOPE_API_KEY  # Linux/Mac")
                print("     !echo %DASHSCOPE_API_KEY%  # Windows")
                print("  2. 直接传入正确的 API key:")
                print("     generator = QwenGenerator(api_key='sk-your-real-key')")
                print(f"{'='*60}")
            raise ValueError("API key 格式错误！请检查是否使用了正确的 API key")
        
        # 打印 API key 前4位用于调试（安全：不暴露完整 key）
        print(f"API key 前缀: {self.api_key[:4]}... (长度: {len(self.api_key)})")
        
        # 配置 httpx 客户端以支持 UTF-8 编码
        http_client = httpx.Client(
            timeout=httpx.Timeout(60.0, connect=10.0),
        )
        
        # 设置默认 headers，覆盖 OpenAI SDK 的 X-Stainless-* headers
        # 这些 headers 可能包含系统信息，如果系统语言是中文，会导致 httpx 编码错误
        default_headers = {
            "x-stainless-os": "Linux",  # 覆盖默认的系统名称，避免中文编码问题
            "x-stainless-arch": "x86_64",
            "x-stainless-lang": "python",
            "x-stainless-runtime": "CPython",
        }
        
        # 初始化 OpenAI 客户端（兼容阿里百炼 API）
        self.client = OpenAI(
            api_key=self.api_key,
            base_url=Config.DASHSCOPE_BASE_URL,
            http_client=http_client,
            default_headers=default_headers,
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
# LLM 生成器（本地 Llama 模型 - Colab 专用）
# ============================================

class LlamaGenerator:
    """本地 Llama 模型生成器 - 适用于 Colab 环境"""
    
    def __init__(
        self, 
        model_path: str = None, 
        device: str = None,
        load_in_8bit: bool = True,
        load_in_4bit: bool = False
    ):
        """
        初始化 Llama 生成器
        
        Args:
            model_path: 模型路径（默认使用 Config.LLAMA_MODEL_PATH）
            device: 运行设备
            load_in_8bit: 是否使用 8bit 量化（节省显存）
            load_in_4bit: 是否使用 4bit 量化（更省显存）
        """
        self.model_path = model_path or Config.LLAMA_MODEL_PATH
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        
        print(f"正在加载 Llama 模型: {self.model_path}")
        print(f"设备: {self.device}")
        
        # 检查模型路径是否存在
        if not os.path.exists(self.model_path):
            raise ValueError(
                f"模型路径不存在: {self.model_path}\n"
                f"请先下载模型或检查路径是否正确。\n"
                f"下载方法: 使用 llama/download_model.py"
            )
        
        
        # 加载 tokenizer
        from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
        
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_path,
            trust_remote_code=True
        )
        
        # 设置 pad_token
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        
        
        # 量化配置
        quantization_config = None
        if load_in_4bit:
            from transformers import BitsAndBytesConfig
            quantization_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.float16,
                bnb_4bit_use_double_quant=True,
                bnb_4bit_quant_type="nf4"
            )
            print("使用 4bit 量化")
        elif load_in_8bit:
            from transformers import BitsAndBytesConfig
            quantization_config = BitsAndBytesConfig(
                load_in_8bit=True
            )
            print("使用 8bit 量化")
        else:
            print("使用全精度 (FP16)")
        
        # 加载模型
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_path,
            torch_dtype=torch.float16,
            device_map="auto" if self.device == "cuda" else None,
            quantization_config=quantization_config,
            trust_remote_code=True
        )
        
        if self.device == "cpu":
            self.model = self.model.to(self.device)
        
        self.model.eval()
        
        # 创建 pipeline
        self.pipe = pipeline(
            "text-generation",
            model=self.model,
            tokenizer=self.tokenizer,
            torch_dtype=torch.float16,
            device_map="auto" if self.device == "cuda" else None
        )
        
        print(f"Llama 模型加载完成！")
        
        # 保存引用
        self.pipeline_module = pipeline
    
    def format_context(self, documents: List[Dict]) -> str:
        """格式化检索到的文档"""
        context_parts = []
        for i, doc in enumerate(documents, 1):
            title = doc.get('title', 'Unknown')
            context_parts.append(f"[Document {i}] {title}\n{doc['text']}\n")
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
        
        # Llama-3-Instruct 格式的 prompt
        prompt = f"""You are a helpful AI assistant. Answer the question based on the provided documents.
If the documents don't contain enough information, say "I cannot answer this question based on the provided documents."

Documents:
{context}

Question: {question}

Please provide a concise and accurate answer:"""
        
        try:
            # 使用 Llama-3-Instruct 的对话格式
            messages = [
                {"role": "system", "content": "You are a helpful AI assistant that answers questions based on provided documents."},
                {"role": "user", "content": prompt}
            ]
            
            # 应用聊天模板
            input_ids = self.tokenizer.apply_chat_template(
                messages, 
                add_generation_prompt=True,
                return_tensors="pt"
            ).to(self.model.device)
            
            # 生成
            with torch.no_grad():
                outputs = self.model.generate(
                    input_ids,
                    max_new_tokens=256,
                    do_sample=True,
                    temperature=0.7,
                    top_p=0.9,
                    pad_token_id=self.tokenizer.eos_token_id
                )
            
            # 解码
            generated_text = self.tokenizer.decode(
                outputs[0][input_ids.shape[-1]:], 
                skip_special_tokens=True
            )
            
            return {
                "answer": generated_text.strip(),
                "prompt": prompt,
                "model": f"Llama-3.1-8B-Instruct"
            }
            
        except Exception as e:
            error_msg = str(e)
            error_type = type(e).__name__
            print(f"\n生成错误 [{error_type}]: {error_msg}")
            return {
                "answer": "",
                "prompt": prompt,
                "model": "Llama-3.1-8B-Instruct",
                "error": error_msg,
                "error_type": error_type
            }
    
    def get_name(self) -> str:
        return "Llama-3.1-8B-Instruct"


# ============================================
# 通用 LLM 生成器工厂
# ============================================

def create_generator(
    generator_type: str = "api",
    model: str = None,
    api_key: str = None,
    model_path: str = None,
    device: str = None,
    **kwargs
):
    """
    创建生成器的工厂函数
    
    Args:
        generator_type: 生成器类型，按部署方式区分
            - "api": 使用云端 API（阿里百炼 Qwen）
            - "local": 使用本地 HuggingFace 模型（支持 Qwen/Llama 等）
            - 向后兼容: "qwen" 等同于 "api", "llama" 等同于 "local"
        model: API 模型名称，默认 "qwen-plus" (仅 api 类型)
        api_key: DashScope API Key (仅 api 类型)
        model_path: 本地模型路径 (仅 local 类型)
        device: 运行设备 (仅 local 类型)
        
    Returns:
        生成器实例
    """
    generator_type = generator_type.lower()
    
    # 向后兼容映射
    type_mapping = {
        "qwen": "api",
        "llama": "local"
    }
    generator_type = type_mapping.get(generator_type, generator_type)
    
    if generator_type == "api":
        return QwenGenerator(
            api_key=api_key,
            model=model or Config.QWEN_MODEL
        )
    elif generator_type == "local":
        return LlamaGenerator(
            model_path=model_path,
            device=device
        )
    else:
        raise ValueError(f"不支持的生成器类型: {generator_type}. 可选: api, local")


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
                {
                    "id": doc['id'],
                    "title": doc.get('title', ''),
                    "text": doc.get('text', ''),
                    "score": score
                }
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


def compute_retrieval_metrics(retrieved_docs: List[Dict], evidence_doc: str) -> Dict[str, float]:
    """
    计算检索指标：Recall@K, Precision@K, F1@K, Hit@K
    
    适用于 FinanceBench 单文档检索场景
    
    Args:
        retrieved_docs: 检索到的文档列表，每个包含 'id' 字段
        evidence_doc: 正确证据文档的名称（单个）
        
    Returns:
        包含 recall@k, precision@k, f1@k, hit@k 的字典
    """
    # 检索到的文档 ID 集合
    retrieved_ids = [doc['id'] for doc in retrieved_docs]
    retrieved_set = set(retrieved_ids)
    
    # 正确文档（FinanceBench 是单文档检索）
    gold_set = {evidence_doc} if evidence_doc else set()
    
    # 检查是否命中正确文档
    hit = evidence_doc in retrieved_set
    
    # Recall@K: 如果正确文档在检索结果中，则为 1，否则为 0
    # 对于单文档检索，Recall@K 退化为 Hit@K
    recall = 1.0 if hit else 0.0
    
    # Precision@K: 命中数 / 检索数（单文档场景下最多为 1/K）
    precision = 1.0 / len(retrieved_ids) if hit and retrieved_ids else 0.0
    
    # F1@K: Recall 和 Precision 的调和平均
    if recall + precision > 0:
        f1 = 2 * recall * precision / (recall + precision)
    else:
        f1 = 0.0
    
    # Hit@K: 是否命中正确文档（0 或 1）
    hit_at_k = 1.0 if hit else 0.0
    
    return {
        'recall@k': recall,
        'precision@k': precision,
        'f1@k': f1,
        'hit@k': hit_at_k
    }


def evaluate(rag_system: RAGSystem, examples: List[Dict]) -> Dict:
    """
    评估 RAG 系统
    
    使用检索指标（Recall@K, Precision@K, F1@K, Hit@K）评估
    适用于 FinanceBench 单文档检索场景
    
    Args:
        rag_system: RAG 系统实例
        examples: 测试示例列表，每个包含 'question', 'answer', 'evidence_doc' 字段
        
    Returns:
        评估结果字典
    """
    print(f"\n开始评估 {rag_system.retriever.get_name()} + {rag_system.generator.get_name()}...")
    
    f1_scores = []
    recall_scores = []
    precision_scores = []
    f1_at_k_scores = []
    hit_at_k_scores = []
    
    for example in tqdm(examples, desc="评估中", ascii=False):
        result = rag_system.answer(example['question'])
        
        # 计算生成答案的 F1 分数
        f1 = compute_f1(result['answer'], example['answer'])
        f1_scores.append(f1)
        
        # 计算检索指标（使用 evidence_doc 而非 supporting_facts）
        retrieval_metrics = compute_retrieval_metrics(
            result['retrieved_documents'], 
            example.get('evidence_doc', '')  # FinanceBench 使用单文档
        )
        recall_scores.append(retrieval_metrics['recall@k'])
        precision_scores.append(retrieval_metrics['precision@k'])
        f1_at_k_scores.append(retrieval_metrics['f1@k'])
        hit_at_k_scores.append(retrieval_metrics['hit@k'])
    
    return {
        "f1_score": sum(f1_scores) / len(f1_scores),
        "recall@k": sum(recall_scores) / len(recall_scores),
        "precision@k": sum(precision_scores) / len(precision_scores),
        "f1@k": sum(f1_at_k_scores) / len(f1_at_k_scores),
        "hit@k": sum(hit_at_k_scores) / len(hit_at_k_scores),  # 新增 Hit@K
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
    parser.add_argument("--generator-type", type=str, default="api", choices=["api", "local", "qwen", "llama"],
                        help="生成器类型: api (云端API) 或 local (本地模型)。向后兼容: qwen=api, llama=local")
    parser.add_argument("--model", type=str, default="qwen-plus",
                        help="API 模型名称 (仅当 --generator-type=api 时有效)")
    parser.add_argument("--model-path", type=str, default=Config.LLAMA_MODEL_PATH,
                        help="本地模型路径 (仅当 --generator-type=local 时有效)")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu",
                        help="运行设备")
    
    # 使用 parse_known_args() 以兼容 Jupyter/Colab 环境
    args, _ = parser.parse_known_args()
    
    print("=" * 60)
    print("RAG 系统实验 - FinanceBench")
    print("=" * 60)
    print(f"检索方法: {args.retriever}")
    print(f"测试样本: {args.subset}")
    print(f"Top-K: {args.top_k}")
    print(f"生成器类型: {args.generator_type}")
    # 向后兼容
    gen_type = args.generator_type.lower()
    type_mapping = {"qwen": "api", "llama": "local"}
    gen_type = type_mapping.get(gen_type, gen_type)
    
    if gen_type == "api":
        print(f"生成模型: {args.model}")
    else:
        print(f"模型路径: {args.model_path}")
    print(f"设备: {args.device}")
    print("=" * 60)
    
    # 1. 加载数据
    examples, corpus = load_financebench(subset_size=args.subset)
    
    # 确保缓存目录存在
    os.makedirs(Config.CACHE_DIR, exist_ok=True)
    os.makedirs(Config.RESULTS_DIR, exist_ok=True)
    
    # 2. 初始化检索器
    retrievers = {}
    
    if args.retriever in ["bm25", "both"]:
        bm25_cache = os.path.join(Config.CACHE_DIR, "bm25_index_finance.pkl")
        retrievers["bm25"] = BM25Retriever(corpus, cache_path=bm25_cache)
    
    if args.retriever in ["contriever", "both"]:
        contriever_cache = os.path.join(Config.CACHE_DIR, "contriever_embeddings_finance.pkl")
        retrievers["contriever"] = ContrieverRetriever(
            corpus, 
            cache_path=contriever_cache,
            device=args.device
        )
    
    # 3. 初始化生成器
    if gen_type == "local":
        generator = LlamaGenerator(
            model_path=args.model_path,
            device=args.device
        )
    else:
        generator = QwenGenerator(model=args.model)
    
    # 4. 运行实验
    generator_name = args.model if args.generator_type == "qwen" else "llama"
    results = []
    for retriever_name, retriever in retrievers.items():
        print(f"\n{'='*60}")
        print(f"测试配置: {retriever_name} + {generator_name}")
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
        print(f"  Hit@K:       {eval_result['hit@k']:.4f}")
    
    # 5. 保存结果
    result_file = os.path.join(
        Config.RESULTS_DIR, 
        f"results_{args.retriever}_{generator_name}_{args.subset}.json"
    )
    with open(result_file, 'w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    print(f"\n结果已保存至: {result_file}")
    
    # 6. 对比展示
    if len(results) > 1:
        print("\n" + "=" * 60)
        print("检索方法对比:")
        print("=" * 80)
        print(f"{'检索器':<12} {'F1 Score':<12} {'Recall@K':<12} {'Precision@K':<12} {'Hit@K':<12}")
        print("-" * 80)
        for r in results:
            print(f"{r['retriever']:<12} {r['f1_score']:<12.4f} {r['recall@k']:<12.4f} {r['precision@k']:<12.4f} {r['hit@k']:<12.4f}")


if __name__ == "__main__":
    main()


# ============================================
# Colab 专用：BM25 vs Contriever 对比实验
# ============================================

def run_comparison_experiment(
    subset_size: int = 50,
    top_k: int = 3,
    model: str = "qwen-plus",
    api_key: str = None,
    device: str = None,
    use_retrievers: List[str] = None,
    generator_type: str = "api",
    model_path: str = None
) -> Dict:
    """
    Colab 专用函数：运行 BM25 vs Contriever 对比实验 (FinanceBench)
    
    在 Colab 中直接调用：
        # 使用云端 API
        from rag_finance import run_comparison_experiment
        results = run_comparison_experiment(api_key="sk-xxx")
        
        # 使用本地模型（如 Qwen2.5-7B-Instruct）
        results = run_comparison_experiment(
            generator_type="local",
            model_path="/content/models/Qwen2.5-7B-Instruct"
        )
        
        # 快速调试：只用 BM25（跳过 Contriever 模型下载）
        results = run_comparison_experiment(use_retrievers=["bm25"])
    
    Args:
        subset_size: 测试样本数量，默认 50
        top_k: 检索文档数量，默认 3
        model: API 模型名称，默认 "qwen-plus" (仅当 generator_type="api" 时有效)
        api_key: DashScope API Key（如未提供，从环境变量读取）
        device: 运行设备（默认自动检测）
        use_retrievers: 要使用的检索器列表，可选 ["bm25", "contriever"]，默认两者都用
        generator_type: 生成器类型，"api" 或 "local"（向后兼容: "qwen"=api, "llama"=local）
        model_path: 本地模型路径 (仅当 generator_type="local" 时有效)
        
    Returns:
        包含检索器评估结果的字典
    """
    # 向后兼容映射
    type_mapping = {"qwen": "api", "llama": "local"}
    generator_type = type_mapping.get(generator_type.lower(), generator_type.lower())
    
    # 设置 API Key (仅 API 模式需要)
    if api_key and generator_type == "api":
        os.environ['DASHSCOPE_API_KEY'] = api_key
    
    # 检测设备
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    
    print("=" * 70)
    print("🚀 RAG 系统对比实验 - FinanceBench (BM25 vs Contriever)")
    print("=" * 70)
    print(f"测试样本: {subset_size}")
    print(f"Top-K: {top_k}")
    print(f"生成器类型: {generator_type.upper()}")
    if generator_type == "api":
        print(f"生成模型: {model}")
    else:
        print(f"模型路径: {model_path or Config.LLAMA_MODEL_PATH}")
    print(f"设备: {device}")
    print("=" * 70)
    
    # 1. 加载数据
    examples, corpus = load_financebench(subset_size=subset_size)
    
    # 确保目录存在
    os.makedirs(Config.CACHE_DIR, exist_ok=True)
    os.makedirs(Config.RESULTS_DIR, exist_ok=True)
    
    # 默认使用两个检索器
    if use_retrievers is None:
        use_retrievers = ["bm25", "contriever"]
    use_retrievers = [r.lower() for r in use_retrievers]
    
    # 2. 初始化检索器
    print("\n" + "="*70)
    print("📦 初始化检索器...")
    print("="*70)
    
    retrievers = []
    
    # BM25
    if "bm25" in use_retrievers:
        print("\n[BM25] 初始化 BM25 检索器...")
        bm25_cache = os.path.join(Config.CACHE_DIR, "bm25_index_finance.pkl")
        bm25_retriever = BM25Retriever(corpus, cache_path=bm25_cache)
        retrievers.append(("BM25", bm25_retriever))
    
    # Contriever
    if "contriever" in use_retrievers:
        print("\n[Contriever] 初始化 Contriever 检索器...")
        contriever_cache = os.path.join(Config.CACHE_DIR, "contriever_embeddings_finance.pkl")
        contriever_retriever = ContrieverRetriever(
            corpus, 
            cache_path=contriever_cache,
            device=device
        )
        retrievers.append(("Contriever", contriever_retriever))
    
    # 3. 初始化生成器
    print(f"\n🤖 初始化生成器: {generator_type.upper()}")
    if generator_type == "local":
        generator = LlamaGenerator(
            model_path=model_path,
            device=device
        )
    else:
        generator = QwenGenerator(model=model)
    
    # 4. 运行实验
    results = {}
    
    for name, retriever in retrievers:
        print(f"\n{'='*70}")
        print(f"🔬 测试配置: {name} + {model}")
        print(f"{'='*70}")
        
        # 创建 RAG 系统
        rag = RAGSystem(retriever, generator, top_k=top_k)
        
        # 评估
        eval_result = evaluate(rag, examples)
        results[name] = eval_result
        
        # 打印单次结果
        print(f"\n📊 {name} 评估结果:")
        print(f"   F1 Score:    {eval_result['f1_score']:.4f}")
        print(f"   Recall@K:   {eval_result['recall@k']:.4f}")
        print(f"   Precision@K: {eval_result['precision@k']:.4f}")
        print(f"   Hit@K:      {eval_result['hit@k']:.4f}")
    
    # 5. 保存结果
    from datetime import datetime
    result_file = os.path.join(
        Config.RESULTS_DIR, 
        f"comparison_finance_{subset_size}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    )
    with open(result_file, 'w', encoding='utf-8') as f:
        json.dump(list(results.values()), f, ensure_ascii=False, indent=2)
    print(f"\n💾 结果已保存至: {result_file}")
    
    # 6. 对比展示
    print("\n" + "="*70)
    print("📈 最终对比结果")
    print("="*70)
    print(f"{'检索器':<15} {'F1 Score':<12} {'Recall@K':<12} {'Precision@K':<14} {'Hit@K':<12}")
    print("-"*70)
    for name, r in results.items():
        print(f"{r['retriever']:<15} {r['f1_score']:<12.4f} {r['recall@k']:<12.4f} {r['precision@k']:<14.4f} {r['hit@k']:<12.4f}")
    print("="*70)
    
    # 找出最佳检索器
    best_hit = max(results.keys(), key=lambda k: results[k]['hit@k'])
    best_f1 = max(results.keys(), key=lambda k: results[k]['f1_score'])
    
    print(f"\n🏆 最佳 Hit@K:    {best_hit} ({results[best_hit]['hit@k']:.4f})")
    print(f"🏆 最佳 F1 Score: {best_f1} ({results[best_f1]['f1_score']:.4f})")
    
    # 计算提升幅度（只有当两个检索器都存在时才计算）
    if 'BM25' in results and 'Contriever' in results:
        hit_improve = (results['Contriever']['hit@k'] - results['BM25']['hit@k']) / results['BM25']['hit@k'] * 100 if results['BM25']['hit@k'] > 0 else 0
        f1_improve = (results['Contriever']['f1_score'] - results['BM25']['f1_score']) / results['BM25']['f1_score'] * 100 if results['BM25']['f1_score'] > 0 else 0
        
        print(f"\n📊 Contriever vs BM25 提升:")
        print(f"   Hit@K:    {'+' if hit_improve >= 0 else ''}{hit_improve:.2f}%")
        print(f"   F1 Score: {'+' if f1_improve >= 0 else ''}{f1_improve:.2f}%")
    
    return results


# ============================================
# Colab 专用：一键运行 Llama 实验
# ============================================

def run_llama_experiment(
    subset_size: int = 50,
    top_k: int = 3,
    llama_model_path: str = "/content/models/llama-3.1-8b",
    use_retrievers: List[str] = None,
    use_8bit: bool = True
) -> Dict:
    """
    Colab 专用：使用本地 Llama 模型运行 RAG 实验 (FinanceBench)
    
    使用方法（在 Colab 中）:
        # 1. 先下载模型（如果还没有）
        from huggingface_hub import snapshot_download, login
        login(token="your_hf_token")
        snapshot_download(
            repo_id="meta-llama/Meta-Llama-3.1-8B-Instruct",
            local_dir="/content/models/llama-3.1-8b"
        )
        
        # 2. 运行实验
        from rag_finance import run_llama_experiment
        results = run_llama_experiment(
            subset_size=10,  # 快速测试用少量数据
            llama_model_path="/content/models/llama-3.1-8b"
        )
    
    Args:
        subset_size: 测试样本数量
        top_k: 检索文档数量
        llama_model_path: Llama 模型路径
        use_retrievers: 使用的检索器 ["bm25", "contriever"]
        use_8bit: 是否使用 8bit 量化（节省显存）
        
    Returns:
        实验结果字典
    """
    return run_comparison_experiment(
        subset_size=subset_size,
        top_k=top_k,
        generator_type="llama",
        llama_model_path=llama_model_path,
        use_retrievers=use_retrievers
    )
