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
    DATASET_NAME = "hotpotqa/hotpot_qa"
    DATASET_SPLIT = "validation"  # 使用 validation 分割（dev 集）
    SUBSET_SIZE = 100  # 测试样本数量
    
    # 检索配置
    RETRIEVAL_K = 3  # 每次检索返回 top-k 文档
    
    # 模型配置
    CONTRIEVER_MODEL = "facebook/contriever-msmarco"
    
    # 阿里百炼 API 配置
    # 请在此处填写你的 API Key，或通过环境变量设置
    DASHSCOPE_API_KEY = os.environ.get("DASHSCOPE_API_KEY", "sk-0f66e1b456ab47f889da87e04a6512a6")
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
    # "distractor" 是 HotpotQA 的困难模式：
    # 每个问题有 10 个候选文档，其中只有 2 个是真正需要的支持文档，其他 8 个是干扰项
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
        
        # 加载模型（带重试机制）
        print(f"加载 Contriever 模型: {model_name}")
        max_retries = 10  # 增加重试次数
        import time
        for attempt in range(max_retries):
            try:
                print(f"  尝试加载 ({attempt + 1}/{max_retries})...")
                self.tokenizer = AutoTokenizer.from_pretrained(
                    model_name, 
                    resume_download=True,
                    force_download=False
                )
                self.model = AutoModel.from_pretrained(
                    model_name, 
                    resume_download=True,
                    force_download=False
                )
                print("  模型加载成功！")
                break
            except Exception as e:
                if attempt < max_retries - 1:
                    print(f"  加载失败: {type(e).__name__}")
                    wait_time = 10 + attempt * 5  # 递增等待时间
                    print(f"  等待 {wait_time} 秒后重试...")
                    time.sleep(wait_time)
                else:
                    print(f"  加载模型失败，请检查网络连接")
                    print(f"  提示: 如果本地网络不稳定，建议在 Colab 上运行")
                    raise
        
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
            dtype=torch.float16,
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
            dtype=torch.float16,
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
            )
            
            # 获取模型设备（device_map="auto" 时 model.device 可能不存在）
            try:
                device = self.model.device
            except AttributeError:
                device = next(self.model.parameters()).device
            input_ids = input_ids.to(device)
            
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
    generator_type: str = "qwen",
    model: str = None,
    api_key: str = None,
    model_path: str = None,
    device: str = None,
    **kwargs
):
    """
    创建生成器的工厂函数
    
    Args:
        generator_type: 生成器类型 ("qwen" 或 "llama")
        model: Qwen 模型名称 (仅 qwen 类型)
        api_key: DashScope API Key (仅 qwen 类型)
        model_path: Llama 模型路径 (仅 llama 类型)
        device: 运行设备 (仅 llama 类型)
        
    Returns:
        生成器实例
    """
    generator_type = generator_type.lower()
    
    if generator_type == "qwen":
        return QwenGenerator(
            api_key=api_key,
            model=model or Config.QWEN_MODEL
        )
    elif generator_type == "llama":
        return LlamaGenerator(
            model_path=model_path,
            device=device
        )
    else:
        raise ValueError(f"不支持的生成器类型: {generator_type}. 可选: qwen, llama")


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
    parser.add_argument("--generator-type", type=str, default="qwen", choices=["qwen", "llama"],
                        help="生成器类型: qwen (API) 或 llama (本地)")
    parser.add_argument("--model", type=str, default="qwen-plus",
                        help="Qwen 模型名称 (仅当 --generator-type=qwen 时有效)")
    parser.add_argument("--llama-model-path", type=str, default=Config.LLAMA_MODEL_PATH,
                        help="Llama 模型路径 (仅当 --generator-type=llama 时有效)")
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
    print(f"生成器类型: {args.generator_type}")
    if args.generator_type == "qwen":
        print(f"生成模型: {args.model}")
    else:
        print(f"模型路径: {args.llama_model_path}")
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
    if args.generator_type == "llama":
        generator = LlamaGenerator(
            model_path=args.llama_model_path,
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
        print(f"{'检索器':<12} {'F1 Score':<12} {'Recall@K':<12} {'Precision@K':<12} {'F1@K':<12}")
        print("-" * 80)
        for r in results:
            print(f"{r['retriever']:<12} {r['f1_score']:<12.4f} {r['recall@k']:<12.4f} {r['precision@k']:<12.4f} {r['f1@k']:<12.4f}")


if __name__ == "__main__":
    main()


# ============================================
# Colab 专用：BM25 vs Contriever 对比实验
# ============================================

def run_comparison_experiment(
    subset_size: int = 100,
    top_k: int = 3,
    model: str = "qwen-plus",
    api_key: str = None,
    device: str = None,
    use_retrievers: List[str] = None,
    generator_type: str = "qwen",
    llama_model_path: str = None
) -> Dict:
    """
    Colab 专用函数：运行 BM25 vs Contriever 对比实验
    
    在 Colab 中直接调用：
        # 使用 Qwen (API)
        from rag_hotpotqa import run_comparison_experiment
        results = run_comparison_experiment(api_key="sk-xxx")
        
        # 使用本地 Llama 模型
        results = run_comparison_experiment(
            generator_type="llama",
            llama_model_path="/content/models/llama-3.1-8b"
        )
        
        # 快速调试：只用 BM25（跳过 Contriever 模型下载）
        results = run_comparison_experiment(api_key="sk-xxx", use_retrievers=["bm25"])
    
    Args:
        subset_size: 测试样本数量，默认 100
        top_k: 检索文档数量，默认 3
        model: Qwen 模型名称，默认 "qwen-plus" (仅当 generator_type="qwen" 时有效)
        api_key: DashScope API Key（如未提供，从环境变量读取）
        device: 运行设备（默认自动检测）
        use_retrievers: 要使用的检索器列表，可选 ["bm25", "contriever"]，默认两者都用
        generator_type: 生成器类型，"qwen" 或 "llama"
        llama_model_path: Llama 模型路径 (仅当 generator_type="llama" 时有效)
        
    Returns:
        包含检索器评估结果的字典
    """
    # 设置 API Key (仅 Qwen 需要)
    if api_key and generator_type == "qwen":
        os.environ['DASHSCOPE_API_KEY'] = api_key
    
    # 检测设备
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    
    print("=" * 70)
    print("🚀 RAG 系统对比实验 - BM25 vs Contriever")
    print("=" * 70)
    print(f"测试样本: {subset_size}")
    print(f"Top-K: {top_k}")
    print(f"生成器类型: {generator_type.upper()}")
    if generator_type == "qwen":
        print(f"生成模型: {model}")
    else:
        print(f"模型路径: {llama_model_path or Config.LLAMA_MODEL_PATH}")
    print(f"设备: {device}")
    print("=" * 70)
    
    # 1. 加载数据
    examples, corpus = load_hotpotqa(subset_size=subset_size)
    
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
        bm25_cache = os.path.join(Config.CACHE_DIR, "bm25_index.pkl")
        bm25_retriever = BM25Retriever(corpus, cache_path=bm25_cache)
        retrievers.append(("BM25", bm25_retriever))
    
    # Contriever
    if "contriever" in use_retrievers:
        print("\n[Contriever] 初始化 Contriever 检索器...")
        contriever_cache = os.path.join(Config.CACHE_DIR, "contriever_embeddings.pkl")
        contriever_retriever = ContrieverRetriever(
            corpus, 
            cache_path=contriever_cache,
            device=device
        )
        retrievers.append(("Contriever", contriever_retriever))
    
    # 3. 初始化生成器
    print(f"\n🤖 初始化生成器: {generator_type.upper()}")
    if generator_type.lower() == "llama":
        generator = LlamaGenerator(
            model_path=llama_model_path,
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
        print(f"   F1@K:       {eval_result['f1@k']:.4f}")
    
    # 5. 保存结果
    from datetime import datetime
    result_file = os.path.join(
        Config.RESULTS_DIR, 
        f"comparison_{subset_size}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    )
    with open(result_file, 'w', encoding='utf-8') as f:
        json.dump(list(results.values()), f, ensure_ascii=False, indent=2)
    print(f"\n💾 结果已保存至: {result_file}")
    
    # 6. 对比展示
    print("\n" + "="*70)
    print("📈 最终对比结果")
    print("="*70)
    print(f"{'检索器':<15} {'F1 Score':<12} {'Recall@K':<12} {'Precision@K':<14} {'F1@K':<12}")
    print("-"*70)
    for name, r in results.items():
        print(f"{r['retriever']:<15} {r['f1_score']:<12.4f} {r['recall@k']:<12.4f} {r['precision@k']:<14.4f} {r['f1@k']:<12.4f}")
    print("="*70)
    
    # 找出最佳检索器
    best_recall = max(results.keys(), key=lambda k: results[k]['recall@k'])
    best_f1k = max(results.keys(), key=lambda k: results[k]['f1@k'])
    best_f1 = max(results.keys(), key=lambda k: results[k]['f1_score'])
    
    print(f"\n🏆 最佳 Recall@K: {best_recall} ({results[best_recall]['recall@k']:.4f})")
    print(f"🏆 最佳 F1@K:     {best_f1k} ({results[best_f1k]['f1@k']:.4f})")
    print(f"🏆 最佳 F1 Score: {best_f1} ({results[best_f1]['f1_score']:.4f})")
    
    # 计算提升幅度（只有当两个检索器都存在时才计算）
    if 'BM25' in results and 'Contriever' in results:
        recall_improve = (results['Contriever']['recall@k'] - results['BM25']['recall@k']) / results['BM25']['recall@k'] * 100 if results['BM25']['recall@k'] > 0 else 0
        f1k_improve = (results['Contriever']['f1@k'] - results['BM25']['f1@k']) / results['BM25']['f1@k'] * 100 if results['BM25']['f1@k'] > 0 else 0
        
        print(f"\n📊 Contriever vs BM25 提升:")
        print(f"   Recall@K: {'+' if recall_improve >= 0 else ''}{recall_improve:.2f}%")
        print(f"   F1@K:     {'+' if f1k_improve >= 0 else ''}{f1k_improve:.2f}%")
    
    return results


# ============================================
# Colab 专用：一键运行 Llama 实验
# ============================================

def run_llama_bm25_quick(
    subset_size: int = 10,
    top_k: int = 3,
    llama_model_path: str = "/content/models/llama-3.1-8b"
) -> Dict:
    """
    Colab 快速测试：只运行 Llama + BM25（跳过 Qwen 和 Contriever）
    
    使用方法（在 Colab 中）:
        # 复制整个代码块到 Colab 单元格直接运行
        exec(open('rag_hotpotqa.py').read())
        results = run_llama_bm25_quick(subset_size=10)
    
    Args:
        subset_size: 测试样本数量
        top_k: 检索文档数量
        llama_model_path: Llama 模型路径
        
    Returns:
        实验结果字典
    """
    return run_comparison_experiment(
        subset_size=subset_size,
        top_k=top_k,
        generator_type="llama",
        llama_model_path=llama_model_path,
        use_retrievers=["bm25"]  # 只用 BM25
    )


def run_llama_experiment(
    subset_size: int = 100,
    top_k: int = 3,
    llama_model_path: str = "/content/models/llama-3.1-8b",
    use_retrievers: List[str] = None,
    use_8bit: bool = True
) -> Dict:
    """
    Colab 专用：使用本地 Llama 模型运行 RAG 实验
    
    使用方法（在 Colab 中）:
        # 1. 先下载模型（如果还没有）
        from huggingface_hub import snapshot_download, login
        login(token="your_hf_token")
        snapshot_download(
            repo_id="meta-llama/Meta-Llama-3.1-8B-Instruct",
            local_dir="/content/models/llama-3.1-8b"
        )
        
        # 2. 运行实验
        from rag_hotpotqa import run_llama_experiment
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
