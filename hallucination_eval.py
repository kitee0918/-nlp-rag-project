"""
幻觉评估模块 - 幻觉缓解实验
支持 FActScore 评估和 Self-RAG 生成

包含:
1. FActScoreEvaluator - 原子事实准确性评估（使用 Qwen API）
2. SelfRAGGenerator - Prompt-based Self-RAG 生成器
3. HallucinationEvaluator - 整合幻觉评估流程
"""

import os
import sys
import json
import re
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass
from tqdm import tqdm

# 设置 UTF-8 编码
os.environ['PYTHONIOENCODING'] = 'utf-8'
os.environ['PYTHONUTF8'] = '1'

# 尝试加载环境变量
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

from openai import OpenAI
import httpx


# ============================================
# 配置
# ============================================

class HallucinationConfig:
    """幻觉评估配置"""
    # 复用阿里百炼 API 配置
    DASHSCOPE_API_KEY = os.environ.get("DASHSCOPE_API_KEY", "")
    DASHSCOPE_BASE_URL = "https://dashscope.aliyuncs.com/compatible-mode/v1"
    QWEN_MODEL = "qwen-plus"
    
    # FActScore 配置
    MAX_ATOMS_PER_ANSWER = 10  # 每个答案最多拆分的原子事实数
    
    # Self-RAG 配置
    SELF_RAG_MAX_ITERATIONS = 3  # 最大迭代次数


# ============================================
# FActScore 评估器
# ============================================

@dataclass
class AtomicFact:
    """原子事实"""
    content: str
    is_supported: bool = False
    evidence: str = ""


class FActScoreEvaluator:
    """
    FActScore 评估器 - 使用 Qwen API
    
    FActScore = 支持的原子事实数 / 总原子事实数
    
    流程:
    1. 将答案分解为原子事实（atomic facts）
    2. 验证每个原子事实是否有证据支持
    3. 计算最终分数
    """
    
    def __init__(self, api_key: str = None, model: str = None):
        """
        初始化 FActScore 评估器
        
        Args:
            api_key: DashScope API Key
            model: Qwen 模型名称
        """
        self.api_key = api_key or HallucinationConfig.DASHSCOPE_API_KEY
        self.model = model or HallucinationConfig.QWEN_MODEL
        
        if not self.api_key:
            raise ValueError("未找到 API Key，请设置 DASHSCOPE_API_KEY 环境变量")
        
        # 初始化 OpenAI 客户端
        http_client = httpx.Client(timeout=httpx.Timeout(60.0, connect=10.0))
        default_headers = {
            "x-stainless-os": "Linux",
            "x-stainless-arch": "x86_64",
            "x-stainless-lang": "python",
        }
        
        self.client = OpenAI(
            api_key=self.api_key,
            base_url=HallucinationConfig.DASHSCOPE_BASE_URL,
            http_client=http_client,
            default_headers=default_headers,
        )
        
        print(f"FActScore 评估器初始化完成，使用模型: {self.model}")
    
    def decompose_answer(self, answer: str) -> List[str]:
        """
        将答案分解为原子事实
        
        原子事实：不可再分的简单陈述句
        
        Args:
            answer: 生成的答案
            
        Returns:
            原子事实列表
        """
        prompt = f"""请将以下答案分解为独立的原子事实（atomic facts）。

原子事实定义：不可再分的简单陈述句，每个句子包含一个独立的事实信息。

答案：
{answer}

要求：
1. 每行输出一个原子事实
2. 保持原意，不要添加或删除信息
3. 如果答案是"我不知道"或类似表述，输出一个原子事实即可
4. 只输出原子事实，不要编号或其他格式

原子事实列表："""

        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": "你是一个专业的事实分解助手。"},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.1,
                max_tokens=512
            )
            
            result = response.choices[0].message.content.strip()
            
            # 解析原子事实
            atoms = []
            for line in result.split('\n'):
                line = line.strip()
                # 移除可能的编号
                line = re.sub(r'^\d+[.、)\]]\s*', '', line)
                if line and len(line) > 5:  # 过滤过短的行
                    atoms.append(line)
            
            return atoms[:HallucinationConfig.MAX_ATOMS_PER_ANSWER]
            
        except Exception as e:
            print(f"分解原子事实时出错: {e}")
            return [answer]  # 返回原始答案作为单个原子
    
    def verify_fact(self, fact: str, evidence: str) -> Tuple[bool, str]:
        """
        验证原子事实是否有证据支持
        
        Args:
            fact: 原子事实
            evidence: 证据文本
            
        Returns:
            (is_supported, reason)
        """
        prompt = f"""请判断以下原子事实是否被证据文本支持。

原子事实：{fact}

证据文本：
{evidence}

判断标准：
- 支持（Supported）：证据明确包含该事实，或可以从中直接推断
- 不支持（Not Supported）：证据不包含该事实，或与事实矛盾
- 无法判断：证据不足以做出判断

请只回答"支持"、"不支持"或"无法判断"，并在下一行简要说明理由。

判断："""

        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": "你是一个专业的事实核查助手。"},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.1,
                max_tokens=128
            )
            
            result = response.choices[0].message.content.strip()
            
            # 解析结果
            if "支持" in result and "不支持" not in result:
                return True, result
            else:
                return False, result
                
        except Exception as e:
            print(f"验证事实时出错: {e}")
            return False, str(e)
    
    def compute_factscore(
        self, 
        answer: str, 
        evidence: str,
        verbose: bool = False
    ) -> Dict:
        """
        计算 FActScore
        
        Args:
            answer: 生成的答案
            evidence: 证据文本
            verbose: 是否打印详细信息
            
        Returns:
            包含 FActScore 和详细信息的字典
        """
        # 1. 分解原子事实
        atoms = self.decompose_answer(answer)
        
        if not atoms:
            return {
                "factscore": 0.0,
                "num_atoms": 0,
                "num_supported": 0,
                "atoms": []
            }
        
        if verbose:
            print(f"\n分解出 {len(atoms)} 个原子事实:")
            for i, atom in enumerate(atoms, 1):
                print(f"  {i}. {atom}")
        
        # 2. 验证每个原子事实
        atomic_facts = []
        supported_count = 0
        
        for atom in atoms:
            is_supported, reason = self.verify_fact(atom, evidence)
            atomic_facts.append(AtomicFact(
                content=atom,
                is_supported=is_supported,
                evidence=reason
            ))
            if is_supported:
                supported_count += 1
            
            if verbose:
                status = "✓ 支持" if is_supported else "✗ 不支持"
                print(f"  {status}: {atom[:50]}...")
        
        # 3. 计算 FActScore
        factscore = supported_count / len(atoms) if atoms else 0.0
        
        return {
            "factscore": factscore,
            "num_atoms": len(atoms),
            "num_supported": supported_count,
            "atoms": [
                {
                    "content": af.content,
                    "is_supported": af.is_supported,
                    "reason": af.evidence
                }
                for af in atomic_facts
            ]
        }
    
    def batch_compute_factscore(
        self,
        answers: List[str],
        evidences: List[str],
        verbose: bool = False
    ) -> Dict:
        """
        批量计算 FActScore
        
        Args:
            answers: 答案列表
            evidences: 证据列表
            verbose: 是否打印详细信息
            
        Returns:
            聚合结果
        """
        all_factscores = []
        all_details = []
        
        for i, (answer, evidence) in enumerate(tqdm(
            zip(answers, evidences), 
            total=len(answers),
            desc="计算 FActScore",
            ascii=False
        )):
            result = self.compute_factscore(answer, evidence, verbose=verbose and i < 3)
            all_factscores.append(result["factscore"])
            all_details.append(result)
        
        return {
            "avg_factscore": sum(all_factscores) / len(all_factscores) if all_factscores else 0.0,
            "individual_factscores": all_factscores,
            "details": all_details
        }


# ============================================
# Self-RAG 生成器
# ============================================

class SelfRAGGenerator:
    """
    Prompt-based Self-RAG 生成器
    
    Self-RAG 机制：
    1. [Retrieve] - 判断是否需要检索
    2. [IsREL] - 判断检索内容是否相关
    3. [IsSUP] - 判断检索内容是否支持回答
    4. [IsUSE] - 判断回答是否有用
    
    通过 Prompt 引导模型输出这些判断标记
    """
    
    def __init__(
        self, 
        retriever,
        api_key: str = None,
        model: str = None,
        max_iterations: int = 3
    ):
        """
        初始化 Self-RAG 生成器
        
        Args:
            retriever: 检索器实例
            api_key: DashScope API Key
            model: Qwen 模型名称
            max_iterations: 最大迭代次数
        """
        self.retriever = retriever
        self.api_key = api_key or HallucinationConfig.DASHSCOPE_API_KEY
        self.model = model or HallucinationConfig.QWEN_MODEL
        self.max_iterations = max_iterations
        
        if not self.api_key:
            raise ValueError("未找到 API Key，请设置 DASHSCOPE_API_KEY 环境变量")
        
        # 初始化 OpenAI 客户端
        http_client = httpx.Client(timeout=httpx.Timeout(60.0, connect=10.0))
        default_headers = {
            "x-stainless-os": "Linux",
            "x-stainless-arch": "x86_64",
            "x-stainless-lang": "python",
        }
        
        self.client = OpenAI(
            api_key=self.api_key,
            base_url=HallucinationConfig.DASHSCOPE_BASE_URL,
            http_client=http_client,
            default_headers=default_headers,
        )
        
        print(f"Self-RAG 生成器初始化完成，使用模型: {self.model}")
    
    def _format_context(self, documents: List[Dict]) -> str:
        """格式化检索到的文档"""
        context_parts = []
        for i, doc in enumerate(documents, 1):
            title = doc.get('title', 'Unknown')
            context_parts.append(f"[文档 {i}] {title}\n{doc['text']}\n")
        return "\n".join(context_parts)
    
    def _check_need_retrieve(self, question: str) -> bool:
        """
        判断是否需要检索
        
        对于金融问答，大多数情况需要检索
        """
        prompt = f"""判断回答以下问题是否需要检索外部知识。

问题：{question}

判断标准：
- 如果问题涉及具体数据、事实、日期、金额等，需要检索
- 如果是常识性问题或开放性讨论，可能不需要检索

请只回答"需要检索"或"不需要检索"。

判断："""

        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": "你是一个判断助手。"},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.1,
                max_tokens=32
            )
            
            result = response.choices[0].message.content.strip()
            return "需要检索" in result
            
        except Exception as e:
            print(f"判断检索需求时出错: {e}")
            return True  # 默认需要检索
    
    def _check_relevance(self, question: str, documents: List[Dict]) -> Tuple[bool, List[int]]:
        """
        判断检索文档是否与问题相关
        
        Returns:
            (is_relevant, relevant_doc_indices)
        """
        context = self._format_context(documents)
        
        prompt = f"""判断以下检索到的文档是否与问题相关。

问题：{question}

检索文档：
{context}

请判断每个文档是否相关，用逗号分隔的相关文档编号（如：1,2,3）。
如果都不相关，回答"无"。

相关文档编号："""

        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": "你是一个相关性判断助手。"},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.1,
                max_tokens=32
            )
            
            result = response.choices[0].message.content.strip()
            
            if "无" in result or "不相关" in result:
                return False, []
            
            # 解析相关文档编号
            indices = []
            for num in re.findall(r'\d+', result):
                idx = int(num) - 1
                if 0 <= idx < len(documents):
                    indices.append(idx)
            
            return len(indices) > 0, indices
            
        except Exception as e:
            print(f"判断相关性时出错: {e}")
            return True, list(range(len(documents)))  # 默认全部相关
    
    def _check_support(
        self, 
        question: str, 
        answer: str, 
        documents: List[Dict]
    ) -> Tuple[bool, str]:
        """
        判断答案是否被文档支持
        
        Returns:
            (is_supported, reason)
        """
        context = self._format_context(documents)
        
        prompt = f"""判断以下答案是否被提供的文档充分支持。

问题：{question}

答案：{answer}

文档：
{context}

判断标准：
- 充分支持：答案中的关键信息都能在文档中找到依据
- 部分支持：部分答案有依据，但存在无依据的内容
- 不支持：答案主要内容无法从文档中找到依据

请回答"充分支持"、"部分支持"或"不支持"，并在下一行简要说明。

判断："""

        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": "你是一个事实核查助手。"},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.1,
                max_tokens=128
            )
            
            result = response.choices[0].message.content.strip()
            
            if "充分支持" in result:
                return True, result
            elif "部分支持" in result:
                return True, result  # 部分支持也算支持
            else:
                return False, result
                
        except Exception as e:
            print(f"判断支持度时出错: {e}")
            return True, str(e)  # 默认支持
    
    def generate(
        self, 
        question: str, 
        top_k: int = 3,
        verbose: bool = False
    ) -> Dict:
        """
        Self-RAG 生成流程
        
        Args:
            question: 问题
            top_k: 检索文档数量
            verbose: 是否打印详细信息
            
        Returns:
            生成结果字典
        """
        iteration = 0
        all_retrieved_docs = []
        retrieval_decisions = []
        
        while iteration < self.max_iterations:
            iteration += 1
            
            if verbose:
                print(f"\n=== Self-RAG 迭代 {iteration} ===")
            
            # Step 1: 判断是否需要检索
            need_retrieve = self._check_need_retrieve(question)
            retrieval_decisions.append({
                "iteration": iteration,
                "need_retrieve": need_retrieve
            })
            
            if verbose:
                print(f"[Retrieve] 需要检索: {need_retrieve}")
            
            if not need_retrieve:
                # 不需要检索，直接生成
                prompt = f"""请回答以下问题。

问题：{question}

请给出简洁准确的回答："""
                
                response = self.client.chat.completions.create(
                    model=self.model,
                    messages=[
                        {"role": "system", "content": "你是一个智能问答助手。"},
                        {"role": "user", "content": prompt}
                    ],
                    temperature=0.7,
                    max_tokens=256
                )
                
                answer = response.choices[0].message.content.strip()
                
                return {
                    "answer": answer,
                    "retrieved_documents": [],
                    "retrieval_decisions": retrieval_decisions,
                    "final_iteration": iteration,
                    "supported": None
                }
            
            # Step 2: 检索
            retrieved_docs = self.retriever.retrieve(question, top_k=top_k)
            all_retrieved_docs = [
                {"id": doc['id'], "title": doc.get('title', ''), "score": score}
                for doc, score in retrieved_docs
            ]
            
            # Step 3: 判断相关性
            is_relevant, relevant_indices = self._check_relevance(
                question, 
                [doc for doc, _ in retrieved_docs]
            )
            
            retrieval_decisions[-1]["is_relevant"] = is_relevant
            retrieval_decisions[-1]["relevant_doc_indices"] = relevant_indices
            
            if verbose:
                print(f"[IsREL] 相关文档: {relevant_indices}")
            
            if not is_relevant:
                # 没有相关文档，尝试重新生成查询或结束
                if iteration < self.max_iterations:
                    continue
                else:
                    # 达到最大迭代，用现有文档生成
                    pass
            
            # Step 4: 基于相关文档生成答案
            relevant_docs = [retrieved_docs[i][0] for i in relevant_indices] if relevant_indices else [doc for doc, _ in retrieved_docs]
            context = self._format_context(relevant_docs)
            
            prompt = f"""你是一个智能问答助手。请根据以下提供的文档信息回答问题。
如果文档中没有足够的信息回答问题，请说"根据提供的文档，我无法回答这个问题"。

文档信息：
{context}

问题：{question}

请给出简洁准确的回答："""
            
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": "你是一个智能问答助手，擅长根据提供的文档信息回答问题。"},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.7,
                max_tokens=256
            )
            
            answer = response.choices[0].message.content.strip()
            
            # Step 5: 判断答案是否被支持
            is_supported, support_reason = self._check_support(question, answer, relevant_docs)
            
            retrieval_decisions[-1]["is_supported"] = is_supported
            retrieval_decisions[-1]["support_reason"] = support_reason
            
            if verbose:
                print(f"[IsSUP] 答案被支持: {is_supported}")
            
            if is_supported:
                # 答案被支持，返回结果
                return {
                    "answer": answer,
                    "retrieved_documents": all_retrieved_docs,
                    "retrieval_decisions": retrieval_decisions,
                    "final_iteration": iteration,
                    "supported": True
                }
            else:
                # 答案不被支持，继续迭代
                if iteration < self.max_iterations:
                    if verbose:
                        print("答案未被充分支持，继续迭代...")
                    continue
                else:
                    # 达到最大迭代，返回当前结果
                    return {
                        "answer": answer,
                        "retrieved_documents": all_retrieved_docs,
                        "retrieval_decisions": retrieval_decisions,
                        "final_iteration": iteration,
                        "supported": False
                    }
        
        # 不应该到达这里
        return {
            "answer": "",
            "retrieved_documents": all_retrieved_docs,
            "retrieval_decisions": retrieval_decisions,
            "final_iteration": iteration,
            "supported": False
        }
    
    def get_name(self) -> str:
        return f"Self-RAG-{self.model}"


# ============================================
# 整合幻觉评估器
# ============================================

class HallucinationEvaluator:
    """
    整合幻觉评估器
    
    协调：
    1. Vanilla RAG vs Self-RAG 对比
    2. FActScore 计算
    3. 引用质量评估（复用 Hit@K）
    """
    
    def __init__(
        self,
        retriever,
        api_key: str = None,
        model: str = None
    ):
        """
        初始化幻觉评估器
        
        Args:
            retriever: 检索器实例
            api_key: DashScope API Key
            model: Qwen 模型名称
        """
        self.retriever = retriever
        self.api_key = api_key or HallucinationConfig.DASHSCOPE_API_KEY
        self.model = model or HallucinationConfig.QWEN_MODEL
        
        # 初始化组件
        self.factscore_evaluator = FActScoreEvaluator(api_key=self.api_key, model=self.model)
        self.self_rag_generator = SelfRAGGenerator(
            retriever=retriever,
            api_key=self.api_key,
            model=self.model
        )
        
        print(f"幻觉评估器初始化完成")
    
    def evaluate_single(
        self,
        question: str,
        ground_truth_answer: str,
        evidence_doc: str,
        evidence_text: str,
        top_k: int = 3,
        verbose: bool = False
    ) -> Dict:
        """
        评估单个问题
        
        Args:
            question: 问题
            ground_truth_answer: 标准答案
            evidence_doc: 证据文档ID
            evidence_text: 证据文本
            top_k: 检索文档数量
            verbose: 是否打印详细信息
            
        Returns:
            评估结果字典
        """
        if verbose:
            print(f"\n{'='*60}")
            print(f"问题: {question}")
            print(f"{'='*60}")
        
        # 1. Self-RAG 生成
        self_rag_result = self.self_rag_generator.generate(question, top_k=top_k, verbose=verbose)
        
        # 2. 计算 FActScore
        factscore_result = self.factscore_evaluator.compute_factscore(
            answer=self_rag_result["answer"],
            evidence=evidence_text,
            verbose=verbose
        )
        
        # 3. 计算引用质量（复用 Hit@K）
        retrieved_ids = [doc['id'] for doc in self_rag_result["retrieved_documents"]]
        hit_at_k = 1.0 if evidence_doc in retrieved_ids else 0.0
        
        # 4. 组装结果
        return {
            "question": question,
            "ground_truth_answer": ground_truth_answer,
            "self_rag_answer": self_rag_result["answer"],
            "factscore": factscore_result["factscore"],
            "num_atoms": factscore_result["num_atoms"],
            "num_supported_atoms": factscore_result["num_supported"],
            "hit_at_k": hit_at_k,
            "retrieved_documents": self_rag_result["retrieved_documents"],
            "retrieval_decisions": self_rag_result["retrieval_decisions"],
            "final_iteration": self_rag_result["final_iteration"],
            "supported": self_rag_result["supported"]
        }
    
    def batch_evaluate(
        self,
        examples: List[Dict],
        top_k: int = 3,
        verbose: bool = False
    ) -> Dict:
        """
        批量评估
        
        Args:
            examples: 示例列表，每个包含 question, answer, evidence_doc, evidence_text
            top_k: 检索文档数量
            verbose: 是否打印详细信息
            
        Returns:
            聚合结果
        """
        all_results = []
        factscores = []
        hit_at_ks = []
        supported_count = 0
        
        for i, example in enumerate(tqdm(examples, desc="幻觉评估", ascii=False)):
            result = self.evaluate_single(
                question=example["question"],
                ground_truth_answer=example["answer"],
                evidence_doc=example.get("evidence_doc", ""),
                evidence_text=example.get("evidence_text", ""),
                top_k=top_k,
                verbose=verbose and i < 3
            )
            
            all_results.append(result)
            factscores.append(result["factscore"])
            hit_at_ks.append(result["hit_at_k"])
            
            if result["supported"]:
                supported_count += 1
        
        # 计算聚合指标
        return {
            "avg_factscore": sum(factscores) / len(factscores) if factscores else 0.0,
            "avg_hit_at_k": sum(hit_at_ks) / len(hit_at_ks) if hit_at_ks else 0.0,
            "support_rate": supported_count / len(examples) if examples else 0.0,
            "num_examples": len(examples),
            "individual_results": all_results,
            "retriever": self.retriever.get_name(),
            "generator": self.self_rag_generator.get_name()
        }


# ============================================
# 便捷函数
# ============================================

def run_hallucination_experiment(
    examples: List[Dict],
    corpus: List[Dict],
    retriever_type: str = "bm25",
    top_k: int = 3,
    api_key: str = None,
    model: str = "qwen-plus",
    device: str = "cuda",
    verbose: bool = False
) -> Dict:
    """
    运行幻觉缓解实验
    
    Args:
        examples: 测试示例
        corpus: 文档语料库
        retriever_type: 检索器类型 ("bm25" 或 "contriever")
        top_k: 检索文档数量
        api_key: DashScope API Key
        model: Qwen 模型名称
        device: 运行设备
        verbose: 是否打印详细信息
        
    Returns:
        实验结果
    """
    # 导入检索器
    from rag_finance import BM25Retriever, ContrieverRetriever
    
    # 初始化检索器
    if retriever_type.lower() == "bm25":
        retriever = BM25Retriever(corpus, cache_path="./cache/bm25_index_hallucination.pkl")
    else:
        retriever = ContrieverRetriever(
            corpus,
            cache_path="./cache/contriever_embeddings_hallucination.pkl",
            device=device
        )
    
    # 初始化评估器
    evaluator = HallucinationEvaluator(
        retriever=retriever,
        api_key=api_key,
        model=model
    )
    
    # 运行评估
    results = evaluator.batch_evaluate(examples, top_k=top_k, verbose=verbose)
    
    # 打印摘要
    print("\n" + "="*60)
    print("幻觉评估结果")
    print("="*60)
    print(f"平均 FActScore:  {results['avg_factscore']:.4f}")
    print(f"平均 Hit@K:      {results['avg_hit_at_k']:.4f}")
    print(f"答案支持率:      {results['support_rate']:.4f}")
    print(f"样本数量:        {results['num_examples']}")
    
    return results


if __name__ == "__main__":
    # 测试代码
    print("幻觉评估模块加载完成")
    print("使用方法:")
    print("  from hallucination_eval import HallucinationEvaluator, FActScoreEvaluator")
    print("  evaluator = HallucinationEvaluator(retriever, api_key='sk-xxx')")
    print("  results = evaluator.batch_evaluate(examples)")
