"""提示词模板集中定义。"""

# 通用法律文档抽取式 QA 提示词
LEGAL_QA_PROMPT = """You are a STRICT extractive QA system for legal documents.

TASK:
Find answers to the QUESTION using ONLY the provided DOCUMENTS.

CRITICAL RULES:
- Answers MUST be copied from the DOCUMENTS.
- Answers can be spans or full sentences, but MUST appear verbatim in the DOCUMENTS.
- DO NOT rephrase, summarize, or explain.
- DO NOT use external knowledge.
- Each answer MUST be supported by at least one document.

CITATION RULES:
- Each answer MUST include the document ID it comes from.
- Use the format: (Doc X)

OUTPUT FORMAT (STRICT JSON):
{{
  "answers": [
    {{
      "text": "<copied span or sentence>",
      "doc_id": <document id>
    }}
  ]
}}

SPECIAL RULES:
- If no answer exists, return:
{{
  "answers": []
}}

- DO NOT output anything outside the JSON.
- DO NOT add explanations.

DOCUMENTS:
{context}

QUESTION:
{question}

OUTPUT:
"""

# FinanceBench 专用金融问答提示词 —— from guming shan
FINANCEBENCH_QA_PROMPT = """You are a financial question answering system.

TASK:
Answer the QUESTION using ONLY the provided DOCUMENTS.

CRITICAL RULES:
- You MUST base your answer ONLY on the DOCUMENTS.
- Do NOT use external knowledge.
- Do NOT guess or infer missing facts.
- If the documents do not contain enough information, return empty.

ANSWER GUIDELINES:

1. Yes/No questions:
   - Answer ONLY "Yes" or "No"
   - You may add ONE short supporting phrase from the documents

2. Numeric questions:
   - Return ONLY the number
   - Remove currency symbols, commas, and text
   - Do NOT add explanation

3. Comparison / reasoning questions:
   - Use ONLY information explicitly supported by documents
   - Keep answer short and direct
   - Do NOT invent financial metrics

4. All other questions:
   - Return a concise phrase or sentence from documents

CITATION RULE:
- Each answer MUST include the supporting document id

OUTPUT FORMAT (STRICT JSON):
{{
  "answers": [
    {{
      "text": "<final answer>",
      "doc_id": <document id>
    }}
  ]
}}

SPECIAL RULES:
- If answer cannot be directly supported by the documents:
{{
  "answers": []
}}

- DO NOT output anything outside JSON
- DO NOT include explanations, notes, or code

DOCUMENTS:
{context}

QUESTION:
{question}

OUTPUT:
"""

# 检索需求判断提示词（Self-RAG 风格）
NEED_RETRIEVE_PROMPT = """判断回答以下问题是否需要检索外部知识。

问题：{question}

请只回答"需要检索"或"不需要检索"。"""


# 导入 FinanceBench 提示词模板
def build_prompt_financebench(question, docs=None):
    if docs is None:
        context = "NO DOCUMENTS PROVIDED"
    else:
        context = "\n\n".join(
            [f"[Doc {d['id']}] {d['text']}" for d in docs]
        )
    
    # 替换模板中的占位符
    prompt = FINANCEBENCH_QA_PROMPT.format(
        context=context,
        question=question
    )
    return prompt




__all__ = [
    "LEGAL_QA_PROMPT",
    "FINANCEBENCH_QA_PROMPT",
    "NEED_RETRIEVE_PROMPT",
]
