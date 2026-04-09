import string, re
from collections import Counter

def normalise(text):
    """Lowercase + remove punctuation + collapse whitespace."""
    text = text.lower()
    text = text.translate(str.maketrans('', '', string.punctuation))
    return ' '.join(text.split())


# ── STAGE 1: RETRIEVAL METRICS (no LLM)
def retrieval_em(retrieved_docs, gold_answers):
    """1 if any retrieved chunk CONTAINS the exact gold answer span."""
    for doc in retrieved_docs:
        chunk_norm = normalise(doc.page_content)
        for gold in gold_answers:
            if normalise(gold) in chunk_norm:
                return 1
    return 0


def recall_at_k(retrieved_docs, gold_answer_text):
    """1.0 if any of the k retrieved chunks contains the gold span."""
    gold_norm = normalise(gold_answer_text)
    for doc in retrieved_docs:
        if gold_norm in normalise(doc.page_content):
            return 1.0
    return 0.0


# ── STAGE 2: GENERATION METRICS (LLM output)
def generation_em(prediction, gold_answers):
    """1 if LLM prediction exactly matches any gold answer (normalised)."""
    pred_norm = normalise(prediction)
    return int(any(normalise(a) == pred_norm for a in gold_answers))


def generation_f1(prediction, gold_answers):
    """Best token-level F1 between LLM prediction and gold answers."""
    def _f1(pred, gold):
        pred_tokens = normalise(pred).split()
        gold_tokens = normalise(gold).split()
        common      = Counter(pred_tokens) & Counter(gold_tokens)
        n_common    = sum(common.values())
        if n_common == 0:
            return 0.0
        precision = n_common / len(pred_tokens)
        recall    = n_common / len(gold_tokens)
        return 2 * precision * recall / (precision + recall)
    return max(_f1(prediction, a) for a in gold_answers) if gold_answers else 0.0


# ── FActScore (lightweight)
def factscore_fast(prediction, retrieved_docs, model, tokenizer, max_claims=3):
    """Fraction of atomic claims in prediction supported by retrieved chunks."""
    if not prediction.strip() or not retrieved_docs:
        return 0.0, 0

    decompose_prompt = (
        f'List up to {max_claims} short factual claims from this text.\n'
        f'Output ONLY a numbered list.\n\nText: {prediction[:300]}\n\nClaims:\n1.'
    )
    inputs = tokenizer(decompose_prompt, return_tensors='pt',
                       truncation=True, max_length=512).to(model.device)
    with torch.no_grad():
        out = model.generate(**inputs, max_new_tokens=64,
                             do_sample=False, pad_token_id=tokenizer.eos_token_id)
    raw = tokenizer.decode(out[0][inputs['input_ids'].shape[1]:],
                           skip_special_tokens=True).strip()
    claims = []
    for line in raw.split('\n'):
        c = re.sub(r'^[\d]+[.)\s]+', '', line.strip())
        if len(c) > 10:
            claims.append(c)
        if len(claims) >= max_claims:
            break
    if not claims:
        claims = [prediction[:200]]

    context   = '\n'.join([f'[C{i+1}]: {d.page_content[:300]}'
                             for i, d in enumerate(retrieved_docs)])
    supported = 0
    for claim in claims:
        verify_prompt = (
            f'Context:\n{context}\n\n'
            f'Is this claim supported by the context above?\n'
            f'Claim: {claim}\nAnswer (yes/no):'
        )
        inputs = tokenizer(verify_prompt, return_tensors='pt',
                           truncation=True, max_length=1024).to(model.device)
        with torch.no_grad():
            out = model.generate(**inputs, max_new_tokens=5,
                                 do_sample=False, pad_token_id=tokenizer.eos_token_id)
        verdict = tokenizer.decode(out[0][inputs['input_ids'].shape[1]:],
                                   skip_special_tokens=True).strip().lower()
        if 'yes' in verdict or 'support' in verdict:
            supported += 1

    return supported / len(claims), len(claims)

# ── Citation Precision (Traditional RAG)
def citation_precision(answer_text, retrieved_docs):
    citations = re.findall(r'\[Source (\d+)\]', answer_text)
    if not citations: return None
    stopwords = {'the','a','an','is','are','was','were','of','in','to',
                 'and','or','that','this','it','for','on','with','by'}
    grounded = 0
    for cite_num in citations:
        idx = int(cite_num) - 1
        if idx >= len(retrieved_docs): continue
        pat = r'([^.!?]{0,150}\[Source ' + cite_num + r'\][^.!?]{0,150})'
        matches = re.findall(pat, answer_text)
        if not matches: continue
        claim_tok = set(normalise(matches[0]).split()) - stopwords
        chunk_tok = set(normalise(retrieved_docs[idx].page_content).split()) - stopwords
        if claim_tok and len(claim_tok & chunk_tok)/len(claim_tok) > 0.15:
            grounded += 1
    return grounded / len(citations)



# ── Sanity checks
gold    = ['State of Arizona']
chunk_y = Document(page_content='governed by the laws of the State of Arizona')
chunk_n = Document(page_content='parties agree to arbitration in California')
print('✅ Metrics ready')
print(f'   retrieval_em([chunk WITH gold])  : {retrieval_em([chunk_y], gold)} ← 1 expected')
print(f'   retrieval_em([chunk WITHOUT gold]): {retrieval_em([chunk_n], gold)} ← 0 expected')
print(f'   generation_em(exact match)        : {generation_em("State of Arizona", gold)} ← 1 expected')
print(f'   generation_f1(partial match)      : {generation_f1("The law is Arizona", gold):.3f}')