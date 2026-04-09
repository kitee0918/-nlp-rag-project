import re
from collections import defaultdict


# utils
def normalize(text):
    if text is None:
        return ""
    text = text.lower()
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def to_pred_list(pred_answer):
    """
    unify prediction format -> List[{"text": str}]
    """
    if pred_answer is None:
        return []

    if isinstance(pred_answer, str):
        return [{"text": pred_answer}]

    if isinstance(pred_answer, list):
        if len(pred_answer) == 0:
            return []
        if isinstance(pred_answer[0], str):
            return [{"text": x} for x in pred_answer]
        return pred_answer

    return []


# document support check
def is_supported(text, documents):
    text = normalize(text)
    for doc in documents:
        if text in normalize(doc):
            return True
    return False


def split_claims(text):
    sents = re.split(r"[。.!?;]", text)
    return [s.strip() for s in sents if s.strip()]


def doc_contains_answer(doc_text, answers):
    doc_text = normalize(doc_text)
    for a in answers:
        if normalize(a) in doc_text:
            return True
    return False

# metrics
def hit_rate(preds, golds):
    if not preds or not golds:
        return 0

    preds = [normalize(p["text"]) for p in preds]
    golds = [normalize(g) for g in golds]

    for g in golds:
        for p in preds:
            if g in p or p in g:
                return 1
    return 0


def hallucination_rate(preds, documents):
    if not preds:
        return 0

    total_claims = 0
    bad_claims = 0

    for p in preds:
        claims = split_claims(p["text"])
        total_claims += len(claims)

        for c in claims:
            if not is_supported(c, documents):
                bad_claims += 1

    return bad_claims / max(total_claims, 1)


def factscore(preds, documents):
    if not preds:
        return 0

    total = 0
    supported = 0

    for p in preds:
        claims = split_claims(p["text"])
        total += len(claims)

        for c in claims:
            if is_supported(c, documents):
                supported += 1

    return supported / max(total, 1)


def citation_metrics(pred_docs, gold_docs, gold_answers):
    """
    pred_docs: model retrieved docs
    gold_docs: GT docs (context)
    gold_answers: GT answers
    """

    if not pred_docs:
        return 0, 0

    # Precision: 检索到的 doc 有多少 actually useful
    useful = 0
    for d in pred_docs:
        text = d.get("text", "")
        if doc_contains_answer(text, gold_answers):
            useful += 1

    precision = useful / len(pred_docs)

    # Recall: gold answer 是否至少被一个检索 doc 覆盖
    hit = 0
    for g in gold_answers:
        for d in pred_docs:
            if normalize(g) in normalize(d.get("text", "")):
                hit += 1
                break

    recall = hit / len(gold_answers) if gold_answers else 0

    return precision, recall


# grouping
def build_gold_index(dataset):
    gold_index = defaultdict(list)

    for item in dataset:
        gold_index[item["question"]].append(item)

    return gold_index


# main eval
def evaluate_all(dataset, model_outputs):

    gold_index = build_gold_index(dataset)

    results = {
        "hit_rate": 0,
        "citation_precision": 0,
        "citation_recall": 0,
        "hallucination_rate": 0,
        "factscore": 0,
        "retrieval_coverage": 0
    }

    n = 0

    for pred_obj in model_outputs:
        question = pred_obj["question"]

        if question not in gold_index:
            continue

        items = gold_index[question]
        n += 1

        # prediction
        pred_answer = pred_obj.get("pred_answer")
        pred_docs = pred_obj.get("docs", [])

        preds = to_pred_list(pred_answer)

        # merge gold
        gold_answers = []
        gold_docs = []

        for item in items:
            gold_answers.extend(item.get("answers", []))
            gold_docs.extend(item.get("documents", []))

        gold_answers = list(set(gold_answers))

        # metrics
        results["hit_rate"] += hit_rate(preds, gold_answers)

        cp, cr = citation_metrics(pred_docs, gold_docs, gold_answers)
        results["citation_precision"] += cp
        results["citation_recall"] += cr

        results["hallucination_rate"] += hallucination_rate(preds, gold_docs)

        results["factscore"] += factscore(preds, gold_docs)

    for k in results:
        results[k] /= max(n, 1)

    return results