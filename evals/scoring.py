"""Scoring for the evaluation"""
import re
import string
from collections import Counter

YES_NO = {"yes", "no"}
MIN_WORD_RECALL = 0.8


def normalize(text):
    """SQuAD-style: lowercase, drop punctuation and articles (a/an/the), collapse whitespace."""
    text = text.lower()
    text = "".join(ch for ch in text if ch not in string.punctuation)
    text = re.sub(r"\b(a|an|the)\b", " ", text)
    return " ".join(text.split())


def is_yes_no(ground_truth):
    return normalize(ground_truth) in YES_NO


def word_recall(text, ground_truth):
    """Share of the ground truth's words that also appear in the text"""
    gt = Counter(normalize(ground_truth).split())
    if not gt:
        return 0.0
    found = sum((gt & Counter(normalize(text).split())).values())
    return found / sum(gt.values())


def contains(answer, ground_truth):
    """True if at least MIN_WORD_RECALL of the ground truth's words appear in the answer"""
    gt, ans = normalize(ground_truth), normalize(answer)
    if not gt:
        return False
    if gt in YES_NO:
        return ans.split()[:1] == [gt]
    return word_recall(answer, ground_truth) >= MIN_WORD_RECALL


def token_f1(answer, ground_truth):
    """Word-overlap F1 between answer and ground truth (SQuAD)."""
    ans, gt = normalize(answer).split(), normalize(ground_truth).split()
    common = sum((Counter(ans) & Counter(gt)).values())
    if not ans or not gt or common == 0:
        return 0.0
    precision, recall = common / len(ans), common / len(gt)
    return 2 * precision * recall / (precision + recall)


def retrieval_hit(chunk_texts, ground_truth):
    """True if any retrieved chunk contains at least MIN_WORD_RECALL of the ground truth's words"""
    if is_yes_no(ground_truth) or not normalize(ground_truth):
        return None
    return any(word_recall(t, ground_truth) >= MIN_WORD_RECALL for t in chunk_texts)


JUDGE_SYSTEM = "You are a strict grader. You compare an answer with a reference answer and reply with one word."

JUDGE_TEMPLATE = """Question: {question}
Reference answer: {ground_truth}
Answer to grade: {answer}

Is the answer to grade correct? It is CORRECT if it states the same key fact as the reference answer (different \
wording, more detail or a short explanation are fine). It is INCORRECT if it gives a different or contradictory \
fact, is vague, or says it doesn't know.
Reply with exactly one word: CORRECT or INCORRECT."""


def judge_messages(question, ground_truth, answer):
    return [
        {"role": "system", "content": JUDGE_SYSTEM},
        {"role": "user", "content": JUDGE_TEMPLATE.format(question=question, ground_truth=ground_truth, answer=answer)},
    ]


def parse_verdict(reply):
    """True for CORRECT, False for INCORRECT, None if the reply is neither."""
    word = re.sub(r"[^A-Z]", "", reply.strip().split()[0].upper()) if reply.strip() else ""
    if word == "INCORRECT":
        return False
    if word == "CORRECT":
        return True
    return None
