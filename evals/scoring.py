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
JUDGE_TEMPLATE = """Grade the answer against the reference answer.

Rules:
1. CORRECT if the answer states the same key fact as the reference. Different wording, extra detail, or a more \
specific or more general name for the same thing are fine.
2. If the answer gives the reference fact together with other correct items, it is CORRECT.
3. Yes/no references: work out what fact the reference means for this question, then check the answer states the \
same fact. Judge the facts in the answer, not only its first word. For a question like "Did X not do Y?", \
check whether the answer agrees with the reference on whether X did Y.
4. INCORRECT if the answer gives a different or contradictory fact, is too vague to contain the fact, or says the \
information isn't available (unless it then states the reference fact).

Examples:
Question: Is a mouse bigger than a cat? | Reference: no | Answer: No, a mouse is much smaller than a cat. -> CORRECT
Question: Which planets have rings? | Reference: Saturn | Answer: Saturn, Jupiter, Uranus and Neptune all have rings. -> CORRECT
Question: Who runs each state? | Reference: each state's governor | Answer: State governors run the states. -> CORRECT
Question: What share of people are farmers? | Reference: two-thirds | Answer: About one-third are farmers. -> INCORRECT

Now grade this one:
Question: {question}
Reference: {ground_truth}
Answer: {answer}

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
