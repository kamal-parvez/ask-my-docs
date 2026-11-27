"""Evaluate the RAG pipeline on evals/questions.csv against evals/documents.txt

Run from the project root:
    python evals/run_eval.py                     # full run
    python evals/run_eval.py --limit 5           # quick check on the first 5 questions
    python evals/run_eval.py --no-baseline       # RAG answers only
    python evals/run_eval.py --reingest          # rebuild the eval collection (after changing chunk size/overlap)
    python evals/run_eval.py --rejudge evals/results/<file>.jsonl [--judge-model phi4:14b]
"""
import argparse
import csv
import json
import sys
import tempfile
import time
from datetime import datetime
from pathlib import Path

EVAL_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(EVAL_DIR.parent))  # the project root, for config / chat / rag

import httpx
import ollama

import config
import rag
from chat import Chat
from scoring import MIN_WORD_RECALL, contains, judge_messages, parse_verdict, retrieval_hit, token_f1

DOCUMENTS_FILE = EVAL_DIR / "documents.txt"
QUESTIONS_FILE = EVAL_DIR / "questions.csv"
RESULTS_DIR = EVAL_DIR / "results"
EVAL_COLLECTION = "EvalChunk"


class EvalError(Exception):
    """Stops the run with a message (no traceback)."""


def configure():
    """Point the pipeline at the eval collection and make answers repeatable"""
    config.WEAVIATE_COLLECTION = EVAL_COLLECTION
    config.DOCUMENTS_DIR = Path(tempfile.mkdtemp(prefix="eval-docs-"))  # ingest's file copy stays out of documents/
    config.TEMPERATURE = 0.0


def settings(baseline, n_questions):
    return {
        "type": "settings",
        "started": datetime.now().astimezone().isoformat(timespec="seconds"),
        "model": config.MODEL,
        "judge_model": None,  # set in phase 2
        "embed_model": config.EMBED_MODEL,
        "chunk_size": config.CHUNK_SIZE,
        "chunk_overlap": config.CHUNK_OVERLAP,
        "top_k": config.RAG_TOP_K,
        "min_score": config.RAG_MIN_SCORE,
        "alpha": config.RAG_ALPHA,
        "num_ctx": config.NUM_CTX,
        "temperature": config.TEMPERATURE,
        "think": config.THINK,
        "system_prompt": config.SYSTEM_PROMPT,
        "collection": config.WEAVIATE_COLLECTION,
        "questions": n_questions,
        "baseline": baseline,
    }


def check_models(client, names):
    """Fail early if Ollama is down or a model isn't installed"""
    try:
        installed = {m.model for m in client.list().models}
    except (ConnectionError, httpx.ConnectError) as e:
        raise EvalError("Can't reach Ollama. Is it running? (try `ollama serve`)") from e
    missing = [n for n in names if (n if ":" in n else f"{n}:latest") not in installed]
    if missing:
        raise EvalError("Missing Ollama model(s): " + ", ".join(missing) + ". Install with `ollama pull <name>`.")


def load_questions(limit):
    with QUESTIONS_FILE.open(encoding="utf-8", newline="") as f:
        rows = [{"question": r["question"].strip(), "ground_truth": r["ground_truth"].strip()} for r in csv.DictReader(f)]
    return rows[:limit] if limit else rows


def ensure_ingested(reingest):
    docs = rag.list_documents()
    if docs and not reingest:
        print(f"Using collection {EVAL_COLLECTION}: {sum(n or 0 for n in docs.values())} chunks "
              f"(pass --reingest if chunk size/overlap changed)")
        return
    print(f"Ingesting {DOCUMENTS_FILE.name} into {EVAL_COLLECTION} (chunk size {config.CHUNK_SIZE}, "
          f"overlap {config.CHUNK_OVERLAP}); this takes a few minutes...")
    start = time.perf_counter()
    n = rag.ingest(DOCUMENTS_FILE)
    print(f"Ingested {n} chunks in {time.perf_counter() - start:.0f}s")


def ask(chat, question):
    """Send one question, return (answer, seconds)."""
    start = time.perf_counter()
    reply = "".join(chat.send(question))
    return reply.strip(), round(time.perf_counter() - start, 2)


def answer_all(questions, baseline, path):
    """Phase 1: answer every question and write each result as soon as it's ready"""
    captured = {}

    def retriever(question):
        found = rag.retrieve_with_prompt(question)
        if found.warning:  # the app falls back to plain chat; the evaluation must not
            raise EvalError(found.warning)
        captured["found"] = found
        return found

    with path.open("w", encoding="utf-8") as out:
        out.write(json.dumps(settings(baseline, len(questions)), ensure_ascii=False) + "\n")
        for i, q in enumerate(questions, 1):
            answer, seconds = ask(Chat(retriever=retriever), q["question"])  # fresh chat: no history between questions
            found = captured.pop("found")
            result = {
                "type": "result",
                "id": i,
                **q,
                "rag": {
                    "answer": answer,
                    "seconds": seconds,
                    "sources": found.sources,
                    "chunks": [
                        {"source": h.chunk.source, "score": round(h.score, 4), "semantic": h.semantic,
                         "keyword": h.keyword, "text": h.chunk.text}
                        for h in found.hits
                    ],
                },
            }
            if baseline:
                answer, seconds = ask(Chat(), q["question"])
                result["baseline"] = {"answer": answer, "seconds": seconds}
            out.write(json.dumps(result, ensure_ascii=False) + "\n")
            out.flush()
            base = f" | baseline {result['baseline']['seconds']:.1f}s" if baseline else ""
            print(f"[{i}/{len(questions)}] rag {result['rag']['seconds']:.1f}s, "
                  f"{len(found.hits)} chunks{base} | {q['question'][:70]}")


def read_results(path):
    # Split on "\n" only: splitlines() also splits on characters like \x85 and  , which json.dumps leaves
    # unescaped inside strings (ensure_ascii=False), and some chunks of documents.txt contain them
    text = path.read_text(encoding="utf-8")
    lines = [json.loads(line) for line in text.split("\n") if line.strip()]
    if not lines or lines[0].get("type") != "settings":
        raise EvalError(f"{path} isn't a results file (no settings line)")
    return lines[0], [r for r in lines[1:] if r.get("type") == "result"]


def unload(client, model):
    """Free the answer model's memory before the judge loads"""
    try:
        client.generate(model=model, keep_alive=0)
    except Exception:  # not loaded, or already gone: nothing to free
        pass


def judge(client, model, question, ground_truth, answer):
    response = client.chat(
        model=model,
        messages=judge_messages(question, ground_truth, answer),
        options={"temperature": 0, "num_ctx": 4096, "num_predict": 8},
    )
    reply = response["message"]["content"].strip()
    return parse_verdict(reply), reply


def score_all(client, run_settings, results, judge_model, path):
    """Phase 2: string metrics and judge verdicts for every answer"""
    unload(client, run_settings["model"])
    run_settings = {**run_settings, "judge_model": judge_model,
                    "judged": datetime.now().astimezone().isoformat(timespec="seconds")}
    for i, r in enumerate(results, 1):
        r["retrieval_hit"] = retrieval_hit([c["text"] for c in r["rag"]["chunks"]], r["ground_truth"])
        for mode in ("rag", "baseline"):
            if mode not in r:
                continue
            answer = r[mode]["answer"]
            verdict, reply = judge(client, judge_model, r["question"], r["ground_truth"], answer)
            r[mode]["score"] = {
                "judge": verdict,
                "judge_reply": reply,
                "contains": contains(answer, r["ground_truth"]),
                "f1": round(token_f1(answer, r["ground_truth"]), 4),
            }
        print(f"[{i}/{len(results)}] judged", end="\r")
    print()
    with path.open("w", encoding="utf-8") as out:
        for line in [run_settings, *results]:
            out.write(json.dumps(line, ensure_ascii=False) + "\n")
    return run_settings


def pct(part, whole):
    return f"{part}/{whole} ({100 * part / whole:.0f}%)" if whole else "n/a"


def summarize(run_settings, results):
    n = len(results)
    s = run_settings
    print(f"\n=== {n} questions | model {s['model']} | judge {s['judge_model']} | chunks {s['chunk_size']}/"
          f"{s['chunk_overlap']} | k {s['top_k']} | alpha {s['alpha']} | min score {s['min_score']} ===")

    hits = [r["retrieval_hit"] for r in results if r["retrieval_hit"] is not None]
    print(f"Retrieval hit (>={MIN_WORD_RECALL:.0%} of answer words in a retrieved chunk): {pct(sum(hits), len(hits))}  "
          f"[{n - len(hits)} yes/no questions not checked]")
    print(f"Sources shown: {pct(sum(bool(r['rag']['sources']) for r in results), n)}")

    for mode in ("rag", "baseline"):
        scored = [r[mode]["score"] for r in results if mode in r]
        if not scored:
            continue
        verdicts = [x["judge"] for x in scored]
        seconds = [r[mode]["seconds"] for r in results if mode in r]
        print(f"{mode:>8}: judge {pct(verdicts.count(True), len(scored))} | "
              f"contains {pct(sum(x['contains'] for x in scored), len(scored))} | "
              f"F1 {sum(x['f1'] for x in scored) / len(scored):.2f} | "
              f"avg {sum(seconds) / len(seconds):.1f}s | unparsed verdicts {verdicts.count(None)}")

    # Did RAG fail at retrieval or at answering?
    with_hit = [r for r in results if r["retrieval_hit"] is True]
    without = [r for r in results if r["retrieval_hit"] is False]
    print(f"RAG judge-correct when retrieval hit: {pct(sum(r['rag']['score']['judge'] is True for r in with_hit), len(with_hit))}"
          f" | when it missed: {pct(sum(r['rag']['score']['judge'] is True for r in without), len(without))}")

    disagree = [r for r in results
                if r["rag"]["score"]["judge"] is not None and r["rag"]["score"]["judge"] != r["rag"]["score"]["contains"]]
    if disagree:
        print(f"\nJudge and Contains disagree on {len(disagree)} RAG answers (spot-check these):")
        for r in disagree:
            verdict = "CORRECT" if r["rag"]["score"]["judge"] else "INCORRECT"
            print(f"  #{r['id']} judge {verdict} | Q: {r['question'][:60]} | truth: {r['ground_truth'][:40]} | "
                  f"answer: {r['rag']['answer'][:80]!r}")


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--limit", type=int, help="only the first N questions")
    parser.add_argument("--no-baseline", action="store_true", help="skip the answers without RAG")
    parser.add_argument("--reingest", action="store_true", help="rebuild the eval collection before answering")
    parser.add_argument("--rejudge", type=Path, metavar="RESULTS", help="score a saved run again (phase 2 only)")
    parser.add_argument("--judge-model", default=config.EVAL_JUDGE_MODEL, help="default: %(default)s")
    args = parser.parse_args()

    configure()
    client = ollama.Client(host=config.HOST)
    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    try:
        if args.rejudge:
            check_models(client, [args.judge_model])
            run_settings, results = read_results(args.rejudge)
            path = args.rejudge.with_name(f"{args.rejudge.stem}_rejudged-{stamp}.jsonl")
        else:
            baseline = not args.no_baseline
            check_models(client, [config.MODEL, config.EMBED_MODEL, args.judge_model])
            questions = load_questions(args.limit)
            ensure_ingested(args.reingest)
            RESULTS_DIR.mkdir(parents=True, exist_ok=True)
            path = RESULTS_DIR / f"{stamp}.jsonl"
            try:
                answer_all(questions, baseline, path)
            except KeyboardInterrupt:
                raise EvalError(f"Interrupted; answers so far are in {path} (score them with --rejudge)")
            run_settings, results = read_results(path)
        run_settings = score_all(client, run_settings, results, args.judge_model, path)
    except EvalError as e:
        sys.exit(str(e))
    except rag.RAG_ERRORS as e:
        sys.exit(rag.error_message(e))
    except ollama.ResponseError as e:
        sys.exit(f"Ollama error: {e.error}")
    summarize(run_settings, results)
    print(f"\nResults: {path}")


if __name__ == "__main__":
    main()
