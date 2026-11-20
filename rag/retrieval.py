"""Retrieval: find the chunks relevant to a question"""
from dataclasses import dataclass, field

import logging

import config
from .embeddings import EmbeddingModelMissing, embed_query
from .prompting import build_prompt
from .store import VectorStoreUnavailable, search

log = logging.getLogger("chatbot.rag")

def retrieve(query, k=None, min_score=None):
    """Chunks most relevant to the query (hybrid search), best first. A chunk is kept if its cosine similarity is at
    least `min_score`, or if the keyword search also matched it. Returns a list of Hit, or empty if nothing is relevant (or nothing is ingested).
    """
    k = config.RAG_TOP_K if k is None else k
    min_score = config.RAG_MIN_SCORE if min_score is None else min_score
    found = search(query, embed_query(query), k, config.RAG_ALPHA)
    kept = []
    for h in found:
        keep = h.score >= min_score or h.keyword
        found_by = "both" if h.semantic and h.keyword else "keyword" if h.keyword else "semantic"
        where = h.chunk.source if h.chunk.page is None else f"{h.chunk.source} p.{h.chunk.page}"
        log.debug(f"Chunk {'kept' if keep else 'dropped'} | score {h.score:.2f} | found by: {found_by} | {where}\n{h.chunk.text}\n")
        if keep:
            kept.append(h)
    return kept


@dataclass
class Retrieval:
    """What Chat needs for one question: the message to send, where it came from, any problem."""
    prompt: str  # the question, with context added when something relevant was found
    sources: list[str] = field(default_factory=list)  # e.g. ["handbook.pdf, p.5"]
    warning: str | None = None  # set when documents couldn't be searched


def _sources(hits):
    """One entry per file, with its pages from best to worst match: ["handbook.pdf, p.28-29, 5-6, 13", "notes.txt"]"""
    pages = {}  # file -> [page label]
    for h in hits:
        c = h.chunk
        found = pages.setdefault(c.source, [])
        if c.page is not None:
            found.append(str(c.page) if c.page_end in (None, c.page) else f"{c.page}-{c.page_end}")
    return [
        f"{name}, p." + ", ".join(dict.fromkeys(found)) if found else name
        for name, found in pages.items()
    ]


def retrieve_with_prompt(question):
    """Retrieve context for a question and build the message to send"""
    try:
        hits = retrieve(question)
    except VectorStoreUnavailable:
        log.warning("Weaviate isn't running; answering without documents")
        return Retrieval(question, warning="Document store (Weaviate) isn't running; answering without documents.")
    except EmbeddingModelMissing as e:
        log.warning("Embedding model %s isn't installed; answering without documents", e)
        return Retrieval(question, warning=f"Embedding model {e} isn't installed; answering without documents.")
    sources = _sources(hits)
    best_score = f"{hits[0].score:.2f}" if hits else "none"
    log.info(f"Question: {question} | chunks used: {len(hits)} | best score: {best_score} | sources: {sources}")
    return Retrieval(build_prompt(question, hits), sources)
