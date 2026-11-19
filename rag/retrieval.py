"""Retrieval: find the chunks relevant to a question"""
from dataclasses import dataclass, field

import logging

import config
from .embeddings import EmbeddingModelMissing, embed_query
from .prompting import build_prompt
from .store import VectorStoreUnavailable, search

log = logging.getLogger("chatbot.rag")

def retrieve(query, k=None, min_score=None):
    """Chunks most similar to the query, best first, keeping only those scoring at least `min_score`.
    Returns a list of Hit (chunk + score) or empty if nothing is relevant (or nothing is ingested).
    """
    k = config.RAG_TOP_K if k is None else k
    min_score = config.RAG_MIN_SCORE if min_score is None else min_score
    found = search(embed_query(query), k)
    for h in found:
        status = "kept" if h.score >= min_score else "dropped"
        where = h.chunk.source if h.chunk.page is None else f"{h.chunk.source} p.{h.chunk.page}"
        log.debug(f"Chunk {status} | score {h.score:.2f} | {where}\n{h.chunk.text}\n")
    return [hit for hit in found if hit.score >= min_score]


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
