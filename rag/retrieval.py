"""Retrieval: find the chunks relevant to a question"""
import config
from .embeddings import embed_query
from .store import search


def retrieve(query, k=None, min_score=None):
    """Chunks most similar to the query, best first, keeping only those scoring at least `min_score`.
    Returns a list of Hit (chunk + score) or empty if nothing is relevant (or nothing is ingested).
    """
    k = config.RAG_TOP_K if k is None else k
    min_score = config.RAG_MIN_SCORE if min_score is None else min_score
    return [hit for hit in search(embed_query(query), k) if hit.score >= min_score]
