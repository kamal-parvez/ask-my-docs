"""RAG building blocks: embeddings"""
import httpx
import numpy as np
import ollama

import config
from chat import OllamaUnavailable

# nomic-embed-text works best when told whether text is a document or a search query
DOC_PREFIX = "search_document: "
QUERY_PREFIX = "search_query: "


class EmbeddingModelMissing(Exception):
    """The embedding model isn't installed (run `ollama pull nomic-embed-text`)."""


def _embed(texts):
    """Embed texts, returns an array of vectors, one vector per text"""
    if not texts:
        return np.zeros((0, 0), dtype=np.float32)
    try:
        client = ollama.Client(host=config.HOST)
        response = client.embed(model=config.EMBED_MODEL, input=texts)
    except (ConnectionError, httpx.ConnectError) as e:
        raise OllamaUnavailable from e
    except ollama.ResponseError as e:
        if e.status_code == 404:
            raise EmbeddingModelMissing(config.EMBED_MODEL) from e
        raise
    vectors = np.array(response.embeddings, dtype=np.float32)
    # Unit length, so cosine similarity later is just a dot product
    return vectors / np.linalg.norm(vectors, axis=1, keepdims=True)


def embed_documents(texts):
    """Embed document chunks for storing"""
    return _embed([DOC_PREFIX + t for t in texts])


def embed_query(text):
    """Embed a user query"""
    return _embed([QUERY_PREFIX + text])[0]
