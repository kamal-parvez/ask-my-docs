"""RAG: ingest documents into a vector store and retrieve from it

documents.py  - read files and clean them
chunking.py   - split text into overlapping chunks
embeddings.py - text to vectors (Ollama)
store.py      - Weaviate storage and vector search
retrieval.py  - find the chunks relevant to a question
prompting.py  - put the chunks into the message sent to the model
"""
import shutil
from pathlib import Path

import numpy as np

import config
from chat import OllamaUnavailable
from .chunking import Chunk, chunk_file, split_text
from .documents import SUPPORTED_TYPES, DocumentError
from .embeddings import EmbeddingModelMissing, embed_documents, embed_query
from .prompting import build_prompt, visible_sources
from .retrieval import Retrieval, retrieve, retrieve_with_prompt
from .store import Hit, VectorStoreUnavailable, delete_chunks, list_documents, replace_document

EMBED_BATCH = 32  # chunks embedded per Ollama call


def ingest(path):
    """Copy a file into documents/, split and embed it, and store it.
    Re-ingesting a file replaces its old chunks. Returns the number of chunks stored.
    """
    path = Path(path)
    if not path.is_file():
        raise DocumentError(f"File not found: {path}")
    chunks = chunk_file(path)  # validates the file before anything is copied or stored
    vectors = np.vstack([
        embed_documents([c.text for c in chunks[i:i + EMBED_BATCH]])
        for i in range(0, len(chunks), EMBED_BATCH)
    ])

    docs_dir = config.DOCUMENTS_DIR
    docs_dir.mkdir(parents=True, exist_ok=True)
    target = docs_dir / path.name
    if not (target.exists() and target.samefile(path)):  # skip if it's already there
        shutil.copyfile(path, target)

    replace_document(path.name, chunks, vectors)
    return len(chunks)


def delete_document(name):
    """Remove a document's chunks from the store, and its copy in documents/."""
    delete_chunks(name)
    (config.DOCUMENTS_DIR / name).unlink(missing_ok=True)


# Everything ingest / list / delete can raise, for callers that show a message instead of crashing
RAG_ERRORS = (DocumentError, VectorStoreUnavailable, EmbeddingModelMissing, OllamaUnavailable)


def error_message(e):
    """A short, user-facing message for one of RAG_ERRORS."""
    if isinstance(e, VectorStoreUnavailable):
        return "Document store (Weaviate) isn't running. Start it with `docker compose up -d`."
    if isinstance(e, EmbeddingModelMissing):
        return f"Embedding model {e} isn't installed. Run `ollama pull {e}`."
    if isinstance(e, OllamaUnavailable):
        return "Can't reach Ollama. Is it running? (try `ollama serve`)"
    return str(e)
