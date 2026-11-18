"""The Weaviate vector store: connect, replace/list/delete documents' chunks, vector search"""
from contextlib import contextmanager
from dataclasses import dataclass
from typing import cast

import weaviate
from weaviate.classes.aggregate import GroupByAggregate
from weaviate.classes.config import Configure, DataType, Property, Tokenization, VectorDistances
from weaviate.classes.data import DataObject
from weaviate.classes.query import Filter, MetadataQuery
from weaviate.exceptions import WeaviateConnectionError

import config
from .chunking import Chunk

COLLECTION = "Chunk"


@dataclass
class Hit:
    """A search result: a chunk and its cosine similarity to the query (higher is closer)."""
    chunk: Chunk
    score: float


class VectorStoreUnavailable(Exception):
    """Weaviate isn't reachable (run `docker compose up -d`)."""


@contextmanager
def _store():
    """Connect to Weaviate and make sure the Chunk collection exists."""
    try:
        client = weaviate.connect_to_local(
            host=config.WEAVIATE_HOST, port=config.WEAVIATE_PORT, grpc_port=config.WEAVIATE_GRPC_PORT
        )
    except WeaviateConnectionError as e:
        raise VectorStoreUnavailable from e
    try:
        if not client.collections.exists(COLLECTION):
            client.collections.create(
                COLLECTION,
                vector_config=Configure.Vectors.self_provided(  # we supply our own vectors
                    vector_index_config=Configure.VectorIndex.hnsw(distance_metric=VectorDistances.COSINE)
                ),
                properties=[
                    Property(name="text", data_type=DataType.TEXT, skip_vectorization=True),
                    # FIELD: match the whole file name, not individual words
                    Property(name="source", data_type=DataType.TEXT, tokenization=Tokenization.FIELD),
                    Property(name="page", data_type=DataType.INT),
                    Property(name="page_end", data_type=DataType.INT),
                ],
            )
        yield client.collections.get(COLLECTION)
    finally:
        client.close()


def replace_document(name, chunks, vectors):
    """Store a document's chunks (with their vectors), replacing any earlier chunks of that file."""
    with _store() as collection:
        collection.data.delete_many(where=Filter.by_property("source").equal(name))
        collection.data.insert_many([
            DataObject(
                properties={"text": c.text, "source": c.source, "page": c.page, "page_end": c.page_end},
                vector=v.tolist(),
            )
            for c, v in zip(chunks, vectors)
        ])


def list_documents():
    """Ingested documents as {file name: number of chunks}."""
    with _store() as collection:
        result = collection.aggregate.over_all(group_by=GroupByAggregate(prop="source"))
        return {g.grouped_by.value: g.total_count for g in result.groups}


def delete_chunks(name):
    """Remove all stored chunks of a document."""
    with _store() as collection:
        collection.data.delete_many(where=Filter.by_property("source").equal(name))


def search(vector, k):
    """The k chunks closest to a query vector, best first (no relevance cutoff here)."""
    with _store() as collection:
        response = collection.query.near_vector(
            near_vector=vector.tolist(), limit=k, return_metadata=MetadataQuery(distance=True)
        )
        return [
            Hit(
                Chunk(
                    text=cast(str, o.properties["text"]),
                    source=cast(str, o.properties["source"]),
                    page=cast("int | None", o.properties["page"]),
                    page_end=cast("int | None", o.properties["page_end"]),
                ),
                1 - cast(float, o.metadata.distance),  # cosine distance
            )
            for o in response.objects
        ]
