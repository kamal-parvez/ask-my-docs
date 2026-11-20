"""Chatbot configuration, loaded from environment variables or a .env file, with default values."""
import os
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

MODEL = os.getenv("CHATBOT_MODEL", "qwen3.5:4b")
HOST = os.getenv("OLLAMA_HOST", "http://localhost:11434")
SYSTEM_PROMPT = os.getenv(
    "CHATBOT_SYSTEM_PROMPT",
    "You are a helpful, concise assistant. Keep answers short unless asked for detail.",
)
TEMPERATURE = float(os.getenv("CHATBOT_TEMPERATURE", "0.7"))
MAX_TURNS = int(os.getenv("CHATBOT_MAX_TURNS", "10"))  # last N user/assistant exchanges kept
# Context window in tokens. Ollama's default may be as low as 4096, too small once retrieved chunks are added
NUM_CTX = int(os.getenv("CHATBOT_NUM_CTX", "8192"))
THINK = os.getenv("CHATBOT_THINK", "false").lower() in {"1", "true", "yes"}

EMBED_MODEL = os.getenv("CHATBOT_EMBED_MODEL", "nomic-embed-text")  # used by RAG to embed text
# RAG chunking, both in characters (the overlap is ~10% of the chunk size)
CHUNK_SIZE = int(os.getenv("CHATBOT_CHUNK_SIZE", "1200"))
CHUNK_OVERLAP = int(os.getenv("CHATBOT_CHUNK_OVERLAP", "120"))

# RAG retrieval: how many chunks to fetch, and the minimum cosine similarity (0-1) to use one
RAG_TOP_K = int(os.getenv("CHATBOT_RAG_TOP_K", "4"))
RAG_MIN_SCORE = float(os.getenv("CHATBOT_RAG_MIN_SCORE", "0.62"))  # tuned on the handbook, see plan.md
# Hybrid search: share of the semantic (vector) part vs the keyword part (1 = vector only, 0 = keyword only)
RAG_ALPHA = float(os.getenv("CHATBOT_RAG_ALPHA", "0.6"))

# Weaviate vector store
WEAVIATE_HOST = os.getenv("CHATBOT_WEAVIATE_HOST", "localhost")
WEAVIATE_PORT = int(os.getenv("CHATBOT_WEAVIATE_PORT", "8080"))
WEAVIATE_GRPC_PORT = int(os.getenv("CHATBOT_WEAVIATE_GRPC_PORT", "50051"))

# Where source documents are copied to when ingested
DOCUMENTS_DIR = Path(os.getenv("CHATBOT_DOCUMENTS_DIR", Path(__file__).parent / "documents"))

# Logging
LOG_LEVEL = os.getenv("CHATBOT_LOG_LEVEL", "INFO")
LOG_DIR = Path(os.getenv("CHATBOT_LOG_DIR", Path(__file__).parent / "logs"))

# Where saved chats live (one JSON file each); defaults to ./conversations next to this file
CONVERSATIONS_DIR = Path(
    os.getenv("CHATBOT_CONVERSATIONS_DIR", Path(__file__).parent / "conversations")
)
