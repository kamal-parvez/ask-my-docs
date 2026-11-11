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
THINK = os.getenv("CHATBOT_THINK", "false").lower() in {"1", "true", "yes"}
# Where saved chats live (one JSON file each); defaults to ./conversations next to this file
CONVERSATIONS_DIR = Path(
    os.getenv("CHATBOT_CONVERSATIONS_DIR", Path(__file__).parent / "conversations")
)
