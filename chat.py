"""Ollama wrapper: holds the conversation state and talks to the model"""
import uuid
from dataclasses import dataclass
from datetime import datetime

import httpx
import ollama

import config


class OllamaUnavailable(Exception):
    """Ollama server can't be reached."""


class ModelNotFound(Exception):
    """Requested model isn't installed."""


@dataclass
class ModelInfo:
    """An installed Ollama model's name and size in GB."""
    name: str
    size_gb: float


class Chat:
    """One conversation: message history, current model, and streamed replies."""

    def __init__(self, model=config.MODEL, system_prompt=config.SYSTEM_PROMPT):
        self.client = ollama.Client(host=config.HOST)
        self.model = model
        self.system_prompt = system_prompt
        self.history = []
        self.reset()

    def reset(self):
        """Start a new, empty chat"""
        now = datetime.now().astimezone()
        self.id = now.strftime("%Y%m%d-%H%M%S-") + uuid.uuid4().hex[:4]
        self.created = now.isoformat(timespec="seconds")
        self.history = [{"role": "system", "content": self.system_prompt}]

    def load(self, data):
        """Continue a saved chat"""
        self.reset()
        self.id = data["id"]
        self.created = data.get("created", self.created)
        self.model = data.get("model") or self.model
        self.history += data["messages"]

    def _window(self):
        """Messages sent to the model: system prompt + last MAX_TURNS exchanges + new user message"""
        system, rest = self.history[:1], self.history[1:]
        limit = config.MAX_TURNS * 2 + 1  # N full exchanges + the message just added
        while len(rest) > limit:
            rest = rest[2:]  # drop whole user/assistant pairs
        return system + rest

    def list_models(self):
        try:
            models = self.client.list().models
        except (ConnectionError, httpx.ConnectError) as e:
            raise OllamaUnavailable from e
        return [
            ModelInfo(m.model, (m.size or 0) / 1e9)
            for m in models
            if m.model  # skip entries without a name
        ]

    def set_model(self, name):
        """Switching the model"""
        installed = [m.name for m in self.list_models()]
        # Ollama treats a bare name like "qwen3.5" as "qwen3.5:latest"
        full = name if ":" in name else f"{name}:latest"
        if full not in installed:
            raise ModelNotFound(name)
        self.model = full
        return full

    def send(self, user_input):
        """Add a user message, yield reply chunks and record the full reply"""
        self.history.append({"role": "user", "content": user_input})
        reply = ""
        try:
            # Stream the response, the caller prints each chunk as it arrives
            # think = False disables reasoning mode for faster replies
            stream = self.client.chat(
                model=self.model,
                messages=self._window(),
                stream=True,
                think=config.THINK,
                options={"temperature": config.TEMPERATURE},
            )
            for chunk in stream:
                piece = chunk["message"]["content"]
                reply += piece    # accumulate the full reply for the chat history
                yield piece
        except (ConnectionError, httpx.ConnectError) as e:
            self.history.pop()
            raise OllamaUnavailable from e
        except BaseException:  # ResponseError, KeyboardInterrupt, generator close
            self.history.pop() # Remove last user msg if response failed to be consistent
            raise
        self.history.append({"role": "assistant", "content": reply})
