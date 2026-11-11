"""Saves and loads conversations as JSON files, one file per chat"""
import json
import os
import re
from dataclasses import dataclass
from datetime import datetime

import config

_ID_RE = re.compile(r"[\w-]+")  # ids become file names, so allow only safe characters
TITLE_LENGTH = 50


class ChatNotFound(Exception):
    """Saved chat doesn't exist or can't be read."""


@dataclass
class ChatSummary:
    """What a chat list needs: id, title, last update and model."""
    id: str
    title: str
    updated: str
    model: str


def _path(chat_id):
    if not _ID_RE.fullmatch(chat_id):
        raise ChatNotFound(chat_id)
    return config.CONVERSATIONS_DIR / f"{chat_id}.json"


def _title(messages):
    """First user message, shortened."""
    for m in messages:
        if m["role"] == "user":
            text = " ".join(m["content"].split())
            return text[:TITLE_LENGTH] + ("…" if len(text) > TITLE_LENGTH else "")
    return "New chat"


def exists(chat_id):
    return _path(chat_id).is_file()


def save(chat):
    """Write the chat to disk. Returns False if there's nothing worth saving yet."""
    messages = chat.history[1:]  # the system prompt isn't stored; the current one is used on load
    if not any(m["role"] == "assistant" for m in messages):
        return False
    data = {
        "id": chat.id,
        "title": _title(messages),
        "model": chat.model,
        "created": chat.created,
        "updated": datetime.now().astimezone().isoformat(timespec="seconds"),
        "messages": messages,
    }
    path = _path(chat.id)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")
    os.replace(tmp, path)  # atomic: a crash never leaves a half-written chat
    return True


def load(chat_id):
    """Return the saved chat"""
    try:
        data = json.loads(_path(chat_id).read_text(encoding="utf-8"))
    except (OSError, ValueError) as e:
        raise ChatNotFound(chat_id) from e
    if not isinstance(data, dict) or "id" not in data or "messages" not in data:
        raise ChatNotFound(chat_id)
    return data


def list_chats():
    """Saved chats, most recently updated first"""
    if not config.CONVERSATIONS_DIR.is_dir():
        return []
    chats = []
    for path in config.CONVERSATIONS_DIR.glob("*.json"):
        try:
            d = load(path.stem)
            chats.append(ChatSummary(d["id"], d.get("title", "New chat"),
                                     d.get("updated", ""), d.get("model", "")))
        except ChatNotFound:
            continue
    return sorted(chats, key=lambda c: (c.updated, c.id), reverse=True)


def delete(chat_id):
    """Permanently remove a saved chat."""
    _path(chat_id).unlink(missing_ok=True)
