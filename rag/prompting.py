"""Prompting: turn a question and its retrieved chunks into the message sent to the model"""

INSTRUCTIONS = (
    "Use the context below if it helps answer the question. If it is not relevant, ignore it and answer normally "
    "from your own knowledge. Do not mention that you were given context. Keep the answer short."
)


def _label(chunk):
    """'file.pdf, page 5' / 'file.pdf, pages 5-6' / 'notes.txt' (no pages)"""
    if chunk.page is None:
        return chunk.source
    if chunk.page_end in (None, chunk.page):
        return f"{chunk.source}, page {chunk.page}"
    return f"{chunk.source}, pages {chunk.page}-{chunk.page_end}"


def build_prompt(question, hits):
    """The user message to send: the question with the retrieved context.
    With no hits the question is returned unchanged, so it's a normal chat message.
    """
    if not hits:
        return question
    context = "\n\n".join(f"[{_label(h.chunk)}]\n{h.chunk.text}" for h in hits)
    return f"{INSTRUCTIONS}\n\nContext:\n{context}\n\nQuestion: {question}"


def visible_sources(message):
    """Sources to show under a saved assistant message (saved only when retrieval was confident)."""
    return message.get("sources", [])
