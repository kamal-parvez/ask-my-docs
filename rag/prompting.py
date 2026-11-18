"""Prompting: turn a question and its retrieved chunks into the message sent to the model"""

INSTRUCTIONS = (
    "Use only facts stated in the context below to answer the question; do not guess. "
    "If the context does not clearly answer it, reply exactly: \"I couldn't find that in the documents.\" "
    "Keep the answer short. Do not mention the context or cite sources."
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
