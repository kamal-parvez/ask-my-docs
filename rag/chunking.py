"""Splitting cleaned text into overlapping chunks that remember their file and page"""
from dataclasses import dataclass
from pathlib import Path

import config
from .documents import DocumentError, clean_pages, load_pages

# Split on the biggest boundary first, and fall back to smaller ones only when needed
SEPARATORS = ["\n\n", "\n", ". ", " "]


@dataclass
class Chunk:
    """A piece of a document, with where it came from (for citations)."""
    text: str
    source: str  # file name
    page: int | None  # first page the chunk touches (PDFs only)
    page_end: int | None  # last page the chunk touches (PDFs only)



def _pieces(text, size, separators):
    """Split text into pieces of at most `size` characters, each keeping its trailing separator"""
    if len(text) <= size:
        return [text]
    for i, sep in enumerate(separators):
        if sep in text:
            parts = text.split(sep)
            pieces = [p + sep for p in parts[:-1]] + [parts[-1]]
            result = []
            for piece in pieces:
                if piece:
                    result.extend(_pieces(piece, size, separators[i + 1:]))
            return result
    # No separator left (e.g. one huge unbroken string): cut at the size limit
    return [text[i:i + size] for i in range(0, len(text), size)]


def _merge(pieces, size, overlap):
    """Pack (text, page) pieces into chunks of at most `size` characters.
    Each new chunk starts with the last pieces of the previous one, up to `overlap` characters.
    Returns a list of chunks, each a list of (text, page) pieces.
    """
    chunks, current, length = [], [], 0
    for piece in pieces:
        if current and length + len(piece[0]) > size:
            chunks.append(current)
            keep, kept = [], 0
            for old in reversed(current):  # carry over the tail as overlap
                if kept + len(old[0]) > overlap:
                    break
                keep.insert(0, old)
                kept += len(old[0])
            current, length = keep, kept
            while current and length + len(piece[0]) > size:  # overlap must leave room
                length -= len(current.pop(0)[0])
        current.append(piece)
        length += len(piece[0])
    if current:
        chunks.append(current)
    return chunks


def split_text(text, size=None, overlap=None):
    """Recursively split plain text into chunks of at most `size` characters."""
    size = size or config.CHUNK_SIZE
    overlap = config.CHUNK_OVERLAP if overlap is None else overlap
    if not 0 <= overlap < size:
        raise ValueError("chunk overlap must be at least 0 and smaller than the chunk size")
    pieces = [(p, None) for p in _pieces(text, size, SEPARATORS)]
    chunks = ("".join(t for t, _ in group).strip() for group in _merge(pieces, size, overlap))
    return [c for c in chunks if c]


def chunk_file(path, size=None, overlap=None):
    """Load a .pdf/.txt/.md file and split it into chunks"""
    path = Path(path)
    size = size or config.CHUNK_SIZE
    overlap = config.CHUNK_OVERLAP if overlap is None else overlap
    if not 0 <= overlap < size:
        raise ValueError("chunk overlap must be at least 0 and smaller than the chunk size")

    pieces = []  # (text, page) for the whole document, so chunks can span page breaks
    for page, text in clean_pages(load_pages(path)):
        if text:
            pieces += [(p, page) for p in _pieces(text + "\n\n", size, SEPARATORS)]

    chunks = []
    for group in _merge(pieces, size, overlap):
        text = "".join(t for t, _ in group).strip()
        if text:
            chunks.append(Chunk(text, path.name, group[0][1], group[-1][1]))
    if not chunks:
        raise DocumentError(f"{path.name} has no text to index")
    return chunks
