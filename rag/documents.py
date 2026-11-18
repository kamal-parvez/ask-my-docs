"""Reading files: load pages of text from a PDF/txt/md file and clean them up"""
import re
from collections import Counter

from pypdf import PdfReader
from pypdf.errors import PyPdfError

SUPPORTED_TYPES = {".pdf", ".txt", ".md"}


class DocumentError(Exception):
    """A file can't be used: unsupported type, unreadable, or no extractable text."""



def load_pages(path):
    """Read a file as a list of (page number, text) for PDFs or (None, text) for .md or .txt"""
    suffix = path.suffix.lower()
    if suffix == ".pdf":
        try:
            reader = PdfReader(path)
            pages = [(n, page.extract_text() or "") for n, page in enumerate(reader.pages, 1)]
        except (PyPdfError, OSError) as e:
            raise DocumentError(f"Couldn't read {path.name}: {e}") from e
        if not any(text.strip() for _, text in pages):
            raise DocumentError(f"{path.name} has no extractable text (a scanned PDF?)")
        return pages
    if suffix in SUPPORTED_TYPES:
        try:
            return [(None, path.read_text(encoding="utf-8", errors="replace"))]
        except OSError as e:
            raise DocumentError(f"Couldn't read {path.name}: {e}") from e
    raise DocumentError(f"Unsupported file type '{suffix}' (supported: {', '.join(sorted(SUPPORTED_TYPES))})")


def _line_key(line):
    """A line with its numbers masked, so 'Page 5' and 'Page 6' count as the same line."""
    return re.sub(r"\d+", "#", line.strip())


def clean_pages(pages):
    """Cleanup: drop repeated headers/footers/page numbers, join hyphenated line breaks."""
    if len(pages) >= 4:
        counts = Counter()
        for _, text in pages:
            counts.update({_line_key(line) for line in text.splitlines() if 0 < len(line.strip()) < 100})
        threshold = max(3, int(len(pages) * 0.4))  # on at least 40% of the pages
        repeated = {key for key, n in counts.items() if n >= threshold}
        pages = [
            (n, "\n".join(line for line in text.splitlines() if _line_key(line) not in repeated))
            for n, text in pages
        ]
    cleaned = []
    for n, text in pages:
        text = re.sub(r"(?<=[a-z])-\n(?=[a-z])", "", text)  # "exam-\nple" -> "example"
        text = re.sub(r"[ \t]+", " ", text)
        text = re.sub(r"\n{3,}", "\n\n", text)
        cleaned.append((n, text.strip()))
    return cleaned
