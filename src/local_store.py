"""Shared local helpers for the individual RAG tasks.

The assignment recommends hosted/vector services, but the test environment should
also run offline. These helpers keep a small local corpus built from markdown.
"""

from __future__ import annotations

import math
import re
from pathlib import Path


PROJECT_DIR = Path(__file__).parent.parent
STANDARDIZED_DIR = PROJECT_DIR / "data" / "standardized"


def tokenize(text: str) -> list[str]:
    return re.findall(r"[\wÀ-ỹ]+", text.lower(), flags=re.UNICODE)


def load_markdown_documents() -> list[dict]:
    documents: list[dict] = []
    for md_file in sorted(STANDARDIZED_DIR.rglob("*.md")):
        if md_file.name.startswith("."):
            continue
        content = md_file.read_text(encoding="utf-8").strip()
        if not content:
            continue
        doc_type = "legal" if "legal" in md_file.parts else "news"
        documents.append(
            {
                "content": content,
                "metadata": {
                    "source": md_file.stem,
                    "filename": md_file.name,
                    "path": str(md_file.relative_to(PROJECT_DIR)),
                    "type": doc_type,
                },
            }
        )
    return documents


def cosine_sparse(left: dict[str, float], right: dict[str, float]) -> float:
    if not left or not right:
        return 0.0
    dot = sum(value * right.get(term, 0.0) for term, value in left.items())
    left_norm = math.sqrt(sum(value * value for value in left.values()))
    right_norm = math.sqrt(sum(value * value for value in right.values()))
    if left_norm == 0 or right_norm == 0:
        return 0.0
    return dot / (left_norm * right_norm)


def source_label(metadata: dict) -> str:
    source = metadata.get("source") or metadata.get("filename") or "unknown-source"
    year_match = re.search(r"(20\d{2}|19\d{2})", source)
    if year_match:
        return f"{source}, {year_match.group(1)}"
    if metadata.get("type") == "news":
        return f"{source}, 2026"
    return str(source)
