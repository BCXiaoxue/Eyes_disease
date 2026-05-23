"""
Lightweight local RAG retrieval.

This module intentionally avoids HuggingFace, SentenceTransformer, and ChromaDB
at runtime. It reads knowledge/*.md and retrieves relevant chunks with a small
lexical scorer, so AI consultation cannot crash Streamlit by downloading or
loading embedding weights.
"""

from __future__ import annotations

import math
import re
from functools import lru_cache
from pathlib import Path
from typing import Dict, List


KNOWLEDGE_DIR = Path(__file__).resolve().parents[1] / "knowledge"
CHUNK_SIZE = 700
CHUNK_OVERLAP = 120

STOPWORDS = {
    "the", "and", "for", "with", "that", "this", "from", "are", "was", "were",
    "patient", "患者", "症状", "检查", "建议", "眼科", "疾病", "可能", "需要",
}


def _tokenize(text: str) -> List[str]:
    latin = re.findall(r"[A-Za-z][A-Za-z0-9_+-]{1,}", text.lower())
    cjk = re.findall(r"[\u4e00-\u9fff]{2,}", text)
    tokens = latin + cjk
    return [token for token in tokens if token not in STOPWORDS]


def _chunk_text(text: str, size: int = CHUNK_SIZE, overlap: int = CHUNK_OVERLAP) -> List[str]:
    chunks = []
    start = 0
    while start < len(text):
        chunk = text[start : start + size].strip()
        if len(chunk) > 40:
            chunks.append(chunk)
        start += size - overlap
    return chunks


@lru_cache(maxsize=1)
def _load_chunks() -> List[Dict[str, object]]:
    chunks: List[Dict[str, object]] = []
    for md_file in sorted(KNOWLEDGE_DIR.glob("*.md")):
        try:
            text = md_file.read_text(encoding="utf-8")
        except UnicodeDecodeError:
            text = md_file.read_text(encoding="utf-8", errors="ignore")
        for idx, chunk in enumerate(_chunk_text(text)):
            tokens = _tokenize(chunk)
            chunks.append(
                {
                    "id": f"{md_file.stem}_{idx}",
                    "source": md_file.name,
                    "text": chunk,
                    "tokens": tokens,
                    "token_set": set(tokens),
                }
            )
    return chunks


def retrieve(query: str, n_results: int = 4) -> List[str]:
    """Return top-N relevant knowledge chunks using lightweight lexical scoring."""
    chunks = _load_chunks()
    query_tokens = _tokenize(query)
    if not chunks or not query_tokens:
        return []

    query_set = set(query_tokens)
    scored = []
    for chunk in chunks:
        token_set = chunk["token_set"]
        overlap = query_set & token_set
        if not overlap:
            continue
        token_hits = sum(query_tokens.count(token) for token in overlap)
        density = len(overlap) / math.sqrt(max(len(token_set), 1))
        score = token_hits + density
        scored.append((score, chunk))

    scored.sort(key=lambda item: item[0], reverse=True)
    return [f"来源：{item[1]['source']}\n{item[1]['text']}" for item in scored[:n_results]]


def build_context(query: str, n_results: int = 4) -> str:
    """Return a formatted RAG context block ready for LLM prompt injection."""
    chunks = retrieve(query, n_results)
    if not chunks:
        return ""
    lines = ["【眼科知识库参考（来源：本地医学文档，仅作辅助依据）】"]
    for idx, chunk in enumerate(chunks, 1):
        lines.append(f"\n[{idx}]\n{chunk}")
    return "\n".join(lines)


def rag_status() -> dict:
    chunks = _load_chunks()
    doc_files = list(KNOWLEDGE_DIR.glob("*.md"))
    return {
        "available": bool(chunks),
        "chunks": len(chunks),
        "doc_files": len(doc_files),
        "embed_model": "local lexical retrieval",
        "index_path": str(KNOWLEDGE_DIR),
        "index_built": bool(chunks),
        "fingerprint_match": True,
        "reason": "" if chunks else "knowledge 目录中没有可检索的 Markdown 文档",
    }


def rebuild_index() -> dict:
    _load_chunks.cache_clear()
    chunks = _load_chunks()
    return {"ok": bool(chunks), "chunks": len(chunks), "model": "local lexical retrieval"}
