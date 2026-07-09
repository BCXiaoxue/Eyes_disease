"""
Lightweight local RAG retrieval.

The app uses only local Markdown files from knowledge/*.md. Retrieval stays
dependency-free and avoids runtime model downloads, while still returning
structured evidence for the UI and prompt construction.
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
    "patient", "症状", "检查", "建议", "眼科", "疾病", "可能", "需要", "患者",
}

DISEASE_ALIASES = {
    "N": ["正常", "normal", "无明显异常"],
    "D": ["糖尿病视网膜病变", "糖网", "糖尿病眼病", "diabetic retinopathy", "dr", "npdr", "pdr"],
    "G": ["青光眼", "glaucoma", "眼压", "视野缺损", "rnfl"],
    "C": ["白内障", "cataract", "晶状体混浊", "眩光"],
    "A": ["amd", "年龄相关性黄斑变性", "黄斑变性", "黄斑", "玻璃膜疣"],
    "H": ["高血压视网膜病变", "高血压眼底", "hypertensive retinopathy", "血压"],
    "M": ["病理性近视", "高度近视", "pathologic myopia", "myopic", "cnv"],
    "O": ["其他眼病", "其他疾病", "other disease"],
}

SOURCE_BOOSTS = {
    "D": "diabetic_retinopathy.md",
    "G": "glaucoma.md",
    "C": "cataract.md",
    "A": "amd.md",
    "H": "hypertensive_retinopathy.md",
    "M": "pathologic_myopia.md",
}

RED_FLAG_TERMS = {
    "突然视力下降", "突发视力下降", "突然失明", "飞蚊", "闪光", "眼痛", "外伤",
    "突发视野缺损", "幕布", "恶心", "头痛", "急诊", "红旗",
}


def _tokenize(text: str) -> List[str]:
    latin = re.findall(r"[A-Za-z][A-Za-z0-9_+-]{1,}", text.lower())
    cjk = re.findall(r"[\u4e00-\u9fff]{2,}", text)
    tokens = latin + cjk
    return [token for token in tokens if token not in STOPWORDS]


def _expand_query(query: str) -> List[str]:
    tokens = _tokenize(query)
    query_lower = query.lower()
    for aliases in DISEASE_ALIASES.values():
        if any(alias.lower() in query_lower for alias in aliases):
            tokens.extend(_tokenize(" ".join(aliases)))
    if any(term in query for term in RED_FLAG_TERMS):
        tokens.extend(["转诊", "急诊", "红旗", "referral", "standards"])
    return tokens


def _query_label_hits(query: str) -> List[str]:
    query_lower = query.lower()
    hits = []
    for label, aliases in DISEASE_ALIASES.items():
        if any(alias.lower() in query_lower for alias in aliases):
            hits.append(label)
    return hits


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
        title = next((line.lstrip("# ").strip() for line in text.splitlines() if line.startswith("#")), md_file.stem)
        source_tokens = _tokenize(f"{md_file.stem} {md_file.name} {title}")
        for idx, chunk in enumerate(_chunk_text(text)):
            tokens = _tokenize(chunk)
            chunks.append(
                {
                    "id": f"{md_file.stem}_{idx}",
                    "source": md_file.name,
                    "title": title,
                    "text": chunk,
                    "tokens": tokens,
                    "token_set": set(tokens),
                    "source_tokens": source_tokens,
                    "source_token_set": set(source_tokens),
                }
            )
    return chunks


def retrieve(
    query: str,
    n_results: int = 4,
    *,
    unique_sources: bool = True,
    min_score: float = 0.75,
) -> List[Dict[str, object]]:
    """Return ranked local evidence with optional source diversification."""
    chunks = _load_chunks()
    query_tokens = _expand_query(query)
    if not chunks or not query_tokens:
        return []

    query_set = set(query_tokens)
    label_hits = _query_label_hits(query)
    has_red_flag = any(term in query for term in RED_FLAG_TERMS)
    scored = []
    for chunk in chunks:
        token_set = chunk["token_set"]
        source_token_set = chunk["source_token_set"]
        overlap = query_set & token_set
        source_overlap = query_set & source_token_set
        if not overlap and not source_overlap:
            continue
        token_hits = sum(query_tokens.count(token) for token in overlap)
        density = len(overlap) / math.sqrt(max(len(token_set), 1))
        title_boost = len(source_overlap) * 2.5
        source_boost = 0.0
        for label in label_hits:
            if SOURCE_BOOSTS.get(label) == chunk["source"]:
                source_boost += 8.0
        if has_red_flag and chunk["source"] == "referral_standards.md":
            source_boost += 9.0
        score = token_hits + density + title_boost + source_boost
        if score >= min_score:
            scored.append((score, overlap | source_overlap, chunk))

    scored.sort(key=lambda item: item[0], reverse=True)
    if has_red_flag:
        referral = next((item for item in scored if item[2]["source"] == "referral_standards.md"), None)
        if referral is not None:
            scored.remove(referral)
            scored.insert(0, referral)

    results: List[Dict[str, object]] = []
    seen_sources = set()
    for score, matched_terms, chunk in scored:
        if unique_sources and chunk["source"] in seen_sources:
            continue
        seen_sources.add(chunk["source"])
        results.append(
            {
                "citation_id": str(chunk["id"]),
                "chunk_id": chunk["id"],
                "source": chunk["source"],
                "title": chunk["title"],
                "score": round(float(score), 4),
                "matched_terms": sorted(matched_terms),
                "text": chunk["text"],
            }
        )
        if len(results) >= n_results:
            break
    return results


def build_context(query: str, n_results: int = 4) -> str:
    """Return a formatted RAG context block ready for LLM prompt injection."""
    chunks = retrieve(query, n_results)
    if not chunks:
        return ""
    lines = ["【眼科知识库参考（来源：本地医学文档，仅作辅助依据）】"]
    for idx, chunk in enumerate(chunks, 1):
        terms = "、".join(chunk["matched_terms"][:8])
        lines.append(
            f"\n[R{idx}] 引用ID：{chunk['citation_id']}\n"
            f"来源：{chunk['source']} | 标题：{chunk['title']} | 分数：{chunk['score']} | 命中：{terms}\n"
            f"{chunk['text']}"
        )
    return "\n".join(lines)


def explain_query(query: str, n_results: int = 4) -> Dict[str, object]:
    """Return retrieval metadata for UI diagnostics and evidence display."""
    results = retrieve(query, n_results)
    terms = []
    for result in results:
        terms.extend(result["matched_terms"])
    return {
        "query": query,
        "results": results,
        "sources": sorted({result["source"] for result in results}),
        "top_terms": sorted(set(terms))[:12],
        "result_count": len(results),
        "citation_ids": [result["citation_id"] for result in results],
    }


def rag_status() -> dict:
    chunks = _load_chunks()
    doc_files = list(KNOWLEDGE_DIR.glob("*.md"))
    return {
        "available": bool(chunks),
        "chunks": len(chunks),
        "doc_files": len(doc_files),
        "embed_model": "local lexical retrieval with clinical boosts",
        "index_path": str(KNOWLEDGE_DIR),
        "index_built": bool(chunks),
        "fingerprint_match": True,
        "reason": "" if chunks else "knowledge 目录中没有可检索的 Markdown 文档",
    }


def rebuild_index() -> dict:
    _load_chunks.cache_clear()
    chunks = _load_chunks()
    return {"ok": bool(chunks), "chunks": len(chunks), "model": "local lexical retrieval with clinical boosts"}
