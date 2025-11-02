import re
from typing import Dict, List, Optional, Tuple

ARTICLE_HEADER_RE = re.compile(r"^제\s*(\d+)\s*조")
CLAUSE_RE = re.compile(r"^\s*(\d+)\s*\.")  # '1.' 형태 항 번호가 있는 경우 보조
ITEM_RE = re.compile(r"^\s*(?:제\s*)?(\d+)\s*호")


def _normalize_text(text: str) -> str:
    return re.sub(r"\s+", " ", text.strip())


def split_statute_sentences_into_articles(sentences: List[str]) -> List[Dict[str, str]]:
    """
    Input: sentences array from statute JSON.
    Output: list of article blocks: {heading, text}
    """
    blocks: List[Dict[str, str]] = []
    current: Optional[Dict[str, str]] = None
    for raw in sentences:
        s = raw.rstrip("\n")
        if ARTICLE_HEADER_RE.match(s):
            if current:
                blocks.append(current)
            current = {"heading": s, "text": ""}
        else:
            if current is None:
                # preamble text before first article; skip or collect into intro
                continue
            current["text"] += ("\n" if current["text"] else "") + s
    if current:
        blocks.append(current)
    return blocks


def explode_article_to_hierarchy(article_heading: str, article_body: str) -> List[Dict[str, Optional[str]]]:
    """
    Produce rows at minimal unit: item > clause > article.
    Returns: list dicts with article_no, clause_no, item_no, heading, text
    """
    m = ARTICLE_HEADER_RE.match(article_heading)
    article_no = m.group(1) if m else None

    # Split by line; detect subunits heuristically (항/호 표기 다양성 고려)
    lines = [ln for ln in article_body.splitlines() if ln.strip()]

    # First attempt: detect '제n항' boundaries
    clause_splits: List[Tuple[Optional[str], List[str]]] = []
    buffer: List[str] = []
    current_clause: Optional[str] = None
    CLAUSE_HEADER_RE = re.compile(r"^제\s*(\d+)\s*항")
    for ln in lines:
        m2 = CLAUSE_HEADER_RE.match(ln)
        if m2:
            if buffer:
                clause_splits.append((current_clause, buffer))
                buffer = []
            current_clause = m2.group(1)
            # keep the line but without the header marker
            rest = ln[m2.end():].strip()
            if rest:
                buffer.append(rest)
        else:
            buffer.append(ln)
    if buffer:
        clause_splits.append((current_clause, buffer))

    # If no explicit clause found, fall back to a single clause None
    if all(clause is None for clause, _ in clause_splits):
        clause_splits = [(None, lines)]

    results: List[Dict[str, Optional[str]]] = []
    for clause_no, clause_lines in clause_splits:
        # Try to explode into items (호)
        items: List[Tuple[Optional[str], List[str]]] = []
        buf2: List[str] = []
        current_item: Optional[str] = None
        for ln in clause_lines:
            mi = ITEM_RE.match(ln)
            if mi:
                if buf2:
                    items.append((current_item, buf2))
                    buf2 = []
                current_item = mi.group(1)
                rest = ln[mi.end():].strip()
                if rest:
                    buf2.append(rest)
            else:
                buf2.append(ln)
        if buf2:
            items.append((current_item, buf2))

        # If no items, create clause-level row; else item-level rows
        if all(item_no is None for item_no, _ in items):
            text = _normalize_text("\n".join(clause_lines))
            if text:
                results.append({
                    "article_no": article_no,
                    "clause_no": clause_no,
                    "item_no": None,
                    "heading": article_heading,
                    "text": text,
                })
        else:
            for item_no, item_lines in items:
                text = _normalize_text("\n".join(item_lines))
                if text:
                    results.append({
                        "article_no": article_no,
                        "clause_no": clause_no,
                        "item_no": item_no,
                        "heading": article_heading,
                        "text": text,
                    })

    # Backup: if nothing produced, fallback to article row
    if not results:
        results.append({
            "article_no": article_no,
            "clause_no": None,
            "item_no": None,
            "heading": article_heading,
            "text": _normalize_text(article_body),
        })
    return results


def chunk_statute(sentences: List[str], min_chars: int = 200, max_chars: int = 1200, overlap_ratio: float = 0.2) -> List[Dict]:
    """
    Build chunks at item/clause/article with small overlap between adjacent siblings.
    Returns: list of chunks with meta {level, article_no, clause_no, item_no}
    """
    chunks: List[Dict] = []
    articles = split_statute_sentences_into_articles(sentences)
    for art in articles:
        rows = explode_article_to_hierarchy(art["heading"], art["text"])
        # Simple overlap: concatenate neighbor tail/head by ratio
        for idx, row in enumerate(rows):
            text = row["text"]
            # enforce size; if too long, naive split
            if len(text) > max_chars:
                start = 0
                i = 0
                while start < len(text):
                    end = min(len(text), start + max_chars)
                    seg = text[start:end]
                    chunks.append({
                        "level": "item" if row["item_no"] else ("clause" if row["clause_no"] else "article"),
                        "article_no": row["article_no"],
                        "clause_no": row["clause_no"],
                        "item_no": row["item_no"],
                        "chunk_index": i,
                        "text": seg,
                    })
                    if end >= len(text):
                        break
                    overlap = int(max_chars * overlap_ratio)
                    start = end - overlap
                    i += 1
            else:
                chunks.append({
                    "level": "item" if row["item_no"] else ("clause" if row["clause_no"] else "article"),
                    "article_no": row["article_no"],
                    "clause_no": row["clause_no"],
                    "item_no": row["item_no"],
                    "chunk_index": 0,
                    "text": text,
                })
    return chunks


def chunk_paragraphs(paragraphs: List[str], min_chars: int = 400, max_chars: int = 1800, overlap_ratio: float = 0.25) -> List[Dict]:
    """
    Sliding window over paragraphs; 1-3 paragraphs per chunk target, with overlap.
    Returns chunk list with {chunk_index, start_para, end_para, text}
    """
    paras = [p.strip() for p in paragraphs if p and p.strip()]
    chunks: List[Dict] = []
    if not paras:
        return chunks

    i = 0
    chunk_idx = 0
    while i < len(paras):
        buf: List[str] = []
        start_i = i
        while i < len(paras) and len("\n".join(buf + [paras[i]])) < max_chars and (len(buf) < 3 or len("\n".join(buf)) < min_chars):
            buf.append(paras[i])
            i += 1
        text = "\n\n".join(buf)
        chunks.append({
            "chunk_index": chunk_idx,
            "start_para": start_i,
            "end_para": i - 1,
            "text": text,
        })
        chunk_idx += 1
        if i >= len(paras):
            break
        # overlap by ratio of paragraphs in last chunk
        overlap = max(1, int(len(buf) * overlap_ratio))
        i = max(start_i + 1, i - overlap)

    return chunks
