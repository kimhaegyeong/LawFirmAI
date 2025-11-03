import argparse
import json
import os
import sqlite3
from typing import Any, Dict, List
import sys
from pathlib import Path

# Ensure project root is on sys.path so `scripts.*` imports work from any CWD
_CURRENT_FILE = Path(__file__).resolve()
_PROJECT_ROOT = _CURRENT_FILE.parents[2]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from scripts.utils.text_chunker import chunk_paragraphs
from scripts.utils.embeddings import SentenceEmbedder


def ensure_domain(conn: sqlite3.Connection, name: str) -> int:
    cur = conn.execute("SELECT id FROM domains WHERE name=?", (name,))
    row = cur.fetchone()
    if row:
        return row[0]
    cur = conn.execute("INSERT INTO domains(name) VALUES(?)", (name,))
    return cur.lastrowid


def insert_decision(conn: sqlite3.Connection, domain_id: int, meta: Dict[str, Any]) -> int:
    cur = conn.execute(
        """
        INSERT OR IGNORE INTO decisions(domain_id, org, doc_id, decision_date, result)
        VALUES(?,?,?,?,?)
        """,
        (
            domain_id,
            meta.get("document_type"),
            meta.get("doc_id"),
            meta.get("decision_date"),
            meta.get("result"),
        ),
    )
    if cur.lastrowid:
        decision_id = cur.lastrowid
    else:
        decision_id = conn.execute("SELECT id FROM decisions WHERE doc_id=?", (meta.get("doc_id"),)).fetchone()[0]
    return decision_id


def insert_paragraphs(conn: sqlite3.Connection, decision_id: int, paragraphs: List[str]) -> List[int]:
    ids: List[int] = []
    for i, p in enumerate(paragraphs):
        cur = conn.execute(
            "INSERT OR REPLACE INTO decision_paragraphs(decision_id, para_index, text) VALUES(?,?,?)",
            (decision_id, i, p),
        )
        ids.append(cur.lastrowid if cur.lastrowid else conn.execute(
            "SELECT id FROM decision_paragraphs WHERE decision_id=? AND para_index=?", (decision_id, i)
        ).fetchone()[0])
    return ids


def insert_chunks_and_embeddings(
    conn: sqlite3.Connection,
    decision_id: int,
    paragraphs: List[str],
    embedder: SentenceEmbedder,
    batch: int = 64,
):
    chunks = chunk_paragraphs(paragraphs)
    rows_to_embed: List[Dict] = []
    for ch in chunks:
        cur = conn.execute(
            """
            INSERT INTO text_chunks(source_type, source_id, level, chunk_index, start_char, end_char, overlap_chars, text, token_count, meta)
            VALUES(?,?,?,?,?,?,?,?,?,NULL)
            """,
            (
                "decision_paragraph",
                decision_id,
                "paragraph",
                ch.get("chunk_index", 0),
                None,
                None,
                None,
                ch.get("text"),
                None,
            ),
        )
        rows_to_embed.append({"id": cur.lastrowid, "text": ch.get("text", "")})

    texts = [r["text"] for r in rows_to_embed]
    if not texts:
        return
    vecs = embedder.encode(texts, batch_size=batch)
    for r, v in zip(rows_to_embed, vecs):
        conn.execute(
            "INSERT INTO embeddings(chunk_id, model, dim, vector) VALUES(?,?,?,?)",
            (r["id"], embedder.model.name_or_path, vecs.shape[1], v.tobytes()),
        )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--db", default=os.path.join("data", "lawfirm_v2.db"))
    parser.add_argument("--json", required=True)
    parser.add_argument("--domain", default="민사�?)
    parser.add_argument("--model", default="snunlp/KR-SBERT-V40K-klueNLI-augSTS")
    args = parser.parse_args()

    with open(args.json, "r", encoding="utf-8") as f:
        meta = json.load(f)

    os.makedirs(os.path.dirname(args.db), exist_ok=True)
    with sqlite3.connect(args.db) as conn:
        conn.execute("PRAGMA foreign_keys = ON")
        domain_id = ensure_domain(conn, args.domain)
        decision_id = insert_decision(conn, domain_id, meta)
        insert_paragraphs(conn, decision_id, meta.get("sentences", []))
        conn.commit()

        embedder = SentenceEmbedder(args.model)
        insert_chunks_and_embeddings(conn, decision_id, meta.get("sentences", []), embedder)
        conn.commit()

    print("Ingested decision:", meta.get("doc_id"))


if __name__ == "__main__":
    main()
