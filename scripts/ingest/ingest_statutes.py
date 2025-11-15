import argparse
import json
import os
import sqlite3
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

# Ensure project root is on sys.path so `scripts.*` imports work from any CWD
_CURRENT_FILE = Path(__file__).resolve()
_PROJECT_ROOT = _CURRENT_FILE.parents[2]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from scripts.utils.embeddings import SentenceEmbedder
from scripts.utils.text_chunker import chunk_statute
from scripts.utils.chunking.factory import ChunkingFactory
from scripts.utils.embedding_version_manager import EmbeddingVersionManager


def ensure_domain(conn: sqlite3.Connection, name: str) -> int:
    cur = conn.execute("SELECT id FROM domains WHERE name=?", (name,))
    row = cur.fetchone()
    if row:
        return row[0]
    cur = conn.execute("INSERT INTO domains(name) VALUES(?)", (name,))
    return cur.lastrowid


def ensure_statute(conn: sqlite3.Connection, domain_id: int, meta: Dict[str, Any]) -> int:
    cur = conn.execute(
        "SELECT id FROM statutes WHERE domain_id=? AND name=?",
        (domain_id, meta["statute_name"]),
    )
    row = cur.fetchone()
    if row:
        return row[0]
    cur = conn.execute(
        """
        INSERT INTO statutes(domain_id, name, abbrv, statute_type, proclamation_date, effective_date, category)
        VALUES(?,?,?,?,?,?,?)
        """,
        (
            domain_id,
            meta.get("statute_name"),
            meta.get("statute_abbrv"),
            meta.get("statute_type"),
            meta.get("proclamation_date"),
            meta.get("effective_date"),
            meta.get("statute_category"),
        ),
    )
    return cur.lastrowid


def insert_statute_articles(conn: sqlite3.Connection, statute_id: int, meta: Dict[str, Any]) -> List[int]:
    sentences: List[str] = meta.get("sentences", [])
    # build hierarchy rows via chunker pre-parse (will later chunk again for embeddings)
    from scripts.utils.text_chunker import (
        explode_article_to_hierarchy,
        split_statute_sentences_into_articles,
    )

    article_blocks = split_statute_sentences_into_articles(sentences)
    ids: List[int] = []
    for blk in article_blocks:
        rows = explode_article_to_hierarchy(blk["heading"], blk["text"])
        for r in rows:
            cur = conn.execute(
                """
                INSERT INTO statute_articles(statute_id, article_no, clause_no, item_no, heading, text)
                VALUES(?,?,?,?,?,?)
                """,
                (
                    statute_id,
                    r.get("article_no"),
                    r.get("clause_no"),
                    r.get("item_no"),
                    r.get("heading"),
                    r.get("text"),
                ),
            )
            ids.append(cur.lastrowid)
    return ids


def insert_chunks_and_embeddings(
    conn: sqlite3.Connection,
    statute_article_ids: List[int],
    sentences: List[str],
    embedder: SentenceEmbedder,
    batch: int = 64,
    chunking_strategy: str = "standard",
    query_type: Optional[str] = None,
    replace_existing: bool = True,
):
    import logging
    logger = logging.getLogger(__name__)
    
    # ?????? ?? ????
    db_path = conn.execute("PRAGMA database_list").fetchone()[2]
    version_manager = EmbeddingVersionManager(db_path)
    
    # ?? ?? ?? ?? ??
    active_version = version_manager.get_active_version(chunking_strategy)
    if not active_version:
        model_name = getattr(embedder.model, 'name_or_path', 'snunlp/KR-SBERT-V40K-klueNLI-augSTS')
        version_name = f"v1.0.0-{chunking_strategy}"
        version_id = version_manager.register_version(
            version_name=version_name,
            chunking_strategy=chunking_strategy,
            model_name=model_name,
            description=f"{chunking_strategy} ?? ??",
            set_active=True
        )
    else:
        version_id = active_version['id']
    
    # ?? ?? ?? (?? ?? ??) - ? statute_article_id?? ??
    if replace_existing:
        for article_id in statute_article_ids:
            deleted_chunks, deleted_embeddings = version_manager.delete_chunks_by_version(
                source_type="statute_article",
                source_id=article_id
            )
            if deleted_chunks > 0:
                logger.info(f"Deleted {deleted_chunks} existing chunks for statute_article_id={article_id}")
    
    # ?? ?? ??
    strategy = ChunkingFactory.create_strategy(
        strategy_name=chunking_strategy,
        query_type=query_type
    )
    
    # ?? ?? (statute_article? ? ?? ID ??)
    source_id = statute_article_ids[0] if statute_article_ids else 0
    chunk_results = strategy.chunk(
        content=sentences,
        source_type="statute_article",
        source_id=source_id
    )
    
    # ChunkResult? ?? ???? ?? (?? ???)
    chunks = []
    for chunk_result in chunk_results:
        chunk_dict = {
            "text": chunk_result.text,
            "chunk_index": chunk_result.chunk_index,
            **chunk_result.metadata
        }
        chunks.append(chunk_dict)
    # Map chunks back to nearest statute_articles by article/clause/item identifiers
    # Build lookup from keys -> list of rowids inserted for that key in order of insertion
    cur = conn.execute(
        "SELECT id, article_no, clause_no, item_no FROM statute_articles WHERE id IN (" + ",".join(
            [str(i) for i in statute_article_ids]
        ) + ") ORDER BY id"
    )
    key_to_ids: Dict[str, List[int]] = {}
    for row in cur.fetchall():
        key = f"{row[1]}|{row[2]}|{row[3]}"
        key_to_ids.setdefault(key, []).append(row[0])

    # Prepare next chunk_index per source_id to avoid UNIQUE(source_type, source_id, chunk_index) collisions
    # Initialize with current max(chunk_index)+1 if rows already exist
    next_index_by_source: Dict[int, int] = {}
    if statute_article_ids:
        placeholders = ",".join(["?"] * len(statute_article_ids))
        for sid, max_idx in conn.execute(
            f"SELECT source_id, COALESCE(MAX(chunk_index), -1) FROM text_chunks WHERE source_type=? AND source_id IN ({placeholders}) GROUP BY source_id",
            ("statute_article", *statute_article_ids,),
        ):
            next_index_by_source[int(sid)] = int(max_idx) + 1

    rows_to_embed: List[Dict] = []
    for ch in chunks:
        key = f"{ch.get('article_no')}|{ch.get('clause_no')}|{ch.get('item_no')}"
        candidates = key_to_ids.get(key) or []
        source_id = candidates[0] if candidates else statute_article_ids[0]
        # Assign sequential chunk_index per source_id
        current_idx = next_index_by_source.get(source_id, 0)
        # ??????? ?? ?? ??
        metadata = ch.get("metadata", {})
        cur2 = conn.execute(
            """
            INSERT INTO text_chunks(
                source_type, source_id, level, chunk_index, 
                start_char, end_char, overlap_chars, text, token_count, meta,
                chunking_strategy, chunk_size_category, chunk_group_id, query_type, original_document_id, embedding_version_id
            ) VALUES(?,?,?,?,?,?,?,?,?,NULL,?,?,?,?,?,?)
            """,
            (
                "statute_article",
                source_id,
                ch.get("level"),
                current_idx,
                None,
                None,
                None,
                ch.get("text"),
                None,
                metadata.get("chunking_strategy"),
                metadata.get("chunk_size_category"),
                metadata.get("chunk_group_id"),
                metadata.get("query_type"),
                metadata.get("original_document_id"),
                version_id
            ),
        )
        # Advance next index for this source
        next_index_by_source[source_id] = current_idx + 1
        chunk_id = cur2.lastrowid
        rows_to_embed.append({"id": chunk_id, "text": ch.get("text", "")})

    texts = [r["text"] for r in rows_to_embed]
    if not texts:
        return
    vecs = embedder.encode(texts, batch_size=batch)
    for r, v in zip(rows_to_embed, vecs):
        conn.execute(
            "INSERT INTO embeddings(chunk_id, model, dim, vector, version_id) VALUES(?,?,?,?,?)",
            (r["id"], embedder.model.name_or_path, vecs.shape[1], v.tobytes(), version_id),
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
        statute_id = ensure_statute(conn, domain_id, meta)
        article_ids = insert_statute_articles(conn, statute_id, meta)
        conn.commit()

        embedder = SentenceEmbedder(args.model)
        insert_chunks_and_embeddings(conn, article_ids, meta.get("sentences", []), embedder)
        conn.commit()

    print("Ingested statute:", meta.get("statute_name"))


if __name__ == "__main__":
    main()
