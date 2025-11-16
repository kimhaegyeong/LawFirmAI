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
from scripts.utils.chunking.factory import ChunkingFactory
from scripts.utils.embeddings import SentenceEmbedder
from scripts.utils.embedding_version_manager import EmbeddingVersionManager
from scripts.utils.reference_statute_extractor import ReferenceStatuteExtractor


def ensure_domain(conn: sqlite3.Connection, name: str) -> int:
    cur = conn.execute("SELECT id FROM domains WHERE name=?", (name,))
    row = cur.fetchone()
    if row:
        return row[0]
    cur = conn.execute("INSERT INTO domains(name) VALUES(?)", (name,))
    return cur.lastrowid


def insert_decision(conn: sqlite3.Connection, domain_id: int, meta: Dict[str, Any]) -> int:
    """Decision ?? ? ID ?? (???? ?? ??)"""
    # ???? ??
    extractor = ReferenceStatuteExtractor()
    full_text = meta.get("full_text", "") or meta.get("content", "")
    if not full_text and meta.get("sentences"):
        full_text = "\n".join(meta.get("sentences", []))
    
    reference_statutes = extractor.extract_from_content(full_text)
    reference_statutes_json = extractor.to_json(reference_statutes) if reference_statutes else None
    
    cur = conn.execute(
        """
        INSERT OR IGNORE INTO decisions(domain_id, org, doc_id, decision_date, result, reference_statutes)
        VALUES(?,?,?,?,?,?)
        """,
        (
            domain_id,
            meta.get("document_type"),
            meta.get("doc_id"),
            meta.get("decision_date"),
            meta.get("result"),
            reference_statutes_json,
        ),
    )
    if cur.lastrowid:
        decision_id = cur.lastrowid
    else:
        decision_id = conn.execute("SELECT id FROM decisions WHERE doc_id=?", (meta.get("doc_id"),)).fetchone()[0]
        # ?? ???? ??? ???? ????
        if reference_statutes_json:
            conn.execute(
                "UPDATE decisions SET reference_statutes = ? WHERE id = ?",
                (reference_statutes_json, decision_id)
            )
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
    
    # ?? ?? ?? (?? ?? ??)
    if replace_existing:
        deleted_chunks, deleted_embeddings = version_manager.delete_chunks_by_version(
            source_type="decision_paragraph",
            source_id=decision_id
        )
        if deleted_chunks > 0:
            logger.info(f"Deleted {deleted_chunks} existing chunks for decision_id={decision_id}")
    
    # ?? ?? ??
    strategy = ChunkingFactory.create_strategy(
        strategy_name=chunking_strategy,
        query_type=query_type
    )
    
    # ?? ??
    chunk_results = strategy.chunk(
        content=paragraphs,
        source_type="decision_paragraph",
        source_id=decision_id
    )
    
    # decision ????? ?? (?? ??? ???? ??)
    decision_metadata = None
    try:
        import json
        cursor_meta = conn.execute("""
            SELECT org, doc_id
            FROM decisions
            WHERE id = ?
        """, (decision_id,))
        row = cursor_meta.fetchone()
        if row:
            decision_metadata = {
                'org': row['org'],
                'doc_id': row['doc_id']
            }
    except Exception as e:
        logger.debug(f"Failed to get decision metadata for decision_id={decision_id}: {e}")
    
    # ????? JSON ??
    meta_json = None
    if decision_metadata:
        try:
            meta_json = json.dumps(decision_metadata, ensure_ascii=False)
        except Exception as e:
            logger.debug(f"Failed to serialize decision metadata for decision_id={decision_id}: {e}")
    
    rows_to_embed: List[Dict] = []
    for chunk_result in chunk_results:
        metadata = chunk_result.metadata
        cur = conn.execute(
            """
            INSERT INTO text_chunks(
                source_type, source_id, level, chunk_index, 
                start_char, end_char, overlap_chars, text, token_count, meta,
                chunking_strategy, chunk_size_category, chunk_group_id, query_type, original_document_id, embedding_version_id
            ) VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)
            """,
            (
                "decision_paragraph",
                decision_id,
                "paragraph",
                chunk_result.chunk_index,
                None,
                None,
                None,
                chunk_result.text,
                None,
                meta_json,  # ????? JSON ??
                metadata.get("chunking_strategy"),
                metadata.get("chunk_size_category"),
                metadata.get("chunk_group_id"),
                metadata.get("query_type"),
                metadata.get("original_document_id"),
                version_id
            ),
        )
        rows_to_embed.append({"id": cur.lastrowid, "text": chunk_result.text})

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
        decision_id = insert_decision(conn, domain_id, meta)
        insert_paragraphs(conn, decision_id, meta.get("sentences", []))
        conn.commit()

        embedder = SentenceEmbedder(args.model)
        insert_chunks_and_embeddings(conn, decision_id, meta.get("sentences", []), embedder)
        conn.commit()

    print("Ingested decision:", meta.get("doc_id"))


if __name__ == "__main__":
    main()
