import argparse
import hashlib
import json
import os
import shutil
import sqlite3
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

# Ensure project root is on sys.path so `scripts.*` imports work from any CWD
_CURRENT_FILE = Path(__file__).resolve()
_PROJECT_ROOT = _CURRENT_FILE.parents[2]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from scripts.utils.embeddings import SentenceEmbedder
from scripts.utils.text_chunker import chunk_paragraphs
from scripts.utils.chunking.factory import ChunkingFactory
from scripts.utils.embedding_version_manager import EmbeddingVersionManager
from scripts.utils.reference_statute_extractor import ReferenceStatuteExtractor


def ensure_domain(conn: sqlite3.Connection, name: str) -> int:
    """?ë©???ì¸ ë°??ì±"""
    cur = conn.execute("SELECT id FROM domains WHERE name=?", (name,))
    row = cur.fetchone()
    if row:
        return row[0]
    cur = conn.execute("INSERT INTO domains(name) VALUES(?)", (name,))
    return cur.lastrowid


def calculate_file_hash(file_path: str) -> str:
    """
    ?ì¼ ?´ì ê³ì° (SHA256)

    Args:
        file_path: ?ì¼ ê²½ë¡

    Returns:
        str: ?ì¼??SHA256 ?´ìê°?(hex)
    """
    sha256_hash = hashlib.sha256()
    try:
        with open(file_path, "rb") as f:
            for byte_block in iter(lambda: f.read(4096), b""):
                sha256_hash.update(byte_block)
        return sha256_hash.hexdigest()
    except Exception as e:
        raise IOError(f"Error calculating file hash for {file_path}: {e}")


def check_file_processed(conn: sqlite3.Connection, file_path: str, file_hash: str) -> bool:
    """
    sources ?ì´ë¸ì???ì¼???´ë? ì²ë¦¬?ì?ì? ?ì¸

    Args:
        conn: ?°ì´?°ë² ?´ì¤ ?°ê²°
        file_path: ?ì¼ ê²½ë¡
        file_hash: ?ì¼ ?´ì

    Returns:
        bool: ?´ë? ì²ë¦¬??ê²½ì° True
    """
    cur = conn.execute(
        "SELECT id FROM sources WHERE source_type='case' AND path=? AND hash=?",
        (file_path, file_hash)
    )
    return cur.fetchone() is not None


def insert_case(conn: sqlite3.Connection, domain_id: int, meta: Dict[str, Any]) -> int:
    """Case ?? ? ID ?? (???? ?? ??)"""
    # ???? ??
    extractor = ReferenceStatuteExtractor()
    full_text = meta.get("full_text", "") or meta.get("content", "")
    if not full_text and meta.get("sentences"):
        full_text = "\n".join(meta.get("sentences", []))
    
    reference_statutes = extractor.extract_from_content(full_text)
    reference_statutes_json = extractor.to_json(reference_statutes) if reference_statutes else None
    
    cur = conn.execute(
        """
        INSERT OR IGNORE INTO cases(domain_id, doc_id, court, case_type, casenames, announce_date, reference_statutes)
        VALUES(?,?,?,?,?,?,?)
        """,
        (
            domain_id,
            meta.get("doc_id"),
            meta.get("normalized_court"),
            meta.get("casetype"),
            meta.get("casenames"),
            meta.get("announce_date"),
            reference_statutes_json,
        ),
    )
    if cur.lastrowid:
        case_id = cur.lastrowid
    else:
        case_id = conn.execute("SELECT id FROM cases WHERE doc_id=?", (meta.get("doc_id"),)).fetchone()[0]
        # ?? ???? ??? ???? ????
        if reference_statutes_json:
            conn.execute(
                "UPDATE cases SET reference_statutes = ? WHERE id = ?",
                (reference_statutes_json, case_id)
            )
    return case_id


def is_case_complete(
    conn: sqlite3.Connection,
    case_id: int,
    expected_para_count: int
) -> bool:
    """
    Caseê° ?ì ???ì¬?ì?ì? ?ì¸

    ?ì ??ê²ì¦???ª©:
    1. paragraphs ê°ìê° ?ì ê°ì? ?¼ì¹
    2. chunksê° ì¡´ì¬
    3. embeddingsê° chunksë§í¼ ì¡´ì¬

    Args:
        conn: ?°ì´?°ë² ?´ì¤ ?°ê²°
        case_id: Case ID
        expected_para_count: ?ì paragraph ê°ì

    Returns:
        bool: ?ì ???ì¬??ê²½ì° True
    """
    # 1. Paragraphs ê°ì ?ì¸
    para_count = conn.execute(
        "SELECT COUNT(*) FROM case_paragraphs WHERE case_id=?",
        (case_id,)
    ).fetchone()[0]

    if para_count != expected_para_count:
        return False

    # 2. Chunks ?ì¸ (ìµì 1ê°??´ì ?ì´????
    chunk_count = conn.execute(
        "SELECT COUNT(*) FROM text_chunks WHERE source_type='case_paragraph' AND source_id=?",
        (case_id,)
    ).fetchone()[0]

    if chunk_count == 0:
        return False

    # 3. Embeddings ?ì¸ (ëª¨ë  chunk??embedding???ì´????
    embedding_count = conn.execute(
        """SELECT COUNT(*) FROM embeddings e
           JOIN text_chunks tc ON e.chunk_id = tc.id
           WHERE tc.source_type='case_paragraph' AND tc.source_id=?""",
        (case_id,)
    ).fetchone()[0]

    return embedding_count == chunk_count


def cleanup_incomplete_case(conn: sqlite3.Connection, case_id: int):
    """
    ë¶ë¶??ì¬??case ?°ì´???ë¦¬

    CASCADEë¡??ë ?? ?ì?ë§? ëªì?ì¼ë¡??ë¦¬?ì¬ ë¶ë¶???  ?íë¥?ë°©ì?

    Args:
        conn: ?°ì´?°ë² ?´ì¤ ?°ê²°
        case_id: ?? ??Case ID
    """
    # 1. Embeddings ??  (chunks??FK ì°¸ì¡°)
    conn.execute(
        """DELETE FROM embeddings
           WHERE chunk_id IN (
               SELECT id FROM text_chunks
               WHERE source_type='case_paragraph' AND source_id=?
           )""",
        (case_id,)
    )

    # 2. Text chunks ?? 
    conn.execute(
        "DELETE FROM text_chunks WHERE source_type='case_paragraph' AND source_id=?",
        (case_id,)
    )

    # 3. Case paragraphs ??  (CASCADEë¡??ë ?? ?ì?ë§?ëªì??
    conn.execute("DELETE FROM case_paragraphs WHERE case_id=?", (case_id,))

    # 4. Case ?? 
    conn.execute("DELETE FROM cases WHERE id=?", (case_id,))


def insert_paragraphs(conn: sqlite3.Connection, case_id: int, paragraphs: List[str]) -> List[int]:
    """Paragraphs ë°°ì¹ ?½ì (?±ë¥ ìµì ??"""
    if not paragraphs:
        return []

    # ë°°ì¹ INSERTë¡?ìµì ??
    data = [(case_id, i, p) for i, p in enumerate(paragraphs)]
    conn.executemany(
        "INSERT OR REPLACE INTO case_paragraphs(case_id, para_index, text) VALUES(?,?,?)",
        data
    )

    # ?½ì??ID ì¡°í (ë°°ì¹ ì¡°íë¡?ìµì ??
    placeholders = ",".join(["?"] * len(paragraphs))
    indices = list(range(len(paragraphs)))
    rows = conn.execute(
        f"SELECT id, para_index FROM case_paragraphs WHERE case_id=? AND para_index IN ({placeholders})",
        (case_id, *indices)
    ).fetchall()

    # para_index ?ì?ë¡??ë ¬?ì¬ ë°í
    id_map = {para_index: id_val for id_val, para_index in rows}
    return [id_map[i] for i in range(len(paragraphs))]


def insert_chunks_and_embeddings(
    conn: sqlite3.Connection,
    case_id: int,
    paragraphs: List[str],
    embedder: SentenceEmbedder,
    batch: int = 128,
    chunking_strategy: str = "standard",
    query_type: Optional[str] = None,
    replace_existing: bool = True,
):
    """
    Chunks ë°?Embeddings ë°°ì¹ ?½ì (?±ë¥ ìµì ??

    Args:
        conn: ?°ì´?°ë² ?´ì¤ ?°ê²°
        case_id: Case ID
        paragraphs: Paragraph ë¦¬ì¤??
        embedder: Sentence embedder
        batch: Embedding ë°°ì¹ ?¬ê¸° (ê¸°ë³¸ê°? 128)
    """
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
            source_type="case_paragraph",
            source_id=case_id
        )
        if deleted_chunks > 0:
            logger.info(f"Deleted {deleted_chunks} existing chunks for case_id={case_id}")
    
    # ?? ?? ??
    strategy = ChunkingFactory.create_strategy(
        strategy_name=chunking_strategy,
        query_type=query_type
    )
    
    chunk_results = strategy.chunk(
        content=paragraphs,
        source_type="case_paragraph",
        source_id=case_id
    )
    
    if not chunk_results:
        return
    
    # 기존 chunks가 있는 경우 chunk_index 충돌 방지
    max_idx = conn.execute(
        "SELECT COALESCE(MAX(chunk_index), -1) FROM text_chunks WHERE source_type='case_paragraph' AND source_id=?",
        (case_id,)
    ).fetchone()[0]
    next_chunk_index = int(max_idx) + 1

    # Chunks 배치 입력
    chunk_ids = []
    texts_to_embed = []
    
    # case 메타데이터 조회 (모든 청크에 공통으로 사용)
    case_metadata = None
    try:
        import json
        cursor_meta = conn.execute("""
            SELECT doc_id, casenames, court
            FROM cases
            WHERE id = ?
        """, (case_id,))
        row = cursor_meta.fetchone()
        if row:
            case_metadata = {
                'doc_id': row['doc_id'],
                'casenames': row['casenames'],
                'court': row['court']
            }
    except Exception as e:
        logger.debug(f"Failed to get case metadata for case_id={case_id}: {e}")
    
    # 메타데이터 JSON 생성
    meta_json = None
    if case_metadata:
        try:
            meta_json = json.dumps(case_metadata, ensure_ascii=False)
        except Exception as e:
            logger.debug(f"Failed to serialize case metadata for case_id={case_id}: {e}")
    
    for i, chunk_result in enumerate(chunk_results):
        chunk_idx = next_chunk_index + i
        metadata = chunk_result.metadata
        
        cursor = conn.execute(
            """INSERT INTO text_chunks(
                source_type, source_id, level, chunk_index, 
                start_char, end_char, overlap_chars, text, token_count, meta,
                chunking_strategy, chunk_size_category, chunk_group_id, 
                query_type, original_document_id, embedding_version_id
            ) VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)""",
            (
                "case_paragraph",
                case_id,
                metadata.get("level", "paragraph"),
                chunk_idx,
                None, None, None,
                chunk_result.text,
                None,
                meta_json,  # 메타데이터 JSON 저장
                metadata.get("chunking_strategy"),
                metadata.get("chunk_size_category"),
                metadata.get("chunk_group_id"),
                metadata.get("query_type"),
                metadata.get("original_document_id"),
                version_id
            )
        )
        
        chunk_id = cursor.lastrowid
        chunk_ids.append(chunk_id)
        texts_to_embed.append(chunk_result.text)

    # Embeddings 생성 (배치 처리)
    if texts_to_embed:
        vecs = embedder.encode(texts_to_embed, batch_size=batch)
        dim = vecs.shape[1] if len(vecs.shape) > 1 else vecs.shape[0]
        model_name = getattr(embedder.model, 'name_or_path', 'snunlp/KR-SBERT-V40K-klueNLI-augSTS')

        # Embeddings 배치 입력
        embedding_data = [
            (chunk_id, model_name, dim, vec.tobytes(), version_id)
            for chunk_id, vec in zip(chunk_ids, vecs)
        ]

        conn.executemany(
            "INSERT INTO embeddings(chunk_id, model, dim, vector, version_id) VALUES(?,?,?,?,?)",
            embedding_data
        )


def ingest_case_with_duplicate_protection(
    conn: sqlite3.Connection,
    file_path: str,
    domain_name: str,
    meta: Dict[str, Any],
    embedder: SentenceEmbedder,
    force: bool = False,
    batch_size: int = 128,
    skip_hash: bool = False,
) -> Tuple[bool, str]:
    """
    ì¤ë³µ ë°©ì? ë¡ì§???¬í¨??case ?ì¬ (?±ë¥ ìµì ??

    Args:
        conn: ?°ì´?°ë² ?´ì¤ ?°ê²°
        file_path: JSON ?ì¼ ê²½ë¡
        domain_name: ?ë©???´ë¦
        meta: Case ë©í??°ì´??
        embedder: Sentence embedder
        force: ê°ì  ?¬ì ???¬ë?
        batch_size: Embedding ë°°ì¹ ?¬ê¸°
        skip_hash: ?´ì ê³ì° ?¤íµ (doc_id ì²´í¬ë§??í)

    Returns:
        Tuple[bool, str]: (?±ê³µ ?¬ë?, ë©ìì§)
    """
    doc_id = meta.get("doc_id")
    if not doc_id:
        return False, "doc_id is required"

    expected_para_count = len(meta.get("sentences", []))
    file_hash = None

    # ê°ì  ?¬ì ??ëª¨ëê° ?ë ê²½ì° ì¤ë³µ ?ì¸
    if not force:
        # Layer 1: doc_id ê¸°ë° ì¤ë³µ ?ì¸ (??ë¹ ë¦ - ?¸ë±???¬ì©)
        existing_case = conn.execute(
            "SELECT id FROM cases WHERE doc_id=?", (doc_id,)
        ).fetchone()

        if existing_case:
            case_id = existing_case[0]
            if is_case_complete(conn, case_id, expected_para_count):
                # ?ì ???ì¬??ê²½ì° - ?´ì ê³ì°???ì??ê²½ì°?ë§ ?í
                if not skip_hash:
                    try:
                        file_hash = calculate_file_hash(file_path)
                        conn.execute(
                            "INSERT OR REPLACE INTO sources(source_type, path, hash) VALUES(?,?,?)",
                            ("case", file_path, file_hash)
                        )
                    except Exception:
                        pass  # ?´ì ê³ì° ?¤í¨?´ë ?¤íµ? ê³ì
                return False, f"Case already fully ingested: {doc_id}"
            else:
                # ë¶ë¶??ì¬??ê²½ì° - ?ë¦¬ ???¬ì ??
                cleanup_incomplete_case(conn, case_id)
        else:
            # doc_idê° ?ì¼ë©??ì¼ ?´ì ê¸°ë° ?ì¸ (???ë¦¬ì§ë§??í)
            if not skip_hash:
                try:
                    file_hash = calculate_file_hash(file_path)
                    if check_file_processed(conn, file_path, file_hash):
                        return False, f"File already processed: {file_path}"
                except Exception as e:
                    # ?´ì ê³ì° ?¤í¨ ??ê³ì ì§í
                    pass
    else:
        # ê°ì  ?¬ì ??ëª¨ë - ê¸°ì¡´ ?°ì´???ë¦¬
        existing_case = conn.execute(
            "SELECT id FROM cases WHERE doc_id=?", (doc_id,)
        ).fetchone()
        if existing_case:
            cleanup_incomplete_case(conn, existing_case[0])
            # sources?ì???ê±°
            conn.execute(
                "DELETE FROM sources WHERE source_type='case' AND path=?",
                (file_path,)
            )

    # ?¸ë?? ?ì (?ì??ë³´ì¥)
    try:
        # Domain ?ì¸
        domain_id = ensure_domain(conn, domain_name)

        # Case ?½ì
        case_id = insert_case(conn, domain_id, meta)

        # Paragraphs ?½ì
        insert_paragraphs(conn, case_id, meta.get("sentences", []))

        # Chunks ë°?Embeddings ?½ì
        insert_chunks_and_embeddings(
            conn, case_id, meta.get("sentences", []), embedder, batch=batch_size
        )

        # Sources ?ì´ë¸ì ê¸°ë¡ (?±ê³µ ?ìë§? ?´ìê° ?ë ê²½ì°?ë§ ê³ì°)
        if file_hash is None:
            try:
                file_hash = calculate_file_hash(file_path)
            except Exception:
                file_hash = ""  # ?´ì ê³ì° ?¤í¨?´ë ê³ì ì§í

        if file_hash:
            conn.execute(
                "INSERT INTO sources(source_type, path, hash) VALUES(?,?,?)",
                ("case", file_path, file_hash)
            )

        # ì»¤ë°? ?¸ì¶?ê? ë°°ì¹ë¡?ì²ë¦¬?????ëë¡?ì£¼ì ì²ë¦¬
        # conn.commit()
        return True, f"Successfully ingested case: {doc_id}"

    except Exception as e:
        conn.rollback()
        return False, f"Error ingesting case {doc_id}: {e}"


def find_json_files(folder_path: str, recursive: bool = True) -> List[str]:
    """
    ?´ë?ì JSON ?ì¼ ì°¾ê¸°

    Args:
        folder_path: ?´ë ê²½ë¡
        recursive: ?¬ê??ì¼ë¡?ê²?í ì§ ?¬ë?

    Returns:
        List[str]: ì°¾ì? JSON ?ì¼ ê²½ë¡ ë¦¬ì¤??
    """
    folder = Path(folder_path)
    if not folder.exists() or not folder.is_dir():
        return []

    json_files = []
    if recursive:
        json_files = list(folder.rglob("*.json"))
    else:
        json_files = list(folder.glob("*.json"))

    # complete ?´ë???ì¼? ?ì¸
    json_files = [str(f) for f in json_files if "complete" not in str(f)]

    return sorted(json_files)


def process_single_file(
    conn: sqlite3.Connection,
    file_path: str,
    domain_name: str,
    embedder: SentenceEmbedder,
    force: bool,
    batch_size: int,
    no_move: bool,
    skip_hash: bool = False,
    auto_commit: bool = True,
) -> Tuple[bool, str]:
    """
    ?¨ì¼ ?ì¼ ì²ë¦¬

    Args:
        conn: ?°ì´?°ë² ?´ì¤ ?°ê²°
        file_path: JSON ?ì¼ ê²½ë¡
        domain_name: ?ë©???´ë¦
        embedder: Sentence embedder
        force: ê°ì  ?¬ì ???¬ë?
        batch_size: Embedding ë°°ì¹ ?¬ê¸°
        no_move: ?ì¼ ?´ë ë¹í?±í ?¬ë?

    Returns:
        Tuple[bool, str]: (?±ê³µ ?¬ë?, ë©ìì§)
    """
    abs_file_path = os.path.abspath(file_path)

    # JSON ?ì¼ ë¡ë
    try:
        with open(abs_file_path, "r", encoding="utf-8") as f:
            meta = json.load(f)
    except Exception as e:
        return False, f"Error loading JSON file: {e}"

    # ?ì¬ ?í
    success, message = ingest_case_with_duplicate_protection(
        conn, abs_file_path, domain_name, meta, embedder, force, batch_size, skip_hash
    )

    # ì»¤ë° (auto_commit??True??ê²½ì°)
    if auto_commit and success:
        conn.commit()

    # ?ì¼ ?´ë ì²ë¦¬
    if success:
        if not no_move:
            move_to_complete_folder(abs_file_path)
        return True, message
    else:
        # ?´ë? ?ë£???ì¼???´ë
        if not no_move and ("already" in message.lower() or "processed" in message.lower()):
            if Path(abs_file_path).exists():
                move_to_complete_folder(abs_file_path)
        return False, message


def move_to_complete_folder(file_path: str) -> Optional[str]:
    """
    ?ì ?ë£???ì¼??complete ?´ëë¡??´ë

    Args:
        file_path: ?´ë???ì¼ ê²½ë¡

    Returns:
        Optional[str]: ?´ë???ì¼ ê²½ë¡ (?¤í¨ ??None)
    """
    try:
        file_path_obj = Path(file_path)
        if not file_path_obj.exists():
            print(f"Warning: File does not exist: {file_path}")
            return None

        # ?ë³¸ ?ì¼???ë ? ë¦¬ ê¸°ì??¼ë¡ complete ?´ë ?ì±
        parent_dir = file_path_obj.parent
        complete_dir = parent_dir / "complete"
        complete_dir.mkdir(exist_ok=True)

        # ?´ë???ì¼ ê²½ë¡
        destination = complete_dir / file_path_obj.name

        # ?ì¼???ì¼???´ë? ?ë ê²½ì° ì²ë¦¬
        if destination.exists():
            # ??ì¤?¬í ì¶ê??ì¬ ì¤ë³µ ë°©ì?
            stem = file_path_obj.stem
            suffix = file_path_obj.suffix
            timestamp = os.path.getmtime(file_path)
            dt = datetime.fromtimestamp(timestamp)
            new_name = f"{stem}_{dt.strftime('%Y%m%d_%H%M%S')}{suffix}"
            destination = complete_dir / new_name

        # ?ì¼ ?´ë
        shutil.move(str(file_path_obj), str(destination))
        print(f"??Moved to complete folder: {destination}")
        return str(destination)

    except Exception as e:
        print(f"Error moving file to complete folder: {e}")
        return None


def main():
    parser = argparse.ArgumentParser(
        description="Ingest case data from JSON file into lawfirm_v2.db",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # ?¨ì¼ ?ì¼ ?ì¬
  python scripts/ingest/ingest_cases.py --file "data/aihub/.../ë¯¼ì¬ë²??ê²°ë¬?1.json" --domain "ë¯¼ì¬ë²?

  # ?´ë ë°°ì¹ ì²ë¦¬ (?¬ê? ê²??
  python scripts/ingest/ingest_cases.py --folder "data/aihub/.../TS_01. ë¯¼ì¬ë²?001. ?ê²°ë¬? --domain "ë¯¼ì¬ë²?

  # ?´ë ë°°ì¹ ì²ë¦¬ (?ì¬ ?´ëë§? ?ì ?´ë ?ì¸)
  python scripts/ingest/ingest_cases.py --folder "data/aihub/..." --domain "ë¯¼ì¬ë²? --no-recursive

  # ê°ì  ?¬ì ??
  python scripts/ingest/ingest_cases.py --file "data/aihub/.../ë¯¼ì¬ë²??ê²°ë¬?1.json" --domain "ë¯¼ì¬ë²? --force

  # ?ë£ ?ì¼ ?´ë ë¹í?±í
  python scripts/ingest/ingest_cases.py --folder "data/aihub/..." --domain "ë¯¼ì¬ë²? --no-move

  # ??ë°°ì¹ ?¬ê¸°ë¡?ë¹ ë¥´ê²?ì²ë¦¬
  python scripts/ingest/ingest_cases.py --folder "data/aihub/..." --domain "ë¯¼ì¬ë²? --batch-size 256

  # ?ë ìµì ???µì (ë¹ ë¥¸ ì²ë¦¬)
  python scripts/ingest/ingest_cases.py --folder "data/aihub/..." --domain "ë¯¼ì¬ë²? --commit-batch 20 --quiet

  # ìµë? ?ë (????ë°°ì¹ ì»¤ë°, quiet ëª¨ë)
  python scripts/ingest/ingest_cases.py --folder "data/aihub/..." --domain "ë¯¼ì¬ë²? --commit-batch 50 --quiet --batch-size 256
        """
    )
    parser.add_argument("--db", default=os.path.join("data", "lawfirm_v2.db"),
                        help="Database path (default: data/lawfirm_v2.db)")

    # ?ë ¥ ?ì¤ ?µì (?ì¼ ?ë ?´ë)
    input_group = parser.add_mutually_exclusive_group(required=True)
    input_group.add_argument("--file", "--json",
                             dest="input_path",
                             help="Path to JSON file containing case data (deprecated: use --file)")
    input_group.add_argument("--folder",
                             dest="input_path",
                             help="Path to folder containing JSON files (recursively searches for *.json)")

    parser.add_argument("--domain", default="ë¯¼ì¬ë²?,
                        help="Domain name (default: ë¯¼ì¬ë²?")
    parser.add_argument("--model", default="snunlp/KR-SBERT-V40K-klueNLI-augSTS",
                        help="Sentence embedding model (default: snunlp/KR-SBERT-V40K-klueNLI-augSTS)")
    parser.add_argument("--batch-size", type=int, default=128,
                        help="Embedding batch size (default: 128, larger = faster but uses more memory)")
    parser.add_argument("--force", action="store_true",
                        help="Force re-ingestion even if already processed")
    parser.add_argument("--no-move", action="store_true",
                        help="Do not move file to complete folder after successful ingestion")
    parser.add_argument("--no-recursive", action="store_true",
                        help="Do not search subdirectories when using --folder")
    parser.add_argument("--commit-batch", type=int, default=10,
                        help="Commit every N files (default: 10, larger = faster but riskier)")
    parser.add_argument("--quiet", action="store_true",
                        help="Quiet mode: less output during batch processing")
    args = parser.parse_args()

    # ?ë ¥ ê²½ë¡ ?ì¸
    if not os.path.exists(args.input_path):
        print(f"Error: Path not found: {args.input_path}")
        sys.exit(1)

    # ?°ì´?°ë² ?´ì¤ ?ë ? ë¦¬ ?ì±
    os.makedirs(os.path.dirname(args.db), exist_ok=True)

    with sqlite3.connect(args.db) as conn:
        # ?°ì´?°ë² ?´ì¤ ?±ë¥ ìµì ???¤ì 
        conn.execute("PRAGMA foreign_keys = ON")
        conn.execute("PRAGMA journal_mode = WAL")  # Write-Ahead Logging ëª¨ë
        conn.execute("PRAGMA synchronous = NORMAL")  # ?±ë¥ ?¥ì???í ?ê¸°???ë²¨
        conn.execute("PRAGMA cache_size = -64000")  # 64MB ìºì ?¬ê¸°
        conn.execute("PRAGMA temp_store = MEMORY")  # ?ì ?°ì´?°ë? ë©ëª¨ë¦¬ì ???

        # Embedder ì´ê¸°??(??ë²ë§ ì´ê¸°?í???¬ì¬??
        embedder = SentenceEmbedder(args.model)

        # ?ì¼ ?ë ?´ë ì²ë¦¬
        input_path_obj = Path(args.input_path)

        if input_path_obj.is_file():
            # ?¨ì¼ ?ì¼ ì²ë¦¬
            print(f"Processing single file: {args.input_path}")
            success, message = process_single_file(
                conn, args.input_path, args.domain, embedder,
                args.force, args.batch_size, args.no_move
            )

            if success:
                print(f"??{message}")
            else:
                print(f"??{message}")
                if not args.force and "already" in message.lower():
                    print("  Use --force to re-ingest")

        elif input_path_obj.is_dir():
            # ?´ë ë°°ì¹ ì²ë¦¬
            json_files = find_json_files(args.input_path, recursive=not args.no_recursive)

            if not json_files:
                print(f"No JSON files found in: {args.input_path}")
                sys.exit(0)

            total = len(json_files)
            print(f"\n{'='*60}")
            print(f"Batch Processing: {total} JSON files found")
            print(f"Folder: {args.input_path}")
            print(f"Recursive: {not args.no_recursive}")
            print(f"{'='*60}\n")

            stats = {
                "total": total,
                "success": 0,
                "skipped": 0,
                "failed": 0,
                "errors": []
            }

            start_time = datetime.now()

            # ë°°ì¹ ì»¤ë° ?¤ì  (?±ë¥ ?¥ì)
            commit_batch_size = args.commit_batch
            processed_count = 0
            last_progress_time = datetime.now()
            progress_interval = 10 if args.quiet else 5  # quiet ëª¨ë?ì?????ê² ì¶ë ¥

            for idx, file_path in enumerate(json_files, 1):
                file_name = Path(file_path).name

                # ê°ì?ë ì§í ?í© ?ì (?ë¬´ ?ì£¼ ì¶ë ¥?ì? ?ì)
                current_time = datetime.now()
                time_since_last_progress = (current_time - last_progress_time).total_seconds()

                should_print = (idx == 1 or idx == total or
                              time_since_last_progress >= progress_interval or
                              (not args.quiet and idx % 10 == 0))

                if should_print:
                    elapsed_so_far = (current_time - start_time).total_seconds()
                    avg_time = elapsed_so_far / idx if idx > 0 else 0
                    remaining = (total - idx) * avg_time if idx > 0 else 0
                    progress_pct = (idx / total) * 100
                    print(f"[{idx}/{total}] ({progress_pct:.1f}%) Processing: {file_name} "
                          f"[ETA: {remaining:.0f}s, Avg: {avg_time:.2f}s]")
                    last_progress_time = current_time

                try:
                    # ë°°ì¹ ì²ë¦¬ ëª¨ë: ?´ì ?¤íµ, ë°°ì¹ ì»¤ë°
                    success, message = process_single_file(
                        conn, file_path, args.domain, embedder,
                        args.force, args.batch_size, args.no_move,
                        skip_hash=True,  # ?´ì ê³ì° ?¤íµ (?ë ?¥ì)
                        auto_commit=False  # ë°°ì¹ ì»¤ë° ?¬ì©
                    )

                    if success:
                        stats["success"] += 1
                        processed_count += 1

                        # ë°°ì¹ ì»¤ë°
                        if processed_count >= commit_batch_size:
                            conn.commit()
                            processed_count = 0
                    else:
                        if "already" in message.lower() or "processed" in message.lower():
                            stats["skipped"] += 1
                            # ?¤íµ??ê²½ì°?ë sources ?ë°?´í¸ë¥??í´ ì»¤ë° ?ì?????ì
                            if processed_count > 0:
                                conn.commit()
                                processed_count = 0
                        else:
                            stats["failed"] += 1
                            stats["errors"].append({"file": file_name, "error": message})
                            # ?ë¬ ë°ì ??ë¡¤ë°±?ì? ?ê³  ê³ì ì§í
                            if processed_count > 0:
                                conn.commit()
                                processed_count = 0

                except Exception as e:
                    stats["failed"] += 1
                    stats["errors"].append({"file": file_name, "error": str(e)})
                    # ?ë¬ ë°ì ??ë¡¤ë°±?ì? ?ê³  ê³ì ì§í
                    if processed_count > 0:
                        conn.commit()
                        processed_count = 0

            # ?¨ì? ë³ê²½ì¬??ì»¤ë°
            if processed_count > 0:
                conn.commit()

            # ?µê³ ì¶ë ¥
            elapsed = (datetime.now() - start_time).total_seconds()
            print(f"\n{'='*60}")
            print("Batch Processing Summary:")
            print(f"  Total files:    {stats['total']}")
            print(f"  ??Success:      {stats['success']}")
            print(f"  ??Skipped:      {stats['skipped']}")
            print(f"  ??Failed:        {stats['failed']}")
            print(f"  Time elapsed:    {elapsed:.2f} seconds")
            if stats['total'] > 0:
                print(f"  Avg time/file:  {elapsed/stats['total']:.2f} seconds")
            print(f"{'='*60}")

            if stats["errors"]:
                print("\nErrors:")
                for err in stats["errors"][:10]:  # ìµë? 10ê°ë§ ì¶ë ¥
                    print(f"  - {err['file']}: {err['error']}")
                if len(stats["errors"]) > 10:
                    print(f"  ... and {len(stats['errors']) - 10} more errors")

            # ?¤í¨???ì¼???ì¼ë©?exit code 1
            if stats["failed"] > 0:
                sys.exit(1)
        else:
            print(f"Error: Path is neither a file nor a directory: {args.input_path}")
            sys.exit(1)


if __name__ == "__main__":
    main()
