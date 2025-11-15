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
from scripts.utils.reference_statute_extractor import ReferenceStatuteExtractor


def ensure_domain(conn: sqlite3.Connection, name: str) -> int:
    """?„ë©”???•ì¸ ë°??ì„±"""
    cur = conn.execute("SELECT id FROM domains WHERE name=?", (name,))
    row = cur.fetchone()
    if row:
        return row[0]
    cur = conn.execute("INSERT INTO domains(name) VALUES(?)", (name,))
    return cur.lastrowid


def calculate_file_hash(file_path: str) -> str:
    """
    ?Œì¼ ?´ì‹œ ê³„ì‚° (SHA256)

    Args:
        file_path: ?Œì¼ ê²½ë¡œ

    Returns:
        str: ?Œì¼??SHA256 ?´ì‹œê°?(hex)
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
    sources ?Œì´ë¸”ì—???Œì¼???´ë? ì²˜ë¦¬?˜ì—ˆ?”ì? ?•ì¸

    Args:
        conn: ?°ì´?°ë² ?´ìŠ¤ ?°ê²°
        file_path: ?Œì¼ ê²½ë¡œ
        file_hash: ?Œì¼ ?´ì‹œ

    Returns:
        bool: ?´ë? ì²˜ë¦¬??ê²½ìš° True
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
    Caseê°€ ?„ì „???ìž¬?˜ì—ˆ?”ì? ?•ì¸

    ?„ì „??ê²€ì¦???ª©:
    1. paragraphs ê°œìˆ˜ê°€ ?ˆìƒ ê°œìˆ˜?€ ?¼ì¹˜
    2. chunksê°€ ì¡´ìž¬
    3. embeddingsê°€ chunksë§Œí¼ ì¡´ìž¬

    Args:
        conn: ?°ì´?°ë² ?´ìŠ¤ ?°ê²°
        case_id: Case ID
        expected_para_count: ?ˆìƒ paragraph ê°œìˆ˜

    Returns:
        bool: ?„ì „???ìž¬??ê²½ìš° True
    """
    # 1. Paragraphs ê°œìˆ˜ ?•ì¸
    para_count = conn.execute(
        "SELECT COUNT(*) FROM case_paragraphs WHERE case_id=?",
        (case_id,)
    ).fetchone()[0]

    if para_count != expected_para_count:
        return False

    # 2. Chunks ?•ì¸ (ìµœì†Œ 1ê°??´ìƒ ?ˆì–´????
    chunk_count = conn.execute(
        "SELECT COUNT(*) FROM text_chunks WHERE source_type='case_paragraph' AND source_id=?",
        (case_id,)
    ).fetchone()[0]

    if chunk_count == 0:
        return False

    # 3. Embeddings ?•ì¸ (ëª¨ë“  chunk??embedding???ˆì–´????
    embedding_count = conn.execute(
        """SELECT COUNT(*) FROM embeddings e
           JOIN text_chunks tc ON e.chunk_id = tc.id
           WHERE tc.source_type='case_paragraph' AND tc.source_id=?""",
        (case_id,)
    ).fetchone()[0]

    return embedding_count == chunk_count


def cleanup_incomplete_case(conn: sqlite3.Connection, case_id: int):
    """
    ë¶€ë¶??ìž¬??case ?°ì´???•ë¦¬

    CASCADEë¡??ë™ ?? œ?˜ì?ë§? ëª…ì‹œ?ìœ¼ë¡??•ë¦¬?˜ì—¬ ë¶€ë¶??? œ ?íƒœë¥?ë°©ì?

    Args:
        conn: ?°ì´?°ë² ?´ìŠ¤ ?°ê²°
        case_id: ?? œ??Case ID
    """
    # 1. Embeddings ?? œ (chunks??FK ì°¸ì¡°)
    conn.execute(
        """DELETE FROM embeddings
           WHERE chunk_id IN (
               SELECT id FROM text_chunks
               WHERE source_type='case_paragraph' AND source_id=?
           )""",
        (case_id,)
    )

    # 2. Text chunks ?? œ
    conn.execute(
        "DELETE FROM text_chunks WHERE source_type='case_paragraph' AND source_id=?",
        (case_id,)
    )

    # 3. Case paragraphs ?? œ (CASCADEë¡??ë™ ?? œ?˜ì?ë§?ëª…ì‹œ??
    conn.execute("DELETE FROM case_paragraphs WHERE case_id=?", (case_id,))

    # 4. Case ?? œ
    conn.execute("DELETE FROM cases WHERE id=?", (case_id,))


def insert_paragraphs(conn: sqlite3.Connection, case_id: int, paragraphs: List[str]) -> List[int]:
    """Paragraphs ë°°ì¹˜ ?½ìž… (?±ëŠ¥ ìµœì ??"""
    if not paragraphs:
        return []

    # ë°°ì¹˜ INSERTë¡?ìµœì ??
    data = [(case_id, i, p) for i, p in enumerate(paragraphs)]
    conn.executemany(
        "INSERT OR REPLACE INTO case_paragraphs(case_id, para_index, text) VALUES(?,?,?)",
        data
    )

    # ?½ìž…??ID ì¡°íšŒ (ë°°ì¹˜ ì¡°íšŒë¡?ìµœì ??
    placeholders = ",".join(["?"] * len(paragraphs))
    indices = list(range(len(paragraphs)))
    rows = conn.execute(
        f"SELECT id, para_index FROM case_paragraphs WHERE case_id=? AND para_index IN ({placeholders})",
        (case_id, *indices)
    ).fetchall()

    # para_index ?œì„œ?€ë¡??•ë ¬?˜ì—¬ ë°˜í™˜
    id_map = {para_index: id_val for id_val, para_index in rows}
    return [id_map[i] for i in range(len(paragraphs))]


def insert_chunks_and_embeddings(
    conn: sqlite3.Connection,
    case_id: int,
    paragraphs: List[str],
    embedder: SentenceEmbedder,
    batch: int = 128,
):
    """
    Chunks ë°?Embeddings ë°°ì¹˜ ?½ìž… (?±ëŠ¥ ìµœì ??

    Args:
        conn: ?°ì´?°ë² ?´ìŠ¤ ?°ê²°
        case_id: Case ID
        paragraphs: Paragraph ë¦¬ìŠ¤??
        embedder: Sentence embedder
        batch: Embedding ë°°ì¹˜ ?¬ê¸° (ê¸°ë³¸ê°? 128)
    """
    chunks = chunk_paragraphs(paragraphs)
    if not chunks:
        return

    # ê¸°ì¡´ chunksê°€ ?ˆëŠ” ê²½ìš° chunk_index ì¶©ëŒ ë°©ì?
    max_idx = conn.execute(
        "SELECT COALESCE(MAX(chunk_index), -1) FROM text_chunks WHERE source_type='case_paragraph' AND source_id=?",
        (case_id,)
    ).fetchone()[0]
    next_chunk_index = int(max_idx) + 1

    # Chunks ë°°ì¹˜ ?½ìž…
    chunk_data = []
    for i, ch in enumerate(chunks):
        chunk_idx = next_chunk_index + i
        chunk_data.append((
            "case_paragraph",
            case_id,
            "paragraph",
            chunk_idx,
            None,
            None,
            None,
            ch.get("text"),
            None,
        ))

    # ë°°ì¹˜ INSERT
    conn.executemany(
        """INSERT INTO text_chunks(source_type, source_id, level, chunk_index, start_char, end_char, overlap_chars, text, token_count, meta)
           VALUES(?,?,?,?,?,?,?,?,?,NULL)""",
        chunk_data
    )

    # ?½ìž…??chunk IDs ì¡°íšŒ (ë°°ì¹˜ ì¡°íšŒ)
    chunk_indices = list(range(next_chunk_index, next_chunk_index + len(chunks)))
    placeholders = ",".join(["?"] * len(chunk_indices))
    chunk_rows = conn.execute(
        f"SELECT id, chunk_index FROM text_chunks WHERE source_type='case_paragraph' AND source_id=? AND chunk_index IN ({placeholders})",
        (case_id, *chunk_indices)
    ).fetchall()

    # chunk_index ?œì„œ?€ë¡??•ë ¬
    chunk_id_map = {idx: id_val for id_val, idx in chunk_rows}
    chunk_ids = [chunk_id_map[idx] for idx in chunk_indices]
    texts = [ch.get("text", "") for ch in chunks]

    # Embeddings ?ì„± (ë°°ì¹˜ ì²˜ë¦¬)
    vecs = embedder.encode(texts, batch_size=batch)
    dim = vecs.shape[1] if len(vecs.shape) > 1 else vecs.shape[0]
    model_name = embedder.model.name_or_path

    # Embeddings ë°°ì¹˜ ?½ìž…
    embedding_data = [
        (chunk_id, model_name, dim, vec.tobytes())
        for chunk_id, vec in zip(chunk_ids, vecs)
    ]

    conn.executemany(
        "INSERT INTO embeddings(chunk_id, model, dim, vector) VALUES(?,?,?,?)",
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
    ì¤‘ë³µ ë°©ì? ë¡œì§???¬í•¨??case ?ìž¬ (?±ëŠ¥ ìµœì ??

    Args:
        conn: ?°ì´?°ë² ?´ìŠ¤ ?°ê²°
        file_path: JSON ?Œì¼ ê²½ë¡œ
        domain_name: ?„ë©”???´ë¦„
        meta: Case ë©”í??°ì´??
        embedder: Sentence embedder
        force: ê°•ì œ ?¬ì ???¬ë?
        batch_size: Embedding ë°°ì¹˜ ?¬ê¸°
        skip_hash: ?´ì‹œ ê³„ì‚° ?¤í‚µ (doc_id ì²´í¬ë§??˜í–‰)

    Returns:
        Tuple[bool, str]: (?±ê³µ ?¬ë?, ë©”ì‹œì§€)
    """
    doc_id = meta.get("doc_id")
    if not doc_id:
        return False, "doc_id is required"

    expected_para_count = len(meta.get("sentences", []))
    file_hash = None

    # ê°•ì œ ?¬ì ??ëª¨ë“œê°€ ?„ë‹Œ ê²½ìš° ì¤‘ë³µ ?•ì¸
    if not force:
        # Layer 1: doc_id ê¸°ë°˜ ì¤‘ë³µ ?•ì¸ (??ë¹ ë¦„ - ?¸ë±???¬ìš©)
        existing_case = conn.execute(
            "SELECT id FROM cases WHERE doc_id=?", (doc_id,)
        ).fetchone()

        if existing_case:
            case_id = existing_case[0]
            if is_case_complete(conn, case_id, expected_para_count):
                # ?„ì „???ìž¬??ê²½ìš° - ?´ì‹œ ê³„ì‚°???„ìš”??ê²½ìš°?ë§Œ ?˜í–‰
                if not skip_hash:
                    try:
                        file_hash = calculate_file_hash(file_path)
                        conn.execute(
                            "INSERT OR REPLACE INTO sources(source_type, path, hash) VALUES(?,?,?)",
                            ("case", file_path, file_hash)
                        )
                    except Exception:
                        pass  # ?´ì‹œ ê³„ì‚° ?¤íŒ¨?´ë„ ?¤í‚µ?€ ê³„ì†
                return False, f"Case already fully ingested: {doc_id}"
            else:
                # ë¶€ë¶??ìž¬??ê²½ìš° - ?•ë¦¬ ???¬ì ??
                cleanup_incomplete_case(conn, case_id)
        else:
            # doc_idê°€ ?†ìœ¼ë©??Œì¼ ?´ì‹œ ê¸°ë°˜ ?•ì¸ (???ë¦¬ì§€ë§??•í™•)
            if not skip_hash:
                try:
                    file_hash = calculate_file_hash(file_path)
                    if check_file_processed(conn, file_path, file_hash):
                        return False, f"File already processed: {file_path}"
                except Exception as e:
                    # ?´ì‹œ ê³„ì‚° ?¤íŒ¨ ??ê³„ì† ì§„í–‰
                    pass
    else:
        # ê°•ì œ ?¬ì ??ëª¨ë“œ - ê¸°ì¡´ ?°ì´???•ë¦¬
        existing_case = conn.execute(
            "SELECT id FROM cases WHERE doc_id=?", (doc_id,)
        ).fetchone()
        if existing_case:
            cleanup_incomplete_case(conn, existing_case[0])
            # sources?ì„œ???œê±°
            conn.execute(
                "DELETE FROM sources WHERE source_type='case' AND path=?",
                (file_path,)
            )

    # ?¸ëžœ??…˜ ?œìž‘ (?ìž??ë³´ìž¥)
    try:
        # Domain ?•ì¸
        domain_id = ensure_domain(conn, domain_name)

        # Case ?½ìž…
        case_id = insert_case(conn, domain_id, meta)

        # Paragraphs ?½ìž…
        insert_paragraphs(conn, case_id, meta.get("sentences", []))

        # Chunks ë°?Embeddings ?½ìž…
        insert_chunks_and_embeddings(
            conn, case_id, meta.get("sentences", []), embedder, batch=batch_size
        )

        # Sources ?Œì´ë¸”ì— ê¸°ë¡ (?±ê³µ ?œì—ë§? ?´ì‹œê°€ ?†ëŠ” ê²½ìš°?ë§Œ ê³„ì‚°)
        if file_hash is None:
            try:
                file_hash = calculate_file_hash(file_path)
            except Exception:
                file_hash = ""  # ?´ì‹œ ê³„ì‚° ?¤íŒ¨?´ë„ ê³„ì† ì§„í–‰

        if file_hash:
            conn.execute(
                "INSERT INTO sources(source_type, path, hash) VALUES(?,?,?)",
                ("case", file_path, file_hash)
            )

        # ì»¤ë°‹?€ ?¸ì¶œ?ê? ë°°ì¹˜ë¡?ì²˜ë¦¬?????ˆë„ë¡?ì£¼ì„ ì²˜ë¦¬
        # conn.commit()
        return True, f"Successfully ingested case: {doc_id}"

    except Exception as e:
        conn.rollback()
        return False, f"Error ingesting case {doc_id}: {e}"


def find_json_files(folder_path: str, recursive: bool = True) -> List[str]:
    """
    ?´ë”?ì„œ JSON ?Œì¼ ì°¾ê¸°

    Args:
        folder_path: ?´ë” ê²½ë¡œ
        recursive: ?¬ê??ìœ¼ë¡?ê²€?‰í• ì§€ ?¬ë?

    Returns:
        List[str]: ì°¾ì? JSON ?Œì¼ ê²½ë¡œ ë¦¬ìŠ¤??
    """
    folder = Path(folder_path)
    if not folder.exists() or not folder.is_dir():
        return []

    json_files = []
    if recursive:
        json_files = list(folder.rglob("*.json"))
    else:
        json_files = list(folder.glob("*.json"))

    # complete ?´ë”???Œì¼?€ ?œì™¸
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
    ?¨ì¼ ?Œì¼ ì²˜ë¦¬

    Args:
        conn: ?°ì´?°ë² ?´ìŠ¤ ?°ê²°
        file_path: JSON ?Œì¼ ê²½ë¡œ
        domain_name: ?„ë©”???´ë¦„
        embedder: Sentence embedder
        force: ê°•ì œ ?¬ì ???¬ë?
        batch_size: Embedding ë°°ì¹˜ ?¬ê¸°
        no_move: ?Œì¼ ?´ë™ ë¹„í™œ?±í™” ?¬ë?

    Returns:
        Tuple[bool, str]: (?±ê³µ ?¬ë?, ë©”ì‹œì§€)
    """
    abs_file_path = os.path.abspath(file_path)

    # JSON ?Œì¼ ë¡œë“œ
    try:
        with open(abs_file_path, "r", encoding="utf-8") as f:
            meta = json.load(f)
    except Exception as e:
        return False, f"Error loading JSON file: {e}"

    # ?ìž¬ ?˜í–‰
    success, message = ingest_case_with_duplicate_protection(
        conn, abs_file_path, domain_name, meta, embedder, force, batch_size, skip_hash
    )

    # ì»¤ë°‹ (auto_commit??True??ê²½ìš°)
    if auto_commit and success:
        conn.commit()

    # ?Œì¼ ?´ë™ ì²˜ë¦¬
    if success:
        if not no_move:
            move_to_complete_folder(abs_file_path)
        return True, message
    else:
        # ?´ë? ?„ë£Œ???Œì¼???´ë™
        if not no_move and ("already" in message.lower() or "processed" in message.lower()):
            if Path(abs_file_path).exists():
                move_to_complete_folder(abs_file_path)
        return False, message


def move_to_complete_folder(file_path: str) -> Optional[str]:
    """
    ?‘ì—… ?„ë£Œ???Œì¼??complete ?´ë”ë¡??´ë™

    Args:
        file_path: ?´ë™???Œì¼ ê²½ë¡œ

    Returns:
        Optional[str]: ?´ë™???Œì¼ ê²½ë¡œ (?¤íŒ¨ ??None)
    """
    try:
        file_path_obj = Path(file_path)
        if not file_path_obj.exists():
            print(f"Warning: File does not exist: {file_path}")
            return None

        # ?ë³¸ ?Œì¼???”ë ‰? ë¦¬ ê¸°ì??¼ë¡œ complete ?´ë” ?ì„±
        parent_dir = file_path_obj.parent
        complete_dir = parent_dir / "complete"
        complete_dir.mkdir(exist_ok=True)

        # ?´ë™???Œì¼ ê²½ë¡œ
        destination = complete_dir / file_path_obj.name

        # ?™ì¼???Œì¼???´ë? ?ˆëŠ” ê²½ìš° ì²˜ë¦¬
        if destination.exists():
            # ?€?„ìŠ¤?¬í”„ ì¶”ê??˜ì—¬ ì¤‘ë³µ ë°©ì?
            stem = file_path_obj.stem
            suffix = file_path_obj.suffix
            timestamp = os.path.getmtime(file_path)
            dt = datetime.fromtimestamp(timestamp)
            new_name = f"{stem}_{dt.strftime('%Y%m%d_%H%M%S')}{suffix}"
            destination = complete_dir / new_name

        # ?Œì¼ ?´ë™
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
  # ?¨ì¼ ?Œì¼ ?ìž¬
  python scripts/ingest/ingest_cases.py --file "data/aihub/.../ë¯¼ì‚¬ë²??ê²°ë¬?1.json" --domain "ë¯¼ì‚¬ë²?

  # ?´ë” ë°°ì¹˜ ì²˜ë¦¬ (?¬ê? ê²€??
  python scripts/ingest/ingest_cases.py --folder "data/aihub/.../TS_01. ë¯¼ì‚¬ë²?001. ?ê²°ë¬? --domain "ë¯¼ì‚¬ë²?

  # ?´ë” ë°°ì¹˜ ì²˜ë¦¬ (?„ìž¬ ?´ë”ë§? ?˜ìœ„ ?´ë” ?œì™¸)
  python scripts/ingest/ingest_cases.py --folder "data/aihub/..." --domain "ë¯¼ì‚¬ë²? --no-recursive

  # ê°•ì œ ?¬ì ??
  python scripts/ingest/ingest_cases.py --file "data/aihub/.../ë¯¼ì‚¬ë²??ê²°ë¬?1.json" --domain "ë¯¼ì‚¬ë²? --force

  # ?„ë£Œ ?Œì¼ ?´ë™ ë¹„í™œ?±í™”
  python scripts/ingest/ingest_cases.py --folder "data/aihub/..." --domain "ë¯¼ì‚¬ë²? --no-move

  # ??ë°°ì¹˜ ?¬ê¸°ë¡?ë¹ ë¥´ê²?ì²˜ë¦¬
  python scripts/ingest/ingest_cases.py --folder "data/aihub/..." --domain "ë¯¼ì‚¬ë²? --batch-size 256

  # ?ë„ ìµœì ???µì…˜ (ë¹ ë¥¸ ì²˜ë¦¬)
  python scripts/ingest/ingest_cases.py --folder "data/aihub/..." --domain "ë¯¼ì‚¬ë²? --commit-batch 20 --quiet

  # ìµœë? ?ë„ (????ë°°ì¹˜ ì»¤ë°‹, quiet ëª¨ë“œ)
  python scripts/ingest/ingest_cases.py --folder "data/aihub/..." --domain "ë¯¼ì‚¬ë²? --commit-batch 50 --quiet --batch-size 256
        """
    )
    parser.add_argument("--db", default=os.path.join("data", "lawfirm_v2.db"),
                        help="Database path (default: data/lawfirm_v2.db)")

    # ?…ë ¥ ?ŒìŠ¤ ?µì…˜ (?Œì¼ ?ëŠ” ?´ë”)
    input_group = parser.add_mutually_exclusive_group(required=True)
    input_group.add_argument("--file", "--json",
                             dest="input_path",
                             help="Path to JSON file containing case data (deprecated: use --file)")
    input_group.add_argument("--folder",
                             dest="input_path",
                             help="Path to folder containing JSON files (recursively searches for *.json)")

    parser.add_argument("--domain", default="ë¯¼ì‚¬ë²?,
                        help="Domain name (default: ë¯¼ì‚¬ë²?")
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

    # ?…ë ¥ ê²½ë¡œ ?•ì¸
    if not os.path.exists(args.input_path):
        print(f"Error: Path not found: {args.input_path}")
        sys.exit(1)

    # ?°ì´?°ë² ?´ìŠ¤ ?”ë ‰? ë¦¬ ?ì„±
    os.makedirs(os.path.dirname(args.db), exist_ok=True)

    with sqlite3.connect(args.db) as conn:
        # ?°ì´?°ë² ?´ìŠ¤ ?±ëŠ¥ ìµœì ???¤ì •
        conn.execute("PRAGMA foreign_keys = ON")
        conn.execute("PRAGMA journal_mode = WAL")  # Write-Ahead Logging ëª¨ë“œ
        conn.execute("PRAGMA synchronous = NORMAL")  # ?±ëŠ¥ ?¥ìƒ???„í•œ ?™ê¸°???ˆë²¨
        conn.execute("PRAGMA cache_size = -64000")  # 64MB ìºì‹œ ?¬ê¸°
        conn.execute("PRAGMA temp_store = MEMORY")  # ?„ì‹œ ?°ì´?°ë? ë©”ëª¨ë¦¬ì— ?€??

        # Embedder ì´ˆê¸°??(??ë²ˆë§Œ ì´ˆê¸°?”í•˜???¬ì‚¬??
        embedder = SentenceEmbedder(args.model)

        # ?Œì¼ ?ëŠ” ?´ë” ì²˜ë¦¬
        input_path_obj = Path(args.input_path)

        if input_path_obj.is_file():
            # ?¨ì¼ ?Œì¼ ì²˜ë¦¬
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
            # ?´ë” ë°°ì¹˜ ì²˜ë¦¬
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

            # ë°°ì¹˜ ì»¤ë°‹ ?¤ì • (?±ëŠ¥ ?¥ìƒ)
            commit_batch_size = args.commit_batch
            processed_count = 0
            last_progress_time = datetime.now()
            progress_interval = 10 if args.quiet else 5  # quiet ëª¨ë“œ?ì„œ?????ê²Œ ì¶œë ¥

            for idx, file_path in enumerate(json_files, 1):
                file_name = Path(file_path).name

                # ê°„ì†Œ?”ëœ ì§„í–‰ ?í™© ?œì‹œ (?ˆë¬´ ?ì£¼ ì¶œë ¥?˜ì? ?ŠìŒ)
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
                    # ë°°ì¹˜ ì²˜ë¦¬ ëª¨ë“œ: ?´ì‹œ ?¤í‚µ, ë°°ì¹˜ ì»¤ë°‹
                    success, message = process_single_file(
                        conn, file_path, args.domain, embedder,
                        args.force, args.batch_size, args.no_move,
                        skip_hash=True,  # ?´ì‹œ ê³„ì‚° ?¤í‚µ (?ë„ ?¥ìƒ)
                        auto_commit=False  # ë°°ì¹˜ ì»¤ë°‹ ?¬ìš©
                    )

                    if success:
                        stats["success"] += 1
                        processed_count += 1

                        # ë°°ì¹˜ ì»¤ë°‹
                        if processed_count >= commit_batch_size:
                            conn.commit()
                            processed_count = 0
                    else:
                        if "already" in message.lower() or "processed" in message.lower():
                            stats["skipped"] += 1
                            # ?¤í‚µ??ê²½ìš°?ë„ sources ?…ë°?´íŠ¸ë¥??„í•´ ì»¤ë°‹ ?„ìš”?????ˆìŒ
                            if processed_count > 0:
                                conn.commit()
                                processed_count = 0
                        else:
                            stats["failed"] += 1
                            stats["errors"].append({"file": file_name, "error": message})
                            # ?ëŸ¬ ë°œìƒ ??ë¡¤ë°±?˜ì? ?Šê³  ê³„ì† ì§„í–‰
                            if processed_count > 0:
                                conn.commit()
                                processed_count = 0

                except Exception as e:
                    stats["failed"] += 1
                    stats["errors"].append({"file": file_name, "error": str(e)})
                    # ?ëŸ¬ ë°œìƒ ??ë¡¤ë°±?˜ì? ?Šê³  ê³„ì† ì§„í–‰
                    if processed_count > 0:
                        conn.commit()
                        processed_count = 0

            # ?¨ì? ë³€ê²½ì‚¬??ì»¤ë°‹
            if processed_count > 0:
                conn.commit()

            # ?µê³„ ì¶œë ¥
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
                for err in stats["errors"][:10]:  # ìµœë? 10ê°œë§Œ ì¶œë ¥
                    print(f"  - {err['file']}: {err['error']}")
                if len(stats["errors"]) > 10:
                    print(f"  ... and {len(stats['errors']) - 10} more errors")

            # ?¤íŒ¨???Œì¼???ˆìœ¼ë©?exit code 1
            if stats["failed"] > 0:
                sys.exit(1)
        else:
            print(f"Error: Path is neither a file nor a directory: {args.input_path}")
            sys.exit(1)


if __name__ == "__main__":
    main()
