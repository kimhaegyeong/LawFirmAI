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


def ensure_domain(conn: sqlite3.Connection, name: str) -> int:
    """도메인 확인 및 생성"""
    cur = conn.execute("SELECT id FROM domains WHERE name=?", (name,))
    row = cur.fetchone()
    if row:
        return row[0]
    cur = conn.execute("INSERT INTO domains(name) VALUES(?)", (name,))
    return cur.lastrowid


def calculate_file_hash(file_path: str) -> str:
    """
    파일 해시 계산 (SHA256)

    Args:
        file_path: 파일 경로

    Returns:
        str: 파일의 SHA256 해시값 (hex)
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
    sources 테이블에서 파일이 이미 처리되었는지 확인

    Args:
        conn: 데이터베이스 연결
        file_path: 파일 경로
        file_hash: 파일 해시

    Returns:
        bool: 이미 처리된 경우 True
    """
    cur = conn.execute(
        "SELECT id FROM sources WHERE source_type='case' AND path=? AND hash=?",
        (file_path, file_hash)
    )
    return cur.fetchone() is not None


def insert_case(conn: sqlite3.Connection, domain_id: int, meta: Dict[str, Any]) -> int:
    """Case 삽입 및 ID 반환"""
    cur = conn.execute(
        """
        INSERT OR IGNORE INTO cases(domain_id, doc_id, court, case_type, casenames, announce_date)
        VALUES(?,?,?,?,?,?)
        """,
        (
            domain_id,
            meta.get("doc_id"),
            meta.get("normalized_court"),
            meta.get("casetype"),
            meta.get("casenames"),
            meta.get("announce_date"),
        ),
    )
    if cur.lastrowid:
        case_id = cur.lastrowid
    else:
        case_id = conn.execute("SELECT id FROM cases WHERE doc_id=?", (meta.get("doc_id"),)).fetchone()[0]
    return case_id


def is_case_complete(
    conn: sqlite3.Connection,
    case_id: int,
    expected_para_count: int
) -> bool:
    """
    Case가 완전히 적재되었는지 확인

    완전성 검증 항목:
    1. paragraphs 개수가 예상 개수와 일치
    2. chunks가 존재
    3. embeddings가 chunks만큼 존재

    Args:
        conn: 데이터베이스 연결
        case_id: Case ID
        expected_para_count: 예상 paragraph 개수

    Returns:
        bool: 완전히 적재된 경우 True
    """
    # 1. Paragraphs 개수 확인
    para_count = conn.execute(
        "SELECT COUNT(*) FROM case_paragraphs WHERE case_id=?",
        (case_id,)
    ).fetchone()[0]

    if para_count != expected_para_count:
        return False

    # 2. Chunks 확인 (최소 1개 이상 있어야 함)
    chunk_count = conn.execute(
        "SELECT COUNT(*) FROM text_chunks WHERE source_type='case_paragraph' AND source_id=?",
        (case_id,)
    ).fetchone()[0]

    if chunk_count == 0:
        return False

    # 3. Embeddings 확인 (모든 chunk에 embedding이 있어야 함)
    embedding_count = conn.execute(
        """SELECT COUNT(*) FROM embeddings e
           JOIN text_chunks tc ON e.chunk_id = tc.id
           WHERE tc.source_type='case_paragraph' AND tc.source_id=?""",
        (case_id,)
    ).fetchone()[0]

    return embedding_count == chunk_count


def cleanup_incomplete_case(conn: sqlite3.Connection, case_id: int):
    """
    부분 적재된 case 데이터 정리

    CASCADE로 자동 삭제되지만, 명시적으로 정리하여 부분 삭제 상태를 방지

    Args:
        conn: 데이터베이스 연결
        case_id: 삭제할 Case ID
    """
    # 1. Embeddings 삭제 (chunks의 FK 참조)
    conn.execute(
        """DELETE FROM embeddings
           WHERE chunk_id IN (
               SELECT id FROM text_chunks
               WHERE source_type='case_paragraph' AND source_id=?
           )""",
        (case_id,)
    )

    # 2. Text chunks 삭제
    conn.execute(
        "DELETE FROM text_chunks WHERE source_type='case_paragraph' AND source_id=?",
        (case_id,)
    )

    # 3. Case paragraphs 삭제 (CASCADE로 자동 삭제되지만 명시적)
    conn.execute("DELETE FROM case_paragraphs WHERE case_id=?", (case_id,))

    # 4. Case 삭제
    conn.execute("DELETE FROM cases WHERE id=?", (case_id,))


def insert_paragraphs(conn: sqlite3.Connection, case_id: int, paragraphs: List[str]) -> List[int]:
    """Paragraphs 배치 삽입 (성능 최적화)"""
    if not paragraphs:
        return []

    # 배치 INSERT로 최적화
    data = [(case_id, i, p) for i, p in enumerate(paragraphs)]
    conn.executemany(
        "INSERT OR REPLACE INTO case_paragraphs(case_id, para_index, text) VALUES(?,?,?)",
        data
    )

    # 삽입된 ID 조회 (배치 조회로 최적화)
    placeholders = ",".join(["?"] * len(paragraphs))
    indices = list(range(len(paragraphs)))
    rows = conn.execute(
        f"SELECT id, para_index FROM case_paragraphs WHERE case_id=? AND para_index IN ({placeholders})",
        (case_id, *indices)
    ).fetchall()

    # para_index 순서대로 정렬하여 반환
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
    Chunks 및 Embeddings 배치 삽입 (성능 최적화)

    Args:
        conn: 데이터베이스 연결
        case_id: Case ID
        paragraphs: Paragraph 리스트
        embedder: Sentence embedder
        batch: Embedding 배치 크기 (기본값: 128)
    """
    chunks = chunk_paragraphs(paragraphs)
    if not chunks:
        return

    # 기존 chunks가 있는 경우 chunk_index 충돌 방지
    max_idx = conn.execute(
        "SELECT COALESCE(MAX(chunk_index), -1) FROM text_chunks WHERE source_type='case_paragraph' AND source_id=?",
        (case_id,)
    ).fetchone()[0]
    next_chunk_index = int(max_idx) + 1

    # Chunks 배치 삽입
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

    # 배치 INSERT
    conn.executemany(
        """INSERT INTO text_chunks(source_type, source_id, level, chunk_index, start_char, end_char, overlap_chars, text, token_count, meta)
           VALUES(?,?,?,?,?,?,?,?,?,NULL)""",
        chunk_data
    )

    # 삽입된 chunk IDs 조회 (배치 조회)
    chunk_indices = list(range(next_chunk_index, next_chunk_index + len(chunks)))
    placeholders = ",".join(["?"] * len(chunk_indices))
    chunk_rows = conn.execute(
        f"SELECT id, chunk_index FROM text_chunks WHERE source_type='case_paragraph' AND source_id=? AND chunk_index IN ({placeholders})",
        (case_id, *chunk_indices)
    ).fetchall()

    # chunk_index 순서대로 정렬
    chunk_id_map = {idx: id_val for id_val, idx in chunk_rows}
    chunk_ids = [chunk_id_map[idx] for idx in chunk_indices]
    texts = [ch.get("text", "") for ch in chunks]

    # Embeddings 생성 (배치 처리)
    vecs = embedder.encode(texts, batch_size=batch)
    dim = vecs.shape[1] if len(vecs.shape) > 1 else vecs.shape[0]
    model_name = embedder.model.name_or_path

    # Embeddings 배치 삽입
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
    중복 방지 로직이 포함된 case 적재 (성능 최적화)

    Args:
        conn: 데이터베이스 연결
        file_path: JSON 파일 경로
        domain_name: 도메인 이름
        meta: Case 메타데이터
        embedder: Sentence embedder
        force: 강제 재적재 여부
        batch_size: Embedding 배치 크기
        skip_hash: 해시 계산 스킵 (doc_id 체크만 수행)

    Returns:
        Tuple[bool, str]: (성공 여부, 메시지)
    """
    doc_id = meta.get("doc_id")
    if not doc_id:
        return False, "doc_id is required"

    expected_para_count = len(meta.get("sentences", []))
    file_hash = None

    # 강제 재적재 모드가 아닌 경우 중복 확인
    if not force:
        # Layer 1: doc_id 기반 중복 확인 (더 빠름 - 인덱스 사용)
        existing_case = conn.execute(
            "SELECT id FROM cases WHERE doc_id=?", (doc_id,)
        ).fetchone()

        if existing_case:
            case_id = existing_case[0]
            if is_case_complete(conn, case_id, expected_para_count):
                # 완전히 적재된 경우 - 해시 계산이 필요한 경우에만 수행
                if not skip_hash:
                    try:
                        file_hash = calculate_file_hash(file_path)
                        conn.execute(
                            "INSERT OR REPLACE INTO sources(source_type, path, hash) VALUES(?,?,?)",
                            ("case", file_path, file_hash)
                        )
                    except Exception:
                        pass  # 해시 계산 실패해도 스킵은 계속
                return False, f"Case already fully ingested: {doc_id}"
            else:
                # 부분 적재된 경우 - 정리 후 재적재
                cleanup_incomplete_case(conn, case_id)
        else:
            # doc_id가 없으면 파일 해시 기반 확인 (더 느리지만 정확)
            if not skip_hash:
                try:
                    file_hash = calculate_file_hash(file_path)
                    if check_file_processed(conn, file_path, file_hash):
                        return False, f"File already processed: {file_path}"
                except Exception as e:
                    # 해시 계산 실패 시 계속 진행
                    pass
    else:
        # 강제 재적재 모드 - 기존 데이터 정리
        existing_case = conn.execute(
            "SELECT id FROM cases WHERE doc_id=?", (doc_id,)
        ).fetchone()
        if existing_case:
            cleanup_incomplete_case(conn, existing_case[0])
            # sources에서도 제거
            conn.execute(
                "DELETE FROM sources WHERE source_type='case' AND path=?",
                (file_path,)
            )

    # 트랜잭션 시작 (원자성 보장)
    try:
        # Domain 확인
        domain_id = ensure_domain(conn, domain_name)

        # Case 삽입
        case_id = insert_case(conn, domain_id, meta)

        # Paragraphs 삽입
        insert_paragraphs(conn, case_id, meta.get("sentences", []))

        # Chunks 및 Embeddings 삽입
        insert_chunks_and_embeddings(
            conn, case_id, meta.get("sentences", []), embedder, batch=batch_size
        )

        # Sources 테이블에 기록 (성공 시에만, 해시가 없는 경우에만 계산)
        if file_hash is None:
            try:
                file_hash = calculate_file_hash(file_path)
            except Exception:
                file_hash = ""  # 해시 계산 실패해도 계속 진행

        if file_hash:
            conn.execute(
                "INSERT INTO sources(source_type, path, hash) VALUES(?,?,?)",
                ("case", file_path, file_hash)
            )

        # 커밋은 호출자가 배치로 처리할 수 있도록 주석 처리
        # conn.commit()
        return True, f"Successfully ingested case: {doc_id}"

    except Exception as e:
        conn.rollback()
        return False, f"Error ingesting case {doc_id}: {e}"


def find_json_files(folder_path: str, recursive: bool = True) -> List[str]:
    """
    폴더에서 JSON 파일 찾기

    Args:
        folder_path: 폴더 경로
        recursive: 재귀적으로 검색할지 여부

    Returns:
        List[str]: 찾은 JSON 파일 경로 리스트
    """
    folder = Path(folder_path)
    if not folder.exists() or not folder.is_dir():
        return []

    json_files = []
    if recursive:
        json_files = list(folder.rglob("*.json"))
    else:
        json_files = list(folder.glob("*.json"))

    # complete 폴더의 파일은 제외
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
    단일 파일 처리

    Args:
        conn: 데이터베이스 연결
        file_path: JSON 파일 경로
        domain_name: 도메인 이름
        embedder: Sentence embedder
        force: 강제 재적재 여부
        batch_size: Embedding 배치 크기
        no_move: 파일 이동 비활성화 여부

    Returns:
        Tuple[bool, str]: (성공 여부, 메시지)
    """
    abs_file_path = os.path.abspath(file_path)

    # JSON 파일 로드
    try:
        with open(abs_file_path, "r", encoding="utf-8") as f:
            meta = json.load(f)
    except Exception as e:
        return False, f"Error loading JSON file: {e}"

    # 적재 수행
    success, message = ingest_case_with_duplicate_protection(
        conn, abs_file_path, domain_name, meta, embedder, force, batch_size, skip_hash
    )

    # 커밋 (auto_commit이 True인 경우)
    if auto_commit and success:
        conn.commit()

    # 파일 이동 처리
    if success:
        if not no_move:
            move_to_complete_folder(abs_file_path)
        return True, message
    else:
        # 이미 완료된 파일도 이동
        if not no_move and ("already" in message.lower() or "processed" in message.lower()):
            if Path(abs_file_path).exists():
                move_to_complete_folder(abs_file_path)
        return False, message


def move_to_complete_folder(file_path: str) -> Optional[str]:
    """
    작업 완료된 파일을 complete 폴더로 이동

    Args:
        file_path: 이동할 파일 경로

    Returns:
        Optional[str]: 이동된 파일 경로 (실패 시 None)
    """
    try:
        file_path_obj = Path(file_path)
        if not file_path_obj.exists():
            print(f"Warning: File does not exist: {file_path}")
            return None

        # 원본 파일의 디렉토리 기준으로 complete 폴더 생성
        parent_dir = file_path_obj.parent
        complete_dir = parent_dir / "complete"
        complete_dir.mkdir(exist_ok=True)

        # 이동할 파일 경로
        destination = complete_dir / file_path_obj.name

        # 동일한 파일이 이미 있는 경우 처리
        if destination.exists():
            # 타임스탬프 추가하여 중복 방지
            stem = file_path_obj.stem
            suffix = file_path_obj.suffix
            timestamp = os.path.getmtime(file_path)
            dt = datetime.fromtimestamp(timestamp)
            new_name = f"{stem}_{dt.strftime('%Y%m%d_%H%M%S')}{suffix}"
            destination = complete_dir / new_name

        # 파일 이동
        shutil.move(str(file_path_obj), str(destination))
        print(f"✓ Moved to complete folder: {destination}")
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
  # 단일 파일 적재
  python scripts/ingest/ingest_cases.py --file "data/aihub/.../민사법_판결문_1.json" --domain "민사법"

  # 폴더 배치 처리 (재귀 검색)
  python scripts/ingest/ingest_cases.py --folder "data/aihub/.../TS_01. 민사법_001. 판결문" --domain "민사법"

  # 폴더 배치 처리 (현재 폴더만, 하위 폴더 제외)
  python scripts/ingest/ingest_cases.py --folder "data/aihub/..." --domain "민사법" --no-recursive

  # 강제 재적재
  python scripts/ingest/ingest_cases.py --file "data/aihub/.../민사법_판결문_1.json" --domain "민사법" --force

  # 완료 파일 이동 비활성화
  python scripts/ingest/ingest_cases.py --folder "data/aihub/..." --domain "민사법" --no-move

  # 큰 배치 크기로 빠르게 처리
  python scripts/ingest/ingest_cases.py --folder "data/aihub/..." --domain "민사법" --batch-size 256

  # 속도 최적화 옵션 (빠른 처리)
  python scripts/ingest/ingest_cases.py --folder "data/aihub/..." --domain "민사법" --commit-batch 20 --quiet

  # 최대 속도 (더 큰 배치 커밋, quiet 모드)
  python scripts/ingest/ingest_cases.py --folder "data/aihub/..." --domain "민사법" --commit-batch 50 --quiet --batch-size 256
        """
    )
    parser.add_argument("--db", default=os.path.join("data", "lawfirm_v2.db"),
                        help="Database path (default: data/lawfirm_v2.db)")

    # 입력 소스 옵션 (파일 또는 폴더)
    input_group = parser.add_mutually_exclusive_group(required=True)
    input_group.add_argument("--file", "--json",
                             dest="input_path",
                             help="Path to JSON file containing case data (deprecated: use --file)")
    input_group.add_argument("--folder",
                             dest="input_path",
                             help="Path to folder containing JSON files (recursively searches for *.json)")

    parser.add_argument("--domain", default="민사법",
                        help="Domain name (default: 민사법)")
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

    # 입력 경로 확인
    if not os.path.exists(args.input_path):
        print(f"Error: Path not found: {args.input_path}")
        sys.exit(1)

    # 데이터베이스 디렉토리 생성
    os.makedirs(os.path.dirname(args.db), exist_ok=True)

    with sqlite3.connect(args.db) as conn:
        # 데이터베이스 성능 최적화 설정
        conn.execute("PRAGMA foreign_keys = ON")
        conn.execute("PRAGMA journal_mode = WAL")  # Write-Ahead Logging 모드
        conn.execute("PRAGMA synchronous = NORMAL")  # 성능 향상을 위한 동기화 레벨
        conn.execute("PRAGMA cache_size = -64000")  # 64MB 캐시 크기
        conn.execute("PRAGMA temp_store = MEMORY")  # 임시 데이터를 메모리에 저장

        # Embedder 초기화 (한 번만 초기화하여 재사용)
        embedder = SentenceEmbedder(args.model)

        # 파일 또는 폴더 처리
        input_path_obj = Path(args.input_path)

        if input_path_obj.is_file():
            # 단일 파일 처리
            print(f"Processing single file: {args.input_path}")
            success, message = process_single_file(
                conn, args.input_path, args.domain, embedder,
                args.force, args.batch_size, args.no_move
            )

            if success:
                print(f"✓ {message}")
            else:
                print(f"⊘ {message}")
                if not args.force and "already" in message.lower():
                    print("  Use --force to re-ingest")

        elif input_path_obj.is_dir():
            # 폴더 배치 처리
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

            # 배치 커밋 설정 (성능 향상)
            commit_batch_size = args.commit_batch
            processed_count = 0
            last_progress_time = datetime.now()
            progress_interval = 10 if args.quiet else 5  # quiet 모드에서는 더 적게 출력

            for idx, file_path in enumerate(json_files, 1):
                file_name = Path(file_path).name

                # 간소화된 진행 상황 표시 (너무 자주 출력하지 않음)
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
                    # 배치 처리 모드: 해시 스킵, 배치 커밋
                    success, message = process_single_file(
                        conn, file_path, args.domain, embedder,
                        args.force, args.batch_size, args.no_move,
                        skip_hash=True,  # 해시 계산 스킵 (속도 향상)
                        auto_commit=False  # 배치 커밋 사용
                    )

                    if success:
                        stats["success"] += 1
                        processed_count += 1

                        # 배치 커밋
                        if processed_count >= commit_batch_size:
                            conn.commit()
                            processed_count = 0
                    else:
                        if "already" in message.lower() or "processed" in message.lower():
                            stats["skipped"] += 1
                            # 스킵된 경우에도 sources 업데이트를 위해 커밋 필요할 수 있음
                            if processed_count > 0:
                                conn.commit()
                                processed_count = 0
                        else:
                            stats["failed"] += 1
                            stats["errors"].append({"file": file_name, "error": message})
                            # 에러 발생 시 롤백하지 않고 계속 진행
                            if processed_count > 0:
                                conn.commit()
                                processed_count = 0

                except Exception as e:
                    stats["failed"] += 1
                    stats["errors"].append({"file": file_name, "error": str(e)})
                    # 에러 발생 시 롤백하지 않고 계속 진행
                    if processed_count > 0:
                        conn.commit()
                        processed_count = 0

            # 남은 변경사항 커밋
            if processed_count > 0:
                conn.commit()

            # 통계 출력
            elapsed = (datetime.now() - start_time).total_seconds()
            print(f"\n{'='*60}")
            print("Batch Processing Summary:")
            print(f"  Total files:    {stats['total']}")
            print(f"  ✓ Success:      {stats['success']}")
            print(f"  ⊘ Skipped:      {stats['skipped']}")
            print(f"  ✗ Failed:        {stats['failed']}")
            print(f"  Time elapsed:    {elapsed:.2f} seconds")
            if stats['total'] > 0:
                print(f"  Avg time/file:  {elapsed/stats['total']:.2f} seconds")
            print(f"{'='*60}")

            if stats["errors"]:
                print("\nErrors:")
                for err in stats["errors"][:10]:  # 최대 10개만 출력
                    print(f"  - {err['file']}: {err['error']}")
                if len(stats["errors"]) > 10:
                    print(f"  ... and {len(stats['errors']) - 10} more errors")

            # 실패한 파일이 있으면 exit code 1
            if stats["failed"] > 0:
                sys.exit(1)
        else:
            print(f"Error: Path is neither a file nor a directory: {args.input_path}")
            sys.exit(1)


if __name__ == "__main__":
    main()
