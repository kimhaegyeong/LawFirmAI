"""
기존 데이터에 참조조문 추가 마이그레이션 스크립트

사용법:
    python scripts/migrations/migrate_reference_statutes.py --db data/lawfirm_v2.db
"""
import argparse
import sqlite3
import sys
from pathlib import Path

# 프로젝트 루트를 sys.path에 추가
_CURRENT_FILE = Path(__file__).resolve()
_PROJECT_ROOT = _CURRENT_FILE.parents[2]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from scripts.utils.reference_statute_extractor import ReferenceStatuteExtractor


def migrate_cases(conn: sqlite3.Connection, extractor: ReferenceStatuteExtractor, force: bool = False):
    """Cases 테이블 마이그레이션"""
    print("Migrating cases...")
    
    if force:
        cursor = conn.execute("SELECT id, doc_id FROM cases")
        print("  Force mode: Re-extracting all cases")
    else:
        cursor = conn.execute("SELECT id, doc_id FROM cases WHERE reference_statutes IS NULL")
    cases = cursor.fetchall()
    
    total = len(cases)
    if total == 0:
        print("  No cases to migrate")
        return
    
    print(f"  Found {total} cases to migrate")
    
    migrated = 0
    for idx, (case_id, doc_id) in enumerate(cases, 1):
        if idx % 100 == 0:
            print(f"  Progress: {idx}/{total} ({idx*100//total}%)")
        
        # 전체 텍스트 가져오기
        para_cursor = conn.execute(
            "SELECT text FROM case_paragraphs WHERE case_id = ? ORDER BY para_index",
            (case_id,)
        )
        paragraphs = para_cursor.fetchall()
        full_text = "\n".join([p[0] for p in paragraphs])
        
        if not full_text:
            continue
        
        # 참조조문 추출
        reference_statutes = extractor.extract_from_content(full_text)
        reference_statutes_json = extractor.to_json(reference_statutes) if reference_statutes else None
        
        # 업데이트
        conn.execute(
            "UPDATE cases SET reference_statutes = ? WHERE id = ?",
            (reference_statutes_json, case_id)
        )
        
        if reference_statutes:
            migrated += 1
    
    conn.commit()
    print(f"  Migrated {migrated}/{total} cases")


def migrate_decisions(conn: sqlite3.Connection, extractor: ReferenceStatuteExtractor, force: bool = False):
    """Decisions 테이블 마이그레이션"""
    print("Migrating decisions...")
    
    if force:
        cursor = conn.execute("SELECT id, doc_id FROM decisions")
        print("  Force mode: Re-extracting all decisions")
    else:
        cursor = conn.execute("SELECT id, doc_id FROM decisions WHERE reference_statutes IS NULL")
    decisions = cursor.fetchall()
    
    total = len(decisions)
    if total == 0:
        print("  No decisions to migrate")
        return
    
    print(f"  Found {total} decisions to migrate")
    
    migrated = 0
    for idx, (decision_id, doc_id) in enumerate(decisions, 1):
        if idx % 100 == 0:
            print(f"  Progress: {idx}/{total} ({idx*100//total}%)")
        
        # 전체 텍스트 가져오기
        para_cursor = conn.execute(
            "SELECT text FROM decision_paragraphs WHERE decision_id = ? ORDER BY para_index",
            (decision_id,)
        )
        paragraphs = para_cursor.fetchall()
        full_text = "\n".join([p[0] for p in paragraphs])
        
        if not full_text:
            continue
        
        # 참조조문 추출
        reference_statutes = extractor.extract_from_content(full_text)
        reference_statutes_json = extractor.to_json(reference_statutes) if reference_statutes else None
        
        # 업데이트
        conn.execute(
            "UPDATE decisions SET reference_statutes = ? WHERE id = ?",
            (reference_statutes_json, decision_id)
        )
        
        if reference_statutes:
            migrated += 1
    
    conn.commit()
    print(f"  Migrated {migrated}/{total} decisions")


def migrate_interpretations(conn: sqlite3.Connection, extractor: ReferenceStatuteExtractor, force: bool = False):
    """Interpretations 테이블 마이그레이션"""
    print("Migrating interpretations...")
    
    if force:
        cursor = conn.execute("SELECT id, doc_id FROM interpretations")
        print("  Force mode: Re-extracting all interpretations")
    else:
        cursor = conn.execute("SELECT id, doc_id FROM interpretations WHERE reference_statutes IS NULL")
    interpretations = cursor.fetchall()
    
    total = len(interpretations)
    if total == 0:
        print("  No interpretations to migrate")
        return
    
    print(f"  Found {total} interpretations to migrate")
    
    migrated = 0
    for idx, (interp_id, doc_id) in enumerate(interpretations, 1):
        if idx % 100 == 0:
            print(f"  Progress: {idx}/{total} ({idx*100//total}%)")
        
        # 전체 텍스트 가져오기
        para_cursor = conn.execute(
            "SELECT text FROM interpretation_paragraphs WHERE interpretation_id = ? ORDER BY para_index",
            (interp_id,)
        )
        paragraphs = para_cursor.fetchall()
        full_text = "\n".join([p[0] for p in paragraphs])
        
        if not full_text:
            continue
        
        # 참조조문 추출
        reference_statutes = extractor.extract_from_content(full_text)
        reference_statutes_json = extractor.to_json(reference_statutes) if reference_statutes else None
        
        # 업데이트
        conn.execute(
            "UPDATE interpretations SET reference_statutes = ? WHERE id = ?",
            (reference_statutes_json, interp_id)
        )
        
        if reference_statutes:
            migrated += 1
    
    conn.commit()
    print(f"  Migrated {migrated}/{total} interpretations")


def main():
    parser = argparse.ArgumentParser(
        description="Migrate existing data to add reference_statutes field"
    )
    parser.add_argument(
        "--db",
        default="./data/lawfirm_v2.db",
        help="Database path (default: ./data/lawfirm_v2.db)"
    )
    parser.add_argument(
        "--type",
        choices=["cases", "decisions", "interpretations", "all"],
        default="all",
        help="Type of data to migrate (default: all)"
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Force re-extraction of all records (including already extracted ones)"
    )
    
    args = parser.parse_args()
    
    db_path = Path(args.db)
    if not db_path.exists():
        print(f"Error: Database not found at {db_path}")
        sys.exit(1)
    
    print(f"Migrating reference statutes for {args.type} in {db_path}")
    print("=" * 60)
    
    extractor = ReferenceStatuteExtractor()
    
    with sqlite3.connect(str(db_path)) as conn:
        # 마이그레이션 스키마 확인
        try:
            conn.execute("SELECT reference_statutes FROM cases LIMIT 1")
        except sqlite3.OperationalError:
            print("Error: reference_statutes column not found. Please run migration 003_add_reference_statutes.sql first")
            sys.exit(1)
        
        if args.type in ["cases", "all"]:
            migrate_cases(conn, extractor, force=args.force)
        
        if args.type in ["decisions", "all"]:
            migrate_decisions(conn, extractor, force=args.force)
        
        if args.type in ["interpretations", "all"]:
            migrate_interpretations(conn, extractor, force=args.force)
    
    print("=" * 60)
    print("Migration completed!")


if __name__ == "__main__":
    main()

