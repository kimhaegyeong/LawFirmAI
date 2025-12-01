"""
재임베딩 완료 여부 확인 스크립트
"""
import sqlite3
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

def check_complete(db_path: str, version_id: int, threshold: float = 0.99):
    """
    재임베딩 완료 여부 확인
    
    Args:
        db_path: 데이터베이스 경로
        version_id: 버전 ID
        threshold: 완료 임계값 (0.99 = 99% 완료 시 완료로 간주)
    
    Returns:
        bool: 완료 여부
    """
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    
    # 전체 문서 수 조회 (기존 버전 기준)
    cursor = conn.execute("""
        SELECT COUNT(DISTINCT source_type || '_' || source_id) as total
        FROM text_chunks
        WHERE embedding_version_id = 1
    """)
    total_docs = cursor.fetchone()[0]
    
    # 재임베딩된 문서 수 조회
    cursor = conn.execute("""
        SELECT COUNT(DISTINCT source_type || '_' || source_id) as processed
        FROM text_chunks
        WHERE embedding_version_id = ?
    """, (version_id,))
    processed_docs = cursor.fetchone()[0]
    
    conn.close()
    
    if total_docs == 0:
        return False, 0.0
    
    progress = processed_docs / total_docs
    is_complete = progress >= threshold
    
    return is_complete, progress

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="재임베딩 완료 여부 확인")
    parser.add_argument("--db", default="data/lawfirm_v2.db", help="데이터베이스 경로")
    parser.add_argument("--version-id", type=int, required=True, help="버전 ID")
    parser.add_argument("--threshold", type=float, default=0.99, help="완료 임계값")
    
    args = parser.parse_args()
    
    is_complete, progress = check_complete(args.db, args.version_id, args.threshold)
    
    if is_complete:
        print(f"✓ 재임베딩 완료! ({progress*100:.2f}%)")
        sys.exit(0)
    else:
        print(f"재임베딩 진행 중... ({progress*100:.2f}%)")
        sys.exit(1)

