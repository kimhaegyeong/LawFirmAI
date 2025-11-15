"""
재임베딩 진행 상황 모니터링 스크립트
"""
import sqlite3
import sys
from pathlib import Path
from datetime import datetime

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from scripts.utils.embedding_version_manager import EmbeddingVersionManager

def monitor_progress(db_path: str, version_id: int):
    """
    재임베딩 진행 상황 모니터링
    
    Args:
        db_path: 데이터베이스 경로
        version_id: 버전 ID
    """
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    
    version_manager = EmbeddingVersionManager(db_path)
    version_info = version_manager.get_version_statistics(version_id)
    
    if not version_info:
        print(f"✗ 버전 ID {version_id}를 찾을 수 없습니다.")
        return
    
    print("=" * 80)
    print(f"재임베딩 진행 상황 모니터링: {version_info['version_name']}")
    print("=" * 80)
    
    # 전체 문서 수 조회
    cursor = conn.execute("""
        SELECT 
            source_type,
            COUNT(DISTINCT source_id) as total_documents
        FROM (
            SELECT DISTINCT source_type, source_id 
            FROM text_chunks 
            WHERE embedding_version_id = 1
        )
        GROUP BY source_type
    """)
    
    total_docs_by_type = {}
    total_docs = 0
    for row in cursor.fetchall():
        source_type = row['source_type']
        count = row['total_documents']
        total_docs_by_type[source_type] = count
        total_docs += count
    
    # 재임베딩된 문서 수 조회
    cursor = conn.execute("""
        SELECT 
            source_type,
            COUNT(DISTINCT source_id) as processed_documents,
            COUNT(*) as chunk_count
        FROM text_chunks
        WHERE embedding_version_id = ?
        GROUP BY source_type
    """, (version_id,))
    
    processed_docs_by_type = {}
    total_processed = 0
    total_chunks = 0
    
    for row in cursor.fetchall():
        source_type = row['source_type']
        processed = row['processed_documents']
        chunks = row['chunk_count']
        processed_docs_by_type[source_type] = {
            'documents': processed,
            'chunks': chunks
        }
        total_processed += processed
        total_chunks += chunks
    
    # 진행률 계산
    print(f"\n전체 문서 수: {total_docs}")
    print(f"재임베딩 완료: {total_processed} ({total_processed/total_docs*100:.1f}%)" if total_docs > 0 else "재임베딩 완료: 0")
    print(f"생성된 청크 수: {total_chunks}")
    print()
    
    print("=" * 80)
    print("문서 타입별 진행 상황")
    print("=" * 80)
    
    for source_type in sorted(set(list(total_docs_by_type.keys()) + list(processed_docs_by_type.keys()))):
        total = total_docs_by_type.get(source_type, 0)
        processed_info = processed_docs_by_type.get(source_type, {'documents': 0, 'chunks': 0})
        processed = processed_info['documents']
        chunks = processed_info['chunks']
        
        progress = (processed / total * 100) if total > 0 else 0
        
        print(f"\n{source_type}:")
        print(f"  전체 문서: {total}")
        print(f"  완료: {processed} ({progress:.1f}%)")
        print(f"  청크 수: {chunks}")
        if processed > 0:
            print(f"  문서당 평균 청크: {chunks / processed:.1f}")
        
        # 진행 바
        bar_length = 40
        filled = int(bar_length * progress / 100)
        bar = "█" * filled + "░" * (bar_length - filled)
        print(f"  [{bar}] {progress:.1f}%")
    
    conn.close()
    
    print("\n" + "=" * 80)
    print(f"마지막 업데이트: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 80)

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="재임베딩 진행 상황 모니터링")
    parser.add_argument("--db", default="data/lawfirm_v2.db", help="데이터베이스 경로")
    parser.add_argument("--version-id", type=int, required=True, help="버전 ID")
    
    args = parser.parse_args()
    
    monitor_progress(args.db, args.version_id)

