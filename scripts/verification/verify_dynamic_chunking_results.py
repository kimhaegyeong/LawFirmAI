"""
다이나믹 청킹 재임베딩 결과 검증 스크립트
"""
import sqlite3
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from scripts.utils.embedding_version_manager import EmbeddingVersionManager

def verify_dynamic_chunking_results(db_path: str, version_id: int):
    """
    다이나믹 청킹 재임베딩 결과 검증
    
    Args:
        db_path: 데이터베이스 경로
        version_id: 버전 ID
    """
    print("=" * 80)
    print("다이나믹 청킹 재임베딩 결과 검증")
    print("=" * 80)
    
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    
    version_manager = EmbeddingVersionManager(db_path)
    version_info = version_manager.get_version_statistics(version_id)
    
    if not version_info:
        print(f"✗ 버전 ID {version_id}를 찾을 수 없습니다.")
        return False
    
    print(f"\n버전 정보: {version_info['version_name']}")
    print(f"청킹 전략: {version_info['chunking_strategy']}")
    
    # 버전별 청크 통계
    cursor = conn.execute("""
        SELECT 
            source_type,
            COUNT(DISTINCT source_id) as document_count,
            COUNT(*) as chunk_count,
            AVG(LENGTH(text)) as avg_chunk_length,
            MIN(LENGTH(text)) as min_chunk_length,
            MAX(LENGTH(text)) as max_chunk_length
        FROM text_chunks
        WHERE embedding_version_id = ?
        GROUP BY source_type
    """, (version_id,))
    
    print("\n" + "=" * 80)
    print("문서 타입별 통계")
    print("=" * 80)
    
    total_docs = 0
    total_chunks = 0
    
    for row in cursor.fetchall():
        source_type = row['source_type']
        doc_count = row['document_count']
        chunk_count = row['chunk_count']
        avg_length = row['avg_chunk_length']
        min_length = row['min_chunk_length']
        max_length = row['max_chunk_length']
        
        print(f"\n{source_type}:")
        print(f"  문서 수: {doc_count}")
        print(f"  청크 수: {chunk_count}")
        print(f"  평균 청크 길이: {avg_length:.0f}자")
        print(f"  최소 청크 길이: {min_length}자")
        print(f"  최대 청크 길이: {max_length}자")
        print(f"  문서당 평균 청크 수: {chunk_count / doc_count:.1f}")
        
        total_docs += doc_count
        total_chunks += chunk_count
    
    print("\n" + "=" * 80)
    print("전체 통계")
    print("=" * 80)
    print(f"총 문서 수: {total_docs}")
    print(f"총 청크 수: {total_chunks}")
    print(f"문서당 평균 청크 수: {total_chunks / total_docs:.1f}" if total_docs > 0 else "문서당 평균 청크 수: 0")
    
    # 청킹 메타데이터 확인
    cursor = conn.execute("""
        SELECT 
            chunking_strategy,
            COUNT(*) as count
        FROM text_chunks
        WHERE embedding_version_id = ?
        GROUP BY chunking_strategy
    """, (version_id,))
    
    print("\n" + "=" * 80)
    print("청킹 전략 확인")
    print("=" * 80)
    for row in cursor.fetchall():
        print(f"  {row['chunking_strategy']}: {row['count']}개")
    
    # 문서 타입별 청킹 크기 분포 확인
    cursor = conn.execute("""
        SELECT 
            source_type,
            CASE 
                WHEN LENGTH(text) < 800 THEN 'small'
                WHEN LENGTH(text) < 1500 THEN 'medium'
                ELSE 'large'
            END as size_category,
            COUNT(*) as count
        FROM text_chunks
        WHERE embedding_version_id = ?
        GROUP BY source_type, size_category
        ORDER BY source_type, size_category
    """, (version_id,))
    
    print("\n" + "=" * 80)
    print("청크 크기 분포 (문서 타입별)")
    print("=" * 80)
    current_source = None
    for row in cursor.fetchall():
        source_type = row['source_type']
        if source_type != current_source:
            if current_source is not None:
                print()
            print(f"{source_type}:")
            current_source = source_type
        print(f"  {row['size_category']}: {row['count']}개")
    
    # 샘플 청크 확인
    cursor = conn.execute("""
        SELECT 
            source_type,
            source_id,
            chunk_index,
            LENGTH(text) as text_length,
            text
        FROM text_chunks
        WHERE embedding_version_id = ?
        ORDER BY source_type, source_id, chunk_index
        LIMIT 5
    """, (version_id,))
    
    print("\n" + "=" * 80)
    print("샘플 청크 (처음 5개)")
    print("=" * 80)
    for row in cursor.fetchall():
        print(f"\n{row['source_type']}/{row['source_id']} (chunk_index: {row['chunk_index']})")
        print(f"  길이: {row['text_length']}자")
        print(f"  내용: {row['text'][:100]}...")
    
    conn.close()
    
    print("\n" + "=" * 80)
    print("✓ 검증 완료")
    print("=" * 80)
    
    return True

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="다이나믹 청킹 재임베딩 결과 검증")
    parser.add_argument("--db", default="data/lawfirm_v2.db", help="데이터베이스 경로")
    parser.add_argument("--version-id", type=int, required=True, help="버전 ID")
    
    args = parser.parse_args()
    
    verify_dynamic_chunking_results(args.db, args.version_id)

