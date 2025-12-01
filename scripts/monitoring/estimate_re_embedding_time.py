"""
재임베딩 예상 소요 시간 계산 스크립트
"""
import sqlite3
import sys
from pathlib import Path
from datetime import datetime, timedelta

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from scripts.utils.embedding_version_manager import EmbeddingVersionManager

def estimate_time(db_path: str, version_id: int):
    """
    재임베딩 예상 소요 시간 계산
    
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
    print("재임베딩 예상 소요 시간 계산")
    print("=" * 80)
    print()
    
    # 전체 문서 수 조회
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
    
    # 버전 생성 시간 조회
    cursor = conn.execute("""
        SELECT created_at
        FROM embedding_versions
        WHERE id = ?
    """, (version_id,))
    created_at_row = cursor.fetchone()
    
    conn.close()
    
    if total_docs == 0:
        print("✗ 전체 문서 수를 찾을 수 없습니다.")
        return
    
    progress = processed_docs / total_docs
    remaining_docs = total_docs - processed_docs
    
    print(f"전체 문서 수: {total_docs:,}개")
    print(f"처리 완료: {processed_docs:,}개 ({progress*100:.2f}%)")
    print(f"남은 문서: {remaining_docs:,}개")
    print()
    
    # 시간 계산
    if processed_docs > 0 and created_at_row:
        created_at_str = created_at_row['created_at']
        try:
            created_at = datetime.strptime(created_at_str, '%Y-%m-%d %H:%M:%S')
            elapsed = datetime.now() - created_at
            elapsed_seconds = elapsed.total_seconds()
            
            # 문서당 평균 처리 시간
            avg_time_per_doc = elapsed_seconds / processed_docs
            
            # 남은 시간 계산
            remaining_seconds = avg_time_per_doc * remaining_docs
            remaining_time = timedelta(seconds=int(remaining_seconds))
            
            print("=" * 80)
            print("시간 분석")
            print("=" * 80)
            print(f"시작 시간: {created_at.strftime('%Y-%m-%d %H:%M:%S')}")
            print(f"경과 시간: {str(elapsed).split('.')[0]}")
            print(f"문서당 평균 처리 시간: {avg_time_per_doc:.2f}초")
            print()
            print("=" * 80)
            print("예상 남은 시간")
            print("=" * 80)
            
            if remaining_seconds < 60:
                print(f"약 {int(remaining_seconds)}초")
            elif remaining_seconds < 3600:
                minutes = int(remaining_seconds / 60)
                print(f"약 {minutes}분 ({remaining_time})")
            else:
                hours = int(remaining_seconds / 3600)
                minutes = int((remaining_seconds % 3600) / 60)
                print(f"약 {hours}시간 {minutes}분 ({remaining_time})")
            
            # 예상 완료 시간
            estimated_completion = datetime.now() + remaining_time
            print(f"예상 완료 시간: {estimated_completion.strftime('%Y-%m-%d %H:%M:%S')}")
            
            # 처리 속도
            docs_per_hour = processed_docs / (elapsed_seconds / 3600) if elapsed_seconds > 0 else 0
            print(f"처리 속도: 약 {docs_per_hour:.1f} 문서/시간")
            
        except Exception as e:
            print(f"시간 계산 오류: {e}")
            print("수동 계산:")
            print(f"  남은 문서: {remaining_docs:,}개")
            print(f"  예상 소요 시간: 약 2-4시간 (CPU 기준)")
    else:
        print("=" * 80)
        print("예상 소요 시간")
        print("=" * 80)
        print(f"남은 문서: {remaining_docs:,}개")
        print("예상 소요 시간: 약 2-4시간 (CPU 기준)")
        print("(처리된 문서가 없어 정확한 시간 계산 불가)")
    
    print()
    print("=" * 80)
    print("진행률")
    print("=" * 80)
    bar_length = 50
    filled = int(bar_length * progress)
    bar = "█" * filled + "░" * (bar_length - filled)
    print(f"[{bar}] {progress*100:.2f}%")
    print()

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="재임베딩 예상 소요 시간 계산")
    parser.add_argument("--db", default="data/lawfirm_v2.db", help="데이터베이스 경로")
    parser.add_argument("--version-id", type=int, required=True, help="버전 ID")
    
    args = parser.parse_args()
    
    estimate_time(args.db, args.version_id)

