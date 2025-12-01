"""
재임베딩 성능 확인 스크립트
"""
import sqlite3
import sys
from pathlib import Path
from datetime import datetime, timedelta

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from scripts.utils.embedding_version_manager import EmbeddingVersionManager

def check_performance(db_path: str, version_id: int):
    """
    재임베딩 성능 확인
    
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
    print("재임베딩 성능 확인")
    print("=" * 80)
    print(f"버전: {version_info['version_name']}")
    print()
    
    # 전체 문서 수
    cursor = conn.execute("""
        SELECT COUNT(DISTINCT source_type || '_' || source_id) as total
        FROM text_chunks
        WHERE embedding_version_id = 1
    """)
    total_docs = cursor.fetchone()[0]
    
    # 재임베딩된 문서 수
    cursor = conn.execute("""
        SELECT COUNT(DISTINCT source_type || '_' || source_id) as processed
        FROM text_chunks
        WHERE embedding_version_id = ?
    """, (version_id,))
    processed_docs = cursor.fetchone()[0]
    
    # 버전 생성 시간
    cursor = conn.execute("""
        SELECT created_at
        FROM embedding_versions
        WHERE id = ?
    """, (version_id,))
    created_at_row = cursor.fetchone()
    
    conn.close()
    
    progress = processed_docs / total_docs if total_docs > 0 else 0
    remaining_docs = total_docs - processed_docs
    
    print(f"전체 문서: {total_docs:,}개")
    print(f"처리 완료: {processed_docs:,}개 ({progress*100:.2f}%)")
    print(f"남은 문서: {remaining_docs:,}개")
    print()
    
    # 성능 분석
    if processed_docs > 0 and created_at_row:
        created_at_str = created_at_row['created_at']
        try:
            created_at = datetime.strptime(created_at_str, '%Y-%m-%d %H:%M:%S')
            elapsed = datetime.now() - created_at
            elapsed_seconds = elapsed.total_seconds()
            
            avg_time_per_doc = elapsed_seconds / processed_docs
            docs_per_hour = processed_docs / (elapsed_seconds / 3600) if elapsed_seconds > 0 else 0
            
            remaining_seconds = avg_time_per_doc * remaining_docs
            remaining_time = timedelta(seconds=int(remaining_seconds))
            
            print("=" * 80)
            print("성능 분석")
            print("=" * 80)
            print(f"시작 시간: {created_at.strftime('%Y-%m-%d %H:%M:%S')}")
            print(f"경과 시간: {str(elapsed).split('.')[0]}")
            print(f"문서당 평균 처리 시간: {avg_time_per_doc:.2f}초")
            print(f"처리 속도: {docs_per_hour:.1f} 문서/시간")
            print()
            print("=" * 80)
            print("예상 남은 시간")
            print("=" * 80)
            
            if remaining_seconds < 3600:
                print(f"약 {int(remaining_seconds / 60)}분")
            else:
                hours = int(remaining_seconds / 3600)
                minutes = int((remaining_seconds % 3600) / 60)
                print(f"약 {hours}시간 {minutes}분")
            
            estimated_completion = datetime.now() + remaining_time
            print(f"예상 완료 시간: {estimated_completion.strftime('%Y-%m-%d %H:%M:%S')}")
            print()
            
            # 성능 평가
            print("=" * 80)
            print("성능 평가")
            print("=" * 80)
            
            if avg_time_per_doc < 5:
                status = "✓ 매우 빠름"
                color = "Green"
            elif avg_time_per_doc < 15:
                status = "✓ 빠름"
                color = "Green"
            elif avg_time_per_doc < 30:
                status = "⚠ 보통"
                color = "Yellow"
            else:
                status = "✗ 느림"
                color = "Red"
            
            print(f"문서당 처리 시간: {avg_time_per_doc:.2f}초 - {status}")
            
            if avg_time_per_doc > 30:
                print()
                print("개선 권장사항:")
                print("1. 배치 크기 증가: --doc-batch-size 100 --embedding-batch-size 512")
                print("2. GPU 사용 확인")
                print("3. 데이터베이스 최적화 확인")
            
        except Exception as e:
            print(f"시간 계산 오류: {e}")
    
    print()
    print("=" * 80)
    print("진행률")
    print("=" * 80)
    bar_length = 50
    filled = int(bar_length * progress)
    bar = "█" * filled + "░" * (bar_length - filled)
    print(f"[{bar}] {progress*100:.2f}%")

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="재임베딩 성능 확인")
    parser.add_argument("--db", default="data/lawfirm_v2.db", help="데이터베이스 경로")
    parser.add_argument("--version-id", type=int, required=True, help="버전 ID")
    
    args = parser.parse_args()
    
    check_performance(args.db, args.version_id)

