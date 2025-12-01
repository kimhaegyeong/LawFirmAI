"""
재임베딩 실행 속도 모니터링 스크립트
"""
import sqlite3
import sys
import time
from pathlib import Path
from datetime import datetime, timedelta

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from scripts.utils.embedding_version_manager import EmbeddingVersionManager

def monitor_speed(db_path: str, version_id: int, interval: int = 60):
    """
    재임베딩 실행 속도 모니터링
    
    Args:
        db_path: 데이터베이스 경로
        version_id: 버전 ID
        interval: 모니터링 간격 (초)
    """
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    
    version_manager = EmbeddingVersionManager(db_path)
    version_info = version_manager.get_version_statistics(version_id)
    
    if not version_info:
        print(f"✗ 버전 ID {version_id}를 찾을 수 없습니다.")
        return
    
    print("=" * 80)
    print("재임베딩 실행 속도 모니터링")
    print("=" * 80)
    print(f"버전: {version_info['version_name']}")
    print(f"모니터링 간격: {interval}초")
    print("=" * 80)
    print()
    
    # 초기 상태 기록
    cursor = conn.execute("""
        SELECT COUNT(DISTINCT source_type || '_' || source_id) as processed
        FROM text_chunks
        WHERE embedding_version_id = ?
    """, (version_id,))
    initial_count = cursor.fetchone()[0]
    initial_time = datetime.now()
    
    print(f"시작 시간: {initial_time.strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"초기 처리된 문서 수: {initial_count:,}개")
    print()
    print("모니터링 시작... (Ctrl+C로 중단)")
    print("-" * 80)
    
    try:
        iteration = 0
        while True:
            time.sleep(interval)
            iteration += 1
            
            # 현재 상태 조회
            cursor = conn.execute("""
                SELECT COUNT(DISTINCT source_type || '_' || source_id) as processed
                FROM text_chunks
                WHERE embedding_version_id = ?
            """, (version_id,))
            current_count = cursor.fetchone()[0]
            
            current_time = datetime.now()
            elapsed = current_time - initial_time
            processed = current_count - initial_count
            
            if processed > 0:
                # 처리 속도 계산
                docs_per_second = processed / elapsed.total_seconds()
                docs_per_minute = docs_per_second * 60
                docs_per_hour = docs_per_minute * 60
                avg_time_per_doc = elapsed.total_seconds() / processed
                
                # 전체 문서 수 조회
                cursor = conn.execute("""
                    SELECT COUNT(DISTINCT source_type || '_' || source_id) as total
                    FROM text_chunks
                    WHERE embedding_version_id = 1
                """)
                total_docs = cursor.fetchone()[0]
                remaining_docs = total_docs - current_count
                
                if docs_per_hour > 0:
                    remaining_hours = remaining_docs / docs_per_hour
                    remaining_time = timedelta(hours=remaining_hours)
                else:
                    remaining_time = timedelta(days=999)
                
                progress = (current_count / total_docs * 100) if total_docs > 0 else 0
                
                print(f"[{iteration}] {current_time.strftime('%H:%M:%S')} | "
                      f"처리: {current_count:,}/{total_docs:,} ({progress:.2f}%) | "
                      f"이번 구간: +{processed:,}개 | "
                      f"속도: {docs_per_hour:.1f} 문서/시간 | "
                      f"문서당: {avg_time_per_doc:.2f}초 | "
                      f"남은 시간: {str(remaining_time).split('.')[0]}")
            else:
                print(f"[{iteration}] {current_time.strftime('%H:%M:%S')} | 처리된 문서 없음 (대기 중...)")
            
            # 초기 상태 업데이트 (다음 구간 계산을 위해)
            initial_count = current_count
            initial_time = current_time
            
    except KeyboardInterrupt:
        print()
        print("-" * 80)
        print("모니터링 중단")
        
        # 최종 상태
        cursor = conn.execute("""
            SELECT COUNT(DISTINCT source_type || '_' || source_id) as processed
            FROM text_chunks
            WHERE embedding_version_id = ?
        """, (version_id,))
        final_count = cursor.fetchone()[0]
        
        total_processed = final_count - initial_count
        total_elapsed = (datetime.now() - initial_time).total_seconds()
        
        if total_processed > 0:
            print(f"총 처리된 문서: {total_processed:,}개")
            print(f"총 경과 시간: {str(timedelta(seconds=int(total_elapsed))).split('.')[0]}")
            print(f"평균 속도: {total_processed / (total_elapsed / 3600):.1f} 문서/시간")
            print(f"문서당 평균 시간: {total_elapsed / total_processed:.2f}초")
    
    conn.close()

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="재임베딩 실행 속도 모니터링")
    parser.add_argument("--db", default="data/lawfirm_v2.db", help="데이터베이스 경로")
    parser.add_argument("--version-id", type=int, required=True, help="버전 ID")
    parser.add_argument("--interval", type=int, default=60, help="모니터링 간격 (초, 기본값: 60)")
    
    args = parser.parse_args()
    
    monitor_speed(args.db, args.version_id, args.interval)

