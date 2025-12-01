"""
pgvector 인덱스 사용률 모니터링 스크립트
"""

import sys
from pathlib import Path
from datetime import datetime

project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

try:
    from lawfirm_langgraph.core.data.db_adapter import DatabaseAdapter
except ImportError:
    from core.data.db_adapter import DatabaseAdapter

def get_database_url():
    """환경 변수에서 데이터베이스 URL 가져오기 (POSTGRES_* 환경 변수 조합)"""
    import os
    from urllib.parse import quote_plus
    
    # DATABASE_URL이 명시적으로 설정되어 있으면 사용
    database_url = os.getenv("DATABASE_URL")
    if database_url:
        return database_url
    
    # PostgreSQL 환경변수 조합 (프로젝트 루트 .env 파일의 설정 우선 사용)
    postgres_host = os.getenv("POSTGRES_HOST", "localhost")
    postgres_port = os.getenv("POSTGRES_PORT", "5432")
    postgres_db = os.getenv("POSTGRES_DB", "lawfirmai_local")
    postgres_user = os.getenv("POSTGRES_USER", "lawfirmai")
    postgres_password = os.getenv("POSTGRES_PASSWORD", "local_password")
    
    # URL 인코딩 (특수문자 처리)
    encoded_password = quote_plus(postgres_password)
    
    # PostgreSQL URL 생성
    database_url = f"postgresql://{postgres_user}:{encoded_password}@{postgres_host}:{postgres_port}/{postgres_db}"
    return database_url

def get_index_statistics(db_adapter: DatabaseAdapter) -> list:
    """인덱스 통계 정보 조회"""
    with db_adapter.get_connection_context() as conn:
        cursor = conn.cursor()
        
        # pgvector 인덱스 통계 정보 조회
        cursor.execute("""
            SELECT 
                schemaname,
                relname as tablename,
                indexrelname as indexname,
                idx_scan as index_scans,
                idx_tup_read as tuples_read,
                idx_tup_fetch as tuples_fetched,
                pg_size_pretty(pg_relation_size(indexrelid)) as index_size
            FROM pg_stat_user_indexes
            WHERE indexrelid IN (
                SELECT oid FROM pg_class
                WHERE relname IN (
                    SELECT indexname FROM pg_indexes
                    WHERE indexdef LIKE '%vector%' OR indexdef LIKE '%hnsw%' OR indexdef LIKE '%ivfflat%'
                )
            )
            ORDER BY idx_scan DESC, relname, indexrelname
        """)
        
        results = cursor.fetchall()
        return results

def get_table_statistics(db_adapter: DatabaseAdapter) -> list:
    """테이블 통계 정보 조회"""
    with db_adapter.get_connection_context() as conn:
        cursor = conn.cursor()
        
        # 벡터 테이블 통계 정보 조회
        cursor.execute("""
            SELECT 
                schemaname,
                relname as tablename,
                seq_scan as sequential_scans,
                seq_tup_read as sequential_tuples_read,
                idx_scan as index_scans,
                idx_tup_fetch as index_tuples_fetched,
                n_tup_ins as tuples_inserted,
                n_tup_upd as tuples_updated,
                n_tup_del as tuples_deleted,
                n_live_tup as live_tuples,
                n_dead_tup as dead_tuples,
                last_vacuum,
                last_autovacuum,
                last_analyze,
                last_autoanalyze
            FROM pg_stat_user_tables
            WHERE relname IN (
                'statute_embeddings',
                'precedent_chunks',
                'embeddings',
                'interpretation_paragraphs',
                'decision_paragraphs'
            )
            ORDER BY relname
        """)
        
        results = cursor.fetchall()
        return results

def main():
    """메인 함수"""
    print("=" * 80)
    print("pgvector 인덱스 사용률 모니터링")
    print("=" * 80)
    print(f"모니터링 시간: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print()
    
    # 데이터베이스 연결
    database_url = get_database_url()
    db_adapter = DatabaseAdapter(database_url)
    
    # 인덱스 통계 정보
    print("=" * 80)
    print("1. 인덱스 사용 통계")
    print("=" * 80)
    print()
    
    index_stats = get_index_statistics(db_adapter)
    
    if not index_stats:
        print("❌ 인덱스 통계 정보를 가져올 수 없습니다")
    else:
        print(f"{'테이블':<30} {'인덱스':<50} {'스캔 횟수':<15} {'읽은 튜플':<15} {'인덱스 크기':<15}")
        print("-" * 125)
        
        total_scans = 0
        for row in index_stats:
            if isinstance(row, dict):
                schema = row.get('schemaname', '')
                table = row.get('tablename', '')
                index = row.get('indexname', '')
                scans = row.get('index_scans', 0)
                tuples_read = row.get('tuples_read', 0)
                size = row.get('index_size', '')
            else:
                schema, table, index, scans, tuples_read, tuples_fetched, size = row
            
            table_name = f"{schema}.{table}"
            total_scans += scans if isinstance(scans, (int, float)) else 0
            
            print(f"{table_name:<30} {index:<50} {scans:<15} {tuples_read:<15} {size:<15}")
        
        print("-" * 125)
        print(f"총 인덱스 스캔 횟수: {total_scans:,}")
        print()
    
    # 테이블 통계 정보
    print("=" * 80)
    print("2. 테이블 통계 정보")
    print("=" * 80)
    print()
    
    table_stats = get_table_statistics(db_adapter)
    
    if not table_stats:
        print("❌ 테이블 통계 정보를 가져올 수 없습니다")
    else:
        print(f"{'테이블':<30} {'순차 스캔':<15} {'인덱스 스캔':<15} {'인덱스 비율':<15} {'라이브 튜플':<15}")
        print("-" * 90)
        
        for row in table_stats:
            if isinstance(row, dict):
                schema = row.get('schemaname', '')
                table = row.get('tablename', '')
                seq_scans = row.get('sequential_scans', 0)
                idx_scans = row.get('index_scans', 0)
                live_tuples = row.get('live_tuples', 0)
            else:
                schema = row[0]
                table = row[1]
                seq_scans = row[2]
                idx_scans = row[5]
                live_tuples = row[9]
            
            table_name = f"{schema}.{table}"
            total_scans = seq_scans + idx_scans
            index_ratio = (idx_scans / total_scans * 100) if total_scans > 0 else 0
            
            print(f"{table_name:<30} {seq_scans:<15} {idx_scans:<15} {index_ratio:>13.1f}% {live_tuples:<15}")
        
        print("-" * 90)
        print()
    
    # 권장 사항
    print("=" * 80)
    print("3. 권장 사항")
    print("=" * 80)
    print()
    
    if index_stats:
        unused_indexes = [r for r in index_stats if (r[3] if isinstance(r, tuple) else r.get('index_scans', 0)) == 0]
        if unused_indexes:
            print(f"⚠️  사용되지 않는 인덱스: {len(unused_indexes)}개")
            for row in unused_indexes[:5]:  # 최대 5개만 표시
                if isinstance(row, dict):
                    index = row.get('indexname', '')
                else:
                    index = row[2]
                print(f"   - {index}")
            print()
        else:
            print("✅ 모든 인덱스가 사용되고 있습니다")
            print()
    
    if table_stats:
        low_index_ratio_tables = []
        for row in table_stats:
            if isinstance(row, dict):
                table = row.get('tablename', '')
                seq_scans = row.get('sequential_scans', 0)
                idx_scans = row.get('index_scans', 0)
            else:
                table = row[1]
                seq_scans = row[2]
                idx_scans = row[5]
            
            total_scans = seq_scans + idx_scans
            if total_scans > 10:  # 최소 10회 이상 스캔이 있는 경우만
                index_ratio = (idx_scans / total_scans * 100) if total_scans > 0 else 0
                if index_ratio < 50:  # 인덱스 사용률이 50% 미만
                    low_index_ratio_tables.append((table, index_ratio))
        
        if low_index_ratio_tables:
            print(f"⚠️  인덱스 사용률이 낮은 테이블: {len(low_index_ratio_tables)}개")
            for table, ratio in low_index_ratio_tables:
                print(f"   - {table}: {ratio:.1f}%")
            print("   💡 통계 정보 업데이트 권장: python scripts/rag/update_pgvector_stats.py")
            print()
        else:
            print("✅ 모든 테이블의 인덱스 사용률이 양호합니다")
            print()

if __name__ == "__main__":
    main()

