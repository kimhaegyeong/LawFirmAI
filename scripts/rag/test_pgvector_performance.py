"""
pgvector 인덱스 성능 테스트 스크립트
"""

import sys
import time
from pathlib import Path
from typing import List, Dict, Any

project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

try:
    from lawfirm_langgraph.core.data.db_adapter import DatabaseAdapter
    from lawfirm_langgraph.core.search.engines.semantic_search_engine_v2 import SemanticSearchEngineV2
except ImportError:
    from core.data.db_adapter import DatabaseAdapter
    from core.search.engines.semantic_search_engine_v2 import SemanticSearchEngineV2

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

def check_index_usage(query: str, db_adapter: DatabaseAdapter) -> Dict[str, Any]:
    """인덱스 사용 여부 확인"""
    with db_adapter.get_connection_context() as conn:
        cursor = conn.cursor()
        
        # EXPLAIN ANALYZE 실행
        explain_query = f"EXPLAIN (ANALYZE, BUFFERS, VERBOSE) {query}"
        cursor.execute(explain_query)
        explain_result = cursor.fetchall()
        
        # 결과 분석
        plan_text = "\n".join([str(row) for row in explain_result])
        
        # 인덱스 사용 여부 확인
        uses_index = False
        index_name = None
        if "Index Scan" in plan_text or "Bitmap Index Scan" in plan_text:
            uses_index = True
            # 인덱스 이름 추출
            for row in explain_result:
                row_str = str(row)
                if "Index Scan" in row_str or "Bitmap Index Scan" in row_str:
                    # 인덱스 이름 추출 시도
                    if "idx_" in row_str:
                        parts = row_str.split("idx_")
                        if len(parts) > 1:
                            index_part = parts[1].split()[0] if parts[1].split() else ""
                            index_name = f"idx_{index_part}"
                    break
        
        return {
            "uses_index": uses_index,
            "index_name": index_name,
            "plan": plan_text
        }

def test_search_performance(search_engine: SemanticSearchEngineV2, query: str, k: int = 10) -> Dict[str, Any]:
    """검색 성능 테스트"""
    results = []
    
    # 워밍업 (첫 검색 제외)
    try:
        search_engine.search(query, k=1)
    except:
        pass
    
    # 실제 성능 측정 (3회 평균)
    times = []
    result_counts = []
    
    for i in range(3):
        start_time = time.time()
        try:
            search_results = search_engine.search(query, k=k)
            end_time = time.time()
            elapsed = end_time - start_time
            times.append(elapsed)
            result_counts.append(len(search_results) if search_results else 0)
        except Exception as e:
            print(f"  ⚠️  검색 실패 (시도 {i+1}): {e}")
            continue
    
    if not times:
        return {
            "success": False,
            "error": "모든 검색 시도 실패"
        }
    
    avg_time = sum(times) / len(times)
    min_time = min(times)
    max_time = max(times)
    avg_results = sum(result_counts) / len(result_counts) if result_counts else 0
    
    return {
        "success": True,
        "query": query,
        "k": k,
        "avg_time": avg_time,
        "min_time": min_time,
        "max_time": max_time,
        "avg_results": avg_results,
        "times": times
    }

def test_vector_query_performance(db_adapter: DatabaseAdapter, table_name: str, vector_column: str, k: int = 10) -> Dict[str, Any]:
    """벡터 쿼리 성능 테스트 (인덱스 사용 여부 확인)"""
    # 샘플 벡터 가져오기
    with db_adapter.get_connection_context() as conn:
        cursor = conn.cursor()
        
        # 테이블에서 첫 번째 벡터 가져오기
        cursor.execute(f"""
            SELECT {vector_column}
            FROM {table_name}
            WHERE {vector_column} IS NOT NULL
            LIMIT 1
        """)
        
        row = cursor.fetchone()
        if not row:
            return {
                "success": False,
                "error": f"{table_name}에 벡터가 없습니다"
            }
        
        sample_vector = row[0] if isinstance(row, tuple) else row.get(vector_column)
        if not sample_vector:
            return {
                "success": False,
                "error": f"{table_name}에서 벡터를 가져올 수 없습니다"
            }
    
    # 벡터 쿼리 성능 테스트
    query = f"""
        SELECT id, {vector_column} <=> %s::vector AS distance
        FROM {table_name}
        WHERE {vector_column} IS NOT NULL
        ORDER BY distance
        LIMIT {k}
    """
    
    # 워밍업
    try:
        with db_adapter.get_connection_context() as conn:
            cursor = conn.cursor()
            cursor.execute(query, (sample_vector,))
            cursor.fetchall()
    except:
        pass
    
    # 성능 측정 (3회 평균)
    times = []
    for i in range(3):
        start_time = time.time()
        try:
            with db_adapter.get_connection_context() as conn:
                cursor = conn.cursor()
                cursor.execute(query, (sample_vector,))
                results = cursor.fetchall()
                end_time = time.time()
                elapsed = end_time - start_time
                times.append(elapsed)
        except Exception as e:
            print(f"  ⚠️  쿼리 실패 (시도 {i+1}): {e}")
            continue
    
    if not times:
        return {
            "success": False,
            "error": "모든 쿼리 시도 실패"
        }
    
    # 인덱스 사용 여부 확인
    index_info = check_index_usage(query.replace("%s::vector", f"'{sample_vector}'::vector"), db_adapter)
    
    avg_time = sum(times) / len(times)
    min_time = min(times)
    max_time = max(times)
    
    return {
        "success": True,
        "table_name": table_name,
        "k": k,
        "avg_time": avg_time,
        "min_time": min_time,
        "max_time": max_time,
        "uses_index": index_info["uses_index"],
        "index_name": index_info["index_name"],
        "times": times
    }

def main():
    """메인 함수"""
    print("=" * 80)
    print("pgvector 인덱스 성능 테스트")
    print("=" * 80)
    print()
    
    # 데이터베이스 연결
    database_url = get_database_url()
    db_adapter = DatabaseAdapter(database_url)
    
    # 검색 엔진 초기화
    print("검색 엔진 초기화 중...")
    try:
        search_engine = SemanticSearchEngineV2()
        print("✅ 검색 엔진 초기화 완료\n")
    except Exception as e:
        print(f"❌ 검색 엔진 초기화 실패: {e}")
        return
    
    # 테스트 쿼리
    test_queries = [
        "계약 해지",
        "손해배상",
        "임대차 계약",
        "부동산 매매",
        "상속"
    ]
    
    print("=" * 80)
    print("1. 검색 성능 테스트")
    print("=" * 80)
    print()
    
    search_results = []
    for query in test_queries:
        print(f"📝 쿼리: '{query}'")
        result = test_search_performance(search_engine, query, k=10)
        if result["success"]:
            print(f"  ⏱️  평균 시간: {result['avg_time']:.3f}초 (최소: {result['min_time']:.3f}초, 최대: {result['max_time']:.3f}초)")
            print(f"  📊 평균 결과 수: {result['avg_results']:.1f}개")
            search_results.append(result)
        else:
            print(f"  ❌ 실패: {result.get('error', '알 수 없는 오류')}")
        print()
    
    print("=" * 80)
    print("2. 벡터 쿼리 성능 테스트 (인덱스 사용 여부 확인)")
    print("=" * 80)
    print()
    
    # 테스트할 테이블
    test_tables = [
        ("statute_embeddings", "embedding_vector"),
        ("precedent_chunks", "embedding_vector"),
        ("embeddings", "vector")
    ]
    
    vector_results = []
    for table_name, vector_column in test_tables:
        print(f"📊 테이블: {table_name}")
        result = test_vector_query_performance(db_adapter, table_name, vector_column, k=10)
        if result["success"]:
            print(f"  ⏱️  평균 시간: {result['avg_time']:.3f}초 (최소: {result['min_time']:.3f}초, 최대: {result['max_time']:.3f}초)")
            if result["uses_index"]:
                print(f"  ✅ 인덱스 사용: {result['index_name']}")
            else:
                print(f"  ⚠️  인덱스 미사용")
            vector_results.append(result)
        else:
            print(f"  ❌ 실패: {result.get('error', '알 수 없는 오류')}")
        print()
    
    # 결과 요약
    print("=" * 80)
    print("3. 성능 테스트 결과 요약")
    print("=" * 80)
    print()
    
    if search_results:
        avg_search_time = sum(r["avg_time"] for r in search_results) / len(search_results)
        print(f"검색 성능:")
        print(f"  평균 검색 시간: {avg_search_time:.3f}초")
        print(f"  최소 검색 시간: {min(r['min_time'] for r in search_results):.3f}초")
        print(f"  최대 검색 시간: {max(r['max_time'] for r in search_results):.3f}초")
        print()
    
    if vector_results:
        indexed_tables = [r for r in vector_results if r["uses_index"]]
        non_indexed_tables = [r for r in vector_results if not r["uses_index"]]
        
        print(f"벡터 쿼리 성능:")
        if indexed_tables:
            avg_indexed_time = sum(r["avg_time"] for r in indexed_tables) / len(indexed_tables)
            print(f"  인덱스 사용 테이블 평균: {avg_indexed_time:.3f}초 ({len(indexed_tables)}개)")
        if non_indexed_tables:
            avg_non_indexed_time = sum(r["avg_time"] for r in non_indexed_tables) / len(non_indexed_tables)
            print(f"  인덱스 미사용 테이블 평균: {avg_non_indexed_time:.3f}초 ({len(non_indexed_tables)}개)")
        
        if indexed_tables and non_indexed_tables:
            speedup = avg_non_indexed_time / avg_indexed_time
            print(f"  성능 향상: {speedup:.2f}배")
        print()
        
        print("인덱스 사용 현황:")
        for r in vector_results:
            status = "✅ 사용" if r["uses_index"] else "❌ 미사용"
            index_info = f" ({r['index_name']})" if r["uses_index"] and r["index_name"] else ""
            print(f"  {r['table_name']}: {status}{index_info}")

if __name__ == "__main__":
    main()

