#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
데이터베이스 연결 최적화 테스트 스크립트
연결 유지 시간 모니터링 및 쿼리 성능을 테스트합니다.
"""

import os
import sys
from pathlib import Path

# 프로젝트 루트를 Python 경로에 추가
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

# .env 파일 로드
try:
    from utils.env_loader import ensure_env_loaded
    ensure_env_loaded(project_root)
except ImportError:
    try:
        from dotenv import load_dotenv
        root_env = project_root / ".env"
        if root_env.exists():
            load_dotenv(dotenv_path=str(root_env), override=False)
    except ImportError:
        pass

from lawfirm_langgraph.core.data.db_adapter import DatabaseAdapter
from lawfirm_langgraph.core.utils.logger import get_logger
import time
import os
from urllib.parse import quote_plus


def build_database_url():
    """데이터베이스 URL 구성"""
    # DATABASE_URL 환경변수 확인
    db_url = os.getenv('DATABASE_URL')
    if db_url and db_url.strip():
        return db_url
    
    # POSTGRES_* 환경변수로 구성
    host = os.getenv('POSTGRES_HOST', 'localhost')
    port = os.getenv('POSTGRES_PORT', '5432')
    db = os.getenv('POSTGRES_DB', 'lawfirmai_local')
    user = os.getenv('POSTGRES_USER', 'lawfirmai')
    password = os.getenv('POSTGRES_PASSWORD', 'local_password')
    
    # password는 URL 인코딩 필요
    encoded_password = quote_plus(password)
    url = f"postgresql://{user}:{encoded_password}@{host}:{port}/{db}"
    return url

logger = get_logger(__name__)


def test_connection_monitoring():
    """연결 유지 시간 모니터링 테스트"""
    print("=" * 80)
    print("연결 유지 시간 모니터링 테스트")
    print("=" * 80)
    print()
    
    db_url = build_database_url()
    if not db_url:
        print("❌ 데이터베이스 URL을 구성할 수 없습니다.")
        return False
    
    adapter = DatabaseAdapter(db_url)
    
    # 테스트 쿼리 실행
    test_queries = [
        "SELECT COUNT(*) FROM precedent_chunks LIMIT 1",
        "SELECT id, embedding_version FROM precedent_chunks WHERE id IN (1, 2, 3, 4, 5)",
        "SELECT id FROM precedent_chunks WHERE embedding_version = 1 LIMIT 10",
    ]
    
    print("테스트 쿼리 실행 중...")
    print()
    
    for i, query in enumerate(test_queries, 1):
        print(f"[{i}/{len(test_queries)}] 쿼리 실행: {query[:60]}...")
        start_time = time.time()
        
        try:
            with adapter.get_connection_context() as conn:
                cursor = conn.cursor()
                cursor.execute(query)
                result = cursor.fetchall()
                elapsed = time.time() - start_time
                
                print(f"  ✅ 실행 완료: {len(result)}개 결과, {elapsed:.3f}초")
        except Exception as e:
            elapsed = time.time() - start_time
            print(f"  ❌ 실행 실패 ({elapsed:.3f}초): {e}")
        
        print()
    
    print("=" * 80)
    print("테스트 완료")
    print("=" * 80)
    
    # adapter는 마지막에 한 번만 닫음
    return True


def test_batch_query():
    """배치 쿼리 테스트"""
    print("=" * 80)
    print("배치 쿼리 테스트")
    print("=" * 80)
    print()
    
    db_url = build_database_url()
    if not db_url:
        print("❌ 데이터베이스 URL을 구성할 수 없습니다.")
        return False
    
    adapter = DatabaseAdapter(db_url)
    
    # 배치 쿼리 테스트
    chunk_ids = list(range(1, 21))  # 20개의 chunk_id
    
    print(f"배치 쿼리 테스트: {len(chunk_ids)}개의 chunk_id 조회")
    print()
    
    # 개별 쿼리 방식 (비교용)
    print("1. 개별 쿼리 방식 (비교용):")
    start_time = time.time()
    individual_results = {}
    
    try:
        with adapter.get_connection_context() as conn:
            for chunk_id in chunk_ids[:5]:  # 처음 5개만 테스트
                cursor = conn.cursor()
                cursor.execute(
                    "SELECT id, embedding_version FROM precedent_chunks WHERE id = %s",
                    (chunk_id,)
                )
                row = cursor.fetchone()
                if row:
                    individual_results[chunk_id] = row.get('embedding_version') if hasattr(row, 'get') else row[0]
    except Exception as e:
        print(f"  ❌ 개별 쿼리 실행 중 오류: {e}")
    
    individual_time = time.time() - start_time
    print(f"  실행 시간: {individual_time:.3f}초")
    print()
    
    # 배치 쿼리 방식
    print("2. 배치 쿼리 방식:")
    start_time = time.time()
    batch_results = {}
    
    try:
        with adapter.get_connection_context() as conn:
            placeholders = ','.join(['%s'] * len(chunk_ids))
            cursor = conn.cursor()
            cursor.execute(
                f"SELECT id, embedding_version FROM precedent_chunks WHERE id IN ({placeholders})",
                chunk_ids
            )
            rows = cursor.fetchall()
            for row in rows:
                if hasattr(row, 'keys'):
                    chunk_id = row['id']
                    embedding_version = row.get('embedding_version')
                else:
                    chunk_id = row[0] if len(row) > 0 else None
                    embedding_version = row[1] if len(row) > 1 else None
                if chunk_id:
                    batch_results[chunk_id] = embedding_version
    except Exception as e:
        print(f"  ❌ 배치 쿼리 실행 중 오류: {e}")
    
    batch_time = time.time() - start_time
    print(f"  실행 시간: {batch_time:.3f}초")
    print(f"  결과 수: {len(batch_results)}개")
    print()
    
    # 성능 비교
    if individual_time > 0 and batch_time > 0:
        improvement = ((individual_time - batch_time) / individual_time) * 100
        speedup = individual_time / batch_time
        print(f"성능 개선: {improvement:.1f}% (배치 쿼리가 {speedup:.1f}배 빠름)")
    elif batch_time == 0:
        print("성능 개선: 배치 쿼리가 매우 빠름 (< 0.001초)")
    else:
        print("성능 비교: 시간 측정 불가")
    
    print("=" * 80)
    print("배치 쿼리 테스트 완료")
    print("=" * 80)
    
    adapter.close()
    return True


def test_connection_pool():
    """연결 풀 상태 테스트"""
    print("=" * 80)
    print("연결 풀 상태 테스트")
    print("=" * 80)
    print()
    
    db_url = build_database_url()
    if not db_url:
        print("❌ 데이터베이스 URL을 구성할 수 없습니다.")
        return False
    
    adapter = DatabaseAdapter(db_url)
    
    # 연결 풀 상태 조회
    pool_status = adapter.get_pool_status()
    
    print("연결 풀 상태:")
    print(f"  상태: {pool_status.get('status', 'unknown')}")
    print(f"  최소 연결 수: {pool_status.get('minconn', 'N/A')}")
    print(f"  최대 연결 수: {pool_status.get('maxconn', 'N/A')}")
    print(f"  활성 연결 수: {pool_status.get('active_connections', 'N/A')}")
    print(f"  사용 가능 연결 수: {pool_status.get('available_connections', 'N/A')}")
    print(f"  사용률: {pool_status.get('utilization', 0) * 100:.1f}%")
    print()
    
    print("=" * 80)
    print("연결 풀 상태 테스트 완료")
    print("=" * 80)
    
    # adapter는 마지막에 한 번만 닫음
    return True


if __name__ == "__main__":
    try:
        print("데이터베이스 연결 최적화 테스트 시작")
        print()
        
        # 테스트 실행
        results = []
        
        results.append(("연결 유지 시간 모니터링", test_connection_monitoring()))
        print()
        
        results.append(("배치 쿼리", test_batch_query()))
        print()
        
        results.append(("연결 풀 상태", test_connection_pool()))
        print()
        
        # 결과 요약
        print("=" * 80)
        print("테스트 결과 요약")
        print("=" * 80)
        for test_name, success in results:
            status = "✅ 통과" if success else "❌ 실패"
            print(f"{test_name}: {status}")
        
        all_passed = all(success for _, success in results)
        print()
        if all_passed:
            print("✅ 모든 테스트 통과!")
        else:
            print("⚠️  일부 테스트 실패")
        
        # 마지막에 adapter 닫기
        try:
            db_url = build_database_url()
            if db_url:
                adapter = DatabaseAdapter(db_url)
                adapter.close()
        except Exception:
            pass  # 이미 닫혔을 수 있음
        
        sys.exit(0 if all_passed else 1)
        
    except Exception as e:
        logger.error(f"테스트 실행 실패: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

