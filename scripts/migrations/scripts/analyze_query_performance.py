#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
쿼리 성능 분석 스크립트
주요 쿼리의 실행 계획을 분석하여 성능 최적화 포인트를 찾습니다.
"""

import os
import sys
from pathlib import Path

# 프로젝트 루트를 Python 경로에 추가
project_root = Path(__file__).parent.parent.parent.parent
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

# 프로젝트 루트의 migrations utils 사용
sys.path.insert(0, str(project_root))
try:
    from scripts.migrations.utils.database import build_database_url
except ImportError:
    # 직접 경로로 임포트
    import importlib.util
    utils_path = project_root / "scripts" / "migrations" / "utils" / "database.py"
    spec = importlib.util.spec_from_file_location("database_utils", str(utils_path))
    database_utils = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(database_utils)
    build_database_url = database_utils.build_database_url

logger = get_logger(__name__)


def get_database_url() -> str:
    """데이터베이스 URL 가져오기 (환경 변수 조합)"""
    db_url = build_database_url()
    if not db_url:
        raise ValueError(
            "데이터베이스 URL을 구성할 수 없습니다. "
            "DATABASE_URL 또는 POSTGRES_* 환경변수(POSTGRES_HOST, POSTGRES_PORT, POSTGRES_DB, POSTGRES_USER, POSTGRES_PASSWORD)를 설정하세요."
        )
    return db_url


def analyze_queries():
    """주요 쿼리 성능 분석"""
    db_url = get_database_url()
    adapter = DatabaseAdapter(db_url)
    
    # 분석할 주요 쿼리 목록
    queries = [
        {
            "name": "precedent_chunks - embedding_vector IS NOT NULL",
            "query": """
                SELECT
                    pc.id,
                    pc.embedding_vector,
                    pc.chunk_content,
                    pc.precedent_content_id,
                    pc.chunk_index,
                    pc.metadata,
                    pc.embedding_version
                FROM precedent_chunks pc
                WHERE pc.embedding_vector IS NOT NULL
                LIMIT 100
            """,
            "params": None
        },
        {
            "name": "precedent_chunks - id IN (배치 조회)",
            "query": """
                SELECT
                    pc.id,
                    pc.embedding_version
                FROM precedent_chunks pc
                WHERE pc.id IN (%s, %s, %s, %s, %s)
            """,
            "params": (1, 2, 3, 4, 5)
        },
        {
            "name": "precedent_chunks - embedding_version 필터링",
            "query": """
                SELECT
                    pc.id,
                    pc.embedding_version
                FROM precedent_chunks pc
                WHERE pc.embedding_version = %s
                LIMIT 100
            """,
            "params": (1,)
        },
        {
            "name": "embeddings - version_id 필터링",
            "query": """
                SELECT chunk_id
                FROM embeddings
                WHERE version_id = %s
                ORDER BY chunk_id
                LIMIT 100
            """,
            "params": (1,)
        },
        {
            "name": "embeddings - chunk_id와 version_id 복합",
            "query": """
                SELECT chunk_id, version_id
                FROM embeddings
                WHERE chunk_id = %s AND version_id = %s
            """,
            "params": (1, 1)
        },
        {
            "name": "statute_embeddings - embedding_vector IS NOT NULL",
            "query": """
                SELECT
                    se.article_id,
                    se.embedding_vector,
                    se.embedding_version
                FROM statute_embeddings se
                WHERE se.embedding_vector IS NOT NULL
                LIMIT 100
            """,
            "params": None
        },
        {
            "name": "statute_embeddings - article_id IN (배치 조회)",
            "query": """
                SELECT article_id
                FROM statute_embeddings
                WHERE article_id IN (%s, %s, %s, %s, %s)
            """,
            "params": (1, 2, 3, 4, 5)
        },
        {
            "name": "precedent_contents - id 조회",
            "query": """
                SELECT
                    pcc.id,
                    pcc.precedent_id,
                    pcc.section_type,
                    pcc.section_content
                FROM precedent_contents pcc
                WHERE pcc.id = %s
            """,
            "params": (1,)
        },
        {
            "name": "precedent_contents - text 필드 조회",
            "query": """
                SELECT id
                FROM precedent_contents
                WHERE text IS NOT NULL AND text != ''
                LIMIT 100
            """,
            "params": None
        },
        {
            "name": "embedding_versions - is_active와 data_type",
            "query": """
                SELECT version
                FROM embedding_versions
                WHERE is_active = TRUE AND data_type = %s
            """,
            "params": ('precedents',)
        },
        {
            "name": "precedent_chunks JOIN precedent_contents",
            "query": """
                SELECT
                    pc.id,
                    pcc.precedent_id,
                    p.section_content
                FROM precedent_chunks pc
                JOIN precedent_contents pcc ON pc.precedent_content_id = pcc.id
                JOIN precedents p ON pcc.precedent_id = p.id
                WHERE pc.id = %s
            """,
            "params": (1,)
        }
    ]
    
    print("=" * 80)
    print("쿼리 성능 분석 시작")
    print("=" * 80)
    print()
    
    results = []
    
    for i, query_info in enumerate(queries, 1):
        print(f"[{i}/{len(queries)}] 분석 중: {query_info['name']}")
        print("-" * 80)
        
        try:
            result = adapter.analyze_query_performance(
                query_info['query'],
                query_info['params']
            )
            
            results.append({
                'name': query_info['name'],
                'result': result
            })
            
            print(f"실행 시간: {result['execution_time']:.3f}초")
            print("\n실행 계획:")
            print(result['explain_plan'])
            print()
            
        except Exception as e:
            logger.error(f"쿼리 분석 실패: {query_info['name']}")
            logger.error(f"에러: {e}")
            print(f"❌ 분석 실패: {e}\n")
    
    # 요약 출력
    print("=" * 80)
    print("분석 요약")
    print("=" * 80)
    print()
    
    slow_queries = [
        (r['name'], r['result']['execution_time'])
        for r in results
        if r['result']['execution_time'] > 0.1
    ]
    
    if slow_queries:
        print("⚠️  느린 쿼리 (0.1초 이상):")
        for name, time_taken in sorted(slow_queries, key=lambda x: x[1], reverse=True):
            print(f"  - {name}: {time_taken:.3f}초")
    else:
        print("✅ 모든 쿼리가 0.1초 이하로 실행됩니다.")
    
    print()
    print("=" * 80)
    print("분석 완료")
    print("=" * 80)
    
    adapter.close()


if __name__ == "__main__":
    try:
        analyze_queries()
    except Exception as e:
        logger.error(f"스크립트 실행 실패: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

