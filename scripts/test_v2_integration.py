# -*- coding: utf-8 -*-
"""
lawfirm_v2.db 통합 테스트 스크립트
실제 데이터베이스와 연동하여 검색 기능 검증
"""

import os
import sys
from pathlib import Path

# 프로젝트 루트 경로 추가
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

# lawfirm_langgraph 경로 추가
lawfirm_langgraph_path = project_root / "lawfirm_langgraph"
sys.path.insert(0, str(lawfirm_langgraph_path))

import logging

from core.services.search.hybrid_search_engine_v2 import HybridSearchEngineV2
from core.services.search.semantic_search_engine_v2 import SemanticSearchEngineV2
from core.utils.config import Config
from lawfirm_langgraph.langgraph_core.services.legal_data_connector_v2 import (
    LegalDataConnectorV2,
    route_query,
)

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def test_query_routing():
    """Query Router 테스트"""
    print("\n=== Query Router 테스트 ===")

    test_queries = [
        ("제3조 알려줘", "text2sql"),
        ("2023-10-19 시행", "text2sql"),
        ("대법원 판례", "text2sql"),
        ("부당이득 반환 요건", "vector"),
        ("계약 해지 절차", "vector"),
    ]

    for query, expected in test_queries:
        result = route_query(query)
        status = "✅" if result == expected else "❌"
        print(f"{status} '{query}' → {result} (expected: {expected})")


def test_fts_search(config):
    """FTS5 검색 테스트"""
    print("\n=== FTS5 검색 테스트 ===")

    db_path = config.database_path
    if not os.path.exists(db_path):
        print(f"⚠️ 데이터베이스가 없습니다: {db_path}")
        print("먼저 데이터를 적재하세요:")
        print("  python scripts/ingest/ingest_statutes.py --json <path> --domain 민사법")
        return

    connector = LegalDataConnectorV2(db_path)

    # 법령 검색
    print("\n1. 법령 FTS 검색:")
    results = connector.search_statutes_fts("인지", limit=5)
    print(f"   결과: {len(results)}개")
    for i, r in enumerate(results[:3], 1):
        print(f"   {i}. [{r['relevance_score']:.3f}] {r['source']} - {r['content'][:50]}...")

    # 판례 검색
    print("\n2. 판례 FTS 검색:")
    results = connector.search_cases_fts("손해배상", limit=5)
    print(f"   결과: {len(results)}개")
    for i, r in enumerate(results[:3], 1):
        print(f"   {i}. [{r['relevance_score']:.3f}] {r['source']} - {r['content'][:50]}...")

    # 통합 검색 (라우팅)
    print("\n3. 통합 검색 (자동 라우팅):")
    results = connector.search_documents("제1조", limit=5)
    print(f"   Text2SQL 라우팅 결과: {len(results)}개")

    results = connector.search_documents("부당이득", limit=5)
    print(f"   Vector 라우팅 결과: {len(results)}개 (SemanticSearchEngineV2로 위임)")


def test_semantic_search(config):
    """벡터 검색 테스트"""
    print("\n=== 벡터 의미 검색 테스트 ===")

    db_path = config.database_path
    if not os.path.exists(db_path):
        print(f"⚠️ 데이터베이스가 없습니다: {db_path}")
        return

    try:
        engine = SemanticSearchEngineV2(db_path)

        if not engine.embedder:
            print("⚠️ 임베딩 모델을 로드할 수 없습니다. 모델 다운로드가 필요할 수 있습니다.")
            return

        print("\n벡터 검색 실행 중...")
        results = engine.search(
            query="부당이득 반환 청구 요건",
            k=5,
            similarity_threshold=0.3
        )

        print(f"결과: {len(results)}개")
        for i, r in enumerate(results[:3], 1):
            print(f"   {i}. [{r['score']:.3f}] {r['source']}")
            print(f"      {r['text'][:80]}...")

    except Exception as e:
        print(f"❌ 벡터 검색 실패: {e}")
        import traceback
        traceback.print_exc()


def test_hybrid_search(config):
    """하이브리드 검색 테스트"""
    print("\n=== 하이브리드 검색 테스트 ===")

    db_path = config.database_path
    if not os.path.exists(db_path):
        print(f"⚠️ 데이터베이스가 없습니다: {db_path}")
        return

    try:
        engine = HybridSearchEngineV2(db_path)

        print("\n하이브리드 검색 실행 중...")
        result = engine.search(
            query="인지",
            search_types=["law"],
            max_results=10,
            include_exact=True,
            include_semantic=True
        )

        print(f"총 결과: {result['total']}개")
        print(f"FTS5 결과: {result['exact_count']}개")
        print(f"벡터 결과: {result['semantic_count']}개")

        for i, r in enumerate(result['results'][:3], 1):
            print(f"\n   {i}. [{r.get('search_type', 'unknown')}] {r.get('relevance_score', 0):.3f}")
            text = r.get('text', r.get('content', ''))[:80]
            print(f"      {text}...")

    except Exception as e:
        print(f"❌ 하이브리드 검색 실패: {e}")
        import traceback
        traceback.print_exc()


def check_database_status(config):
    """데이터베이스 상태 확인"""
    print("\n=== 데이터베이스 상태 확인 ===")

    db_path = config.database_path

    if not os.path.exists(db_path):
        print(f"❌ 데이터베이스 파일이 없습니다: {db_path}")
        print("\n초기화 방법:")
        print("  python scripts/init_lawfirm_v2_db.py")
        return False

    print(f"✅ 데이터베이스 존재: {db_path}")

    import sqlite3
    try:
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()

        # 테이블 확인
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table' ORDER BY name")
        tables = [row[0] for row in cursor.fetchall()]
        print(f"\n테이블 수: {len(tables)}")

        # 주요 테이블 데이터 확인
        checks = [
            ("domains", "SELECT COUNT(*) FROM domains"),
            ("statutes", "SELECT COUNT(*) FROM statutes"),
            ("statute_articles", "SELECT COUNT(*) FROM statute_articles"),
            ("cases", "SELECT COUNT(*) FROM cases"),
            ("case_paragraphs", "SELECT COUNT(*) FROM case_paragraphs"),
            ("text_chunks", "SELECT COUNT(*) FROM text_chunks"),
            ("embeddings", "SELECT COUNT(*) FROM embeddings"),
        ]

        print("\n데이터 통계:")
        for name, query in checks:
            try:
                cursor.execute(query)
                count = cursor.fetchone()[0]
                print(f"  {name}: {count}개")
            except:
                print(f"  {name}: 테이블 없음")

        # FTS5 테이블 확인
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name LIKE '%_fts'")
        fts_tables = [row[0] for row in cursor.fetchall()]
        print(f"\nFTS5 테이블: {len(fts_tables)}개")
        for table in fts_tables:
            print(f"  - {table}")

        conn.close()
        return True

    except Exception as e:
        print(f"❌ 데이터베이스 접근 실패: {e}")
        return False


def main():
    """메인 테스트 실행"""
    print("=" * 60)
    print("lawfirm_v2.db 통합 테스트")
    print("=" * 60)

    # V2 DB 경로 명시적 설정
    v2_db_path = "./data/lawfirm_v2.db"
    print(f"\n테스트 대상 데이터베이스: {v2_db_path}")

    # Config 객체 생성 (내부적으로 사용)
    config = Config()
    # 테스트에서는 v2 경로를 직접 사용
    config.database_path = v2_db_path

    # 데이터베이스 상태 확인
    if not check_database_status(config):
        print("\n⚠️ 데이터베이스가 준비되지 않았습니다. 테스트를 건너뜁니다.")
        return

    # 테스트 실행
    test_query_routing()
    test_fts_search(config)
    test_semantic_search(config)
    test_hybrid_search(config)

    print("\n" + "=" * 60)
    print("테스트 완료")
    print("=" * 60)


if __name__ == "__main__":
    main()
