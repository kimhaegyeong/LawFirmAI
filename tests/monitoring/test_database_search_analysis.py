#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
데이터베이스 검색 및 검색 결과 전달 분석
"""

import sys
import os
from pathlib import Path

# 프로젝트 루트 경로 추가
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from source.data.database import DatabaseManager
from core.services.search.semantic_search_engine import SemanticSearchEngine
from core.services.search.keyword_search_engine import KeywordSearchEngine

def test_database_connection():
    """데이터베이스 연결 테스트"""
    print("\n" + "="*80)
    print("데이터베이스 검색 분석")
    print("="*80 + "\n")

    try:
        # 데이터베이스 연결
        db_manager = DatabaseManager()

        # 테이블 목록 확인
        tables = db_manager.execute_query("SELECT name FROM sqlite_master WHERE type='table'")
        print(f"📊 데이터베이스 테이블 수: {len(tables)}")
        for table in tables:
            table_name = table.get('name', '')

            # 테이블 행 수 확인
            try:
                count_result = db_manager.execute_query(f"SELECT COUNT(*) as count FROM {table_name}")
                row_count = count_result[0].get('count', 0) if count_result else 0
                print(f"   - {table_name}: {row_count}개 행")

                # 샘플 데이터 확인 (최대 3개)
                if row_count > 0:
                    sample = db_manager.execute_query(f"SELECT * FROM {table_name} LIMIT 3")
                    if sample:
                        print(f"     샘플 데이터: {len(sample)}개")
            except Exception as e:
                print(f"   - {table_name}: 확인 불가 ({e})")

        print()

    except Exception as e:
        print(f"❌ 데이터베이스 연결 실패: {e}")
        return False

    return True


def test_search_engines():
    """검색 엔진 테스트"""
    print("\n" + "="*80)
    print("검색 엔진 분석")
    print("="*80 + "\n")

    try:
        # Semantic Search Engine
        print("🔍 Semantic Search Engine:")
        semantic_search = SemanticSearchEngine()
        if semantic_search:
            print(f"   - 초기화: ✅")
            # FAISS 인덱스 확인
            if hasattr(semantic_search, 'faiss_index') and semantic_search.faiss_index:
                print(f"   - FAISS 인덱스: ✅")
            else:
                print(f"   - FAISS 인덱스: ❌ (없음)")
        else:
            print(f"   - 초기화: ❌")

        print()

        # Keyword Search Engine
        print("🔍 Keyword Search Engine:")
        keyword_search = KeywordSearchEngine()
        if keyword_search:
            print(f"   - 초기화: ✅")
        else:
            print(f"   - 초기화: ❌")

        print()

        # 테스트 검색
        test_query = "손해배상"
        print(f"📝 테스트 검색: '{test_query}'")
        print()

        # 키워드 검색 테스트
        try:
            keyword_results = keyword_search.search(test_query, limit=5)
            print(f"🔎 키워드 검색 결과: {len(keyword_results)}개")
            if keyword_results:
                for i, result in enumerate(keyword_results[:3], 1):
                    print(f"   {i}. {result.get('source', 'Unknown')[:50]}")
                    print(f"      점수: {result.get('relevance_score', 0.0):.3f}")
            else:
                print("   ⚠️ 검색 결과 없음")
        except Exception as e:
            print(f"   ❌ 키워드 검색 실패: {e}")

        print()

        # 의미 검색 테스트 (FAISS 인덱스가 있는 경우)
        try:
            semantic_results = semantic_search.search(test_query, top_k=5)
            print(f"🔎 의미 검색 결과: {len(semantic_results)}개")
            if semantic_results:
                for i, result in enumerate(semantic_results[:3], 1):
                    print(f"   {i}. {result.get('source', 'Unknown')[:50]}")
                    print(f"      점수: {result.get('relevance_score', 0.0):.3f}")
            else:
                print("   ⚠️ 검색 결과 없음 (FAISS 인덱스가 없을 수 있음)")
        except Exception as e:
            print(f"   ❌ 의미 검색 실패: {e}")

        print()

    except Exception as e:
        print(f"❌ 검색 엔진 테스트 실패: {e}")
        import traceback
        traceback.print_exc()
        return False

    return True


if __name__ == "__main__":
    print("="*80)
    print("데이터베이스 및 검색 엔진 분석")
    print("="*80)

    db_ok = test_database_connection()
    search_ok = test_search_engines()

    print("\n" + "="*80)
    print("종합 결과")
    print("="*80)
    print(f"데이터베이스: {'✅' if db_ok else '❌'}")
    print(f"검색 엔진: {'✅' if search_ok else '❌'}")
    print("="*80 + "\n")


