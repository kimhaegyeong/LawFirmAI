#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
?�이?�베?�스 검??�?검??결과 ?�달 분석
"""

import sys
import os
from pathlib import Path

# ?�로?�트 루트 경로 추�?
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from source.data.database import DatabaseManager
from source.services.semantic_search_engine import SemanticSearchEngine
# Note: keyword_search_engine might not exist in source, check if needed

def test_database_connection():
    """?�이?�베?�스 ?�결 ?�스??""
    print("\n" + "="*80)
    print("?�이?�베?�스 검??분석")
    print("="*80 + "\n")

    try:
        # ?�이?�베?�스 ?�결
        db_manager = DatabaseManager()

        # ?�이�?목록 ?�인
        tables = db_manager.execute_query("SELECT name FROM sqlite_master WHERE type='table'")
        print(f"?�� ?�이?�베?�스 ?�이�??? {len(tables)}")
        for table in tables:
            table_name = table.get('name', '')

            # ?�이�??????�인
            try:
                count_result = db_manager.execute_query(f"SELECT COUNT(*) as count FROM {table_name}")
                row_count = count_result[0].get('count', 0) if count_result else 0
                print(f"   - {table_name}: {row_count}�???)

                # ?�플 ?�이???�인 (최�? 3�?
                if row_count > 0:
                    sample = db_manager.execute_query(f"SELECT * FROM {table_name} LIMIT 3")
                    if sample:
                        print(f"     ?�플 ?�이?? {len(sample)}�?)
            except Exception as e:
                print(f"   - {table_name}: ?�인 불�? ({e})")

        print()

    except Exception as e:
        print(f"???�이?�베?�스 ?�결 ?�패: {e}")
        return False

    return True


def test_search_engines():
    """검???�진 ?�스??""
    print("\n" + "="*80)
    print("검???�진 분석")
    print("="*80 + "\n")

    try:
        # Semantic Search Engine
        print("?�� Semantic Search Engine:")
        semantic_search = SemanticSearchEngine()
        if semantic_search:
            print(f"   - 초기?? ??)
            # FAISS ?�덱???�인
            if hasattr(semantic_search, 'faiss_index') and semantic_search.faiss_index:
                print(f"   - FAISS ?�덱?? ??)
            else:
                print(f"   - FAISS ?�덱?? ??(?�음)")
        else:
            print(f"   - 초기?? ??)

        print()

        # Keyword Search Engine
        print("?�� Keyword Search Engine:")
        keyword_search = KeywordSearchEngine()
        if keyword_search:
            print(f"   - 초기?? ??)
        else:
            print(f"   - 초기?? ??)

        print()

        # ?�스??검??
        test_query = "?�해배상"
        print(f"?�� ?�스??검?? '{test_query}'")
        print()

        # ?�워??검???�스??
        try:
            keyword_results = keyword_search.search(test_query, limit=5)
            print(f"?�� ?�워??검??결과: {len(keyword_results)}�?)
            if keyword_results:
                for i, result in enumerate(keyword_results[:3], 1):
                    print(f"   {i}. {result.get('source', 'Unknown')[:50]}")
                    print(f"      ?�수: {result.get('relevance_score', 0.0):.3f}")
            else:
                print("   ?�️ 검??결과 ?�음")
        except Exception as e:
            print(f"   ???�워??검???�패: {e}")

        print()

        # ?��? 검???�스??(FAISS ?�덱?��? ?�는 경우)
        try:
            semantic_results = semantic_search.search(test_query, top_k=5)
            print(f"?�� ?��? 검??결과: {len(semantic_results)}�?)
            if semantic_results:
                for i, result in enumerate(semantic_results[:3], 1):
                    print(f"   {i}. {result.get('source', 'Unknown')[:50]}")
                    print(f"      ?�수: {result.get('relevance_score', 0.0):.3f}")
            else:
                print("   ?�️ 검??결과 ?�음 (FAISS ?�덱?��? ?�을 ???�음)")
        except Exception as e:
            print(f"   ???��? 검???�패: {e}")

        print()

    except Exception as e:
        print(f"??검???�진 ?�스???�패: {e}")
        import traceback
        traceback.print_exc()
        return False

    return True


if __name__ == "__main__":
    print("="*80)
    print("?�이?�베?�스 �?검???�진 분석")
    print("="*80)

    db_ok = test_database_connection()
    search_ok = test_search_engines()

    print("\n" + "="*80)
    print("종합 결과")
    print("="*80)
    print(f"?�이?�베?�스: {'?? if db_ok else '??}")
    print(f"검???�진: {'?? if search_ok else '??}")
    print("="*80 + "\n")


