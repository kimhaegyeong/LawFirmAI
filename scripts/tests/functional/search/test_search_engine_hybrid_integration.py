#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
검색 엔진 하이브리드 청킹 통합 테스트

SemanticSearchEngineV2의 하이브리드 청킹 지원 기능을 테스트합니다.
"""
import sys
from pathlib import Path

# 프로젝트 루트를 sys.path에 추가
_CURRENT_FILE = Path(__file__).resolve()
_PROJECT_ROOT = _CURRENT_FILE.parents[1]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from lawfirm_langgraph.core.search.engines.semantic_search_engine_v2 import SemanticSearchEngineV2


def test_hybrid_chunking_search():
    """하이브리드 청킹 검색 테스트"""
    print("=" * 80)
    print("검색 엔진 하이브리드 청킹 통합 테스트")
    print("=" * 80)
    
    # 검색 엔진 초기화
    db_path = "data/lawfirm_v2.db"
    engine = SemanticSearchEngineV2(db_path=db_path)
    
    if not engine.is_available():
        print("❌ 검색 엔진을 사용할 수 없습니다.")
        return
    
    test_query = "전세금 반환 보증"
    print(f"\n검색 쿼리: {test_query}\n")
    
    # 1. 기본 검색 (모든 청킹 전략)
    print("1. 기본 검색 (모든 청킹 전략)")
    print("-" * 80)
    results = engine.search(
        query=test_query,
        k=5,
        similarity_threshold=0.5
    )
    print(f"   결과 수: {len(results)}")
    for i, result in enumerate(results[:3], 1):
        print(f"   결과 {i}:")
        print(f"     유사도: {result.get('similarity', 0):.4f}")
        print(f"     청킹 전략: {result.get('metadata', {}).get('chunking_strategy', 'N/A')}")
        print(f"     크기 카테고리: {result.get('metadata', {}).get('chunk_size_category', 'N/A')}")
        print(f"     그룹 ID: {str(result.get('metadata', {}).get('chunk_group_id', 'N/A'))[:30]}")
    
    # 2. 하이브리드 청킹만 검색 (중복 제거 활성화)
    print("\n2. 하이브리드 청킹 검색 (중복 제거 활성화)")
    print("-" * 80)
    results_hybrid = engine.search(
        query=test_query,
        k=5,
        similarity_threshold=0.5,
        deduplicate_by_group=True
    )
    # 하이브리드 청킹 결과만 필터링
    hybrid_results = [r for r in results_hybrid if r.get('metadata', {}).get('chunking_strategy') == 'hybrid']
    print(f"   하이브리드 청킹 결과 수: {len(hybrid_results)}")
    for i, result in enumerate(hybrid_results[:3], 1):
        print(f"   결과 {i}:")
        print(f"     유사도: {result.get('similarity', 0):.4f}")
        print(f"     크기 카테고리: {result.get('metadata', {}).get('chunk_size_category', 'N/A')}")
        print(f"     그룹 ID: {str(result.get('metadata', {}).get('chunk_group_id', 'N/A'))[:30]}")
    
    # 3. 특정 크기 카테고리만 검색
    print("\n3. 특정 크기 카테고리 검색 (small)")
    print("-" * 80)
    results_small = engine.search(
        query=test_query,
        k=5,
        similarity_threshold=0.5,
        chunk_size_category='small'
    )
    print(f"   Small 카테고리 결과 수: {len(results_small)}")
    for i, result in enumerate(results_small[:3], 1):
        print(f"   결과 {i}:")
        print(f"     유사도: {result.get('similarity', 0):.4f}")
        print(f"     크기 카테고리: {result.get('metadata', {}).get('chunk_size_category', 'N/A')}")
    
    print("\n" + "=" * 80)
    print("✅ 검색 엔진 하이브리드 청킹 통합 테스트 완료!")
    print("=" * 80)


if __name__ == '__main__':
    test_hybrid_chunking_search()

