#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
data_type별 pgvector 검색 통합 테스트 (PostgreSQL 전용)

이 테스트는 실제 PostgreSQL 데이터베이스에 연결하여
data_type별로 올바른 버전이 선택되고 검색이 정상적으로 동작하는지 확인합니다.
"""

import sys
import os
from pathlib import Path

# 프로젝트 경로 설정 (하위 폴더로 이동하여 parent 하나 추가)
script_dir = Path(__file__).parent
tests_dir = script_dir.parent
unit_dir = tests_dir.parent
lawfirm_langgraph_dir = unit_dir.parent
project_root = lawfirm_langgraph_dir.parent

if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))
if str(lawfirm_langgraph_dir) not in sys.path:
    sys.path.insert(0, str(lawfirm_langgraph_dir))

# 환경 변수 로드
try:
    from utils.env_loader import ensure_env_loaded
    ensure_env_loaded(project_root)
except ImportError:
    pass
except Exception:
    pass

from lawfirm_langgraph.core.search.engines.semantic_search_engine_v2 import SemanticSearchEngineV2
from lawfirm_langgraph.core.utils.logger import get_logger

logger = get_logger(__name__)


def test_statutes_search_with_data_type():
    """statutes data_type으로 검색 테스트"""
    print("\n" + "=" * 80)
    print("테스트 1: statutes data_type 검색")
    print("=" * 80)
    
    try:
        engine = SemanticSearchEngineV2()
        
        # statutes 활성 버전 확인
        statutes_version_id = engine._get_active_embedding_version_id(data_type='statutes')
        print(f"\n📊 statutes 활성 버전 ID: {statutes_version_id}")
        
        if not statutes_version_id:
            print("⚠️ statutes 활성 버전이 없습니다. 테스트를 건너뜁니다.")
            return False
        
        # statute_article로 검색
        query = "계약 해지"
        k = 5
        source_types = ['statute_article']
        
        print(f"\n🔍 검색 실행:")
        print(f"   Query: {query}")
        print(f"   Source Types: {source_types}")
        print(f"   K: {k}")
        
        results = engine.search(
            query=query,
            k=k,
            source_types=source_types,
            similarity_threshold=0.5
        )
        
        print(f"\n📊 검색 결과: {len(results)}개")
        
        if results:
            print("✅ statutes 검색 성공!")
            
            # 결과 상세 정보
            for i, result in enumerate(results[:3], 1):
                if isinstance(result, dict):
                    print(f"\n   결과 {i}:")
                    print(f"      ID: {result.get('chunk_id', result.get('id', 'N/A'))}")
                    print(f"      Source Type: {result.get('source_type', 'N/A')}")
                    print(f"      Similarity: {result.get('similarity', 'N/A')}")
                    if 'embedding_version' in result:
                        print(f"      Embedding Version: {result.get('embedding_version', 'N/A')}")
                elif isinstance(result, tuple):
                    print(f"\n   결과 {i}:")
                    print(f"      {result}")
            
            # 버전 확인
            versions_found = set()
            for result in results:
                if isinstance(result, dict):
                    version = result.get('embedding_version')
                    if version:
                        versions_found.add(version)
            
            if versions_found:
                print(f"\n   발견된 버전: {versions_found}")
                if statutes_version_id in versions_found:
                    print(f"   ✅ 활성 버전 {statutes_version_id}이 검색 결과에 포함됨")
                else:
                    print(f"   ⚠️ 활성 버전 {statutes_version_id}이 검색 결과에 없음")
            
            return True
        else:
            print("⚠️ statutes 검색 결과 없음")
            return False
            
    except Exception as e:
        print(f"❌ statutes 검색 실패: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_precedents_search_with_data_type():
    """precedents data_type으로 검색 테스트"""
    print("\n" + "=" * 80)
    print("테스트 2: precedents data_type 검색")
    print("=" * 80)
    
    try:
        engine = SemanticSearchEngineV2()
        
        # precedents 활성 버전 확인
        precedents_version_id = engine._get_active_embedding_version_id(data_type='precedents')
        print(f"\n📊 precedents 활성 버전 ID: {precedents_version_id}")
        
        if not precedents_version_id:
            print("⚠️ precedents 활성 버전이 없습니다. 테스트를 건너뜁니다.")
            return False
        
        # case_paragraph로 검색
        query = "계약 해지"
        k = 5
        source_types = ['case_paragraph']
        
        print(f"\n🔍 검색 실행:")
        print(f"   Query: {query}")
        print(f"   Source Types: {source_types}")
        print(f"   K: {k}")
        
        results = engine.search(
            query=query,
            k=k,
            source_types=source_types,
            similarity_threshold=0.5
        )
        
        print(f"\n📊 검색 결과: {len(results)}개")
        
        if results:
            print("✅ precedents 검색 성공!")
            
            # 결과 상세 정보
            for i, result in enumerate(results[:3], 1):
                if isinstance(result, dict):
                    print(f"\n   결과 {i}:")
                    print(f"      ID: {result.get('chunk_id', result.get('id', 'N/A'))}")
                    print(f"      Source Type: {result.get('source_type', 'N/A')}")
                    print(f"      Similarity: {result.get('similarity', 'N/A')}")
                    if 'embedding_version' in result:
                        print(f"      Embedding Version: {result.get('embedding_version', 'N/A')}")
                elif isinstance(result, tuple):
                    print(f"\n   결과 {i}:")
                    print(f"      {result}")
            
            # 버전 확인
            versions_found = set()
            for result in results:
                if isinstance(result, dict):
                    version = result.get('embedding_version')
                    if version:
                        versions_found.add(version)
            
            if versions_found:
                print(f"\n   발견된 버전: {versions_found}")
                if precedents_version_id in versions_found:
                    print(f"   ✅ 활성 버전 {precedents_version_id}이 검색 결과에 포함됨")
                else:
                    print(f"   ⚠️ 활성 버전 {precedents_version_id}이 검색 결과에 없음")
            
            return True
        else:
            print("⚠️ precedents 검색 결과 없음")
            return False
            
    except Exception as e:
        print(f"❌ precedents 검색 실패: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_mixed_search_without_data_type():
    """혼합 검색 (data_type 지정 없음) 테스트"""
    print("\n" + "=" * 80)
    print("테스트 3: 혼합 검색 (data_type 지정 없음)")
    print("=" * 80)
    
    try:
        engine = SemanticSearchEngineV2()
        
        # 활성 버전 확인
        any_version_id = engine._get_active_embedding_version_id()
        statutes_version_id = engine._get_active_embedding_version_id(data_type='statutes')
        precedents_version_id = engine._get_active_embedding_version_id(data_type='precedents')
        
        print(f"\n📊 활성 버전 정보:")
        print(f"   전체 활성 버전 ID: {any_version_id}")
        print(f"   statutes 활성 버전 ID: {statutes_version_id}")
        print(f"   precedents 활성 버전 ID: {precedents_version_id}")
        
        # 혼합 검색 (source_types에 여러 타입 포함)
        query = "계약 해지"
        k = 10
        source_types = ['statute_article', 'case_paragraph']
        
        print(f"\n🔍 검색 실행:")
        print(f"   Query: {query}")
        print(f"   Source Types: {source_types} (혼합)")
        print(f"   K: {k}")
        
        results = engine.search(
            query=query,
            k=k,
            source_types=source_types,
            similarity_threshold=0.5
        )
        
        print(f"\n📊 검색 결과: {len(results)}개")
        
        if results:
            print("✅ 혼합 검색 성공!")
            
            # 타입별 분포 확인
            type_counts = {}
            for result in results:
                if isinstance(result, dict):
                    source_type = result.get('source_type', 'unknown')
                    type_counts[source_type] = type_counts.get(source_type, 0) + 1
            
            print(f"\n   타입별 분포: {type_counts}")
            
            # statutes와 precedents가 모두 포함되어 있는지 확인
            has_statutes = any('statute' in st.lower() for st in type_counts.keys())
            has_precedents = any('case' in st.lower() or 'precedent' in st.lower() for st in type_counts.keys())
            
            if has_statutes and has_precedents:
                print("   ✅ statutes와 precedents가 모두 검색 결과에 포함됨")
            elif has_statutes:
                print("   ⚠️ statutes만 검색 결과에 포함됨")
            elif has_precedents:
                print("   ⚠️ precedents만 검색 결과에 포함됨")
            
            return True
        else:
            print("⚠️ 혼합 검색 결과 없음")
            return False
            
    except Exception as e:
        print(f"❌ 혼합 검색 실패: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_version_selection_logic():
    """버전 선택 로직 확인"""
    print("\n" + "=" * 80)
    print("테스트 4: 버전 선택 로직 확인")
    print("=" * 80)
    
    try:
        engine = SemanticSearchEngineV2()
        
        # 각 data_type별 활성 버전 조회
        statutes_version_id = engine._get_active_embedding_version_id(data_type='statutes')
        precedents_version_id = engine._get_active_embedding_version_id(data_type='precedents')
        any_version_id = engine._get_active_embedding_version_id()
        
        print(f"\n📊 활성 버전 조회 결과:")
        print(f"   statutes 활성 버전 ID: {statutes_version_id}")
        print(f"   precedents 활성 버전 ID: {precedents_version_id}")
        print(f"   전체 활성 버전 ID (data_type 지정 없음): {any_version_id}")
        
        # data_type 결정 로직 테스트
        print(f"\n📊 data_type 결정 로직 테스트:")
        
        test_cases = [
            (['statute_article'], 'statutes'),
            (['statute_articles'], 'statutes'),
            (['case_paragraph'], 'precedents'),
            (['precedent_content'], 'precedents'),
            (['statute_article', 'case_paragraph'], None),
        ]
        
        for source_types, expected_data_type in test_cases:
            data_type = engine._determine_data_type_from_source_types(source_types)
            status = "✅" if data_type == expected_data_type else "❌"
            print(f"   {status} {source_types} -> {data_type} (expected: {expected_data_type})")
        
        # statutes와 precedents가 다른 버전을 사용하는지 확인
        if statutes_version_id and precedents_version_id:
            if statutes_version_id != precedents_version_id:
                print(f"\n   ✅ statutes와 precedents가 서로 다른 버전 사용")
                print(f"      (statutes: {statutes_version_id}, precedents: {precedents_version_id})")
            else:
                print(f"\n   ℹ️ statutes와 precedents가 같은 버전 사용")
                print(f"      (버전 ID: {statutes_version_id})")
        
        return True
        
    except Exception as e:
        print(f"❌ 버전 선택 로직 확인 실패: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """모든 테스트 실행"""
    print("=" * 80)
    print("data_type별 pgvector 검색 통합 테스트")
    print("=" * 80)
    print("\n이 테스트는 실제 PostgreSQL 데이터베이스에 연결합니다.")
    print("환경 변수가 올바르게 설정되어 있는지 확인하세요.\n")
    
    results = []
    
    # 테스트 실행
    results.append(("버전 선택 로직 확인", test_version_selection_logic()))
    results.append(("statutes 검색", test_statutes_search_with_data_type()))
    results.append(("precedents 검색", test_precedents_search_with_data_type()))
    results.append(("혼합 검색", test_mixed_search_without_data_type()))
    
    # 결과 요약
    print("\n" + "=" * 80)
    print("테스트 결과 요약")
    print("=" * 80)
    
    passed = 0
    failed = 0
    
    for test_name, result in results:
        status = "✅ 통과" if result else "❌ 실패"
        print(f"{status}: {test_name}")
        if result:
            passed += 1
        else:
            failed += 1
    
    print(f"\n총 {len(results)}개 테스트: {passed}개 통과, {failed}개 실패")
    print("=" * 80)
    
    return failed == 0


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)

