# -*- coding: utf-8 -*-
"""
검색 점수 정규화 통합 테스트 스크립트

실제 검색 쿼리를 실행하여 모든 검색 결과의 점수가 0.0~1.0 범위인지 확인합니다.
"""

import sys
import os
from pathlib import Path

# 프로젝트 루트 경로 추가
project_root = Path(__file__).parent.parent.parent.parent.parent
sys.path.insert(0, str(project_root))
sys.path.insert(0, str(project_root / "lawfirm_langgraph"))

try:
    from lawfirm_langgraph.core.search.handlers.search_handler import SearchHandler
    from lawfirm_langgraph.core.search.utils.score_utils import normalize_score
    from lawfirm_langgraph.core.utils.logger import get_logger
except ImportError as e:
    print(f"❌ Import 오류: {e}")
    sys.exit(1)

logger = get_logger(__name__)


def check_score_range(score: float, field_name: str, doc_id: str = "unknown") -> bool:
    """점수가 0.0~1.0 범위인지 확인"""
    if score is None:
        return True
    
    score_float = float(score)
    if score_float < 0.0 or score_float > 1.0:
        logger.error(
            f"❌ [SCORE RANGE VIOLATION] {field_name} out of range: "
            f"score={score_float:.3f}, doc_id={doc_id}"
        )
        return False
    return True


def test_search_result_scores():
    """검색 결과의 모든 점수가 0.0~1.0 범위인지 테스트"""
    print("\n" + "="*80)
    print("검색 점수 정규화 통합 테스트")
    print("="*80)
    
    # 테스트 쿼리 목록
    test_queries = [
        "계약 해지 사유에 대해 알려주세요",
        "민법 제1조의 내용은 무엇인가요?",
        "대법원 판례를 검색해주세요",
        "임대차 계약서 작성 시 주의사항",
        "형법 제250조 살인죄",
    ]
    
    try:
        # SearchHandler 초기화
        search_handler = SearchHandler()
        print("✅ SearchHandler 초기화 완료")
        
        total_results = 0
        violations = 0
        score_fields_checked = {
            "relevance_score": 0,
            "similarity": 0,
            "score": 0,
            "final_weighted_score": 0,
            "combined_score": 0,
        }
        
        for query_idx, query in enumerate(test_queries, 1):
            print(f"\n📋 테스트 쿼리 {query_idx}/{len(test_queries)}: {query}")
            
            try:
                # 검색 실행
                semantic_results, semantic_count = search_handler.semantic_search(
                    query=query,
                    limit=10,
                    query_type_str="general_question"
                )
                
                keyword_results, keyword_count = search_handler.keyword_search(
                    query=query,
                    query_type_str="general_question",
                    limit=10
                )
                
                # 결과 병합
                merged_results = search_handler.merge_and_rerank_search_results(
                    semantic_results=semantic_results,
                    keyword_results=keyword_results,
                    query=query,
                    optimized_queries={"query_type": "general_question"},
                    rerank_params={"top_k": 10}
                )
                
                print(f"   검색 결과: {len(merged_results)}개")
                
                # 각 결과의 점수 확인
                for result_idx, result in enumerate(merged_results):
                    total_results += 1
                    doc_id = result.get("id", f"result_{result_idx}")
                    
                    # 모든 점수 필드 확인
                    for field_name in score_fields_checked.keys():
                        if field_name in result:
                            score = result[field_name]
                            score_fields_checked[field_name] += 1
                            
                            if not check_score_range(score, field_name, doc_id):
                                violations += 1
                                print(f"      ⚠️ {field_name}: {score:.3f} (범위 초과)")
                            elif score > 1.0:
                                print(f"      ⚠️ {field_name}: {score:.3f} (정규화 필요)")
                
            except Exception as e:
                logger.error(f"❌ 쿼리 '{query}' 실행 중 오류: {e}")
                continue
        
        # 결과 요약
        print("\n" + "="*80)
        print("테스트 결과 요약")
        print("="*80)
        print(f"총 검색 결과 수: {total_results}")
        print(f"점수 범위 위반: {violations}")
        print(f"\n점수 필드별 확인 횟수:")
        for field_name, count in score_fields_checked.items():
            print(f"  - {field_name}: {count}회")
        
        if violations == 0:
            print("\n✅ 모든 검색 결과의 점수가 0.0~1.0 범위 내에 있습니다!")
            return True
        else:
            print(f"\n❌ {violations}개의 점수 범위 위반이 발견되었습니다.")
            return False
            
    except Exception as e:
        logger.error(f"❌ 테스트 실행 중 오류: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_score_normalization_utility():
    """점수 정규화 유틸리티 함수 테스트"""
    print("\n" + "="*80)
    print("점수 정규화 유틸리티 테스트")
    print("="*80)
    
    test_cases = [
        (0.0, 0.0),
        (0.5, 0.5),
        (1.0, 1.0),
        (1.1, None),  # 1.0 이하로 정규화되어야 함
        (1.2, None),  # 1.0 이하로 정규화되어야 함
        (2.0, None),  # 1.0 이하로 정규화되어야 함
        (-0.1, 0.0),  # 0.0 이상으로 정규화되어야 함
    ]
    
    passed = 0
    failed = 0
    
    for input_score, expected in test_cases:
        normalized = normalize_score(input_score)
        
        # 범위 확인
        if 0.0 <= normalized <= 1.0:
            if expected is None:
                # 정규화만 확인
                print(f"✅ {input_score:.1f} -> {normalized:.3f} (정규화됨)")
                passed += 1
            elif abs(normalized - expected) < 0.001:
                print(f"✅ {input_score:.1f} -> {normalized:.3f} (예상값: {expected:.1f})")
                passed += 1
            else:
                print(f"❌ {input_score:.1f} -> {normalized:.3f} (예상값: {expected:.1f})")
                failed += 1
        else:
            print(f"❌ {input_score:.1f} -> {normalized:.3f} (범위 초과!)")
            failed += 1
    
    print(f"\n결과: {passed}개 통과, {failed}개 실패")
    return failed == 0


if __name__ == "__main__":
    print("\n" + "="*80)
    print("검색 점수 정규화 통합 테스트 시작")
    print("="*80)
    
    # 유틸리티 테스트
    utility_test_passed = test_score_normalization_utility()
    
    # 통합 테스트
    integration_test_passed = test_search_result_scores()
    
    # 최종 결과
    print("\n" + "="*80)
    print("최종 결과")
    print("="*80)
    
    if utility_test_passed and integration_test_passed:
        print("✅ 모든 테스트 통과!")
        sys.exit(0)
    else:
        print("❌ 일부 테스트 실패")
        if not utility_test_passed:
            print("  - 점수 정규화 유틸리티 테스트 실패")
        if not integration_test_passed:
            print("  - 검색 결과 점수 범위 테스트 실패")
        sys.exit(1)

