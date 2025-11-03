#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
데이터베이스 기반 키워드 관리 시스템 테스트
"""

import os
import sys
from typing import Any, Dict, List

# 프로젝트 루트 디렉토리를 Python 경로에 추가
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))

from lawfirm_langgraph.langgraph_core.services.answer_structure_enhancer import AnswerStructureEnhancer
from core.services.search.database_keyword_manager import DatabaseKeywordManager


def test_database_keyword_manager():
    """데이터베이스 키워드 관리자 테스트"""
    print("=" * 60)
    print("데이터베이스 키워드 관리자 테스트")
    print("=" * 60)

    db_manager = DatabaseKeywordManager()

    # 1. 질문 유형 조회 테스트
    print("\n1. 질문 유형 조회 테스트")
    question_types = db_manager.get_all_question_types()
    print(f"   등록된 질문 유형 수: {len(question_types)}")
    for qt in question_types[:5]:  # 상위 5개만 표시
        print(f"   - {qt['type_name']}: {qt['display_name']}")

    # 2. 키워드 조회 테스트
    print("\n2. 키워드 조회 테스트")
    test_types = ["precedent_search", "contract_review", "divorce_procedure"]

    for q_type in test_types:
        keywords = db_manager.get_keywords_for_type(q_type, limit=5)
        print(f"   {q_type}: {len(keywords)}개 키워드")
        for kw in keywords[:3]:  # 상위 3개만 표시
            print(f"     - {kw['keyword']} ({kw['weight_level']}, {kw['weight_value']})")

    # 3. 키워드 검색 테스트
    print("\n3. 키워드 검색 테스트")
    search_terms = ["판례", "계약서", "이혼"]

    for term in search_terms:
        results = db_manager.search_keywords(term, limit=5)
        print(f"   '{term}' 검색 결과: {len(results)}개")
        for result in results[:2]:  # 상위 2개만 표시
            print(f"     - {result['question_type']}: {result['keyword']} ({result['weight_level']})")

    # 4. 패턴 조회 테스트
    print("\n4. 패턴 조회 테스트")
    for q_type in test_types:
        patterns = db_manager.get_patterns_for_type(q_type)
        print(f"   {q_type}: {len(patterns)}개 패턴")
        for pattern in patterns[:2]:  # 상위 2개만 표시
            print(f"     - {pattern['pattern'][:50]}...")

    # 5. 통계 조회 테스트
    print("\n5. 통계 조회 테스트")
    stats = db_manager.get_keyword_statistics()
    print(f"   전체 키워드 수: {stats.get('total_keywords', 0)}")
    print(f"   고가중치 키워드: {stats.get('high_weight_count', 0)}")
    print(f"   중가중치 키워드: {stats.get('medium_weight_count', 0)}")
    print(f"   저가중치 키워드: {stats.get('low_weight_count', 0)}")

    print("\n✅ 데이터베이스 키워드 관리자 테스트 완료!")


def test_answer_structure_enhancer():
    """답변 구조화 향상 시스템 테스트"""
    print("\n" + "=" * 60)
    print("답변 구조화 향상 시스템 테스트 (DB 기반)")
    print("=" * 60)

    enhancer = AnswerStructureEnhancer()

    # 테스트 케이스들
    test_cases = [
        {
            "question": "부동산 매매 계약서에서 위약금 조항을 검토해주세요",
            "question_type": "contract_review",
            "domain": "민사법"
        },
        {
            "question": "이혼 절차는 어떻게 진행되나요?",
            "question_type": "divorce_procedure",
            "domain": "가족법"
        },
        {
            "question": "대법원 판례를 찾아주세요",
            "question_type": "precedent_search",
            "domain": "일반"
        },
        {
            "question": "상속 절차에 대해 알고 싶습니다",
            "question_type": "inheritance_procedure",
            "domain": "가족법"
        },
        {
            "question": "법률 용어의 의미를 설명해주세요",
            "question_type": "term_explanation",
            "domain": "일반"
        },
        {
            "question": "노동 분쟁 해결 방법을 알려주세요",
            "question_type": "labor_dispute",
            "domain": "노동법"
        },
        {
            "question": "형사 사건 처리 절차는 어떻게 되나요?",
            "question_type": "criminal_case",
            "domain": "형사법"
        },
        {
            "question": "법률 상담을 받고 싶습니다",
            "question_type": "legal_advice",
            "domain": "일반"
        },
        {
            "question": "법령에 대해 문의드립니다",
            "question_type": "law_inquiry",
            "domain": "일반"
        },
        {
            "question": "일반적인 법률 질문입니다",
            "question_type": "general",
            "domain": "일반"
        }
    ]

    success_count = 0
    total_count = len(test_cases)

    for i, case in enumerate(test_cases, 1):
        print(f"\n테스트 케이스 {i}: {case['question']}")

        try:
            # 구조화 향상 실행
            result = enhancer.enhance_answer_structure(
                answer="테스트 답변입니다.",
                question_type=case['question_type'],
                question=case['question'],
                domain=case['domain']
            )

            if 'error' in result:
                print(f"   ❌ 오류: {result['error']}")
            else:
                print(f"   ✅ 성공!")
                print(f"     질문 유형: {result.get('question_type', 'Unknown')}")
                print(f"     템플릿: {result.get('template_used', 'Unknown')}")

                quality_metrics = result.get('quality_metrics', {})
                print(f"     구조화 점수: {quality_metrics.get('structure_score', 0.0):.2f}")
                print(f"     전체 점수: {quality_metrics.get('overall_score', 0.0):.2f}")

                success_count += 1

        except Exception as e:
            print(f"   ❌ 예외 발생: {e}")

    print(f"\n" + "=" * 60)
    print(f"답변 구조화 향상 테스트 결과: {success_count}/{total_count} 성공")
    print(f"성공률: {(success_count/total_count)*100:.1f}%")
    print("=" * 60)

    return success_count == total_count


def test_question_type_mapping():
    """질문 유형 매핑 테스트"""
    print("\n" + "=" * 60)
    print("질문 유형 매핑 테스트 (DB 기반)")
    print("=" * 60)

    enhancer = AnswerStructureEnhancer()

    # 매핑 테스트 케이스들
    mapping_tests = [
        ("판례를 찾아주세요", "precedent_search"),
        ("계약서를 검토해주세요", "contract_review"),
        ("이혼 절차를 알려주세요", "divorce_procedure"),
        ("상속 절차는 어떻게 되나요?", "inheritance_procedure"),
        ("범죄 처벌에 대해 알고 싶습니다", "criminal_case"),
        ("노동 분쟁 해결 방법", "labor_dispute"),
        ("절차 안내를 요청합니다", "procedure_guide"),
        ("용어의 의미를 설명해주세요", "term_explanation"),
        ("법률 조언을 받고 싶습니다", "legal_advice"),
        ("법령에 대해 문의드립니다", "law_inquiry"),
        ("일반적인 질문입니다", "general_question")
    ]

    correct_mappings = 0
    total_mappings = len(mapping_tests)

    for question, expected_type in mapping_tests:
        try:
            mapped_type = enhancer._map_question_type("", question)
            mapped_type_name = mapped_type.value if hasattr(mapped_type, 'value') else str(mapped_type)

            is_correct = mapped_type_name == expected_type
            status = "✅" if is_correct else "❌"

            print(f"{status} '{question}' -> {mapped_type_name} (예상: {expected_type})")

            if is_correct:
                correct_mappings += 1

        except Exception as e:
            print(f"❌ '{question}' -> 오류: {e}")

    print(f"\n" + "=" * 60)
    print(f"질문 유형 매핑 테스트 결과: {correct_mappings}/{total_mappings} 정확")
    print(f"정확도: {(correct_mappings/total_mappings)*100:.1f}%")
    print("=" * 60)

    return correct_mappings >= total_mappings * 0.8  # 80% 이상 정확도


def test_performance():
    """성능 테스트"""
    print("\n" + "=" * 60)
    print("성능 테스트")
    print("=" * 60)

    import time

    db_manager = DatabaseKeywordManager()
    enhancer = AnswerStructureEnhancer()

    # 데이터베이스 조회 성능 테스트
    print("\n1. 데이터베이스 조회 성능 테스트")

    start_time = time.time()
    for _ in range(100):
        keywords = db_manager.get_keywords_for_type("precedent_search", limit=10)
    db_time = time.time() - start_time
    print(f"   100회 키워드 조회: {db_time:.3f}초 (평균: {db_time/100*1000:.1f}ms)")

    # 질문 유형 매핑 성능 테스트
    print("\n2. 질문 유형 매핑 성능 테스트")

    test_questions = [
        "판례를 찾아주세요",
        "계약서를 검토해주세요",
        "이혼 절차를 알려주세요",
        "상속 절차는 어떻게 되나요?",
        "범죄 처벌에 대해 알고 싶습니다"
    ]

    start_time = time.time()
    for _ in range(50):
        for question in test_questions:
            mapped_type = enhancer._map_question_type("", question)
    mapping_time = time.time() - start_time
    print(f"   250회 질문 유형 매핑: {mapping_time:.3f}초 (평균: {mapping_time/250*1000:.1f}ms)")

    print("\n✅ 성능 테스트 완료!")


def main():
    """메인 테스트 함수"""
    print("데이터베이스 기반 키워드 관리 시스템 종합 테스트")

    try:
        # 1. 데이터베이스 키워드 관리자 테스트
        test_database_keyword_manager()

        # 2. 답변 구조화 향상 시스템 테스트
        structure_test_passed = test_answer_structure_enhancer()

        # 3. 질문 유형 매핑 테스트
        mapping_test_passed = test_question_type_mapping()

        # 4. 성능 테스트
        test_performance()

        # 전체 결과
        print(f"\n" + "=" * 60)
        print("전체 테스트 결과")
        print("=" * 60)
        print(f"답변 구조화 향상: {'✅ 통과' if structure_test_passed else '❌ 실패'}")
        print(f"질문 유형 매핑: {'✅ 통과' if mapping_test_passed else '❌ 실패'}")

        if structure_test_passed and mapping_test_passed:
            print("🎉 모든 테스트가 통과했습니다!")
            print("데이터베이스 기반 키워드 관리 시스템이 정상적으로 작동합니다.")
        else:
            print("⚠️ 일부 테스트가 실패했습니다.")
            print("추가 수정이 필요할 수 있습니다.")

        return structure_test_passed and mapping_test_passed

    except Exception as e:
        print(f"❌ 테스트 실행 중 오류 발생: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    main()
