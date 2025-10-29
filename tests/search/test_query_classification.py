#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
질의 분류 테스트 스크립트
실제 법률 질의를 대상으로 데이터베이스 기반 질문 유형 매핑 시스템 테스트
"""

import os
import sys
from typing import Any, Dict, List, Tuple

# 프로젝트 루트 디렉토리를 Python 경로에 추가
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))

from source.services.answer_structure_enhancer import AnswerStructureEnhancer
from source.services.database_keyword_manager import DatabaseKeywordManager


def get_real_world_queries() -> List[Dict[str, Any]]:
    """실제 법률 질의 데이터"""
    return [
        # 판례 검색 관련
        {
            "query": "부동산 매매 계약 해지 시 위약금 관련 대법원 판례를 찾아주세요",
            "expected_type": "precedent_search",
            "category": "판례 검색",
            "difficulty": "high"
        },
        {
            "query": "이혼 시 재산분할 관련 유사 판례가 있나요?",
            "expected_type": "precedent_search",
            "category": "판례 검색",
            "difficulty": "medium"
        },
        {
            "query": "근로자 해고 관련 최근 판례를 알려주세요",
            "expected_type": "precedent_search",
            "category": "판례 검색",
            "difficulty": "medium"
        },

        # 계약서 검토 관련
        {
            "query": "부동산 매매 계약서의 위약금 조항을 검토해주세요",
            "expected_type": "contract_review",
            "category": "계약서 검토",
            "difficulty": "high"
        },
        {
            "query": "임대차 계약서에서 수정이 필요한 부분이 있나요?",
            "expected_type": "contract_review",
            "category": "계약서 검토",
            "difficulty": "medium"
        },
        {
            "query": "근로계약서의 불리한 조항을 확인해주세요",
            "expected_type": "contract_review",
            "category": "계약서 검토",
            "difficulty": "medium"
        },

        # 이혼 절차 관련
        {
            "query": "협의이혼 절차는 어떻게 진행되나요?",
            "expected_type": "divorce_procedure",
            "category": "이혼 절차",
            "difficulty": "low"
        },
        {
            "query": "재판이혼 신청 시 필요한 서류는 무엇인가요?",
            "expected_type": "divorce_procedure",
            "category": "이혼 절차",
            "difficulty": "medium"
        },
        {
            "query": "이혼 시 양육권과 면접교섭권은 어떻게 결정되나요?",
            "expected_type": "divorce_procedure",
            "category": "이혼 절차",
            "difficulty": "high"
        },

        # 상속 절차 관련
        {
            "query": "상속 절차는 어떻게 진행되나요?",
            "expected_type": "inheritance_procedure",
            "category": "상속 절차",
            "difficulty": "low"
        },
        {
            "query": "유언이 있는 경우 상속 절차는 어떻게 되나요?",
            "expected_type": "inheritance_procedure",
            "category": "상속 절차",
            "difficulty": "medium"
        },
        {
            "query": "상속세 신고는 언제까지 해야 하나요?",
            "expected_type": "inheritance_procedure",
            "category": "상속 절차",
            "difficulty": "medium"
        },

        # 형사 사건 관련
        {
            "query": "사기죄로 고소당했는데 어떻게 해야 하나요?",
            "expected_type": "criminal_case",
            "category": "형사 사건",
            "difficulty": "high"
        },
        {
            "query": "교통사고로 과실치상상죄가 적용되나요?",
            "expected_type": "criminal_case",
            "category": "형사 사건",
            "difficulty": "medium"
        },
        {
            "query": "형사사건에서 변호인 선임은 필수인가요?",
            "expected_type": "criminal_case",
            "category": "형사 사건",
            "difficulty": "low"
        },

        # 노동 분쟁 관련
        {
            "query": "부당해고로 노동위원회에 신청하려고 합니다",
            "expected_type": "labor_dispute",
            "category": "노동 분쟁",
            "difficulty": "medium"
        },
        {
            "query": "임금체불로 인한 구제 절차를 알려주세요",
            "expected_type": "labor_dispute",
            "category": "노동 분쟁",
            "difficulty": "medium"
        },
        {
            "query": "근로시간 위반으로 인한 분쟁 해결 방법은?",
            "expected_type": "labor_dispute",
            "category": "노동 분쟁",
            "difficulty": "medium"
        },

        # 절차 안내 관련
        {
            "query": "소액사건심판절차는 어떻게 신청하나요?",
            "expected_type": "procedure_guide",
            "category": "절차 안내",
            "difficulty": "medium"
        },
        {
            "query": "민사조정 신청 방법과 절차를 알려주세요",
            "expected_type": "procedure_guide",
            "category": "절차 안내",
            "difficulty": "medium"
        },
        {
            "query": "가정법원 이혼조정 신청은 어떻게 하나요?",
            "expected_type": "procedure_guide",
            "category": "절차 안내",
            "difficulty": "medium"
        },

        # 용어 해설 관련
        {
            "query": "불법행위의 의미와 구성요건을 설명해주세요",
            "expected_type": "term_explanation",
            "category": "용어 해설",
            "difficulty": "high"
        },
        {
            "query": "손해배상과 위자료의 차이점은 무엇인가요?",
            "expected_type": "term_explanation",
            "category": "용어 해설",
            "difficulty": "medium"
        },
        {
            "query": "채권과 채무의 개념을 쉽게 설명해주세요",
            "expected_type": "term_explanation",
            "category": "용어 해설",
            "difficulty": "low"
        },

        # 법률 조언 관련
        {
            "query": "계약 위반으로 손해를 입었는데 어떻게 해야 하나요?",
            "expected_type": "legal_advice",
            "category": "법률 조언",
            "difficulty": "high"
        },
        {
            "query": "이웃과의 소음 분쟁 해결 방법을 조언해주세요",
            "expected_type": "legal_advice",
            "category": "법률 조언",
            "difficulty": "medium"
        },
        {
            "query": "직장에서 성희롱을 당했는데 어떻게 대처해야 하나요?",
            "expected_type": "legal_advice",
            "category": "법률 조언",
            "difficulty": "high"
        },

        # 법률 문의 관련
        {
            "query": "민법 제750조의 내용을 알려주세요",
            "expected_type": "law_inquiry",
            "category": "법률 문의",
            "difficulty": "medium"
        },
        {
            "query": "근로기준법에서 정한 최저임금은 얼마인가요?",
            "expected_type": "law_inquiry",
            "category": "법률 문의",
            "difficulty": "low"
        },
        {
            "query": "형법 제257조의 처벌 기준은 어떻게 되나요?",
            "expected_type": "law_inquiry",
            "category": "법률 문의",
            "difficulty": "medium"
        },

        # 일반 질문 관련
        {
            "query": "법률 상담은 어디서 받을 수 있나요?",
            "expected_type": "general_question",
            "category": "일반 질문",
            "difficulty": "low"
        },
        {
            "query": "변호사 선임 비용은 얼마나 드나요?",
            "expected_type": "general_question",
            "category": "일반 질문",
            "difficulty": "low"
        },
        {
            "query": "법원에서 소송을 제기하려면 어떻게 해야 하나요?",
            "expected_type": "general_question",
            "category": "일반 질문",
            "difficulty": "medium"
        }
    ]


def test_query_classification():
    """질의 분류 테스트"""
    print("=" * 80)
    print("실제 법률 질의 분류 테스트")
    print("=" * 80)

    enhancer = AnswerStructureEnhancer()
    queries = get_real_world_queries()

    # 결과 저장
    results = {
        "total": len(queries),
        "correct": 0,
        "incorrect": 0,
        "by_category": {},
        "by_difficulty": {},
        "detailed_results": []
    }

    print(f"\n총 {len(queries)}개의 질의를 테스트합니다...\n")

    for i, query_data in enumerate(queries, 1):
        query = query_data["query"]
        expected_type = query_data["expected_type"]
        category = query_data["category"]
        difficulty = query_data["difficulty"]

        print(f"테스트 {i:2d}: {query}")

        try:
            # 질문 유형 매핑
            mapped_type = enhancer._map_question_type("", query)
            mapped_type_name = mapped_type.value if hasattr(mapped_type, 'value') else str(mapped_type)

            # 결과 판정
            is_correct = mapped_type_name == expected_type
            status = "✅" if is_correct else "❌"

            print(f"         {status} 예상: {expected_type} | 실제: {mapped_type_name}")

            # 통계 업데이트
            if is_correct:
                results["correct"] += 1
            else:
                results["incorrect"] += 1

            # 카테고리별 통계
            if category not in results["by_category"]:
                results["by_category"][category] = {"total": 0, "correct": 0}
            results["by_category"][category]["total"] += 1
            if is_correct:
                results["by_category"][category]["correct"] += 1

            # 난이도별 통계
            if difficulty not in results["by_difficulty"]:
                results["by_difficulty"][difficulty] = {"total": 0, "correct": 0}
            results["by_difficulty"][difficulty]["total"] += 1
            if is_correct:
                results["by_difficulty"][difficulty]["correct"] += 1

            # 상세 결과 저장
            results["detailed_results"].append({
                "query": query,
                "expected": expected_type,
                "actual": mapped_type_name,
                "correct": is_correct,
                "category": category,
                "difficulty": difficulty
            })

        except Exception as e:
            print(f"         ❌ 오류 발생: {e}")
            results["incorrect"] += 1

    return results


def analyze_results(results: Dict[str, Any]):
    """결과 분석 및 리포트 생성"""
    print("\n" + "=" * 80)
    print("테스트 결과 분석")
    print("=" * 80)

    # 전체 정확도
    accuracy = (results["correct"] / results["total"]) * 100
    print(f"\n📊 전체 정확도: {accuracy:.1f}% ({results['correct']}/{results['total']})")

    # 카테고리별 정확도
    print(f"\n📋 카테고리별 정확도:")
    for category, stats in results["by_category"].items():
        cat_accuracy = (stats["correct"] / stats["total"]) * 100
        print(f"   {category:12s}: {cat_accuracy:5.1f}% ({stats['correct']:2d}/{stats['total']:2d})")

    # 난이도별 정확도
    print(f"\n🎯 난이도별 정확도:")
    for difficulty, stats in results["by_difficulty"].items():
        diff_accuracy = (stats["correct"] / stats["total"]) * 100
        print(f"   {difficulty:8s}: {diff_accuracy:5.1f}% ({stats['correct']:2d}/{stats['total']:2d})")

    # 오분류 분석
    print(f"\n❌ 오분류 사례:")
    incorrect_cases = [r for r in results["detailed_results"] if not r["correct"]]

    for case in incorrect_cases:
        print(f"   질의: {case['query'][:50]}...")
        print(f"   예상: {case['expected']} | 실제: {case['actual']} | 카테고리: {case['category']}")
        print()

    return accuracy


def test_edge_cases():
    """엣지 케이스 테스트"""
    print("\n" + "=" * 80)
    print("엣지 케이스 테스트")
    print("=" * 80)

    enhancer = AnswerStructureEnhancer()

    edge_cases = [
        {
            "query": "판례와 계약서 모두 관련된 질문입니다",
            "description": "복합 키워드"
        },
        {
            "query": "법률",
            "description": "매우 짧은 질의"
        },
        {
            "query": "이혼하면서 상속도 같이 처리하고 싶은데 계약서도 검토받고 판례도 찾아주세요",
            "description": "매우 긴 복합 질의"
        },
        {
            "query": "123456789",
            "description": "숫자만 포함"
        },
        {
            "query": "!@#$%^&*()",
            "description": "특수문자만 포함"
        },
        {
            "query": "",
            "description": "빈 질의"
        },
        {
            "query": "법률상담변호사계약서이혼상속형사노동절차용어조언문의",
            "description": "모든 키워드 포함"
        }
    ]

    print(f"\n엣지 케이스 {len(edge_cases)}개를 테스트합니다...\n")

    for i, case in enumerate(edge_cases, 1):
        query = case["query"]
        description = case["description"]

        print(f"엣지 케이스 {i}: {description}")
        print(f"   질의: '{query}'")

        try:
            mapped_type = enhancer._map_question_type("", query)
            mapped_type_name = mapped_type.value if hasattr(mapped_type, 'value') else str(mapped_type)

            print(f"   결과: {mapped_type_name}")
            print(f"   상태: {'✅ 정상 처리' if mapped_type_name else '❌ 오류'}")

        except Exception as e:
            print(f"   상태: ❌ 오류 발생 - {e}")

        print()


def test_performance_with_real_queries():
    """실제 질의로 성능 테스트"""
    print("\n" + "=" * 80)
    print("실제 질의 성능 테스트")
    print("=" * 80)

    import time

    enhancer = AnswerStructureEnhancer()
    queries = get_real_world_queries()

    # 성능 테스트
    print(f"\n{len(queries)}개의 실제 질의로 성능을 측정합니다...")

    start_time = time.time()
    for query_data in queries:
        mapped_type = enhancer._map_question_type("", query_data["query"])
    end_time = time.time()

    total_time = end_time - start_time

    if len(queries) == 0:
        print(f"\n⚠️  질의 데이터가 없어 성능 테스트를 건너뜁니다.")
        return

    if total_time == 0:
        total_time = 0.001  # 최소값 설정하여 division by zero 방지

    avg_time = total_time / len(queries) * 1000  # ms로 변환

    print(f"\n📈 성능 결과:")
    print(f"   전체 처리 시간: {total_time:.3f}초")
    print(f"   평균 처리 시간: {avg_time:.1f}ms/질의")
    print(f"   처리량: {len(queries)/total_time:.1f}질의/초")


def main():
    """메인 테스트 함수"""
    print("법률 질의 분류 시스템 종합 테스트")

    try:
        # 1. 실제 질의 분류 테스트
        results = test_query_classification()

        # 2. 결과 분석
        accuracy = analyze_results(results)

        # 3. 엣지 케이스 테스트
        test_edge_cases()

        # 4. 성능 테스트
        test_performance_with_real_queries()

        # 최종 평가
        print("\n" + "=" * 80)
        print("최종 평가")
        print("=" * 80)

        if accuracy >= 90:
            grade = "A+ (우수)"
        elif accuracy >= 80:
            grade = "A (양호)"
        elif accuracy >= 70:
            grade = "B (보통)"
        elif accuracy >= 60:
            grade = "C (미흡)"
        else:
            grade = "D (불량)"

        print(f"🎯 전체 정확도: {accuracy:.1f}%")
        print(f"📊 등급: {grade}")

        if accuracy >= 80:
            print("✅ 시스템이 실용적으로 사용 가능한 수준입니다.")
        else:
            print("⚠️ 시스템 개선이 필요합니다.")

        print("\n🎉 질의 분류 테스트가 완료되었습니다!")

        return accuracy >= 80

    except Exception as e:
        print(f"❌ 테스트 실행 중 오류 발생: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    main()
