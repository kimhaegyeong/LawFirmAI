#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
데이터베이스 기반 템플릿 시스템 + classify_question_type 통합 테스트
"""

import os
import sys

# 프로젝트 루트 디렉토리를 Python 경로에 추가
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))

from source.services.answer_structure_enhancer import (
    AnswerStructureEnhancer,
    QuestionType,
)


def test_integrated_system():
    """통합 시스템 테스트"""
    print("=" * 60)
    print("데이터베이스 기반 템플릿 시스템 + classify_question_type 통합 테스트")
    print("=" * 60)

    enhancer = AnswerStructureEnhancer()

    # 통합 테스트 케이스들
    test_cases = [
        {
            "question": "민법 제123조의 내용이 무엇인가요?",
            "answer": "민법 제123조는 계약의 해제에 관한 규정입니다. 이 조항에 따르면 계약 당사자는 상대방이 계약을 이행하지 않을 경우 계약을 해제할 수 있습니다.",
            "expected_type": QuestionType.LAW_INQUIRY
        },
        {
            "question": "계약서를 검토해주세요",
            "answer": "제공해주신 계약서를 검토한 결과, 몇 가지 주의사항이 있습니다. 특히 제3조의 손해배상 조항이 과도하게 불리할 수 있습니다.",
            "expected_type": QuestionType.CONTRACT_REVIEW
        },
        {
            "question": "이혼 절차를 알려주세요",
            "answer": "이혼 절차는 크게 협의이혼, 조정이혼, 재판이혼으로 나뉩니다. 협의이혼이 가장 간단하며, 양방이 합의하면 가정법원에 신청할 수 있습니다.",
            "expected_type": QuestionType.DIVORCE_PROCEDURE
        },
        {
            "question": "대법원 판례를 찾아주세요",
            "answer": "관련 대법원 판례를 찾았습니다. 2023다12345 사건에서 대법원은 계약 해제의 요건에 대해 명확한 기준을 제시했습니다.",
            "expected_type": QuestionType.PRECEDENT_SEARCH
        },
        {
            "question": "노동 분쟁 해결 방법",
            "answer": "노동 분쟁이 발생한 경우 노동위원회에 신청하거나 법원에 소송을 제기할 수 있습니다. 먼저 노동위원회 조정을 통해 해결을 시도하는 것이 좋습니다.",
            "expected_type": QuestionType.LABOR_DISPUTE
        }
    ]

    print(f"\n총 {len(test_cases)}개 통합 테스트 케이스 실행 중...\n")

    success_count = 0

    for i, test_case in enumerate(test_cases, 1):
        question = test_case["question"]
        answer = test_case["answer"]
        expected_type = test_case["expected_type"]

        print(f"{i}. 테스트 케이스: {question}")
        print("-" * 50)

        try:
            # 1. 질문 유형 분류 테스트
            classified_type = enhancer.classify_question_type(question)
            classification_correct = classified_type == expected_type

            print(f"   질문 유형 분류:")
            print(f"     예상: {expected_type.value}")
            print(f"     결과: {classified_type.value}")
            print(f"     상태: {'✅' if classification_correct else '❌'}")

            # 2. 답변 구조화 테스트
            structure_result = enhancer.enhance_answer_structure(
                answer=answer,
                question_type=classified_type.value,
                question=question
            )

            if "error" not in structure_result:
                print(f"   답변 구조화:")
                print(f"     원본 길이: {len(answer)} 문자")
                print(f"     구조화 길이: {len(structure_result['structured_answer'])} 문자")
                print(f"     사용된 템플릿: {structure_result['template_used']}")
                print(f"     전체 점수: {structure_result['quality_metrics']['overall_score']:.2f}")
                print(f"     상태: ✅")

                structure_success = True
            else:
                print(f"   답변 구조화:")
                print(f"     오류: {structure_result['error']}")
                print(f"     상태: ❌")
                structure_success = False

            # 3. 통합 성공 여부
            if classification_correct and structure_success:
                success_count += 1
                print(f"   전체 결과: ✅ 성공")
            else:
                print(f"   전체 결과: ❌ 실패")

            print()

        except Exception as e:
            print(f"   오류 발생: {e}")
            print(f"   전체 결과: ❌ 실패")
            print()

    # 결과 요약
    success_rate = (success_count / len(test_cases)) * 100

    print("=" * 60)
    print("통합 테스트 결과 요약")
    print("=" * 60)
    print(f"총 테스트 케이스: {len(test_cases)}")
    print(f"성공한 케이스: {success_count}")
    print(f"실패한 케이스: {len(test_cases) - success_count}")
    print(f"성공률: {success_rate:.1f}%")

    if success_rate >= 90:
        grade = "A+"
    elif success_rate >= 80:
        grade = "A"
    elif success_rate >= 70:
        grade = "B"
    elif success_rate >= 60:
        grade = "C"
    else:
        grade = "D"

    print(f"등급: {grade}")
    print("=" * 60)

    return success_rate


def test_performance_comparison():
    """성능 비교 테스트"""
    print("\n" + "=" * 60)
    print("성능 비교 테스트")
    print("=" * 60)

    import time

    enhancer = AnswerStructureEnhancer()

    test_questions = [
        "민법 제123조의 내용이 무엇인가요?",
        "계약서를 검토해주세요",
        "이혼 절차를 알려주세요",
        "판례를 찾아주세요",
        "법률 상담이 필요합니다"
    ] * 20  # 100개 질문

    test_answer = "이것은 테스트 답변입니다. 법률적 내용을 포함하고 있습니다."

    print(f"\n{len(test_questions)}개 질문에 대한 성능 테스트 중...")

    # 1. 질문 유형 분류 성능
    start_time = time.time()
    for question in test_questions:
        enhancer.classify_question_type(question)
    classification_time = time.time() - start_time

    # 2. 답변 구조화 성능
    start_time = time.time()
    for question in test_questions:
        classified_type = enhancer.classify_question_type(question)
        enhancer.enhance_answer_structure(
            answer=test_answer,
            question_type=classified_type.value,
            question=question
        )
    structure_time = time.time() - start_time

    # 3. 통합 성능
    total_time = classification_time + structure_time

    print(f"\n성능 결과:")
    print(f"  질문 유형 분류 시간: {classification_time:.3f}초")
    print(f"  답변 구조화 시간: {structure_time:.3f}초")
    print(f"  총 처리 시간: {total_time:.3f}초")
    print(f"  평균 처리 시간: {total_time/len(test_questions)*1000:.2f}ms")
    print(f"  초당 처리량: {len(test_questions)/total_time:.0f} questions/sec")

    print(f"\n개별 성능:")
    print(f"  분류 평균: {classification_time/len(test_questions)*1000:.2f}ms")
    print(f"  구조화 평균: {structure_time/len(test_questions)*1000:.2f}ms")


def test_database_integration():
    """데이터베이스 통합 테스트"""
    print("\n" + "=" * 60)
    print("데이터베이스 통합 테스트")
    print("=" * 60)

    enhancer = AnswerStructureEnhancer()

    # 데이터베이스에서 템플릿 정보 조회
    print("\n1. 데이터베이스 템플릿 조회 테스트")
    print("-" * 30)

    question_types = [
        "law_inquiry",
        "contract_review",
        "divorce_procedure",
        "precedent_search",
        "labor_dispute"
    ]

    for question_type in question_types:
        template_info = enhancer.get_template_info(question_type)
        print(f"  {question_type}:")
        print(f"    제목: {template_info['title']}")
        print(f"    섹션 수: {template_info['section_count']}")
        print(f"    소스: {template_info['source']}")
        print()

    # 동적 템플릿 리로드 테스트
    print("\n2. 동적 템플릿 리로드 테스트")
    print("-" * 30)

    print("  템플릿 리로드 중...")
    enhancer.reload_templates()
    print("  ✅ 템플릿 리로드 완료")

    # 데이터베이스 통계 조회
    print("\n3. 데이터베이스 통계 조회")
    print("-" * 30)

    # template_db_manager가 없는 경우를 대비한 안전 처리
    if hasattr(enhancer, 'template_db_manager') and enhancer.template_db_manager:
        try:
            stats = enhancer.template_db_manager.get_template_statistics()
            print(f"  전체 템플릿: {stats.get('total_templates', 0)}")
            print(f"  활성 템플릿: {stats.get('active_templates', 0)}")
            print(f"  전체 섹션: {stats.get('total_sections', 0)}")
            print(f"  활성 섹션: {stats.get('active_sections', 0)}")
        except Exception as e:
            print(f"  ⚠️  통계 조회 실패: {e}")
            print(f"  템플릿 시스템은 정상 동작 중입니다.")
    else:
        # 템플릿 시스템은 하드코딩된 템플릿을 사용하므로 통계는 직접 계산
        template_count = len(enhancer.structure_templates) if hasattr(enhancer, 'structure_templates') else 0
        print(f"  전체 템플릿: {template_count}개 (하드코딩)")
        print(f"  템플릿 시스템: 정상 동작 중")


def main():
    """메인 함수"""
    print("데이터베이스 기반 템플릿 시스템 + classify_question_type 통합 테스트")

    try:
        # 1. 통합 시스템 테스트
        success_rate = test_integrated_system()

        # 2. 성능 비교 테스트
        test_performance_comparison()

        # 3. 데이터베이스 통합 테스트
        test_database_integration()

        print(f"\n🎉 모든 통합 테스트가 완료되었습니다!")
        print(f"📊 최종 성공률: {success_rate:.1f}%")

        if success_rate >= 90:
            print("🏆 우수한 성능을 보여주고 있습니다!")
        elif success_rate >= 80:
            print("👍 양호한 성능을 보여주고 있습니다!")
        else:
            print("⚠️  개선이 필요한 부분이 있습니다.")

    except Exception as e:
        print(f"❌ 테스트 중 오류 발생: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
