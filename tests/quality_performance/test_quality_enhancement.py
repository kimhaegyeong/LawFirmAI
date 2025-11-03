#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
답변 품질 향상 시스템 테스트
단기 개선 방안 적용 결과 검증
"""

import json
import os
import sys
from datetime import datetime

# 프로젝트 루트 디렉토리를 Python 경로에 추가
project_root = os.path.join(os.path.dirname(__file__), '..', '..')
sys.path.insert(0, project_root)

from source.services.answer_quality_enhancer import AnswerQualityEnhancer
from lawfirm_langgraph.langgraph_core.services.answer_structure_enhancer import AnswerStructureEnhancer
from source.services.context_quality_enhancer import ContextQualityEnhancer
from source.services.keyword_coverage_enhancer import KeywordCoverageEnhancer
from source.services.legal_term_validator import LegalTermValidator


def test_keyword_coverage_enhancement():
    """키워드 포함도 향상 테스트"""
    print("=" * 60)
    print("키워드 포함도 향상 테스트")
    print("=" * 60)

    enhancer = KeywordCoverageEnhancer()

    # 테스트 케이스
    test_cases = [
        {
            "answer": "계약서 검토는 중요합니다. 당사자와 조건을 확인해야 합니다.",
            "query_type": "contract_review",
            "question": "계약서 검토 시 주의사항은 무엇인가요?",
            "expected_improvement": True
        },
        {
            "answer": "이혼 절차는 복잡합니다. 법원에 신청해야 합니다.",
            "query_type": "divorce_procedure",
            "question": "이혼 절차는 어떻게 진행되나요?",
            "expected_improvement": True
        }
    ]

    for i, case in enumerate(test_cases, 1):
        print(f"\n테스트 케이스 {i}:")
        print(f"질문: {case['question']}")
        print(f"원본 답변: {case['answer']}")

        # 키워드 포함도 분석
        analysis = enhancer.analyze_keyword_coverage(
            case['answer'], case['query_type'], case['question']
        )

        print(f"현재 포함도: {analysis.get('current_coverage', 0.0):.2f}")
        print(f"목표 포함도: {analysis.get('target_coverage', 0.7):.2f}")
        print(f"개선 필요: {analysis.get('needs_improvement', False)}")

        # 향상 제안
        enhancement = enhancer.enhance_keyword_coverage(
            case['answer'], case['query_type'], case['question']
        )

        if enhancement.get('status') == 'needs_improvement':
            print("개선 제안:")
            for action in enhancement.get('action_plan', []):
                print(f"  - {action}")
        else:
            print("목표 포함도를 달성했습니다!")


def test_answer_structure_enhancement():
    """답변 구조화 향상 테스트"""
    print("\n" + "=" * 60)
    print("답변 구조화 향상 테스트")
    print("=" * 60)

    enhancer = AnswerStructureEnhancer()

    # 테스트 케이스
    test_cases = [
        {
            "answer": "계약서 검토는 중요합니다. 당사자와 조건을 확인해야 합니다. 법적 근거도 필요합니다.",
            "question_type": "contract_review",
            "question": "계약서 검토 시 주의사항은 무엇인가요?",
            "domain": "민사법"
        },
        {
            "answer": "이혼 절차는 협의이혼, 조정이혼, 재판이혼이 있습니다. 각각 다른 절차를 따릅니다.",
            "question_type": "divorce_procedure",
            "question": "이혼 절차는 어떻게 진행되나요?",
            "domain": "가족법"
        }
    ]

    for i, case in enumerate(test_cases, 1):
        print(f"\n테스트 케이스 {i}:")
        print(f"질문: {case['question']}")
        print(f"원본 답변: {case['answer']}")

        # 구조화 향상
        enhancement = enhancer.enhance_answer_structure(
            case['answer'], case['question_type'], case['question'], case['domain']
        )

        if 'structured_answer' in enhancement:
            print(f"구조화된 답변:\n{enhancement['structured_answer']}")

        quality_metrics = enhancement.get('quality_metrics', {})
        print(f"구조화 품질 점수: {quality_metrics.get('overall_score', 0.0):.2f}")


def test_legal_term_validation():
    """법률 용어 정확성 검증 테스트"""
    print("\n" + "=" * 60)
    print("법률 용어 정확성 검증 테스트")
    print("=" * 60)

    validator = LegalTermValidator()

    # 테스트 케이스
    test_cases = [
        {
            "answer": "계약은 당사자 간의 의사표시의 합치에 의하여 성립합니다. 민법 제105조에 따르면 계약은 당사자 간의 의사표시의 합치에 의하여 성립합니다.",
            "domain": "민사법",
            "expected_accuracy": 0.8
        },
        {
            "answer": "이혼은 혼인관계를 해소하는 법률행위입니다. 협의이혼, 조정이혼, 재판이혼의 방법이 있습니다.",
            "domain": "가족법",
            "expected_accuracy": 0.7
        }
    ]

    for i, case in enumerate(test_cases, 1):
        print(f"\n테스트 케이스 {i}:")
        print(f"원본 답변: {case['answer']}")

        # 용어 정확성 검증
        validation = validator.validate_legal_terms(case['answer'], case['domain'])

        print(f"추출된 용어: {validation.get('extracted_terms', [])}")
        print(f"전체 정확성: {validation.get('overall_accuracy', 0.0):.2f}")

        # 향상 제안
        enhancement = validator.enhance_term_accuracy(case['answer'], case['domain'])

        if enhancement.get('status') == 'needs_improvement':
            print("개선 제안:")
            for action in enhancement.get('action_plan', []):
                print(f"  - {action}")


def test_context_quality_enhancement():
    """컨텍스트 품질 향상 테스트"""
    print("\n" + "=" * 60)
    print("컨텍스트 품질 향상 테스트")
    print("=" * 60)

    enhancer = ContextQualityEnhancer()

    # 테스트 검색 결과
    test_search_results = [
        {
            "title": "계약서 검토 가이드",
            "content": "계약서 검토 시 당사자, 목적, 조건, 기간을 확인해야 합니다. 민법 제105조에 따르면 계약은 당사자 간의 의사표시의 합치에 의하여 성립합니다.",
            "source": "법무부"
        },
        {
            "title": "계약서 작성 요령",
            "content": "계약서 작성 시 명확한 조건과 기간을 명시해야 합니다. 해지 조건과 위약금 조항도 포함하는 것이 좋습니다.",
            "source": "법률신문"
        },
        {
            "title": "계약 분쟁 해결",
            "content": "계약 분쟁 발생 시 조정, 중재, 소송의 방법이 있습니다. 각각의 장단점을 고려하여 선택해야 합니다.",
            "source": "대법원"
        }
    ]

    query = "계약서 검토 시 주의사항은 무엇인가요?"
    question_type = "contract_review"
    domain = "민사법"

    print(f"질문: {query}")
    print(f"검색 결과 수: {len(test_search_results)}")

    # 컨텍스트 품질 향상
    enhancement = enhancer.enhance_context_quality(
        test_search_results, query, question_type, domain
    )

    if 'optimized_context' in enhancement:
        print(f"최적화된 컨텍스트 길이: {enhancement['optimized_context'].get('total_length', 0)}")
        print(f"선택된 소스 수: {enhancement['optimized_context'].get('source_count', 0)}")

    quality_metrics = enhancement.get('quality_metrics', {})
    print(f"컨텍스트 품질 점수: {quality_metrics.get('overall_quality', 0.0):.2f}")


def test_integrated_quality_enhancement():
    """통합 품질 향상 테스트"""
    print("\n" + "=" * 60)
    print("통합 품질 향상 테스트")
    print("=" * 60)

    enhancer = AnswerQualityEnhancer()

    # 테스트 케이스
    test_case = {
        "answer": "계약서 검토는 중요합니다. 당사자와 조건을 확인해야 합니다.",
        "query": "계약서 검토 시 주의사항은 무엇인가요?",
        "question_type": "contract_review",
        "domain": "민사법",
        "search_results": [
            {
                "title": "계약서 검토 가이드",
                "content": "계약서 검토 시 당사자, 목적, 조건, 기간을 확인해야 합니다. 민법 제105조에 따르면 계약은 당사자 간의 의사표시의 합치에 의하여 성립합니다.",
                "source": "법무부"
            }
        ],
        "sources": [
            {"title": "계약서 검토 가이드", "content": "계약서 검토 시 당사자, 목적, 조건, 기간을 확인해야 합니다.", "source": "법무부"}
        ]
    }

    print(f"질문: {test_case['query']}")
    print(f"원본 답변: {test_case['answer']}")

    # 통합 품질 향상
    enhancement_result = enhancer.enhance_answer_quality(
        test_case['answer'],
        test_case['query'],
        test_case['question_type'],
        test_case['domain'],
        test_case['search_results'],
        test_case['sources']
    )

    if 'error' not in enhancement_result:
        print(f"\n향상된 답변:\n{enhancement_result.get('enhanced_answer', '')}")

        # 품질 향상 보고서
        quality_report = enhancer.get_quality_report(enhancement_result)

        print(f"\n품질 향상 보고서:")
        print(f"원본 품질: {quality_report['summary']['original_quality']:.2f}")
        print(f"최종 품질: {quality_report['summary']['final_quality']:.2f}")
        print(f"개선 효과: {quality_report['summary']['improvement']:.2f}%")

        print(f"\n상세 메트릭:")
        for metric, data in quality_report['detailed_metrics'].items():
            print(f"  {metric}: {data['original']:.2f} → {data['final']:.2f} ({data['improvement']:.2f}%)")

        print(f"\n권장사항:")
        for recommendation in quality_report['recommendations']:
            print(f"  - {recommendation}")
    else:
        print(f"오류 발생: {enhancement_result['error']}")


def main():
    """메인 테스트 함수"""
    print("LawFirmAI 답변 품질 향상 시스템 테스트")
    print("=" * 60)
    print(f"테스트 시작 시간: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

    try:
        # 1. 키워드 포함도 향상 테스트
        test_keyword_coverage_enhancement()

        # 2. 답변 구조화 향상 테스트
        test_answer_structure_enhancement()

        # 3. 법률 용어 정확성 검증 테스트
        test_legal_term_validation()

        # 4. 컨텍스트 품질 향상 테스트
        test_context_quality_enhancement()

        # 5. 통합 품질 향상 테스트
        test_integrated_quality_enhancement()

        print("\n" + "=" * 60)
        print("모든 테스트가 완료되었습니다!")
        print("=" * 60)

    except Exception as e:
        print(f"\n테스트 중 오류 발생: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
