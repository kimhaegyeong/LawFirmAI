#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
구조화 품질 오류 수정 검증 테스트
"""

import sys
import os

# 프로젝트 루트 디렉토리를 Python 경로에 추가
sys.path.append(os.path.join(os.path.dirname(__file__), 'source'))

from source.services.answer_structure_enhancer import AnswerStructureEnhancer


def test_structure_enhancement():
    """구조화 향상 테스트"""
    print("=" * 60)
    print("구조화 품질 오류 수정 검증 테스트")
    print("=" * 60)
    
    enhancer = AnswerStructureEnhancer()
    
    # 테스트 케이스들
    test_cases = [
        {
            "name": "정상 케이스",
            "answer": "계약서 검토는 중요합니다. 당사자와 조건을 확인해야 합니다.",
            "question_type": "contract_review",
            "question": "계약서 검토 시 주의사항은 무엇인가요?",
            "domain": "민사법"
        },
        {
            "name": "빈 답변 케이스",
            "answer": "",
            "question_type": "general",
            "question": "테스트 질문",
            "domain": "general"
        },
        {
            "name": "None 답변 케이스",
            "answer": None,
            "question_type": "general",
            "question": "테스트 질문",
            "domain": "general"
        },
        {
            "name": "짧은 답변 케이스",
            "answer": "네",
            "question_type": "general",
            "question": "테스트 질문",
            "domain": "general"
        },
        {
            "name": "이혼 절차 케이스",
            "answer": "이혼 절차는 협의이혼, 조정이혼, 재판이혼이 있습니다.",
            "question_type": "divorce_procedure",
            "question": "이혼 절차는 어떻게 진행되나요?",
            "domain": "가족법"
        }
    ]
    
    success_count = 0
    total_count = len(test_cases)
    
    for i, case in enumerate(test_cases, 1):
        print(f"\n테스트 케이스 {i}: {case['name']}")
        print(f"질문: {case['question']}")
        print(f"답변: {case['answer']}")
        
        try:
            # 구조화 향상 실행
            result = enhancer.enhance_answer_structure(
                case['answer'],
                case['question_type'],
                case['question'],
                case['domain']
            )
            
            if 'error' in result:
                print(f"❌ 오류 발생: {result['error']}")
            else:
                print(f"✅ 성공!")
                print(f"   질문 유형: {result.get('question_type', 'Unknown')}")
                print(f"   템플릿: {result.get('template_used', 'Unknown')}")
                
                quality_metrics = result.get('quality_metrics', {})
                print(f"   구조화 점수: {quality_metrics.get('structure_score', 0.0):.2f}")
                print(f"   완성도 점수: {quality_metrics.get('completeness_score', 0.0):.2f}")
                print(f"   전체 점수: {quality_metrics.get('overall_score', 0.0):.2f}")
                
                success_count += 1
        
        except Exception as e:
            print(f"❌ 예외 발생: {e}")
            import traceback
            traceback.print_exc()
    
    print(f"\n" + "=" * 60)
    print(f"테스트 결과: {success_count}/{total_count} 성공")
    print(f"성공률: {(success_count/total_count)*100:.1f}%")
    print("=" * 60)
    
    return success_count == total_count


def test_edge_cases():
    """엣지 케이스 테스트"""
    print("\n" + "=" * 60)
    print("엣지 케이스 테스트")
    print("=" * 60)
    
    enhancer = AnswerStructureEnhancer()
    
    # 엣지 케이스들
    edge_cases = [
        {
            "name": "매우 긴 답변",
            "answer": "A" * 10000,
            "question_type": "general",
            "question": "테스트",
            "domain": "general"
        },
        {
            "name": "특수 문자 포함",
            "answer": "!@#$%^&*()_+{}|:<>?[]\\;'\",./",
            "question_type": "general",
            "question": "테스트",
            "domain": "general"
        },
        {
            "name": "한글만 포함",
            "answer": "안녕하세요법률상담입니다",
            "question_type": "general",
            "question": "테스트",
            "domain": "general"
        },
        {
            "name": "숫자만 포함",
            "answer": "1234567890",
            "question_type": "general",
            "question": "테스트",
            "domain": "general"
        }
    ]
    
    success_count = 0
    
    for i, case in enumerate(edge_cases, 1):
        print(f"\n엣지 케이스 {i}: {case['name']}")
        
        try:
            result = enhancer.enhance_answer_structure(
                case['answer'],
                case['question_type'],
                case['question'],
                case['domain']
            )
            
            if 'error' not in result:
                print(f"✅ 성공!")
                success_count += 1
            else:
                print(f"❌ 오류: {result['error']}")
        
        except Exception as e:
            print(f"❌ 예외: {e}")
    
    print(f"\n엣지 케이스 테스트 결과: {success_count}/{len(edge_cases)} 성공")
    return success_count == len(edge_cases)


def main():
    """메인 테스트 함수"""
    print("구조화 품질 오류 수정 검증 테스트 시작")
    
    # 기본 테스트
    basic_test_passed = test_structure_enhancement()
    
    # 엣지 케이스 테스트
    edge_test_passed = test_edge_cases()
    
    # 전체 결과
    print(f"\n" + "=" * 60)
    print("전체 테스트 결과")
    print("=" * 60)
    print(f"기본 테스트: {'✅ 통과' if basic_test_passed else '❌ 실패'}")
    print(f"엣지 케이스 테스트: {'✅ 통과' if edge_test_passed else '❌ 실패'}")
    
    if basic_test_passed and edge_test_passed:
        print("🎉 모든 테스트가 통과했습니다!")
        print("구조화 품질 오류가 성공적으로 수정되었습니다.")
    else:
        print("⚠️ 일부 테스트가 실패했습니다.")
        print("추가 수정이 필요할 수 있습니다.")
    
    return basic_test_passed and edge_test_passed


if __name__ == "__main__":
    main()
