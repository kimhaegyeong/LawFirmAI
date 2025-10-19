#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
새로 추가된 classify_question_type 메서드 테스트
"""

import sys
import os

# 프로젝트 루트 디렉토리를 Python 경로에 추가
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from source.services.answer_structure_enhancer import AnswerStructureEnhancer, QuestionType


def test_classify_question_type():
    """새로 추가된 classify_question_type 메서드 테스트"""
    print("=" * 60)
    print("classify_question_type 메서드 테스트")
    print("=" * 60)
    
    enhancer = AnswerStructureEnhancer()
    
    # 테스트 케이스들
    test_cases = [
        # 법률 문의 테스트
        ("민법 제123조의 내용이 무엇인가요?", QuestionType.LAW_INQUIRY),
        ("형법 제250조 처벌 기준은?", QuestionType.LAW_INQUIRY),
        ("근로기준법 제15조 의미는?", QuestionType.LAW_INQUIRY),
        ("상법 제123조 해석해주세요", QuestionType.LAW_INQUIRY),
        ("헌법 제10조 내용은?", QuestionType.LAW_INQUIRY),
        ("특허법 제25조 규정은?", QuestionType.LAW_INQUIRY),
        
        # 판례 검색 테스트
        ("대법원 판례를 찾아주세요", QuestionType.PRECEDENT_SEARCH),
        ("관련 판례가 있나요?", QuestionType.PRECEDENT_SEARCH),
        ("고등법원 판결을 알려주세요", QuestionType.PRECEDENT_SEARCH),
        ("지방법원 판례 검색", QuestionType.PRECEDENT_SEARCH),
        
        # 계약서 검토 테스트
        ("계약서를 검토해주세요", QuestionType.CONTRACT_REVIEW),
        ("이 계약 조항이 불리한가요?", QuestionType.CONTRACT_REVIEW),
        ("계약서 수정이 필요한가요?", QuestionType.CONTRACT_REVIEW),
        
        # 이혼 절차 테스트
        ("이혼 절차를 알려주세요", QuestionType.DIVORCE_PROCEDURE),
        ("협의이혼 방법은?", QuestionType.DIVORCE_PROCEDURE),
        ("재판이혼 절차는?", QuestionType.DIVORCE_PROCEDURE),
        ("이혼절차 신청 방법", QuestionType.DIVORCE_PROCEDURE),
        
        # 상속 절차 테스트
        ("상속 절차를 알려주세요", QuestionType.INHERITANCE_PROCEDURE),
        ("유산 분할 방법은?", QuestionType.INHERITANCE_PROCEDURE),
        ("상속인 확인 방법", QuestionType.INHERITANCE_PROCEDURE),
        ("상속세 신고 절차", QuestionType.INHERITANCE_PROCEDURE),
        ("유언 검인 절차", QuestionType.INHERITANCE_PROCEDURE),
        ("상속포기 방법", QuestionType.INHERITANCE_PROCEDURE),
        
        # 형사 사건 테스트
        ("사기죄 구성요건은?", QuestionType.CRIMINAL_CASE),
        ("절도 범죄 처벌은?", QuestionType.CRIMINAL_CASE),
        ("강도 사건 대응 방법", QuestionType.CRIMINAL_CASE),
        ("살인죄 형량은?", QuestionType.CRIMINAL_CASE),
        ("형사 사건 절차", QuestionType.CRIMINAL_CASE),
        
        # 노동 분쟁 테스트
        ("노동 분쟁 해결 방법", QuestionType.LABOR_DISPUTE),
        ("근로 시간 규정은?", QuestionType.LABOR_DISPUTE),
        ("임금 체불 대응", QuestionType.LABOR_DISPUTE),
        ("부당해고 구제 방법", QuestionType.LABOR_DISPUTE),
        ("해고 통보 대응", QuestionType.LABOR_DISPUTE),
        ("노동위원회 신청", QuestionType.LABOR_DISPUTE),
        
        # 절차 안내 테스트
        ("소송 절차를 알려주세요", QuestionType.PROCEDURE_GUIDE),
        ("민사조정 신청 방법", QuestionType.PROCEDURE_GUIDE),
        ("소액사건 절차는?", QuestionType.PROCEDURE_GUIDE),
        ("어떻게 신청하나요?", QuestionType.PROCEDURE_GUIDE),
        
        # 법률 용어 설명 테스트
        ("법인격의 의미는?", QuestionType.TERM_EXPLANATION),
        ("소멸시효 정의는?", QuestionType.TERM_EXPLANATION),
        ("무효와 취소의 개념", QuestionType.TERM_EXPLANATION),
        ("무엇이 계약인가요?", QuestionType.TERM_EXPLANATION),
        ("뜻을 설명해주세요", QuestionType.TERM_EXPLANATION),
        
        # 법률 자문 테스트
        ("어떻게 대응해야 하나요?", QuestionType.LEGAL_ADVICE),
        ("권리 구제 방법은?", QuestionType.LEGAL_ADVICE),
        ("의무 이행 방법", QuestionType.LEGAL_ADVICE),
        ("해야 할 일은?", QuestionType.LEGAL_ADVICE),
        
        # 일반 질문 테스트
        ("안녕하세요", QuestionType.GENERAL_QUESTION),
        ("도움이 필요합니다", QuestionType.GENERAL_QUESTION),
        ("질문이 있습니다", QuestionType.GENERAL_QUESTION),
    ]
    
    # 테스트 실행
    correct_count = 0
    total_count = len(test_cases)
    
    print(f"\n총 {total_count}개 테스트 케이스 실행 중...\n")
    
    for i, (question, expected_type) in enumerate(test_cases, 1):
        try:
            result_type = enhancer.classify_question_type(question)
            is_correct = result_type == expected_type
            
            if is_correct:
                correct_count += 1
                status = "✅"
            else:
                status = "❌"
            
            print(f"{i:2d}. {status} 질문: {question}")
            print(f"    예상: {expected_type.value}")
            print(f"    결과: {result_type.value}")
            
            if not is_correct:
                print(f"    ⚠️  불일치!")
            print()
            
        except Exception as e:
            print(f"{i:2d}. ❌ 오류: {question}")
            print(f"    오류 메시지: {e}")
            print()
    
    # 결과 요약
    accuracy = (correct_count / total_count) * 100
    
    print("=" * 60)
    print("테스트 결과 요약")
    print("=" * 60)
    print(f"총 테스트 케이스: {total_count}")
    print(f"정확한 분류: {correct_count}")
    print(f"부정확한 분류: {total_count - correct_count}")
    print(f"정확도: {accuracy:.1f}%")
    
    if accuracy >= 90:
        grade = "A+"
    elif accuracy >= 80:
        grade = "A"
    elif accuracy >= 70:
        grade = "B"
    elif accuracy >= 60:
        grade = "C"
    else:
        grade = "D"
    
    print(f"등급: {grade}")
    
    # 질문 유형별 정확도 분석
    print(f"\n질문 유형별 분석:")
    type_stats = {}
    for question, expected_type in test_cases:
        if expected_type not in type_stats:
            type_stats[expected_type] = {"total": 0, "correct": 0}
        type_stats[expected_type]["total"] += 1
    
    # 실제 분류 결과로 정확도 계산
    for question, expected_type in test_cases:
        try:
            result_type = enhancer.classify_question_type(question)
            if result_type == expected_type:
                type_stats[expected_type]["correct"] += 1
        except:
            pass
    
    for question_type, stats in type_stats.items():
        accuracy = (stats["correct"] / stats["total"]) * 100
        print(f"  {question_type.value}: {stats['correct']}/{stats['total']} ({accuracy:.1f}%)")
    
    print("=" * 60)
    
    return accuracy


def test_edge_cases():
    """엣지 케이스 테스트"""
    print("\n" + "=" * 60)
    print("엣지 케이스 테스트")
    print("=" * 60)
    
    enhancer = AnswerStructureEnhancer()
    
    edge_cases = [
        ("", QuestionType.GENERAL_QUESTION),  # 빈 문자열
        ("   ", QuestionType.GENERAL_QUESTION),  # 공백만
        ("민법", QuestionType.GENERAL_QUESTION),  # 단어만
        ("제123조", QuestionType.LAW_INQUIRY),  # 조문만
        ("123조", QuestionType.GENERAL_QUESTION),  # 숫자+조문
        ("민법 제", QuestionType.GENERAL_QUESTION),  # 불완전한 조문
        ("제조", QuestionType.GENERAL_QUESTION),  # 잘못된 조문
        ("민법 제123조 제456항", QuestionType.LAW_INQUIRY),  # 복합 조문
        ("민법과 형법", QuestionType.LAW_INQUIRY),  # 여러 법령
        ("판례와 계약서", QuestionType.PRECEDENT_SEARCH),  # 여러 키워드 (우선순위)
    ]
    
    print(f"\n엣지 케이스 {len(edge_cases)}개 테스트 중...\n")
    
    for i, (question, expected_type) in enumerate(edge_cases, 1):
        try:
            result_type = enhancer.classify_question_type(question)
            is_correct = result_type == expected_type
            status = "✅" if is_correct else "❌"
            
            print(f"{i:2d}. {status} 질문: '{question}'")
            print(f"    예상: {expected_type.value}")
            print(f"    결과: {result_type.value}")
            print()
            
        except Exception as e:
            print(f"{i:2d}. ❌ 오류: '{question}'")
            print(f"    오류 메시지: {e}")
            print()


def test_performance():
    """성능 테스트"""
    print("\n" + "=" * 60)
    print("classify_question_type 성능 테스트")
    print("=" * 60)
    
    import time
    
    enhancer = AnswerStructureEnhancer()
    
    # 테스트 질문들
    test_questions = [
        "민법 제123조의 내용이 무엇인가요?",
        "계약서를 검토해주세요",
        "이혼 절차를 알려주세요",
        "판례를 찾아주세요",
        "법률 상담이 필요합니다",
        "상속 절차는 어떻게 되나요?",
        "노동 분쟁 해결 방법",
        "형사 사건 대응 방법",
        "법률 용어 설명해주세요",
        "소송 절차를 알려주세요"
    ] * 100  # 1000개 질문
    
    print(f"\n{len(test_questions)}개 질문 분류 성능 테스트 중...")
    
    start_time = time.time()
    results = []
    
    for question in test_questions:
        try:
            result = enhancer.classify_question_type(question)
            results.append(result)
        except Exception as e:
            print(f"오류 발생: {e}")
            results.append(QuestionType.GENERAL_QUESTION)
    
    end_time = time.time()
    total_time = end_time - start_time
    
    print(f"\n성능 결과:")
    print(f"  총 처리 시간: {total_time:.3f}초")
    print(f"  평균 처리 시간: {total_time/len(test_questions)*1000:.2f}ms")
    print(f"  초당 처리량: {len(test_questions)/total_time:.0f} questions/sec")
    
    # 결과 분포
    result_counts = {}
    for result in results:
        result_counts[result] = result_counts.get(result, 0) + 1
    
    print(f"\n분류 결과 분포:")
    for question_type, count in sorted(result_counts.items(), key=lambda x: x[1], reverse=True):
        percentage = (count / len(results)) * 100
        print(f"  {question_type.value}: {count}개 ({percentage:.1f}%)")


def main():
    """메인 함수"""
    print("classify_question_type 메서드 종합 테스트")
    
    try:
        # 1. 기본 기능 테스트
        accuracy = test_classify_question_type()
        
        # 2. 엣지 케이스 테스트
        test_edge_cases()
        
        # 3. 성능 테스트
        test_performance()
        
        print(f"\n🎉 모든 테스트가 완료되었습니다!")
        print(f"📊 최종 정확도: {accuracy:.1f}%")
        
    except Exception as e:
        print(f"❌ 테스트 중 오류 발생: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
