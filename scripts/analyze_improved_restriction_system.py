#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ImprovedLegalRestrictionSystem 분석 스크립트
"""

import sys
import os

# 프로젝트 루트 경로 추가
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

def analyze_improved_restriction_system():
    """ImprovedLegalRestrictionSystem 분석"""
    print("=" * 60)
    print("ImprovedLegalRestrictionSystem 분석")
    print("=" * 60)
    
    try:
        from source.services.improved_legal_restriction_system import ImprovedLegalRestrictionSystem
        
        # 테스트 질문들
        test_questions = [
            "계약서 작성에 대해 알려주세요.",
            "이혼 절차는 어떻게 진행되나요?",
            "형사 사건에서 변호사 선임은 필수인가요?",
            "부동산 매매 계약 시 주의사항이 있나요?",
            "제가 겪고 있는 문제에 대해 조언해주세요.",  # 개인적 조언 요청
        ]
        
        restriction_system = ImprovedLegalRestrictionSystem()
        
        print("\n" + "-" * 40)
        print("질문별 제한 상태 분석")
        print("-" * 40)
        
        for i, question in enumerate(test_questions, 1):
            print(f"\n[질문 {i}] {question}")
            print("-" * 30)
            
            # 제한 검사
            result = restriction_system.check_restrictions(question)
            
            print(f"제한 여부: {'제한됨' if result.is_restricted else '허용됨'}")
            print(f"제한 수준: {result.restriction_level}")
            print(f"신뢰도: {result.confidence:.3f}")
            print(f"맥락 분석: {result.context_analysis.context_type}")
            print(f"추론: {result.reasoning}")
            
            if result.matched_rules:
                print(f"매칭된 규칙: {len(result.matched_rules)}개")
                for rule in result.matched_rules[:2]:  # 처음 2개만 표시
                    print(f"  - {rule.id}: {rule.area}")
            
            if result.matched_patterns:
                print(f"매칭된 패턴: {len(result.matched_patterns)}개")
                for pattern in result.matched_patterns[:2]:  # 처음 2개만 표시
                    print(f"  - {pattern}")
        
        print("\n" + "-" * 40)
        print("예외 패턴 확인")
        print("-" * 40)
        
        # 예외 패턴 확인
        print(f"컴파일된 예외 패턴 수: {len(restriction_system.compiled_exceptions)}")
        
        # 일반적인 법률 정보 요청에 대한 예외 패턴 테스트
        general_questions = [
            "계약서 작성에 대해 알려주세요.",
            "이혼 절차는 어떻게 진행되나요?",
            "형사 사건에서 변호사 선임은 필수인가요?",
        ]
        
        for question in general_questions:
            exception_matched = restriction_system._check_exceptions(question)
            print(f"\n질문: {question}")
            print(f"예외 패턴 매칭: {exception_matched if exception_matched else '없음'}")
        
    except Exception as e:
        print(f"❌ 분석 실패: {e}")
        import traceback
        traceback.print_exc()
    
    print("\n" + "=" * 60)
    print("ImprovedLegalRestrictionSystem 분석 완료!")
    print("=" * 60)

if __name__ == "__main__":
    # Windows 콘솔 인코딩 설정
    if sys.platform == "win32":
        import codecs
        sys.stdout = codecs.getwriter("utf-8")(sys.stdout.detach())
        sys.stderr = codecs.getwriter("utf-8")(sys.stderr.detach())
    
    analyze_improved_restriction_system()

