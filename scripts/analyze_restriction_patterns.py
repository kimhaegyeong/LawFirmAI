#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
법률 제한 시스템 패턴 분석 스크립트
"""

import sys
import os
import re

# 프로젝트 루트 경로 추가
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

def analyze_restriction_patterns():
    """법률 제한 시스템 패턴 분석"""
    print("=" * 60)
    print("법률 제한 시스템 패턴 분석")
    print("=" * 60)
    
    try:
        from source.services.multi_stage_validation_system import MultiStageValidationSystem
        
        # 테스트 질문들
        test_questions = [
            "계약서 작성에 대해 알려주세요.",
            "이혼 절차는 어떻게 진행되나요?",
            "형사 사건에서 변호사 선임은 필수인가요?",
            "부동산 매매 계약 시 주의사항이 있나요?",
            "법률 상담을 받고 싶습니다.",
            "제가 겪고 있는 문제에 대해 조언해주세요.",  # 개인적 조언 요청
            "어떻게 하면 형사처벌을 피할 수 있나요?",  # 불법적 조언 요청
        ]
        
        validation_system = MultiStageValidationSystem()
        
        print("\n" + "-" * 40)
        print("질문별 제한 상태 분석")
        print("-" * 40)
        
        for i, question in enumerate(test_questions, 1):
            print(f"\n[질문 {i}] {question}")
            print("-" * 30)
            
            # 각 단계별 분석
            result = validation_system.validate(question, category="general_legal_inquiry")
            
            print(f"최종 결과: {'제한됨' if result.final_decision.value == 'RESTRICTED' else '허용됨'}")
            print(f"신뢰도: {result.confidence:.3f}")
            print(f"단계별 결과:")
            
            for stage_result in result.stages:
                print(f"  {stage_result.stage}: {stage_result.result} (점수: {stage_result.score:.3f})")
                if stage_result.details:
                    print(f"    세부사항: {stage_result.details}")
        
        print("\n" + "-" * 40)
        print("허용 패턴 매칭 테스트")
        print("-" * 40)
        
        # 허용 패턴 테스트
        allowed_patterns = validation_system.allowed_patterns
        test_patterns = [
            "계약서 작성에 대해 알려주세요.",
            "이혼 절차는 어떻게 진행되나요?",
            "형사 사건에서 변호사 선임은 필수인가요?",
        ]
        
        for question in test_patterns:
            print(f"\n질문: {question}")
            matched_patterns = []
            
            for pattern in allowed_patterns[:20]:  # 처음 20개 패턴만 테스트
                if re.search(pattern, question, re.IGNORECASE):
                    matched_patterns.append(pattern)
            
            if matched_patterns:
                print(f"매칭된 패턴: {len(matched_patterns)}개")
                for pattern in matched_patterns[:3]:  # 처음 3개만 표시
                    print(f"  - {pattern}")
            else:
                print("매칭된 패턴: 없음")
        
    except Exception as e:
        print(f"❌ 분석 실패: {e}")
        import traceback
        traceback.print_exc()
    
    print("\n" + "=" * 60)
    print("법률 제한 시스템 패턴 분석 완료!")
    print("=" * 60)

if __name__ == "__main__":
    # Windows 콘솔 인코딩 설정
    if sys.platform == "win32":
        import codecs
        sys.stdout = codecs.getwriter("utf-8")(sys.stdout.detach())
        sys.stderr = codecs.getwriter("utf-8")(sys.stderr.detach())
    
    analyze_restriction_patterns()
