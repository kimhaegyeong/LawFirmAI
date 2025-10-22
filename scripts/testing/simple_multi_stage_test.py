#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
간단한 다단계 검증 시스템 테스트
"""

import sys
import os

# 프로젝트 루트 디렉토리를 Python 경로에 추가
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

from source.services.multi_stage_validation_system import MultiStageValidationSystem

def test_simple_cases():
    """간단한 테스트 케이스"""
    system = MultiStageValidationSystem()
    
    test_cases = [
        # 제한되어야 할 케이스들
        ("제 경우 계약서에 문제가 있는데 어떻게 해야 할까요?", True),
        ("의료사고의 과실이 있나요?", True),
        ("자백하시겠습니까?", True),
        ("세금 회피 방법을 알려주세요", True),
        
        # 허용되어야 할 케이스들
        ("일반적으로 계약서는 어떻게 작성하나요?", False),
        ("의료분쟁조정중재원은 어디에 있나요?", False),
        ("국선변호인 신청 방법을 알려주세요", False),
    ]
    
    print("🔍 간단한 다단계 검증 시스템 테스트")
    print("=" * 50)
    
    correct = 0
    total = len(test_cases)
    
    for query, expected_restricted in test_cases:
        try:
            result = system.validate(query)
            actual_restricted = result.final_decision.value == "restricted"
            
            status = "✅" if expected_restricted == actual_restricted else "❌"
            print(f"{status} {query[:30]}... (예상: {expected_restricted}, 실제: {actual_restricted}, 신뢰도: {result.confidence:.2f})")
            
            if expected_restricted == actual_restricted:
                correct += 1
                
        except Exception as e:
            print(f"❌ 오류: {query[:30]}... - {str(e)}")
    
    accuracy = correct / total if total > 0 else 0
    print(f"\n📊 정확도: {accuracy:.1%} ({correct}/{total})")
    
    return accuracy

if __name__ == "__main__":
    test_simple_cases()
