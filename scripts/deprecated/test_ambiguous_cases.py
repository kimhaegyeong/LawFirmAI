#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
애매한 케이스 테스트
"""

import sys
import os

# 프로젝트 루트 디렉토리를 Python 경로에 추가
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

from source.services.multi_stage_validation_system import MultiStageValidationSystem

def test_ambiguous_cases():
    """애매한 케이스 테스트"""
    system = MultiStageValidationSystem()
    
    ambiguous_cases = [
        "어떻게 해야 할까요?",
        "법적 문제가 있어요",
        "도움이 필요해요",
        "문제가 생겼어요",
        "어떤 방법이 있을까요?",
        "해결 방법을 알려주세요",
        "무엇을 해야 할까요?",
        "조언이 필요해요",
        "어떻게 대응해야 할까요?",
        "방법을 알려주세요"
    ]
    
    print("🔍 애매한 케이스 테스트")
    print("=" * 50)
    
    correct = 0
    total = len(ambiguous_cases)
    
    for query in ambiguous_cases:
        try:
            result = system.validate(query)
            actual_restricted = result.final_decision.value == "restricted"
            expected_restricted = False  # 애매한 케이스는 모두 허용되어야 함
            
            status = "✅" if expected_restricted == actual_restricted else "❌"
            print(f"{status} {query} (예상: {expected_restricted}, 실제: {actual_restricted}, 신뢰도: {result.confidence:.2f})")
            
            if expected_restricted == actual_restricted:
                correct += 1
                
        except Exception as e:
            print(f"❌ 오류: {query} - {str(e)}")
    
    accuracy = correct / total if total > 0 else 0
    print(f"\n📊 애매한 케이스 정확도: {accuracy:.1%} ({correct}/{total})")
    
    return accuracy

if __name__ == "__main__":
    test_ambiguous_cases()
