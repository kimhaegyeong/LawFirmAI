#!/usr/bin/env python3
"""
분류기 성능 비교 테스트
기존 분류기와 하이브리드 분류기의 성능 비교
"""

import sys
import os
import time
from pathlib import Path

# 프로젝트 루트 디렉토리를 Python 경로에 추가
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

from source.services.question_classifier import QuestionClassifier
from source.services.integrated_hybrid_classifier import IntegratedHybridQuestionClassifier

def test_classifier_performance():
    """분류기 성능 비교 테스트"""
    print("=" * 60)
    print("분류기 성능 비교 테스트")
    print("=" * 60)
    
    # 분류기 초기화
    print("분류기 초기화 중...")
    old_classifier = QuestionClassifier()
    hybrid_classifier = IntegratedHybridQuestionClassifier()
    print("✓ 분류기 초기화 완료")
    
    # 테스트 질문들
    test_questions = [
        "민법 제750조 손해배상에 대해 알려주세요",
        "계약서를 검토해주세요",
        "이혼 절차는 어떻게 되나요?",
        "어떻게 대응해야 하나요?",
        "관련 판례를 찾아주세요",
        "무엇이 불법행위인가요?",
        "상속 절차를 알려주세요",
        "노동 분쟁 해결 방법",
        "형사 사건 절차",
        "용어의 의미를 설명해주세요",
        "회사에서 직원을 해고할 때 어떤 절차를 따라야 하나요?",
        "부동산 매매 계약서에 문제가 있어서 손해를 입었습니다",
        "이혼할 때 자녀의 양육권을 어떻게 결정하나요?",
        "상속받은 부동산을 다른 형제들과 나누려고 합니다",
        "근로시간이 법정 기준을 초과하는데 임금을 받지 못했습니다",
        "계약서에 명시되지 않은 추가 비용을 요구받고 있습니다",
        "사업자 등록을 하지 않고 사업을 했는데 문제가 될까요?",
        "임대차 계약 기간이 끝났는데 계속 거주할 수 있나요?",
        "회사에서 부당한 해고를 당했을 때 구제받을 방법이 있나요?",
        "상속 포기를 하고 싶은데 어떤 절차가 필요한가요?"
    ]
    
    print(f"총 {len(test_questions)}개 질문으로 성능 테스트")
    
    # 기존 분류기 테스트
    print("\n1. 기존 분류기 성능 테스트")
    start_time = time.time()
    
    old_results = []
    for question in test_questions:
        result = old_classifier.classify_question(question)
        old_results.append(result)
    
    old_time = time.time() - start_time
    print(f"기존 분류기 처리 시간: {old_time:.3f}초")
    print(f"평균 처리 시간: {old_time/len(test_questions)*1000:.2f}ms")
    
    # 하이브리드 분류기 테스트
    print("\n2. 하이브리드 분류기 성능 테스트")
    start_time = time.time()
    
    hybrid_results = []
    for question in test_questions:
        result = hybrid_classifier.classify(question)
        hybrid_results.append(result)
    
    hybrid_time = time.time() - start_time
    print(f"하이브리드 분류기 처리 시간: {hybrid_time:.3f}초")
    print(f"평균 처리 시간: {hybrid_time/len(test_questions)*1000:.2f}ms")
    
    # 성능 비교
    print("\n3. 성능 비교")
    if old_time > 0:
        speed_improvement = ((old_time - hybrid_time) / old_time * 100)
        print(f"속도 개선: {speed_improvement:.1f}%")
    
    # 분류 결과 비교
    print("\n4. 분류 결과 비교")
    print("질문별 분류 결과:")
    print("-" * 80)
    print(f"{'질문':<30} {'기존':<15} {'하이브리드':<15} {'신뢰도':<10}")
    print("-" * 80)
    
    for i, question in enumerate(test_questions):
        old_result = old_results[i]
        hybrid_result = hybrid_results[i]
        
        old_type = old_result.question_type if hasattr(old_result, 'question_type') else 'unknown'
        hybrid_type = hybrid_result.question_type.value
        hybrid_confidence = hybrid_result.confidence
        
        print(f"{question[:28]:<30} {old_type:<15} {hybrid_type:<15} {hybrid_confidence:.3f}")
    
    # 하이브리드 분류기 통계
    print("\n5. 하이브리드 분류기 통계")
    hybrid_stats = hybrid_classifier.get_stats()
    print("하이브리드 분류기 통계:")
    for key, value in hybrid_stats.items():
        print(f"  {key}: {value}")
    
    # 정확도 분석
    print("\n6. 정확도 분석")
    same_classification = 0
    different_classification = 0
    
    for i in range(len(test_questions)):
        old_result = old_results[i]
        hybrid_result = hybrid_results[i]
        
        old_type = old_result.question_type if hasattr(old_result, 'question_type') else 'unknown'
        hybrid_type = hybrid_result.question_type.value
        
        if old_type == hybrid_type:
            same_classification += 1
        else:
            different_classification += 1
    
    total = len(test_questions)
    agreement_rate = (same_classification / total) * 100
    
    print(f"동일한 분류: {same_classification}/{total} ({agreement_rate:.1f}%)")
    print(f"다른 분류: {different_classification}/{total} ({100-agreement_rate:.1f}%)")
    
    print("\n✓ 성능 테스트 완료!")

if __name__ == "__main__":
    test_classifier_performance()
