#!/usr/bin/env python3
"""
통합된 하이브리드 분류기 훈련 스크립트
"""

import sys
import os
from pathlib import Path

# 프로젝트 루트 디렉토리를 Python 경로에 추가
project_root = Path(__file__).parent.parent.parent
sys.path.append(str(project_root))

from source.services.integrated_hybrid_classifier import IntegratedHybridQuestionClassifier
from source.services.unified_question_types import UnifiedQuestionType
from scripts.ml_training.create_integrated_training_data import create_integrated_training_data

def train_integrated_classifier():
    """통합된 하이브리드 분류기 훈련"""
    print("=" * 60)
    print("통합된 하이브리드 분류기 훈련 시작")
    print("=" * 60)
    
    # 하이브리드 분류기 생성
    classifier = IntegratedHybridQuestionClassifier(confidence_threshold=0.7)
    
    # 훈련 데이터 생성
    print("훈련 데이터 생성 중...")
    training_data = create_integrated_training_data()
    print(f"총 {len(training_data)}개의 훈련 데이터 생성")
    
    # ML 모델 훈련
    print("\nML 모델 훈련 중...")
    classifier.train_ml_model(training_data)
    
    # 테스트
    test_questions = [
        "민법 제123조의 내용이 무엇인가요?",
        "계약서를 검토해주세요",
        "이혼 절차를 알려주세요",
        "어떻게 대응해야 하나요?",
        "판례를 찾아주세요",
        "무엇이 계약인가요?"
    ]
    
    print("\n하이브리드 분류 테스트:")
    print("=" * 60)
    
    for question in test_questions:
        result = classifier.classify(question)
        print(f"질문: {question}")
        print(f"분류: {result.question_type.value}")
        print(f"신뢰도: {result.confidence:.3f}")
        print(f"방법: {result.method}")
        print(f"이유: {result.reasoning}")
        print("-" * 40)
    
    # 통계 출력
    print("\n성능 통계:")
    stats = classifier.get_stats()
    for key, value in stats.items():
        print(f"{key}: {value}")
    
    print("\n훈련 완료!")

if __name__ == "__main__":
    train_integrated_classifier()
