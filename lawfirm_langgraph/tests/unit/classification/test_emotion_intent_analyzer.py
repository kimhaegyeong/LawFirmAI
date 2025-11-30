# -*- coding: utf-8 -*-
"""
EmotionIntentAnalyzer 테스트
하이브리드 방식 (패턴 기반 + KoBERT) 및 배치 처리 테스트
"""

import pytest
import time
import os
from typing import List

try:
    from lawfirm_langgraph.core.classification.analyzers.emotion_intent_analyzer import (
        EmotionIntentAnalyzer,
        EmotionType,
        IntentType,
        UrgencyLevel
    )
except ImportError:
    from core.classification.analyzers.emotion_intent_analyzer import (
        EmotionIntentAnalyzer,
        EmotionType,
        IntentType,
        UrgencyLevel
    )


class TestEmotionIntentAnalyzer:
    """EmotionIntentAnalyzer 테스트 클래스"""
    
    @pytest.fixture
    def analyzer(self):
        """분석기 인스턴스 생성"""
        return EmotionIntentAnalyzer(use_ml_model=True)
    
    @pytest.fixture
    def analyzer_pattern_only(self):
        """패턴 기반만 사용하는 분석기"""
        return EmotionIntentAnalyzer(use_ml_model=False)
    
    @pytest.fixture
    def test_texts(self):
        """테스트 텍스트 리스트"""
        return [
            "손해배상 청구 방법을 알려주세요",
            "급해요! 오늘까지 답변해주세요!",
            "감사합니다. 정말 도움이 되었어요",
            "이해가 안 돼요. 더 자세히 설명해주세요",
            "왜 이런 문제가 생겼나요? 정말 화나네요",
            "추가로 궁금한 것이 있어요",
            "정확히 말하면 어떤 절차인가요?"
        ]
    
    def test_analyze_emotion_single(self, analyzer):
        """단일 감정 분석 테스트"""
        text = "급해요! 오늘까지 답변해주세요!"
        result = analyzer.analyze_emotion(text)
        
        assert result is not None
        assert result.primary_emotion in EmotionType
        assert 0.0 <= result.confidence <= 1.0
        assert 0.0 <= result.intensity <= 1.0
        assert result.reasoning is not None
        assert len(result.emotion_scores) > 0
    
    def test_analyze_intent_single(self, analyzer):
        """단일 의도 분석 테스트"""
        text = "손해배상 청구 방법을 알려주세요"
        result = analyzer.analyze_intent(text)
        
        assert result is not None
        assert result.primary_intent in IntentType
        assert 0.0 <= result.confidence <= 1.0
        assert result.urgency_level in UrgencyLevel
        assert result.reasoning is not None
        assert len(result.intent_scores) > 0
    
    def test_analyze_emotion_batch(self, analyzer, test_texts):
        """배치 감정 분석 테스트"""
        results = analyzer.analyze_emotion_batch(test_texts, batch_size=4)
        
        assert len(results) == len(test_texts)
        for result in results:
            assert result is not None
            assert result.primary_emotion in EmotionType
            assert 0.0 <= result.confidence <= 1.0
    
    def test_analyze_intent_batch(self, analyzer, test_texts):
        """배치 의도 분석 테스트"""
        results = analyzer.analyze_intent_batch(test_texts, batch_size=4)
        
        assert len(results) == len(test_texts)
        for result in results:
            assert result is not None
            assert result.primary_intent in IntentType
            assert 0.0 <= result.confidence <= 1.0
            assert result.urgency_level in UrgencyLevel
    
    def test_batch_performance(self, analyzer, test_texts):
        """배치 처리 성능 테스트"""
        # 단일 처리 시간 측정
        start_time = time.time()
        single_results = [analyzer.analyze_emotion(text) for text in test_texts]
        single_time = time.time() - start_time
        
        # 배치 처리 시간 측정
        start_time = time.time()
        batch_results = analyzer.analyze_emotion_batch(test_texts, batch_size=4)
        batch_time = time.time() - start_time
        
        print(f"\n단일 처리 시간: {single_time:.4f}초")
        print(f"배치 처리 시간: {batch_time:.4f}초")
        print(f"성능 개선: {single_time/batch_time:.2f}배")
        
        # 배치 처리가 더 빠르거나 비슷해야 함 (ML 모델 사용 시)
        # 패턴 기반만 사용 시에는 비슷한 성능
        assert len(single_results) == len(batch_results)
    
    def test_pattern_only_mode(self, analyzer_pattern_only):
        """패턴 기반만 사용 모드 테스트"""
        text = "급해요! 오늘까지 답변해주세요!"
        result = analyzer_pattern_only.analyze_emotion(text)
        
        assert result is not None
        assert result.primary_emotion in EmotionType
        # 패턴 기반만 사용하므로 ML 모델 관련 추론이 없어야 함
        assert "ML 모델" not in result.reasoning or "패턴" in result.reasoning
    
    def test_hybrid_mode(self, analyzer):
        """하이브리드 모드 테스트"""
        # 신뢰도가 낮은 텍스트 (ML 모델 사용될 가능성 높음)
        text = "이해가 안 돼요. 더 자세히 설명해주세요"
        result = analyzer.analyze_emotion(text)
        
        assert result is not None
        assert result.primary_emotion in EmotionType
        
        # 하이브리드 모드에서는 패턴 또는 ML 결과가 있을 수 있음
        assert result.reasoning is not None
    
    def test_urgency_assessment(self, analyzer):
        """긴급도 평가 테스트"""
        urgent_text = "급해요! 오늘까지 답변해주세요!"
        normal_text = "손해배상 청구 방법을 알려주세요"
        
        urgent_result = analyzer.analyze_intent(urgent_text)
        normal_result = analyzer.analyze_intent(normal_text)
        
        assert urgent_result.urgency_level in [UrgencyLevel.HIGH, UrgencyLevel.CRITICAL]
        assert normal_result.urgency_level in [UrgencyLevel.LOW, UrgencyLevel.MEDIUM]
    
    def test_response_tone(self, analyzer):
        """응답 톤 결정 테스트"""
        text = "이해가 안 돼요. 더 자세히 설명해주세요"
        emotion = analyzer.analyze_emotion(text)
        intent = analyzer.analyze_intent(text)
        
        tone = analyzer.get_contextual_response_tone(emotion, intent)
        
        assert tone is not None
        assert tone.tone_type in ["empathetic", "professional", "supportive", "urgent", "casual"]
        assert 0.0 <= tone.empathy_level <= 1.0
        assert 0.0 <= tone.formality_level <= 1.0
        assert isinstance(tone.urgency_response, bool)
        assert tone.explanation_depth in ["simple", "medium", "detailed"]
    
    def test_empty_text(self, analyzer):
        """빈 텍스트 처리 테스트"""
        result = analyzer.analyze_emotion("")
        assert result is not None
        assert result.primary_emotion == EmotionType.NEUTRAL
    
    def test_batch_empty_list(self, analyzer):
        """빈 리스트 배치 처리 테스트"""
        results = analyzer.analyze_emotion_batch([])
        assert results == []
    
    def test_model_cache_integration(self, analyzer):
        """ModelCacheManager 통합 테스트"""
        # 첫 번째 로드
        text1 = "급해요! 오늘까지 답변해주세요!"
        result1 = analyzer.analyze_emotion(text1)
        
        # 두 번째 로드 (캐시 사용)
        text2 = "감사합니다. 정말 도움이 되었어요"
        result2 = analyzer.analyze_emotion(text2)
        
        # 두 결과 모두 정상적으로 반환되어야 함
        assert result1 is not None
        assert result2 is not None


def test_standalone():
    """독립 실행 테스트"""
    print("\n" + "="*60)
    print("EmotionIntentAnalyzer 테스트")
    print("="*60)
    
    analyzer = EmotionIntentAnalyzer(use_ml_model=True)
    
    test_texts = [
        "손해배상 청구 방법을 알려주세요",
        "급해요! 오늘까지 답변해주세요!",
        "감사합니다. 정말 도움이 되었어요",
        "이해가 안 돼요. 더 자세히 설명해주세요",
        "왜 이런 문제가 생겼나요? 정말 화나네요"
    ]
    
    print("\n[단일 분석 테스트]")
    for i, text in enumerate(test_texts[:3], 1):
        print(f"\n{i}. 텍스트: {text}")
        
        emotion = analyzer.analyze_emotion(text)
        intent = analyzer.analyze_intent(text)
        
        print(f"   감정: {emotion.primary_emotion.value} (신뢰도: {emotion.confidence:.2f})")
        print(f"   의도: {intent.primary_intent.value} (신뢰도: {intent.confidence:.2f})")
        print(f"   긴급도: {intent.urgency_level.value}")
        print(f"   추론: {emotion.reasoning[:50]}...")
    
    print("\n[배치 분석 테스트]")
    start_time = time.time()
    emotion_results = analyzer.analyze_emotion_batch(test_texts, batch_size=4)
    batch_time = time.time() - start_time
    
    print(f"배치 처리 시간: {batch_time:.4f}초")
    print(f"처리된 텍스트 수: {len(emotion_results)}")
    
    for i, (text, result) in enumerate(zip(test_texts, emotion_results), 1):
        print(f"{i}. {text[:30]}... → {result.primary_emotion.value} ({result.confidence:.2f})")
    
    print("\n[성능 비교]")
    start_time = time.time()
    single_results = [analyzer.analyze_emotion(text) for text in test_texts]
    single_time = time.time() - start_time
    
    print(f"단일 처리: {single_time:.4f}초")
    print(f"배치 처리: {batch_time:.4f}초")
    if batch_time > 0:
        print(f"성능 개선: {single_time/batch_time:.2f}배")
    
    print("\n" + "="*60)
    print("테스트 완료!")
    print("="*60)


if __name__ == "__main__":
    test_standalone()

