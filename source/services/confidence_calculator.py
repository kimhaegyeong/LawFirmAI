# -*- coding: utf-8 -*-
"""
Confidence Calculator
신뢰도 계산기
"""

import logging
from typing import Dict, Any, Optional, List
from dataclasses import dataclass
from enum import Enum

logger = logging.getLogger(__name__)


class ConfidenceLevel(Enum):
    """신뢰도 레벨"""
    VERY_HIGH = "very_high"
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"
    VERY_LOW = "very_low"


@dataclass
class ConfidenceInfo:
    """신뢰도 정보"""
    confidence: float
    level: ConfidenceLevel
    factors: Dict[str, float]
    explanation: str
    
    @property
    def reliability_level(self) -> str:
        """신뢰도 레벨을 문자열로 반환"""
        return self.level.value


class ConfidenceCalculator:
    """신뢰도 계산기"""
    
    def __init__(self):
        """계산기 초기화"""
        self.logger = logging.getLogger(__name__)
        self.logger.info("ConfidenceCalculator initialized")
        
        # 신뢰도 계산 가중치
        self.weights = {
            'answer_length': 0.2,
            'source_count': 0.3,
            'question_type': 0.2,
            'answer_quality': 0.3
        }
    
    def calculate_confidence(self, 
                           answer: str,
                           sources: List[Dict[str, Any]],
                           question_type: str = "general") -> ConfidenceInfo:
        """
        신뢰도 계산
        
        Args:
            answer: 생성된 답변
            sources: 참조 소스
            question_type: 질문 유형
            
        Returns:
            ConfidenceInfo: 신뢰도 정보
        """
        factors = {}
        
        # 답변 길이 기반 신뢰도
        factors['answer_length'] = self._calculate_length_confidence(answer)
        
        # 소스 수 기반 신뢰도
        factors['source_count'] = self._calculate_source_confidence(sources)
        
        # 질문 유형 기반 신뢰도
        factors['question_type'] = self._calculate_type_confidence(question_type)
        
        # 답변 품질 기반 신뢰도
        factors['answer_quality'] = self._calculate_quality_confidence(answer)
        
        # 전체 신뢰도 계산
        overall_confidence = sum(factors.values()) / len(factors)
        
        # 신뢰도 레벨 결정
        level = self._determine_confidence_level(overall_confidence)
        
        # 설명 생성
        explanation = self._generate_explanation(factors, overall_confidence)
        
        return ConfidenceInfo(
            confidence=overall_confidence,
            level=level,
            factors=factors,
            explanation=explanation
        )
    
    def _calculate_length_confidence(self, answer: str) -> float:
        """답변 길이 기반 신뢰도"""
        length = len(answer)
        
        if length < 50:
            return 0.3
        elif length < 100:
            return 0.5
        elif length < 200:
            return 0.7
        elif length < 500:
            return 0.9
        else:
            return 1.0
    
    def _calculate_source_confidence(self, sources: List[Dict[str, Any]]) -> float:
        """소스 수 기반 신뢰도"""
        source_count = len(sources)
        
        if source_count == 0:
            return 0.1
        elif source_count == 1:
            return 0.6
        elif source_count <= 3:
            return 0.8
        elif source_count <= 5:
            return 0.9
        else:
            return 0.7  # 너무 많은 소스는 오히려 혼란
    
    def _calculate_type_confidence(self, question_type: str) -> float:
        """질문 유형 기반 신뢰도"""
        type_confidence_map = {
            "precedent_search": 0.9,
            "law_inquiry": 0.8,
            "general_question": 0.7,
            "analysis_request": 0.6,
            "comparison_request": 0.5
        }
        
        return type_confidence_map.get(question_type, 0.6)
    
    def _calculate_quality_confidence(self, answer: str) -> float:
        """답변 품질 기반 신뢰도"""
        # 간단한 품질 지표들
        quality_score = 0.0
        
        # 법률 용어 포함 여부
        legal_terms = ["법", "조항", "판례", "법원", "판결", "소송", "계약", "손해배상"]
        if any(term in answer for term in legal_terms):
            quality_score += 0.3
        
        # 구체적인 설명 포함 여부
        if "따라서" in answer or "결론적으로" in answer or "판단컨대" in answer:
            quality_score += 0.3
        
        # 근거 제시 여부
        if "법률에 따르면" in answer or "판례에 따르면" in answer:
            quality_score += 0.4
        
        return min(quality_score, 1.0)
    
    def _determine_confidence_level(self, confidence_score: float) -> ConfidenceLevel:
        """신뢰도 점수에 따른 레벨 결정"""
        if confidence_score >= 0.9:
            return ConfidenceLevel.VERY_HIGH
        elif confidence_score >= 0.7:
            return ConfidenceLevel.HIGH
        elif confidence_score >= 0.5:
            return ConfidenceLevel.MEDIUM
        elif confidence_score >= 0.3:
            return ConfidenceLevel.LOW
        else:
            return ConfidenceLevel.VERY_LOW
    
    def _generate_explanation(self, factors: Dict[str, float], overall_confidence: float) -> str:
        """신뢰도 설명 생성"""
        explanations = []
        
        if factors.get('answer_length', 0) > 0.7:
            explanations.append("답변 길이가 충분합니다")
        
        if factors.get('source_count', 0) > 0.7:
            explanations.append("충분한 출처가 참조되었습니다")
        
        if factors.get('question_type', 0) > 0.7:
            explanations.append("질문 유형에 적합한 답변입니다")
        
        if factors.get('answer_quality', 0) > 0.7:
            explanations.append("답변 품질이 양호합니다")
        
        if not explanations:
            explanations.append("제공된 정보가 제한적입니다")
        
        return f"신뢰도: {overall_confidence:.2f}. " + ", ".join(explanations)