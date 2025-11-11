# -*- coding: utf-8 -*-
"""답변 포맷터 설정"""

from dataclasses import dataclass
from typing import Dict, Tuple


@dataclass
class AnswerLengthConfig:
    """답변 길이 설정"""
    simple_question: Tuple[int, int] = (500, 3000)
    term_explanation: Tuple[int, int] = (800, 3500)
    legal_analysis: Tuple[int, int] = (1500, 5000)
    complex_question: Tuple[int, int] = (2000, 8000)
    default: Tuple[int, int] = (800, 4000)
    
    def get_targets(self, query_type: str) -> Tuple[int, int]:
        """질의 유형에 따른 목표 길이 반환"""
        return getattr(self, query_type, self.default)


@dataclass
class ConfidenceConfig:
    """신뢰도 설정"""
    
    def __post_init__(self):
        if not hasattr(self, 'min_confidence_by_type'):
            self.min_confidence_by_type = {
                "simple_question": 0.75,
                "term_explanation": 0.80,
                "legal_analysis": 0.75,
                "complex_question": 0.70,
                "general_question": 0.70
            }
        
        if not hasattr(self, 'confidence_boost_factors'):
            self.confidence_boost_factors = {
                "source_count_5+": 0.08,
                "source_count_3+": 0.05,
                "source_count_1+": 0.02,
                "answer_length_500+": 0.05,
                "answer_length_200+": 0.03,
                "answer_length_100+": 0.01,
                "citation_count_3+": 0.10,
                "citation_count_2+": 0.08,
                "citation_count_1+": 0.03
            }
    
    def get_min_confidence(self, query_type: str) -> float:
        """질의 유형에 따른 최소 신뢰도 반환"""
        return self.min_confidence_by_type.get(query_type, 0.70)

