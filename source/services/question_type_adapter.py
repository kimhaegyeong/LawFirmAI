#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
질문 유형 호환성 어댑터
기존 시스템과 새로운 하이브리드 분류기 간의 호환성 제공
"""

from typing import Union, Optional
from .question_classifier import QuestionType as OldQuestionType
from .answer_structure_enhancer import QuestionType as NewQuestionType
from .unified_question_types import UnifiedQuestionType

class QuestionTypeAdapter:
    """질문 유형 호환성 어댑터"""
    
    @staticmethod
    def convert_to_unified(old_type: Union[OldQuestionType, NewQuestionType, str]) -> UnifiedQuestionType:
        """기존 질문 유형을 통합 유형으로 변환"""
        # 문자열인 경우 직접 변환
        if isinstance(old_type, str):
            return UnifiedQuestionType.from_string(old_type)
        
        # 기존 QuestionType 매핑
        old_mapping = {
            OldQuestionType.PRECEDENT_SEARCH: UnifiedQuestionType.PRECEDENT_SEARCH,
            OldQuestionType.LAW_INQUIRY: UnifiedQuestionType.LAW_INQUIRY,
            OldQuestionType.LEGAL_ADVICE: UnifiedQuestionType.LEGAL_ADVICE,
            OldQuestionType.PROCEDURE_GUIDE: UnifiedQuestionType.PROCEDURE_GUIDE,
            OldQuestionType.TERM_EXPLANATION: UnifiedQuestionType.TERM_EXPLANATION,
            OldQuestionType.GENERAL_QUESTION: UnifiedQuestionType.GENERAL_QUESTION,
        }
        
        # 새로운 QuestionType 매핑
        new_mapping = {
            NewQuestionType.CONTRACT_REVIEW: UnifiedQuestionType.CONTRACT_REVIEW,
            NewQuestionType.DIVORCE_PROCEDURE: UnifiedQuestionType.DIVORCE_PROCEDURE,
            NewQuestionType.INHERITANCE_PROCEDURE: UnifiedQuestionType.INHERITANCE_PROCEDURE,
            NewQuestionType.CRIMINAL_CASE: UnifiedQuestionType.CRIMINAL_CASE,
            NewQuestionType.LABOR_DISPUTE: UnifiedQuestionType.LABOR_DISPUTE,
            NewQuestionType.PRECEDENT_SEARCH: UnifiedQuestionType.PRECEDENT_SEARCH,
            NewQuestionType.LAW_INQUIRY: UnifiedQuestionType.LAW_INQUIRY,
            NewQuestionType.LEGAL_ADVICE: UnifiedQuestionType.LEGAL_ADVICE,
            NewQuestionType.PROCEDURE_GUIDE: UnifiedQuestionType.PROCEDURE_GUIDE,
            NewQuestionType.TERM_EXPLANATION: UnifiedQuestionType.TERM_EXPLANATION,
            NewQuestionType.GENERAL_QUESTION: UnifiedQuestionType.GENERAL_QUESTION,
        }
        
        # 매핑에서 찾기
        if old_type in old_mapping:
            return old_mapping[old_type]
        elif old_type in new_mapping:
            return new_mapping[old_type]
        
        # 기본값 반환
        return UnifiedQuestionType.GENERAL_QUESTION
    
    @staticmethod
    def convert_from_unified(unified_type: UnifiedQuestionType) -> str:
        """통합 유형을 문자열로 변환 (기존 시스템 호환)"""
        return unified_type.value
    
    @staticmethod
    def get_law_weight(unified_type: UnifiedQuestionType) -> float:
        """법률 가중치 계산 (기존 시스템 호환)"""
        law_heavy_types = [
            UnifiedQuestionType.LAW_INQUIRY,
            UnifiedQuestionType.PROCEDURE_GUIDE,
            UnifiedQuestionType.TERM_EXPLANATION
        ]
        return 0.8 if unified_type in law_heavy_types else 0.5
    
    @staticmethod
    def get_precedent_weight(unified_type: UnifiedQuestionType) -> float:
        """판례 가중치 계산 (기존 시스템 호환)"""
        precedent_heavy_types = [
            UnifiedQuestionType.PRECEDENT_SEARCH,
            UnifiedQuestionType.LEGAL_ADVICE,
            UnifiedQuestionType.CRIMINAL_CASE
        ]
        return 0.8 if unified_type in precedent_heavy_types else 0.5
