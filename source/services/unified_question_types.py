#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
통합된 질문 유형 분류
기존 시스템과 호환되는 통합된 질문 유형 정의
"""

from enum import Enum

class UnifiedQuestionType(Enum):
    """통합된 질문 유형 분류"""
    # 기존 question_classifier.py의 유형들
    PRECEDENT_SEARCH = "precedent_search"
    LAW_INQUIRY = "law_inquiry"
    LEGAL_ADVICE = "legal_advice"
    PROCEDURE_GUIDE = "procedure_guide"
    TERM_EXPLANATION = "term_explanation"
    GENERAL_QUESTION = "general_question"
    
    # answer_structure_enhancer.py의 추가 유형들
    CONTRACT_REVIEW = "contract_review"
    DIVORCE_PROCEDURE = "divorce_procedure"
    INHERITANCE_PROCEDURE = "inheritance_procedure"
    CRIMINAL_CASE = "criminal_case"
    LABOR_DISPUTE = "labor_dispute"

    @classmethod
    def from_string(cls, value: str) -> 'UnifiedQuestionType':
        """문자열에서 UnifiedQuestionType으로 변환"""
        for question_type in cls:
            if question_type.value == value:
                return question_type
        return cls.GENERAL_QUESTION

    def to_domain(self) -> str:
        """질문 유형을 도메인으로 매핑"""
        domain_mapping = {
            UnifiedQuestionType.CONTRACT_REVIEW: "civil_law",
            UnifiedQuestionType.DIVORCE_PROCEDURE: "family_law",
            UnifiedQuestionType.INHERITANCE_PROCEDURE: "family_law",
            UnifiedQuestionType.CRIMINAL_CASE: "criminal_law",
            UnifiedQuestionType.LABOR_DISPUTE: "labor_law",
            UnifiedQuestionType.LAW_INQUIRY: "general",
            UnifiedQuestionType.PRECEDENT_SEARCH: "general",
            UnifiedQuestionType.LEGAL_ADVICE: "general",
            UnifiedQuestionType.PROCEDURE_GUIDE: "general",
            UnifiedQuestionType.TERM_EXPLANATION: "general",
            UnifiedQuestionType.GENERAL_QUESTION: "general"
        }
        return domain_mapping.get(self, "general")

    def get_description(self) -> str:
        """질문 유형 설명 반환"""
        descriptions = {
            UnifiedQuestionType.PRECEDENT_SEARCH: "판례 검색 - 관련 판례나 사건을 찾는 질문",
            UnifiedQuestionType.LAW_INQUIRY: "법률 문의 - 법률 조문이나 법령에 대한 질문",
            UnifiedQuestionType.LEGAL_ADVICE: "법적 조언 - 구체적인 해결방법이나 조언을 요청하는 질문",
            UnifiedQuestionType.PROCEDURE_GUIDE: "절차 안내 - 법적 절차나 신청 방법에 대한 질문",
            UnifiedQuestionType.TERM_EXPLANATION: "용어 해설 - 법률 용어나 개념에 대한 설명 요청",
            UnifiedQuestionType.CONTRACT_REVIEW: "계약서 검토 - 계약서 분석 및 검토 요청",
            UnifiedQuestionType.DIVORCE_PROCEDURE: "이혼 절차 - 이혼 관련 절차 및 방법 문의",
            UnifiedQuestionType.INHERITANCE_PROCEDURE: "상속 절차 - 상속 관련 절차 및 방법 문의",
            UnifiedQuestionType.CRIMINAL_CASE: "형사 사건 - 형사 범죄 관련 문의",
            UnifiedQuestionType.LABOR_DISPUTE: "노동 분쟁 - 노동 관련 분쟁 및 문제 문의",
            UnifiedQuestionType.GENERAL_QUESTION: "일반 질문 - 기타 법률 관련 일반적인 질문"
        }
        return descriptions.get(self, "알 수 없는 질문 유형")
