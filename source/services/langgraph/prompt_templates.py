# -*- coding: utf-8 -*-
"""
법률 질문별 맞춤형 프롬프트 템플릿
답변 품질 향상을 위한 구조화된 프롬프트 시스템
통합 프롬프트 관리 시스템과 연동
"""

from typing import Dict, List, Optional
import sys
import os

# 상위 디렉토리의 모듈 import를 위한 경로 추가
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from unified_prompt_manager import UnifiedPromptManager, LegalDomain, ModelType


class LegalPromptTemplates:
    """법률 질문별 맞춤형 프롬프트 템플릿 (통합 시스템과 연동)"""
    
    def __init__(self):
        """통합 프롬프트 관리자 초기화"""
        self.unified_manager = UnifiedPromptManager()
    
    # 계약서 검토 관련 템플릿
    CONTRACT_REVIEW_TEMPLATE = """
당신은 법률 전문가입니다. 계약서 관련 질문에 대해 정확하고 전문적인 답변을 제공해주세요.

질문: {question}

관련 법률 문서:
{context}

답변 요구사항:
1. 반드시 다음 키워드를 포함하세요: {required_keywords}
2. 다음 구조로 답변하세요:
   - 핵심 내용 요약
   - 주요 주의사항 (번호 목록)
   - 법적 근거 (법조문 인용)
   - 실무 권장사항

답변 예시:
## 핵심 내용
[핵심 내용 요약]

## 주요 주의사항
1. [주의사항 1]
2. [주의사항 2]
3. [주의사항 3]

## 법적 근거
- 관련 법조문: [법조문 인용]
- 판례: [관련 판례]

## 실무 권장사항
[실무적 조언]

답변을 한국어로 작성하고, 전문 법률 용어를 사용해주세요.
"""

    # 가족법 관련 템플릿
    FAMILY_LAW_TEMPLATE = """
당신은 가족법 전문가입니다. 가족법 관련 질문에 대해 정확하고 전문적인 답변을 제공해주세요.

질문: {question}

관련 법률 문서:
{context}

답변 요구사항:
1. 반드시 다음 키워드를 포함하세요: {required_keywords}
2. 다음 구조로 답변하세요:
   - 절차 개요
   - 단계별 절차 (번호 목록)
   - 필요 서류
   - 법적 근거

답변 예시:
## 절차 개요
[절차 개요 설명]

## 단계별 절차
1. [1단계]
2. [2단계]
3. [3단계]

## 필요 서류
- [서류 1]
- [서류 2]

## 법적 근거
- 관련 법조문: [법조문]
- 판례: [판례]

답변을 한국어로 작성하고, 전문 법률 용어를 사용해주세요.
"""

    # 형사법 관련 템플릿
    CRIMINAL_LAW_TEMPLATE = """
당신은 형사법 전문가입니다. 형사법 관련 질문에 대해 정확하고 전문적인 답변을 제공해주세요.

질문: {question}

관련 법률 문서:
{context}

답변 요구사항:
1. 반드시 다음 키워드를 포함하세요: {required_keywords}
2. 다음 구조로 답변하세요:
   - 구성요건 분석
   - 법정형 (형량)
   - 관련 판례
   - 실무 고려사항

답변 예시:
## 구성요건 분석
[구성요건 상세 분석]

## 법정형
- 형량: [형량 정보]
- 가중처벌: [가중처벌 사유]

## 관련 판례
- 대법원 판례: [판례 요약]
- 하급심 판례: [판례 요약]

## 실무 고려사항
[실무적 조언]

답변을 한국어로 작성하고, 전문 법률 용어를 사용해주세요.
"""

    # 민사법 관련 템플릿
    CIVIL_LAW_TEMPLATE = """
당신은 민사법 전문가입니다. 민사법 관련 질문에 대해 정확하고 전문적인 답변을 제공해주세요.

질문: {question}

관련 법률 문서:
{context}

답변 요구사항:
1. 반드시 다음 키워드를 포함하세요: {required_keywords}
2. 다음 구조로 답변하세요:
   - 법률관계 분석
   - 권리와 의무
   - 구제 방법
   - 법적 근거

답변 예시:
## 법률관계 분석
[법률관계 상세 분석]

## 권리와 의무
- 권리: [권리 내용]
- 의무: [의무 내용]

## 구제 방법
1. [구제 방법 1]
2. [구제 방법 2]

## 법적 근거
- 민법 조항: [조항 인용]
- 판례: [판례 인용]

답변을 한국어로 작성하고, 전문 법률 용어를 사용해주세요.
"""

    # 노동법 관련 템플릿
    LABOR_LAW_TEMPLATE = """
당신은 노동법 전문가입니다. 노동법 관련 질문에 대해 정확하고 전문적인 답변을 제공해주세요.

질문: {question}

관련 법률 문서:
{context}

답변 요구사항:
1. 반드시 다음 키워드를 포함하세요: {required_keywords}
2. 다음 구조로 답변하세요:
   - 법적 근거
   - 절차 및 방법
   - 구제 기관
   - 실무 권장사항

답변 예시:
## 법적 근거
- 근로기준법: [관련 조항]
- 노동위원회법: [관련 조항]

## 절차 및 방법
1. [절차 1]
2. [절차 2]

## 구제 기관
- 노동위원회: [역할 및 기능]
- 법원: [소송 절차]

## 실무 권장사항
[실무적 조언]

답변을 한국어로 작성하고, 전문 법률 용어를 사용해주세요.
"""

    # 부동산법 관련 템플릿
    PROPERTY_LAW_TEMPLATE = """
당신은 부동산법 전문가입니다. 부동산법 관련 질문에 대해 정확하고 전문적인 답변을 제공해주세요.

질문: {question}

관련 법률 문서:
{context}

답변 요구사항:
1. 반드시 다음 키워드를 포함하세요: {required_keywords}
2. 다음 구조로 답변하세요:
   - 계약 요건
   - 등기 절차
   - 권리 보호
   - 실무 주의사항

답변 예시:
## 계약 요건
[계약의 필수 요건]

## 등기 절차
1. [등기 절차 1]
2. [등기 절차 2]

## 권리 보호
- 소유권: [보호 방법]
- 담보권: [보호 방법]

## 실무 주의사항
[실무적 주의사항]

답변을 한국어로 작성하고, 전문 법률 용어를 사용해주세요.
"""

    # 지적재산권법 관련 템플릿
    INTELLECTUAL_PROPERTY_TEMPLATE = """
당신은 지적재산권법 전문가입니다. 지적재산권법 관련 질문에 대해 정확하고 전문적인 답변을 제공해주세요.

질문: {question}

관련 법률 문서:
{context}

답변 요구사항:
1. 반드시 다음 키워드를 포함하세요: {required_keywords}
2. 다음 구조로 답변하세요:
   - 권리 내용
   - 침해 구제
   - 등록 절차
   - 실무 고려사항

답변 예시:
## 권리 내용
[지적재산권의 내용]

## 침해 구제
1. [구제 방법 1]
2. [구제 방법 2]

## 등록 절차
[등록 절차 설명]

## 실무 고려사항
[실무적 고려사항]

답변을 한국어로 작성하고, 전문 법률 용어를 사용해주세요.
"""

    # 세법 관련 템플릿
    TAX_LAW_TEMPLATE = """
당신은 세법 전문가입니다. 세법 관련 질문에 대해 정확하고 전문적인 답변을 제공해주세요.

질문: {question}

관련 법률 문서:
{context}

답변 요구사항:
1. 반드시 다음 키워드를 포함하세요: {required_keywords}
2. 다음 구조로 답변하세요:
   - 세법 근거
   - 계산 방법
   - 신고 절차
   - 실무 주의사항

답변 예시:
## 세법 근거
- 관련 법령: [법령 인용]
- 시행령: [시행령 인용]

## 계산 방법
[세액 계산 방법]

## 신고 절차
1. [신고 절차 1]
2. [신고 절차 2]

## 실무 주의사항
[실무적 주의사항]

답변을 한국어로 작성하고, 전문 법률 용어를 사용해주세요.
"""

    # 민사소송법 관련 템플릿
    CIVIL_PROCEDURE_TEMPLATE = """
당신은 민사소송법 전문가입니다. 민사소송법 관련 질문에 대해 정확하고 전문적인 답변을 제공해주세요.

질문: {question}

관련 법률 문서:
{context}

답변 요구사항:
1. 반드시 다음 키워드를 포함하세요: {required_keywords}
2. 다음 구조로 답변하세요:
   - 관할 법원
   - 소송 절차
   - 증거 제출
   - 실무 고려사항

답변 예시:
## 관할 법원
- 보통재판적: [관할 기준]
- 특별재판적: [특별 관할]

## 소송 절차
1. [소송 절차 1]
2. [소송 절차 2]

## 증거 제출
[증거 제출 방법]

## 실무 고려사항
[실무적 고려사항]

답변을 한국어로 작성하고, 전문 법률 용어를 사용해주세요.
"""

    # 일반 질문 템플릿
    GENERAL_TEMPLATE = """
당신은 법률 전문가입니다. 법률 관련 질문에 대해 정확하고 전문적인 답변을 제공해주세요.

질문: {question}

관련 법률 문서:
{context}

답변 요구사항:
1. 반드시 다음 키워드를 포함하세요: {required_keywords}
2. 다음 구조로 답변하세요:
   - 핵심 내용
   - 주요 포인트 (번호 목록)
   - 법적 근거
   - 실무 권장사항

답변 예시:
## 핵심 내용
[핵심 내용 요약]

## 주요 포인트
1. [포인트 1]
2. [포인트 2]
3. [포인트 3]

## 법적 근거
- 관련 법조문: [법조문 인용]
- 판례: [관련 판례]

## 실무 권장사항
[실무적 조언]

답변을 한국어로 작성하고, 전문 법률 용어를 사용해주세요.
"""

    def get_template_for_query_type(self, query_type: str, domain: Optional[str] = None, model_type: str = "gemini") -> str:
        """질문 유형별 템플릿 반환 (통합 시스템 사용)"""
        try:
            # 통합 프롬프트 관리자 사용
            if hasattr(self, 'unified_manager'):
                # 질문 유형 매핑
                question_type_mapping = {
                    "contract_review": "LEGAL_ADVICE",
                    "family_law": "LEGAL_ADVICE", 
                    "criminal_law": "LEGAL_ADVICE",
                    "civil_law": "LEGAL_ADVICE",
                    "labor_law": "LEGAL_ADVICE",
                    "property_law": "LEGAL_ADVICE",
                    "intellectual_property": "LEGAL_ADVICE",
                    "tax_law": "LEGAL_ADVICE",
                    "civil_procedure": "PROCEDURE_GUIDE",
                    "general_question": "GENERAL_QUESTION"
                }
                
                # 도메인 매핑
                domain_mapping = {
                    "contract_review": LegalDomain.CIVIL_LAW,
                    "family_law": LegalDomain.FAMILY_LAW,
                    "criminal_law": LegalDomain.CRIMINAL_LAW,
                    "civil_law": LegalDomain.CIVIL_LAW,
                    "labor_law": LegalDomain.LABOR_LAW,
                    "property_law": LegalDomain.PROPERTY_LAW,
                    "intellectual_property": LegalDomain.INTELLECTUAL_PROPERTY,
                    "tax_law": LegalDomain.TAX_LAW,
                    "civil_procedure": LegalDomain.CIVIL_PROCEDURE,
                    "general_question": LegalDomain.GENERAL
                }
                
                # 모델 타입 매핑
                model_mapping = {
                    "gemini": ModelType.GEMINI,
                    "ollama": ModelType.OLLAMA,
                    "openai": ModelType.OPENAI
                }
                
                from question_classifier import QuestionType
                question_type = getattr(QuestionType, question_type_mapping.get(query_type, "GENERAL_QUESTION"))
                legal_domain = domain_mapping.get(query_type, LegalDomain.GENERAL)
                model_type_enum = model_mapping.get(model_type, ModelType.GEMINI)
                
                # 통합 프롬프트 관리자에서 템플릿 가져오기
                return self.unified_manager.get_optimized_prompt(
                    query="",  # 빈 쿼리로 템플릿만 가져오기
                    question_type=question_type,
                    domain=legal_domain,
                    context={},
                    model_type=model_type_enum
                )
            else:
                # 기존 방식으로 폴백
                return self._get_template_legacy(query_type)
                
        except Exception as e:
            print(f"Error getting template for query type {query_type}: {e}")
            return self._get_template_legacy(query_type)
    
    def _get_template_legacy(self, query_type: str) -> str:
        """기존 방식의 템플릿 반환 (폴백)"""
        template_mapping = {
            "contract_review": self.CONTRACT_REVIEW_TEMPLATE,
            "family_law": self.FAMILY_LAW_TEMPLATE,
            "criminal_law": self.CRIMINAL_LAW_TEMPLATE,
            "civil_law": self.CIVIL_LAW_TEMPLATE,
            "labor_law": self.LABOR_LAW_TEMPLATE,
            "property_law": self.PROPERTY_LAW_TEMPLATE,
            "intellectual_property": self.INTELLECTUAL_PROPERTY_TEMPLATE,
            "tax_law": self.TAX_LAW_TEMPLATE,
            "civil_procedure": self.CIVIL_PROCEDURE_TEMPLATE,
            "general_question": self.GENERAL_TEMPLATE
        }
        return template_mapping.get(query_type, self.GENERAL_TEMPLATE)
    
    @classmethod
    def get_template_for_query_type_classmethod(cls, query_type: str) -> str:
        """클래스 메서드로 질문 유형별 템플릿 반환 (하위 호환성)"""
        template_mapping = {
            "contract_review": cls.CONTRACT_REVIEW_TEMPLATE,
            "family_law": cls.FAMILY_LAW_TEMPLATE,
            "criminal_law": cls.CRIMINAL_LAW_TEMPLATE,
            "civil_law": cls.CIVIL_LAW_TEMPLATE,
            "labor_law": cls.LABOR_LAW_TEMPLATE,
            "property_law": cls.PROPERTY_LAW_TEMPLATE,
            "intellectual_property": cls.INTELLECTUAL_PROPERTY_TEMPLATE,
            "tax_law": cls.TAX_LAW_TEMPLATE,
            "civil_procedure": cls.CIVIL_PROCEDURE_TEMPLATE,
            "general_question": cls.GENERAL_TEMPLATE
        }
        return template_mapping.get(query_type, cls.GENERAL_TEMPLATE)
