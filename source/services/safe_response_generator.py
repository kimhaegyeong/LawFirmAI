# -*- coding: utf-8 -*-
"""
Safe Response Generator
안전한 대안 답변 생성 시스템
"""

import logging
from typing import Dict, List, Optional, Any, Tuple
from enum import Enum
from dataclasses import dataclass
from datetime import datetime
import random

from .legal_restriction_system import RestrictionResult, RestrictionLevel, LegalArea
from .content_filter_engine import FilterResult, IntentType, ContextType

logger = logging.getLogger(__name__)


class ResponseType(Enum):
    """답변 유형"""
    GENERAL_INFO = "general_info"           # 일반 정보
    PROCEDURE_GUIDE = "procedure_guide"     # 절차 안내
    STATUTE_REFERENCE = "statute_reference"  # 법령 참조
    PRECEDENT_INFO = "precedent_info"        # 판례 정보
    EXPERT_REFERRAL = "expert_referral"      # 전문가 추천
    DISCLAIMER = "disclaimer"                # 면책 조항


class SafetyLevel(Enum):
    """안전 수준"""
    HIGH = "high"        # 높은 안전성
    MEDIUM = "medium"    # 중간 안전성
    LOW = "low"          # 낮은 안전성


@dataclass
class SafeResponse:
    """안전한 답변"""
    content: str
    response_type: ResponseType
    safety_level: SafetyLevel
    disclaimer: Optional[str]
    expert_referral: Optional[str]
    additional_info: List[str]
    confidence: float


@dataclass
class ResponseTemplate:
    """답변 템플릿"""
    id: str
    response_type: ResponseType
    safety_level: SafetyLevel
    template: str
    placeholders: List[str]
    conditions: List[str]
    disclaimer: Optional[str]
    expert_referral: Optional[str]


class SafeResponseGenerator:
    """안전한 대안 답변 생성 시스템"""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.response_templates = self._initialize_response_templates()
        self.safe_responses = self._initialize_safe_responses()
        self.disclaimers = self._initialize_disclaimers()
        self.expert_referrals = self._initialize_expert_referrals()
        
    def _initialize_response_templates(self) -> List[ResponseTemplate]:
        """답변 템플릿 초기화"""
        return [
            # 일반 정보 제공 템플릿
            ResponseTemplate(
                id="general_info_001",
                response_type=ResponseType.GENERAL_INFO,
                safety_level=SafetyLevel.HIGH,
                template="'{topic}'에 대한 일반적인 정보를 안내드리겠습니다. {general_info}",
                placeholders=["topic", "general_info"],
                conditions=["general_info_request", "low_risk"],
                disclaimer="이 정보는 일반적인 참고용이며, 구체적인 사안은 전문가와 상담하시기 바랍니다.",
                expert_referral="더 정확한 정보가 필요하시면 해당 분야 전문가와 상담하시는 것을 권합니다."
            ),
            
            # 절차 안내 템플릿
            ResponseTemplate(
                id="procedure_guide_001",
                response_type=ResponseType.PROCEDURE_GUIDE,
                safety_level=SafetyLevel.HIGH,
                template="'{procedure}' 절차에 대해 안내드리겠습니다. {steps}",
                placeholders=["procedure", "steps"],
                conditions=["procedure_inquiry", "low_risk"],
                disclaimer="절차는 상황에 따라 달라질 수 있으니 관련 기관에 확인하시기 바랍니다.",
                expert_referral="복잡한 절차의 경우 전문가의 도움을 받으시는 것이 좋습니다."
            ),
            
            # 법령 참조 템플릿
            ResponseTemplate(
                id="statute_reference_001",
                response_type=ResponseType.STATUTE_REFERENCE,
                safety_level=SafetyLevel.MEDIUM,
                template="관련 법령 '{statute}'에 따르면 {content}",
                placeholders=["statute", "content"],
                conditions=["statute_reference", "medium_risk"],
                disclaimer="법령의 해석과 적용은 구체적인 사안에 따라 달라질 수 있습니다.",
                expert_referral="법령의 정확한 해석이 필요하시면 법무 전문가와 상담하시기 바랍니다."
            ),
            
            # 판례 정보 템플릿
            ResponseTemplate(
                id="precedent_info_001",
                response_type=ResponseType.PRECEDENT_INFO,
                safety_level=SafetyLevel.MEDIUM,
                template="관련 판례 '{precedent}'에서는 {content}",
                placeholders=["precedent", "content"],
                conditions=["precedent_search", "medium_risk"],
                disclaimer="판례는 구체적인 사안과 다를 수 있으니 참고용으로만 활용하시기 바랍니다.",
                expert_referral="유사한 사안의 경우 변호사와 상담하시는 것이 좋습니다."
            ),
            
            # 전문가 추천 템플릿
            ResponseTemplate(
                id="expert_referral_001",
                response_type=ResponseType.EXPERT_REFERRAL,
                safety_level=SafetyLevel.HIGH,
                template="'{area}' 분야의 전문가와 상담하시는 것을 권합니다. {referral_info}",
                placeholders=["area", "referral_info"],
                conditions=["high_risk", "specific_case"],
                disclaimer="구체적인 사안에 대한 정확한 조언은 전문가와 상담을 통해 받으시기 바랍니다.",
                expert_referral=None
            ),
            
            # 면책 조항 템플릿
            ResponseTemplate(
                id="disclaimer_001",
                response_type=ResponseType.DISCLAIMER,
                safety_level=SafetyLevel.HIGH,
                template="죄송합니다. '{topic}'에 대한 구체적인 조언은 제공할 수 없습니다. {alternative}",
                placeholders=["topic", "alternative"],
                conditions=["critical_risk", "legal_advice_request"],
                disclaimer="법률 자문은 변호사와 상담을 통해 받으시기 바랍니다.",
                expert_referral="변호사와 상담하시는 것을 강력히 권합니다."
            )
        ]
    
    def _initialize_safe_responses(self) -> Dict[str, List[str]]:
        """안전한 응답 초기화"""
        return {
            "general_legal_info": [
                "법률은 사회 질서를 유지하고 시민의 권리를 보호하기 위한 규범입니다.",
                "법령은 국회에서 제정되고 대통령이 공포하는 법률과 시행령, 시행규칙으로 구성됩니다.",
                "법원은 법령에 따라 사건을 심리하고 판결을 내립니다.",
                "변호사는 법률 전문가로서 법률 자문과 소송 대리를 담당합니다."
            ],
            
            "procedure_info": [
                "법적 절차는 일반적으로 신청, 심리, 결정의 단계로 진행됩니다.",
                "소송 절차는 민사소송법과 형사소송법에 따라 규정되어 있습니다.",
                "행정 절차는 행정절차법에 따라 진행됩니다.",
                "각 절차마다 필요한 서류와 기간이 정해져 있습니다."
            ],
            
            "statute_info": [
                "민법은 사법 관계의 기본법입니다.",
                "형법은 범죄와 형벌에 관한 법률입니다.",
                "상법은 상사 관계에 관한 법률입니다.",
                "노동법은 근로자와 사용자 간의 관계를 규정합니다."
            ],
            
            "precedent_info": [
                "대법원 판례는 하급심 법원의 판단 기준이 됩니다.",
                "판례는 법령의 해석과 적용에 대한 법원의 견해를 보여줍니다.",
                "유사한 사건의 판례를 참고할 수 있습니다.",
                "판례는 법률의 보완적 역할을 합니다."
            ],
            
            "expert_referral": [
                "변호사는 법률 전문가로서 정확한 법률 자문을 제공할 수 있습니다.",
                "법무법인에서는 다양한 분야의 전문 변호사를 보유하고 있습니다.",
                "국선변호인 제도는 형사사건에서 변호사가 없는 피고인을 위해 마련되었습니다.",
                "법률구조공단에서는 소송비용 지원을 받을 수 있습니다."
            ]
        }
    
    def _initialize_disclaimers(self) -> Dict[str, str]:
        """면책 조항 초기화"""
        return {
            "general": "이 정보는 일반적인 참고용이며, 구체적인 사안은 전문가와 상담하시기 바랍니다.",
            "legal_advice": "법률 자문은 변호사와 상담을 통해 받으시기 바랍니다.",
            "medical_legal": "의료사고 관련 구체적인 판단은 의료분쟁조정중재원이나 전문 의료소송 변호사와 상담하시기 바랍니다.",
            "criminal_case": "형사사건 관련 구체적인 조언은 변호사와 상담하시는 것이 좋습니다.",
            "tax_legal": "세법 관련 구체적인 사안은 세무 전문가와 상담하시기 바랍니다.",
            "contract_legal": "계약 관련 구체적인 사안은 법무 전문가와 상담하시기 바랍니다."
        }
    
    def _initialize_expert_referrals(self) -> Dict[str, str]:
        """전문가 추천 초기화"""
        return {
            "general_legal": "변호사와 상담하시는 것을 권합니다.",
            "medical_legal": "의료분쟁조정중재원이나 전문 의료소송 변호사와 상담하시기 바랍니다.",
            "criminal_legal": "국선변호인 신청이나 변호사와 상담하시는 것이 좋습니다.",
            "tax_legal": "세무 전문가나 국세청에 문의하시기 바랍니다.",
            "contract_legal": "법무 전문가나 변호사와 상담하시는 것을 권합니다.",
            "labor_legal": "노동부나 노동 전문 변호사와 상담하시기 바랍니다."
        }
    
    def generate_safe_response(self, query: str, restriction_result: RestrictionResult, 
                             filter_result: FilterResult) -> SafeResponse:
        """안전한 답변 생성"""
        try:
            self.logger.info(f"Generating safe response for query: {query[:100]}...")
            
            # 상황 분석
            situation = self._analyze_situation(restriction_result, filter_result)
            
            # 적절한 템플릿 선택
            template = self._select_template(situation, query)
            
            # 답변 생성
            response_content = self._generate_response_content(template, query, situation)
            
            # 면책 조항 추가
            disclaimer = self._get_disclaimer(situation)
            
            # 전문가 추천 추가
            expert_referral = self._get_expert_referral(situation)
            
            # 추가 정보 제공
            additional_info = self._get_additional_info(situation)
            
            # 신뢰도 계산
            confidence = self._calculate_confidence(situation, template)
            
            safe_response = SafeResponse(
                content=response_content,
                response_type=template.response_type,
                safety_level=template.safety_level,
                disclaimer=disclaimer,
                expert_referral=expert_referral,
                additional_info=additional_info,
                confidence=confidence
            )
            
            self.logger.info(f"Safe response generated. Type: {template.response_type.value}")
            return safe_response
            
        except Exception as e:
            self.logger.error(f"Error generating safe response: {e}")
            return self._get_fallback_response()
    
    def _analyze_situation(self, restriction_result: RestrictionResult, 
                          filter_result: FilterResult) -> Dict[str, Any]:
        """상황 분석"""
        situation = {
            "restriction_level": restriction_result.restriction_level,
            "restricted_areas": [rule.area for rule in restriction_result.matched_rules],
            "intent_type": filter_result.intent_analysis.intent_type,
            "context_type": filter_result.intent_analysis.context_type,
            "risk_level": filter_result.intent_analysis.risk_level,
            "is_blocked": filter_result.is_blocked
        }
        
        # 위험도 결정
        if restriction_result.restriction_level == RestrictionLevel.CRITICAL:
            situation["safety_level"] = SafetyLevel.LOW
        elif restriction_result.restriction_level == RestrictionLevel.HIGH:
            situation["safety_level"] = SafetyLevel.MEDIUM
        else:
            situation["safety_level"] = SafetyLevel.HIGH
        
        return situation
    
    def _select_template(self, situation: Dict[str, Any], query: str) -> ResponseTemplate:
        """적절한 템플릿 선택"""
        # 위험도가 높으면 면책 조항 템플릿 사용
        if situation["safety_level"] == SafetyLevel.LOW:
            return self._get_template_by_id("disclaimer_001")
        
        # 의도 유형에 따른 템플릿 선택
        intent_type = situation["intent_type"]
        
        if intent_type == IntentType.GENERAL_INFO_REQUEST:
            return self._get_template_by_id("general_info_001")
        elif intent_type == IntentType.PROCEDURE_INQUIRY:
            return self._get_template_by_id("procedure_guide_001")
        elif intent_type == IntentType.STATUTE_REFERENCE:
            return self._get_template_by_id("statute_reference_001")
        elif intent_type == IntentType.PRECEDENT_SEARCH:
            return self._get_template_by_id("precedent_info_001")
        elif intent_type in [IntentType.LEGAL_ADVICE_REQUEST, IntentType.CASE_SPECIFIC_QUESTION]:
            return self._get_template_by_id("expert_referral_001")
        else:
            return self._get_template_by_id("general_info_001")
    
    def _get_template_by_id(self, template_id: str) -> ResponseTemplate:
        """ID로 템플릿 가져오기"""
        for template in self.response_templates:
            if template.id == template_id:
                return template
        
        # 기본 템플릿 반환
        return self.response_templates[0]
    
    def _generate_response_content(self, template: ResponseTemplate, query: str, 
                                 situation: Dict[str, Any]) -> str:
        """답변 내용 생성"""
        try:
            # 플레이스홀더 값 생성
            placeholder_values = {}
            
            for placeholder in template.placeholders:
                if placeholder == "topic":
                    placeholder_values[placeholder] = self._extract_topic(query)
                elif placeholder == "general_info":
                    placeholder_values[placeholder] = self._get_general_info(situation)
                elif placeholder == "procedure":
                    placeholder_values[placeholder] = self._extract_procedure(query)
                elif placeholder == "steps":
                    placeholder_values[placeholder] = self._get_procedure_steps(situation)
                elif placeholder == "statute":
                    placeholder_values[placeholder] = self._extract_statute(query)
                elif placeholder == "content":
                    placeholder_values[placeholder] = self._get_statute_content(situation)
                elif placeholder == "precedent":
                    placeholder_values[placeholder] = self._extract_precedent(query)
                elif placeholder == "area":
                    placeholder_values[placeholder] = self._get_expert_area(situation)
                elif placeholder == "referral_info":
                    placeholder_values[placeholder] = self._get_referral_info(situation)
                elif placeholder == "alternative":
                    placeholder_values[placeholder] = self._get_alternative_info(situation)
            
            # 템플릿에 값 적용
            content = template.template
            for placeholder, value in placeholder_values.items():
                content = content.replace(f"{{{placeholder}}}", value)
            
            return content
            
        except Exception as e:
            self.logger.error(f"Error generating response content: {e}")
            return "일반적인 법률 정보를 안내드릴 수 있습니다."
    
    def _extract_topic(self, query: str) -> str:
        """주제 추출"""
        # 간단한 키워드 추출
        legal_keywords = ["계약", "소송", "법령", "판례", "절차", "권리", "의무"]
        
        for keyword in legal_keywords:
            if keyword in query:
                return keyword
        
        return "법률"
    
    def _get_general_info(self, situation: Dict[str, Any]) -> str:
        """일반 정보 가져오기"""
        general_infos = self.safe_responses["general_legal_info"]
        return random.choice(general_infos)
    
    def _extract_procedure(self, query: str) -> str:
        """절차 추출"""
        procedure_keywords = ["신청", "제출", "처리", "심리", "결정"]
        
        for keyword in procedure_keywords:
            if keyword in query:
                return f"{keyword} 절차"
        
        return "법적 절차"
    
    def _get_procedure_steps(self, situation: Dict[str, Any]) -> str:
        """절차 단계 가져오기"""
        procedure_infos = self.safe_responses["procedure_info"]
        return random.choice(procedure_infos)
    
    def _extract_statute(self, query: str) -> str:
        """법령 추출"""
        statute_keywords = ["민법", "형법", "상법", "노동법", "행정법"]
        
        for keyword in statute_keywords:
            if keyword in query:
                return keyword
        
        return "관련 법령"
    
    def _get_statute_content(self, situation: Dict[str, Any]) -> str:
        """법령 내용 가져오기"""
        statute_infos = self.safe_responses["statute_info"]
        return random.choice(statute_infos)
    
    def _extract_precedent(self, query: str) -> str:
        """판례 추출"""
        return "관련 판례"
    
    def _get_expert_area(self, situation: Dict[str, Any]) -> str:
        """전문가 분야 가져오기"""
        if LegalArea.MEDICAL_LEGAL in situation["restricted_areas"]:
            return "의료법"
        elif LegalArea.CRIMINAL_CASE in situation["restricted_areas"]:
            return "형사법"
        elif LegalArea.TAX_EVASION in situation["restricted_areas"]:
            return "세법"
        else:
            return "법률"
    
    def _get_referral_info(self, situation: Dict[str, Any]) -> str:
        """추천 정보 가져오기"""
        expert_infos = self.safe_responses["expert_referral"]
        return random.choice(expert_infos)
    
    def _get_alternative_info(self, situation: Dict[str, Any]) -> str:
        """대안 정보 가져오기"""
        return "일반적인 법률 정보나 절차는 안내드릴 수 있습니다."
    
    def _get_disclaimer(self, situation: Dict[str, Any]) -> str:
        """면책 조항 가져오기"""
        if LegalArea.MEDICAL_LEGAL in situation["restricted_areas"]:
            return self.disclaimers["medical_legal"]
        elif LegalArea.CRIMINAL_CASE in situation["restricted_areas"]:
            return self.disclaimers["criminal_case"]
        elif LegalArea.TAX_EVASION in situation["restricted_areas"]:
            return self.disclaimers["tax_legal"]
        elif LegalArea.CONTRACT_MANIPULATION in situation["restricted_areas"]:
            return self.disclaimers["contract_legal"]
        else:
            return self.disclaimers["general"]
    
    def _get_expert_referral(self, situation: Dict[str, Any]) -> str:
        """전문가 추천 가져오기"""
        if LegalArea.MEDICAL_LEGAL in situation["restricted_areas"]:
            return self.expert_referrals["medical_legal"]
        elif LegalArea.CRIMINAL_CASE in situation["restricted_areas"]:
            return self.expert_referrals["criminal_legal"]
        elif LegalArea.TAX_EVASION in situation["restricted_areas"]:
            return self.expert_referrals["tax_legal"]
        elif LegalArea.CONTRACT_MANIPULATION in situation["restricted_areas"]:
            return self.expert_referrals["contract_legal"]
        else:
            return self.expert_referrals["general_legal"]
    
    def _get_additional_info(self, situation: Dict[str, Any]) -> List[str]:
        """추가 정보 가져오기"""
        additional_info = []
        
        # 일반적인 법률 정보 추가
        if situation["safety_level"] == SafetyLevel.HIGH:
            additional_info.append("관련 법령이나 판례를 참고하실 수 있습니다.")
            additional_info.append("법원이나 관련 기관에 문의하시면 더 자세한 정보를 얻을 수 있습니다.")
        
        return additional_info
    
    def _calculate_confidence(self, situation: Dict[str, Any], template: ResponseTemplate) -> float:
        """신뢰도 계산"""
        base_confidence = 0.8
        
        # 안전 수준에 따른 조정
        if template.safety_level == SafetyLevel.HIGH:
            base_confidence += 0.1
        elif template.safety_level == SafetyLevel.MEDIUM:
            base_confidence += 0.0
        else:
            base_confidence -= 0.2
        
        # 위험도에 따른 조정
        if situation["risk_level"] == "low":
            base_confidence += 0.1
        elif situation["risk_level"] == "high":
            base_confidence -= 0.1
        elif situation["risk_level"] == "critical":
            base_confidence -= 0.2
        
        return max(0.0, min(1.0, base_confidence))
    
    def _get_fallback_response(self) -> SafeResponse:
        """폴백 응답"""
        return SafeResponse(
            content="죄송합니다. 구체적인 법률 자문은 변호사와 상담하시는 것이 좋습니다.",
            response_type=ResponseType.DISCLAIMER,
            safety_level=SafetyLevel.HIGH,
            disclaimer="법률 자문은 변호사와 상담을 통해 받으시기 바랍니다.",
            expert_referral="변호사와 상담하시는 것을 강력히 권합니다.",
            additional_info=["일반적인 법률 정보는 안내드릴 수 있습니다."],
            confidence=0.9
        )
    
    def get_safe_response_statistics(self) -> Dict[str, Any]:
        """안전한 답변 통계 정보"""
        return {
            "total_templates": len(self.response_templates),
            "response_types": [rt.value for rt in ResponseType],
            "safety_levels": [sl.value for sl in SafetyLevel],
            "total_safe_responses": sum(len(responses) for responses in self.safe_responses.values()),
            "total_disclaimers": len(self.disclaimers),
            "total_expert_referrals": len(self.expert_referrals)
        }
