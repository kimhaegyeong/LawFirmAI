# -*- coding: utf-8 -*-
"""
Intent-Based Processing System
의도별 세분화된 처리 로직
"""

import logging
from typing import Dict, List, Optional, Any
from enum import Enum
from dataclasses import dataclass
from datetime import datetime

from .improved_legal_restriction_system import ImprovedRestrictionResult, ContextType, RestrictionLevel, ContextAnalysis

logger = logging.getLogger(__name__)


class IntentType(Enum):
    """의도 유형"""
    GENERAL_INFO_REQUEST = "general_info_request"       # 일반 정보 요청
    PROCEDURE_INQUIRY = "procedure_inquiry"             # 절차 문의
    STATUTE_REFERENCE = "statute_reference"             # 법령 참조
    PRECEDENT_SEARCH = "precedent_search"               # 판례 검색
    LEGAL_ADVICE_REQUEST = "legal_advice_request"       # 법률 자문 요청
    CASE_SPECIFIC_QUESTION = "case_specific_question"    # 구체적 사건 질문
    SUSPICIOUS_REQUEST = "suspicious_request"           # 의심스러운 요청


class ResponseType(Enum):
    """응답 유형"""
    GENERAL_INFO = "general_info"           # 일반 정보
    PROCEDURE_GUIDE = "procedure_guide"     # 절차 안내
    STATUTE_INFO = "statute_info"           # 법령 정보
    PRECEDENT_INFO = "precedent_info"       # 판례 정보
    SAFE_RESPONSE = "safe_response"         # 안전한 응답
    EXPERT_REFERRAL = "expert_referral"     # 전문가 추천
    BLOCKED_RESPONSE = "blocked_response"   # 차단된 응답


@dataclass
class IntentAnalysis:
    """의도 분석 결과"""
    intent_type: IntentType
    confidence: float
    keywords: List[str]
    patterns: List[str]
    context_type: ContextType
    risk_level: str
    reasoning: str


@dataclass
class ProcessingResult:
    """처리 결과"""
    allowed: bool
    response_type: ResponseType
    confidence: float
    message: str
    safe_alternatives: List[str]
    disclaimer: Optional[str]
    expert_referral: Optional[str]
    reasoning: str


class IntentBasedProcessor:
    """의도별 처리기"""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.intent_patterns = self._initialize_intent_patterns()
        self.response_templates = self._initialize_response_templates()
        
    def _initialize_intent_patterns(self) -> Dict[IntentType, Dict[str, Any]]:
        """의도 패턴 초기화"""
        return {
            IntentType.GENERAL_INFO_REQUEST: {
                "patterns": [
                    r"일반적으로\s*(어떻게|무엇을|해야|어떤)",
                    r"보통\s*(어떻게|무엇을|해야|어떤)",
                    r"일반적인\s*(절차|방법|과정|정보)",
                    r"법령\s*(어떻게|무엇을|해야|어떤)",
                    r"법률\s*(어떻게|무엇을|해야|어떤)",
                    r"관련\s*법\s*(어떻게|무엇을|해야|어떤)",
                    r"법률\s*정보\s*(를|을)\s*(알려주세요|찾아주세요)",
                    r"법률\s*상식\s*(을|를)\s*(알려주세요|찾아주세요)"
                ],
                "keywords": [
                    "일반적으로", "보통", "일반적인", "법령", "법률", "관련 법",
                    "법률 정보", "법률 상식", "기본 정보"
                ],
                "risk_level": "low",
                "threshold": 0.3
            },
            
            IntentType.PROCEDURE_INQUIRY: {
                "patterns": [
                    r"절차\s*(어떻게|무엇을|해야|어떤)",
                    r"신청\s*(어떻게|무엇을|해야|어떤)",
                    r"제출\s*(어떻게|무엇을|해야|어떤)",
                    r"처리\s*(어떻게|무엇을|해야|어떤)",
                    r"어디에\s*(신청|제출|문의|접수)",
                    r"어떤\s*서류\s*(필요|준비|제출)",
                    r"어떤\s*기관\s*(에|에서)\s*(신청|문의)",
                    r"어떤\s*부서\s*(에|에서)\s*(신청|문의)",
                    r"신청\s*방법\s*(을|를)\s*(알려주세요|찾아주세요)",
                    r"제출\s*방법\s*(을|를)\s*(알려주세요|찾아주세요)"
                ],
                "keywords": [
                    "절차", "신청", "제출", "처리", "어디에", "어떤 서류",
                    "어떤 기관", "어떤 부서", "신청 방법", "제출 방법"
                ],
                "risk_level": "low",
                "threshold": 0.3
            },
            
            IntentType.STATUTE_REFERENCE: {
                "patterns": [
                    r"법령\s*(참조|인용|적용|찾기)",
                    r"법조문\s*(참조|인용|적용|찾기)",
                    r"관련\s*법령\s*(참조|인용|적용|찾기)",
                    r"적용\s*법령\s*(참조|인용|적용|찾기)",
                    r"법률\s*(참조|인용|적용|찾기)",
                    r"어떤\s*법령\s*(이|가)\s*(적용|관련)",
                    r"어떤\s*법조문\s*(이|가)\s*(적용|관련)",
                    r"법령\s*(을|를)\s*(찾아주세요|알려주세요)",
                    r"법조문\s*(을|를)\s*(찾아주세요|알려주세요)"
                ],
                "keywords": [
                    "법령", "법조문", "관련 법령", "적용 법령", "법률",
                    "어떤 법령", "어떤 법조문", "참조", "인용", "적용"
                ],
                "risk_level": "low",
                "threshold": 0.3
            },
            
            IntentType.PRECEDENT_SEARCH: {
                "patterns": [
                    r"판례\s*(참조|인용|적용|찾기)",
                    r"대법원\s*(판례|판결)",
                    r"법원\s*(판례|판결)",
                    r"관련\s*판례\s*(참조|인용|적용|찾기)",
                    r"유사\s*사건\s*(판례|판결)",
                    r"어떤\s*판례\s*(이|가)\s*(관련|적용)",
                    r"판례\s*(을|를)\s*(찾아주세요|알려주세요)",
                    r"대법원\s*판례\s*(을|를)\s*(찾아주세요|알려주세요)",
                    r"법원\s*판례\s*(을|를)\s*(찾아주세요|알려주세요)"
                ],
                "keywords": [
                    "판례", "대법원", "법원", "관련 판례", "유사 사건",
                    "어떤 판례", "참조", "인용", "적용"
                ],
                "risk_level": "low",
                "threshold": 0.3
            },
            
            IntentType.LEGAL_ADVICE_REQUEST: {
                "patterns": [
                    r"제\s*경우\s*(어떻게|무엇을|해야|어떤)",
                    r"저는\s*(어떻게|무엇을|해야|어떤)",
                    r"내\s*사건\s*(어떻게|무엇을|해야|어떤)",
                    r"이런\s*상황\s*(어떻게|무엇을|해야|어떤)",
                    r"소송\s*(할까요|해야\s*할까요|하지\s*않을까요)",
                    r"변호사\s*(고용|선임|상담)\s*(해야|할까요)",
                    r"법적\s*조치\s*(어떻게|무엇을|해야)",
                    r"권리\s*(주장|행사)\s*(어떻게|무엇을|해야)",
                    r"어떻게\s*해야\s*(하나요|할까요)",
                    r"무엇을\s*해야\s*(하나요|할까요)"
                ],
                "keywords": [
                    "제 경우", "저는", "내 사건", "이런 상황", "소송",
                    "변호사", "법적 조치", "권리 주장", "권리 행사",
                    "어떻게 해야", "무엇을 해야"
                ],
                "risk_level": "high",
                "threshold": 0.5
            },
            
            IntentType.CASE_SPECIFIC_QUESTION: {
                "patterns": [
                    r"구체적으로\s*(어떻게|무엇을|해야|어떤)",
                    r"실제\s*사건\s*(어떻게|무엇을|해야|어떤)",
                    r"이\s*사건\s*(어떻게|무엇을|해야|어떤)",
                    r"현재\s*상황\s*(어떻게|무엇을|해야|어떤)",
                    r"진행\s*중인\s*(사건|소송|분쟁)",
                    r"당사자\s*(어떻게|무엇을|해야|어떤)",
                    r"구체적\s*사안\s*(어떻게|무엇을|해야|어떤)",
                    r"실제\s*상황\s*(어떻게|무엇을|해야|어떤)"
                ],
                "keywords": [
                    "구체적으로", "실제 사건", "이 사건", "현재 상황",
                    "진행 중인", "당사자", "구체적 사안", "실제 상황"
                ],
                "risk_level": "high",
                "threshold": 0.5
            },
            
            IntentType.SUSPICIOUS_REQUEST: {
                "patterns": [
                    r"탈법\s*(방법|수단|기법)",
                    r"법망\s*(빠져나가기|회피)",
                    r"법적\s*구멍\s*(이용|활용)",
                    r"우회\s*(방법|수단|기법)",
                    r"회피\s*(방법|수단|기법)",
                    r"조작\s*(방법|수단|기법)",
                    r"위조\s*(방법|수단|기법)",
                    r"세금\s*회피\s*(방법|수단)",
                    r"위장\s*(방법|수단|기법)",
                    r"가짜\s*(방법|수단|기법)"
                ],
                "keywords": [
                    "탈법", "법망", "법적 구멍", "우회", "회피", "조작", "위조",
                    "세금 회피", "위장", "가짜", "허위", "거짓"
                ],
                "risk_level": "critical",
                "threshold": 0.3
            }
        }
    
    def _initialize_response_templates(self) -> Dict[ResponseType, Dict[str, Any]]:
        """응답 템플릿 초기화"""
        return {
            ResponseType.GENERAL_INFO: {
                "message": "일반적인 법률 정보를 제공할 수 있습니다.",
                "confidence": 0.9,
                "safe_alternatives": [
                    "관련 법령이나 판례를 참고하실 수 있습니다.",
                    "법원이나 관련 기관에 문의하시면 더 자세한 정보를 얻을 수 있습니다."
                ],
                "disclaimer": "이 정보는 일반적인 참고용이며, 구체적인 사안은 전문가와 상담하시기 바랍니다.",
                "expert_referral": None
            },
            
            ResponseType.PROCEDURE_GUIDE: {
                "message": "법적 절차에 대해 안내드릴 수 있습니다.",
                "confidence": 0.8,
                "safe_alternatives": [
                    "관련 기관에 직접 문의하시면 정확한 절차를 안내받을 수 있습니다.",
                    "필요한 서류나 조건은 상황에 따라 달라질 수 있습니다."
                ],
                "disclaimer": "절차는 상황에 따라 달라질 수 있으니 관련 기관에 확인하시기 바랍니다.",
                "expert_referral": "복잡한 절차의 경우 전문가의 도움을 받으시는 것이 좋습니다."
            },
            
            ResponseType.STATUTE_INFO: {
                "message": "관련 법령 정보를 제공할 수 있습니다.",
                "confidence": 0.8,
                "safe_alternatives": [
                    "법령의 해석과 적용은 구체적인 사안에 따라 달라질 수 있습니다.",
                    "최신 법령 정보는 법제처 홈페이지에서 확인하실 수 있습니다."
                ],
                "disclaimer": "법령의 해석과 적용은 구체적인 사안에 따라 달라질 수 있습니다.",
                "expert_referral": "법령의 정확한 해석이 필요하시면 법무 전문가와 상담하시기 바랍니다."
            },
            
            ResponseType.PRECEDENT_INFO: {
                "message": "관련 판례 정보를 제공할 수 있습니다.",
                "confidence": 0.8,
                "safe_alternatives": [
                    "판례는 구체적인 사안과 다를 수 있으니 참고용으로만 활용하시기 바랍니다.",
                    "최신 판례는 법원 홈페이지에서 확인하실 수 있습니다."
                ],
                "disclaimer": "판례는 구체적인 사안과 다를 수 있으니 참고용으로만 활용하시기 바랍니다.",
                "expert_referral": "유사한 사안의 경우 변호사와 상담하시는 것이 좋습니다."
            },
            
            ResponseType.SAFE_RESPONSE: {
                "message": "구체적인 법률 자문은 변호사와 상담하시는 것이 좋습니다.",
                "confidence": 0.9,
                "safe_alternatives": [
                    "일반적인 법률 정보나 절차는 안내드릴 수 있습니다.",
                    "관련 법령이나 판례를 참고하실 수 있습니다."
                ],
                "disclaimer": "법률 자문은 변호사와 상담을 통해 받으시기 바랍니다.",
                "expert_referral": "변호사와 상담하시는 것을 강력히 권합니다."
            },
            
            ResponseType.EXPERT_REFERRAL: {
                "message": "해당 분야의 전문가와 상담하시는 것을 권합니다.",
                "confidence": 0.9,
                "safe_alternatives": [
                    "일반적인 법률 정보나 절차는 안내드릴 수 있습니다.",
                    "관련 기관에 문의하시면 도움이 될 것입니다."
                ],
                "disclaimer": "구체적인 사안에 대한 정확한 조언은 전문가와 상담을 통해 받으시기 바랍니다.",
                "expert_referral": "전문가와 상담하시는 것을 강력히 권합니다."
            },
            
            ResponseType.BLOCKED_RESPONSE: {
                "message": "법적으로 부적절한 요청이 감지되었습니다.",
                "confidence": 1.0,
                "safe_alternatives": [
                    "합법적인 방법으로 도움을 받으시기 바랍니다.",
                    "일반적인 법률 정보나 절차는 안내드릴 수 있습니다."
                ],
                "disclaimer": "합법적인 방법으로 법률 서비스를 이용하시기 바랍니다.",
                "expert_referral": "법률 상담을 통해 올바른 방법을 안내받으시기 바랍니다."
            }
        }
    
    def analyze_intent(self, query: str) -> IntentAnalysis:
        """의도 분석"""
        try:
            query_lower = query.lower()
            intent_scores = {}
            
            # 각 의도 유형별 점수 계산
            for intent_type, config in self.intent_patterns.items():
                score = 0.0
                matched_keywords = []
                matched_patterns = []
                
                # 키워드 매칭
                for keyword in config["keywords"]:
                    if keyword.lower() in query_lower:
                        matched_keywords.append(keyword)
                        score += 0.3
                
                # 패턴 매칭
                import re
                for pattern in config["patterns"]:
                    try:
                        if re.search(pattern, query, re.IGNORECASE):
                            matched_patterns.append(pattern)
                            score += 0.4
                    except re.error:
                        continue
                
                intent_scores[intent_type] = {
                    "score": score,
                    "keywords": matched_keywords,
                    "patterns": matched_patterns,
                    "risk_level": config["risk_level"],
                    "threshold": config["threshold"]
                }
            
            # 가장 높은 점수의 의도 선택
            best_intent = max(intent_scores.items(), key=lambda x: x[1]["score"])
            intent_type = best_intent[0]
            intent_data = best_intent[1]
            
            # 맥락 분석 (간단한 버전)
            context_type = self._analyze_context(query)
            
            # 위험도 결정
            risk_level = intent_data["risk_level"]
            
            # 신뢰도 계산
            confidence = min(intent_data["score"], 1.0)
            
            # 추론 과정 생성
            reasoning = self._generate_reasoning(intent_type, intent_data, context_type)
            
            return IntentAnalysis(
                intent_type=intent_type,
                confidence=confidence,
                keywords=intent_data["keywords"],
                patterns=intent_data["patterns"],
                context_type=context_type,
                risk_level=risk_level,
                reasoning=reasoning
            )
            
        except Exception as e:
            self.logger.error(f"Error analyzing intent: {e}")
            return IntentAnalysis(
                intent_type=IntentType.GENERAL_INFO_REQUEST,
                confidence=0.0,
                keywords=[],
                patterns=[],
                context_type=ContextType.GENERAL_CURIOSITY,
                risk_level="low",
                reasoning=f"Error: {str(e)}"
            )
    
    def _analyze_context(self, query: str) -> ContextType:
        """맥락 분석"""
        query_lower = query.lower()
        
        # 개인적 표현 지표
        personal_indicators = [
            "제 경우", "저는", "내 사건", "이런 상황", "현재 상황",
            "진행 중인", "당사자", "구체적 사안", "실제 사건"
        ]
        
        # 일반적 표현 지표
        general_indicators = [
            "일반적으로", "보통", "일반적인", "법령", "법률", 
            "관련 법", "절차", "신청", "제출", "처리"
        ]
        
        # 가상적 표현 지표
        hypothetical_indicators = [
            "만약", "가정", "예를 들어", "가상의", "만일",
            "상상해보면", "가정해보면", "예시로"
        ]
        
        # 점수 계산
        personal_score = sum(1 for indicator in personal_indicators if indicator in query_lower)
        general_score = sum(1 for indicator in general_indicators if indicator in query_lower)
        hypothetical_score = sum(1 for indicator in hypothetical_indicators if indicator in query_lower)
        
        # 맥락 유형 결정
        if personal_score > 0:
            return ContextType.PERSONAL_CASE
        elif general_score > 0:
            return ContextType.GENERAL_CURIOSITY
        elif hypothetical_score > 0:
            return ContextType.HYPOTHETICAL
        else:
            return ContextType.GENERAL_CURIOSITY
    
    def _generate_reasoning(self, intent_type: IntentType, intent_data: Dict, context_type: ContextType) -> str:
        """추론 과정 생성"""
        reasoning_parts = []
        
        # 의도 분석 근거
        if intent_data["keywords"]:
            reasoning_parts.append(f"키워드 매칭: {', '.join(intent_data['keywords'])}")
        
        if intent_data["patterns"]:
            reasoning_parts.append(f"패턴 매칭: {len(intent_data['patterns'])}개")
        
        # 맥락 분석 근거
        reasoning_parts.append(f"맥락 유형: {context_type.value}")
        
        # 위험도 근거
        reasoning_parts.append(f"위험도: {intent_data['risk_level']}")
        
        return " | ".join(reasoning_parts)
    
    def process_by_intent(self, query: str, restriction_result: ImprovedRestrictionResult) -> ProcessingResult:
        """의도별 처리"""
        try:
            # 의도 분석
            intent_analysis = self.analyze_intent(query)
            
            # 절차 관련 질문에 대한 특별 처리 (더욱 관대한 처리)
            if any(keyword in query.lower() for keyword in ["절차", "방법", "과정", "규정", "제도"]):
                # 절차 관련 질문은 더욱 관대하게 처리
                if restriction_result.is_restricted:
                    # 제한되어 있어도 절차 관련이면 허용
                    return ProcessingResult(
                        allowed=True,
                        response_type=ResponseType.GENERAL_INFO,
                        confidence=0.8,
                        message="절차 관련 질문에 대해 일반적인 정보를 제공할 수 있습니다.",
                        safe_alternatives=["관련 법령이나 판례를 참고하실 수 있습니다."],
                        disclaimer="구체적인 사안은 전문가와 상담하시기 바랍니다.",
                        expert_referral="변호사와 상담하시는 것을 권합니다.",
                        reasoning="절차 관련 질문 특별 처리 (관대한 처리)"
                    )
                else:
                    # 제한되지 않았으면 그대로 허용
                    return self._process_allowed_query(intent_analysis, restriction_result)
            
            # 제한 결과와 의도 분석을 종합하여 처리
            if restriction_result.is_restricted:
                return self._process_restricted_query(intent_analysis, restriction_result)
            else:
                return self._process_allowed_query(intent_analysis, restriction_result)
                
        except Exception as e:
            self.logger.error(f"Error processing by intent: {e}")
            return ProcessingResult(
                allowed=False,
                response_type=ResponseType.SAFE_RESPONSE,
                confidence=0.0,
                message="처리 중 오류가 발생했습니다.",
                safe_alternatives=["일반적인 법률 정보는 안내드릴 수 있습니다."],
                disclaimer="구체적인 사안은 전문가와 상담하시기 바랍니다.",
                expert_referral="변호사와 상담하시는 것을 권합니다.",
                reasoning=f"Error: {str(e)}"
            )
    
    def _process_restricted_query(self, intent_analysis: IntentAnalysis, 
                                restriction_result: ImprovedRestrictionResult) -> ProcessingResult:
        """제한된 질문 처리"""
        if intent_analysis.intent_type == IntentType.SUSPICIOUS_REQUEST:
            template = self.response_templates[ResponseType.BLOCKED_RESPONSE]
            return ProcessingResult(
                allowed=False,
                response_type=ResponseType.BLOCKED_RESPONSE,
                confidence=template["confidence"],
                message=template["message"],
                safe_alternatives=template["safe_alternatives"],
                disclaimer=template["disclaimer"],
                expert_referral=template["expert_referral"],
                reasoning=f"의심스러운 요청 감지: {intent_analysis.reasoning}"
            )
        else:
            template = self.response_templates[ResponseType.SAFE_RESPONSE]
            return ProcessingResult(
                allowed=False,
                response_type=ResponseType.SAFE_RESPONSE,
                confidence=template["confidence"],
                message=template["message"],
                safe_alternatives=template["safe_alternatives"],
                disclaimer=template["disclaimer"],
                expert_referral=template["expert_referral"],
                reasoning=f"제한된 질문: {intent_analysis.reasoning}"
            )
    
    def _process_allowed_query(self, intent_analysis: IntentAnalysis, 
                             restriction_result: ImprovedRestrictionResult) -> ProcessingResult:
        """허용된 질문 처리"""
        if intent_analysis.intent_type == IntentType.GENERAL_INFO_REQUEST:
            template = self.response_templates[ResponseType.GENERAL_INFO]
            response_type = ResponseType.GENERAL_INFO
        elif intent_analysis.intent_type == IntentType.PROCEDURE_INQUIRY:
            template = self.response_templates[ResponseType.PROCEDURE_GUIDE]
            response_type = ResponseType.PROCEDURE_GUIDE
        elif intent_analysis.intent_type == IntentType.STATUTE_REFERENCE:
            template = self.response_templates[ResponseType.STATUTE_INFO]
            response_type = ResponseType.STATUTE_INFO
        elif intent_analysis.intent_type == IntentType.PRECEDENT_SEARCH:
            template = self.response_templates[ResponseType.PRECEDENT_INFO]
            response_type = ResponseType.PRECEDENT_INFO
        elif intent_analysis.intent_type in [IntentType.LEGAL_ADVICE_REQUEST, IntentType.CASE_SPECIFIC_QUESTION]:
            template = self.response_templates[ResponseType.EXPERT_REFERRAL]
            response_type = ResponseType.EXPERT_REFERRAL
        else:
            template = self.response_templates[ResponseType.GENERAL_INFO]
            response_type = ResponseType.GENERAL_INFO
        
        return ProcessingResult(
            allowed=True,
            response_type=response_type,
            confidence=template["confidence"],
            message=template["message"],
            safe_alternatives=template["safe_alternatives"],
            disclaimer=template["disclaimer"],
            expert_referral=template["expert_referral"],
            reasoning=f"허용된 질문: {intent_analysis.reasoning}"
        )
    
    def get_processing_statistics(self) -> Dict[str, Any]:
        """처리 통계 정보"""
        return {
            "intent_types": [intent.value for intent in IntentType],
            "response_types": [response.value for response in ResponseType],
            "total_patterns": sum(len(config["patterns"]) for config in self.intent_patterns.values()),
            "total_keywords": sum(len(config["keywords"]) for config in self.intent_patterns.values()),
            "response_templates": len(self.response_templates)
        }
