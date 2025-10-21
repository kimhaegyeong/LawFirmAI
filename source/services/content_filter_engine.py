# -*- coding: utf-8 -*-
"""
Content Filter Engine
콘텐츠 필터링 엔진 - 고급 의도 분석 및 맥락 이해
"""

import re
import logging
from typing import Dict, List, Optional, Any, Tuple
from enum import Enum
from dataclasses import dataclass
from datetime import datetime
import json

logger = logging.getLogger(__name__)


class IntentType(Enum):
    """의도 유형"""
    LEGAL_ADVICE_REQUEST = "legal_advice_request"      # 법률 자문 요청
    CASE_SPECIFIC_QUESTION = "case_specific_question"  # 구체적 사건 질문
    GENERAL_INFO_REQUEST = "general_info_request"       # 일반 정보 요청
    PROCEDURE_INQUIRY = "procedure_inquiry"            # 절차 문의
    STATUTE_REFERENCE = "statute_reference"             # 법령 참조
    PRECEDENT_SEARCH = "precedent_search"               # 판례 검색
    SUSPICIOUS_REQUEST = "suspicious_request"           # 의심스러운 요청


class ContextType(Enum):
    """맥락 유형"""
    PERSONAL_CASE = "personal_case"          # 개인 사건
    HYPOTHETICAL = "hypothetical"            # 가상의 상황
    ACADEMIC = "academic"                    # 학술적 질문
    PROFESSIONAL = "professional"           # 전문가 질문
    GENERAL_CURIOSITY = "general_curiosity"  # 일반적 호기심


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
class FilterResult:
    """필터링 결과"""
    is_blocked: bool
    block_reason: Optional[str]
    intent_analysis: IntentAnalysis
    safe_alternatives: List[str]
    warning_level: str
    timestamp: datetime


class ContentFilterEngine:
    """콘텐츠 필터링 엔진"""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.intent_patterns = self._initialize_intent_patterns()
        self.context_indicators = self._initialize_context_indicators()
        self.risk_keywords = self._initialize_risk_keywords()
        
    def _initialize_intent_patterns(self) -> Dict[IntentType, Dict[str, Any]]:
        """의도 패턴 초기화"""
        return {
            IntentType.LEGAL_ADVICE_REQUEST: {
                "patterns": [
                    r"제\s*경우\s*(어떻게|무엇을|해야)",
                    r"저는\s*(어떻게|무엇을|해야)",
                    r"내\s*사건\s*(어떻게|무엇을|해야)",
                    r"이런\s*상황\s*(어떻게|무엇을|해야)",
                    r"소송\s*(할까요|해야\s*할까요|하지\s*않을까요)",
                    r"변호사\s*(고용|선임|상담)\s*(해야|할까요)",
                    r"법적\s*조치\s*(어떻게|무엇을|해야)",
                    r"권리\s*(주장|행사)\s*(어떻게|무엇을|해야)"
                ],
                "keywords": [
                    "제 경우", "저는", "내 사건", "이런 상황", "소송", "변호사",
                    "법적 조치", "권리 주장", "권리 행사", "어떻게 해야", "무엇을 해야"
                ],
                "risk_level": "high"
            },
            
            IntentType.CASE_SPECIFIC_QUESTION: {
                "patterns": [
                    r"구체적으로\s*(어떻게|무엇을|해야)",
                    r"실제\s*사건\s*(어떻게|무엇을|해야)",
                    r"이\s*사건\s*(어떻게|무엇을|해야)",
                    r"현재\s*상황\s*(어떻게|무엇을|해야)",
                    r"진행\s*중인\s*(사건|소송|분쟁)",
                    r"당사자\s*(어떻게|무엇을|해야)"
                ],
                "keywords": [
                    "구체적으로", "실제 사건", "이 사건", "현재 상황",
                    "진행 중인", "당사자", "구체적 사안"
                ],
                "risk_level": "high"
            },
            
            IntentType.GENERAL_INFO_REQUEST: {
                "patterns": [
                    r"일반적으로\s*(어떻게|무엇을|해야)",
                    r"보통\s*(어떻게|무엇을|해야)",
                    r"일반적인\s*(절차|방법|과정)",
                    r"법령\s*(어떻게|무엇을|해야)",
                    r"법률\s*(어떻게|무엇을|해야)",
                    r"관련\s*법\s*(어떻게|무엇을|해야)"
                ],
                "keywords": [
                    "일반적으로", "보통", "일반적인", "법령", "법률", "관련 법"
                ],
                "risk_level": "low"
            },
            
            IntentType.PROCEDURE_INQUIRY: {
                "patterns": [
                    r"절차\s*(어떻게|무엇을|해야)",
                    r"신청\s*(어떻게|무엇을|해야)",
                    r"제출\s*(어떻게|무엇을|해야)",
                    r"처리\s*(어떻게|무엇을|해야)",
                    r"어디에\s*(신청|제출|문의)",
                    r"어떤\s*서류\s*(필요|준비)"
                ],
                "keywords": [
                    "절차", "신청", "제출", "처리", "어디에", "어떤 서류"
                ],
                "risk_level": "low"
            },
            
            IntentType.STATUTE_REFERENCE: {
                "patterns": [
                    r"법령\s*(참조|인용|적용)",
                    r"법조문\s*(참조|인용|적용)",
                    r"관련\s*법령\s*(참조|인용|적용)",
                    r"적용\s*법령\s*(참조|인용|적용)",
                    r"법률\s*(참조|인용|적용)"
                ],
                "keywords": [
                    "법령", "법조문", "관련 법령", "적용 법령", "법률"
                ],
                "risk_level": "low"
            },
            
            IntentType.PRECEDENT_SEARCH: {
                "patterns": [
                    r"판례\s*(참조|인용|적용)",
                    r"대법원\s*(판례|판결)",
                    r"법원\s*(판례|판결)",
                    r"관련\s*판례\s*(참조|인용|적용)",
                    r"유사\s*사건\s*(판례|판결)"
                ],
                "keywords": [
                    "판례", "대법원", "법원", "관련 판례", "유사 사건"
                ],
                "risk_level": "low"
            },
            
            IntentType.SUSPICIOUS_REQUEST: {
                "patterns": [
                    r"탈법\s*(방법|수단|기법)",
                    r"법망\s*(빠져나가기|회피)",
                    r"법적\s*구멍\s*(이용|활용)",
                    r"우회\s*(방법|수단|기법)",
                    r"회피\s*(방법|수단|기법)",
                    r"조작\s*(방법|수단|기법)",
                    r"위조\s*(방법|수단|기법)"
                ],
                "keywords": [
                    "탈법", "법망", "법적 구멍", "우회", "회피", "조작", "위조"
                ],
                "risk_level": "critical"
            }
        }
    
    def _initialize_context_indicators(self) -> Dict[ContextType, List[str]]:
        """맥락 지표 초기화"""
        return {
            ContextType.PERSONAL_CASE: [
                "제 경우", "저는", "내 사건", "이런 상황", "현재 상황",
                "진행 중인", "당사자", "구체적 사안", "실제 사건"
            ],
            ContextType.HYPOTHETICAL: [
                "만약", "가정", "예를 들어", "가상의", "만일",
                "상상해보면", "가정해보면", "예시로"
            ],
            ContextType.ACADEMIC: [
                "학술", "연구", "논문", "학위", "과제",
                "공부", "학습", "교육", "강의"
            ],
            ContextType.PROFESSIONAL: [
                "전문가", "변호사", "법무", "법률", "업무",
                "직업", "전문", "업계"
            ],
            ContextType.GENERAL_CURIOSITY: [
                "궁금", "호기심", "알고 싶", "이해하고 싶",
                "배우고 싶", "공부하고 싶"
            ]
        }
    
    def _initialize_risk_keywords(self) -> Dict[str, List[str]]:
        """위험 키워드 초기화"""
        return {
            "critical": [
                "탈법", "법망", "법적 구멍", "우회", "회피", "조작", "위조",
                "세금 회피", "탈세", "위장", "가짜", "허위", "거짓"
            ],
            "high": [
                "소송", "변호사", "법적 조치", "권리 주장", "권리 행사",
                "의료사고", "의료과실", "장애등급", "형량", "자백", "부인"
            ],
            "medium": [
                "구체적", "실제", "현재", "진행 중", "당사자",
                "법적 판단", "과실", "책임", "유죄", "무죄"
            ],
            "low": [
                "일반적", "보통", "법령", "법률", "관련 법",
                "절차", "신청", "제출", "처리", "판례"
            ]
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
                    "risk_level": config["risk_level"]
                }
            
            # 가장 높은 점수의 의도 선택
            best_intent = max(intent_scores.items(), key=lambda x: x[1]["score"])
            intent_type = best_intent[0]
            intent_data = best_intent[1]
            
            # 맥락 분석
            context_type = self._analyze_context(query)
            
            # 위험도 결정
            risk_level = intent_data["risk_level"]
            if context_type == ContextType.PERSONAL_CASE:
                risk_level = self._escalate_risk_level(risk_level)
            
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
        
        # 각 맥락 유형별 점수 계산
        context_scores = {}
        for context_type, indicators in self.context_indicators.items():
            score = 0
            for indicator in indicators:
                if indicator.lower() in query_lower:
                    score += 1
            context_scores[context_type] = score
        
        # 가장 높은 점수의 맥락 선택
        best_context = max(context_scores.items(), key=lambda x: x[1])
        
        # 점수가 0이면 일반적 호기심으로 분류
        if best_context[1] == 0:
            return ContextType.GENERAL_CURIOSITY
        
        return best_context[0]
    
    def _escalate_risk_level(self, current_level: str) -> str:
        """위험도 상향 조정"""
        escalation_map = {
            "low": "medium",
            "medium": "high",
            "high": "critical",
            "critical": "critical"
        }
        return escalation_map.get(current_level, current_level)
    
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
    
    def filter_content(self, query: str, response: str = "") -> FilterResult:
        """콘텐츠 필터링"""
        try:
            # 의도 분석
            intent_analysis = self.analyze_intent(query)
            
            # 차단 여부 결정
            is_blocked = self._should_block(intent_analysis, query, response)
            
            # 차단 사유
            block_reason = None
            if is_blocked:
                block_reason = self._get_block_reason(intent_analysis)
            
            # 안전한 대안 생성
            safe_alternatives = self._generate_safe_alternatives(intent_analysis, query)
            
            # 경고 수준 결정
            warning_level = self._determine_warning_level(intent_analysis)
            
            return FilterResult(
                is_blocked=is_blocked,
                block_reason=block_reason,
                intent_analysis=intent_analysis,
                safe_alternatives=safe_alternatives,
                warning_level=warning_level,
                timestamp=datetime.now()
            )
            
        except Exception as e:
            self.logger.error(f"Error filtering content: {e}")
            return FilterResult(
                is_blocked=False,
                block_reason=None,
                intent_analysis=IntentAnalysis(
                    intent_type=IntentType.GENERAL_INFO_REQUEST,
                    confidence=0.0,
                    keywords=[],
                    patterns=[],
                    context_type=ContextType.GENERAL_CURIOSITY,
                    risk_level="low",
                    reasoning=f"Error: {str(e)}"
                ),
                safe_alternatives=[],
                warning_level="low",
                timestamp=datetime.now()
            )
    
    def _should_block(self, intent_analysis: IntentAnalysis, query: str, response: str) -> bool:
        """차단 여부 결정"""
        # 위험도 기반 차단
        if intent_analysis.risk_level == "critical":
            return True
        
        # 개인 사건 + 높은 위험도
        if (intent_analysis.context_type == ContextType.PERSONAL_CASE and 
            intent_analysis.risk_level in ["high", "medium"]):
            return True
        
        # 의심스러운 요청
        if intent_analysis.intent_type == IntentType.SUSPICIOUS_REQUEST:
            return True
        
        # 법률 자문 요청 + 높은 신뢰도
        if (intent_analysis.intent_type == IntentType.LEGAL_ADVICE_REQUEST and 
            intent_analysis.confidence > 0.7):
            return True
        
        # 구체적 사건 질문 + 높은 신뢰도
        if (intent_analysis.intent_type == IntentType.CASE_SPECIFIC_QUESTION and 
            intent_analysis.confidence > 0.6):
            return True
        
        return False
    
    def _get_block_reason(self, intent_analysis: IntentAnalysis) -> str:
        """차단 사유 생성"""
        reasons = {
            IntentType.LEGAL_ADVICE_REQUEST: "구체적인 법률 자문 요청으로 인한 차단",
            IntentType.CASE_SPECIFIC_QUESTION: "구체적 사건에 대한 질문으로 인한 차단",
            IntentType.SUSPICIOUS_REQUEST: "의심스러운 요청으로 인한 차단"
        }
        
        return reasons.get(intent_analysis.intent_type, "위험도가 높은 요청으로 인한 차단")
    
    def _generate_safe_alternatives(self, intent_analysis: IntentAnalysis, query: str) -> List[str]:
        """안전한 대안 생성"""
        alternatives = []
        
        if intent_analysis.intent_type == IntentType.LEGAL_ADVICE_REQUEST:
            alternatives.extend([
                "일반적인 법률 정보나 절차를 안내드릴 수 있습니다.",
                "관련 법령이나 판례를 참고하실 수 있습니다.",
                "구체적인 사안은 변호사와 상담하시는 것을 권합니다."
            ])
        
        elif intent_analysis.intent_type == IntentType.CASE_SPECIFIC_QUESTION:
            alternatives.extend([
                "일반적인 절차나 방법을 안내드릴 수 있습니다.",
                "관련 법령의 일반적인 적용 원칙을 설명드릴 수 있습니다.",
                "구체적인 사안은 전문가와 상담하시는 것을 권합니다."
            ])
        
        elif intent_analysis.intent_type == IntentType.SUSPICIOUS_REQUEST:
            alternatives.extend([
                "법적 절차나 원칙에 대한 일반적인 정보를 안내드릴 수 있습니다.",
                "관련 법령의 정당한 적용 방법을 설명드릴 수 있습니다."
            ])
        
        else:
            alternatives.extend([
                "일반적인 법률 정보를 안내드릴 수 있습니다.",
                "관련 절차나 방법을 설명드릴 수 있습니다."
            ])
        
        return alternatives
    
    def _determine_warning_level(self, intent_analysis: IntentAnalysis) -> str:
        """경고 수준 결정"""
        if intent_analysis.risk_level == "critical":
            return "critical"
        elif intent_analysis.risk_level == "high":
            return "high"
        elif intent_analysis.risk_level == "medium":
            return "medium"
        else:
            return "low"
    
    def get_filter_statistics(self) -> Dict[str, Any]:
        """필터 통계 정보"""
        return {
            "intent_types": [intent.value for intent in IntentType],
            "context_types": [context.value for context in ContextType],
            "risk_levels": ["critical", "high", "medium", "low"],
            "total_patterns": sum(len(config["patterns"]) for config in self.intent_patterns.values()),
            "total_keywords": sum(len(config["keywords"]) for config in self.intent_patterns.values())
        }
