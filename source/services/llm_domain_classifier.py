# -*- coding: utf-8 -*-
"""
LLM 기반 도메인 분류 시스템
하드코딩된 키워드 매칭 대신 LLM을 활용한 지능적 도메인 분류
"""

import logging
import json
from typing import Dict, List, Any, Optional
from datetime import datetime

logger = logging.getLogger(__name__)


class LLMDomainClassifier:
    """LLM을 활용한 도메인 분류 시스템"""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.conversation_history = {}
        
        # 도메인별 응답 생성기 매핑 (나중에 설정됨)
        self.domain_mappings = {}
        
        self.logger.info("LLMDomainClassifier initialized")
    
    def set_domain_mappings(self, mappings: Dict[str, Any]):
        """도메인별 응답 생성기 매핑 설정"""
        self.domain_mappings = mappings
    
    def classify_domain_with_llm(self, message: str, query_analysis: Dict[str, Any]) -> str:
        """LLM을 사용하여 메시지의 도메인 분류"""
        
        try:
            from .gemini_client import GeminiClient
            gemini_client = GeminiClient()
            
            prompt = f"""다음 사용자 질문을 분석하여 가장 적절한 법률 도메인을 분류해주세요.

사용자 질문: "{message}"
현재 분석된 도메인: {query_analysis.get('domain', 'unknown')}
질문 유형: {query_analysis.get('question_type', 'unknown')}

분류할 도메인:
1. contract: 계약서 작성, 계약 관련 문제, 계약 조건, 계약서 검토 등
2. real_estate: 부동산 매매, 임대차, 등기, 부동산 거래, 부동산 계약 등
3. family_law: 이혼, 상속, 양육권, 재산분할, 가족관계, 친권 등
4. civil_law: 손해배상, 불법행위, 민사소송, 계약 위반, 채권채무 등
5. general: 일반적인 법률 질문, 법률 조문 문의, 법령 해석 등

응답 형식: JSON으로만 응답해주세요.
{{
    "domain": "도메인명",
    "confidence": 0.0-1.0,
    "reasoning": "분류 이유"
}}"""
            
            response = gemini_client.generate(prompt)
            result = self._parse_domain_response(response.response)
            
            detected_domain = result.get("domain", "general")
            confidence = result.get("confidence", 0.5)
            
            self.logger.info(f"LLM classified domain: {detected_domain} (confidence: {confidence})")
            return detected_domain
            
        except Exception as e:
            self.logger.error(f"LLM domain classification failed: {e}")
            return self._fallback_domain_classification(message)
    
    def _parse_domain_response(self, llm_response: str) -> Dict[str, Any]:
        """LLM 응답 파싱"""
        try:
            if "{" in llm_response and "}" in llm_response:
                json_start = llm_response.find("{")
                json_end = llm_response.rfind("}") + 1
                json_str = llm_response[json_start:json_end]
                return json.loads(json_str)
        except Exception as e:
            self.logger.error(f"Failed to parse domain response: {e}")
        
        return {"domain": "general", "confidence": 0.5}
    
    def _fallback_domain_classification(self, message: str) -> str:
        """폴백 도메인 분류 (기존 키워드 방식)"""
        message_lower = message.lower()
        
        # 계약서 관련 키워드
        contract_keywords = ["계약서", "계약", "작성", "만들", "체결", "조건", "검토"]
        if any(keyword in message_lower for keyword in contract_keywords):
            return "contract"
        
        # 부동산 관련 키워드
        real_estate_keywords = ["부동산", "매매", "임대차", "등기", "아파트", "주택", "토지"]
        if any(keyword in message_lower for keyword in real_estate_keywords):
            return "real_estate"
        
        # 가족법 관련 키워드
        family_law_keywords = ["이혼", "상속", "양육권", "재산분할", "가족", "친권", "위자료"]
        if any(keyword in message_lower for keyword in family_law_keywords):
            return "family_law"
        
        # 민사법 관련 키워드
        civil_law_keywords = ["손해배상", "청구", "배상", "피해", "불법행위", "민사소송", "채권"]
        if any(keyword in message_lower for keyword in civil_law_keywords):
            return "civil_law"
        
        return "general"


class ContextAwareDomainClassifier(LLMDomainClassifier):
    """컨텍스트를 고려한 도메인 분류기"""
    
    def __init__(self):
        super().__init__()
        self.logger.info("ContextAwareDomainClassifier initialized")
    
    def classify_domain_with_context(self, message: str, query_analysis: Dict[str, Any], session_id: str) -> str:
        """대화 히스토리를 고려한 도메인 분류"""
        
        # 이전 대화 컨텍스트 가져오기
        context = self.conversation_history.get(session_id, [])
        context_summary = self._summarize_context(context)
        
        try:
            from .gemini_client import GeminiClient
            gemini_client = GeminiClient()
            
            prompt = f"""다음 사용자 질문을 분석하여 가장 적절한 법률 도메인을 분류해주세요.

사용자 질문: "{message}"
현재 분석된 도메인: {query_analysis.get('domain', 'unknown')}
질문 유형: {query_analysis.get('question_type', 'unknown')}

대화 컨텍스트: {context_summary}

분류할 도메인:
1. contract: 계약서 작성, 계약 관련 문제, 계약 조건, 계약서 검토 등
2. real_estate: 부동산 매매, 임대차, 등기, 부동산 거래, 부동산 계약 등
3. family_law: 이혼, 상속, 양육권, 재산분할, 가족관계, 친권 등
4. civil_law: 손해배상, 불법행위, 민사소송, 계약 위반, 채권채무 등
5. general: 일반적인 법률 질문, 법률 조문 문의, 법령 해석 등

응답 형식: JSON으로만 응답해주세요.
{{
    "domain": "도메인명",
    "confidence": 0.0-1.0,
    "reasoning": "분류 이유",
    "context_influence": "컨텍스트가 분류에 미친 영향"
}}"""
            
            response = gemini_client.generate(prompt)
            result = self._parse_domain_response(response.response)
            
            detected_domain = result.get("domain", "general")
            confidence = result.get("confidence", 0.5)
            
            # 대화 히스토리 업데이트
            self._update_conversation_history(session_id, message, detected_domain)
            
            self.logger.info(f"Context-aware LLM classified domain: {detected_domain} (confidence: {confidence})")
            return detected_domain
            
        except Exception as e:
            self.logger.error(f"Context-aware domain classification failed: {e}")
            return super().classify_domain_with_llm(message, query_analysis)
    
    def _summarize_context(self, context: List[Dict[str, Any]]) -> str:
        """대화 컨텍스트 요약"""
        if not context:
            return "이전 대화 없음"
        
        recent_topics = [item.get("domain", "unknown") for item in context[-3:]]
        return f"최근 대화 주제: {', '.join(recent_topics)}"
    
    def _update_conversation_history(self, session_id: str, message: str, domain: str):
        """대화 히스토리 업데이트"""
        if session_id not in self.conversation_history:
            self.conversation_history[session_id] = []
        
        self.conversation_history[session_id].append({
            "message": message,
            "domain": domain,
            "timestamp": datetime.now().isoformat()
        })
        
        # 최대 10개 대화만 유지
        if len(self.conversation_history[session_id]) > 10:
            self.conversation_history[session_id] = self.conversation_history[session_id][-10:]
    
    def get_conversation_context(self, session_id: str) -> List[Dict[str, Any]]:
        """대화 컨텍스트 반환"""
        return self.conversation_history.get(session_id, [])
    
    def clear_conversation_history(self, session_id: str):
        """대화 히스토리 초기화"""
        if session_id in self.conversation_history:
            del self.conversation_history[session_id]
