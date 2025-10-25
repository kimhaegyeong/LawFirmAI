# -*- coding: utf-8 -*-
"""
지능형 답변 스타일 시스템
사용자 의도와 상황을 분석하여 적절한 답변 스타일을 동적으로 결정하는 시스템
"""

import logging
import json
from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime
from enum import Enum

logger = logging.getLogger(__name__)


class ResponseStyle(Enum):
    """답변 스타일 유형"""
    CONCISE = "concise"          # 간결한 답변
    DETAILED = "detailed"        # 상세한 답변
    INTERACTIVE = "interactive"  # 대화형 답변
    PROFESSIONAL = "professional" # 전문적인 답변
    FRIENDLY = "friendly"        # 친근한 답변


class ResponseStyleAnalyzer:
    """LLM 기반 사용자 의도 분석 및 스타일 결정"""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        
        # 스타일별 특성 정의
        self.style_characteristics = {
            ResponseStyle.CONCISE: {
                "description": "간결하고 핵심만 담은 답변",
                "features": ["요약", "핵심 포인트", "간단한 절차", "짧은 설명"],
                "use_cases": ["빠른 정보 확인", "요약 필요", "핵심만 알고 싶음"]
            },
            ResponseStyle.DETAILED: {
                "description": "구체적이고 상세한 답변",
                "features": ["예시 포함", "단계별 설명", "주의사항", "배경 설명"],
                "use_cases": ["완전한 이해", "실무 적용", "복잡한 상황"]
            },
            ResponseStyle.INTERACTIVE: {
                "description": "대화형이고 단계별 가이드 답변",
                "features": ["질문 포함", "단계별 진행", "상황 파악", "다음 단계 제안"],
                "use_cases": ["처음 시작", "도움 필요", "단계별 진행"]
            },
            ResponseStyle.PROFESSIONAL: {
                "description": "전문적이고 법률적 답변",
                "features": ["법률 용어", "조문 인용", "판례 참조", "정확한 표현"],
                "use_cases": ["전문가 대상", "법률 근거 필요", "정확성 중요"]
            },
            ResponseStyle.FRIENDLY: {
                "description": "친근하고 이해하기 쉬운 답변",
                "features": ["일상 언어", "친근한 톤", "쉬운 설명", "격려"],
                "use_cases": ["일반인 대상", "부담 없음", "친근함 필요"]
            }
        }
        
        self.logger.info("ResponseStyleAnalyzer initialized")
    
    def analyze_user_intent_with_llm(self, message: str, query_analysis: Dict[str, Any]) -> ResponseStyle:
        """LLM을 사용하여 사용자 의도 분석 및 스타일 결정"""
        try:
            from .gemini_client import GeminiClient
            gemini_client = GeminiClient()
            
            # 사용자 메시지와 컨텍스트 정보 구성
            context_info = {
                "message": message,
                "domain": query_analysis.get("domain", "general"),
                "question_type": query_analysis.get("question_type", "general"),
                "urgency": query_analysis.get("urgency", "normal"),
                "complexity": query_analysis.get("complexity", "medium")
            }
            
            # LLM 프롬프트 구성
            prompt = self._create_style_analysis_prompt(context_info)
            
            # LLM 호출
            response = gemini_client.generate(prompt)
            
            # 응답 파싱 및 스타일 결정
            detected_style = self._parse_llm_response(response.response)
            
            self.logger.info(f"LLM detected style: {detected_style} for message: {message[:50]}...")
            return detected_style
            
        except Exception as e:
            self.logger.error(f"LLM-based style analysis failed: {e}")
            # 폴백: 키워드 기반 분석
            return self._fallback_keyword_analysis(message, query_analysis)
    
    def _create_style_analysis_prompt(self, context_info: Dict[str, Any]) -> str:
        """스타일 분석을 위한 LLM 프롬프트 생성"""
        
        prompt = f"""사용자의 질문을 분석하여 가장 적절한 답변 스타일을 결정해주세요.

사용자 질문: "{context_info['message']}"
도메인: {context_info['domain']}
질문 유형: {context_info['question_type']}
긴급도: {context_info['urgency']}
복잡도: {context_info['complexity']}

답변 스타일 옵션:
1. concise: 간결하고 핵심만 담은 답변 (요약, 핵심 포인트, 간단한 절차)
2. detailed: 구체적이고 상세한 답변 (예시 포함, 단계별 설명, 주의사항)
3. interactive: 대화형이고 단계별 가이드 답변 (질문 포함, 단계별 진행, 상황 파악)
4. professional: 전문적이고 법률적 답변 (법률 용어, 조문 인용, 정확한 표현)
5. friendly: 친근하고 이해하기 쉬운 답변 (일상 언어, 친근한 톤, 쉬운 설명)

분석 기준:
- 사용자가 "간단히", "요약", "핵심만" 등을 언급하면 → concise
- 사용자가 "자세히", "구체적으로", "예시와 함께" 등을 언급하면 → detailed
- 사용자가 "도와", "가이드", "어떻게 시작" 등을 언급하면 → interactive
- 사용자가 법률 용어를 사용하거나 전문가처럼 질문하면 → professional
- 일반적인 질문이거나 친근함을 원하는 경우 → friendly

응답 형식: JSON으로만 응답해주세요.
{{
    "detected_style": "스타일명",
    "confidence": 0.0-1.0,
    "reasoning": "선택 이유"
}}"""
        
        return prompt
    
    def _parse_llm_response(self, llm_response: str) -> ResponseStyle:
        """LLM 응답을 파싱하여 스타일 결정"""
        try:
            # JSON 응답 추출 시도
            if "{" in llm_response and "}" in llm_response:
                json_start = llm_response.find("{")
                json_end = llm_response.rfind("}") + 1
                json_str = llm_response[json_start:json_end]
                
                result = json.loads(json_str)
                detected_style = result.get("detected_style", "detailed")
                confidence = result.get("confidence", 0.8)
                
                self.logger.info(f"LLM analysis result: {detected_style} (confidence: {confidence})")
                
                # 스타일 매핑
                style_mapping = {
                    "concise": ResponseStyle.CONCISE,
                    "detailed": ResponseStyle.DETAILED,
                    "interactive": ResponseStyle.INTERACTIVE,
                    "professional": ResponseStyle.PROFESSIONAL,
                    "friendly": ResponseStyle.FRIENDLY
                }
                
                return style_mapping.get(detected_style, ResponseStyle.DETAILED)
            
            # JSON 파싱 실패 시 텍스트에서 스타일 추출
            llm_response_lower = llm_response.lower()
            if "concise" in llm_response_lower:
                return ResponseStyle.CONCISE
            elif "interactive" in llm_response_lower:
                return ResponseStyle.INTERACTIVE
            elif "professional" in llm_response_lower:
                return ResponseStyle.PROFESSIONAL
            elif "friendly" in llm_response_lower:
                return ResponseStyle.FRIENDLY
            else:
                return ResponseStyle.DETAILED
                
        except Exception as e:
            self.logger.error(f"Failed to parse LLM response: {e}")
            return ResponseStyle.DETAILED
    
    def _fallback_keyword_analysis(self, message: str, query_analysis: Dict[str, Any]) -> ResponseStyle:
        """LLM 실패 시 키워드 기반 폴백 분석"""
        message_lower = message.lower()
        
        # 간결한 답변 요청
        concise_keywords = ["간단히", "요약", "핵심만", "짧게", "담백하게", "한 줄로", "요점만"]
        if any(keyword in message_lower for keyword in concise_keywords):
            return ResponseStyle.CONCISE
        
        # 상세한 답변 요청
        detailed_keywords = ["자세히", "구체적으로", "예시와 함께", "상세히", "완전히", "모든"]
        if any(keyword in message_lower for keyword in detailed_keywords):
            return ResponseStyle.DETAILED
        
        # 대화형 답변 요청
        interactive_keywords = ["도와", "가이드", "단계별", "차근차근", "어떻게 시작", "처음"]
        if any(keyword in message_lower for keyword in interactive_keywords):
            return ResponseStyle.INTERACTIVE
        
        # 전문적 답변 요청
        professional_keywords = ["법률", "조문", "판례", "법원", "변호사", "전문가"]
        if any(keyword in message_lower for keyword in professional_keywords):
            return ResponseStyle.PROFESSIONAL
        
        # 긴급도나 복잡도 기반 판단
        urgency = query_analysis.get("urgency", "normal")
        complexity = query_analysis.get("complexity", "medium")
        
        if urgency == "high":
            return ResponseStyle.CONCISE
        elif complexity == "high":
            return ResponseStyle.DETAILED
        else:
            return ResponseStyle.FRIENDLY


class DynamicResponseGenerator:
    """AI를 활용한 동적 답변 생성"""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.logger.info("DynamicResponseGenerator initialized")
    
    def generate_style_aware_response(self, 
                                    base_content: str, 
                                    style: ResponseStyle, 
                                    context: Dict[str, Any]) -> str:
        """스타일과 컨텍스트를 고려한 동적 답변 생성"""
        try:
            from .gemini_client import GeminiClient
            gemini_client = GeminiClient()
            
            # 스타일별 프롬프트 생성
            style_prompt = self._create_style_prompt(style, context)
            
            # 절차 우선 요구 감지
            message = context.get('message', '')
            needs_procedure_first = self._detect_procedure_request(message)
            
            # 전체 프롬프트 구성 - 절차 우선 요구 시 특별 처리
            if needs_procedure_first:
                full_prompt = f"""
**절대 규칙: 절차 설명이 먼저 와야 합니다!**

{style_prompt}

질문: {message}
원본 정보: {base_content}

**중요한 지시사항:**
1. **절대 "상황:", "예시:", "구체적인 예시" 등으로 시작하지 마세요**
2. **반드시 방법/절차부터 설명하세요**
3. 그 다음에 예시나 사례를 제시하세요
4. 마지막에 팁이나 주의사항을 제공하세요

답변 순서:
## [제목: 질문에 대한 직접적 답변]

### 1. 기본 절차/방법
- 단계별 절차 설명
- 필요한 서류나 조건
- 법적 근거

### 2. 구체적 예시
- 실제 사례 (예시용)
- 단계별 진행 과정

### 3. 실무 팁 및 주의사항
- 중요한 포인트
- 주의할 점

위 순서를 반드시 지켜서 답변을 생성해주세요.
"""
            else:
                # 기존 프롬프트 사용
                full_prompt = f"""
{style_prompt}

원본 법률 정보: {base_content}
사용자 상황: {context.get('user_situation', {})}
질문: {message}

위 정보를 바탕으로 사용자가 원하는 스타일로 답변을 생성해주세요.
"""
            
            # LLM 호출
            response = gemini_client.generate(full_prompt)
            
            self.logger.info(f"Generated {style.value} style response")
            return response.response
            
        except Exception as e:
            self.logger.error(f"Dynamic response generation failed: {e}")
            # 폴백: 기본 스타일 적용
            return self._apply_basic_style(base_content, style)
    
    def _create_style_prompt(self, style: ResponseStyle, context: Dict[str, Any]) -> str:
        """스타일별 프롬프트 생성"""
        
        base_instructions = """
**절대적으로 지켜야 할 답변 순서 규칙:**
1. **절대 예시부터 시작하지 마세요**
2. **반드시 방법/절차를 먼저 설명**해야 합니다
3. 그 다음에 구체적인 예시나 사례를 제시합니다
4. 마지막에 실무 팁이나 주의사항을 제공합니다

**중요한 주의사항:**
- 사용자가 "방법", "절차", "과정", "어떻게"를 묻는 경우 반드시 절차 설명이 먼저 와야 합니다
- 예시나 사례는 절차 설명 이후에만 제시하세요
- 절차 설명 없이 바로 예시로 시작하는 것은 절대 금지입니다
- "상황:", "예시:", "구체적인 예시" 등으로 시작하는 것은 절대 금지입니다

답변 구조:
1. 먼저 방법/절차 설명
2. 그 다음에 예시나 사례 제시
3. 마지막에 팁이나 주의사항
"""
        
        style_prompts = {
            ResponseStyle.CONCISE: f"""{base_instructions}
간결하고 핵심적인 답변을 생성하세요:
- 핵심 절차만 간단히 나열 (3-5개 포인트)
- 불필요한 예시나 설명 제거
- 단계별로 명확하게 구분
- 요약 형태로 정리
- 예시는 간단한 사례 하나만 포함""",
            
            ResponseStyle.DETAILED: f"""{base_instructions}
상세하고 포괄적인 답변을 생성하세요:
- 모든 절차를 단계별로 상세 설명
- 법적 근거와 조문 포함
- 다양한 예시와 사례 제시
- 실무 팁과 주의사항 상세 포함
- 배경 설명과 법적 근거 포함""",
            
            ResponseStyle.INTERACTIVE: f"""{base_instructions}
대화형이고 참여를 유도하는 답변을 생성하세요:
- 질문을 통한 상황 파악 유도
- 단계별 확인 질문 포함
- 구체적인 예시로 이해도 높임
- 다음 단계 안내 포함
- 친근하고 도움이 되는 톤 사용""",
            
            ResponseStyle.PROFESSIONAL: f"""{base_instructions}
전문적이고 정확한 답변을 생성하세요:
- 법률 조문과 판례 인용
- 정확한 법률 용어 사용
- 체계적인 절차 설명
- 전문가 수준의 분석과 조언
- 정확하고 명확한 표현""",
            
            ResponseStyle.FRIENDLY: f"""{base_instructions}
친근하고 이해하기 쉬운 답변을 생성하세요:
- 쉬운 말로 설명하되 법률 용어는 정확히 사용
- 격려와 위로의 메시지 포함
- 일상적인 예시로 설명
- 실용적인 조언 제공
- 친근하고 격려하는 톤 사용"""
        }
        
        return style_prompts.get(style, style_prompts[ResponseStyle.DETAILED])
    
    def _detect_procedure_request(self, message: str) -> bool:
        """절차 요청 감지"""
        procedure_keywords = [
            "방법", "절차", "과정", "어떻게", "방법을", "절차를", "과정을",
            "어떻게 해야", "어떻게 하면", "어떻게 하는", "어떻게 할",
            "어떻게 해야 하는", "어떻게 하면 되는", "청구 방법", "신청 방법",
            "진행 방법", "처리 방법", "해결 방법"
        ]
        return any(keyword in message.lower() for keyword in procedure_keywords)
    
    def _apply_basic_style(self, content: str, style: ResponseStyle) -> str:
        """기본 스타일 적용 (폴백)"""
        if style == ResponseStyle.CONCISE:
            # 간단한 요약 형태
            lines = content.split('\n')
            key_points = [line.strip() for line in lines if line.strip() and not line.startswith('#')]
            return '\n'.join(key_points[:5])  # 최대 5개 포인트
        
        elif style == ResponseStyle.FRIENDLY:
            # 친근한 인사말 추가
            return f"안녕하세요! {content}"
        
        else:
            return content


class IntelligentResponseStyleSystem:
    """지능형 답변 스타일 시스템 통합 클래스"""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.style_analyzer = ResponseStyleAnalyzer()
        self.response_generator = DynamicResponseGenerator()
        
        # 사용자 선호도 학습을 위한 저장소
        self.user_preferences = {}
        self.conversation_history = {}
        
        self.logger.info("IntelligentResponseStyleSystem initialized")
    
    def determine_optimal_style(self, 
                              message: str, 
                              query_analysis: Dict[str, Any],
                              session_id: str) -> ResponseStyle:
        """최적의 답변 스타일 결정"""
        
        # LLM 기반 사용자 의도 분석
        detected_style = self.style_analyzer.analyze_user_intent_with_llm(message, query_analysis)
        
        # 사용자 선호도 학습 적용
        if session_id in self.user_preferences:
            preference_style = self.user_preferences[session_id].get("preferred_style")
            if preference_style:
                # 선호도와 감지된 스타일을 조합
                detected_style = self._combine_preference_and_detection(preference_style, detected_style)
        
        self.logger.info(f"Optimal style determined: {detected_style.value}")
        return detected_style
    
    def generate_adaptive_response(self, 
                                 base_content: str,
                                 message: str,
                                 query_analysis: Dict[str, Any],
                                 session_id: str) -> str:
        """적응형 답변 생성"""
        
        # 최적 스타일 결정
        optimal_style = self.determine_optimal_style(message, query_analysis, session_id)
        
        # 컨텍스트 정보 구성
        context = {
            "user_situation": query_analysis.get("user_situation", {}),
            "conversation_history": self.conversation_history.get(session_id, []),
            "user_preferences": self.user_preferences.get(session_id, {}),
            "message": message,
            "session_id": session_id
        }
        
        # 동적 답변 생성
        response = self.response_generator.generate_style_aware_response(
            base_content, optimal_style, context
        )
        
        # 대화 히스토리 업데이트
        self._update_conversation_history(session_id, message, response, optimal_style)
        
        return response
    
    def _combine_preference_and_detection(self, preference: str, detected: ResponseStyle) -> ResponseStyle:
        """사용자 선호도와 감지된 스타일 조합"""
        # 선호도가 강하면 선호도 우선, 아니면 감지된 스타일 사용
        preference_mapping = {
            "concise": ResponseStyle.CONCISE,
            "detailed": ResponseStyle.DETAILED,
            "interactive": ResponseStyle.INTERACTIVE,
            "professional": ResponseStyle.PROFESSIONAL,
            "friendly": ResponseStyle.FRIENDLY
        }
        
        preferred_style = preference_mapping.get(preference, detected)
        
        # 선호도와 감지된 스타일이 다르면 선호도 우선 (70% 확률)
        if preferred_style != detected:
            return preferred_style
        else:
            return detected
    
    def _update_conversation_history(self, session_id: str, message: str, response: str, style: ResponseStyle):
        """대화 히스토리 업데이트"""
        if session_id not in self.conversation_history:
            self.conversation_history[session_id] = []
        
        self.conversation_history[session_id].append({
            "timestamp": datetime.now().isoformat(),
            "message": message,
            "response": response,
            "style": style.value
        })
        
        # 최대 10개 대화만 유지
        if len(self.conversation_history[session_id]) > 10:
            self.conversation_history[session_id] = self.conversation_history[session_id][-10:]
    
    def learn_user_preference(self, session_id: str, feedback: str):
        """사용자 피드백으로부터 선호도 학습"""
        if session_id not in self.user_preferences:
            self.user_preferences[session_id] = {}
        
        # 피드백 분석 (간단한 키워드 기반)
        feedback_lower = feedback.lower()
        
        if any(word in feedback_lower for word in ["간단", "요약", "짧게"]):
            self.user_preferences[session_id]["preferred_style"] = "concise"
        elif any(word in feedback_lower for word in ["자세", "구체", "상세"]):
            self.user_preferences[session_id]["preferred_style"] = "detailed"
        elif any(word in feedback_lower for word in ["친근", "쉽게", "부담"]):
            self.user_preferences[session_id]["preferred_style"] = "friendly"
        
        self.logger.info(f"Learned user preference for session {session_id}: {self.user_preferences[session_id]}")
