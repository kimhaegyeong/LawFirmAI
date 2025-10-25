# -*- coding: utf-8 -*-
"""
COT(Chain of Thought) 기반 동적 템플릿 생성기
질문에 맞춘 정적 템플릿 대신 COT 방식으로 동적 템플릿 생성
"""

import logging
import json
from typing import Dict, List, Any, Optional
from datetime import datetime

logger = logging.getLogger(__name__)


class COTTemplateGenerator:
    """COT 기반 동적 템플릿 생성기"""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.logger.info("COTTemplateGenerator initialized")
    
    def generate_cot_template(self, message: str, domain: str, style: str, context: Dict[str, Any]) -> str:
        """COT 방식으로 동적 템플릿 생성"""
        
        try:
            from .gemini_client import GeminiClient
            gemini_client = GeminiClient()
            
            cot_prompt = f"""다음 사용자 질문에 대해 Chain of Thought 방식으로 답변 템플릿을 생성해주세요.

사용자 질문: "{message}"
도메인: {domain}
응답 스타일: {style}
사용자 상황: {context.get('user_situation', {})}

생성 과정:
1. 질문 분석: 사용자가 무엇을 원하는지 파악
2. 핵심 정보 추출: 답변에 필요한 핵심 법률 정보 식별
3. 구조 설계: 논리적 흐름에 따른 답변 구조 설계
4. 스타일 적용: 요청된 스타일에 맞는 표현 방식 적용

응답 형식:
{{
    "question_analysis": "사용자 질문 분석",
    "core_information": ["핵심 정보 1", "핵심 정보 2", ...],
    "logical_structure": ["단계 1", "단계 2", ...],
    "style_adaptation": "스타일별 표현 방식",
    "final_template": "최종 템플릿"
}}"""
            
            response = gemini_client.generate(cot_prompt)
            result = self._parse_cot_response(response.response)
            
            self.logger.info(f"COT template generated for domain: {domain}, style: {style}")
            return result
            
        except Exception as e:
            self.logger.error(f"COT template generation failed: {e}")
            return self._generate_fallback_template(domain, style)
    
    def _parse_cot_response(self, llm_response: str) -> str:
        """COT 응답 파싱하여 템플릿 추출"""
        try:
            if "{" in llm_response and "}" in llm_response:
                json_start = llm_response.find("{")
                json_end = llm_response.rfind("}") + 1
                json_str = llm_response[json_start:json_end]
                result = json.loads(json_str)
                return result.get("final_template", "")
        except Exception as e:
            self.logger.error(f"Failed to parse COT response: {e}")
        
        return ""
    
    def _generate_fallback_template(self, domain: str, style: str) -> str:
        """폴백 템플릿 생성"""
        fallback_templates = {
            "real_estate": {
                "interactive": "부동산 매매 절차 안내\n\n**절차:**\n1. 매물 확인\n2. 계약 체결\n3. 등기 신청\n4. 명도 완료\n\n**예시:** 구체적 사례와 실무 팁을 포함하여 설명드리겠습니다.",
                "concise": "부동산 매매 절차\n\n**방법:**\n1. 매물 확인 → 2. 계약 체결 → 3. 등기 신청 → 4. 명도 완료\n\n**예시:** 간단한 사례",
                "detailed": "부동산 매매 절차 상세 안내\n\n**절차:** 단계별 상세 설명과 법적 근거, 주의사항을 포함하여 안내드리겠습니다.\n\n**예시:** 구체적 사례"
            },
            "contract": {
                "interactive": "계약서 작성 방법 안내\n\n**절차:**\n1. 당사자 정보\n2. 계약 내용\n3. 조건 명시\n4. 분쟁 해결\n\n**예시:** 구체적 사례와 작성 팁을 포함하여 설명드리겠습니다.",
                "concise": "계약서 작성\n\n**방법:**\n필수 요소: 당사자, 내용, 조건, 분쟁해결\n\n**예시:** 간단한 사례",
                "detailed": "계약서 작성 상세 가이드\n\n**절차:** 법적 요건, 작성 방법, 주의사항을 상세히 안내드리겠습니다.\n\n**예시:** 구체적 사례"
            },
            "family_law": {
                "interactive": "가족법 관련 절차 안내\n\n**절차:**\n1. 신청 준비\n2. 서류 제출\n3. 심리 진행\n4. 결정 통보\n\n**예시:** 구체적 사례와 실무 팁을 포함하여 설명드리겠습니다.",
                "concise": "가족법 절차\n\n**방법:**\n1. 신청 → 2. 제출 → 3. 심리 → 4. 결정\n\n**예시:** 간단한 사례",
                "detailed": "가족법 절차 상세 안내\n\n**절차:** 법적 근거, 절차, 주의사항을 상세히 안내드리겠습니다.\n\n**예시:** 구체적 사례"
            },
            "civil_law": {
                "interactive": "민사법 관련 절차 안내\n\n**절차:**\n1. 손해 확인\n2. 증거 수집\n3. 청구서 작성\n4. 소송 제기\n\n**예시:** 구체적 사례와 실무 팁을 포함하여 설명드리겠습니다.",
                "concise": "민사법 절차\n\n**방법:**\n1. 확인 → 2. 수집 → 3. 작성 → 4. 제기\n\n**예시:** 간단한 사례",
                "detailed": "민사법 절차 상세 안내\n\n**절차:** 법적 근거, 절차, 주의사항을 상세히 안내드리겠습니다.\n\n**예시:** 구체적 사례"
            },
            "general": {
                "interactive": "법률 관련 도움 안내\n\n**절차:**\n1. 문제 파악\n2. 법적 근거\n3. 해결 방안\n4. 실행\n\n**예시:** 구체적 사례와 실무 팁을 포함하여 설명드리겠습니다.",
                "concise": "법률 도움\n\n**방법:**\n문제 파악 → 법적 근거 → 해결 방안\n\n**예시:** 간단한 사례",
                "detailed": "법률 문제 상세 안내\n\n**절차:** 법적 근거, 해결 방안, 주의사항을 상세히 안내드리겠습니다.\n\n**예시:** 구체적 사례"
            }
        }
        
        domain_templates = fallback_templates.get(domain, {})
        return domain_templates.get(style, "해당 도메인의 답변을 생성하겠습니다.")


class AdvancedCOTTemplateGenerator(COTTemplateGenerator):
    """고급 COT 템플릿 생성기 - 다단계 사고 과정"""
    
    def __init__(self):
        super().__init__()
        self.logger.info("AdvancedCOTTemplateGenerator initialized")
    
    def generate_advanced_cot_template(self, message: str, domain: str, style: str, context: Dict[str, Any]) -> str:
        """다단계 COT 방식으로 템플릿 생성"""
        
        try:
            # 1단계: 질문 의도 분석
            intent_analysis = self._analyze_user_intent(message, domain)
            
            # 2단계: 법률 정보 매핑
            legal_info_mapping = self._map_legal_information(intent_analysis, domain)
            
            # 3단계: 답변 구조 설계
            response_structure = self._design_response_structure(legal_info_mapping, style)
            
            # 4단계: 템플릿 생성
            final_template = self._generate_final_template(response_structure, style, context)
            
            self.logger.info(f"Advanced COT template generated for domain: {domain}, style: {style}")
            return final_template
            
        except Exception as e:
            self.logger.error(f"Advanced COT template generation failed: {e}")
            return super().generate_cot_template(message, domain, style, context)
    
    def _analyze_user_intent(self, message: str, domain: str) -> Dict[str, Any]:
        """사용자 의도 분석"""
        try:
            from .gemini_client import GeminiClient
            gemini_client = GeminiClient()
            
            intent_prompt = f"""사용자 질문의 의도를 분석해주세요.

질문: "{message}"
도메인: {domain}

분석할 요소:
1. 정보 요청 유형 (절차, 법적 근거, 예시, 비교 등)
2. 상세도 요구 수준 (간단, 상세, 전체)
3. 응답 형태 선호도 (단계별, 예시 중심, 비교형 등)
4. 긴급도 (즉시 필요, 학습 목적, 참고용)

응답 형식:
{{
    "information_type": "절차/법적근거/예시/비교",
    "detail_level": "간단/상세/전체",
    "response_format": "단계별/예시중심/비교형",
    "urgency": "높음/보통/낮음",
    "specific_needs": ["구체적 필요사항 1", "구체적 필요사항 2"]
}}"""
            
            response = gemini_client.generate(intent_prompt)
            result = self._parse_json_response(response.response)
            
            # 절차 우선 요구 여부 강제 설정
            procedure_keywords = ["방법", "절차", "과정", "어떻게", "방법을", "절차를", "과정을", "어떻게 해야", "어떻게 하면", "어떻게 하는", "어떻게 할", "어떻게 해야 하는", "어떻게 하면 되는"]
            if any(keyword in message.lower() for keyword in procedure_keywords):
                result["needs_procedure_first"] = True
                result["information_type"] = "절차"
                result["response_format"] = "단계별"
            
            return result
            
        except Exception as e:
            self.logger.error(f"User intent analysis failed: {e}")
            return {
                "information_type": "절차",
                "detail_level": "상세",
                "response_format": "단계별",
                "urgency": "보통",
                "specific_needs": [],
                "needs_procedure_first": True
            }
    
    def _map_legal_information(self, intent_analysis: Dict[str, Any], domain: str) -> List[str]:
        """법률 정보 매핑"""
        info_type = intent_analysis.get("information_type", "절차")
        detail_level = intent_analysis.get("detail_level", "상세")
        
        # 도메인별 법률 정보 데이터베이스
        legal_info_db = {
            "real_estate": {
                "절차": ["매물확인", "계약체결", "등기신청", "명도정산"],
                "법적근거": ["민법", "부동산등기법", "공인중개사법"],
                "예시": ["아파트매매", "상가매매", "토지매매"],
                "비교": ["매매vs임대차", "아파트vs상가", "신축vs중고"]
            },
            "contract": {
                "절차": ["계약서작성", "조건협의", "서명체결", "이행"],
                "법적근거": ["민법", "상법", "근로기준법"],
                "예시": ["용역계약", "근로계약", "매매계약"],
                "비교": ["유효vs무효", "해제vs해지", "위약vs손해배상"]
            },
            "family_law": {
                "절차": ["이혼신청", "재산분할", "양육권결정", "위자료청구"],
                "법적근거": ["민법", "가사소송법", "가족관계등록법"],
                "예시": ["협의이혼", "재판상이혼", "양육권분쟁"],
                "비교": ["협의vs재판", "친권vs양육권", "위자료vs재산분할"]
            },
            "civil_law": {
                "절차": ["손해확인", "증거수집", "청구서작성", "소송제기"],
                "법적근거": ["민법", "민사소송법", "소비자보호법"],
                "예시": ["불법행위", "계약위반", "제품하자"],
                "비교": ["과실vs고의", "직접손해vs간접손해", "재산손해vs정신손해"]
            },
            "general": {
                "절차": ["문제파악", "법적근거", "해결방안", "실행"],
                "법적근거": ["관련법령", "판례", "해석"],
                "예시": ["일반사례", "특수사례", "실무사례"],
                "비교": ["법적vs사실적", "이론vs실무", "일반vs특수"]
            }
        }
        
        domain_info = legal_info_db.get(domain, {})
        return domain_info.get(info_type, [])
    
    def _design_response_structure(self, legal_info: List[str], style: str) -> Dict[str, Any]:
        """답변 구조 설계 - 방법을 먼저 설명하고 예시는 뒤에"""
        structure_templates = {
            "interactive": {
                "opening": "방법 개요 (절차 우선)",
                "main_content": "단계별 절차 상세 설명",
                "example": "실제 사례 (절차 적용 예시)",
                "tips": "실무 팁",
                "closing": "추가 도움 제안"
            },
            "concise": {
                "opening": "핵심 방법 (절차 요약)",
                "main_content": "필수 절차 나열",
                "example": "간단 예시 (절차 적용)",
                "closing": "간단한 마무리"
            },
            "detailed": {
                "opening": "전체 방법 개요 (절차 중심)",
                "main_content": "상세 절차 단계별 설명",
                "legal_basis": "법적 근거",
                "example": "구체적 사례 (절차 적용)",
                "tips": "실무 팁",
                "closing": "주의사항"
            },
            "professional": {
                "opening": "법적 방법 개요 (절차 중심)",
                "main_content": "법적 근거 및 절차 상세",
                "precedent": "관련 판례",
                "example": "법적 사례 (절차 적용)",
                "closing": "법적 주의사항"
            },
            "friendly": {
                "opening": "친근한 인사",
                "main_content": "쉬운 방법 설명 (절차 중심)",
                "example": "일상 예시 (절차 적용)",
                "tips": "실용적 팁",
                "closing": "격려 메시지"
            }
        }
        
        return structure_templates.get(style, structure_templates["interactive"])
    
    def _detect_procedure_request(self, message: str) -> bool:
        """절차 요청 감지"""
        procedure_keywords = [
            "방법", "절차", "과정", "어떻게", "방법을", "절차를", "과정을",
            "어떻게 해야", "어떻게 하면", "어떻게 하는", "어떻게 할",
            "어떻게 해야 하는", "어떻게 하면 되는", "청구 방법", "신청 방법",
            "진행 방법", "처리 방법", "해결 방법"
        ]
        return any(keyword in message.lower() for keyword in procedure_keywords)
    
    def _generate_final_template(self, structure: Dict[str, Any], style: str, context: Dict[str, Any]) -> str:
        """최종 템플릿 생성"""
        try:
            from .gemini_client import GeminiClient
            gemini_client = GeminiClient()
            
            # 절차 우선 요구 감지
            message = context.get('message', '')
            needs_procedure_first = self._detect_procedure_request(message)
            
            if needs_procedure_first:
                template_prompt = f"""
**절대 규칙: 절차 설명이 먼저 와야 합니다!**

구조: {structure}
스타일: {style}
사용자 상황: {context}
질문: {message}

**절대적으로 지켜야 할 답변 순서 규칙:**
1. **절대 예시부터 시작하지 마세요**
2. **반드시 방법/절차를 먼저 설명**해야 합니다
3. 그 다음에 구체적인 예시나 사례를 제시합니다
4. 마지막에 실무 팁이나 주의사항을 제공합니다

**중요한 주의사항:**
- 사용자가 "방법", "절차", "과정", "어떻게"를 묻는 경우 반드시 절차 설명이 먼저 와야 합니다
- 예시나 사례는 절차 설명 이후에만 제시하세요
- 절차 설명 없이 바로 예시로 시작하는 것은 절대 금지입니다

**답변 구조 예시 (반드시 이 순서를 따르세요):**
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

**템플릿 생성 시 절대 금지사항:**
- 예시나 사례부터 시작하는 구조
- "상황:", "예시:", "구체적인 예시" 등으로 시작하는 구조
- 절차 설명 없이 바로 사례를 제시하는 구조

**중요: 위 순서를 반드시 지켜서 템플릿을 생성해주세요!**

템플릿만 생성해주세요:
"""
            else:
                # 기존 프롬프트 사용
                template_prompt = f"""다음 구조에 따라 답변 템플릿을 생성해주세요.

구조: {structure}
스타일: {style}
사용자 상황: {context}

템플릿 요구사항:
1. 자연스러운 한국어 표현
2. 법률 전문용어는 그대로 사용 (쉬운 말로 바꾸지 말 것)
3. 단계별로 명확한 구분
4. 사용자 친화적인 톤
5. **구체적인 예시나 내용은 제외하고 템플릿 구조만 생성**

템플릿만 생성해주세요:"""
            
            response = gemini_client.generate(template_prompt)
            return response.response
            
        except Exception as e:
            self.logger.error(f"Final template generation failed: {e}")
            return "해당 도메인의 답변을 생성하겠습니다."
    
    def _parse_json_response(self, llm_response: str) -> Dict[str, Any]:
        """JSON 응답 파싱"""
        try:
            if "{" in llm_response and "}" in llm_response:
                json_start = llm_response.find("{")
                json_end = llm_response.rfind("}") + 1
                json_str = llm_response[json_start:json_end]
                return json.loads(json_str)
        except Exception as e:
            self.logger.error(f"Failed to parse JSON response: {e}")
        
        return {}
