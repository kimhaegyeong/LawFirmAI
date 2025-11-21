# -*- coding: utf-8 -*-
"""
워크플로우 프롬프트 빌더
질문 유형 분류, 법률 분야 추출, 복잡도 평가 등의 프롬프트 생성
"""

import logging
try:
    from lawfirm_langgraph.core.utils.logger import get_logger
except ImportError:
    from core.utils.logger import get_logger
from typing import Any, Dict, List, Optional

logger = get_logger(__name__)


class WorkflowPromptBuilder:
    """워크플로우 프롬프트 빌더"""
    
    def __init__(self, logger: Optional[logging.Logger] = None):
        self.logger = logger or logging.getLogger(__name__)
    
    def build_question_type_prompt(self, query: str) -> str:
        """질문 유형 분류 프롬프트 생성"""
        return f"""다음 법률 질문의 유형을 분류해주세요.

질문: {query}

**중요**: 질문의 핵심 의도를 정확히 파악하여 가장 적합한 유형을 선택하세요.

다음 유형 중 하나를 선택하세요:
1. precedent_search - 판례, 사건, 법원 판결, 판시사항 관련
2. law_inquiry - 법률 조문, 법령, 규정의 내용을 묻는 질문
3. legal_advice - 법률 조언, 해석, 권리 구제 방법을 묻는 질문 (권리와 의무, 계약서 작성 등 포함)
4. procedure_guide - 법적 절차, 소송 방법, 대응 방법을 묻는 질문
5. term_explanation - 법률 용어의 정의나 의미를 묻는 질문
6. general_question - 범용적인 법률 질문

**특별 지침**:
- "권리와 의무", "계약서 작성", "법률 조언" 등이 포함된 질문은 legal_advice로 분류하되, 관련 법령 조문과 판례를 함께 검색해야 함을 명시하세요.
- 질문에 특정 법령명이나 조문이 언급되지 않아도, 질문의 핵심 개념과 관련된 법령을 추론하여 검색 범위를 제안하세요.
- 예: "임대차 계약서 작성" → 민법, 임대차보호법 관련 법령 조문과 판례 검색 필요

다음 형식으로 응답해주세요:
{{
    "question_type": "precedent_search" | "law_inquiry" | "legal_advice" | "procedure_guide" | "term_explanation" | "general_question",
    "confidence": 0.0-1.0,
    "reasoning": "판단 근거 (한국어)",
    "suggested_laws": ["관련 법령명1", "관련 법령명2"],
    "needs_both_law_and_precedent": true | false
}}
"""
    
    def build_legal_field_prompt(self, query: str, question_type: str) -> str:
        """법률 분야 추출 프롬프트 생성"""
        return f"""다음 질문과 질문 유형을 바탕으로 법률 분야를 추출해주세요.

질문: {query}
질문 유형: {question_type}

**중요**: 질문의 핵심 키워드를 분석하여 가장 적합한 법률 분야를 선택하세요.

법률 분야 예시:
- family_law (가족법): 이혼, 양육권, 상속, 부양 등
- civil_law (민법): 계약, 손해배상, 물권, 채권, 임대차, 부동산 등
- corporate_law (기업법): 회사법, 상법, 금융법 등
- intellectual_property (지적재산권): 특허, 상표, 저작권 등
- criminal_law (형법): 형사소송, 범죄 등
- labor_law (노동법): 근로법, 근로기준법 등
- administrative_law (행정법): 행정처분, 행정소송 등
- general (일반): 분류되지 않는 경우

**특별 지침**:
- "임대차", "임대인", "임차인" → civil_law (민법)
- "계약서 작성", "계약", "권리", "의무" → civil_law (민법)
- 질문에 여러 법률 분야가 관련될 수 있으면 가장 직접적으로 관련된 분야를 선택하세요.

다음 형식으로 응답해주세요:
{{
    "legal_field": "family_law" | "civil_law" | "corporate_law" | "intellectual_property" | "criminal_law" | "labor_law" | "administrative_law" | "general",
    "confidence": 0.0-1.0,
    "reasoning": "판단 근거 (한국어)",
    "related_laws": ["관련 법령명1", "관련 법령명2"]
}}
"""
    
    def build_complexity_prompt(self, query: str, question_type: str, legal_field: str) -> str:
        """복잡도 평가 프롬프트 생성"""
        return f"""다음 질문의 복잡도를 평가해주세요.

질문: {query}
질문 유형: {question_type}
법률 분야: {legal_field if legal_field else "미지정"}

다음 복잡도 중 하나를 선택하세요:
1. simple (간단):
   - 단순 인사말: "안녕하세요", "고마워요" 등
   - 매우 간단한 법률 용어 정의 (10자 이내, 일반 상식 수준)
   - 검색이 불필요한 경우

2. moderate (중간):
   - 특정 법령 조문 조회: "민법 제123조", "형법 제250조" 등
   - 단일 법률 개념 질문: "계약이란?", "손해배상의 요건은?"
   - 단일 판례 검색: "XX 사건 판례"
   - 검색이 필요하지만 단순한 경우

3. complex (복잡):
   - 비교 분석 질문: "계약 해지와 해제의 차이", "이혼과 재혼의 차이"
   - 절차/방법 질문: "이혼 절차는?", "소송 방법은?"
   - 다중 법령/판례 필요: "손해배상 관련 최근 판례와 법령"
   - 복합적 법률 분석: "계약 해지 시 위약금과 손해배상"
   - 검색과 분석이 모두 필요한 경우

다음 형식으로 응답해주세요:
{{
    "complexity": "simple" | "moderate" | "complex",
    "confidence": 0.0-1.0,
    "reasoning": "판단 근거 (한국어)"
}}
"""
    
    def build_search_necessity_prompt(self, query: str, complexity: str) -> str:
        """검색 필요성 판단 프롬프트 생성"""
        return f"""다음 질문의 검색 필요성을 판단해주세요.

질문: {query}
복잡도: {complexity}

검색이 필요한 경우:
- simple이 아닌 경우 (moderate 또는 complex)
- 법률 조문, 판례, 규정을 찾아야 하는 경우
- 최신 정보가 필요한 경우

검색이 불필요한 경우:
- simple 복잡도인 경우
- 일반적인 법률 상식으로 답변 가능한 경우
- 단순 인사말이나 정의 질문

다음 형식으로 응답해주세요:
{{
    "needs_search": true | false,
    "confidence": 0.0-1.0,
    "reasoning": "판단 근거 (한국어)"
}}
"""
    
    def build_classification_chain_steps(
        self,
        query: str,
        build_question_type_prompt_func=None,
        build_legal_field_prompt_func=None,
        build_complexity_prompt_func=None,
        build_search_necessity_prompt_func=None
    ) -> List[Dict[str, Any]]:
        """분류 체인 스텝 정의"""
        from core.processing.parsers.response_parsers import ClassificationParser
        
        chain_steps = []
        
        def build_question_type_prompt(prev_output, initial_input):
            query_value = prev_output.get("query") if isinstance(prev_output, dict) else (initial_input.get("query") if isinstance(initial_input, dict) else "")
            if not query_value:
                query_value = str(prev_output) if not isinstance(prev_output, dict) else ""
            if build_question_type_prompt_func:
                return build_question_type_prompt_func(query_value)
            return self.build_question_type_prompt(query_value)
        
        chain_steps.append({
            "name": "question_type_classification",
            "prompt_builder": build_question_type_prompt,
            "input_extractor": lambda prev: {"query": query} if isinstance(prev, dict) or not prev else prev,
            "output_parser": lambda response, prev: ClassificationParser.parse_question_type_response(response),
            "validator": lambda output: output and isinstance(output, dict) and "question_type" in output,
            "required": True
        })
        
        def build_legal_field_prompt(prev_output, initial_input):
            if not isinstance(prev_output, dict):
                return None
            question_type = prev_output.get("question_type", "")
            query_value = initial_input.get("query") if isinstance(initial_input, dict) else query
            if not question_type:
                return None
            if build_legal_field_prompt_func:
                return build_legal_field_prompt_func(query_value, question_type)
            return self.build_legal_field_prompt(query_value, question_type)
        
        chain_steps.append({
            "name": "legal_field_extraction",
            "prompt_builder": build_legal_field_prompt,
            "input_extractor": lambda prev: prev,
            "output_parser": lambda response, prev: ClassificationParser.parse_legal_field_response(response),
            "validator": lambda output: output is None or (isinstance(output, dict) and "legal_field" in output),
            "required": False,
            "skip_if": lambda prev: not isinstance(prev, dict) or not prev.get("question_type")
        })
        
        def build_complexity_prompt(prev_output, initial_input):
            if not isinstance(prev_output, dict):
                prev_output = {}
            question_type = prev_output.get("question_type", "")
            legal_field = prev_output.get("legal_field", "")
            query_value = initial_input.get("query") if isinstance(initial_input, dict) else query
            if not question_type:
                if isinstance(prev_output, dict):
                    question_type = prev_output.get("question_type", "")
            if build_complexity_prompt_func:
                return build_complexity_prompt_func(query_value, question_type, legal_field)
            return self.build_complexity_prompt(query_value, question_type, legal_field)
        
        chain_steps.append({
            "name": "complexity_assessment",
            "prompt_builder": build_complexity_prompt,
            "input_extractor": lambda prev: prev,
            "output_parser": lambda response, prev: ClassificationParser.parse_complexity_response(response),
            "validator": lambda output: output and isinstance(output, dict) and "complexity" in output,
            "required": True
        })
        
        def build_search_necessity_prompt(prev_output, initial_input):
            if not isinstance(prev_output, dict):
                return None
            complexity = prev_output.get("complexity", "")
            query_value = initial_input.get("query") if isinstance(initial_input, dict) else query
            if not complexity:
                return None
            if build_search_necessity_prompt_func:
                return build_search_necessity_prompt_func(query_value, complexity)
            return self.build_search_necessity_prompt(query_value, complexity)
        
        chain_steps.append({
            "name": "search_necessity_assessment",
            "prompt_builder": build_search_necessity_prompt,
            "input_extractor": lambda prev: prev,
            "output_parser": lambda response, prev: ClassificationParser.parse_search_necessity_response(response),
            "validator": lambda output: output is None or (isinstance(output, dict) and "needs_search" in output),
            "required": False,
            "skip_if": lambda prev: not isinstance(prev, dict) or not prev.get("complexity")
        })
        
        return chain_steps

