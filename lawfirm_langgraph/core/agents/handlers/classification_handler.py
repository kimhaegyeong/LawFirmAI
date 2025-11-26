# -*- coding: utf-8 -*-
"""
분류 핸들러 모듈
질문 유형, 복잡도, 법률 분야, 문서 유형 분류 로직을 독립 모듈로 분리
"""

import logging
import re
import time

# QueryComplexity는 legal_workflow_enhanced.py에 정의되어 있음
# 임시로 여기에 정의 (나중에 별도 모듈로 분리 가능)
from enum import Enum
from typing import Any, Dict, Optional, Tuple

try:
    from lawfirm_langgraph.core.agents.parsers.response_parsers import ClassificationParser
except ImportError:
    from core.agents.parsers.response_parsers import ClassificationParser
try:
    from lawfirm_langgraph.core.workflow.utils.workflow_constants import WorkflowConstants
except ImportError:
    from core.workflow.utils.workflow_constants import WorkflowConstants
try:
    from lawfirm_langgraph.core.workflow.utils.workflow_utils import WorkflowUtils
except ImportError:
    from core.workflow.utils.workflow_utils import WorkflowUtils
try:
    from core.classification.classifiers.question_classifier import QuestionType
except ImportError:
    # 호환성을 위한 fallback
    from core.services.question_classifier import QuestionType


class QueryComplexity(str, Enum):
    """질문 복잡도"""
    SIMPLE = "simple"
    MODERATE = "moderate"
    COMPLEX = "complex"


class ClassificationHandler:
    """
    질문 분류 및 복잡도 평가 클래스

    LLM 기반 질문 유형 분류, 복잡도 평가, 법률 분야 추출 등을 처리합니다.
    """

    def __init__(
        self,
        llm: Any,
        llm_fast: Any,
        stats: Optional[Dict[str, Any]] = None,
        logger: Optional[logging.Logger] = None
    ):
        """
        ClassificationHandler 초기화

        Args:
            llm: 메인 LLM 인스턴스
            llm_fast: 빠른 LLM 인스턴스 (간단한 질문용)
            stats: 통계 딕셔너리 (선택적)
            logger: 로거 (없으면 자동 생성)
        """
        self.llm = llm
        self.llm_fast = llm_fast
        self.stats = stats
        self.logger = logger or logging.getLogger(__name__)

        # 캐시 초기화
        self._complexity_cache: Dict[str, Tuple[QueryComplexity, bool]] = {}
        self._classification_cache: Dict[str, Tuple[QuestionType, float, QueryComplexity, bool]] = {}
        self._comprehensive_cache: Dict[str, Dict[str, Any]] = {}  # 통합 분류 결과 캐시

    def classify_with_llm(self, query: str, max_retries: int = 2) -> Tuple[QuestionType, float]:
        """LLM 기반 분류"""
        classification_prompt = f"""다음 법률 질문을 질문 유형으로 분류해주세요.

질문: {query}

분류 가능한 유형:
1. precedent_search - 판례, 사건, 법원 판결, 판시사항 관련
2. law_inquiry - 법률 조문, 법령, 규정의 내용을 묻는 질문
3. legal_advice - 법률 조언, 해석, 권리 구제 방법을 묻는 질문
4. procedure_guide - 법적 절차, 소송 방법, 대응 방법을 묻는 질문
5. term_explanation - 법률 용어의 정의나 의미를 묻는 질문
6. general_question - 범용적인 법률 질문

중요: 질문의 핵심 의도를 파악하여 가장 적합한 유형 하나만 선택하세요.
응답 형식: 유형명만 답변 (예: legal_advice)"""

        # 재시도 로직
        for attempt in range(max_retries):
            try:
                response = self.llm.invoke(classification_prompt)
                response_content = WorkflowUtils.extract_response_content(response)
                classification_result = response_content.strip().lower().replace(" ", "")

                question_type_mapping = {
                    "precedent_search": QuestionType.PRECEDENT_SEARCH,
                    "law_inquiry": QuestionType.LAW_INQUIRY,
                    "legal_advice": QuestionType.LEGAL_ADVICE,
                    "procedure_guide": QuestionType.PROCEDURE_GUIDE,
                    "term_explanation": QuestionType.TERM_EXPLANATION,
                    "general_question": QuestionType.GENERAL_QUESTION,
                }

                for key, question_type in question_type_mapping.items():
                    if key in classification_result or key.replace("_", "") in classification_result:
                        return question_type, WorkflowConstants.LLM_CLASSIFICATION_CONFIDENCE

                self.logger.warning(f"Could not classify query, LLM response: {classification_result}")
                return QuestionType.GENERAL_QUESTION, WorkflowConstants.DEFAULT_CONFIDENCE
            except Exception as e:
                if attempt < max_retries - 1:
                    self.logger.debug(f"LLM classification attempt {attempt + 1} failed, retrying: {e}")
                    continue
                else:
                    self.logger.warning(f"LLM classification failed after {max_retries} attempts: {e}")
                    return QuestionType.GENERAL_QUESTION, WorkflowConstants.DEFAULT_CONFIDENCE

        return QuestionType.GENERAL_QUESTION, WorkflowConstants.DEFAULT_CONFIDENCE

    def fallback_classification(self, query: str) -> Tuple[QuestionType, float]:
        """폴백 키워드 기반 분류"""
        self.logger.info("Using fallback keyword-based classification")
        query_lower = query.lower()

        if any(k in query_lower for k in ["판례", "사건", "판결"]):
            return QuestionType.PRECEDENT_SEARCH, WorkflowConstants.FALLBACK_CONFIDENCE
        elif any(k in query_lower for k in ["법률", "조문", "법령", "규정"]):
            return QuestionType.LAW_INQUIRY, WorkflowConstants.FALLBACK_CONFIDENCE
        elif any(k in query_lower for k in ["절차", "방법", "대응"]):
            return QuestionType.PROCEDURE_GUIDE, WorkflowConstants.FALLBACK_CONFIDENCE
        else:
            return QuestionType.GENERAL_QUESTION, WorkflowConstants.DEFAULT_CONFIDENCE

    def _hardcoded_complexity_classification(self, query: str) -> Tuple[QueryComplexity, bool]:
        """하드코딩된 키워드 기반 복잡도 분류 (폴백용)"""
        if not query:
            return QueryComplexity.MODERATE, True

        import re
        query_lower = query.lower()

        # 1. 법령 조문 패턴 감지 (제XX조, XX조 등)
        # 인코딩 문제로 깨진 경우에도 패턴 감지 가능하도록 개선
        article_patterns = [
            r'제\s*\d+\s*조',  # 제750조
            r'\d+\s*조',  # 750조
            r'제\s*\d+조',  # 제750조 (공백 없음)
        ]
        has_article = any(re.search(pattern, query) for pattern in article_patterns)

        # 2. 복잡한 질문 키워드 체크 (설명 요청, 비교 분석 등)
        complex_keywords = [
            "비교", "차이", "어떻게", "방법", "절차", "사례", "판례 비교",
            "설명해주세요", "설명해드려", "에 대해 설명", "설명해", "설명해줘",
            "알려주세요", "알려줘", "알려드려", "에 대해 알려",
            "해석", "의미", "내용", "규정", "조항"
        ]
        has_complex_keyword = any(keyword in query_lower for keyword in complex_keywords)

        # 3. 법률 키워드 포함 시 검색 필수
        legal_keywords = ["법", "법령", "법률", "조", "항", "판례", "판결", "소송", "계약", "손해"]
        has_legal_keyword = any(keyword in query_lower for keyword in legal_keywords)

        # 4. 복잡도 판단
        # 법령 조문 + 설명 요청 = complex
        if has_article and has_complex_keyword:
            self.logger.debug(f"Complex 질문 감지: 법령 조문 + 설명 요청 (query: {query[:50]}...)")
            return QueryComplexity.COMPLEX, True
        
        # 법령 조문만 있으면 moderate
        if has_article:
            self.logger.debug(f"Moderate 질문 감지: 법령 조문 포함 (query: {query[:50]}...)")
            return QueryComplexity.MODERATE, True
        
        # 법률 키워드 + 복잡한 질문 키워드 = complex
        if has_legal_keyword and has_complex_keyword:
            self.logger.debug(f"Complex 질문 감지: 법률 키워드 + 복잡한 질문 키워드 (query: {query[:50]}...)")
            return QueryComplexity.COMPLEX, True

        # 5. 법률 키워드만 있으면 moderate
        if has_legal_keyword:
            self.logger.debug(f"Moderate 질문 감지: 법률 키워드 포함 (query: {query[:50]}...)")
            return QueryComplexity.MODERATE, True

        # 6. 인사말 체크 (단순 인사말만 simple)
        simple_greetings = ["안녕", "고마워", "감사", "안녕하세요", "고마워요", "감사합니다"]
        # "설명"은 제외 (설명 요청은 complex)
        if any(pattern in query_lower for pattern in simple_greetings):
            if len(query) < 20 and not has_legal_keyword:
                return QueryComplexity.SIMPLE, False

        # 기본값 (moderate)
        return QueryComplexity.MODERATE, True

    def parse_comprehensive_classification_response(self, response: str) -> Optional[Dict[str, Any]]:
        """통합 분류 응답 파싱 (질문 유형, 복잡도, 법률 분야, 키워드 등 모든 정보)"""
        import json

        try:
            # JSON 추출 시도 (여러 중괄호 패턴 지원)
            json_match = re.search(r'\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\}', response, re.DOTALL)
            if json_match:
                json_str = json_match.group(0)
                result = json.loads(json_str)

                # 필수 필드 검증
                if "complexity" in result:
                    # 기본값 설정
                    if "needs_search" not in result:
                        result["needs_search"] = result.get("complexity") != "simple"
                    if "confidence" not in result:
                        result["confidence"] = 0.85
                    if "question_type" not in result:
                        result["question_type"] = "general_question"
                    if "legal_field" not in result:
                        result["legal_field"] = "general"
                    if "extracted_keywords" not in result:
                        result["extracted_keywords"] = []
                    if "legal_articles" not in result:
                        result["legal_articles"] = []
                    if "precedent_citations" not in result:
                        result["precedent_citations"] = []
                    if "search_strategy" not in result:
                        result["search_strategy"] = "hybrid"
                    if "expected_answer_length" not in result:
                        result["expected_answer_length"] = "medium"
                    if "priority" not in result:
                        result["priority"] = "medium"
                    
                    return result
        except json.JSONDecodeError:
            pass

        # 정규식으로 부분 추출 시도 (폴백)
        response_lower = response.lower()
        
        # 복잡도 추출
        complexity = "moderate"
        if '"simple"' in response_lower or "'simple'" in response_lower:
            complexity = "simple"
        elif '"complex"' in response_lower or "'complex'" in response_lower:
            complexity = "complex"

        # 질문 유형 추출
        question_type = "general_question"
        type_patterns = {
            "precedent_search": ["precedent_search", "precedent", "판례"],
            "law_inquiry": ["law_inquiry", "law", "법령", "조문"],
            "legal_advice": ["legal_advice", "advice", "조언"],
            "procedure_guide": ["procedure_guide", "procedure", "절차"],
            "term_explanation": ["term_explanation", "term", "용어"],
        }

        for qtype, patterns in type_patterns.items():
            if any(pattern in response_lower for pattern in patterns):
                question_type = qtype
                break

        return {
            "question_type": question_type,
            "complexity": complexity,
            "needs_search": complexity != "simple",
            "confidence": 0.85,
            "legal_field": "general",
            "extracted_keywords": [],
            "legal_articles": [],
            "precedent_citations": [],
            "search_strategy": "hybrid",
            "expected_answer_length": "medium",
            "priority": "medium",
            "reasoning": "정규식 추출 (JSON 파싱 실패)"
        }

    def classify_comprehensive_with_llm(self, query: str) -> Optional[Dict[str, Any]]:
        """
        LLM을 한 번 호출하여 질문에 대한 모든 정보를 통합 분류
        
        한 번의 질의로 추출하는 정보:
        - 질문 유형 (question_type)
        - 복잡도 (complexity)
        - 법률 분야 (legal_field)
        - 검색 필요성 (needs_search)
        - 신뢰도 (confidence)
        - 추출된 키워드 (extracted_keywords)
        - 법률 조문 참조 (legal_articles)
        - 판례 참조 (precedent_citations)
        - 검색 전략 제안 (search_strategy)
        - 예상 답변 길이 (expected_answer_length)
        - 우선순위 (priority)
        
        Returns:
            Optional[Dict[str, Any]]: 통합 분류 결과 (실패 시 None)
        """
        try:
            # 캐시 키 생성
            cache_key = f"comprehensive:{query}"

            # 캐시 확인
            if cache_key in self._comprehensive_cache:
                self.logger.debug(f"Using cached comprehensive classification for: {query[:50]}...")
                if self.stats:
                    self.stats['comprehensive_cache_hits'] = self.stats.get('comprehensive_cache_hits', 0) + 1
                return self._comprehensive_cache[cache_key]

            if self.stats:
                self.stats['comprehensive_cache_misses'] = self.stats.get('comprehensive_cache_misses', 0) + 1

            start_time = time.time()

            # 통합 프롬프트 생성
            comprehensive_prompt = f"""다음 법률 질문을 종합적으로 분석하여 모든 정보를 추출해주세요.

질문: {query}

## 분석 항목

### 1. 질문 유형 분류
다음 유형 중 가장 적합한 하나를 선택하세요:
- precedent_search: 판례, 사건, 법원 판결, 판시사항 관련
- law_inquiry: 법률 조문, 법령, 규정의 내용을 묻는 질문
- legal_advice: 법률 조언, 해석, 권리 구제 방법을 묻는 질문
- procedure_guide: 법적 절차, 소송 방법, 대응 방법을 묻는 질문
- term_explanation: 법률 용어의 정의나 의미를 묻는 질문
- general_question: 범용적인 법률 질문

### 2. 복잡도 분류
- simple: 단순 인사말, 매우 간단한 용어 정의 (검색 불필요)
- moderate: 특정 법령 조문 조회, 단일 법률 개념 질문
- complex: 비교 분석, 절차/방법, 다중 법령/판례 필요, 복합적 법률 분석

### 3. 법률 분야 추출
다음 분야 중 하나 이상을 선택하세요 (주요 분야만):
- civil: 민사법
- criminal: 형사법
- family: 가족법
- commercial: 상법
- labor: 노동법
- administrative: 행정법
- constitutional: 헌법
- intellectual_property: 지적재산권
- general: 일반

### 4. 검색 필요성 판단
- needs_search: true/false
- search_reasoning: 검색이 필요한 이유 또는 불필요한 이유 (한국어)

### 5. 추출된 키워드
질문에서 핵심 키워드 3-10개를 추출하세요. (배열 형식)

### 6. 법률 조문 참조
질문에서 언급된 법률 조문을 추출하세요 (예: "민법 제750조" → {{"law_name": "민법", "article_number": "750"}})
배열 형식으로, 없으면 빈 배열

### 7. 판례 참조
질문에서 언급된 판례나 사건을 추출하세요. 배열 형식으로, 없으면 빈 배열

### 8. 검색 전략 제안
- semantic: 의미적 검색 우선
- keyword: 키워드 검색 우선
- hybrid: 하이브리드 검색 권장
- statute_only: 법령 조문만 검색
- precedent_only: 판례만 검색

### 9. 예상 답변 길이
- short: 50-200자
- medium: 200-500자
- long: 500-1000자
- very_long: 1000자 이상

### 10. 우선순위
- high: 긴급한 법률 조언 필요
- medium: 일반적인 질문
- low: 단순 정보 조회

## 응답 형식 (JSON)

다음 형식으로 응답해주세요:
{{
    "question_type": "law_inquiry",
    "complexity": "moderate",
    "legal_field": "civil",
    "needs_search": true,
    "search_reasoning": "특정 법령 조문 내용을 확인하기 위해 검색이 필요함",
    "confidence": 0.90,
    "extracted_keywords": ["민법", "제750조", "손해배상", "불법행위"],
    "legal_articles": [
        {{"law_name": "민법", "article_number": "750"}}
    ],
    "precedent_citations": [],
    "search_strategy": "hybrid",
    "expected_answer_length": "medium",
    "priority": "medium",
    "reasoning": "민법 제750조 손해배상에 대한 설명 요청으로, 법령 조문 확인과 관련 법리 해석이 필요함"
}}

중요: 법률 정보의 정확성이 중요하므로, 불확실하면 moderate 이상으로 판단하세요."""

            # LLM 호출 (빠른 모델 사용)
            llm = self.llm_fast if self.llm_fast else self.llm
            response = llm.invoke(comprehensive_prompt)
            response_content = WorkflowUtils.extract_response_content(response)

            # JSON 파싱
            comprehensive_result = self.parse_comprehensive_classification_response(response_content)

            if comprehensive_result:
                # QueryComplexity enum으로 변환
                complexity_mapping = {
                    "simple": QueryComplexity.SIMPLE,
                    "moderate": QueryComplexity.MODERATE,
                    "complex": QueryComplexity.COMPLEX,
                }
                complexity_str = comprehensive_result.get("complexity", "moderate")
                comprehensive_result["complexity"] = complexity_mapping.get(complexity_str, QueryComplexity.MODERATE)

                # QuestionType enum으로 변환
                question_type_mapping = {
                    "precedent_search": QuestionType.PRECEDENT_SEARCH,
                    "law_inquiry": QuestionType.LAW_INQUIRY,
                    "legal_advice": QuestionType.LEGAL_ADVICE,
                    "procedure_guide": QuestionType.PROCEDURE_GUIDE,
                    "term_explanation": QuestionType.TERM_EXPLANATION,
                    "general_question": QuestionType.GENERAL_QUESTION,
                }
                question_type_str = comprehensive_result.get("question_type", "general_question")
                comprehensive_result["question_type"] = question_type_mapping.get(
                    question_type_str, QuestionType.GENERAL_QUESTION
                )

                elapsed_time = time.time() - start_time

                reasoning = comprehensive_result.get("reasoning", "")
                self.logger.info(
                    f"✅ [COMPREHENSIVE LLM CLASSIFICATION] "
                    f"question_type={comprehensive_result['question_type'].value}, "
                    f"complexity={comprehensive_result['complexity'].value}, "
                    f"legal_field={comprehensive_result.get('legal_field', 'N/A')}, "
                    f"keywords={len(comprehensive_result.get('extracted_keywords', []))}, "
                    f"confidence={comprehensive_result.get('confidence', 0.0):.2f}, "
                    f"reasoning: {reasoning[:100] if reasoning else 'N/A'}... "
                    f"(시간: {elapsed_time:.3f}s)"
                )

                # 캐시에 저장 (최대 100개)
                if len(self._comprehensive_cache) >= 100:
                    # 오래된 항목 제거 (FIFO)
                    oldest_key = next(iter(self._comprehensive_cache))
                    del self._comprehensive_cache[oldest_key]

                self._comprehensive_cache[cache_key] = comprehensive_result

                # 성능 메트릭 업데이트
                if self.stats:
                    count = self.stats.get('comprehensive_classifications', 0) + 1
                    self.stats['comprehensive_classifications'] = count
                    current_avg = self.stats.get('avg_comprehensive_classification_time', 0.0)
                    self.stats['avg_comprehensive_classification_time'] = (current_avg * (count - 1) + elapsed_time) / count

                return comprehensive_result
            else:
                self.logger.warning("Comprehensive classification parsing failed")
                return None

        except Exception as e:
            self.logger.warning(f"Comprehensive classification failed: {e}")
            if self.stats:
                self.stats['comprehensive_classification_errors'] = self.stats.get('comprehensive_classification_errors', 0) + 1
            return None

    def fallback_complexity_classification(self, query: str) -> Tuple[QueryComplexity, bool]:
        """
        LLM 기반 통합 분류를 사용하는 복잡도 분류 (폴백 메커니즘 포함)
        
        한 번의 LLM 호출로 복잡도와 검색 필요성을 포함한 모든 정보를 추출합니다.
        LLM 실패 시 하드코딩 로직을 폴백으로 사용합니다.
        
        Returns:
            Tuple[QueryComplexity, bool]: (복잡도, 검색 필요 여부)
        """
        if not query:
            return QueryComplexity.MODERATE, True

        # 1단계: 캐시 확인
        cache_key = f"comprehensive:{query}"
        if cache_key in self._comprehensive_cache:
            result = self._comprehensive_cache[cache_key]
            complexity = result.get("complexity")
            needs_search = result.get("needs_search", True)
            
            # QueryComplexity enum인지 확인
            if isinstance(complexity, QueryComplexity):
                return complexity, needs_search
            # 문자열인 경우 변환
            elif isinstance(complexity, str):
                complexity_mapping = {
                    "simple": QueryComplexity.SIMPLE,
                    "moderate": QueryComplexity.MODERATE,
                    "complex": QueryComplexity.COMPLEX,
                }
                return complexity_mapping.get(complexity, QueryComplexity.MODERATE), needs_search

        # 2단계: LLM 기반 통합 분류 시도
        try:
            comprehensive_result = self.classify_comprehensive_with_llm(query)
            
            if comprehensive_result:
                complexity = comprehensive_result.get("complexity")
                needs_search = comprehensive_result.get("needs_search", True)
                
                # QueryComplexity enum인지 확인
                if isinstance(complexity, QueryComplexity):
                    return complexity, needs_search
                # 문자열인 경우 변환
                elif isinstance(complexity, str):
                    complexity_mapping = {
                        "simple": QueryComplexity.SIMPLE,
                        "moderate": QueryComplexity.MODERATE,
                        "complex": QueryComplexity.COMPLEX,
                    }
                    return complexity_mapping.get(complexity, QueryComplexity.MODERATE), needs_search
        except Exception as e:
            self.logger.warning(f"LLM comprehensive classification failed: {e}, using hardcoded fallback")
            if self.stats:
                self.stats['comprehensive_fallback_count'] = self.stats.get('comprehensive_fallback_count', 0) + 1
        
        # 3단계: 하드코딩 폴백 (기존 로직 유지)
        self.logger.debug("Using hardcoded fallback for complexity classification")
        return self._hardcoded_complexity_classification(query)

    def parse_complexity_response(self, response: str) -> Dict[str, Any]:
        """복잡도 평가 응답 파싱"""
        import json

        try:
            # JSON 부분 추출 (중괄호 감싸진 부분)
            json_match = re.search(r'\{[^{}]*"complexity"[^{}]*\}', response, re.DOTALL)
            if json_match:
                json_str = json_match.group(0)
                result = json.loads(json_str)

                # 검증
                if "complexity" in result:
                    # needs_search 기본값 설정
                    if "needs_search" not in result:
                        # simple이 아닌 경우 기본적으로 검색 필요
                        result["needs_search"] = result.get("complexity") != "simple"
                    return result
        except json.JSONDecodeError:
            # JSON 파싱 실패 시 정규식으로 추출 시도
            pass

        # 정규식으로 복잡도 추출 (폴백)
        response_lower = response.lower()
        if '"complexity":' in response_lower or "'complexity':" in response_lower:
            if '"simple"' in response_lower or "'simple'" in response_lower:
                return {
                    "complexity": "simple",
                    "needs_search": False,
                    "reasoning": "LLM 응답에서 simple 추출"
                }
            elif '"complex"' in response_lower or "'complex'" in response_lower:
                return {
                    "complexity": "complex",
                    "needs_search": True,
                    "reasoning": "LLM 응답에서 complex 추출"
                }

        # 기본값 (moderate)
        return {
            "complexity": "moderate",
            "needs_search": True,
            "reasoning": "LLM 응답 파싱 실패로 기본값 사용"
        }

    def parse_unified_classification_response(self, response: str) -> Optional[Dict[str, Any]]:
        """통합 분류 응답 파싱 (질문 유형 + 복잡도)"""
        import json

        try:
            # JSON 부분 추출 (question_type과 complexity 모두 포함)
            json_match = re.search(r'\{[^{}]*"question_type"[^{}]*"complexity"[^{}]*\}', response, re.DOTALL)
            if not json_match:
                # question_type과 complexity가 다른 순서일 수 있음
                json_match = re.search(
                    r'\{[^{}]*(?:question_type|complexity)[^{}]*(?:question_type|complexity)[^{}]*\}',
                    response,
                    re.DOTALL
                )
            if json_match:
                json_str = json_match.group(0)
                result = json.loads(json_str)

                # 필수 필드 검증
                if "question_type" in result and "complexity" in result:
                    # 기본값 설정
                    if "confidence" not in result:
                        result["confidence"] = 0.85
                    if "needs_search" not in result:
                        result["needs_search"] = result.get("complexity") != "simple"
                    return result
        except json.JSONDecodeError:
            pass

        # 정규식으로 추출 시도 (폴백)
        response_lower = response.lower()

        # 질문 유형 추출
        question_type = "general_question"
        type_patterns = {
            "precedent_search": ["precedent_search", "precedent", "판례"],
            "law_inquiry": ["law_inquiry", "law", "법령", "조문"],
            "legal_advice": ["legal_advice", "advice", "조언"],
            "procedure_guide": ["procedure_guide", "procedure", "절차"],
            "term_explanation": ["term_explanation", "term", "용어"],
        }

        for qtype, patterns in type_patterns.items():
            if any(pattern in response_lower for pattern in patterns):
                question_type = qtype
                break

        # 복잡도 추출
        complexity = "moderate"
        if '"simple"' in response_lower or "'simple'" in response_lower or "simple" in response_lower:
            complexity = "simple"
        elif '"complex"' in response_lower or "'complex'" in response_lower or "complex" in response_lower:
            complexity = "complex"

        return {
            "question_type": question_type,
            "confidence": 0.85,
            "complexity": complexity,
            "needs_search": complexity != "simple",
            "reasoning": "정규식 추출 (JSON 파싱 실패)"
        }

    def parse_legal_field_response(self, response: str) -> Optional[Dict[str, Any]]:
        """법률 분야 추출 응답 파싱"""
        try:
            import json

            # JSON 추출
            json_match = re.search(r'\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\}', response, re.DOTALL)
            if json_match:
                json_str = json_match.group(0)
                return json.loads(json_str)

            return None
        except Exception as e:
            self.logger.warning(f"Failed to parse legal field response: {e}")
            return None

    def parse_document_type_response(self, response: str) -> Dict[str, Any]:
        """문서 유형 확인 응답 파싱"""
        try:
            import json

            # JSON 추출
            json_match = re.search(r'\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\}', response, re.DOTALL)
            if json_match:
                json_str = json_match.group(0)
                return json.loads(json_str)

            # 기본값
            return {
                "document_type": "general_legal_document",
                "confidence": 0.7,
                "reasoning": "JSON 파싱 실패"
            }
        except Exception as e:
            self.logger.warning(f"Failed to parse document type response: {e}")
            return {
                "document_type": "general_legal_document",
                "confidence": 0.7,
                "reasoning": f"파싱 에러: {e}"
            }

    def classify_complexity_with_llm(self, query: str, query_type: str = "") -> Tuple[QueryComplexity, bool]:
        """
        LLM 기반 복잡도 분류

        Args:
            query: 사용자 질문
            query_type: 이미 분류된 질문 유형 (선택적)

        Returns:
            Tuple[QueryComplexity, bool]: (복잡도, 검색 필요 여부)
        """
        try:
            # 캐시 키 생성
            cache_key = f"complexity:{query}:{query_type}"

            # 캐시 확인
            if cache_key in self._complexity_cache:
                self.logger.debug(f"Using cached complexity classification for: {query[:50]}...")
                if self.stats:
                    self.stats['complexity_cache_hits'] = self.stats.get('complexity_cache_hits', 0) + 1
                return self._complexity_cache[cache_key]

            if self.stats:
                self.stats['complexity_cache_misses'] = self.stats.get('complexity_cache_misses', 0) + 1

            start_time = time.time()

            # 프롬프트 생성
            complexity_prompt = f"""다음 법률 질문의 복잡도를 판단해주세요.

질문: {query}
질문 유형: {query_type if query_type else "미분류"}

복잡도 기준:
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

중요 판단 기준:
- 법률 키워드(법, 법령, 조, 판례, 계약, 소송 등) 포함 시 → moderate 이상
- 법률 용어 정의 질문이라도 검색으로 확인 필요 → moderate 이상
- 비교, 절차, 방법 등 복잡한 분석 필요 → complex
- 인사말만 → simple (검색 불필요)

응답 형식 (JSON):
{{
    "complexity": "simple" | "moderate" | "complex",
    "needs_search": true | false,
    "reasoning": "판단 근거 (한국어)"
}}

중요: 법률 정보의 정확성이 중요하므로, 불확실하면 moderate 이상으로 판단하세요."""

            # LLM 호출 (빠른 모델 사용)
            llm = self.llm_fast if self.llm_fast else self.llm
            response = llm.invoke(complexity_prompt)
            response_content = WorkflowUtils.extract_response_content(response)

            # JSON 파싱
            complexity_result = ClassificationParser.parse_complexity_response(response_content)

            if complexity_result:
                complexity_str = complexity_result.get("complexity", "moderate")
                needs_search = complexity_result.get("needs_search", True)
                reasoning = complexity_result.get("reasoning", "")

                # QueryComplexity enum으로 변환
                complexity_mapping = {
                    "simple": QueryComplexity.SIMPLE,
                    "moderate": QueryComplexity.MODERATE,
                    "complex": QueryComplexity.COMPLEX,
                }
                complexity = complexity_mapping.get(complexity_str, QueryComplexity.MODERATE)

                elapsed_time = time.time() - start_time

                self.logger.info(
                    f"✅ [LLM COMPLEXITY CLASSIFICATION] "
                    f"complexity={complexity.value}, needs_search={needs_search}, "
                    f"reasoning: {reasoning[:100] if reasoning else 'N/A'}... "
                    f"(시간: {elapsed_time:.3f}s)"
                )

                result = (complexity, needs_search)

                # 캐시에 저장 (최대 100개)
                if len(self._complexity_cache) >= 100:
                    # 오래된 항목 제거 (FIFO)
                    oldest_key = next(iter(self._complexity_cache))
                    del self._complexity_cache[oldest_key]

                self._complexity_cache[cache_key] = result

                # 성능 메트릭 업데이트
                if self.stats:
                    self.stats['llm_complexity_classifications'] = self.stats.get('llm_complexity_classifications', 0) + 1
                    current_avg = self.stats.get('avg_complexity_classification_time', 0.0)
                    count = self.stats.get('llm_complexity_classifications', 1)
                    # 이동 평균 계산
                    self.stats['avg_complexity_classification_time'] = (current_avg * (count - 1) + elapsed_time) / count

                return result
            else:
                # 파싱 실패 시 폴백
                self.logger.warning("LLM complexity classification parsing failed, using fallback")
                if self.stats:
                    self.stats['complexity_fallback_count'] = self.stats.get('complexity_fallback_count', 0) + 1
                return self.fallback_complexity_classification(query)

        except Exception as e:
            self.logger.warning(f"LLM complexity classification failed: {e}, using fallback")
            if self.stats:
                self.stats['complexity_fallback_count'] = self.stats.get('complexity_fallback_count', 0) + 1
            return self.fallback_complexity_classification(query)

    def classify_query_and_complexity_with_llm(self, query: str) -> Tuple[QuestionType, float, QueryComplexity, bool]:
        """
        LLM을 한 번 호출하여 질문 유형과 복잡도를 동시에 분류

        Returns:
            Tuple[QuestionType, float, QueryComplexity, bool]: (질문 유형, 신뢰도, 복잡도, 검색 필요 여부)
        """
        try:
            # 캐시 키 생성
            cache_key = f"query_and_complexity:{query}"

            # 캐시 확인
            if cache_key in self._classification_cache:
                self.logger.debug(f"Using cached unified classification for: {query[:50]}...")
                if self.stats:
                    self.stats['complexity_cache_hits'] = self.stats.get('complexity_cache_hits', 0) + 1
                return self._classification_cache[cache_key]

            if self.stats:
                self.stats['complexity_cache_misses'] = self.stats.get('complexity_cache_misses', 0) + 1

            start_time = time.time()

            # 통합 프롬프트 생성
            unified_prompt = f"""다음 법률 질문을 분석하여 질문 유형과 복잡도를 동시에 분류해주세요.

질문: {query}

## 1. 질문 유형 분류

다음 유형 중 하나를 선택하세요:
1. precedent_search - 판례, 사건, 법원 판결, 판시사항 관련
2. law_inquiry - 법률 조문, 법령, 규정의 내용을 묻는 질문
3. legal_advice - 법률 조언, 해석, 권리 구제 방법을 묻는 질문
4. procedure_guide - 법적 절차, 소송 방법, 대응 방법을 묻는 질문
5. term_explanation - 법률 용어의 정의나 의미를 묻는 질문
6. general_question - 범용적인 법률 질문

## 2. 복잡도 분류

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

## 중요 판단 기준

- 법률 키워드(법, 법령, 조, 판례, 계약, 소송 등) 포함 시 → moderate 이상
- 법률 용어 정의 질문이라도 검색으로 확인 필요 → moderate 이상
- 비교, 절차, 방법 등 복잡한 분석 필요 → complex
- 인사말만 → simple (검색 불필요)

## 응답 형식 (JSON)

다음 형식으로 응답해주세요:
{{
    "question_type": "precedent_search" | "law_inquiry" | "legal_advice" | "procedure_guide" | "term_explanation" | "general_question",
    "confidence": 0.0-1.0,
    "complexity": "simple" | "moderate" | "complex",
    "needs_search": true | false,
    "reasoning": "판단 근거 (한국어)"
}}

중요: 법률 정보의 정확성이 중요하므로, 불확실하면 moderate 이상으로 판단하세요."""

            # LLM 호출 (빠른 모델 사용)
            llm = self.llm_fast if self.llm_fast else self.llm
            response = llm.invoke(unified_prompt)
            response_content = WorkflowUtils.extract_response_content(response)

            # JSON 파싱
            result = self.parse_unified_classification_response(response_content)

            if result:
                # QuestionType 변환
                question_type_mapping = {
                    "precedent_search": QuestionType.PRECEDENT_SEARCH,
                    "law_inquiry": QuestionType.LAW_INQUIRY,
                    "legal_advice": QuestionType.LEGAL_ADVICE,
                    "procedure_guide": QuestionType.PROCEDURE_GUIDE,
                    "term_explanation": QuestionType.TERM_EXPLANATION,
                    "general_question": QuestionType.GENERAL_QUESTION,
                }
                question_type = question_type_mapping.get(
                    result.get("question_type", "general_question"),
                    QuestionType.GENERAL_QUESTION
                )
                confidence = float(result.get("confidence", 0.85))

                # QueryComplexity 변환
                complexity_mapping = {
                    "simple": QueryComplexity.SIMPLE,
                    "moderate": QueryComplexity.MODERATE,
                    "complex": QueryComplexity.COMPLEX,
                }
                complexity = complexity_mapping.get(
                    result.get("complexity", "moderate"),
                    QueryComplexity.MODERATE
                )
                needs_search = result.get("needs_search", True)
                reasoning = result.get("reasoning", "")

                elapsed_time = time.time() - start_time

                self.logger.info(
                    f"✅ [UNIFIED LLM CLASSIFICATION] "
                    f"question_type={question_type.value}, complexity={complexity.value}, "
                    f"needs_search={needs_search}, confidence={confidence:.2f}, "
                    f"reasoning: {reasoning[:100] if reasoning else 'N/A'}... "
                    f"(시간: {elapsed_time:.3f}s)"
                )

                result_tuple = (question_type, confidence, complexity, needs_search)

                # 캐시에 저장 (최대 100개)
                if len(self._classification_cache) >= 100:
                    oldest_key = next(iter(self._classification_cache))
                    del self._classification_cache[oldest_key]

                self._classification_cache[cache_key] = result_tuple

                # 성능 메트릭 업데이트
                if self.stats:
                    self.stats['llm_complexity_classifications'] = self.stats.get('llm_complexity_classifications', 0) + 1
                    current_avg = self.stats.get('avg_complexity_classification_time', 0.0)
                    count = self.stats.get('llm_complexity_classifications', 1)
                    self.stats['avg_complexity_classification_time'] = (current_avg * (count - 1) + elapsed_time) / count

                return result_tuple
            else:
                # 파싱 실패 시 폴백
                self.logger.warning("Unified classification parsing failed, using fallback")
                if self.stats:
                    self.stats['complexity_fallback_count'] = self.stats.get('complexity_fallback_count', 0) + 1
                question_type, confidence = self.fallback_classification(query)
                complexity, needs_search = self.fallback_complexity_classification(query)
                return (question_type, confidence, complexity, needs_search)

        except Exception as e:
            self.logger.warning(f"Unified classification failed: {e}, using fallback")
            if self.stats:
                self.stats['complexity_fallback_count'] = self.stats.get('complexity_fallback_count', 0) + 1
            question_type, confidence = self.fallback_classification(query)
            complexity, needs_search = self.fallback_complexity_classification(query)
            return (question_type, confidence, complexity, needs_search)
