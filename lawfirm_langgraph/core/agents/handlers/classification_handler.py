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
from typing import Any, Dict, List, Optional, Tuple

from core.agents.parsers.response_parsers import ClassificationParser
from core.agents.workflow_constants import WorkflowConstants
from core.agents.workflow_utils import WorkflowUtils
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

    def fallback_complexity_classification(self, query: str) -> Tuple[QueryComplexity, bool]:
        """폴백 키워드 기반 복잡도 분류"""
        if not query:
            return QueryComplexity.MODERATE, True

        query_lower = query.lower()

        # 1. 인사말 체크
        simple_greetings = ["안녕", "고마워", "감사", "도움", "설명", "안녕하세요", "고마워요", "감사합니다"]
        if any(pattern in query_lower for pattern in simple_greetings):
            if len(query) < 20:
                return QueryComplexity.SIMPLE, False

        # 2. 법률 키워드 포함 시 검색 필수
        legal_keywords = ["법", "법령", "법률", "조", "항", "판례", "판결", "소송", "계약", "손해"]
        if any(keyword in query_lower for keyword in legal_keywords):
            # 복잡한 질문 키워드 체크
            complex_keywords = ["비교", "차이", "어떻게", "방법", "절차", "사례", "판례 비교"]
            if any(keyword in query_lower for keyword in complex_keywords):
                return QueryComplexity.COMPLEX, True
            else:
                return QueryComplexity.MODERATE, True

        # 기본값
        return QueryComplexity.MODERATE, True

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
