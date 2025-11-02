# -*- coding: utf-8 -*-
"""
워크플로우 라우팅 모듈
LangGraph 워크플로우의 조건부 라우팅 로직을 독립 모듈로 분리
"""

import logging
from typing import Any, Dict, Optional

from core.agents.state_definitions import LegalWorkflowState
from core.agents.workflow_constants import (
    QualityThresholds,
    RetryConfig,
    WorkflowConstants,
)
from core.agents.workflow_utils import WorkflowUtils


class QueryComplexity:
    """질문 복잡도 Enum 대체 클래스"""
    SIMPLE = "simple"
    MODERATE = "moderate"
    COMPLEX = "complex"
    MULTI_HOP = "multi_hop"


class WorkflowRoutes:
    """
    워크플로우 라우팅 클래스

    LangGraph 워크플로우의 조건부 엣지 함수들을 제공합니다.
    """

    def __init__(
        self,
        retry_manager: Any,
        answer_generator: Any = None,
        ai_keyword_generator: Any = None,
        logger: Optional[logging.Logger] = None
    ):
        """
        WorkflowRoutes 초기화

        Args:
            retry_manager: RetryCounterManager 인스턴스
            answer_generator: AnswerGenerator 인스턴스 (fallback 답변 생성용)
            ai_keyword_generator: AIKeywordGenerator 인스턴스 (키워드 확장용)
            logger: 로거 인스턴스 (없으면 자동 생성)
        """
        self.retry_manager = retry_manager
        self.answer_generator = answer_generator
        self.ai_keyword_generator = ai_keyword_generator
        self.logger = logger or logging.getLogger(__name__)

    def route_by_complexity(self, state: LegalWorkflowState) -> str:
        """복잡도에 따라 라우팅"""
        # 디버깅: state 구조 확인
        state_keys = list(state.keys()) if isinstance(state, dict) else []
        print(f"[DEBUG] _route_by_complexity: state keys={state_keys[:15]}")
        print(f"[DEBUG] _route_by_complexity: state type={type(state).__name__}")

        # 여러 방법으로 complexity 확인 (우선순위: 최상위 > classification 그룹 > _get_state_value)
        complexity = None

        # 1. 최상위 레벨 직접 확인 (가장 빠르고 확실함)
        if isinstance(state, dict) and "query_complexity" in state:
            complexity = state["query_complexity"]
            print(f"[DEBUG] _route_by_complexity: [1] via top-level direct={complexity}")

        # 2. common 그룹 확인 (reducer가 보존하는 그룹)
        if not complexity and isinstance(state, dict) and "common" in state:
            if isinstance(state["common"], dict):
                complexity = state["common"].get("query_complexity")
                print(f"[DEBUG] _route_by_complexity: [2] via common group={complexity}")

        # 3. metadata 확인 (reducer가 보존하는 그룹)
        if not complexity and isinstance(state, dict) and "metadata" in state:
            if isinstance(state["metadata"], dict):
                complexity = state["metadata"].get("query_complexity")
                print(f"[DEBUG] _route_by_complexity: [3] via metadata={complexity}")

        # 4. classification 그룹 확인
        if not complexity and isinstance(state, dict) and "classification" in state:
            if isinstance(state["classification"], dict):
                complexity = state["classification"].get("query_complexity")
                print(f"[DEBUG] _route_by_complexity: [4] via classification group={complexity}")

        # 5. _get_state_value 사용 (마지막 시도)
        if not complexity:
            complexity = WorkflowUtils.get_state_value(state, "query_complexity", None)
            print(f"[DEBUG] _route_by_complexity: [5] via _get_state_value={complexity}")

        # 기본값 (문자열로 저장됨)
        if not complexity:
            complexity = QueryComplexity.MODERATE
            print(f"[DEBUG] _route_by_complexity: [4] using default={complexity}")

        # Enum인 경우 값으로 변환
        if hasattr(complexity, 'value'):
            complexity = complexity.value

        # 디버깅 로그
        self.logger.info(f"🔀 [ROUTE] 복잡도: {complexity}, 라우팅 결정 중...")
        print(f"[DEBUG] _route_by_complexity: FINAL complexity={complexity}")

        # 문자열 비교 (state에 저장된 값은 문자열)
        if complexity == QueryComplexity.SIMPLE or complexity == "simple":
            self.logger.info(f"✅ [ROUTE] 간단한 질문 → direct_answer")
            print(f"[DEBUG] _route_by_complexity: ✅ returning 'simple'")
            return "simple"
        elif complexity == QueryComplexity.MODERATE or complexity == "moderate":
            self.logger.info(f"🔄 [ROUTE] 중간 질문 → classification_parallel")
            print(f"[DEBUG] _route_by_complexity: 🔄 returning 'moderate'")
            return "moderate"
        else:
            self.logger.info(f"🔀 [ROUTE] 복잡한 질문 → classification_parallel")
            print(f"[DEBUG] _route_by_complexity: 🔀 returning 'complex'")
            return "complex"

    def route_expert(self, state: LegalWorkflowState) -> LegalWorkflowState:
        """전문가 서브그래프로 라우팅"""
        try:
            # 복잡도 평가
            complexity = self.assess_complexity(state)
            state["complexity_level"] = complexity

            # 전문가 라우팅 필요 여부
            requires_expert = (
                complexity == "complex" and
                state.get("legal_field") in ["family", "corporate", "intellectual_property"]
            )
            state["requires_expert"] = requires_expert

            if requires_expert:
                # 전문가 서브그래프 결정
                expert_map = {
                    "family": "family_law_expert",
                    "corporate": "corporate_law_expert",
                    "intellectual_property": "ip_law_expert"
                }
                state["expert_subgraph"] = expert_map.get(state.get("legal_field"))
                self.logger.info(f"Routing to expert: {state['expert_subgraph']}")
            else:
                state["expert_subgraph"] = None

            return state

        except Exception as e:
            self.logger.error(f"전문가 라우팅 중 오류: {e}")
            state["complexity_level"] = "simple"
            state["requires_expert"] = False
            state["expert_subgraph"] = None
            return state

    def assess_complexity(self, state: LegalWorkflowState) -> str:
        """질문 복잡도 평가"""
        # 복잡도 지표들
        query = WorkflowUtils.get_state_value(state, "query", "")
        extracted_keywords = WorkflowUtils.get_state_value(state, "extracted_keywords", [])
        indicators = {
            "query_length": len(query),
            "num_keywords": len(extracted_keywords),
            "has_document": bool(WorkflowUtils.get_state_value(state, "uploaded_document")),
            "high_urgency": state.get("urgency_level") in ["high", "critical"],
            "multiple_legal_issues": len(state.get("potential_issues", [])) > 2
        }

        complexity_score = 0

        # 점수 계산
        if indicators["query_length"] > 200:
            complexity_score += 2
        if indicators["num_keywords"] > 10:
            complexity_score += 2
        if indicators["has_document"]:
            complexity_score += 3
        if indicators["high_urgency"]:
            complexity_score += 1
        if indicators["multiple_legal_issues"]:
            complexity_score += 2

        # 복잡도 판정
        if complexity_score >= 7:
            return "complex"
        elif complexity_score >= 4:
            return "medium"
        else:
            return "simple"

    def should_analyze_document(self, state: LegalWorkflowState) -> str:
        """문서 분석 필요 여부 결정"""
        if state.get("uploaded_document"):
            return "analyze"
        return "skip"

    def should_skip_search(self, state: LegalWorkflowState) -> str:
        """검색 실행 건너뛰기 여부 결정 (캐시 히트)"""
        cache_hit = WorkflowUtils.get_state_value(state, "search_cache_hit", False)
        if cache_hit:
            return "skip"
        return "continue"

    def should_skip_search_adaptive(self, state: LegalWorkflowState) -> str:
        """Adaptive RAG: 질문 복잡도에 따라 검색 스킵 결정"""
        # 캐시 히트 체크 (기존 로직)
        cache_hit = WorkflowUtils.get_state_value(state, "search_cache_hit", False)
        if cache_hit:
            return "skip"

        # 복잡도 기반 스킵 결정
        needs_search = WorkflowUtils.get_state_value(state, "needs_search", True)
        complexity = WorkflowUtils.get_state_value(state, "query_complexity", QueryComplexity.MODERATE)

        # Enum인 경우 값으로 변환
        if hasattr(complexity, 'value'):
            complexity = complexity.value

        if not needs_search or complexity == QueryComplexity.SIMPLE or complexity == "simple":
            self.logger.info(f"⏭️ 검색 스킵: 간단한 질문 (복잡도: {complexity})")
            return "skip"

        return "continue"

    def should_expand_keywords_ai(self, state: LegalWorkflowState) -> str:
        """AI 키워드 확장 여부 결정"""
        # AI 확장 조건:
        # 1. AI 키워드 생성기가 초기화되어 있는가
        # 2. 추출된 키워드가 충분히 있는가
        # 3. 질문 복잡도가 충분히 높은가

        if not self.ai_keyword_generator:
            return "skip"

        keywords = WorkflowUtils.get_state_value(state, "extracted_keywords", [])
        if len(keywords) < 3:
            return "skip"

        # 복잡한 질문인 경우 확장
        query_type = WorkflowUtils.get_state_value(state, "query_type", "")
        complex_types = ["precedent_search", "law_inquiry", "legal_advice"]

        if query_type in complex_types:
            return "expand"

        return "skip"

    def should_retry_generation(self, state: LegalWorkflowState) -> str:
        """
        1단계: 답변 생성 후 재시도 여부 결정

        중요: 조건부 엣지 함수는 상태를 수정할 수 없으므로,
        카운터 증가는 prepare_search_query 노드에서 처리합니다.
        여기서는 재시도 여부만 판단합니다.

        재시도 전략:
        - 최대 2회 재시도
        - 에러가 있거나 답변이 너무 짧으면 문서 검색부터 재시도
        - 재시도 횟수 초과 시 포맷팅으로 진행

        Returns:
            "validate": 정상 진행
            "retry_search": 검색부터 재시도
            "format": 포맷팅으로 진행
        """
        # 재시도 카운터 조회 (통합 헬퍼 사용)
        retry_counts = self.retry_manager.get_retry_counts(state)
        generation_retry_count = retry_counts["generation"]
        total_retry_count = retry_counts["total"]

        # 전역 재시도 횟수 체크
        if total_retry_count >= RetryConfig.MAX_TOTAL_RETRIES:
            self.logger.warning(
                f"Maximum total retry count ({RetryConfig.MAX_TOTAL_RETRIES}) reached. "
                "Proceeding to formatting."
            )
            return "format"

        # 생성 재시도 횟수 체크
        if generation_retry_count >= RetryConfig.MAX_GENERATION_RETRIES:
            self.logger.warning(
                f"Maximum generation retry count ({RetryConfig.MAX_GENERATION_RETRIES}) reached. "
                "Proceeding to formatting."
            )
            return "format"

        # 이미 재시도한 경우 즉시 종료 (무한 루프 방지)
        if generation_retry_count > 0:
            self.logger.info(
                f"✅ [RETRY LIMIT] Already retried {generation_retry_count}/{RetryConfig.MAX_GENERATION_RETRIES} times. "
                "Proceeding to formatting to prevent infinite loop."
            )
            return "format"

        # 답변 및 에러 확인
        answer = WorkflowUtils.normalize_answer(WorkflowUtils.get_state_value(state, "answer", ""))
        errors = WorkflowUtils.get_state_value(state, "errors", [])

        answer_len = len(answer)
        has_errors = len(errors) > 0
        is_short_answer = answer_len < WorkflowConstants.MIN_ANSWER_LENGTH_GENERATION

        # 재시도 필요 여부 판단
        if has_errors or is_short_answer:
            retry_reasons = []
            if has_errors:
                retry_reasons.append(f"errors={len(errors)}")
            if is_short_answer:
                retry_reasons.append(f"answer_len={answer_len} < {WorkflowConstants.MIN_ANSWER_LENGTH_GENERATION}")

            # 답변 내용 상세 로깅 (재시도 원인 파악용)
            self.logger.warning(
                f"⚠️ [SHORT ANSWER DETECTED] Answer length: {answer_len} characters\n"
                f"   Full answer content: '{answer}'\n"
                f"   Answer type: {type(answer).__name__}\n"
                f"   Answer repr: {repr(answer)}\n"
                f"   Error count: {len(errors)}"
            )

            # 답변 미리보기 로깅 (안전하게 처리)
            if isinstance(answer, str):
                answer_preview = (answer[:100] + "...") if len(answer) > 100 else answer
            else:
                answer_str = str(answer)
                answer_preview = (answer_str[:100] + "...") if len(answer_str) > 100 else answer_str
            self.logger.info(
                f"🔄 [RETRY DECISION] Retry needed ({', '.join(retry_reasons)}). "
                f"Will retry from document retrieval. [Preview: '{answer_preview}']"
            )
            return "retry_search"  # 검색부터 재시도 (카운터는 prepare_search_query에서 증가)

        # 정상적인 경우 검증으로 진행
        return "validate"

    def should_retry_validation(
        self,
        state: LegalWorkflowState,
        answer_generator: Any = None
    ) -> str:
        """
        2단계: 품질 검증 후 재시도 여부 결정

        재시도 전략:
        - 최대 1회 재시도
        - 법령 검증 실패 → 검색 재시도
        - 답변이 짧음 → 답변 생성 재시도
        - 그 외 → 구조 강화만 시도 또는 수락

        Args:
            state: 워크플로우 상태
            answer_generator: AnswerGenerator 인스턴스 (fallback 답변 생성용)

        Returns:
            "accept": 통과 또는 재시도 불필요
            "retry_generate": 답변 생성 재시도
            "retry_search": 검색부터 재시도
        """
        # answer_generator가 제공되지 않으면 self.answer_generator 사용
        if answer_generator is None:
            answer_generator = self.answer_generator

        # 품질 메타데이터 조회 (통합 헬퍼 사용)
        quality_meta = WorkflowUtils.get_quality_metadata(state)
        quality_check_passed = quality_meta["quality_check_passed"]
        quality_score = quality_meta["quality_score"]

        # 품질 메타데이터 상세 로깅 (디버깅용)
        self.logger.debug(
            f"🔍 [QUALITY METADATA READ] From _should_retry_validation:\n"
            f"   quality_check_passed: {quality_check_passed}\n"
            f"   quality_score: {quality_score:.2f}\n"
            f"   quality_meta dict: {quality_meta}"
        )

        # 재시도 카운터 조회 (통합 헬퍼 사용)
        retry_counts = self.retry_manager.get_retry_counts(state)
        validation_retry_count = retry_counts["validation"]
        total_retry_count = retry_counts["total"]

        # 전역 재시도 횟수 체크
        if total_retry_count >= RetryConfig.MAX_TOTAL_RETRIES:
            self.logger.warning(
                f"Maximum total retry count ({RetryConfig.MAX_TOTAL_RETRIES}) reached. "
                "Accepting answer despite quality issues."
            )
            return "accept"

        # 품질 검증 통과 시 즉시 accept (무한 루프 방지)
        if quality_check_passed:
            self.logger.info(
                f"✅ [QUALITY PASS] Quality check passed (score={quality_score:.2f}). "
                "Accepting answer without retry."
            )
            return "accept"

        # 무한 루프 방지: 이미 재시도한 경우 accept
        if validation_retry_count > 0:
            self.logger.warning(
                f"⛔ [HARD STOP] Validation retry already attempted ({validation_retry_count}/{RetryConfig.MAX_VALIDATION_RETRIES}). "
                "Accepting answer to prevent infinite loop."
            )
            return "accept"

        # 최대 재시도 횟수 초과 시 폴백 처리
        if validation_retry_count >= RetryConfig.MAX_VALIDATION_RETRIES:
            answer = WorkflowUtils.normalize_answer(WorkflowUtils.get_state_value(state, "answer", ""))
            answer_len = len(answer)

            if not answer or answer_len < 20:
                if answer_generator:
                    fallback_answer = answer_generator.generate_fallback_answer(state)
                    WorkflowUtils.set_state_value(state, "answer", fallback_answer)
                    self.logger.warning(
                        f"Maximum validation retry count ({RetryConfig.MAX_VALIDATION_RETRIES}) reached. "
                        f"Generated fallback answer (length: {len(fallback_answer)})"
                    )
                else:
                    self.logger.warning(
                        f"Maximum validation retry count ({RetryConfig.MAX_VALIDATION_RETRIES}) reached. "
                        "AnswerGenerator not available, cannot generate fallback answer."
                    )
            else:
                self.logger.warning(
                    f"Maximum validation retry count ({RetryConfig.MAX_VALIDATION_RETRIES}) reached. "
                    f"Accepting existing answer (length: {answer_len})"
                )
            return "accept"

        # 재시도 전략: 문제 유형에 따라 다른 재시도 방법 선택
        answer = WorkflowUtils.normalize_answer(WorkflowUtils.get_state_value(state, "answer", ""))
        answer_len = len(answer)
        legal_validity = WorkflowUtils.get_state_value(state, "legal_validity_check", True)

        # 메타데이터에서 품질 체크 정보 가져오기
        metadata = WorkflowUtils.get_state_value(state, "metadata", {})
        quality_metadata = metadata.get("quality_metadata", {}) if isinstance(metadata, dict) else {}
        quality_checks = quality_metadata.get("quality_checks", {})

        # 개선 가능성 평가 (AnswerGenerator 사용)
        improvement_potential = None
        if answer_generator:
            improvement_potential = answer_generator.assess_improvement_potential(
                quality_score,
                quality_checks,
                state
            )
            # 호환성을 위해 반환 형식 변환
            improvement_potential = {
                "should_retry": improvement_potential.get("potential", 0.0) >= 0.3,
                "confidence": improvement_potential.get("potential", 0.0),
                "best_strategy": improvement_potential.get("strategy") or "retry_generate",
                "reasons": improvement_potential.get("reasons", [])
            }

        # quality_score 기반 동적 임계값 설정
        if quality_score >= QualityThresholds.HIGH_QUALITY_THRESHOLD:
            min_length = QualityThresholds.HIGH_QUALITY_MIN_LENGTH
        elif quality_score >= QualityThresholds.MEDIUM_QUALITY_THRESHOLD:
            min_length = QualityThresholds.MEDIUM_QUALITY_MIN_LENGTH
        else:
            min_length = QualityThresholds.LOW_QUALITY_MIN_LENGTH

        # 재시도 필요성 분류
        retry_reasons = []
        if not legal_validity:
            retry_reasons.append("legal_validity_failed")
        if answer_len < min_length:
            retry_reasons.append(f"answer_too_short({answer_len} < {min_length})")
        if quality_score < QualityThresholds.MEDIUM_QUALITY_THRESHOLD:
            retry_reasons.append(f"low_quality_score({quality_score:.2f} < {QualityThresholds.MEDIUM_QUALITY_THRESHOLD})")

        # 재시도 결정 (개선 가능성 기반)
        if retry_reasons and validation_retry_count < RetryConfig.MAX_VALIDATION_RETRIES:
            # 개선 가능성이 높으면 재시도
            if improvement_potential and improvement_potential.get("should_retry"):
                retry_strategy = improvement_potential.get("best_strategy")

                # 피드백 저장 (다음 노드에서 사용)
                if not isinstance(metadata, dict):
                    metadata = {}
                metadata["retry_feedback"] = {
                    "previous_score": quality_score,
                    "failed_checks": [k for k, v in quality_checks.items() if not v],
                    "improvement_potential": improvement_potential,
                    "retry_strategy": retry_strategy
                }
                WorkflowUtils.set_state_value(state, "metadata", metadata)

                # 답변 내용 상세 로깅
                answer_preview = ""
                if isinstance(answer, str):
                    answer_preview = answer[:200]
                elif isinstance(answer, (dict, list)):
                    answer_preview = str(answer)[:200]
                else:
                    answer_preview = str(answer)[:200] if answer else ""

                self.logger.warning(
                    f"⚠️ [VALIDATION RETRY] Answer analysis:\n"
                    f"   Answer length: {answer_len} characters (min required: {min_length})\n"
                    f"   Quality score: {quality_score:.2f} (threshold: {QualityThresholds.MEDIUM_QUALITY_THRESHOLD})\n"
                    f"   Legal validity: {legal_validity}\n"
                    f"   Answer preview: {answer_preview}\n"
                    f"   Improvement potential: {improvement_potential.get('confidence', 0.0):.2f}\n"
                    f"   Best strategy: {retry_strategy}\n"
                    f"   Reasons: {improvement_potential.get('reasons', [])}\n"
                    f"   Full answer content: '{answer}'\n"
                    f"   Answer type: {type(answer).__name__}"
                )

                # 접지/인용 부족 시 검색 재시도를 우선 적용
                try:
                    has_sources = bool(WorkflowUtils.get_state_value(state, "sources", [])) or bool(WorkflowUtils.get_state_value(state, "retrieved_docs", []))
                except Exception:
                    has_sources = True

                if not legal_validity or not has_sources or retry_strategy == "retry_search":
                    # 법령 검증 실패 → 검색 재시도
                    self.logger.info(
                        f"🔄 [RETRY] Reasons: {', '.join(retry_reasons)}. "
                        f"Will retry search (count: {validation_retry_count}/{RetryConfig.MAX_VALIDATION_RETRIES})"
                    )
                    return "retry_search"
                elif answer_len < min_length or quality_score < QualityThresholds.MEDIUM_QUALITY_THRESHOLD:
                    # 답변이 짧거나 품질이 낮음 → 답변 생성 재시도
                    self.logger.info(
                        f"🔄 [RETRY] Reasons: {', '.join(retry_reasons)}. "
                        f"Will retry generation (count: {validation_retry_count}/{RetryConfig.MAX_VALIDATION_RETRIES})"
                    )
                    return "retry_generate"
            else:
                # 개선 가능성이 낮으면 수락
                if improvement_potential:
                    self.logger.info(
                        f"⚠️ [NO IMPROVEMENT POTENTIAL] Quality improvement unlikely. "
                        f"Score: {quality_score:.2f}, Potential: {improvement_potential.get('confidence', 0.0):.2f}, "
                        f"Reasons: {improvement_potential.get('reasons', [])}"
                    )
                return "accept"

        # 재시도 필요 없음
        self.logger.info(
            f"Quality check failed but no retry needed "
            f"(validation_retry_count: {validation_retry_count}/{RetryConfig.MAX_VALIDATION_RETRIES}, "
            f"quality_score: {quality_score:.2f}, answer_len: {answer_len}). "
            "Proceeding with enhancement."
        )
        return "accept"
