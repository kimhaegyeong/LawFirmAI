# -*- coding: utf-8 -*-
"""
답변 포맷팅 모듈
LangGraph 워크플로우의 답변 포맷팅 및 최종 응답 준비 로직을 독립 모듈로 분리
"""

import logging
import time
from datetime import datetime
from typing import Any, Dict, List, Optional, Union

from core.agents.state_definitions import LegalWorkflowState
from core.agents.workflow_utils import WorkflowUtils

# Constants for processing steps
MAX_PROCESSING_STEPS = 50


def prune_processing_steps(steps: List[Dict[str, Any]], max_items: int = MAX_PROCESSING_STEPS) -> List[Dict[str, Any]]:
    """처리 단계 목록 축소"""
    if not isinstance(steps, list):
        return []
    if len(steps) <= max_items:
        return steps
    # 최근 항목들만 유지
    return steps[-max_items:]


class AnswerFormatterHandler:
    """
    답변 포맷팅 클래스

    LangGraph 워크플로우의 답변 포맷팅 및 최종 응답 준비 기능을 제공합니다.
    """

    def __init__(
        self,
        keyword_mapper: Any = None,
        answer_structure_enhancer: Any = None,
        answer_formatter: Any = None,
        confidence_calculator: Any = None,
        reasoning_extractor: Any = None,
        answer_generator: Any = None,
        logger: Optional[logging.Logger] = None
    ):
        """
        AnswerFormatter 초기화

        Args:
            keyword_mapper: LegalKeywordMapper 인스턴스
            answer_structure_enhancer: AnswerStructureEnhancer 인스턴스
            answer_formatter: AnswerFormatter 인스턴스 (시각적 포맷팅용)
            confidence_calculator: ConfidenceCalculator 인스턴스
            reasoning_extractor: ReasoningExtractor 인스턴스
            answer_generator: AnswerGenerator 인스턴스 (파이프라인 추적용)
            logger: 로거 인스턴스 (없으면 자동 생성)
        """
        self.keyword_mapper = keyword_mapper
        self.answer_structure_enhancer = answer_structure_enhancer
        self.answer_formatter = answer_formatter
        self.confidence_calculator = confidence_calculator
        self.reasoning_extractor = reasoning_extractor
        self.answer_generator = answer_generator
        self.logger = logger or logging.getLogger(__name__)

    def format_answer(self, state: LegalWorkflowState) -> LegalWorkflowState:
        """통합된 답변 포맷팅: 구조화 + 시각적 포맷팅"""
        try:
            start_time = time.time()

            answer = WorkflowUtils.get_state_value(state, "answer", "")
            query = WorkflowUtils.get_state_value(state, "query", "")
            query_type = WorkflowUtils.get_state_value(state, "query_type", "general_question")
            confidence = WorkflowUtils.get_state_value(state, "confidence", 0.0)

            # 1단계: 답변 구조화 및 법적 근거 강화
            structured_answer = answer
            if self.answer_structure_enhancer:
                try:
                    retrieved_docs = state.get("retrieved_docs", [])
                    legal_references = state.get("legal_references", [])
                    legal_citations = state.get("legal_citations", [])

                    enhanced_result = self.answer_structure_enhancer.enhance_answer_structure(
                        answer=answer,
                        question_type=query_type,
                        question=query,
                        domain="general",
                        retrieved_docs=retrieved_docs if retrieved_docs else None,
                        legal_references=legal_references if legal_references else None,
                        legal_citations=legal_citations if legal_citations else None
                    )

                    if enhanced_result and "structured_answer" in enhanced_result:
                        structured_answer = WorkflowUtils.normalize_answer(enhanced_result["structured_answer"])

                        quality_metrics = enhanced_result.get("quality_metrics", {})
                        if quality_metrics:
                            WorkflowUtils.set_state_value(state, "quality_metrics", quality_metrics)
                            confidence = quality_metrics.get("overall_score", confidence)
                            WorkflowUtils.set_state_value(state, "structure_confidence", confidence)

                        if "legal_citations" in enhanced_result:
                            WorkflowUtils.set_state_value(state, "legal_citations", enhanced_result["legal_citations"])

                        self.logger.info("Answer structure enhanced successfully")
                except Exception as e:
                    self.logger.warning(f"AnswerStructureEnhancer failed: {e}")
                    structured_answer = WorkflowUtils.normalize_answer(answer)

            # 2단계: 시각적 포맷팅 (이모지 + 섹션 구조)
            if self.answer_formatter:
                try:
                    from core.services.search.question_classifier import QuestionType as QType
                    question_type_mapping = {
                        "precedent_search": QType.PRECEDENT_SEARCH,
                        "law_inquiry": QType.LAW_INQUIRY,
                        "legal_advice": QType.LEGAL_ADVICE,
                        "procedure_guide": QType.PROCEDURE_GUIDE,
                        "term_explanation": QType.TERM_EXPLANATION,
                        "general_question": QType.GENERAL_QUESTION,
                    }
                    q_type = question_type_mapping.get(query_type, QType.GENERAL_QUESTION)

                    from core.services.enhancement.confidence_calculator import ConfidenceInfo
                    final_confidence = state.get("structure_confidence") or confidence
                    confidence_info = ConfidenceInfo(
                        confidence=final_confidence,
                        level=self.map_confidence_level(final_confidence),
                        factors={"answer_quality": final_confidence},
                        explanation=f"신뢰도: {final_confidence:.1%}"
                    )

                    sources = {
                        "law_results": [
                            d for d in state.get("retrieved_docs", [])
                            if isinstance(d, dict) and (
                                "law" in str(d.get("type", "")).lower() or
                                "law" in str(d.get("source", "")).lower()
                            )
                        ],
                        "precedent_results": [
                            d for d in state.get("retrieved_docs", [])
                            if isinstance(d, dict) and (
                                "precedent" in str(d.get("type", "")).lower() or
                                "precedent" in str(d.get("source", "")).lower()
                            )
                        ]
                    }

                    formatted_result = self.answer_formatter.format_answer(
                        raw_answer=structured_answer,
                        question_type=q_type,
                        sources=sources,
                        confidence=confidence_info
                    )

                    if formatted_result and formatted_result.formatted_content:
                        final_answer = WorkflowUtils.normalize_answer(formatted_result.formatted_content)
                        state["answer"] = final_answer
                        state["format_metadata"] = formatted_result.metadata
                        self.logger.info("Visual formatting applied successfully")
                    else:
                        state["answer"] = structured_answer

                except Exception as e:
                    self.logger.warning(f"AnswerFormatter failed: {e}")
                    state["answer"] = structured_answer
            else:
                state["answer"] = structured_answer

            WorkflowUtils.update_processing_time(state, start_time)
            WorkflowUtils.add_step(state, "포맷팅", "답변 구조화 및 포맷팅 완료")

        except Exception as e:
            WorkflowUtils.handle_error(state, str(e), "답변 포맷팅 중 오류 발생")
            answer = WorkflowUtils.get_state_value(state, "answer", "")
            normalized_answer = WorkflowUtils.normalize_answer(answer)
            state["answer"] = normalized_answer

        return state

    def format_answer_part(self, state: LegalWorkflowState) -> str:
        """
        Part 1: 답변 포맷팅 로직만 처리

        Args:
            state: LegalWorkflowState 객체

        Returns:
            str: 포맷팅된 답변
        """
        format_start_time = time.time()

        answer = WorkflowUtils.get_state_value(state, "answer", "")
        query = WorkflowUtils.get_state_value(state, "query", "")
        query_type = WorkflowUtils.get_state_value(state, "query_type", "general_question")
        confidence = WorkflowUtils.get_state_value(state, "confidence", 0.0)

        # 추론 과정 분리 (LLM 응답에서 추론 과정과 실제 답변 분리)
        extraction_start_time = time.time()
        reasoning_info = self.reasoning_extractor.extract_reasoning(answer) if self.reasoning_extractor else {}
        actual_answer = None
        extraction_method = "none"

        # 추출 방법 우선순위에 따라 재시도
        if self.reasoning_extractor:
            extraction_methods = [
                ("output_section", lambda: self.reasoning_extractor.extract_by_output_section(answer)),
                ("reasoning_removed", lambda: self.reasoning_extractor.extract_by_removing_reasoning(answer, reasoning_info)),
                ("partial_cleaning", lambda: self.reasoning_extractor.extract_by_partial_cleaning(answer)),
            ]

            for method_name, extract_func in extraction_methods:
                try:
                    extracted = extract_func()
                    if extracted and extracted.strip() and extracted != answer:
                        actual_answer = extracted
                        extraction_method = method_name
                        break
                except Exception as e:
                    self.logger.debug(f"Extraction method {method_name} failed: {e}")
                    continue

        if not actual_answer or not actual_answer.strip():
            actual_answer = answer
            extraction_method = "fallback"

        extraction_time = time.time() - extraction_start_time

        # Step별 추출 성공 여부 확인
        step_extraction_status = {
            "step1": {"extracted": bool(reasoning_info.get("step1")), "length": len(reasoning_info.get("step1", ""))},
            "step2": {"extracted": bool(reasoning_info.get("step2")), "length": len(reasoning_info.get("step2", ""))},
            "step3": {"extracted": bool(reasoning_info.get("step3")), "length": len(reasoning_info.get("step3", ""))},
        }

        # 추론 과정 분리 후 품질 검증
        quality_metrics = {}
        if self.reasoning_extractor:
            quality_metrics = self.reasoning_extractor.verify_extraction_quality(
                original_answer=answer,
                actual_answer=actual_answer,
                reasoning_info=reasoning_info
            )

        # 추론 과정을 메타데이터에 저장
        if reasoning_info.get("has_reasoning") or extraction_method != "none":
            if "metadata" not in state:
                state["metadata"] = {}
            if "debug" not in state["metadata"]:
                state["metadata"]["debug"] = {}

            REASONING_MAX_LENGTH = 5000
            STEP_MAX_LENGTH = 2000

            full_reasoning = reasoning_info.get("reasoning", "")
            reasoning_stored = full_reasoning
            if len(full_reasoning) > REASONING_MAX_LENGTH:
                summary_length = REASONING_MAX_LENGTH // 2
                reasoning_stored = (
                    full_reasoning[:summary_length] +
                    "\n\n... (중간 생략) ...\n\n" +
                    full_reasoning[-summary_length:]
                )

            step1 = reasoning_info.get("step1", "")[:STEP_MAX_LENGTH] + ("... (생략)" if len(reasoning_info.get("step1", "")) > STEP_MAX_LENGTH else "")
            step2 = reasoning_info.get("step2", "")[:STEP_MAX_LENGTH] + ("... (생략)" if len(reasoning_info.get("step2", "")) > STEP_MAX_LENGTH else "")
            step3 = reasoning_info.get("step3", "")[:STEP_MAX_LENGTH] + ("... (생략)" if len(reasoning_info.get("step3", "")) > STEP_MAX_LENGTH else "")

            state["metadata"]["debug"]["reasoning"] = {
                "extraction_success": reasoning_info.get("has_reasoning", False),
                "extraction_method": extraction_method,
                "extraction_time_ms": round(extraction_time * 1000, 2),
                "full_reasoning": reasoning_stored,
                "step1": step1,
                "step2": step2,
                "step3": step3,
                "step_extraction_status": step_extraction_status,
                "reasoning_section_count": reasoning_info.get("reasoning_section_count", 1),
                "quality_metrics": quality_metrics,
                "extracted_at": datetime.now().isoformat(),
                "original_answer_length": len(answer),
                "actual_answer_length": len(actual_answer),
                "reasoning_length": len(full_reasoning),
                "extraction_ratio": round(len(actual_answer) / len(answer), 3) if answer else 0.0,
                "memory_optimized": len(full_reasoning) > REASONING_MAX_LENGTH,
            }

        # 실제 답변에서 추론 과정 키워드 정리
        if self.reasoning_extractor:
            cleaned_actual_answer = self.reasoning_extractor.clean_reasoning_keywords(actual_answer if actual_answer else answer)
        else:
            cleaned_actual_answer = actual_answer if actual_answer else answer

        # 1단계: 답변 구조화 및 법적 근거 강화
        structured_answer = cleaned_actual_answer
        if self.answer_structure_enhancer:
            try:
                retrieved_docs = state.get("retrieved_docs", [])
                legal_references = state.get("legal_references", [])
                legal_citations = state.get("legal_citations", [])

                enhanced_result = self.answer_structure_enhancer.enhance_answer_structure(
                    answer=structured_answer,
                    question_type=query_type,
                    question=query,
                    domain="general",
                    retrieved_docs=retrieved_docs if retrieved_docs else None,
                    legal_references=legal_references if legal_references else None,
                    legal_citations=legal_citations if legal_citations else None
                )

                if enhanced_result and "structured_answer" in enhanced_result:
                    enhanced_answer = WorkflowUtils.normalize_answer(enhanced_result["structured_answer"])

                    # Enhancer 결과에서 추론 과정이 재포함되었는지 확인
                    if self.reasoning_extractor:
                        enhanced_reasoning_info = self.reasoning_extractor.extract_reasoning(enhanced_answer)
                        if enhanced_reasoning_info.get("has_reasoning"):
                            self.logger.warning("AnswerStructureEnhancer re-included reasoning process. Re-extracting...")
                            enhanced_actual_answer = self.reasoning_extractor.extract_actual_answer(enhanced_answer)

                            if enhanced_actual_answer and enhanced_actual_answer.strip():
                                structured_answer = enhanced_actual_answer
                                enhanced_quality_metrics = self.reasoning_extractor.verify_extraction_quality(
                                    original_answer=enhanced_answer,
                                    actual_answer=enhanced_actual_answer,
                                    reasoning_info=enhanced_reasoning_info
                                )
                                if "metadata" in state and "debug" in state["metadata"] and "reasoning" in state["metadata"]["debug"]:
                                    existing_quality = state["metadata"]["debug"]["reasoning"].get("quality_metrics", {})
                                    if enhanced_quality_metrics.get("score", 1.0) < existing_quality.get("score", 1.0):
                                        self.logger.warning(
                                            f"Enhanced answer quality degraded "
                                            f"(before: {existing_quality.get('score', 1.0):.2f}, "
                                            f"after: {enhanced_quality_metrics.get('score', 1.0):.2f})"
                                        )
                                    state["metadata"]["debug"]["reasoning"]["enhanced_quality_metrics"] = enhanced_quality_metrics
                            else:
                                self.logger.warning("Failed to re-extract reasoning from enhancer output. Using original.")
                        else:
                            structured_answer = enhanced_answer
                            if self.reasoning_extractor:
                                enhanced_quality_metrics = self.reasoning_extractor.verify_extraction_quality(
                                    original_answer=structured_answer,
                                    actual_answer=structured_answer,
                                    reasoning_info=enhanced_reasoning_info
                                )
                                if "metadata" in state and "debug" in state["metadata"] and "reasoning" in state["metadata"]["debug"]:
                                    state["metadata"]["debug"]["reasoning"]["enhanced_quality_metrics"] = enhanced_quality_metrics
                    else:
                        structured_answer = enhanced_answer

                    quality_metrics_enhanced = enhanced_result.get("quality_metrics", {})
                    if quality_metrics_enhanced:
                        WorkflowUtils.set_state_value(state, "quality_metrics", quality_metrics_enhanced)
                        confidence = quality_metrics_enhanced.get("overall_score", confidence)
                        WorkflowUtils.set_state_value(state, "structure_confidence", confidence)

                    if "legal_citations" in enhanced_result:
                        WorkflowUtils.set_state_value(state, "legal_citations", enhanced_result["legal_citations"])

                    self.logger.info("Answer structure enhanced successfully")
            except Exception as e:
                self.logger.warning(f"AnswerStructureEnhancer failed: {e}")
                structured_answer = WorkflowUtils.normalize_answer(structured_answer)

        # 2단계: 시각적 포맷팅 (이모지 + 섹션 구조)
        formatted_answer = structured_answer
        if self.answer_formatter:
            try:
                from core.services.search.question_classifier import QuestionType as QType
                question_type_mapping = {
                    "precedent_search": QType.PRECEDENT_SEARCH,
                    "law_inquiry": QType.LAW_INQUIRY,
                    "legal_advice": QType.LEGAL_ADVICE,
                    "procedure_guide": QType.PROCEDURE_GUIDE,
                    "term_explanation": QType.TERM_EXPLANATION,
                    "general_question": QType.GENERAL_QUESTION,
                }
                q_type = question_type_mapping.get(query_type, QType.GENERAL_QUESTION)

                from core.services.enhancement.confidence_calculator import ConfidenceInfo
                final_confidence = state.get("structure_confidence") or confidence
                confidence_info = ConfidenceInfo(
                    confidence=final_confidence,
                    level=self.map_confidence_level(final_confidence),
                    factors={"answer_quality": final_confidence},
                    explanation=f"신뢰도: {final_confidence:.1%}"
                )

                sources = {
                    "law_results": [
                        d for d in state.get("retrieved_docs", [])
                        if isinstance(d, dict) and (
                            "law" in str(d.get("type", "")).lower() or
                            "law" in str(d.get("source", "")).lower()
                        )
                    ],
                    "precedent_results": [
                        d for d in state.get("retrieved_docs", [])
                        if isinstance(d, dict) and (
                            "precedent" in str(d.get("type", "")).lower() or
                            "precedent" in str(d.get("source", "")).lower()
                        )
                    ]
                }

                formatted_result = self.answer_formatter.format_answer(
                    raw_answer=structured_answer,
                    question_type=q_type,
                    sources=sources,
                    confidence=confidence_info
                )

                if formatted_result and formatted_result.formatted_content:
                    formatted_answer = WorkflowUtils.normalize_answer(formatted_result.formatted_content)
                    state["format_metadata"] = formatted_result.metadata
                    self.logger.info("Visual formatting applied successfully")
                else:
                    formatted_answer = structured_answer
            except Exception as e:
                self.logger.warning(f"AnswerFormatter failed: {e}")
                formatted_answer = structured_answer

        WorkflowUtils.update_processing_time(state, format_start_time)
        WorkflowUtils.add_step(state, "포맷팅", "답변 구조화 및 포맷팅 완료")

        return formatted_answer

    def prepare_final_response(self, state: LegalWorkflowState) -> LegalWorkflowState:
        """최종 응답 상태 준비"""
        try:
            start_time = time.time()

            # 파이프라인 품질 추적
            if self.answer_generator:
                self.answer_generator.track_search_to_answer_pipeline(state)

            # Final pruning
            if len(state.get("processing_steps", [])) > MAX_PROCESSING_STEPS:
                state["processing_steps"] = prune_processing_steps(
                    state["processing_steps"],
                    max_items=MAX_PROCESSING_STEPS
                )

            errors = WorkflowUtils.get_state_value(state, "errors", [])
            if len(errors) > 10:
                WorkflowUtils.set_state_value(state, "errors", errors[-10:])

            # 신뢰도 계산
            answer_value = WorkflowUtils.normalize_answer(state.get("answer", ""))

            sources_list = []
            for doc in state.get("retrieved_docs", []):
                if isinstance(doc, dict):
                    sources_list.append(doc)

            query_type = WorkflowUtils.get_state_value(state, "query_type", "general")

            # ConfidenceCalculator를 사용하여 신뢰도 계산
            calculated_confidence = None
            if self.confidence_calculator and answer_value:
                try:
                    confidence_info = self.confidence_calculator.calculate_confidence(
                        answer=answer_value,
                        sources=sources_list,
                        question_type=query_type
                    )
                    calculated_confidence = confidence_info.confidence
                    self.logger.info(f"ConfidenceCalculator: confidence={calculated_confidence:.3f}, factors={confidence_info.factors}")
                except Exception as e:
                    self.logger.warning(f"ConfidenceCalculator failed: {e}")

            existing_confidence = state.get("structure_confidence") or state.get("confidence", 0.0)

            if calculated_confidence is not None:
                final_confidence = calculated_confidence
            else:
                final_confidence = existing_confidence

            # 기본 신뢰도 보장
            min_confidence = 0.25 if (answer_value and sources_list) else (0.15 if answer_value else 0.05)
            final_confidence = max(final_confidence, min_confidence)

            # 키워드 포함도 기반 보정
            keyword_coverage = self.calculate_keyword_coverage(state, answer_value)
            keyword_boost = keyword_coverage * 0.3
            adjusted_confidence = min(0.95, final_confidence + keyword_boost)

            # 소스 개수 기반 추가 보정
            if sources_list:
                source_count = len(sources_list)
                if source_count >= 5:
                    adjusted_confidence = min(0.95, adjusted_confidence + 0.05)
                elif source_count >= 3:
                    adjusted_confidence = min(0.95, adjusted_confidence + 0.03)
                elif source_count >= 1:
                    adjusted_confidence = min(0.95, adjusted_confidence + 0.01)

            # 답변 길이 기반 추가 보정
            if answer_value:
                answer_length = len(answer_value)
                if answer_length >= 500:
                    adjusted_confidence = min(0.95, adjusted_confidence + 0.05)
                elif answer_length >= 200:
                    adjusted_confidence = min(0.95, adjusted_confidence + 0.03)
                elif answer_length >= 100:
                    adjusted_confidence = min(0.95, adjusted_confidence + 0.01)

            state["confidence"] = adjusted_confidence

            # Phase 3: 최종 answer를 문자열로 수렴 - 타입 확인 후 필요시만 정규화
            current_answer = state.get("answer", "")
            if not isinstance(current_answer, str):
                # 이미 포맷팅된 answer는 정규화 불필요, 타입 검증만 수행
                try:
                    state["answer"] = WorkflowUtils.normalize_answer(current_answer)
                except Exception:
                    state["answer"] = str(current_answer) if current_answer else ""

            # sources 추출
            final_sources_list = []
            for doc in state.get("retrieved_docs", []):
                source = doc.get("source", "Unknown") if isinstance(doc, dict) else str(doc) if doc else "Unknown"
                if isinstance(source, str):
                    final_sources_list.append(source)
                elif source is not None:
                    try:
                        final_sources_list.append(str(source))
                    except Exception:
                        final_sources_list.append("Unknown")
            state["sources"] = list(set(final_sources_list))

            # 법적 참조 정보 추가
            if "legal_references" not in state:
                state["legal_references"] = []

            # 메타데이터 설정
            self.set_metadata(state, answer_value, keyword_coverage)

            # 임시 라우팅 플래그/피드백 제거
            try:
                metadata = state.get("metadata", {})
                if isinstance(metadata, dict):
                    for k in ("force_rag_fallback", "router_feedback"):
                        metadata.pop(k, None)
                    state["metadata"] = metadata
            except Exception:
                pass

            # sources 표준화 및 중복 제거
            try:
                src = state.get("sources", [])
                norm = []
                seen = set()
                if isinstance(src, list):
                    for s in src:
                        if isinstance(s, dict):
                            key = (s.get("type"), s.get("sql") or s.get("title") or s.get("url"))
                            if key in seen:
                                continue
                            seen.add(key)
                            norm.append(s)
                        elif isinstance(s, str):
                            if s in seen:
                                continue
                            seen.add(s)
                            norm.append(s)
                state["sources"] = norm[:10]
            except Exception:
                pass

            WorkflowUtils.update_processing_time(state, start_time)
            WorkflowUtils.add_step(state, "최종 준비", "최종 응답 준비 완료")

            # Final pruning after adding last step
            if len(state.get("processing_steps", [])) > MAX_PROCESSING_STEPS:
                state["processing_steps"] = prune_processing_steps(
                    state["processing_steps"],
                    max_items=MAX_PROCESSING_STEPS
                )

            self.logger.info(f"Final response prepared with confidence: {adjusted_confidence:.3f}")

        except Exception as e:
            WorkflowUtils.handle_error(state, str(e), "최종 준비 중 오류 발생")

        return state

    def prepare_final_response_part(
        self,
        state: LegalWorkflowState,
        query_complexity: Optional[str],
        needs_search: bool
    ) -> None:
        """
        Part 2: 최종 응답 준비 로직만 처리

        Args:
            state: LegalWorkflowState 객체
            query_complexity: 보존할 query_complexity 값
            needs_search: 보존할 needs_search 값
        """
        final_start_time = time.time()

        # query_complexity 보존 및 저장
        if query_complexity:
            self.preserve_and_store_values(state, query_complexity, needs_search)

        # 파이프라인 품질 추적
        if self.answer_generator:
            self.answer_generator.track_search_to_answer_pipeline(state)

        # Final pruning
        if len(state.get("processing_steps", [])) > MAX_PROCESSING_STEPS:
            state["processing_steps"] = prune_processing_steps(
                state["processing_steps"],
                max_items=MAX_PROCESSING_STEPS
            )

        errors = WorkflowUtils.get_state_value(state, "errors", [])
        if len(errors) > 10:
            WorkflowUtils.set_state_value(state, "errors", errors[-10:])

        # 신뢰도 계산
        answer_value = WorkflowUtils.normalize_answer(state.get("answer", ""))

        sources_list = []
        for doc in state.get("retrieved_docs", []):
            if isinstance(doc, dict):
                sources_list.append(doc)

        query_type = WorkflowUtils.get_state_value(state, "query_type", "general")

        # ConfidenceCalculator를 사용하여 신뢰도 계산
        calculated_confidence = None
        if self.confidence_calculator and answer_value:
            try:
                confidence_info = self.confidence_calculator.calculate_confidence(
                    answer=answer_value,
                    sources=sources_list,
                    question_type=query_type
                )
                calculated_confidence = confidence_info.confidence
                self.logger.info(f"ConfidenceCalculator: confidence={calculated_confidence:.3f}, factors={confidence_info.factors}")
            except Exception as e:
                self.logger.warning(f"ConfidenceCalculator failed: {e}")

        existing_confidence = state.get("structure_confidence") or state.get("confidence", 0.0)

        if calculated_confidence is not None:
            final_confidence = calculated_confidence
        else:
            final_confidence = existing_confidence

        # 기본 신뢰도 보장
        min_confidence = 0.25 if (answer_value and sources_list) else (0.15 if answer_value else 0.05)
        final_confidence = max(final_confidence, min_confidence)

        # 키워드 포함도 기반 보정
        keyword_coverage = self.calculate_keyword_coverage(state, answer_value)
        keyword_boost = keyword_coverage * 0.3
        adjusted_confidence = min(0.95, final_confidence + keyword_boost)

        # 소스 개수 기반 추가 보정
        if sources_list:
            source_count = len(sources_list)
            if source_count >= 5:
                adjusted_confidence = min(0.95, adjusted_confidence + 0.05)
            elif source_count >= 3:
                adjusted_confidence = min(0.95, adjusted_confidence + 0.03)
            elif source_count >= 1:
                adjusted_confidence = min(0.95, adjusted_confidence + 0.01)

        # 답변 길이 기반 추가 보정
        if answer_value:
            answer_length = len(answer_value)
            if answer_length >= 500:
                adjusted_confidence = min(0.95, adjusted_confidence + 0.05)
            elif answer_length >= 200:
                adjusted_confidence = min(0.95, adjusted_confidence + 0.03)
            elif answer_length >= 100:
                adjusted_confidence = min(0.95, adjusted_confidence + 0.01)

        state["confidence"] = adjusted_confidence

        # 최종 answer를 문자열로 수렴
        try:
            state["answer"] = WorkflowUtils.normalize_answer(state.get("answer", ""))
        except Exception:
            state["answer"] = str(state.get("answer", ""))

        # sources 추출
        final_sources_list = []
        for doc in state.get("retrieved_docs", []):
            source = doc.get("source", "Unknown") if isinstance(doc, dict) else str(doc) if doc else "Unknown"
            if isinstance(source, str):
                final_sources_list.append(source)
            elif source is not None:
                try:
                    final_sources_list.append(str(source))
                except Exception:
                    final_sources_list.append("Unknown")
        state["sources"] = list(set(final_sources_list))

        # 법적 참조 정보 추가
        if "legal_references" not in state:
            state["legal_references"] = []

        # 메타데이터 설정
        self.set_metadata(state, answer_value, keyword_coverage)

        # 임시 라우팅 플래그/피드백 제거
        try:
            metadata = state.get("metadata", {})
            if isinstance(metadata, dict):
                for k in ("force_rag_fallback", "router_feedback"):
                    metadata.pop(k, None)
                state["metadata"] = metadata
        except Exception:
            pass

        # sources 표준화 및 중복 제거
        try:
            src = state.get("sources", [])
            norm = []
            seen = set()
            if isinstance(src, list):
                for s in src:
                    if isinstance(s, dict):
                        key = (s.get("type"), s.get("sql") or s.get("title") or s.get("url"))
                        if key in seen:
                            continue
                        seen.add(key)
                        norm.append(s)
                    elif isinstance(s, str):
                        if s in seen:
                            continue
                        seen.add(s)
                        norm.append(s)
            state["sources"] = norm[:10]
        except Exception:
            pass

        WorkflowUtils.update_processing_time(state, final_start_time)
        WorkflowUtils.add_step(state, "최종 준비", "최종 응답 준비 완료")

        # Final pruning after adding last step
        if len(state.get("processing_steps", [])) > MAX_PROCESSING_STEPS:
            state["processing_steps"] = prune_processing_steps(
                state["processing_steps"],
                max_items=MAX_PROCESSING_STEPS
            )

    def format_and_prepare_final(self, state: LegalWorkflowState) -> LegalWorkflowState:
        """통합된 답변 포맷팅 및 최종 준비 (format_answer + prepare_final_response)"""
        try:
            overall_start_time = time.time()

            # 보존할 값 추출
            preserved_values = self.extract_preserved_values(state)
            query_complexity = preserved_values["query_complexity"]
            needs_search = preserved_values["needs_search"]

            # Part 1: 포맷팅
            formatted_answer = self.format_answer_part(state)
            # Phase 1/Phase 3: _set_answer_safely는 legal_workflow_enhanced에 있으므로,
            # 여기서는 정규화만 확인하고 필요시 업데이트 (format_answer_part에서 이미 정규화되었을 수 있음)
            if not isinstance(formatted_answer, str):
                formatted_answer = WorkflowUtils.normalize_answer(formatted_answer)
            state["answer"] = formatted_answer

            # Part 2: 최종 준비
            self.prepare_final_response_part(state, query_complexity, needs_search)

            elapsed = time.time() - overall_start_time
            confidence = state.get("confidence", 0.0)
            self.logger.info(
                f"format_and_prepare_final completed in {elapsed:.2f}s, "
                f"confidence: {confidence:.3f}"
            )

        except Exception as e:
            WorkflowUtils.handle_error(state, str(e), "답변 포맷팅 및 최종 준비 중 오류 발생")
            answer = WorkflowUtils.get_state_value(state, "answer", "")

            # 에러 발생 시에도 추론 과정 분리 시도
            if self.reasoning_extractor:
                try:
                    reasoning_info = self.reasoning_extractor.extract_reasoning(answer)
                    if reasoning_info.get("has_reasoning"):
                        actual_answer = self.reasoning_extractor.extract_actual_answer(answer)
                        if actual_answer and actual_answer.strip():
                            state["answer"] = WorkflowUtils.normalize_answer(actual_answer)
                except Exception:
                    pass

            # Phase 5: 에러 처리 통일 - answer 복원 로직 개선
            if not state.get("answer"):
                # answer가 없으면 정규화하여 설정
                state["answer"] = WorkflowUtils.normalize_answer(answer) if answer else ""
            elif answer and state.get("answer") != answer:
                # answer가 있지만 원본과 다르면 정규화만 수행
                current_answer = state.get("answer")
                if not isinstance(current_answer, str):
                    state["answer"] = WorkflowUtils.normalize_answer(current_answer)

        return state

    def extract_preserved_values(self, state: LegalWorkflowState) -> Dict[str, Any]:
        """
        state에서 보존해야 하는 값들을 한 번에 추출

        Args:
            state: LegalWorkflowState 객체

        Returns:
            Dict[str, Any]: {
                "query_complexity": str | None,
                "needs_search": bool
            }
        """
        preserved = {
            "query_complexity": None,
            "needs_search": True
        }

        # 우선순위별로 검색
        search_paths = [
            (None, "query_complexity", "needs_search"),  # 직접 접근
            ("common", "query_complexity", "needs_search"),  # common 그룹
            ("metadata", "query_complexity", "needs_search"),  # metadata 그룹
            ("classification", "query_complexity", "needs_search")  # classification 그룹
        ]

        for path in search_paths:
            group_key, complexity_key, search_key = path

            if group_key is None:
                if isinstance(state, dict):
                    if complexity_key in state and not preserved["query_complexity"]:
                        preserved["query_complexity"] = state.get(complexity_key)
                    if search_key in state:
                        preserved["needs_search"] = state.get(search_key, True)
            else:
                if isinstance(state, dict) and group_key in state:
                    group = state[group_key]
                    if isinstance(group, dict):
                        if not preserved["query_complexity"] and complexity_key in group:
                            preserved["query_complexity"] = group.get(complexity_key)
                        if search_key in group:
                            preserved["needs_search"] = group.get(search_key, True)

            if preserved["query_complexity"] is not None:
                break

        return preserved

    def preserve_and_store_values(
        self,
        state: LegalWorkflowState,
        query_complexity: Optional[str],
        needs_search: bool
    ) -> None:
        """
        추출한 값을 state의 모든 필요한 위치에 저장

        Args:
            state: LegalWorkflowState 객체
            query_complexity: 보존할 query_complexity 값
            needs_search: 보존할 needs_search 값
        """
        if not query_complexity:
            return

        # 저장할 위치들
        storage_locations = [state]

        # 그룹별로 저장
        for group_key in ["common", "metadata", "classification"]:
            if group_key not in state:
                state[group_key] = {}
            storage_locations.append(state[group_key])

        for location in storage_locations:
            if isinstance(location, dict):
                location["query_complexity"] = query_complexity
                location["needs_search"] = needs_search

    def map_confidence_level(self, confidence: float):
        """신뢰도 점수에 따른 레벨 매핑"""
        from core.services.enhancement.confidence_calculator import ConfidenceLevel

        if confidence >= 0.9:
            return ConfidenceLevel.VERY_HIGH
        elif confidence >= 0.7:
            return ConfidenceLevel.HIGH
        elif confidence >= 0.5:
            return ConfidenceLevel.MEDIUM
        elif confidence >= 0.3:
            return ConfidenceLevel.LOW
        else:
            return ConfidenceLevel.VERY_LOW

    def calculate_keyword_coverage(
        self,
        state: LegalWorkflowState,
        answer: Union[str, Dict[str, Any], None]
    ) -> float:
        """키워드 포함도 계산"""
        try:
            answer_str = WorkflowUtils.normalize_answer(answer)

            if not isinstance(answer_str, str):
                answer_str = str(answer_str) if answer_str else ""

            if not answer_str:
                return 0.0

            query = WorkflowUtils.get_state_value(state, "query", "")
            query_type = WorkflowUtils.get_state_value(state, "query_type", "")
            if not self.keyword_mapper:
                return 1.0

            required_keywords = self.keyword_mapper.get_keywords_for_question(query, query_type)

            if not required_keywords:
                return 1.0

            return self.keyword_mapper.calculate_keyword_coverage(answer_str, required_keywords)
        except Exception as e:
            self.logger.warning(f"Keyword coverage calculation failed: {e}")
            return 0.0

    def set_metadata(
        self,
        state: LegalWorkflowState,
        answer: Union[str, Dict[str, Any], None],
        keyword_coverage: float
    ) -> None:
        """메타데이터 설정"""
        try:
            answer_str = WorkflowUtils.normalize_answer(answer)

            if not isinstance(answer_str, str):
                answer_str = str(answer_str) if answer_str else ""

            if not answer_str:
                answer_str = ""

            query = WorkflowUtils.get_state_value(state, "query", "")
            query_type = WorkflowUtils.get_state_value(state, "query_type", "")
            required_keywords = []
            if self.keyword_mapper:
                try:
                    required_keywords = self.keyword_mapper.get_keywords_for_question(query, query_type)
                    if required_keywords is None:
                        required_keywords = []
                    elif not isinstance(required_keywords, (list, tuple)):
                        required_keywords = [str(required_keywords)]
                except Exception as e:
                    self.logger.warning(f"Keyword retrieval failed: {e}")

            missing_keywords = []
            if answer_str and required_keywords and self.keyword_mapper:
                try:
                    missing_keywords = self.keyword_mapper.get_missing_keywords(answer_str, list(required_keywords))
                except Exception as e:
                    self.logger.warning(f"Missing keyword calculation failed: {e}")

            metadata = WorkflowUtils.get_state_value(state, "metadata", {})
            if not isinstance(metadata, dict):
                metadata = {}
            metadata.update({
                "keyword_coverage": keyword_coverage,
                "required_keywords_count": len(required_keywords) if required_keywords else 0,
                "matched_keywords_count": len(required_keywords) - len(missing_keywords) if required_keywords else 0,
                "response_length": len(answer_str) if answer_str else 0,
                "query_type": query_type
            })
            WorkflowUtils.set_state_value(state, "metadata", metadata)
        except Exception as e:
            self.logger.warning(f"Metadata setting failed: {e}")
