# -*- coding: utf-8 -*-
"""
답변 포맷팅 모듈
LangGraph 워크플로우의 답변 포맷팅 및 최종 응답 준비 로직을 독립 모듈로 분리
"""

import logging
import os
import re
import time
from datetime import datetime
from typing import Any, Dict, List, Optional, Union

try:
    from lawfirm_langgraph.core.agents.state_definitions import LegalWorkflowState
except ImportError:
    from core.agents.state_definitions import LegalWorkflowState
try:
    from lawfirm_langgraph.core.workflow.utils.workflow_utils import WorkflowUtils
except ImportError:
    from core.workflow.utils.workflow_utils import WorkflowUtils
try:
    from lawfirm_langgraph.core.agents.validators.quality_validators import AnswerValidator
except ImportError:
    from core.agents.validators.quality_validators import AnswerValidator

from .config.formatter_config import AnswerLengthConfig, ConfidenceConfig
from .managers.confidence_manager import ConfidenceManager
from .extractors.source_extractor import SourceExtractor
from .cleaners.answer_cleaner import AnswerCleaner
from .formatters.length_adjuster import AnswerLengthAdjuster

# Constants for processing steps
MAX_PROCESSING_STEPS = 50

# 개선: 검색 결과 수 증가에 맞춰 제한 상수 정의
MAX_SOURCES_LIMIT = 15  # sources, sources_detail 제한 (10 → 15)
MAX_LEGAL_REFERENCES_LIMIT = 15  # legal_references 제한 (10 → 15)
MAX_RELATED_QUESTIONS_LIMIT = 10  # related_questions 제한 (5 → 10)
MAX_SOURCES_DISPLAY_LIMIT = 10  # 답변 내 sources 표시 제한 (5 → 10)

# 답변 길이 목표 (질의 유형별) - 개선: 최대 길이 추가 증가 (하위 호환성 유지)
ANSWER_LENGTH_TARGETS = {
    "simple_question": (500, 3000),
    "term_explanation": (800, 3500),
    "legal_analysis": (1500, 5000),
    "complex_question": (2000, 8000),
    "default": (800, 4000)
}


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
        if self.logger.level > logging.INFO:
            self.logger.setLevel(logging.INFO)
        
        # 리팩토링된 컴포넌트 초기화
        self.length_config = AnswerLengthConfig()
        self.confidence_config = ConfidenceConfig()
        self.confidence_manager = ConfidenceManager(self.confidence_config, self.logger)
        self.source_extractor = SourceExtractor(self.logger)
        self.answer_cleaner = AnswerCleaner(self.logger)
        self.length_adjuster = AnswerLengthAdjuster(self.length_config, self.logger)
        
        # Config 및 LegalDataConnectorV2 인스턴스 초기화 (os 변수 오류 방지)
        # 지연 초기화로 변경하여 os 변수 오류 방지
        self._config = None
        self._connector = None
        self._config_initialized = False

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
                    try:
                        from core.classification.classifiers.question_classifier import (
                            QuestionType as QType,
                        )
                    except ImportError:
                        # 호환성을 위한 fallback
                        from core.services.question_classifier import (
                            QuestionType as QType,
                        )
                    question_type_mapping = {
                        "precedent_search": QType.PRECEDENT_SEARCH,
                        "law_inquiry": QType.LAW_INQUIRY,
                        "legal_advice": QType.LEGAL_ADVICE,
                        "procedure_guide": QType.PROCEDURE_GUIDE,
                        "term_explanation": QType.TERM_EXPLANATION,
                        "general_question": QType.GENERAL_QUESTION,
                    }
                    q_type = question_type_mapping.get(query_type, QType.GENERAL_QUESTION)

                    from core.generation.validators.confidence_calculator import (
                        ConfidenceInfo,
                    )
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
                from core.services.question_classifier import (
                    QuestionType as QType,
                )
                question_type_mapping = {
                    "precedent_search": QType.PRECEDENT_SEARCH,
                    "law_inquiry": QType.LAW_INQUIRY,
                    "legal_advice": QType.LEGAL_ADVICE,
                    "procedure_guide": QType.PROCEDURE_GUIDE,
                    "term_explanation": QType.TERM_EXPLANATION,
                    "general_question": QType.GENERAL_QUESTION,
                }
                q_type = question_type_mapping.get(query_type, QType.GENERAL_QUESTION)

                from core.generation.validators.confidence_calculator import (
                    ConfidenceInfo,
                )
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

        # 답변 반환 전 최종 정리: 중복 헤더 제거 및 신뢰도 값 통일 (즉시 적용)
        import re
        if formatted_answer and isinstance(formatted_answer, str):
            lines = formatted_answer.split('\n')
            cleaned_lines = []
            seen_answer_header = False

            for line in lines:
                # "## 답변" 헤더는 한 번만 유지
                if re.match(r'^##\s*답변\s*$', line, re.IGNORECASE):
                    if not seen_answer_header:
                        cleaned_lines.append(line)
                        seen_answer_header = True
                    continue
                # "###" 로 시작하고 "답변"이 포함된 줄 제거 (이모지 포함)
                elif re.match(r'^###\s*.*답변', line, re.IGNORECASE):
                    continue
                else:
                    cleaned_lines.append(line)

            formatted_answer = '\n'.join(cleaned_lines)

            # 신뢰도 값 통일 (리팩토링된 메서드 사용)
            current_confidence = state.get("confidence", 0.0)
            if current_confidence > 0:
                formatted_answer = self.confidence_manager.replace_in_text(formatted_answer, current_confidence)

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

            # sources 추출 (개선: source_type별 상세 정보 추출)
            # 주의: prepare_final_response_part에서 이미 sources를 생성했을 수 있으므로,
            # sources가 이미 있으면 덮어쓰지 않음
            existing_sources = state.get("sources", [])
            
            if existing_sources and len(existing_sources) > 0:
                # sources가 이미 있으면 그대로 사용 (덮어쓰기 방지)
                # prepare_final_response_part에서 생성한 sources를 보존
                self.logger.info(f"[PREPARE_FINAL_RESPONSE] Using existing sources ({len(existing_sources)} items) from prepare_final_response_part - skipping source generation")
                # sources 생성 로직을 완전히 건너뛰고 다음 단계로 진행
                sources_skipped = True
            else:
                sources_skipped = False
                # sources가 없으면 생성
                final_sources_list = []
                final_sources_detail = []
                seen_sources = set()

                # 통일된 포맷터 및 검증기 초기화
                try:
                    from ...generation.formatters.unified_source_formatter import UnifiedSourceFormatter
                    from ...generation.validators.source_validator import SourceValidator
                    formatter = UnifiedSourceFormatter()
                    validator = SourceValidator()
                except ImportError:
                    formatter = None
                    validator = None

                for doc in state.get("retrieved_docs", []):
                    if not isinstance(doc, dict):
                        continue

                    source = None
                    source_type = doc.get("type") or doc.get("source_type") or doc.get("metadata", {}).get("source_type", "")
                    metadata = doc.get("metadata", {}) if isinstance(doc.get("metadata"), dict) else {}
                    
                    # 통일된 포맷터로 상세 정보 생성
                    source_info_detail = None
                    if formatter and source_type:
                        try:
                            # doc과 metadata를 병합하여 포맷터에 전달
                            merged_metadata = {**metadata}
                            for key in ["statute_name", "law_name", "article_no", "article_number", "clause_no", "item_no",
                                       "court", "doc_id", "casenames", "org", "title", "announce_date", "decision_date", "response_date"]:
                                if key in doc:
                                    merged_metadata[key] = doc[key]
                            
                            source_info_detail = formatter.format_source(source_type, merged_metadata)
                            
                            # 검증 수행
                            if validator:
                                validation_result = validator.validate_source(source_type, merged_metadata)
                                source_info_detail.validation = validation_result
                        except Exception as e:
                            self.logger.warning(f"Error formatting source detail: {e}")
                    
                    # 1. statute_article (법령 조문) 처리
                    if source_type == "statute_article":
                        statute_name = (
                            doc.get("statute_name") or
                            doc.get("law_name") or
                            metadata.get("statute_name") or
                            metadata.get("law_name")
                        )
                        
                        if statute_name:
                            article_no = (
                                doc.get("article_no") or
                                doc.get("article_number") or
                                metadata.get("article_no") or
                                metadata.get("article_number")
                            )
                            clause_no = doc.get("clause_no") or metadata.get("clause_no")
                            item_no = doc.get("item_no") or metadata.get("item_no")
                            
                            source_parts = [statute_name]
                            if article_no:
                                source_parts.append(article_no)
                            if clause_no:
                                source_parts.append(f"제{clause_no}항")
                            if item_no:
                                source_parts.append(f"제{item_no}호")
                            
                            source = " ".join(source_parts)
                    
                    # 2. case_paragraph (판례) 처리
                    elif source_type == "case_paragraph":
                        court = doc.get("court") or metadata.get("court")
                        casenames = doc.get("casenames") or metadata.get("casenames")
                        doc_id = doc.get("doc_id") or metadata.get("doc_id")
                        
                        if court or casenames:
                            source_parts = []
                            if court:
                                source_parts.append(court)
                            if casenames:
                                source_parts.append(casenames)
                            if doc_id:
                                source_parts.append(f"({doc_id})")
                            source = " ".join(source_parts)
                    
                    # 3. decision_paragraph (결정례) 처리
                    elif source_type == "decision_paragraph":
                        org = doc.get("org") or metadata.get("org")
                        doc_id = doc.get("doc_id") or metadata.get("doc_id")
                        
                        if org:
                            source_parts = [org]
                            if doc_id:
                                source_parts.append(f"({doc_id})")
                            source = " ".join(source_parts)
                    
                    # 4. interpretation_paragraph (해석례) 처리
                    elif source_type == "interpretation_paragraph":
                        org = doc.get("org") or metadata.get("org")
                        title = doc.get("title") or metadata.get("title")
                        
                        if org or title:
                            source_parts = []
                            if org:
                                source_parts.append(org)
                            if title:
                                source_parts.append(title)
                            source = " ".join(source_parts)
                    
                    # 5. 기존 로직 (source_type이 없는 경우 또는 위에서 source를 찾지 못한 경우)
                    if not source:
                        source_raw = (
                            doc.get("statute_name") or
                            doc.get("law_name") or
                            doc.get("source_name") or
                            doc.get("source")
                        )
                        
                        if source_raw and isinstance(source_raw, str):
                            source_lower = source_raw.lower().strip()
                            invalid_sources = ["semantic", "keyword", "unknown", "fts", "vector", "search", "text2sql", ""]
                            # 한글 법령명은 2자 이상이면 유효 (예: "민법", "형법")
                            if source_lower not in invalid_sources and len(source_lower) >= 2:
                                source = source_raw.strip()
                        
                        if not source:
                            source = (
                                metadata.get("statute_name") or
                                metadata.get("statute_abbrv") or
                                metadata.get("law_name") or
                                metadata.get("court") or
                                metadata.get("org") or
                                metadata.get("title")
                            )
                        
                        if not source:
                            content = doc.get("content", "") or doc.get("text", "")
                            if isinstance(content, str) and content:
                                import re
                                law_pattern = re.search(r'([가-힣]+법)\s*(?:제\d+조)?', content[:200])
                                if law_pattern:
                                    source = law_pattern.group(1)

                    # 소스 문자열 변환 및 중복 제거
                    if source:
                        if isinstance(source, str):
                            source_str = source.strip()
                        else:
                            try:
                                source_str = str(source).strip()
                            except Exception:
                                source_str = None
                        
                        # 검색 타입 필터링 (최종 검증)
                        if source_str:
                            source_lower = source_str.lower().strip()
                            invalid_sources = ["semantic", "keyword", "unknown", "fts", "vector", "search", "text2sql", ""]
                            # 한글 법령명은 2자 이상이면 유효 (예: "민법", "형법")
                            if source_lower not in invalid_sources and len(source_lower) >= 2:
                                if source_str not in seen_sources and source_str != "Unknown":
                                    final_sources_list.append(source_str)
                                    seen_sources.add(source_str)
                                    
                                    # sources_detail 추가
                                    if source_info_detail:
                                        final_sources_detail.append({
                                            "name": source_info_detail.name,
                                            "type": source_info_detail.type,
                                            "url": source_info_detail.url or "",
                                            "metadata": source_info_detail.metadata or {}
                                        })
                
                # sources가 없어서 생성한 경우에만 state에 저장
                state["sources"] = final_sources_list[:MAX_SOURCES_LIMIT]
                state["sources_detail"] = final_sources_detail[:MAX_SOURCES_LIMIT]

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

            # sources 표준화 및 중복 제거 (개선: 포맷팅 향상)
            # 주의: prepare_final_response_part에서 이미 sources를 정규화했을 수 있으므로,
            # existing_sources가 있으면 표준화 로직 건너뛰기
            if not sources_skipped:
                try:
                    src = state.get("sources", [])
                    if not src or not isinstance(src, list):
                        pass
                    elif len(src) > 0 and isinstance(src[0], dict):
                        # 딕셔너리 형태의 sources만 정규화
                        norm = []
                        seen = set()
                        for s in src:
                            if isinstance(s, dict):
                                # dict 형식의 소스는 더 자세한 정보 추출
                                source_name = s.get("statute_name") or s.get("law_name") or s.get("title") or s.get("source_name")
                                article = s.get("article_number") or s.get("article")
                                if source_name:
                                    formatted_source = f"{source_name}"
                                    if article:
                                        formatted_source += f" {article}"
                                else:
                                    formatted_source = str(s.get("sql") or s.get("url") or s.get("type", "Unknown"))

                                key = formatted_source.lower()
                                if key in seen:
                                    continue
                                seen.add(key)
                                norm.append(formatted_source)
                            elif isinstance(s, str):
                                if s in seen:
                                    continue
                                seen.add(s.lower())
                                norm.append(s)

                        # 정렬 (긴 이름 우선) 및 제한
                        state["sources"] = sorted(norm[:MAX_SOURCES_LIMIT], key=len, reverse=True)
                    # 이미 문자열 리스트인 경우 정규화 불필요 (prepare_final_response_part에서 이미 정규화됨)
                except Exception as e:
                    self.logger.warning(f"Error formatting sources: {e}")
                    pass
            else:
                self.logger.debug("[PREPARE_FINAL_RESPONSE] Skipping sources normalization (existing sources preserved)")

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

    def _extract_metadata_sections(self, answer_text: str) -> Dict[str, str]:
        """답변 텍스트에서 메타 정보 섹션 추출"""
        import re

        metadata = {
            "confidence_info": "",
            "reference_materials": "",
            "disclaimer": ""
        }

        if not answer_text or not isinstance(answer_text, str):
            return metadata

        # 신뢰도 정보 섹션 추출
        confidence_match = re.search(
            r'###\s*💡\s*신뢰도정보.*?(?=\n###|\n---|\Z)',
            answer_text,
            flags=re.DOTALL | re.IGNORECASE
        )
        if confidence_match:
            metadata["confidence_info"] = confidence_match.group(0).strip()

        # 참고 자료 섹션 추출
        reference_match = re.search(
            r'###\s*📚\s*참고\s*자료.*?(?=\n###|\n---|\Z)',
            answer_text,
            flags=re.DOTALL | re.IGNORECASE
        )
        if reference_match:
            metadata["reference_materials"] = reference_match.group(0).strip()

        # 면책 조항 섹션 추출 (--- 이후 부분)
        disclaimer_match = re.search(
            r'---\s*\n\s*💼\s*\*\*면책\s*조항\*\*.*?(?=\n###|\Z)',
            answer_text,
            flags=re.DOTALL | re.IGNORECASE
        )
        if disclaimer_match:
            metadata["disclaimer"] = disclaimer_match.group(0).strip()

        return metadata

    def _remove_metadata_sections(self, answer_text: str) -> str:
        """답변 텍스트에서 메타 정보 섹션 제거 (리팩토링된 메서드 사용)"""
        return self.answer_cleaner.remove_metadata_sections(answer_text)

    def _remove_answer_header(self, answer_text: str) -> str:
        """답변 텍스트에서 '## 답변' 헤더 제거 (리팩토링된 메서드 사용)"""
        return self.answer_cleaner.remove_answer_header(answer_text)

    def _remove_intermediate_text(self, answer_text: str) -> str:
        """중간 생성 텍스트 제거 (리팩토링된 메서드 사용)"""
        return self.answer_cleaner.remove_intermediate_text(answer_text)

    def _adjust_answer_length(
        self,
        answer: str,
        query_type: str,
        query_complexity: str,
        grounding_score: Optional[float] = None,
        quality_score: Optional[float] = None
    ) -> str:
        """답변 길이를 질의 유형에 맞게 조절 (리팩토링된 메서드 사용)"""
        return self.length_adjuster.adjust_length(
            answer, query_type, query_complexity, grounding_score, quality_score
        )

    def _calculate_consistent_confidence(
        self,
        base_confidence: float,
        query_type: str,
        query_complexity: str,
        grounding_score: Optional[float] = None,
        source_coverage: Optional[float] = None
    ) -> float:
        """
        일관된 신뢰도 계산

        Args:
            base_confidence: 기본 신뢰도
            query_type: 질의 유형
            query_complexity: 질의 복잡도
            grounding_score: 검증 점수 (선택적)
            source_coverage: 소스 커버리지 (선택적)

        Returns:
            조정된 신뢰도
        """
        # 1. 기본 신뢰도 조정
        confidence = base_confidence

        # 2. 질의 복잡도에 따른 조정
        complexity_adjustments = {
            "simple": 0.05,      # 간단한 질의: +5%
            "moderate": 0.0,      # 보통: 변화 없음
            "complex": -0.05      # 복잡한 질의: -5%
        }
        confidence += complexity_adjustments.get(query_complexity or "moderate", 0.0)

        # 개선 사항 8: 신뢰도 70% 달성 - 검증 점수 패널티 완화 및 보너스 추가
        # 3. 검증 점수에 따른 조정 (있는 경우) - 패널티 완화
        if grounding_score is not None:
            if grounding_score < 0.5:
                # 낮은 grounding_score에 대한 패널티 완화 (0.3 -> 0.2)
                confidence -= (0.5 - grounding_score) * 0.2
            elif grounding_score >= 0.5:
                # 높은 grounding_score에 대한 보너스 추가
                grounding_bonus = (grounding_score - 0.5) * 0.15  # 0.5 이상일 때 보너스
                confidence += grounding_bonus
                self.logger.info(f"[CONFIDENCE CALC] Grounding bonus applied: +{grounding_bonus:.3f}")

        # 4. 소스 커버리지에 따른 조정 (있는 경우) - 패널티 완화 및 보너스 추가
        if source_coverage is not None:
            if source_coverage < 0.3:
                # 낮은 source_coverage에 대한 패널티 완화 (0.2 -> 0.15)
                confidence -= (0.3 - source_coverage) * 0.15
            elif source_coverage >= 0.3:
                # 높은 source_coverage에 대한 보너스 추가
                coverage_bonus = (source_coverage - 0.3) * 0.1  # 0.3 이상일 때 보너스
                confidence += coverage_bonus
                self.logger.info(f"[CONFIDENCE CALC] Source coverage bonus applied: +{coverage_bonus:.3f}")

        # 5. 범위 제한 (0.0 ~ 1.0)
        confidence = max(0.0, min(1.0, confidence))

        # 개선 사항 8: 질의 유형별 최소 신뢰도 설정 및 강제 보장
        min_confidence_by_type = {
            "simple_question": 0.75,
            "term_explanation": 0.80,
            "legal_analysis": 0.75,
            "complex_question": 0.70,
            "general_question": 0.70
        }
        min_confidence = min_confidence_by_type.get(query_type, 0.70)

        # 최소 신뢰도보다 낮으면 최소 신뢰도로 조정 (70% 달성 보장)
        # 단, 신뢰도가 낮은 이유를 로깅하여 추적 가능하도록 개선
        if confidence < min_confidence:
            # 신뢰도가 낮은 이유 분석
            reasons = []
            if grounding_score is not None and grounding_score < 0.5:
                reasons.append(f"낮은 grounding_score ({grounding_score:.2%})")
            if source_coverage is not None and source_coverage < 0.3:
                reasons.append(f"낮은 source_coverage ({source_coverage:.2%})")
            if base_confidence < 0.5:
                reasons.append(f"낮은 base_confidence ({base_confidence:.2%})")
            
            reason_str = ", ".join(reasons) if reasons else "알 수 없는 이유"
            self.logger.warning(
                f"신뢰도가 최소 기준({min_confidence:.2%})보다 낮음: {confidence:.2%}. "
                f"최소 신뢰도로 조정합니다. (이유: {reason_str})"
            )
            # 최소 신뢰도로 조정 (70% 달성 보장)
            confidence = min_confidence

        return confidence

    def _extract_source_from_content(self, content: str) -> Optional[str]:
        """Content에서 source 추출 (강화된 키워드 추출)"""
        if not content or not isinstance(content, str):
            return None
        
        import re
        
        # 1. 법령명 추출 (더 다양한 패턴)
        law_patterns = [
            r'([가-힣]+법)\s*(?:제\d+조)?',
            r'([가-힣]+법령)',
            r'([가-힣]+규칙)',
            r'([가-힣]+시행령)'
        ]
        for pattern in law_patterns:
            match = re.search(pattern, content[:500])
            if match:
                return match.group(1)
        
        # 2. 판례/법원 정보 추출
        court_patterns = [
            r'(대법원|지방법원|고등법원|특허법원|가정법원|행정법원)',
            r'([가-힣]+고등법원)',
            r'([가-힣]+지방법원)'
        ]
        for pattern in court_patterns:
            match = re.search(pattern, content[:500])
            if match:
                court = match.group(1)
                # 판례 번호도 함께 추출 시도
                case_num = re.search(r'(\d{4}[가-힣]\d+)', content[:500])
                if case_num:
                    return f"{court} {case_num.group(1)}"
                return court
        
        # 3. 기관명 추출
        org_patterns = [
            r'([가-힣]+부)',
            r'([가-힣]+청)',
            r'([가-힣]+원)'
        ]
        for pattern in org_patterns:
            match = re.search(pattern, content[:300])
            if match:
                return match.group(1)
        
        # 4. 첫 문장의 핵심 단어 추출
        if len(content.strip()) > 20:
            first_sentence = content.split('。')[0].split('.')[0].split('!')[0][:100]
            # 한글 단어만 추출 (2자 이상)
            words = re.findall(r'[가-힣]{2,}', first_sentence)
            if words:
                return words[0]  # 첫 번째 의미있는 단어
        
        # 5. content의 처음 50자 사용 (최후의 수단)
        if len(content.strip()) > 10:
            return content[:50].strip() + "..."
        
        return None
    
    def _combine_fields_for_source(self, doc: Dict[str, Any], metadata: Dict[str, Any], source_type: Optional[str]) -> Optional[str]:
        """복합 필드 조합 방식으로 source 생성"""
        source_parts = []
        
        # source_type 기반 접두사
        if source_type:
            type_prefix = {
                "statute_article": "법령",
                "case_paragraph": "판례",
                "decision_paragraph": "결정례",
                "interpretation_paragraph": "해석례"
            }.get(source_type, "")
            if type_prefix:
                source_parts.append(type_prefix)
        
        # 여러 필드를 조합
        fields_to_try = [
            ("statute_name", "law_name"),
            ("title", "case_name", "casenames"),
            ("court", "org"),
            ("doc_id", "id", "case_id", "decision_id")
        ]
        
        for field_group in fields_to_try:
            for field in field_group:
                value = doc.get(field) or metadata.get(field)
                if value and isinstance(value, str) and len(value.strip()) >= 2:
                    source_parts.append(value.strip()[:30])  # 최대 30자
                    break
            if source_parts:
                break
        
        if source_parts:
            return " ".join(source_parts)
        
        return None
    
    def _generate_hash_based_source(self, content: str, doc_index: int) -> str:
        """해시 기반 고유 식별자 생성"""
        import hashlib
        
        if content and len(content.strip()) > 10:
            # content의 처음 100자를 해시하여 고유 식별자 생성
            content_hash = hashlib.md5(content[:100].encode('utf-8')).hexdigest()[:8]
            return f"문서 #{content_hash}"
        else:
            # content가 없으면 인덱스 기반
            return f"문서 {doc_index}"

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
        try:
            self.logger.info("[PREPARE_FINAL_RESPONSE_PART] Starting prepare_final_response_part")
            self.logger.info(f"[PREPARE_FINAL_RESPONSE_PART] State keys: {list(state.keys())[:15]}")

            query_type = self._restore_query_type_enhanced(state)
            
            if query_complexity:
                self.preserve_and_store_values(state, query_complexity, needs_search)

            if self.answer_generator:
                self.answer_generator.track_search_to_answer_pipeline(state)

            if len(state.get("processing_steps", [])) > MAX_PROCESSING_STEPS:
                state["processing_steps"] = prune_processing_steps(
                    state["processing_steps"],
                    max_items=MAX_PROCESSING_STEPS
                )

            errors = WorkflowUtils.get_state_value(state, "errors", [])
            if len(errors) > 10:
                WorkflowUtils.set_state_value(state, "errors", errors[-10:])

            answer_value = self._recover_and_validate_answer(state)
            
            retrieved_docs = self._restore_retrieved_docs_enhanced(state)
            sources_list = [doc for doc in retrieved_docs if isinstance(doc, dict)]

            query_type = WorkflowUtils.get_state_value(state, "query_type", "general")
            query_complexity = WorkflowUtils.get_state_value(state, "query_complexity", "moderate")
            needs_search = WorkflowUtils.get_state_value(state, "needs_search", True)

            keyword_coverage = self._calculate_and_set_confidence(
                state, answer_value, sources_list, query_type, query_complexity, needs_search
            )

            final_sources_list, final_sources_detail, legal_refs = self._extract_and_process_sources(state)

            state["sources"] = final_sources_list[:MAX_SOURCES_LIMIT]
            state["sources_detail"] = final_sources_detail[:MAX_SOURCES_LIMIT]
            state["legal_references"] = legal_refs[:MAX_LEGAL_REFERENCES_LIMIT] if isinstance(legal_refs, list) else legal_refs
            
            final_sources_detail = self._generate_fallback_sources_detail_if_needed(
                final_sources_detail, final_sources_list, retrieved_docs, answer_value
            )
            state["sources_detail"] = final_sources_detail[:MAX_SOURCES_LIMIT]

            self._extract_and_store_related_questions(state)

            self.set_metadata(state, answer_value, keyword_coverage)
            
            # 메모리 최적화: 중간 데이터 정리
            self._cleanup_intermediate_data(state)
        except Exception as e:
            self.logger.error(f"[PREPARE_FINAL_RESPONSE_PART] Error in prepare_final_response_part: {e}", exc_info=True)
            # 에러 발생 시에도 최소한의 상태는 유지
            if "answer" not in state:
                state["answer"] = ""
            if "sources" not in state:
                state["sources"] = []
            if "legal_references" not in state:
                state["legal_references"] = []
            if "sources_detail" not in state:
                state["sources_detail"] = []
    
    def _cleanup_intermediate_data(self, state: LegalWorkflowState) -> None:
        """메모리 최적화: 중간 데이터 정리"""
        try:
            # retrieved_docs는 이미 sources로 변환되었으므로 크기 제한
            if "retrieved_docs" in state and isinstance(state["retrieved_docs"], list):
                if len(state["retrieved_docs"]) > MAX_SOURCES_LIMIT:
                    state["retrieved_docs"] = state["retrieved_docs"][:MAX_SOURCES_LIMIT]
            
            # processing_steps는 이미 제한되어 있지만, 추가로 확인
            if "processing_steps" in state and isinstance(state["processing_steps"], list):
                if len(state["processing_steps"]) > MAX_PROCESSING_STEPS:
                    state["processing_steps"] = prune_processing_steps(
                        state["processing_steps"],
                        max_items=MAX_PROCESSING_STEPS
                    )
            
            # errors 리스트 크기 제한
            if "errors" in state and isinstance(state["errors"], list):
                if len(state["errors"]) > 10:
                    state["errors"] = state["errors"][-10:]
            
            self.logger.debug("[PREPARE_FINAL_RESPONSE_PART] Cleaned up intermediate data")
        except Exception as e:
            self.logger.warning(f"[PREPARE_FINAL_RESPONSE_PART] Error during cleanup: {e}")
    
    def _recover_and_validate_answer(self, state: LegalWorkflowState) -> str:
        """답변 복구 및 검증"""
        answer_value = WorkflowUtils.normalize_answer(state.get("answer", ""))
        
        if not answer_value or len(answer_value.strip()) < 10:
            raw_answer = state.get("answer", "")
            self.logger.warning(
                f"[PREPARE_FINAL_RESPONSE_PART] ⚠️ Answer is too short or empty: "
                f"normalized_length={len(answer_value) if answer_value else 0}, "
                f"raw_answer_length={len(raw_answer) if raw_answer else 0}, "
                f"raw_answer_preview={repr(raw_answer[:100]) if raw_answer else 'None'}, "
                f"state_answer_type={type(state.get('answer')).__name__ if state.get('answer') else 'None'}"
            )
            
            answer_value = self._recover_answer_from_state(state, answer_value)
            
            if not answer_value or len(answer_value.strip()) < 10:
                answer_value = self._generate_fallback_answer(state, answer_value)
        
        return answer_value
    
    def _recover_answer_from_state(self, state: LegalWorkflowState, current_answer: str) -> str:
        """여러 위치에서 답변 복구 시도"""
        answer_candidates = [
            state.get("answer", ""),
            state.get("common", {}).get("answer", "") if isinstance(state.get("common"), dict) else "",
            state.get("metadata", {}).get("answer", "") if isinstance(state.get("metadata"), dict) else "",
        ]
        
        for i, candidate in enumerate(answer_candidates):
            if candidate and len(str(candidate).strip()) > len(current_answer):
                recovered = WorkflowUtils.normalize_answer(str(candidate))
                self.logger.info(f"[PREPARE_FINAL_RESPONSE_PART] Recovered answer from candidate {i}: length={len(recovered)}")
                return recovered
        
        return current_answer
    
    def _generate_fallback_answer(self, state: LegalWorkflowState, current_answer: str) -> str:
        """Fallback 답변 생성"""
        self.logger.warning("[PREPARE_FINAL_RESPONSE_PART] ⚠️ Answer recovery failed, attempting fallback answer generation")
        
        try:
            if self.answer_generator:
                fallback_answer = self.answer_generator.generate_fallback_answer(state)
                if fallback_answer and len(fallback_answer.strip()) >= 10:
                    answer_value = WorkflowUtils.normalize_answer(fallback_answer)
                    state["answer"] = answer_value
                    self.logger.info(f"[PREPARE_FINAL_RESPONSE_PART] Generated fallback answer: length={len(answer_value)}")
                    return answer_value
            
            return self._generate_simple_fallback_answer(state)
        except Exception as e:
            self.logger.error(f"[PREPARE_FINAL_RESPONSE_PART] Fallback answer generation failed: {e}")
            query = state.get("query", "")
            answer_value = f"질문 '{query}'에 대한 답변을 준비 중입니다."
            state["answer"] = answer_value
            return answer_value
    
    def _generate_simple_fallback_answer(self, state: LegalWorkflowState) -> str:
        """retrieved_docs 기반 간단한 답변 생성"""
        retrieved_docs_temp = self._restore_retrieved_docs_enhanced(state)
        if not retrieved_docs_temp or len(retrieved_docs_temp) == 0:
            query = state.get("query", "")
            return f"질문 '{query}'에 대한 답변을 준비 중입니다."
        
        query = state.get("query", "")
        doc_summaries = []
        for doc in retrieved_docs_temp[:3]:
            if isinstance(doc, dict):
                content = doc.get("content", "") or doc.get("text", "")
                if content and len(content) > 50:
                    summary = content[:200] + "..." if len(content) > 200 else content
                    doc_summaries.append(summary)
        
        if doc_summaries:
            simple_answer = f"질문 '{query}'에 대한 답변을 준비했습니다.\n\n" + "\n\n".join(doc_summaries)
        else:
            simple_answer = f"질문 '{query}'에 대한 답변을 생성하는 중 문제가 발생했습니다. 검색된 문서 {len(retrieved_docs_temp)}개를 참고하여 답변을 준비했습니다."
        
        state["answer"] = simple_answer
        self.logger.info(f"[PREPARE_FINAL_RESPONSE_PART] Generated simple fallback answer: length={len(simple_answer)}")
        return simple_answer
    
    def _generate_fallback_sources_detail_if_needed(
        self,
        final_sources_detail: List[Dict[str, Any]],
        final_sources_list: List[str],
        retrieved_docs: List[Dict[str, Any]],
        answer_value: str
    ) -> List[Dict[str, Any]]:
        """sources_detail이 비어있을 때 fallback 생성"""
        if final_sources_detail and len(final_sources_detail) > 0:
            return final_sources_detail
        
        self.logger.warning(
            f"[PREPARE_FINAL_RESPONSE_PART] ⚠️ sources_detail is empty: "
            f"sources_count={len(final_sources_list)}, "
            f"retrieved_docs_count={len(retrieved_docs)}, "
            f"answer_length={len(answer_value) if answer_value else 0}"
        )
        
        if not final_sources_list or len(final_sources_list) == 0:
            return []
        
        self.logger.info("[PREPARE_FINAL_RESPONSE_PART] Attempting to generate sources_detail from sources")
        fallback_sources_detail = []
        
        for source_str in final_sources_list[:MAX_SOURCES_LIMIT]:
            if not source_str or not isinstance(source_str, str) or len(source_str.strip()) == 0:
                continue
            
            matching_doc = self._find_matching_doc_for_source(source_str, retrieved_docs)
            detail_dict = self._create_source_detail_dict(source_str, matching_doc)
            fallback_sources_detail.append(detail_dict)
        
        if fallback_sources_detail:
            self.logger.info(f"[PREPARE_FINAL_RESPONSE_PART] Generated {len(fallback_sources_detail)} fallback sources_detail from sources")
        else:
            self.logger.warning("[PREPARE_FINAL_RESPONSE_PART] Failed to generate fallback sources_detail")
        
        return fallback_sources_detail
    
    def _find_matching_doc_for_source(self, source_str: str, retrieved_docs: List[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
        """retrieved_docs에서 source와 매칭되는 doc 찾기"""
        for doc in retrieved_docs:
            if not isinstance(doc, dict):
                continue
            
            doc_source = doc.get("source") or doc.get("title") or doc.get("doc_id") or ""
            if source_str in str(doc_source) or str(doc_source) in source_str:
                return doc
        
        return None
    
    def _create_source_detail_dict(self, source_str: str, matching_doc: Optional[Dict[str, Any]]) -> Dict[str, Any]:
        """source detail 딕셔너리 생성"""
        detail_dict = {
            "name": source_str,
            "type": matching_doc.get("type") or matching_doc.get("source_type") or "unknown" if matching_doc else "unknown",
            "url": "",
            "metadata": matching_doc.get("metadata", {}) if matching_doc else {}
        }
        
        if matching_doc:
            content = matching_doc.get("content") or matching_doc.get("text") or ""
            if content:
                detail_dict["content"] = content
        
        return detail_dict
    
    def _calculate_and_set_confidence(
        self,
        state: LegalWorkflowState,
        answer_value: str,
        sources_list: List[Dict[str, Any]],
        query_type: str,
        query_complexity: str,
        needs_search: bool
    ) -> float:
        """신뢰도 계산 및 설정"""
        calculated_confidence = None
        if self.confidence_calculator and answer_value:
            try:
                confidence_info = self.confidence_calculator.calculate_confidence(
                    answer=answer_value,
                    sources=sources_list,
                    question_type=query_type
                )
                calculated_confidence = confidence_info.confidence
                
                if not needs_search and not sources_list:
                    if calculated_confidence < 0.60:
                        calculated_confidence = max(calculated_confidence * 1.2, 0.60)
                        self.logger.info(f"[CONFIDENCE CALC] Direct answer confidence adjusted: {calculated_confidence:.3f} (no search)")
                    else:
                        self.logger.info(f"[CONFIDENCE CALC] Direct answer confidence: {calculated_confidence:.3f} (no search)")
                
                self.logger.info(f"ConfidenceCalculator: confidence={calculated_confidence:.3f}, factors={confidence_info.factors}")
            except Exception as e:
                self.logger.warning(f"ConfidenceCalculator failed: {e}")

        existing_confidence = state.get("structure_confidence") or state.get("confidence", 0.0)
        final_confidence = calculated_confidence if calculated_confidence is not None else existing_confidence

        search_quality_score = self._get_search_quality_score(state)
        quality_boost = search_quality_score * 0.3

        search_failed = state.get("search_failed", False)
        if not needs_search:
            if answer_value:
                answer_length = len(answer_value)
                if answer_length >= 200:
                    base_min_confidence = 0.70
                elif answer_length >= 100:
                    base_min_confidence = 0.65
                elif answer_length >= 50:
                    base_min_confidence = 0.60
                else:
                    base_min_confidence = 0.55
                self.logger.info(f"[CONFIDENCE CALC] Direct answer (no search): base_min_confidence={base_min_confidence:.3f}, answer_length={answer_length}")
            else:
                base_min_confidence = 0.50
        elif search_failed:
            base_min_confidence = 0.20 if answer_value else 0.10
            self.logger.warning(f"[CONFIDENCE CALC] Search failed, using lower base confidence: {base_min_confidence}")
        else:
            base_min_confidence = 0.35 if (answer_value and sources_list and len(sources_list) >= 3 and search_quality_score > 0.3) else \
                                   0.30 if (answer_value and sources_list) else \
                                   0.25 if answer_value else 0.15

        final_confidence = max(final_confidence, base_min_confidence) + quality_boost

        keyword_coverage = self.calculate_keyword_coverage(state, answer_value)
        keyword_boost = keyword_coverage * 0.3
        adjusted_confidence = min(0.95, final_confidence + keyword_boost)

        if sources_list:
            source_count = len(sources_list)
            if source_count >= 5:
                adjusted_confidence = min(0.95, adjusted_confidence + 0.08)
            elif source_count >= 3:
                adjusted_confidence = min(0.95, adjusted_confidence + 0.05)
            elif source_count >= 1:
                adjusted_confidence = min(0.95, adjusted_confidence + 0.02)

        if answer_value:
            answer_length = len(answer_value)
            if answer_length >= 500:
                adjusted_confidence = min(0.95, adjusted_confidence + 0.05)
            elif answer_length >= 200:
                adjusted_confidence = min(0.95, adjusted_confidence + 0.03)
            elif answer_length >= 100:
                adjusted_confidence = min(0.95, adjusted_confidence + 0.01)

        grounding_score = state.get("grounding_score")
        source_coverage = state.get("source_coverage")

        citation_count = 0
        if answer_value:
            citation_patterns = [
                r'[가-힣]+법\s*제?\s*\d+\s*조',
                r'\[법령:\s*[^\]]+\]',
                r'제\d+조',
            ]
            unique_citations = set()
            for pattern in citation_patterns:
                matches = re.findall(pattern, answer_value)
                for match in matches:
                    unique_citations.add(match)
            citation_count = len(unique_citations)

        citation_boost = 0.0
        if citation_count >= 3:
            citation_boost = 0.10
            self.logger.info(f"[CONFIDENCE CALC] Citation boost applied: {citation_count} citations found (+{citation_boost})")
        elif citation_count >= 2:
            citation_boost = 0.08
            self.logger.info(f"[CONFIDENCE CALC] Citation boost applied: {citation_count} citations found (+{citation_boost})")
        elif citation_count >= 1:
            citation_boost = 0.03
            self.logger.info(f"[CONFIDENCE CALC] Citation boost applied: {citation_count} citation found (+{citation_boost})")

        grounding_boost = 0.0
        if grounding_score is not None:
            grounding_boost = float(grounding_score) * 0.15
            self.logger.info(f"[CONFIDENCE CALC] Grounding boost applied: grounding_score={grounding_score:.3f} (+{grounding_boost:.3f})")

        adjusted_confidence_with_validation = min(0.95, adjusted_confidence + citation_boost + grounding_boost)

        final_adjusted_confidence = self._calculate_consistent_confidence(
            base_confidence=adjusted_confidence_with_validation,
            query_type=query_type,
            query_complexity=query_complexity or "moderate",
            grounding_score=grounding_score if (needs_search or grounding_score is not None) else None,
            source_coverage=source_coverage if (needs_search or source_coverage is not None) else None
        )

        state["confidence"] = final_adjusted_confidence

        current_answer = state.get("answer", "")
        if current_answer and isinstance(current_answer, str) and final_adjusted_confidence > 0:
            state["answer"] = self.confidence_manager.replace_in_text(current_answer, final_adjusted_confidence)

        try:
            state["answer"] = WorkflowUtils.normalize_answer(state.get("answer", ""))
        except Exception:
            state["answer"] = str(state.get("answer", ""))

        if final_adjusted_confidence > 0 and state.get("answer"):
            current_answer = state.get("answer", "")
            if isinstance(current_answer, str):
                state["answer"] = self.confidence_manager.replace_in_text(current_answer, final_adjusted_confidence)

        return keyword_coverage

    def _get_search_quality_score(self, state: LegalWorkflowState) -> float:
        """search_quality 점수 추출"""
        search_quality_dict = None
        
        try:
            from core.workflow.state.state_helpers import get_field
            search_quality_dict = get_field(state, "search_quality")
            if not search_quality_dict or not isinstance(search_quality_dict, dict):
                search_quality_dict = get_field(state, "search_quality_evaluation")
        except Exception as e:
            self.logger.debug(f"Failed to get search_quality via get_field: {e}")
        
        if not search_quality_dict or not isinstance(search_quality_dict, dict):
            search_quality_dict = state.get("search_quality", {})
        if not search_quality_dict or not isinstance(search_quality_dict, dict):
            if "search" in state and isinstance(state.get("search"), dict):
                search_quality_dict = state["search"].get("search_quality", {}) or state["search"].get("search_quality_evaluation", {})
        if not search_quality_dict or not isinstance(search_quality_dict, dict):
            if "common" in state and isinstance(state.get("common"), dict):
                if "search" in state["common"] and isinstance(state["common"]["search"], dict):
                    search_quality_dict = state["common"]["search"].get("search_quality", {}) or state["common"]["search"].get("search_quality_evaluation", {})
        if not search_quality_dict or not isinstance(search_quality_dict, dict):
            search_quality_dict = state.get("search_quality_evaluation", {})
        if not search_quality_dict or not isinstance(search_quality_dict, dict):
            metadata = state.get("metadata", {})
            if isinstance(metadata, dict):
                search_quality_dict = metadata.get("search_quality", {}) or metadata.get("search_quality_evaluation", {})
        
        if not search_quality_dict or not isinstance(search_quality_dict, dict):
            try:
                from core.agents.node_wrappers import _global_search_results_cache
                if _global_search_results_cache and isinstance(_global_search_results_cache, dict):
                    if "search" in _global_search_results_cache and isinstance(_global_search_results_cache["search"], dict):
                        cached_quality = _global_search_results_cache["search"].get("search_quality", {}) or \
                                        _global_search_results_cache["search"].get("search_quality_evaluation", {})
                        if cached_quality and isinstance(cached_quality, dict):
                            search_quality_dict = cached_quality
                            self.logger.info(f"✅ [CONFIDENCE CALC] Found search_quality in global cache: {list(cached_quality.keys())}")
            except Exception as e:
                self.logger.debug(f"Failed to get search_quality from global cache: {e}")
        
        if search_quality_dict and isinstance(search_quality_dict, dict):
            search_quality_score = search_quality_dict.get("overall_quality", 0.0)
        elif search_quality_dict and isinstance(search_quality_dict, (int, float)):
            search_quality_score = float(search_quality_dict)
        else:
            search_quality_score = 0.0
        
        self.logger.info(f"[CONFIDENCE CALC] search_quality_score: {search_quality_score:.3f}")
        return search_quality_score

    def _extract_and_process_sources(
        self,
        state: LegalWorkflowState
    ) -> tuple[List[str], List[Dict[str, Any]], List[str]]:
        """sources 추출 및 처리"""
        final_sources_list = []
        final_sources_detail = []
        seen_sources = set()
        legal_refs = []
        seen_legal_refs = set()

        try:
            from ...generation.formatters.unified_source_formatter import UnifiedSourceFormatter
            from ...generation.validators.source_validator import SourceValidator
            formatter = UnifiedSourceFormatter()
            validator = SourceValidator()
        except ImportError:
            formatter = None
            validator = None

        retrieved_docs_list = self._restore_retrieved_docs_enhanced(state)
        total_docs = len(retrieved_docs_list)
        self.logger.info(f"[SOURCES] Processing {total_docs} retrieved_docs in prepare_final_response_part")

        sources_created_count = 0
        sources_failed_count = 0

        for doc_index, doc in enumerate(retrieved_docs_list, 1):
            if not isinstance(doc, dict):
                self.logger.warning(f"[SOURCES] Doc {doc_index}/{total_docs} is not a dict, skipping")
                sources_failed_count += 1
                continue

            source = None
            source_created = False
            source_type = doc.get("type") or doc.get("source_type") or doc.get("metadata", {}).get("source_type", "")
            metadata = doc.get("metadata", {}) if isinstance(doc.get("metadata"), dict) else {}
            # doc_id 추출: 여러 위치에서 확인 (우선순위 순)
            doc_id = (
                doc.get("doc_id") or 
                metadata.get("doc_id") or 
                metadata.get("case_id") or 
                metadata.get("decision_id") or 
                metadata.get("interpretation_id") or
                metadata.get("id") or
                ""
            )
            
            # 디버깅: case_paragraph인데 doc_id가 없으면 로깅
            if source_type == "case_paragraph" and not doc_id:
                self.logger.warning(
                    f"[SOURCES] ⚠️ case_paragraph에 doc_id가 없습니다 (doc {doc_index}/{total_docs}). "
                    f"doc keys: {list(doc.keys())}, metadata keys: {list(metadata.keys())}, "
                    f"doc.get('doc_id'): {doc.get('doc_id')}, metadata.get('doc_id'): {metadata.get('doc_id')}"
                )
        
            if not source_type:
                content_for_inference = doc.get("content", "") or doc.get("text", "")
                if isinstance(content_for_inference, str) and len(content_for_inference) > 10:
                    import re
                    if re.search(r'[가-힣]+법\s*제\s*\d+\s*조', content_for_inference[:500]):
                        source_type = "statute_article"
                    elif re.search(r'(대법원|지방법원|고등법원|법원)\s*\d+[가-힣]+\s*\d+', content_for_inference[:500]) or \
                         re.search(r'선고\s*\d+[가-힣]+\s*\d+', content_for_inference[:500]):
                        source_type = "case_paragraph"
                    elif re.search(r'(결정|의결)', content_for_inference[:500]):
                        source_type = "decision_paragraph"
                    elif re.search(r'(해석|의견|질의)', content_for_inference[:500]):
                        source_type = "interpretation_paragraph"
        
            self.logger.info(f"[SOURCES] Processing doc {doc_index}/{total_docs}: type={source_type or 'none'}, doc_id={doc_id or 'none'}")

            source_info_detail = None
            formatter_error = None
            if formatter and source_type:
                try:
                    merged_metadata = {**metadata}
                    for key in ["statute_name", "law_name", "article_no", "article_number", "clause_no", "item_no",
                               "court", "doc_id", "casenames", "org", "title", "announce_date", "decision_date", "response_date"]:
                        if key in doc:
                            merged_metadata[key] = doc[key]
                    
                    source_info_detail = formatter.format_source(source_type, merged_metadata)
                    
                    if validator:
                        validation_result = validator.validate_source(source_type, merged_metadata)
                        source_info_detail.validation = validation_result
                except Exception as e:
                    formatter_error = str(e)
                    self.logger.warning(f"[SOURCES_DETAIL] Error formatting source detail for doc {doc_index}/{total_docs}: {e}")

            source = self._create_source_from_doc(doc, metadata, source_type, doc_id)
            
            if source:
                source_str = str(source).strip() if isinstance(source, str) else str(source).strip()
                source_lower = source_str.lower().strip()
                invalid_sources = ["semantic", "keyword", "unknown", "fts", "vector", "search", "text2sql", ""]
                
                is_valid_source = False
                if source_type and source_type in ["statute_article", "case_paragraph", "decision_paragraph", "interpretation_paragraph"]:
                    if source_lower not in invalid_sources and len(source_lower) >= 1:
                        is_valid_source = True
                    elif source_lower not in invalid_sources:
                        is_valid_source = True
                else:
                    if source_lower not in invalid_sources and len(source_lower) >= 1:
                        is_valid_source = True
                    elif any(ord(c) >= 0xAC00 and ord(c) <= 0xD7A3 for c in source_lower) or any(c.isdigit() for c in source_lower):
                        is_valid_source = True
                
                if is_valid_source:
                    source_key = f"{source_str}::{doc_id}" if doc_id else source_str
                    if source_key not in seen_sources and source_str != "Unknown":
                        final_sources_list.append(source_str)
                        seen_sources.add(source_key)
                        source_created = True
                        sources_created_count += 1
                        self.logger.info(f"[SOURCES] ✅ Successfully created source for doc {doc_index}/{total_docs}: {source_str}")
                        
                        if source_type == "statute_article":
                            legal_ref = self._extract_legal_ref_from_source(source_str, doc, metadata)
                            if legal_ref and legal_ref not in seen_legal_refs:
                                legal_refs.append(legal_ref)
                                seen_legal_refs.add(legal_ref)
                        
                        detail_dict = self._create_source_detail_dict(
                            source_str, source_type, source_info_detail, doc, metadata, formatter_error
                        )
                        if detail_dict:
                            final_sources_detail.append(detail_dict)
            
            if not source_created:
                source = self._create_fallback_source(doc, metadata, source_type, doc_id, doc_index)
                if source:
                    source_str = str(source).strip()
                    source_key = f"{source_str}::{doc_id}" if doc_id else source_str
                    if source_key not in seen_sources:
                        final_sources_list.append(source_str)
                        seen_sources.add(source_key)
                        source_created = True
                        sources_created_count += 1
                        self.logger.info(f"[SOURCES] ✅ Generated fallback source for doc {doc_index}/{total_docs}: {source_str}")
                        
                        detail_dict = self._create_source_detail_dict(
                            source_str, source_type, None, doc, metadata, None
                        )
                        if detail_dict:
                            final_sources_detail.append(detail_dict)
            
            if not source_created:
                # 개선: final fallback으로 source가 생성되므로 sources_failed_count 증가하지 않음
                # 더 구체적인 fallback source 생성 (doc_id, content 일부 등 활용)
                content_preview = ""
                if isinstance(doc, dict):
                    content = doc.get("content", "") or doc.get("text", "")
                    if content and isinstance(content, str) and len(content) > 10:
                        content_preview = content[:50].strip().replace("\n", " ")
                
                if doc_id:
                    final_fallback_source = f"문서 {doc_id}"
                elif content_preview:
                    final_fallback_source = f"문서 {doc_index}: {content_preview}"
                else:
                    final_fallback_source = f"문서 {doc_index}"
                
                # 중복 체크를 위해 더 구체적인 키 사용
                source_key = f"{final_fallback_source}::{doc_id}::{doc_index}" if doc_id else f"{final_fallback_source}::{doc_index}"
                
                if source_key not in seen_sources:
                    final_sources_list.append(final_fallback_source)
                    seen_sources.add(source_key)
                    source_created = True
                    sources_created_count += 1
                    self.logger.warning(f"[SOURCES] ⚠️ CRITICAL: Using final fallback for doc {doc_index}/{total_docs}: {final_fallback_source}")
                    
                    # final fallback으로도 detail 생성 보장 (source와 동시에 추가)
                    detail_dict = self._create_source_detail_dict(
                        final_fallback_source, source_type, None, doc, metadata, None
                    )
                    if detail_dict:
                        final_sources_detail.append(detail_dict)
                else:
                    # 중복이지만 다른 형태로 source 생성 시도
                    alt_fallback_source = f"참고문서 {doc_index}"
                    alt_source_key = f"{alt_fallback_source}::{doc_id}::{doc_index}" if doc_id else f"{alt_fallback_source}::{doc_index}"
                    if alt_source_key not in seen_sources:
                        final_sources_list.append(alt_fallback_source)
                        seen_sources.add(alt_source_key)
                        source_created = True
                        sources_created_count += 1
                        self.logger.warning(f"[SOURCES] ⚠️ CRITICAL: Using alternative fallback for doc {doc_index}/{total_docs}: {alt_fallback_source}")
                        
                        detail_dict = self._create_source_detail_dict(
                            alt_fallback_source, source_type, None, doc, metadata, None
                        )
                        if detail_dict:
                            final_sources_detail.append(detail_dict)
                    else:
                        # 실제로 source가 생성되지 않은 경우에만 실패 카운트 증가
                        sources_failed_count += 1
                        self.logger.error(f"[SOURCES] ❌ Failed to create source for doc {doc_index}/{total_docs} (even with final fallback)")
            
            # 최종 검증: source_created가 False이면 강제로 source 생성
            if not source_created:
                forced_source = f"참고자료 {doc_index}"
                forced_source_key = f"{forced_source}::{doc_id}::{doc_index}" if doc_id else f"{forced_source}::{doc_index}"
                if forced_source_key not in seen_sources:
                    final_sources_list.append(forced_source)
                    seen_sources.add(forced_source_key)
                    sources_created_count += 1
                    self.logger.warning(f"[SOURCES] ⚠️ FORCED: Created source for doc {doc_index}/{total_docs}: {forced_source}")
                    
                    detail_dict = self._create_source_detail_dict(
                        forced_source, source_type, None, doc, metadata, None
                    )
                    if detail_dict:
                        final_sources_detail.append(detail_dict)
                else:
                    sources_failed_count += 1
                    self.logger.error(f"[SOURCES] ❌ Failed to create source for doc {doc_index}/{total_docs} (even with forced creation)")

        conversion_rate = (sources_created_count / total_docs * 100) if total_docs > 0 else 0
        self.logger.info(f"[SOURCES] 📊 Conversion statistics: {sources_created_count}/{total_docs} docs converted ({conversion_rate:.1f}%), failed: {sources_failed_count}")

        normalized_sources = self._normalize_sources(final_sources_list)
        
        # sources와 sources_detail 개수 동기화 보장 (개선: 더 정확한 매칭)
        # sources_detail이 더 많으면 sources에 맞춰 조정
        if len(final_sources_detail) > len(normalized_sources):
            self.logger.warning(f"[SOURCES] ⚠️ sources_detail({len(final_sources_detail)}) > sources({len(normalized_sources)}), trimming sources_detail")
            # sources와 매칭되는 sources_detail만 유지 (이름 기반 매칭)
            matched_details = []
            for source_str in normalized_sources:
                matched = False
                for detail in final_sources_detail:
                    detail_name = detail.get("name", "")
                    if source_str == detail_name or source_str in detail_name or detail_name in source_str:
                        matched_details.append(detail)
                        matched = True
                        break
                if not matched:
                    # 매칭되지 않은 경우 기본 detail 생성
                    matched_details.append({
                        "name": source_str,
                        "type": "unknown",
                        "url": "",
                        "metadata": {}
                    })
            final_sources_detail = matched_details[:len(normalized_sources)]
        # sources가 더 많으면 sources_detail을 생성
        elif len(normalized_sources) > len(final_sources_detail):
            self.logger.warning(f"[SOURCES] ⚠️ sources({len(normalized_sources)}) > sources_detail({len(final_sources_detail)}), generating missing sources_detail")
            # 기존 sources_detail과 매칭되지 않은 sources에 대해 detail 생성
            existing_names = {detail.get("name", "") for detail in final_sources_detail}
            for idx in range(len(final_sources_detail), len(normalized_sources)):
                source_str = normalized_sources[idx]
                if source_str and source_str not in existing_names:
                    # retrieved_docs에서 해당 source와 매칭되는 doc 찾기
                    matching_doc = None
                    for doc in retrieved_docs_list:
                        if isinstance(doc, dict):
                            doc_source = self._create_source_from_doc(
                                doc, 
                                doc.get("metadata", {}), 
                                doc.get("type") or doc.get("source_type", ""),
                                doc.get("doc_id")
                            )
                            if doc_source and str(doc_source).strip() == source_str:
                                matching_doc = doc
                                break
                    
                    # 매칭된 doc이 있으면 상세 정보 포함하여 detail 생성
                    if matching_doc:
                        detail_dict = self._create_source_detail_dict(
                            source_str,
                            matching_doc.get("type") or matching_doc.get("source_type", ""),
                            None,
                            matching_doc,
                            matching_doc.get("metadata", {}),
                            None
                        )
                    else:
                        # 기본 sources_detail 생성
                        detail_dict = {
                            "name": source_str,
                            "type": "unknown",
                            "url": "",
                            "metadata": {}
                        }
                    if detail_dict:
                        final_sources_detail.append(detail_dict)
                        self.logger.info(f"[SOURCES] Generated missing sources_detail[{idx}]: {source_str}")
        
        # sources 배열에서 판례명 추출하여 sources_detail의 name 및 metadata 보완
        if len(normalized_sources) > 0 and len(final_sources_detail) > 0:
            import re
            for idx, detail in enumerate(final_sources_detail):
                if idx >= len(normalized_sources):
                    continue
                    
                source_str = normalized_sources[idx]
                if not source_str or not str(source_str).strip() or source_str == "판례":
                    continue
                
                detail_name = detail.get("name") or ""
                detail_type = detail.get("type", "")
                metadata = detail.get("metadata", {})
                if not isinstance(metadata, dict):
                    metadata = {}
                    detail["metadata"] = metadata
                
                # name이 "판례"이거나 비어있는 경우, 또는 metadata가 비어있는 경우 보완
                needs_name_update = detail_name in ("판례", "")
                needs_metadata_update = (
                    detail_type == "case_paragraph" and 
                    (not str(metadata.get("court", "")).strip() and 
                     not str(metadata.get("doc_id", "")).strip() and 
                     not str(metadata.get("casenames", "")).strip())
                )
                
                if needs_name_update or needs_metadata_update:
                    # "판례명 (case_xxx)" 형식에서 판례명 추출 (정규표현식 사용)
                    pattern = r'^(.+?)\s*\(([^)]+)\)\s*$'
                    match = re.match(pattern, str(source_str).strip())
                    
                    if match:
                        case_name = match.group(1).strip()
                        doc_id_match = match.group(2).strip()
                        
                        # case_ 접두사 제거
                        clean_doc_id = doc_id_match.replace("case_", "").strip()
                        
                        # metadata 보완
                        if clean_doc_id:
                            if not detail.get("case_number"):
                                detail["case_number"] = clean_doc_id
                            if not metadata.get("doc_id"):
                                metadata["doc_id"] = clean_doc_id
                        
                        # name 보완
                        if case_name and needs_name_update:
                            detail["name"] = case_name
                            if not detail.get("case_name"):
                                detail["case_name"] = case_name
                            if not metadata.get("casenames"):
                                metadata["casenames"] = case_name
                        elif case_name and needs_metadata_update:
                            # name은 이미 있지만 metadata가 비어있는 경우
                            if not metadata.get("casenames"):
                                metadata["casenames"] = case_name
                            if not detail.get("case_name"):
                                detail["case_name"] = case_name
                        
                        # URL 생성 (doc_id가 있으면)
                        if not detail.get("url") and clean_doc_id:
                            from core.generation.formatters.unified_source_formatter import UnifiedSourceFormatter
                            formatter = UnifiedSourceFormatter()
                            detail["url"] = formatter._generate_case_url(clean_doc_id, metadata)
                    else:
                        # 괄호가 없으면 전체를 판례명으로 사용
                        clean_source = str(source_str).strip()
                        if clean_source:
                            if needs_name_update:
                                detail["name"] = clean_source
                            if not detail.get("case_name"):
                                detail["case_name"] = clean_source
                            if not metadata.get("casenames"):
                                metadata["casenames"] = clean_source
                    
                    if detail.get("name") and detail.get("name") != "판례":
                        self.logger.info(f"[SOURCES] Enhanced source detail[{idx}] from sources array: name={detail.get('name')}, case_name={detail.get('case_name')}, case_number={detail.get('case_number')}")
        
        # 개선: Legal References 추출 로깅 강화
        legal_refs_from_sources = self.source_extractor.extract_legal_references_from_sources_detail(final_sources_detail)
        legal_refs_from_docs = self.source_extractor.extract_legal_references_from_docs(retrieved_docs_list)
        
        self.logger.info(f"[LEGAL_REFS] Extracted {len(legal_refs_from_sources)} legal references from sources_detail")
        self.logger.info(f"[LEGAL_REFS] Extracted {len(legal_refs_from_docs)} legal references from retrieved_docs")
        
        legal_refs.extend(legal_refs_from_sources)
        legal_refs.extend(legal_refs_from_docs)
        
        # 중복 제거
        seen_legal_refs_set = set(seen_legal_refs)
        unique_legal_refs = []
        for ref in legal_refs:
            if ref not in seen_legal_refs_set:
                unique_legal_refs.append(ref)
                seen_legal_refs_set.add(ref)
        
        legal_refs = unique_legal_refs
        self.logger.info(f"[LEGAL_REFS] Total unique legal references: {len(legal_refs)}")

        def convert_numpy_types(obj):
            import numpy as np
            if isinstance(obj, (np.integer, np.int64, np.int32, np.int16, np.int8)):
                return int(obj)
            elif isinstance(obj, (np.floating, np.float64, np.float32, np.float16)):
                return float(obj)
            elif isinstance(obj, np.ndarray):
                return obj.tolist()
            elif isinstance(obj, dict):
                return {k: convert_numpy_types(v) for k, v in obj.items()}
            elif isinstance(obj, (list, tuple)):
                return [convert_numpy_types(item) for item in obj]
            return obj

        normalized_sources_clean = [convert_numpy_types(s) for s in normalized_sources[:MAX_SOURCES_LIMIT]]
        final_sources_detail_clean = [convert_numpy_types(d) for d in final_sources_detail[:MAX_SOURCES_LIMIT]]
        
        if final_sources_detail_clean:
            full_texts_map = self._get_full_texts_batch(final_sources_detail_clean)
            
            for detail in final_sources_detail_clean:
                source_type = detail.get("type")
                doc_id = detail.get("case_number") or detail.get("decision_number") or detail.get("interpretation_number")
                metadata = detail.get("metadata", {})
                
                if not detail.get("content") or len(detail.get("content", "")) < 500:
                    if source_type == "case_paragraph" and doc_id and doc_id in full_texts_map:
                        detail["content"] = full_texts_map[doc_id]
                        self.logger.debug(f"Added full text to case detail: {len(detail['content'])} chars")
                    elif source_type == "decision_paragraph" and doc_id and doc_id in full_texts_map:
                        detail["content"] = full_texts_map[doc_id]
                        self.logger.debug(f"Added full text to decision detail: {len(detail['content'])} chars")
                    elif source_type == "interpretation_paragraph" and doc_id and doc_id in full_texts_map:
                        detail["content"] = full_texts_map[doc_id]
                        self.logger.debug(f"Added full text to interpretation detail: {len(detail['content'])} chars")
                    elif source_type == "statute_article":
                        statute_id = metadata.get("statute_id")
                        article_no = detail.get("article_no") or metadata.get("article_no")
                        if statute_id and article_no:
                            key = f"{statute_id}_{article_no}"
                            if key in full_texts_map:
                                detail["content"] = full_texts_map[key]
                                self.logger.debug(f"Added full text to statute detail: {len(detail['content'])} chars")

        if "common" not in state:
            state["common"] = {}
        if not isinstance(state["common"], dict):
            state["common"] = {}
        state["common"]["sources"] = normalized_sources_clean

        metadata = state.get("metadata", {})
        if not isinstance(metadata, dict):
            metadata = {}
        metadata["sources"] = normalized_sources_clean
        metadata["sources_detail"] = final_sources_detail_clean
        metadata["legal_references"] = legal_refs[:MAX_LEGAL_REFERENCES_LIMIT]
        state["metadata"] = metadata
        
        # 최상위 레벨에도 저장 (workflow_service에서 추출하기 위해)
        state["sources_detail"] = final_sources_detail_clean
        state["legal_references"] = legal_refs[:MAX_LEGAL_REFERENCES_LIMIT]
        
        # common 그룹에도 저장
        if "common" not in state:
            state["common"] = {}
        if not isinstance(state["common"], dict):
            state["common"] = {}
        state["common"]["sources_detail"] = final_sources_detail_clean
        state["common"]["legal_references"] = legal_refs[:MAX_LEGAL_REFERENCES_LIMIT]

        self.logger.info(f"[SOURCES] ✅ Final sources saved to state: {len(normalized_sources_clean)} sources, {len(final_sources_detail_clean)} details, {len(legal_refs[:MAX_LEGAL_REFERENCES_LIMIT])} legal refs")
        
        # sources 데이터 상세 로깅 (개발 모드에서만)
        if os.getenv("DEBUG_SOURCES", "false").lower() == "true" or self.logger.level <= logging.DEBUG:
            self.logger.info("[SOURCES_TEST] ===== Sources Data Analysis =====")
            self.logger.info(f"[SOURCES_TEST] Sources count: {len(normalized_sources_clean)}")
            self.logger.info(f"[SOURCES_TEST] Sources detail count: {len(final_sources_detail_clean)}")
            self.logger.info(f"[SOURCES_TEST] Legal references count: {len(legal_refs[:MAX_LEGAL_REFERENCES_LIMIT])}")
            
            # sources 배열 상세 로깅
            for idx, source in enumerate(normalized_sources_clean[:5], 1):
                self.logger.info(f"[SOURCES_TEST] Source[{idx}]: {source}")
            
            # sources_detail 상세 로깅
            for idx, detail in enumerate(final_sources_detail_clean[:5], 1):
                detail_info = {
                    "name": detail.get("name"),
                    "type": detail.get("type"),
                    "case_name": detail.get("case_name"),
                    "case_number": detail.get("case_number"),
                    "court": detail.get("court"),
                    "url": detail.get("url"),
                    "metadata": detail.get("metadata", {}),
                }
                self.logger.info(f"[SOURCES_TEST] SourceDetail[{idx}]: {detail_info}")
            
            # sources와 sources_detail 매칭 확인
            if len(normalized_sources_clean) != len(final_sources_detail_clean):
                self.logger.warning(f"[SOURCES_TEST] ⚠️ Count mismatch: sources={len(normalized_sources_clean)}, sources_detail={len(final_sources_detail_clean)}")
            
            # 비어있는 metadata 확인
            empty_metadata_count = 0
            for detail in final_sources_detail_clean:
                metadata = detail.get("metadata", {})
                if isinstance(metadata, dict):
                    # 판례의 경우 court, doc_id, casenames가 모두 비어있는지 확인
                    if detail.get("type") == "case_paragraph":
                        court = metadata.get("court") or ""
                        doc_id = metadata.get("doc_id") or ""
                        casenames = metadata.get("casenames") or ""
                        if not str(court).strip() and not str(doc_id).strip() and not str(casenames).strip():
                            empty_metadata_count += 1
                            self.logger.warning(f"[SOURCES_TEST] ⚠️ Empty metadata detected: {detail.get('name')}")
            
            if empty_metadata_count > 0:
                self.logger.warning(f"[SOURCES_TEST] ⚠️ Total empty metadata count: {empty_metadata_count}")
            
            self.logger.info("[SOURCES_TEST] ===== End Sources Data Analysis =====")

        return normalized_sources_clean, final_sources_detail_clean, legal_refs[:MAX_LEGAL_REFERENCES_LIMIT]

    def _create_source_from_doc(
        self,
        doc: Dict[str, Any],
        metadata: Dict[str, Any],
        source_type: str,
        doc_id: Optional[str]
    ) -> Optional[str]:
        """doc에서 source 생성"""
        if source_type == "statute_article":
            statute_name = (
                doc.get("statute_name") or
                doc.get("law_name") or
                metadata.get("statute_name") or
                metadata.get("law_name")
            )
            if statute_name:
                article_no = (
                    doc.get("article_no") or
                    doc.get("article_number") or
                    metadata.get("article_no") or
                    metadata.get("article_number")
                )
                clause_no = doc.get("clause_no") or metadata.get("clause_no")
                item_no = doc.get("item_no") or metadata.get("item_no")
                
                source_parts = [statute_name]
                if article_no:
                    article_no_str = str(article_no) if article_no else ""
                    if article_no_str.startswith("제") and article_no_str.endswith("조"):
                        source_parts.append(article_no_str)
                    else:
                        article_no_clean = article_no_str.strip()
                        if article_no_clean:
                            source_parts.append(f"제{article_no_clean}조")
                if clause_no:
                    source_parts.append(f"제{clause_no}항")
                if item_no:
                    source_parts.append(f"제{item_no}호")
                return " ".join(source_parts)
        
        elif source_type == "case_paragraph":
            court = doc.get("court") or metadata.get("court") or metadata.get("court_name") or metadata.get("court_type")
            casenames = doc.get("casenames") or metadata.get("casenames") or metadata.get("case_name") or metadata.get("title")
            if court or casenames or doc_id:
                source_parts = []
                if court:
                    source_parts.append(court)
                if casenames:
                    source_parts.append(casenames)
                if doc_id:
                    source_parts.append(f"({doc_id})")
                if not court and not casenames and doc_id:
                    source_parts.insert(0, "판례")
                return " ".join(source_parts) if source_parts else None
        
        elif source_type == "decision_paragraph":
            org = doc.get("org") or metadata.get("org") or metadata.get("org_name") or metadata.get("organization")
            if org or doc_id:
                source_parts = []
                if org:
                    source_parts.append(org)
                if doc_id:
                    source_parts.append(f"({doc_id})")
                if not org and doc_id:
                    source_parts.insert(0, "결정례")
                return " ".join(source_parts) if source_parts else None
        
        elif source_type == "interpretation_paragraph":
            org = doc.get("org") or metadata.get("org")
            title = doc.get("title") or metadata.get("title")
            if org or title:
                source_parts = []
                if org:
                    source_parts.append(org)
                if title:
                    source_parts.append(title)
                return " ".join(source_parts)
        
        source_raw = (
            doc.get("statute_name") or
            doc.get("law_name") or
            doc.get("source_name") or
            doc.get("source")
        )
        
        if source_raw and isinstance(source_raw, str):
            source_lower = source_raw.lower().strip()
            invalid_sources = ["semantic", "keyword", "unknown", "fts", "vector", "search", "text2sql", ""]
            if source_lower not in invalid_sources and len(source_lower) >= 1:
                return source_raw.strip()
        
        source = (
            metadata.get("statute_name") or
            metadata.get("statute_abbrv") or
            metadata.get("law_name") or
            metadata.get("court") or
            metadata.get("org") or
            metadata.get("title")
        )
        
        if not source:
            content = doc.get("content", "") or doc.get("text", "")
            if isinstance(content, str) and content:
                law_pattern = re.search(r'([가-힣]+법)\s*(?:제\d+조)?', content[:200])
                if law_pattern:
                    return law_pattern.group(1)
        
        return source

    def _create_fallback_source(
        self,
        doc: Dict[str, Any],
        metadata: Dict[str, Any],
        source_type: Optional[str],
        doc_id: Optional[str],
        doc_index: int
    ) -> Optional[str]:
        """fallback source 생성 (개선: 더 많은 필드에서 source 추출)"""
        # 방법 1: source_type과 doc_id 조합
        if source_type:
            type_names = {
                "case_paragraph": "판례",
                "decision_paragraph": "결정례",
                "interpretation_paragraph": "해석례",
                "statute_article": "법령"
            }
            type_name = type_names.get(source_type, "문서")
            
            # doc_id가 있으면 조합
            if doc_id:
                return f"{type_name} ({doc_id})"
            
            # doc_id가 없어도 다른 필드로 조합 시도
            statute_name = doc.get("statute_name") or metadata.get("statute_name")
            article_no = doc.get("article_no") or metadata.get("article_no")
            court = doc.get("court") or metadata.get("court")
            org = doc.get("org") or metadata.get("org")
            title = doc.get("title") or metadata.get("title")
            
            if source_type == "statute_article" and statute_name:
                if article_no:
                    return f"{statute_name} 제{article_no}조"
                return statute_name
            elif source_type == "case_paragraph" and court:
                if title:
                    return f"{court} {title[:30]}"
                return court
            elif source_type == "interpretation_paragraph" and org:
                if title:
                    return f"{org} {title[:30]}"
                return org
            elif title and isinstance(title, str) and len(title.strip()) >= 2:
                return f"{type_name}: {title.strip()[:50]}"
        
        # 방법 2: 여러 필드에서 source 추출 시도
        title = (
            doc.get("title") or 
            metadata.get("title") or 
            metadata.get("case_name") or 
            metadata.get("casenames") or
            doc.get("casenames")
        )
        statute_name = doc.get("statute_name") or metadata.get("statute_name") or doc.get("law_name") or metadata.get("law_name")
        article_no = doc.get("article_no") or metadata.get("article_no") or doc.get("article_number") or metadata.get("article_number")
        court = doc.get("court") or metadata.get("court") or doc.get("ccourt") or metadata.get("ccourt")
        org = doc.get("org") or metadata.get("org")
        content = doc.get("content", "") or doc.get("text", "")
        
        # 조합 시도
        if statute_name and article_no:
            return f"{statute_name} 제{article_no}조"
        elif statute_name:
            return statute_name
        elif court and title:
            return f"{court} {title[:30]}"
        elif court:
            return court
        elif org and title:
            return f"{org} {title[:30]}"
        elif org:
            return org
        elif title and isinstance(title, str) and len(title.strip()) >= 2:
            return title.strip()[:50]
        elif doc_id:
            return f"문서 ({doc_id})"
        elif content and isinstance(content, str) and len(content.strip()) > 10:
            extracted = self._extract_source_from_content(content)
            if extracted:
                return extracted
            # content에서 의미있는 부분 추출
            content_preview = content[:50].strip().replace("\n", " ")
            if content_preview:
                return content_preview
        else:
            return f"문서 {doc_index}"

    def _get_full_text_from_database(
        self,
        source_type: Optional[str],
        doc: Dict[str, Any],
        metadata: Dict[str, Any]
    ) -> Optional[str]:
        """
        데이터베이스에서 전체 본문 조회
        
        Args:
            source_type: 문서 타입 (case_paragraph, decision_paragraph, interpretation_paragraph, statute_article)
            doc: 검색 결과 문서
            metadata: 메타데이터
        
        Returns:
            전체 본문 텍스트 또는 None
        """
        try:
            # Config 및 connector 인스턴스 지연 초기화 (os 변수 오류 방지)
            if not self._config_initialized:
                try:
                    from core.utils.config import Config
                    from core.search.connectors.legal_data_connector_v2 import LegalDataConnectorV2
                    self._config = Config()
                    self._connector = LegalDataConnectorV2(self._config.database_path)
                    self._config_initialized = True
                except Exception as e:
                    self.logger.warning(f"Failed to initialize Config/Connector: {e}")
                    return None
            
            if self._config is None or self._connector is None:
                self.logger.warning("Config or Connector not initialized, cannot get full text")
                return None
            
            conn = self._connector._get_connection()
            cursor = conn.cursor()
            
            full_text = None
            
            try:
                if source_type in ["case_paragraph", "precedent_content"]:  # 🔥 레거시 지원
                    doc_id = doc.get("doc_id") or metadata.get("doc_id") or metadata.get("case_id")
                    if doc_id:
                        # precedent_contents에서 텍스트를 가져옴 (PostgreSQL)
                        # 🔥 레거시: case_paragraphs 테이블은 더 이상 사용하지 않음
                        cursor.execute("""
                            SELECT STRING_AGG(pcc.section_content, E'\\n\\n' ORDER BY pcc.section_index) as full_text
                            FROM precedents p
                            JOIN precedent_contents pcc ON pcc.precedent_id = p.id
                            WHERE p.doc_id = %s
                            GROUP BY p.doc_id
                        """, (doc_id,))
                        row = cursor.fetchone()
                        if row and row[0]:
                            full_text = row[0]
                
                elif source_type == "decision_paragraph":
                    doc_id = doc.get("doc_id") or metadata.get("doc_id") or metadata.get("decision_id")
                    if doc_id:
                        cursor.execute("""
                            SELECT GROUP_CONCAT(dp.text, '\n\n') as full_text
                            FROM decision_paragraphs dp
                            JOIN decisions d ON dp.decision_id = d.id
                            WHERE d.doc_id = ?
                            GROUP BY d.doc_id
                            ORDER BY dp.para_index
                        """, (doc_id,))
                        row = cursor.fetchone()
                        if row and row[0]:
                            full_text = row[0]
                
                elif source_type == "interpretation_paragraph":
                    doc_id = doc.get("doc_id") or metadata.get("doc_id") or metadata.get("interpretation_id")
                    if doc_id:
                        cursor.execute("""
                            SELECT GROUP_CONCAT(ip.text, '\n\n') as full_text
                            FROM interpretation_paragraphs ip
                            JOIN interpretations i ON ip.interpretation_id = i.id
                            WHERE i.doc_id = ?
                            GROUP BY i.doc_id
                            ORDER BY ip.para_index
                        """, (doc_id,))
                        row = cursor.fetchone()
                        if row and row[0]:
                            full_text = row[0]
                
                elif source_type == "statute_article":
                    statute_id = doc.get("statute_id") or metadata.get("statute_id")
                    article_no = doc.get("article_no") or metadata.get("article_no") or metadata.get("article_number")
                    if statute_id and article_no:
                        # 🔥 개선: statutes_articles 테이블만 사용 (statute_articles는 레거시, 삭제됨)
                        cursor.execute("""
                            SELECT article_content
                            FROM statutes_articles
                            WHERE statute_id = ? AND article_no = ?
                        """, (statute_id, article_no))
                        row = cursor.fetchone()
                        if row and row[0]:
                            full_text = row[0]
            finally:
                # 연결 풀링 사용 시 close() 호출하지 않음
                if hasattr(self._connector, '_connection_pool') and self._connector._connection_pool:
                    # 연결 풀링 사용 중이면 close() 호출하지 않음
                    pass
                else:
                    # 직접 연결인 경우에만 close()
                    try:
                        conn.close()
                    except Exception:
                        pass
            
            return full_text if full_text and len(full_text.strip()) > 0 else None
            
        except Exception as e:
            self.logger.warning(f"Failed to get full text from database: {e}")
            return None

    def _get_full_texts_batch(
        self,
        source_details: List[Dict[str, Any]]
    ) -> Dict[str, str]:
        """
        여러 문서의 전체 본문을 배치로 조회
        
        Args:
            source_details: source_detail 딕셔너리 리스트
        
        Returns:
            {doc_id: full_text} 딕셔너리
        """
        try:
            # Config 및 connector 인스턴스 지연 초기화 (os 변수 오류 방지)
            if not self._config_initialized:
                try:
                    from core.utils.config import Config
                    from core.search.connectors.legal_data_connector_v2 import LegalDataConnectorV2
                    self._config = Config()
                    self._connector = LegalDataConnectorV2(self._config.database_path)
                    self._config_initialized = True
                except Exception as e:
                    self.logger.warning(f"Failed to initialize Config/Connector: {e}")
                    return {}
            
            if self._config is None or self._connector is None:
                self.logger.warning("Config or Connector not initialized, cannot get full texts")
                return {}
            
            conn = self._connector._get_connection()
            cursor = conn.cursor()
            
            full_texts = {}
            
            try:
                cases = []
                decisions = []
                interpretations = []
                statutes = []
                
                for detail in source_details:
                    source_type = detail.get("type")
                    doc_id = detail.get("case_number") or detail.get("decision_number") or detail.get("interpretation_number")
                    metadata = detail.get("metadata", {})
                    
                    if source_type == "case_paragraph" and doc_id:
                        cases.append((doc_id, detail))
                    elif source_type == "decision_paragraph" and doc_id:
                        decisions.append((doc_id, detail))
                    elif source_type == "interpretation_paragraph" and doc_id:
                        interpretations.append((doc_id, detail))
                    elif source_type == "statute_article":
                        statute_id = metadata.get("statute_id")
                        article_no = detail.get("article_no") or metadata.get("article_no")
                        if statute_id and article_no:
                            statutes.append((statute_id, article_no, detail))
                
                if cases:
                    doc_ids = [doc_id for doc_id, _ in cases]
                    if doc_ids:
                        placeholders = ','.join(['%s'] * len(doc_ids))  # 🔥 PostgreSQL: %s 사용
                        # precedent_contents에서 텍스트를 가져옴 (PostgreSQL)
                        # 🔥 레거시: case_paragraphs 테이블은 더 이상 사용하지 않음
                        cursor.execute(f"""
                            SELECT p.doc_id, STRING_AGG(pcc.section_content, E'\\n\\n' ORDER BY pcc.section_index) as full_text
                            FROM precedents p
                            JOIN precedent_contents pcc ON pcc.precedent_id = p.id
                            WHERE p.doc_id IN ({placeholders})
                            GROUP BY p.doc_id
                            ORDER BY p.doc_id
                        """, doc_ids)
                        for row in cursor.fetchall():
                            if row[0] and row[1]:
                                full_texts[row[0]] = row[1]
                
                if decisions:
                    doc_ids = [doc_id for doc_id, _ in decisions]
                    if doc_ids:
                        placeholders = ','.join(['?'] * len(doc_ids))
                        cursor.execute(f"""
                            SELECT d.doc_id, GROUP_CONCAT(dp.text, '\n\n') as full_text
                            FROM decision_paragraphs dp
                            JOIN decisions d ON dp.decision_id = d.id
                            WHERE d.doc_id IN ({placeholders})
                            GROUP BY d.doc_id
                            ORDER BY d.doc_id, dp.para_index
                        """, doc_ids)
                        for row in cursor.fetchall():
                            if row[0] and row[1]:
                                full_texts[row[0]] = row[1]
                
                if interpretations:
                    doc_ids = [doc_id for doc_id, _ in interpretations]
                    if doc_ids:
                        placeholders = ','.join(['?'] * len(doc_ids))
                        cursor.execute(f"""
                            SELECT i.doc_id, GROUP_CONCAT(ip.text, '\n\n') as full_text
                            FROM interpretation_paragraphs ip
                            JOIN interpretations i ON ip.interpretation_id = i.id
                            WHERE i.doc_id IN ({placeholders})
                            GROUP BY i.doc_id
                            ORDER BY i.doc_id, ip.para_index
                        """, doc_ids)
                        for row in cursor.fetchall():
                            if row[0] and row[1]:
                                full_texts[row[0]] = row[1]
                
                if statutes:
                    for statute_id, article_no, detail in statutes:
                        # 🔥 개선: statutes_articles 테이블만 사용 (statute_articles는 레거시, 삭제됨)
                        cursor.execute("""
                            SELECT article_content
                            FROM statutes_articles
                            WHERE statute_id = ? AND article_no = ?
                        """, (statute_id, article_no))
                        row = cursor.fetchone()
                        if row and row[0]:
                            key = f"{statute_id}_{article_no}"
                            full_texts[key] = row[0]
            finally:
                # 연결 풀링 사용 시 close() 호출하지 않음
                if hasattr(self._connector, '_connection_pool') and self._connector._connection_pool:
                    # 연결 풀링 사용 중이면 close() 호출하지 않음
                    pass
                else:
                    # 직접 연결인 경우에만 close()
                    try:
                        conn.close()
                    except Exception:
                        pass
            
            return full_texts
            
        except Exception as e:
            self.logger.warning(f"Failed to get full texts in batch: {e}")
            return {}

    def _create_source_detail_dict(
        self,
        source_str: str,
        source_type: Optional[str],
        source_info_detail: Any,
        doc: Dict[str, Any],
        metadata: Dict[str, Any],
        formatter_error: Optional[str]
    ) -> Optional[Dict[str, Any]]:
        """source_detail 딕셔너리 생성"""
        if source_info_detail:
            detail_dict = {
                "name": source_info_detail.name,
                "type": source_info_detail.type,
                "url": source_info_detail.url or "",
                "metadata": source_info_detail.metadata or {}
            }
            
            if source_info_detail.metadata:
                meta = source_info_detail.metadata
                if source_type == "statute_article":
                    if meta.get("statute_name"):
                        detail_dict["statute_name"] = meta["statute_name"]
                    if meta.get("article_no"):
                        detail_dict["article_no"] = meta["article_no"]
                elif source_type == "case_paragraph":
                    # doc_id 추출: 여러 위치에서 확인 (우선순위 순)
                    doc_metadata = doc.get("metadata", {}) if isinstance(doc.get("metadata"), dict) else {}
                    doc_id = (
                        meta.get("doc_id") or 
                        doc.get("doc_id") or 
                        doc_metadata.get("doc_id") or
                        doc_metadata.get("case_id") or
                        metadata.get("doc_id") or 
                        metadata.get("case_id") or
                        ""
                    )
                    detail_dict["case_number"] = doc_id
                    # doc_id가 없으면 로깅
                    if not doc_id:
                        self.logger.warning(f"[_create_source_detail_dict] case_paragraph에 doc_id가 없습니다. doc keys: {list(doc.keys())}, meta keys: {list(meta.keys()) if isinstance(meta, dict) else []}, metadata keys: {list(metadata.keys()) if isinstance(metadata, dict) else []}")
                    if meta.get("court"):
                        detail_dict["court"] = meta["court"]
                    # casenames를 case_name으로 변환
                    if meta.get("casenames"):
                        detail_dict["case_name"] = meta["casenames"]
                elif source_type == "decision_paragraph":
                    if meta.get("doc_id"):
                        detail_dict["decision_number"] = meta["doc_id"]
                    if meta.get("org"):
                        detail_dict["org"] = meta["org"]
                elif source_type == "interpretation_paragraph":
                    if meta.get("doc_id"):
                        detail_dict["interpretation_number"] = meta["doc_id"]
                    if meta.get("org"):
                        detail_dict["org"] = meta["org"]
            
            content = doc.get("content") or doc.get("text") or ""
            
            if not content or len(content.strip()) < 500:
                full_text = self._get_full_text_from_database(source_type, doc, source_info_detail.metadata or {})
                if full_text:
                    content = full_text
                    self.logger.debug(f"Retrieved full text from database for {source_type}: {len(content)} chars")
            
            if content:
                detail_dict["content"] = content
            
            return detail_dict
        else:
            detail_dict = {
                "name": source_str,
                "type": source_type or "unknown",
                "url": "",
                "metadata": metadata
            }
            
            if source_type == "statute_article":
                statute_name = doc.get("statute_name") or doc.get("law_name") or metadata.get("statute_name") or metadata.get("law_name")
                article_no = doc.get("article_no") or doc.get("article_number") or metadata.get("article_no") or metadata.get("article_number")
                if statute_name:
                    detail_dict["statute_name"] = statute_name
                if article_no:
                    detail_dict["article_no"] = article_no
            elif source_type == "case_paragraph":
                # doc_id 추출: 여러 위치에서 확인 (우선순위 순)
                doc_metadata = doc.get("metadata", {}) if isinstance(doc.get("metadata"), dict) else {}
                doc_id = (
                    doc.get("doc_id") or 
                    doc_metadata.get("doc_id") or
                    doc_metadata.get("case_id") or
                    metadata.get("doc_id") or 
                    metadata.get("case_id") or
                    ""
                )
                detail_dict["case_number"] = doc_id
                # doc_id가 없으면 로깅
                if not doc_id:
                    self.logger.warning(f"[_create_source_detail_dict] case_paragraph에 doc_id가 없습니다 (source_info_detail 없음). doc keys: {list(doc.keys())}, metadata keys: {list(metadata.keys()) if isinstance(metadata, dict) else []}")
                if doc.get("court") or metadata.get("court"):
                    detail_dict["court"] = doc.get("court") or metadata.get("court")
                # casenames를 case_name으로 변환
                casenames = doc.get("casenames") or metadata.get("casenames")
                if casenames:
                    detail_dict["case_name"] = casenames
            elif source_type == "decision_paragraph":
                doc_id = doc.get("doc_id") or metadata.get("doc_id") or metadata.get("decision_id")
                if doc_id:
                    detail_dict["decision_number"] = doc_id
                if doc.get("org") or metadata.get("org"):
                    detail_dict["org"] = doc.get("org") or metadata.get("org")
            elif source_type == "interpretation_paragraph":
                doc_id = doc.get("doc_id") or metadata.get("doc_id") or metadata.get("interpretation_id")
                if doc_id:
                    detail_dict["interpretation_number"] = doc_id
                if doc.get("org") or metadata.get("org"):
                    detail_dict["org"] = doc.get("org") or metadata.get("org")
            
            content = doc.get("content") or doc.get("text") or ""
            
            if not content or len(content.strip()) < 500:
                full_text = self._get_full_text_from_database(source_type, doc, metadata)
                if full_text:
                    content = full_text
                    self.logger.debug(f"Retrieved full text from database for {source_type}: {len(content)} chars")
            
            if content:
                detail_dict["content"] = content
            
            return detail_dict

    def _extract_legal_ref_from_source(
        self,
        source_str: str,
        doc: Dict[str, Any],
        metadata: Dict[str, Any]
    ) -> Optional[str]:
        """source_str에서 legal_reference 추출"""
        statute_pattern = r'([가-힣]+법)\s*(?:제\s*(\d+)\s*조)?'
        match = re.search(statute_pattern, source_str)
        if match:
            statute_name = match.group(1)
            article_no = match.group(2)
            if article_no:
                return f"{statute_name} 제{article_no}조"
            else:
                return statute_name
        
        statute_name = (
            doc.get("statute_name") or
            doc.get("law_name") or
            metadata.get("statute_name") or
            metadata.get("law_name")
        )
        article_no = (
            doc.get("article_no") or
            doc.get("article_number") or
            metadata.get("article_no") or
            metadata.get("article_number")
        )
        if statute_name:
            if article_no:
                return f"{statute_name} 제{article_no}조"
            else:
                return statute_name
        
        return None

    def _normalize_sources(self, sources_list: List[str]) -> List[str]:
        """sources 정규화"""
        normalized_sources = []
        for source in sources_list[:MAX_SOURCES_LIMIT]:
            try:
                if isinstance(source, dict):
                    source_str = (source.get("source") or 
                                 source.get("name") or 
                                 source.get("title") or 
                                 str(source.get("type", "Unknown")))
                    if source_str and isinstance(source_str, str) and source_str.strip():
                        normalized_sources.append(source_str.strip())
                elif isinstance(source, str):
                    if source.strip():
                        normalized_sources.append(source.strip())
                else:
                    source_str = str(source)
                    if source_str.strip():
                        normalized_sources.append(source_str.strip())
            except Exception as e:
                self.logger.warning(f"[SOURCES] Error normalizing source: {e}")
                continue
        
        normalized_sources = [s for s in normalized_sources if s and len(s.strip()) > 0]
        seen_normalized = set()
        normalized_sources_unique = []
        for s in normalized_sources:
            if s not in seen_normalized:
                normalized_sources_unique.append(s)
                seen_normalized.add(s)
        
        return normalized_sources_unique

    def _extract_and_store_related_questions(self, state: LegalWorkflowState) -> None:
        """related_questions 추출 및 저장"""
        related_questions = []
        
        metadata = state.get("metadata", {})
        if isinstance(metadata, dict) and "related_questions" in metadata:
            related_questions = metadata.get("related_questions", [])
            if isinstance(related_questions, list) and len(related_questions) > 0:
                self.logger.info(f"[RELATED_QUESTIONS] Found {len(related_questions)} related_questions in metadata")
        
        if not related_questions:
            if "common" in state and isinstance(state.get("common"), dict):
                common_metadata = state["common"].get("metadata", {})
                if isinstance(common_metadata, dict) and "related_questions" in common_metadata:
                    related_questions = common_metadata.get("related_questions", [])
                    if isinstance(related_questions, list) and len(related_questions) > 0:
                        self.logger.info(f"[RELATED_QUESTIONS] Found {len(related_questions)} related_questions in common.metadata")
        
        if not related_questions and "related_questions" in state:
            related_questions = state.get("related_questions", [])
            if isinstance(related_questions, list) and len(related_questions) > 0:
                self.logger.info(f"[RELATED_QUESTIONS] Found {len(related_questions)} related_questions in top-level state")
        else:
            phase_info = state.get("phase_info", {})
            if isinstance(phase_info, dict) and "phase2" in phase_info:
                phase2 = phase_info.get("phase2", {})
                if isinstance(phase2, dict) and "flow_tracking_info" in phase2:
                    flow_tracking = phase2.get("flow_tracking_info", {})
                    if isinstance(flow_tracking, dict) and "suggested_questions" in flow_tracking:
                        suggested_questions = flow_tracking.get("suggested_questions", [])
                        if isinstance(suggested_questions, list) and len(suggested_questions) > 0:
                            if isinstance(suggested_questions[0], dict):
                                related_questions = [q.get("question", "") for q in suggested_questions if q.get("question")]
                            else:
                                related_questions = [str(q) for q in suggested_questions if q]
                            self.logger.info(f"[RELATED_QUESTIONS] Extracted {len(related_questions)} related_questions from phase_info")
        
        if not related_questions:
            try:
                query = state.get("query", "")
                answer = state.get("answer", "")
                if query:
                    related_questions = self._generate_related_questions(query, answer or "")
                    if related_questions:
                        self.logger.info(f"[RELATED_QUESTIONS] Generated {len(related_questions)} related_questions using template: {related_questions[:3]}")
            except Exception as e:
                self.logger.warning(f"[RELATED_QUESTIONS] Failed to generate related_questions: {e}", exc_info=True)
        
        if related_questions:
            if not isinstance(metadata, dict):
                metadata = {}
            metadata["related_questions"] = related_questions
            state["metadata"] = metadata
            # 성능 최적화: 여러 위치에 저장하여 손실 방지
            state["related_questions"] = related_questions
            if "common" not in state:
                state["common"] = {}
            if "metadata" not in state["common"]:
                state["common"]["metadata"] = {}
            state["common"]["metadata"]["related_questions"] = related_questions
            self.logger.info(f"[RELATED_QUESTIONS] Saved {len(related_questions)} related_questions to multiple locations: {related_questions[:3]}")
        else:
            self.logger.warning("[RELATED_QUESTIONS] No related_questions found after all attempts")

    def _restore_retrieved_docs_enhanced(self, state: LegalWorkflowState) -> List[Dict[str, Any]]:
        """
        retrieved_docs를 여러 위치에서 복구 (공통 로직)
        
        Returns:
            복구된 retrieved_docs 리스트
        """
        retrieved_docs_list = state.get("retrieved_docs", [])
        restore_locations = []
        
        if not retrieved_docs_list:
            # 1. search 그룹에서 확인
            if "search" in state and isinstance(state["search"], dict):
                retrieved_docs_list = state["search"].get("retrieved_docs", [])
                if retrieved_docs_list:
                    restore_locations.append("search.retrieved_docs")
        
        if not retrieved_docs_list:
            # 2. common.search 그룹에서 확인
            if "common" in state and isinstance(state["common"], dict):
                if "search" in state["common"] and isinstance(state["common"]["search"], dict):
                    retrieved_docs_list = state["common"]["search"].get("retrieved_docs", [])
                    if retrieved_docs_list:
                        restore_locations.append("common.search.retrieved_docs")
        
        if not retrieved_docs_list:
            # 3. merged_documents에서 확인
            merged_docs = state.get("merged_documents", [])
            if merged_docs:
                retrieved_docs_list = merged_docs
                restore_locations.append("merged_documents")
        
        if not retrieved_docs_list:
            # 4. search.merged_documents에서 확인
            if "search" in state and isinstance(state["search"], dict):
                merged_docs = state["search"].get("merged_documents", [])
                if merged_docs:
                    retrieved_docs_list = merged_docs
                    restore_locations.append("search.merged_documents")
        
        if not retrieved_docs_list:
            # 5. state_helpers의 get_retrieved_docs 사용
            try:
                from core.agents.state_helpers import get_retrieved_docs
                retrieved_docs_list = get_retrieved_docs(state)
                if retrieved_docs_list:
                    restore_locations.append("state_helpers.get_retrieved_docs")
            except (ImportError, AttributeError) as e:
                self.logger.debug(f"[SOURCES] Could not use state_helpers.get_retrieved_docs: {e}")
        
        if not retrieved_docs_list:
            # 6. global cache에서 확인 (여러 경로)
            try:
                from core.agents.node_wrappers import _global_search_results_cache
                if _global_search_results_cache:
                    # 여러 경로에서 시도
                    cached_docs = (
                        _global_search_results_cache.get("retrieved_docs", []) or
                        _global_search_results_cache.get("search", {}).get("retrieved_docs", []) if isinstance(_global_search_results_cache.get("search"), dict) else [] or
                        _global_search_results_cache.get("common", {}).get("search", {}).get("retrieved_docs", []) if isinstance(_global_search_results_cache.get("common"), dict) and isinstance(_global_search_results_cache["common"].get("search"), dict) else []
                    )
                    if cached_docs:
                        retrieved_docs_list = cached_docs
                        restore_locations.append("global_cache")
            except (ImportError, AttributeError) as e:
                self.logger.debug(f"[SOURCES] Could not access global cache: {e}")
        
        # 복구된 retrieved_docs를 state에 저장 (표준화)
        if retrieved_docs_list:
            # 표준화: 항상 state["retrieved_docs"]에 저장
            state["retrieved_docs"] = retrieved_docs_list
            # 다른 위치에도 저장하여 일관성 보장
            if "search" not in state:
                state["search"] = {}
            state["search"]["retrieved_docs"] = retrieved_docs_list
            if "common" not in state:
                state["common"] = {}
            if "search" not in state["common"]:
                state["common"]["search"] = {}
            state["common"]["search"]["retrieved_docs"] = retrieved_docs_list
            
            self.logger.info(f"[SOURCES] ✅ Restored {len(retrieved_docs_list)} retrieved_docs from: {', '.join(restore_locations) if restore_locations else 'top-level'}")
        else:
            self.logger.warning(f"[SOURCES] ⚠️ No retrieved_docs found (state keys: {list(state.keys())[:10]})")
        
        return retrieved_docs_list
    
    def _restore_query_type_enhanced(self, state: LegalWorkflowState) -> str:
        """
        query_type을 여러 위치에서 복구 (공통 로직)
        
        Returns:
            복구된 query_type 문자열
        """
        query_type = state.get("query_type", "")
        if not query_type:
            # classification 그룹에서 확인
            if "classification" in state and isinstance(state["classification"], dict):
                query_type = state["classification"].get("query_type", "")
            # common.classification 그룹에서 확인
            if not query_type and "common" in state and isinstance(state["common"], dict):
                if "classification" in state["common"] and isinstance(state["common"]["classification"], dict):
                    query_type = state["common"]["classification"].get("query_type", "")
            # metadata에서 확인
            if not query_type:
                metadata = state.get("metadata", {})
                if isinstance(metadata, dict):
                    query_type = metadata.get("query_type", "")
            # global cache에서 확인
            if not query_type:
                try:
                    from core.agents.node_wrappers import _global_search_results_cache
                    if _global_search_results_cache:
                        query_type = (
                            _global_search_results_cache.get("common", {}).get("classification", {}).get("query_type", "") or
                            _global_search_results_cache.get("metadata", {}).get("query_type", "") or
                            _global_search_results_cache.get("classification", {}).get("query_type", "") or
                            _global_search_results_cache.get("query_type", "") or
                            ""
                        )
                except (ImportError, AttributeError):
                    pass
            # 기본값 설정
            if not query_type:
                query_type = "general_question"
                self.logger.warning(f"[QUERY_TYPE] ⚠️ query_type not found, using default: {query_type}")
            else:
                self.logger.info(f"[QUERY_TYPE] ✅ Restored query_type: {query_type}")
                # state에 저장
                state["query_type"] = query_type
        
        return query_type
    
    def _generate_related_questions(self, query: str, answer: str) -> List[str]:
        """관련 질문 생성 (템플릿 기반 - 개선: 더 다양하고 관련성 높은 질문 생성)"""
        related_questions = []
        
        # 질문에서 핵심 키워드 추출
        query_lower = query.lower()
        
        # 답변에서 핵심 키워드 추출 (개선)
        answer_keywords = []
        if answer:
            # 법령 조문 추출
            law_pattern = r'([가-힣]+법)\s*제?\s*(\d+)\s*조'
            law_matches = re.findall(law_pattern, answer)
            for law_name, article_no in law_matches[:3]:
                answer_keywords.append(f"{law_name} 제{article_no}조")
        
        # 법령 관련 질문 패턴
        if any(keyword in query_lower for keyword in ["법령", "법률", "조문", "조", "항"]):
            if answer_keywords:
                for law_ref in answer_keywords[:2]:
                    related_questions.append(f"{law_ref}의 구체적인 내용은 무엇인가요?")
                    related_questions.append(f"{law_ref}와 관련된 판례는 무엇이 있나요?")
            else:
                related_questions.append(f"{query}에 대한 다른 법령도 확인해볼까요?")
                related_questions.append(f"{query}와 관련된 판례도 찾아볼까요?")
            related_questions.append(f"{query}의 적용 요건은 무엇인가요?")
            related_questions.append(f"{query}와 관련된 실무 사례는 무엇이 있나요?")
        
        # 판례 관련 질문 패턴
        elif any(keyword in query_lower for keyword in ["판례", "판결", "사건", "대법원"]):
            related_questions.append(f"{query}와 유사한 사건의 판례도 찾아볼까요?")
            related_questions.append(f"{query}에 대한 법령 조문도 확인해볼까요?")
            related_questions.append(f"{query}의 판결 요지는 무엇인가요?")
            related_questions.append(f"{query}와 관련된 다른 판례는 무엇이 있나요?")
        
        # 손해배상 관련 질문 패턴
        elif any(keyword in query_lower for keyword in ["손해배상", "배상", "손해", "청구"]):
            related_questions.append("손해배상 청구의 절차는 어떻게 되나요?")
            related_questions.append("손해배상의 범위는 어떻게 결정되나요?")
            related_questions.append("손해배상과 관련된 판례도 찾아볼까요?")
            related_questions.append("손해배상 청구 시 필요한 증거는 무엇인가요?")
            related_questions.append("손해배상의 소멸시효는 어떻게 되나요?")
        
        # 전세금 관련 질문 패턴
        elif any(keyword in query_lower for keyword in ["전세금", "전세", "보증금", "반환"]):
            related_questions.append("전세금 반환 보증의 가입 조건은 어떻게 되나요?")
            related_questions.append("전세금 반환 보증을 통해 보증받을 수 있는 최대 금액은 얼마인가요?")
            related_questions.append("전세 계약 만료 시 전세금을 돌려받지 못했을 때, 전세금 반환 보증을 통해 어떻게 보상받을 수 있나요?")
            related_questions.append("전세금 반환 보증 가입 시 필요한 서류는 무엇인가요?")
            related_questions.append("전세금 반환 보증의 종류에는 어떤 것들이 있나요?")
        
        # 임대차 관련 질문 패턴
        elif any(keyword in query_lower for keyword in ["임대차", "임대", "계약", "해지"]):
            related_questions.append("임대차 계약 해지 시 주의사항은 무엇인가요?")
            related_questions.append("임대차 계약 해지 시 보증금 반환은 어떻게 되나요?")
            related_questions.append("임대차 계약 해지와 관련된 판례는 무엇이 있나요?")
            related_questions.append("임대차 계약 해지 시 손해배상은 어떻게 되나요?")
            related_questions.append("임대차 계약 해지 시 필요한 절차는 무엇인가요?")
        
        # 일반적인 관련 질문 (개선: 더 구체적인 질문)
        if len(related_questions) < 3:
            # query에서 핵심 키워드 추출
            core_keywords = []
            for keyword in ["법령", "조문", "판례", "계약", "손해", "보증", "반환", "해지", "청구"]:
                if keyword in query:
                    core_keywords.append(keyword)
            
            if core_keywords:
                keyword_str = core_keywords[0]
                related_questions.append(f"{keyword_str}와 관련된 다른 질문이 있으신가요?")
                related_questions.append(f"{keyword_str}에 대한 더 자세한 정보가 필요하신가요?")
                related_questions.append(f"{keyword_str}의 실무 적용 사례는 무엇이 있나요?")
            else:
                related_questions.append(f"{query}에 대한 더 자세한 정보가 필요하신가요?")
                related_questions.append(f"{query}와 관련된 다른 질문이 있으신가요?")
                related_questions.append(f"{query}의 실무 적용 사례는 무엇이 있나요?")
        
        return related_questions[:MAX_RELATED_QUESTIONS_LIMIT]

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
            self.logger.info("[FORMAT_AND_PREPARE_FINAL] Calling prepare_final_response_part")
            sources_before = len(state.get("sources", []))
            self.prepare_final_response_part(state, query_complexity, needs_search)
            sources_after = len(state.get("sources", []))
            self.logger.info(f"[FORMAT_AND_PREPARE_FINAL] prepare_final_response_part completed: sources {sources_before} -> {sources_after}, legal_references={len(state.get('legal_references', []))}")

            # Part 3: 최종 후처리 (성능 최적화: 필수 작업만 수행)
            final_answer = state.get("answer", "")
            if final_answer and len(final_answer) > 100:  # 성능 최적화: 짧은 답변은 후처리 스킵
                import re

                # 중복 헤더 제거 (리팩토링된 메서드 사용)
                final_answer = self.answer_cleaner.remove_duplicate_headers(final_answer)

                # 연속된 빈 줄 정리 (3개 이상 -> 2개) - 성능 최적화: 한 번만 수행
                if '\n\n\n' in final_answer:
                    final_answer = re.sub(r'\n{3,}', '\n\n', final_answer)

                # 공백 없는 텍스트 수정 (성능 최적화: 패턴이 있을 때만 수행)
                # 한글 + 영문/숫자 사이에 공백 추가 (패턴이 있을 때만)
                if re.search(r'[가-힣][A-Za-z0-9]', final_answer) or re.search(r'[A-Za-z0-9][가-힣]', final_answer):
                    final_answer = re.sub(r'([가-힣])([A-Za-z0-9])', r'\1 \2', final_answer)
                    final_answer = re.sub(r'([A-Za-z0-9])([가-힣])', r'\1 \2', final_answer)
                # 특정 패턴 수정 (법령명 + 조항) - 패턴이 있을 때만
                if re.search(r'[가-힣]+법[가-힣]', final_answer):
                    final_answer = re.sub(r'([가-힣]+법)([가-힣])', r'\1 \2', final_answer)
                # "의", "및", "와", "과" 앞뒤 공백 보장 - 패턴이 있을 때만
                if re.search(r'[가-힣](의|및|와|과|에서|으로|에게)[가-힣]', final_answer):
                    final_answer = re.sub(r'([가-힣])(의|및|와|과|에서|으로|에게)([가-힣])', r'\1 \2 \3', final_answer)

                # 답변 내부의 하드코딩된 신뢰도 값 교체 (리팩토링된 메서드 사용)
                current_confidence = state.get("confidence", 0.0)
                if current_confidence > 0:
                    final_answer = self.confidence_manager.replace_in_text(final_answer, current_confidence)

                # "참고자료" 섹션이 "관련 정보를 찾을 수 없습니다"로 표시된 경우 소스 정보로 교체
                sources_list = state.get("sources", [])
                if sources_list and len(sources_list) > 0:
                    # 실제 소스 이름만 추출 (검색 타입 제외)
                    valid_sources = [s for s in sources_list if isinstance(s, str) and
                                   s.lower() not in ["semantic", "keyword", "unknown", "fts", "vector", ""]]

                    if valid_sources:
                        sources_text = "\n".join([f"- {source}" for source in valid_sources[:MAX_SOURCES_DISPLAY_LIMIT]])
                        # "참고자료" 섹션 교체
                        final_answer = re.sub(
                            r'###\s*📚\s*참고자료.*?관련 정보를 찾을 수 없습니다\.',
                            f'### 📚 참고자료\n\n{sources_text}',
                            final_answer,
                            flags=re.DOTALL | re.IGNORECASE
                        )
                        # "참고자료" 섹션이 비어있는 경우도 처리
                        final_answer = re.sub(
                            r'###\s*📚\s*참고자료\s*\n+\s*관련 정보를 찾을 수 없습니다\.',
                            f'### 📚 참고자료\n\n{sources_text}',
                            final_answer,
                            flags=re.IGNORECASE
                        )
                    else:
                        # 유효한 소스가 없으면 "참고자료" 섹션 제거
                        final_answer = re.sub(
                            r'###\s*📚\s*참고자료.*?(?=\n###|\n---|\Z)',
                            '',
                            final_answer,
                            flags=re.DOTALL | re.IGNORECASE
                        )
                else:
                    # 소스가 없으면 "참고자료" 섹션 제거
                    final_answer = re.sub(
                        r'###\s*📚\s*참고자료.*?(?=\n###|\n---|\Z)',
                        '',
                        final_answer,
                        flags=re.DOTALL | re.IGNORECASE
                    )

                # 최종 후처리: 중복 헤더가 여전히 있는지 확인하고 제거 (추가 안전장치)
                # "###" 로 시작하고 "답변"이 포함된 줄이 있는지 확인
                if '###' in final_answer and '답변' in final_answer:
                    lines_final = final_answer.split('\n')
                    final_cleaned = []
                    for line in lines_final:
                        # "###" 로 시작하고 "답변"이 포함된 줄은 제거
                        if re.match(r'^###\s*.*답변', line, re.IGNORECASE):
                            continue
                        final_cleaned.append(line)
                    final_answer = '\n'.join(final_cleaned)

                # 신뢰도 값 최종 교체 (리팩토링된 메서드 사용)
                current_confidence = state.get("confidence", 0.0)
                if current_confidence > 0:
                    final_answer = self.confidence_manager.replace_in_text(final_answer, current_confidence)

                # 메타 정보 섹션 추출 및 분리 (신뢰도 섹션 교체 후)
                metadata_sections = self._extract_metadata_sections(final_answer)

                # answer에서 메타 정보 섹션 제거
                before_metadata = len(final_answer)
                clean_answer = self._remove_metadata_sections(final_answer)
                after_metadata = len(clean_answer)
                self.logger.debug(f"[ANSWER CLEANUP] After metadata removal: {before_metadata} -> {after_metadata} chars")

                # 중간 생성 텍스트 제거 (STEP 0, 원본 답변, 질문 정보 등)
                before_intermediate = len(clean_answer)
                clean_answer = self._remove_intermediate_text(clean_answer)
                after_intermediate = len(clean_answer)
                self.logger.debug(f"[ANSWER CLEANUP] After intermediate removal: {before_intermediate} -> {after_intermediate} chars")

                # '## 답변' 헤더 제거
                before_header = len(clean_answer)
                clean_answer = self._remove_answer_header(clean_answer)
                after_header = len(clean_answer)
                self.logger.debug(f"[ANSWER CLEANUP] After header removal: {before_header} -> {after_header} chars")

                # 답변 길이 조절 (질의 유형에 맞게)
                # 개선: grounding_score와 quality_score를 전달하여 품질 기반 동적 조정
                query_type = WorkflowUtils.get_state_value(state, "query_type", "general_question")
                query_complexity = WorkflowUtils.get_state_value(state, "complexity_level", "moderate")
                # grounding_score는 나중에 계산되므로, 이전 검증 결과가 있으면 사용
                grounding_score = state.get("grounding_score")
                if grounding_score is None:
                    # 이전 검증 결과가 없으면 None으로 전달 (품질 기반 조정 없음)
                    grounding_score = None
                # quality_score는 현재 사용하지 않지만, 향후 확장 가능
                quality_score = None
                clean_answer = self._adjust_answer_length(clean_answer, query_type, query_complexity, grounding_score, quality_score)

                # 디버깅 로그
                self.logger.info(f"[ANSWER CLEANUP] Original length: {len(final_answer)}, Clean length: {len(clean_answer)}")
                self.logger.info(f"[ANSWER CLEANUP] Has confidence_info: {bool(metadata_sections.get('confidence_info'))}")
                self.logger.info(f"[ANSWER CLEANUP] Has reference_materials: {bool(metadata_sections.get('reference_materials'))}")
                self.logger.info(f"[ANSWER CLEANUP] Has disclaimer: {bool(metadata_sections.get('disclaimer'))}")

                # state에 정리된 answer 저장
                state["answer"] = clean_answer.strip()

                # 메타 정보를 별도 필드로 저장
                state["confidence_info"] = metadata_sections.get("confidence_info", "")
                state["reference_materials"] = metadata_sections.get("reference_materials", "")
                state["disclaimer"] = metadata_sections.get("disclaimer", "")

                # 최종 답변 검증 (개선: 테스트 및 검증 로직 추가)
                # 검색 결과 기반 검증 추가 (Hallucination 방지)
                # 개선: retrieved_docs를 여러 위치에서 검색
                retrieved_docs = state.get("retrieved_docs", [])
                if not retrieved_docs:
                    # search 그룹에서 확인
                    if "search" in state and isinstance(state["search"], dict):
                        retrieved_docs = state["search"].get("retrieved_docs", [])
                if not retrieved_docs:
                    # common.search 그룹에서 확인
                    if "common" in state and isinstance(state["common"], dict):
                        if "search" in state["common"] and isinstance(state["common"]["search"], dict):
                            retrieved_docs = state["common"]["search"].get("retrieved_docs", [])
                if not retrieved_docs:
                    # state_helpers의 get_retrieved_docs 사용
                    try:
                        from core.agents.state_helpers import get_retrieved_docs
                        retrieved_docs = get_retrieved_docs(state)
                    except (ImportError, AttributeError):
                        pass
                if not retrieved_docs:
                    # global cache에서 확인
                    try:
                        from core.agents.node_wrappers import _global_search_results_cache
                        if _global_search_results_cache:
                            retrieved_docs = _global_search_results_cache.get("retrieved_docs", [])
                    except (ImportError, AttributeError):
                        pass
                
                self.logger.info(f"[GROUNDING VERIFICATION] Retrieved docs count: {len(retrieved_docs) if retrieved_docs else 0}")
                
                query = WorkflowUtils.get_state_value(state, "query", "")
                
                # needs_search 확인 (direct_answer 노드의 경우 검색이 없음)
                needs_search = WorkflowUtils.get_state_value(state, "needs_search", True)

                # 검색 결과 기반 검증 수행
                # direct_answer 노드의 경우 (needs_search=False) 검색이 없으므로 grounding_score 계산 건너뛰기
                if not needs_search and not retrieved_docs:
                    # direct_answer 노드: 검색 없이 직접 답변 생성
                    # grounding_score는 None으로 설정하여 신뢰도 계산 시 패널티 방지
                    self.logger.info(
                        "답변 검증 결과: grounding_score=N/A (direct_answer, no search), "
                        "unverified_count=0"
                    )
                    state["grounding_score"] = None  # None으로 설정하여 패널티 방지
                    state["source_coverage"] = None  # None으로 설정하여 패널티 방지
                else:
                    # 검색 결과가 있는 경우 정상적인 검증 수행
                    # 개선: 원본 답변(final_answer)로 검증하여 잘린 답변으로 인한 grounding_score 저하 방지
                    verification_answer = final_answer if len(final_answer) > len(clean_answer) else clean_answer
                    self.logger.debug(f"[GROUNDING VERIFICATION] Using {'original' if verification_answer == final_answer else 'cleaned'} answer for verification (length: {len(verification_answer)})")
                    
                    source_verification_result = AnswerValidator.validate_answer_source_verification(
                        answer=verification_answer,
                        retrieved_docs=retrieved_docs,
                        query=query
                    )

                    # 검증 결과에 따라 신뢰도 조정
                    if source_verification_result.get("needs_review", False):
                        self.logger.warning(
                            f"답변 검증 결과: grounding_score={source_verification_result.get('grounding_score', 0):.2f}, "
                            f"unverified_count={source_verification_result.get('unverified_count', 0)}"
                        )

                        # 신뢰도 조정 적용
                        current_confidence = state.get("confidence", 0.8)
                        penalty = source_verification_result.get("confidence_penalty", 0.0)
                        adjusted_confidence = max(0.0, current_confidence - penalty)
                        state["confidence"] = adjusted_confidence

                        # 검증되지 않은 섹션을 로그에 기록
                        unverified = source_verification_result.get("unverified_sentences", [])
                        if unverified:
                            self.logger.warning(
                                f"검증되지 않은 문장 {len(unverified)}개 발견. "
                                f"샘플: {unverified[0].get('sentence', '')[:50]}..."
                            )
                    else:
                        self.logger.info(
                            f"답변 검증 통과: grounding_score={source_verification_result.get('grounding_score', 0):.2f}"
                        )

                    # 검증 결과를 state에 저장 (신뢰도 계산에 사용)
                    state["grounding_score"] = source_verification_result.get("grounding_score", 0.0)
                    state["source_coverage"] = source_verification_result.get("source_coverage", 0.0)

                    # 개선: grounding_score 계산 후 답변 길이 재조정 (품질이 높으면 더 긴 답변 허용)
                    # 개선: _adjust_answer_length 함수를 재사용하여 섹션 기반 스마트 트렁케이션 적용
                    calculated_grounding_score = state.get("grounding_score")
                    if calculated_grounding_score is not None and calculated_grounding_score >= 0.5:
                        # grounding_score가 높으면 답변 길이를 재조정하여 더 긴 답변 허용
                        original_clean_length = len(clean_answer)
                        # 원본 답변(final_answer)이 더 길면, grounding_score에 따라 더 긴 답변 허용
                        if len(final_answer) > len(clean_answer):
                            # 개선: _adjust_answer_length 함수를 재사용하여 섹션 기반 스마트 트렁케이션 적용
                            # grounding_score에 따라 더 긴 길이로 재조정
                            if calculated_grounding_score >= 0.7:
                                # 높은 grounding_score: 원본 답변의 최대 150%까지 허용
                                # _adjust_answer_length 함수를 재사용하여 섹션 기반 스마트 트렁케이션 적용
                                re_adjusted_answer = self._adjust_answer_length(
                                    final_answer,
                                    query_type,
                                    query_complexity,
                                    calculated_grounding_score,
                                    None
                                )
                                # 재조정된 답변이 원본보다 길면 사용
                                if len(re_adjusted_answer) > len(clean_answer):
                                    clean_answer = re_adjusted_answer
                                    self.logger.info(f"[ANSWER LENGTH] Re-adjusted after grounding_score calculation: {original_clean_length} -> {len(clean_answer)} chars (grounding_score: {calculated_grounding_score:.2f}, using smart truncation)")
                                else:
                                    # 재조정이 효과가 없으면 원본 답변 사용
                                    clean_answer = final_answer
                                    self.logger.info(f"[ANSWER LENGTH] Re-adjusted after grounding_score calculation: {original_clean_length} -> {len(clean_answer)} chars (grounding_score: {calculated_grounding_score:.2f})")
                            elif calculated_grounding_score >= 0.5:
                                # 중간 grounding_score: 원본 답변의 최대 120%까지 허용
                                # _adjust_answer_length 함수를 재사용하여 섹션 기반 스마트 트렁케이션 적용
                                re_adjusted_answer = self._adjust_answer_length(
                                    final_answer,
                                    query_type,
                                    query_complexity,
                                    calculated_grounding_score,
                                    None
                                )
                                # 재조정된 답변이 원본보다 길면 사용
                                if len(re_adjusted_answer) > len(clean_answer):
                                    clean_answer = re_adjusted_answer
                                    self.logger.debug(f"[ANSWER LENGTH] Re-adjusted after grounding_score calculation: {original_clean_length} -> {len(clean_answer)} chars (grounding_score: {calculated_grounding_score:.2f}, using smart truncation)")
                                else:
                                    # 재조정이 효과가 없으면 원본 답변 사용
                                    clean_answer = final_answer
                                    self.logger.debug(f"[ANSWER LENGTH] Re-adjusted after grounding_score calculation: {original_clean_length} -> {len(clean_answer)} chars (grounding_score: {calculated_grounding_score:.2f})")
                        # 재조정된 답변을 state에 저장
                        state["answer"] = clean_answer.strip()

                # 기존 답변 검증 수행
                validation_result = self._validate_final_answer(clean_answer, retrieved_docs, query)
                if validation_result.get("has_issues"):
                    self.logger.warning(f"Answer validation issues: {validation_result.get('issues', [])}")
                    # 검증 실패해도 답변은 유지 (로그만 기록)

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

    def _validate_final_answer(
        self,
        answer: str,
        retrieved_docs: Optional[List[Dict[str, Any]]] = None,
        query: Optional[str] = None
    ) -> Dict[str, Any]:
        """최종 답변 검증 (강화된 버전)"""
        try:
            issues = []
            warnings = []
            quality_metrics = {}

            if not answer or len(answer.strip()) == 0:
                issues.append("답변이 비어있습니다")
                return {"has_issues": True, "issues": issues, "warnings": warnings, "quality_metrics": quality_metrics}

            # 중복 헤더 확인 (개선)
            headers = re.findall(r'^#{1,3}\s+(.+)$', answer, re.MULTILINE)
            header_counts = {}
            duplicate_headers = []
            for header in headers:
                clean_header = re.sub(r'[📖⚖️💼💡📚📋⭐📌🔍💬🎯📊📝📄⏰🔗⚠️❗✅🚨🎉💯🔔]+\s*', '', header).strip().lower()
                normalized = re.sub(r'\s+', ' ', clean_header)
                header_counts[normalized] = header_counts.get(normalized, 0) + 1
                if header_counts[normalized] > 1:
                    duplicate_headers.append(normalized)

            # 중복 헤더 제거 시도 (자동 수정)
            if duplicate_headers:
                warnings.append(f"중복 헤더 발견: {', '.join(set(duplicate_headers))}")

            # 빈 섹션 확인 (개선)
            sections = re.findall(r'^###\s+(.+)$\n(.*?)(?=^###|$)', answer, re.MULTILINE | re.DOTALL)
            empty_sections = []
            for section_title, section_content in sections:
                clean_content = section_content.strip()
                if not clean_content or clean_content == "관련 정보를 찾을 수 없습니다." or len(clean_content) < 10:
                    empty_sections.append(section_title.strip())

            if empty_sections:
                warnings.append(f"빈 섹션: {', '.join(empty_sections)}")

            # 답변 길이 확인 (개선)
            answer_length = len(answer)
            quality_metrics["answer_length"] = answer_length
            if answer_length < 50:
                issues.append("답변이 너무 짧습니다 (50자 미만)")
            elif answer_length > 5000:
                warnings.append("답변이 너무 깁니다 (5000자 초과)")
            elif answer_length < 100:
                warnings.append("답변이 다소 짧습니다 (100자 미만)")

            # 법적 인용 확인 (개선)
            legal_citations = len(re.findall(r'\[법령:|\[판례:|제\d+조', answer))
            quality_metrics["legal_citations"] = legal_citations
            if legal_citations == 0:
                warnings.append("법적 인용이 없습니다")

            # 구조 품질 점수 (개선)
            structure_score = 0.0
            if len(headers) > 0:
                structure_score += 0.3  # 헤더가 있으면 구조화됨
            if len(re.findall(r'\n\n', answer)) > 2:
                structure_score += 0.2  # 단락 구분이 있으면 가독성 향상
            if re.search(r'\d+\.|-\s+', answer):
                structure_score += 0.2  # 목록이 있으면 구조화됨
            if len(answer) > 200:
                structure_score += 0.3  # 충분한 내용이 있음

            quality_metrics["structure_score"] = structure_score

            # 가독성 점수 계산 (개선)
            readability_score = structure_score * 100
            quality_metrics["readability_score"] = readability_score

            # 전체 품질 점수 계산
            quality_score = structure_score
            if legal_citations > 0:
                quality_score += 0.2  # 법적 인용 보너스
            if answer_length >= 200 and answer_length <= 2000:
                quality_score += 0.1  # 적절한 길이 보너스

            quality_metrics["quality_score"] = min(1.0, quality_score)

            return {
                "has_issues": len(issues) > 0,
                "issues": issues,
                "warnings": warnings,
                "readability_score": readability_score,
                "header_count": len(headers),
                "quality_metrics": quality_metrics,
                "answer_length": len(answer)
            }

        except Exception as e:
            self.logger.error(f"Error validating final answer: {e}")
            return {"has_issues": False, "issues": [], "warnings": [f"검증 오류: {str(e)}"]}

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
        from core.generation.validators.confidence_calculator import ConfidenceLevel

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
