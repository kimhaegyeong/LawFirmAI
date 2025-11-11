# -*- coding: utf-8 -*-
"""
답변 포맷팅 모듈
LangGraph 워크플로우의 답변 포맷팅 및 최종 응답 준비 로직을 독립 모듈로 분리
"""

import logging
import re
import time
from datetime import datetime
from typing import Any, Dict, List, Optional, Union

from core.agents.state_definitions import LegalWorkflowState
from core.agents.workflow_utils import WorkflowUtils
from core.agents.validators.quality_validators import AnswerValidator

from .config.formatter_config import AnswerLengthConfig, ConfidenceConfig
from .managers.confidence_manager import ConfidenceManager
from .extractors.source_extractor import SourceExtractor
from .cleaners.answer_cleaner import AnswerCleaner
from .formatters.length_adjuster import AnswerLengthAdjuster

# Constants for processing steps
MAX_PROCESSING_STEPS = 50

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
            final_sources_list = []
            final_sources_detail = []
            seen_sources = set()

            # 통일된 포맷터 및 검증기 초기화
            try:
                from ...services.unified_source_formatter import UnifiedSourceFormatter
                from ...services.source_validator import SourceValidator
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

            state["sources"] = final_sources_list[:10]  # 최대 10개만 (하위 호환성)
            state["sources_detail"] = final_sources_detail[:10]  # 최대 10개만 (신규 필드)

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
            try:
                src = state.get("sources", [])
                norm = []
                seen = set()
                if isinstance(src, list):
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

                # 최대 10개로 제한하고 정렬 (긴 이름 우선)
                state["sources"] = sorted(norm[:10], key=len, reverse=True)
            except Exception as e:
                self.logger.warning(f"Error formatting sources: {e}")
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
        self.logger.warning("[PREPARE_FINAL_RESPONSE_PART] Starting prepare_final_response_part")
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

        # retrieved_docs 복구 (여러 위치에서 검색)
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
        
        # 복구된 retrieved_docs를 state에 저장
        if retrieved_docs:
            state["retrieved_docs"] = retrieved_docs
            self.logger.info(f"[SOURCES] Restored {len(retrieved_docs)} retrieved_docs in prepare_final_response_part")
        else:
            self.logger.warning(f"[SOURCES] No retrieved_docs found in prepare_final_response_part")

        sources_list = []
        for doc in retrieved_docs:
            if isinstance(doc, dict):
                sources_list.append(doc)

        query_type = WorkflowUtils.get_state_value(state, "query_type", "general")
        query_complexity = WorkflowUtils.get_state_value(state, "query_complexity", "moderate")

        # needs_search 확인 (direct_answer 노드의 경우 검색이 없음)
        needs_search = WorkflowUtils.get_state_value(state, "needs_search", True)

        # ConfidenceCalculator를 사용하여 신뢰도 계산
        # direct_answer 노드의 경우 (needs_search=False) ConfidenceCalculator가 낮은 신뢰도를 계산할 수 있으므로 조정
        calculated_confidence = None
        if self.confidence_calculator and answer_value:
            try:
                confidence_info = self.confidence_calculator.calculate_confidence(
                    answer=answer_value,
                    sources=sources_list,
                    question_type=query_type
                )
                calculated_confidence = confidence_info.confidence
                
                # direct_answer 노드의 경우 (검색 없음) 신뢰도 보정
                if not needs_search and not sources_list:
                    # 검색 없이 직접 답변 생성한 경우 신뢰도 보정
                    # ConfidenceCalculator는 소스가 없으면 낮게 계산하므로, 직접 답변의 경우 보정 필요
                    if calculated_confidence < 0.60:
                        # 낮은 신뢰도는 직접 답변의 특성을 고려하여 보정
                        calculated_confidence = max(calculated_confidence * 1.2, 0.60)  # 최소 60% 보장
                        self.logger.info(f"[CONFIDENCE CALC] Direct answer confidence adjusted: {calculated_confidence:.3f} (no search)")
                    else:
                        self.logger.info(f"[CONFIDENCE CALC] Direct answer confidence: {calculated_confidence:.3f} (no search)")
                
                self.logger.info(f"ConfidenceCalculator: confidence={calculated_confidence:.3f}, factors={confidence_info.factors}")
            except Exception as e:
                self.logger.warning(f"ConfidenceCalculator failed: {e}")

        existing_confidence = state.get("structure_confidence") or state.get("confidence", 0.0)

        if calculated_confidence is not None:
            final_confidence = calculated_confidence
        else:
            final_confidence = existing_confidence

        # 기본 신뢰도 보장 (개선: 검색 품질 점수 반영)
        # search_quality를 여러 위치에서 찾기 (개선: search 그룹과 common 그룹도 확인)
        search_quality_score = 0.0
        search_quality_dict = state.get("search_quality", {})
        if not search_quality_dict or not isinstance(search_quality_dict, dict):
            # search 그룹에서 찾기
            if "search" in state and isinstance(state.get("search"), dict):
                search_quality_dict = state["search"].get("search_quality", {}) or state["search"].get("search_quality_evaluation", {})
        if not search_quality_dict or not isinstance(search_quality_dict, dict):
            # common.search 그룹에서 찾기
            if "common" in state and isinstance(state.get("common"), dict):
                if "search" in state["common"] and isinstance(state["common"]["search"], dict):
                    search_quality_dict = state["common"]["search"].get("search_quality", {}) or state["common"]["search"].get("search_quality_evaluation", {})
        if not search_quality_dict or not isinstance(search_quality_dict, dict):
            # search_quality_evaluation에서 찾기
            search_quality_dict = state.get("search_quality_evaluation", {})
        if not search_quality_dict or not isinstance(search_quality_dict, dict):
            # metadata에서 찾기
            metadata = state.get("metadata", {})
            if isinstance(metadata, dict):
                search_quality_dict = metadata.get("search_quality", {}) or metadata.get("search_quality_evaluation", {})
        
        # 전역 캐시에서도 찾기 (우선순위 높임 - 개선)
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
        
        # 로깅 추가
        self.logger.info(f"[CONFIDENCE CALC] search_quality_score: {search_quality_score:.3f} (from search_quality dict: {bool(search_quality_dict)}, keys: {list(search_quality_dict.keys()) if search_quality_dict else []})")
        
        quality_boost = search_quality_score * 0.3  # 검색 품질 점수 30% 반영 (20% -> 30%로 상향)
        
        # 검색 결과가 있고 품질이 좋으면 기본 신뢰도 상향
        # 검색 결과가 없을 때도 기본 신뢰도 보장 (개선)
        # direct_answer 노드의 경우 (needs_search=False) 다른 기준 적용
        search_failed = state.get("search_failed", False)
        if not needs_search:
            # direct_answer 노드: 검색 없이 직접 답변 생성
            # 답변 품질에 따라 기본 신뢰도 설정
            if answer_value:
                answer_length = len(answer_value)
                if answer_length >= 200:
                    base_min_confidence = 0.70  # 충분한 길이의 답변
                elif answer_length >= 100:
                    base_min_confidence = 0.65  # 적절한 길이의 답변
                elif answer_length >= 50:
                    base_min_confidence = 0.60  # 짧은 답변
                else:
                    base_min_confidence = 0.55  # 너무 짧은 답변
                self.logger.info(f"[CONFIDENCE CALC] Direct answer (no search): base_min_confidence={base_min_confidence:.3f}, answer_length={answer_length}")
            else:
                base_min_confidence = 0.50
        elif search_failed:
            # 검색 실패(데이터베이스 문제 등)인 경우 기본 신뢰도 낮게 설정
            base_min_confidence = 0.20 if answer_value else 0.10
            self.logger.warning(f"[CONFIDENCE CALC] Search failed, using lower base confidence: {base_min_confidence}")
        else:
            # 정상적인 경우 (기본 신뢰도 상향)
            base_min_confidence = 0.35 if (answer_value and sources_list and len(sources_list) >= 3 and search_quality_score > 0.3) else \
                                   0.30 if (answer_value and sources_list) else \
                                   0.25 if answer_value else 0.15  # 검색 결과가 없어도 답변이 있으면 최소 25% (0.20 -> 0.25)
        
        final_confidence = max(final_confidence, base_min_confidence) + quality_boost

        # 키워드 포함도 기반 보정
        keyword_coverage = self.calculate_keyword_coverage(state, answer_value)
        keyword_boost = keyword_coverage * 0.3
        adjusted_confidence = min(0.95, final_confidence + keyword_boost)

        # 소스 개수 기반 추가 보정 (개선: 더 많은 소스일수록 높은 보정)
        if sources_list:
            source_count = len(sources_list)
            if source_count >= 5:
                adjusted_confidence = min(0.95, adjusted_confidence + 0.08)  # 0.05 -> 0.08
            elif source_count >= 3:
                adjusted_confidence = min(0.95, adjusted_confidence + 0.05)  # 0.03 -> 0.05
            elif source_count >= 1:
                adjusted_confidence = min(0.95, adjusted_confidence + 0.02)  # 0.01 -> 0.02

        # 답변 길이 기반 추가 보정
        if answer_value:
            answer_length = len(answer_value)
            if answer_length >= 500:
                adjusted_confidence = min(0.95, adjusted_confidence + 0.05)
            elif answer_length >= 200:
                adjusted_confidence = min(0.95, adjusted_confidence + 0.03)
            elif answer_length >= 100:
                adjusted_confidence = min(0.95, adjusted_confidence + 0.01)

        # 일관된 신뢰도 계산 적용 (검증 점수 반영)
        # 검색 결과 기반 검증 점수 가져오기 (있는 경우)
        grounding_score = state.get("grounding_score")
        source_coverage = state.get("source_coverage")
        
        # 문서 인용 점수 계산 (개선: 2개 이상 인용 시 보정 추가)
        citation_count = 0
        if answer_value:
            import re
            # 법령 조문 인용 패턴
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
        
        # 문서 인용 점수 보정 증가 (개선: 2개 이상 인용 시 +0.08)
        citation_boost = 0.0
        if citation_count >= 3:
            citation_boost = 0.10  # 0.05 -> 0.10
            self.logger.info(f"[CONFIDENCE CALC] Citation boost applied: {citation_count} citations found (+{citation_boost})")
        elif citation_count >= 2:
            citation_boost = 0.08  # 0.05 -> 0.08
            self.logger.info(f"[CONFIDENCE CALC] Citation boost applied: {citation_count} citations found (+{citation_boost})")
        elif citation_count >= 1:
            citation_boost = 0.03  # 0.02 -> 0.03
            self.logger.info(f"[CONFIDENCE CALC] Citation boost applied: {citation_count} citation found (+{citation_boost})")
        
        # grounding_score 반영 비율 증가 (개선: 10% -> 15%)
        grounding_boost = 0.0
        if grounding_score is not None:
            grounding_boost = float(grounding_score) * 0.15  # 0.10 -> 0.15
            self.logger.info(f"[CONFIDENCE CALC] Grounding boost applied: grounding_score={grounding_score:.3f} (+{grounding_boost:.3f})")
        
        adjusted_confidence_with_validation = min(0.95, adjusted_confidence + citation_boost + grounding_boost)

        # 일관된 신뢰도로 최종 조정
        # direct_answer 노드의 경우 (needs_search=False) grounding_score가 None이므로 패널티 적용 안 함
        # grounding_score가 None이고 검색이 없는 경우에는 패널티를 적용하지 않도록 함
        final_adjusted_confidence = self._calculate_consistent_confidence(
            base_confidence=adjusted_confidence_with_validation,
            query_type=query_type,
            query_complexity=query_complexity or "moderate",
            grounding_score=grounding_score if (needs_search or grounding_score is not None) else None,  # 검색 없으면 None으로 전달하여 패널티 방지
            source_coverage=source_coverage if (needs_search or source_coverage is not None) else None  # 검색 없으면 None으로 전달하여 패널티 방지
        )

        state["confidence"] = final_adjusted_confidence

        # 신뢰도 값 설정 직후 답변 텍스트의 신뢰도 값 교체 (리팩토링된 메서드 사용)
        current_answer = state.get("answer", "")
        if current_answer and isinstance(current_answer, str) and final_adjusted_confidence > 0:
            state["answer"] = self.confidence_manager.replace_in_text(current_answer, final_adjusted_confidence)

        # 최종 answer를 문자열로 수렴
        try:
            state["answer"] = WorkflowUtils.normalize_answer(state.get("answer", ""))
        except Exception:
            state["answer"] = str(state.get("answer", ""))

        # normalize_answer 호출 이후 신뢰도 값 다시 교체 (정규화로 인한 손실 방지, 리팩토링된 메서드 사용)
        if final_adjusted_confidence > 0 and state.get("answer"):
            current_answer = state.get("answer", "")
            if isinstance(current_answer, str):
                state["answer"] = self.confidence_manager.replace_in_text(current_answer, final_adjusted_confidence)

        # sources 추출 (prepare_final_response와 동일한 로직 사용)
        final_sources_list = []
        final_sources_detail = []
        seen_sources = set()
        legal_refs = []
        seen_legal_refs = set()

        # 통일된 포맷터 및 검증기 초기화
        try:
            from ...services.unified_source_formatter import UnifiedSourceFormatter
            from ...services.source_validator import SourceValidator
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
                        # article_no가 문자열이 아니면 문자열로 변환
                        article_no_str = str(article_no) if article_no else ""
                        # article_no가 이미 "제2조" 형식이면 그대로 사용, 아니면 "제{article_no}조" 형식으로 변환
                        if article_no_str.startswith("제") and article_no_str.endswith("조"):
                            source_parts.append(article_no_str)
                        else:
                            # article_no에서 숫자만 추출
                            article_no_clean = article_no_str.strip()
                            if article_no_clean:
                                source_parts.append(f"제{article_no_clean}조")
                    if clause_no:
                        source_parts.append(f"제{clause_no}항")
                    if item_no:
                        source_parts.append(f"제{item_no}호")
                    
                    source = " ".join(source_parts)
            
            # 2. case_paragraph (판례) 처리
            elif source_type == "case_paragraph":
                court = doc.get("court") or metadata.get("court")
                casenames = doc.get("casenames") or metadata.get("casenames")
                doc_id = doc.get("doc_id") or metadata.get("doc_id") or metadata.get("case_id") or doc.get("id") or metadata.get("id")
                
                # court나 casenames가 없으면 다른 필드에서 찾기
                if not court and not casenames:
                    # metadata에서 추가 필드 확인
                    court = metadata.get("court_name") or metadata.get("court_type")
                    casenames = metadata.get("case_name") or metadata.get("title")
                
                if court or casenames or doc_id:
                    source_parts = []
                    if court:
                        source_parts.append(court)
                    if casenames:
                        source_parts.append(casenames)
                    if doc_id:
                        source_parts.append(f"({doc_id})")
                    # court나 casenames가 없어도 doc_id만 있으면 "판례 (doc_id)" 형태로 생성
                    if not court and not casenames and doc_id:
                        source_parts.insert(0, "판례")
                    source = " ".join(source_parts) if source_parts else None
            
            # 3. decision_paragraph (결정례) 처리
            elif source_type == "decision_paragraph":
                org = doc.get("org") or metadata.get("org")
                doc_id = doc.get("doc_id") or metadata.get("doc_id") or metadata.get("decision_id") or doc.get("id") or metadata.get("id")
                
                # org가 없으면 다른 필드에서 찾기
                if not org:
                    org = metadata.get("org_name") or metadata.get("organization")
                
                if org or doc_id:
                    source_parts = []
                    if org:
                        source_parts.append(org)
                    if doc_id:
                        source_parts.append(f"({doc_id})")
                    # org가 없어도 doc_id만 있으면 "결정례 (doc_id)" 형태로 생성
                    if not org and doc_id:
                        source_parts.insert(0, "결정례")
                    source = " ".join(source_parts) if source_parts else None
            
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
                    if source_lower not in invalid_sources and len(source_lower) >= 2:
                        source = source_raw.strip()
                
                if not source:
                    source = (
                        metadata.get("statute_name") or
                        metadata.get("statute_abbrv") or
                        metadata.get("law_name") or
                        metadata.get("court") or
                        metadata.get("court_name") or
                        metadata.get("org") or
                        metadata.get("org_name") or
                        metadata.get("title") or
                        metadata.get("case_name")
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
                    # source_type이 있으면 더 관대하게 처리 (source_type 기반으로 생성된 source는 유효)
                    is_valid_source = False
                    if source_type and source_type in ["statute_article", "case_paragraph", "decision_paragraph", "interpretation_paragraph"]:
                        # source_type 기반으로 생성된 source는 유효한 것으로 간주 (최소 1자 이상)
                        if source_lower not in invalid_sources and len(source_lower) >= 1:
                            is_valid_source = True
                    else:
                        # source_type이 없거나 일반적인 경우 기존 로직 사용 (최소 2자 이상)
                        if source_lower not in invalid_sources and len(source_lower) >= 2:
                            is_valid_source = True
                    
                    if is_valid_source:
                        if source_str not in seen_sources and source_str != "Unknown":
                            final_sources_list.append(source_str)
                            seen_sources.add(source_str)
                            
                            # statute_article 타입 문서의 경우 legal_references에도 추가
                            if source_type == "statute_article" and source_str:
                                # source_str에서 이미 "제{article_no}조" 형식으로 변환된 경우 그대로 사용
                                if source_str not in seen_legal_refs:
                                    legal_refs.append(source_str)
                                    seen_legal_refs.add(source_str)
                            
                            # sources_detail 추가
                            if source_info_detail:
                                detail_dict = {
                                    "name": source_info_detail.name,
                                    "type": source_info_detail.type,
                                    "url": source_info_detail.url or "",
                                    "metadata": source_info_detail.metadata or {}
                                }
                                
                                # metadata의 정보를 최상위 레벨로 추출
                                if source_info_detail.metadata:
                                    meta = source_info_detail.metadata
                                    
                                    # 법령 조문인 경우
                                    if source_type == "statute_article":
                                        if meta.get("statute_name"):
                                            detail_dict["statute_name"] = meta["statute_name"]
                                        if meta.get("article_no"):
                                            detail_dict["article_no"] = meta["article_no"]
                                        if meta.get("clause_no"):
                                            detail_dict["clause_no"] = meta["clause_no"]
                                        if meta.get("item_no"):
                                            detail_dict["item_no"] = meta["item_no"]
                                    
                                    # 판례인 경우
                                    elif source_type == "case_paragraph":
                                        if meta.get("doc_id"):
                                            detail_dict["case_number"] = meta["doc_id"]
                                        if meta.get("court"):
                                            detail_dict["court"] = meta["court"]
                                        if meta.get("casenames"):
                                            detail_dict["case_name"] = meta["casenames"]
                                    
                                    # 결정례인 경우
                                    elif source_type == "decision_paragraph":
                                        if meta.get("doc_id"):
                                            detail_dict["decision_number"] = meta["doc_id"]
                                        if meta.get("org"):
                                            detail_dict["org"] = meta["org"]
                                        if meta.get("decision_date"):
                                            detail_dict["decision_date"] = meta["decision_date"]
                                        if meta.get("result"):
                                            detail_dict["result"] = meta["result"]
                                    
                                    # 해석례인 경우
                                    elif source_type == "interpretation_paragraph":
                                        if meta.get("doc_id"):
                                            detail_dict["interpretation_number"] = meta["doc_id"]
                                        if meta.get("org"):
                                            detail_dict["org"] = meta["org"]
                                        if meta.get("title"):
                                            detail_dict["title"] = meta["title"]
                                        if meta.get("response_date"):
                                            detail_dict["response_date"] = meta["response_date"]
                                
                                # 상세본문 추가 (doc에서 text 또는 content 가져오기)
                                content = doc.get("content") or doc.get("text") or ""
                                if content:
                                    detail_dict["content"] = content
                                
                                final_sources_detail.append(detail_dict)
            elif source_type and source_type in ["case_paragraph", "decision_paragraph"]:
                # source_type이 있지만 source가 생성되지 않은 경우 디버깅
                self.logger.debug(f"[SOURCES DEBUG] source_type={source_type}, but source is None. doc_id={doc.get('doc_id') or metadata.get('doc_id') or metadata.get('case_id') or metadata.get('decision_id') or metadata.get('id')}")

        state["sources"] = final_sources_list[:10]  # 최대 10개만 (하위 호환성)
        state["sources_detail"] = final_sources_detail[:10]  # 최대 10개만 (신규 필드)
        
        # 디버깅: sources_detail 생성 결과 로깅
        if len(final_sources_detail) > 0:
            self.logger.info(f"[SOURCES_DETAIL] Generated {len(final_sources_detail)} sources_detail entries")
            for i, detail in enumerate(final_sources_detail[:3], 1):
                if isinstance(detail, dict):
                    self.logger.debug(f"[SOURCES_DETAIL] {i}. {detail.get('name', 'N/A')} (type: {detail.get('type', 'N/A')})")
        else:
            self.logger.warning(f"[SOURCES_DETAIL] No sources_detail generated from {len(state.get('retrieved_docs', []))} retrieved_docs")
        
        # 디버깅: sources 생성 결과 로깅
        if len(final_sources_list) > 0:
            self.logger.info(f"[SOURCES] Generated {len(final_sources_list)} sources: {final_sources_list[:5]}")
        else:
            retrieved_docs_count = len(state.get("retrieved_docs", []))
            self.logger.warning(f"[SOURCES] No sources generated from {retrieved_docs_count} retrieved_docs")

        # 법적 참조 정보 추가 (sources 생성 시점에 함께 생성)
        # statute_article 타입 문서의 sources를 legal_references로 사용
        # sources_detail에서 legal_references 추출 (리팩토링된 메서드 사용)
        legal_refs_from_detail = self.source_extractor.extract_legal_references_from_sources_detail(final_sources_detail)
        legal_refs.extend(legal_refs_from_detail)
        seen_legal_refs.update(legal_refs_from_detail)
        
        # sources_detail에서 찾지 못한 경우, retrieved_docs에서 직접 추출
        if len(legal_refs) == 0:
            legal_refs_from_docs = self.source_extractor.extract_legal_references_from_docs(state.get("retrieved_docs", []))
            legal_refs.extend(legal_refs_from_docs)
            seen_legal_refs.update(legal_refs_from_docs)
        
        state["legal_references"] = legal_refs[:10]  # 최대 10개만
        
        # 디버깅: legal_references 생성 결과 로깅
        if len(legal_refs) > 0:
            self.logger.info(f"[LEGAL_REFERENCES] Generated {len(legal_refs)} legal references: {legal_refs[:5]}")
        else:
            retrieved_docs_count = len(state.get("retrieved_docs", []))
            # statute_article 타입 문서 개수 확인
            statute_articles = [doc for doc in state.get("retrieved_docs", []) if isinstance(doc, dict) and (doc.get("type") == "statute_article" or doc.get("source_type") == "statute_article" or doc.get("metadata", {}).get("source_type") == "statute_article")]
            statute_articles_count = len(statute_articles)
            if statute_articles_count > 0:
                # statute_article 문서의 필드 확인
                sample_doc = statute_articles[0]
                statute_name = sample_doc.get("statute_name") or sample_doc.get("law_name") or sample_doc.get("metadata", {}).get("statute_name") or sample_doc.get("metadata", {}).get("law_name")
                self.logger.warning(f"[LEGAL_REFERENCES] No legal references generated from {retrieved_docs_count} retrieved_docs (statute_article: {statute_articles_count}개)")
                self.logger.warning(f"[LEGAL_REFERENCES] Sample statute_article doc: type={sample_doc.get('type')}, statute_name={statute_name}, article_no={sample_doc.get('article_no')}, metadata={sample_doc.get('metadata', {})}")
            else:
                self.logger.debug(f"[LEGAL_REFERENCES] No legal references generated from {retrieved_docs_count} retrieved_docs (no statute_article documents)")

        # related_questions 추출 (metadata에서 또는 phase_info에서 또는 LLM으로 생성)
        related_questions = []
        metadata = state.get("metadata", {})
        if isinstance(metadata, dict) and "related_questions" in metadata:
            related_questions = metadata.get("related_questions", [])
            if isinstance(related_questions, list) and len(related_questions) > 0:
                self.logger.warning(f"[RELATED_QUESTIONS] Found {len(related_questions)} related_questions in metadata")
        else:
            # phase_info에서 추출 시도
            phase_info = state.get("phase_info", {})
            self.logger.debug(f"[RELATED_QUESTIONS] Checking phase_info: {'present' if phase_info else 'missing'}, type: {type(phase_info)}")
            if isinstance(phase_info, dict):
                self.logger.debug(f"[RELATED_QUESTIONS] phase_info keys: {list(phase_info.keys())}")
                if "phase2" in phase_info:
                    phase2 = phase_info.get("phase2", {})
                    self.logger.debug(f"[RELATED_QUESTIONS] phase2 keys: {list(phase2.keys()) if isinstance(phase2, dict) else 'N/A'}")
                    if isinstance(phase2, dict) and "flow_tracking_info" in phase2:
                        flow_tracking = phase2.get("flow_tracking_info", {})
                        self.logger.debug(f"[RELATED_QUESTIONS] flow_tracking_info keys: {list(flow_tracking.keys()) if isinstance(flow_tracking, dict) else 'N/A'}")
                        if isinstance(flow_tracking, dict) and "suggested_questions" in flow_tracking:
                            suggested_questions = flow_tracking.get("suggested_questions", [])
                            self.logger.debug(f"[RELATED_QUESTIONS] suggested_questions: {len(suggested_questions) if isinstance(suggested_questions, list) else 'N/A'} items")
                            if isinstance(suggested_questions, list) and len(suggested_questions) > 0:
                                # 각 항목이 딕셔너리인 경우 "question" 필드 추출
                                if isinstance(suggested_questions[0], dict):
                                    related_questions = [q.get("question", "") for q in suggested_questions if q.get("question")]
                                else:
                                    related_questions = [str(q) for q in suggested_questions if q]
                                self.logger.info(f"[RELATED_QUESTIONS] Extracted {len(related_questions)} related_questions from phase_info")
                        else:
                            self.logger.debug(f"[RELATED_QUESTIONS] suggested_questions not found in flow_tracking_info")
                    else:
                        self.logger.debug(f"[RELATED_QUESTIONS] flow_tracking_info not found in phase2")
                else:
                    self.logger.debug(f"[RELATED_QUESTIONS] phase2 not found in phase_info")
        
        # related_questions가 없으면 템플릿 기반 생성 시도 (phase_info에 의존하지 않음)
        if not related_questions:
            try:
                query = state.get("query", "")
                answer = state.get("answer", "")
                self.logger.debug(f"[RELATED_QUESTIONS] Attempting to generate related_questions: query={query[:50] if query else 'None'}, answer={answer[:50] if answer else 'None'}")
                if query:
                    # answer가 없어도 query만으로 관련 질문 생성 가능
                    if not answer:
                        answer = ""  # 빈 문자열로 설정
                    # 간단한 템플릿 기반 관련 질문 생성
                    related_questions = self._generate_related_questions(query, answer)
                    if related_questions:
                        self.logger.info(f"[RELATED_QUESTIONS] Generated {len(related_questions)} related_questions using template: {related_questions[:3]}")
                else:
                    self.logger.debug(f"[RELATED_QUESTIONS] Cannot generate related_questions: query is empty")
            except Exception as e:
                self.logger.warning(f"[RELATED_QUESTIONS] Failed to generate related_questions: {e}", exc_info=True)
        
        # related_questions를 metadata에 저장
        if related_questions:
            if not isinstance(metadata, dict):
                metadata = {}
            metadata["related_questions"] = related_questions
            state["metadata"] = metadata
            self.logger.info(f"[RELATED_QUESTIONS] Saved {len(related_questions)} related_questions to metadata")
        else:
            self.logger.debug(f"[RELATED_QUESTIONS] No related_questions found (metadata keys: {list(metadata.keys()) if isinstance(metadata, dict) else 'N/A'})")

        # 메타데이터 설정
        self.set_metadata(state, answer_value, keyword_coverage)
    
    def _generate_related_questions(self, query: str, answer: str) -> List[str]:
        """관련 질문 생성 (템플릿 기반)"""
        related_questions = []
        
        # 질문에서 핵심 키워드 추출
        query_lower = query.lower()
        
        # 법령 관련 질문 패턴
        if any(keyword in query_lower for keyword in ["법령", "법률", "조문", "조", "항"]):
            related_questions.append(f"{query}에 대한 다른 법령도 확인해볼까요?")
            related_questions.append(f"{query}와 관련된 판례도 찾아볼까요?")
        
        # 판례 관련 질문 패턴
        elif any(keyword in query_lower for keyword in ["판례", "판결", "사건", "대법원"]):
            related_questions.append(f"{query}와 유사한 사건의 판례도 찾아볼까요?")
            related_questions.append(f"{query}에 대한 법령 조문도 확인해볼까요?")
        
        # 손해배상 관련 질문 패턴
        elif any(keyword in query_lower for keyword in ["손해배상", "배상", "손해", "청구"]):
            related_questions.append("손해배상 청구의 절차는 어떻게 되나요?")
            related_questions.append("손해배상의 범위는 어떻게 결정되나요?")
            related_questions.append("손해배상과 관련된 판례도 찾아볼까요?")
        
        # 일반적인 관련 질문
        if len(related_questions) < 3:
            related_questions.append(f"{query}에 대한 더 자세한 정보가 필요하신가요?")
            related_questions.append(f"{query}와 관련된 다른 질문이 있으신가요?")
        
        return related_questions[:5]

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
            self.logger.warning("[FORMAT_AND_PREPARE_FINAL] Calling prepare_final_response_part")
            self.prepare_final_response_part(state, query_complexity, needs_search)
            self.logger.warning(f"[FORMAT_AND_PREPARE_FINAL] prepare_final_response_part completed, legal_references={len(state.get('legal_references', []))}")

            # Part 3: 최종 후처리 (리팩토링된 메서드 사용)
            final_answer = state.get("answer", "")
            if final_answer:
                import re

                # 중복 헤더 제거 (리팩토링된 메서드 사용)
                final_answer = self.answer_cleaner.remove_duplicate_headers(final_answer)

                # 연속된 빈 줄 정리 (3개 이상 -> 2개)
                final_answer = re.sub(r'\n{3,}', '\n\n', final_answer)

                # 공백 없는 텍스트 수정 (예: "민사법상계약해지의요건" -> "민사법상 계약 해지의 요건")
                # 한글 + 영문/숫자 사이에 공백 추가
                final_answer = re.sub(r'([가-힣])([A-Za-z0-9])', r'\1 \2', final_answer)
                final_answer = re.sub(r'([A-Za-z0-9])([가-힣])', r'\1 \2', final_answer)
                # 특정 패턴 수정 (법령명 + 조항)
                final_answer = re.sub(r'([가-힣]+법)([가-힣])', r'\1 \2', final_answer)
                # "의", "및", "와", "과" 앞뒤 공백 보장
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
                        sources_text = "\n".join([f"- {source}" for source in valid_sources[:5]])
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
                        f"답변 검증 결과: grounding_score=N/A (direct_answer, no search), "
                        f"unverified_count=0"
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
