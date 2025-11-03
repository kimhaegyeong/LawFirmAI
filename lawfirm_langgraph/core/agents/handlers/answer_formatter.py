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

# Constants for processing steps
MAX_PROCESSING_STEPS = 50

# 답변 길이 목표 (질의 유형별)
ANSWER_LENGTH_TARGETS = {
    "simple_question": (500, 1000),      # 간단한 질의: 500-1000자
    "term_explanation": (800, 1500),     # 용어 설명: 800-1500자
    "legal_analysis": (1500, 2500),      # 법률 분석: 1500-2500자
    "complex_question": (2000, 3500),    # 복잡한 질의: 2000-3500자
    "default": (800, 2000)               # 기본값: 800-2000자
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

                    from core.services.confidence_calculator import (
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

                from core.services.confidence_calculator import (
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

            # 신뢰도 값 통일 (현재 state의 confidence 값으로 교체) - format_answer_part 단계에서도 적용
            current_confidence = state.get("confidence", 0.0)
            if current_confidence > 0:
                confidence_str = f"{current_confidence:.1%}"
                # 신뢰도 레벨 결정
                if current_confidence >= 0.8:
                    level = "high"
                    emoji = "🟢"
                elif current_confidence >= 0.6:
                    level = "medium"
                    emoji = "🟡"
                else:
                    level = "low"
                    emoji = "🟠"

                # 반복 적용하여 모든 신뢰도 패턴 교체
                for _ in range(5):  # 더 많이 반복
                    formatted_answer = re.sub(r'\*\*신뢰도:\s*[\d.]+%\*\*', f'**신뢰도: {confidence_str}**', formatted_answer, flags=re.IGNORECASE)
                    formatted_answer = re.sub(r'🟡\s*\*\*신뢰도:\s*[\d.]+%\*\*', f'{emoji} **신뢰도: {confidence_str}**', formatted_answer, flags=re.IGNORECASE)
                    formatted_answer = re.sub(r'🟠\s*\*\*신뢰도:\s*[\d.]+%\*\*', f'{emoji} **신뢰도: {confidence_str}**', formatted_answer, flags=re.IGNORECASE)
                    formatted_answer = re.sub(r'🟢\s*\*\*신뢰도:\s*[\d.]+%\*\*', f'{emoji} **신뢰도: {confidence_str}**', formatted_answer, flags=re.IGNORECASE)
                    formatted_answer = re.sub(r'신뢰도:\s*[\d.]+%', f'신뢰도: {confidence_str}', formatted_answer, flags=re.IGNORECASE)
                    formatted_answer = re.sub(r'답변품질:\s*[\d.]+%', f'답변 품질: {confidence_str}', formatted_answer, flags=re.IGNORECASE)

                # "신뢰도정보" 섹션도 교체 (format_answer_part 단계에서)
                new_confidence_section = f'### 💡 신뢰도정보\n{emoji} **신뢰도: {confidence_str}** ({level})\n\n**상세점수:**\n- 답변 품질: {confidence_str}\n\n**설명:** 신뢰도: {confidence_str}'

                # 섹션을 직접 찾아 교체
                lines = formatted_answer.split('\n')
                new_lines = []
                in_confidence_section = False

                for line in lines:
                    if re.match(r'^###\s*💡\s*신뢰도정보', line, re.IGNORECASE):
                        in_confidence_section = True
                        new_lines.append(new_confidence_section)
                        continue

                    if in_confidence_section:
                        if line.strip() == '---' or line.strip().startswith('💼') or re.match(r'^###\s+', line):
                            in_confidence_section = False
                            new_lines.append(line)
                        continue

                    new_lines.append(line)

                formatted_answer = '\n'.join(new_lines)

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

            # sources 추출 (개선: 메타데이터에서도 소스 정보 추출)
            final_sources_list = []
            seen_sources = set()

            for doc in state.get("retrieved_docs", []):
                if not isinstance(doc, dict):
                    continue

                # 다양한 필드에서 소스 추출 시도
                source = None

                # 1. 직접 source 필드 확인 (단, "semantic", "keyword" 같은 검색 타입은 제외)
                # 우선순위: statute_name > law_name > source_name > source
                source_raw = (
                    doc.get("statute_name") or
                    doc.get("law_name") or
                    doc.get("source_name") or
                    doc.get("source")
                )

                # 검색 타입이 아닌 실제 소스명만 추출
                if source_raw and isinstance(source_raw, str):
                    source_lower = source_raw.lower().strip()
                    # 검색 타입 필터링 (더 포괄적)
                    invalid_sources = ["semantic", "keyword", "unknown", "fts", "vector", "search", "text2sql", ""]
                    if source_lower not in invalid_sources and len(source_lower) > 2:
                        source = source_raw.strip()
                    else:
                        source = None
                else:
                    source = None

                # law_name, statute_name도 별도로 확인 (위에서 확인했지만 재확인)
                if not source:
                    law_name = doc.get("law_name") or doc.get("statute_name")
                    if law_name and isinstance(law_name, str) and law_name.strip() and len(law_name.strip()) > 2:
                        source = law_name.strip()

                # 2. metadata에서 소스 정보 추출
                if not source:
                    metadata = doc.get("metadata", {})
                    if isinstance(metadata, dict):
                        source = (
                            metadata.get("statute_name") or
                            metadata.get("statute_abbrv") or
                            metadata.get("law_name") or
                            metadata.get("court") or
                            metadata.get("org") or
                            metadata.get("title")
                        )

                # 3. content나 text에서 법령명 추출 시도 (정규식 패턴)
                if not source:
                    content = doc.get("content", "") or doc.get("text", "")
                    if isinstance(content, str) and content:
                        import re
                        # 법령명 패턴 찾기 (예: "민법 제550조", "형법 제257조" 등)
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

                    if source_str and source_str not in seen_sources and source_str != "Unknown":
                        final_sources_list.append(source_str)
                        seen_sources.add(source_str)

            state["sources"] = final_sources_list[:10]  # 최대 10개만

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
        """답변 텍스트에서 메타 정보 섹션 제거 (줄 단위 직접 처리)"""
        import re

        if not answer_text or not isinstance(answer_text, str):
            return answer_text

        lines = answer_text.split('\n')
        cleaned_lines = []
        in_confidence_section = False
        in_reference_section = False
        in_disclaimer_section = False

        for line in lines:
            # 신뢰도 정보 섹션 시작
            if re.match(r'^###\s*💡\s*신뢰도정보', line, re.IGNORECASE):
                in_confidence_section = True
                continue

            # 참고 자료 섹션 시작
            if re.match(r'^###\s*📚\s*참고\s*자료', line, re.IGNORECASE):
                in_reference_section = True
                continue

            # 면책 조항 섹션 시작 (--- 또는 💼)
            if line.strip() == '---' or re.match(r'^\s*💼\s*\*\*면책\s*조항\*\*', line, re.IGNORECASE):
                in_disclaimer_section = True
                continue

            # 섹션 종료 확인
            if in_confidence_section:
                # 다음 ### 섹션이나 --- 나오면 종료
                if re.match(r'^###\s+', line) or line.strip() == '---':
                    in_confidence_section = False
                    # 이 줄은 건너뛰기
                    continue
                # 섹션 내부는 모두 건너뛰기
                continue

            if in_reference_section:
                # 다음 ### 섹션이나 --- 나오면 종료
                if re.match(r'^###\s+', line) or line.strip() == '---':
                    in_reference_section = False
                    # 이 줄은 건너뛰기
                    continue
                # 섹션 내부는 모두 건너뛰기
                continue

            if in_disclaimer_section:
                # 면책 조항 섹션은 끝까지 모두 건너뛰기
                continue

            # 남아있는 메타 정보 패턴 제거 (상세 점수, 설명 등)
            if re.match(r'^\*\*상세\s*점수:\*\*', line, re.IGNORECASE):
                continue
            if re.match(r'^\*\*설명:\*\*', line, re.IGNORECASE):
                continue
            if re.match(r'^-\s*답변\s*품질:', line, re.IGNORECASE):
                continue
            if re.match(r'^-\s*신뢰도:', line, re.IGNORECASE):
                continue

            # 메타 정보 섹션이 아닌 경우만 추가
            cleaned_lines.append(line)

        cleaned_text = '\n'.join(cleaned_lines)

        # 연속된 빈 줄 정리
        cleaned_text = re.sub(r'\n{3,}', '\n\n', cleaned_text)

        # 남아있는 메타 정보 패턴 추가 제거
        # "**상세 점수:**" 섹션 제거
        cleaned_text = re.sub(r'\*\*상세\s*점수:\*\*.*?\n', '', cleaned_text, flags=re.DOTALL | re.IGNORECASE)
        # "- 답변 품질:" 패턴 제거
        cleaned_text = re.sub(r'-\s*답변\s*품질:\s*[\d.]+%?\s*\n?', '', cleaned_text, flags=re.IGNORECASE | re.MULTILINE)
        # "**설명:**" 패턴 제거
        cleaned_text = re.sub(r'\*\*설명:\*\*\s*신뢰도:.*?\n?', '', cleaned_text, flags=re.IGNORECASE | re.MULTILINE)
        # "- 신뢰도:" 패턴 제거
        cleaned_text = re.sub(r'-\s*신뢰도:\s*[\d.]+%?\s*\n?', '', cleaned_text, flags=re.IGNORECASE | re.MULTILINE)

        # 연속된 빈 줄 재정리
        cleaned_text = re.sub(r'\n{3,}', '\n\n', cleaned_text)

        return cleaned_text.strip()

    def _remove_answer_header(self, answer_text: str) -> str:
        """답변 텍스트에서 '## 답변' 헤더 제거"""
        import re

        if not answer_text or not isinstance(answer_text, str):
            return answer_text

        # '## 답변' 헤더 제거 (단독 라인으로 있는 경우)
        answer_text = re.sub(r'^##\s*답변\s*\n+', '', answer_text, flags=re.MULTILINE | re.IGNORECASE)

        # 앞부분의 빈 줄 제거
        answer_text = answer_text.lstrip('\n')

        return answer_text

    def _remove_intermediate_text(self, answer_text: str) -> str:
        """
        중간 생성 텍스트 제거 (STEP 0, 원본 답변, 질문 정보 등)

        Args:
            answer_text: 원본 답변 텍스트

        Returns:
            중간 텍스트가 제거된 답변
        """
        import re

        if not answer_text or not isinstance(answer_text, str):
            return answer_text

        lines = answer_text.split('\n')
        cleaned_lines = []
        skip_section = False

        # 제거할 패턴 목록
        skip_patterns = [
            r'^##\s*STEP\s*0',
            r'^##\s*원본\s*품질\s*평가',
            r'^##\s*질문\s*정보',
            r'^##\s*원본\s*답변',
            r'^\*\*질문\*\*:',
            r'^\*\*질문\s*유형\*\*:',
            r'^평가\s*결과',
            r'원본\s*에\s*개선이\s*필요하면',
            r'^\*\*평가\s*결\s*과\s*에\s*따른\s*작업',
        ]

        for i, line in enumerate(lines):
            # 섹션 시작 패턴 확인
            is_section_start = False
            for pattern in skip_patterns:
                if re.match(pattern, line, re.IGNORECASE):
                    skip_section = True
                    is_section_start = True
                    self.logger.debug(f"[INTERMEDIATE TEXT REMOVAL] Found skip pattern: {line[:50]}")
                    break

            if is_section_start:
                continue

            # 섹션 종료 확인 (다음 ## 헤더 또는 실제 답변 시작)
            if skip_section:
                # 다음 ## 헤더가 나오거나, 실제 답변 시작 패턴 확인
                if re.match(r'^##\s+[가-힣]', line):  # 실제 답변 섹션 시작
                    skip_section = False
                    # 이 줄은 포함 (하지만 패턴에 매칭되지 않는 경우만)
                    if not any(re.match(p, line, re.IGNORECASE) for p in skip_patterns):
                        cleaned_lines.append(line)
                    continue

                # 체크리스트 패턴 제거 (• [ ] 형태)
                if re.match(r'^\s*[•\-\*]\s*\[.*?\].*?', line):
                    continue

                # "안녕하세요" 같은 인사말 뒤에 오는 불필요한 텍스트도 제거
                if re.match(r'^안녕하세요.*?궁금하시군요', line, re.IGNORECASE):
                    continue

                # 섹션 내부의 다른 줄들은 모두 건너뛰기
                continue
            else:
                # 일반 텍스트 추가 (체크리스트 패턴 필터링)
                if re.match(r'^\s*[•\-\*]\s*\[.*?\].*?', line):
                    continue

                # 체크박스 패턴 제거 (• [ ] 법적 정보가 충분하고...)
                if re.search(r'\[.*?\].*?(충분|명확|일관|포함)', line):
                    continue

                cleaned_lines.append(line)

        cleaned_text = '\n'.join(cleaned_lines)

        # 연속된 빈 줄 정리
        cleaned_text = re.sub(r'\n{3,}', '\n\n', cleaned_text)

        # 앞뒤 공백 제거
        cleaned_text = cleaned_text.strip()

        self.logger.debug(f"[INTERMEDIATE TEXT REMOVAL] Removed sections, original: {len(answer_text)}, cleaned: {len(cleaned_text)}")

        return cleaned_text

    def _adjust_answer_length(
        self,
        answer: str,
        query_type: str,
        query_complexity: str
    ) -> str:
        """
        답변 길이를 질의 유형에 맞게 조절

        Args:
            answer: 원본 답변
            query_type: 질의 유형
            query_complexity: 질의 복잡도

        Returns:
            조절된 답변
        """
        import re

        if not answer:
            return answer

        current_length = len(answer)

        # 목표 길이 결정
        if query_complexity == "simple":
            min_len, max_len = ANSWER_LENGTH_TARGETS.get("simple_question", (500, 1000))
        elif query_complexity == "complex":
            min_len, max_len = ANSWER_LENGTH_TARGETS.get("complex_question", (2000, 3500))
        else:
            targets = ANSWER_LENGTH_TARGETS.get(query_type, ANSWER_LENGTH_TARGETS["default"])
            min_len, max_len = targets

        # 길이가 적절한 경우 그대로 반환
        if min_len <= current_length <= max_len:
            self.logger.debug(f"[ANSWER LENGTH] Length OK: {current_length} (target: {min_len}-{max_len})")
            return answer

        # 너무 긴 경우: 핵심 내용만 추출
        if current_length > max_len:
            self.logger.info(f"[ANSWER LENGTH] Too long: {current_length}, adjusting to max {max_len}")
            # 섹션별로 분리
            sections = re.split(r'\n\n+', answer)

            # 각 섹션의 중요도 평가 (법령 인용, 판례 등 포함 여부)
            important_sections = []
            other_sections = []

            for section in sections:
                if (re.search(r'\[법령:', section) or
                    re.search(r'대법원', section) or
                    re.search(r'제\s*\d+\s*조', section)):
                    important_sections.append(section)
                else:
                    other_sections.append(section)

            # 중요 섹션 우선 포함
            result = []
            current_len = 0

            for section in important_sections:
                if current_len + len(section) <= max_len:
                    result.append(section)
                    current_len += len(section)
                else:
                    # 섹션 일부만 포함
                    remaining = max_len - current_len - 10  # 여유 공간
                    if remaining > 100:  # 최소 100자 이상은 포함
                        result.append(section[:remaining] + "...")
                    break

            # 여유가 있으면 다른 섹션도 포함
            for section in other_sections:
                if current_len + len(section) <= max_len:
                    result.append(section)
                    current_len += len(section)
                else:
                    break

            adjusted_answer = '\n\n'.join(result)
            self.logger.info(f"[ANSWER LENGTH] Adjusted: {len(answer)} -> {len(adjusted_answer)}")
            return adjusted_answer

        # 너무 짧은 경우: 이미 최소 길이로 생성된 것이므로 그대로 반환
        # (추가 생성은 LLM 호출이 필요하므로 여기서는 하지 않음)
        self.logger.debug(f"[ANSWER LENGTH] Too short: {current_length} (target: {min_len}-{max_len}), keeping as is")
        return answer

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

        # 3. 검증 점수에 따른 조정 (있는 경우)
        if grounding_score is not None and grounding_score < 0.8:
            confidence -= (0.8 - grounding_score) * 0.3  # 최대 30% 감소

        # 4. 소스 커버리지에 따른 조정 (있는 경우)
        if source_coverage is not None and source_coverage < 0.5:
            confidence -= (0.5 - source_coverage) * 0.2  # 최대 20% 감소

        # 5. 범위 제한 (0.0 ~ 1.0)
        confidence = max(0.0, min(1.0, confidence))

        # 6. 질의 유형별 최소 신뢰도 설정
        min_confidence_by_type = {
            "simple_question": 0.75,
            "term_explanation": 0.80,
            "legal_analysis": 0.75,
            "complex_question": 0.70,
            "general_question": 0.70
        }
        min_confidence = min_confidence_by_type.get(query_type, 0.70)

        # 최소 신뢰도보다 낮으면 경고 (하지만 강제로 올리지는 않음)
        if confidence < min_confidence:
            self.logger.warning(
                f"신뢰도가 최소 기준({min_confidence:.2%})보다 낮음: {confidence:.2%}"
            )

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

        # 일관된 신뢰도 계산 적용 (검증 점수 반영)
        # 검색 결과 기반 검증 점수 가져오기 (있는 경우)
        grounding_score = state.get("grounding_score")
        source_coverage = state.get("source_coverage")

        # 일관된 신뢰도로 최종 조정
        final_adjusted_confidence = self._calculate_consistent_confidence(
            base_confidence=adjusted_confidence,
            query_type=query_type,
            query_complexity=query_complexity or "moderate",
            grounding_score=grounding_score,
            source_coverage=source_coverage
        )

        state["confidence"] = final_adjusted_confidence

        # 신뢰도 값 설정 직후 답변 텍스트의 신뢰도 값 교체 (중요: prepare_final_response_part에서)
        import re
        current_answer = state.get("answer", "")
        if current_answer and isinstance(current_answer, str) and final_adjusted_confidence > 0:
            confidence_str = f"{final_adjusted_confidence:.1%}"
            # 신뢰도 레벨 결정
            if final_adjusted_confidence >= 0.8:
                level = "high"
                emoji = "🟢"
            elif final_adjusted_confidence >= 0.6:
                level = "medium"
                emoji = "🟡"
            else:
                level = "low"
                emoji = "🟠"

            # 반복 적용하여 모든 신뢰도 패턴 교체
            for _ in range(5):
                current_answer = re.sub(r'\*\*신뢰도:\s*[\d.]+%\*\*', f'**신뢰도: {confidence_str}**', current_answer, flags=re.IGNORECASE)
                current_answer = re.sub(r'🟡\s*\*\*신뢰도:\s*[\d.]+%\*\*', f'{emoji} **신뢰도: {confidence_str}**', current_answer, flags=re.IGNORECASE)
                current_answer = re.sub(r'🟠\s*\*\*신뢰도:\s*[\d.]+%\*\*', f'{emoji} **신뢰도: {confidence_str}**', current_answer, flags=re.IGNORECASE)
                current_answer = re.sub(r'🟢\s*\*\*신뢰도:\s*[\d.]+%\*\*', f'{emoji} **신뢰도: {confidence_str}**', current_answer, flags=re.IGNORECASE)
                current_answer = re.sub(r'신뢰도:\s*[\d.]+%', f'신뢰도: {confidence_str}', current_answer, flags=re.IGNORECASE)
                current_answer = re.sub(r'답변품질:\s*[\d.]+%', f'답변 품질: {confidence_str}', current_answer, flags=re.IGNORECASE)
                # 레벨도 함께 교체
                current_answer = re.sub(r'\*\*신뢰도:\s*[\d.]+%\*\*\s*\(low\)', f'**신뢰도: {confidence_str}** ({level})', current_answer, flags=re.IGNORECASE)
                current_answer = re.sub(r'\*\*신뢰도:\s*[\d.]+%\*\*\s*\(medium\)', f'**신뢰도: {confidence_str}** ({level})', current_answer, flags=re.IGNORECASE)
                current_answer = re.sub(r'\*\*신뢰도:\s*[\d.]+%\*\*\s*\(high\)', f'**신뢰도: {confidence_str}** ({level})', current_answer, flags=re.IGNORECASE)
                current_answer = re.sub(r'🟢\s*\*\*신뢰도:\s*[\d.]+%\*\*\s*\(low\)', f'{emoji} **신뢰도: {confidence_str}** ({level})', current_answer, flags=re.IGNORECASE)
                current_answer = re.sub(r'🟡\s*\*\*신뢰도:\s*[\d.]+%\*\*\s*\(low\)', f'{emoji} **신뢰도: {confidence_str}** ({level})', current_answer, flags=re.IGNORECASE)
                current_answer = re.sub(r'🟠\s*\*\*신뢰도:\s*[\d.]+%\*\*\s*\(low\)', f'{emoji} **신뢰도: {confidence_str}** ({level})', current_answer, flags=re.IGNORECASE)

            # "신뢰도정보" 섹션 직접 교체
            new_confidence_section = f'### 💡 신뢰도정보\n{emoji} **신뢰도: {confidence_str}** ({level})\n\n**상세점수:**\n- 답변 품질: {confidence_str}\n\n**설명:** 신뢰도: {confidence_str}'

            lines = current_answer.split('\n')
            new_lines = []
            in_confidence_section = False

            for line in lines:
                if re.match(r'^###\s*💡\s*신뢰도정보', line, re.IGNORECASE):
                    in_confidence_section = True
                    new_lines.append(new_confidence_section)
                    continue

                if in_confidence_section:
                    if line.strip() == '---' or line.strip().startswith('💼') or re.match(r'^###\s+', line):
                        in_confidence_section = False
                        new_lines.append(line)
                    continue

                new_lines.append(line)

            state["answer"] = '\n'.join(new_lines)

        # 최종 answer를 문자열로 수렴
        try:
            state["answer"] = WorkflowUtils.normalize_answer(state.get("answer", ""))
        except Exception:
            state["answer"] = str(state.get("answer", ""))

        # normalize_answer 호출 이후 신뢰도 값 다시 교체 (정규화로 인한 손실 방지)
        if final_adjusted_confidence > 0 and state.get("answer"):
            current_answer = state.get("answer", "")
            if isinstance(current_answer, str):
                confidence_str = f"{final_adjusted_confidence:.1%}"
                if final_adjusted_confidence >= 0.8:
                    level = "high"
                    emoji = "🟢"
                elif final_adjusted_confidence >= 0.6:
                    level = "medium"
                    emoji = "🟡"
                else:
                    level = "low"
                    emoji = "🟠"

                # 반복 적용하여 모든 신뢰도 패턴 교체
                for _ in range(5):
                    current_answer = re.sub(r'\*\*신뢰도:\s*[\d.]+%\*\*', f'**신뢰도: {confidence_str}**', current_answer, flags=re.IGNORECASE)
                    current_answer = re.sub(r'🟡\s*\*\*신뢰도:\s*[\d.]+%\*\*', f'{emoji} **신뢰도: {confidence_str}**', current_answer, flags=re.IGNORECASE)
                    current_answer = re.sub(r'🟠\s*\*\*신뢰도:\s*[\d.]+%\*\*', f'{emoji} **신뢰도: {confidence_str}**', current_answer, flags=re.IGNORECASE)
                    current_answer = re.sub(r'🟢\s*\*\*신뢰도:\s*[\d.]+%\*\*', f'{emoji} **신뢰도: {confidence_str}**', current_answer, flags=re.IGNORECASE)
                    current_answer = re.sub(r'신뢰도:\s*[\d.]+%', f'신뢰도: {confidence_str}', current_answer, flags=re.IGNORECASE)
                    current_answer = re.sub(r'답변품질:\s*[\d.]+%', f'답변 품질: {confidence_str}', current_answer, flags=re.IGNORECASE)
                    # 레벨도 함께 교체
                    current_answer = re.sub(r'\*\*신뢰도:\s*[\d.]+%\*\*\s*\(low\)', f'**신뢰도: {confidence_str}** ({level})', current_answer, flags=re.IGNORECASE)
                    current_answer = re.sub(r'\*\*신뢰도:\s*[\d.]+%\*\*\s*\(medium\)', f'**신뢰도: {confidence_str}** ({level})', current_answer, flags=re.IGNORECASE)
                    current_answer = re.sub(r'\*\*신뢰도:\s*[\d.]+%\*\*\s*\(high\)', f'**신뢰도: {confidence_str}** ({level})', current_answer, flags=re.IGNORECASE)
                    current_answer = re.sub(r'🟢\s*\*\*신뢰도:\s*[\d.]+%\*\*\s*\(low\)', f'{emoji} **신뢰도: {confidence_str}** ({level})', current_answer, flags=re.IGNORECASE)
                    current_answer = re.sub(r'🟡\s*\*\*신뢰도:\s*[\d.]+%\*\*\s*\(low\)', f'{emoji} **신뢰도: {confidence_str}** ({level})', current_answer, flags=re.IGNORECASE)
                    current_answer = re.sub(r'🟠\s*\*\*신뢰도:\s*[\d.]+%\*\*\s*\(low\)', f'{emoji} **신뢰도: {confidence_str}** ({level})', current_answer, flags=re.IGNORECASE)

                # "신뢰도정보" 섹션 직접 교체 (다시)
                new_confidence_section = f'### 💡 신뢰도정보\n{emoji} **신뢰도: {confidence_str}** ({level})\n\n**상세점수:**\n- 답변 품질: {confidence_str}\n\n**설명:** 신뢰도: {confidence_str}'

                lines = current_answer.split('\n')
                new_lines = []
                in_confidence_section = False

                for line in lines:
                    if re.match(r'^###\s*💡\s*신뢰도정보', line, re.IGNORECASE):
                        in_confidence_section = True
                        new_lines.append(new_confidence_section)
                        continue

                    if in_confidence_section:
                        if line.strip() == '---' or line.strip().startswith('💼') or re.match(r'^###\s+', line):
                            in_confidence_section = False
                            new_lines.append(line)
                        continue

                    new_lines.append(line)

                state["answer"] = '\n'.join(new_lines)

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

            # Part 3: 최종 후처리 (개선: 중복 헤더, 빈 섹션, 불필요한 형식 제거)
            final_answer = state.get("answer", "")
            if final_answer:
                import re

                # 중복 헤더 제거 (개선된 버전)
                lines = final_answer.split('\n')
                result_lines = []
                seen_headers = set()
                skip_next_empty = False

                for i, line in enumerate(lines):
                    header_match = re.match(r'^(#{1,3})\s+(.+)', line)
                    if header_match:
                        level = len(header_match.group(1))
                        header_text = header_match.group(2).strip()

                        # 이모지 및 특수문자 제거
                        clean_header = re.sub(r'[📖⚖️💼💡📚📋⭐📌🔍💬🎯📊📝📄⏰🔗⚠️❗✅🚨🎉💯🔔]+\s*', '', header_text).strip()

                        # "답변", "답" 같은 단어만 포함된 헤더는 더 일반적으로 처리
                        normalized_header = re.sub(r'\s+', ' ', clean_header.lower())

                        # 중복 확인 (같은 레벨, 같은 제목)
                        header_key = f"{level}:{normalized_header}"

                        # 특정 중복 패턴 제거
                        if normalized_header in ["답변", "answer", "답"]:
                            # 이미 "답변" 헤더가 있으면 중복 제거
                            if "답변" in seen_headers or "answer" in seen_headers:
                                skip_next_empty = True
                                continue

                        if header_key in seen_headers:
                            skip_next_empty = True
                            continue

                        seen_headers.add(normalized_header)
                        seen_headers.add(header_key)
                        skip_next_empty = False
                    elif skip_next_empty and line.strip() == "":
                        # 중복 헤더 다음의 빈 줄도 제거
                        continue
                    else:
                        skip_next_empty = False

                    result_lines.append(line)

                final_answer = '\n'.join(result_lines)

                # 중복 헤더 제거 (더 강력한 방식 - 줄 단위 직접 처리)
                lines = final_answer.split('\n')
                cleaned_lines = []
                seen_answer_header = False
                i = 0

                while i < len(lines):
                    line = lines[i]
                    # "## 답변" 헤더는 한 번만 유지
                    if re.match(r'^##\s*답변\s*$', line, re.IGNORECASE):
                        if not seen_answer_header:
                            cleaned_lines.append(line)
                            seen_answer_header = True
                        # 다음 줄이 "###"로 시작하면 건너뛰기
                        if i + 1 < len(lines) and re.match(r'^###\s*.*답변', lines[i + 1], re.IGNORECASE):
                            i += 2  # "## 답변"과 "### 답변" 모두 건너뛰기
                            continue
                        else:
                            i += 1
                            continue
                    # "###" 로 시작하고 "답변"이 포함된 줄 제거
                    elif re.match(r'^###\s*.*답변', line, re.IGNORECASE):
                        i += 1
                        continue  # 이 줄은 건너뛰기
                    else:
                        cleaned_lines.append(line)
                        i += 1

                final_answer = '\n'.join(cleaned_lines)

                # 추가 패턴 제거 (정규식으로 남은 것들 처리)
                # "## 답변" 바로 다음에 오는 "###" 헤더 제거
                final_answer = re.sub(
                    r'(##\s*답변\s*\n+)(###\s*.*답변\s*\n+)',
                    r'\1',
                    final_answer,
                    flags=re.MULTILINE | re.IGNORECASE
                )

                # 연속된 "## 답변" 패턴 제거
                final_answer = re.sub(
                    r'##\s*답변\s*\n+\s*##\s*답변',
                    '## 답변',
                    final_answer,
                    flags=re.IGNORECASE | re.MULTILINE
                )

                # 빈 섹션 정리 (헤더만 있고 내용 없는 섹션)
                final_answer = re.sub(r'###\s+[^\n]+\s*\n\s*\n(?=###|$)', '', final_answer, flags=re.MULTILINE)

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

                # 답변 내부의 하드코딩된 신뢰도 값 교체 (state의 confidence로 통일)
                current_confidence = state.get("confidence", 0.0)
                if current_confidence > 0:
                    # 답변 내부의 모든 신뢰도 패턴 찾기 및 교체 (개선: 더 포괄적인 패턴)
                    confidence_str = f"{current_confidence:.1%}"

                    # 다양한 신뢰도 패턴 교체 (더 포괄적이고 강력한 패턴, 모든 경우를 찾기 위해 반복 적용)
                    # 이모지 포함 패턴 (우선 처리, 더 포괄적인 패턴)
                    final_answer = re.sub(r'🟠\s*\*\*신뢰도:\s*[\d.]+%\*\*\s*\(low\)', f'🟡 **신뢰도: {confidence_str}**', final_answer, flags=re.IGNORECASE)
                    final_answer = re.sub(r'🟡\s*\*\*신뢰도:\s*[\d.]+%\*\*\s*\(medium\)', f'🟡 **신뢰도: {confidence_str}**', final_answer, flags=re.IGNORECASE)
                    final_answer = re.sub(r'🟠\s*\*\*신뢰도:\s*[\d.]+%\*\*', f'🟡 **신뢰도: {confidence_str}**', final_answer, flags=re.IGNORECASE)
                    final_answer = re.sub(r'🟡\s*\*\*신뢰도:\s*[\d.]+%\*\*', f'🟡 **신뢰도: {confidence_str}**', final_answer, flags=re.IGNORECASE)

                    # 볼드 패턴 (더 포괄적)
                    final_answer = re.sub(r'\*\*신뢰도:\s*[\d.]+%\*\*\s*\(low\)', f'**신뢰도: {confidence_str}**', final_answer, flags=re.IGNORECASE)
                    final_answer = re.sub(r'\*\*신뢰도:\s*[\d.]+%\*\*\s*\(medium\)', f'**신뢰도: {confidence_str}**', final_answer, flags=re.IGNORECASE)
                    final_answer = re.sub(r'\*\*신뢰도:\s*[\d.]+%\*\*', f'**신뢰도: {confidence_str}**', final_answer, flags=re.IGNORECASE)

                    # 일반 패턴 (더 포괄적)
                    final_answer = re.sub(r'신뢰도:\s*[\d.]+%\s*\(low\)', f'신뢰도: {confidence_str}', final_answer, flags=re.IGNORECASE)
                    final_answer = re.sub(r'신뢰도:\s*[\d.]+%\s*\(medium\)', f'신뢰도: {confidence_str}', final_answer, flags=re.IGNORECASE)
                    final_answer = re.sub(r'신뢰도:\s*[\d.]+%\s*\(high\)', f'신뢰도: {confidence_str}', final_answer, flags=re.IGNORECASE)
                    # 가장 일반적인 패턴 (모든 숫자 패턴 매칭, 여러 번 적용)
                    for _ in range(3):  # 여러 번 적용하여 모든 인스턴스 교체
                        final_answer = re.sub(r'신뢰도:\s*[\d.]+%', f'신뢰도: {confidence_str}', final_answer, flags=re.IGNORECASE)
                        final_answer = re.sub(r'\*\*신뢰도:\s*[\d.]+%\*\*', f'**신뢰도: {confidence_str}**', final_answer, flags=re.IGNORECASE)
                        final_answer = re.sub(r'🟡\s*\*\*신뢰도:\s*[\d.]+%\*\*', f'🟡 **신뢰도: {confidence_str}**', final_answer, flags=re.IGNORECASE)

                    # % 없는 패턴
                    final_answer = re.sub(r'신뢰도:\s*[\d.]+\s*\(low\)', f'신뢰도: {confidence_str}', final_answer, flags=re.IGNORECASE)
                    final_answer = re.sub(r'신뢰도:\s*[\d.]+\s*\(medium\)', f'신뢰도: {confidence_str}', final_answer, flags=re.IGNORECASE)
                    final_answer = re.sub(r'신뢰도:\s*[\d.]+(?:\s|$|\))', f'신뢰도: {confidence_str}', final_answer, flags=re.IGNORECASE)

                    # 답변 품질 패턴
                    final_answer = re.sub(r'답변품질:\s*[\d.]+%', f'답변 품질: {confidence_str}', final_answer, flags=re.IGNORECASE)
                    final_answer = re.sub(r'답변\s*품질:\s*[\d.]+%', f'답변 품질: {confidence_str}', final_answer, flags=re.IGNORECASE)

                    # 상세점수 패턴도 교체
                    final_answer = re.sub(r'상세점수:.*?답변품질:\s*[\d.]+%', f'상세점수:\n- 답변 품질: {confidence_str}', final_answer, flags=re.IGNORECASE | re.DOTALL)

                    # "신뢰도정보" 섹션 전체 교체 시도
                    final_answer = re.sub(
                        r'###\s*💡\s*신뢰도정보.*?(?=\n###|\n---|\Z)',
                        f'### 💡 신뢰도정보\n🟡 **신뢰도: {confidence_str}** (medium)\n\n**설명:** 신뢰도: {confidence_str}',
                        final_answer,
                        flags=re.DOTALL | re.IGNORECASE
                    )

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

                # 신뢰도 값 최종 교체 (추가 안전장치 - 더 강력한 패턴 매칭)
                current_confidence = state.get("confidence", 0.0)
                if current_confidence > 0:
                    confidence_str = f"{current_confidence:.1%}"
                    # 신뢰도 레벨 결정
                    if current_confidence >= 0.8:
                        level = "high"
                        emoji = "🟢"
                    elif current_confidence >= 0.6:
                        level = "medium"
                        emoji = "🟡"
                    else:
                        level = "low"
                        emoji = "🟠"

                    # 모든 신뢰도 패턴 최종 교체 (반복 적용, 더 포괄적인 패턴)
                    for _ in range(10):  # 충분히 반복 적용
                        # 가장 일반적인 패턴 우선
                        final_answer = re.sub(r'\*\*신뢰도:\s*[\d.]+%\*\*', f'**신뢰도: {confidence_str}**', final_answer, flags=re.IGNORECASE)
                        final_answer = re.sub(r'🟡\s*\*\*신뢰도:\s*[\d.]+%\*\*', f'{emoji} **신뢰도: {confidence_str}**', final_answer, flags=re.IGNORECASE)
                        final_answer = re.sub(r'🟠\s*\*\*신뢰도:\s*[\d.]+%\*\*', f'{emoji} **신뢰도: {confidence_str}**', final_answer, flags=re.IGNORECASE)
                        final_answer = re.sub(r'🟢\s*\*\*신뢰도:\s*[\d.]+%\*\*', f'{emoji} **신뢰도: {confidence_str}**', final_answer, flags=re.IGNORECASE)
                        final_answer = re.sub(r'신뢰도:\s*[\d.]+%', f'신뢰도: {confidence_str}', final_answer, flags=re.IGNORECASE)
                        final_answer = re.sub(r'답변품질:\s*[\d.]+%', f'답변 품질: {confidence_str}', final_answer, flags=re.IGNORECASE)

                    # "신뢰도정보" 섹션 전체를 찾아서 교체 (더 강력한 방법 - 직접 섹션 찾기)
                    # 섹션 전체를 새로운 내용으로 교체
                    new_confidence_section = f'### 💡 신뢰도정보\n{emoji} **신뢰도: {confidence_str}** ({level})\n\n**상세점수:**\n- 답변 품질: {confidence_str}\n\n**설명:** 신뢰도: {confidence_str}'

                    # 더 직접적인 방법: "### 💡 신뢰도정보"로 시작하는 섹션을 직접 찾아 교체
                    lines = final_answer.split('\n')
                    new_lines = []
                    in_confidence_section = False

                    for i, line in enumerate(lines):
                        # "### 💡 신뢰도정보" 또는 "###💡신뢰도정보"로 시작하는 줄 찾기
                        if re.match(r'^###\s*💡\s*신뢰도정보', line, re.IGNORECASE):
                            in_confidence_section = True
                            new_lines.append(new_confidence_section)
                            continue

                        # 신뢰도 섹션 내부이면 건너뛰기 (다음 섹션 시작까지)
                        if in_confidence_section:
                            # "---" 또는 "💼" 또는 다음 "###" 섹션 시작까지 건너뛰기
                            if line.strip() == '---' or line.strip().startswith('💼') or re.match(r'^###\s+', line):
                                in_confidence_section = False
                                # 섹션 종료 후 이 줄은 포함
                                new_lines.append(line)
                            # 그 외는 모두 건너뛰기
                            continue

                        new_lines.append(line)

                    final_answer = '\n'.join(new_lines)

                    # 추가로 정규식으로도 시도 (fallback)
                    if '### 💡 신뢰도정보' in final_answer or '###💡신뢰도정보' in final_answer:
                        # 정규식으로 한 번 더 교체 시도
                        patterns = [
                            r'###\s*💡\s*신뢰도정보.*?(?=\n---|\n💼|\Z)',
                            r'###\s*💡\s*신뢰도정보.*?(?=\n###|\Z)',
                            r'###\s*💡\s*신뢰도정보[^\n]*\n.*?(?=\n---|\n💼|\Z)',
                        ]

                        for pattern in patterns:
                            if re.search(pattern, final_answer, flags=re.DOTALL | re.IGNORECASE):
                                final_answer = re.sub(
                                    pattern,
                                    new_confidence_section,
                                    final_answer,
                                    flags=re.DOTALL | re.IGNORECASE
                                )
                                break

                # 메타 정보 섹션 추출 및 분리 (신뢰도 섹션 교체 후)
                metadata_sections = self._extract_metadata_sections(final_answer)

                # answer에서 메타 정보 섹션 제거
                clean_answer = self._remove_metadata_sections(final_answer)

                # 중간 생성 텍스트 제거 (STEP 0, 원본 답변, 질문 정보 등)
                clean_answer = self._remove_intermediate_text(clean_answer)

                # '## 답변' 헤더 제거
                clean_answer = self._remove_answer_header(clean_answer)

                # 답변 길이 조절 (질의 유형에 맞게)
                query_type = WorkflowUtils.get_state_value(state, "query_type", "general_question")
                query_complexity = WorkflowUtils.get_state_value(state, "complexity_level", "moderate")
                clean_answer = self._adjust_answer_length(clean_answer, query_type, query_complexity)

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
                retrieved_docs = state.get("retrieved_docs", [])
                query = WorkflowUtils.get_state_value(state, "query", "")

                # 검색 결과 기반 검증 수행
                source_verification_result = AnswerValidator.validate_answer_source_verification(
                    answer=clean_answer,
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
        from core.services.confidence_calculator import ConfidenceLevel

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
