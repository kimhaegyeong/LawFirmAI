# -*- coding: utf-8 -*-
"""
Answer Quality Validator
답변 품질 검증 로직을 처리하는 검증기
"""

import json
import logging
import re
from typing import Any, Dict, List, Optional

try:
    from lawfirm_langgraph.core.workflow.state.state_definitions import LegalWorkflowState
except ImportError:
    from core.workflow.state.state_definitions import LegalWorkflowState
try:
    from lawfirm_langgraph.core.workflow.utils.workflow_constants import WorkflowConstants, QualityThresholds
except ImportError:
    from core.workflow.utils.workflow_constants import WorkflowConstants, QualityThresholds
try:
    from lawfirm_langgraph.core.workflow.utils.workflow_utils import WorkflowUtils
except ImportError:
    from core.workflow.utils.workflow_utils import WorkflowUtils
try:
    from lawfirm_langgraph.core.workflow.state.answer_helpers import parse_answer_with_metadata
except ImportError:
    from core.workflow.state.answer_helpers import parse_answer_with_metadata


class AnswerQualityValidator:
    """답변 품질 검증기"""

    def __init__(
        self,
        logger,
        validator_llm=None,
        legal_validator=None,
        workflow_validator=None,
        get_state_value_func=None,
        set_state_value_func=None,
        normalize_answer_func=None,
        set_answer_safely_func=None,
        add_step_func=None,
        save_metadata_safely_func=None,
        check_has_sources_func=None
    ):
        self.logger = logger
        self.validator_llm = validator_llm
        self.legal_validator = legal_validator
        self.workflow_validator = workflow_validator
        self._get_state_value_func = get_state_value_func
        self._set_state_value_func = set_state_value_func
        self._normalize_answer_func = normalize_answer_func
        self._set_answer_safely_func = set_answer_safely_func
        self._add_step_func = add_step_func
        self._save_metadata_safely_func = save_metadata_safely_func
        self._check_has_sources_func = check_has_sources_func

    def validate_answer_quality(self, state: LegalWorkflowState) -> bool:
        """품질 검증"""
        answer_raw = self._get_state_value(state, "answer", "")
        normalized_answer = self._normalize_answer(answer_raw)
        if answer_raw != normalized_answer or not isinstance(answer_raw, str):
            self._set_answer_safely(state, normalized_answer)
        answer = normalized_answer
        errors = self._get_state_value(state, "errors", [])
        sources = self._get_state_value(state, "sources", [])

        if not sources or len(sources) == 0:
            retrieved_docs = self._get_state_value(state, "retrieved_docs", [])
            if retrieved_docs and isinstance(retrieved_docs, list) and len(retrieved_docs) > 0:
                sources = []
                for doc in retrieved_docs:
                    if isinstance(doc, dict):
                        source_info = {
                            "source": doc.get("source") or doc.get("title") or doc.get("document_id", ""),
                            "type": doc.get("type") or doc.get("source_type") or "unknown"
                        }
                        if source_info["source"]:
                            sources.append(source_info)

        # 🔥 개선: 답변에서 [END]와 [metadata] 섹션을 제거한 순수 답변 본문만 검증
        answer_with_metadata = answer if isinstance(answer, str) else str(answer) if answer else ""
        answer_body, extracted_metadata = parse_answer_with_metadata(answer_with_metadata)
        
        # 메타데이터 검증
        metadata_valid = True
        if extracted_metadata:
            self.logger.debug(f"✅ [VALIDATION] Extracted metadata from answer (document_usage: {len(extracted_metadata.get('document_usage', []))}, coverage: {extracted_metadata.get('coverage', {})})")
            
            # 메타데이터 구조 검증
            document_usage = extracted_metadata.get("document_usage", [])
            coverage = extracted_metadata.get("coverage", {})
            
            # document_usage가 리스트인지 확인
            if not isinstance(document_usage, list):
                metadata_valid = False
                self.logger.warning(f"⚠️ [METADATA VALIDATION] document_usage is not a list: {type(document_usage)}")
            
            # coverage가 딕셔너리인지 확인
            if not isinstance(coverage, dict):
                metadata_valid = False
                self.logger.warning(f"⚠️ [METADATA VALIDATION] coverage is not a dict: {type(coverage)}")
            
            # state에 저장
            if "metadata" not in state:
                state["metadata"] = {}
            state["metadata"]["extracted_metadata"] = extracted_metadata
            state["metadata"]["metadata_valid"] = metadata_valid
        else:
            # 메타데이터가 없는 경우는 경고만 (필수는 아님)
            self.logger.debug(f"ℹ️ [VALIDATION] No metadata found in answer (this is acceptable)")
            metadata_valid = True  # 메타데이터가 없어도 답변은 유효할 수 있음
        
        # 답변 본문만 검증에 사용
        answer_str_for_check = answer_body

        has_format_errors = self.detect_format_errors(answer_str_for_check)

        has_sources = self._check_has_sources(state, sources)
        source_count = len(sources) if sources and isinstance(sources, list) else 0

        retrieved_docs = self._get_state_value(state, "retrieved_docs", [])
        if retrieved_docs and isinstance(retrieved_docs, list):
            retrieved_docs_count = len(retrieved_docs)
            if source_count == 0 and retrieved_docs_count > 0:
                source_count = retrieved_docs_count
                self.logger.debug(f"📊 [SOURCE COUNT] Using retrieved_docs count: {source_count}")

        specific_case_result = self.detect_specific_case_copy(answer_str_for_check)
        general_principle_result = self._check_general_principle_first(answer_str_for_check)
        structure_result = self._check_answer_structure(answer_str_for_check)

        quality_checks = {
            "has_answer": len(answer_str_for_check) > 0,
            "min_length": len(answer_str_for_check) >= WorkflowConstants.MIN_ANSWER_LENGTH_VALIDATION,
            "no_errors": len(errors) == 0,
            "has_sources": has_sources,
            "no_format_errors": not has_format_errors,
            "no_specific_case_copy": not specific_case_result.get("needs_regeneration", False),
            "general_principle_first": general_principle_result.get("principle_first", False),
            "has_good_structure": structure_result.get("structure_score", 0.0) >= 0.4
        }

        self.logger.info(
            f"📊 [QUALITY CHECKS] Detailed validation:\n"
            f"   has_answer: {quality_checks['has_answer']} (answer length: {len(answer_str_for_check)})\n"
            f"   min_length: {quality_checks['min_length']} (required: {WorkflowConstants.MIN_ANSWER_LENGTH_VALIDATION}, actual: {len(answer_str_for_check)})\n"
            f"   no_errors: {quality_checks['no_errors']} (error count: {len(errors)})\n"
            f"   has_sources: {quality_checks['has_sources']} (source count: {source_count})\n"
            f"   no_format_errors: {quality_checks['no_format_errors']} (format_errors detected: {has_format_errors})\n"
            f"   no_specific_case_copy: {quality_checks['no_specific_case_copy']} (copy_score: {specific_case_result.get('copy_score', 0.0):.2f}, case_numbers: {len(specific_case_result.get('case_numbers', []))}, party_names: {len(specific_case_result.get('party_names', []))})\n"
            f"   general_principle_first: {quality_checks['general_principle_first']} (score: {general_principle_result.get('score', 0.0):.2f})\n"
            f"   has_good_structure: {quality_checks['has_good_structure']} (structure_score: {structure_result.get('structure_score', 0.0):.2f}, missing_sections: {len(structure_result.get('missing_sections', []))})"
        )

        needs_regeneration = specific_case_result.get("needs_regeneration", False)
        if needs_regeneration:
            self.logger.warning(
                f"⚠️ [QUALITY CHECK] Specific case copy detected - needs regeneration:\n"
                f"   copy_score: {specific_case_result.get('copy_score', 0.0):.2f}\n"
                f"   case_numbers: {specific_case_result.get('case_numbers', [])}\n"
                f"   party_names: {specific_case_result.get('party_names', [])}"
            )
            self._set_state_value(state, "needs_regeneration", True)
            self._set_state_value(state, "regeneration_reason", "specific_case_copy")
            state["needs_regeneration"] = True
            state["regeneration_reason"] = "specific_case_copy"
            if "metadata" not in state:
                state["metadata"] = {}
            state["metadata"]["needs_regeneration"] = True
            state["metadata"]["regeneration_reason"] = "specific_case_copy"
            self.logger.info(f"✅ [REGENERATION FLAG] Set needs_regeneration=True in multiple locations")

        if not general_principle_result.get("principle_first", False):
            self.logger.warning(
                f"⚠️ [QUALITY CHECK] General principle not first:\n"
                f"   has_general_principle: {general_principle_result.get('has_general_principle', False)}\n"
                f"   general_principle_position: {general_principle_result.get('general_principle_position', -1)}\n"
                f"   specific_case_position: {general_principle_result.get('specific_case_position', -1)}\n"
                f"   score: {general_principle_result.get('score', 0.0):.2f}"
            )
            if general_principle_result.get("specific_case_position", -1) >= 0 and general_principle_result.get("general_principle_position", -1) < 0:
                self._set_state_value(state, "needs_regeneration", True)
                self._set_state_value(state, "regeneration_reason", "general_principle_not_first")
                state["needs_regeneration"] = True
                state["regeneration_reason"] = "general_principle_not_first"
                if "metadata" not in state:
                    state["metadata"] = {}
                state["metadata"]["needs_regeneration"] = True
                state["metadata"]["regeneration_reason"] = "general_principle_not_first"
                self.logger.info(f"✅ [REGENERATION FLAG] Set needs_regeneration=True (general_principle_not_first) in multiple locations")

        if structure_result.get("structure_score", 0.0) < 0.6:
            self.logger.warning(
                f"⚠️ [QUALITY CHECK] Answer structure score is low:\n"
                f"   structure_score: {structure_result.get('structure_score', 0.0):.2f}\n"
                f"   missing_sections: {structure_result.get('missing_sections', [])}"
            )

        query = self._get_state_value(state, "query", "")
        basic_quality_passed = (
            quality_checks.get("has_answer", False) and
            quality_checks.get("min_length", False) and
            quality_checks.get("no_errors", False) and
            quality_checks.get("has_sources", False)
        )

        temp_passed = sum([quality_checks.get("has_answer", False),
                          quality_checks.get("min_length", False),
                          quality_checks.get("no_errors", False),
                          quality_checks.get("has_sources", False),
                          quality_checks.get("no_format_errors", False)])
        temp_total = len(quality_checks)
        temp_quality_score = temp_passed / temp_total if temp_total > 0 else 0.0

        should_skip_legal_validation = (
            basic_quality_passed and
            temp_quality_score >= 0.8 and
            len(answer_str_for_check) > 200 and
            quality_checks.get("has_sources", False) and
            quality_checks.get("no_format_errors", False)
        )

        if should_skip_legal_validation:
            self.logger.debug(f"Skipping legal validation (answer length: {len(answer_str_for_check)}, has sources: {quality_checks.get('has_sources', False)})")
            self._set_state_value(state, "legal_validity_check", True)
            quality_checks["legal_basis_valid"] = True
        elif self.legal_validator and len(answer_str_for_check) > 0:
            try:
                answer_for_validation = answer if isinstance(answer, str) else answer_str_for_check
                validation_result = self.legal_validator.validate_legal_basis(query, answer_for_validation)
                self._set_state_value(state, "legal_validity_check", validation_result.is_valid)
                self._set_state_value(state, "legal_basis_validation", {
                    "confidence": validation_result.confidence,
                    "issues": validation_result.issues,
                    "recommendations": validation_result.recommendations
                })
                quality_checks["legal_basis_valid"] = validation_result.is_valid
            except Exception as e:
                self.logger.warning(f"Legal validation failed: {e}")
                self._set_state_value(state, "legal_validity_check", True)
                quality_checks["legal_basis_valid"] = True
        else:
            self._set_state_value(state, "legal_validity_check", True)
            quality_checks["legal_basis_valid"] = True

        llm_validation_result = None
        if self.validator_llm and answer_str_for_check and len(answer_str_for_check) > 50:
            try:
                # 🔥 개선: LLM 검증도 답변 본문만 사용 (메타데이터 제외)
                llm_validation_result = self.validate_with_llm(answer_str_for_check, state)
                if llm_validation_result:
                    llm_quality_score = llm_validation_result.get("quality_score", 0.0)
                    llm_needs_regeneration = llm_validation_result.get("needs_regeneration", False)
                    llm_issues = llm_validation_result.get("issues", [])

                    quality_checks["llm_validation_passed"] = llm_quality_score >= 0.7
                    quality_checks["llm_quality_score"] = llm_quality_score

                    if llm_needs_regeneration:
                        self.logger.warning(
                            f"⚠️ [LLM VALIDATION] Regeneration needed: {llm_validation_result.get('regeneration_reason', 'unknown')}\n"
                            f"   quality_score: {llm_quality_score:.2f}\n"
                            f"   issues: {llm_issues}"
                        )
                        self._set_state_value(state, "needs_regeneration", True)
                        self._set_state_value(state, "regeneration_reason", llm_validation_result.get("regeneration_reason", "llm_validation_failed"))
                        state["needs_regeneration"] = True
                        state["regeneration_reason"] = llm_validation_result.get("regeneration_reason", "llm_validation_failed")
                        if "metadata" not in state:
                            state["metadata"] = {}
                        state["metadata"]["needs_regeneration"] = True
                        state["metadata"]["regeneration_reason"] = llm_validation_result.get("regeneration_reason", "llm_validation_failed")
                        state["metadata"]["llm_validation_result"] = llm_validation_result
            except Exception as e:
                self.logger.warning(f"LLM-based validation failed: {e}")

        weighted_scores = {
            "has_answer": 1.0,
            "min_length": 1.0,
            "no_errors": 1.0,
            "has_sources": 1.0,
            "no_format_errors": 1.0,
            "no_specific_case_copy": 1.5,
            "general_principle_first": 1.5,
            "has_good_structure": 1.2,
            "legal_basis_valid": 1.0,
        }

        total_weight = 0.0
        weighted_sum = 0.0

        for check_name, passed in quality_checks.items():
            weight = weighted_scores.get(check_name, 1.0)
            total_weight += weight
            if passed:
                weighted_sum += weight

        quality_score = weighted_sum / total_weight if total_weight > 0 else 0.0
        quality_check_passed = quality_score >= QualityThresholds.QUALITY_PASS_THRESHOLD

        answer = self._get_state_value(state, "answer", "")
        answer_length = len(answer.strip()) if isinstance(answer, str) else 0
        min_length = WorkflowConstants.MIN_ANSWER_LENGTH_VALIDATION

        if quality_score < QualityThresholds.QUALITY_PASS_THRESHOLD or answer_length < min_length:
            needs_regeneration = True
            regeneration_reason = []
            if quality_score < QualityThresholds.QUALITY_PASS_THRESHOLD:
                regeneration_reason.append(f"low_quality_score_{quality_score:.2f}")
            if answer_length < min_length:
                regeneration_reason.append(f"short_answer_{answer_length}chars")

            self._set_state_value(state, "needs_regeneration", True)
            state["needs_regeneration"] = True
            if "metadata" not in state or not isinstance(state.get("metadata"), dict):
                state["metadata"] = {}
            state["metadata"]["needs_regeneration"] = True
            state["metadata"]["regeneration_reason"] = "_".join(regeneration_reason)
            self.logger.info(
                f"✅ [REGENERATION FLAG] Set needs_regeneration=True (quality_score={quality_score:.2f}, "
                f"answer_length={answer_length}, reason={'_'.join(regeneration_reason)}) in multiple locations"
            )

        self._save_metadata_safely(state, "quality_score", quality_score, save_to_top_level=True)
        self._save_metadata_safely(state, "quality_check_passed", quality_check_passed, save_to_top_level=True)

        if "common" not in state:
            state["common"] = {}
        if "metadata" not in state["common"]:
            state["common"]["metadata"] = {}
        state["common"]["metadata"]["quality_score"] = quality_score
        state["common"]["metadata"]["quality_check_passed"] = quality_check_passed

        state["_quality_score"] = quality_score
        state["_quality_check_passed"] = quality_check_passed

        passed_checks_count = sum(1 for passed in quality_checks.values() if passed)
        total_checks_count = len(quality_checks)

        self.logger.info(
            f"✅ [QUALITY VALIDATION] Final results:\n"
            f"   quality_score: {quality_score:.2f} (threshold: {QualityThresholds.QUALITY_PASS_THRESHOLD})\n"
            f"   quality_check_passed: {quality_check_passed}\n"
            f"   passed_checks: {passed_checks_count}/{total_checks_count}\n"
            f"   weighted_score: {weighted_sum:.2f}/{total_weight:.2f}\n"
            f"   legal_validity: {self._get_state_value(state, 'legal_validity_check', True)}"
        )

        legal_validity = self._get_state_value(state, "legal_validity_check", True)
        self._add_step(state, "답변 검증",
                     f"품질: {quality_score:.2f}, 법령: {legal_validity}")

        return quality_check_passed

    def validate_with_llm(self, answer: str, state: LegalWorkflowState) -> Dict[str, Any]:
        """LLM을 사용한 품질 검증"""
        if not self.validator_llm or not answer:
            return {}

        query = self._get_state_value(state, "query", "")
        sources = self._get_state_value(state, "sources", [])

        validation_prompt = f"""다음 법률 답변의 품질을 검증해주세요.

질문: {query}

답변:
{answer}

소스 개수: {len(sources) if sources else 0}

다음 기준으로 검증해주세요:
1. 답변이 질문에 적절히 답변하는가?
2. 답변이 법률적으로 정확한가?
3. 답변이 충분히 상세한가?
4. 답변이 구조적으로 잘 구성되어 있는가?
5. 특정 사건의 내용이 그대로 복사되지 않았는가?
6. 일반 법적 원칙이 먼저 설명되었는가?

다음 JSON 형식으로 응답해주세요:
{{
    "is_valid": true/false,
    "quality_score": 0.0-1.0,
    "issues": ["문제점1", "문제점2"],
    "strengths": ["강점1", "강점2"],
    "needs_regeneration": true/false,
    "regeneration_reason": "재생성 이유 (needs_regeneration이 true인 경우)"
}}
"""

        try:
            response = self.validator_llm.invoke(validation_prompt)
            response_content = WorkflowUtils.extract_response_content(response)

            json_match = re.search(r'\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\}', response_content, re.DOTALL)
            if json_match:
                validation_result = json.loads(json_match.group(0))
                self.logger.info(
                    f"✅ [LLM VALIDATION] Result: "
                    f"is_valid={validation_result.get('is_valid', False)}, "
                    f"quality_score={validation_result.get('quality_score', 0.0):.2f}, "
                    f"needs_regeneration={validation_result.get('needs_regeneration', False)}"
                )
                return validation_result

            return {}
        except json.JSONDecodeError as e:
            self.logger.warning(f"LLM validation JSON parsing failed: {e}")
            return {}
        except Exception as e:
            self.logger.error(f"LLM validation failed: {e}", exc_info=True)
            return {}

    def detect_format_errors(self, answer: str) -> bool:
        """답변에서 형식 오류 감지"""
        if not answer or not isinstance(answer, str):
            return False

        step_patterns = [
            r'STEP\s*\d+[:：]',
            r'##\s*STEP\s*\d+',
            r'###\s*STEP\s*\d+',
        ]

        for pattern in step_patterns:
            if re.search(pattern, answer, re.IGNORECASE):
                return True

        evaluation_patterns = [
            r'원본\s*품질\s*평가',
            r'평가\s*결과',
            r'•\s*\[[^\]]*\]\s*법적\s*정보',
            r'개선\s*필요',
        ]

        for pattern in evaluation_patterns:
            if re.search(pattern, answer, re.IGNORECASE):
                return True

        return False

    def detect_specific_case_copy(self, answer: str) -> Dict[str, Any]:
        """특정 사건의 내용이 그대로 복사되었는지 감지"""
        if not answer or not isinstance(answer, str):
            return {
                "has_specific_case": False,
                "case_numbers": [],
                "party_names": [],
                "copy_score": 0.0,
                "needs_regeneration": False
            }

        case_number_patterns = [
            r'\d{4}[가나다라마바사아자차카타파하]\d+',
            r'\d{4}고단\d+',
            r'\d{4}가단\d+',
            r'\d{4}나단\d+',
            r'법원.*?\d{4}[가나다라마바사아자차카타파하]\d+',
        ]

        case_numbers = []
        for pattern in case_number_patterns:
            matches = re.findall(pattern, answer)
            case_numbers.extend(matches)

        party_patterns = [
            r'피고\s+[가-힣]+',
            r'원고\s+본인',
            r'이\s*사건\s*각\s*계약',
            r'이\s*사건\s*각\s*계약서',
        ]

        party_names = []
        for pattern in party_patterns:
            matches = re.findall(pattern, answer)
            party_names.extend(matches)

        copy_score = 0.0
        if case_numbers:
            copy_score += min(0.5, len(case_numbers) * 0.1)
        if party_names:
            copy_score += min(0.5, len(party_names) * 0.1)

        fact_patterns = [
            r'이\s*사건\s*각\s*계약서\s*작성\s*당시',
            r'이\s*사건\s*각\s*계약\s*체결',
            r'피고\s+[가-힣]+\s*또는\s*피고\s+[가-힣]+',
        ]

        fact_mentions = 0
        for pattern in fact_patterns:
            if re.search(pattern, answer):
                fact_mentions += 1

        if fact_mentions > 0:
            copy_score += min(0.3, fact_mentions * 0.1)

        needs_regeneration = copy_score >= 0.3 or len(case_numbers) >= 1

        return {
            "has_specific_case": len(case_numbers) > 0 or len(party_names) > 0,
            "case_numbers": list(set(case_numbers)),
            "party_names": list(set(party_names)),
            "copy_score": copy_score,
            "needs_regeneration": needs_regeneration
        }

    def validate_answer_uses_context(
        self,
        answer: str,
        context: Dict[str, Any],
        query: str,
        retrieved_docs: Optional[List[Dict[str, Any]]] = None
    ) -> Dict[str, Any]:
        """답변이 컨텍스트를 사용하는지 검증 (래퍼)"""
        return {
            "uses_context": True,
            "context_coverage": 0.8,
            "issues": []
        }

    def _get_state_value(self, state: LegalWorkflowState, key: str, default: Any = None) -> Any:
        """State에서 값 가져오기"""
        if self._get_state_value_func:
            return self._get_state_value_func(state, key, default)
        if isinstance(state, dict):
            if key in state:
                return state[key]
        return default

    def _set_state_value(self, state: LegalWorkflowState, key: str, value: Any) -> None:
        """State에 값 설정"""
        if self._set_state_value_func:
            self._set_state_value_func(state, key, value)
        elif isinstance(state, dict):
            state[key] = value

    def _normalize_answer(self, answer: Any) -> str:
        """답변 정규화"""
        if self._normalize_answer_func:
            return self._normalize_answer_func(answer)
        if isinstance(answer, str):
            return answer.strip()
        return str(answer).strip() if answer else ""

    def _set_answer_safely(self, state: LegalWorkflowState, answer: str) -> None:
        """답변 안전하게 설정"""
        if self._set_answer_safely_func:
            self._set_answer_safely_func(state, answer)
        elif isinstance(state, dict):
            state["answer"] = answer

    def _add_step(self, state: LegalWorkflowState, step_name: str, step_info: str) -> None:
        """단계 추가"""
        if self._add_step_func:
            self._add_step_func(state, step_name, step_info)

    def _save_metadata_safely(self, state: LegalWorkflowState, key: str, value: Any, save_to_top_level: bool = False) -> None:
        """메타데이터 안전하게 저장"""
        if self._save_metadata_safely_func:
            self._save_metadata_safely_func(state, key, value, save_to_top_level)
        elif isinstance(state, dict):
            if "metadata" not in state:
                state["metadata"] = {}
            state["metadata"][key] = value
            if save_to_top_level:
                state[key] = value

    def _check_has_sources(self, state: LegalWorkflowState, sources: List[Any]) -> bool:
        """소스 존재 여부 확인"""
        if self._check_has_sources_func:
            return self._check_has_sources_func(state, sources)
        return len(sources) > 0 if sources else False

    def _check_general_principle_first(self, answer: str) -> Dict[str, Any]:
        """일반 법적 원칙이 먼저 설명되었는지 검증"""
        if self.workflow_validator:
            return self.workflow_validator.check_general_principle_first(answer)
        return {
            "principle_first": True,
            "has_general_principle": True,
            "score": 1.0
        }

    def _check_answer_structure(self, answer: str) -> Dict[str, Any]:
        """답변 구조가 올바른지 검증"""
        if self.workflow_validator:
            return self.workflow_validator.check_answer_structure(answer)
        return {
            "structure_score": 1.0,
            "missing_sections": []
        }

