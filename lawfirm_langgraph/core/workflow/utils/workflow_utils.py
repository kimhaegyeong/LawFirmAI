# -*- coding: utf-8 -*-
"""
워크플로우 유틸리티 함수 집합
상태 관리, 파싱, 정규화 등의 공통 기능 제공
"""

import json
import logging
import re
import time
from typing import Any, Dict, List, Optional, Tuple

from core.processing.extractors.extractors import ResponseExtractor
from core.workflow.state.state_definitions import LegalWorkflowState
from core.workflow.state.state_helpers import ensure_state_group, get_field, set_field
from core.workflow.state.state_utils import (
    MAX_PROCESSING_STEPS,
    prune_processing_steps,
)
from core.classification.classifiers.question_classifier import QuestionType
from core.agents.prompt_builders.unified_prompt_manager import LegalDomain


class WorkflowUtils:
    """
    워크플로우 유틸리티 함수 집합

    정적 메서드로 구현하여 어디서든 사용 가능하도록 설계
    """

    @staticmethod
    def get_state_value(state: LegalWorkflowState, key: str, default: Any = None) -> Any:
        """
        State에서 값을 안전하게 가져오기 (flat/nested 모두 지원)

        state_helpers의 get_field를 사용하여 일관된 접근 제공
        개선: query_type 전용 검색 로직 추가

        Args:
            state: State 객체 (flat 또는 nested)
            key: 접근할 키
            default: 기본값

        Returns:
            State에서 가져온 값 또는 기본값
        """
        # query_type 전용 검색 로직 (개선: 여러 위치에서 검색)
        if key == "query_type":
            return WorkflowUtils._get_query_type_enhanced(state, default)
        
        result = get_field(state, key)
        return result if result is not None else default
    
    @staticmethod
    def _get_query_type_enhanced(state: LegalWorkflowState, default: Any = None) -> Any:
        """
        query_type을 여러 위치에서 검색 (개선된 로직)
        
        검색 순서:
        1. 최상위 레벨: state.get("query_type")
        2. classification 그룹: state["classification"]["query_type"]
        3. common.classification 그룹: state["common"]["classification"]["query_type"]
        4. metadata 그룹: state["metadata"]["query_type"]
        5. common.metadata 그룹: state["common"]["metadata"]["query_type"]
        6. Global cache: _global_classification_cache
        
        Args:
            state: LegalWorkflowState
            default: 기본값
            
        Returns:
            query_type 값 또는 기본값
        """
        import logging
        logger = logging.getLogger(__name__)
        
        # 1. 최상위 레벨에서 검색
        if "query_type" in state:
            value = state.get("query_type")
            if value:
                logger.debug(f"[QUERY_TYPE] Found in top-level: {value}")
                return value
        
        # 2. classification 그룹에서 검색
        if "classification" in state and isinstance(state["classification"], dict):
            value = state["classification"].get("query_type")
            if value:
                logger.debug(f"[QUERY_TYPE] Found in classification group: {value}")
                return value
        
        # 3. common.classification 그룹에서 검색
        if "common" in state and isinstance(state.get("common"), dict):
            common = state["common"]
            if "classification" in common and isinstance(common["classification"], dict):
                value = common["classification"].get("query_type")
                if value:
                    logger.debug(f"[QUERY_TYPE] Found in common.classification group: {value}")
                    return value
        
        # 4. metadata 그룹에서 검색
        if "metadata" in state and isinstance(state.get("metadata"), dict):
            value = state["metadata"].get("query_type")
            if value:
                logger.debug(f"[QUERY_TYPE] Found in metadata group: {value}")
                return value
        
        # 5. common.metadata 그룹에서 검색
        if "common" in state and isinstance(state.get("common"), dict):
            common = state["common"]
            if "metadata" in common and isinstance(common.get("metadata"), dict):
                value = common["metadata"].get("query_type")
                if value:
                    logger.debug(f"[QUERY_TYPE] Found in common.metadata group: {value}")
                    return value
        
        # 6. Global cache에서 검색
        try:
            from core.shared.wrappers.node_wrappers import _global_search_results_cache
            if _global_search_results_cache:
                # 여러 위치에서 검색
                cached_value = (
                    _global_search_results_cache.get("common", {}).get("classification", {}).get("query_type") or
                    _global_search_results_cache.get("metadata", {}).get("query_type") or
                    _global_search_results_cache.get("classification", {}).get("query_type") or
                    _global_search_results_cache.get("query_type")
                )
                if cached_value:
                    logger.debug(f"[QUERY_TYPE] Found in global cache: {cached_value}")
                    # state에도 복원하여 다음 노드에서 사용 가능하도록
                    WorkflowUtils.set_state_value(state, "query_type", cached_value)
                    return cached_value
        except (ImportError, AttributeError, TypeError) as e:
            logger.debug(f"[QUERY_TYPE] Could not access global cache: {e}")
        
        # 모든 위치에서 찾지 못한 경우
        logger.debug(f"[QUERY_TYPE] Not found in any location, using default: {default}")
        return default

    @staticmethod
    def set_state_value(state: LegalWorkflowState, key: str, value: Any, logger: Optional[logging.Logger] = None) -> None:
        """
        State에 값을 안전하게 설정하기 (flat/nested 모두 지원)

        state_helpers의 set_field를 사용하여 일관된 설정 제공
        필요한 State 그룹이 없으면 자동으로 초기화합니다.

        Args:
            state: State 객체 (flat 또는 nested)
            key: 설정할 키
            value: 설정할 값
            logger: 로거 (선택사항)
        """
        # 중요: metadata 전체 딕셔너리를 설정할 때 query_complexity 보존
        if key == "metadata" and isinstance(value, dict):
            # 기존 metadata에서 query_complexity와 needs_search 보존
            existing_metadata = WorkflowUtils.get_state_value(state, "metadata", {})
            if isinstance(existing_metadata, dict):
                preserved_complexity = existing_metadata.get("query_complexity")
                preserved_needs_search = existing_metadata.get("needs_search")
                # 보존된 값 복원
                if preserved_complexity:
                    value["query_complexity"] = preserved_complexity
                if preserved_needs_search is not None:
                    value["needs_search"] = preserved_needs_search

        # Classification 필드인 경우 그룹 초기화
        if key in ["query_type", "confidence", "legal_field", "legal_domain",
                   "urgency_level", "urgency_reasoning", "emergency_type",
                   "complexity_level", "requires_expert", "expert_subgraph"]:
            ensure_state_group(state, "classification")
        # Search 필드인 경우 그룹 초기화
        elif key in ["search_query", "extracted_keywords", "ai_keyword_expansion", "retrieved_docs",
                     "optimized_queries", "search_params", "semantic_results", "keyword_results",
                     "semantic_count", "keyword_count", "merged_documents", "keyword_weights",
                     "prompt_optimized_context"]:
            ensure_state_group(state, "search")
        # Analysis 필드인 경우 그룹 초기화
        elif key in ["analysis", "legal_references", "legal_citations"]:
            ensure_state_group(state, "analysis")
        # Answer 필드인 경우 그룹 초기화
        elif key in ["answer", "sources", "structure_confidence"]:
            ensure_state_group(state, "answer")
        # Document 필드인 경우 그룹 초기화
        elif key in ["document_type", "document_analysis", "key_clauses", "potential_issues"]:
            ensure_state_group(state, "document")
        # MultiTurn 필드인 경우 그룹 초기화
        elif key in ["is_multi_turn", "multi_turn_confidence", "conversation_history", "conversation_context"]:
            ensure_state_group(state, "multi_turn")
        # Validation 필드인 경우 그룹 초기화
        elif key in ["legal_validity_check", "legal_basis_validation", "outdated_laws"]:
            ensure_state_group(state, "validation")
        # Control 필드인 경우 그룹 초기화
        elif key in ["retry_count", "quality_check_passed", "needs_enhancement"]:
            ensure_state_group(state, "control")
        # Common 필드는 항상 존재
        elif key in ["processing_steps", "errors", "metadata", "processing_time", "tokens_used"]:
            ensure_state_group(state, "common")

        set_field(state, key, value)

    @staticmethod
    def update_processing_time(state: LegalWorkflowState, start_time: float) -> float:
        """처리 시간 업데이트"""
        processing_time = time.time() - start_time
        current_time = WorkflowUtils.get_state_value(state, "processing_time", 0.0)
        WorkflowUtils.set_state_value(state, "processing_time", current_time + processing_time)
        return processing_time

    @staticmethod
    def add_step(state: LegalWorkflowState, step_prefix: str, step_message: str) -> None:
        """처리 단계 추가 (중복 방지 및 pruning)"""
        processing_steps = WorkflowUtils.get_state_value(state, "processing_steps", [])
        if not processing_steps:
            processing_steps = []
            WorkflowUtils.set_state_value(state, "processing_steps", processing_steps)

        if not any(step_prefix in step for step in processing_steps):
            processing_steps.append(step_message)
            WorkflowUtils.set_state_value(state, "processing_steps", processing_steps)

        # Always prune if too many steps (check on every add)
        if len(processing_steps) > MAX_PROCESSING_STEPS:
            pruned_steps = prune_processing_steps(
                processing_steps,
                max_items=MAX_PROCESSING_STEPS
            )
            WorkflowUtils.set_state_value(state, "processing_steps", pruned_steps)

    @staticmethod
    def handle_error(state: LegalWorkflowState, error_msg: str, context: str = "",
                     logger: Optional[logging.Logger] = None) -> None:
        """에러 처리 헬퍼"""
        full_error = f"{context}: {error_msg}" if context else error_msg

        # errors 리스트 가져오기 및 초기화
        errors = WorkflowUtils.get_state_value(state, "errors", [])
        if not errors:
            errors = []
            WorkflowUtils.set_state_value(state, "errors", errors)
        errors.append(full_error)
        WorkflowUtils.set_state_value(state, "errors", errors)

        # processing_steps에 추가
        WorkflowUtils.add_step(state, "ERROR", full_error)

        if logger:
            logger.error(full_error)

    @staticmethod
    def normalize_answer(answer_raw: Any) -> str:
        """
        답변을 안전하게 문자열로 변환하고 형식 오류를 제거하는 통합 메서드 (오류 처리 강화)

        Args:
            answer_raw: 답변 (str, dict, 또는 다른 타입)

        Returns:
            정규화된 답변 문자열
        """
        try:
            if answer_raw is None:
                return ""
            if isinstance(answer_raw, str):
                answer = answer_raw
                # JSON 문자열 형식 파싱 ({"answer": "..."} 또는 {"answer": {...}})
                answer_stripped = answer.strip()
                if answer_stripped.startswith('{') and '"answer"' in answer_stripped:
                    try:
                        # 불완전한 JSON 처리 (닫는 괄호가 없는 경우)
                        if not answer_stripped.endswith('}'):
                            # 마지막 닫는 괄호 추가 시도
                            test_json = answer_stripped + '}'
                            try:
                                json_data = json.loads(test_json)
                            except:
                                # JSON 끝 부분 찾기
                                last_brace = answer_stripped.rfind('}')
                                if last_brace > 0:
                                    test_json = answer_stripped[:last_brace+1]
                                    json_data = json.loads(test_json)
                                else:
                                    raise ValueError("Invalid JSON format")
                        else:
                            json_data = json.loads(answer_stripped)
                        
                        if isinstance(json_data, dict):
                            parsed_answer = json_data.get("answer", "")
                            if isinstance(parsed_answer, dict):
                                parsed_answer = json.dumps(parsed_answer, ensure_ascii=False)
                            if parsed_answer and isinstance(parsed_answer, str) and parsed_answer.strip():
                                answer = parsed_answer
                            elif not parsed_answer:
                                # answer 키가 없거나 비어있으면 전체 JSON을 문자열로 변환
                                answer = answer_stripped
                    except (json.JSONDecodeError, ValueError):
                        # JSON 파싱 실패 시 원본 문자열에서 "answer" 키의 값 추출 시도
                        import re
                        # 멀티라인 JSON 패턴 시도
                        match = re.search(r'"answer"\s*:\s*"([^"]*(?:\\.[^"]*)*)"', answer_stripped, re.DOTALL)
                        if not match:
                            # 더 넓은 패턴 시도 (닫는 따옴표 전까지)
                            match = re.search(r'"answer"\s*:\s*"([^"]*(?:\\.[^"]*)*)"', answer_stripped)
                        if match:
                            answer = match.group(1).replace('\\n', '\n').replace('\\"', '"').replace('\\/', '/')
                        else:
                            # JSON 형식이지만 파싱 실패한 경우 원본에서 "answer" 키 제거 시도
                            # {"answer": "..."} 형식에서 "..." 부분만 추출
                            if answer_stripped.startswith('{"answer"'):
                                # 첫 번째 따옴표 이후부터 마지막 닫는 괄호 전까지 추출
                                start_idx = answer_stripped.find('"answer"') + len('"answer"')
                                start_idx = answer_stripped.find('"', start_idx) + 1
                                end_idx = answer_stripped.rfind('}')
                                if end_idx > start_idx:
                                    answer = answer_stripped[start_idx:end_idx].strip().strip('"')
                                    # 이스케이프 문자 처리
                                    answer = answer.replace('\\n', '\n').replace('\\"', '"').replace('\\/', '/')
            elif isinstance(answer_raw, dict):
                # dict에서 content나 answer 키를 찾거나, 전체 dict를 문자열로 변환
                content = answer_raw.get("content") or answer_raw.get("answer")
                if content:
                    # content가 여전히 dict일 수 있으므로 재귀적으로 처리
                    if isinstance(content, str):
                        answer = content
                    elif isinstance(content, dict):
                        answer = content.get("content", content.get("answer", str(content)))
                    else:
                        answer = str(content)
                else:
                    answer = str(answer_raw)
            elif isinstance(answer_raw, list):
                # list인 경우 첫 번째 항목 사용
                if answer_raw:
                    return WorkflowUtils.normalize_answer(answer_raw[0])
                return ""
            else:
                answer = str(answer_raw) if answer_raw else ""
            
            # 형식 오류 제거 (STEP, 평가 템플릿 등)
            answer = WorkflowUtils._remove_format_errors(answer)
            
            # 특정 사건 내용 제거 후처리
            answer = WorkflowUtils._remove_specific_case_details(answer)
            
            return answer
        except Exception as e:
            # 오류 발생 시 원본 답변 반환 (최소한의 처리)
            import logging
            logger = logging.getLogger(__name__)
            logger.warning(f"답변 정규화 중 오류 발생: {e}, 원본 답변 반환")
            if isinstance(answer_raw, str):
                return answer_raw.strip()
            return str(answer_raw) if answer_raw else ""
    
    @staticmethod
    def _remove_format_errors(answer: str) -> str:
        """
        답변에서 형식 오류 제거 (STEP, 평가 템플릿 등)
        
        Args:
            answer: 원본 답변
            
        Returns:
            정리된 답변
        """
        if not answer or not isinstance(answer, str):
            return answer
        
        # STEP 패턴 제거 (예: "STEP 0:", "## STEP 0:", "### STEP 0:" 등)
        step_patterns = [
            r'^##\s*STEP\s*\d+[:：]\s*.*?\n',
            r'^###\s*STEP\s*\d+[:：]\s*.*?\n',
            r'^STEP\s*\d+[:：]\s*.*?\n',
            r'##\s*STEP\s*\d+[:：]\s*원본\s*품질\s*평가.*?\n',
            r'##\s*STEP\s*\d+[:：]\s*.*?평가.*?\n',
            r'STEP\s*\d+[:：]',  # 줄 시작이 아닌 경우도 제거
            r'##\s*STEP\s*\d+',  # 마크다운 헤더 형식
            r'###\s*STEP\s*\d+',  # 마크다운 헤더 형식
        ]
        
        for pattern in step_patterns:
            answer = re.sub(pattern, '', answer, flags=re.MULTILINE | re.IGNORECASE)
        
        # 평가 템플릿 제거 (예: "• [ ] 법적 정보가 충분하고 정확한가?")
        evaluation_patterns = [
            r'•\s*\[[^\]]*\]\s*.*?•\s*\*\*.*?\*\*.*?\n',
            r'평가\s*결과.*?\n',
            r'원본\s*품질\s*평가.*?\n',
            r'개선\s*필요.*?\n',
            r'-?\s*\[[^\]]*\]\s*법적\s*정보.*?\n',  # 체크리스트 형식
            r'-?\s*\[[^\]]*\]\s*구조.*?\n',  # 체크리스트 형식
            r'-?\s*\[[^\]]*\]\s*어투.*?\n',  # 체크리스트 형식
            r'-?\s*\[[^\]]*\]\s*예시.*?\n',  # 체크리스트 형식
        ]
        
        for pattern in evaluation_patterns:
            answer = re.sub(pattern, '', answer, flags=re.MULTILINE | re.IGNORECASE | re.DOTALL)
        
        # 프롬프트 지시사항 제거
        prompt_instructions = [
            r'##\s*작업\s*지시.*?\n',
            r'##\s*지시사항.*?\n',
            r'다음\s*단계.*?\n',
            r'평가\s*기준.*?\n',
            r'작업\s*방법.*?\n',  # 작업 방법 섹션 제거
            r'내부\s*참고용.*?\n',  # 내부 참고용 제거
        ]
        
        for pattern in prompt_instructions:
            answer = re.sub(pattern, '', answer, flags=re.MULTILINE | re.IGNORECASE | re.DOTALL)
        
        # 특정 문구 제거 (개선: 답변 시작 부분의 불필요한 문구 제거)
        unwanted_phrases = [
            r'^주어진\s*문서를\s*바탕으로\s*답변드리면[:：]\s*\n?',
            r'^문서를\s*바탕으로\s*답변드리면[:：]\s*\n?',
            r'^주어진\s*문서를\s*바탕으로\s*답변하면[:：]\s*\n?',
            r'^문서를\s*바탕으로\s*답변하면[:：]\s*\n?',
            r'^주어진\s*문서를\s*참고하여\s*답변드리면[:：]\s*\n?',
            r'^문서를\s*참고하여\s*답변드리면[:：]\s*\n?',
        ]
        
        for pattern in unwanted_phrases:
            answer = re.sub(pattern, '', answer, flags=re.MULTILINE | re.IGNORECASE)
        
        # 연속된 빈 줄 정리
        answer = re.sub(r'\n{3,}', '\n\n', answer)
        
        return answer.strip()
    
    @staticmethod
    def _remove_specific_case_details(answer: str) -> str:
        """
        답변에서 특정 사건의 세부사항을 제거하고 법적 원칙만 유지
        
        Args:
            answer: 원본 답변
            
        Returns:
            정리된 답변
        """
        if not answer or not isinstance(answer, str):
            return answer
        
        # 답변이 너무 짧으면 후처리 건너뛰기 (최소 100자 이상일 때만 후처리)
        if len(answer) < 100:
            return answer
        
        original_length = len(answer)
        
        # 답변 시작 부분의 특정 사건 내용을 우선적으로 제거 (더 공격적으로)
        # 처음 200자에서 "[문서: ...]" 패턴을 먼저 제거
        first_200 = answer[:200] if len(answer) > 200 else answer
        if re.search(r'\[문서', first_200):
            # 처음 200자에서 "[문서: ...]" 패턴 제거 (모든 변형 포함)
            first_200_cleaned = re.sub(
                r'\[문서[:\s]*[^\]]*\]\s*\n?',
                '',
                first_200
            )
            # "[문서: ...]" 패턴이 제거된 후 빈 줄 정리
            first_200_cleaned = re.sub(r'\n{2,}', '\n', first_200_cleaned).strip()
            # 정리된 처음 200자와 나머지 부분 결합
            if len(first_200_cleaned) < len(first_200):
                answer = first_200_cleaned + answer[200:]
        
        # 처음 200자에서 "나아가"로 시작하는 문장 제거 (더 공격적으로)
        first_200 = answer[:200] if len(answer) > 200 else answer
        if re.search(r'나아가', first_200):
            # 처음 200자에서 "나아가"로 시작하는 문장 제거 (여러 문장 포함)
            first_200_cleaned = re.sub(
                r'나아가[^.]*\.',
                '',
                first_200
            )
            # "나아가"로 시작하는 문장이 여러 개일 수 있으므로 한 번 더 제거
            first_200_cleaned = re.sub(
                r'나아가[^.]*\.',
                '',
                first_200_cleaned
            )
            # "나아가" 문장이 제거된 후 빈 줄 정리
            first_200_cleaned = re.sub(r'\n{2,}', '\n', first_200_cleaned).strip()
            # 정리된 처음 200자와 나머지 부분 결합
            if len(first_200_cleaned) < len(first_200):
                answer = first_200_cleaned + answer[200:]
        
        # 처음 200자에서 "이 사건"으로 시작하는 문장 제거
        first_200 = answer[:200] if len(answer) > 200 else answer
        if re.search(r'이\s*사건', first_200):
            first_200_cleaned = re.sub(
                r'이\s*사건[^.]*\.',
                '',
                first_200
            )
            first_200_cleaned = re.sub(r'\n{2,}', '\n', first_200_cleaned).strip()
            if len(first_200_cleaned) < len(first_200):
                answer = first_200_cleaned + answer[200:]
        
        # 전체 답변에서 특정 사건번호 패턴 제거 (개선)
        case_number_patterns = [
            r'\d{4}[가나다라마바사아자차카타파하]\d+',  # 기본 사건번호 패턴
            r'\d{4}고단\d+',  # 고단 사건번호
            r'\d{4}가단\d+',  # 가단 사건번호
            r'\d{4}나단\d+',  # 나단 사건번호
        ]
        
        for pattern in case_number_patterns:
            # 사건번호가 포함된 문장 전체 제거 (더 공격적으로)
            answer = re.sub(
                r'[^.]*' + pattern + r'[^.]*\.',
                '',
                answer
            )
            # 사건번호만 제거 (문장 중간에 있는 경우)
            answer = re.sub(pattern, '', answer)
        
        # 처음 500자 내에 특정 사건번호가 있으면 제거
        first_500 = answer[:500] if len(answer) > 500 else answer
        if re.search(r'[가-힣]*지방법원[가-힣]*\s*-\s*\d{4}[가나다라마바사아자차카타파하]\d+', first_500):
            # 처음 500자에서 "[문서: ...]" 패턴 제거 (더 공격적으로)
            first_500_cleaned = re.sub(
                r'\[문서[:\s]*[^\]]*[가-힣]*지방법원[가-힣]*[^\]]*-\s*\d{4}[가나다라마바사아자차카타파하]\d+[^\]]*\]\s*\n?',
                '',
                first_500
            )
            # 처음 500자에서 특정 사건번호가 포함된 문장 제거 (더 공격적으로)
            first_500_cleaned = re.sub(
                r'[^.]*[가-힣]*지방법원[^.]*-\s*\d{4}[가나다라마바사아자차카타파하]\d+[^.]*\.',
                '',
                first_500_cleaned
            )
            # 처음 500자에서 "나아가"로 시작하는 특정 사건 사실관계 서술 제거 (더 공격적으로)
            first_500_cleaned = re.sub(
                r'나아가[^.]*이\s*사건[^.]*\.',
                '',
                first_500_cleaned
            )
            # 처음 500자에서 "나아가"로 시작하는 문장 전체 제거 (특정 사건 사실관계 서술)
            first_500_cleaned = re.sub(
                r'나아가[^.]*\.',
                '',
                first_500_cleaned
            )
            # 처음 500자에서 "이 사건"으로 시작하는 문장 제거
            first_500_cleaned = re.sub(
                r'이\s*사건[^.]*\.',
                '',
                first_500_cleaned
            )
            # 처음 500자에서 특정 당사자명이 포함된 문장 제거
            first_500_cleaned = re.sub(
                r'[^.]*피고\s+[가-힣]+[^.]*\.',
                '',
                first_500_cleaned
            )
            first_500_cleaned = re.sub(
                r'[^.]*원고\s+본인[^.]*\.',
                '',
                first_500_cleaned
            )
            # 정리된 처음 500자와 나머지 부분 결합
            answer = first_500_cleaned + answer[500:]
        
        # "[문서: ...]" 패턴 제거 (특정 사건번호 포함) - 정규식 개선
        # 더 포괄적인 패턴으로 수정 - 모든 "[문서: ...]" 패턴 제거
        answer = re.sub(
            r'\[문서[:\s]*[^\]]*\]',
            '',
            answer
        )
        # "[문서: 대전지방법원 대전지방법원-2014가단3882]" 같은 중복 패턴 제거
        answer = re.sub(
            r'\[문서[:\s]*[^\]]*[가-힣]*지방법원[가-힣]*\s+[가-힣]*지방법원[가-힣]*[^\]]*-\s*\d{4}[가나다라마바사아자차카타파하]\d+[^\]]*\]',
            '',
            answer
        )
        # "[문서: ...]" 패턴이 남아있으면 한 번 더 제거
        answer = re.sub(
            r'\[문서[:\s]*[^\]]*[가-힣]*지방법원[가-힣]*[^\]]*-\s*\d{4}[가나다라마바사아자차카타파하]\d+[^\]]*\]',
            '',
            answer
        )
        
        # 특정 사건번호 제거 (하지만 판례 인용 형식은 유지)
        # 예: "대전지방법원-2014가단43882" 제거하되 "[판례: 대전지방법원 2014가단43882]"는 유지
        answer = re.sub(
            r'(?<!\[판례[:\s])(?<!\[문서[:\s])[가-힣]*지방법원[가-힣]*\s*-\s*\d{4}[가나다라마바사아자차카타파하]\d+',
            '',
            answer
        )
        
        # 특정 사건번호가 포함된 문장 제거 (더 선택적으로 - 문장이 특정 사건에만 집중하는 경우만)
        # 단, 일반 법적 원칙을 설명하는 문장은 유지
        # 예: "이 사건 각 계약서 작성 당시..." 같은 패턴만 제거
        answer = re.sub(
            r'[^.]*이\s*사건[^.]*[가-힣]*지방법원[^.]*-\s*\d{4}[가나다라마바사아자차카타파하]\d+[^.]*\.',
            '',
            answer
        )
        
        # 특정 당사자명을 일반적인 용어로 대체 (제거하지 않고 대체)
        answer = re.sub(
            r'피고\s+[가-힣]+(?:\s+또는\s+피고\s+[가-힣]+)?',
            '당사자',
            answer
        )
        answer = re.sub(
            r'원고\s+본인',
            '당사자',
            answer
        )
        
        # 특정 사건의 사실관계 서술을 일반적인 용어로 대체
        answer = re.sub(
            r'이\s*사건\s*각\s*계약서\s*작성\s*당시',
            '계약서 작성 시',
            answer
        )
        answer = re.sub(
            r'이\s*사건\s*각\s*계약',
            '계약',
            answer
        )
        
        # 후처리 후 답변이 너무 짧아지면 원본 반환 (50% 이상 제거된 경우)
        if len(answer) < original_length * 0.5:
            return answer  # 원본 반환 대신 처리된 답변 반환 (무한 루프 방지)
        
        # 연속된 빈 줄 정리
        answer = re.sub(r'\n{3,}', '\n\n', answer)
        
        return answer.strip()

    @staticmethod
    def save_metadata_safely(state: LegalWorkflowState, key: str, value: Any,
                             save_to_top_level: bool = False) -> None:
        """
        메타데이터를 모든 경로에 안전하게 저장

        Args:
            state: LegalWorkflowState
            key: 메타데이터 키
            value: 메타데이터 값
            save_to_top_level: 최상위 레벨에도 저장할지 여부 (조건부 엣지 접근용)
        """
        # metadata 설정
        if "metadata" not in state or not isinstance(state.get("metadata"), dict):
            state["metadata"] = {}
        # 중요: query_complexity와 needs_search 보존
        preserved_complexity = state.get("metadata", {}).get("query_complexity")
        preserved_needs_search = state.get("metadata", {}).get("needs_search")
        state["metadata"] = dict(state["metadata"])  # 복사본 생성
        # 보존된 값 복원
        if preserved_complexity:
            state["metadata"]["query_complexity"] = preserved_complexity
        if preserved_needs_search is not None:
            state["metadata"]["needs_search"] = preserved_needs_search
        state["metadata"][key] = value

        # common.metadata 설정 (상태 최적화에서 항상 포함됨)
        if "common" not in state or not isinstance(state.get("common"), dict):
            state["common"] = {}
        if "metadata" not in state["common"]:
            state["common"]["metadata"] = {}
        state["common"]["metadata"] = dict(state["common"]["metadata"])  # 복사본 생성
        state["common"]["metadata"][key] = value

        # 특정 키는 top-level에도 저장 (조건부 엣지 접근용)
        if save_to_top_level and isinstance(state, dict):
            state[f"_{key}"] = value

    @staticmethod
    def get_quality_metadata(state: LegalWorkflowState) -> Dict[str, Any]:
        """
        품질 검증 메타데이터를 모든 경로에서 안전하게 읽기

        Args:
            state: LegalWorkflowState

        Returns:
            품질 메타데이터 딕셔너리 (quality_check_passed, quality_score)
        """
        quality_check_passed = None
        quality_score = None
        
        # 디버깅: state 구조 확인
        state_keys = list(state.keys()) if isinstance(state, dict) else []
        
        # 1순위: 최상위 레벨 (조건부 엣지에서 가장 확실하게 접근 가능)
        if isinstance(state, dict):
            if "_quality_check_passed" in state:
                quality_check_passed = state.get("_quality_check_passed")
            if "_quality_score" in state:
                quality_score = state.get("_quality_score")

        # 2순위: common.metadata (상태 최적화에서 항상 포함됨)
        if quality_check_passed is None or quality_score is None:
            if "common" in state and isinstance(state.get("common"), dict):
                common_meta = state["common"].get("metadata", {})
                if isinstance(common_meta, dict):
                    if quality_check_passed is None:
                        quality_check_passed = common_meta.get("quality_check_passed")
                    if quality_score is None:
                        quality_score = common_meta.get("quality_score")

        # 3순위: metadata 그룹 직접 접근
        if quality_check_passed is None or quality_score is None:
            if "metadata" in state and isinstance(state.get("metadata"), dict):
                metadata = state["metadata"]
                if quality_check_passed is None:
                    quality_check_passed = metadata.get("quality_check_passed")
                if quality_score is None:
                    quality_score = metadata.get("quality_score")

        # 4순위: 일반 경로 (get_field를 통한 접근)
        if quality_check_passed is None:
            quality_check_passed = WorkflowUtils.get_state_value(state, "quality_check_passed")
        if quality_score is None:
            quality_score = WorkflowUtils.get_state_value(state, "quality_score")

        # 기본값 설정 (None인 경우에만)
        if quality_check_passed is None:
            quality_check_passed = False
        if quality_score is None:
            quality_score = 0.0

        return {
            "quality_check_passed": bool(quality_check_passed),
            "quality_score": float(quality_score)
        }

    @staticmethod
    def extract_response_content(response: Any) -> str:
        """응답에서 내용 추출"""
        return ResponseExtractor.extract_response_content(response)

    @staticmethod
    def get_query_type_str(query_type: Any) -> str:
        """QueryType을 문자열로 변환"""
        return query_type.value if hasattr(query_type, 'value') else str(query_type)

    @staticmethod
    def normalize_query_type_for_prompt(query_type: Any, logger: Optional[logging.Logger] = None) -> str:
        """질문 유형을 프롬프트용 표준 문자열로 변환"""
        if not query_type:
            return "general_question"

        # 문자열로 변환
        if hasattr(query_type, 'value'):
            query_type_str = query_type.value
        elif hasattr(query_type, 'name'):
            query_type_str = query_type.name.lower()
        else:
            query_type_str = str(query_type).lower()

        # 표준 형태로 변환 (snake_case)
        query_type_mapping = {
            "precedent_search": "precedent_search",
            "law_inquiry": "law_inquiry",
            "legal_advice": "legal_advice",
            "document_analysis": "document_analysis",
            "procedure_guide": "procedure_guide",
            "term_explanation": "term_explanation",
            "general_question": "general_question",
            # 변형 형태 매핑
            "precedent": "precedent_search",
            "law": "law_inquiry",
            "advice": "legal_advice",
            "analysis": "document_analysis",
            "procedure": "procedure_guide",
            "term": "term_explanation",
            "general": "general_question",
        }

        # 매핑이 있으면 사용, 없으면 원본 반환 (하지만 소문자로)
        normalized = query_type_mapping.get(query_type_str, query_type_str)

        # 유효한 query_type 목록에 없으면 general_question으로
        valid_types = ["precedent_search", "law_inquiry", "legal_advice",
                      "document_analysis", "procedure_guide", "term_explanation", "general_question"]
        if normalized not in valid_types:
            if logger:
                logger.debug(f"Unknown query_type '{query_type_str}', defaulting to 'general_question'")
            normalized = "general_question"

        return normalized

    @staticmethod
    def get_domain_from_query_type(query_type: str) -> str:
        """
        질문 유형에서 도메인 추출

        현재 지원 도메인만 반환:
        - 민사법 (CIVIL_LAW)
        - 지식재산권법 (INTELLECTUAL_PROPERTY)
        - 행정법 (ADMINISTRATIVE_LAW)
        - 형사법 (CRIMINAL_LAW)

        이외는 기타/일반으로 처리
        """
        domain_mapping = {
            "precedent_search": "민사법",
            "law_inquiry": "민사법",
            "legal_advice": "민사법",
            "procedure_guide": "기타/일반",  # 절차 가이드는 기타로 처리
            "term_explanation": "기타/일반",
            "general_question": "기타/일반"
        }
        return domain_mapping.get(query_type, "기타/일반")

    @staticmethod
    def get_supported_domains() -> List[LegalDomain]:
        """현재 지원되는 도메인 목록 반환"""
        return [
            LegalDomain.CIVIL_LAW,
            LegalDomain.INTELLECTUAL_PROPERTY,
            LegalDomain.ADMINISTRATIVE_LAW,
            LegalDomain.CRIMINAL_LAW
        ]

    @staticmethod
    def is_supported_domain(domain: Optional[LegalDomain]) -> bool:
        """도메인이 지원되는지 확인"""
        if domain is None:
            return False
        return domain in WorkflowUtils.get_supported_domains()

    @staticmethod
    def get_question_type_and_domain(query_type: Any, query: str = "",
                                     logger: Optional[logging.Logger] = None) -> Tuple[QuestionType, Optional[LegalDomain]]:
        """
        질문 유형과 도메인 매핑 - LegalDomain enum 반환

        Args:
            query_type: 질문 유형 (문자열 또는 QuestionType enum)
            query: 사용자 질문 내용 (도메인 추출용, 선택사항)
            logger: 로거 (선택사항)

        현재 지원 도메인:
        - 민사법 (CIVIL_LAW)
        - 지식재산권법 (INTELLECTUAL_PROPERTY)
        - 행정법 (ADMINISTRATIVE_LAW)
        - 형사법 (CRIMINAL_LAW)

        이외의 모든 도메인은 기타(GENERAL)로 처리됩니다.
        """
        # 1. QuestionType 추출 (query_type 문자열을 enum으로 변환)
        question_type = WorkflowUtils.normalize_question_type(query_type, logger)

        # 2. 도메인 추출 (질문 내용에서 지원 도메인만 필터링)
        domain = WorkflowUtils.extract_supported_domain_from_query(query)

        # 로깅: QuestionType과 Domain 매핑 결과
        if logger:
            logger.info(
                f"📋 [QUESTION TYPE & DOMAIN] "
                f"query_type='{query_type}', "
                f"normalized_question_type={question_type.name if hasattr(question_type, 'name') else question_type}, "
                f"extracted_domain={domain.value if domain else 'None'}"
            )

        return (question_type, domain)

    @staticmethod
    def normalize_question_type(query_type: Any, logger: Optional[logging.Logger] = None) -> QuestionType:
        """query_type을 QuestionType enum으로 정규화"""
        # 이미 QuestionType enum인 경우
        if isinstance(query_type, QuestionType):
            return query_type

        # 문자열인 경우 매핑
        if isinstance(query_type, str):
            query_type_lower = query_type.lower().strip()

            # 직접 매핑
            type_mapping = {
                "precedent_search": QuestionType.PRECEDENT_SEARCH,
                "law_inquiry": QuestionType.LAW_INQUIRY,
                "legal_advice": QuestionType.LEGAL_ADVICE,
                "procedure_guide": QuestionType.PROCEDURE_GUIDE,
                "term_explanation": QuestionType.TERM_EXPLANATION,
                "general_question": QuestionType.GENERAL_QUESTION,
                "general": QuestionType.GENERAL_QUESTION,
            }

            # 직접 매핑 시도
            if query_type_lower in type_mapping:
                return type_mapping[query_type_lower]

            # QuestionType enum의 value로 찾기
            for qt in QuestionType:
                if qt.value.lower() == query_type_lower:
                    return qt

            # QuestionType enum의 name으로 찾기
            try:
                return QuestionType[query_type.upper()]
            except (KeyError, AttributeError):
                pass

        # 기본값
        if logger:
            logger.warning(f"⚠️ [QUESTION TYPE] Unknown query_type: '{query_type}', defaulting to GENERAL_QUESTION")
        return QuestionType.GENERAL_QUESTION

    @staticmethod
    def extract_supported_domain_from_query(query: str) -> Optional[LegalDomain]:
        """질문 내용에서 지원되는 도메인만 추출"""
        if not query or not isinstance(query, str):
            return LegalDomain.GENERAL

        query_lower = query.lower()

        # 지원 도메인별 키워드 매핑
        domain_keywords = {
            LegalDomain.CIVIL_LAW: [
                "민사", "계약", "손해배상", "채권", "채무", "임대차",
                "상속", "부동산", "계약서", "민법"
            ],
            LegalDomain.CRIMINAL_LAW: [
                "형사", "범죄", "처벌", "형량", "형법", "벌금",
                "징역", "교통사고", "절도", "사기", "폭행"
            ],
            LegalDomain.ADMINISTRATIVE_LAW: [
                "행정", "행정처분", "행정소송", "행정심판", "허가",
                "신고", "공무원", "행정법"
            ],
            LegalDomain.INTELLECTUAL_PROPERTY: [
                "특허", "상표", "저작권", "지적재산", "지식재산",
                "디자인", "영업비밀", "지적재산권"
            ]
        }

        # 도메인별 점수 계산
        domain_scores = {}
        for domain, keywords in domain_keywords.items():
            score = sum(1 for keyword in keywords if keyword in query_lower)
            if score > 0:
                domain_scores[domain] = score

        # 가장 높은 점수의 도메인 반환
        if domain_scores:
            return max(domain_scores.items(), key=lambda x: x[1])[0]

        # 기본값
        return LegalDomain.GENERAL

    @staticmethod
    def parse_validation_response(response: str, logger: Optional[logging.Logger] = None) -> Dict[str, Any]:
        """검증 응답 파싱"""
        try:
            # JSON 추출
            json_match = re.search(r'\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\}', response, re.DOTALL)
            if json_match:
                json_str = json_match.group(0)
                return json.loads(json_str)

            # JSON이 없으면 간단한 파싱
            result = {
                "is_valid": "invalid" not in response.lower() and "문제" not in response,
                "quality_score": 0.7 if "good" in response.lower() or "좋" in response else 0.5,
                "issues": [],
                "strengths": [],
                "recommendations": []
            }

            # 문제점 추출 시도
            if "문제" in response:
                issues_match = re.findall(r'문제[점]?\s*[:\-]\s*([^\n]+)', response)
                result["issues"] = issues_match[:5]

            return result
        except Exception as e:
            if logger:
                logger.warning(f"Failed to parse validation response: {e}")
            return {
                "is_valid": True,
                "quality_score": 0.7,
                "issues": [],
                "strengths": [],
                "recommendations": []
            }

    @staticmethod
    def parse_improvement_instructions(response: str, logger: Optional[logging.Logger] = None) -> Optional[Dict[str, Any]]:
        """개선 지시 파싱"""
        try:
            # None이면 건너뛰기
            if not response or "needs_improvement" not in response.lower():
                return None

            # JSON 추출
            json_match = re.search(r'\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\}', response, re.DOTALL)
            if json_match:
                json_str = json_match.group(0)
                result = json.loads(json_str)
                if result.get("needs_improvement", False):
                    return result

            return None
        except Exception as e:
            if logger:
                logger.warning(f"Failed to parse improvement instructions: {e}")
            return None

    @staticmethod
    def parse_final_validation_response(response: str, logger: Optional[logging.Logger] = None) -> Optional[Dict[str, Any]]:
        """최종 검증 응답 파싱"""
        try:
            # JSON 추출
            json_match = re.search(r'\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\}', response, re.DOTALL)
            if json_match:
                json_str = json_match.group(0)
                return json.loads(json_str)

            return None
        except Exception as e:
            if logger:
                logger.warning(f"Failed to parse final validation response: {e}")
            return None

    @staticmethod
    def parse_query_type_analysis_response(response: str, logger: Optional[logging.Logger] = None) -> Dict[str, Any]:
        """질문 유형 분석 응답 파싱"""
        try:
            # JSON 추출
            json_match = re.search(r'\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\}', response, re.DOTALL)
            if json_match:
                json_str = json_match.group(0)
                return json.loads(json_str)

            # 기본값
            return {
                "query_type": "simple_question",
                "confidence": 0.7,
                "reasoning": "JSON 파싱 실패"
            }
        except Exception as e:
            if logger:
                logger.warning(f"Failed to parse query type analysis response: {e}")
            return {
                "query_type": "simple_question",
                "confidence": 0.7,
                "reasoning": f"파싱 에러: {e}"
            }

    @staticmethod
    def parse_quality_validation_response(response: str, logger: Optional[logging.Logger] = None) -> Dict[str, Any]:
        """답변 품질 검증 응답 파싱"""
        try:
            # JSON 추출
            json_match = re.search(r'\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\}', response, re.DOTALL)
            if json_match:
                json_str = json_match.group(0)
                return json.loads(json_str)

            # 기본값
            return {
                "is_valid": True,
                "quality_score": 0.8,
                "issues": [],
                "needs_improvement": False
            }
        except Exception as e:
            if logger:
                logger.warning(f"Failed to parse quality validation response: {e}")
            return {
                "is_valid": True,
                "quality_score": 0.8,
                "issues": [],
                "needs_improvement": False
            }

    @staticmethod
    def get_category_mapping() -> Dict[str, List[str]]:
        """카테고리 매핑 반환"""
        return {
            "precedent_search": ["family_law", "civil_law", "criminal_law"],
            "law_inquiry": ["family_law", "civil_law", "contract_review"],
            "legal_advice": ["family_law", "civil_law", "labor_law"],
            "procedure_guide": ["civil_procedure", "family_law", "labor_law"],
            "term_explanation": ["civil_law", "family_law", "contract_review"],
            "general_question": ["civil_law", "family_law", "contract_review"]
        }
    
    @staticmethod
    def extract_chain_results(chain_history: List[Dict[str, Any]]) -> Dict[str, Any]:
        """체인 실행 결과 추출"""
        question_type_result = None
        legal_field_result = None
        complexity_result = None
        search_necessity_result = None
        
        for step in chain_history:
            step_name = step.get("step_name")
            if step.get("success"):
                if step_name == "question_type_classification":
                    question_type_result = step.get("output", {})
                elif step_name == "legal_field_extraction":
                    legal_field_result = step.get("output", {})
                elif step_name == "complexity_assessment":
                    complexity_result = step.get("output", {})
                elif step_name == "search_necessity_assessment":
                    search_necessity_result = step.get("output", {})
        
        return {
            "question_type_result": question_type_result,
            "legal_field_result": legal_field_result,
            "complexity_result": complexity_result,
            "search_necessity_result": search_necessity_result
        }
    
    @staticmethod
    def convert_chain_results(
        question_type_result: Dict[str, Any],
        complexity_result: Dict[str, Any],
        search_necessity_result: Dict[str, Any]
    ) -> Tuple[QuestionType, float, 'QueryComplexity', bool]:
        """체인 결과를 반환 형식으로 변환"""
        from core.workflow.state.workflow_types import QueryComplexity
        
        if not question_type_result or not isinstance(question_type_result, dict):
            raise ValueError("Question type classification failed")
        
        question_type_mapping = {
            "precedent_search": QuestionType.PRECEDENT_SEARCH,
            "law_inquiry": QuestionType.LAW_INQUIRY,
            "legal_advice": QuestionType.LEGAL_ADVICE,
            "procedure_guide": QuestionType.PROCEDURE_GUIDE,
            "term_explanation": QuestionType.TERM_EXPLANATION,
            "general_question": QuestionType.GENERAL_QUESTION,
        }
        question_type_str = question_type_result.get("question_type", "general_question")
        classified_type = question_type_mapping.get(question_type_str, QuestionType.GENERAL_QUESTION)
        confidence = float(question_type_result.get("confidence", 0.85))
        
        if complexity_result and isinstance(complexity_result, dict):
            complexity_str = complexity_result.get("complexity", "moderate")
        else:
            complexity_str = "moderate"
        
        complexity_mapping = {
            "simple": QueryComplexity.SIMPLE,
            "moderate": QueryComplexity.MODERATE,
            "complex": QueryComplexity.COMPLEX,
        }
        complexity = complexity_mapping.get(complexity_str, QueryComplexity.MODERATE)
        
        if search_necessity_result and isinstance(search_necessity_result, dict):
            needs_search = search_necessity_result.get("needs_search", True)
        else:
            needs_search = complexity != QueryComplexity.SIMPLE
        
        return (classified_type, confidence, complexity, needs_search)