# -*- coding: utf-8 -*-
"""
State Utils Mixin
State 관련 유틸리티 메서드들을 제공하는 Mixin 클래스
"""

from typing import Any, Dict, List, Tuple

from core.workflow.state.state_definitions import LegalWorkflowState
from core.workflow.utils.workflow_utils import WorkflowUtils


class StateUtilsMixin:
    """State 관련 유틸리티 메서드들을 제공하는 Mixin 클래스"""
    
    def _get_state_value(self, state: LegalWorkflowState, key: str, default: Any = None) -> Any:
        """WorkflowUtils.get_state_value 래퍼"""
        return WorkflowUtils.get_state_value(state, key, default)

    def _set_state_value(self, state: LegalWorkflowState, key: str, value: Any) -> None:
        """WorkflowUtils.set_state_value 래퍼"""
        WorkflowUtils.set_state_value(state, key, value, self.logger)

    def _update_processing_time(self, state: LegalWorkflowState, start_time: float):
        """WorkflowUtils.update_processing_time 래퍼"""
        return WorkflowUtils.update_processing_time(state, start_time)

    def _add_step(self, state: LegalWorkflowState, step_prefix: str, step_message: str):
        """WorkflowUtils.add_step 래퍼"""
        WorkflowUtils.add_step(state, step_prefix, step_message)

    def _handle_error(self, state: LegalWorkflowState, error_msg: str, context: str = ""):
        """WorkflowUtils.handle_error 래퍼"""
        WorkflowUtils.handle_error(state, error_msg, context, self.logger)

    def _normalize_answer(self, answer_raw: Any) -> str:
        """WorkflowUtils.normalize_answer 래퍼"""
        return WorkflowUtils.normalize_answer(answer_raw)

    def _ensure_input_group(self, state: LegalWorkflowState) -> None:
        """input 그룹 보장 (중복 코드 제거)"""
        if "input" not in state or not isinstance(state.get("input"), dict):
            state["input"] = {}
    
    def _restore_query_from_state(self, state: LegalWorkflowState) -> Tuple[str, str]:
        """state에서 query와 session_id 복원 (중복 코드 제거)"""
        query_value = None
        session_id_value = None
        
        if "input" in state and isinstance(state.get("input"), dict):
            query_value = state["input"].get("query", "")
            session_id_value = state["input"].get("session_id", "")
        
        if not query_value:
            query_value = state.get("query", "")
            session_id_value = state.get("session_id", "")
        
        if not query_value and "search" in state and isinstance(state.get("search"), dict):
            query_value = state["search"].get("search_query", "")
        
        return query_value or "", session_id_value or ""
    
    def _normalize_query_encoding(self, query: Any) -> str:
        """쿼리 인코딩 정규화 (중복 코드 제거)"""
        if not query:
            return ""
        
        try:
            if isinstance(query, str):
                return query.encode('utf-8', errors='replace').decode('utf-8')
            elif isinstance(query, bytes):
                return query.decode('utf-8', errors='replace')
            else:
                return str(query)
        except Exception:
            return str(query) if query else ""
    
    def _preserve_metadata(self, state: LegalWorkflowState, keys: List[str]) -> Dict[str, Any]:
        """metadata 보존 (중복 코드 제거)"""
        preserved = {}
        metadata = state.get("metadata", {})
        if isinstance(metadata, dict):
            for key in keys:
                if key in metadata:
                    preserved[key] = metadata[key]
        return preserved
    
    def _set_answer_safely(self, state: LegalWorkflowState, answer: Any) -> None:
        """
        Answer를 안전하게 저장하는 헬퍼 메서드

        - normalize_answer를 자동으로 호출하여 문자열 보장
        - 타입 검증 및 로깅 추가
        - 모든 노드에서 일관된 answer 저장 보장

        Args:
            state: LegalWorkflowState
            answer: 저장할 answer 값 (str, dict, 또는 다른 타입)
        """
        # 타입 확인 및 정규화
        original_type = type(answer).__name__
        normalized_answer = self._normalize_answer(answer)
        final_type = type(normalized_answer).__name__

        # 타입 변환이 발생한 경우 로깅
        if original_type != final_type or original_type != 'str':
            self.logger.debug(
                f"[ANSWER TYPE] Type conversion: {original_type} → {final_type} "
                f"(length: {len(normalized_answer)})"
            )

        # answer 저장
        self._set_state_value(state, "answer", normalized_answer)

        # answer 길이 로깅 (너무 짧은 경우 경고)
        if len(normalized_answer) < 10:
            self.logger.warning(
                f"[ANSWER WARNING] Answer is very short (length: {len(normalized_answer)})"
            )

    def _save_metadata_safely(self, state: LegalWorkflowState, key: str, value: Any,
                             save_to_top_level: bool = False) -> None:
        """WorkflowUtils.save_metadata_safely 래퍼"""
        WorkflowUtils.save_metadata_safely(state, key, value, save_to_top_level)

    def _get_quality_metadata(self, state: LegalWorkflowState) -> Dict[str, Any]:
        """WorkflowUtils.get_quality_metadata 래퍼"""
        return WorkflowUtils.get_quality_metadata(state)

    def _get_category_mapping(self) -> Dict[str, List[str]]:
        """WorkflowUtils.get_category_mapping 래퍼"""
        return WorkflowUtils.get_category_mapping()
    
    def _preserve_metadata_and_common_state(self, state: LegalWorkflowState) -> None:
        """metadata와 common state 보존 (중복 코드 제거)"""
        query_value = self._get_state_value(state, "query", "")
        session_id_value = self._get_state_value(state, "session_id", "")
        if "input" not in state:
            state["input"] = {}
        state["input"]["query"] = query_value or state.get("input", {}).get("query", "")
        state["input"]["session_id"] = session_id_value or state.get("input", {}).get("session_id", "")

