# -*- coding: utf-8 -*-
"""
Workflow Types and Utility Classes
워크플로우 타입 및 유틸리티 클래스 정의
"""

from enum import Enum
from typing import Any, Dict

try:
    from lawfirm_langgraph.core.workflow.state.state_definitions import LegalWorkflowState
except ImportError:
    from core.workflow.state.state_definitions import LegalWorkflowState

try:
    from lawfirm_langgraph.core.workflow.utils.workflow_constants import RetryConfig
except ImportError:
    from core.workflow.utils.workflow_constants import RetryConfig


class QueryComplexity(str, Enum):
    """질문 복잡도"""
    SIMPLE = "simple"           # 검색 불필요 (일반 상식, 정의)
    MODERATE = "moderate"       # 단일 검색 필요
    COMPLEX = "complex"         # 다중 검색 필요
    MULTI_HOP = "multi_hop"     # 추론 체인 필요


class RetryCounterManager:
    """재시도 카운터를 안전하게 관리하는 클래스"""

    def __init__(self, logger):
        self.logger = logger

    def get_retry_counts(self, state: LegalWorkflowState) -> Dict[str, int]:
        """
        모든 경로에서 재시도 카운터 안전하게 읽기

        Args:
            state: LegalWorkflowState

        Returns:
            재시도 카운터 딕셔너리 (generation, validation, total)
        """
        generation_retry = 0
        validation_retry = 0

        # 1순위: common.metadata (상태 최적화에서 항상 포함됨)
        if "common" in state and isinstance(state.get("common"), dict):
            common_meta = state["common"].get("metadata", {})
            if isinstance(common_meta, dict):
                generation_retry = max(generation_retry, common_meta.get("generation_retry_count", 0))
                validation_retry = max(validation_retry, common_meta.get("validation_retry_count", 0))

        # 2순위: 최상위 레벨 retry_count
        top_level_retry = state.get("retry_count", 0)
        generation_retry = max(generation_retry, top_level_retry)

        # 3순위: 최상위 레벨 _generation_retry_count, _validation_retry_count
        if isinstance(state, dict):
            generation_retry = max(generation_retry, state.get("_generation_retry_count", 0))
            validation_retry = max(validation_retry, state.get("_validation_retry_count", 0))

        # 4순위: metadata 직접 확인
        metadata = state.get("metadata", {})
        if isinstance(metadata, dict):
            generation_retry = max(generation_retry, metadata.get("generation_retry_count", 0))
            validation_retry = max(validation_retry, metadata.get("validation_retry_count", 0))

        total = generation_retry + validation_retry

        return {
            "generation": generation_retry,
            "validation": validation_retry,
            "total": total
        }

    def increment_retry_count(self, state: LegalWorkflowState, retry_type: str) -> int:
        """
        재시도 카운터 안전하게 증가 (모든 경로에 저장)

        Args:
            state: LegalWorkflowState
            retry_type: "generation" 또는 "validation"

        Returns:
            증가后的 재시도 횟수
        """
        counts = self.get_retry_counts(state)

        if retry_type == "generation":
            new_count = counts["generation"] + 1
            if new_count > RetryConfig.MAX_GENERATION_RETRIES:
                self.logger.warning(
                    f"Generation retry count would exceed limit: {new_count} > {RetryConfig.MAX_GENERATION_RETRIES}"
                )
                new_count = RetryConfig.MAX_GENERATION_RETRIES
        elif retry_type == "validation":
            new_count = counts["validation"] + 1
            if new_count > RetryConfig.MAX_VALIDATION_RETRIES:
                self.logger.warning(
                    f"Validation retry count would exceed limit: {new_count} > {RetryConfig.MAX_VALIDATION_RETRIES}"
                )
                new_count = RetryConfig.MAX_VALIDATION_RETRIES
        else:
            self.logger.error(f"Unknown retry_type: {retry_type}")
            return counts.get(retry_type, 0)

        # 모든 경로에 저장
        self._save_retry_count(state, retry_type, new_count)

        self.logger.info(
            f"✅ [RETRY] {retry_type.capitalize()} retry count: {counts[retry_type]} → {new_count}"
        )

        return new_count

    def _save_retry_count(self, state: LegalWorkflowState, retry_type: str, count: int) -> None:
        """재시도 카운터를 모든 경로에 저장"""
        key = f"{retry_type}_retry_count"

        # metadata에 저장
        if "metadata" not in state or not isinstance(state.get("metadata"), dict):
            state["metadata"] = {}
        state["metadata"][key] = count

        # common.metadata에 저장
        if "common" not in state or not isinstance(state.get("common"), dict):
            state["common"] = {}
        if "metadata" not in state["common"]:
            state["common"]["metadata"] = {}
        state["common"]["metadata"][key] = count

        # 최상위 레벨에 저장 (조건부 엣지 접근용)
        if retry_type == "generation":
            state["retry_count"] = count
            state["_generation_retry_count"] = count
        elif retry_type == "validation":
            state["_validation_retry_count"] = count

    def should_allow_retry(self, state: LegalWorkflowState, retry_type: str) -> bool:
        """
        재시도 허용 여부 확인

        Args:
            state: LegalWorkflowState
            retry_type: "generation" 또는 "validation"

        Returns:
            재시도 허용 여부
        """
        counts = self.get_retry_counts(state)

        # 전역 재시도 횟수 체크
        if counts["total"] >= RetryConfig.MAX_TOTAL_RETRIES:
            self.logger.warning(
                f"Maximum total retry count ({RetryConfig.MAX_TOTAL_RETRIES}) reached"
            )
            return False

        # 개별 재시도 횟수 체크
        if retry_type == "generation":
            if counts["generation"] >= RetryConfig.MAX_GENERATION_RETRIES:
                return False
        elif retry_type == "validation":
            if counts["validation"] >= RetryConfig.MAX_VALIDATION_RETRIES:
                return False

        return True

