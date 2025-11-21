# -*- coding: utf-8 -*-
"""
Answer Routes
답변 생성 관련 라우팅 함수들
"""

import logging
try:
    from lawfirm_langgraph.core.utils.logger import get_logger
except ImportError:
    from core.utils.logger import get_logger
from typing import Any, Optional

from core.agents.state_definitions import LegalWorkflowState
from core.workflow.utils.workflow_utils import WorkflowUtils
from core.workflow.utils.workflow_constants import (
    QualityThresholds,
    RetryConfig,
    WorkflowConstants,
)


logger = get_logger(__name__)


class AnswerRoutes:
    """답변 생성 관련 라우팅 클래스"""
    
    def __init__(
        self,
        retry_manager: Any,
        answer_generator: Any = None,
        logger_instance: Optional[logging.Logger] = None
    ):
        """
        AnswerRoutes 초기화
        
        Args:
            retry_manager: RetryCounterManager 인스턴스
            answer_generator: AnswerGenerator 인스턴스 (선택적)
            logger_instance: 로거 인스턴스
        """
        self.retry_manager = retry_manager
        self.answer_generator = answer_generator
        self.logger = logger_instance or logger
    
    def should_retry_validation(
        self,
        state: LegalWorkflowState,
        answer_generator: Any = None
    ) -> str:
        """
        검증 후 재시도 여부 결정
        
        Args:
            state: 워크플로우 상태
            answer_generator: AnswerGenerator 인스턴스 (선택적)
        
        Returns:
            "accept", "retry_generate", 또는 "retry_search"
        """
        # answer_generator가 제공되지 않으면 self.answer_generator 사용
        if answer_generator is None:
            answer_generator = self.answer_generator
        
        # 품질 메타데이터 조회
        quality_meta = WorkflowUtils.get_quality_metadata(state)
        quality_check_passed = quality_meta["quality_check_passed"]
        quality_score = quality_meta["quality_score"]
        
        # 재시도 카운터 조회
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
        
        # 품질 검증 통과 시 즉시 accept
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
        
        # 개선 가능성 평가
        improvement_potential = None
        if answer_generator:
            improvement_potential = answer_generator.assess_improvement_potential(
                quality_score,
                quality_checks,
                state
            )
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
        
        # 재시도 결정
        if retry_reasons and validation_retry_count < RetryConfig.MAX_VALIDATION_RETRIES:
            if improvement_potential and improvement_potential.get("should_retry"):
                retry_strategy = improvement_potential.get("best_strategy")
                
                # 접지/인용 부족 시 검색 재시도를 우선 적용
                has_sources = bool(WorkflowUtils.get_state_value(state, "sources", [])) or bool(WorkflowUtils.get_state_value(state, "retrieved_docs", []))
                
                if not legal_validity or not has_sources or retry_strategy == "retry_search":
                    self.logger.info(
                        f"🔄 [RETRY] Reasons: {', '.join(retry_reasons)}. "
                        f"Will retry search (count: {validation_retry_count}/{RetryConfig.MAX_VALIDATION_RETRIES})"
                    )
                    return "retry_search"
                elif answer_len < min_length or quality_score < QualityThresholds.MEDIUM_QUALITY_THRESHOLD:
                    self.logger.info(
                        f"🔄 [RETRY] Reasons: {', '.join(retry_reasons)}. "
                        f"Will retry generation (count: {validation_retry_count}/{RetryConfig.MAX_VALIDATION_RETRIES})"
                    )
                    return "retry_generate"
            else:
                if improvement_potential:
                    self.logger.info(
                        f"⚠️ [NO IMPROVEMENT POTENTIAL] Quality improvement unlikely. "
                        f"Score: {quality_score:.2f}, Potential: {improvement_potential.get('confidence', 0.0):.2f}"
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
    
    def should_skip_final_node(self, state: LegalWorkflowState) -> str:
        """
        최종 노드 스킵 여부 결정
        
        Args:
            state: 워크플로우 상태
        
        Returns:
            "skip" 또는 "finalize"
        """
        # 스트리밍 노드에서 이미 검증/포맷팅이 완료된 경우 스킵
        answer = WorkflowUtils.get_state_value(state, "answer", "")
        if answer and len(answer) > 100:
            # 이미 충분한 답변이 있으면 스킵
            return "skip"
        return "finalize"

