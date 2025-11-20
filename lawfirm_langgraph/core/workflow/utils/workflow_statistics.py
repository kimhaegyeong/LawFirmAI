# -*- coding: utf-8 -*-
"""
Workflow Statistics
워크플로우 통계 관리 모듈
"""

import logging
from typing import Any, Dict, Optional

from core.workflow.state.state_definitions import LegalWorkflowState

logger = logging.getLogger(__name__)


class WorkflowStatistics:
    """워크플로우 통계 관리 클래스"""

    def __init__(self, enable_statistics: bool = True):
        self.enable_statistics = enable_statistics
        self.stats = self._initialize_statistics() if enable_statistics else None

    def _initialize_statistics(self) -> Dict[str, Any]:
        """통계 초기화"""
        return {
            'total_queries': 0,
            'total_documents_retrieved': 0,
            'avg_response_time': 0.0,
            'avg_confidence': 0.0,
            'total_errors': 0,
            'llm_complexity_classifications': 0,
            'complexity_cache_hits': 0,
            'complexity_cache_misses': 0,
            'complexity_fallback_count': 0,
            'avg_complexity_classification_time': 0.0,
            # 통합 분류 메트릭 (최적화)
            'unified_classification_calls': 0,
            'unified_classification_llm_calls': 0,
            'avg_unified_classification_time': 0.0,
            'total_unified_classification_time': 0.0,
            # 폴백 원인 분류
            'fallback_reasons': {}
        }

    def update_statistics(self, state: LegalWorkflowState, config=None):
        """통계 업데이트 (이동 평균 사용)"""
        if not self.stats:
            return

        try:
            self.stats['total_queries'] += 1
            processing_time = state.get("processing_time", 0.0)
            confidence = state.get("confidence", 0.0)
            docs_count = len(state.get("retrieved_docs", []))
            errors_count = len(state.get("errors", []))

            # 이동 평균 계산
            alpha = config.stats_update_alpha if config and hasattr(config, 'stats_update_alpha') else 0.1

            if self.stats['total_queries'] == 1:
                self.stats['avg_response_time'] = processing_time
                self.stats['avg_confidence'] = confidence
            else:
                # 이동 평균 업데이트
                self.stats['avg_response_time'] = (
                    (1 - alpha) * self.stats['avg_response_time'] +
                    alpha * processing_time
                )
                self.stats['avg_confidence'] = (
                    (1 - alpha) * self.stats['avg_confidence'] +
                    alpha * confidence
                )

            # 누적 통계
            self.stats['total_documents_retrieved'] += docs_count
            self.stats['total_errors'] += errors_count

            logger.debug(
                f"Statistics updated: queries={self.stats['total_queries']}, "
                f"avg_time={self.stats['avg_response_time']:.2f}s, "
                f"avg_conf={self.stats['avg_confidence']:.2f}"
            )
        except Exception as e:
            logger.warning(f"Statistics update failed: {e}")

    def get_statistics(self) -> Dict[str, Any]:
        """통계 조회"""
        if not self.stats:
            return {"enabled": False}

        return {
            "enabled": True,
            "total_queries": self.stats['total_queries'],
            "total_documents_retrieved": self.stats['total_documents_retrieved'],
            "avg_response_time": round(self.stats['avg_response_time'], 3),
            "avg_confidence": round(self.stats['avg_confidence'], 3),
            "total_errors": self.stats['total_errors'],
            "avg_docs_per_query": (
                round(self.stats['total_documents_retrieved'] / self.stats['total_queries'], 2)
                if self.stats['total_queries'] > 0 else 0
            )
        }

    def reset_statistics(self):
        """통계 초기화"""
        if self.enable_statistics:
            self.stats = self._initialize_statistics()
            logger.info("Statistics reset")

