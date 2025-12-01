# -*- coding: utf-8 -*-
"""
검색 성능 모니터링 시스템
검색 시간, 품질 메트릭, 캐시 히트율 추적
"""

import logging
try:
    from lawfirm_langgraph.core.utils.logger import get_logger
except ImportError:
    from core.utils.logger import get_logger
import time
from collections import deque
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, List, Optional

logger = get_logger(__name__)


@dataclass
class SearchMetrics:
    """검색 성능 메트릭"""
    query: str
    timestamp: float = field(default_factory=time.time)
    search_time: float = 0.0
    semantic_count: int = 0
    keyword_count: int = 0
    total_results: int = 0
    average_score: float = 0.0
    min_score: float = 0.0
    max_score: float = 0.0
    cache_hit: bool = False
    error: Optional[str] = None


class SearchPerformanceMonitor:
    """검색 성능 모니터링 클래스"""

    def __init__(self, max_history: int = 1000):
        """
        성능 모니터 초기화

        Args:
            max_history: 최대 기록 수
        """
        self.max_history = max_history
        self.search_history: deque = deque(maxlen=max_history)
        self.cache_stats = {
            "hits": 0,
            "misses": 0,
            "total": 0
        }
        self.logger = get_logger(__name__)

    def record_search(
        self,
        query: str,
        search_time: float,
        semantic_count: int,
        keyword_count: int,
        total_results: int,
        scores: List[float],
        cache_hit: bool = False,
        error: Optional[str] = None
    ):
        """
        검색 메트릭 기록

        Args:
            query: 검색 쿼리
            search_time: 검색 시간 (초)
            semantic_count: Semantic 검색 결과 수
            keyword_count: Keyword 검색 결과 수
            total_results: 최종 결과 수
            scores: 관련도 점수 리스트
            cache_hit: 캐시 히트 여부
            error: 에러 메시지 (있는 경우)
        """
        try:
            avg_score = sum(scores) / len(scores) if scores else 0.0
            min_score = min(scores) if scores else 0.0
            max_score = max(scores) if scores else 0.0

            metrics = SearchMetrics(
                query=query,
                search_time=search_time,
                semantic_count=semantic_count,
                keyword_count=keyword_count,
                total_results=total_results,
                average_score=avg_score,
                min_score=min_score,
                max_score=max_score,
                cache_hit=cache_hit,
                error=error
            )

            self.search_history.append(metrics)

            # 캐시 통계 업데이트
            if cache_hit:
                self.cache_stats["hits"] += 1
            else:
                self.cache_stats["misses"] += 1
            self.cache_stats["total"] += 1

        except Exception as e:
            self.logger.warning(f"Error recording search metrics: {e}")

    def get_cache_hit_rate(self) -> float:
        """캐시 히트율 계산"""
        total = self.cache_stats["total"]
        if total == 0:
            return 0.0
        return self.cache_stats["hits"] / total

    def get_average_search_time(self, recent_n: int = 100) -> float:
        """최근 N개 검색의 평균 검색 시간"""
        if len(self.search_history) == 0:
            return 0.0

        recent_searches = list(self.search_history)[-recent_n:]
        total_time = sum(m.search_time for m in recent_searches)
        return total_time / len(recent_searches)

    def get_average_result_count(self, recent_n: int = 100) -> float:
        """최근 N개 검색의 평균 결과 수"""
        if len(self.search_history) == 0:
            return 0.0

        recent_searches = list(self.search_history)[-recent_n:]
        total_results = sum(m.total_results for m in recent_searches)
        return total_results / len(recent_searches)

    def get_average_score(self, recent_n: int = 100) -> float:
        """최근 N개 검색의 평균 관련도 점수"""
        if len(self.search_history) == 0:
            return 0.0

        recent_searches = list(self.search_history)[-recent_n:]
        avg_scores = [m.average_score for m in recent_searches if m.average_score > 0]
        if not avg_scores:
            return 0.0
        return sum(avg_scores) / len(avg_scores)

    def get_error_rate(self, recent_n: int = 100) -> float:
        """최근 N개 검색의 에러율"""
        if len(self.search_history) == 0:
            return 0.0

        recent_searches = list(self.search_history)[-recent_n:]
        errors = sum(1 for m in recent_searches if m.error is not None)
        return errors / len(recent_searches)

    def get_performance_summary(self) -> Dict[str, Any]:
        """성능 요약 정보 반환"""
        return {
            "total_searches": len(self.search_history),
            "cache_hit_rate": self.get_cache_hit_rate(),
            "average_search_time": self.get_average_search_time(),
            "average_result_count": self.get_average_result_count(),
            "average_score": self.get_average_score(),
            "error_rate": self.get_error_rate(),
            "cache_stats": self.cache_stats.copy()
        }

    def log_performance_summary(self):
        """성능 요약 로그 출력"""
        summary = self.get_performance_summary()
        self.logger.info(
            f"Search Performance Summary:\n"
            f"  Total searches: {summary['total_searches']}\n"
            f"  Cache hit rate: {summary['cache_hit_rate']:.2%}\n"
            f"  Average search time: {summary['average_search_time']:.3f}s\n"
            f"  Average result count: {summary['average_result_count']:.1f}\n"
            f"  Average score: {summary['average_score']:.3f}\n"
            f"  Error rate: {summary['error_rate']:.2%}"
        )
