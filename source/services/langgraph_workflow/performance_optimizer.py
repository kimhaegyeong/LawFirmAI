# -*- coding: utf-8 -*-
"""
성능 최적화기
LangGraph 워크플로우 성능 최적화를 위한 컴포넌트
"""

import logging
import time
import asyncio
import psutil
import gc
from typing import Dict, Any, List, Optional
from datetime import datetime, timedelta
from functools import wraps
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass
class PerformanceMetrics:
    """성능 메트릭 데이터 클래스"""
    step_name: str
    start_time: float
    end_time: Optional[float] = None
    duration: Optional[float] = None
    memory_usage_mb: float = 0.0
    cpu_usage_percent: float = 0.0
    cache_hits: int = 0
    cache_misses: int = 0
    error_count: int = 0
    retry_count: int = 0


class PerformanceOptimizer:
    """성능 최적화기"""

    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.metrics_history: List[PerformanceMetrics] = []
        self.cache_stats = {"hits": 0, "misses": 0}
        self.performance_thresholds = {
            "max_step_duration": 30.0,  # 30초
            "max_memory_usage_mb": 1024,  # 1GB
            "max_cpu_usage_percent": 80.0,
            "min_cache_hit_rate": 0.7
        }

        # 캐시 시스템 초기화
        self.cache = DocumentCache()

        self.logger.info("PerformanceOptimizer initialized")

    def measure_performance(self, step_name: str):
        """성능 측정 데코레이터"""
        def decorator(func):
            @wraps(func)
            async def async_wrapper(*args, **kwargs):
                metrics = PerformanceMetrics(
                    step_name=step_name,
                    start_time=time.time()
                )

                try:
                    # 메모리 사용량 측정 시작
                    process = psutil.Process()
                    memory_before = process.memory_info().rss / 1024 / 1024
                    cpu_before = process.cpu_percent()

                    # 함수 실행
                    result = await func(*args, **kwargs)

                    # 메트릭 업데이트
                    metrics.end_time = time.time()
                    metrics.duration = metrics.end_time - metrics.start_time
                    metrics.memory_usage_mb = process.memory_info().rss / 1024 / 1024 - memory_before
                    metrics.cpu_usage_percent = process.cpu_percent() - cpu_before

                    # 성능 검사
                    self._check_performance_thresholds(metrics)

                    # 메트릭 저장
                    self.metrics_history.append(metrics)

                    return result

                except Exception as e:
                    metrics.error_count += 1
                    metrics.end_time = time.time()
                    metrics.duration = metrics.end_time - metrics.start_time
                    self.metrics_history.append(metrics)
                    raise e

            @wraps(func)
            def sync_wrapper(*args, **kwargs):
                metrics = PerformanceMetrics(
                    step_name=step_name,
                    start_time=time.time()
                )

                try:
                    # 메모리 사용량 측정 시작
                    process = psutil.Process()
                    memory_before = process.memory_info().rss / 1024 / 1024
                    cpu_before = process.cpu_percent()

                    # 함수 실행
                    result = func(*args, **kwargs)

                    # 메트릭 업데이트
                    metrics.end_time = time.time()
                    metrics.duration = metrics.end_time - metrics.start_time
                    metrics.memory_usage_mb = process.memory_info().rss / 1024 / 1024 - memory_before
                    metrics.cpu_usage_percent = process.cpu_percent() - cpu_before

                    # 성능 검사
                    self._check_performance_thresholds(metrics)

                    # 메트릭 저장
                    self.metrics_history.append(metrics)

                    return result

                except Exception as e:
                    metrics.error_count += 1
                    metrics.end_time = time.time()
                    metrics.duration = metrics.end_time - metrics.start_time
                    self.metrics_history.append(metrics)
                    raise e

            return async_wrapper if asyncio.iscoroutinefunction(func) else sync_wrapper
        return decorator

    def _check_performance_thresholds(self, metrics: PerformanceMetrics):
        """성능 임계값 검사"""
        warnings = []

        if metrics.duration and metrics.duration > self.performance_thresholds["max_step_duration"]:
            warnings.append(f"Step {metrics.step_name} took {metrics.duration:.2f}s (threshold: {self.performance_thresholds['max_step_duration']}s)")

        if metrics.memory_usage_mb > self.performance_thresholds["max_memory_usage_mb"]:
            warnings.append(f"Step {metrics.step_name} used {metrics.memory_usage_mb:.2f}MB (threshold: {self.performance_thresholds['max_memory_usage_mb']}MB)")

        if metrics.cpu_usage_percent > self.performance_thresholds["max_cpu_usage_percent"]:
            warnings.append(f"Step {metrics.step_name} used {metrics.cpu_usage_percent:.2f}% CPU (threshold: {self.performance_thresholds['max_cpu_usage_percent']}%)")

        if warnings:
            self.logger.warning(f"Performance warnings for {metrics.step_name}: {'; '.join(warnings)}")

    def get_performance_summary(self, last_n_minutes: int = 10) -> Dict[str, Any]:
        """성능 요약 정보"""
        cutoff_time = time.time() - (last_n_minutes * 60)
        recent_metrics = [m for m in self.metrics_history if m.start_time >= cutoff_time]

        if not recent_metrics:
            return {"message": "No recent metrics available"}

        # 기본 통계
        total_steps = len(recent_metrics)
        total_duration = sum(m.duration or 0 for m in recent_metrics)
        avg_duration = total_duration / total_steps if total_steps > 0 else 0

        # 메모리 통계
        total_memory = sum(m.memory_usage_mb for m in recent_metrics)
        avg_memory = total_memory / total_steps if total_steps > 0 else 0
        max_memory = max(m.memory_usage_mb for m in recent_metrics)

        # CPU 통계
        total_cpu = sum(m.cpu_usage_percent for m in recent_metrics)
        avg_cpu = total_cpu / total_steps if total_steps > 0 else 0
        max_cpu = max(m.cpu_usage_percent for m in recent_metrics)

        # 오류 통계
        total_errors = sum(m.error_count for m in recent_metrics)
        error_rate = total_errors / total_steps if total_steps > 0 else 0

        # 캐시 통계
        cache_hit_rate = self.cache_stats["hits"] / (self.cache_stats["hits"] + self.cache_stats["misses"]) if (self.cache_stats["hits"] + self.cache_stats["misses"]) > 0 else 0

        return {
            "time_range_minutes": last_n_minutes,
            "total_steps": total_steps,
            "total_duration_seconds": total_duration,
            "average_duration_seconds": avg_duration,
            "total_memory_mb": total_memory,
            "average_memory_mb": avg_memory,
            "max_memory_mb": max_memory,
            "total_cpu_percent": total_cpu,
            "average_cpu_percent": avg_cpu,
            "max_cpu_percent": max_cpu,
            "total_errors": total_errors,
            "error_rate": error_rate,
            "cache_hit_rate": cache_hit_rate,
            "performance_score": self._calculate_performance_score(recent_metrics)
        }

    def _calculate_performance_score(self, metrics: List[PerformanceMetrics]) -> float:
        """성능 점수 계산 (0-100)"""
        if not metrics:
            return 0.0

        score = 100.0

        # 실행 시간 점수 (40% 가중치)
        avg_duration = sum(m.duration or 0 for m in metrics) / len(metrics)
        if avg_duration > self.performance_thresholds["max_step_duration"]:
            duration_score = max(0, 40 - (avg_duration - self.performance_thresholds["max_step_duration"]) * 2)
        else:
            duration_score = 40
        score = min(score, duration_score)

        # 메모리 사용량 점수 (30% 가중치)
        avg_memory = sum(m.memory_usage_mb for m in metrics) / len(metrics)
        if avg_memory > self.performance_thresholds["max_memory_usage_mb"]:
            memory_score = max(0, 30 - (avg_memory - self.performance_thresholds["max_memory_usage_mb"]) / 10)
        else:
            memory_score = 30
        score = min(score, memory_score)

        # 오류율 점수 (20% 가중치)
        error_rate = sum(m.error_count for m in metrics) / len(metrics)
        error_score = max(0, 20 - error_rate * 20)
        score = min(score, error_score)

        # 캐시 히트율 점수 (10% 가중치)
        cache_hit_rate = self.cache_stats["hits"] / (self.cache_stats["hits"] + self.cache_stats["misses"]) if (self.cache_stats["hits"] + self.cache_stats["misses"]) > 0 else 0
        cache_score = cache_hit_rate * 10
        score = min(score, cache_score)

        return max(0, score)

    def optimize_memory_usage(self):
        """메모리 사용량 최적화"""
        try:
            # 가비지 컬렉션 실행
            collected = gc.collect()

            # 메모리 사용량 확인
            process = psutil.Process()
            memory_usage = process.memory_info().rss / 1024 / 1024

            self.logger.info(f"Memory optimization completed: {collected} objects collected, {memory_usage:.2f}MB used")

            return {
                "objects_collected": collected,
                "memory_usage_mb": memory_usage,
                "optimization_time": time.time()
            }

        except Exception as e:
            self.logger.error(f"Memory optimization failed: {e}")
            return {"error": str(e)}

    def get_step_performance_analysis(self, step_name: str) -> Dict[str, Any]:
        """특정 단계의 성능 분석"""
        step_metrics = [m for m in self.metrics_history if m.step_name == step_name]

        if not step_metrics:
            return {"message": f"No metrics found for step: {step_name}"}

        # 통계 계산
        durations = [m.duration for m in step_metrics if m.duration is not None]
        memory_usages = [m.memory_usage_mb for m in step_metrics]
        cpu_usages = [m.cpu_usage_percent for m in step_metrics]
        error_counts = [m.error_count for m in step_metrics]

        analysis = {
            "step_name": step_name,
            "total_executions": len(step_metrics),
            "average_duration": sum(durations) / len(durations) if durations else 0,
            "min_duration": min(durations) if durations else 0,
            "max_duration": max(durations) if durations else 0,
            "average_memory_mb": sum(memory_usages) / len(memory_usages) if memory_usages else 0,
            "max_memory_mb": max(memory_usages) if memory_usages else 0,
            "average_cpu_percent": sum(cpu_usages) / len(cpu_usages) if cpu_usages else 0,
            "max_cpu_percent": max(cpu_usages) if cpu_usages else 0,
            "total_errors": sum(error_counts),
            "error_rate": sum(error_counts) / len(step_metrics),
            "performance_trend": self._calculate_performance_trend(step_metrics)
        }

        return analysis

    def _calculate_performance_trend(self, metrics: List[PerformanceMetrics]) -> str:
        """성능 트렌드 계산"""
        if len(metrics) < 2:
            return "insufficient_data"

        # 최근 절반과 이전 절반 비교
        mid_point = len(metrics) // 2
        recent_metrics = metrics[mid_point:]
        older_metrics = metrics[:mid_point]

        recent_avg_duration = sum(m.duration or 0 for m in recent_metrics) / len(recent_metrics)
        older_avg_duration = sum(m.duration or 0 for m in older_metrics) / len(older_metrics)

        if recent_avg_duration < older_avg_duration * 0.9:
            return "improving"
        elif recent_avg_duration > older_avg_duration * 1.1:
            return "degrading"
        else:
            return "stable"

    def get_optimization_recommendations(self) -> List[Dict[str, Any]]:
        """최적화 권장사항 생성"""
        recommendations = []

        # 최근 성능 데이터 분석
        recent_metrics = self.metrics_history[-50:] if len(self.metrics_history) > 50 else self.metrics_history

        if not recent_metrics:
            return [{"type": "info", "message": "No performance data available for analysis"}]

        # 실행 시간 분석
        avg_duration = sum(m.duration or 0 for m in recent_metrics) / len(recent_metrics)
        if avg_duration > self.performance_thresholds["max_step_duration"]:
            recommendations.append({
                "type": "performance",
                "priority": "high",
                "title": "실행 시간 최적화 필요",
                "description": f"평균 실행 시간이 {avg_duration:.2f}초로 임계값을 초과합니다.",
                "suggestions": [
                    "병렬 처리 도입 검토",
                    "캐싱 전략 개선",
                    "불필요한 연산 제거"
                ]
            })

        # 메모리 사용량 분석
        avg_memory = sum(m.memory_usage_mb for m in recent_metrics) / len(recent_metrics)
        if avg_memory > self.performance_thresholds["max_memory_usage_mb"]:
            recommendations.append({
                "type": "memory",
                "priority": "high",
                "title": "메모리 사용량 최적화 필요",
                "description": f"평균 메모리 사용량이 {avg_memory:.2f}MB로 임계값을 초과합니다.",
                "suggestions": [
                    "메모리 풀 사용",
                    "대용량 객체 재사용",
                    "가비지 컬렉션 최적화"
                ]
            })

        # 오류율 분석
        error_rate = sum(m.error_count for m in recent_metrics) / len(recent_metrics)
        if error_rate > 0.1:  # 10% 이상
            recommendations.append({
                "type": "reliability",
                "priority": "high",
                "title": "오류율 개선 필요",
                "description": f"오류율이 {error_rate:.2%}로 높습니다.",
                "suggestions": [
                    "예외 처리 강화",
                    "재시도 로직 개선",
                    "입력 검증 강화"
                ]
            })

        # 캐시 히트율 분석
        cache_hit_rate = self.cache_stats["hits"] / (self.cache_stats["hits"] + self.cache_stats["misses"]) if (self.cache_stats["hits"] + self.cache_stats["misses"]) > 0 else 0
        if cache_hit_rate < self.performance_thresholds["min_cache_hit_rate"]:
            recommendations.append({
                "type": "caching",
                "priority": "medium",
                "title": "캐시 전략 개선 필요",
                "description": f"캐시 히트율이 {cache_hit_rate:.2%}로 낮습니다.",
                "suggestions": [
                    "캐시 키 전략 개선",
                    "캐시 TTL 조정",
                    "캐시 크기 증가"
                ]
            })

        # 기본 권장사항
        if not recommendations:
            recommendations.append({
                "type": "info",
                "priority": "low",
                "title": "성능 상태 양호",
                "description": "현재 성능 지표가 모두 정상 범위 내에 있습니다.",
                "suggestions": [
                    "정기적인 성능 모니터링 유지",
                    "새로운 기능 추가 시 성능 영향 평가"
                ]
            })

        return recommendations

    def record_cache_hit(self):
        """캐시 히트 기록"""
        self.cache_stats["hits"] += 1

    def record_cache_miss(self):
        """캐시 미스 기록"""
        self.cache_stats["misses"] += 1

    def clear_metrics_history(self, older_than_hours: int = 24):
        """오래된 메트릭 히스토리 정리"""
        cutoff_time = time.time() - (older_than_hours * 3600)
        original_count = len(self.metrics_history)
        self.metrics_history = [m for m in self.metrics_history if m.start_time >= cutoff_time]
        removed_count = original_count - len(self.metrics_history)

        self.logger.info(f"Cleared {removed_count} old metrics (older than {older_than_hours} hours)")
        return removed_count

    def export_metrics(self, file_path: str):
        """메트릭 데이터 내보내기"""
        try:
            import json

            export_data = {
                "export_time": datetime.now().isoformat(),
                "metrics_count": len(self.metrics_history),
                "cache_stats": self.cache_stats,
                "performance_thresholds": self.performance_thresholds,
                "metrics": [
                    {
                        "step_name": m.step_name,
                        "start_time": m.start_time,
                        "end_time": m.end_time,
                        "duration": m.duration,
                        "memory_usage_mb": m.memory_usage_mb,
                        "cpu_usage_percent": m.cpu_usage_percent,
                        "cache_hits": m.cache_hits,
                        "cache_misses": m.cache_misses,
                        "error_count": m.error_count,
                        "retry_count": m.retry_count
                    }
                    for m in self.metrics_history
                ]
            }

            with open(file_path, 'w', encoding='utf-8') as f:
                json.dump(export_data, f, indent=2, ensure_ascii=False)

            self.logger.info(f"Metrics exported to {file_path}")
            return True

        except Exception as e:
            self.logger.error(f"Failed to export metrics: {e}")
            return False


class DocumentCache:
    """문서 캐시 클래스"""

    def __init__(self):
        self.cache = {}
        self.logger = logging.getLogger(__name__)

    def get_cached_documents(self, query: str, query_type: str) -> Optional[List[Dict[str, Any]]]:
        """캐시된 문서 가져오기"""
        cache_key = f"{query}_{query_type}"
        if cache_key in self.cache:
            self.logger.debug(f"Cache hit for query: {query}")
            return self.cache[cache_key]
        return None

    def cache_documents(self, query: str, query_type: str, documents: List[Dict[str, Any]]):
        """문서 캐시 저장"""
        cache_key = f"{query}_{query_type}"
        self.cache[cache_key] = documents
        self.logger.debug(f"Cached documents for query: {query}")

    def clear_cache(self):
        """캐시 클리어"""
        self.cache.clear()
        self.logger.info("Document cache cleared")
