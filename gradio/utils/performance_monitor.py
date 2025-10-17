# -*- coding: utf-8 -*-
"""
성능 모니터링 유틸리티
HuggingFace Spaces 환경에서 성능을 모니터링합니다.
"""

import time
import logging
import psutil
from typing import Dict, Any, List, Optional
from datetime import datetime, timedelta
from collections import deque
from dataclasses import dataclass

logger = logging.getLogger(__name__)

@dataclass
class PerformanceMetrics:
    """성능 메트릭 데이터 클래스"""
    timestamp: datetime
    response_time: float
    memory_usage: float
    cpu_usage: float
    success: bool
    error_message: Optional[str] = None

class PerformanceMonitor:
    """성능 모니터링 클래스"""
    
    def __init__(self, max_history: int = 1000):
        """
        성능 모니터링기 초기화
        
        Args:
            max_history: 최대 기록 수 (기본값: 1000)
        """
        self.max_history = max_history
        self.logger = logging.getLogger(__name__)
        
        # 성능 메트릭 저장
        self.metrics_history = deque(maxlen=max_history)
        
        # 통계 정보
        self.total_requests = 0
        self.successful_requests = 0
        self.failed_requests = 0
        
        # 응답 시간 통계
        self.response_times = deque(maxlen=max_history)
        
        # 에러 통계
        self.error_counts = {}
        
        # 시작 시간
        self.start_time = datetime.now()
    
    def log_request(self, 
                    response_time: float, 
                    success: bool = True, 
                    error_message: Optional[str] = None) -> None:
        """요청 로깅"""
        # 현재 시스템 상태 측정
        memory_usage = psutil.virtual_memory().percent
        cpu_usage = psutil.cpu_percent()
        
        # 메트릭 생성
        metric = PerformanceMetrics(
            timestamp=datetime.now(),
            response_time=response_time,
            memory_usage=memory_usage,
            cpu_usage=cpu_usage,
            success=success,
            error_message=error_message
        )
        
        # 메트릭 저장
        self.metrics_history.append(metric)
        self.response_times.append(response_time)
        
        # 통계 업데이트
        self.total_requests += 1
        if success:
            self.successful_requests += 1
        else:
            self.failed_requests += 1
            if error_message:
                self.error_counts[error_message] = self.error_counts.get(error_message, 0) + 1
        
        # 로깅
        if success:
            self.logger.debug(f"Request completed: {response_time:.3f}s, memory: {memory_usage:.1f}%")
        else:
            self.logger.warning(f"Request failed: {response_time:.3f}s, error: {error_message}")
    
    def get_response_time_stats(self) -> Dict[str, float]:
        """응답 시간 통계 반환"""
        if not self.response_times:
            return {
                "avg": 0.0,
                "min": 0.0,
                "max": 0.0,
                "p50": 0.0,
                "p95": 0.0,
                "p99": 0.0
            }
        
        sorted_times = sorted(self.response_times)
        n = len(sorted_times)
        
        return {
            "avg": sum(sorted_times) / n,
            "min": min(sorted_times),
            "max": max(sorted_times),
            "p50": sorted_times[int(n * 0.5)],
            "p95": sorted_times[int(n * 0.95)],
            "p99": sorted_times[int(n * 0.99)]
        }
    
    def get_success_rate(self) -> float:
        """성공률 반환"""
        if self.total_requests == 0:
            return 0.0
        return self.successful_requests / self.total_requests
    
    def get_error_rate(self) -> float:
        """에러율 반환"""
        if self.total_requests == 0:
            return 0.0
        return self.failed_requests / self.total_requests
    
    def get_uptime(self) -> timedelta:
        """업타임 반환"""
        return datetime.now() - self.start_time
    
    def get_recent_metrics(self, minutes: int = 5) -> List[PerformanceMetrics]:
        """최근 N분간의 메트릭 반환"""
        cutoff_time = datetime.now() - timedelta(minutes=minutes)
        return [metric for metric in self.metrics_history if metric.timestamp >= cutoff_time]
    
    def get_memory_stats(self) -> Dict[str, Any]:
        """메모리 통계 반환"""
        if not self.metrics_history:
            return {
                "avg": 0.0,
                "min": 0.0,
                "max": 0.0,
                "current": 0.0
            }
        
        memory_values = [metric.memory_usage for metric in self.metrics_history]
        current_memory = psutil.virtual_memory().percent
        
        return {
            "avg": sum(memory_values) / len(memory_values),
            "min": min(memory_values),
            "max": max(memory_values),
            "current": current_memory
        }
    
    def get_cpu_stats(self) -> Dict[str, Any]:
        """CPU 통계 반환"""
        if not self.metrics_history:
            return {
                "avg": 0.0,
                "min": 0.0,
                "max": 0.0,
                "current": 0.0
            }
        
        cpu_values = [metric.cpu_usage for metric in self.metrics_history]
        current_cpu = psutil.cpu_percent()
        
        return {
            "avg": sum(cpu_values) / len(cpu_values),
            "min": min(cpu_values),
            "max": max(cpu_values),
            "current": current_cpu
        }
    
    def get_error_stats(self) -> Dict[str, Any]:
        """에러 통계 반환"""
        return {
            "total_errors": self.failed_requests,
            "error_rate": self.get_error_rate(),
            "error_counts": self.error_counts,
            "most_common_error": max(self.error_counts.items(), key=lambda x: x[1])[0] if self.error_counts else None
        }
    
    def get_overall_stats(self) -> Dict[str, Any]:
        """전체 통계 반환"""
        return {
            "uptime": str(self.get_uptime()),
            "total_requests": self.total_requests,
            "successful_requests": self.successful_requests,
            "failed_requests": self.failed_requests,
            "success_rate": self.get_success_rate(),
            "error_rate": self.get_error_rate(),
            "response_time_stats": self.get_response_time_stats(),
            "memory_stats": self.get_memory_stats(),
            "cpu_stats": self.get_cpu_stats(),
            "error_stats": self.get_error_stats(),
            "start_time": self.start_time.isoformat(),
            "current_time": datetime.now().isoformat()
        }
    
    def get_health_status(self) -> Dict[str, Any]:
        """헬스 상태 반환"""
        response_stats = self.get_response_time_stats()
        memory_stats = self.get_memory_stats()
        cpu_stats = self.get_cpu_stats()
        
        # 헬스 체크 기준
        avg_response_time = response_stats["avg"]
        current_memory = memory_stats["current"]
        current_cpu = cpu_stats["current"]
        success_rate = self.get_success_rate()
        
        # 상태 결정
        status = "healthy"
        issues = []
        
        if avg_response_time > 5.0:  # 5초 이상
            status = "degraded"
            issues.append(f"High response time: {avg_response_time:.2f}s")
        
        if current_memory > 90:  # 90% 이상
            status = "degraded"
            issues.append(f"High memory usage: {current_memory:.1f}%")
        
        if current_cpu > 90:  # 90% 이상
            status = "degraded"
            issues.append(f"High CPU usage: {current_cpu:.1f}%")
        
        if success_rate < 0.95:  # 95% 미만
            status = "degraded"
            issues.append(f"Low success rate: {success_rate:.1%}")
        
        return {
            "status": status,
            "issues": issues,
            "avg_response_time": avg_response_time,
            "current_memory": current_memory,
            "current_cpu": current_cpu,
            "success_rate": success_rate,
            "timestamp": datetime.now().isoformat()
        }
    
    def reset_stats(self):
        """통계 초기화"""
        self.metrics_history.clear()
        self.response_times.clear()
        self.error_counts.clear()
        self.total_requests = 0
        self.successful_requests = 0
        self.failed_requests = 0
        self.start_time = datetime.now()
        self.logger.info("Performance statistics reset")

# 전역 성능 모니터 인스턴스
performance_monitor = PerformanceMonitor()

def get_performance_monitor() -> PerformanceMonitor:
    """성능 모니터 인스턴스 반환"""
    return performance_monitor

def monitor_request(func):
    """요청 모니터링 데코레이터"""
    def wrapper(*args, **kwargs):
        start_time = time.time()
        success = True
        error_message = None
        
        try:
            result = func(*args, **kwargs)
            return result
        except Exception as e:
            success = False
            error_message = str(e)
            raise
        finally:
            response_time = time.time() - start_time
            performance_monitor.log_request(response_time, success, error_message)
    
    return wrapper
