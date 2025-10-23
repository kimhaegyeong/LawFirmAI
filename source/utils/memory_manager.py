# -*- coding: utf-8 -*-
"""
Memory Manager
메모리 사용량 모니터링 및 최적화를 위한 관리자 클래스
"""

import os
import gc
import time
import weakref
import psutil
import threading
from typing import Dict, List, Any, Optional, Callable
from datetime import datetime, timedelta
from dataclasses import dataclass
from ..utils.logger import get_logger

logger = get_logger(__name__)


@dataclass
class MemoryMetrics:
    """메모리 메트릭 데이터 클래스"""
    timestamp: datetime
    rss_mb: float  # 실제 메모리 사용량 (MB)
    vms_mb: float  # 가상 메모리 사용량 (MB)
    cpu_percent: float  # CPU 사용률
    memory_percent: float  # 시스템 메모리 사용률
    available_mb: float  # 사용 가능한 메모리 (MB)


@dataclass
class MemoryAlert:
    """메모리 알림 데이터 클래스"""
    alert_type: str  # 'high_usage', 'leak_detected', 'cleanup_needed'
    message: str
    severity: str  # 'low', 'medium', 'high', 'critical'
    timestamp: datetime
    metrics: MemoryMetrics


class MemoryManager:
    """메모리 관리자 클래스"""
    
    def __init__(self, 
                 max_memory_mb: int = 1024,
                 alert_threshold: float = 0.8,
                 cleanup_threshold: float = 0.9,
                 monitoring_interval: int = 30):
        """
        메모리 관리자 초기화
        
        Args:
            max_memory_mb: 최대 허용 메모리 사용량 (MB)
            alert_threshold: 알림 임계값 (비율)
            cleanup_threshold: 정리 임계값 (비율)
            monitoring_interval: 모니터링 간격 (초)
        """
        self.max_memory_mb = max_memory_mb
        self.alert_threshold = alert_threshold
        self.cleanup_threshold = cleanup_threshold
        self.monitoring_interval = monitoring_interval
        
        self.logger = get_logger(__name__)
        
        # 메모리 메트릭 히스토리
        self.metrics_history: List[MemoryMetrics] = []
        self.max_history_size = 1000
        
        # 추적 중인 객체들
        self.tracked_objects: List[weakref.ref] = []
        self.object_registry: Dict[str, Any] = {}
        
        # 알림 콜백 함수들
        self.alert_callbacks: List[Callable[[MemoryAlert], None]] = []
        
        # 모니터링 스레드
        self.monitoring_thread: Optional[threading.Thread] = None
        self.monitoring_active = False
        
        # 메모리 정리 통계
        self.cleanup_stats = {
            'total_cleanups': 0,
            'last_cleanup_time': None,
            'objects_freed': 0,
            'memory_freed_mb': 0.0
        }
        
        self.logger.info(f"MemoryManager 초기화 완료 - 최대 메모리: {max_memory_mb}MB")
    
    def start_monitoring(self):
        """메모리 모니터링 시작"""
        if self.monitoring_active:
            self.logger.warning("메모리 모니터링이 이미 실행 중입니다.")
            return
        
        self.monitoring_active = True
        self.monitoring_thread = threading.Thread(
            target=self._monitoring_loop,
            daemon=True,
            name="MemoryMonitor"
        )
        self.monitoring_thread.start()
        self.logger.info("메모리 모니터링 시작")
    
    def stop_monitoring(self):
        """메모리 모니터링 중지"""
        self.monitoring_active = False
        if self.monitoring_thread:
            self.monitoring_thread.join(timeout=5)
        self.logger.info("메모리 모니터링 중지")
    
    def _monitoring_loop(self):
        """메모리 모니터링 루프"""
        while self.monitoring_active:
            try:
                metrics = self.get_current_metrics()
                self.metrics_history.append(metrics)
                
                # 히스토리 크기 제한
                if len(self.metrics_history) > self.max_history_size:
                    self.metrics_history = self.metrics_history[-self.max_history_size:]
                
                # 메모리 사용량 체크
                self._check_memory_usage(metrics)
                
                # 메모리 누수 감지
                self._detect_memory_leak(metrics)
                
                time.sleep(self.monitoring_interval)
                
            except Exception as e:
                self.logger.error(f"메모리 모니터링 중 오류: {e}")
                time.sleep(self.monitoring_interval)
    
    def get_current_metrics(self) -> MemoryMetrics:
        """현재 메모리 메트릭 수집"""
        try:
            process = psutil.Process()
            memory_info = process.memory_info()
            system_memory = psutil.virtual_memory()
            
            return MemoryMetrics(
                timestamp=datetime.now(),
                rss_mb=memory_info.rss / 1024 / 1024,
                vms_mb=memory_info.vms / 1024 / 1024,
                cpu_percent=process.cpu_percent(),
                memory_percent=system_memory.percent,
                available_mb=system_memory.available / 1024 / 1024
            )
        except Exception as e:
            self.logger.error(f"메모리 메트릭 수집 실패: {e}")
            return MemoryMetrics(
                timestamp=datetime.now(),
                rss_mb=0.0,
                vms_mb=0.0,
                cpu_percent=0.0,
                memory_percent=0.0,
                available_mb=0.0
            )
    
    def _check_memory_usage(self, metrics: MemoryMetrics):
        """메모리 사용량 체크 및 알림"""
        usage_ratio = metrics.rss_mb / self.max_memory_mb
        
        if usage_ratio >= self.cleanup_threshold:
            # 정리 필요
            alert = MemoryAlert(
                alert_type='cleanup_needed',
                message=f"메모리 사용량이 정리 임계값을 초과했습니다. ({metrics.rss_mb:.1f}MB / {self.max_memory_mb}MB)",
                severity='high',
                timestamp=metrics.timestamp,
                metrics=metrics
            )
            self._trigger_alert(alert)
            self.perform_cleanup()
            
        elif usage_ratio >= self.alert_threshold:
            # 알림 필요
            alert = MemoryAlert(
                alert_type='high_usage',
                message=f"메모리 사용량이 높습니다. ({metrics.rss_mb:.1f}MB / {self.max_memory_mb}MB)",
                severity='medium',
                timestamp=metrics.timestamp,
                metrics=metrics
            )
            self._trigger_alert(alert)
    
    def _detect_memory_leak(self, metrics: MemoryMetrics):
        """메모리 누수 감지"""
        if len(self.metrics_history) < 10:
            return
        
        # 최근 10개 메트릭의 평균 증가율 계산
        recent_metrics = self.metrics_history[-10:]
        memory_values = [m.rss_mb for m in recent_metrics]
        
        # 선형 증가 추세 감지
        if len(memory_values) >= 5:
            trend = self._calculate_trend(memory_values)
            if trend > 5.0:  # 5MB/분 이상 증가
                alert = MemoryAlert(
                    alert_type='leak_detected',
                    message=f"메모리 누수 가능성이 감지되었습니다. 증가율: {trend:.1f}MB/분",
                    severity='critical',
                    timestamp=metrics.timestamp,
                    metrics=metrics
                )
                self._trigger_alert(alert)
                self.perform_cleanup()
    
    def _calculate_trend(self, values: List[float]) -> float:
        """값들의 증가 추세 계산 (MB/분)"""
        if len(values) < 2:
            return 0.0
        
        # 간단한 선형 회귀로 기울기 계산
        n = len(values)
        x_sum = sum(range(n))
        y_sum = sum(values)
        xy_sum = sum(i * values[i] for i in range(n))
        x2_sum = sum(i * i for i in range(n))
        
        if n * x2_sum - x_sum * x_sum == 0:
            return 0.0
        
        slope = (n * xy_sum - x_sum * y_sum) / (n * x2_sum - x_sum * x_sum)
        return slope * self.monitoring_interval / 60.0  # MB/분으로 변환
    
    def _trigger_alert(self, alert: MemoryAlert):
        """알림 트리거"""
        self.logger.warning(f"메모리 알림 [{alert.severity}]: {alert.message}")
        
        # 등록된 콜백 함수들 실행
        for callback in self.alert_callbacks:
            try:
                callback(alert)
            except Exception as e:
                self.logger.error(f"알림 콜백 실행 실패: {e}")
    
    def add_alert_callback(self, callback: Callable[[MemoryAlert], None]):
        """알림 콜백 함수 추가"""
        self.alert_callbacks.append(callback)
    
    def track_object(self, obj: Any, name: Optional[str] = None) -> str:
        """객체 추적 시작"""
        if name is None:
            name = f"obj_{id(obj)}_{type(obj).__name__}"
        
        # WeakRef로 객체 추적
        ref = weakref.ref(obj, lambda r: self._on_object_deleted(name))
        self.tracked_objects.append(ref)
        self.object_registry[name] = ref
        
        self.logger.debug(f"객체 추적 시작: {name}")
        return name
    
    def _on_object_deleted(self, name: str):
        """객체 삭제 시 호출"""
        if name in self.object_registry:
            del self.object_registry[name]
        self.logger.debug(f"객체 삭제됨: {name}")
    
    def untrack_object(self, name: str):
        """객체 추적 중지"""
        if name in self.object_registry:
            del self.object_registry[name]
            self.logger.debug(f"객체 추적 중지: {name}")
    
    def perform_cleanup(self) -> Dict[str, Any]:
        """메모리 정리 수행"""
        cleanup_start_time = time.time()
        initial_memory = self.get_current_metrics().rss_mb
        
        # 1. WeakRef 정리
        self._cleanup_weakrefs()
        
        # 2. 가비지 컬렉션 강제 실행
        collected = gc.collect()
        
        # 3. 캐시 정리 (있는 경우)
        self._cleanup_caches()
        
        # 4. 메트릭 히스토리 정리
        self._cleanup_metrics_history()
        
        cleanup_end_time = time.time()
        final_memory = self.get_current_metrics().rss_mb
        
        # 정리 통계 업데이트
        memory_freed = initial_memory - final_memory
        self.cleanup_stats.update({
            'total_cleanups': self.cleanup_stats['total_cleanups'] + 1,
            'last_cleanup_time': datetime.now(),
            'objects_freed': self.cleanup_stats['objects_freed'] + collected,
            'memory_freed_mb': self.cleanup_stats['memory_freed_mb'] + memory_freed
        })
        
        result = {
            'success': True,
            'initial_memory_mb': initial_memory,
            'final_memory_mb': final_memory,
            'memory_freed_mb': memory_freed,
            'objects_collected': collected,
            'cleanup_duration': cleanup_end_time - cleanup_start_time,
            'timestamp': datetime.now()
        }
        
        self.logger.info(f"메모리 정리 완료: {memory_freed:.1f}MB 해제, {collected}개 객체 수집")
        return result
    
    def _cleanup_weakrefs(self):
        """WeakRef 정리"""
        initial_count = len(self.tracked_objects)
        self.tracked_objects = [ref for ref in self.tracked_objects if ref() is not None]
        final_count = len(self.tracked_objects)
        
        if initial_count != final_count:
            self.logger.debug(f"WeakRef 정리: {initial_count - final_count}개 참조 제거")
    
    def _cleanup_caches(self):
        """캐시 정리 (구현체에 따라 다름)"""
        # 실제 캐시 정리 로직은 구현체에 따라 달라짐
        pass
    
    def _cleanup_metrics_history(self):
        """메트릭 히스토리 정리"""
        if len(self.metrics_history) > self.max_history_size // 2:
            self.metrics_history = self.metrics_history[-self.max_history_size // 2:]
            self.logger.debug("메트릭 히스토리 정리 완료")
    
    def get_memory_usage(self) -> Dict[str, Any]:
        """현재 메모리 사용량 정보 반환"""
        metrics = self.get_current_metrics()
        
        return {
            'current_memory_mb': metrics.rss_mb,
            'max_memory_mb': self.max_memory_mb,
            'usage_percentage': (metrics.rss_mb / self.max_memory_mb) * 100,
            'virtual_memory_mb': metrics.vms_mb,
            'cpu_percent': metrics.cpu_percent,
            'system_memory_percent': metrics.memory_percent,
            'available_memory_mb': metrics.available_mb,
            'tracked_objects_count': len(self.tracked_objects),
            'registry_size': len(self.object_registry),
            'metrics_history_size': len(self.metrics_history),
            'cleanup_stats': self.cleanup_stats.copy(),
            'monitoring_active': self.monitoring_active,
            'timestamp': metrics.timestamp
        }
    
    def get_memory_trend(self, hours: int = 1) -> Dict[str, Any]:
        """메모리 사용량 추세 분석"""
        cutoff_time = datetime.now() - timedelta(hours=hours)
        recent_metrics = [m for m in self.metrics_history if m.timestamp >= cutoff_time]
        
        if len(recent_metrics) < 2:
            return {'trend': 0.0, 'message': '데이터 부족'}
        
        memory_values = [m.rss_mb for m in recent_metrics]
        trend = self._calculate_trend(memory_values)
        
        return {
            'trend_mb_per_minute': trend,
            'data_points': len(recent_metrics),
            'time_range_hours': hours,
            'min_memory_mb': min(memory_values),
            'max_memory_mb': max(memory_values),
            'avg_memory_mb': sum(memory_values) / len(memory_values),
            'trend_direction': 'increasing' if trend > 0 else 'decreasing' if trend < 0 else 'stable'
        }
    
    def force_garbage_collection(self) -> Dict[str, Any]:
        """강제 가비지 컬렉션 실행"""
        initial_memory = self.get_current_metrics().rss_mb
        
        # 모든 세대의 가비지 컬렉션 실행
        collected = []
        for generation in range(3):
            collected.append(gc.collect(generation))
        
        final_memory = self.get_current_metrics().rss_mb
        
        return {
            'success': True,
            'initial_memory_mb': initial_memory,
            'final_memory_mb': final_memory,
            'memory_freed_mb': initial_memory - final_memory,
            'objects_collected_by_generation': collected,
            'total_objects_collected': sum(collected),
            'timestamp': datetime.now()
        }
    
    def get_system_info(self) -> Dict[str, Any]:
        """시스템 메모리 정보 반환"""
        try:
            memory = psutil.virtual_memory()
            swap = psutil.swap_memory()
            
            return {
                'total_memory_mb': memory.total / 1024 / 1024,
                'available_memory_mb': memory.available / 1024 / 1024,
                'used_memory_mb': memory.used / 1024 / 1024,
                'memory_percent': memory.percent,
                'total_swap_mb': swap.total / 1024 / 1024,
                'used_swap_mb': swap.used / 1024 / 1024,
                'swap_percent': swap.percent,
                'timestamp': datetime.now()
            }
        except Exception as e:
            self.logger.error(f"시스템 정보 수집 실패: {e}")
            return {'error': str(e)}
    
    def __del__(self):
        """소멸자 - 모니터링 중지"""
        self.stop_monitoring()


# 전역 메모리 관리자 인스턴스
_memory_manager_instance: Optional[MemoryManager] = None


def get_memory_manager(max_memory_mb: int = 1024) -> MemoryManager:
    """메모리 관리자 싱글톤 인스턴스 반환"""
    global _memory_manager_instance
    
    if _memory_manager_instance is None:
        _memory_manager_instance = MemoryManager(max_memory_mb=max_memory_mb)
        _memory_manager_instance.start_monitoring()
    
    return _memory_manager_instance


def cleanup_memory_manager():
    """메모리 관리자 정리"""
    global _memory_manager_instance
    
    if _memory_manager_instance:
        _memory_manager_instance.stop_monitoring()
        _memory_manager_instance = None
