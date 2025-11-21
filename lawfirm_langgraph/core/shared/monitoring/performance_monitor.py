# -*- coding: utf-8 -*-
"""
성능 모니터링 시스템
"""

import time
import logging
try:
    from lawfirm_langgraph.core.utils.logger import get_logger
except ImportError:
    from core.utils.logger import get_logger
import json
import psutil
from typing import Dict, Any, List, Optional
from dataclasses import dataclass, asdict
from datetime import datetime, timedelta
from pathlib import Path
import threading
from collections import defaultdict, deque

logger = get_logger(__name__)

@dataclass
class PerformanceMetric:
    """성능 메트릭"""
    timestamp: datetime
    operation: str
    duration: float
    success: bool
    memory_usage: float
    cpu_usage: float
    metadata: Dict[str, Any]

@dataclass
class SystemStats:
    """시스템 통계"""
    timestamp: datetime
    memory_usage: float
    cpu_usage: float
    disk_usage: float
    active_connections: int
    cache_hit_rate: float
    avg_response_time: float

class PerformanceMonitor:
    """성능 모니터"""
    
    def __init__(self, 
                 max_metrics: int = 10000,
                 max_system_stats: int = 1000,
                 monitoring_interval: int = 60):
        """성능 모니터 초기화"""
        self.logger = get_logger(__name__)
        
        # 메트릭 저장소
        self.metrics = deque(maxlen=max_metrics)
        self.system_stats = deque(maxlen=max_system_stats)
        
        # 통계 데이터
        self.operation_stats = defaultdict(list)
        self.hourly_stats = defaultdict(list)
        
        # 모니터링 설정
        self.monitoring_interval = monitoring_interval
        self.is_monitoring = False
        self.monitor_thread = None
        
        # 성능 임계값
        self.thresholds = {
            'response_time_warning': 5.0,  # 5초
            'response_time_critical': 10.0,  # 10초
            'memory_usage_warning': 80.0,  # 80%
            'memory_usage_critical': 90.0,  # 90%
            'cpu_usage_warning': 80.0,  # 80%
            'cpu_usage_critical': 95.0  # 95%
        }
        
        self.logger.info("PerformanceMonitor initialized")
    
    def start_monitoring(self):
        """모니터링 시작"""
        if not self.is_monitoring:
            self.is_monitoring = True
            self.monitor_thread = threading.Thread(target=self._monitor_loop, daemon=True)
            self.monitor_thread.start()
            self.logger.info("Performance monitoring started")
    
    def stop_monitoring(self):
        """모니터링 중지"""
        self.is_monitoring = False
        if self.monitor_thread:
            self.monitor_thread.join(timeout=5)
        self.logger.info("Performance monitoring stopped")
    
    def _monitor_loop(self):
        """모니터링 루프"""
        while self.is_monitoring:
            try:
                self._collect_system_stats()
                time.sleep(self.monitoring_interval)
            except Exception as e:
                self.logger.error(f"Error in monitoring loop: {e}")
                time.sleep(5)
    
    def _collect_system_stats(self):
        """시스템 통계 수집"""
        try:
            # 메모리 사용량
            memory = psutil.virtual_memory()
            memory_usage = memory.percent
            
            # CPU 사용량
            cpu_usage = psutil.cpu_percent(interval=1)
            
            # 디스크 사용량
            disk = psutil.disk_usage('/')
            disk_usage = (disk.used / disk.total) * 100
            
            # 활성 연결 수 (네트워크)
            connections = len(psutil.net_connections())
            
            # 캐시 히트율 계산
            cache_hit_rate = self._calculate_cache_hit_rate()
            
            # 평균 응답 시간 계산
            avg_response_time = self._calculate_avg_response_time()
            
            stats = SystemStats(
                timestamp=datetime.now(),
                memory_usage=memory_usage,
                cpu_usage=cpu_usage,
                disk_usage=disk_usage,
                active_connections=connections,
                cache_hit_rate=cache_hit_rate,
                avg_response_time=avg_response_time
            )
            
            self.system_stats.append(stats)
            
            # 임계값 체크
            self._check_thresholds(stats)
            
        except Exception as e:
            self.logger.error(f"Error collecting system stats: {e}")
    
    def _calculate_cache_hit_rate(self) -> float:
        """캐시 히트율 계산"""
        if not self.metrics:
            return 0.0
        
        recent_metrics = list(self.metrics)[-100:]  # 최근 100개 메트릭
        cache_hits = sum(1 for m in recent_metrics if m.metadata.get('cache_hit', False))
        total_requests = len(recent_metrics)
        
        return (cache_hits / total_requests) * 100 if total_requests > 0 else 0.0
    
    def _calculate_avg_response_time(self) -> float:
        """평균 응답 시간 계산"""
        if not self.metrics:
            return 0.0
        
        recent_metrics = list(self.metrics)[-100:]  # 최근 100개 메트릭
        total_time = sum(m.duration for m in recent_metrics)
        
        return total_time / len(recent_metrics) if recent_metrics else 0.0
    
    def _check_thresholds(self, stats: SystemStats):
        """임계값 체크"""
        warnings = []
        
        if stats.memory_usage > self.thresholds['memory_usage_critical']:
            warnings.append(f"CRITICAL: Memory usage {stats.memory_usage:.1f}%")
        elif stats.memory_usage > self.thresholds['memory_usage_warning']:
            warnings.append(f"WARNING: Memory usage {stats.memory_usage:.1f}%")
        
        if stats.cpu_usage > self.thresholds['cpu_usage_critical']:
            warnings.append(f"CRITICAL: CPU usage {stats.cpu_usage:.1f}%")
        elif stats.cpu_usage > self.thresholds['cpu_usage_warning']:
            warnings.append(f"WARNING: CPU usage {stats.cpu_usage:.1f}%")
        
        if stats.avg_response_time > self.thresholds['response_time_critical']:
            warnings.append(f"CRITICAL: Avg response time {stats.avg_response_time:.2f}s")
        elif stats.avg_response_time > self.thresholds['response_time_warning']:
            warnings.append(f"WARNING: Avg response time {stats.avg_response_time:.2f}s")
        
        for warning in warnings:
            self.logger.warning(warning)
    
    def log_request(self, 
                   operation: str, 
                   duration: float, 
                   success: bool = True,
                   metadata: Optional[Dict[str, Any]] = None):
        """요청 로그 기록"""
        try:
            # 시스템 리소스 사용량
            memory = psutil.virtual_memory()
            cpu = psutil.cpu_percent()
            
            metric = PerformanceMetric(
                timestamp=datetime.now(),
                operation=operation,
                duration=duration,
                success=success,
                memory_usage=memory.percent,
                cpu_usage=cpu,
                metadata=metadata or {}
            )
            
            self.metrics.append(metric)
            
            # 연산별 통계 업데이트
            self.operation_stats[operation].append(metric)
            
            # 시간별 통계 업데이트
            hour_key = metric.timestamp.strftime('%Y-%m-%d-%H')
            self.hourly_stats[hour_key].append(metric)
            
            # 성능 임계값 체크
            if duration > self.thresholds['response_time_critical']:
                self.logger.warning(f"CRITICAL: {operation} took {duration:.2f}s")
            elif duration > self.thresholds['response_time_warning']:
                self.logger.warning(f"WARNING: {operation} took {duration:.2f}s")
            
        except Exception as e:
            self.logger.error(f"Error logging request: {e}")
    
    def get_performance_summary(self, hours: int = 24) -> Dict[str, Any]:
        """성능 요약 반환"""
        try:
            cutoff_time = datetime.now() - timedelta(hours=hours)
            recent_metrics = [m for m in self.metrics if m.timestamp >= cutoff_time]
            
            if not recent_metrics:
                return {"error": "No metrics available"}
            
            # 기본 통계
            total_requests = len(recent_metrics)
            successful_requests = sum(1 for m in recent_metrics if m.success)
            success_rate = (successful_requests / total_requests) * 100
            
            # 응답 시간 통계
            durations = [m.duration for m in recent_metrics]
            avg_duration = sum(durations) / len(durations)
            max_duration = max(durations)
            min_duration = min(durations)
            
            # 메모리 사용량 통계
            memory_usage = [m.memory_usage for m in recent_metrics]
            avg_memory = sum(memory_usage) / len(memory_usage)
            max_memory = max(memory_usage)
            
            # CPU 사용량 통계
            cpu_usage = [m.cpu_usage for m in recent_metrics]
            avg_cpu = sum(cpu_usage) / len(cpu_usage)
            max_cpu = max(cpu_usage)
            
            # 연산별 통계
            operation_stats = {}
            for operation, metrics in self.operation_stats.items():
                op_metrics = [m for m in metrics if m.timestamp >= cutoff_time]
                if op_metrics:
                    op_durations = [m.duration for m in op_metrics]
                    operation_stats[operation] = {
                        'count': len(op_metrics),
                        'avg_duration': sum(op_durations) / len(op_durations),
                        'max_duration': max(op_durations),
                        'success_rate': (sum(1 for m in op_metrics if m.success) / len(op_metrics)) * 100
                    }
            
            return {
                'period_hours': hours,
                'total_requests': total_requests,
                'success_rate': success_rate,
                'response_time': {
                    'avg': avg_duration,
                    'max': max_duration,
                    'min': min_duration
                },
                'memory_usage': {
                    'avg': avg_memory,
                    'max': max_memory
                },
                'cpu_usage': {
                    'avg': avg_cpu,
                    'max': max_cpu
                },
                'operation_stats': operation_stats,
                'cache_hit_rate': self._calculate_cache_hit_rate(),
                'thresholds': self.thresholds
            }
            
        except Exception as e:
            self.logger.error(f"Error generating performance summary: {e}")
            return {"error": str(e)}
    
    def get_system_health(self) -> Dict[str, Any]:
        """시스템 상태 반환"""
        try:
            if not self.system_stats:
                return {"error": "No system stats available"}
            
            latest_stats = self.system_stats[-1]
            
            # 상태 레벨 결정
            health_status = "HEALTHY"
            if (latest_stats.memory_usage > self.thresholds['memory_usage_critical'] or
                latest_stats.cpu_usage > self.thresholds['cpu_usage_critical'] or
                latest_stats.avg_response_time > self.thresholds['response_time_critical']):
                health_status = "CRITICAL"
            elif (latest_stats.memory_usage > self.thresholds['memory_usage_warning'] or
                  latest_stats.cpu_usage > self.thresholds['cpu_usage_warning'] or
                  latest_stats.avg_response_time > self.thresholds['response_time_warning']):
                health_status = "WARNING"
            
            return {
                'status': health_status,
                'timestamp': latest_stats.timestamp.isoformat(),
                'memory_usage': latest_stats.memory_usage,
                'cpu_usage': latest_stats.cpu_usage,
                'disk_usage': latest_stats.disk_usage,
                'active_connections': latest_stats.active_connections,
                'cache_hit_rate': latest_stats.cache_hit_rate,
                'avg_response_time': latest_stats.avg_response_time,
                'thresholds': self.thresholds
            }
            
        except Exception as e:
            self.logger.error(f"Error getting system health: {e}")
            return {"error": str(e)}
    
    def export_metrics(self, filepath: str, hours: int = 24):
        """메트릭 내보내기"""
        try:
            cutoff_time = datetime.now() - timedelta(hours=hours)
            recent_metrics = [m for m in self.metrics if m.timestamp >= cutoff_time]
            
            export_data = {
                'export_timestamp': datetime.now().isoformat(),
                'period_hours': hours,
                'metrics': [asdict(m) for m in recent_metrics],
                'system_stats': [asdict(s) for s in self.system_stats if s.timestamp >= cutoff_time]
            }
            
            with open(filepath, 'w', encoding='utf-8') as f:
                json.dump(export_data, f, ensure_ascii=False, indent=2, default=str)
            
            self.logger.info(f"Metrics exported to {filepath}")
            
        except Exception as e:
            self.logger.error(f"Error exporting metrics: {e}")
    
    def clear_old_metrics(self, days: int = 7):
        """오래된 메트릭 정리"""
        try:
            cutoff_time = datetime.now() - timedelta(days=days)
            
            # 메트릭 정리
            self.metrics = deque([m for m in self.metrics if m.timestamp >= cutoff_time], 
                               maxlen=self.metrics.maxlen)
            
            # 시스템 통계 정리
            self.system_stats = deque([s for s in self.system_stats if s.timestamp >= cutoff_time],
                                    maxlen=self.system_stats.maxlen)
            
            # 연산별 통계 정리
            for operation in self.operation_stats:
                self.operation_stats[operation] = [m for m in self.operation_stats[operation] 
                                                 if m.timestamp >= cutoff_time]
            
            # 시간별 통계 정리
            for hour_key in list(self.hourly_stats.keys()):
                if datetime.strptime(hour_key, '%Y-%m-%d-%H') < cutoff_time:
                    del self.hourly_stats[hour_key]
            
            self.logger.info(f"Cleared metrics older than {days} days")
            
        except Exception as e:
            self.logger.error(f"Error clearing old metrics: {e}")

class PerformanceContext:
    """성능 측정 컨텍스트 매니저"""
    
    def __init__(self, monitor: PerformanceMonitor, operation: str, metadata: Optional[Dict[str, Any]] = None):
        self.monitor = monitor
        self.operation = operation
        self.metadata = metadata or {}
        self.start_time = None
        self.success = True
    
    def __enter__(self):
        self.start_time = time.time()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.start_time:
            duration = time.time() - self.start_time
            self.success = exc_type is None
            
            self.monitor.log_request(
                operation=self.operation,
                duration=duration,
                success=self.success,
                metadata=self.metadata
            )
    
    def add_metadata(self, key: str, value: Any):
        """메타데이터 추가"""
        self.metadata[key] = value
