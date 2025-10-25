# -*- coding: utf-8 -*-
"""
Performance Monitor
성능 모니터링 시스템

시스템의 성능 메트릭을 수집하고 분석하는 모듈
"""

import logging
import time
import psutil
import threading
from typing import Dict, List, Any, Optional
from dataclasses import dataclass, field
from datetime import datetime
import json
from pathlib import Path


@dataclass
class PerformanceMetric:
    """성능 메트릭 데이터 클래스"""
    timestamp: float
    query_type: str
    processing_time: float
    confidence: float
    response_length: int
    memory_usage_mb: float
    cpu_usage_percent: float
    success: bool
    error_message: Optional[str] = None


@dataclass
class SystemMetrics:
    """시스템 메트릭 데이터 클래스"""
    timestamp: float
    memory_usage_mb: float
    cpu_usage_percent: float
    disk_usage_percent: float
    active_threads: int


class PerformanceMonitor:
    """성능 모니터링 클래스"""
    
    def __init__(self, config=None):
        self.config = config
        self.logger = logging.getLogger(__name__)
        
        # 메트릭 저장소
        self.metrics: List[PerformanceMetric] = []
        self.system_metrics: List[SystemMetrics] = []
        
        # 설정
        self.max_metrics_count = 1000  # 최대 저장 메트릭 수
        self.metrics_file = Path("data/ml_metrics/performance_metrics.json")
        self.system_metrics_file = Path("data/ml_metrics/system_metrics.json")
        
        # 스레드 안전성을 위한 락
        self._metrics_lock = threading.Lock()
        self._system_lock = threading.Lock()
        
        # 시스템 모니터링 활성화 여부
        self.system_monitoring_enabled = True
        self._monitoring_thread = None
        self._stop_monitoring = False
        
        # 메트릭 파일 디렉토리 생성
        self.metrics_file.parent.mkdir(parents=True, exist_ok=True)
        
        # 기존 메트릭 로드
        self._load_metrics()
        
        # 시스템 모니터링 시작
        if self.system_monitoring_enabled:
            self._start_system_monitoring()
        
        self.logger.info("Performance Monitor 초기화 완료")
    
    def log_response_metrics(self, query_type: str, processing_time: float, 
                           confidence: float, response_length: int, 
                           success: bool = True, error_message: str = None):
        """응답 메트릭 로깅 (긴급 최적화 버전)"""
        try:
            # 시스템 리소스 정보 수집
            process = psutil.Process()
            memory_mb = process.memory_info().rss / (1024**2)
            cpu_percent = process.cpu_percent()
            
            # 메트릭 생성
            metric = PerformanceMetric(
                timestamp=time.time(),
                query_type=query_type,
                processing_time=processing_time,
                confidence=confidence,
                response_length=response_length,
                memory_usage_mb=memory_mb,
                cpu_usage_percent=cpu_percent,
                success=success,
                error_message=error_message
            )
            
            # 메트릭 저장
            with self._metrics_lock:
                self.metrics.append(metric)
                
                # 최대 개수 초과 시 오래된 메트릭 제거 (더 적게 유지)
                if len(self.metrics) > 50:  # 기존 max_metrics_count에서 50으로 축소
                    self.metrics = self.metrics[-50:]
            
            # 성능 경고 체크 (긴급 최적화)
            if processing_time > 10.0:  # 10초 초과 시 경고
                self.logger.warning(f"느린 응답 감지: {processing_time:.2f}초 (타입: {query_type})")
            
            if memory_mb > 1000:  # 1GB 초과 시 경고
                self.logger.warning(f"높은 메모리 사용량: {memory_mb:.1f}MB")
            
            # 주기적으로 파일에 저장 (더 자주 저장)
            if len(self.metrics) % 5 == 0:  # 5개마다 저장 (기존 10개에서 축소)
                self._save_metrics()
                
        except Exception as e:
            self.logger.error(f"Failed to log response metrics: {e}")
    
    def _collect_system_metrics(self):
        """시스템 메트릭 수집"""
        try:
            # 메모리 사용량
            memory = psutil.virtual_memory()
            memory_mb = memory.used / (1024**2)
            
            # CPU 사용량
            cpu_percent = psutil.cpu_percent(interval=1)
            
            # 디스크 사용량
            disk = psutil.disk_usage('/')
            disk_percent = (disk.used / disk.total) * 100
            
            # 활성 스레드 수
            active_threads = threading.active_count()
            
            # 시스템 메트릭 생성
            system_metric = SystemMetrics(
                timestamp=time.time(),
                memory_usage_mb=memory_mb,
                cpu_usage_percent=cpu_percent,
                disk_usage_percent=disk_percent,
                active_threads=active_threads
            )
            
            # 시스템 메트릭 저장
            with self._system_lock:
                self.system_metrics.append(system_metric)
                
                # 최대 개수 초과 시 오래된 메트릭 제거
                if len(self.system_metrics) > 100:  # 시스템 메트릭은 더 적게 저장
                    self.system_metrics = self.system_metrics[-100:]
            
        except Exception as e:
            self.logger.error(f"Failed to collect system metrics: {e}")
    
    def _start_system_monitoring(self):
        """시스템 모니터링 시작"""
        def monitor_loop():
            while not self._stop_monitoring:
                self._collect_system_metrics()
                time.sleep(60)  # 1분마다 수집
        
        self._monitoring_thread = threading.Thread(target=monitor_loop, daemon=True)
        self._monitoring_thread.start()
        self.logger.info("시스템 모니터링 시작됨")
    
    def stop_system_monitoring(self):
        """시스템 모니터링 중지"""
        self._stop_monitoring = True
        if self._monitoring_thread:
            self._monitoring_thread.join(timeout=5)
        self.logger.info("시스템 모니터링 중지됨")
    
    def get_performance_summary(self) -> Dict[str, Any]:
        """성능 요약 반환"""
        with self._metrics_lock:
            if not self.metrics:
                return {"message": "No metrics available"}
            
            # 쿼리 타입별 통계
            query_type_stats = {}
            for metric in self.metrics:
                query_type = metric.query_type
                if query_type not in query_type_stats:
                    query_type_stats[query_type] = {
                        'count': 0,
                        'total_processing_time': 0,
                        'total_confidence': 0,
                        'total_response_length': 0,
                        'success_count': 0,
                        'error_count': 0
                    }
                
                stats = query_type_stats[query_type]
                stats['count'] += 1
                stats['total_processing_time'] += metric.processing_time
                stats['total_confidence'] += metric.confidence
                stats['total_response_length'] += metric.response_length
                
                if metric.success:
                    stats['success_count'] += 1
                else:
                    stats['error_count'] += 1
            
            # 평균 계산
            for query_type, stats in query_type_stats.items():
                count = stats['count']
                stats['avg_processing_time'] = stats['total_processing_time'] / count
                stats['avg_confidence'] = stats['total_confidence'] / count
                stats['avg_response_length'] = stats['total_response_length'] / count
                stats['success_rate'] = stats['success_count'] / count
            
            # 전체 통계
            total_metrics = len(self.metrics)
            total_success = sum(1 for m in self.metrics if m.success)
            total_processing_time = sum(m.processing_time for m in self.metrics)
            total_confidence = sum(m.confidence for m in self.metrics)
            
            return {
                'total_queries': total_metrics,
                'success_rate': total_success / total_metrics if total_metrics > 0 else 0,
                'avg_processing_time': total_processing_time / total_metrics if total_metrics > 0 else 0,
                'avg_confidence': total_confidence / total_metrics if total_metrics > 0 else 0,
                'query_type_stats': query_type_stats,
                'last_updated': datetime.fromtimestamp(self.metrics[-1].timestamp).isoformat()
            }
    
    def get_system_summary(self) -> Dict[str, Any]:
        """시스템 요약 반환"""
        with self._system_lock:
            if not self.system_metrics:
                return {"message": "No system metrics available"}
            
            latest = self.system_metrics[-1]
            avg_memory = sum(m.memory_usage_mb for m in self.system_metrics) / len(self.system_metrics)
            avg_cpu = sum(m.cpu_usage_percent for m in self.system_metrics) / len(self.system_metrics)
            avg_disk = sum(m.disk_usage_percent for m in self.system_metrics) / len(self.system_metrics)
            
            return {
                'current_memory_mb': latest.memory_usage_mb,
                'current_cpu_percent': latest.cpu_usage_percent,
                'current_disk_percent': latest.disk_usage_percent,
                'current_threads': latest.active_threads,
                'avg_memory_mb': avg_memory,
                'avg_cpu_percent': avg_cpu,
                'avg_disk_percent': avg_disk,
                'last_updated': datetime.fromtimestamp(latest.timestamp).isoformat()
            }
    
    def _save_metrics(self):
        """메트릭을 파일에 저장"""
        try:
            with self._metrics_lock:
                metrics_data = [
                    {
                        'timestamp': m.timestamp,
                        'query_type': m.query_type,
                        'processing_time': m.processing_time,
                        'confidence': m.confidence,
                        'response_length': m.response_length,
                        'memory_usage_mb': m.memory_usage_mb,
                        'cpu_usage_percent': m.cpu_usage_percent,
                        'success': m.success,
                        'error_message': m.error_message
                    }
                    for m in self.metrics
                ]
            
            with open(self.metrics_file, 'w', encoding='utf-8') as f:
                json.dump(metrics_data, f, ensure_ascii=False, indent=2)
                
        except Exception as e:
            self.logger.error(f"Failed to save metrics: {e}")
    
    def _load_metrics(self):
        """파일에서 메트릭 로드"""
        try:
            if self.metrics_file.exists():
                with open(self.metrics_file, 'r', encoding='utf-8') as f:
                    metrics_data = json.load(f)
                
                self.metrics = [
                    PerformanceMetric(
                        timestamp=m['timestamp'],
                        query_type=m['query_type'],
                        processing_time=m['processing_time'],
                        confidence=m['confidence'],
                        response_length=m['response_length'],
                        memory_usage_mb=m['memory_usage_mb'],
                        cpu_usage_percent=m['cpu_usage_percent'],
                        success=m['success'],
                        error_message=m.get('error_message')
                    )
                    for m in metrics_data
                ]
                
                self.logger.info(f"Loaded {len(self.metrics)} performance metrics")
                
        except Exception as e:
            self.logger.error(f"Failed to load metrics: {e}")
    
    def export_metrics(self, file_path: str = None):
        """메트릭을 외부 파일로 내보내기"""
        if file_path is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            file_path = f"data/ml_metrics/performance_export_{timestamp}.json"
        
        try:
            summary = self.get_performance_summary()
            system_summary = self.get_system_summary()
            
            export_data = {
                'performance_summary': summary,
                'system_summary': system_summary,
                'export_timestamp': datetime.now().isoformat()
            }
            
            with open(file_path, 'w', encoding='utf-8') as f:
                json.dump(export_data, f, ensure_ascii=False, indent=2)
            
            self.logger.info(f"Metrics exported to {file_path}")
            return file_path
            
        except Exception as e:
            self.logger.error(f"Failed to export metrics: {e}")
            return None
    
    def cleanup_old_metrics(self, days: int = 7):
        """오래된 메트릭 정리"""
        cutoff_time = time.time() - (days * 24 * 60 * 60)
        
        with self._metrics_lock:
            original_count = len(self.metrics)
            self.metrics = [m for m in self.metrics if m.timestamp > cutoff_time]
            removed_count = original_count - len(self.metrics)
            
        with self._system_lock:
            original_system_count = len(self.system_metrics)
            self.system_metrics = [m for m in self.system_metrics if m.timestamp > cutoff_time]
            removed_system_count = original_system_count - len(self.system_metrics)
        
        self.logger.info(f"Cleaned up {removed_count} performance metrics and {removed_system_count} system metrics")
        self._save_metrics()
    
    def __del__(self):
        """소멸자 - 모니터링 중지 및 메트릭 저장"""
        self.stop_system_monitoring()
        self._save_metrics()
