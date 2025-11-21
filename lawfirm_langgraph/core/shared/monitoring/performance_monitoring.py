# -*- coding: utf-8 -*-
"""
성능 모니터링 및 알림 시스템
실시간 성능 모니터링과 알림 기능을 제공합니다.
"""

import asyncio
import logging
try:
    from lawfirm_langgraph.core.utils.logger import get_logger
except ImportError:
    from core.utils.logger import get_logger
import json
import time
from typing import Dict, Any, List, Optional, Callable
from datetime import datetime, timedelta
from dataclasses import dataclass, asdict
from enum import Enum
import threading
from pathlib import Path

logger = get_logger(__name__)

class AlertLevel(Enum):
    """알림 레벨"""
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"

class MetricType(Enum):
    """메트릭 유형"""
    RESPONSE_TIME = "response_time"
    MEMORY_USAGE = "memory_usage"
    CPU_USAGE = "cpu_usage"
    ERROR_RATE = "error_rate"
    SUCCESS_RATE = "success_rate"
    THROUGHPUT = "throughput"

@dataclass
class Alert:
    """알림 데이터 클래스"""
    id: str
    timestamp: datetime
    level: AlertLevel
    metric_type: MetricType
    message: str
    value: float
    threshold: float
    resolved: bool = False
    resolved_at: Optional[datetime] = None

@dataclass
class MetricThreshold:
    """메트릭 임계값 데이터 클래스"""
    metric_type: MetricType
    warning_threshold: float
    error_threshold: float
    critical_threshold: float
    enabled: bool = True

class AlertManager:
    """알림 관리 클래스"""
    
    def __init__(self):
        self.logger = get_logger(__name__)
        self.alerts = []
        self.thresholds = self._get_default_thresholds()
        self.alert_callbacks = []
        self.alert_history = []
        
    def _get_default_thresholds(self) -> Dict[MetricType, MetricThreshold]:
        """기본 임계값 설정"""
        return {
            MetricType.RESPONSE_TIME: MetricThreshold(
                metric_type=MetricType.RESPONSE_TIME,
                warning_threshold=3.0,  # 3초
                error_threshold=5.0,    # 5초
                critical_threshold=10.0 # 10초
            ),
            MetricType.MEMORY_USAGE: MetricThreshold(
                metric_type=MetricType.MEMORY_USAGE,
                warning_threshold=80.0,  # 80%
                error_threshold=90.0,    # 90%
                critical_threshold=95.0   # 95%
            ),
            MetricType.CPU_USAGE: MetricThreshold(
                metric_type=MetricType.CPU_USAGE,
                warning_threshold=80.0,  # 80%
                error_threshold=90.0,    # 90%
                critical_threshold=95.0   # 95%
            ),
            MetricType.ERROR_RATE: MetricThreshold(
                metric_type=MetricType.ERROR_RATE,
                warning_threshold=5.0,   # 5%
                error_threshold=10.0,    # 10%
                critical_threshold=20.0  # 20%
            ),
            MetricType.SUCCESS_RATE: MetricThreshold(
                metric_type=MetricType.SUCCESS_RATE,
                warning_threshold=95.0,  # 95%
                error_threshold=90.0,    # 90%
                critical_threshold=80.0  # 80%
            ),
            MetricType.THROUGHPUT: MetricThreshold(
                metric_type=MetricType.THROUGHPUT,
                warning_threshold=0.1,  # 0.1 req/s
                error_threshold=0.05,    # 0.05 req/s
                critical_threshold=0.01   # 0.01 req/s
            )
        }
    
    def add_alert_callback(self, callback: Callable[[Alert], None]):
        """알림 콜백 추가"""
        self.alert_callbacks.append(callback)
    
    def check_metric(self, metric_type: MetricType, value: float) -> Optional[Alert]:
        """메트릭 체크 및 알림 생성"""
        if metric_type not in self.thresholds:
            return None
        
        threshold = self.thresholds[metric_type]
        if not threshold.enabled:
            return None
        
        # 알림 레벨 결정
        level = None
        if value >= threshold.critical_threshold:
            level = AlertLevel.CRITICAL
        elif value >= threshold.error_threshold:
            level = AlertLevel.ERROR
        elif value >= threshold.warning_threshold:
            level = AlertLevel.WARNING
        
        if level is None:
            return None
        
        # 기존 알림 확인 (같은 메트릭에 대한 중복 알림 방지)
        existing_alert = self._find_active_alert(metric_type)
        if existing_alert and existing_alert.level.value >= level.value:
            return None
        
        # 새 알림 생성
        alert = Alert(
            id=f"{metric_type.value}_{int(time.time())}",
            timestamp=datetime.now(),
            level=level,
            metric_type=metric_type,
            message=f"{metric_type.value} threshold exceeded: {value:.2f} >= {threshold.warning_threshold:.2f}",
            value=value,
            threshold=threshold.warning_threshold
        )
        
        # 알림 저장
        self.alerts.append(alert)
        self.alert_history.append(alert)
        
        # 알림 콜백 실행
        for callback in self.alert_callbacks:
            try:
                callback(alert)
            except Exception as e:
                self.logger.error(f"Alert callback failed: {e}")
        
        # 로깅
        self.logger.warning(f"Alert triggered: {alert.message}")
        
        return alert
    
    def _find_active_alert(self, metric_type: MetricType) -> Optional[Alert]:
        """활성 알림 찾기"""
        for alert in reversed(self.alerts):
            if alert.metric_type == metric_type and not alert.resolved:
                return alert
        return None
    
    def resolve_alert(self, alert_id: str):
        """알림 해결"""
        for alert in self.alerts:
            if alert.id == alert_id and not alert.resolved:
                alert.resolved = True
                alert.resolved_at = datetime.now()
                self.logger.info(f"Alert resolved: {alert_id}")
                break
    
    def get_active_alerts(self) -> List[Alert]:
        """활성 알림 반환"""
        return [alert for alert in self.alerts if not alert.resolved]
    
    def get_alert_stats(self) -> Dict[str, Any]:
        """알림 통계 반환"""
        active_alerts = self.get_active_alerts()
        
        stats = {
            "total_alerts": len(self.alerts),
            "active_alerts": len(active_alerts),
            "resolved_alerts": len([a for a in self.alerts if a.resolved]),
            "alerts_by_level": {},
            "alerts_by_metric": {}
        }
        
        # 레벨별 통계
        for level in AlertLevel:
            count = len([a for a in self.alerts if a.level == level])
            stats["alerts_by_level"][level.value] = count
        
        # 메트릭별 통계
        for metric_type in MetricType:
            count = len([a for a in self.alerts if a.metric_type == metric_type])
            stats["alerts_by_metric"][metric_type.value] = count
        
        return stats

class PerformanceMonitor:
    """성능 모니터링 클래스"""
    
    def __init__(self, check_interval: int = 30):
        """
        성능 모니터링기 초기화
        
        Args:
            check_interval: 체크 간격 (초)
        """
        self.check_interval = check_interval
        self.logger = get_logger(__name__)
        self.alert_manager = AlertManager()
        self.monitoring = False
        self.monitor_thread = None
        
        # 메트릭 저장
        self.metrics_history = []
        self.current_metrics = {}
        
        # 통계
        self.request_count = 0
        self.error_count = 0
        self.start_time = datetime.now()
        
    def start_monitoring(self):
        """모니터링 시작"""
        if self.monitoring:
            return
        
        self.monitoring = True
        self.monitor_thread = threading.Thread(target=self._monitoring_loop, daemon=True)
        self.monitor_thread.start()
        self.logger.info("Performance monitoring started")
    
    def stop_monitoring(self):
        """모니터링 중지"""
        self.monitoring = False
        if self.monitor_thread:
            self.monitor_thread.join()
        self.logger.info("Performance monitoring stopped")
    
    def _monitoring_loop(self):
        """모니터링 루프"""
        while self.monitoring:
            try:
                self._collect_metrics()
                self._check_thresholds()
                time.sleep(self.check_interval)
            except Exception as e:
                self.logger.error(f"Monitoring loop error: {e}")
                time.sleep(self.check_interval)
    
    def _collect_metrics(self):
        """메트릭 수집"""
        import psutil
        
        # 시스템 메트릭
        memory_percent = psutil.virtual_memory().percent
        cpu_percent = psutil.cpu_percent()
        
        # 애플리케이션 메트릭
        uptime = (datetime.now() - self.start_time).total_seconds()
        throughput = self.request_count / uptime if uptime > 0 else 0
        error_rate = (self.error_count / self.request_count * 100) if self.request_count > 0 else 0
        success_rate = 100 - error_rate
        
        # 평균 응답 시간 (실제 구현에서는 응답 시간 히스토리에서 계산)
        avg_response_time = 0.0  # TODO: 실제 응답 시간 계산
        
        # 메트릭 저장
        metrics = {
            MetricType.MEMORY_USAGE: memory_percent,
            MetricType.CPU_USAGE: cpu_percent,
            MetricType.THROUGHPUT: throughput,
            MetricType.ERROR_RATE: error_rate,
            MetricType.SUCCESS_RATE: success_rate,
            MetricType.RESPONSE_TIME: avg_response_time
        }
        
        self.current_metrics = metrics
        self.metrics_history.append({
            "timestamp": datetime.now().isoformat(),
            "metrics": metrics
        })
        
        # 최근 1000개 기록만 유지
        if len(self.metrics_history) > 1000:
            self.metrics_history = self.metrics_history[-1000:]
    
    def _check_thresholds(self):
        """임계값 체크"""
        for metric_type, value in self.current_metrics.items():
            self.alert_manager.check_metric(metric_type, value)
    
    def log_request(self, response_time: float, success: bool = True):
        """요청 로깅"""
        self.request_count += 1
        if not success:
            self.error_count += 1
        
        # 응답 시간 체크
        self.alert_manager.check_metric(MetricType.RESPONSE_TIME, response_time)
    
    def add_alert_callback(self, callback: Callable[[Alert], None]):
        """알림 콜백 추가"""
        self.alert_manager.add_alert_callback(callback)
    
    def get_current_metrics(self) -> Dict[str, Any]:
        """현재 메트릭 반환"""
        return {
            "timestamp": datetime.now().isoformat(),
            "metrics": self.current_metrics,
            "uptime": (datetime.now() - self.start_time).total_seconds(),
            "request_count": self.request_count,
            "error_count": self.error_count
        }
    
    def get_metrics_history(self, hours: int = 24) -> List[Dict[str, Any]]:
        """메트릭 히스토리 반환"""
        cutoff_time = datetime.now() - timedelta(hours=hours)
        return [
            record for record in self.metrics_history
            if datetime.fromisoformat(record["timestamp"]) >= cutoff_time
        ]
    
    def get_alert_stats(self) -> Dict[str, Any]:
        """알림 통계 반환"""
        return self.alert_manager.get_alert_stats()
    
    def get_health_status(self) -> Dict[str, Any]:
        """헬스 상태 반환"""
        active_alerts = self.alert_manager.get_active_alerts()
        
        # 상태 결정
        if any(alert.level == AlertLevel.CRITICAL for alert in active_alerts):
            status = "critical"
        elif any(alert.level == AlertLevel.ERROR for alert in active_alerts):
            status = "error"
        elif any(alert.level == AlertLevel.WARNING for alert in active_alerts):
            status = "warning"
        else:
            status = "healthy"
        
        return {
            "status": status,
            "active_alerts": len(active_alerts),
            "current_metrics": self.current_metrics,
            "uptime": (datetime.now() - self.start_time).total_seconds(),
            "timestamp": datetime.now().isoformat()
        }

class NotificationService:
    """알림 서비스 클래스"""
    
    def __init__(self):
        self.logger = get_logger(__name__)
        self.notification_methods = []
    
    def add_notification_method(self, method: Callable[[Alert], None]):
        """알림 방법 추가"""
        self.notification_methods.append(method)
    
    def send_notification(self, alert: Alert):
        """알림 전송"""
        for method in self.notification_methods:
            try:
                method(alert)
            except Exception as e:
                self.logger.error(f"Notification method failed: {e}")
    
    def log_notification(self, alert: Alert):
        """로그 알림"""
        self.logger.warning(f"ALERT [{alert.level.value.upper()}]: {alert.message}")
    
    def file_notification(self, alert: Alert, file_path: str = "logs/alerts.json"):
        """파일 알림"""
        try:
            alert_data = asdict(alert)
            alert_data["timestamp"] = alert.timestamp.isoformat()
            if alert.resolved_at:
                alert_data["resolved_at"] = alert.resolved_at.isoformat()
            
            Path(file_path).parent.mkdir(parents=True, exist_ok=True)
            
            with open(file_path, "a", encoding="utf-8") as f:
                f.write(json.dumps(alert_data, ensure_ascii=False) + "\n")
                
        except Exception as e:
            self.logger.error(f"File notification failed: {e}")

# 전역 인스턴스
performance_monitor = PerformanceMonitor()
notification_service = NotificationService()

# 기본 알림 방법 설정
notification_service.add_notification_method(notification_service.log_notification)
notification_service.add_notification_method(notification_service.file_notification)

# 성능 모니터에 알림 서비스 연결
performance_monitor.add_alert_callback(notification_service.send_notification)

def get_performance_monitor() -> PerformanceMonitor:
    """성능 모니터 인스턴스 반환"""
    return performance_monitor

def get_notification_service() -> NotificationService:
    """알림 서비스 인스턴스 반환"""
    return notification_service

def start_monitoring():
    """모니터링 시작"""
    performance_monitor.start_monitoring()

def stop_monitoring():
    """모니터링 중지"""
    performance_monitor.stop_monitoring()
