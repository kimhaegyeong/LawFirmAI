# -*- coding: utf-8 -*-
"""
Real-time Memory Monitoring System
실시간 메모리 모니터링 시스템
"""

import time
import threading
import queue
import json
import psutil
from typing import Dict, List, Any, Optional, Callable
from datetime import datetime, timedelta
from dataclasses import dataclass, asdict
from collections import deque
import matplotlib.pyplot as plt
import io
import base64
from ..logger import get_logger
from ..memory_manager import MemoryMetrics, MemoryAlert
from ..weakref_cleanup import get_weakref_registry

logger = get_logger(__name__)


@dataclass
class MonitoringConfig:
    """모니터링 설정"""
    interval_seconds: int = 30
    max_history_size: int = 1000
    alert_threshold_mb: float = 800.0
    critical_threshold_mb: float = 1000.0
    enable_auto_cleanup: bool = True
    enable_trend_analysis: bool = True
    enable_visualization: bool = True


@dataclass
class MonitoringData:
    """모니터링 데이터"""
    timestamp: datetime
    memory_mb: float
    cpu_percent: float
    process_count: int
    thread_count: int
    gc_counts: Dict[int, int]
    weakref_stats: Dict[str, Any]


class RealTimeMemoryMonitor:
    """실시간 메모리 모니터"""

    def __init__(self, config: Optional[MonitoringConfig] = None):
        self.config = config or MonitoringConfig()
        self.logger = get_logger(__name__)

        # 모니터링 데이터 저장소
        self.monitoring_data: deque = deque(maxlen=self.config.max_history_size)

        # 알림 큐
        self.alert_queue: queue.Queue = queue.Queue()

        # 콜백 함수들
        self.alert_callbacks: List[Callable[[MemoryAlert], None]] = []
        self.data_callbacks: List[Callable[[MonitoringData], None]] = []

        # 모니터링 스레드
        self.monitoring_thread: Optional[threading.Thread] = None
        self.is_monitoring = False

        # 통계 데이터
        self.stats = {
            'total_samples': 0,
            'alert_count': 0,
            'cleanup_count': 0,
            'start_time': None,
            'last_sample_time': None
        }

        # WeakRef 등록소 참조
        self.weakref_registry = get_weakref_registry()

        self.logger.info("RealTimeMemoryMonitor 초기화 완료")

    def start_monitoring(self):
        """모니터링 시작"""
        if self.is_monitoring:
            self.logger.warning("모니터링이 이미 실행 중입니다.")
            return

        self.is_monitoring = True
        self.stats['start_time'] = datetime.now()

        self.monitoring_thread = threading.Thread(
            target=self._monitoring_loop,
            daemon=True,
            name="MemoryMonitor"
        )
        self.monitoring_thread.start()

        self.logger.info("실시간 메모리 모니터링 시작")

    def stop_monitoring(self):
        """모니터링 중지"""
        self.is_monitoring = False

        if self.monitoring_thread:
            self.monitoring_thread.join(timeout=5)

        self.logger.info("실시간 메모리 모니터링 중지")

    def _monitoring_loop(self):
        """모니터링 루프"""
        while self.is_monitoring:
            try:
                # 모니터링 데이터 수집
                data = self._collect_monitoring_data()
                self.monitoring_data.append(data)

                # 통계 업데이트
                self.stats['total_samples'] += 1
                self.stats['last_sample_time'] = data.timestamp

                # 알림 체크
                self._check_alerts(data)

                # 데이터 콜백 실행
                for callback in self.data_callbacks:
                    try:
                        callback(data)
                    except Exception as e:
                        self.logger.error(f"데이터 콜백 실행 실패: {e}")

                # 자동 정리 (필요시)
                if self.config.enable_auto_cleanup:
                    self._auto_cleanup_if_needed(data)

                time.sleep(self.config.interval_seconds)

            except Exception as e:
                self.logger.error(f"모니터링 루프 오류: {e}")
                time.sleep(self.config.interval_seconds)

    def _collect_monitoring_data(self) -> MonitoringData:
        """모니터링 데이터 수집"""
        try:
            process = psutil.Process()
            memory_info = process.memory_info()

            # 가비지 컬렉션 통계
            import gc
            gc_counts = {}
            for generation in range(3):
                gc_counts[generation] = gc.get_count()[generation]

            # WeakRef 통계
            weakref_stats = self.weakref_registry.get_registry_stats()

            return MonitoringData(
                timestamp=datetime.now(),
                memory_mb=memory_info.rss / 1024 / 1024,
                cpu_percent=process.cpu_percent(),
                process_count=len(psutil.pids()),
                thread_count=process.num_threads(),
                gc_counts=gc_counts,
                weakref_stats=weakref_stats
            )

        except Exception as e:
            self.logger.error(f"모니터링 데이터 수집 실패: {e}")
            # 기본값 반환
            return MonitoringData(
                timestamp=datetime.now(),
                memory_mb=0.0,
                cpu_percent=0.0,
                process_count=0,
                thread_count=0,
                gc_counts={},
                weakref_stats={}
            )

    def _check_alerts(self, data: MonitoringData):
        """알림 체크"""
        try:
            alerts = []

            # 메모리 사용량 체크
            if data.memory_mb >= self.config.critical_threshold_mb:
                alert = MemoryAlert(
                    alert_type='critical_memory',
                    message=f"메모리 사용량이 위험 수준입니다: {data.memory_mb:.1f}MB",
                    severity='critical',
                    timestamp=data.timestamp,
                    metrics=None  # MemoryMetrics는 별도로 생성 필요
                )
                alerts.append(alert)

            elif data.memory_mb >= self.config.alert_threshold_mb:
                alert = MemoryAlert(
                    alert_type='high_memory',
                    message=f"메모리 사용량이 높습니다: {data.memory_mb:.1f}MB",
                    severity='high',
                    timestamp=data.timestamp,
                    metrics=None
                )
                alerts.append(alert)

            # CPU 사용률 체크
            if data.cpu_percent > 90.0:
                alert = MemoryAlert(
                    alert_type='high_cpu',
                    message=f"CPU 사용률이 높습니다: {data.cpu_percent:.1f}%",
                    severity='medium',
                    timestamp=data.timestamp,
                    metrics=None
                )
                alerts.append(alert)

            # 알림 처리
            for alert in alerts:
                self._process_alert(alert)

        except Exception as e:
            self.logger.error(f"알림 체크 실패: {e}")

    def _process_alert(self, alert: MemoryAlert):
        """알림 처리"""
        try:
            # 알림 큐에 추가
            self.alert_queue.put(alert)

            # 통계 업데이트
            self.stats['alert_count'] += 1

            # 콜백 함수 실행
            for callback in self.alert_callbacks:
                try:
                    callback(alert)
                except Exception as e:
                    self.logger.error(f"알림 콜백 실행 실패: {e}")

            self.logger.warning(f"메모리 알림: {alert.message}")

        except Exception as e:
            self.logger.error(f"알림 처리 실패: {e}")

    def _auto_cleanup_if_needed(self, data: MonitoringData):
        """필요시 자동 정리"""
        try:
            if data.memory_mb >= self.config.critical_threshold_mb:
                self.logger.info("자동 메모리 정리 실행")

                # WeakRef 정리
                cleanup_result = self.weakref_registry.force_cleanup()

                # 가비지 컬렉션
                import gc
                collected = gc.collect()

                # 통계 업데이트
                self.stats['cleanup_count'] += 1

                self.logger.info(f"자동 정리 완료: {collected}개 객체 수집")

        except Exception as e:
            self.logger.error(f"자동 정리 실패: {e}")

    def add_alert_callback(self, callback: Callable[[MemoryAlert], None]):
        """알림 콜백 추가"""
        self.alert_callbacks.append(callback)

    def add_data_callback(self, callback: Callable[[MonitoringData], None]):
        """데이터 콜백 추가"""
        self.data_callbacks.append(callback)

    def get_current_status(self) -> Dict[str, Any]:
        """현재 상태 반환"""
        try:
            if not self.monitoring_data:
                return {'status': 'no_data', 'message': '모니터링 데이터가 없습니다.'}

            latest_data = self.monitoring_data[-1]

            return {
                'status': 'monitoring' if self.is_monitoring else 'stopped',
                'latest_data': asdict(latest_data),
                'stats': self.stats.copy(),
                'config': asdict(self.config),
                'data_points': len(self.monitoring_data),
                'alert_queue_size': self.alert_queue.qsize(),
                'timestamp': datetime.now()
            }

        except Exception as e:
            self.logger.error(f"상태 조회 실패: {e}")
            return {'status': 'error', 'error': str(e)}

    def get_memory_trend(self, hours: int = 1) -> Dict[str, Any]:
        """메모리 사용량 추세 분석"""
        try:
            if not self.config.enable_trend_analysis:
                return {'trend_analysis_disabled': True}

            cutoff_time = datetime.now() - timedelta(hours=hours)
            recent_data = [d for d in self.monitoring_data if d.timestamp >= cutoff_time]

            if len(recent_data) < 2:
                return {'trend': 0.0, 'message': '분석할 데이터가 부족합니다.'}

            # 메모리 사용량 추세 계산
            memory_values = [d.memory_mb for d in recent_data]
            timestamps = [d.timestamp for d in recent_data]

            # 선형 추세 계산
            trend = self._calculate_linear_trend(memory_values)

            # 통계 정보
            min_memory = min(memory_values)
            max_memory = max(memory_values)
            avg_memory = sum(memory_values) / len(memory_values)

            return {
                'trend_mb_per_hour': trend,
                'data_points': len(recent_data),
                'time_range_hours': hours,
                'min_memory_mb': min_memory,
                'max_memory_mb': max_memory,
                'avg_memory_mb': avg_memory,
                'trend_direction': 'increasing' if trend > 0 else 'decreasing' if trend < 0 else 'stable',
                'latest_memory_mb': memory_values[-1],
                'timestamp': datetime.now()
            }

        except Exception as e:
            self.logger.error(f"추세 분석 실패: {e}")
            return {'error': str(e)}

    def _calculate_linear_trend(self, values: List[float]) -> float:
        """선형 추세 계산 (MB/시간)"""
        if len(values) < 2:
            return 0.0

        n = len(values)
        x_sum = sum(range(n))
        y_sum = sum(values)
        xy_sum = sum(i * values[i] for i in range(n))
        x2_sum = sum(i * i for i in range(n))

        if n * x2_sum - x_sum * x_sum == 0:
            return 0.0

        slope = (n * xy_sum - x_sum * y_sum) / (n * x2_sum - x_sum * x_sum)
        return slope * self.config.interval_seconds / 3600.0  # MB/시간으로 변환

    def generate_memory_chart(self, hours: int = 1) -> Optional[str]:
        """메모리 사용량 차트 생성 (Base64 인코딩된 이미지)"""
        try:
            if not self.config.enable_visualization:
                return None

            cutoff_time = datetime.now() - timedelta(hours=hours)
            recent_data = [d for d in self.monitoring_data if d.timestamp >= cutoff_time]

            if len(recent_data) < 2:
                return None

            # 차트 생성
            timestamps = [d.timestamp for d in recent_data]
            memory_values = [d.memory_mb for d in recent_data]
            cpu_values = [d.cpu_percent for d in recent_data]

            fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8))

            # 메모리 사용량 차트
            ax1.plot(timestamps, memory_values, 'b-', linewidth=2, label='Memory Usage (MB)')
            ax1.axhline(y=self.config.alert_threshold_mb, color='orange', linestyle='--', label='Alert Threshold')
            ax1.axhline(y=self.config.critical_threshold_mb, color='red', linestyle='--', label='Critical Threshold')
            ax1.set_ylabel('Memory Usage (MB)')
            ax1.set_title(f'Memory Usage Trend (Last {hours} hours)')
            ax1.legend()
            ax1.grid(True, alpha=0.3)

            # CPU 사용률 차트
            ax2.plot(timestamps, cpu_values, 'r-', linewidth=2, label='CPU Usage (%)')
            ax2.set_ylabel('CPU Usage (%)')
            ax2.set_xlabel('Time')
            ax2.set_title('CPU Usage Trend')
            ax2.legend()
            ax2.grid(True, alpha=0.3)

            plt.tight_layout()

            # 이미지를 Base64로 인코딩
            buffer = io.BytesIO()
            plt.savefig(buffer, format='png', dpi=150, bbox_inches='tight')
            buffer.seek(0)
            image_base64 = base64.b64encode(buffer.getvalue()).decode()
            plt.close(fig)

            return image_base64

        except Exception as e:
            self.logger.error(f"차트 생성 실패: {e}")
            return None

    def export_monitoring_data(self, format: str = 'json') -> str:
        """모니터링 데이터 내보내기"""
        try:
            data = {
                'config': asdict(self.config),
                'stats': self.stats,
                'monitoring_data': [asdict(d) for d in self.monitoring_data],
                'export_timestamp': datetime.now().isoformat()
            }

            if format.lower() == 'json':
                return json.dumps(data, indent=2, default=str)
            else:
                raise ValueError(f"지원하지 않는 형식: {format}")

        except Exception as e:
            self.logger.error(f"데이터 내보내기 실패: {e}")
            return json.dumps({'error': str(e)})

    def get_alerts_summary(self) -> Dict[str, Any]:
        """알림 요약 정보"""
        try:
            alerts = []
            while not self.alert_queue.empty():
                try:
                    alert = self.alert_queue.get_nowait()
                    alerts.append(asdict(alert))
                except queue.Empty:
                    break

            # 알림 통계
            alert_stats = {}
            for alert in alerts:
                alert_type = alert['alert_type']
                alert_stats[alert_type] = alert_stats.get(alert_type, 0) + 1

            return {
                'total_alerts': len(alerts),
                'alert_stats': alert_stats,
                'recent_alerts': alerts[-10:] if alerts else [],  # 최근 10개
                'stats': self.stats,
                'timestamp': datetime.now()
            }

        except Exception as e:
            self.logger.error(f"알림 요약 생성 실패: {e}")
            return {'error': str(e)}

    def __del__(self):
        """소멸자 - 모니터링 중지"""
        self.stop_monitoring()


# 전역 모니터 인스턴스
_memory_monitor_instance: Optional[RealTimeMemoryMonitor] = None


def get_memory_monitor(config: Optional[MonitoringConfig] = None) -> RealTimeMemoryMonitor:
    """메모리 모니터 싱글톤 인스턴스 반환"""
    global _memory_monitor_instance

    if _memory_monitor_instance is None:
        _memory_monitor_instance = RealTimeMemoryMonitor(config)
        _memory_monitor_instance.start_monitoring()

    return _memory_monitor_instance


def cleanup_memory_monitor():
    """메모리 모니터 정리"""
    global _memory_monitor_instance

    if _memory_monitor_instance:
        _memory_monitor_instance.stop_monitoring()
        _memory_monitor_instance = None
