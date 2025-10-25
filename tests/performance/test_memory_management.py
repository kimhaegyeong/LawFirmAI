# -*- coding: utf-8 -*-
"""
Memory Management Test Suite
메모리 관리 시스템 테스트 코드
"""

import pytest
import asyncio
import time
import gc
import weakref
from unittest.mock import Mock, patch, MagicMock
from datetime import datetime, timedelta
import sys
import os

# 테스트 대상 모듈 import
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))

from source.utils.memory_manager import MemoryManager, MemoryMetrics, MemoryAlert, get_memory_manager
from source.utils.weakref_cleanup import WeakRefRegistry, MemoryOptimizer, get_weakref_registry
from source.utils.realtime_memory_monitor import RealTimeMemoryMonitor, MonitoringConfig, get_memory_monitor
from source.services.enhanced_chat_service import EnhancedChatService
from source.utils.config import Config


class TestMemoryManager:
    """메모리 관리자 테스트"""
    
    def setup_method(self):
        """테스트 설정"""
        self.memory_manager = MemoryManager(max_memory_mb=100, alert_threshold=0.7, cleanup_threshold=0.9)
    
    def teardown_method(self):
        """테스트 정리"""
        if self.memory_manager:
            self.memory_manager.stop_monitoring()
    
    def test_memory_manager_initialization(self):
        """메모리 관리자 초기화 테스트"""
        assert self.memory_manager.max_memory_mb == 100
        assert self.memory_manager.alert_threshold == 0.7
        assert self.memory_manager.cleanup_threshold == 0.9
        assert len(self.memory_manager.metrics_history) == 0
        assert len(self.memory_manager.tracked_objects) == 0
    
    def test_get_current_metrics(self):
        """현재 메모리 메트릭 수집 테스트"""
        metrics = self.memory_manager.get_current_metrics()
        
        assert isinstance(metrics, MemoryMetrics)
        assert metrics.rss_mb >= 0
        assert metrics.vms_mb >= 0
        assert metrics.cpu_percent >= 0
        assert metrics.memory_percent >= 0
        assert metrics.available_mb >= 0
        assert isinstance(metrics.timestamp, datetime)
    
    def test_memory_usage_check(self):
        """메모리 사용량 체크 테스트"""
        # 낮은 메모리 사용량 시뮬레이션
        low_memory_metrics = MemoryMetrics(
            timestamp=datetime.now(),
            rss_mb=50.0,  # 50MB (임계값 70MB 이하)
            vms_mb=100.0,
            cpu_percent=10.0,
            memory_percent=50.0,
            available_mb=1000.0
        )
        
        # 알림이 발생하지 않아야 함
        initial_alert_count = self.memory_manager.stats['alert_count']
        self.memory_manager._check_memory_usage(low_memory_metrics)
        assert self.memory_manager.stats['alert_count'] == initial_alert_count
        
        # 높은 메모리 사용량 시뮬레이션
        high_memory_metrics = MemoryMetrics(
            timestamp=datetime.now(),
            rss_mb=80.0,  # 80MB (임계값 70MB 초과)
            vms_mb=150.0,
            cpu_percent=20.0,
            memory_percent=60.0,
            available_mb=800.0
        )
        
        self.memory_manager._check_memory_usage(high_memory_metrics)
        assert self.memory_manager.stats['alert_count'] > initial_alert_count
    
    def test_perform_cleanup(self):
        """메모리 정리 테스트"""
        # 테스트용 객체 생성
        test_objects = []
        for i in range(10):
            obj = {'data': f'test_data_{i}', 'number': i}
            test_objects.append(obj)
            self.memory_manager.track_object(obj, f'test_obj_{i}')
        
        # 정리 전 상태
        initial_refs = len(self.memory_manager.tracked_objects)
        
        # 정리 실행
        cleanup_result = self.memory_manager.perform_cleanup()
        
        assert cleanup_result['success'] is True
        assert 'memory_freed_mb' in cleanup_result
        assert 'objects_collected' in cleanup_result
        assert self.memory_manager.stats['total_cleanups'] > 0
    
    def test_force_garbage_collection(self):
        """강제 가비지 컬렉션 테스트"""
        # 테스트용 객체 생성
        test_objects = []
        for i in range(100):
            obj = {'data': f'large_data_{i}' * 1000}
            test_objects.append(obj)
        
        # 가비지 컬렉션 실행
        gc_result = self.memory_manager.force_garbage_collection()
        
        assert gc_result['success'] is True
        assert 'total_objects_collected' in gc_result
        assert 'memory_freed_mb' in gc_result
    
    def test_memory_trend_calculation(self):
        """메모리 추세 계산 테스트"""
        # 테스트용 메트릭 데이터 생성
        base_time = datetime.now()
        test_metrics = []
        
        for i in range(10):
            metrics = MemoryMetrics(
                timestamp=base_time + timedelta(minutes=i),
                rss_mb=100.0 + i * 5.0,  # 증가하는 추세
                vms_mb=200.0 + i * 10.0,
                cpu_percent=20.0,
                memory_percent=50.0,
                available_mb=1000.0
            )
            test_metrics.append(metrics)
        
        self.memory_manager.metrics_history = test_metrics
        
        # 추세 계산
        trend = self.memory_manager._calculate_trend([m.rss_mb for m in test_metrics])
        
        assert trend > 0  # 증가 추세여야 함
    
    def test_alert_callback(self):
        """알림 콜백 테스트"""
        callback_called = False
        received_alert = None
        
        def test_callback(alert):
            nonlocal callback_called, received_alert
            callback_called = True
            received_alert = alert
        
        # 콜백 등록
        self.memory_manager.add_alert_callback(test_callback)
        
        # 테스트 알림 생성
        test_alert = MemoryAlert(
            alert_type='test_alert',
            message='테스트 알림',
            severity='medium',
            timestamp=datetime.now(),
            metrics=None
        )
        
        # 알림 트리거
        self.memory_manager._trigger_alert(test_alert)
        
        assert callback_called is True
        assert received_alert == test_alert


class TestWeakRefRegistry:
    """WeakRef 등록소 테스트"""
    
    def setup_method(self):
        """테스트 설정"""
        self.registry = WeakRefRegistry()
    
    def test_object_registration(self):
        """객체 등록 테스트"""
        test_obj = {'data': 'test_data', 'number': 42}
        
        # 객체 등록
        result = self.registry.register_object(test_obj, 'test_object')
        
        assert result is True
        assert 'test_object' in self.registry._weak_refs
        assert 'test_object' in self.registry._object_info
        
        # 등록된 객체 가져오기
        retrieved_obj = self.registry.get_object('test_object')
        assert retrieved_obj == test_obj
    
    def test_object_unregistration(self):
        """객체 등록 해제 테스트"""
        test_obj = {'data': 'test_data'}
        
        # 객체 등록
        self.registry.register_object(test_obj, 'test_object')
        assert 'test_object' in self.registry._weak_refs
        
        # 객체 등록 해제
        result = self.registry.unregister_object('test_object')
        
        assert result is True
        assert 'test_object' not in self.registry._weak_refs
        assert 'test_object' not in self.registry._object_info
    
    def test_dead_reference_cleanup(self):
        """죽은 참조 정리 테스트"""
        # 객체 생성 및 등록
        test_obj = {'data': 'test_data'}
        self.registry.register_object(test_obj, 'test_object')
        
        initial_count = len(self.registry._weak_refs)
        
        # 객체 삭제
        del test_obj
        gc.collect()  # 가비지 컬렉션 강제 실행
        
        # 죽은 참조 정리
        cleanup_result = self.registry.cleanup_dead_references()
        
        assert cleanup_result['success'] is True
        assert cleanup_result['cleaned_refs'] > 0
        assert len(self.registry._weak_refs) < initial_count
    
    def test_registry_stats(self):
        """등록소 통계 테스트"""
        # 테스트 객체들 등록
        for i in range(5):
            obj = {'data': f'test_data_{i}'}
            self.registry.register_object(obj, f'test_obj_{i}')
        
        # 통계 조회
        stats = self.registry.get_registry_stats()
        
        assert 'total_registered' in stats
        assert 'alive_objects' in stats
        assert 'dead_references' in stats
        assert 'type_distribution' in stats
        assert stats['total_registered'] == 5
        assert stats['alive_objects'] == 5
    
    def test_access_tracking(self):
        """접근 추적 테스트"""
        test_obj = {'data': 'test_data'}
        self.registry.register_object(test_obj, 'test_object')
        
        # 여러 번 접근
        for _ in range(3):
            self.registry.get_object('test_object')
        
        # 접근 통계 확인
        obj_info = self.registry.get_object_info('test_object')
        assert obj_info.access_count == 3
        
        stats = self.registry.get_registry_stats()
        assert stats['total_accesses'] == 3


class TestMemoryOptimizer:
    """메모리 최적화 도구 테스트"""
    
    def setup_method(self):
        """테스트 설정"""
        self.registry = WeakRefRegistry()
        self.optimizer = MemoryOptimizer(self.registry)
    
    def test_optimize_memory_usage(self):
        """메모리 사용량 최적화 테스트"""
        # 테스트 객체들 등록
        for i in range(10):
            obj = {'data': f'test_data_{i}' * 1000}  # 큰 객체
            self.registry.register_object(obj, f'large_obj_{i}')
        
        # 최적화 실행
        result = self.optimizer.optimize_memory_usage(max_memory_mb=1.0, cleanup_threshold=0.8)
        
        assert 'optimization_needed' in result
        assert 'initial_memory_mb' in result
        assert 'final_memory_mb' in result
        assert 'memory_freed_mb' in result
    
    def test_cleanup_low_access_objects(self):
        """낮은 접근 빈도 객체 정리 테스트"""
        # 오래된 객체 등록
        old_time = datetime.now() - timedelta(hours=25)
        
        # Mock을 사용하여 시간 조작
        with patch('source.utils.weakref_cleanup.datetime') as mock_datetime:
            mock_datetime.now.return_value = old_time
            
            obj = {'data': 'old_data'}
            self.registry.register_object(obj, 'old_object')
            
            # 접근 빈도 낮게 설정
            obj_info = self.registry._object_info['old_object']
            obj_info.access_count = 2  # 낮은 접근 빈도
            obj_info.last_accessed = old_time
        
        # 정리 실행
        result = self.optimizer._cleanup_low_access_objects(min_access_count=5, max_age_hours=24)
        
        assert result['strategy'] == 'low_access_cleanup'
        assert 'cleaned_count' in result


class TestRealTimeMemoryMonitor:
    """실시간 메모리 모니터 테스트"""
    
    def setup_method(self):
        """테스트 설정"""
        config = MonitoringConfig(
            interval_seconds=1,  # 빠른 테스트를 위해 1초로 설정
            max_history_size=100,
            alert_threshold_mb=50.0,
            critical_threshold_mb=80.0,
            enable_auto_cleanup=False,  # 테스트 중 자동 정리 비활성화
            enable_trend_analysis=True,
            enable_visualization=False  # 테스트 중 시각화 비활성화
        )
        self.monitor = RealTimeMemoryMonitor(config)
    
    def teardown_method(self):
        """테스트 정리"""
        if self.monitor:
            self.monitor.stop_monitoring()
    
    def test_monitor_initialization(self):
        """모니터 초기화 테스트"""
        assert self.monitor.config.interval_seconds == 1
        assert self.monitor.config.alert_threshold_mb == 50.0
        assert self.monitor.config.critical_threshold_mb == 80.0
        assert len(self.monitor.monitoring_data) == 0
        assert self.monitor.is_monitoring is False
    
    def test_collect_monitoring_data(self):
        """모니터링 데이터 수집 테스트"""
        data = self.monitor._collect_monitoring_data()
        
        assert isinstance(data.timestamp, datetime)
        assert data.memory_mb >= 0
        assert data.cpu_percent >= 0
        assert data.process_count >= 0
        assert data.thread_count >= 0
        assert isinstance(data.gc_counts, dict)
        assert isinstance(data.weakref_stats, dict)
    
    def test_alert_checking(self):
        """알림 체크 테스트"""
        # 낮은 메모리 사용량 데이터
        low_memory_data = self.monitor._collect_monitoring_data()
        low_memory_data.memory_mb = 30.0  # 임계값 이하
        
        initial_alert_count = self.monitor.stats['alert_count']
        self.monitor._check_alerts(low_memory_data)
        assert self.monitor.stats['alert_count'] == initial_alert_count
        
        # 높은 메모리 사용량 데이터
        high_memory_data = self.monitor._collect_monitoring_data()
        high_memory_data.memory_mb = 60.0  # 임계값 초과
        
        self.monitor._check_alerts(high_memory_data)
        assert self.monitor.stats['alert_count'] > initial_alert_count
    
    def test_memory_trend_analysis(self):
        """메모리 추세 분석 테스트"""
        # 테스트용 데이터 생성
        base_time = datetime.now()
        for i in range(10):
            data = self.monitor._collect_monitoring_data()
            data.timestamp = base_time + timedelta(minutes=i)
            data.memory_mb = 100.0 + i * 2.0  # 증가하는 추세
            self.monitor.monitoring_data.append(data)
        
        # 추세 분석
        trend_result = self.monitor.get_memory_trend(hours=1)
        
        assert 'trend_mb_per_hour' in trend_result
        assert 'data_points' in trend_result
        assert 'trend_direction' in trend_result
        assert trend_result['data_points'] == 10
        assert trend_result['trend_direction'] == 'increasing'
    
    def test_callback_system(self):
        """콜백 시스템 테스트"""
        data_callback_called = False
        alert_callback_called = False
        
        def data_callback(data):
            nonlocal data_callback_called
            data_callback_called = True
        
        def alert_callback(alert):
            nonlocal alert_callback_called
            alert_callback_called = True
        
        # 콜백 등록
        self.monitor.add_data_callback(data_callback)
        self.monitor.add_alert_callback(alert_callback)
        
        # 테스트 데이터 생성
        test_data = self.monitor._collect_monitoring_data()
        test_data.memory_mb = 60.0  # 알림 발생시키기
        
        # 데이터 콜백 테스트
        for callback in self.monitor.data_callbacks:
            callback(test_data)
        
        # 알림 콜백 테스트
        test_alert = MemoryAlert(
            alert_type='test',
            message='테스트 알림',
            severity='medium',
            timestamp=datetime.now(),
            metrics=None
        )
        self.monitor._process_alert(test_alert)
        
        assert data_callback_called is True
        assert alert_callback_called is True


class TestEnhancedChatServiceMemoryIntegration:
    """EnhancedChatService 메모리 통합 테스트"""
    
    def setup_method(self):
        """테스트 설정"""
        # Mock Config 생성
        self.mock_config = Mock(spec=Config)
        self.mock_config.max_memory_mb = 100
        self.mock_config.alert_threshold = 0.8
        
        # 의존성 모킹
        with patch('source.services.enhanced_chat_service.DatabaseManager'), \
             patch('source.services.enhanced_chat_service.LegalVectorStore'), \
             patch('source.services.enhanced_chat_service.OptimizedModelManager'), \
             patch('source.services.enhanced_chat_service.HybridSearchEngine'), \
             patch('source.services.enhanced_chat_service.QuestionClassifier'), \
             patch('source.services.enhanced_chat_service.ImprovedAnswerGenerator'):
            
            self.service = EnhancedChatService(self.mock_config)
    
    def test_memory_management_initialization(self):
        """메모리 관리 초기화 테스트"""
        assert hasattr(self.service, 'memory_manager')
        assert hasattr(self.service, 'weakref_registry')
        assert hasattr(self.service, 'memory_monitor')
        assert hasattr(self.service, '_track_component')
    
    def test_perform_memory_cleanup(self):
        """메모리 정리 테스트"""
        cleanup_result = self.service.perform_memory_cleanup()
        
        assert 'success' in cleanup_result
        assert 'total_memory_freed_mb' in cleanup_result
        assert 'cleanup_details' in cleanup_result
        assert 'timestamp' in cleanup_result
    
    def test_get_memory_status(self):
        """메모리 상태 조회 테스트"""
        status = self.service.get_memory_status()
        
        assert 'service_name' in status
        assert 'timestamp' in status
        assert 'memory_manager' in status or 'error' in status
    
    def test_optimize_memory_usage(self):
        """메모리 사용량 최적화 테스트"""
        optimization_result = self.service.optimize_memory_usage(target_memory_mb=50.0)
        
        assert 'optimization_needed' in optimization_result
        assert 'initial_memory_mb' in optimization_result
        assert 'final_memory_mb' in optimization_result
        assert 'memory_freed_mb' in optimization_result
    
    def test_generate_memory_report(self):
        """메모리 리포트 생성 테스트"""
        report = self.service.generate_memory_report()
        
        assert 'report_timestamp' in report
        assert 'service_info' in report
        assert 'memory_status' in report
        assert 'recommendations' in report
    
    def test_component_tracking(self):
        """컴포넌트 추적 테스트"""
        # 테스트 객체 생성
        test_obj = {'data': 'test_component'}
        
        # 컴포넌트 추적
        tracked_name = self.service._track_component(test_obj, 'test_component')
        
        # 추적 확인
        if self.service.weakref_registry:
            retrieved_obj = self.service.weakref_registry.get_object(tracked_name)
            assert retrieved_obj == test_obj


class TestMemoryManagementIntegration:
    """메모리 관리 통합 테스트"""
    
    def test_singleton_patterns(self):
        """싱글톤 패턴 테스트"""
        # 메모리 관리자 싱글톤 테스트
        manager1 = get_memory_manager()
        manager2 = get_memory_manager()
        assert manager1 is manager2
        
        # WeakRef 등록소 싱글톤 테스트
        registry1 = get_weakref_registry()
        registry2 = get_weakref_registry()
        assert registry1 is registry2
        
        # 메모리 모니터 싱글톤 테스트
        monitor1 = get_memory_monitor()
        monitor2 = get_memory_monitor()
        assert monitor1 is monitor2
    
    def test_memory_leak_detection(self):
        """메모리 누수 감지 테스트"""
        manager = get_memory_manager()
        
        # 메모리 사용량 증가 시뮬레이션
        for i in range(5):
            metrics = manager.get_current_metrics()
            metrics.rss_mb = 100.0 + i * 20.0  # 증가하는 메모리 사용량
            manager.metrics_history.append(metrics)
        
        # 누수 감지 테스트
        latest_metrics = manager.get_current_metrics()
        latest_metrics.rss_mb = 200.0
        manager._detect_memory_leak(latest_metrics)
        
        # 알림이 발생했는지 확인
        assert manager.stats['alert_count'] > 0
    
    def test_end_to_end_memory_management(self):
        """전체 메모리 관리 플로우 테스트"""
        # 1. 메모리 관리자 초기화
        manager = get_memory_manager()
        registry = get_weakref_registry()
        monitor = get_memory_monitor()
        
        # 2. 테스트 객체 생성 및 등록
        test_objects = []
        for i in range(10):
            obj = {'data': f'test_data_{i}' * 1000}
            test_objects.append(obj)
            registry.register_object(obj, f'test_obj_{i}')
        
        # 3. 메모리 상태 확인
        initial_status = manager.get_memory_usage()
        assert initial_status['tracked_objects_count'] == 10
        
        # 4. 메모리 정리 실행
        cleanup_result = manager.perform_cleanup()
        assert cleanup_result['success'] is True
        
        # 5. 정리 후 상태 확인
        final_status = manager.get_memory_usage()
        assert final_status['tracked_objects_count'] <= 10
        
        # 6. 모니터링 데이터 확인
        monitor_status = monitor.get_current_status()
        assert 'latest_data' in monitor_status or 'error' in monitor_status


if __name__ == '__main__':
    # 테스트 실행
    pytest.main([__file__, '-v', '--tb=short'])
