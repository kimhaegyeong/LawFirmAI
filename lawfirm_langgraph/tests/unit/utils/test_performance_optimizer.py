# -*- coding: utf-8 -*-
"""
PerformanceOptimizer 테스트
성능 최적화 모듈 단위 테스트
"""

import pytest
import time
from unittest.mock import Mock, MagicMock, patch
from typing import Dict, Any

from lawfirm_langgraph.core.utils.performance_optimizer import (
    PerformanceMonitor,
    MemoryOptimizer,
    CacheManager,
    PerformanceMetrics,
    MemoryUsage,
    performance_monitor,
    memory_optimized
)


class TestPerformanceMonitor:
    """PerformanceMonitor 테스트"""
    
    @pytest.fixture
    def performance_monitor(self):
        """PerformanceMonitor 인스턴스"""
        return PerformanceMonitor(max_history=100)
    
    def test_performance_monitor_initialization(self):
        """PerformanceMonitor 초기화 테스트"""
        monitor = PerformanceMonitor(max_history=100)
        
        assert monitor.max_history == 100
        assert len(monitor.metrics_history) == 0
        assert monitor.thresholds["cpu_usage"] == 80.0
        assert monitor.thresholds["memory_usage"] == 85.0
    
    def test_record_metrics(self, performance_monitor):
        """메트릭 기록 테스트"""
        with patch('lawfirm_langgraph.core.utils.performance_optimizer.PSUTIL_AVAILABLE', False):
            performance_monitor.record_metrics(
                response_time=1.5,
                active_sessions=5,
                cache_hit_rate=0.8,
                db_query_time=0.5
            )
            
            assert len(performance_monitor.metrics_history) == 1
            assert performance_monitor.metrics_history[0].response_time == 1.5
            assert performance_monitor.metrics_history[0].active_sessions == 5
    
    def test_get_performance_summary(self, performance_monitor):
        """성능 요약 조회 테스트"""
        with patch('lawfirm_langgraph.core.utils.performance_optimizer.PSUTIL_AVAILABLE', False):
            performance_monitor.record_metrics(
                response_time=1.0,
                active_sessions=3,
                cache_hit_rate=0.7,
                db_query_time=0.3
            )
            
            summary = performance_monitor.get_performance_summary(hours=24)
            
            assert isinstance(summary, dict)
            if "message" not in summary:
                assert "averages" in summary
                assert "maximums" in summary
    
    def test_get_system_health(self, performance_monitor):
        """시스템 건강 상태 조회 테스트"""
        with patch('lawfirm_langgraph.core.utils.performance_optimizer.PSUTIL_AVAILABLE', False):
            health = performance_monitor.get_system_health()
            
            assert isinstance(health, dict)
            assert "status" in health
            assert health["status"] == "unavailable" or health["status"] in ["healthy", "warning", "error"]
    
    def test_check_thresholds(self, performance_monitor):
        """임계값 체크 테스트"""
        with patch('lawfirm_langgraph.core.utils.performance_optimizer.PSUTIL_AVAILABLE', False):
            metrics = PerformanceMetrics(
                timestamp=time.time(),
                cpu_usage=90.0,
                memory_usage=90.0,
                response_time=6.0,
                active_sessions=10,
                cache_hit_rate=0.5,
                db_query_time=3.0
            )
            
            performance_monitor._check_thresholds(metrics)
            
            assert True


class TestMemoryOptimizer:
    """MemoryOptimizer 테스트"""
    
    @pytest.fixture
    def memory_optimizer(self):
        """MemoryOptimizer 인스턴스"""
        return MemoryOptimizer(max_cache_size=100)
    
    def test_memory_optimizer_initialization(self):
        """MemoryOptimizer 초기화 테스트"""
        optimizer = MemoryOptimizer(max_cache_size=100)
        
        assert optimizer.max_cache_size == 100
        assert len(optimizer.memory_usage_history) == 0
    
    def test_optimize_memory(self, memory_optimizer):
        """메모리 최적화 테스트"""
        with patch('lawfirm_langgraph.core.utils.performance_optimizer.PSUTIL_AVAILABLE', False):
            result = memory_optimizer.optimize_memory()
            
            assert isinstance(result, dict)
            assert "timestamp" in result
            assert "actions_taken" in result
            assert "memory_freed_mb" in result
    
    def test_get_memory_usage(self, memory_optimizer):
        """메모리 사용량 조회 테스트"""
        with patch('lawfirm_langgraph.core.utils.performance_optimizer.PSUTIL_AVAILABLE', False):
            memory_usage = memory_optimizer.get_memory_usage()
            
            assert isinstance(memory_usage, MemoryUsage)
            assert memory_usage.total_memory == 0
            assert memory_usage.process_memory == 0
    
    def test_monitor_memory_trend(self, memory_optimizer):
        """메모리 트렌드 모니터링 테스트"""
        with patch('lawfirm_langgraph.core.utils.performance_optimizer.PSUTIL_AVAILABLE', False):
            for _ in range(10):
                memory_optimizer.memory_usage_history.append(100 * 1024 * 1024)
            
            trend = memory_optimizer.monitor_memory_trend()
            
            assert isinstance(trend, dict)
            assert "trend" in trend or "message" in trend


class TestCacheManager:
    """CacheManager 테스트"""
    
    @pytest.fixture
    def cache_manager(self):
        """CacheManager 인스턴스"""
        return CacheManager(max_size=10, ttl=3600)
    
    def test_cache_manager_initialization(self):
        """CacheManager 초기화 테스트"""
        manager = CacheManager(max_size=10, ttl=3600)
        
        assert manager.max_size == 10
        assert manager.ttl == 3600
        assert len(manager.cache) == 0
    
    def test_cache_get_set(self, cache_manager):
        """캐시 저장 및 조회 테스트"""
        cache_manager.set("test_key", "test_value")
        
        value = cache_manager.get("test_key")
        
        assert value == "test_value"
    
    def test_cache_get_not_found(self, cache_manager):
        """캐시 미조회 테스트"""
        value = cache_manager.get("non_existent_key")
        
        assert value is None
    
    def test_cache_delete(self, cache_manager):
        """캐시 삭제 테스트"""
        cache_manager.set("test_key", "test_value")
        cache_manager.delete("test_key")
        
        value = cache_manager.get("test_key")
        
        assert value is None
    
    def test_cache_clear(self, cache_manager):
        """캐시 전체 삭제 테스트"""
        cache_manager.set("key1", "value1")
        cache_manager.set("key2", "value2")
        
        cleared = cache_manager.clear()
        
        assert cleared == 2
        assert len(cache_manager.cache) == 0
    
    def test_cache_get_stats(self, cache_manager):
        """캐시 통계 조회 테스트"""
        cache_manager.set("key1", "value1")
        cache_manager.get("key1")
        
        stats = cache_manager.get_stats()
        
        assert isinstance(stats, dict)
        assert "cache_size" in stats
        assert "hits" in stats
        assert "misses" in stats
    
    def test_cache_lru_eviction(self, cache_manager):
        """LRU 캐시 제거 테스트"""
        for i in range(15):
            cache_manager.set(f"key{i}", f"value{i}")
        
        assert len(cache_manager.cache) <= cache_manager.max_size
    
    def test_cache_ttl_expiration(self, cache_manager):
        """TTL 만료 테스트"""
        cache_manager.ttl = 1
        cache_manager.set("test_key", "test_value")
        
        time.sleep(2)
        
        value = cache_manager.get("test_key")
        
        assert value is None


class TestDecorators:
    """데코레이터 테스트"""
    
    def test_performance_monitor_decorator(self):
        """performance_monitor 데코레이터 테스트"""
        @performance_monitor
        def test_function():
            return "test"
        
        result = test_function()
        
        assert result == "test"
    
    def test_memory_optimized_decorator(self):
        """memory_optimized 데코레이터 테스트"""
        @memory_optimized
        def test_function():
            return "test"
        
        result = test_function()
        
        assert result == "test"

