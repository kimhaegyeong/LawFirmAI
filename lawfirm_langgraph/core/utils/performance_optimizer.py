# -*- coding: utf-8 -*-
"""
성능 최적화 및 메모리 관리 모듈
LawFirmAI의 성능을 최적화하고 메모리 사용량을 관리합니다.
"""

# Global logger 사용
try:
    from lawfirm_langgraph.core.utils.logger import get_logger
except ImportError:
    from core.utils.logger import get_logger
import time
import gc
from typing import Dict, List, Any, Optional, Callable
from dataclasses import dataclass
from datetime import datetime, timedelta
from collections import defaultdict, deque
import asyncio
from functools import wraps

# psutil 선택적 import
try:
    import psutil
    PSUTIL_AVAILABLE = True
except ImportError:
    PSUTIL_AVAILABLE = False
    psutil = None

logger = get_logger(__name__)


@dataclass
class PerformanceMetrics:
    """성능 메트릭"""
    timestamp: datetime
    cpu_usage: float
    memory_usage: float
    response_time: float
    active_sessions: int
    cache_hit_rate: float
    db_query_time: float


@dataclass
class MemoryUsage:
    """메모리 사용량 정보"""
    total_memory: int
    used_memory: int
    available_memory: int
    memory_percentage: float
    process_memory: int
    cache_memory: int


class PerformanceMonitor:
    """성능 모니터링 클래스"""
    
    def __init__(self, max_history: int = 1000):
        """
        성능 모니터 초기화
        
        Args:
            max_history: 최대 메트릭 저장 개수
        """
        self.logger = get_logger(__name__)
        self.max_history = max_history
        self.metrics_history = deque(maxlen=max_history)
        self.start_time = datetime.now()
        
        # 성능 임계값 설정
        self.thresholds = {
            "cpu_usage": 80.0,  # CPU 사용률 80% 이상 시 경고
            "memory_usage": 85.0,  # 메모리 사용률 85% 이상 시 경고
            "response_time": 5.0,  # 응답 시간 5초 이상 시 경고
            "cache_hit_rate": 0.7,  # 캐시 히트율 70% 미만 시 경고
            "db_query_time": 2.0  # DB 쿼리 시간 2초 이상 시 경고
        }
        
        self.logger.trace("PerformanceMonitor initialized")
    
    def record_metrics(self, response_time: float, active_sessions: int, 
                      cache_hit_rate: float = 0.0, db_query_time: float = 0.0):
        """성능 메트릭 기록"""
        try:
            # 시스템 리소스 사용량 조회
            if not PSUTIL_AVAILABLE:
                cpu_usage = 0.0
                memory_percent = 0.0
            else:
                cpu_usage = psutil.cpu_percent(interval=0.1)
                memory = psutil.virtual_memory()
                memory_percent = memory.percent
            
            metrics = PerformanceMetrics(
                timestamp=datetime.now(),
                cpu_usage=cpu_usage,
                memory_usage=memory_percent,
                response_time=response_time,
                active_sessions=active_sessions,
                cache_hit_rate=cache_hit_rate,
                db_query_time=db_query_time
            )
            
            self.metrics_history.append(metrics)
            
            # 임계값 초과 시 경고
            self._check_thresholds(metrics)
            
        except Exception as e:
            self.logger.error(f"Error recording metrics: {e}")
    
    def get_performance_summary(self, hours: int = 24) -> Dict[str, Any]:
        """성능 요약 조회"""
        try:
            cutoff_time = datetime.now() - timedelta(hours=hours)
            recent_metrics = [m for m in self.metrics_history if m.timestamp >= cutoff_time]
            
            if not recent_metrics:
                return {"message": "No recent metrics available"}
            
            # 평균값 계산
            avg_cpu = sum(m.cpu_usage for m in recent_metrics) / len(recent_metrics)
            avg_memory = sum(m.memory_usage for m in recent_metrics) / len(recent_metrics)
            avg_response_time = sum(m.response_time for m in recent_metrics) / len(recent_metrics)
            avg_cache_hit_rate = sum(m.cache_hit_rate for m in recent_metrics) / len(recent_metrics)
            avg_db_query_time = sum(m.db_query_time for m in recent_metrics) / len(recent_metrics)
            
            # 최대값 계산
            max_cpu = max(m.cpu_usage for m in recent_metrics)
            max_memory = max(m.memory_usage for m in recent_metrics)
            max_response_time = max(m.response_time for m in recent_metrics)
            
            # 최근 세션 수
            latest_sessions = recent_metrics[-1].active_sessions if recent_metrics else 0
            
            return {
                "period_hours": hours,
                "total_requests": len(recent_metrics),
                "averages": {
                    "cpu_usage": round(avg_cpu, 2),
                    "memory_usage": round(avg_memory, 2),
                    "response_time": round(avg_response_time, 3),
                    "cache_hit_rate": round(avg_cache_hit_rate, 3),
                    "db_query_time": round(avg_db_query_time, 3)
                },
                "maximums": {
                    "cpu_usage": round(max_cpu, 2),
                    "memory_usage": round(max_memory, 2),
                    "response_time": round(max_response_time, 3)
                },
                "current_active_sessions": latest_sessions,
                "uptime_hours": (datetime.now() - self.start_time).total_seconds() / 3600,
                "status": self._get_overall_status(avg_cpu, avg_memory, avg_response_time)
            }
            
        except Exception as e:
            self.logger.error(f"Error getting performance summary: {e}")
            return {"error": str(e)}
    
    def get_system_health(self) -> Dict[str, Any]:
        """시스템 건강 상태 조회"""
        try:
            if not PSUTIL_AVAILABLE:
                return {
                    "status": "unavailable",
                    "error": "psutil not available",
                    "timestamp": datetime.now().isoformat()
                }
            
            # 현재 시스템 상태
            cpu_usage = psutil.cpu_percent(interval=0.1)
            memory = psutil.virtual_memory()
            disk = psutil.disk_usage('/')
            
            # 프로세스 정보
            process = psutil.Process()
            process_memory = process.memory_info().rss / 1024 / 1024  # MB
            
            # 최근 메트릭에서 트렌드 분석
            recent_metrics = list(self.metrics_history)[-10:] if self.metrics_history else []
            
            health_status = "healthy"
            warnings = []
            
            # 임계값 체크
            if cpu_usage > self.thresholds["cpu_usage"]:
                health_status = "warning"
                warnings.append(f"High CPU usage: {cpu_usage:.1f}%")
            
            if memory.percent > self.thresholds["memory_usage"]:
                health_status = "warning"
                warnings.append(f"High memory usage: {memory.percent:.1f}%")
            
            if recent_metrics:
                avg_response_time = sum(m.response_time for m in recent_metrics) / len(recent_metrics)
                if avg_response_time > self.thresholds["response_time"]:
                    health_status = "warning"
                    warnings.append(f"Slow response time: {avg_response_time:.2f}s")
            
            return {
                "status": health_status,
                "timestamp": datetime.now().isoformat(),
                "system": {
                    "cpu_usage": round(cpu_usage, 2),
                    "memory_usage": round(memory.percent, 2),
                    "memory_available_gb": round(memory.available / 1024 / 1024 / 1024, 2),
                    "disk_usage": round(disk.percent, 2),
                    "disk_free_gb": round(disk.free / 1024 / 1024 / 1024, 2)
                },
                "process": {
                    "memory_mb": round(process_memory, 2),
                    "cpu_percent": round(process.cpu_percent(), 2),
                    "threads": process.num_threads(),
                    "open_files": len(process.open_files())
                },
                "warnings": warnings,
                "uptime_hours": (datetime.now() - self.start_time).total_seconds() / 3600
            }
            
        except Exception as e:
            self.logger.error(f"Error getting system health: {e}")
            return {"status": "error", "error": str(e)}
    
    def _check_thresholds(self, metrics: PerformanceMetrics):
        """임계값 체크 및 경고"""
        warnings = []
        
        if metrics.cpu_usage > self.thresholds["cpu_usage"]:
            warnings.append(f"High CPU usage: {metrics.cpu_usage:.1f}%")
        
        if metrics.memory_usage > self.thresholds["memory_usage"]:
            warnings.append(f"High memory usage: {metrics.memory_usage:.1f}%")
        
        if metrics.response_time > self.thresholds["response_time"]:
            warnings.append(f"Slow response time: {metrics.response_time:.2f}s")
        
        if metrics.cache_hit_rate < self.thresholds["cache_hit_rate"]:
            warnings.append(f"Low cache hit rate: {metrics.cache_hit_rate:.1%}")
        
        if metrics.db_query_time > self.thresholds["db_query_time"]:
            warnings.append(f"Slow DB query time: {metrics.db_query_time:.2f}s")
        
        if warnings:
            self.logger.warning(f"Performance warnings: {'; '.join(warnings)}")
    
    def _get_overall_status(self, avg_cpu: float, avg_memory: float, avg_response_time: float) -> str:
        """전체 상태 판단"""
        if (avg_cpu > self.thresholds["cpu_usage"] or 
            avg_memory > self.thresholds["memory_usage"] or 
            avg_response_time > self.thresholds["response_time"]):
            return "degraded"
        elif (avg_cpu > self.thresholds["cpu_usage"] * 0.8 or 
              avg_memory > self.thresholds["memory_usage"] * 0.8 or 
              avg_response_time > self.thresholds["response_time"] * 0.8):
            return "warning"
        else:
            return "healthy"


class MemoryOptimizer:
    """메모리 최적화 클래스"""
    
    def __init__(self, max_cache_size: int = 1000):
        """
        메모리 최적화기 초기화
        
        Args:
            max_cache_size: 최대 캐시 크기
        """
        self.logger = get_logger(__name__)
        self.max_cache_size = max_cache_size
        
        # 캐시 관리
        self.cache_stats = defaultdict(int)
        self.cache_access_times = defaultdict(float)
        
        # 메모리 사용량 추적
        self.memory_usage_history = deque(maxlen=100)
        
        # 가비지 컬렉션 설정
        gc.set_threshold(700, 10, 10)  # 더 적극적인 GC
        
        self.logger.trace("MemoryOptimizer initialized")
    
    def optimize_memory(self) -> Dict[str, Any]:
        """메모리 최적화 수행"""
        try:
            optimization_results = {
                "timestamp": datetime.now().isoformat(),
                "actions_taken": [],
                "memory_freed_mb": 0,
                "cache_cleared": 0
            }
            
            # 1. 가비지 컬렉션 실행
            before_gc = self._get_memory_usage()
            collected = gc.collect()
            after_gc = self._get_memory_usage()
            
            memory_freed = before_gc - after_gc
            optimization_results["memory_freed_mb"] = round(memory_freed / 1024 / 1024, 2)
            optimization_results["actions_taken"].append(f"Garbage collection: {collected} objects collected")
            
            # 2. 오래된 캐시 정리
            cache_cleared = self._cleanup_old_cache()
            optimization_results["cache_cleared"] = cache_cleared
            if cache_cleared > 0:
                optimization_results["actions_taken"].append(f"Cache cleanup: {cache_cleared} entries removed")
            
            # 3. 메모리 사용량 기록
            self.memory_usage_history.append(after_gc)
            
            # 4. 메모리 압박 상황 체크
            if after_gc > 1024 * 1024 * 1024:  # 1GB 이상
                self.logger.warning(f"High memory usage detected: {after_gc / 1024 / 1024:.1f}MB")
                optimization_results["actions_taken"].append("High memory usage warning")
            
            return optimization_results
            
        except Exception as e:
            self.logger.error(f"Error optimizing memory: {e}")
            return {"error": str(e)}
    
    def get_memory_usage(self) -> MemoryUsage:
        """메모리 사용량 조회"""
        try:
            if not PSUTIL_AVAILABLE:
                return MemoryUsage(0, 0, 0, 0, 0, 0)
            
            memory = psutil.virtual_memory()
            process = psutil.Process()
            process_memory = process.memory_info().rss
            
            # 캐시 메모리 추정 (간단한 계산)
            cache_memory = len(self.cache_access_times) * 1024  # 각 캐시 항목당 1KB 추정
            
            return MemoryUsage(
                total_memory=memory.total,
                used_memory=memory.used,
                available_memory=memory.available,
                memory_percentage=memory.percent,
                process_memory=process_memory,
                cache_memory=cache_memory
            )
            
        except Exception as e:
            self.logger.error(f"Error getting memory usage: {e}")
            return MemoryUsage(0, 0, 0, 0, 0, 0)
    
    def monitor_memory_trend(self) -> Dict[str, Any]:
        """메모리 사용량 트렌드 분석"""
        try:
            if len(self.memory_usage_history) < 10:
                return {"message": "Insufficient data for trend analysis"}
            
            recent_usage = list(self.memory_usage_history)[-10:]
            
            # 트렌드 계산
            if len(recent_usage) >= 2:
                trend = "increasing" if recent_usage[-1] > recent_usage[0] else "decreasing"
                trend_rate = (recent_usage[-1] - recent_usage[0]) / recent_usage[0] * 100
            else:
                trend = "stable"
                trend_rate = 0
            
            # 평균 사용량
            avg_usage = sum(recent_usage) / len(recent_usage)
            
            # 메모리 누수 가능성 체크
            memory_leak_warning = False
            if trend == "increasing" and trend_rate > 10:  # 10% 이상 증가
                memory_leak_warning = True
            
            return {
                "trend": trend,
                "trend_rate_percent": round(trend_rate, 2),
                "average_usage_mb": round(avg_usage / 1024 / 1024, 2),
                "current_usage_mb": round(recent_usage[-1] / 1024 / 1024, 2),
                "memory_leak_warning": memory_leak_warning,
                "recommendations": self._get_memory_recommendations(trend, trend_rate, avg_usage)
            }
            
        except Exception as e:
            self.logger.error(f"Error analyzing memory trend: {e}")
            return {"error": str(e)}
    
    def _get_memory_usage(self) -> int:
        """현재 메모리 사용량 조회 (바이트)"""
        try:
            if not PSUTIL_AVAILABLE:
                return 0
            process = psutil.Process()
            return process.memory_info().rss
        except Exception:
            return 0
    
    def _cleanup_old_cache(self) -> int:
        """오래된 캐시 정리"""
        try:
            current_time = time.time()
            cutoff_time = current_time - 3600  # 1시간 전
            
            # 오래된 캐시 항목 제거
            old_entries = [key for key, access_time in self.cache_access_times.items() 
                          if access_time < cutoff_time]
            
            for key in old_entries:
                del self.cache_access_times[key]
                if key in self.cache_stats:
                    del self.cache_stats[key]
            
            return len(old_entries)
            
        except Exception as e:
            self.logger.error(f"Error cleaning up cache: {e}")
            return 0
    
    def _get_memory_recommendations(self, trend: str, trend_rate: float, avg_usage: int) -> List[str]:
        """메모리 최적화 권장사항"""
        recommendations = []
        
        if trend == "increasing" and trend_rate > 20:
            recommendations.append("메모리 사용량이 급격히 증가하고 있습니다. 가비지 컬렉션을 더 자주 실행하세요.")
        
        if avg_usage > 1024 * 1024 * 1024:  # 1GB 이상
            recommendations.append("메모리 사용량이 높습니다. 캐시 크기를 줄이거나 불필요한 데이터를 정리하세요.")
        
        if trend == "increasing" and trend_rate > 10:
            recommendations.append("메모리 누수 가능성이 있습니다. 코드를 검토하여 메모리 해제를 확인하세요.")
        
        if not recommendations:
            recommendations.append("메모리 사용량이 정상 범위 내에 있습니다.")
        
        return recommendations


class CacheManager:
    """캐시 관리 클래스"""
    
    def __init__(self, max_size: int = 1000, ttl: int = 3600):
        """
        캐시 관리자 초기화
        
        Args:
            max_size: 최대 캐시 크기
            ttl: 캐시 생존 시간 (초)
        """
        self.logger = get_logger(__name__)
        self.max_size = max_size
        self.ttl = ttl
        
        # LRU 캐시 구현
        self.cache = {}
        self.access_order = deque()
        self.access_times = {}
        
        # 캐시 통계
        self.hits = 0
        self.misses = 0
        
        self.logger.trace(f"CacheManager initialized (max_size={max_size}, ttl={ttl})")
    
    def get(self, key: str) -> Optional[Any]:
        """캐시에서 값 조회"""
        try:
            if key in self.cache:
                # TTL 체크
                if time.time() - self.access_times[key] > self.ttl:
                    self._remove_key(key)
                    self.misses += 1
                    return None
                
                # 접근 시간 업데이트
                self.access_times[key] = time.time()
                self._update_access_order(key)
                self.hits += 1
                return self.cache[key]
            else:
                self.misses += 1
                return None
                
        except Exception as e:
            self.logger.error(f"Error getting cache value: {e}")
            return None
    
    def set(self, key: str, value: Any) -> bool:
        """캐시에 값 저장"""
        try:
            # 캐시 크기 체크
            if len(self.cache) >= self.max_size and key not in self.cache:
                self._evict_lru()
            
            self.cache[key] = value
            self.access_times[key] = time.time()
            self._update_access_order(key)
            
            return True
            
        except Exception as e:
            self.logger.error(f"Error setting cache value: {e}")
            return False
    
    def delete(self, key: str) -> bool:
        """캐시에서 값 삭제"""
        try:
            if key in self.cache:
                self._remove_key(key)
                return True
            return False
            
        except Exception as e:
            self.logger.error(f"Error deleting cache value: {e}")
            return False
    
    def clear(self) -> int:
        """캐시 전체 삭제"""
        try:
            count = len(self.cache)
            self.cache.clear()
            self.access_order.clear()
            self.access_times.clear()
            return count
            
        except Exception as e:
            self.logger.error(f"Error clearing cache: {e}")
            return 0
    
    def get_stats(self) -> Dict[str, Any]:
        """캐시 통계 조회"""
        try:
            total_requests = self.hits + self.misses
            hit_rate = self.hits / total_requests if total_requests > 0 else 0
            
            return {
                "cache_size": len(self.cache),
                "max_size": self.max_size,
                "hits": self.hits,
                "misses": self.misses,
                "hit_rate": round(hit_rate, 3),
                "ttl_seconds": self.ttl,
                "oldest_entry": min(self.access_times.values()) if self.access_times else None,
                "newest_entry": max(self.access_times.values()) if self.access_times else None
            }
            
        except Exception as e:
            self.logger.error(f"Error getting cache stats: {e}")
            return {}
    
    def _remove_key(self, key: str):
        """키 제거"""
        if key in self.cache:
            del self.cache[key]
        if key in self.access_times:
            del self.access_times[key]
        if key in self.access_order:
            self.access_order.remove(key)
    
    def _update_access_order(self, key: str):
        """접근 순서 업데이트"""
        if key in self.access_order:
            self.access_order.remove(key)
        self.access_order.append(key)
    
    def _evict_lru(self):
        """LRU 항목 제거"""
        if self.access_order:
            oldest_key = self.access_order.popleft()
            self._remove_key(oldest_key)


def performance_monitor(func: Callable) -> Callable:
    """성능 모니터링 데코레이터"""
    @wraps(func)
    async def async_wrapper(*args, **kwargs):
        start_time = time.time()
        try:
            result = await func(*args, **kwargs)
            return result
        finally:
            execution_time = time.time() - start_time
            logger.trace(f"{func.__name__} executed in {execution_time:.3f}s")
    
    @wraps(func)
    def sync_wrapper(*args, **kwargs):
        start_time = time.time()
        try:
            result = func(*args, **kwargs)
            return result
        finally:
            execution_time = time.time() - start_time
            logger.trace(f"{func.__name__} executed in {execution_time:.3f}s")
    
    if asyncio.iscoroutinefunction(func):
        return async_wrapper
    else:
        return sync_wrapper


def memory_optimized(func: Callable) -> Callable:
    """메모리 최적화 데코레이터"""
    @wraps(func)
    def wrapper(*args, **kwargs):
        # 함수 실행 전 메모리 상태
        before_memory = 0
        if PSUTIL_AVAILABLE:
            try:
                before_memory = psutil.Process().memory_info().rss
            except Exception:
                pass
        
        try:
            result = func(*args, **kwargs)
            return result
        finally:
            if PSUTIL_AVAILABLE:
                try:
                    # 함수 실행 후 메모리 상태
                    after_memory = psutil.Process().memory_info().rss
                    memory_diff = after_memory - before_memory
                    
                    if memory_diff > 10 * 1024 * 1024:  # 10MB 이상 증가
                        logger.warning(f"{func.__name__} used {memory_diff / 1024 / 1024:.1f}MB memory")
                    
                    # 메모리 사용량이 높으면 가비지 컬렉션 실행
                    if psutil.virtual_memory().percent > 80:
                        gc.collect()
                except Exception:
                    pass
    
    return wrapper


# 전역 인스턴스
performance_monitor_instance = PerformanceMonitor()
memory_optimizer_instance = MemoryOptimizer()
cache_manager_instance = CacheManager()


# 테스트 함수
def test_performance_optimization():
    """성능 최적화 테스트"""
    print("=== 성능 최적화 테스트 ===")
    
    # 1. 성능 모니터 테스트
    print("\n1. 성능 모니터 테스트")
    performance_monitor_instance.record_metrics(
        response_time=1.5,
        active_sessions=5,
        cache_hit_rate=0.85,
        db_query_time=0.3
    )
    
    summary = performance_monitor_instance.get_performance_summary()
    print(f"성능 요약: {summary}")
    
    health = performance_monitor_instance.get_system_health()
    print(f"시스템 건강 상태: {health['status']}")
    
    # 2. 메모리 최적화 테스트
    print("\n2. 메모리 최적화 테스트")
    optimization_result = memory_optimizer_instance.optimize_memory()
    print(f"메모리 최적화 결과: {optimization_result}")
    
    memory_usage = memory_optimizer_instance.get_memory_usage()
    print(f"메모리 사용량: {memory_usage.memory_percentage:.1f}%")
    
    trend = memory_optimizer_instance.monitor_memory_trend()
    print(f"메모리 트렌드: {trend}")
    
    # 3. 캐시 관리 테스트
    print("\n3. 캐시 관리 테스트")
    cache_manager_instance.set("test_key", "test_value")
    cached_value = cache_manager_instance.get("test_key")
    print(f"캐시된 값: {cached_value}")
    
    cache_stats = cache_manager_instance.get_stats()
    print(f"캐시 통계: {cache_stats}")
    
    # 4. 데코레이터 테스트
    print("\n4. 데코레이터 테스트")
    
    @performance_monitor
    def test_function():
        time.sleep(0.1)
        return "test result"
    
    @memory_optimized
    def memory_intensive_function():
        data = [i for i in range(10000)]
        return len(data)
    
    result1 = test_function()
    result2 = memory_intensive_function()
    print(f"테스트 함수 결과: {result1}")
    print(f"메모리 집약적 함수 결과: {result2}")
    
    print("\n테스트 완료")


if __name__ == "__main__":
    test_performance_optimization()
