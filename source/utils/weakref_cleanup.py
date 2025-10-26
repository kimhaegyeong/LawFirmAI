# -*- coding: utf-8 -*-
"""
WeakRef Memory Cleanup System
WeakRef를 활용한 메모리 정리 시스템
"""

import weakref
import gc
import threading
import time
from typing import Dict, List, Any, Optional, Callable, Set
from datetime import datetime
from dataclasses import dataclass
from .logger import get_logger

logger = get_logger(__name__)


@dataclass
class ObjectInfo:
    """객체 정보 데이터 클래스"""
    name: str
    obj_type: str
    created_at: datetime
    last_accessed: datetime
    access_count: int
    size_estimate: int  # 바이트 단위 추정 크기


class WeakRefRegistry:
    """WeakRef 기반 객체 등록 시스템"""

    def __init__(self):
        self.logger = get_logger(__name__)

        # WeakRef 저장소
        self._weak_refs: Dict[str, weakref.ref] = {}
        self._object_info: Dict[str, ObjectInfo] = {}

        # 콜백 함수 저장소
        self._cleanup_callbacks: Dict[str, Callable] = {}

        # 접근 통계
        self._access_stats: Dict[str, int] = {}

        # 스레드 안전성을 위한 락
        self._lock = threading.RLock()

        self.logger.info("WeakRefRegistry 초기화 완료")

    def register_object(self,
                       obj: Any,
                       name: str,
                       cleanup_callback: Optional[Callable] = None,
                       size_estimate: Optional[int] = None) -> bool:
        """
        객체를 WeakRef로 등록

        Args:
            obj: 등록할 객체
            name: 객체 식별자
            cleanup_callback: 객체 삭제 시 호출될 콜백 함수
            size_estimate: 객체 크기 추정값 (바이트)

        Returns:
            등록 성공 여부
        """
        with self._lock:
            try:
                # 기존 등록 확인
                if name in self._weak_refs:
                    self.logger.warning(f"객체 '{name}'이 이미 등록되어 있습니다.")
                    return False

                # WeakRef 생성 (콜백 함수 포함)
                def cleanup_callback_wrapper(ref):
                    self._on_object_deleted(name)
                    if cleanup_callback:
                        try:
                            cleanup_callback(name)
                        except Exception as e:
                            self.logger.error(f"정리 콜백 실행 실패 ({name}): {e}")

                weak_ref = weakref.ref(obj, cleanup_callback_wrapper)

                # 등록
                self._weak_refs[name] = weak_ref
                self._object_info[name] = ObjectInfo(
                    name=name,
                    obj_type=type(obj).__name__,
                    created_at=datetime.now(),
                    last_accessed=datetime.now(),
                    access_count=0,
                    size_estimate=size_estimate or self._estimate_object_size(obj)
                )

                if cleanup_callback:
                    self._cleanup_callbacks[name] = cleanup_callback

                self.logger.debug(f"객체 등록 완료: {name} ({type(obj).__name__})")
                return True

            except Exception as e:
                self.logger.error(f"객체 등록 실패 ({name}): {e}")
                return False

    def unregister_object(self, name: str) -> bool:
        """객체 등록 해제"""
        with self._lock:
            try:
                if name in self._weak_refs:
                    del self._weak_refs[name]
                    del self._object_info[name]

                    if name in self._cleanup_callbacks:
                        del self._cleanup_callbacks[name]

                    if name in self._access_stats:
                        del self._access_stats[name]

                    self.logger.debug(f"객체 등록 해제 완료: {name}")
                    return True
                else:
                    self.logger.warning(f"등록되지 않은 객체: {name}")
                    return False

            except Exception as e:
                self.logger.error(f"객체 등록 해제 실패 ({name}): {e}")
                return False

    def get_object(self, name: str) -> Optional[Any]:
        """등록된 객체 가져오기"""
        with self._lock:
            if name not in self._weak_refs:
                return None

            weak_ref = self._weak_refs[name]
            obj = weak_ref()

            if obj is None:
                # 객체가 이미 삭제됨
                self._cleanup_dead_reference(name)
                return None

            # 접근 통계 업데이트
            self._update_access_stats(name)

            return obj

    def _on_object_deleted(self, name: str):
        """객체 삭제 시 호출되는 콜백"""
        with self._lock:
            self.logger.debug(f"객체 삭제됨: {name}")

            # 정리 작업
            if name in self._object_info:
                del self._object_info[name]

            if name in self._access_stats:
                del self._access_stats[name]

    def _cleanup_dead_reference(self, name: str):
        """죽은 참조 정리"""
        with self._lock:
            if name in self._weak_refs:
                del self._weak_refs[name]

            if name in self._object_info:
                del self._object_info[name]

            if name in self._cleanup_callbacks:
                del self._cleanup_callbacks[name]

            if name in self._access_stats:
                del self._access_stats[name]

            self.logger.debug(f"죽은 참조 정리 완료: {name}")

    def _update_access_stats(self, name: str):
        """접근 통계 업데이트"""
        if name in self._object_info:
            self._object_info[name].last_accessed = datetime.now()
            self._object_info[name].access_count += 1

        self._access_stats[name] = self._access_stats.get(name, 0) + 1

    def _estimate_object_size(self, obj: Any) -> int:
        """객체 크기 추정"""
        try:
            import sys
            return sys.getsizeof(obj)
        except Exception:
            return 0

    def cleanup_dead_references(self) -> Dict[str, Any]:
        """죽은 참조들 정리"""
        with self._lock:
            initial_count = len(self._weak_refs)
            dead_refs = []

            # 죽은 참조 찾기
            for name, weak_ref in list(self._weak_refs.items()):
                if weak_ref() is None:
                    dead_refs.append(name)

            # 죽은 참조 정리
            for name in dead_refs:
                self._cleanup_dead_reference(name)

            final_count = len(self._weak_refs)
            cleaned_count = initial_count - final_count

            result = {
                'success': True,
                'initial_refs': initial_count,
                'final_refs': final_count,
                'cleaned_refs': cleaned_count,
                'timestamp': datetime.now()
            }

            if cleaned_count > 0:
                self.logger.info(f"죽은 참조 정리 완료: {cleaned_count}개 참조 제거")

            return result

    def get_registry_stats(self) -> Dict[str, Any]:
        """등록소 통계 반환"""
        with self._lock:
            total_refs = len(self._weak_refs)
            alive_refs = sum(1 for ref in self._weak_refs.values() if ref() is not None)
            dead_refs = total_refs - alive_refs

            # 객체 타입별 통계
            type_stats = {}
            for info in self._object_info.values():
                obj_type = info.obj_type
                type_stats[obj_type] = type_stats.get(obj_type, 0) + 1

            # 크기 통계
            total_size = sum(info.size_estimate for info in self._object_info.values())

            # 접근 통계
            total_accesses = sum(self._access_stats.values())
            avg_accesses = total_accesses / len(self._access_stats) if self._access_stats else 0

            return {
                'total_registered': total_refs,
                'alive_objects': alive_refs,
                'dead_references': dead_refs,
                'type_distribution': type_stats,
                'total_size_bytes': total_size,
                'total_size_mb': total_size / 1024 / 1024,
                'total_accesses': total_accesses,
                'average_accesses': avg_accesses,
                'most_accessed': max(self._access_stats.items(), key=lambda x: x[1]) if self._access_stats else None,
                'timestamp': datetime.now()
            }

    def get_object_info(self, name: str) -> Optional[ObjectInfo]:
        """특정 객체 정보 반환"""
        with self._lock:
            return self._object_info.get(name)

    def list_all_objects(self) -> List[Dict[str, Any]]:
        """모든 등록된 객체 목록 반환"""
        with self._lock:
            objects = []
            for name, weak_ref in self._weak_refs.items():
                obj = weak_ref()
                info = self._object_info.get(name)

                if info:
                    objects.append({
                        'name': name,
                        'type': info.obj_type,
                        'alive': obj is not None,
                        'created_at': info.created_at,
                        'last_accessed': info.last_accessed,
                        'access_count': info.access_count,
                        'size_bytes': info.size_estimate,
                        'current_accesses': self._access_stats.get(name, 0)
                    })

            return objects

    def force_cleanup(self) -> Dict[str, Any]:
        """강제 정리 수행"""
        with self._lock:
            # 죽은 참조 정리
            cleanup_result = self.cleanup_dead_references()

            # 가비지 컬렉션 실행
            collected = gc.collect()

            # 통계 업데이트
            stats = self.get_registry_stats()

            result = {
                'success': True,
                'cleanup_result': cleanup_result,
                'garbage_collected': collected,
                'registry_stats': stats,
                'timestamp': datetime.now()
            }

            self.logger.info(f"강제 정리 완료: {cleanup_result['cleaned_refs']}개 참조, {collected}개 객체 수집")

            return result


class MemoryOptimizer:
    """메모리 최적화 도구"""

    def __init__(self, registry: WeakRefRegistry):
        self.registry = registry
        self.logger = get_logger(__name__)

    def optimize_memory_usage(self,
                             max_memory_mb: float = 100.0,
                             cleanup_threshold: float = 0.8) -> Dict[str, Any]:
        """메모리 사용량 최적화"""
        try:
            stats = self.registry.get_registry_stats()
            current_memory_mb = stats['total_size_mb']

            if current_memory_mb <= max_memory_mb * cleanup_threshold:
                return {
                    'optimization_needed': False,
                    'current_memory_mb': current_memory_mb,
                    'threshold_mb': max_memory_mb * cleanup_threshold,
                    'message': '메모리 사용량이 임계값 이하입니다.'
                }

            # 최적화 전략 실행
            optimization_results = []

            # 1. 죽은 참조 정리
            cleanup_result = self.registry.cleanup_dead_references()
            optimization_results.append(cleanup_result)

            # 2. 접근 빈도가 낮은 객체 정리
            low_access_cleanup = self._cleanup_low_access_objects()
            optimization_results.append(low_access_cleanup)

            # 3. 큰 객체 우선 정리
            large_object_cleanup = self._cleanup_large_objects()
            optimization_results.append(large_object_cleanup)

            # 4. 가비지 컬렉션 실행
            gc_result = gc.collect()

            # 최종 통계
            final_stats = self.registry.get_registry_stats()

            result = {
                'optimization_needed': True,
                'initial_memory_mb': current_memory_mb,
                'final_memory_mb': final_stats['total_size_mb'],
                'memory_freed_mb': current_memory_mb - final_stats['total_size_mb'],
                'optimization_results': optimization_results,
                'garbage_collected': gc_result,
                'final_stats': final_stats,
                'timestamp': datetime.now()
            }

            self.logger.info(f"메모리 최적화 완료: {result['memory_freed_mb']:.1f}MB 해제")

            return result

        except Exception as e:
            self.logger.error(f"메모리 최적화 실패: {e}")
            return {
                'success': False,
                'error': str(e),
                'timestamp': datetime.now()
            }

    def _cleanup_low_access_objects(self,
                                  min_access_count: int = 5,
                                  max_age_hours: int = 24) -> Dict[str, Any]:
        """접근 빈도가 낮은 객체 정리"""
        try:
            objects = self.registry.list_all_objects()
            cutoff_time = datetime.now() - timedelta(hours=max_age_hours)

            cleanup_candidates = []
            for obj_info in objects:
                if (obj_info['access_count'] < min_access_count and
                    obj_info['last_accessed'] < cutoff_time):
                    cleanup_candidates.append(obj_info['name'])

            cleaned_count = 0
            for name in cleanup_candidates:
                if self.registry.unregister_object(name):
                    cleaned_count += 1

            return {
                'strategy': 'low_access_cleanup',
                'candidates_found': len(cleanup_candidates),
                'cleaned_count': cleaned_count,
                'min_access_count': min_access_count,
                'max_age_hours': max_age_hours
            }

        except Exception as e:
            self.logger.error(f"낮은 접근 빈도 객체 정리 실패: {e}")
            return {'strategy': 'low_access_cleanup', 'error': str(e)}

    def _cleanup_large_objects(self,
                             min_size_mb: float = 1.0,
                             max_age_hours: int = 12) -> Dict[str, Any]:
        """큰 객체 정리"""
        try:
            objects = self.registry.list_all_objects()
            cutoff_time = datetime.now() - timedelta(hours=max_age_hours)
            min_size_bytes = min_size_mb * 1024 * 1024

            cleanup_candidates = []
            for obj_info in objects:
                if (obj_info['size_bytes'] > min_size_bytes and
                    obj_info['last_accessed'] < cutoff_time):
                    cleanup_candidates.append(obj_info['name'])

            cleaned_count = 0
            total_size_freed = 0
            for name in cleanup_candidates:
                obj_info = self.registry.get_object_info(name)
                if obj_info and self.registry.unregister_object(name):
                    cleaned_count += 1
                    total_size_freed += obj_info.size_estimate

            return {
                'strategy': 'large_object_cleanup',
                'candidates_found': len(cleanup_candidates),
                'cleaned_count': cleaned_count,
                'size_freed_bytes': total_size_freed,
                'size_freed_mb': total_size_freed / 1024 / 1024,
                'min_size_mb': min_size_mb,
                'max_age_hours': max_age_hours
            }

        except Exception as e:
            self.logger.error(f"큰 객체 정리 실패: {e}")
            return {'strategy': 'large_object_cleanup', 'error': str(e)}


# 전역 WeakRef 등록소 인스턴스
_weakref_registry_instance: Optional[WeakRefRegistry] = None


def get_weakref_registry() -> WeakRefRegistry:
    """WeakRef 등록소 싱글톤 인스턴스 반환"""
    global _weakref_registry_instance

    if _weakref_registry_instance is None:
        _weakref_registry_instance = WeakRefRegistry()

    return _weakref_registry_instance


def cleanup_weakref_registry():
    """WeakRef 등록소 정리"""
    global _weakref_registry_instance

    if _weakref_registry_instance:
        _weakref_registry_instance.force_cleanup()
        _weakref_registry_instance = None
