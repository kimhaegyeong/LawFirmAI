# -*- coding: utf-8 -*-
"""
Cache Manager
캐싱 시스템 관리 모듈
"""

import logging
import time
import json
import hashlib
import threading
import asyncio
from typing import Any, Dict, Optional, List, Union
from dataclasses import dataclass, asdict
from datetime import datetime, timedelta
from collections import OrderedDict
import pickle
import os
from pathlib import Path

logger = logging.getLogger(__name__)


@dataclass
class CacheEntry:
    """캐시 엔트리 데이터 클래스"""
    key: str
    value: Any
    created_at: float
    expires_at: Optional[float] = None
    access_count: int = 0
    last_accessed: float = 0.0
    size_bytes: int = 0
    
    def is_expired(self) -> bool:
        """캐시 엔트리가 만료되었는지 확인"""
        if self.expires_at is None:
            return False
        return time.time() > self.expires_at
    
    def touch(self):
        """접근 시간 업데이트"""
        self.last_accessed = time.time()
        self.access_count += 1


class CacheManager:
    """캐시 관리자 클래스"""
    
    def __init__(self, 
                 max_size_mb: int = 100,
                 default_ttl_seconds: int = 3600,
                 enable_persistence: bool = True,
                 cache_dir: str = "data/cache"):
        """
        캐시 관리자 초기화
        
        Args:
            max_size_mb: 최대 캐시 크기 (MB)
            default_ttl_seconds: 기본 TTL (초)
            enable_persistence: 영구 저장 활성화
            cache_dir: 캐시 디렉토리
        """
        self.max_size_bytes = max_size_mb * 1024 * 1024
        self.default_ttl = default_ttl_seconds
        self.enable_persistence = enable_persistence
        self.cache_dir = Path(cache_dir)
        
        # 메모리 캐시
        self._cache: OrderedDict[str, CacheEntry] = OrderedDict()
        self._lock = threading.RLock()
        
        # 통계
        self._stats = {
            'hits': 0,
            'misses': 0,
            'evictions': 0,
            'total_size_bytes': 0
        }
        
        # 캐시 디렉토리 생성
        if self.enable_persistence:
            self.cache_dir.mkdir(parents=True, exist_ok=True)
        
        logger.info(f"CacheManager initialized: max_size={max_size_mb}MB, ttl={default_ttl_seconds}s")
    
    def _generate_key(self, prefix: str, *args, **kwargs) -> str:
        """캐시 키 생성"""
        # 인자들을 문자열로 변환하여 해시 생성
        key_data = {
            'prefix': prefix,
            'args': args,
            'kwargs': sorted(kwargs.items()) if kwargs else {}
        }
        
        key_string = json.dumps(key_data, sort_keys=True, ensure_ascii=False)
        return hashlib.md5(key_string.encode('utf-8')).hexdigest()
    
    def _calculate_size(self, value: Any) -> int:
        """값의 크기 계산 (바이트)"""
        try:
            if isinstance(value, (str, bytes)):
                return len(value)
            elif isinstance(value, (int, float)):
                return 8
            elif isinstance(value, (list, tuple, dict)):
                return len(pickle.dumps(value))
            else:
                return len(pickle.dumps(value))
        except Exception:
            return 1024  # 기본값
    
    def _evict_lru(self):
        """LRU 방식으로 캐시 엔트리 제거"""
        with self._lock:
            if not self._cache:
                return
            
            # 가장 오래된 엔트리 제거
            oldest_key = next(iter(self._cache))
            oldest_entry = self._cache.pop(oldest_key)
            
            self._stats['total_size_bytes'] -= oldest_entry.size_bytes
            self._stats['evictions'] += 1
            
            logger.debug(f"Evicted cache entry: {oldest_key}")
    
    def _make_space(self, required_bytes: int):
        """필요한 공간 확보"""
        while (self._stats['total_size_bytes'] + required_bytes > self.max_size_bytes and 
               self._cache):
            self._evict_lru()
    
    def get(self, key: str) -> Optional[Any]:
        """캐시에서 값 가져오기"""
        with self._lock:
            if key not in self._cache:
                self._stats['misses'] += 1
                return None
            
            entry = self._cache[key]
            
            # 만료 확인
            if entry.is_expired():
                del self._cache[key]
                self._stats['total_size_bytes'] -= entry.size_bytes
                self._stats['misses'] += 1
                return None
            
            # LRU 업데이트
            entry.touch()
            self._cache.move_to_end(key)
            
            self._stats['hits'] += 1
            return entry.value
    
    def set(self, key: str, value: Any, ttl_seconds: Optional[int] = None) -> bool:
        """캐시에 값 저장"""
        try:
            ttl = ttl_seconds or self.default_ttl
            expires_at = time.time() + ttl if ttl > 0 else None
            size_bytes = self._calculate_size(value)
            
            with self._lock:
                # 기존 엔트리 제거
                if key in self._cache:
                    old_entry = self._cache.pop(key)
                    self._stats['total_size_bytes'] -= old_entry.size_bytes
                
                # 공간 확보
                self._make_space(size_bytes)
                
                # 새 엔트리 생성
                entry = CacheEntry(
                    key=key,
                    value=value,
                    created_at=time.time(),
                    expires_at=expires_at,
                    size_bytes=size_bytes
                )
                entry.touch()
                
                self._cache[key] = entry
                self._stats['total_size_bytes'] += size_bytes
                
                # 영구 저장
                if self.enable_persistence:
                    self._persist_entry(key, entry)
                
                return True
                
        except Exception as e:
            logger.error(f"Failed to set cache entry {key}: {e}")
            return False
    
    def delete(self, key: str) -> bool:
        """캐시에서 값 삭제"""
        with self._lock:
            if key in self._cache:
                entry = self._cache.pop(key)
                self._stats['total_size_bytes'] -= entry.size_bytes
                
                # 영구 저장소에서도 삭제
                if self.enable_persistence:
                    self._delete_persisted_entry(key)
                
                return True
            return False
    
    def clear(self):
        """캐시 전체 삭제"""
        with self._lock:
            self._cache.clear()
            self._stats['total_size_bytes'] = 0
            
            # 영구 저장소도 삭제
            if self.enable_persistence:
                self._clear_persistent_cache()
    
    def _persist_entry(self, key: str, entry: CacheEntry):
        """캐시 엔트리를 영구 저장소에 저장"""
        try:
            cache_file = self.cache_dir / f"{key}.cache"
            
            # 직렬화 가능한 데이터만 저장
            persist_data = {
                'key': entry.key,
                'value': entry.value,
                'created_at': entry.created_at,
                'expires_at': entry.expires_at,
                'access_count': entry.access_count,
                'last_accessed': entry.last_accessed,
                'size_bytes': entry.size_bytes
            }
            
            with open(cache_file, 'wb') as f:
                pickle.dump(persist_data, f)
                
        except Exception as e:
            logger.warning(f"Failed to persist cache entry {key}: {e}")
    
    def _load_persisted_entry(self, key: str) -> Optional[CacheEntry]:
        """영구 저장소에서 캐시 엔트리 로드"""
        try:
            cache_file = self.cache_dir / f"{key}.cache"
            
            if not cache_file.exists():
                return None
            
            with open(cache_file, 'rb') as f:
                persist_data = pickle.load(f)
            
            # 만료 확인
            if persist_data.get('expires_at') and time.time() > persist_data['expires_at']:
                cache_file.unlink()  # 만료된 파일 삭제
                return None
            
            return CacheEntry(**persist_data)
            
        except Exception as e:
            logger.warning(f"Failed to load persisted cache entry {key}: {e}")
            return None
    
    def _delete_persisted_entry(self, key: str):
        """영구 저장소에서 캐시 엔트리 삭제"""
        try:
            cache_file = self.cache_dir / f"{key}.cache"
            if cache_file.exists():
                cache_file.unlink()
        except Exception as e:
            logger.warning(f"Failed to delete persisted cache entry {key}: {e}")
    
    def _clear_persistent_cache(self):
        """영구 저장소 캐시 전체 삭제"""
        try:
            for cache_file in self.cache_dir.glob("*.cache"):
                cache_file.unlink()
        except Exception as e:
            logger.warning(f"Failed to clear persistent cache: {e}")
    
    def get_stats(self) -> Dict[str, Any]:
        """캐시 통계 반환"""
        with self._lock:
            total_requests = self._stats['hits'] + self._stats['misses']
            hit_rate = self._stats['hits'] / total_requests if total_requests > 0 else 0
            
            return {
                'hits': self._stats['hits'],
                'misses': self._stats['misses'],
                'hit_rate': hit_rate,
                'evictions': self._stats['evictions'],
                'total_size_bytes': self._stats['total_size_bytes'],
                'total_size_mb': self._stats['total_size_bytes'] / (1024 * 1024),
                'max_size_mb': self.max_size_bytes / (1024 * 1024),
                'entry_count': len(self._cache),
                'utilization_rate': self._stats['total_size_bytes'] / self.max_size_bytes
            }
    
    def cleanup_expired(self):
        """만료된 캐시 엔트리 정리"""
        with self._lock:
            expired_keys = []
            
            for key, entry in self._cache.items():
                if entry.is_expired():
                    expired_keys.append(key)
            
            for key in expired_keys:
                entry = self._cache.pop(key)
                self._stats['total_size_bytes'] -= entry.size_bytes
                self._delete_persisted_entry(key)
            
            if expired_keys:
                logger.info(f"Cleaned up {len(expired_keys)} expired cache entries")


# 전역 캐시 매니저 인스턴스
_cache_manager = None

def get_cache_manager() -> CacheManager:
    """전역 캐시 매니저 인스턴스 반환"""
    global _cache_manager
    if _cache_manager is None:
        _cache_manager = CacheManager()
    return _cache_manager


def cached(prefix: str, ttl_seconds: Optional[int] = None):
    """캐싱 데코레이터"""
    def decorator(func):
        def wrapper(*args, **kwargs):
            cache_manager = get_cache_manager()
            
            # self 객체 제외하고 키 생성
            if args and hasattr(args[0], '__class__'):
                # 인스턴스 메서드인 경우 self 제외
                cache_key = cache_manager._generate_key(prefix, func.__name__, *args[1:], **kwargs)
            else:
                cache_key = cache_manager._generate_key(prefix, func.__name__, *args, **kwargs)
            
            # 캐시에서 확인
            cached_result = cache_manager.get(cache_key)
            if cached_result is not None:
                return cached_result
            
            # 함수 실행
            result = func(*args, **kwargs)
            
            # 결과 캐싱 (비동기 함수의 경우 결과만 캐싱)
            if asyncio.iscoroutine(result):
                # 비동기 함수의 경우 결과를 기다린 후 캐싱
                async def async_wrapper():
                    actual_result = await result
                    cache_manager.set(cache_key, actual_result, ttl_seconds)
                    return actual_result
                return async_wrapper()
            else:
                cache_manager.set(cache_key, result, ttl_seconds)
                return result
        return wrapper
    return decorator


class QueryCache:
    """쿼리 전용 캐시 클래스"""
    
    def __init__(self, cache_manager: CacheManager):
        self.cache_manager = cache_manager
        self.query_prefix = "query"
        self.embedding_prefix = "embedding"
        self.search_prefix = "search"
    
    def get_query_result(self, query: str, top_k: int = 10) -> Optional[List[Dict[str, Any]]]:
        """쿼리 결과 캐시에서 가져오기"""
        cache_key = self.cache_manager._generate_key(
            self.query_prefix, query, top_k
        )
        return self.cache_manager.get(cache_key)
    
    def set_query_result(self, query: str, result: List[Dict[str, Any]], 
                        top_k: int = 10, ttl_seconds: int = 1800) -> bool:
        """쿼리 결과 캐시에 저장"""
        cache_key = self.cache_manager._generate_key(
            self.query_prefix, query, top_k
        )
        return self.cache_manager.set(cache_key, result, ttl_seconds)
    
    def get_embedding(self, text: str) -> Optional[List[float]]:
        """임베딩 캐시에서 가져오기"""
        cache_key = self.cache_manager._generate_key(
            self.embedding_prefix, text
        )
        return self.cache_manager.get(cache_key)
    
    def set_embedding(self, text: str, embedding: List[float], 
                     ttl_seconds: int = 7200) -> bool:
        """임베딩 캐시에 저장"""
        cache_key = self.cache_manager._generate_key(
            self.embedding_prefix, text
        )
        return self.cache_manager.set(cache_key, embedding, ttl_seconds)
    
    def get_search_result(self, query: str, filters: Dict = None, 
                         top_k: int = 10) -> Optional[List[Dict[str, Any]]]:
        """검색 결과 캐시에서 가져오기"""
        cache_key = self.cache_manager._generate_key(
            self.search_prefix, query, filters, top_k
        )
        return self.cache_manager.get(cache_key)
    
    def set_search_result(self, query: str, result: List[Dict[str, Any]], 
                         filters: Dict = None, top_k: int = 10, 
                         ttl_seconds: int = 1800) -> bool:
        """검색 결과 캐시에 저장"""
        cache_key = self.cache_manager._generate_key(
            self.search_prefix, query, filters, top_k
        )
        return self.cache_manager.set(cache_key, result, ttl_seconds)
