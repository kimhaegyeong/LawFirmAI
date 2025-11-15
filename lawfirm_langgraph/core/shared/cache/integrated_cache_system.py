#!/usr/bin/env python3
"""
통합 캐싱 시스템
질문 분류, 검색 결과, 답변 생성을 위한 다층 캐싱
"""

import hashlib
import json
import time
import threading
from typing import Dict, List, Any, Optional, Union, Callable
from dataclasses import dataclass, asdict
from collections import OrderedDict
import pickle
import os

@dataclass
class CacheEntry:
    """캐시 엔트리"""
    data: Any
    timestamp: float
    ttl: float
    access_count: int = 0
    last_access: float = 0.0

class LRUCache:
    """LRU 캐시 구현"""
    
    def __init__(self, max_size: int = 1000):
        self.max_size = max_size
        self.cache: OrderedDict[str, CacheEntry] = OrderedDict()
        self.lock = threading.RLock()
    
    def get(self, key: str) -> Optional[Any]:
        """캐시에서 값 가져오기"""
        with self.lock:
            if key not in self.cache:
                return None
            
            entry = self.cache[key]
            
            # TTL 확인
            if time.time() - entry.timestamp > entry.ttl:
                del self.cache[key]
                return None
            
            # 접근 정보 업데이트
            entry.access_count += 1
            entry.last_access = time.time()
            
            # LRU 순서 업데이트
            self.cache.move_to_end(key)
            
            return entry.data
    
    def put(self, key: str, value: Any, ttl: float = 300.0):
        """캐시에 값 저장"""
        with self.lock:
            # 캐시 크기 제한
            if len(self.cache) >= self.max_size:
                # 가장 오래된 항목 제거
                self.cache.popitem(last=False)
            
            entry = CacheEntry(
                data=value,
                timestamp=time.time(),
                ttl=ttl,
                access_count=1,
                last_access=time.time()
            )
            
            self.cache[key] = entry
    
    def clear(self):
        """캐시 정리"""
        with self.lock:
            self.cache.clear()
    
    def get_stats(self) -> Dict[str, Any]:
        """캐시 통계 반환"""
        with self.lock:
            if not self.cache:
                return {
                    'size': 0,
                    'max_size': self.max_size,
                    'hit_rate': 0.0,
                    'avg_access_count': 0.0
                }
            
            total_access = sum(entry.access_count for entry in self.cache.values())
            avg_access = total_access / len(self.cache)
            
            return {
                'size': len(self.cache),
                'max_size': self.max_size,
                'avg_access_count': avg_access,
                'total_access': total_access
            }

class PersistentCache:
    """영구 캐시 (디스크 저장)"""
    
    def __init__(self, cache_dir: str = "cache", max_size: int = 10000):
        self.cache_dir = cache_dir
        self.max_size = max_size
        self.lock = threading.RLock()
        
        # 캐시 디렉토리 생성
        os.makedirs(cache_dir, exist_ok=True)
        
        # 메모리 인덱스
        self.index: Dict[str, str] = {}  # key -> filename
        self.access_times: Dict[str, float] = {}
    
    def _get_filename(self, key: str) -> str:
        """키에 대한 파일명 생성"""
        hash_key = hashlib.md5(key.encode()).hexdigest()
        return os.path.join(self.cache_dir, f"{hash_key}.cache")
    
    def get(self, key: str) -> Optional[Any]:
        """캐시에서 값 가져오기"""
        with self.lock:
            if key not in self.index:
                return None
            
            filename = self.index[key]
            if not os.path.exists(filename):
                del self.index[key]
                if key in self.access_times:
                    del self.access_times[key]
                return None
            
            try:
                with open(filename, 'rb') as f:
                    entry = pickle.load(f)
                
                # TTL 확인
                if time.time() - entry.timestamp > entry.ttl:
                    os.remove(filename)
                    del self.index[key]
                    if key in self.access_times:
                        del self.access_times[key]
                    return None
                
                # 접근 시간 업데이트
                self.access_times[key] = time.time()
                
                return entry.data
                
            except Exception as e:
                print(f"Cache read error: {e}")
                return None
    
    def put(self, key: str, value: Any, ttl: float = 3600.0):
        """캐시에 값 저장"""
        with self.lock:
            # 캐시 크기 제한
            if len(self.index) >= self.max_size:
                self._evict_oldest()
            
            filename = self._get_filename(key)
            
            entry = CacheEntry(
                data=value,
                timestamp=time.time(),
                ttl=ttl,
                access_count=1,
                last_access=time.time()
            )
            
            try:
                with open(filename, 'wb') as f:
                    pickle.dump(entry, f)
                
                self.index[key] = filename
                self.access_times[key] = time.time()
                
            except Exception as e:
                print(f"Cache write error: {e}")
    
    def _evict_oldest(self):
        """가장 오래된 항목 제거"""
        if not self.access_times:
            return
        
        oldest_key = min(self.access_times.keys(), key=lambda k: self.access_times[k])
        filename = self.index[oldest_key]
        
        try:
            os.remove(filename)
        except OSError:
            pass
        
        del self.index[oldest_key]
        del self.access_times[oldest_key]
    
    def clear(self):
        """캐시 정리"""
        with self.lock:
            for filename in self.index.values():
                try:
                    os.remove(filename)
                except OSError:
                    pass
            
            self.index.clear()
            self.access_times.clear()

class IntegratedCacheSystem:
    """통합 캐싱 시스템"""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or {}
        
        # 캐시 레벨 설정
        self.question_classification_cache = LRUCache(
            max_size=self.config.get('classification_cache_size', 500)
        )
        self.search_results_cache = LRUCache(
            max_size=self.config.get('search_cache_size', 1000)
        )
        self.answer_generation_cache = LRUCache(
            max_size=self.config.get('answer_cache_size', 200)
        )
        
        # 영구 캐시 (긴 TTL 데이터용)
        self.persistent_cache = PersistentCache(
            cache_dir=self.config.get('persistent_cache_dir', 'cache'),
            max_size=self.config.get('persistent_cache_size', 5000)
        )
        
        # 성능 통계
        self.stats = {
            'classification_hits': 0,
            'classification_misses': 0,
            'search_hits': 0,
            'search_misses': 0,
            'answer_hits': 0,
            'answer_misses': 0,
            'persistent_hits': 0,
            'persistent_misses': 0
        }
    
    def _generate_key(self, prefix: str, *args) -> str:
        """캐시 키 생성"""
        key_string = f"{prefix}:{':'.join(str(arg) for arg in args)}"
        return hashlib.md5(key_string.encode()).hexdigest()
    
    def get_question_classification(self, query: str) -> Optional[Any]:
        """질문 분류 결과 가져오기"""
        key = self._generate_key("classification", query)
        result = self.question_classification_cache.get(key)
        
        if result is not None:
            self.stats['classification_hits'] += 1
        else:
            self.stats['classification_misses'] += 1
        
        return result
    
    def put_question_classification(self, query: str, classification: Any, ttl: float = 1800.0):
        """질문 분류 결과 저장"""
        key = self._generate_key("classification", query)
        self.question_classification_cache.put(key, classification, ttl)
    
    def get_search_results(self, query: str, question_type: str, max_results: int) -> Optional[List[Any]]:
        """검색 결과 가져오기"""
        key = self._generate_key("search", query, question_type, max_results)
        result = self.search_results_cache.get(key)
        
        if result is not None:
            self.stats['search_hits'] += 1
        else:
            self.stats['search_misses'] += 1
        
        return result
    
    def put_search_results(self, query: str, question_type: str, max_results: int, 
                          results: List[Any], ttl: float = 600.0):
        """검색 결과 저장"""
        key = self._generate_key("search", query, question_type, max_results)
        self.search_results_cache.put(key, results, ttl)
    
    def get_answer(self, query: str, question_type: str, context_hash: str) -> Optional[str]:
        """생성된 답변 가져오기"""
        key = self._generate_key("answer", query, question_type, context_hash)
        result = self.answer_generation_cache.get(key)
        
        if result is not None:
            self.stats['answer_hits'] += 1
        else:
            self.stats['answer_misses'] += 1
        
        return result
    
    def put_answer(self, query: str, question_type: str, context_hash: str, 
                   answer: str, ttl: float = 3600.0):
        """생성된 답변 저장"""
        key = self._generate_key("answer", query, question_type, context_hash)
        self.answer_generation_cache.put(key, answer, ttl)
    
    def get_persistent(self, key: str) -> Optional[Any]:
        """영구 캐시에서 값 가져오기"""
        result = self.persistent_cache.get(key)
        
        if result is not None:
            self.stats['persistent_hits'] += 1
        else:
            self.stats['persistent_misses'] += 1
        
        return result
    
    def put_persistent(self, key: str, value: Any, ttl: float = 86400.0):
        """영구 캐시에 값 저장"""
        self.persistent_cache.put(key, value, ttl)
    
    def cached_function(self, cache_type: str, ttl: float = 300.0):
        """함수 캐싱 데코레이터"""
        def decorator(func: Callable):
            def wrapper(*args, **kwargs):
                # 캐시 키 생성
                cache_key = self._generate_key(cache_type, func.__name__, str(args), str(kwargs))
                
                # 캐시 확인
                if cache_type == "persistent":
                    result = self.persistent_cache.get(cache_key)
                else:
                    result = self.question_classification_cache.get(cache_key)
                
                if result is not None:
                    return result
                
                # 함수 실행
                result = func(*args, **kwargs)
                
                # 결과 캐싱
                if cache_type == "persistent":
                    self.persistent_cache.put(cache_key, result, ttl)
                else:
                    self.question_classification_cache.put(cache_key, result, ttl)
                
                return result
            
            return wrapper
        return decorator
    
    def get_stats(self) -> Dict[str, Any]:
        """전체 캐시 통계 반환"""
        total_classification = self.stats['classification_hits'] + self.stats['classification_misses']
        total_search = self.stats['search_hits'] + self.stats['search_misses']
        total_answer = self.stats['answer_hits'] + self.stats['answer_misses']
        total_persistent = self.stats['persistent_hits'] + self.stats['persistent_misses']
        
        return {
            'classification_cache': {
                'hit_rate': self.stats['classification_hits'] / total_classification if total_classification > 0 else 0,
                'stats': self.question_classification_cache.get_stats()
            },
            'search_cache': {
                'hit_rate': self.stats['search_hits'] / total_search if total_search > 0 else 0,
                'stats': self.search_results_cache.get_stats()
            },
            'answer_cache': {
                'hit_rate': self.stats['answer_hits'] / total_answer if total_answer > 0 else 0,
                'stats': self.answer_generation_cache.get_stats()
            },
            'persistent_cache': {
                'hit_rate': self.stats['persistent_hits'] / total_persistent if total_persistent > 0 else 0,
                'size': len(self.persistent_cache.index)
            },
            'overall_stats': self.stats
        }
    
    def clear_all(self):
        """모든 캐시 정리"""
        self.question_classification_cache.clear()
        self.search_results_cache.clear()
        self.answer_generation_cache.clear()
        self.persistent_cache.clear()
        
        # 통계 초기화
        self.stats = {
            'classification_hits': 0,
            'classification_misses': 0,
            'search_hits': 0,
            'search_misses': 0,
            'answer_hits': 0,
            'answer_misses': 0,
            'persistent_hits': 0,
            'persistent_misses': 0
        }

# 전역 캐시 시스템 인스턴스
cache_system = IntegratedCacheSystem()
