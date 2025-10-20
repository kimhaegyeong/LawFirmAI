#!/usr/bin/env python3
"""
최적화된 모델 관리자
싱글톤 패턴과 지연 로딩을 통한 성능 최적화
"""

import threading
import time
from typing import Dict, Optional, Any
from functools import lru_cache
import logging

# 외부 라이브러리 임포트
try:
    from sentence_transformers import SentenceTransformer
    SENTENCE_TRANSFORMERS_AVAILABLE = True
except ImportError:
    SENTENCE_TRANSFORMERS_AVAILABLE = False

try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

class OptimizedModelManager:
    """최적화된 모델 관리자 (싱글톤 패턴)"""
    
    _instance = None
    _lock = threading.Lock()
    
    def __new__(cls):
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
                    cls._instance._initialized = False
        return cls._instance
    
    def __init__(self):
        if self._initialized:
            return
            
        self._initialized = True
        self.logger = logging.getLogger(__name__)
        
        # 모델 캐시
        self._models: Dict[str, Any] = {}
        self._model_locks: Dict[str, threading.Lock] = {}
        self._loading_flags: Dict[str, bool] = {}
        
        # 성능 통계
        self.stats = {
            'model_loads': 0,
            'cache_hits': 0,
            'total_load_time': 0.0
        }
    
    def get_sentence_transformer(self, model_name: str, 
                                enable_quantization: bool = True,
                                device: str = "cpu") -> Optional[SentenceTransformer]:
        """Sentence-BERT 모델 가져오기 (캐싱 적용)"""
        
        # 캐시 확인
        cache_key = f"{model_name}_{device}_{enable_quantization}"
        if cache_key in self._models:
            self.stats['cache_hits'] += 1
            return self._models[cache_key]
        
        # 로딩 중인지 확인
        if cache_key in self._loading_flags and self._loading_flags[cache_key]:
            # 다른 스레드가 로딩 중이면 대기
            if cache_key not in self._model_locks:
                self._model_locks[cache_key] = threading.Lock()
            
            with self._model_locks[cache_key]:
                if cache_key in self._models:
                    return self._models[cache_key]
        
        # 모델 로딩
        return self._load_sentence_transformer(model_name, enable_quantization, device, cache_key)
    
    def _load_sentence_transformer(self, model_name: str, 
                                 enable_quantization: bool, 
                                 device: str, 
                                 cache_key: str) -> Optional[SentenceTransformer]:
        """Sentence-BERT 모델 로딩"""
        
        if not SENTENCE_TRANSFORMERS_AVAILABLE:
            self.logger.error("SentenceTransformers not available")
            return None
        
        # 로딩 플래그 설정
        self._loading_flags[cache_key] = True
        
        try:
            start_time = time.time()
            
            # 모델 로딩
            model = SentenceTransformer(model_name, device=device)
            
            # 양자화 적용
            if enable_quantization and TORCH_AVAILABLE:
                try:
                    if hasattr(model, 'model') and hasattr(model.model, 'half'):
                        model.model = model.model.half()
                        self.logger.info(f"Model {model_name} quantized to Float16")
                except Exception as e:
                    self.logger.warning(f"Quantization failed for {model_name}: {e}")
            
            # 캐시에 저장
            self._models[cache_key] = model
            
            # 통계 업데이트
            load_time = time.time() - start_time
            self.stats['model_loads'] += 1
            self.stats['total_load_time'] += load_time
            
            self.logger.info(f"Model {model_name} loaded in {load_time:.2f}s")
            return model
            
        except Exception as e:
            self.logger.error(f"Failed to load model {model_name}: {e}")
            return None
        finally:
            self._loading_flags[cache_key] = False
    
    @lru_cache(maxsize=128)
    def get_cached_model(self, model_name: str, device: str = "cpu") -> Optional[SentenceTransformer]:
        """LRU 캐시를 사용한 모델 가져오기"""
        return self.get_sentence_transformer(model_name, device=device)
    
    def preload_models(self, model_configs: Dict[str, Dict[str, Any]]):
        """모델들을 미리 로딩"""
        for model_name, config in model_configs.items():
            device = config.get('device', 'cpu')
            enable_quantization = config.get('enable_quantization', True)
            
            # 백그라운드에서 로딩
            threading.Thread(
                target=self.get_sentence_transformer,
                args=(model_name, enable_quantization, device),
                daemon=True
            ).start()
    
    def get_stats(self) -> Dict[str, Any]:
        """성능 통계 반환"""
        avg_load_time = (
            self.stats['total_load_time'] / self.stats['model_loads'] 
            if self.stats['model_loads'] > 0 else 0
        )
        
        return {
            **self.stats,
            'avg_load_time': avg_load_time,
            'cached_models': len(self._models),
            'cache_hit_rate': (
                self.stats['cache_hits'] / (self.stats['cache_hits'] + self.stats['model_loads'])
                if (self.stats['cache_hits'] + self.stats['model_loads']) > 0 else 0
            )
        }
    
    def clear_cache(self):
        """모델 캐시 정리"""
        self._models.clear()
        self._loading_flags.clear()
        self.stats = {
            'model_loads': 0,
            'cache_hits': 0,
            'total_load_time': 0.0
        }

# 전역 인스턴스
model_manager = OptimizedModelManager()
