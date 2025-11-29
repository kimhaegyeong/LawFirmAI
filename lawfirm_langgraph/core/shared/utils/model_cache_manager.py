# -*- coding: utf-8 -*-
"""
모델 캐시 매니저 (싱글톤 패턴)
동일한 모델을 중복 로드하지 않도록 관리
"""

import threading
import logging
from typing import Optional, Dict, Any

try:
    from lawfirm_langgraph.core.utils.logger import get_logger
except ImportError:
    from core.utils.logger import get_logger

try:
    from sentence_transformers import SentenceTransformer
    SENTENCE_TRANSFORMERS_AVAILABLE = True
except ImportError:
    SENTENCE_TRANSFORMERS_AVAILABLE = False

logger = get_logger(__name__)


class ModelCacheManager:
    """모델 캐시 매니저 (싱글톤 패턴)"""
    
    _instance: Optional['ModelCacheManager'] = None
    _lock = threading.Lock()
    
    def __new__(cls):
        """싱글톤 패턴 구현"""
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
                    cls._instance._initialized = False
        return cls._instance
    
    def __init__(self):
        """초기화 (한 번만 실행)"""
        if self._initialized:
            return
        
        self._models: Dict[str, Any] = {}
        self._model_locks: Dict[str, threading.Lock] = {}
        self._cache_lock = threading.Lock()
        self._initialized = True
        logger.debug("✅ [MODEL CACHE] ModelCacheManager initialized")
    
    def get_model(
        self,
        model_name: str,
        fallback_model_name: Optional[str] = None,
        device: Optional[str] = None,
        model_kwargs: Optional[Dict[str, Any]] = None
    ) -> Optional[Any]:
        """
        모델 가져오기 (캐시에서 로드 또는 새로 로드)
        
        Args:
            model_name: 모델명
            fallback_model_name: 폴백 모델명 (기본값: paraphrase-multilingual-MiniLM-L12-v2)
            device: 디바이스 ("cpu", "cuda" 등, None이면 기본값 사용)
            model_kwargs: 모델 로딩 시 추가 옵션 (dict)
            
        Returns:
            SentenceTransformer 모델 인스턴스 또는 None
        """
        if not SENTENCE_TRANSFORMERS_AVAILABLE:
            logger.warning("⚠️ [MODEL CACHE] SentenceTransformers not available")
            return None
        
        # 캐시 확인
        with self._cache_lock:
            if model_name in self._models:
                logger.trace(f"✅ [MODEL CACHE] Cache hit for model: {model_name}")
                return self._models[model_name]
        
        # 모델별 락 생성 (없으면)
        if model_name not in self._model_locks:
            with self._cache_lock:
                if model_name not in self._model_locks:
                    self._model_locks[model_name] = threading.Lock()
        
        # 모델 로드 (동시 로드 방지)
        with self._model_locks[model_name]:
            # 다시 한 번 확인 (다른 스레드가 로드했을 수 있음)
            if model_name in self._models:
                logger.trace(f"✅ [MODEL CACHE] Cache hit (after lock) for model: {model_name}")
                return self._models[model_name]
            
            try:
                logger.info(f"🔄 [MODEL CACHE] Loading model: {model_name}")
                
                # device와 model_kwargs를 사용하여 모델 로드
                if device and model_kwargs:
                    model = SentenceTransformer(model_name, device=device, model_kwargs=model_kwargs)
                elif device:
                    model = SentenceTransformer(model_name, device=device)
                elif model_kwargs:
                    model = SentenceTransformer(model_name, model_kwargs=model_kwargs)
                else:
                    model = SentenceTransformer(model_name)
                
                with self._cache_lock:
                    self._models[model_name] = model
                
                logger.info(f"✅ [MODEL CACHE] Successfully loaded and cached model: {model_name}")
                return model
                
            except Exception as e:
                logger.warning(f"⚠️ [MODEL CACHE] Failed to load {model_name}: {e}")
                
                # 폴백 모델 시도
                fallback = fallback_model_name or "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
                
                if fallback in self._models:
                    logger.info(f"✅ [MODEL CACHE] Using cached fallback model: {fallback}")
                    return self._models[fallback]
                
                try:
                    logger.info(f"🔄 [MODEL CACHE] Loading fallback model: {fallback}")
                    
                    # 폴백 모델도 동일한 옵션으로 로드 시도
                    if device and model_kwargs:
                        fallback_model = SentenceTransformer(fallback, device=device, model_kwargs=model_kwargs)
                    elif device:
                        fallback_model = SentenceTransformer(fallback, device=device)
                    elif model_kwargs:
                        fallback_model = SentenceTransformer(fallback, model_kwargs=model_kwargs)
                    else:
                        fallback_model = SentenceTransformer(fallback)
                    
                    with self._cache_lock:
                        self._models[fallback] = fallback_model
                    
                    logger.info(f"✅ [MODEL CACHE] Successfully loaded and cached fallback model: {fallback}")
                    return fallback_model
                    
                except Exception as e2:
                    logger.error(f"❌ [MODEL CACHE] Failed to load fallback model {fallback}: {e2}")
                    return None
    
    def clear_cache(self, model_name: Optional[str] = None):
        """
        캐시 삭제
        
        Args:
            model_name: 삭제할 모델명 (None이면 전체 삭제)
        """
        with self._cache_lock:
            if model_name:
                if model_name in self._models:
                    del self._models[model_name]
                    logger.info(f"🗑️ [MODEL CACHE] Cleared cache for model: {model_name}")
            else:
                self._models.clear()
                logger.info("🗑️ [MODEL CACHE] Cleared all model cache")
    
    def get_cached_models(self) -> list:
        """캐시된 모델 목록 반환"""
        with self._cache_lock:
            return list(self._models.keys())


def get_model_cache_manager() -> ModelCacheManager:
    """ModelCacheManager 싱글톤 인스턴스 반환"""
    return ModelCacheManager()

