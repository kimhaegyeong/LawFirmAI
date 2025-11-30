# -*- coding: utf-8 -*-
"""
모델 캐시 매니저 (싱글톤 패턴)
동일한 모델을 중복 로드하지 않도록 관리
"""

import threading
import logging
import os
import re
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

# Transformers 모델 지원
TRANSFORMERS_AVAILABLE = False
try:
    from transformers import AutoTokenizer, AutoModelForSequenceClassification
    import torch
    TRANSFORMERS_AVAILABLE = True
except ImportError:
    TRANSFORMERS_AVAILABLE = False

logger = get_logger(__name__)


def _normalize_model_name(model_name: str) -> str:
    """
    모델 이름 정규화 (따옴표 제거 및 공백 제거)
    
    Args:
        model_name: 원본 모델 이름 (예: '"woong0322/ko-legal-sbert-finetuned"')
        
    Returns:
        정규화된 모델 이름 (예: "woong0322/ko-legal-sbert-finetuned")
    """
    # 앞뒤 따옴표 및 공백 제거
    normalized = model_name.strip().strip('"').strip("'")
    return normalized


def _normalize_model_name_for_cache(model_name: str) -> str:
    """
    모델 이름을 캐시 디렉토리 이름으로 사용하기 위해 정규화
    
    Args:
        model_name: 원본 모델 이름 (예: "woong0322/ko-legal-sbert-finetuned")
        
    Returns:
        정규화된 모델 이름 (예: "woong0322_ko_legal_sbert_finetuned")
    """
    # 먼저 모델 이름 정규화 (따옴표 제거)
    normalized = _normalize_model_name(model_name)
    # 슬래시와 하이픈을 언더스코어로 변경
    normalized = re.sub(r'[/-]', '_', normalized)
    # 특수문자 제거 (알파벳, 숫자, 언더스코어만 허용)
    normalized = re.sub(r'[^a-zA-Z0-9_]', '', normalized)
    return normalized


def _filter_model_kwargs(model_kwargs: Dict[str, Any]) -> Dict[str, Any]:
    """
    SentenceTransformer가 지원하는 model_kwargs만 필터링
    
    Args:
        model_kwargs: 원본 model_kwargs
        
    Returns:
        필터링된 model_kwargs
    """
    # SentenceTransformer가 직접 지원하지 않는 파라미터 제거
    # 이 파라미터들은 내부 transformers 모델에만 전달되지만,
    # SentenceTransformer는 이를 직접 파라미터로 받지 않음
    unsupported_params = {
        'low_cpu_mem_usage',  # SentenceTransformer가 직접 지원하지 않음
        'device_map',  # SentenceTransformer는 device 파라미터 사용
        'dtype',  # SentenceTransformer가 직접 지원하지 않음 (일부 버전에서 에러 발생)
        'torch_dtype',  # SentenceTransformer가 직접 지원하지 않음
        'use_safetensors',  # SentenceTransformer가 직접 지원하지 않음 (일부 버전에서 에러 발생)
        'trust_remote_code',  # SentenceTransformer가 직접 지원하지 않음 (일부 버전에서 에러 발생)
        'local_files_only',  # SentenceTransformer가 직접 지원하지 않음 (일부 버전에서 에러 발생)
    }
    
    filtered_kwargs = {
        k: v for k, v in model_kwargs.items()
        if k not in unsupported_params
    }
    
    if filtered_kwargs != model_kwargs:
        removed = set(model_kwargs.keys()) - set(filtered_kwargs.keys())
        logger.debug(f"🔧 [MODEL CACHE] Filtered unsupported params: {removed}")
    
    return filtered_kwargs


def _get_cache_folder(model_name: str, base_cache_dir: Optional[str] = None) -> str:
    """
    모델별 캐시 디렉토리 경로 생성
    
    Args:
        model_name: 모델 이름
        base_cache_dir: 기본 캐시 디렉토리 (None이면 기본값 사용)
        
    Returns:
        캐시 디렉토리 경로
    """
    if base_cache_dir is None:
        # 기본 캐시 디렉토리: ~/.cache/huggingface/transformers 또는 ./model_cache
        default_cache = os.getenv("HF_HOME", os.path.expanduser("~/.cache/huggingface"))
        base_cache_dir = os.getenv("MODEL_CACHE_DIR", os.path.join(default_cache, "sentence_transformers"))
    
    # 모델 이름 정규화
    normalized_name = _normalize_model_name_for_cache(model_name)
    cache_folder = os.path.join(base_cache_dir, normalized_name)
    
    # 디렉토리 생성
    os.makedirs(cache_folder, exist_ok=True)
    
    return cache_folder


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
        self._transformers_models: Dict[str, Dict[str, Any]] = {}  # {model_name: {"model": model, "tokenizer": tokenizer}}
        self._model_locks: Dict[str, threading.Lock] = {}
        self._cache_lock = threading.Lock()
        self._initialized = True
        logger.debug("✅ [MODEL CACHE] ModelCacheManager initialized")
    
    def get_model(
        self,
        model_name: str,
        fallback_model_name: Optional[str] = None,
        device: Optional[str] = None,
        model_kwargs: Optional[Dict[str, Any]] = None,
        cache_folder: Optional[str] = None
    ) -> Optional[Any]:
        """
        모델 가져오기 (캐시에서 로드 또는 새로 로드)
        
        Args:
            model_name: 모델명
            fallback_model_name: 폴백 모델명 (기본값: paraphrase-multilingual-MiniLM-L12-v2)
            device: 디바이스 ("cpu", "cuda" 등, None이면 기본값 사용)
            model_kwargs: 모델 로딩 시 추가 옵션 (dict)
            cache_folder: 명시적 캐시 폴더 경로 (None이면 자동 생성)
            
        Returns:
            SentenceTransformer 모델 인스턴스 또는 None
        """
        if not SENTENCE_TRANSFORMERS_AVAILABLE:
            logger.warning("⚠️ [MODEL CACHE] SentenceTransformers not available")
            return None
        
        # 모델 이름 정규화 (따옴표 제거)
        normalized_model_name = _normalize_model_name(model_name)
        
        # 캐시 확인 (정규화된 이름 사용)
        with self._cache_lock:
            if normalized_model_name in self._models:
                logger.trace(f"✅ [MODEL CACHE] Cache hit for model: {normalized_model_name}")
                return self._models[normalized_model_name]
        
        # 모델별 락 생성 (없으면) - 정규화된 이름 사용
        if normalized_model_name not in self._model_locks:
            with self._cache_lock:
                if normalized_model_name not in self._model_locks:
                    self._model_locks[normalized_model_name] = threading.Lock()
        
        # 모델 로드 (동시 로드 방지)
        with self._model_locks[normalized_model_name]:
            # 다시 한 번 확인 (다른 스레드가 로드했을 수 있음)
            if normalized_model_name in self._models:
                logger.trace(f"✅ [MODEL CACHE] Cache hit (after lock) for model: {normalized_model_name}")
                return self._models[normalized_model_name]
            
            try:
                logger.info(f"🔄 [MODEL CACHE] Loading model: {normalized_model_name} (original: {model_name})")
                
                # 캐시 폴더 설정
                if cache_folder is None:
                    cache_folder = _get_cache_folder(normalized_model_name)
                
                # model_kwargs 필터링 및 cache_folder 추가
                if model_kwargs is None:
                    model_kwargs = {}
                else:
                    # SentenceTransformer가 지원하지 않는 파라미터 필터링
                    model_kwargs = _filter_model_kwargs(model_kwargs)
                
                # cache_folder가 이미 설정되지 않은 경우에만 추가
                if 'cache_folder' not in model_kwargs:
                    model_kwargs = {**model_kwargs, 'cache_folder': cache_folder}
                
                logger.debug(f"📁 [MODEL CACHE] Using cache folder: {cache_folder}")
                
                # device와 model_kwargs를 사용하여 모델 로드 (정규화된 이름 사용)
                if device:
                    model = SentenceTransformer(normalized_model_name, device=device, **model_kwargs)
                else:
                    model = SentenceTransformer(normalized_model_name, **model_kwargs)
                
                with self._cache_lock:
                    self._models[normalized_model_name] = model
                
                logger.info(f"✅ [MODEL CACHE] Successfully loaded and cached model: {normalized_model_name} (original: {model_name})")
                return model
                
            except Exception as e:
                logger.warning(f"⚠️ [MODEL CACHE] Failed to load model '{normalized_model_name}' (original: '{model_name}'): {e}")
                
                # 폴백 모델 시도
                fallback_raw = fallback_model_name or "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
                fallback = _normalize_model_name(fallback_raw)
                
                if fallback in self._models:
                    logger.info(f"✅ [MODEL CACHE] Using cached fallback model: {fallback} (original: {fallback_raw})")
                    return self._models[fallback]
                
                # 폴백 모델 로딩 시도 (최대 2회: 필터링된 파라미터, 최소 파라미터)
                fallback_cache_folder = cache_folder or _get_cache_folder(fallback)
                
                # 첫 번째 시도: 필터링된 파라미터 사용
                fallback_model_kwargs = {**model_kwargs} if model_kwargs else {}
                fallback_model_kwargs = _filter_model_kwargs(fallback_model_kwargs)
                if 'cache_folder' not in fallback_model_kwargs:
                    fallback_model_kwargs['cache_folder'] = fallback_cache_folder
                
                try:
                    logger.info(f"🔄 [MODEL CACHE] Loading fallback model: {fallback} (original: {fallback_raw})")
                    
                    # 폴백 모델도 동일한 옵션으로 로드 시도
                    if device:
                        fallback_model = SentenceTransformer(fallback, device=device, **fallback_model_kwargs)
                    else:
                        fallback_model = SentenceTransformer(fallback, **fallback_model_kwargs)
                    
                    with self._cache_lock:
                        self._models[fallback] = fallback_model
                    
                    logger.info(f"✅ [MODEL CACHE] Successfully loaded and cached fallback model: {fallback} (original: {fallback_raw})")
                    return fallback_model
                    
                except Exception as e2:
                    logger.warning(f"⚠️ [MODEL CACHE] Failed to load fallback model '{fallback}' (original: '{fallback_raw}') with filtered kwargs: {e2}")
                    
                    # 두 번째 시도: 최소 파라미터만 사용 (cache_folder만)
                    try:
                        logger.info(f"🔄 [MODEL CACHE] Retrying fallback model with minimal kwargs: {fallback}")
                        minimal_kwargs = {'cache_folder': fallback_cache_folder}
                        
                        if device:
                            fallback_model = SentenceTransformer(fallback, device=device, **minimal_kwargs)
                        else:
                            fallback_model = SentenceTransformer(fallback, **minimal_kwargs)
                        
                        with self._cache_lock:
                            self._models[fallback] = fallback_model
                        
                        logger.info(f"✅ [MODEL CACHE] Successfully loaded fallback model with minimal kwargs: {fallback} (original: {fallback_raw})")
                        return fallback_model
                        
                    except Exception as e3:
                        logger.error(f"❌ [MODEL CACHE] Failed to load fallback model '{fallback}' (original: '{fallback_raw}') even with minimal kwargs: {e3}")
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
    
    def get_transformers_model(
        self,
        model_name: str,
        model_type: str = "AutoModelForSequenceClassification",
        device: Optional[str] = None,
        cache_dir: Optional[str] = None
    ) -> Optional[Dict[str, Any]]:
        """
        Transformers 모델 가져오기 (캐시에서 로드 또는 새로 로드)
        
        Args:
            model_name: 모델명 (예: "monologg/kobert")
            model_type: 모델 타입 ("AutoModelForSequenceClassification", "AutoModel" 등)
            device: 디바이스 ("cpu", "cuda" 등, None이면 자동 감지)
            cache_dir: 캐시 디렉토리 (None이면 기본값 사용)
            
        Returns:
            {"model": model, "tokenizer": tokenizer} 또는 None
        """
        if not TRANSFORMERS_AVAILABLE:
            logger.warning("⚠️ [MODEL CACHE] Transformers not available")
            return None
        
        # 모델 이름 정규화
        normalized_model_name = _normalize_model_name(model_name)
        
        # 캐시 확인
        with self._cache_lock:
            if normalized_model_name in self._transformers_models:
                logger.trace(f"✅ [MODEL CACHE] Cache hit for transformers model: {normalized_model_name}")
                return self._transformers_models[normalized_model_name]
        
        # 모델별 락 생성
        if normalized_model_name not in self._model_locks:
            with self._cache_lock:
                if normalized_model_name not in self._model_locks:
                    self._model_locks[normalized_model_name] = threading.Lock()
        
        # 모델 로드 (동시 로드 방지)
        with self._model_locks[normalized_model_name]:
            # 다시 한 번 확인
            if normalized_model_name in self._transformers_models:
                logger.trace(f"✅ [MODEL CACHE] Cache hit (after lock) for transformers model: {normalized_model_name}")
                return self._transformers_models[normalized_model_name]
            
            try:
                logger.info(f"🔄 [MODEL CACHE] Loading transformers model: {normalized_model_name} (type: {model_type})")
                
                # 캐시 디렉토리 설정
                if cache_dir is None:
                    cache_dir = os.getenv("HF_HOME", os.path.expanduser("~/.cache/huggingface"))
                
                # Tokenizer 로드
                tokenizer = AutoTokenizer.from_pretrained(
                    normalized_model_name,
                    cache_dir=cache_dir,
                    trust_remote_code=True  # KoBERT 등 커스텀 코드 모델 지원
                )
                
                # Model 로드
                if model_type == "AutoModelForSequenceClassification":
                    model = AutoModelForSequenceClassification.from_pretrained(
                        normalized_model_name,
                        cache_dir=cache_dir,
                        trust_remote_code=True  # KoBERT 등 커스텀 코드 모델 지원
                    )
                else:
                    from transformers import AutoModel
                    model = AutoModel.from_pretrained(
                        normalized_model_name,
                        cache_dir=cache_dir,
                        trust_remote_code=True  # KoBERT 등 커스텀 코드 모델 지원
                    )
                
                # 평가 모드로 설정
                model.eval()
                
                # 디바이스 설정
                if device is None:
                    device = "cuda" if torch.cuda.is_available() else "cpu"
                
                if device == "cuda" and torch.cuda.is_available():
                    model = model.cuda()
                    logger.debug(f"📱 [MODEL CACHE] Model loaded on GPU: {normalized_model_name}")
                else:
                    logger.debug(f"📱 [MODEL CACHE] Model loaded on CPU: {normalized_model_name}")
                
                # 캐시에 저장
                model_dict = {
                    "model": model,
                    "tokenizer": tokenizer,
                    "device": device
                }
                
                with self._cache_lock:
                    self._transformers_models[normalized_model_name] = model_dict
                
                logger.info(f"✅ [MODEL CACHE] Successfully loaded and cached transformers model: {normalized_model_name}")
                return model_dict
                
            except Exception as e:
                logger.warning(f"⚠️ [MODEL CACHE] Failed to load transformers model '{normalized_model_name}': {e}")
                return None
    
    def clear_transformers_cache(self, model_name: Optional[str] = None):
        """
        Transformers 모델 캐시 삭제
        
        Args:
            model_name: 삭제할 모델명 (None이면 전체 삭제)
        """
        with self._cache_lock:
            if model_name:
                normalized_name = _normalize_model_name(model_name)
                if normalized_name in self._transformers_models:
                    # GPU 메모리 정리
                    model_dict = self._transformers_models[normalized_name]
                    if "model" in model_dict:
                        del model_dict["model"]
                    if "tokenizer" in model_dict:
                        del model_dict["tokenizer"]
                    del self._transformers_models[normalized_name]
                    logger.info(f"🗑️ [MODEL CACHE] Cleared transformers cache for model: {normalized_name}")
            else:
                # 전체 삭제
                for normalized_name, model_dict in self._transformers_models.items():
                    if "model" in model_dict:
                        del model_dict["model"]
                    if "tokenizer" in model_dict:
                        del model_dict["tokenizer"]
                self._transformers_models.clear()
                logger.info("🗑️ [MODEL CACHE] Cleared all transformers model cache")
    
    def get_cached_transformers_models(self) -> list:
        """캐시된 Transformers 모델 목록 반환"""
        with self._cache_lock:
            return list(self._transformers_models.keys())


def get_model_cache_manager() -> ModelCacheManager:
    """ModelCacheManager 싱글톤 인스턴스 반환"""
    return ModelCacheManager()

