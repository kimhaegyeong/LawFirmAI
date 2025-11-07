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
        
        # 모델 로딩 전에 HuggingFace 로깅 비활성화
        import os
        import warnings
        
        # 환경 변수 설정
        original_verbosity = os.environ.get('TRANSFORMERS_VERBOSITY', None)
        original_progress_bars = os.environ.get('HF_HUB_DISABLE_PROGRESS_BARS', None)
        
        os.environ['TRANSFORMERS_VERBOSITY'] = 'error'
        os.environ['HF_HUB_DISABLE_PROGRESS_BARS'] = '1'
        os.environ['HF_HUB_DISABLE_EXPERIMENTAL_WARNING'] = '1'
        
        # 로깅 레벨 임시 변경
        transformers_logger = logging.getLogger('transformers')
        sentence_transformers_logger = logging.getLogger('sentence_transformers')
        hf_hub_logger = logging.getLogger('huggingface_hub')
        
        original_levels = {
            'transformers': transformers_logger.level,
            'sentence_transformers': sentence_transformers_logger.level,
            'huggingface_hub': hf_hub_logger.level,
        }
        
        # 로깅 레벨을 ERROR로 설정
        transformers_logger.setLevel(logging.ERROR)
        sentence_transformers_logger.setLevel(logging.ERROR)
        hf_hub_logger.setLevel(logging.ERROR)
        
        # 로딩 플래그 설정
        self._loading_flags[cache_key] = True
        
        try:
            start_time = time.time()
            
            # meta tensor 오류 방지를 위한 환경 변수 설정
            import os
            original_env = {}
            try:
                # meta device 사용 방지
                original_env['HF_HUB_DISABLE_EXPERIMENTAL_WARNING'] = os.environ.get('HF_HUB_DISABLE_EXPERIMENTAL_WARNING', None)
                os.environ['HF_HUB_DISABLE_EXPERIMENTAL_WARNING'] = '1'
                
                # device_map 사용 방지
                original_env['HF_DEVICE_MAP'] = os.environ.get('HF_DEVICE_MAP', None)
                if 'HF_DEVICE_MAP' in os.environ:
                    del os.environ['HF_DEVICE_MAP']
            except Exception:
                pass
            
            # 경고 메시지 비활성화하고 모델 로딩
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                # 모델 로딩 (로깅이 비활성화된 상태)
                # meta tensor 오류 방지를 위한 추가 옵션 설정
                try:
                    # torch dtype 설정 (torch가 사용 가능한 경우)
                    torch_dtype = None
                    if TORCH_AVAILABLE:
                        torch_dtype = torch.float32
                    
                    # 방법 1: CPU에 먼저 로드 (가장 안전한 방법)
                    if device != "cpu":
                        # GPU가 있어도 안정성을 위해 CPU에 먼저 로드
                        self.logger.debug(f"Loading SentenceTransformer model {model_name} on CPU first (to avoid meta tensor errors)...")
                        model_kwargs = {
                            "low_cpu_mem_usage": False,  # meta device 사용 방지
                            "device_map": None,  # device_map 사용 안 함
                        }
                        if torch_dtype is not None:
                            model_kwargs["torch_dtype"] = torch_dtype
                        
                        model = SentenceTransformer(
                            model_name, 
                            device="cpu",
                            model_kwargs=model_kwargs
                        )
                        # CPU에서 GPU로 이동 (필요한 경우)
                        if device == "cuda" and TORCH_AVAILABLE and torch.cuda.is_available():
                            try:
                                # 모델을 GPU로 이동
                                model = model.to(device)
                                self.logger.info(f"Model moved from CPU to {device}")
                            except Exception as move_error:
                                # GPU 이동 실패 시 CPU에 유지
                                self.logger.warning(f"Failed to move model to {device}, keeping on CPU: {move_error}")
                                device = "cpu"
                    else:
                        # CPU 사용
                        model_kwargs = {
                            "low_cpu_mem_usage": False,  # meta device 사용 방지
                            "device_map": None,  # device_map 사용 안 함
                        }
                        if torch_dtype is not None:
                            model_kwargs["torch_dtype"] = torch_dtype
                        
                        model = SentenceTransformer(
                            model_name, 
                            device="cpu",
                            model_kwargs=model_kwargs
                        )
                except Exception as load_error:
                    # meta tensor 오류 발생 시 대체 방법 시도
                    if "meta tensor" in str(load_error).lower() or "to_empty" in str(load_error).lower():
                        self.logger.warning(f"Meta tensor error detected, trying alternative loading method: {load_error}")
                        # 대체 방법: 더 명시적인 옵션으로 로드
                        model_kwargs = {
                            "low_cpu_mem_usage": False,
                            "device_map": None,
                            "trust_remote_code": True,  # 원격 코드 신뢰
                        }
                        if TORCH_AVAILABLE:
                            model_kwargs["torch_dtype"] = torch.float32
                        
                        model = SentenceTransformer(
                            model_name, 
                            device="cpu",  # 항상 CPU에 먼저 로드
                            model_kwargs=model_kwargs
                        )
                        # CPU에서 GPU로 이동 (필요한 경우)
                        if device == "cuda" and TORCH_AVAILABLE and torch.cuda.is_available():
                            try:
                                model = model.to(device)
                            except Exception:
                                # GPU 이동 실패 시 CPU에 유지
                                device = "cpu"
                    else:
                        # 다른 오류는 그대로 전파
                        raise
            
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
            
            # 환경 변수 복원
            if original_verbosity is not None:
                os.environ['TRANSFORMERS_VERBOSITY'] = original_verbosity
            else:
                os.environ.pop('TRANSFORMERS_VERBOSITY', None)
            
            if original_progress_bars is not None:
                os.environ['HF_HUB_DISABLE_PROGRESS_BARS'] = original_progress_bars
            else:
                os.environ.pop('HF_HUB_DISABLE_PROGRESS_BARS', None)
            
            # 로깅 레벨 복원
            transformers_logger.setLevel(original_levels['transformers'])
            sentence_transformers_logger.setLevel(original_levels['sentence_transformers'])
            hf_hub_logger.setLevel(original_levels['huggingface_hub'])
    
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
