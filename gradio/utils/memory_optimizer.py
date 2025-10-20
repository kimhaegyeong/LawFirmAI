# -*- coding: utf-8 -*-
"""
메모리 최적화 유틸리티
HuggingFace Spaces 환경에서 메모리 사용량을 최적화합니다.
"""

import gc
import torch
import psutil
import logging
from contextlib import contextmanager
from typing import Optional, Dict, Any
from datetime import datetime

logger = logging.getLogger(__name__)

class MemoryOptimizer:
    """메모리 최적화 클래스"""
    
    def __init__(self, max_memory_percent: float = 85.0):
        """
        메모리 최적화기 초기화
        
        Args:
            max_memory_percent: 최대 메모리 사용률 (기본값: 85%)
        """
        self.max_memory_percent = max_memory_percent
        self.logger = logging.getLogger(__name__)
        self.memory_history = []
        self.optimization_count = 0
        
    @contextmanager
    def memory_efficient_inference(self):
        """메모리 효율적인 추론 컨텍스트"""
        # 추론 전 메모리 정리
        initial_memory = self._cleanup_memory()
        
        try:
            yield
        finally:
            # 추론 후 메모리 정리
            final_memory = self._cleanup_memory()
            
            # 메모리 사용량 로깅
            if initial_memory and final_memory:
                memory_diff = final_memory - initial_memory
                self.logger.debug(f"Memory usage change: {memory_diff:.1f}MB")
    
    def _cleanup_memory(self) -> Optional[float]:
        """메모리 정리 및 사용량 반환"""
        # 가비지 컬렉션 실행
        collected = gc.collect()
        
        # CUDA 캐시 정리 (GPU 사용 시)
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        # 메모리 사용량 측정
        memory_usage = self.get_memory_usage_mb()
        
        if collected > 0:
            self.logger.debug(f"Garbage collected {collected} objects, memory: {memory_usage:.1f}MB")
        
        return memory_usage
    
    def get_memory_usage_mb(self) -> float:
        """현재 메모리 사용량을 MB 단위로 반환"""
        process = psutil.Process()
        memory_info = process.memory_info()
        return memory_info.rss / 1024 / 1024  # MB 단위
    
    def get_memory_usage_percent(self) -> float:
        """현재 메모리 사용률을 퍼센트로 반환"""
        return psutil.virtual_memory().percent
    
    def monitor_memory_usage(self) -> Dict[str, Any]:
        """메모리 사용량 모니터링"""
        memory_percent = self.get_memory_usage_percent()
        memory_mb = self.get_memory_usage_mb()
        
        # 메모리 사용량 기록
        self.memory_history.append({
            "timestamp": datetime.now().isoformat(),
            "percent": memory_percent,
            "mb": memory_mb
        })
        
        # 최근 100개 기록만 유지
        if len(self.memory_history) > 100:
            self.memory_history = self.memory_history[-100:]
        
        # 메모리 사용률이 임계값을 초과하면 정리
        if memory_percent > self.max_memory_percent:
            self.logger.warning(f"High memory usage: {memory_percent:.1f}%")
            self._force_cleanup()
            self.optimization_count += 1
        
        return {
            "percent": memory_percent,
            "mb": memory_mb,
            "optimization_count": self.optimization_count,
            "status": "high" if memory_percent > self.max_memory_percent else "normal"
        }
    
    def _force_cleanup(self):
        """강제 메모리 정리"""
        self.logger.info("Performing forced memory cleanup...")
        
        # 여러 번 가비지 컬렉션 실행
        for _ in range(3):
            collected = gc.collect()
            if collected == 0:
                break
        
        # CUDA 캐시 정리
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        self.logger.info("Forced memory cleanup completed")
    
    def get_memory_stats(self) -> Dict[str, Any]:
        """메모리 통계 정보 반환"""
        current_memory = self.monitor_memory_usage()
        
        # 메모리 사용량 통계
        if self.memory_history:
            memory_percents = [record["percent"] for record in self.memory_history]
            memory_mbs = [record["mb"] for record in self.memory_history]
            
            stats = {
                "current": current_memory,
                "average_percent": sum(memory_percents) / len(memory_percents),
                "max_percent": max(memory_percents),
                "min_percent": min(memory_percents),
                "average_mb": sum(memory_mbs) / len(memory_mbs),
                "max_mb": max(memory_mbs),
                "min_mb": min(memory_mbs),
                "optimization_count": self.optimization_count,
                "history_length": len(self.memory_history)
            }
        else:
            stats = {
                "current": current_memory,
                "optimization_count": self.optimization_count,
                "history_length": 0
            }
        
        return stats

class ModelMemoryManager:
    """모델 메모리 관리 클래스"""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.loaded_models = {}
        self.model_memory_usage = {}
    
    def load_model_efficiently(self, model_name: str, model_loader_func):
        """메모리 효율적으로 모델 로딩"""
        if model_name in self.loaded_models:
            self.logger.info(f"Model {model_name} already loaded")
            return self.loaded_models[model_name]
        
        # 메모리 사용량 측정
        memory_before = psutil.Process().memory_info().rss / 1024 / 1024
        
        try:
            # 모델 로딩
            model = model_loader_func()
            self.loaded_models[model_name] = model
            
            # 메모리 사용량 측정
            memory_after = psutil.Process().memory_info().rss / 1024 / 1024
            memory_used = memory_after - memory_before
            
            self.model_memory_usage[model_name] = memory_used
            self.logger.info(f"Model {model_name} loaded, memory used: {memory_used:.1f}MB")
            
            return model
            
        except Exception as e:
            self.logger.error(f"Failed to load model {model_name}: {e}")
            raise
    
    def unload_model(self, model_name: str):
        """모델 언로딩"""
        if model_name in self.loaded_models:
            del self.loaded_models[model_name]
            if model_name in self.model_memory_usage:
                del self.model_memory_usage[model_name]
            
            # 메모리 정리
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            
            self.logger.info(f"Model {model_name} unloaded")
    
    def get_model_stats(self) -> Dict[str, Any]:
        """모델 통계 정보 반환"""
        total_memory = sum(self.model_memory_usage.values())
        
        return {
            "loaded_models": list(self.loaded_models.keys()),
            "model_memory_usage": self.model_memory_usage,
            "total_model_memory": total_memory,
            "model_count": len(self.loaded_models)
        }

# 전역 메모리 최적화기 인스턴스
memory_optimizer = MemoryOptimizer()
model_memory_manager = ModelMemoryManager()

def get_memory_optimizer() -> MemoryOptimizer:
    """메모리 최적화기 인스턴스 반환"""
    return memory_optimizer

def get_model_memory_manager() -> ModelMemoryManager:
    """모델 메모리 관리자 인스턴스 반환"""
    return model_memory_manager
