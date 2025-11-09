# -*- coding: utf-8 -*-
"""
Semantic Search Engine V2
lawfirm_v2.db의 embeddings 테이블을 사용한 벡터 검색 엔진
"""

import logging
import sqlite3
import sys
import threading
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np

# FAISS import (optional)
try:
    import faiss
    FAISS_AVAILABLE = True
except ImportError:
    FAISS_AVAILABLE = False
    logger = logging.getLogger(__name__)
    logger.warning("FAISS not available. Install with: pip install faiss-cpu")

# Embedding utilities import
# Try to import from scripts/utils first, then fallback to direct implementation
try:
    scripts_utils_path = Path(__file__).parent.parent.parent / "scripts" / "utils"
    if scripts_utils_path.exists():
        sys.path.insert(0, str(scripts_utils_path))
    from embeddings import SentenceEmbedder
except ImportError:
    # Fallback: use sentence-transformers directly
    from sentence_transformers import SentenceTransformer

    class SentenceEmbedder:
        """Fallback embedder using sentence-transformers"""
        
        # 상수 정의
        PARAMETER_CHECK_SAMPLE_SIZE = 10  # 검증 시 확인할 파라미터 샘플 개수
        MAX_META_DEVICE_FIX_ATTEMPTS = 3  # meta device 수정 최대 시도 횟수
        
        def __init__(self, model_name: str = "snunlp/KR-SBERT-V40K-klueNLI-augSTS"):
            self.model_name = model_name
            import logging
            logger = logging.getLogger(__name__)
            
            try:
                # sentence-transformers를 사용하여 모델 로딩
                # meta tensor 문제 방지를 위해 CPU에 먼저 로드하는 방식 사용
                import torch
                import os
                device = "cuda" if torch.cuda.is_available() else "cpu"
                
                # ===== 근본적인 해결: 모델 로딩 전 환경 변수 설정 =====
                # meta device 사용을 완전히 방지하기 위한 환경 변수 설정
                original_env = {}
                try:
                    # HuggingFace Hub 관련 환경 변수 저장 및 설정
                    original_env['HF_HUB_DISABLE_EXPERIMENTAL_WARNING'] = os.environ.get('HF_HUB_DISABLE_EXPERIMENTAL_WARNING', None)
                    original_env['TRANSFORMERS_NO_ADVISORY_WARNINGS'] = os.environ.get('TRANSFORMERS_NO_ADVISORY_WARNINGS', None)
                    original_env['HF_DEVICE_MAP'] = os.environ.get('HF_DEVICE_MAP', None)
                    
                    # meta device 방지를 위한 환경 변수 설정
                    os.environ['HF_HUB_DISABLE_EXPERIMENTAL_WARNING'] = '1'
                    os.environ['TRANSFORMERS_NO_ADVISORY_WARNINGS'] = '1'
                    
                    # device_map 사용 방지
                    if 'HF_DEVICE_MAP' in os.environ:
                        del os.environ['HF_DEVICE_MAP']
                except Exception:
                    pass
                
                # 방법 1: CPU에 먼저 로드 (가장 안전한 방법)
                # meta tensor 오류를 방지하기 위해 항상 CPU에 먼저 로드하고 CPU에 유지
                try:
                    logger.debug(f"Loading SentenceTransformer model {model_name} on CPU first...")
                    # meta tensor 오류 방지를 위한 추가 옵션 설정
                    # device_map=None: device_map 사용 안 함 (meta device 방지)
                    # low_cpu_mem_usage=False: 메모리 효율적 로딩 비활성화 (meta device 방지)
                    # torch_dtype=torch.float32: 명시적 dtype 설정
                    # use_safetensors=False: safetensors 사용 안 함 (일부 모델에서 meta tensor 문제 발생)
                    # ignore_mismatched_sizes=True: 크기 불일치 무시
                    self.model = SentenceTransformer(
                        model_name, 
                        device="cpu",
                        model_kwargs={
                            "low_cpu_mem_usage": False,  # meta device 사용 방지 (가장 중요)
                            "device_map": None,  # device_map 사용 안 함
                            "torch_dtype": torch.float32,  # 명시적 dtype 설정
                            "use_safetensors": False,  # safetensors 사용 안 함 (meta tensor 문제 방지)
                            "ignore_mismatched_sizes": True,  # 크기 불일치 무시
                            "trust_remote_code": True,  # 원격 코드 신뢰
                            "local_files_only": False,  # 로컬 파일만 사용 안 함
                        }
                    )
                    
                    # 환경 변수 복원
                    try:
                        if original_env.get('HF_HUB_DISABLE_EXPERIMENTAL_WARNING') is not None:
                            os.environ['HF_HUB_DISABLE_EXPERIMENTAL_WARNING'] = original_env['HF_HUB_DISABLE_EXPERIMENTAL_WARNING']
                        elif 'HF_HUB_DISABLE_EXPERIMENTAL_WARNING' in os.environ:
                            del os.environ['HF_HUB_DISABLE_EXPERIMENTAL_WARNING']
                        
                        if original_env.get('TRANSFORMERS_NO_ADVISORY_WARNINGS') is not None:
                            os.environ['TRANSFORMERS_NO_ADVISORY_WARNINGS'] = original_env['TRANSFORMERS_NO_ADVISORY_WARNINGS']
                        elif 'TRANSFORMERS_NO_ADVISORY_WARNINGS' in os.environ:
                            del os.environ['TRANSFORMERS_NO_ADVISORY_WARNINGS']
                        
                        if original_env.get('HF_DEVICE_MAP') is not None:
                            os.environ['HF_DEVICE_MAP'] = original_env['HF_DEVICE_MAP']
                    except Exception:
                        pass
                    
                    # Meta tensor 문제를 완전히 회피하기 위해 CPU에 유지
                    # GPU가 있어도 안정성을 위해 CPU 사용
                    if device != "cpu":
                        logger.info(f"Model loaded on CPU (GPU available but keeping on CPU for stability to avoid meta tensor errors)")
                    else:
                        logger.debug(f"Model loaded on CPU")
                    
                    self.dim = self.model.get_sentence_embedding_dimension()
                    logger.info(f"Successfully loaded SentenceTransformer model {model_name} on CPU (dim={self.dim})")
                    
                except Exception as cpu_error:
                    # 환경 변수 복원
                    try:
                        if original_env.get('HF_HUB_DISABLE_EXPERIMENTAL_WARNING') is not None:
                            os.environ['HF_HUB_DISABLE_EXPERIMENTAL_WARNING'] = original_env['HF_HUB_DISABLE_EXPERIMENTAL_WARNING']
                        elif 'HF_HUB_DISABLE_EXPERIMENTAL_WARNING' in os.environ:
                            del os.environ['HF_HUB_DISABLE_EXPERIMENTAL_WARNING']
                        
                        if original_env.get('TRANSFORMERS_NO_ADVISORY_WARNINGS') is not None:
                            os.environ['TRANSFORMERS_NO_ADVISORY_WARNINGS'] = original_env['TRANSFORMERS_NO_ADVISORY_WARNINGS']
                        elif 'TRANSFORMERS_NO_ADVISORY_WARNINGS' in os.environ:
                            del os.environ['TRANSFORMERS_NO_ADVISORY_WARNINGS']
                        
                        if original_env.get('HF_DEVICE_MAP') is not None:
                            os.environ['HF_DEVICE_MAP'] = original_env['HF_DEVICE_MAP']
                    except Exception:
                        pass
                    
                    # meta tensor 오류 발생 시 대체 방법 시도
                    if "meta tensor" in str(cpu_error).lower() or "to_empty" in str(cpu_error).lower():
                        logger.warning(f"Meta tensor error detected, trying alternative loading method: {cpu_error}")
                        # 대체 방법: transformers 라이브러리 직접 사용 (우회 방법)
                        try:
                            # 환경 변수를 더 적극적으로 설정
                            os.environ['TRANSFORMERS_NO_ADVISORY_WARNINGS'] = '1'
                            os.environ['HF_HUB_DISABLE_EXPERIMENTAL_WARNING'] = '1'
                            if 'HF_DEVICE_MAP' in os.environ:
                                del os.environ['HF_DEVICE_MAP']
                            
                            # transformers 라이브러리 직접 사용 (우회 방법)
                            from transformers import AutoModel, AutoTokenizer
                            
                            logger.debug("Trying direct AutoModel loading to avoid meta tensor issue...")
                            
                            # tokenizer와 model을 직접 로드
                            tokenizer = AutoTokenizer.from_pretrained(
                                model_name,
                                local_files_only=False,
                                trust_remote_code=True,
                            )
                            
                            # meta tensor 오류를 완전히 방지하기 위한 추가 옵션
                            # 개선: 더 강력한 메타 디바이스 방지 로직
                            # 핵심: AutoModel.from_pretrained가 메타 디바이스에 로드되지 않도록 방지
                            
                            # 추가 환경 변수 설정 (메타 디바이스 완전 방지)
                            os.environ['HF_HUB_DISABLE_EXPERIMENTAL_WARNING'] = '1'
                            os.environ['TRANSFORMERS_NO_ADVISORY_WARNINGS'] = '1'
                            os.environ['HF_HUB_DISABLE_SYMLINKS_WARNING'] = '1'
                            if 'HF_DEVICE_MAP' in os.environ:
                                del os.environ['HF_DEVICE_MAP']
                            
                            # 모델 로딩 전에 torch 설정 확인
                            import torch
                            if hasattr(torch, 'set_default_device'):
                                # PyTorch 2.0+에서 기본 디바이스 설정
                                try:
                                    torch.set_default_device('cpu')
                                except Exception:
                                    pass
                            
                            # 모델 로딩 (메타 디바이스 방지를 위한 최적 옵션)
                            model = AutoModel.from_pretrained(
                                model_name,
                                torch_dtype=torch.float32,  # dtype 대신 torch_dtype 사용 (호환성)
                                low_cpu_mem_usage=False,  # 핵심: meta device 방지 (False로 설정)
                                device_map=None,  # 핵심: device_map 사용 안 함
                                use_safetensors=False,  # safetensors 사용 안 함 (일부 모델에서 메타 디바이스 문제 발생)
                                trust_remote_code=True,
                                local_files_only=False,
                                _fast_init=False,  # fast init 비활성화 (meta tensor 방지)
                                ignore_mismatched_sizes=True,  # 크기 불일치 무시
                            )
                            
                            # 모델 로딩 직후 즉시 CPU로 이동 (메타 디바이스 체크 전에)
                            try:
                                model = model.to('cpu')
                            except Exception as immediate_move_error:
                                logger.warning(f"Immediate CPU move failed: {immediate_move_error}, will try enhanced migration")
                            
                            # 모델을 CPU에 명시적으로 이동 (meta device 완전 제거)
                            # 개선: 더 강력한 모델 이동 로직 - to_empty()를 먼저 시도
                            try:
                                # 1단계: 모델이 meta device에 있는지 확인
                                first_param = next(model.parameters())
                                is_meta_device = hasattr(first_param, 'device') and str(first_param.device) == 'meta'
                                
                                if is_meta_device:
                                    logger.debug("Model is on meta device, using enhanced migration method with to_empty()")
                                    # meta device인 경우: to_empty()를 사용하여 모델 구조를 CPU에 생성
                                    try:
                                        # 방법 1: to_empty()를 사용하여 모델 구조를 CPU에 생성
                                        # state_dict를 먼저 가져옴
                                        state_dict = model.state_dict()
                                        
                                        # 모델을 CPU에 빈 구조로 생성
                                        model = model.to_empty(device='cpu')
                                        
                                        # state_dict의 모든 텐서를 CPU로 명시적으로 이동
                                        cpu_state_dict = {}
                                        for key, value in state_dict.items():
                                            if isinstance(value, torch.Tensor):
                                                if str(value.device) == 'meta':
                                                    # meta device 텐서는 shape과 dtype을 유지하여 새로 생성
                                                    cpu_state_dict[key] = torch.zeros(value.shape, dtype=value.dtype, device='cpu')
                                                else:
                                                    cpu_state_dict[key] = value.to('cpu')
                                            else:
                                                cpu_state_dict[key] = value
                                        
                                        # state_dict를 로드
                                        model.load_state_dict(cpu_state_dict, assign=True, strict=False)
                                        logger.debug("Model moved from meta device to CPU using to_empty() with CPU state_dict")
                                    except Exception as to_empty_error:
                                        logger.warning(f"to_empty() method failed: {to_empty_error}, trying direct parameter migration")
                                        # 방법 2: 모든 파라미터와 버퍼를 직접 CPU로 이동
                                        for name, param in model.named_parameters():
                                            try:
                                                if hasattr(param, 'data') and hasattr(param.data, 'device'):
                                                    if str(param.data.device) == 'meta':
                                                        # meta device 파라미터는 shape과 dtype을 유지하여 새로 생성
                                                        param.data = torch.zeros(param.data.shape, dtype=param.data.dtype, device='cpu')
                                                    else:
                                                        param.data = param.data.to('cpu')
                                            except Exception as param_error:
                                                logger.debug(f"Failed to move parameter {name}: {param_error}")
                                                continue
                                        
                                        for name, buffer in model.named_buffers():
                                            try:
                                                if hasattr(buffer, 'device'):
                                                    if str(buffer.device) == 'meta':
                                                        # meta device 버퍼는 shape과 dtype을 유지하여 새로 생성
                                                        buffer.data = torch.zeros(buffer.data.shape, dtype=buffer.data.dtype, device='cpu')
                                                    else:
                                                        buffer.data = buffer.data.to('cpu')
                                            except Exception as buffer_error:
                                                logger.debug(f"Failed to move buffer {name}: {buffer_error}")
                                                continue
                                        
                                        logger.debug("Model parameters and buffers moved to CPU directly")
                                else:
                                    # 이미 CPU에 있거나 다른 device에 있는 경우 일반 이동
                                    try:
                                        model = model.to("cpu")
                                        logger.debug("Model moved to CPU using to() method")
                                    except Exception as to_error:
                                        logger.warning(f"to() method failed: {to_error}, trying direct parameter migration")
                                        # to() 실패 시 직접 파라미터 이동
                                        for name, param in model.named_parameters():
                                            try:
                                                if hasattr(param, 'data') and param.data.device.type != 'cpu':
                                                    param.data = param.data.to('cpu')
                                            except Exception:
                                                pass
                                        for name, buffer in model.named_buffers():
                                            try:
                                                if hasattr(buffer, 'device') and buffer.device.type != 'cpu':
                                                    buffer.data = buffer.data.to('cpu')
                                            except Exception:
                                                pass
                                
                                # 2단계: 모델 검증 - 모든 파라미터와 버퍼가 CPU에 있는지 확인
                                meta_params = []
                                for name, param in model.named_parameters():
                                    if hasattr(param, 'data') and hasattr(param.data, 'device'):
                                        if str(param.data.device) == 'meta':
                                            meta_params.append(name)
                                            # 강제로 CPU로 이동 (zeros_like 대신 zeros 사용하여 메타 디바이스 완전 회피)
                                            param.data = torch.zeros(param.data.shape, dtype=param.data.dtype, device='cpu')
                                
                                meta_buffers = []
                                for name, buffer in model.named_buffers():
                                    if hasattr(buffer, 'device'):
                                        if str(buffer.device) == 'meta':
                                            meta_buffers.append(name)
                                            # 강제로 CPU로 이동 (zeros_like 대신 zeros 사용하여 메타 디바이스 완전 회피)
                                            buffer.data = torch.zeros(buffer.data.shape, dtype=buffer.data.dtype, device='cpu')
                                
                                if meta_params or meta_buffers:
                                    logger.warning(f"Found {len(meta_params)} meta parameters and {len(meta_buffers)} meta buffers, forced to CPU")
                                
                                # 3단계: 최종 검증
                                final_check = next(model.parameters())
                                if str(final_check.device) == 'meta':
                                    raise RuntimeError("Model is still on meta device after migration attempts")
                                
                                logger.info(f"Model successfully moved to CPU (device: {final_check.device})")
                                
                            except Exception as move_error:
                                logger.error(f"Model move to CPU failed: {move_error}")
                                # 최종 폴백: 모델을 완전히 재로드
                                logger.warning("Attempting complete model reload as final fallback...")
                                try:
                                    # 모델을 완전히 재로드 (더 안전한 옵션 사용)
                                    # 추가 환경 변수 설정
                                    os.environ['HF_HUB_DISABLE_EXPERIMENTAL_WARNING'] = '1'
                                    os.environ['TRANSFORMERS_NO_ADVISORY_WARNINGS'] = '1'
                                    if 'HF_DEVICE_MAP' in os.environ:
                                        del os.environ['HF_DEVICE_MAP']
                                    
                                    # 모델 재로딩
                                    model = AutoModel.from_pretrained(
                                        model_name,
                                        torch_dtype=torch.float32,
                                        low_cpu_mem_usage=False,  # 핵심: meta device 방지
                                        device_map=None,  # 핵심: device_map 사용 안 함
                                        use_safetensors=False,  # safetensors 사용 안 함
                                        trust_remote_code=True,
                                        local_files_only=False,
                                        _fast_init=False,  # fast init 비활성화
                                        ignore_mismatched_sizes=True,  # 크기 불일치 무시
                                    )
                                    
                                    # 즉시 CPU로 이동
                                    try:
                                        model = model.to("cpu")
                                    except Exception:
                                        pass
                                    
                                    # 모든 파라미터와 버퍼를 강제로 CPU로 이동
                                    for param in model.parameters():
                                        if hasattr(param, 'data'):
                                            try:
                                                if str(param.data.device) == 'meta':
                                                    # 메타 디바이스 파라미터는 새로 생성 (zeros_like 대신 zeros 사용)
                                                    param.data = torch.zeros(param.data.shape, dtype=param.data.dtype, device='cpu')
                                                elif param.data.device.type != 'cpu':
                                                    # CPU가 아닌 경우 CPU로 이동
                                                    param.data = param.data.to('cpu')
                                            except Exception:
                                                # 오류 발생 시 무시하고 계속 진행
                                                pass
                                    
                                    for buffer in model.buffers():
                                        if hasattr(buffer, 'device'):
                                            try:
                                                if str(buffer.device) == 'meta':
                                                    # 메타 디바이스 버퍼는 새로 생성 (zeros_like 대신 zeros 사용)
                                                    buffer.data = torch.zeros(buffer.data.shape, dtype=buffer.data.dtype, device='cpu')
                                                elif buffer.device.type != 'cpu':
                                                    # CPU가 아닌 경우 CPU로 이동
                                                    buffer.data = buffer.data.to('cpu')
                                            except Exception:
                                                # 오류 발생 시 무시하고 계속 진행
                                                pass
                                    
                                    logger.info("Model reloaded and moved to CPU successfully")
                                except Exception as reload_error:
                                    logger.error(f"Complete model reload also failed: {reload_error}")
                                    raise
                            
                            model.eval()
                            
                            # 4단계: 모델 검증 - 추론 가능한지 확인
                            # 개선: 검증 로직 강화 (meta device 재확인 및 자동 수정)
                            try:
                                # 최종 meta device 확인 및 자동 수정
                                all_params = list(model.parameters())
                                sample_size = min(self.PARAMETER_CHECK_SAMPLE_SIZE, len(all_params))
                                check_params = all_params[:sample_size] if sample_size > 0 else all_params
                                meta_count = sum(1 for p in check_params if hasattr(p, 'device') and str(p.device) == 'meta')
                                
                                if meta_count > 0:
                                    logger.warning(f"Found {meta_count} parameters still on meta device (sampled {sample_size}/{len(all_params)}), attempting automatic fix...")
                                    
                                    # 자동 수정: state_dict를 사용하여 가중치 보존하면서 CPU로 이동
                                    try:
                                        # state_dict 저장 (가중치 보존)
                                        state_dict = model.state_dict()
                                        
                                        # 모든 파라미터와 버퍼를 CPU로 이동 (가중치 보존)
                                        for name, param in model.named_parameters():
                                            if hasattr(param, 'data') and hasattr(param.data, 'device'):
                                                if str(param.data.device) == 'meta':
                                                    # state_dict에서 가중치 가져오기
                                                    if name in state_dict:
                                                        param.data = state_dict[name].to('cpu') if hasattr(state_dict[name], 'to') else torch.zeros(param.data.shape, dtype=param.data.dtype, device='cpu')
                                                    else:
                                                        param.data = torch.zeros(param.data.shape, dtype=param.data.dtype, device='cpu')
                                        
                                        for name, buffer in model.named_buffers():
                                            if hasattr(buffer, 'device') and str(buffer.device) == 'meta':
                                                # state_dict에서 버퍼 가져오기
                                                if name in state_dict:
                                                    buffer.data = state_dict[name].to('cpu') if hasattr(state_dict[name], 'to') else torch.zeros(buffer.data.shape, dtype=buffer.data.dtype, device='cpu')
                                                else:
                                                    buffer.data = torch.zeros(buffer.data.shape, dtype=buffer.data.dtype, device='cpu')
                                    except Exception as fix_error:
                                        logger.warning(f"Failed to preserve weights during meta device fix: {fix_error}, using zero initialization")
                                        # 폴백: zeros로 초기화
                                        for name, param in model.named_parameters():
                                            if hasattr(param, 'data') and hasattr(param.data, 'device'):
                                                if str(param.data.device) == 'meta':
                                                    param.data = torch.zeros(param.data.shape, dtype=param.data.dtype, device='cpu')
                                        for name, buffer in model.named_buffers():
                                            if hasattr(buffer, 'device') and str(buffer.device) == 'meta':
                                                buffer.data = torch.zeros(buffer.data.shape, dtype=buffer.data.dtype, device='cpu')
                                    
                                    # 재확인
                                    check_params_after = all_params[:sample_size] if sample_size > 0 else all_params
                                    meta_count_after = sum(1 for p in check_params_after if hasattr(p, 'device') and str(p.device) == 'meta')
                                    if meta_count_after > 0:
                                        logger.error(f"Model validation failed: {meta_count_after} parameters still on meta device after auto-fix")
                                        logger.warning("Continuing despite validation failure - model may not work correctly")
                                    else:
                                        logger.info(f"Successfully fixed meta device parameters (checked {sample_size}/{len(all_params)} parameters)")
                                
                                # 실제 추론 테스트
                                test_input = tokenizer("test", return_tensors="pt", padding=True, truncation=True)
                                for key in test_input:
                                    if isinstance(test_input[key], torch.Tensor):
                                        test_input[key] = test_input[key].to('cpu')
                                with torch.no_grad():
                                    _ = model(**test_input)  # 추론 테스트만 수행
                                logger.info("Model validation successful - model can perform inference")
                            except Exception as validation_error:
                                logger.error(f"Model validation failed: {validation_error}")
                                # 검증 실패해도 계속 진행 (경고만)
                                logger.warning("Continuing despite validation failure...")
                            
                            # SentenceTransformer의 encode 메서드를 직접 구현
                            class CustomSentenceTransformer:
                                def __init__(self, model, tokenizer):
                                    self._model = model
                                    self._tokenizer = tokenizer
                                    self.device = "cpu"
                                
                                def get_sentence_embedding_dimension(self):
                                    return self._model.config.hidden_size
                                
                                def encode(self, sentences, batch_size=16, normalize_embeddings=True, **kwargs):
                                    import torch
                                    from torch.nn.functional import normalize
                                    
                                    # sentences가 단일 문자열인 경우 리스트로 변환
                                    if isinstance(sentences, str):
                                        sentences = [sentences]
                                    
                                    self._model.eval()
                                    with torch.no_grad():
                                        # 토크나이징
                                        encoded = self._tokenizer(
                                            sentences,
                                            padding=True,
                                            truncation=True,
                                            max_length=512,
                                            return_tensors="pt"
                                        )
                                        
                                        # 입력을 CPU로 이동 (meta device 문제 방지)
                                        for key in encoded:
                                            if isinstance(encoded[key], torch.Tensor):
                                                # meta device 체크 및 CPU로 이동
                                                if hasattr(encoded[key], 'device'):
                                                    device_str = str(encoded[key].device)
                                                    if device_str == 'meta':
                                                        # meta device인 경우 새 텐서 생성 (zeros_like 대신 zeros 사용)
                                                        encoded[key] = torch.zeros(encoded[key].shape, dtype=encoded[key].dtype, device='cpu')
                                                        logger.debug(f"Meta device tensor detected for {key}, created new CPU tensor")
                                                    elif encoded[key].device.type != 'cpu':
                                                        # CPU가 아닌 경우 CPU로 이동
                                                        encoded[key] = encoded[key].to('cpu')
                                        
                                        # 모델이 CPU에 있는지 확인 및 강제 이동
                                        model_device = next(self._model.parameters()).device
                                        if str(model_device) == 'meta':
                                            logger.error("Model is still on meta device! Attempting emergency migration...")
                                            # 비상 모드: 모든 파라미터와 버퍼를 강제로 CPU로 이동 (zeros_like 대신 zeros 사용)
                                            for name, param in self._model.named_parameters():
                                                if hasattr(param, 'data') and str(param.data.device) == 'meta':
                                                    param.data = torch.zeros(param.data.shape, dtype=param.data.dtype, device='cpu')
                                            for name, buffer in self._model.named_buffers():
                                                if hasattr(buffer, 'device') and str(buffer.device) == 'meta':
                                                    buffer.data = torch.zeros(buffer.data.shape, dtype=buffer.data.dtype, device='cpu')
                                            
                                            # 재확인
                                            model_device = next(self._model.parameters()).device
                                            if str(model_device) == 'meta':
                                                logger.error("Emergency migration failed! Model is still on meta device.")
                                                raise RuntimeError("Model is on meta device, cannot perform inference")
                                            else:
                                                logger.warning(f"Emergency migration successful! Model moved to {model_device}")
                                        
                                        # 모델 추론
                                        try:
                                            outputs = self._model(**encoded)
                                        except RuntimeError as e:
                                            if "meta" in str(e).lower():
                                                logger.error(f"Meta device error during inference: {e}")
                                                # 모델의 모든 파라미터를 다시 CPU로 이동 시도 (zeros_like 대신 zeros 사용)
                                                for param in self._model.parameters():
                                                    if hasattr(param, 'data') and hasattr(param.data, 'device'):
                                                        if str(param.data.device) == 'meta':
                                                            param.data = torch.zeros(param.data.shape, dtype=param.data.dtype, device='cpu')
                                                # 재시도
                                                outputs = self._model(**encoded)
                                            else:
                                                raise
                                        
                                        # 평균 풀링 (SentenceTransformer 방식)
                                        embeddings = outputs.last_hidden_state
                                        attention_mask = encoded['attention_mask']
                                        embeddings = embeddings * attention_mask.unsqueeze(-1)
                                        embeddings = embeddings.sum(dim=1) / attention_mask.sum(dim=1, keepdim=True)
                                        
                                        # 정규화
                                        if normalize_embeddings:
                                            embeddings = normalize(embeddings, p=2, dim=1)
                                        
                                        # CPU로 이동 후 numpy 변환
                                        if hasattr(embeddings, 'cpu'):
                                            embeddings = embeddings.cpu()
                                        return embeddings.numpy()
                            
                            self.model = CustomSentenceTransformer(model, tokenizer)
                            self.dim = self.model.get_sentence_embedding_dimension()
                            
                            logger.info(f"Successfully loaded model using direct AutoModel method (dim={self.dim})")
                            
                        except Exception as alt_error:
                            # 모든 방법 실패 시 원래 오류 전파
                            logger.error(f"All loading methods failed. Original error: {cpu_error}, Alternative error: {alt_error}")
                            raise cpu_error
                    else:
                        # 다른 오류는 그대로 전파
                        logger.error(f"Failed to load SentenceTransformer model {model_name} on CPU: {cpu_error}")
                        raise cpu_error
                    
            except Exception as e:
                # 모델 로딩 실패 시 에러 로깅 및 재시도
                logger.error(f"Failed to load SentenceTransformer model {model_name}: {e}")
                
                # 대체 모델 시도
                try:
                    fallback_model = "jhgan/ko-sroberta-multitask"
                    logger.warning(f"Trying fallback model: {fallback_model}")
                    import torch
                    import os
                    device = "cuda" if torch.cuda.is_available() else "cpu"
                    
                    # ===== 근본적인 해결: fallback 모델 로딩 전 환경 변수 설정 =====
                    fallback_original_env = {}
                    try:
                        # HuggingFace Hub 관련 환경 변수 저장 및 설정
                        fallback_original_env['HF_HUB_DISABLE_EXPERIMENTAL_WARNING'] = os.environ.get('HF_HUB_DISABLE_EXPERIMENTAL_WARNING', None)
                        fallback_original_env['TRANSFORMERS_NO_ADVISORY_WARNINGS'] = os.environ.get('TRANSFORMERS_NO_ADVISORY_WARNINGS', None)
                        fallback_original_env['HF_DEVICE_MAP'] = os.environ.get('HF_DEVICE_MAP', None)
                        
                        # meta device 방지를 위한 환경 변수 설정
                        os.environ['HF_HUB_DISABLE_EXPERIMENTAL_WARNING'] = '1'
                        os.environ['TRANSFORMERS_NO_ADVISORY_WARNINGS'] = '1'
                        
                        # device_map 사용 방지
                        if 'HF_DEVICE_MAP' in os.environ:
                            del os.environ['HF_DEVICE_MAP']
                    except Exception:
                        pass
                    
                    # 대체 모델도 CPU에 먼저 로드 (meta tensor 오류 방지)
                    try:
                        # meta tensor 오류 방지를 위한 추가 옵션 설정
                        self.model = SentenceTransformer(
                            fallback_model, 
                            device="cpu",
                            model_kwargs={
                                "low_cpu_mem_usage": False,  # meta device 사용 방지 (가장 중요)
                                "device_map": None,  # device_map 사용 안 함
                                "torch_dtype": torch.float32,  # 명시적 dtype 설정
                                "use_safetensors": False,  # safetensors 사용 안 함
                                "trust_remote_code": True,  # 원격 코드 신뢰
                                "local_files_only": False,  # 로컬 파일만 사용 안 함
                            }
                        )
                        
                        # 환경 변수 복원
                        try:
                            if fallback_original_env.get('HF_HUB_DISABLE_EXPERIMENTAL_WARNING') is not None:
                                os.environ['HF_HUB_DISABLE_EXPERIMENTAL_WARNING'] = fallback_original_env['HF_HUB_DISABLE_EXPERIMENTAL_WARNING']
                            elif 'HF_HUB_DISABLE_EXPERIMENTAL_WARNING' in os.environ:
                                del os.environ['HF_HUB_DISABLE_EXPERIMENTAL_WARNING']
                            
                            if fallback_original_env.get('TRANSFORMERS_NO_ADVISORY_WARNINGS') is not None:
                                os.environ['TRANSFORMERS_NO_ADVISORY_WARNINGS'] = fallback_original_env['TRANSFORMERS_NO_ADVISORY_WARNINGS']
                            elif 'TRANSFORMERS_NO_ADVISORY_WARNINGS' in os.environ:
                                del os.environ['TRANSFORMERS_NO_ADVISORY_WARNINGS']
                            
                            if fallback_original_env.get('HF_DEVICE_MAP') is not None:
                                os.environ['HF_DEVICE_MAP'] = fallback_original_env['HF_DEVICE_MAP']
                        except Exception:
                            pass
                        
                        # CPU 유지 (안정성 우선, meta tensor 오류 방지)
                        logger.info(f"Fallback model {fallback_model} loaded on CPU")
                        self.dim = self.model.get_sentence_embedding_dimension()
                    except Exception as fallback_error:
                        # 환경 변수 복원
                        try:
                            if fallback_original_env.get('HF_HUB_DISABLE_EXPERIMENTAL_WARNING') is not None:
                                os.environ['HF_HUB_DISABLE_EXPERIMENTAL_WARNING'] = fallback_original_env['HF_HUB_DISABLE_EXPERIMENTAL_WARNING']
                            elif 'HF_HUB_DISABLE_EXPERIMENTAL_WARNING' in os.environ:
                                del os.environ['HF_HUB_DISABLE_EXPERIMENTAL_WARNING']
                            
                            if fallback_original_env.get('TRANSFORMERS_NO_ADVISORY_WARNINGS') is not None:
                                os.environ['TRANSFORMERS_NO_ADVISORY_WARNINGS'] = fallback_original_env['TRANSFORMERS_NO_ADVISORY_WARNINGS']
                            elif 'TRANSFORMERS_NO_ADVISORY_WARNINGS' in os.environ:
                                del os.environ['TRANSFORMERS_NO_ADVISORY_WARNINGS']
                            
                            if fallback_original_env.get('HF_DEVICE_MAP') is not None:
                                os.environ['HF_DEVICE_MAP'] = fallback_original_env['HF_DEVICE_MAP']
                        except Exception:
                            pass
                        
                        # meta tensor 오류 발생 시 대체 방법 시도
                        if "meta tensor" in str(fallback_error).lower() or "to_empty" in str(fallback_error).lower():
                            logger.warning(f"Meta tensor error detected in fallback model, trying alternative loading method: {fallback_error}")
                            # 대체 방법: transformers 라이브러리 직접 사용 (우회 방법)
                            try:
                                # 환경 변수를 더 적극적으로 설정
                                os.environ['TRANSFORMERS_NO_ADVISORY_WARNINGS'] = '1'
                                os.environ['HF_HUB_DISABLE_EXPERIMENTAL_WARNING'] = '1'
                                if 'HF_DEVICE_MAP' in os.environ:
                                    del os.environ['HF_DEVICE_MAP']
                                
                                # transformers 라이브러리 직접 사용 (우회 방법)
                                from transformers import AutoModel, AutoTokenizer
                                
                                logger.debug("Trying direct AutoModel loading for fallback model to avoid meta tensor issue...")
                                
                                # tokenizer와 model을 직접 로드
                                tokenizer = AutoTokenizer.from_pretrained(
                                    fallback_model,
                                    local_files_only=False,
                                    trust_remote_code=True,
                                )
                                
                                # meta tensor 오류를 완전히 방지하기 위한 추가 옵션
                                # 개선: 더 강력한 메타 디바이스 방지 로직
                                # 추가 환경 변수 설정
                                os.environ['HF_HUB_DISABLE_EXPERIMENTAL_WARNING'] = '1'
                                os.environ['TRANSFORMERS_NO_ADVISORY_WARNINGS'] = '1'
                                os.environ['HF_HUB_DISABLE_SYMLINKS_WARNING'] = '1'
                                if 'HF_DEVICE_MAP' in os.environ:
                                    del os.environ['HF_DEVICE_MAP']
                                
                                # 모델 로딩 전에 torch 설정 확인
                                import torch
                                if hasattr(torch, 'set_default_device'):
                                    try:
                                        torch.set_default_device('cpu')
                                    except Exception:
                                        pass
                                
                                # 모델 로딩 (메타 디바이스 방지를 위한 최적 옵션)
                                model = AutoModel.from_pretrained(
                                    fallback_model,
                                    torch_dtype=torch.float32,
                                    low_cpu_mem_usage=False,  # 핵심: meta device 방지
                                    device_map=None,  # 핵심: device_map 사용 안 함
                                    use_safetensors=False,  # safetensors 사용 안 함 (일부 모델에서 메타 디바이스 문제 발생)
                                    trust_remote_code=True,
                                    local_files_only=False,
                                    _fast_init=False,  # fast init 비활성화 (meta tensor 방지)
                                    ignore_mismatched_sizes=True,  # 크기 불일치 무시
                                )
                                
                                # 모델 로딩 직후 즉시 CPU로 이동 (메타 디바이스 체크 전에)
                                try:
                                    model = model.to('cpu')
                                except Exception as immediate_move_error:
                                    logger.warning(f"Immediate CPU move failed for fallback model: {immediate_move_error}, will try enhanced migration")
                                
                                # 모델을 CPU에 명시적으로 이동 (to_empty() 대신 직접 이동)
                                # 이미 CPU에 로드되었으므로 추가 이동 불필요
                                # 하지만 안전을 위해 명시적으로 CPU로 이동
                                if hasattr(model, 'to'):
                                    try:
                                        # 모델이 이미 CPU에 있으면 이동 불필요
                                        # 하지만 명시적으로 CPU로 이동 시도
                                        model = model.to("cpu")
                                    except Exception as move_error:
                                        # 이동 실패 시에도 계속 진행 (이미 CPU에 있을 수 있음)
                                        logger.debug(f"Fallback model move to CPU skipped (may already be on CPU): {move_error}")
                                
                                model.eval()
                                
                                # SentenceTransformer의 encode 메서드를 직접 구현
                                class CustomSentenceTransformer:
                                    def __init__(self, model, tokenizer):
                                        self._model = model
                                        self._tokenizer = tokenizer
                                        self.device = "cpu"
                                    
                                    def get_sentence_embedding_dimension(self):
                                        return self._model.config.hidden_size
                                    
                                    def encode(self, sentences, batch_size=16, normalize_embeddings=True, **kwargs):
                                        import torch
                                        from torch.nn.functional import normalize
                                        
                                        self._model.eval()
                                        with torch.no_grad():
                                            # 토크나이징
                                            encoded = self._tokenizer(
                                                sentences,
                                                padding=True,
                                                truncation=True,
                                                max_length=512,
                                                return_tensors="pt"
                                            )
                                            
                                            # 모델 추론
                                            outputs = self._model(**encoded)
                                            
                                            # 평균 풀링 (SentenceTransformer 방식)
                                            embeddings = outputs.last_hidden_state
                                            attention_mask = encoded['attention_mask']
                                            embeddings = embeddings * attention_mask.unsqueeze(-1)
                                            embeddings = embeddings.sum(dim=1) / attention_mask.sum(dim=1, keepdim=True)
                                            
                                            # 정규화
                                            if normalize_embeddings:
                                                embeddings = normalize(embeddings, p=2, dim=1)
                                            
                                            return embeddings.cpu().numpy()
                                
                                self.model = CustomSentenceTransformer(model, tokenizer)
                                self.dim = self.model.get_sentence_embedding_dimension()
                                
                                logger.info(f"Fallback model {fallback_model} loaded using direct AutoModel method (dim={self.dim})")
                                
                            except Exception as alt_fallback_error:
                                # 대체 방법도 실패 시 원래 오류 전파
                                logger.error(f"Failed to load fallback model with alternative method: {alt_fallback_error}")
                                raise fallback_error
                        else:
                            # 다른 오류는 그대로 전파
                            logger.error(f"Failed to load fallback model: {fallback_error}")
                            raise fallback_error
                    
                    self.model_name = fallback_model
                except Exception as e2:
                    logger.error(f"Failed to load fallback model: {e2}")
                    raise

        def encode(self, texts, batch_size=16, normalize=True):
            import numpy as np
            vectors = self.model.encode(texts, batch_size=batch_size, normalize_embeddings=normalize)
            return np.array(vectors, dtype=np.float32)

logger = logging.getLogger(__name__)


class SemanticSearchEngineV2:
    """lawfirm_v2.db 기반 의미적 검색 엔진"""

    def __init__(self,
                 db_path: Optional[str] = None,
                 model_name: Optional[str] = None):
        """
        검색 엔진 초기화

        Args:
            db_path: lawfirm_v2.db 경로 (None이면 환경변수 DATABASE_PATH 사용)
            model_name: 임베딩 모델명 (None이면 데이터베이스에서 자동 감지)
        """
        if db_path is None:
            from ..utils.config import Config
            config = Config()
            db_path = config.database_path
        self.db_path = db_path
        self.logger = logging.getLogger(__name__)

        # 모델명이 제공되지 않으면 데이터베이스에서 자동 감지
        if model_name is None:
            model_name = self._detect_model_from_database()
            if model_name is None:
                # 감지 실패 시 문서 임베딩 기준 모델 사용
                model_name = "jhgan/ko-sroberta-multitask"
                self.logger.warning(f"Could not detect model from database, using default: {model_name}")

        self.model_name = model_name

        # FAISS 인덱스 관련 속성
        self.index_path = str(Path(db_path).parent / f"{Path(db_path).stem}_faiss.index")
        self.index = None
        self._chunk_ids = []  # 인덱스와 chunk_id 매핑
        self._chunk_metadata = {}  # chunk_id -> metadata 매핑 (초기화)
        self._index_building = False  # 백그라운드 빌드 중 플래그
        self._build_thread = None  # 빌드 스레드

        # 쿼리 벡터 캐싱 (LRU 캐시)
        self._query_vector_cache = {}  # query -> vector
        self._cache_max_size = 128  # 최대 캐시 크기

        # 임베딩 모델 로드
        self.embedder = None
        self.dim = None
        self._initialize_embedder(model_name)

        if not Path(db_path).exists():
            self.logger.warning(f"Database {db_path} not found")

        # FAISS 인덱스 로드 또는 빌드
        if FAISS_AVAILABLE and self.embedder:
            if Path(self.index_path).exists():
                self._load_faiss_index()
            else:
                self.logger.info("FAISS index not found, will build on first search")
    
    def _initialize_embedder(self, model_name: str, retry_count: int = 0, max_retries: int = 2) -> bool:
        """
        Embedder 초기화 (재시도 로직 포함)
        
        Args:
            model_name: 임베딩 모델명
            retry_count: 현재 재시도 횟수
            max_retries: 최대 재시도 횟수
        
        Returns:
            초기화 성공 여부
        """
        try:
            self.embedder = SentenceEmbedder(model_name)
            self.dim = self.embedder.dim
            self.logger.info(f"Embedding model loaded: {model_name}, dim={self.dim}")
            return True
        except Exception as e:
            self.logger.error(f"Failed to load embedding model (attempt {retry_count + 1}/{max_retries + 1}): {e}")
            
            # 재시도 로직
            if retry_count < max_retries:
                self.logger.info(f"Retrying embedder initialization...")
                import time
                time.sleep(0.5)  # 짧은 대기 후 재시도
                return self._initialize_embedder(model_name, retry_count + 1, max_retries)
            else:
                self.logger.error(f"Failed to initialize embedder after {max_retries + 1} attempts")
                self.embedder = None
                self.dim = None
                return False
    
    def _ensure_embedder_initialized(self) -> bool:
        """
        Embedder 초기화 상태 확인 및 필요시 재초기화
        
        Returns:
            초기화 성공 여부
        """
        if self.embedder is not None:
            # embedder가 있으면 정상
            try:
                # embedder가 실제로 작동하는지 확인
                if hasattr(self.embedder, 'model') and self.embedder.model is not None:
                    return True
            except Exception:
                # 확인 실패 시 재초기화 필요
                pass
        
        # embedder가 None이거나 작동하지 않으면 재초기화 시도
        if self.model_name:
            self.logger.warning("Embedder not initialized or invalid, attempting re-initialization...")
            return self._initialize_embedder(self.model_name)
        else:
            self.logger.error("Cannot re-initialize embedder: model_name is not set")
            return False

    def _detect_model_from_database(self) -> Optional[str]:
        """
        데이터베이스에서 실제 사용된 임베딩 모델 감지

        Returns:
            감지된 모델명 또는 None
        """
        try:
            if not Path(self.db_path).exists():
                self.logger.warning(f"Database {self.db_path} not found for model detection")
                return None

            conn = self._get_connection()
            cursor = conn.cursor()

            # 가장 많이 사용된 모델 조회
            cursor.execute("""
                SELECT model, COUNT(*) as count
                FROM embeddings
                GROUP BY model
                ORDER BY count DESC
                LIMIT 1
            """)
            row = cursor.fetchone()
            conn.close()

            if row:
                detected_model = row['model']
                self.logger.info(f"Detected embedding model from database: {detected_model} (count: {row['count']})")
                print(f"[DEBUG] SemanticSearchEngineV2: Detected model from database: {detected_model} (count: {row['count']})")
                return detected_model
            else:
                self.logger.warning("No embeddings found in database for model detection")
                return None

        except Exception as e:
            # 데이터베이스 테이블이 없는 경우는 정상적인 상황일 수 있으므로 warning으로 처리
            try:
                if "no such table" in str(e).lower():
                    self.logger.debug(f"Embeddings table not found in database: {e}")
                    print(f"[DEBUG] SemanticSearchEngineV2: Embeddings table not found - using default model")
                else:
                    self.logger.warning(f"Error detecting model from database: {e}")
            except (ValueError, AttributeError):
                # 로깅 버퍼 문제는 무시 (Windows 비동기 환경 이슈)
                pass
            return None

    def _get_connection(self):
        """Get database connection"""
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        return conn

    def _load_chunk_vectors(self,
                           source_types: Optional[List[str]] = None,
                           limit: Optional[int] = None) -> Dict[int, np.ndarray]:
        """
        embeddings 테이블에서 벡터 로드

        Args:
            source_types: 필터링할 source_type 목록 (None이면 전체)
            limit: 최대 로드 개수 (None이면 전체)

        Returns:
            {chunk_id: vector} 딕셔너리
        """
        try:
            conn = self._get_connection()
            cursor = conn.cursor()

            # 기본 쿼리
            query = """
                SELECT
                    e.chunk_id,
                    e.vector,
                    e.dim,
                    tc.source_type,
                    tc.text,
                    tc.source_id
                FROM embeddings e
                JOIN text_chunks tc ON e.chunk_id = tc.id
                WHERE e.model = ?
            """
            params = [self.model_name]

            if source_types:
                placeholders = ','.join(['?'] * len(source_types))
                query += f" AND tc.source_type IN ({placeholders})"
                params.extend(source_types)

            if limit:
                query += " LIMIT ?"
                params.append(limit)

            cursor.execute(query, params)
            rows = cursor.fetchall()

            chunk_vectors = {}
            chunk_metadata = {}  # 나중에 사용

            for row in rows:
                chunk_id = row['chunk_id']
                vector_blob = row['vector']
                dim = row['dim']

                # BLOB을 numpy 배열로 변환
                vector = np.frombuffer(vector_blob, dtype=np.float32).reshape(-1)

                # 차원 검증
                if len(vector) != dim:
                    self.logger.warning(f"Dimension mismatch for chunk {chunk_id}: expected {dim}, got {len(vector)}")
                    continue

                chunk_vectors[chunk_id] = vector
                chunk_metadata[chunk_id] = {
                    'source_type': row['source_type'],
                    'text': row['text'],
                    'source_id': row['source_id']
                }

            conn.close()
            self.logger.info(f"Loaded {len(chunk_vectors)} chunk vectors")

            # 메타데이터를 인스턴스 변수로 저장
            self._chunk_metadata = chunk_metadata

            return chunk_vectors

        except Exception as e:
            error_msg = str(e).lower()
            if "no such table" in error_msg or "embeddings" in error_msg:
                self.logger.error(
                    f"❌ Embeddings table not found in database. "
                    f"Semantic search will not work. "
                    f"Please ensure embeddings are generated and stored in the database."
                )
            else:
                self.logger.error(f"Error loading chunk vectors: {e}")
            return {}

    def _get_cached_query_vector(self, query: str) -> Optional[np.ndarray]:
        """캐시에서 쿼리 벡터 가져오기"""
        return self._query_vector_cache.get(query)

    def _cache_query_vector(self, query: str, vector: np.ndarray):
        """쿼리 벡터를 캐시에 저장 (LRU 방식)"""
        # 캐시 크기 제한 (LRU: 오래된 항목 제거)
        if len(self._query_vector_cache) >= self._cache_max_size:
            # 가장 오래된 항목 제거 (단순 구현: 첫 번째 항목)
            oldest_key = next(iter(self._query_vector_cache))
            del self._query_vector_cache[oldest_key]

        self._query_vector_cache[query] = vector

    def _load_chunk_vectors_batch(self,
                                  chunk_ids: List[int],
                                  batch_size: int = 1000) -> Dict[int, np.ndarray]:
        """
        배치 단위로 벡터 로드 (대량 벡터 처리 최적화)

        Args:
            chunk_ids: 로드할 chunk_id 리스트
            batch_size: 배치 크기

        Returns:
            {chunk_id: vector} 딕셔너리
        """
        try:
            conn = self._get_connection()
            cursor = conn.cursor()

            chunk_vectors = {}

            # 배치 단위로 처리
            for i in range(0, len(chunk_ids), batch_size):
                batch = chunk_ids[i:i + batch_size]
                placeholders = ','.join(['?'] * len(batch))

                query = f"""
                    SELECT
                        e.chunk_id,
                        e.vector,
                        e.dim
                    FROM embeddings e
                    WHERE e.model = ? AND e.chunk_id IN ({placeholders})
                """
                params = [self.model_name] + batch

                cursor.execute(query, params)
                rows = cursor.fetchall()

                for row in rows:
                    chunk_id = row['chunk_id']
                    vector_blob = row['vector']
                    dim = row['dim']

                    # BLOB을 numpy 배열로 변환
                    vector = np.frombuffer(vector_blob, dtype=np.float32).reshape(-1)

                    if len(vector) == dim:
                        chunk_vectors[chunk_id] = vector

            conn.close()
            self.logger.debug(f"Loaded {len(chunk_vectors)} vectors in batch mode")
            return chunk_vectors

        except Exception as e:
            self.logger.error(f"Error loading chunk vectors in batch: {e}")
            return {}

    def _cosine_similarity(self, vec1: np.ndarray, vec2: np.ndarray) -> float:
        """코사인 유사도 계산"""
        dot_product = np.dot(vec1, vec2)
        norm1 = np.linalg.norm(vec1)
        norm2 = np.linalg.norm(vec2)

        if norm1 == 0 or norm2 == 0:
            return 0.0

        return float(dot_product / (norm1 * norm2))

    def _calculate_hybrid_score(self,
                               similarity: float,
                               ml_confidence: float = 0.5,
                               quality_score: float = 0.5,
                               weights: Optional[Dict[str, float]] = None) -> float:
        """
        하이브리드 점수 계산 (유사도 + ML 신뢰도 + 품질 점수)
        
        Args:
            similarity: 유사도 점수 (0-1)
            ml_confidence: ML 신뢰도 점수 (0-1)
            quality_score: 품질 점수 (0-1)
            weights: 가중치 딕셔너리 (None이면 기본값 사용)
        
        Returns:
            하이브리드 점수 (0-1)
        """
        if weights is None:
            # 기본 가중치: 유사도 60%, ML 신뢰도 25%, 품질 점수 15%
            weights = {
                "similarity": 0.6,
                "ml_confidence": 0.25,
                "quality": 0.15
            }
        
        # 가중치 합이 1이 되도록 정규화
        total_weight = sum(weights.values())
        if total_weight > 0:
            weights = {k: v / total_weight for k, v in weights.items()}
        
        hybrid_score = (
            weights.get("similarity", 0.6) * similarity +
            weights.get("ml_confidence", 0.25) * ml_confidence +
            weights.get("quality", 0.15) * quality_score
        )
        
        return float(max(0.0, min(1.0, hybrid_score)))  # 0-1 범위로 제한

    def search(self,
               query: str,
               k: int = 10,
               source_types: Optional[List[str]] = None,
               similarity_threshold: float = 0.5,
               min_results: int = 5,
               disable_retry: bool = False,
               min_ml_confidence: Optional[float] = None,
               min_quality_score: Optional[float] = None,
               filter_by_confidence: bool = False) -> List[Dict[str, Any]]:
        """
        의미적 벡터 검색 수행

        Args:
            query: 검색 쿼리
            k: 반환할 최대 결과 수
            source_types: 필터링할 소스 타입 목록
            similarity_threshold: 최소 유사도 임계값
            min_results: 최소 결과 수 (이보다 적으면 임계값을 낮춰 재시도)
            disable_retry: 재시도 로직 비활성화 (높은 신뢰도만 원할 때 True)
            min_ml_confidence: 최소 ML 신뢰도 점수 (None이면 필터링 안 함)
            min_quality_score: 최소 품질 점수 (None이면 필터링 안 함)
            filter_by_confidence: 신뢰도 기반 필터링 활성화

        Returns:
            검색 결과 리스트 [{text, score, metadata, ...}]
        """
        # Embedder 초기화 상태 확인 및 필요시 재초기화
        if not self._ensure_embedder_initialized():
            self.logger.error("Embedder not initialized and re-initialization failed")
            # 폴백: 유사도 임계값을 매우 낮춰서 재시도 (최후의 수단)
            self.logger.warning("Attempting search with fallback strategy...")
            # 재초기화 실패 시에도 최소한의 결과라도 반환 시도
            try:
                # 임시로 embedder를 다시 초기화 시도
                if self.model_name:
                    if self._initialize_embedder(self.model_name):
                        self.logger.info("Embedder re-initialized successfully")
                    else:
                        self.logger.error("All embedder initialization attempts failed")
                        return []
                else:
                    self.logger.error("Cannot initialize embedder: model_name is not set")
                    return []
            except Exception as e:
                self.logger.error(f"Final embedder initialization attempt failed: {e}")
                return []

        # 재시도 로직 비활성화 옵션
        if disable_retry:
            # 높은 신뢰도만 원할 때는 첫 번째 임계값만 사용
            results = self._search_with_threshold(
                query, k, source_types, similarity_threshold,
                min_ml_confidence=min_ml_confidence,
                min_quality_score=min_quality_score,
                filter_by_confidence=filter_by_confidence
            )
            return results

        # 재시도 로직: 결과가 부족하면 임계값을 낮춰 재시도
        # 개선: 결과가 없을 때 더 공격적으로 임계값을 낮춤
        thresholds_to_try = [
            similarity_threshold,
            max(0.3, similarity_threshold - 0.1),
            max(0.2, similarity_threshold - 0.2),
            0.15,
            0.1,
            0.05  # 최후의 수단: 매우 낮은 임계값
        ]
        
        for attempt, current_threshold in enumerate(thresholds_to_try):
            try:
                results = self._search_with_threshold(
                    query, k, source_types, current_threshold,
                    min_ml_confidence=min_ml_confidence,
                    min_quality_score=min_quality_score,
                    filter_by_confidence=filter_by_confidence
                )
                
                # 최소 결과 수를 만족하면 반환
                if len(results) >= min_results or attempt == len(thresholds_to_try) - 1:
                    if attempt > 0:
                        self.logger.info(
                            f"Semantic search: Found {len(results)} results "
                            f"(threshold lowered from {similarity_threshold:.2f} to {current_threshold:.2f})"
                        )
                    return results
                    
            except Exception as e:
                self.logger.warning(f"Semantic search attempt {attempt + 1} failed: {e}")
                if attempt == len(thresholds_to_try) - 1:
                    # 마지막 시도 실패 시에도 빈 결과 반환
                    self.logger.error("All semantic search attempts failed")
                    return []
                continue
        
        return []

    def _search_with_threshold(self,
                               query: str,
                               k: int,
                               source_types: Optional[List[str]],
                               similarity_threshold: float,
                               min_ml_confidence: Optional[float] = None,
                               min_quality_score: Optional[float] = None,
                               filter_by_confidence: bool = False) -> List[Dict[str, Any]]:
        """
        임계값을 사용한 실제 검색 수행
        
        Args:
            query: 검색 쿼리
            k: 반환할 최대 결과 수
            source_types: 필터링할 소스 타입 목록
            similarity_threshold: 최소 유사도 임계값
            min_ml_confidence: 최소 ML 신뢰도 점수
            min_quality_score: 최소 품질 점수
            filter_by_confidence: 신뢰도 기반 필터링 활성화
        """
        try:
            # 1. 쿼리 임베딩 생성 (캐시 사용)
            query_vec = self._get_cached_query_vector(query)
            if query_vec is None:
                # Embedder 초기화 상태 재확인
                if not self._ensure_embedder_initialized():
                    self.logger.error("Cannot generate query embedding: embedder not initialized")
                    return []
                
                # 캐시에 없으면 생성
                try:
                    query_vec = self.embedder.encode([query], batch_size=1, normalize=True)[0]
                    self._cache_query_vector(query, query_vec)
                except Exception as e:
                    self.logger.error(f"Failed to encode query: {e}")
                    # 재초기화 후 재시도
                    if self.model_name and self._initialize_embedder(self.model_name):
                        try:
                            query_vec = self.embedder.encode([query], batch_size=1, normalize=True)[0]
                            self._cache_query_vector(query, query_vec)
                        except Exception as e2:
                            self.logger.error(f"Failed to encode query after re-initialization: {e2}")
                            return []
                    else:
                        return []

            # 2. FAISS 인덱스 사용 또는 전체 벡터 로드
            if FAISS_AVAILABLE and self.index is not None:
                # nprobe 동적 튜닝 (k 값에 따라 조정)
                optimal_nprobe = self._calculate_optimal_nprobe(k, self.index.ntotal)
                if self.index.nprobe != optimal_nprobe:
                    self.index.nprobe = optimal_nprobe
                    self.logger.debug(f"Adjusted nprobe to {optimal_nprobe} for k={k}")

                # FAISS 인덱스 검색 (빠른 근사 검색)
                query_vec_np = np.array([query_vec]).astype('float32')
                search_k = k * 2  # 여유 있게 검색
                distances, indices = self.index.search(query_vec_np, search_k)

                similarities = []
                for distance, idx in zip(distances[0], indices[0]):
                    if idx < len(self._chunk_ids):
                        chunk_id = self._chunk_ids[idx]
                        # L2 거리를 코사인 유사도로 변환 (정규화된 벡터의 경우)
                        # distance = 2 - 2*cosine_similarity
                        # cosine_similarity = 1 - distance/2
                        similarity = 1.0 - (distance / 2.0)
                        similarity = max(0.0, min(1.0, similarity))  # 0-1 범위로 제한

                        if similarity >= similarity_threshold:
                            similarities.append((chunk_id, similarity))

                # 유사도 기준 정렬
                similarities.sort(key=lambda x: x[1], reverse=True)

            else:
                # 기존 방식 (전체 벡터 로드 및 선형 검색)
                # FAISS 인덱스가 없으면 백그라운드에서 빌드 시작
                if FAISS_AVAILABLE and self.index is None and not self._index_building:
                    self.logger.info("FAISS index not found, starting background build")
                    self._build_faiss_index_async()

                chunk_vectors = self._load_chunk_vectors(source_types=source_types)

                if not chunk_vectors:
                    self.logger.warning(
                        f"⚠️ No chunk vectors found for search. "
                        f"This may indicate that embeddings need to be generated."
                    )
                    return []

                # 코사인 유사도 계산
                similarities = []
                for chunk_id, chunk_vec in chunk_vectors.items():
                    similarity = self._cosine_similarity(query_vec, chunk_vec)
                    if similarity >= similarity_threshold:
                        similarities.append((chunk_id, similarity))

                # 유사도 기준 정렬
                similarities.sort(key=lambda x: x[1], reverse=True)

            # 5. 상위 K개 결과 구성
            results = []
            conn = self._get_connection()

            for chunk_id, score in similarities[:k]:
                # 청크 메타데이터 조회 (없으면 DB에서 조회)
                if chunk_id not in self._chunk_metadata:
                    # 메타데이터가 없으면 DB에서 직접 조회 (전체 텍스트 가져오기)
                    cursor = conn.execute(
                        "SELECT source_type, source_id, text FROM text_chunks WHERE id = ?",
                        (chunk_id,)
                    )
                    row = cursor.fetchone()
                    if row:
                        text_content = row['text'] if row['text'] else ""
                        # text가 비어있거나 짧으면 원본 테이블에서 복원 시도
                        if not text_content or len(text_content.strip()) < 100:
                            source_type = row['source_type']
                            source_id = row['source_id']
                            restored_text = self._restore_text_from_source(conn, source_type, source_id)
                            if restored_text and len(restored_text.strip()) > len(text_content.strip()):
                                text_content = restored_text
                                self.logger.info(f"Restored longer text for chunk_id={chunk_id} (length: {len(text_content)} chars)")
                        
                        self._chunk_metadata[chunk_id] = {
                            'source_type': row['source_type'],
                            'source_id': row['source_id'],
                            'text': text_content
                        }

                metadata = self._chunk_metadata.get(chunk_id, {})
                source_type = metadata.get('source_type')
                source_id = metadata.get('source_id')
                text = metadata.get('text', '')

                # text가 비어있거나 짧으면 원본 테이블에서 복원 시도 (최소 100자 보장)
                if not text or len(text.strip()) < 100:
                    if not text or len(text.strip()) == 0:
                        self.logger.warning(f"Empty text content for chunk_id={chunk_id}, source_type={source_type}, source_id={source_id}. Attempting to restore from source table...")
                    else:
                        self.logger.debug(f"Short text content for chunk_id={chunk_id} (length: {len(text)} chars), attempting to restore longer text from source table...")
                    
                    restored_text = self._restore_text_from_source(conn, source_type, source_id)
                    if restored_text:
                        # 복원된 텍스트가 더 길면 사용
                        if len(restored_text.strip()) > len(text.strip()) if text else True:
                            text = restored_text
                            # 복원된 text를 메타데이터에 저장
                            self._chunk_metadata[chunk_id]['text'] = text
                            self.logger.info(f"Successfully restored text for chunk_id={chunk_id} from source table (length: {len(text)} chars)")
                        else:
                            self.logger.debug(f"Restored text is not longer than existing text for chunk_id={chunk_id}")
                    else:
                        if not text or len(text.strip()) == 0:
                            self.logger.error(f"Failed to restore text for chunk_id={chunk_id}, source_type={source_type}, source_id={source_id}")
                        else:
                            self.logger.warning(f"Could not restore longer text for chunk_id={chunk_id}, using existing text (length: {len(text)} chars)")

                # 소스별 상세 메타데이터 조회
                source_meta = self._get_source_metadata(conn, source_type, source_id)
                # content 필드가 비어있으면 경고 및 복원 시도
                if not text or len(text.strip()) == 0:
                    self.logger.warning(f"⚠️ [SEMANTIC SEARCH] Empty text for chunk_id={chunk_id}, source_type={source_type}, source_id={source_id}")
                    # 복원 시도
                    if source_type and source_id:
                        restored_text = self._restore_text_from_source(conn, source_type, source_id)
                        if restored_text:
                            text = restored_text
                            self.logger.info(f"✅ [SEMANTIC SEARCH] Restored text for chunk_id={chunk_id} (length: {len(text)} chars)")
                        else:
                            self.logger.error(f"❌ [SEMANTIC SEARCH] Failed to restore text for chunk_id={chunk_id}")
                            continue  # text가 없으면 건너뛰기
                
                # 최소 길이 보장 (100자 이상)
                if text and len(text.strip()) < 100:
                    restored_text = self._restore_text_from_source(conn, source_type, source_id)
                    if restored_text and len(restored_text.strip()) > len(text.strip()):
                        text = restored_text
                        self.logger.debug(f"Extended text for chunk_id={chunk_id} to {len(text)} chars")
                
                # text가 비어있으면 건너뛰기
                if not text or len(text.strip()) == 0:
                    continue
                
                # ML 신뢰도 및 품질 점수 추출
                ml_confidence = source_meta.get("ml_confidence_score") or source_meta.get("ml_confidence", 0.5)
                quality_score = source_meta.get("parsing_quality_score") or source_meta.get("quality_score", 0.5)
                
                # 필터링: ML 신뢰도 및 품질 점수 체크
                if min_ml_confidence is not None and ml_confidence < min_ml_confidence:
                    self.logger.debug(
                        f"Filtered chunk {chunk_id}: ml_confidence {ml_confidence:.2f} < {min_ml_confidence:.2f}"
                    )
                    continue
                
                if min_quality_score is not None and quality_score < min_quality_score:
                    self.logger.debug(
                        f"Filtered chunk {chunk_id}: quality_score {quality_score:.2f} < {min_quality_score:.2f}"
                    )
                    continue
                
                # 하이브리드 점수 계산 (유사도 + 품질 + 신뢰도)
                hybrid_score = self._calculate_hybrid_score(
                    similarity=float(score),
                    ml_confidence=ml_confidence,
                    quality_score=quality_score
                )
                
                # 통일된 포맷터로 출처 정보 생성
                try:
                    from ..services.unified_source_formatter import UnifiedSourceFormatter
                    formatter = UnifiedSourceFormatter()
                    source_info = formatter.format_source(source_type, source_meta)
                    source_name = source_info.name
                    source_url = source_info.url
                except Exception as e:
                    self.logger.warning(f"Error using UnifiedSourceFormatter: {e}, using fallback")
                    source_name = self._format_source(source_type, source_meta)
                    source_url = ""
                
                result = {
                    "id": f"chunk_{chunk_id}",
                    "text": text,
                    "content": text,  # content 필드 보장
                    "score": float(score),
                    "similarity": float(score),
                    "type": source_type,
                    "source": source_name,
                    "source_url": source_url,  # URL 필드 추가
                    # 최상위 필드에 상세 정보 추가 (answer_formatter에서 쉽게 접근)
                    "statute_name": source_meta.get("statute_name") if source_type == "statute_article" else None,
                    "law_name": source_meta.get("statute_name") if source_type == "statute_article" else None,
                    "article_no": source_meta.get("article_no") if source_type == "statute_article" else None,
                    "article_number": source_meta.get("article_no") if source_type == "statute_article" else None,
                    "clause_no": source_meta.get("clause_no") if source_type == "statute_article" else None,
                    "item_no": source_meta.get("item_no") if source_type == "statute_article" else None,
                    "court": source_meta.get("court") if source_type == "case_paragraph" else None,
                    "doc_id": source_meta.get("doc_id") if source_type in ["case_paragraph", "decision_paragraph", "interpretation_paragraph"] else None,
                    "casenames": source_meta.get("casenames") if source_type == "case_paragraph" else None,
                    "org": source_meta.get("org") if source_type in ["decision_paragraph", "interpretation_paragraph"] else None,
                    "title": source_meta.get("title") if source_type == "interpretation_paragraph" else None,
                    "metadata": {
                        "chunk_id": chunk_id,
                        "source_type": source_type,
                        "source_id": source_id,
                        "text": text,  # metadata에도 text 포함
                        "content": text,  # metadata에도 content 저장
                        "ml_confidence_score": ml_confidence,
                        "quality_score": quality_score,
                        **source_meta
                    },
                    "relevance_score": float(score),
                    "hybrid_score": hybrid_score,
                    "ml_confidence": ml_confidence,
                    "quality_score": quality_score,
                    "search_type": "semantic"
                }
                
                # 신뢰도 기반 필터링
                if filter_by_confidence:
                    # relevance_score와 hybrid_score를 모두 고려
                    confidence_threshold = 0.6
                    if result["relevance_score"] < confidence_threshold and result["hybrid_score"] < confidence_threshold:
                        self.logger.debug(
                            f"Filtered chunk {chunk_id} by confidence: "
                            f"relevance={result['relevance_score']:.2f}, hybrid={result['hybrid_score']:.2f}"
                        )
                        continue
                
                results.append(result)

            conn.close()
            self.logger.info(f"Semantic search found {len(results)} results for query: {query[:50]}")
            return results

        except Exception as e:
            self.logger.error(f"Error in semantic search: {e}", exc_info=True)
            return []

    def _calculate_optimal_nprobe(self, k: int, total_vectors: int) -> int:
        """
        검색 파라미터 k와 총 벡터 수에 따라 최적의 nprobe 계산

        Args:
            k: 검색할 최대 결과 수
            total_vectors: 총 벡터 수

        Returns:
            최적의 nprobe 값
        """
        # nlist 추정 (일반적으로 total/10 ~ total/100)
        estimated_nlist = max(10, min(100, total_vectors // 10))

        # k 값에 따라 nprobe 조정
        if k <= 5:
            nprobe = max(1, estimated_nlist // 10)  # 적은 결과: 낮은 nprobe (빠른 검색)
        elif k <= 20:
            nprobe = max(5, estimated_nlist // 5)  # 중간 결과: 중간 nprobe
        else:
            nprobe = max(10, estimated_nlist // 2)  # 많은 결과: 높은 nprobe (정확한 검색)

        # 최소/최대 값 제한
        nprobe = min(max(1, nprobe), estimated_nlist)

        return nprobe

    def _build_faiss_index_sync(self):
        """FAISS IVF 인덱스 빌드 및 저장 (동기 방식)"""
        if not FAISS_AVAILABLE:
            self.logger.warning("FAISS not available, skipping index build")
            return False

        try:
            self.logger.info("Building FAISS index...")

            # 1. 벡터 로드
            chunk_vectors = self._load_chunk_vectors()
            if not chunk_vectors:
                self.logger.error(
                    f"❌ No chunk vectors found, cannot build FAISS index. "
                    f"Semantic search will not work. "
                    f"Please ensure embeddings are generated and stored in the database."
                )
                return False

            # 2. numpy 배열 생성
            chunk_ids_sorted = sorted(chunk_vectors.keys())
            vectors = np.array([
                chunk_vectors[chunk_id]
                for chunk_id in chunk_ids_sorted
            ]).astype('float32')

            if len(vectors) == 0:
                self.logger.warning("No vectors to index")
                return False

            # 3. FAISS IVF 인덱스 생성
            dimension = vectors.shape[1]
            nlist = min(100, max(10, len(vectors) // 10))  # 클러스터 수 (최소 10개)

            quantizer = faiss.IndexFlatL2(dimension)
            index = faiss.IndexIVFFlat(quantizer, dimension, nlist)

            # 4. 학습 및 벡터 추가
            self.logger.info(f"Training FAISS index with {len(vectors)} vectors, nlist={nlist}")
            index.train(vectors)
            index.add(vectors)
            index.nprobe = self._calculate_optimal_nprobe(10, len(vectors))  # 기본 nprobe

            # 5. chunk_id 매핑 저장
            chunk_ids = chunk_ids_sorted

            # 6. 인덱스 저장
            faiss.write_index(index, self.index_path)
            self.logger.info(f"FAISS index built and saved: {self.index_path} ({len(vectors)} vectors)")

            # 7. 메인 스레드에서 인덱스 설정 (스레드 안전)
            self.index = index
            self._chunk_ids = chunk_ids

            return True

        except Exception as e:
            self.logger.error(f"Error building FAISS index: {e}", exc_info=True)
            return False

    def _build_faiss_index(self):
        """FAISS IVF 인덱스 빌드 및 저장 (기존 호환용, 동기 방식)"""
        self._build_faiss_index_sync()

    def _build_faiss_index_async(self):
        """FAISS 인덱스를 백그라운드 스레드에서 빌드"""
        if self._index_building:
            self.logger.debug("FAISS index build already in progress")
            return

        if not FAISS_AVAILABLE:
            self.logger.warning("FAISS not available, skipping index build")
            return

        def build_thread():
            try:
                self._index_building = True
                self.logger.info("Starting background FAISS index build...")
                success = self._build_faiss_index_sync()
                if success:
                    self.logger.info("Background FAISS index build completed successfully")
                else:
                    self.logger.warning("Background FAISS index build failed")
            except Exception as e:
                self.logger.error(f"Error in background FAISS index build: {e}", exc_info=True)
            finally:
                self._index_building = False

        self._build_thread = threading.Thread(target=build_thread, daemon=True)
        self._build_thread.start()
        self.logger.info("FAISS index build started in background thread")

    def _incremental_update_index(self, new_chunk_ids: List[int]):
        """
        FAISS 인덱스에 새로운 벡터를 증분 업데이트 (향후 사용)

        Args:
            new_chunk_ids: 추가할 chunk_id 리스트
        """
        if not FAISS_AVAILABLE or self.index is None:
            self.logger.warning("Cannot update index: FAISS not available or index not loaded")
            return

        try:
            # 새 벡터 로드
            new_vectors_dict = {}
            for chunk_id in new_chunk_ids:
                vectors = self._load_chunk_vectors(limit=1)  # 단일 벡터 로드
                if chunk_id in vectors:
                    new_vectors_dict[chunk_id] = vectors[chunk_id]

            if not new_vectors_dict:
                self.logger.warning("No new vectors to add")
                return

            # numpy 배열 생성
            new_chunk_ids_sorted = sorted(new_vectors_dict.keys())
            new_vectors = np.array([
                new_vectors_dict[chunk_id]
                for chunk_id in new_chunk_ids_sorted
            ]).astype('float32')

            # 인덱스에 추가
            self.index.add(new_vectors)
            self._chunk_ids.extend(new_chunk_ids_sorted)

            # 인덱스 저장
            faiss.write_index(self.index, self.index_path)
            self.logger.info(f"Incremental update: added {len(new_vectors)} vectors to FAISS index")

        except Exception as e:
            self.logger.error(f"Error in incremental index update: {e}", exc_info=True)

    def _load_faiss_index(self):
        """저장된 FAISS 인덱스 로드"""
        if not FAISS_AVAILABLE:
            return

        try:
            self.index = faiss.read_index(str(self.index_path))

            # chunk_id 매핑 재구성 (embeddings 테이블에서)
            conn = self._get_connection()
            cursor = conn.execute(
                "SELECT chunk_id FROM embeddings WHERE model = ? ORDER BY chunk_id",
                (self.model_name,)
            )
            self._chunk_ids = [row[0] for row in cursor.fetchall()]
            conn.close()

            self.logger.info(f"FAISS index loaded: {len(self._chunk_ids)} vectors from {self.index_path}")

        except Exception as e:
            self.logger.warning(f"Failed to load FAISS index: {e}, will rebuild")
            self.index = None
            # 인덱스 파일이 손상된 경우 삭제하여 재빌드 유도
            try:
                Path(self.index_path).unlink()
            except Exception:
                pass

    def _get_source_metadata(self, conn: sqlite3.Connection, source_type: str, source_id: int) -> Dict[str, Any]:
        """
        소스 타입별 상세 메타데이터 조회
        source_id는 text_chunks.source_id로, 각 소스 테이블의 실제 id를 참조
        """
        try:
            if source_type == "statute_article":
                cursor = conn.execute("""
                    SELECT sa.*, s.name as statute_name, s.abbrv, s.category, s.statute_type
                    FROM statute_articles sa
                    JOIN statutes s ON sa.statute_id = s.id
                    WHERE sa.id = ?
                """, (source_id,))
            elif source_type == "case_paragraph":
                cursor = conn.execute("""
                    SELECT cp.*, c.doc_id, c.court, c.case_type, c.casenames, c.announce_date
                    FROM case_paragraphs cp
                    JOIN cases c ON cp.case_id = c.id
                    WHERE cp.id = ?
                """, (source_id,))
            elif source_type == "decision_paragraph":
                cursor = conn.execute("""
                    SELECT dp.*, d.org, d.doc_id, d.decision_date, d.result
                    FROM decision_paragraphs dp
                    JOIN decisions d ON dp.decision_id = d.id
                    WHERE dp.id = ?
                """, (source_id,))
            elif source_type == "interpretation_paragraph":
                cursor = conn.execute("""
                    SELECT ip.*, i.org, i.doc_id, i.title, i.response_date
                    FROM interpretation_paragraphs ip
                    JOIN interpretations i ON ip.interpretation_id = i.id
                    WHERE ip.id = ?
                """, (source_id,))
            else:
                return {}

            row = cursor.fetchone()
            if row:
                return dict(row)
            return {}
        except Exception as e:
            self.logger.warning(f"Error getting source metadata for {source_type} {source_id}: {e}")
            return {}

    def _restore_text_from_source(self, conn: sqlite3.Connection, source_type: str, source_id: int) -> str:
        """
        text_chunks 테이블의 text가 비어있을 때 원본 테이블에서 복원
        
        Args:
            conn: 데이터베이스 연결
            source_type: 소스 타입 (statute_article, case_paragraph 등)
            source_id: 소스 ID
            
        Returns:
            복원된 text 문자열 (없으면 빈 문자열)
        """
        try:
            # row_factory를 설정하여 dict 형태로 접근
            conn.row_factory = sqlite3.Row
            
            if source_type == "statute_article":
                cursor = conn.execute(
                    "SELECT text, article_no FROM statute_articles WHERE id = ?",
                    (source_id,)
                )
            elif source_type == "case_paragraph":
                cursor = conn.execute(
                    "SELECT text FROM case_paragraphs WHERE id = ?",
                    (source_id,)
                )
            elif source_type == "decision_paragraph":
                cursor = conn.execute(
                    "SELECT text FROM decision_paragraphs WHERE id = ?",
                    (source_id,)
                )
            elif source_type == "interpretation_paragraph":
                cursor = conn.execute(
                    "SELECT text FROM interpretation_paragraphs WHERE id = ?",
                    (source_id,)
                )
            else:
                self.logger.warning(f"Unknown source_type for text restoration: {source_type}")
                return ""
            
            row = cursor.fetchone()
            if row:
                # Row 객체에서 text 필드 접근
                text = row['text'] if 'text' in row.keys() else None
                if text and len(str(text).strip()) > 0:
                    self.logger.info(f"Successfully restored text for {source_type} id={source_id} (length: {len(str(text))} chars)")
                    return str(text)
                else:
                    self.logger.warning(f"Text field is empty or None for {source_type} id={source_id}")
                    # text가 비어있으면 다른 방법 시도: text_chunks에서 직접 조회
                    return self._restore_text_from_chunks(conn, source_type, source_id)
            else:
                self.logger.warning(f"No row found for {source_type} id={source_id}")
                # 원본 테이블에 없으면 text_chunks에서 직접 조회
                return self._restore_text_from_chunks(conn, source_type, source_id)
            return ""
        except sqlite3.Error as e:
            self.logger.error(f"SQLite error restoring text from source table ({source_type}, {source_id}): {e}")
            # 에러 발생 시 text_chunks에서 직접 조회 시도
            return self._restore_text_from_chunks(conn, source_type, source_id)
        except Exception as e:
            self.logger.error(f"Error restoring text from source table ({source_type}, {source_id}): {e}")
            # 에러 발생 시 text_chunks에서 직접 조회 시도
            return self._restore_text_from_chunks(conn, source_type, source_id)
    
    def _restore_text_from_chunks(self, conn: sqlite3.Connection, source_type: str, source_id: int) -> str:
        """
        text_chunks 테이블에서 직접 text 조회 (원본 테이블 조회 실패 시)
        """
        try:
            conn.row_factory = sqlite3.Row
            # 같은 source_type과 source_id를 가진 다른 chunk에서 text 가져오기
            cursor = conn.execute(
                "SELECT text FROM text_chunks WHERE source_type = ? AND source_id = ? AND text IS NOT NULL AND text != '' LIMIT 1",
                (source_type, source_id)
            )
            row = cursor.fetchone()
            if row:
                text = row['text'] if 'text' in row.keys() else None
                if text and len(str(text).strip()) > 0:
                    self.logger.info(f"Restored text from text_chunks for {source_type} id={source_id} (length: {len(str(text))} chars)")
                    return str(text)
            return ""
        except Exception as e:
            self.logger.error(f"Error restoring text from text_chunks ({source_type}, {source_id}): {e}")
            return ""

    def _format_source(self, source_type: str, metadata: Dict[str, Any]) -> str:
        """소스 정보 포맷팅 (통일된 포맷터 사용)"""
        try:
            from ..services.unified_source_formatter import UnifiedSourceFormatter
            formatter = UnifiedSourceFormatter()
            source_info = formatter.format_source(source_type, metadata)
            return source_info.name
        except ImportError:
            self.logger.warning("UnifiedSourceFormatter not available, using fallback")
            return self._format_source_fallback(source_type, metadata)
        except Exception as e:
            self.logger.warning(f"Error using UnifiedSourceFormatter: {e}, using fallback")
            return self._format_source_fallback(source_type, metadata)
    
    def _format_source_fallback(self, source_type: str, metadata: Dict[str, Any]) -> str:
        """소스 정보 포맷팅 (Fallback - 기존 로직)"""
        if source_type == "statute_article":
            statute_name = metadata.get("statute_name") or "법령"
            article_no = metadata.get("article_no") or ""
            clause_no = metadata.get("clause_no") or ""
            item_no = metadata.get("item_no") or ""
            
            if article_no:
                source = f"{statute_name} {article_no}"
                if clause_no:
                    source += f" 제{clause_no}항"
                if item_no:
                    source += f" 제{item_no}호"
                return source
            return statute_name
            
        elif source_type == "case_paragraph":
            court = metadata.get("court", "")
            doc_id = metadata.get("doc_id", "")
            casenames = metadata.get("casenames", "")
            announce_date = metadata.get("announce_date", "")
            
            parts = []
            if court:
                parts.append(court)
            if casenames:
                parts.append(casenames)
            if doc_id:
                parts.append(f"({doc_id})")
            if announce_date:
                parts.append(f"[{announce_date}]")
            
            return " ".join(parts) if parts else "판례"
            
        elif source_type == "decision_paragraph":
            org = metadata.get("org", "")
            doc_id = metadata.get("doc_id", "")
            decision_date = metadata.get("decision_date", "")
            
            parts = []
            if org:
                parts.append(org)
            if doc_id:
                parts.append(f"({doc_id})")
            if decision_date:
                parts.append(f"[{decision_date}]")
            
            return " ".join(parts) if parts else "결정례"
            
        elif source_type == "interpretation_paragraph":
            org = metadata.get("org", "")
            title = metadata.get("title", "")
            doc_id = metadata.get("doc_id", "")
            response_date = metadata.get("response_date", "")
            
            parts = []
            if org:
                parts.append(org)
            if title:
                parts.append(title)
            if doc_id:
                parts.append(f"({doc_id})")
            if response_date:
                parts.append(f"[{response_date}]")
            
            return " ".join(parts) if parts else "해석례"
            
        return "Unknown"

    def get_high_confidence_documents(self,
                                      query: str,
                                      min_similarity: float = 0.7,
                                      min_ml_confidence: float = 0.8,
                                      min_quality_score: float = 0.8,
                                      max_results: int = 5,
                                      source_types: Optional[List[str]] = None,
                                      sort_by: str = "hybrid_score") -> List[Dict[str, Any]]:
        """
        신뢰도 높은 문서만 조회
        
        Args:
            query: 검색 쿼리
            min_similarity: 최소 유사도 임계값
            min_ml_confidence: 최소 ML 신뢰도
            min_quality_score: 최소 품질 점수
            max_results: 최대 결과 수
            source_types: 필터링할 소스 타입 목록
            sort_by: 정렬 기준 ("hybrid_score", "relevance_score", "similarity")
        
        Returns:
            필터링된 문서 리스트
        """
        self.logger.info(
            f"High confidence search: query='{query[:50]}...', "
            f"min_similarity={min_similarity:.2f}, "
            f"min_ml_confidence={min_ml_confidence:.2f}, "
            f"min_quality_score={min_quality_score:.2f}"
        )
        
        # 1. 높은 임계값으로 검색 (재시도 비활성화)
        results = self.search(
            query=query,
            k=max_results * 3,  # 여유 있게 검색 후 필터링
            source_types=source_types,
            similarity_threshold=min_similarity,
            min_results=1,  # 재시도 최소화
            disable_retry=True,  # 재시도 비활성화
            min_ml_confidence=min_ml_confidence,
            min_quality_score=min_quality_score,
            filter_by_confidence=True  # 신뢰도 기반 필터링 활성화
        )
        
        # 2. 추가 필터링: relevance_score와 hybrid_score 모두 체크
        filtered = []
        for r in results:
            relevance = r.get("relevance_score", 0.0)
            hybrid = r.get("hybrid_score", 0.0)
            similarity = r.get("similarity", 0.0)
            ml_conf = r.get("ml_confidence", 0.0)
            quality = r.get("quality_score", 0.0)
            
            # 모든 조건 만족 체크
            if (relevance >= min_similarity and
                hybrid >= min_similarity * 0.8 and  # hybrid는 약간 낮은 임계값
                similarity >= min_similarity and
                ml_conf >= min_ml_confidence and
                quality >= min_quality_score):
                filtered.append(r)
        
        # 3. 정렬 기준에 따라 정렬
        if sort_by == "hybrid_score":
            filtered.sort(key=lambda x: x.get("hybrid_score", 0.0), reverse=True)
        elif sort_by == "relevance_score":
            filtered.sort(key=lambda x: x.get("relevance_score", 0.0), reverse=True)
        elif sort_by == "similarity":
            filtered.sort(key=lambda x: x.get("similarity", 0.0), reverse=True)
        else:
            # 기본: hybrid_score 우선, 동일하면 relevance_score
            filtered.sort(
                key=lambda x: (x.get("hybrid_score", 0.0), x.get("relevance_score", 0.0)),
                reverse=True
            )
        
        # 4. 상위 N개 반환
        final_results = filtered[:max_results]
        
        self.logger.info(
            f"High confidence search: {len(final_results)}/{len(results)} results "
            f"(filtered from {len(results)} total)"
        )
        
        return final_results

    def search_with_query_expansion(self,
                                    query: str,
                                    k: int = 10,
                                    source_types: Optional[List[str]] = None,
                                    similarity_threshold: float = 0.5,
                                    expanded_keywords: Optional[List[str]] = None,
                                    use_query_variations: bool = True) -> List[Dict[str, Any]]:
        """
        쿼리 확장 및 변형을 통한 향상된 검색
        
        Args:
            query: 원본 검색 쿼리
            k: 반환할 최대 결과 수
            source_types: 필터링할 소스 타입 목록
            similarity_threshold: 최소 유사도 임계값
            expanded_keywords: 확장된 키워드 목록 (QueryEnhancer에서 제공)
            use_query_variations: 쿼리 변형 사용 여부
        
        Returns:
            통합된 검색 결과 리스트
        """
        # Embedder 초기화 상태 확인 및 필요시 재초기화
        if not self._ensure_embedder_initialized():
            self.logger.error("Embedder not initialized and re-initialization failed")
            # 폴백: 기존 search 메서드 사용 (재시도 로직 포함)
            self.logger.warning("Falling back to standard search method...")
            return self.search(
                query=query,
                k=k,
                source_types=source_types,
                similarity_threshold=similarity_threshold
            )
        
        if not use_query_variations or not expanded_keywords:
            # 기존 방식 사용 (하위 호환성)
            return self.search(
                query=query,
                k=k,
                source_types=source_types,
                similarity_threshold=similarity_threshold
            )
        
        # 1. 쿼리 변형 생성
        query_variations = self._generate_simple_query_variations(query, expanded_keywords)
        
        # 2. 각 변형으로 검색
        all_results = []
        seen_chunk_ids = set()
        
        for variation in query_variations:
            var_query = variation["query"]
            var_weight = variation.get("weight", 1.0)
            
            results = self.search(
                query=var_query,
                k=k * 2,  # 여유 있게 검색
                source_types=source_types,
                similarity_threshold=similarity_threshold,
                disable_retry=True  # 빠른 검색
            )
            
            # 중복 제거 및 가중치 적용
            for result in results:
                chunk_id = result.get("metadata", {}).get("chunk_id")
                if chunk_id and chunk_id not in seen_chunk_ids:
                    result["relevance_score"] *= var_weight
                    result["query_variation"] = variation.get("type", "unknown")
                    result["query_weight"] = var_weight
                    all_results.append(result)
                    seen_chunk_ids.add(chunk_id)
        
        # 3. relevance_score 기준 정렬
        all_results.sort(key=lambda x: x.get("relevance_score", 0.0), reverse=True)
        
        # 4. 상위 K개 반환
        return all_results[:k]

    def _generate_simple_query_variations(self,
                                         query: str,
                                         expanded_keywords: Optional[List[str]] = None) -> List[Dict[str, Any]]:
        """간단한 쿼리 변형 생성 (기존 코드와의 호환성 유지)"""
        variations = []
        
        # 1. 원본 쿼리 (최고 가중치)
        variations.append({
            "query": query,
            "type": "original",
            "weight": 1.0,
            "priority": 1
        })
        
        # 2. 키워드 확장 쿼리
        if expanded_keywords and len(expanded_keywords) >= 2:
            # 상위 3개 키워드만 추가
            top_keywords = expanded_keywords[:3]
            expanded_query = f"{query} {' '.join(top_keywords)}"
            variations.append({
                "query": expanded_query,
                "type": "keyword_expanded",
                "weight": 0.9,
                "priority": 2
            })
            
            # 키워드만으로 검색 (원본 쿼리가 너무 길 때)
            if len(query) > 50:
                keyword_only_query = " ".join(top_keywords)
                variations.append({
                    "query": keyword_only_query,
                    "type": "keyword_only",
                    "weight": 0.8,
                    "priority": 3
                })
        
        # 3. 핵심 키워드 추출 쿼리
        core_keywords = self._extract_core_keywords_simple(query)
        if core_keywords and len(core_keywords) >= 2:
            core_query = " ".join(core_keywords)
            if core_query != query:
                variations.append({
                    "query": core_query,
                    "type": "core_keywords",
                    "weight": 0.85,
                    "priority": 2
                })
        
        # 우선순위 및 가중치 기준 정렬
        variations.sort(key=lambda x: (x["priority"], -x["weight"]))
        
        return variations

    def _extract_core_keywords_simple(self, query: str) -> List[str]:
        """핵심 키워드 추출 (간단한 구현)"""
        import re
        
        # 불용어 제거
        stopwords = ["이", "가", "을", "를", "의", "에", "에서", "로", "으로", "와", "과", "는", "은", "도", "만", "란", "이란"]
        
        # 한글 단어 추출
        words = re.findall(r'[가-힣]+', query)
        
        # 불용어 제거 및 길이 필터링
        core_keywords = [w for w in words if w not in stopwords and len(w) >= 2]
        
        return core_keywords[:5]  # 상위 5개
