# -*- coding: utf-8 -*-
"""
Semantic Search Engine V2
lawfirm_v2.db의 embeddings 테이블을 사용한 벡터 검색 엔진
"""

import logging
import sqlite3
import sys
import threading
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

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
                    original_env['TRANSFORMERS_NO_ADVISORY_WARNINGS'] = os.environ.get('TRANSFORMERS_NO_ADVISORY_WARNINGS', None)
                    original_env['HF_DEVICE_MAP'] = os.environ.get('HF_DEVICE_MAP', None)
                    
                    # meta device 방지를 위한 환경 변수 설정
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
                            "dtype": torch.float32,  # 명시적 dtype 설정 (torch_dtype deprecated)
                            "use_safetensors": False,  # safetensors 사용 안 함 (meta tensor 문제 방지)
                            "ignore_mismatched_sizes": True,  # 크기 불일치 무시
                            "trust_remote_code": True,  # 원격 코드 신뢰
                            "local_files_only": False,  # 로컬 파일만 사용 안 함
                        }
                    )
                    
                    # 환경 변수 복원
                    try:
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
                                dtype=torch.float32,  # dtype 사용 (torch_dtype deprecated)
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
                                    os.environ['TRANSFORMERS_NO_ADVISORY_WARNINGS'] = '1'
                                    if 'HF_DEVICE_MAP' in os.environ:
                                        del os.environ['HF_DEVICE_MAP']
                                    
                                    # 모델 재로딩
                                    model = AutoModel.from_pretrained(
                                        model_name,
                                        dtype=torch.float32,  # torch_dtype deprecated
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
                        fallback_original_env['TRANSFORMERS_NO_ADVISORY_WARNINGS'] = os.environ.get('TRANSFORMERS_NO_ADVISORY_WARNINGS', None)
                        fallback_original_env['HF_DEVICE_MAP'] = os.environ.get('HF_DEVICE_MAP', None)
                        
                        # meta device 방지를 위한 환경 변수 설정
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
                                "dtype": torch.float32,  # 명시적 dtype 설정 (torch_dtype deprecated)
                                "use_safetensors": False,  # safetensors 사용 안 함
                                "trust_remote_code": True,  # 원격 코드 신뢰
                                "local_files_only": False,  # 로컬 파일만 사용 안 함
                            }
                        )
                        
                        # 환경 변수 복원
                        try:
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
                 model_name: Optional[str] = None,
                 mlflow_run_id: Optional[str] = None,
                 use_mlflow_index: bool = False):
        """
        검색 엔진 초기화

        Args:
            db_path: lawfirm_v2.db 경로 (None이면 환경변수 DATABASE_PATH 사용)
            model_name: 임베딩 모델명 (None이면 데이터베이스에서 자동 감지)
            mlflow_run_id: MLflow run ID (선택, None이면 프로덕션 run 자동 조회)
            use_mlflow_index: MLflow 인덱스 사용 여부
        """
        if db_path is None:
            from core.utils.config import Config
            config = Config()
            db_path = config.database_path
        self.db_path = db_path
        self.logger = logging.getLogger(__name__)
        
        # 설정에서 MLflow 인덱스 사용 여부 확인
        if not use_mlflow_index:
            try:
                from core.utils.config import Config
                config = Config()
                use_mlflow_index = config.use_mlflow_index if hasattr(config, 'use_mlflow_index') else False
                if use_mlflow_index:
                    mlflow_run_id = mlflow_run_id or (config.mlflow_run_id if hasattr(config, 'mlflow_run_id') else None)
            except Exception as e:
                self.logger.debug(f"Could not load config for MLflow index settings: {e}")
        
        self.use_mlflow_index = use_mlflow_index
        self.mlflow_run_id = mlflow_run_id

        # 모델명이 제공되지 않으면 데이터베이스에서 자동 감지
        if model_name is None:
            model_name = self._detect_model_from_database()
            if model_name is None:
                # 감지 실패 시 문서 임베딩 기준 모델 사용
                model_name = "jhgan/ko-sroberta-multitask"
                self.logger.warning(f"Could not detect model from database, using default: {model_name}")

        self.model_name = model_name

        # FAISS 인덱스 관련 속성
        # 기본 경로: data/embeddings/ml_enhanced_ko_sroberta_precedents/ml_enhanced_faiss_index.faiss
        # 여러 경로 시도 (프로젝트 루트 기준)
        possible_paths = [
            Path(db_path).parent.parent / "embeddings" / "ml_enhanced_ko_sroberta_precedents" / "ml_enhanced_faiss_index.faiss",
            Path("data") / "embeddings" / "ml_enhanced_ko_sroberta_precedents" / "ml_enhanced_faiss_index.faiss",
            Path(db_path).parent / "embeddings" / "ml_enhanced_ko_sroberta_precedents" / "ml_enhanced_faiss_index.faiss",
        ]
        
        legacy_index_path = Path(db_path).parent / f"{Path(db_path).stem}_faiss.index"
        
        # 새로 빌드된 인덱스를 우선 사용
        default_index_path = None
        for path in possible_paths:
            if path.exists():
                default_index_path = path
                break
        
        if default_index_path:
            self.index_path = str(default_index_path)
            self.logger.info(f"Using default FAISS index: {self.index_path}")
        elif legacy_index_path.exists():
            # 레거시 경로 (하위 호환성)
            self.index_path = str(legacy_index_path)
            self.logger.info(f"Using legacy FAISS index: {self.index_path}")
        else:
            # 인덱스가 없으면 기본 경로 설정 (나중에 빌드됨)
            self.index_path = str(possible_paths[0])
            self.logger.info(f"No FAISS index found, will use: {self.index_path}")
        
        self.index = None
        self._chunk_ids = []  # 인덱스와 chunk_id 매핑
        self._chunk_metadata = {}  # chunk_id -> metadata 매핑 (초기화)
        self._index_building = False  # 백그라운드 빌드 중 플래그
        self._build_thread = None  # 빌드 스레드
        self.current_faiss_version = None  # 현재 FAISS 버전 (MLflow 전환 시에도 호환성 유지)
        self.faiss_version_manager = None  # FAISS 버전 관리자 (MLflow 전환 시에도 호환성 유지)

        # 쿼리 벡터 캐싱 (LRU 캐시)
        self._query_vector_cache = {}  # query -> vector
        self._cache_max_size = 512  # 최대 캐시 크기 (128 → 512로 증가)
        
        # 메타데이터 캐싱 (성능 개선, TTL 지원)
        self._metadata_cache = {}  # key -> {'data': metadata, 'timestamp': time.time()}
        self._metadata_cache_max_size = 1000  # 최대 캐시 크기
        self._metadata_cache_ttl = 3600  # TTL: 1시간 (초 단위)
        self._metadata_cache_hits = 0  # 캐시 히트 수
        self._metadata_cache_misses = 0  # 캐시 미스 수
        self._metadata_cache_last_cleanup = time.time()  # 마지막 정리 시간
        self._metadata_cache_cleanup_interval = 300  # 정리 간격: 5분
        
        # MLflow 매니저 초기화
        self.mlflow_manager = None
        if self.use_mlflow_index:
            try:
                import sys
                import os
                # scripts/rag 경로 추가 (프로젝트 루트 기준)
                # 여러 경로 시도
                possible_paths = [
                    Path(__file__).resolve().parent.parent.parent.parent / "scripts" / "rag",
                    Path(__file__).parent.parent.parent.parent / "scripts" / "rag",
                    Path.cwd() / "scripts" / "rag",
                    Path.cwd().parent / "scripts" / "rag" if Path.cwd().name != "LawFirmAI" else Path.cwd() / "scripts" / "rag"
                ]
                
                mlflow_manager_imported = False
                for scripts_rag_path in possible_paths:
                    if scripts_rag_path.exists() and (scripts_rag_path / "mlflow_manager.py").exists():
                        if str(scripts_rag_path) not in sys.path:
                            sys.path.insert(0, str(scripts_rag_path))
                        try:
                            from mlflow_manager import MLflowFAISSManager
                            mlflow_manager_imported = True
                            self.logger.debug(f"Successfully imported MLflowFAISSManager from {scripts_rag_path}")
                            break
                        except ImportError:
                            continue
                
                if not mlflow_manager_imported:
                    raise ImportError(f"Could not import mlflow_manager from any of the paths: {[str(p) for p in possible_paths]}")
                from core.utils.config import Config
                config = Config()
                tracking_uri = config.mlflow_tracking_uri if hasattr(config, 'mlflow_tracking_uri') else None
                experiment_name = config.mlflow_experiment_name if hasattr(config, 'mlflow_experiment_name') else "faiss_index_versions"
                self.mlflow_manager = MLflowFAISSManager(
                    experiment_name=experiment_name,
                    tracking_uri=tracking_uri
                )
            except ImportError as e:
                self.logger.warning(f"MLflowFAISSManager not available: {e}")
                self.use_mlflow_index = False
            except Exception as e:
                self.logger.warning(f"Failed to initialize MLflow manager: {e}")
                self.use_mlflow_index = False
        
        # 성능 모니터링 초기화
        self.performance_monitor = None
        self.enable_performance_monitoring = False
        try:
            scripts_utils_path = Path(__file__).parent.parent.parent / "scripts" / "utils"
            if scripts_utils_path.exists():
                sys.path.insert(0, str(scripts_utils_path))
            from version_performance_monitor import VersionPerformanceMonitor
            performance_log_path = Path(db_path).parent / "performance_logs"
            self.performance_monitor = VersionPerformanceMonitor(str(performance_log_path))
            self.enable_performance_monitoring = True
        except ImportError:
            self.logger.debug("VersionPerformanceMonitor not available")

        # 임베딩 모델 로드
        self.embedder = None
        self.dim = None
        self._initialize_embedder(model_name)

        if not Path(db_path).exists():
            self.logger.warning(f"Database {db_path} not found")

        # FAISS 인덱스 로드 또는 빌드
        # 예외 처리를 강화하여 초기화 실패 시에도 서비스가 계속되도록 함
        if FAISS_AVAILABLE and self.embedder:
            try:
                if self.use_mlflow_index and self.mlflow_manager:
                    # MLflow 인덱스 사용 시 MLflow에서 로드
                    # 예외 발생 시에도 초기화는 계속되며, 첫 검색 시 재시도
                    try:
                        self._load_mlflow_index()
                    except Exception as e:
                        self.logger.warning(f"Failed to load MLflow index during initialization: {e}. Will retry on first search.")
                        self.index = None
                elif Path(self.index_path).exists():
                    # 내부 인덱스 사용 시 레거시 경로에서 로드
                    try:
                        self._load_faiss_index()
                    except Exception as e:
                        self.logger.warning(f"Failed to load FAISS index during initialization: {e}. Will retry on first search.")
                        self.index = None
                else:
                    self.logger.info("FAISS index not found, will build on first search")
            except Exception as e:
                self.logger.error(f"Unexpected error during index initialization: {e}", exc_info=True)
                self.index = None
    
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
        except RuntimeError as e:
            # PyTorch 스레드 설정 관련 오류는 재시도하지 않음 (이미 SentenceEmbedder에서 처리됨)
            error_msg = str(e).lower()
            if "cannot set number of interop threads" in error_msg:
                self.logger.warning(f"Thread configuration issue detected, but continuing: {e}")
                # 스레드 설정 문제는 무시하고 모델 로딩 재시도
                if retry_count < max_retries:
                    self.logger.info(f"Retrying embedder initialization (thread config issue)...")
                    import time
                    time.sleep(0.5)
                    return self._initialize_embedder(model_name, retry_count + 1, max_retries)
                else:
                    self.logger.error(f"Failed to initialize embedder after {max_retries + 1} attempts (thread config issue)")
                    self.embedder = None
                    self.dim = None
                    return False
            else:
                self.logger.error(f"Failed to load embedding model (attempt {retry_count + 1}/{max_retries + 1}): {e}")
                if retry_count < max_retries:
                    self.logger.info(f"Retrying embedder initialization...")
                    import time
                    time.sleep(0.5)
                    return self._initialize_embedder(model_name, retry_count + 1, max_retries)
                else:
                    self.logger.error(f"Failed to initialize embedder after {max_retries + 1} attempts")
                    self.embedder = None
                    self.dim = None
                    return False
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
    
    def is_available(self) -> bool:
        """
        Semantic Search 엔진 사용 가능 여부 확인
        
        Returns:
            사용 가능 여부
        """
        if not Path(self.db_path).exists():
            return False
        
        if not self._ensure_embedder_initialized():
            return False
        
        try:
            conn = self._get_connection()
            cursor = conn.cursor()
            cursor.execute("SELECT COUNT(*) FROM embeddings LIMIT 1")
            row = cursor.fetchone()
            conn.close()
            return row is not None and row[0] > 0
        except Exception as e:
            self.logger.debug(f"Error checking embeddings table: {e}")
            return False
    
    def diagnose(self) -> Dict[str, Any]:
        """
        Semantic Search 엔진 상태 진단
        
        Returns:
            진단 결과 딕셔너리
        """
        diagnosis = {
            "available": False,
            "db_exists": False,
            "embedder_initialized": False,
            "faiss_available": FAISS_AVAILABLE,
            "faiss_index_exists": False,
            "embeddings_count": 0,
            "model_name": self.model_name,
            "dim": self.dim,
            "issues": [],
            "recommendations": []
        }
        
        diagnosis["db_exists"] = Path(self.db_path).exists()
        if not diagnosis["db_exists"]:
            diagnosis["issues"].append(f"Database not found: {self.db_path}")
            diagnosis["recommendations"].append("Check database path configuration")
            return diagnosis
        
        diagnosis["embedder_initialized"] = self._ensure_embedder_initialized()
        if not diagnosis["embedder_initialized"]:
            diagnosis["issues"].append("Embedder not initialized")
            diagnosis["recommendations"].append("Check embedding model availability")
            return diagnosis
        
        diagnosis["faiss_index_exists"] = Path(self.index_path).exists()
        if not diagnosis["faiss_index_exists"]:
            diagnosis["issues"].append(f"FAISS index not found: {self.index_path}")
            diagnosis["recommendations"].append("FAISS index will be built on first search")
        
        try:
            conn = self._get_connection()
            cursor = conn.cursor()
            cursor.execute("SELECT COUNT(*) FROM embeddings")
            row = cursor.fetchone()
            conn.close()
            diagnosis["embeddings_count"] = row[0] if row else 0
            
            if diagnosis["embeddings_count"] == 0:
                diagnosis["issues"].append("No embeddings found in database")
                diagnosis["recommendations"].append("Run embedding generation script")
        except Exception as e:
            diagnosis["issues"].append(f"Error checking embeddings: {e}")
            diagnosis["recommendations"].append("Check database schema")
        
        diagnosis["available"] = (
            diagnosis["db_exists"] and
            diagnosis["embedder_initialized"] and
            diagnosis["embeddings_count"] > 0
        )
        
        return diagnosis

    def _get_active_embedding_version_id(self) -> Optional[int]:
        """
        활성 임베딩 버전 ID 조회

        Returns:
            활성 버전 ID 또는 None
        """
        try:
            # 외부 인덱스(MLflow)에서 로드된 embedding_version_id 우선 사용
            if hasattr(self, 'external_index_embedding_version_id') and self.external_index_embedding_version_id:
                self.logger.info(f"✅ Using embedding_version_id from external index: {self.external_index_embedding_version_id}")
                return self.external_index_embedding_version_id
            
            # use_external_index가 True인 경우에도 external_index_embedding_version_id가 없을 수 있음
            # 이 경우 DB에서 활성 버전을 찾되, 경고 로그 출력
            if hasattr(self, 'use_external_index') and self.use_external_index:
                self.logger.warning("⚠️  use_external_index=True but external_index_embedding_version_id is not set. Falling back to DB query.")
            
            if not Path(self.db_path).exists():
                self.logger.debug(f"Database file not found: {self.db_path}")
                return None

            conn = self._get_connection()
            cursor = conn.cursor()

            cursor.execute("""
                SELECT id, version_name, is_active
                FROM embedding_versions
                WHERE is_active = 1
                ORDER BY created_at DESC
                LIMIT 1
            """)
            row = cursor.fetchone()
            conn.close()

            if row:
                version_id = row['id']
                # sqlite3.Row는 dict처럼 .get()을 사용할 수 없으므로 직접 접근
                version_name = row['version_name'] if 'version_name' in row.keys() else f'v{version_id}'
                self.logger.info(f"✅ Active embedding version detected: ID={version_id}, name={version_name}")
                return version_id
            else:
                self.logger.warning("⚠️  No active embedding version found in database")
                return None

        except Exception as e:
            if "no such table" in str(e).lower():
                self.logger.debug(f"embedding_versions table not found: {e}")
            else:
                self.logger.warning(f"⚠️  Error getting active embedding version: {e}")
            return None

    def _get_version_chunk_count(self, version_id: int) -> int:
        """
        특정 버전의 청크 수 조회

        Args:
            version_id: 임베딩 버전 ID

        Returns:
            청크 수
        """
        try:
            if not Path(self.db_path).exists():
                return 0

            conn = self._get_connection()
            cursor = conn.cursor()

            cursor.execute("""
                SELECT COUNT(*) as count
                FROM text_chunks
                WHERE embedding_version_id = ?
            """, (version_id,))
            row = cursor.fetchone()
            conn.close()

            if row:
                return row['count']
            else:
                return 0

        except Exception as e:
            if "no such table" in str(e).lower():
                self.logger.debug(f"text_chunks table not found: {e}")
            else:
                self.logger.debug(f"Error getting version chunk count: {e}")
            return 0

    def _detect_model_from_database(self) -> Optional[str]:
        """
        데이터베이스에서 실제 사용된 임베딩 모델 감지
        활성 버전의 모델을 우선적으로 선택

        Returns:
            감지된 모델명 또는 None
        """
        try:
            if not Path(self.db_path).exists():
                self.logger.warning(f"Database {self.db_path} not found for model detection")
                return None

            conn = self._get_connection()
            cursor = conn.cursor()

            # 먼저 활성 버전의 모델 조회 시도
            active_version_id = self._get_active_embedding_version_id()
            if active_version_id:
                cursor.execute("""
                    SELECT DISTINCT e.model, COUNT(*) as count
                    FROM embeddings e
                    JOIN text_chunks tc ON e.chunk_id = tc.id
                    WHERE tc.embedding_version_id = ?
                    GROUP BY e.model
                    ORDER BY count DESC
                    LIMIT 1
                """, (active_version_id,))
                row = cursor.fetchone()
                if row:
                    detected_model = row['model']
                    self.logger.info(
                        f"Detected embedding model from active version (ID={active_version_id}): "
                        f"{detected_model} (count: {row['count']})"
                    )
                    conn.close()
                    return detected_model

            # 활성 버전에서 모델을 찾지 못한 경우 전체 데이터베이스에서 가장 많이 사용된 모델 조회
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
                if active_version_id:
                    self.logger.warning(
                        f"Active version (ID={active_version_id}) has no embeddings, "
                        f"using most common model from all versions"
                    )
                return detected_model
            else:
                self.logger.warning("No embeddings found in database for model detection")
                return None

        except Exception as e:
            # 데이터베이스 테이블이 없는 경우는 정상적인 상황일 수 있으므로 warning으로 처리
            try:
                if "no such table" in str(e).lower():
                    self.logger.debug(f"Embeddings table not found in database: {e}")
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
                           limit: Optional[int] = None,
                           embedding_version_id: Optional[int] = None) -> Dict[int, np.ndarray]:
        """
        embeddings 테이블에서 벡터 로드

        Args:
            source_types: 필터링할 source_type 목록 (None이면 전체)
            limit: 최대 로드 개수 (None이면 전체)
            embedding_version_id: 임베딩 버전 ID 필터 (None이면 활성 버전만)

        Returns:
            {chunk_id: vector} 딕셔너리
        """
        try:
            conn = self._get_connection()
            cursor = conn.cursor()

            # 기본 쿼리 (청킹 메타데이터 포함)
            query = """
                SELECT
                    e.chunk_id,
                    e.vector,
                    e.dim,
                    tc.source_type,
                    tc.text,
                    tc.source_id,
                    tc.chunk_size_category,
                    tc.chunk_group_id,
                    tc.chunking_strategy,
                    tc.embedding_version_id
                FROM embeddings e
                JOIN text_chunks tc ON e.chunk_id = tc.id
                WHERE 1=1
            """
            params = []

            # 모델 필터링 (embedding_version_id가 지정된 경우는 제외)
            if embedding_version_id is None and self.model_name:
                query += " AND e.model = ?"
                params.append(self.model_name)

            if source_types:
                placeholders = ','.join(['?'] * len(source_types))
                query += f" AND tc.source_type IN ({placeholders})"
                params.extend(source_types)

            # 버전 필터링
            if embedding_version_id is not None:
                query += " AND tc.embedding_version_id = ?"
                params.append(embedding_version_id)
            else:
                # 활성 버전만 조회
                query += """
                    AND tc.embedding_version_id IN (
                        SELECT id FROM embedding_versions WHERE is_active = 1
                    )
                """

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
                row_dict = dict(row)
                chunk_metadata[chunk_id] = {
                    'source_type': row_dict.get('source_type'),
                    'text': row_dict.get('text'),
                    'source_id': row_dict.get('source_id'),
                    'chunk_size_category': row_dict.get('chunk_size_category'),
                    'chunk_group_id': row_dict.get('chunk_group_id'),
                    'chunking_strategy': row_dict.get('chunking_strategy'),
                    'embedding_version_id': row_dict.get('embedding_version_id')  # 버전 정보 추가
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

    def _normalize_query(self, query: str) -> str:
        """쿼리 정규화 (공백, 대소문자)"""
        if not query:
            return ""
        # 공백 정규화 및 소문자 변환
        normalized = ' '.join(query.lower().split())
        return normalized
    
    def _encode_query(self, query: str) -> Optional[np.ndarray]:
        """쿼리 인코딩 (캐시 사용, 재정규화 포함)"""
        query_vec = self._get_cached_query_vector(query)
        if query_vec is not None:
            return query_vec
        
        if not self._ensure_embedder_initialized():
            self.logger.error("Cannot generate query embedding: embedder not initialized")
            return None
        
        try:
            query_vec = self.embedder.encode([query], batch_size=1, normalize=True)[0]
            
            query_norm = np.linalg.norm(query_vec)
            if abs(query_norm - 1.0) > 0.01:
                self.logger.debug(f"Query vector not normalized (norm={query_norm:.4f}), re-normalizing")
                query_vec = query_vec / (query_norm + 1e-9)
            
            self._cache_query_vector(query, query_vec)
            return query_vec
        except Exception as e:
            self.logger.error(f"Failed to encode query: {e}")
            if self.model_name and self._initialize_embedder(self.model_name):
                try:
                    query_vec = self.embedder.encode([query], batch_size=1, normalize=True)[0]
                    query_norm = np.linalg.norm(query_vec)
                    if abs(query_norm - 1.0) > 0.01:
                        self.logger.debug(f"Query vector not normalized (norm={query_norm:.4f}), re-normalizing")
                        query_vec = query_vec / (query_norm + 1e-9)
                    self._cache_query_vector(query, query_vec)
                    return query_vec
                except Exception as e2:
                    self.logger.error(f"Failed to encode query after re-initialization: {e2}")
            return None
    
    def _cleanup_expired_cache(self):
        """만료된 캐시 항목 제거"""
        current_time = time.time()
        
        # 정리 간격 체크 (너무 자주 정리하지 않도록)
        if current_time - self._metadata_cache_last_cleanup < self._metadata_cache_cleanup_interval:
            return
        
        expired_keys = []
        for key, cache_item in self._metadata_cache.items():
            if isinstance(cache_item, dict) and 'timestamp' in cache_item:
                if current_time - cache_item['timestamp'] > self._metadata_cache_ttl:
                    expired_keys.append(key)
            else:
                # 구 형식 (타임스탬프 없음) - 오래된 것으로 간주하고 제거
                expired_keys.append(key)
        
        for key in expired_keys:
            del self._metadata_cache[key]
        
        if expired_keys:
            self.logger.debug(f"Cleaned up {len(expired_keys)} expired cache items")
        
        self._metadata_cache_last_cleanup = current_time
    
    def _get_from_cache(self, key: Any) -> Optional[Any]:
        """캐시에서 항목 가져오기 (TTL 체크 포함)"""
        self._cleanup_expired_cache()
        
        if key not in self._metadata_cache:
            return None
        
        cache_item = self._metadata_cache[key]
        
        # 새 형식 (타임스탬프 포함)
        if isinstance(cache_item, dict) and 'timestamp' in cache_item:
            current_time = time.time()
            if current_time - cache_item['timestamp'] > self._metadata_cache_ttl:
                # 만료됨
                del self._metadata_cache[key]
                return None
            return cache_item.get('data')
        else:
            # 구 형식 (타임스탬프 없음) - 호환성을 위해 반환하되 새 형식으로 변환
            data = cache_item
            self._metadata_cache[key] = {
                'data': data,
                'timestamp': time.time()
            }
            return data
    
    def _set_to_cache(self, key: Any, value: Any):
        """캐시에 항목 저장 (TTL 포함)"""
        # 캐시 크기 제한 체크
        if len(self._metadata_cache) >= self._metadata_cache_max_size:
            # 만료된 항목 먼저 정리 시도
            self._cleanup_expired_cache()
            
            # 여전히 크기 제한 초과 시 LRU 방식으로 제거
            if len(self._metadata_cache) >= self._metadata_cache_max_size:
                oldest_key = next(iter(self._metadata_cache))
                del self._metadata_cache[oldest_key]
        
        # 새 형식으로 저장 (타임스탬프 포함)
        self._metadata_cache[key] = {
            'data': value,
            'timestamp': time.time()
        }

    def _get_cached_query_vector(self, query: str) -> Optional[np.ndarray]:
        """캐시에서 쿼리 벡터 가져오기 (정규화된 쿼리 사용)"""
        normalized_query = self._normalize_query(query)
        return self._query_vector_cache.get(normalized_query)

    def _cache_query_vector(self, query: str, vector: np.ndarray):
        """쿼리 벡터를 캐시에 저장 (LRU 방식, 정규화된 쿼리 사용)"""
        normalized_query = self._normalize_query(query)
        # 캐시 크기 제한 (LRU: 오래된 항목 제거)
        if len(self._query_vector_cache) >= self._cache_max_size:
            # 가장 오래된 항목 제거 (단순 구현: 첫 번째 항목)
            oldest_key = next(iter(self._query_vector_cache))
            del self._query_vector_cache[oldest_key]

        self._query_vector_cache[normalized_query] = vector

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

    def _calculate_similarity_from_distance(self, distance: float) -> float:
        """
        FAISS 인덱스에서 반환된 거리를 유사도 점수로 변환
        
        Args:
            distance: FAISS 인덱스에서 반환된 거리 값
            
        Returns:
            0-1 범위의 유사도 점수
        """
        try:
            import numpy as np
            
            if hasattr(self.index, 'metric_type'):
                if self.index.metric_type == faiss.METRIC_INNER_PRODUCT:
                    # Inner Product: 값이 클수록 유사도가 높음
                    similarity = (float(distance) + 1.0) / 2.0
                elif self.index.metric_type == faiss.METRIC_L2:
                    # L2 거리: 지수 감쇠 함수 사용 (0.85 목표 달성을 위해 스케일 조정)
                    # IndexIVFPQ는 압축 인덱스이므로 더 큰 스케일 사용
                    index_type = type(self.index).__name__ if self.index else ''
                    if 'IndexIVFPQ' in index_type:
                        # IndexIVFPQ: 더 큰 스케일 (거리 값이 클 수 있음, 0.85 목표 달성을 위해 8.0 → 10.0으로 증가)
                        scale = 10.0  # 스케일 증가로 유사도 점수 향상
                        similarity = np.exp(-float(distance) / scale)
                    else:
                        # 일반 L2: 표준 스케일 (2.0 → 3.0으로 증가)
                        scale = 3.0  # 스케일 증가로 유사도 점수 향상
                        similarity = np.exp(-float(distance) / scale)
                else:
                    # 기본: 역변환
                    similarity = 1.0 / (1.0 + float(distance))
            else:
                # metric_type이 없는 경우: 기본 변환
                similarity = 1.0 / (1.0 + float(distance))
        except Exception as e:
            self.logger.debug(f"Error calculating similarity: {e}, using default conversion")
            similarity = 1.0 / (1.0 + float(distance))
        
        return max(0.0, min(1.0, similarity))
    
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
            # 기본 가중치: 유사도 85%, ML 신뢰도 7.5%, 품질 점수 7.5%
            # similarity 가중치를 0.8 → 0.85로 증가하여 실제 유사도가 더 반영되도록 (0.85 목표 달성)
            weights = {
                "similarity": 0.85,
                "ml_confidence": 0.075,
                "quality": 0.075
            }
        
        # 가중치 합이 1이 되도록 정규화
        total_weight = sum(weights.values())
        if total_weight > 0:
            weights = {k: v / total_weight for k, v in weights.items()}
        
        hybrid_score = (
            weights.get("similarity", 0.7) * similarity +
            weights.get("ml_confidence", 0.15) * ml_confidence +
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
               filter_by_confidence: bool = False,
               chunk_size_category: Optional[str] = None,
               deduplicate_by_group: bool = True,
               embedding_version_id: Optional[int] = None,
               faiss_version: Optional[str] = None) -> List[Dict[str, Any]]:
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
            chunk_size_category: 청크 크기 카테고리 필터 (None이면 모든 크기, 'small', 'medium', 'large')
            deduplicate_by_group: 그룹별 중복 제거 활성화 (하이브리드 청킹용)
            embedding_version_id: 임베딩 버전 ID 필터 (None이면 활성 버전만, 특정 ID면 해당 버전만)
            faiss_version: FAISS 버전 이름 (None이면 활성 버전 또는 기본 인덱스)

        Returns:
            검색 결과 리스트 [{text, score, metadata, ...}]
        """
        import time
        import uuid
        
        # 성능 모니터링 시작
        query_id = str(uuid.uuid4())
        start_time = time.time()
        used_version = faiss_version or (getattr(self, 'current_faiss_version', None) or "default")
        
        # 활성 버전 정보 로깅 및 데이터 존재 여부 확인
        active_version_id = self._get_active_embedding_version_id()
        if active_version_id:
            self.logger.info(f"🔍 Semantic search starting - Active embedding version ID: {active_version_id}")
            
            # 활성 버전의 데이터 존재 여부 확인
            chunk_count = self._get_version_chunk_count(active_version_id)
            if chunk_count > 0:
                self.logger.info(f"✅ Active version has {chunk_count} chunks available")
            else:
                self.logger.warning(f"⚠️  Active version (ID={active_version_id}) has no chunks! Searching all versions instead.")
                active_version_id = None
        else:
            self.logger.warning("⚠️  No active embedding version found - using all versions")
        
        # embedding_version_id가 지정되지 않은 경우 활성 버전 사용
        if embedding_version_id is None:
            embedding_version_id = active_version_id
        
        if embedding_version_id:
            # 지정된 버전의 데이터 존재 여부 확인
            chunk_count = self._get_version_chunk_count(embedding_version_id)
            if chunk_count > 0:
                self.logger.info(f"📊 Using embedding version ID: {embedding_version_id} for search ({chunk_count} chunks)")
            else:
                self.logger.warning(f"⚠️  Specified version (ID={embedding_version_id}) has no chunks! Falling back to all versions.")
                embedding_version_id = None
        
        # FAISS 버전이 지정된 경우 해당 버전 로드
        current_version = getattr(self, 'current_faiss_version', None)
        if faiss_version and faiss_version != current_version:
            if hasattr(self, 'faiss_version_manager') and self.faiss_version_manager:
                try:
                    self._load_faiss_index(faiss_version)
                except Exception as e:
                    self.logger.warning(f"Failed to load FAISS version {faiss_version}: {e}")
        
        # 인덱스가 없으면 자동으로 로드 시도 (초기화 실패 시 재시도)
        index_load_failed = False
        if self.index is None and FAISS_AVAILABLE and self.embedder:
            try:
                if self.use_mlflow_index and self.mlflow_manager:
                    self._load_mlflow_index()
                elif faiss_version and hasattr(self, 'faiss_version_manager') and self.faiss_version_manager:
                    self._load_faiss_index(faiss_version)
                elif hasattr(self, 'faiss_version_manager') and self.faiss_version_manager:
                    # 버전 관리자가 있으면 활성 버전 시도
                    self._load_faiss_index()  # 활성 버전 자동 로드
                elif Path(self.index_path).exists():
                    self._load_faiss_index()
            except Exception as e:
                self.logger.warning(f"Failed to load FAISS index during search: {e}")
                self.logger.info("🔄 Falling back to direct database search (no FAISS index)")
                index_load_failed = True
                # 인덱스 로드 실패 시에도 계속 진행 (폴백 로직 사용)
        
        # 인덱스가 여전히 없으면 폴백 모드로 표시
        use_fallback = (self.index is None) or index_load_failed
        if use_fallback:
            self.logger.info("📊 Using fallback search mode: Direct database vector search (slower but reliable)")
        
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
                filter_by_confidence=filter_by_confidence,
                chunk_size_category=chunk_size_category,
                deduplicate_by_group=deduplicate_by_group,
                embedding_version_id=embedding_version_id
            )
            
            # 성능 모니터링 로깅
            elapsed_time = time.time() - start_time
            latency_ms = elapsed_time * 1000
            avg_relevance = sum(r.get('score', 0.0) for r in results) / len(results) if results else 0.0
            
            self.logger.info(
                f"⏱️  Search performance (no retry): {elapsed_time:.3f}s ({latency_ms:.1f}ms), "
                f"results: {len(results)}, avg_score: {avg_relevance:.3f}"
            )
            
            if self.enable_performance_monitoring and self.performance_monitor:
                self.performance_monitor.log_search(
                    version=used_version,
                    query_id=query_id,
                    latency_ms=latency_ms,
                    relevance_score=avg_relevance
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
                    filter_by_confidence=filter_by_confidence,
                    chunk_size_category=chunk_size_category,
                    deduplicate_by_group=deduplicate_by_group,
                    embedding_version_id=embedding_version_id
                )
                
                # 최소 결과 수를 만족하면 반환
                if len(results) >= min_results or attempt == len(thresholds_to_try) - 1:
                    if attempt > 0:
                        self.logger.info(
                            f"Semantic search: Found {len(results)} results "
                            f"(threshold lowered from {similarity_threshold:.2f} to {current_threshold:.2f})"
                        )
                    
                    # 성능 모니터링 로깅
                    elapsed_time = time.time() - start_time
                    latency_ms = elapsed_time * 1000
                    avg_relevance = sum(r.get('score', 0.0) for r in results) / len(results) if results else 0.0
                    
                    self.logger.info(
                        f"⏱️  Search performance: {elapsed_time:.3f}s ({latency_ms:.1f}ms), "
                        f"results: {len(results)}, avg_score: {avg_relevance:.3f}"
                    )
                    
                    if self.enable_performance_monitoring and self.performance_monitor:
                        self.performance_monitor.log_search(
                            version=used_version,
                            query_id=query_id,
                            latency_ms=latency_ms,
                            relevance_score=avg_relevance
                        )
                    
                    return results
                    
            except Exception as e:
                self.logger.warning(f"Semantic search attempt {attempt + 1} failed: {e}")
                if attempt == len(thresholds_to_try) - 1:
                    # 마지막 시도 실패 시에도 빈 결과 반환
                    self.logger.error("All semantic search attempts failed")
                    return []
                continue
        
        # 성능 모니터링 로깅 (실패한 경우)
        if self.enable_performance_monitoring and self.performance_monitor:
            latency_ms = (time.time() - start_time) * 1000
            self.performance_monitor.log_search(
                version=used_version,
                query_id=query_id,
                latency_ms=latency_ms,
                relevance_score=0.0
            )
        
        return []
    
    def search_multiple_versions(
        self,
        query: str,
        versions: List[str],
        k: int = 10,
        source_types: Optional[List[str]] = None,
        similarity_threshold: float = 0.5
    ) -> Dict[str, List[Dict[str, Any]]]:
        """
        여러 FAISS 버전에서 동시 검색
        
        Args:
            query: 검색 쿼리
            versions: 검색할 FAISS 버전 이름 리스트
            k: 각 버전에서 반환할 최대 결과 수
            source_types: 필터링할 소스 타입 목록
            similarity_threshold: 최소 유사도 임계값
        
        Returns:
            Dict[str, List[Dict]]: 버전별 검색 결과
        """
        if not self._ensure_embedder_initialized():
            self.logger.error("Embedder not initialized")
            return {}
        
        try:
            query_vec = self.embedder.encode([query], batch_size=1, normalize=True)[0]
        except Exception as e:
            self.logger.error(f"Failed to encode query: {e}")
            return {}
        
        if self.faiss_version_manager is None:
            self.logger.warning("FAISSVersionManager not available")
            return {}
        
        try:
            scripts_utils_path = Path(__file__).parent.parent.parent / "scripts" / "utils"
            if scripts_utils_path.exists():
                sys.path.insert(0, str(scripts_utils_path))
            from multi_version_search import MultiVersionSearch
            
            multi_search = MultiVersionSearch(self.faiss_version_manager)
            results = multi_search.search_all_versions(
                query_vector=query_vec,
                versions=versions,
                k=k
            )
            
            return results
            
        except Exception as e:
            self.logger.error(f"Failed to search multiple versions: {e}")
            return {}
    
    def ensemble_search(
        self,
        query: str,
        versions: List[str],
        weights: Optional[List[float]] = None,
        k: int = 10,
        source_types: Optional[List[str]] = None
    ) -> List[Dict[str, Any]]:
        """
        여러 FAISS 버전의 검색 결과를 앙상블
        
        Args:
            query: 검색 쿼리
            versions: 검색할 FAISS 버전 이름 리스트
            weights: 버전별 가중치 (None이면 균등 가중치)
            k: 최종 반환할 결과 수
            source_types: 필터링할 소스 타입 목록
        
        Returns:
            List[Dict]: 앙상블된 검색 결과
        """
        if not self._ensure_embedder_initialized():
            self.logger.error("Embedder not initialized")
            return []
        
        try:
            query_vec = self.embedder.encode([query], batch_size=1, normalize=True)[0]
        except Exception as e:
            self.logger.error(f"Failed to encode query: {e}")
            return []
        
        if self.faiss_version_manager is None:
            self.logger.warning("FAISSVersionManager not available")
            return []
        
        try:
            scripts_utils_path = Path(__file__).parent.parent.parent / "scripts" / "utils"
            if scripts_utils_path.exists():
                sys.path.insert(0, str(scripts_utils_path))
            from multi_version_search import MultiVersionSearch
            
            multi_search = MultiVersionSearch(self.faiss_version_manager)
            results = multi_search.ensemble_search(
                query_vector=query_vec,
                versions=versions,
                weights=weights,
                k=k
            )
            
            return results
            
        except Exception as e:
            self.logger.error(f"Failed to ensemble search: {e}")
            return []

    def _search_with_threshold(self,
                               query: str,
                               k: int,
                               source_types: Optional[List[str]],
                               similarity_threshold: float,
                               min_ml_confidence: Optional[float] = None,
                               min_quality_score: Optional[float] = None,
                               filter_by_confidence: bool = False,
                               chunk_size_category: Optional[str] = None,
                               deduplicate_by_group: bool = True,
                               embedding_version_id: Optional[int] = None) -> List[Dict[str, Any]]:
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
            chunk_size_category: 청크 크기 카테고리 필터
            deduplicate_by_group: 그룹별 중복 제거 활성화
            embedding_version_id: 임베딩 버전 ID 필터
        """
        try:
            self.logger.debug(f"_search_with_threshold called: query='{query[:50]}...', k={k}, threshold={similarity_threshold}, version_id={embedding_version_id}")
            import time
            step_times = {}
            step_start = time.time()
            
            # 1. 쿼리 임베딩 생성 (캐시 사용)
            encode_start = time.time()
            query_vec = self._encode_query(query)
            if query_vec is None:
                return []
            step_times['query_encoding'] = time.time() - encode_start
            if step_times['query_encoding'] < 0.001:
                self.logger.debug("⏱️  Query encoding: 0.000s (cached)")
            else:
                self.logger.debug(f"⏱️  Query encoding: {step_times['query_encoding']:.3f}s")

            # 2. FAISS 인덱스 사용 또는 전체 벡터 로드
            search_start = time.time()
            if FAISS_AVAILABLE and self.index is not None:
                # nprobe 동적 튜닝 (k 값에 따라 조정)
                # FAISS 인덱스 검색 (빠른 근사 검색)
                query_vec_np = np.array([query_vec]).astype('float32')
                search_k = k * 5  # 여유 있게 검색 (개선: 2배 → 5배로 확대)
                
                # nprobe 설정 (IndexIVF 계열만 지원)
                if hasattr(self.index, 'nprobe'):
                    optimal_nprobe = self._calculate_optimal_nprobe(k, self.index.ntotal)
                    if self.index.nprobe != optimal_nprobe:
                        self.index.nprobe = optimal_nprobe
                        self.logger.debug(f"Adjusted nprobe to {optimal_nprobe} for k={k}")
                
                distances, indices = self.index.search(query_vec_np, search_k)
                
                # IndexIVFPQ 인덱스 감지
                is_indexivfpq = 'IndexIVFPQ' in type(self.index).__name__
                
                self.logger.info(f"🔍 FAISS search returned {len(indices[0])} results (index_type={type(self.index).__name__}, filtering with version_id={embedding_version_id})")

                similarities = []
                for distance, idx in zip(distances[0], indices[0]):
                    if idx < 0 or idx >= len(self._chunk_ids):
                        continue
                    
                    # chunk_id 추출
                    if self.use_mlflow_index:
                        # MLflow 인덱스 사용 시: self._chunk_ids에서 조회
                        if hasattr(self, '_chunk_ids') and self._chunk_ids and idx < len(self._chunk_ids):
                            chunk_id = self._chunk_ids[idx]
                        else:
                            chunk_id = idx
                    else:
                        chunk_id = self._chunk_ids[idx] if hasattr(self, '_chunk_ids') and self._chunk_ids else idx
                    
                    # source_types 필터링 (FAISS 인덱스 사용 시 사전 필터링)
                    if source_types:
                        # chunk_id의 source_type을 먼저 확인
                        chunk_meta = self._chunk_metadata.get(chunk_id, {})
                        chunk_source_type = chunk_meta.get('source_type')
                        
                        # source_type이 없으면 DB에서 조회 (embedding_version_id도 함께 조회)
                        if not chunk_source_type or chunk_id not in self._chunk_metadata or 'embedding_version_id' not in self._chunk_metadata.get(chunk_id, {}):
                            try:
                                conn_temp = self._get_connection()
                                cursor_temp = conn_temp.execute(
                                    "SELECT source_type, embedding_version_id FROM text_chunks WHERE id = ?",
                                    (chunk_id,)
                                )
                                row_temp = cursor_temp.fetchone()
                                if row_temp:
                                    if not chunk_source_type:
                                        chunk_source_type = row_temp['source_type']
                                    # 캐시에 저장
                                    if chunk_id not in self._chunk_metadata:
                                        self._chunk_metadata[chunk_id] = {}
                                    self._chunk_metadata[chunk_id]['source_type'] = chunk_source_type
                                    
                                    # embedding_version_id 처리 (NULL인 경우 활성 버전 사용)
                                    version_id = row_temp.get('embedding_version_id')
                                    if version_id is None:
                                        active_version_id = self._get_active_embedding_version_id()
                                        if active_version_id:
                                            version_id = active_version_id
                                            self.logger.debug(f"Using active version {active_version_id} for chunk_id={chunk_id} (text_chunks.embedding_version_id is NULL)")
                                    self._chunk_metadata[chunk_id]['embedding_version_id'] = version_id
                                conn_temp.close()
                            except Exception as e:
                                self.logger.debug(f"Failed to get metadata for chunk_id={chunk_id}: {e}")
                                # 예외 발생 시에도 활성 버전 시도
                                try:
                                    active_version_id = self._get_active_embedding_version_id()
                                    if active_version_id:
                                        if chunk_id not in self._chunk_metadata:
                                            self._chunk_metadata[chunk_id] = {}
                                        self._chunk_metadata[chunk_id]['embedding_version_id'] = active_version_id
                                except Exception:
                                    pass
                        
                        # source_type이 source_types에 없으면 건너뛰기
                        if chunk_source_type and chunk_source_type not in source_types:
                            continue
                    
                    # embedding_version_id 필터링 (IndexIVFPQ 인덱스 사용 시 선택적 적용)
                    # is_indexivfpq는 이미 위에서 정의됨 (라인 1944)
                    if embedding_version_id is not None:
                        chunk_version_id = self._chunk_metadata.get(chunk_id, {}).get('embedding_version_id')
                        
                        # _chunk_metadata에 없으면 DB에서 조회
                        if chunk_version_id is None:
                            try:
                                conn_temp = self._get_connection()
                                cursor_temp = conn_temp.execute(
                                    "SELECT embedding_version_id FROM text_chunks WHERE id = ?",
                                    (chunk_id,)
                                )
                                row_temp = cursor_temp.fetchone()
                                if row_temp:
                                    chunk_version_id = row_temp.get('embedding_version_id')
                                    # NULL인 경우 활성 버전 사용
                                    if chunk_version_id is None:
                                        active_version_id = self._get_active_embedding_version_id()
                                        if active_version_id:
                                            chunk_version_id = active_version_id
                                    # 메타데이터에 저장
                                    if chunk_id not in self._chunk_metadata:
                                        self._chunk_metadata[chunk_id] = {}
                                    self._chunk_metadata[chunk_id]['embedding_version_id'] = chunk_version_id
                                conn_temp.close()
                            except Exception as e:
                                self.logger.debug(f"Failed to get embedding_version_id for chunk_id={chunk_id}: {e}")
                                # 예외 발생 시 활성 버전 사용
                                active_version_id = self._get_active_embedding_version_id()
                                if active_version_id:
                                    chunk_version_id = active_version_id
                        
                        # IndexIVFPQ 인덱스 사용 시 필터링 완화 (활성 버전이 없거나 매칭되지 않아도 허용)
                        if chunk_version_id != embedding_version_id:
                            if is_indexivfpq:
                                # IndexIVFPQ 인덱스 사용 시: 활성 버전이 없거나 매칭되지 않아도 경고만 출력하고 계속 진행
                                if chunk_version_id is None:
                                    self.logger.debug(f"IndexIVFPQ: chunk_id={chunk_id} has no version_id, allowing (filtering relaxed)")
                                else:
                                    self.logger.debug(f"IndexIVFPQ: chunk_id={chunk_id} version_id={chunk_version_id} != {embedding_version_id}, allowing (filtering relaxed)")
                            else:
                                # 일반 인덱스는 엄격한 필터링
                                self.logger.debug(f"Filtered out chunk_id={chunk_id}: version_id={chunk_version_id} != {embedding_version_id}")
                                continue
                    
                    similarity = self._calculate_similarity_from_distance(distance)

                    if similarity >= similarity_threshold:
                        similarities.append((chunk_id, similarity))
                        self.logger.debug(f"Added to similarities: chunk_id={chunk_id}, similarity={similarity:.4f}, version_id={chunk_version_id if embedding_version_id is not None else 'N/A'}")

                # 유사도 기준 정렬
                similarities.sort(key=lambda x: x[1], reverse=True)  # similarity는 인덱스 1
                
                step_times['faiss_search'] = time.time() - search_start
                
                # IndexIVFPQ 인덱스 사용 시 상세 로깅
                if is_indexivfpq:
                    self.logger.info(f"🔍 IndexIVFPQ search: {len(indices[0])} FAISS results → {len(similarities)} after filtering (threshold={similarity_threshold:.3f})")
                    if len(similarities) > 0:
                        avg_sim = sum(s[1] for s in similarities) / len(similarities)
                        max_sim = max(s[1] for s in similarities)
                        min_sim = min(s[1] for s in similarities)
                        self.logger.info(f"   Similarity scores: avg={avg_sim:.3f}, max={max_sim:.3f}, min={min_sim:.3f}")
                    else:
                        self.logger.warning(f"   ⚠️  No results after filtering! FAISS returned {len(indices[0])} results but all were filtered out.")
                        self.logger.warning(f"   Possible causes: similarity_threshold too high ({similarity_threshold:.3f}) or embedding_version_id mismatch")
                else:
                    self.logger.debug(f"⏱️  FAISS search: {step_times['faiss_search']:.3f}s, {len(similarities)} results")

            else:
                # 기존 방식 (전체 벡터 로드 및 선형 검색)
                # FAISS 인덱스가 없으면 백그라운드에서 빌드 시작
                if FAISS_AVAILABLE and self.index is None and not self._index_building:
                    self.logger.info("FAISS index not found, starting background build")
                    self._build_faiss_index_async()

                # 활성 버전 필터링 로직 검증
                if embedding_version_id:
                    self.logger.info(f"📋 Loading chunk vectors for embedding_version_id={embedding_version_id}")
                else:
                    active_version_id = self._get_active_embedding_version_id()
                    if active_version_id:
                        self.logger.info(f"📋 Loading chunk vectors for active version (ID={active_version_id})")
                    else:
                        self.logger.warning("📋 Loading chunk vectors from all versions (no active version)")
                
                chunk_vectors = self._load_chunk_vectors(
                    source_types=source_types,
                    embedding_version_id=embedding_version_id
                )

                if not chunk_vectors:
                    self.logger.warning(
                        f"⚠️ No chunk vectors found for search. "
                        f"embedding_version_id={embedding_version_id}, "
                        f"This may indicate that embeddings need to be generated."
                    )
                    return []
                
                # 로드된 청크 수 로깅
                self.logger.info(f"✅ Loaded {len(chunk_vectors)} chunk vectors for search")

                # 코사인 유사도 계산
                similarities = []
                for chunk_id, chunk_vec in chunk_vectors.items():
                    similarity = self._cosine_similarity(query_vec, chunk_vec)
                    if similarity >= similarity_threshold:
                        similarities.append((chunk_id, similarity))

                # 유사도 기준 정렬
                similarities.sort(key=lambda x: x[1], reverse=True)
                
                step_times['vector_search'] = time.time() - search_start
                self.logger.debug(f"⏱️  Vector search (DB): {step_times['vector_search']:.3f}s")

            # 5. 상위 K개 결과 구성
            result_processing_start = time.time()
            results = []
            
            # 텍스트 복원을 위해 DB 연결 필요 (외부 인덱스 사용 시에도)
            conn = self._get_connection()

            self.logger.debug(f"Processing {len(similarities)} similarities, top {k} results")
            
            # 배치 메타데이터 조회를 위한 준비 (최적화: 중복 제거 및 필터링)
            chunk_ids_for_batch = []
            seen_chunk_ids = set()
            
            # 먼저 모든 chunk_id 수집 (중복 제거)
            for similarity_item in similarities[:k]:
                chunk_id, score = similarity_item
                
                if chunk_id not in seen_chunk_ids:
                    chunk_ids_for_batch.append(chunk_id)
                    seen_chunk_ids.add(chunk_id)
            
            # 배치로 chunk_metadata 조회 (캐싱 적용)
            batch_chunk_metadata = {}
            if conn and chunk_ids_for_batch:
                batch_start = time.time()
                batch_chunk_metadata = self._batch_load_chunk_metadata(conn, chunk_ids_for_batch)
                batch_time = time.time() - batch_start
                self.logger.debug(f"Batch loaded metadata for {len(batch_chunk_metadata)} chunks in {batch_time:.3f}s")
            
            # source_items 수집 (source_type, source_id 쌍, 중복 제거)
            source_items_for_batch = []
            seen_source_items = set()
            for chunk_id in chunk_ids_for_batch:
                if chunk_id in batch_chunk_metadata:
                    chunk_meta = batch_chunk_metadata[chunk_id]
                    source_type = chunk_meta.get('source_type')
                    source_id = chunk_meta.get('source_id')
                    if source_type and source_id:
                        # source_id가 문자열인 경우 정수로 변환 시도
                        if isinstance(source_id, str):
                            try:
                                import re
                                numbers = re.findall(r'\d+', str(source_id))
                                if numbers:
                                    source_id = int(numbers[-1])
                                else:
                                    continue
                            except (ValueError, TypeError):
                                continue
                        
                        source_key = (source_type, source_id)
                        if source_key not in seen_source_items:
                            source_items_for_batch.append(source_key)
                            seen_source_items.add(source_key)
            
            # 배치로 source_metadata 조회 (캐싱 적용)
            batch_source_metadata = {}
            if conn and source_items_for_batch:
                batch_start = time.time()
                batch_source_metadata = self._batch_load_source_metadata(conn, source_items_for_batch)
                batch_time = time.time() - batch_start
                self.logger.debug(f"Batch loaded source metadata for {len(batch_source_metadata)} source items in {batch_time:.3f}s")
            
            for similarity_item in similarities[:k]:
                chunk_id, score = similarity_item
                self.logger.debug(f"Processing result: chunk_id={chunk_id}, score={score:.4f}")
                
                # metadata 변수 초기화
                metadata = None
                
                # MLflow 인덱스 사용 시 메타데이터는 _chunk_metadata에 이미 로드됨
                if self.use_mlflow_index and chunk_id in self._chunk_metadata:
                    metadata = self._chunk_metadata[chunk_id]
                    self.logger.debug(f"Found metadata for chunk_id={chunk_id}")
                    text = metadata.get('content', '') or metadata.get('text', '')
                    source_type = metadata.get('type') or metadata.get('source_type', '')
                    
                    # source_type이 없으면 메타데이터에서 추론
                    if not source_type:
                        if metadata.get('case_id') or metadata.get('case_number') or metadata.get('doc_id'):
                            source_type = 'case_paragraph'
                        elif metadata.get('law_id') or metadata.get('law_name') or metadata.get('article_number'):
                            source_type = 'statute_article'
                        elif metadata.get('decision_id') or metadata.get('org'):
                            source_type = 'decision_paragraph'
                        elif metadata.get('interpretation_id'):
                            source_type = 'interpretation_paragraph'
                        
                        # source_meta에 모든 메타데이터 포함
                        source_meta = metadata.copy()
                        source_meta['source_type'] = source_type
                        # source_id 추출: 실제 DB ID를 찾기 위해 text_chunks에서 조회
                        # 외부 인덱스 메타데이터에는 case_number, doc_id 등이 있을 수 있지만 실제 DB ID는 다를 수 있음
                        potential_source_id = metadata.get('case_id') or metadata.get('law_id') or metadata.get('doc_id', '')
                        
                        # DB에서 실제 source_id 찾기 (text_chunks 테이블 사용)
                        actual_source_id = None
                        if conn and source_type and potential_source_id:
                            try:
                                # text_chunks에서 source_type과 다른 필드로 조회
                                if source_type == 'case_paragraph':
                                    # case_number나 doc_id로 조회
                                    case_number = metadata.get('case_number') or metadata.get('doc_id')
                                    if case_number:
                                        cursor = conn.execute(
                                            "SELECT DISTINCT source_id FROM text_chunks WHERE source_type = ? AND (metadata LIKE ? OR metadata LIKE ?) LIMIT 1",
                                            (source_type, f'%{case_number}%', f'%doc_id%{case_number}%')
                                        )
                                        row = cursor.fetchone()
                                        if row:
                                            actual_source_id = row[0]
                                elif source_type == 'statute_article':
                                    # law_id와 article_number로 조회
                                    law_id = metadata.get('law_id')
                                    article_no = metadata.get('article_number') or metadata.get('article_no')
                                    if law_id and article_no:
                                        cursor = conn.execute(
                                            "SELECT DISTINCT source_id FROM text_chunks WHERE source_type = ? AND metadata LIKE ? AND metadata LIKE ? LIMIT 1",
                                            (source_type, f'%law_id%{law_id}%', f'%article_number%{article_no}%')
                                        )
                                        row = cursor.fetchone()
                                        if row:
                                            actual_source_id = row[0]
                                
                                # 조회 실패 시 potential_source_id가 숫자면 그대로 사용
                                if actual_source_id is None:
                                    try:
                                        actual_source_id = int(potential_source_id)
                                    except (ValueError, TypeError):
                                        # 문자열이면 None으로 설정 (복원 시도하지 않음)
                                        actual_source_id = None
                            except Exception as e:
                                self.logger.debug(f"Failed to find actual source_id for chunk_id={chunk_id}: {e}")
                                actual_source_id = None
                        
                        # _chunk_metadata에도 저장 (일관성 유지, 청킹 메타데이터 포함)
                        # DB에서 청킹 메타데이터 조회
                        chunking_meta = {}
                        if conn:
                            try:
                                cursor = conn.execute(
                                    "SELECT chunk_size_category, chunk_group_id, chunking_strategy, embedding_version_id FROM text_chunks WHERE id = ?",
                                    (chunk_id,)
                                )
                                chunk_row = cursor.fetchone()
                                if chunk_row:
                                    chunk_row_dict = dict(chunk_row) if chunk_row else {}
                                    chunking_meta = {
                                        'chunk_size_category': chunk_row_dict.get('chunk_size_category'),
                                        'chunk_group_id': chunk_row_dict.get('chunk_group_id'),
                                        'chunking_strategy': chunk_row_dict.get('chunking_strategy'),
                                        'embedding_version_id': chunk_row_dict.get('embedding_version_id')  # 버전 정보 추가
                                    }
                            except Exception as e:
                                self.logger.debug(f"Failed to load chunking metadata for chunk_id={chunk_id}: {e}")
                        
                        self._chunk_metadata[chunk_id] = {
                            'source_type': source_type,
                            'source_id': actual_source_id if actual_source_id is not None else potential_source_id,
                            'text': text,
                            **chunking_meta,
                            **metadata
                        }
                    else:
                        # idx가 범위를 벗어났거나 _external_metadata가 없는 경우 - DB에서 직접 조회
                        self.logger.debug(f"_external_metadata not available for idx={idx}, loading from DB for chunk_id={chunk_id}")
                        if not conn or not chunk_id:
                            self.logger.warning(f"⚠️  Cannot load chunk_id={chunk_id} from DB (conn={conn is not None}, chunk_id={chunk_id})")
                            continue
                        
                        try:
                            cursor = conn.execute(
                                "SELECT source_type, source_id, text, chunk_size_category, chunk_group_id, chunking_strategy, embedding_version_id FROM text_chunks WHERE id = ?",
                                (chunk_id,)
                            )
                            row = cursor.fetchone()
                            if row:
                                text = row['text'] or ''
                                source_type = row['source_type'] or ''
                                source_id = row['source_id']
                                self.logger.debug(f"Loaded from DB: chunk_id={chunk_id}, source_type={source_type}")
                                
                                # _chunk_metadata에 저장
                                version_id = dict(row).get('embedding_version_id')
                                if version_id is None:
                                    active_version_id = self._get_active_embedding_version_id()
                                    if active_version_id:
                                        version_id = active_version_id
                                
                                self._chunk_metadata[chunk_id] = {
                                    'source_type': source_type,
                                    'source_id': source_id,
                                    'text': text,
                                    'chunk_size_category': dict(row).get('chunk_size_category'),
                                    'chunk_group_id': dict(row).get('chunk_group_id'),
                                    'chunking_strategy': dict(row).get('chunking_strategy'),
                                    'embedding_version_id': version_id
                                }
                            else:
                                self.logger.warning(f"⚠️  chunk_id={chunk_id} not found in database")
                                continue
                        except Exception as e:
                            self.logger.warning(f"⚠️  Failed to load chunk_id={chunk_id} from DB: {e}")
                            continue
                
                if chunk_id not in self._chunk_metadata:
                    # 메타데이터가 없으면 DB에서 직접 조회 (전체 텍스트 및 청킹 메타데이터 가져오기)
                    cursor = conn.execute(
                        "SELECT source_type, source_id, text, chunk_size_category, chunk_group_id, chunking_strategy, embedding_version_id FROM text_chunks WHERE id = ?",
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
                        
                        # embedding_version_id가 NULL인 경우 활성 버전 사용
                        version_id = dict(row).get('embedding_version_id')
                        if version_id is None:
                            active_version_id = self._get_active_embedding_version_id()
                            if active_version_id:
                                version_id = active_version_id
                                self.logger.debug(f"Using active version {version_id} for chunk_id={chunk_id} (text_chunks.embedding_version_id is NULL)")
                        
                        self._chunk_metadata[chunk_id] = {
                            'source_type': row['source_type'],
                            'source_id': row['source_id'],
                            'text': text_content,
                            'chunk_size_category': dict(row).get('chunk_size_category'),
                            'chunk_group_id': dict(row).get('chunk_group_id'),
                            'chunking_strategy': dict(row).get('chunking_strategy'),
                            'embedding_version_id': version_id  # 버전 정보 추가 (NULL인 경우 활성 버전 사용)
                        }

                chunk_metadata = self._chunk_metadata.get(chunk_id, {})
                # chunk_metadata가 비어있으면 DB에서 로드
                if not chunk_metadata and conn:
                    try:
                        cursor = conn.execute(
                            "SELECT source_type, source_id, text, chunk_size_category, chunk_group_id, chunking_strategy, embedding_version_id, meta FROM text_chunks WHERE id = ?",
                            (chunk_id,)
                        )
                        row = cursor.fetchone()
                        if row:
                            version_id = dict(row).get('embedding_version_id')
                            if version_id is None:
                                active_version_id = self._get_active_embedding_version_id()
                                if active_version_id:
                                    version_id = active_version_id
                            
                            # text_chunks.meta 컬럼에서 메타데이터 로드
                            chunk_meta_json = None
                            if row['meta']:
                                try:
                                    import json
                                    chunk_meta_json = json.loads(row['meta'])
                                except Exception as e:
                                    self.logger.debug(f"Failed to parse meta JSON for chunk_id={chunk_id}: {e}")
                            
                            chunk_metadata = {
                                'source_type': row['source_type'],
                                'source_id': row['source_id'],
                                'text': row['text'] if row['text'] else '',
                                'chunk_size_category': dict(row).get('chunk_size_category'),
                                'chunk_group_id': dict(row).get('chunk_group_id'),
                                'chunking_strategy': dict(row).get('chunking_strategy'),
                                'embedding_version_id': version_id
                            }
                            
                            # text_chunks.meta의 메타데이터를 chunk_metadata에 병합
                            if chunk_meta_json:
                                chunk_metadata.update(chunk_meta_json)
                            
                            # self._chunk_metadata에도 저장
                            self._chunk_metadata[chunk_id] = chunk_metadata
                    except Exception as e:
                        self.logger.debug(f"Failed to load chunk_metadata for chunk_id={chunk_id}: {e}")
                
                source_type = chunk_metadata.get('source_type')
                source_id = chunk_metadata.get('source_id')
                # source_id가 문자열인 경우 정수로 변환 시도
                if source_id and isinstance(source_id, str):
                    try:
                        # 문자열에서 숫자 추출 시도 (예: "case_2021도1750" -> 1750)
                        import re
                        numbers = re.findall(r'\d+', str(source_id))
                        if numbers:
                            source_id = int(numbers[-1])  # 마지막 숫자 사용
                        else:
                            # 숫자가 없으면 None으로 설정하여 조회 실패 처리
                            source_id = None
                    except (ValueError, TypeError):
                        source_id = None
                text = chunk_metadata.get('text', '')
                chunk_size_cat = chunk_metadata.get('chunk_size_category')
                chunk_group_id = chunk_metadata.get('chunk_group_id')
                
                # source_types 필터링 (FAISS 인덱스 사용 시 필수)
                if source_types and source_type not in source_types:
                    self.logger.debug(f"Filtered chunk {chunk_id}: source_type {source_type} not in {source_types}")
                    continue
                
                # chunk_size_category 필터링 (하이브리드 청킹 지원)
                if chunk_size_category and chunk_size_cat != chunk_size_category:
                    self.logger.debug(f"Filtered chunk {chunk_id}: chunk_size_category {chunk_size_cat} != {chunk_size_category}")
                    continue
                
                # 텍스트 복원을 위해 전체 메타데이터도 필요
                full_metadata = chunk_metadata.copy()

                # text가 비어있거나 짧으면 원본 테이블에서 복원 시도 (최소 100자 보장)
                text = self._ensure_text_content(
                    conn, chunk_id, text, source_type, source_id, full_metadata
                )

                # 소스별 상세 메타데이터 조회 (배치 조회 결과 사용)
                if conn and source_type and source_id:
                    # 배치로 조회한 chunk_metadata 사용
                    chunk_meta_from_db = None
                    if chunk_id in batch_chunk_metadata:
                        chunk_meta_from_db = batch_chunk_metadata[chunk_id].get('meta', {})
                    
                    # source_id가 정수인지 확인하고 변환
                    source_id_for_query = source_id
                    if source_id and not isinstance(source_id, int):
                        try:
                            if isinstance(source_id, str):
                                import re
                                numbers = re.findall(r'\d+', str(source_id))
                                if numbers:
                                    source_id_for_query = int(numbers[-1])
                                else:
                                    source_id_for_query = None
                            else:
                                source_id_for_query = int(source_id)
                        except (ValueError, TypeError):
                            source_id_for_query = None
                    
                    # 배치로 조회한 source_metadata 사용
                    source_meta = {}
                    if source_id_for_query:
                        source_meta_key = (source_type, source_id_for_query)
                        if source_meta_key in batch_source_metadata:
                            source_meta = batch_source_metadata[source_meta_key].copy()
                        else:
                            # 배치 조회에 없으면 개별 조회 (폴백)
                            source_meta = self._get_source_metadata(conn, source_type, source_id_for_query)
                    
                    # text_chunks.meta의 메타데이터를 우선적으로 사용 (소스 테이블 메타데이터와 병합)
                    if chunk_meta_from_db:
                        # chunk_meta_from_db를 우선, source_meta는 보완용으로 사용
                        merged_meta = {**source_meta, **chunk_meta_from_db}
                        source_meta = merged_meta
                    
                    # 소스 테이블에서 메타데이터 추가 조회 (누락된 필드 보완)
                    # 배치 조회 결과가 비어있거나 필수 필드가 누락된 경우에만 재조회
                    # 배치 조회에서 이미 가져온 경우 재조회 생략 (최적화)
                    if not source_meta or len(source_meta) == 0:
                        # 배치 조회에 없었던 경우에만 개별 조회
                        if source_id_for_query and source_meta_key not in batch_source_metadata:
                            source_meta_from_table = self._get_source_metadata(conn, source_type, source_id_for_query)
                            if source_meta_from_table:
                                source_meta = source_meta_from_table
                                # 캐시에 저장 (TTL 포함)
                                self._set_to_cache(source_meta_key, source_meta)
                    else:
                        # 필수 필드 확인 및 보완 (배치 조회 결과가 있지만 필수 필드가 누락된 경우)
                        needs_reload = False
                        if source_type == "statute_article":
                            if not source_meta.get("statute_name") or not source_meta.get("article_no"):
                                needs_reload = True
                        elif source_type == "case_paragraph":
                            if not source_meta.get("doc_id") or not source_meta.get("casenames") or not source_meta.get("court"):
                                needs_reload = True
                        elif source_type == "decision_paragraph":
                            if not source_meta.get("org") or not source_meta.get("doc_id"):
                                needs_reload = True
                        elif source_type == "interpretation_paragraph":
                            if not source_meta.get("org") or not source_meta.get("doc_id"):
                                needs_reload = True
                        
                        # 필수 필드가 누락된 경우에만 재조회 (배치 조회 결과가 불완전한 경우)
                        if needs_reload and source_id_for_query:
                            source_meta_from_table = self._get_source_metadata(conn, source_type, source_id_for_query)
                            if source_meta_from_table:
                                # 누락된 필드만 보완
                                for key, value in source_meta_from_table.items():
                                    if key not in source_meta or not source_meta[key]:
                                        source_meta[key] = value
                                # 캐시 업데이트 (TTL 포함)
                                cached_data = self._get_from_cache(source_meta_key)
                                if cached_data is not None:
                                    # 기존 캐시 항목 업데이트
                                    cached_data.update(source_meta)
                                    self._set_to_cache(source_meta_key, cached_data)
                                else:
                                    # 새로 저장
                                    self._set_to_cache(source_meta_key, source_meta)
                    
                    # statute_article의 경우 별칭 필드도 설정
                    if source_type == "statute_article":
                        if "statute_name" in source_meta and not source_meta.get("law_name"):
                            source_meta["law_name"] = source_meta["statute_name"]
                        if "article_no" in source_meta and not source_meta.get("article_number"):
                            source_meta["article_number"] = source_meta["article_no"]
                    
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
                    
                    # 타입별 최소 길이 차등 적용 (P1-4: 더욱 완화 - 10자 → 5자)
                    if source_type == 'statute_article':
                        min_text_length = 30
                    elif source_type in ['case_paragraph', 'decision_paragraph']:
                        min_text_length = 5
                    else:
                        min_text_length = 50
                    if text and len(text.strip()) < min_text_length:
                        restored_text = self._restore_text_from_source(conn, source_type, source_id)
                        if restored_text and len(restored_text.strip()) > len(text.strip()):
                            text = restored_text
                            self.logger.debug(f"Extended text for chunk_id={chunk_id} to {len(text)} chars")
                else:
                    # conn, source_type, source_id가 없는 경우 기본값 설정
                    if self.use_mlflow_index and metadata:
                        source_meta = metadata.copy()
                    else:
                        source_meta = {}
                
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
                    from core.generation.formatters.unified_source_formatter import UnifiedSourceFormatter
                    formatter = UnifiedSourceFormatter()
                    source_info = formatter.format_source(source_type, source_meta)
                    source_name = source_info.name
                    source_url = source_info.url
                except Exception as e:
                    self.logger.warning(f"Error using UnifiedSourceFormatter: {e}, using fallback")
                    source_name = self._format_source(source_type, source_meta)
                    source_url = ""
                
                # embedding_version_id 조회 (메타데이터에서 먼저 확인)
                result_embedding_version_id = chunk_metadata.get('embedding_version_id')
                if result_embedding_version_id is None and conn:
                    # 메타데이터에 없으면 DB에서 조회
                    try:
                        cursor = conn.execute(
                            "SELECT embedding_version_id FROM text_chunks WHERE id = ?",
                            (chunk_id,)
                        )
                        version_row = cursor.fetchone()
                        if version_row:
                            result_embedding_version_id = version_row['embedding_version_id']
                            # NULL인 경우 활성 버전 사용
                            if result_embedding_version_id is None:
                                active_version_id = self._get_active_embedding_version_id()
                                if active_version_id:
                                    result_embedding_version_id = active_version_id
                                    self.logger.debug(f"Using active version {active_version_id} for chunk_id={chunk_id} (text_chunks.embedding_version_id is NULL)")
                            # 메타데이터에 저장 (로컬 및 인스턴스 변수 모두)
                            chunk_metadata['embedding_version_id'] = result_embedding_version_id
                            if chunk_id in self._chunk_metadata:
                                self._chunk_metadata[chunk_id]['embedding_version_id'] = result_embedding_version_id
                    except Exception as e:
                        self.logger.debug(f"Failed to get embedding_version_id for chunk_id={chunk_id}: {e}")
                        # 예외 발생 시에도 활성 버전 시도
                        try:
                            active_version_id = self._get_active_embedding_version_id()
                            if active_version_id:
                                result_embedding_version_id = active_version_id
                                chunk_metadata['embedding_version_id'] = active_version_id
                                if chunk_id in self._chunk_metadata:
                                    self._chunk_metadata[chunk_id]['embedding_version_id'] = active_version_id
                        except Exception:
                            pass
                
                # 최종적으로도 None이면 활성 버전 사용
                if result_embedding_version_id is None:
                    active_version_id = self._get_active_embedding_version_id()
                    if active_version_id:
                        result_embedding_version_id = active_version_id
                        chunk_metadata['embedding_version_id'] = active_version_id
                        if chunk_id in self._chunk_metadata:
                            self._chunk_metadata[chunk_id]['embedding_version_id'] = active_version_id
                        self.logger.debug(f"Using active version {active_version_id} for chunk_id={chunk_id} as fallback")
                
                # 쿼리-문서 쌍 직접 유사도 재계산 (상위 결과 보정)
                # FAISS 근사 검색의 오차를 보정하기 위해 직접 코사인 유사도 계산
                direct_similarity = float(score)
                if len(results) < 10:  # 상위 10개 결과에 대해서만 재계산 (성능 고려)
                    try:
                        # _chunk_vectors 초기화 확인
                        if not hasattr(self, '_chunk_vectors'):
                            self._chunk_vectors = {}
                        
                        # 문서 벡터 로드
                        if chunk_id in self._chunk_vectors:
                            doc_vec = self._chunk_vectors[chunk_id]
                        else:
                            # DB에서 벡터 로드
                            with sqlite3.connect(self.db_path) as conn:
                                conn.row_factory = sqlite3.Row
                                cursor = conn.execute(
                                    "SELECT embedding FROM embeddings WHERE chunk_id = ? LIMIT 1",
                                    (chunk_id,)
                                )
                                row = cursor.fetchone()
                                if row:
                                    import pickle
                                    doc_vec = pickle.loads(row['embedding'])
                                    self._chunk_vectors[chunk_id] = doc_vec
                                else:
                                    doc_vec = None
                        
                        if doc_vec is not None and query_vec is not None:
                            # 직접 코사인 유사도 계산
                            doc_vec_norm = np.linalg.norm(doc_vec)
                            if doc_vec_norm > 0:
                                doc_vec_normalized = doc_vec / doc_vec_norm
                                direct_similarity = float(np.dot(query_vec, doc_vec_normalized))
                                # FAISS 점수와 직접 계산 점수의 가중 평균 (직접 계산 점수에 더 높은 가중치, 0.85 목표 달성)
                                # 직접 계산 점수 가중치를 0.8 → 0.9로 증가
                                score = 0.9 * direct_similarity + 0.1 * float(score)
                    except Exception as e:
                        self.logger.debug(f"Failed to recalculate direct similarity for chunk {chunk_id}: {e}")
                
                # 필수 메타데이터 필드 보완 (누락된 경우 재조회)
                if conn and source_type and source_id:
                    # 필수 필드가 누락된 경우 재조회 시도
                    required_fields_missing = False
                    if source_type == "case_paragraph":
                        if not source_meta.get("doc_id") or not source_meta.get("casenames") or not source_meta.get("court"):
                            required_fields_missing = True
                    elif source_type == "decision_paragraph":
                        if not source_meta.get("org") or not source_meta.get("doc_id"):
                            required_fields_missing = True
                    elif source_type == "statute_article":
                        if not source_meta.get("statute_name") or not source_meta.get("article_no"):
                            required_fields_missing = True
                    elif source_type == "interpretation_paragraph":
                        if not source_meta.get("org") or not source_meta.get("doc_id"):
                            required_fields_missing = True
                    
                    if required_fields_missing:
                        # 소스 테이블에서 메타데이터 재조회
                        additional_meta = self._get_source_metadata(conn, source_type, source_id)
                        if additional_meta:
                            # 누락된 필드만 보완
                            for key, value in additional_meta.items():
                                if key not in source_meta or not source_meta[key]:
                                    source_meta[key] = value
                
                result = {
                    "id": f"chunk_{chunk_id}",
                    "text": text,
                    "content": text,  # content 필드 보장
                    "score": float(score),
                    "similarity": float(score),
                    "direct_similarity": direct_similarity,  # 직접 계산된 유사도 추가
                    "type": source_type,
                    "source": source_name,
                    "source_url": source_url,  # URL 필드 추가
                    "embedding_version_id": result_embedding_version_id,  # 버전 정보 추가
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
                        "embedding_version_id": result_embedding_version_id,  # 버전 정보 추가
                        "ml_confidence_score": ml_confidence,
                        "quality_score": quality_score,
                        "chunk_size_category": chunk_size_cat,
                        "chunk_group_id": chunk_group_id,
                        "chunking_strategy": chunk_metadata.get('chunking_strategy'),
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

            # 그룹별 중복 제거 (하이브리드 청킹 지원)
            if deduplicate_by_group:
                seen_groups = {}
                deduplicated_results = []
                for result in results:
                    group_id = result.get("metadata", {}).get("chunk_group_id")
                    if group_id and group_id != 'N/A':
                        if group_id not in seen_groups:
                            seen_groups[group_id] = result
                            deduplicated_results.append(result)
                        elif result.get("relevance_score", 0) > seen_groups[group_id].get("relevance_score", 0):
                            # 더 높은 점수로 교체
                            idx = deduplicated_results.index(seen_groups[group_id])
                            deduplicated_results[idx] = result
                            seen_groups[group_id] = result
                    else:
                        # 그룹 ID가 없으면 그대로 추가
                        deduplicated_results.append(result)
                results = deduplicated_results[:k]
            else:
                results = results[:k]

            if conn:
                conn.close()
            
            step_times['result_processing'] = time.time() - result_processing_start
            step_times['total'] = time.time() - step_start
            
            # 단계별 성능 로깅
            if step_times:
                perf_summary = ", ".join([f"{k}: {v:.3f}s" for k, v in step_times.items() if k != 'total'])
                self.logger.info(f"⏱️  Search step performance: {perf_summary}, total: {step_times.get('total', 0):.3f}s")
            
            # 검색 결과 분석 및 로깅
            if results:
                # 유사도 점수 분포 분석
                scores = [r.get('score', 0.0) for r in results]
                if scores:
                    avg_score = sum(scores) / len(scores)
                    max_score = max(scores)
                    min_score = min(scores)
                    self.logger.info(
                        f"📊 Search results analysis: {len(results)} results, "
                        f"score range: [{min_score:.3f}, {max_score:.3f}], "
                        f"avg: {avg_score:.3f}"
                    )
                    
                    # 버전 분포 분석 및 일관성 검증
                    version_counts = {}
                    for r in results:
                        version_id = r.get('embedding_version_id')
                        if version_id:
                            version_counts[version_id] = version_counts.get(version_id, 0) + 1
                    if version_counts:
                        self.logger.info(f"📊 Embedding version distribution: {version_counts}")
                        
                        # 활성 버전과 일치 여부 확인
                        if embedding_version_id:
                            active_count = version_counts.get(embedding_version_id, 0)
                            total_count = sum(version_counts.values())
                            if active_count < total_count:
                                mismatch_ratio = (total_count - active_count) / total_count * 100
                                self.logger.warning(
                                    f"⚠️  Version mismatch detected: "
                                    f"Expected version {embedding_version_id} but found {mismatch_ratio:.1f}% from other versions"
                                )
                            else:
                                self.logger.info(f"✅ All results are from expected version {embedding_version_id}")
                    else:
                        self.logger.warning("⚠️  No embedding_version_id found in results")
                else:
                    self.logger.warning("⚠️  No scores found in results")
            else:
                self.logger.warning(f"⚠️  No results found for query: {query[:50]}")
                
                # Fallback: threshold를 낮춰서 재시도
                if similarity_threshold > 0.3:
                    self.logger.info(f"🔄 Retrying with lower threshold: {similarity_threshold:.3f} → 0.30")
                    fallback_results = self._search_with_threshold(
                        query, k, source_types, 0.30,
                        min_ml_confidence, min_quality_score, filter_by_confidence,
                        chunk_size_category, deduplicate_by_group, embedding_version_id
                    )
                    if fallback_results:
                        self.logger.info(f"✅ Fallback search found {len(fallback_results)} results")
                        results = fallback_results
                
                # 여전히 결과가 없으면 source_types 필터 제거 후 재시도
                if not results and source_types:
                    self.logger.info(f"🔄 Retrying without source_types filter")
                    fallback_results = self._search_with_threshold(
                        query, k, None, max(0.20, similarity_threshold - 0.10),
                        min_ml_confidence, min_quality_score, filter_by_confidence,
                        chunk_size_category, deduplicate_by_group, embedding_version_id
                    )
                    if fallback_results:
                        self.logger.info(f"✅ Fallback search (no source filter) found {len(fallback_results)} results")
                        results = fallback_results
                
                # 원인 분석 (최종적으로 결과가 없을 때만)
                if not results:
                    self._analyze_no_results_cause(query, embedding_version_id, similarity_threshold, source_types)
            
            # 검색 결과 검증 및 복원 (개선 사항 #1, #2, #3)
            if results:
                results = self._validate_and_fix_search_results(results, embedding_version_id)
            
            self.logger.info(f"✅ Semantic search completed: {len(results)} results for query: {query[:50]}")
            return results

        except Exception as e:
            self.logger.error(f"Error in semantic search: {e}", exc_info=True)
            # 에러 발생 시 상세 디버깅 정보
            self._log_search_error_details(e, query, embedding_version_id)
            return []

    def _analyze_no_results_cause(self, 
                                   query: str, 
                                   embedding_version_id: Optional[int],
                                   similarity_threshold: float,
                                   source_types: Optional[List[str]]):
        """
        검색 결과가 없을 때 원인 분석
        
        Args:
            query: 검색 쿼리
            embedding_version_id: 사용된 버전 ID
            similarity_threshold: 유사도 임계값
            source_types: 필터링된 소스 타입
        """
        try:
            self.logger.info("🔍 Analyzing cause of no search results...")
            
            # 1. 버전 데이터 확인
            if embedding_version_id:
                chunk_count = self._get_version_chunk_count(embedding_version_id)
                if chunk_count == 0:
                    self.logger.warning(f"   ❌ Version {embedding_version_id} has no chunks")
                else:
                    self.logger.info(f"   ✅ Version {embedding_version_id} has {chunk_count} chunks")
            else:
                # 전체 버전 확인
                conn = self._get_connection()
                cursor = conn.cursor()
                cursor.execute("SELECT COUNT(*) as count FROM text_chunks")
                row = cursor.fetchone()
                total_chunks = row['count'] if row else 0
                conn.close()
                if total_chunks == 0:
                    self.logger.warning("   ❌ No chunks found in database")
                else:
                    self.logger.info(f"   ✅ Database has {total_chunks} total chunks")
            
            # 2. 임계값 확인
            self.logger.info(f"   📊 Similarity threshold: {similarity_threshold:.3f}")
            if similarity_threshold > 0.7:
                self.logger.warning("   ⚠️  Threshold is quite high, consider lowering it")
            
            # 3. 소스 타입 필터 확인
            if source_types:
                self.logger.info(f"   📊 Filtered source types: {source_types}")
            else:
                self.logger.info("   📊 No source type filter applied")
            
            # 4. 인덱스 상태 확인
            if self.index is None:
                self.logger.warning("   ⚠️  FAISS index not loaded (using slower DB search)")
            else:
                self.logger.info(f"   ✅ FAISS index loaded ({self.index.ntotal} vectors)")
            
            # 5. Embedder 상태 확인
            if not self.embedder:
                self.logger.error("   ❌ Embedder not initialized")
            else:
                self.logger.info("   ✅ Embedder initialized")
                
        except Exception as e:
            self.logger.debug(f"Error analyzing no results cause: {e}")
    
    def _validate_and_fix_search_results(self, 
                                        results: List[Dict[str, Any]], 
                                        expected_version_id: Optional[int] = None) -> List[Dict[str, Any]]:
        """
        검색 결과 검증 및 복원 (개선 사항 #1, #2, #3)
        
        Args:
            results: 검색 결과 리스트
            expected_version_id: 예상되는 버전 ID
            
        Returns:
            검증 및 복원된 검색 결과 리스트
        """
        validated_results = []
        issues_found = {
            'missing_version_id': 0,
            'missing_metadata': 0,
            'poor_text_quality': 0,
            'version_mismatch': 0
        }
        
        # 복원 시도 통계
        restoration_stats = {
            'version_id_restored': 0,
            'version_id_failed': 0,
            'metadata_restored': 0,
            'metadata_failed': 0,
            'text_restored': 0,
            'text_failed': 0
        }
        
        # 배치 조회로 최적화: embedding_version_id가 없는 결과들의 chunk_id 수집
        chunk_ids_to_check = []
        for i, result in enumerate(results):
            version_id = result.get('embedding_version_id') or result.get('metadata', {}).get('embedding_version_id')
            if version_id is None:
                chunk_id = result.get('metadata', {}).get('chunk_id')
                # id 필드에서도 chunk_id 추출 시도 (chunk_xxx 형식)
                if not chunk_id:
                    result_id = result.get('id', '')
                    if result_id.startswith('chunk_'):
                        try:
                            chunk_id = int(result_id.replace('chunk_', ''))
                        except ValueError:
                            pass
                if chunk_id:
                    # numpy 타입을 Python 정수형으로 변환
                    import numpy as np
                    if isinstance(chunk_id, (np.integer, np.int64, np.int32)):
                        chunk_id = int(chunk_id)
                    chunk_ids_to_check.append(chunk_id)
                else:
                    self.logger.debug(f"Result {i+1}: Could not extract chunk_id from result (id={result.get('id')}, metadata keys={list(result.get('metadata', {}).keys())})")
        
        # 배치 조회
        version_id_map = {}
        if chunk_ids_to_check:
            self.logger.debug(f"Batch loading embedding_version_ids for {len(chunk_ids_to_check)} chunks")
            version_id_map = self._batch_load_embedding_version_ids(chunk_ids_to_check)
            self.logger.debug(f"Batch load result: {len(version_id_map)} chunks found with version_ids")
            if len(version_id_map) < len(chunk_ids_to_check):
                missing = set(chunk_ids_to_check) - set(version_id_map.keys())
                self.logger.debug(f"Missing version_ids for chunks: {list(missing)[:10]}")
        
        for i, result in enumerate(results):
            try:
                # 1. embedding_version_id 검증 및 복원 (개선 사항 #1)
                original_version_id = result.get('embedding_version_id') or result.get('metadata', {}).get('embedding_version_id')
                version_id = original_version_id
                if version_id is None:
                    issues_found['missing_version_id'] += 1
                    # 배치 조회 결과에서 확인
                    chunk_id = result.get('metadata', {}).get('chunk_id')
                    # id 필드에서도 chunk_id 추출 시도 (chunk_xxx 형식)
                    if not chunk_id:
                        result_id = result.get('id', '')
                        if result_id.startswith('chunk_'):
                            try:
                                chunk_id = int(result_id.replace('chunk_', ''))
                            except ValueError:
                                pass
                    if chunk_id:
                        if chunk_id in version_id_map:
                            version_id = version_id_map[chunk_id]
                            if version_id:
                                result['embedding_version_id'] = version_id
                                if 'metadata' not in result:
                                    result['metadata'] = {}
                                result['metadata']['embedding_version_id'] = version_id
                                self.logger.debug(f"Restored embedding_version_id={version_id} for chunk_id={chunk_id} (batch)")
                                restoration_stats['version_id_restored'] += 1
                        else:
                            self.logger.debug(f"chunk_id={chunk_id} not found in version_id_map (map size: {len(version_id_map)})")
                    elif chunk_id:
                        # 배치 조회에 없으면 개별 조회 시도 (폴백)
                        try:
                            conn = self._get_connection()
                            cursor = conn.execute(
                                "SELECT embedding_version_id FROM text_chunks WHERE id = ?",
                                (chunk_id,)
                            )
                            row = cursor.fetchone()
                            if row:
                                version_id = row['embedding_version_id']
                                # NULL인 경우 활성 버전 사용
                                if version_id is None:
                                    active_version_id = self._get_active_embedding_version_id()
                                    if active_version_id:
                                        version_id = active_version_id
                                        self.logger.debug(f"Using active version {version_id} for chunk_id={chunk_id} (text_chunks.embedding_version_id is NULL)")
                                
                                if version_id:
                                    result['embedding_version_id'] = version_id
                                    if 'metadata' not in result:
                                        result['metadata'] = {}
                                    result['metadata']['embedding_version_id'] = version_id
                                    self.logger.debug(f"Restored embedding_version_id={version_id} for chunk_id={chunk_id}")
                                    restoration_stats['version_id_restored'] += 1
                            conn.close()
                        except Exception as e:
                            self.logger.debug(f"Failed to restore embedding_version_id for chunk_id={chunk_id}: {e}")
                    
                    if version_id is None:
                        self.logger.warning(f"⚠️  Result {i+1}: embedding_version_id is missing and could not be restored")
                        restoration_stats['version_id_failed'] += 1
                elif version_id != original_version_id:
                    restoration_stats['version_id_restored'] += 1
                
                # 버전 일관성 검증
                if expected_version_id and version_id and version_id != expected_version_id:
                    issues_found['version_mismatch'] += 1
                    self.logger.debug(
                        f"Version mismatch in result {i+1}: expected {expected_version_id}, got {version_id}"
                    )
                
                # 2. 메타데이터 완전성 검증 (개선 사항 #2)
                source_type = result.get('type') or result.get('source_type') or result.get('metadata', {}).get('source_type')
                if not source_type:
                    issues_found['missing_metadata'] += 1
                    self.logger.warning(f"⚠️  Result {i+1}: source_type is missing")
                    continue  # source_type이 없으면 건너뛰기
                
                # 타입별 필수 필드 검증
                required_fields = self._get_required_metadata_fields(source_type)
                missing_fields = []
                for field in required_fields:
                    # 최상위 레벨, metadata, source_meta 모두 확인 (명시적으로 각각 확인)
                    field_value = result.get(field)
                    if not field_value:
                        metadata_dict = result.get('metadata', {})
                        if isinstance(metadata_dict, dict):
                            field_value = metadata_dict.get(field)
                    if not field_value and 'source_meta' in result:
                        source_meta_dict = result.get('source_meta', {})
                        if isinstance(source_meta_dict, dict):
                            field_value = source_meta_dict.get(field)
                    
                    # statute_article의 경우 별칭 필드도 확인
                    if not field_value and source_type == 'statute_article':
                        if field == 'law_name':
                            field_value = result.get('statute_name')
                            if not field_value:
                                metadata_dict = result.get('metadata', {})
                                if isinstance(metadata_dict, dict):
                                    field_value = metadata_dict.get('statute_name')
                        elif field == 'article_number':
                            field_value = result.get('article_no')
                            if not field_value:
                                metadata_dict = result.get('metadata', {})
                                if isinstance(metadata_dict, dict):
                                    field_value = metadata_dict.get('article_no')
                    
                    # 필드 값이 있는지 확인 (None, 빈 문자열, 빈 리스트 등 제외)
                    if field_value is None or (isinstance(field_value, str) and len(field_value.strip()) == 0):
                        missing_fields.append(field)
                
                if missing_fields:
                    issues_found['missing_metadata'] += 1
                    self.logger.warning(
                        f"⚠️  Result {i+1} ({source_type}): Missing required fields: {missing_fields}"
                    )
                    # 필수 필드 복원 시도
                    self._restore_missing_metadata(result, source_type, missing_fields)
                    
                    # 복원 후 재검증
                    still_missing = []
                    for field in missing_fields:
                        # 최상위 레벨, metadata, source_meta 모두 확인 (명시적으로 각각 확인)
                        field_value = result.get(field)
                        if not field_value:
                            metadata_dict = result.get('metadata', {})
                            if isinstance(metadata_dict, dict):
                                field_value = metadata_dict.get(field)
                        if not field_value and 'source_meta' in result:
                            source_meta_dict = result.get('source_meta', {})
                            if isinstance(source_meta_dict, dict):
                                field_value = source_meta_dict.get(field)
                        
                        # statute_article의 경우 별칭 필드도 확인
                        if not field_value and source_type == 'statute_article':
                            if field == 'law_name':
                                field_value = result.get('statute_name')
                                if not field_value:
                                    metadata_dict = result.get('metadata', {})
                                    if isinstance(metadata_dict, dict):
                                        field_value = metadata_dict.get('statute_name')
                            elif field == 'article_number':
                                field_value = result.get('article_no')
                                if not field_value:
                                    metadata_dict = result.get('metadata', {})
                                    if isinstance(metadata_dict, dict):
                                        field_value = metadata_dict.get('article_no')
                        
                        # 필드 값이 있는지 확인 (None, 빈 문자열, 빈 리스트 등 제외)
                        if field_value is None or (isinstance(field_value, str) and len(field_value.strip()) == 0):
                            self.logger.debug(
                                f"Field {field} still missing after restoration: "
                                f"result.get('{field}')={result.get(field)}, "
                                f"result.get('metadata', {{}}).get('{field}')={result.get('metadata', {}).get(field) if isinstance(result.get('metadata'), dict) else 'N/A'}"
                            )
                            still_missing.append(field)
                        else:
                            self.logger.debug(f"✅ Field {field} found after restoration: {field_value}")
                    
                    if still_missing:
                        self.logger.warning(
                            f"⚠️  Result {i+1} ({source_type}): Still missing fields after restoration: {still_missing}"
                        )
                        restoration_stats['metadata_failed'] += len(still_missing)
                        # 복원된 필드 수 계산
                        restored_count = len(missing_fields) - len(still_missing)
                        if restored_count > 0:
                            restoration_stats['metadata_restored'] += restored_count
                    else:
                        self.logger.debug(
                            f"✅ Result {i+1} ({source_type}): All required fields restored successfully"
                        )
                        restoration_stats['metadata_restored'] += len(missing_fields)
                
                # 3. 텍스트 품질 검증 (개선 사항 #3)
                text = result.get('text') or result.get('content') or result.get('metadata', {}).get('text') or result.get('metadata', {}).get('content')
                if not text or len(str(text).strip()) == 0:
                    issues_found['poor_text_quality'] += 1
                    self.logger.warning(f"⚠️  Result {i+1}: Text is empty")
                    continue  # 텍스트가 없으면 건너뛰기
                
                text_length = len(str(text).strip())
                source_type = result.get('type') or result.get('metadata', {}).get('source_type')
                # 타입별 최소 길이 차등 적용 (P1-4: 더욱 완화 - 10자 → 5자)
                if source_type == 'statute_article':
                    min_text_length = 30
                elif source_type in ['case_paragraph', 'decision_paragraph']:
                    min_text_length = 5
                else:
                    min_text_length = 50
                if text_length < min_text_length:
                    issues_found['poor_text_quality'] += 1
                    self.logger.warning(
                        f"⚠️  Result {i+1}: Text is too short ({text_length} chars, minimum: {min_text_length})"
                    )
                    # 텍스트 복원 시도
                    chunk_id = result.get('metadata', {}).get('chunk_id')
                    source_id = result.get('metadata', {}).get('source_id')
                    if chunk_id and source_id:
                        try:
                            conn = self._get_connection()
                            restored_text = self._ensure_text_content(
                                conn, chunk_id, text, source_type, source_id, result.get('metadata', {})
                            )
                            if restored_text and len(restored_text.strip()) > text_length:
                                result['text'] = restored_text
                                result['content'] = restored_text
                                if 'metadata' in result:
                                    result['metadata']['text'] = restored_text
                                    result['metadata']['content'] = restored_text
                                self.logger.info(
                                    f"✅ Restored text for result {i+1} (length: {len(restored_text)} chars)"
                                )
                                restoration_stats['text_restored'] += 1
                            conn.close()
                        except Exception as e:
                            self.logger.debug(f"Failed to restore text for result {i+1}: {e}")
                    
                    # 복원 후에도 짧으면 건너뛰기 (타입별 최소 길이 차등 적용)
                    final_text = result.get('text') or result.get('content', '')
                    source_type = result.get('type') or result.get('metadata', {}).get('source_type')
                    if source_type == 'statute_article':
                        effective_min_length = 50
                    elif source_type in ['case_paragraph', 'decision_paragraph']:
                        effective_min_length = 30
                    else:
                        effective_min_length = min_text_length
                    if len(final_text.strip()) < effective_min_length:
                        self.logger.warning(f"⚠️  Result {i+1}: Text still too short after restoration, skipping")
                        restoration_stats['text_failed'] += 1
                        continue
                
                # 검증 통과한 결과 추가
                validated_results.append(result)
                
            except Exception as e:
                self.logger.error(f"Error validating result {i+1}: {e}")
                continue
        
        # 검증 결과 요약
        if any(issues_found.values()):
            total_issues = sum(issues_found.values())
            self.logger.warning(
                f"⚠️  Search result validation: {total_issues} issues found "
                f"(missing_version_id: {issues_found['missing_version_id']}, "
                f"missing_metadata: {issues_found['missing_metadata']}, "
                f"poor_text_quality: {issues_found['poor_text_quality']}, "
                f"version_mismatch: {issues_found['version_mismatch']})"
            )
            self.logger.info(
                f"✅ Validated {len(validated_results)}/{len(results)} results passed validation"
            )
        else:
            self.logger.info(f"✅ All {len(validated_results)} results passed validation")
        
        # 복원 통계 로깅
        if any(restoration_stats.values()):
            self.logger.info(
                f"📊 Restoration statistics: "
                f"version_id: {restoration_stats['version_id_restored']} restored, "
                f"{restoration_stats['version_id_failed']} failed; "
                f"metadata: {restoration_stats['metadata_restored']} restored, "
                f"{restoration_stats['metadata_failed']} failed; "
                f"text: {restoration_stats['text_restored']} restored, "
                f"{restoration_stats['text_failed']} failed"
            )
        
        return validated_results
    
    def _batch_load_embedding_version_ids(self, chunk_ids: List[int]) -> Dict[int, Optional[int]]:
        """
        여러 chunk_id의 embedding_version_id를 배치로 조회
        
        Args:
            chunk_ids: 조회할 chunk_id 리스트
            
        Returns:
            {chunk_id: embedding_version_id} 딕셔너리
        """
        if not chunk_ids:
            return {}
        
        try:
            conn = self._get_connection()
            # row_factory 설정 확인
            if not hasattr(conn, 'row_factory') or conn.row_factory is None:
                import sqlite3
                conn.row_factory = sqlite3.Row
            
            # numpy 타입을 Python 정수형으로 변환
            import numpy as np
            chunk_ids_python = [int(cid) if isinstance(cid, (np.integer, np.int64, np.int32)) else cid for cid in chunk_ids]
            placeholders = ",".join(["?"] * len(chunk_ids_python))
            query = f"SELECT id, embedding_version_id FROM text_chunks WHERE id IN ({placeholders})"
            self.logger.debug(f"Batch query: {query[:100]}... with {len(chunk_ids_python)} chunk_ids (sample: {chunk_ids_python[:3]})")
            cursor = conn.execute(query, chunk_ids_python)
            rows = cursor.fetchall()
            
            # 실제로 쿼리가 실행되었는지 확인
            if len(rows) == 0:
                # 샘플 chunk_id로 직접 조회 시도
                sample_id = chunk_ids_python[0] if chunk_ids_python else None
                if sample_id:
                    test_cursor = conn.execute("SELECT id, embedding_version_id FROM text_chunks WHERE id = ?", (sample_id,))
                    test_row = test_cursor.fetchone()
                    if test_row:
                        self.logger.debug(f"Direct query for chunk_id={sample_id} succeeded, but batch query failed")
                    else:
                        self.logger.debug(f"Direct query for chunk_id={sample_id} also returned no rows - chunk may not exist")
            
            conn.close()
            
            result = {}
            active_version_id = self._get_active_embedding_version_id()
            
            self.logger.debug(f"Batch query returned {len(rows)} rows")
            for row in rows:
                version_id = row['embedding_version_id']
                if version_id is None and active_version_id:
                    version_id = active_version_id
                result[row['id']] = version_id
                self.logger.debug(f"  chunk_id={row['id']}, embedding_version_id={version_id}")
            
            if len(result) < len(chunk_ids):
                missing = set(chunk_ids) - set(result.keys())
                self.logger.debug(f"Missing chunks in batch result: {list(missing)[:10]}")
            
            return result
        except Exception as e:
            self.logger.error(f"Failed to batch load embedding_version_ids: {e}", exc_info=True)
            return {}
    
    def _get_required_metadata_fields(self, source_type: str) -> List[str]:
        """
        타입별 필수 메타데이터 필드 반환
        
        Args:
            source_type: 소스 타입
            
        Returns:
            필수 필드 리스트
        """
        required_fields_map = {
            'statute_article': ['statute_name', 'article_no', 'law_name'],
            'case_paragraph': ['doc_id', 'casenames', 'court'],
            'decision_paragraph': ['org', 'doc_id'],
            'interpretation_paragraph': ['title', 'interpretation_id']
        }
        return required_fields_map.get(source_type, [])
    
    def _restore_missing_metadata(self, result: Dict[str, Any], source_type: str, missing_fields: List[str]):
        """
        누락된 메타데이터 복원 시도
        
        Args:
            result: 검색 결과 딕셔너리
            source_type: 소스 타입
            missing_fields: 누락된 필드 리스트
        """
        try:
            chunk_id = result.get('metadata', {}).get('chunk_id')
            source_id = result.get('metadata', {}).get('source_id')
            
            if not chunk_id:
                return
            
            # source_id가 None인 경우 chunk_id로 조회 (강화)
            if not source_id:
                conn = self._get_connection()
                if conn:
                    try:
                        cursor = conn.execute(
                            "SELECT source_id, source_type FROM text_chunks WHERE id = ?",
                            (chunk_id,)
                        )
                        row = cursor.fetchone()
                        if row and row['source_id']:
                            source_id = row['source_id']
                            if not source_type:
                                source_type = row['source_type'] or source_type
                            self.logger.debug(f"✅ Restored source_id={source_id} for chunk_id={chunk_id} from text_chunks")
                        else:
                            # source_id가 None인 경우에도 계속 진행 (다른 방법으로 복원 시도)
                            self.logger.debug(f"⚠️  source_id is None for chunk_id={chunk_id}, will try alternative restoration methods")
                        conn.close()
                    except Exception as e:
                        self.logger.debug(f"Failed to get source_id from text_chunks for chunk_id={chunk_id}: {e}")
                        if conn:
                            conn.close()
                        # source_id가 없어도 계속 진행 (다른 방법으로 복원 시도)
                        pass
                else:
                    # conn이 없어도 계속 진행 (다른 방법으로 복원 시도)
                    pass
            
            # source_id가 여전히 None인 경우에도 복원 시도 (chunk_id로 직접 조회)
            if not source_id and chunk_id:
                # chunk_id로 직접 메타데이터 조회 시도
                conn = self._get_connection()
                if conn:
                    try:
                        # text_chunks에서 직접 조회
                        cursor = conn.execute(
                            "SELECT source_id, source_type, text FROM text_chunks WHERE id = ?",
                            (chunk_id,)
                        )
                        row = cursor.fetchone()
                        if row:
                            if not source_id and row['source_id']:
                                source_id = row['source_id']
                            if not source_type and row['source_type']:
                                source_type = row['source_type']
                            # text도 복원 시도 (Empty text content 해결)
                            if row['text']:
                                restored_text = row['text']
                                if len(restored_text.strip()) > 0:
                                    result['text'] = restored_text
                                    result['content'] = restored_text
                                    if 'metadata' not in result:
                                        result['metadata'] = {}
                                    result['metadata']['text'] = restored_text
                                    result['metadata']['content'] = restored_text
                                    self.logger.debug(f"✅ Restored text for chunk_id={chunk_id} from text_chunks (length: {len(restored_text)} chars)")
                        conn.close()
                    except Exception as e:
                        self.logger.debug(f"Failed to get metadata from text_chunks for chunk_id={chunk_id}: {e}")
                        if conn:
                            conn.close()
            
            # source_id가 여전히 None이면 복원 불가능하므로 경고만 출력하고 계속 진행
            if not source_id:
                self.logger.warning(f"⚠️  source_id is None for chunk_id={chunk_id}, metadata restoration may be incomplete")
                # source_id가 없어도 기본 메타데이터 복원은 시도
            
            conn = self._get_connection()
            if not conn:
                return
            
            # 먼저 text_chunks.meta에서 확인
            chunk_metadata_json = None
            if chunk_id:
                try:
                    cursor_meta = conn.execute(
                        "SELECT meta FROM text_chunks WHERE id = ?",
                        (chunk_id,)
                    )
                    meta_row = cursor_meta.fetchone()
                    if meta_row and meta_row['meta']:
                        try:
                            import json
                            chunk_metadata_json = json.loads(meta_row['meta'])
                        except Exception as e:
                            self.logger.debug(f"Failed to parse meta JSON for chunk_id={chunk_id}: {e}")
                except Exception as e:
                    self.logger.debug(f"Failed to get meta for chunk_id={chunk_id}: {e}")
            
            # _get_source_metadata를 사용하여 소스 테이블에서 메타데이터 조회
            # source_id가 None인 경우에도 복원 시도
            source_meta = {}
            if source_id:
                source_meta = self._get_source_metadata(conn, source_type, source_id)
            elif chunk_id:
                # source_id가 None인 경우 chunk_id로 직접 조회 시도
                try:
                    # text_chunks에서 source_id 복원 시도
                    cursor_source = conn.execute(
                        "SELECT source_id, source_type FROM text_chunks WHERE id = ?",
                        (chunk_id,)
                    )
                    source_row = cursor_source.fetchone()
                    if source_row and source_row['source_id']:
                        restored_source_id = source_row['source_id']
                        if not source_type and source_row['source_type']:
                            source_type = source_row['source_type']
                        # 복원된 source_id로 메타데이터 조회
                        source_meta = self._get_source_metadata(conn, source_type, restored_source_id)
                        source_id = restored_source_id
                        self.logger.debug(f"✅ Restored source_id={source_id} for chunk_id={chunk_id} and retrieved metadata")
                except Exception as e:
                    self.logger.debug(f"Failed to restore source_id and metadata for chunk_id={chunk_id}: {e}")
            
            # source_id가 None인 경우에도 chunk_id로 직접 복원 시도 (우선 처리)
            if not source_id and chunk_id:
                try:
                    # case_paragraph의 경우: doc_id, casenames, court 복원
                    if source_type == 'case_paragraph':
                        cursor_case = conn.execute("""
                            SELECT cp.case_id, c.casenames, c.doc_id, c.court
                            FROM text_chunks tc
                            JOIN case_paragraphs cp ON tc.source_id = cp.id
                            JOIN cases c ON cp.case_id = c.id
                            WHERE tc.id = ? AND tc.source_type = 'case_paragraph'
                        """, (chunk_id,))
                        case_row = cursor_case.fetchone()
                        if case_row:
                            if 'doc_id' in missing_fields and case_row['doc_id']:
                                result['doc_id'] = case_row['doc_id']
                                if 'metadata' not in result:
                                    result['metadata'] = {}
                                result['metadata']['doc_id'] = case_row['doc_id']
                            if 'casenames' in missing_fields and case_row['casenames']:
                                result['casenames'] = case_row['casenames']
                                if 'metadata' not in result:
                                    result['metadata'] = {}
                                result['metadata']['casenames'] = case_row['casenames']
                            if 'court' in missing_fields and case_row['court']:
                                result['court'] = case_row['court']
                                if 'metadata' not in result:
                                    result['metadata'] = {}
                                result['metadata']['court'] = case_row['court']
                            self.logger.debug(f"✅ Restored case metadata for chunk_id={chunk_id} (doc_id, casenames, court)")
                    
                    # decision_paragraph의 경우: org, doc_id 복원
                    elif source_type == 'decision_paragraph':
                        cursor_decision = conn.execute("""
                            SELECT dp.decision_id, d.org, d.doc_id
                            FROM text_chunks tc
                            JOIN decision_paragraphs dp ON tc.source_id = dp.id
                            JOIN decisions d ON dp.decision_id = d.id
                            WHERE tc.id = ? AND tc.source_type = 'decision_paragraph'
                        """, (chunk_id,))
                        decision_row = cursor_decision.fetchone()
                        if decision_row:
                            if 'org' in missing_fields and decision_row['org']:
                                result['org'] = decision_row['org']
                                if 'metadata' not in result:
                                    result['metadata'] = {}
                                result['metadata']['org'] = decision_row['org']
                            if 'doc_id' in missing_fields and decision_row['doc_id']:
                                result['doc_id'] = decision_row['doc_id']
                                if 'metadata' not in result:
                                    result['metadata'] = {}
                                result['metadata']['doc_id'] = decision_row['doc_id']
                            self.logger.debug(f"✅ Restored decision metadata for chunk_id={chunk_id} (org, doc_id)")
                    
                    # interpretation_paragraph의 경우: interpretation_id, doc_id 복원
                    elif source_type == 'interpretation_paragraph':
                        cursor_interp = conn.execute("""
                            SELECT ip.interpretation_id, i.doc_id
                            FROM text_chunks tc
                            JOIN interpretation_paragraphs ip ON tc.source_id = ip.id
                            JOIN interpretations i ON ip.interpretation_id = i.id
                            WHERE tc.id = ? AND tc.source_type = 'interpretation_paragraph'
                        """, (chunk_id,))
                        interp_row = cursor_interp.fetchone()
                        if interp_row:
                            if 'interpretation_id' in missing_fields and interp_row['interpretation_id']:
                                result['interpretation_id'] = interp_row['interpretation_id']
                                if 'metadata' not in result:
                                    result['metadata'] = {}
                                result['metadata']['interpretation_id'] = interp_row['interpretation_id']
                            if 'doc_id' in missing_fields and interp_row['doc_id']:
                                result['doc_id'] = interp_row['doc_id']
                                if 'metadata' not in result:
                                    result['metadata'] = {}
                                result['metadata']['doc_id'] = interp_row['doc_id']
                            self.logger.debug(f"✅ Restored interpretation metadata for chunk_id={chunk_id} (interpretation_id, doc_id)")
                except Exception as e:
                    self.logger.debug(f"Failed to restore metadata via chunk_id for chunk_id={chunk_id}: {e}")
            
            # 누락된 필드 복원 (일반적인 방법)
            for field in missing_fields:
                # 이미 복원된 필드는 건너뛰기
                if result.get(field) or result.get('metadata', {}).get(field):
                    continue
                
                field_value = None
                restoration_source = None
                
                # 1. chunk metadata JSON에서 확인 (우선순위 1)
                if chunk_metadata_json and field in chunk_metadata_json:
                    field_value = chunk_metadata_json[field]
                    restoration_source = 'chunk_meta'
                
                # 2. source_meta에서 확인 (우선순위 2)
                if not field_value and source_meta and field in source_meta:
                    field_value = source_meta[field]
                    restoration_source = 'source_table'
                
                # 3. 같은 source_id의 다른 청크에서 확인 (대안, source_id가 있는 경우만)
                if not field_value and source_id:
                    try:
                        cursor_alt = conn.execute("""
                            SELECT meta FROM text_chunks
                            WHERE source_type = ? AND source_id = ? AND meta IS NOT NULL AND meta != ''
                            LIMIT 1
                        """, (source_type, source_id))
                        alt_row = cursor_alt.fetchone()
                        if alt_row and alt_row['meta']:
                            try:
                                import json
                                alt_metadata = json.loads(alt_row['meta'])
                                if field in alt_metadata and alt_metadata[field]:
                                    field_value = alt_metadata[field]
                                    restoration_source = 'alternative_chunk'
                            except Exception as e:
                                self.logger.debug(f"Failed to parse alternative chunk meta: {e}")
                    except Exception as e:
                        self.logger.debug(f"Failed to query alternative chunk for field {field}: {e}")
                
                # 복원 성공 여부 확인 및 반영
                if field_value:
                    result[field] = field_value
                    if 'metadata' not in result:
                        result['metadata'] = {}
                    result['metadata'][field] = field_value
                    
                    # 별칭 필드도 설정
                    if field == 'statute_name':
                        result['law_name'] = field_value
                        result['metadata']['law_name'] = field_value
                    elif field == 'article_no':
                        result['article_number'] = field_value
                        result['metadata']['article_number'] = field_value
                    
                    self.logger.debug(f"✅ Restored {field}={field_value} for chunk_id={chunk_id} from {restoration_source}")
                else:
                    # statute_article의 경우 추가 시도: source_id로 직접 조회
                    if source_type == 'statute_article' and field in ['statute_name', 'article_no', 'law_name']:
                        try:
                            if field == 'law_name':
                                # law_name은 statute_name과 동일
                                if 'statute_name' in result or result.get('metadata', {}).get('statute_name'):
                                    field_value = result.get('statute_name') or result.get('metadata', {}).get('statute_name')
                                    if field_value:
                                        result['law_name'] = field_value
                                        if 'metadata' not in result:
                                            result['metadata'] = {}
                                        result['metadata']['law_name'] = field_value
                                        self.logger.debug(f"✅ Restored {field}={field_value} for chunk_id={chunk_id} from statute_name alias")
                                        continue
                            
                            # source_id로 직접 조회 시도
                            cursor_direct = conn.execute("""
                                SELECT sa.article_no, s.name as statute_name
                                FROM statute_articles sa
                                JOIN statutes s ON sa.statute_id = s.id
                                WHERE sa.id = ?
                            """, (source_id,))
                            direct_row = cursor_direct.fetchone()
                            if direct_row:
                                if field == 'statute_name' or field == 'law_name':
                                    field_value = direct_row['statute_name']
                                    if field_value:
                                        result['statute_name'] = field_value
                                        result['law_name'] = field_value
                                        if 'metadata' not in result:
                                            result['metadata'] = {}
                                        result['metadata']['statute_name'] = field_value
                                        result['metadata']['law_name'] = field_value
                                        self.logger.debug(f"✅ Restored {field}={field_value} for chunk_id={chunk_id} from direct query")
                                elif field == 'article_no' or field == 'article_number':
                                    field_value = direct_row['article_no']
                                    if field_value:
                                        result['article_no'] = field_value
                                        result['article_number'] = field_value
                                        if 'metadata' not in result:
                                            result['metadata'] = {}
                                        result['metadata']['article_no'] = field_value
                                        result['metadata']['article_number'] = field_value
                                        self.logger.debug(f"✅ Restored {field}={field_value} for chunk_id={chunk_id} from direct query")
                                        # 복원 성공했으므로 다음 필드로
                                        continue
                        except Exception as e:
                            self.logger.debug(f"Failed to restore {field} via direct query for chunk_id={chunk_id}: {e}")
                    
                    # case_paragraph의 경우 추가 시도: source_id로 직접 조회
                    if source_type == 'case_paragraph' and field == 'court' and not field_value:
                        try:
                            # source_id가 None인 경우, chunk_id로 먼저 조회
                            actual_source_id = source_id
                            if not actual_source_id and chunk_id:
                                cursor_source = conn.execute("""
                                    SELECT source_id FROM text_chunks WHERE id = ? AND source_type = 'case_paragraph'
                                """, (chunk_id,))
                                source_row = cursor_source.fetchone()
                                if source_row and source_row['source_id']:
                                    actual_source_id = source_row['source_id']
                            
                            court_row = None
                            
                            # 방법 1: cases 테이블에서 직접 조회 (source_id가 cases.id인 경우)
                            if actual_source_id:
                                cursor_court = conn.execute("""
                                    SELECT court FROM cases WHERE id = ?
                                """, (actual_source_id,))
                                court_row = cursor_court.fetchone()
                            
                            # 방법 2: case_paragraphs를 통한 조회 (source_id가 case_paragraphs.id인 경우)
                            if (not court_row or not court_row['court']) and actual_source_id:
                                cursor_court = conn.execute("""
                                    SELECT c.court
                                    FROM case_paragraphs cp
                                    JOIN cases c ON cp.case_id = c.id
                                    WHERE cp.id = ?
                                """, (actual_source_id,))
                                court_row = cursor_court.fetchone()
                            
                            # 방법 3: text_chunks를 통해 case_id를 찾아서 조회 (source_id가 None이거나 실패한 경우)
                            if (not court_row or not court_row['court']) and chunk_id:
                                try:
                                    # text_chunks -> case_paragraphs -> cases 경로
                                    cursor_chunk = conn.execute("""
                                        SELECT cp.case_id
                                        FROM text_chunks tc
                                        JOIN case_paragraphs cp ON tc.source_id = cp.id
                                        WHERE tc.id = ? AND tc.source_type = 'case_paragraph'
                                    """, (chunk_id,))
                                    chunk_row = cursor_chunk.fetchone()
                                    if chunk_row and chunk_row['case_id']:
                                        case_id = chunk_row['case_id']
                                        cursor_court = conn.execute("""
                                            SELECT court FROM cases WHERE id = ?
                                        """, (case_id,))
                                        court_row = cursor_court.fetchone()
                                except Exception as e:
                                    self.logger.debug(f"Failed to get case_id from text_chunks for chunk_id={chunk_id}: {e}")
                            
                            # 방법 4: chunk_id로 직접 case_paragraphs 조회 (source_id가 없는 경우)
                            if (not court_row or not court_row['court']) and chunk_id and not actual_source_id:
                                try:
                                    cursor_chunk = conn.execute("""
                                        SELECT cp.case_id
                                        FROM text_chunks tc
                                        JOIN case_paragraphs cp ON tc.source_id = cp.id
                                        WHERE tc.id = ? AND tc.source_type = 'case_paragraph'
                                    """, (chunk_id,))
                                    chunk_row = cursor_chunk.fetchone()
                                    if chunk_row and chunk_row['case_id']:
                                        case_id = chunk_row['case_id']
                                        cursor_court = conn.execute("""
                                            SELECT court FROM cases WHERE id = ?
                                        """, (case_id,))
                                        court_row = cursor_court.fetchone()
                                except Exception as e:
                                    self.logger.debug(f"Failed to get case_id via chunk_id for chunk_id={chunk_id}: {e}")
                            
                            if court_row and court_row['court']:
                                field_value = court_row['court']
                                restoration_source = 'direct_court_query'
                                result['court'] = field_value
                                if 'metadata' not in result:
                                    result['metadata'] = {}
                                result['metadata']['court'] = field_value
                                self.logger.debug(f"✅ Restored {field}={field_value} for chunk_id={chunk_id} from direct court query")
                            else:
                                # 모든 방법이 실패한 경우 기본값 설정
                                field_value = "알 수 없음"
                                result['court'] = field_value
                                if 'metadata' not in result:
                                    result['metadata'] = {}
                                result['metadata']['court'] = field_value
                                self.logger.debug(f"⚠️  Could not restore {field} for chunk_id={chunk_id}, using default value")
                        except Exception as e:
                            self.logger.debug(f"Failed to restore {field} via direct query for chunk_id={chunk_id}: {e}")
                            # 예외 발생 시에도 기본값 설정
                            if not result.get('court') and not result.get('metadata', {}).get('court'):
                                result['court'] = "알 수 없음"
                                if 'metadata' not in result:
                                    result['metadata'] = {}
                                result['metadata']['court'] = "알 수 없음"
                    
                    # case_paragraph의 casenames 복원 (source_id=None인 경우도 처리)
                    if source_type == 'case_paragraph' and field == 'casenames' and not field_value:
                        try:
                            actual_source_id = source_id
                            if not actual_source_id and chunk_id:
                                cursor_source = conn.execute("""
                                    SELECT source_id FROM text_chunks WHERE id = ? AND source_type = 'case_paragraph'
                                """, (chunk_id,))
                                source_row = cursor_source.fetchone()
                                if source_row and source_row['source_id']:
                                    actual_source_id = source_row['source_id']
                            
                            # source_id가 없으면 chunk_id로 직접 조회
                            if not actual_source_id and chunk_id:
                                try:
                                    # 방법 1: text_chunks -> case_paragraphs -> cases 경로
                                    cursor_direct = conn.execute("""
                                        SELECT cp.case_id, c.casenames, c.doc_id, c.court
                                        FROM text_chunks tc
                                        JOIN case_paragraphs cp ON tc.source_id = cp.id
                                        JOIN cases c ON cp.case_id = c.id
                                        WHERE tc.id = ? AND tc.source_type = 'case_paragraph'
                                    """, (chunk_id,))
                                    direct_row = cursor_direct.fetchone()
                                    if direct_row:
                                        # casenames 복원
                                        if field == 'casenames' and direct_row['casenames']:
                                            field_value = direct_row['casenames']
                                            result['casenames'] = field_value
                                            if 'metadata' not in result:
                                                result['metadata'] = {}
                                            result['metadata']['casenames'] = field_value
                                            self.logger.debug(f"✅ Restored casenames={field_value} for chunk_id={chunk_id} (via chunk_id)")
                                        # doc_id 복원
                                        if 'doc_id' in missing_fields and direct_row['doc_id']:
                                            result['doc_id'] = direct_row['doc_id']
                                            if 'metadata' not in result:
                                                result['metadata'] = {}
                                            result['metadata']['doc_id'] = direct_row['doc_id']
                                            self.logger.debug(f"✅ Restored doc_id={direct_row['doc_id']} for chunk_id={chunk_id} (via chunk_id)")
                                        # court 복원
                                        if 'court' in missing_fields and direct_row['court']:
                                            result['court'] = direct_row['court']
                                            if 'metadata' not in result:
                                                result['metadata'] = {}
                                            result['metadata']['court'] = direct_row['court']
                                            self.logger.debug(f"✅ Restored court={direct_row['court']} for chunk_id={chunk_id} (via chunk_id)")
                                except Exception as e:
                                    self.logger.debug(f"Failed to restore case metadata via chunk_id for chunk_id={chunk_id}: {e}")
                            
                            if actual_source_id:
                                # case_paragraphs를 통해 cases 조회
                                cursor_casenames = conn.execute("""
                                    SELECT c.casenames
                                    FROM case_paragraphs cp
                                    JOIN cases c ON cp.case_id = c.id
                                    WHERE cp.id = ?
                                """, (actual_source_id,))
                                casenames_row = cursor_casenames.fetchone()
                                if not casenames_row or not casenames_row['casenames']:
                                    # cases 테이블에서 직접 조회
                                    cursor_casenames = conn.execute("""
                                        SELECT casenames FROM cases WHERE id = ?
                                    """, (actual_source_id,))
                                    casenames_row = cursor_casenames.fetchone()
                                
                                if casenames_row and casenames_row['casenames']:
                                    field_value = casenames_row['casenames']
                                    result['casenames'] = field_value
                                    if 'metadata' not in result:
                                        result['metadata'] = {}
                                    result['metadata']['casenames'] = field_value
                                    self.logger.debug(f"✅ Restored casenames={field_value} for chunk_id={chunk_id}")
                        except Exception as e:
                            self.logger.debug(f"Failed to restore casenames for chunk_id={chunk_id}: {e}")
                    
                    # decision_paragraph의 org 복원 (source_id=None인 경우도 처리)
                    if source_type == 'decision_paragraph' and field == 'org' and not field_value:
                        try:
                            actual_source_id = source_id
                            if not actual_source_id and chunk_id:
                                cursor_source = conn.execute("""
                                    SELECT source_id FROM text_chunks WHERE id = ? AND source_type = 'decision_paragraph'
                                """, (chunk_id,))
                                source_row = cursor_source.fetchone()
                                if source_row and source_row['source_id']:
                                    actual_source_id = source_row['source_id']
                            
                            # source_id가 없으면 chunk_id로 직접 조회
                            if not actual_source_id and chunk_id:
                                try:
                                    # 방법 1: text_chunks -> decision_paragraphs -> decisions 경로
                                    cursor_direct = conn.execute("""
                                        SELECT dp.decision_id, d.org, d.doc_id
                                        FROM text_chunks tc
                                        JOIN decision_paragraphs dp ON tc.source_id = dp.id
                                        JOIN decisions d ON dp.decision_id = d.id
                                        WHERE tc.id = ? AND tc.source_type = 'decision_paragraph'
                                    """, (chunk_id,))
                                    direct_row = cursor_direct.fetchone()
                                    if direct_row:
                                        # org 복원
                                        if field == 'org' and direct_row['org']:
                                            field_value = direct_row['org']
                                            result['org'] = field_value
                                            if 'metadata' not in result:
                                                result['metadata'] = {}
                                            result['metadata']['org'] = field_value
                                            self.logger.debug(f"✅ Restored org={field_value} for chunk_id={chunk_id} (via chunk_id)")
                                        # doc_id 복원
                                        if 'doc_id' in missing_fields and direct_row['doc_id']:
                                            result['doc_id'] = direct_row['doc_id']
                                            if 'metadata' not in result:
                                                result['metadata'] = {}
                                            result['metadata']['doc_id'] = direct_row['doc_id']
                                            self.logger.debug(f"✅ Restored doc_id={direct_row['doc_id']} for chunk_id={chunk_id} (via chunk_id)")
                                except Exception as e:
                                    self.logger.debug(f"Failed to restore decision metadata via chunk_id for chunk_id={chunk_id}: {e}")
                            
                            if actual_source_id:
                                # decision_paragraphs를 통해 decisions 조회
                                cursor_org = conn.execute("""
                                    SELECT d.org
                                    FROM decision_paragraphs dp
                                    JOIN decisions d ON dp.decision_id = d.id
                                    WHERE dp.id = ?
                                """, (actual_source_id,))
                                org_row = cursor_org.fetchone()
                                if not org_row or not org_row['org']:
                                    # decisions 테이블에서 직접 조회
                                    cursor_org = conn.execute("""
                                        SELECT org FROM decisions WHERE id = ?
                                    """, (actual_source_id,))
                                    org_row = cursor_org.fetchone()
                                
                                if org_row and org_row['org']:
                                    field_value = org_row['org']
                                    result['org'] = field_value
                                    if 'metadata' not in result:
                                        result['metadata'] = {}
                                    result['metadata']['org'] = field_value
                                    self.logger.debug(f"✅ Restored org={field_value} for chunk_id={chunk_id}")
                        except Exception as e:
                            self.logger.debug(f"Failed to restore org for chunk_id={chunk_id}: {e}")
                    
                    # interpretation_paragraph의 interpretation_id 복원 (source_id=None인 경우도 처리)
                    if source_type == 'interpretation_paragraph' and field == 'interpretation_id' and not field_value:
                        try:
                            actual_source_id = source_id
                            if not actual_source_id and chunk_id:
                                cursor_source = conn.execute("""
                                    SELECT source_id FROM text_chunks WHERE id = ? AND source_type = 'interpretation_paragraph'
                                """, (chunk_id,))
                                source_row = cursor_source.fetchone()
                                if source_row and source_row['source_id']:
                                    actual_source_id = source_row['source_id']
                            
                            # source_id가 없으면 chunk_id로 직접 조회
                            if not actual_source_id and chunk_id:
                                cursor_direct = conn.execute("""
                                    SELECT ip.interpretation_id
                                    FROM text_chunks tc
                                    JOIN interpretation_paragraphs ip ON tc.source_id = ip.id
                                    WHERE tc.id = ? AND tc.source_type = 'interpretation_paragraph'
                                """, (chunk_id,))
                                direct_row = cursor_direct.fetchone()
                                if direct_row and direct_row['interpretation_id']:
                                    field_value = direct_row['interpretation_id']
                                    result['interpretation_id'] = field_value
                                    if 'metadata' not in result:
                                        result['metadata'] = {}
                                    result['metadata']['interpretation_id'] = field_value
                                    self.logger.debug(f"✅ Restored interpretation_id={field_value} for chunk_id={chunk_id} (via chunk_id)")
                            
                            if actual_source_id:
                                # interpretation_paragraphs에서 interpretation_id 조회
                                cursor_interp = conn.execute("""
                                    SELECT interpretation_id FROM interpretation_paragraphs WHERE id = ?
                                """, (actual_source_id,))
                                interp_row = cursor_interp.fetchone()
                                if interp_row and interp_row['interpretation_id']:
                                    field_value = interp_row['interpretation_id']
                                    result['interpretation_id'] = field_value
                                    if 'metadata' not in result:
                                        result['metadata'] = {}
                                    result['metadata']['interpretation_id'] = field_value
                                    self.logger.debug(f"✅ Restored interpretation_id={field_value} for chunk_id={chunk_id}")
                        except Exception as e:
                            self.logger.debug(f"Failed to restore interpretation_id for chunk_id={chunk_id}: {e}")
                    
                    # 최종 확인: 별칭 필드도 확인
                    final_value = result.get(field) or result.get('metadata', {}).get(field)
                    if not final_value and source_type == 'statute_article':
                        if field == 'law_name':
                            final_value = result.get('statute_name') or result.get('metadata', {}).get('statute_name')
                        elif field == 'article_number':
                            final_value = result.get('article_no') or result.get('metadata', {}).get('article_no')
                    
                    if not final_value:
                        self.logger.warning(f"⚠️  Failed to restore {field} for chunk_id={chunk_id}, source_id={source_id}")
            
            conn.close()
            
        except Exception as e:
            self.logger.warning(f"Failed to restore missing metadata for chunk_id={chunk_id}, source_type={source_type}: {e}")

    def _log_search_error_details(self, 
                                  error: Exception,
                                  query: str,
                                  embedding_version_id: Optional[int]):
        """
        검색 에러 발생 시 상세 디버깅 정보 로깅
        
        Args:
            error: 발생한 에러
            query: 검색 쿼리
            embedding_version_id: 사용된 버전 ID
        """
        try:
            self.logger.error("🔍 Detailed error diagnostics:")
            
            # 1. 에러 타입 및 메시지
            self.logger.error(f"   Error type: {type(error).__name__}")
            self.logger.error(f"   Error message: {str(error)}")
            
            # 2. 시스템 상태
            self.logger.info("   System status:")
            self.logger.info(f"      - FAISS available: {FAISS_AVAILABLE}")
            self.logger.info(f"      - Index loaded: {self.index is not None}")
            if self.index:
                self.logger.info(f"      - Index vectors: {self.index.ntotal}")
            self.logger.info(f"      - Embedder initialized: {self.embedder is not None}")
            self.logger.info(f"      - Model name: {self.model_name}")
            
            # 3. 버전 정보
            if embedding_version_id:
                chunk_count = self._get_version_chunk_count(embedding_version_id)
                self.logger.info(f"      - Version ID: {embedding_version_id}")
                self.logger.info(f"      - Version chunks: {chunk_count}")
            
            # 4. 데이터베이스 상태
            try:
                if Path(self.db_path).exists():
                    conn = self._get_connection()
                    cursor = conn.cursor()
                    cursor.execute("SELECT COUNT(*) as count FROM embeddings")
                    row = cursor.fetchone()
                    emb_count = row['count'] if row else 0
                    cursor.execute("SELECT COUNT(*) as count FROM text_chunks")
                    row = cursor.fetchone()
                    chunk_count = row['count'] if row else 0
                    conn.close()
                    self.logger.info(f"      - Database embeddings: {emb_count}")
                    self.logger.info(f"      - Database chunks: {chunk_count}")
                else:
                    self.logger.error(f"      - Database not found: {self.db_path}")
            except Exception as db_error:
                self.logger.error(f"      - Database check failed: {db_error}")
                
        except Exception as e:
            self.logger.debug(f"Error logging search error details: {e}")

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

        # IndexIVFPQ는 압축 인덱스이므로 더 높은 nprobe 필요 (정확도 향상)
        if self.index and 'IndexIVFPQ' in type(self.index).__name__:
            # IndexIVFPQ: 정확도 향상을 위해 nprobe 증가 (50% → 100% 증가로 변경)
            nprobe = int(nprobe * 2.0)  # 100% 증가 (정확도 향상)

        # 최소/최대 값 제한
        nprobe = min(max(1, nprobe), estimated_nlist)

        return nprobe

    def _build_faiss_index_sync(self, embedding_version_id: Optional[int] = None, faiss_version_name: Optional[str] = None):
        """FAISS IVF 인덱스 빌드 및 저장 (동기 방식)"""
        if not FAISS_AVAILABLE:
            self.logger.warning("FAISS not available, skipping index build")
            return False

        try:
            self.logger.info("Building FAISS index...")

            # 1. 벡터 로드
            chunk_vectors = self._load_chunk_vectors(embedding_version_id=embedding_version_id)
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

            # 5. chunk_id 매핑 및 메타데이터 준비
            chunk_ids = chunk_ids_sorted
            id_mapping = {i: chunk_id for i, chunk_id in enumerate(chunk_ids)}
            metadata = [self._chunk_metadata.get(chunk_id, {}) for chunk_id in chunk_ids]
            
            # chunk_ids를 JSON 파일로 저장 (레거시 경로 호환성)
            try:
                import json
                chunk_ids_path = Path(self.index_path).with_suffix('.chunk_ids.json')
                with open(chunk_ids_path, 'w', encoding='utf-8') as f:
                    json.dump(chunk_ids, f, indent=2)
                self.logger.debug(f"Saved chunk_ids to {chunk_ids_path}")
            except Exception as e:
                self.logger.debug(f"Failed to save chunk_ids: {e}")

            # 6. 버전 관리 시스템 사용 여부 확인
            # 외부 인덱스를 사용하지 않고 버전 관리가 가능한 경우
            use_version_manager = (
                self.faiss_version_manager is not None and 
                embedding_version_id is not None and
                not self.use_mlflow_index
            )
            
            self.logger.debug(
                f"Version manager check: faiss_version_manager={self.faiss_version_manager is not None}, "
                f"embedding_version_id={embedding_version_id}, use_mlflow_index={self.use_mlflow_index}, "
                f"use_version_manager={use_version_manager}"
            )
            
            if use_version_manager:
                # EmbeddingVersionManager에서 버전 정보 조회
                try:
                    scripts_utils_path = Path(__file__).parent.parent.parent / "scripts" / "utils"
                    if scripts_utils_path.exists():
                        sys.path.insert(0, str(scripts_utils_path))
                    from embedding_version_manager import EmbeddingVersionManager
                    
                    evm = EmbeddingVersionManager(self.db_path)
                    version_info = evm.get_version_statistics(embedding_version_id)
                    
                    if version_info:
                        # FAISS 버전 이름 생성
                        if faiss_version_name is None:
                            chunking_strategy = version_info.get('chunking_strategy', 'unknown')
                            version_name = version_info.get('version_name', f'v{embedding_version_id}')
                            faiss_version_name = f"{version_name}-{chunking_strategy}"
                        
                        # FAISS 버전 생성 또는 기존 버전 사용
                        version_path = self.faiss_version_manager.get_version_path(faiss_version_name)
                        if version_path is None:
                            # 새 버전 생성
                            chunking_config = {
                                'chunk_size': 1000,
                                'chunk_overlap': 200
                            }
                            embedding_config = {
                                'model': self.model_name,
                                'dimension': dimension
                            }
                            
                            version_path = self.faiss_version_manager.create_version(
                                version_name=faiss_version_name,
                                embedding_version_id=embedding_version_id,
                                chunking_strategy=version_info.get('chunking_strategy', 'standard'),
                                chunking_config=chunking_config,
                                embedding_config=embedding_config,
                                document_count=version_info.get('document_count', 0),
                                total_chunks=len(chunk_ids),
                                status='active'
                            )
                        
                        # FAISS 인덱스 및 메타데이터 저장
                        success = self.faiss_version_manager.save_index(
                            version_name=faiss_version_name,
                            index=index,
                            id_mapping=id_mapping,
                            metadata=metadata
                        )
                        
                        if success:
                            self.current_faiss_version = faiss_version_name
                            self.logger.info(f"FAISS index saved to version: {faiss_version_name}")
                            
                            # 인덱스와 매핑을 인스턴스에 저장
                            self.index = index
                            self._chunk_ids = chunk_ids
                            
                            return True
                        else:
                            self.logger.warning("Failed to save to version manager, falling back to legacy path")
                    else:
                        self.logger.warning(f"Version info not found for embedding_version_id={embedding_version_id}")
                except Exception as e:
                    self.logger.warning(f"Failed to use version manager: {e}, falling back to legacy path")
            
            # 기존 방식으로 저장 (레거시 호환성)
            faiss.write_index(index, self.index_path)
            self.logger.info(f"FAISS index built and saved: {self.index_path} ({len(vectors)} vectors)")
            
            # 인덱스와 매핑을 인스턴스에 저장
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

    def _load_mlflow_index(self):
        """MLflow에서 FAISS 인덱스 로드"""
        if not FAISS_AVAILABLE:
            return
        
        if not self.use_mlflow_index or not self.mlflow_manager:
            self.logger.debug("Skipping MLflow index load: use_mlflow_index=False or mlflow_manager not available")
            return
        
        try:
            # run_id가 없으면 프로덕션 run 자동 조회
            run_id = self.mlflow_run_id
            if not run_id:
                run_id = self.mlflow_manager.get_production_run()
                if run_id:
                    self.logger.info(f"Auto-detected production run: {run_id}")
                    self.mlflow_run_id = run_id
                else:
                    self.logger.warning("No production run found in MLflow. Please specify MLFLOW_RUN_ID.")
                    return
            
            # MLflow에서 인덱스 로드
            index_data = self.mlflow_manager.load_index(run_id)
            if not index_data:
                self.logger.warning(f"Failed to load index from MLflow run: {run_id}")
                return
            
            self.index = index_data['index']
            id_mapping = index_data.get('id_mapping', {})
            metadata = index_data.get('metadata', [])
            stats = index_data.get('stats', {})
            run_info = index_data.get('run_info', {})
            
            # id_mapping을 역으로 변환 (chunk_id 리스트 생성)
            if id_mapping:
                # id_mapping의 키가 문자열일 수 있으므로 정수로 변환
                try:
                    # 키를 정수로 변환 시도
                    int_keys = [int(k) for k in id_mapping.keys() if str(k).isdigit()]
                    if int_keys:
                        max_faiss_id = max(int_keys)
                        self._chunk_ids = [id_mapping.get(str(i), id_mapping.get(i, -1)) for i in range(max_faiss_id + 1)]
                    else:
                        # 키가 정수가 아닌 경우 직접 사용
                        self._chunk_ids = list(id_mapping.values())
                except (ValueError, TypeError):
                    # 변환 실패 시 직접 사용
                    self._chunk_ids = list(id_mapping.values())
                self._chunk_ids = [cid for cid in self._chunk_ids if cid != -1]
            else:
                # id_mapping이 없으면 embeddings 테이블에서 조회
                conn = self._get_connection()
                cursor = conn.execute(
                    "SELECT chunk_id FROM embeddings WHERE model = ? ORDER BY chunk_id",
                    (self.model_name,)
                )
                self._chunk_ids = [row[0] for row in cursor.fetchall()]
                conn.close()
            
            # 메타데이터 처리
            if metadata:
                self._chunk_metadata = {chunk_id: meta for chunk_id, meta in zip(self._chunk_ids, metadata)}
            
            # 인덱스 타입 감지 및 로깅
            index_type = type(self.index).__name__
            self.logger.info(f"Loaded MLflow FAISS index: {index_type} ({self.index.ntotal:,} vectors) from run {run_id}")
            
            # IndexIVF 계열 인덱스 (IndexIVFFlat, IndexIVFPQ 등) 확인
            if hasattr(self.index, 'nprobe'):
                if not hasattr(self.index, 'nprobe') or self.index.nprobe == 1:
                    optimal_nprobe = self._calculate_optimal_nprobe(10, self.index.ntotal)
                    self.index.nprobe = optimal_nprobe
                    self.logger.info(f"Set nprobe to {optimal_nprobe} for {index_type}")
                else:
                    self.logger.info(f"Using existing nprobe={self.index.nprobe} for {index_type}")
            
            # IndexIVFPQ 특별 처리
            if 'IndexIVFPQ' in index_type:
                self.logger.info(f"✅ IndexIVFPQ detected - using compressed index for memory efficiency")
                if hasattr(self.index, 'pq'):
                    m = self.index.pq.M if hasattr(self.index.pq, 'M') else 'unknown'
                    nbits = self.index.pq.nbits if hasattr(self.index.pq, 'nbits') else 'unknown'
                    self.logger.info(f"   PQ parameters: M={m}, nbits={nbits}")
            
            # 버전 정보 로깅
            if run_info:
                version = run_info.get('tags', {}).get('version', 'unknown')
                self.logger.info(f"MLflow version: {version}")
        
        except Exception as e:
            self.logger.warning(f"Failed to load MLflow FAISS index: {e}, will use DB-based index", exc_info=True)
            self.index = None
    
    def _load_faiss_index(self, faiss_version_name: Optional[str] = None):
        """저장된 FAISS 인덱스 로드 (DB 기반 또는 버전 관리 시스템)"""
        if not FAISS_AVAILABLE:
            return

        # 버전 관리 시스템 사용 시도
        if self.faiss_version_manager is not None:
            if faiss_version_name is None:
                faiss_version_name = self.faiss_version_manager.get_active_version()
            
            if faiss_version_name:
                try:
                    index_data = self.faiss_version_manager.load_index(faiss_version_name)
                    if index_data:
                        self.index = index_data['index']
                        id_mapping = index_data.get('id_mapping', {})
                        
                        # id_mapping을 역으로 변환 (chunk_id 리스트 생성)
                        if id_mapping:
                            max_faiss_id = max(id_mapping.keys())
                            self._chunk_ids = [id_mapping.get(i, -1) for i in range(max_faiss_id + 1)]
                            self._chunk_ids = [cid for cid in self._chunk_ids if cid != -1]
                        else:
                            # id_mapping이 없으면 embeddings 테이블에서 조회
                            conn = self._get_connection()
                            cursor = conn.execute(
                                "SELECT chunk_id FROM embeddings WHERE model = ? ORDER BY chunk_id",
                                (self.model_name,)
                            )
                            self._chunk_ids = [row[0] for row in cursor.fetchall()]
                            conn.close()
                        
                        metadata = index_data.get('metadata', [])
                        if metadata:
                            self._chunk_metadata = {chunk_id: meta for chunk_id, meta in zip(self._chunk_ids, metadata)}
                        
                        self.current_faiss_version = faiss_version_name
                        
                        # 인덱스 타입 감지 및 로깅
                        if self.index is not None:
                            index_type = type(self.index).__name__
                            self.logger.info(f"FAISS index loaded from version: {faiss_version_name} ({index_type}, {len(self._chunk_ids)} vectors)")
                            
                            # IndexIVFPQ 감지
                            if 'IndexIVFPQ' in index_type:
                                self.logger.info(f"✅ IndexIVFPQ detected - using compressed index")
                                if hasattr(self.index, 'pq'):
                                    m = self.index.pq.M if hasattr(self.index.pq, 'M') else 'unknown'
                                    nbits = self.index.pq.nbits if hasattr(self.index.pq, 'nbits') else 'unknown'
                                    self.logger.info(f"   PQ parameters: M={m}, nbits={nbits}")
                            
                            # nprobe 설정 (IndexIVF 계열)
                            if hasattr(self.index, 'nprobe'):
                                current_nprobe = getattr(self.index, 'nprobe', 1)
                                if current_nprobe == 1:
                                    optimal_nprobe = self._calculate_optimal_nprobe(10, self.index.ntotal)
                                    self.index.nprobe = optimal_nprobe
                                    self.logger.info(f"Set nprobe to {optimal_nprobe} for {index_type}")
                        else:
                            self.logger.info(f"FAISS index loaded from version: {faiss_version_name} ({len(self._chunk_ids)} vectors)")
                        return
                except Exception as e:
                    self.logger.warning(f"Failed to load from version manager: {e}, trying legacy path")

        # 기존 방식으로 로드 (레거시 호환성)
        try:
            if not Path(self.index_path).exists():
                self.logger.warning(f"FAISS index not found at {self.index_path}")
                return
            
            # FAISS 인덱스 로드 (IndexIVFPQ 포함 모든 타입 지원)
            self.index = faiss.read_index(str(self.index_path))
            
            # 인덱스 타입 감지 및 로깅
            index_type = type(self.index).__name__
            self.logger.info(f"Loaded internal FAISS index: {index_type} ({self.index.ntotal:,} vectors)")
            
            # IndexIVF 계열 인덱스 (IndexIVFFlat, IndexIVFPQ 등) 확인
            if hasattr(self.index, 'nprobe'):
                # 기본 nprobe 설정 (IndexIVFPQ 포함)
                if not hasattr(self.index, 'nprobe') or self.index.nprobe == 1:
                    optimal_nprobe = self._calculate_optimal_nprobe(10, self.index.ntotal)
                    self.index.nprobe = optimal_nprobe
                    self.logger.info(f"Set nprobe to {optimal_nprobe} for {index_type}")
                else:
                    self.logger.info(f"Using existing nprobe={self.index.nprobe} for {index_type}")
            
            # IndexIVFPQ 특별 처리
            if 'IndexIVFPQ' in index_type:
                self.logger.info(f"✅ IndexIVFPQ detected - using compressed index for memory efficiency")
                if hasattr(self.index, 'pq'):
                    m = self.index.pq.M if hasattr(self.index.pq, 'M') else 'unknown'
                    nbits = self.index.pq.nbits if hasattr(self.index.pq, 'nbits') else 'unknown'
                    self.logger.info(f"   PQ parameters: M={m}, nbits={nbits}")

            # chunk_id 매핑 재구성
            # 1. chunk_ids.json 파일에서 로드 시도
            chunk_ids_path = Path(self.index_path).with_suffix('.chunk_ids.json')
            if chunk_ids_path.exists():
                try:
                    import json
                    with open(chunk_ids_path, 'r', encoding='utf-8') as f:
                        self._chunk_ids = json.load(f)
                    self.logger.info(f"Loaded chunk_ids from {chunk_ids_path} ({len(self._chunk_ids)} chunks)")
                except Exception as e:
                    self.logger.warning(f"Failed to load chunk_ids from {chunk_ids_path}: {e}, falling back to DB query")
                    # DB에서 로드
                    conn = self._get_connection()
                    cursor = conn.execute(
                        "SELECT chunk_id FROM embeddings WHERE model = ? ORDER BY chunk_id",
                        (self.model_name,)
                    )
                    self._chunk_ids = [row[0] for row in cursor.fetchall()]
                    conn.close()
            else:
                # chunk_ids.json이 없으면 embeddings 테이블에서 로드
                conn = self._get_connection()
                cursor = conn.execute(
                    "SELECT chunk_id FROM embeddings WHERE model = ? ORDER BY chunk_id",
                    (self.model_name,)
                )
                self._chunk_ids = [row[0] for row in cursor.fetchall()]
                conn.close()
            
            # FAISS 인덱스 크기와 _chunk_ids 길이 일치 확인
            if self.index and hasattr(self.index, 'ntotal'):
                if len(self._chunk_ids) != self.index.ntotal:
                    diff = len(self._chunk_ids) - self.index.ntotal
                    self.logger.warning(
                        f"⚠️  _chunk_ids length ({len(self._chunk_ids)}) != FAISS index ntotal ({self.index.ntotal}). "
                        f"Difference: {diff} chunks. Truncating _chunk_ids to match index size."
                    )
                    if diff > 0:
                        self.logger.warning(f"   ⚠️  {diff} chunks will be excluded from search results. Consider rebuilding the index.")
                    self._chunk_ids = self._chunk_ids[:self.index.ntotal]

            self.logger.info(f"FAISS index loaded: {len(self._chunk_ids)} vectors from {self.index_path}")

        except Exception as e:
            self.logger.warning(f"Failed to load FAISS index: {e}, will rebuild")
            self.index = None
            # 인덱스 파일이 손상된 경우 삭제하여 재빌드 유도
            try:
                Path(self.index_path).unlink()
            except Exception:
                pass

    def _column_exists(self, conn: sqlite3.Connection, table_name: str, column_name: str) -> bool:
        """테이블에 컬럼이 존재하는지 확인"""
        try:
            cursor = conn.execute(f"PRAGMA table_info({table_name})")
            columns = [row[1] for row in cursor.fetchall()]
            return column_name in columns
        except Exception:
            return False

    def _batch_load_chunk_metadata(self, conn: sqlite3.Connection, chunk_ids: List[int]) -> Dict[int, Dict[str, Any]]:
        """
        여러 chunk_id의 text_chunks.meta를 배치로 조회 (캐싱 적용)
        
        Args:
            conn: 데이터베이스 연결
            chunk_ids: 조회할 chunk_id 리스트
            
        Returns:
            chunk_id -> 메타데이터 딕셔너리
        """
        if not chunk_ids:
            return {}
        
        metadata_map = {}
        uncached_ids = []
        
        # 캐시에서 먼저 확인 (TTL 체크 포함)
        for chunk_id in chunk_ids:
            cache_key = f"chunk_{chunk_id}"
            cached_data = self._get_from_cache(cache_key)
            if cached_data is not None:
                metadata_map[chunk_id] = cached_data.copy()
                self._metadata_cache_hits += 1
            else:
                uncached_ids.append(chunk_id)
                self._metadata_cache_misses += 1
        
        # 캐시에 없는 것만 DB에서 조회
        if uncached_ids:
            # 배치 크기 최적화: SQLite의 최대 변수 수 제한 고려 (999개)
            batch_size = min(500, len(uncached_ids))
            for i in range(0, len(uncached_ids), batch_size):
                batch = uncached_ids[i:i + batch_size]
                try:
                    placeholders = ','.join(['?'] * len(batch))
                    cursor = conn.execute(f"""
                        SELECT id, meta, source_type, source_id
                        FROM text_chunks
                        WHERE id IN ({placeholders})
                    """, batch)
                    
                    for row in cursor.fetchall():
                        chunk_id = row['id']
                        meta_json = {}
                        if row['meta']:
                            try:
                                import json
                                meta_json = json.loads(row['meta'])
                            except Exception as e:
                                self.logger.debug(f"Failed to parse meta JSON for chunk_id={chunk_id}: {e}")
                        
                        chunk_meta = {
                            'meta': meta_json,
                            'source_type': row['source_type'],
                            'source_id': row['source_id']
                        }
                        metadata_map[chunk_id] = chunk_meta
                        
                        # 캐시에 저장 (TTL 포함)
                        cache_key = f"chunk_{chunk_id}"
                        self._set_to_cache(cache_key, chunk_meta)
                except Exception as e:
                    self.logger.warning(f"Error in batch_load_chunk_metadata: {e}")
        
        if self._metadata_cache_hits + self._metadata_cache_misses > 0:
            hit_rate = self._metadata_cache_hits / (self._metadata_cache_hits + self._metadata_cache_misses) * 100
            if hit_rate > 0:
                self.logger.info(f"📊 Metadata cache hit rate: {hit_rate:.1f}% ({self._metadata_cache_hits} hits, {self._metadata_cache_misses} misses)")
        
        return metadata_map
    
    def _batch_load_source_metadata(self, conn: sqlite3.Connection, source_items: List[Tuple[str, int]]) -> Dict[Tuple[str, int], Dict[str, Any]]:
        """
        여러 (source_type, source_id) 쌍의 소스 메타데이터를 배치로 조회 (캐싱 적용)
        
        Args:
            conn: 데이터베이스 연결
            source_items: (source_type, source_id) 튜플 리스트
            
        Returns:
            (source_type, source_id) -> 메타데이터 딕셔너리
        """
        if not source_items:
            return {}
        
        metadata_map = {}
        uncached_items = []
        
        # 캐시에서 먼저 확인 (TTL 체크 포함)
        for source_type, source_id in source_items:
            cache_key = (source_type, source_id)
            cached_data = self._get_from_cache(cache_key)
            if cached_data is not None:
                metadata_map[cache_key] = cached_data.copy()
                self._metadata_cache_hits += 1
            else:
                uncached_items.append((source_type, source_id))
                self._metadata_cache_misses += 1
        
        # 캐시에 없는 것만 DB에서 조회
        if not uncached_items:
            return metadata_map
        
        # 타입별로 그룹화
        by_type = {}
        for source_type, source_id in uncached_items:
            if source_type not in by_type:
                by_type[source_type] = []
            by_type[source_type].append(source_id)
        
        # 타입별로 배치 조회 (배치 크기 최적화)
        for source_type, source_ids in by_type.items():
            if not source_ids:
                continue
            
            # 배치 크기 최적화: SQLite의 최대 변수 수 제한 고려 (999개)
            batch_size = min(500, len(source_ids))
            for i in range(0, len(source_ids), batch_size):
                batch = source_ids[i:i + batch_size]
                try:
                    placeholders = ','.join(['?'] * len(batch))
                    
                    if source_type == "statute_article":
                        cursor = conn.execute(f"""
                            SELECT sa.id, sa.article_no, s.name as statute_name, s.abbrv, s.category, s.statute_type
                            FROM statute_articles sa
                            JOIN statutes s ON sa.statute_id = s.id
                            WHERE sa.id IN ({placeholders})
                        """, batch)
                    elif source_type == "case_paragraph":
                        # cases 테이블에서 직접 조회
                        cursor = conn.execute(f"""
                            SELECT c.id, c.doc_id, c.court, c.case_type, c.casenames, c.announce_date
                            FROM cases c
                            WHERE c.id IN ({placeholders})
                        """, batch)
                    elif source_type == "decision_paragraph":
                        # decisions 테이블에서 직접 조회
                        cursor = conn.execute(f"""
                            SELECT d.id, d.org, d.doc_id, d.decision_date, d.result
                            FROM decisions d
                            WHERE d.id IN ({placeholders})
                        """, batch)
                    elif source_type == "interpretation_paragraph":
                        cursor = conn.execute(f"""
                            SELECT ip.id, i.org, i.doc_id, i.title, i.response_date
                            FROM interpretation_paragraphs ip
                            JOIN interpretations i ON ip.interpretation_id = i.id
                            WHERE ip.id IN ({placeholders})
                        """, batch)
                    else:
                        continue
                    
                    for row in cursor.fetchall():
                        source_id = row['id']
                        metadata = dict(row)
                        # statute_article의 경우 별칭 추가
                        if source_type == "statute_article":
                            if "statute_name" in metadata:
                                metadata["law_name"] = metadata["statute_name"]
                            if "article_no" in metadata:
                                metadata["article_number"] = metadata["article_no"]
                        
                        cache_key = (source_type, source_id)
                        metadata_map[cache_key] = metadata
                        
                        # 캐시에 저장 (TTL 포함)
                        self._set_to_cache(cache_key, metadata)
                except Exception as e:
                    self.logger.warning(f"Error in batch_load_source_metadata for {source_type}: {e}")
        
        return metadata_map

    def _get_source_metadata(self, conn: sqlite3.Connection, source_type: str, source_id: int) -> Dict[str, Any]:
        """
        소스 타입별 상세 메타데이터 조회
        source_id는 text_chunks.source_id로, 각 소스 테이블의 실제 id를 참조
        """
        try:
            if source_type == "statute_article":
                # text_chunks.source_id가 statute_articles.id를 참조
                base_columns = "sa.article_no, s.name as statute_name, s.abbrv, s.category, s.statute_type"
                optional_columns = []
                
                if self._column_exists(conn, "statutes", "law_id"):
                    optional_columns.append("s.law_id")
                if self._column_exists(conn, "statutes", "mst"):
                    optional_columns.append("s.mst")
                if self._column_exists(conn, "statutes", "proclamation_number"):
                    optional_columns.append("s.proclamation_number")
                if self._column_exists(conn, "statutes", "effective_date"):
                    optional_columns.append("s.effective_date")
                
                select_clause = base_columns
                if optional_columns:
                    select_clause += ", " + ", ".join(optional_columns)
                
                cursor = conn.execute(f"""
                    SELECT {select_clause}
                    FROM statute_articles sa
                    JOIN statutes s ON sa.statute_id = s.id
                    WHERE sa.id = ?
                """, (source_id,))
            elif source_type == "case_paragraph":
                # text_chunks.source_id가 cases.id를 직접 참조할 수 있으므로 두 가지 경우 모두 처리
                base_columns = "c.doc_id, c.court, c.case_type, c.casenames, c.announce_date"
                optional_columns = []
                
                if self._column_exists(conn, "cases", "precedent_serial_number"):
                    optional_columns.append("c.precedent_serial_number")
                
                select_clause = base_columns
                if optional_columns:
                    select_clause += ", " + ", ".join(optional_columns)
                
                # 먼저 cases 테이블에서 직접 조회 시도 (text_chunks.source_id가 cases.id를 직접 참조하는 경우)
                cursor = conn.execute(f"""
                    SELECT {select_clause}
                    FROM cases c
                    WHERE c.id = ?
                """, (source_id,))
                row = cursor.fetchone()
                
                # case_paragraphs를 통한 조회 시도 (text_chunks.source_id가 case_paragraphs.id를 참조하는 경우)
                if not row:
                    cursor = conn.execute(f"""
                        SELECT {select_clause}
                        FROM case_paragraphs cp
                        JOIN cases c ON cp.case_id = c.id
                        WHERE cp.id = ?
                    """, (source_id,))
                    row = cursor.fetchone()
                
                if row:
                    return dict(row)
                return {}
            elif source_type == "decision_paragraph":
                # text_chunks.source_id가 decisions.id를 직접 참조할 수 있으므로 두 가지 경우 모두 처리
                base_columns = "d.org, d.doc_id, d.decision_date, d.result"
                optional_columns = []
                
                if self._column_exists(conn, "decisions", "decision_serial_number"):
                    optional_columns.append("d.decision_serial_number")
                
                select_clause = base_columns
                if optional_columns:
                    select_clause += ", " + ", ".join(optional_columns)
                
                # 먼저 decisions 테이블에서 직접 조회 시도 (text_chunks.source_id가 decisions.id를 직접 참조하는 경우)
                cursor = conn.execute(f"""
                    SELECT {select_clause}
                    FROM decisions d
                    WHERE d.id = ?
                """, (source_id,))
                row = cursor.fetchone()
                
                # decision_paragraphs를 통한 조회 시도 (text_chunks.source_id가 decision_paragraphs.id를 참조하는 경우)
                if not row:
                    cursor = conn.execute(f"""
                        SELECT {select_clause}
                        FROM decision_paragraphs dp
                        JOIN decisions d ON dp.decision_id = d.id
                        WHERE dp.id = ?
                    """, (source_id,))
                    row = cursor.fetchone()
                
                if row:
                    return dict(row)
                return {}
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
                metadata = dict(row)
                # statute_article의 경우 law_name 별칭 추가
                if source_type == "statute_article" and "statute_name" in metadata:
                    metadata["law_name"] = metadata["statute_name"]
                # article_no의 경우 article_number 별칭 추가
                if source_type == "statute_article" and "article_no" in metadata:
                    metadata["article_number"] = metadata["article_no"]
                return metadata
            else:
                self.logger.debug(f"No row found for {source_type} source_id={source_id}")
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
    
    def _restore_text_from_chunks_by_metadata(self, conn: sqlite3.Connection, source_type: str, metadata: Dict[str, Any]) -> str:
        """
        text_chunks 테이블에서 메타데이터를 사용하여 text 조회 (source_id가 문자열인 경우)
        """
        try:
            conn.row_factory = sqlite3.Row
            import json
            
            # metadata를 JSON 문자열로 변환하여 조회 시도
            metadata_json = json.dumps(metadata, ensure_ascii=False) if metadata else ""
            
            # source_type에 따라 다른 필드로 조회
            if source_type == 'case_paragraph':
                # 다양한 필드명 시도
                case_number = (
                    metadata.get('case_number') or 
                    metadata.get('doc_id') or 
                    metadata.get('case_id') or
                    metadata.get('id')
                )
                
                # source_id가 문자열인 경우 (예: case_2015도19521)
                if isinstance(metadata.get('source_id'), str):
                    source_id_str = metadata.get('source_id')
                    # case_ 접두사 제거
                    if source_id_str.startswith('case_'):
                        case_number = source_id_str[5:]  # 'case_' 제거
                
                if case_number:
                    # 방법 1: metadata JSON에 포함된 경우
                    cursor = conn.execute(
                        "SELECT text FROM text_chunks WHERE source_type = ? AND metadata LIKE ? AND text IS NOT NULL AND text != '' ORDER BY LENGTH(text) DESC LIMIT 1",
                        (source_type, f'%{case_number}%')
                    )
                    row = cursor.fetchone()
                    if row:
                        text = row['text'] if 'text' in row.keys() else None
                        if text and len(str(text).strip()) > 0:
                            self.logger.info(f"Restored text from text_chunks by metadata for {source_type} case_number={case_number} (length: {len(str(text))} chars)")
                            return str(text)
                    
                    # 방법 2: text 필드에 포함된 경우
                    cursor = conn.execute(
                        "SELECT text FROM text_chunks WHERE source_type = ? AND text LIKE ? AND text IS NOT NULL AND text != '' ORDER BY LENGTH(text) DESC LIMIT 1",
                        (source_type, f'%{case_number}%')
                    )
                    row = cursor.fetchone()
                    if row:
                        text = row['text'] if 'text' in row.keys() else None
                        if text and len(str(text).strip()) > 0:
                            self.logger.info(f"Restored text from text_chunks by text field for {source_type} case_number={case_number} (length: {len(str(text))} chars)")
                            return str(text)
            
            elif source_type == 'statute_article':
                law_id = metadata.get('law_id') or metadata.get('statute_id')
                article_no = metadata.get('article_number') or metadata.get('article_no') or metadata.get('article')
                statute_name = metadata.get('statute_name') or metadata.get('law_name')
                
                # law_id와 article_no가 있으면 조회
                if law_id and article_no:
                    cursor = conn.execute(
                        "SELECT text FROM text_chunks WHERE source_type = ? AND metadata LIKE ? AND metadata LIKE ? AND text IS NOT NULL AND text != '' ORDER BY LENGTH(text) DESC LIMIT 1",
                        (source_type, f'%law_id%{law_id}%', f'%article_number%{article_no}%')
                    )
                    row = cursor.fetchone()
                    if row:
                        text = row['text'] if 'text' in row.keys() else None
                        if text and len(str(text).strip()) > 0:
                            self.logger.info(f"Restored text from text_chunks by metadata for {source_type} law_id={law_id} article_no={article_no} (length: {len(str(text))} chars)")
                            return str(text)
                
                # statute_name과 article_no로 조회 시도
                if statute_name and article_no:
                    cursor = conn.execute(
                        "SELECT text FROM text_chunks WHERE source_type = ? AND metadata LIKE ? AND metadata LIKE ? AND text IS NOT NULL AND text != '' ORDER BY LENGTH(text) DESC LIMIT 1",
                        (source_type, f'%{statute_name}%', f'%{article_no}%')
                    )
                    row = cursor.fetchone()
                    if row:
                        text = row['text'] if 'text' in row.keys() else None
                        if text and len(str(text).strip()) > 0:
                            self.logger.info(f"Restored text from text_chunks by statute_name for {source_type} statute_name={statute_name} article_no={article_no} (length: {len(str(text))} chars)")
                            return str(text)
            
            elif source_type == 'decision_paragraph':
                doc_id = metadata.get('doc_id') or metadata.get('decision_id') or metadata.get('id')
                org = metadata.get('org')
                
                if doc_id:
                    # 방법 1: metadata JSON에 포함된 경우
                    cursor = conn.execute(
                        "SELECT text FROM text_chunks WHERE source_type = ? AND metadata LIKE ? AND text IS NOT NULL AND text != '' ORDER BY LENGTH(text) DESC LIMIT 1",
                        (source_type, f'%{doc_id}%')
                    )
                    row = cursor.fetchone()
                    if row:
                        text = row['text'] if 'text' in row.keys() else None
                        if text and len(str(text).strip()) > 0:
                            self.logger.info(f"Restored text from text_chunks by metadata for {source_type} doc_id={doc_id} (length: {len(str(text))} chars)")
                            return str(text)
                    
                    # 방법 2: org와 함께 조회
                    if org:
                        cursor = conn.execute(
                            "SELECT text FROM text_chunks WHERE source_type = ? AND metadata LIKE ? AND metadata LIKE ? AND text IS NOT NULL AND text != '' ORDER BY LENGTH(text) DESC LIMIT 1",
                            (source_type, f'%{doc_id}%', f'%{org}%')
                        )
                        row = cursor.fetchone()
                        if row:
                            text = row['text'] if 'text' in row.keys() else None
                            if text and len(str(text).strip()) > 0:
                                self.logger.info(f"Restored text from text_chunks by metadata for {source_type} doc_id={doc_id} org={org} (length: {len(str(text))} chars)")
                                return str(text)
            
            elif source_type == 'interpretation_paragraph':
                doc_id = metadata.get('doc_id') or metadata.get('interpretation_id') or metadata.get('id')
                title = metadata.get('title')
                org = metadata.get('org')
                
                if doc_id:
                    cursor = conn.execute(
                        "SELECT text FROM text_chunks WHERE source_type = ? AND metadata LIKE ? AND text IS NOT NULL AND text != '' ORDER BY LENGTH(text) DESC LIMIT 1",
                        (source_type, f'%{doc_id}%')
                    )
                    row = cursor.fetchone()
                    if row:
                        text = row['text'] if 'text' in row.keys() else None
                        if text and len(str(text).strip()) > 0:
                            self.logger.info(f"Restored text from text_chunks by metadata for {source_type} doc_id={doc_id} (length: {len(str(text))} chars)")
                            return str(text)
                
                # title로 조회 시도
                if title:
                    cursor = conn.execute(
                        "SELECT text FROM text_chunks WHERE source_type = ? AND (metadata LIKE ? OR text LIKE ?) AND text IS NOT NULL AND text != '' ORDER BY LENGTH(text) DESC LIMIT 1",
                        (source_type, f'%{title}%', f'%{title}%')
                    )
                    row = cursor.fetchone()
                    if row:
                        text = row['text'] if 'text' in row.keys() else None
                        if text and len(str(text).strip()) > 0:
                            self.logger.info(f"Restored text from text_chunks by title for {source_type} title={title} (length: {len(str(text))} chars)")
                            return str(text)
            
            # 모든 source_type에 대한 일반적인 조회 시도 (metadata JSON 전체 검색)
            if metadata_json and len(metadata_json) > 10:
                # metadata의 주요 키-값 쌍으로 조회 시도
                search_terms = []
                for key, value in metadata.items():
                    if value and isinstance(value, (str, int, float)) and len(str(value)) > 2:
                        search_terms.append(str(value))
                
                if search_terms:
                    # 가장 긴 검색어부터 시도
                    search_terms.sort(key=len, reverse=True)
                    for term in search_terms[:3]:  # 상위 3개만 시도
                        cursor = conn.execute(
                            "SELECT text FROM text_chunks WHERE source_type = ? AND metadata LIKE ? AND text IS NOT NULL AND text != '' ORDER BY LENGTH(text) DESC LIMIT 1",
                            (source_type, f'%{term}%')
                        )
                        row = cursor.fetchone()
                        if row:
                            text = row['text'] if 'text' in row.keys() else None
                            if text and len(str(text).strip()) > 0:
                                self.logger.info(f"Restored text from text_chunks by general metadata search for {source_type} term={term} (length: {len(str(text))} chars)")
                                return str(text)
            
            return ""
        except Exception as e:
            self.logger.debug(f"Error restoring text from text_chunks by metadata ({source_type}): {e}")
            return ""
    
    def _ensure_text_content(self,
                            conn: sqlite3.Connection,
                            chunk_id: int,
                            text: str,
                            source_type: str,
                            source_id: Any,
                            full_metadata: Dict[str, Any],
                            min_length: int = 100) -> str:
        """
        텍스트 내용이 충분한지 확인하고 필요시 복원
        
        Args:
            conn: 데이터베이스 연결
            chunk_id: 청크 ID
            text: 현재 텍스트
            source_type: 소스 타입
            source_id: 소스 ID
            full_metadata: 전체 메타데이터
            min_length: 최소 텍스트 길이
            
        Returns:
            복원된 또는 기존 텍스트
        """
        if text and len(text.strip()) >= min_length:
            return text
        
        if not text or len(text.strip()) == 0:
            self.logger.warning(f"Empty text content for chunk_id={chunk_id}, source_type={source_type}, source_id={source_id}. Attempting to restore from source table...")
        else:
            self.logger.debug(f"Short text content for chunk_id={chunk_id} (length: {len(text)} chars), attempting to restore longer text from source table...")
        
        if not conn:
            conn = self._get_connection()
        
        if not conn:
            return text or ""
        
        # source_id가 None인 경우 chunk_id로 조회 (강화)
        if not source_id and chunk_id:
            try:
                cursor = conn.execute(
                    "SELECT source_id, source_type, text FROM text_chunks WHERE id = ?",
                    (chunk_id,)
                )
                row = cursor.fetchone()
                if row:
                    if not source_id and row['source_id']:
                        source_id = row['source_id']
                    if not source_type and row['source_type']:
                        source_type = row['source_type']
                    # text도 복원 시도 (text가 비어있거나 짧은 경우)
                    if row['text'] and (not text or len(text.strip()) < min_length):
                        restored_text_from_db = row['text']
                        if restored_text_from_db and len(restored_text_from_db.strip()) >= min_length:
                            text = restored_text_from_db
                            self.logger.debug(f"✅ Restored text for chunk_id={chunk_id} from text_chunks (length: {len(text)} chars)")
                    self.logger.debug(f"✅ Restored source_id={source_id}, source_type={source_type} for chunk_id={chunk_id} from text_chunks")
            except Exception as e:
                self.logger.debug(f"Failed to get source_id from text_chunks for chunk_id={chunk_id}: {e}")
        
        # source_id가 여전히 None인 경우에도 text_chunks에서 직접 텍스트 복원 시도
        if (not source_type or not source_id) and chunk_id:
            try:
                cursor = conn.execute(
                    "SELECT text FROM text_chunks WHERE id = ?",
                    (chunk_id,)
                )
                row = cursor.fetchone()
                if row and row['text']:
                    restored_text_from_db = row['text']
                    if restored_text_from_db and len(restored_text_from_db.strip()) >= min_length:
                        text = restored_text_from_db
                        self.logger.debug(f"✅ Restored text for chunk_id={chunk_id} from text_chunks (direct query, length: {len(text)} chars)")
                        return text
            except Exception as e:
                self.logger.debug(f"Failed to get text from text_chunks for chunk_id={chunk_id}: {e}")
        
        # source_id가 없으면 텍스트만 반환 (메타데이터 복원은 불가능하지만 텍스트는 유지)
        if not source_type or not source_id:
            if text and len(text.strip()) >= min_length:
                return text
            return text or ""
        
        try:
            if isinstance(source_id, str):
                if source_id.isdigit():
                    actual_source_id = int(source_id)
                    restored_text = self._restore_text_from_source(conn, source_type, actual_source_id)
                else:
                    self.logger.debug(f"source_id is string format: {source_id}, trying metadata lookup")
                    restored_text = self._restore_text_from_chunks_by_metadata(conn, source_type, full_metadata)
                    
                    if not restored_text:
                        extracted_id = self._extract_id_from_source_id_string(source_id, source_type)
                        if extracted_id:
                            self.logger.debug(f"Extracted ID from source_id string: {extracted_id}")
                            if isinstance(extracted_id, int):
                                restored_text = self._restore_text_from_source(conn, source_type, extracted_id)
                            else:
                                restored_text = self._restore_text_from_chunks_by_metadata(conn, source_type, full_metadata)
            else:
                restored_text = self._restore_text_from_source(conn, source_type, source_id)
        except (ValueError, TypeError) as e:
            self.logger.debug(f"Invalid source_id format: {source_id}, trying text_chunks lookup: {e}")
            restored_text = self._restore_text_from_chunks_by_metadata(conn, source_type, full_metadata)
        
        if restored_text and len(restored_text.strip()) > len(text.strip()) if text else True:
            text = restored_text
            if chunk_id in self._chunk_metadata:
                self._chunk_metadata[chunk_id]['text'] = text
            self.logger.info(f"Successfully restored text for chunk_id={chunk_id} from source table (length: {len(text)} chars)")
        elif not text or len(text.strip()) == 0:
            self.logger.error(f"Failed to restore text for chunk_id={chunk_id}, source_type={source_type}, source_id={source_id}")
            if isinstance(source_id, str) and not source_id.isdigit():
                self.logger.debug(f"  - source_id is string format: {source_id}, may need metadata lookup")
                self.logger.debug(f"  - metadata keys: {list(full_metadata.keys())[:10] if full_metadata else 'N/A'}")
        else:
            self.logger.debug(f"Could not restore longer text for chunk_id={chunk_id}, using existing text (length: {len(text)} chars)")
        
        return text or ""
    
    def _extract_id_from_source_id_string(self, source_id: str, source_type: str) -> Optional[Union[int, str]]:
        """
        source_id 문자열에서 실제 ID 추출 (예: case_2015도19521 -> 2015도19521 또는 숫자 ID)
        
        Args:
            source_id: source_id 문자열 (예: case_2015도19521, decision_12345)
            source_type: 소스 타입
            
        Returns:
            추출된 ID (int 또는 str), 추출 실패 시 None
        """
        try:
            if not isinstance(source_id, str):
                return None
            
            # source_type 접두사 제거 (예: case_, decision_, statute_)
            prefixes = {
                'case_paragraph': 'case_',
                'decision_paragraph': 'decision_',
                'statute_article': 'statute_',
                'interpretation_paragraph': 'interpretation_'
            }
            
            prefix = prefixes.get(source_type, '')
            if prefix and source_id.startswith(prefix):
                extracted = source_id[len(prefix):]
            else:
                extracted = source_id
            
            # 숫자만 포함된 경우 int로 변환
            if extracted.isdigit():
                return int(extracted)
            
            # 숫자가 포함된 경우 숫자 부분 추출 시도
            import re
            numbers = re.findall(r'\d+', extracted)
            if numbers:
                # 가장 긴 숫자 시퀀스 사용
                longest_number = max(numbers, key=len)
                if len(longest_number) >= 3:  # 최소 3자리 숫자
                    return int(longest_number)
            
            # 숫자 추출 실패 시 원본 문자열 반환 (metadata 조회에 사용)
            return extracted if extracted else None
            
        except Exception as e:
            self.logger.debug(f"Error extracting ID from source_id string ({source_id}): {e}")
            return None

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
            # 상위 5개 키워드 추가 (개선: 3개 → 5개로 확대)
            top_keywords = expanded_keywords[:5]
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
