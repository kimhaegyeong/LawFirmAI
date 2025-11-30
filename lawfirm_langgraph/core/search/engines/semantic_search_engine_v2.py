# -*- coding: utf-8 -*-
"""
Semantic Search Engine V2
lawfirm_v2.db의 embeddings 테이블을 사용한 벡터 검색 엔진
"""

import gc
import heapq
import logging
try:
    from lawfirm_langgraph.core.utils.logger import get_logger
except ImportError:
    from core.utils.logger import get_logger
# SQLite import 제거 - PostgreSQL만 사용
# import sqlite3
import sys
import threading
import time
import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np

# Database adapter import
try:
    from core.data.db_adapter import DatabaseAdapter
    from core.data.sql_adapter import SQLAdapter
except ImportError:
    try:
        from lawfirm_langgraph.core.data.db_adapter import DatabaseAdapter
        from lawfirm_langgraph.core.data.sql_adapter import SQLAdapter
    except ImportError:
        DatabaseAdapter = None
        SQLAdapter = None

# Vector search adapter import
try:
    from core.search.engines.vector_search_adapter import VectorSearchFactory, PGVECTOR_AVAILABLE
except ImportError:
    try:
        from lawfirm_langgraph.core.search.engines.vector_search_adapter import VectorSearchFactory, PGVECTOR_AVAILABLE
    except ImportError:
        VectorSearchFactory = None
        # pgvector 지원 확인
        try:
            from pgvector.psycopg2 import register_vector
            PGVECTOR_AVAILABLE = True
        except ImportError:
            PGVECTOR_AVAILABLE = False

try:
    from lawfirm_langgraph.core.utils.korean_stopword_processor import KoreanStopwordProcessor
except ImportError:
    try:
        from core.utils.korean_stopword_processor import KoreanStopwordProcessor
    except ImportError:
        KoreanStopwordProcessor = None

# FAISS import (optional, only when VECTOR_SEARCH_METHOD=faiss)
VECTOR_SEARCH_METHOD = os.getenv("VECTOR_SEARCH_METHOD", "pgvector").lower()
if VECTOR_SEARCH_METHOD == "faiss":
    try:
        import faiss
        FAISS_AVAILABLE = True
    except ImportError:
        FAISS_AVAILABLE = False
        logger = logging.getLogger(__name__)
        logger.warning("FAISS not available. Install with: pip install faiss-cpu")
else:
    FAISS_AVAILABLE = False

# ModelCacheManager import
try:
    from lawfirm_langgraph.core.shared.utils.model_cache_manager import get_model_cache_manager, _filter_model_kwargs
except ImportError:
    try:
        from core.shared.utils.model_cache_manager import get_model_cache_manager, _filter_model_kwargs
    except ImportError:
        get_model_cache_manager = None
        _filter_model_kwargs = None

# Score normalization utilities
try:
    from lawfirm_langgraph.core.search.utils.score_utils import normalize_score
except ImportError:
    try:
        from core.search.utils.score_utils import normalize_score
    except ImportError:
        def normalize_score(score: float, min_val: float = 0.0, max_val: float = 1.0) -> float:
            """Fallback normalization function"""
            if score < min_val:
                return float(min_val)
            elif score > max_val:
                excess = score - max_val
                normalized = max_val + (excess / (1.0 + excess * 10))
                return float(max(0.0, min(1.0, normalized)))
            return float(score)

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
        
        def __init__(self, model_name: Optional[str] = None):
            if model_name is None:
                import os
                model_name = os.getenv("EMBEDDING_MODEL")
                if model_name is None:
                    from ...utils.config import get_config
                    config = get_config()
                    model_name = config.embedding_model
            
            # 모델 이름 정리 (따옴표 및 공백 제거)
            if model_name:
                model_name = model_name.strip().strip('"').strip("'").strip()
            
            if not model_name:
                raise ValueError("Model name cannot be empty")
            
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
                
                # ModelCacheManager 사용 (중복 로딩 방지)
                model_kwargs = {
                    "low_cpu_mem_usage": False,  # meta device 사용 방지 (가장 중요)
                    "device_map": None,  # device_map 사용 안 함
                    "dtype": torch.float32,  # 명시적 dtype 설정 (torch_dtype deprecated)
                    "use_safetensors": True,  # safetensors 사용 (모델이 safetensors 형식)
                    "ignore_mismatched_sizes": True,  # 크기 불일치 무시
                    "trust_remote_code": True,  # 원격 코드 신뢰
                    "local_files_only": False,  # 로컬 파일만 사용 안 함
                }
                
                if get_model_cache_manager:
                    try:
                        logger.debug(f"Loading SentenceTransformer model {model_name} via cache manager...")
                        model_cache = get_model_cache_manager()
                        self.model = model_cache.get_model(
                            model_name,
                            device="cpu",
                            model_kwargs=model_kwargs
                        )
                        if self.model:
                            logger.debug(f"✅ Model loaded via cache manager: {model_name}")
                        else:
                            raise ValueError(f"Failed to load model {model_name} via cache manager")
                    except Exception as e:
                        logger.warning(f"⚠️ Failed to load model via cache manager: {e}, falling back to direct load")
                        # 폴백: 직접 로드 (파라미터 필터링 적용)
                        logger.debug(f"Loading SentenceTransformer model {model_name} on CPU first...")
                        filtered_kwargs = _filter_model_kwargs(model_kwargs) if _filter_model_kwargs else {}
                        self.model = SentenceTransformer(
                            model_name, 
                            device="cpu",
                            model_kwargs=filtered_kwargs
                        )
                else:
                    # ModelCacheManager가 없으면 직접 로드 (파라미터 필터링 적용)
                    logger.debug(f"Loading SentenceTransformer model {model_name} on CPU first...")
                    filtered_kwargs = _filter_model_kwargs(model_kwargs) if _filter_model_kwargs else {}
                    self.model = SentenceTransformer(
                        model_name, 
                        device="cpu",
                        model_kwargs=filtered_kwargs
                    )
                
                try:
                    
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
                                use_safetensors=True,  # safetensors 사용 (모델이 safetensors 형식)
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
                                        use_safetensors=True,  # safetensors 사용 (모델이 safetensors 형식)
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
                    from ...utils.config import get_config
                    config = get_config()
                    fallback_model = config.embedding_model
                    
                    # 모델 이름 정리 (따옴표 및 공백 제거)
                    if fallback_model:
                        fallback_model = fallback_model.strip().strip('"').strip("'").strip()
                    
                    if not fallback_model:
                        raise ValueError("Fallback model name cannot be empty")
                    
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
                    
                    # 대체 모델도 ModelCacheManager 사용 (중복 로딩 방지)
                    fallback_model_kwargs = {
                        "low_cpu_mem_usage": False,  # meta device 사용 방지 (가장 중요)
                        "device_map": None,  # device_map 사용 안 함
                        "dtype": torch.float32,  # 명시적 dtype 설정 (torch_dtype deprecated)
                        "use_safetensors": True,  # safetensors 사용 (모델이 safetensors 형식)
                        "trust_remote_code": True,  # 원격 코드 신뢰
                        "local_files_only": False,  # 로컬 파일만 사용 안 함
                    }
                    
                    try:
                        if get_model_cache_manager:
                            try:
                                logger.warning(f"Trying fallback model via cache manager: {fallback_model}")
                                model_cache = get_model_cache_manager()
                                self.model = model_cache.get_model(
                                    fallback_model,
                                    device="cpu",
                                    model_kwargs=fallback_model_kwargs
                                )
                                if self.model:
                                    logger.debug(f"✅ Fallback model loaded via cache manager: {fallback_model}")
                                else:
                                    raise ValueError(f"Failed to load fallback model {fallback_model} via cache manager")
                            except Exception as e:
                                logger.warning(f"⚠️ Failed to load fallback model via cache manager: {e}, falling back to direct load")
                                # 폴백: 직접 로드 (파라미터 필터링 적용)
                                filtered_fallback_kwargs = _filter_model_kwargs(fallback_model_kwargs) if _filter_model_kwargs else {}
                                self.model = SentenceTransformer(
                                    fallback_model, 
                                    device="cpu",
                                    model_kwargs=filtered_fallback_kwargs
                                )
                        else:
                            # ModelCacheManager가 없으면 직접 로드 (파라미터 필터링 적용)
                            filtered_fallback_kwargs = _filter_model_kwargs(fallback_model_kwargs) if _filter_model_kwargs else {}
                            self.model = SentenceTransformer(
                                fallback_model, 
                                device="cpu",
                                model_kwargs=filtered_fallback_kwargs
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
                                    use_safetensors=True,  # safetensors 사용 (모델이 safetensors 형식)
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

logger = get_logger(__name__)


class SemanticSearchEngineV2:
    """lawfirm_v2.db 기반 의미적 검색 엔진"""
    
    # 클래스 상수: relaxed_threshold 계산 비율
    RELAXED_THRESHOLD_RATIO = float(os.getenv("RELAXED_THRESHOLD_RATIO", "0.7"))

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
            use_mlflow_index: MLflow 인덱스 사용 여부 (기본값: True)
        """
        # Database URL 또는 path 설정
        if db_path is None:
            from core.utils.config import get_config
            config = get_config()
            # database_url 우선 사용, 없으면 database_path 사용
            database_url = getattr(config, 'database_url', None)
            if database_url:
                self.database_url = database_url
                self.db_path = None
            else:
                raise ValueError("database_url must be set in config. PostgreSQL is required.")
        else:
            # db_path가 제공된 경우 PostgreSQL URL로 변환 시도
            if db_path and (db_path.startswith('postgresql://') or db_path.startswith('postgres://')):
                self.database_url = db_path
                self.db_path = None
            else:
                raise ValueError(f"Invalid database path: {db_path}. PostgreSQL URL is required (e.g., postgresql://user:password@host:port/database)")
        
        self.logger = get_logger(__name__)
        
        # DatabaseAdapter 초기화 (필수)
        if not DatabaseAdapter:
            raise ImportError("DatabaseAdapter is required. PostgreSQL support is mandatory.")
        
        try:
            self._db_adapter = DatabaseAdapter(self.database_url)
            # DatabaseAdapter 초기화 로그는 DatabaseAdapter 내부에서 출력되므로 중복 방지
            # (캐시에서 재사용 시에는 DEBUG 레벨로만 출력됨)
        except Exception as e:
            raise RuntimeError(f"Failed to initialize DatabaseAdapter: {e}") from e
        
        # 연결 풀은 DatabaseAdapter가 관리
        self._connection_pool = None
        
        # KoreanStopwordProcessor 초기화 (KoNLPy 우선 사용, 싱글톤)
        self.stopword_processor = None
        if KoreanStopwordProcessor:
            try:
                self.stopword_processor = KoreanStopwordProcessor.get_instance()
            except Exception as e:
                self.logger.warning(f"Error initializing KoreanStopwordProcessor: {e}")
        
        # 설정에서 MLflow 인덱스 사용 여부 확인 (기본값: True)
        if use_mlflow_index is False:
            # 명시적으로 False로 설정된 경우 환경 변수나 설정 파일 확인
            import os
            env_use_mlflow = os.getenv("USE_MLFLOW_INDEX", "").lower()
            if env_use_mlflow in ("true", "1", "yes"):
                use_mlflow_index = True
                self.logger.info("USE_MLFLOW_INDEX environment variable is set to true")
            elif env_use_mlflow in ("false", "0", "no"):
                # 명시적으로 False로 설정된 경우
                use_mlflow_index = False
                self.logger.warning("USE_MLFLOW_INDEX is explicitly set to false. MLflow will not be used.")
            else:
                # 환경 변수가 없으면 Config에서 확인 (기본값 True)
                try:
                    from core.utils.config import get_config
                    config = get_config()
                    use_mlflow_index = config.use_mlflow_index if hasattr(config, 'use_mlflow_index') else True
                    if use_mlflow_index:
                        mlflow_run_id = mlflow_run_id or (config.mlflow_run_id if hasattr(config, 'mlflow_run_id') else None)
                        self.logger.info("MLflow index will be used as default")
                except Exception as e:
                    self.logger.debug(f"Could not load config for MLflow index settings: {e}")
                    use_mlflow_index = True  # 기본값
        elif not use_mlflow_index:
            # use_mlflow_index가 None이거나 제공되지 않은 경우 기본값 True 사용
            use_mlflow_index = True
            self.logger.info("MLflow index will be used as default (no explicit setting)")
            try:
                from core.utils.config import get_config
                config = get_config()
                mlflow_run_id = mlflow_run_id or (config.mlflow_run_id if hasattr(config, 'mlflow_run_id') else None)
            except Exception as e:
                self.logger.debug(f"Could not load config for MLflow index settings: {e}")
        
        self.use_mlflow_index = use_mlflow_index
        self.mlflow_run_id = mlflow_run_id

        # 모델명이 제공되지 않은 경우, 환경 변수나 Config에서 먼저 확인 (MLflow 초기화 전)
        if model_name is None:
            import os
            model_name = os.getenv("EMBEDDING_MODEL")
            if model_name:
                self.logger.info(f"환경 변수에서 모델 사용: {model_name}")
            
            # 환경 변수도 없으면 Config 기본값 사용 (임시, MLflow에서 덮어쓸 수 있음)
            if model_name is None:
                from ..utils.config import get_config
                config = get_config()
                model_name = config.embedding_model
                self.logger.debug(f"Config 기본값 사용 (임시): {model_name}")

        # FAISS 인덱스 관련 속성
        # 기본 경로: data/embeddings/ml_enhanced_ko_sroberta_precedents/ml_enhanced_faiss_index.faiss
        # 여러 경로 시도 (프로젝트 루트 기준)
        # PostgreSQL을 사용하는 경우 db_path는 None일 수 있음
        if db_path:
            possible_paths = [
                Path(db_path).parent.parent / "embeddings" / "ml_enhanced_ko_sroberta_precedents" / "ml_enhanced_faiss_index.faiss",
                Path("data") / "embeddings" / "ml_enhanced_ko_sroberta_precedents" / "ml_enhanced_faiss_index.faiss",
                Path(db_path).parent / "embeddings" / "ml_enhanced_ko_sroberta_precedents" / "ml_enhanced_faiss_index.faiss",
            ]
            legacy_index_path = Path(db_path).parent / f"{Path(db_path).stem}_faiss.index"
        else:
            # PostgreSQL을 사용하는 경우 프로젝트 루트 기준 경로 사용
            # 프로젝트 루트 찾기 (semantic_search_engine_v2.py -> engines -> search -> core -> lawfirm_langgraph -> 프로젝트 루트)
            try:
                project_root = Path(__file__).parent.parent.parent.parent.parent
            except Exception:
                # except 블록 안에서도 Path를 사용할 수 있도록 별도 import
                from pathlib import Path as PathClass
                project_root = PathClass(".")
                # except 블록 안에서 PathClass를 사용하도록 수정
                possible_paths = [
                    project_root / "data" / "embeddings" / "ml_enhanced_ko_sroberta_precedents" / "ml_enhanced_faiss_index.faiss",
                    PathClass("data") / "embeddings" / "ml_enhanced_ko_sroberta_precedents" / "ml_enhanced_faiss_index.faiss",
                ]
            else:
                # try 블록이 성공한 경우 Path 사용 가능
                possible_paths = [
                    project_root / "data" / "embeddings" / "ml_enhanced_ko_sroberta_precedents" / "ml_enhanced_faiss_index.faiss",
                    Path("data") / "embeddings" / "ml_enhanced_ko_sroberta_precedents" / "ml_enhanced_faiss_index.faiss",
                ]
            legacy_index_path = None
        
        # 새로 빌드된 인덱스를 우선 사용
        default_index_path = None
        for path in possible_paths:
            if path.exists():
                default_index_path = path
                break
        
        if default_index_path:
            self.index_path = str(default_index_path)
            self.logger.info(f"Using default FAISS index: {self.index_path}")
        elif legacy_index_path and legacy_index_path.exists():
            # 레거시 경로 (하위 호환성)
            self.index_path = str(legacy_index_path)
            self.logger.info(f"Using legacy FAISS index: {self.index_path}")
        else:
            # 인덱스가 없으면 기본 경로 설정 (나중에 빌드됨)
            self.index_path = str(possible_paths[0])
            # pgvector 전용 모드에서는 DEBUG 레벨로 변경 (FAISS 불필요)
            self.logger.debug(f"No FAISS index found (not needed for pgvector mode), will use: {self.index_path}")
        
        # 벡터 검색 방법 선택 (pgvector만 사용하도록 강제)
        import os
        vector_search_method = os.getenv("VECTOR_SEARCH_METHOD", "").lower()
        if not vector_search_method:
            try:
                from ..utils.config import get_config
                config = get_config()
                vector_search_method = getattr(config, 'vector_search_method', 'pgvector').lower()
            except Exception:
                vector_search_method = 'pgvector'  # 기본값: pgvector
        
        # pgvector만 사용하도록 강제
        if vector_search_method not in ['pgvector']:
            self.logger.warning(
                f"⚠️ VECTOR_SEARCH_METHOD={vector_search_method} is not supported. "
                f"Only 'pgvector' is allowed. Forcing to 'pgvector'."
            )
            vector_search_method = 'pgvector'
        
        self.vector_search_method = vector_search_method
        # 벡터 검색 방법 로그 (한 번만 출력, 중복 방지)
        if not hasattr(self, '_vector_search_logged'):
            self.logger.info(f"🔍 Vector search method: {self.vector_search_method} (pgvector only)")
            self._vector_search_logged = True
        
        # pgvector 어댑터 초기화 (pgvector만 사용)
        self.pgvector_adapter = None
        if not PGVECTOR_AVAILABLE:
            raise RuntimeError(
                "❌ pgvector is required but not available. "
                "Please install pgvector: pip install pgvector psycopg2-binary"
            )
        else:
            try:
                # 연결 풀에서 연결 가져오기 (나중에 실제 검색 시 사용)
                # 여기서는 어댑터만 초기화하지 않고, 검색 시마다 생성
                # pgvector 사용 로그는 이미 위에서 출력되었으므로 중복 방지
                self.logger.debug("✅ pgvector will be used for vector search (pgvector only mode)")
            except Exception as e:
                raise RuntimeError(
                    f"❌ Failed to initialize pgvector adapter: {e}. "
                    "pgvector is required for this configuration."
                ) from e
        
        self.index = None
        self._chunk_ids = []  # 인덱스와 chunk_id 매핑
        self._chunk_metadata = {}  # chunk_id -> metadata 매핑 (초기화)
        self._index_building = False  # 백그라운드 빌드 중 플래그
        self._build_thread = None  # 빌드 스레드
        self.current_faiss_version = None  # 현재 FAISS 버전 (MLflow 전환 시에도 호환성 유지)
        self.faiss_version_manager = None  # FAISS 버전 관리자 (MLflow 전환 시에도 호환성 유지)

        # 쿼리 벡터 캐싱 (해시 기반 LRU 캐시)
        try:
            import hashlib
            self._use_hash_cache = True
            self._query_vector_cache = {}  # query_hash -> vector
            self._cache_max_size = 1000  # 최대 캐시 크기 (512 → 1000로 증가)
        except ImportError:
            self._use_hash_cache = False
            self._query_vector_cache = {}  # query -> vector
            self._cache_max_size = 512
        
        # 메타데이터 캐싱 (성능 개선, TTL 지원)
        self._metadata_cache = {}  # key -> {'data': metadata, 'timestamp': time.time()}
        self._metadata_cache_max_size = 1000  # 최대 캐시 크기
        self._metadata_cache_ttl = 3600  # TTL: 1시간 (초 단위)
        self._metadata_cache_hits = 0
        self._metadata_cache_misses = 0  # 캐시 미스 수
        self._metadata_cache_last_cleanup = time.time()  # 마지막 정리 시간
        self._metadata_cache_cleanup_interval = 300  # 정리 간격: 5분
        
        # pgvector 연결 풀 워밍업 (환경 변수로 제어 가능)
        warmup_enabled = os.getenv("PGVECTOR_WARMUP", "true").lower() == "true"
        if warmup_enabled:
            self._warmup_pgvector_connections()
        
        # 메타데이터 캐시 워밍업 (환경 변수로 제어 가능)
        metadata_warmup_enabled = os.getenv("METADATA_CACHE_WARMUP", "true").lower() == "true"
        if metadata_warmup_enabled:
            try:
                self._warmup_metadata_cache()
            except Exception as e:
                self.logger.debug(f"Metadata cache warmup failed (non-critical): {e}")
        
        # MLflow 매니저 지연 로딩 (부팅 속도 개선)
        self.mlflow_manager = None
        self._mlflow_initialized = False
        self._mlflow_config = {
            'use_mlflow_index': self.use_mlflow_index,
            'mlflow_run_id': self.mlflow_run_id
        }
        
        # MLflow 초기화는 실제 사용 시점에 수행 (_get_mlflow_manager 메서드에서)
        if False:  # 지연 로딩: 실제 사용 시점에 초기화
            try:
                import sys
                import os
                # scripts/rag 경로 추가 (프로젝트 루트 기준)
                # 프로젝트 루트 찾기: lawfirm_langgraph/core/search/engines/ -> LawFirmAI/
                current_file = Path(__file__).resolve()
                # 여러 방법으로 프로젝트 루트 찾기
                project_root_candidates = []
                
                # 방법 1: lawfirm_langgraph 디렉토리의 부모 찾기
                for parent in [current_file] + list(current_file.parents):
                    if parent.name == "lawfirm_langgraph":
                        project_root_candidates.append(parent.parent)
                        break
                
                # 방법 2: scripts 디렉토리의 부모 찾기
                for parent in [current_file] + list(current_file.parents):
                    if parent.name == "scripts" and (parent / "rag" / "mlflow_manager.py").exists():
                        project_root_candidates.append(parent.parent)
                        break
                
                # 방법 3: 상대 경로로 계산 (lawfirm_langgraph/core/search/engines -> LawFirmAI)
                project_root_candidates.append(current_file.parent.parent.parent.parent.parent)
                
                # 방법 4: 현재 작업 디렉토리 기준
                cwd = Path.cwd()
                if (cwd / "scripts" / "rag" / "mlflow_manager.py").exists():
                    project_root_candidates.append(cwd)
                if (cwd.parent / "scripts" / "rag" / "mlflow_manager.py").exists():
                    project_root_candidates.append(cwd.parent)
                
                # 가능한 경로 생성
                possible_paths = []
                for root in project_root_candidates:
                    scripts_rag_path = root / "scripts" / "rag"
                    if scripts_rag_path.exists() and (scripts_rag_path / "mlflow_manager.py").exists():
                        possible_paths.append(scripts_rag_path)
                
                # 중복 제거 (순서 유지)
                seen = set()
                unique_paths = []
                for path in possible_paths:
                    path_str = str(path)
                    if path_str not in seen:
                        seen.add(path_str)
                        unique_paths.append(path)
                possible_paths = unique_paths
                
                mlflow_manager_imported = False
                for scripts_rag_path in possible_paths:
                    if scripts_rag_path.exists() and (scripts_rag_path / "mlflow_manager.py").exists():
                        if str(scripts_rag_path) not in sys.path:
                            sys.path.insert(0, str(scripts_rag_path))
                        try:
                            from mlflow_manager import MLflowFAISSManager
                            mlflow_manager_imported = True
                            self.logger.info(f"✅ Successfully imported MLflowFAISSManager from {scripts_rag_path}")
                            break
                        except ImportError as import_err:
                            self.logger.debug(f"Failed to import from {scripts_rag_path}: {import_err}")
                            continue
                
                if not mlflow_manager_imported:
                    raise ImportError(f"Could not import mlflow_manager from any of the paths: {[str(p) for p in possible_paths]}")
                from core.utils.config import get_config
                config = get_config()
                
                # 🔥 개선: MLflow 백엔드 전환 (SQLite 기본값 사용)
                # 🔥 개선: Path를 try 블록 밖에서 import하여 except 블록에서도 사용 가능하도록 수정
                from pathlib import Path as PathModule
                tracking_uri = config.mlflow_tracking_uri if hasattr(config, 'mlflow_tracking_uri') and config.mlflow_tracking_uri else None
                if not tracking_uri:
                    # 환경 변수 확인
                    tracking_uri = os.getenv("MLFLOW_TRACKING_URI")
                    if not tracking_uri:
                        # SQLite 백엔드 사용 (FutureWarning 해결)
                        project_root = PathModule(__file__).resolve().parent.parent.parent.parent.parent
                        default_db_path = project_root / "mlflow" / "mlflow.db"
                        default_db_path.parent.mkdir(parents=True, exist_ok=True)
                        tracking_uri = f"sqlite:///{str(default_db_path).replace(os.sep, '/')}"
                        self.logger.info(f"✅ [MLFLOW] Using SQLite backend: {tracking_uri}")
                
                experiment_name = config.mlflow_experiment_name if hasattr(config, 'mlflow_experiment_name') else "faiss_index_versions"
                self.mlflow_manager = MLflowFAISSManager(
                    experiment_name=experiment_name,
                    tracking_uri=tracking_uri
                )
            except ImportError as e:
                self.logger.warning(f"MLflowFAISSManager not available: {e}")
                self.use_mlflow_index = False
            except Exception as e:
                # 🔥 개선: except 블록에서도 Path를 안전하게 사용
                try:
                    from pathlib import Path as PathModule
                except ImportError:
                    PathModule = None
                self.logger.warning(f"Failed to initialize MLflow manager: {e}")
                self.use_mlflow_index = False
        
        # ✅ 방안 1: MLflow manager 초기화 후 모델 정보 확인 (MLflow 인덱스 사용 시)
        # MLflow 인덱스를 사용할 때는 항상 MLflow의 모델 정보를 최우선 사용
        # 지연 로딩: 모델 로딩 시점에 MLflow 초기화 및 모델 정보 확인
        if False:  # 지연 로딩: _get_mlflow_manager()에서 처리
            try:
                run_id = self.mlflow_run_id
                if not run_id:
                    mlflow_manager = self._get_mlflow_manager()
                if mlflow_manager:
                    run_id = mlflow_manager.get_production_run()
                else:
                    run_id = None
                
                if run_id:
                    import mlflow
                    version_info = None
                    if hasattr(self.mlflow_manager, 'load_version_info_from_local'):
                        self.logger.debug(f"로컬 파일 시스템에서 version_info.json 로드 시도: run_id={run_id}")
                        mlflow_manager = self._get_mlflow_manager()
                        if mlflow_manager:
                            version_info = mlflow_manager.load_version_info_from_local(run_id)
                        else:
                            version_info = None
                        if version_info:
                            self.logger.info("✅ 로컬 파일 시스템에서 version_info.json 직접 로드 완료")
                    
                    if version_info is None:
                        self.logger.debug("로컬 경로에서 version_info.json을 찾을 수 없어 MLflow에서 다운로드 시도")
                        version_info = mlflow.artifacts.load_dict(f"runs:/{run_id}/version_info.json")
                    
                    embedding_config = version_info.get('embedding_config', {})
                    mlflow_model_name = embedding_config.get('model')
                    if mlflow_model_name:
                        # MLflow에서 모델 정보를 찾았으면 무조건 사용 (기존 model_name 덮어쓰기)
                        model_name = mlflow_model_name.strip().strip('"').strip("'")
                        self.logger.info(f"✅ MLflow에서 모델 감지: {model_name} (run_id: {run_id})")
                        # MLflow 모델 정보에서 차원 정보도 확인
                        mlflow_dimension = embedding_config.get('dimension')
                        if mlflow_dimension:
                            self.logger.info(f"   MLflow 인덱스 차원: {mlflow_dimension}")
                    else:
                        self.logger.warning(f"⚠️  MLflow version_info에 모델 정보가 없습니다. embedding_config: {embedding_config}")
                        if model_name is None:
                            self.logger.warning("   환경 변수 또는 Config 기본값을 사용합니다.")
                else:
                    self.logger.warning("⚠️  MLflow run_id를 찾을 수 없습니다.")
                    if model_name is None:
                        self.logger.warning("   환경 변수 또는 Config 기본값을 사용합니다.")
            except Exception as e:
                self.logger.warning(f"⚠️  MLflow에서 모델 정보를 가져올 수 없습니다: {e}")
                if model_name is None:
                    self.logger.warning("   환경 변수 또는 Config 기본값을 사용합니다.")
        
        # MLflow에서 모델을 가져오지 못한 경우 최종 확인
        if model_name is None:
            import os
            model_name = os.getenv("EMBEDDING_MODEL")
            if model_name:
                self.logger.info(f"환경 변수에서 모델 사용: {model_name}")
            
            if model_name is None:
                from ..utils.config import get_config
                config = get_config()
                model_name = config.embedding_model
                self.logger.warning(f"⚠️  MLflow 및 환경 변수에서 모델을 찾을 수 없어 Config 기본값 사용: {model_name}")
        
        # 성능 모니터링 초기화
        self.performance_monitor = None
        self.enable_performance_monitoring = False
        try:
            # Path는 이미 파일 상단에서 import되었으므로 사용 가능
            # 하지만 except 블록에서 Path를 재정의할 수 있으므로, 로컬 변수로 명시적으로 사용
            from pathlib import Path as PathModule
            scripts_utils_path = PathModule(__file__).parent.parent.parent / "scripts" / "utils"
            if scripts_utils_path.exists():
                sys.path.insert(0, str(scripts_utils_path))
            from version_performance_monitor import VersionPerformanceMonitor
            # PostgreSQL을 사용하는 경우 db_path는 None일 수 있음
            if db_path:
                performance_log_path = PathModule(db_path).parent / "performance_logs"
            else:
                # 프로젝트 루트 기준 경로 사용
                try:
                    project_root = PathModule(__file__).parent.parent.parent.parent.parent
                except Exception:
                    # except 블록 안에서도 PathModule을 사용할 수 있도록 별도 import
                    from pathlib import Path as PathClass
                    project_root = PathClass(".")
                performance_log_path = project_root / "data" / "performance_logs"
            self.performance_monitor = VersionPerformanceMonitor(str(performance_log_path))
            self.enable_performance_monitoring = True
        except ImportError:
            self.logger.debug("VersionPerformanceMonitor not available")

        # 모델 이름 최종 설정 (MLflow에서 가져온 경우 업데이트)
        if model_name:
            model_name = model_name.strip().strip('"').strip("'")
        self.model_name = model_name
        
        # 임베딩 모델 로드
        self.embedder = None
        self.dim = None
        self._initialize_embedder(self.model_name)

        if db_path and not Path(db_path).exists():
            self.logger.warning(f"Database {db_path} not found")
        elif not db_path:
            self.logger.debug("Using PostgreSQL database (db_path is None)")

        # 🔥 개선: pgvector를 사용하는 경우 FAISS 인덱스 로드 건너뛰기
        # pgvector는 DB에서 직접 검색하므로 FAISS 인덱스가 필요 없음
        if PGVECTOR_AVAILABLE and self.vector_search_method == 'pgvector':
            self.index = None
        # FAISS 인덱스 로드 (MLflow만 사용)
        elif FAISS_AVAILABLE and self.embedder:
            if self.use_mlflow_index:
                # MLflow 벡터 스토어 사용 (기본값)
                if not self.mlflow_manager:
                    self.logger.warning("⚠️ MLflow manager가 초기화되지 않았습니다. DB 기반 인덱스로 폴백합니다.")
                    try:
                        self._load_faiss_index()
                        if self.index is not None:
                            self.logger.info("✅ DB 기반 인덱스 초기화 성공 (MLflow 매니저 없음으로 인한 폴백)")
                            self.use_mlflow_index = False
                        else:
                            self.logger.warning("⚠️ DB 기반 인덱스도 로드 실패. 인덱스는 나중에 빌드될 수 있습니다.")
                    except Exception as fallback_error:
                        self.logger.warning(f"⚠️ DB 기반 인덱스 폴백 실패: {fallback_error}. 인덱스는 나중에 빌드될 수 있습니다.")
                else:
                    try:
                        self._load_mlflow_index()
                        self.logger.info("✅ MLflow 벡터 스토어 로드 성공")
                    except (RuntimeError, ImportError, Exception) as e:
                        self.logger.warning(f"⚠️ MLflow 인덱스 초기화 실패: {e}")
                        self.logger.info("🔄 DB 기반 인덱스로 폴백 시도 중...")
                        # MLflow 인덱스 사용 불가 시 DB 기반 인덱스로 폴백
                        try:
                            self._load_faiss_index()
                            if self.index is not None:
                                self.logger.info("✅ DB 기반 인덱스 초기화 성공 (MLflow 폴백)")
                                # MLflow 사용 비활성화
                                self.use_mlflow_index = False
                            else:
                                self.logger.warning("⚠️ DB 기반 인덱스도 로드 실패. 인덱스는 나중에 빌드될 수 있습니다.")
                        except Exception as fallback_error:
                            self.logger.warning(f"⚠️ DB 기반 인덱스 폴백 실패: {fallback_error}. 인덱스는 나중에 빌드될 수 있습니다.")
                            # 초기화 단계에서는 에러를 발생시키지 않고 경고만 출력
                            # 인덱스는 나중에 검색 시 자동으로 빌드됨
            else:
                # MLflow가 비활성화된 경우 (인덱스 빌드 모드 등)
                # 인덱스는 나중에 빌드되거나 로드될 수 있으므로 에러를 발생시키지 않음
                self.logger.info("ℹ️  MLflow 인덱스 비활성화됨 (인덱스 빌드 모드 또는 다른 용도)")
                self.index = None
    
    def _get_mlflow_manager(self):
        """MLflow Manager 지연 로딩 (부팅 속도 개선)"""
        if not self._mlflow_initialized:
            if self._mlflow_config.get('use_mlflow_index', False):
                try:
                    import sys
                    import os
                    # scripts/rag 경로 추가 (프로젝트 루트 기준)
                    current_file = Path(__file__).resolve()
                    project_root_candidates = []
                    
                    # 방법 1: lawfirm_langgraph 디렉토리의 부모 찾기
                    for parent in [current_file] + list(current_file.parents):
                        if parent.name == "lawfirm_langgraph":
                            project_root_candidates.append(parent.parent)
                            break
                    
                    # 방법 2: scripts 디렉토리의 부모 찾기
                    for parent in [current_file] + list(current_file.parents):
                        if parent.name == "scripts" and (parent / "rag" / "mlflow_manager.py").exists():
                            project_root_candidates.append(parent.parent)
                            break
                    
                    # 방법 3: 상대 경로로 계산
                    project_root_candidates.append(current_file.parent.parent.parent.parent.parent)
                    
                    # 방법 4: 현재 작업 디렉토리 기준
                    cwd = Path.cwd()
                    if (cwd / "scripts" / "rag" / "mlflow_manager.py").exists():
                        project_root_candidates.append(cwd)
                    if (cwd.parent / "scripts" / "rag" / "mlflow_manager.py").exists():
                        project_root_candidates.append(cwd.parent)
                    
                    # 가능한 경로 생성
                    possible_paths = []
                    for root in project_root_candidates:
                        scripts_rag_path = root / "scripts" / "rag"
                        if scripts_rag_path.exists() and (scripts_rag_path / "mlflow_manager.py").exists():
                            possible_paths.append(scripts_rag_path)
                    
                    # 중복 제거
                    seen = set()
                    unique_paths = []
                    for path in possible_paths:
                        path_str = str(path)
                        if path_str not in seen:
                            seen.add(path_str)
                            unique_paths.append(path)
                    possible_paths = unique_paths
                    
                    mlflow_manager_imported = False
                    for scripts_rag_path in possible_paths:
                        if scripts_rag_path.exists() and (scripts_rag_path / "mlflow_manager.py").exists():
                            if str(scripts_rag_path) not in sys.path:
                                sys.path.insert(0, str(scripts_rag_path))
                            try:
                                from mlflow_manager import MLflowFAISSManager
                                mlflow_manager_imported = True
                                self.logger.info(f"✅ Successfully imported MLflowFAISSManager from {scripts_rag_path} (lazy loading)")
                                break
                            except ImportError as import_err:
                                self.logger.debug(f"Failed to import from {scripts_rag_path}: {import_err}")
                                continue
                    
                    if not mlflow_manager_imported:
                        raise ImportError(f"Could not import mlflow_manager from any of the paths: {[str(p) for p in possible_paths]}")
                    
                    from core.utils.config import get_config
                    config = get_config()
                    
                    from pathlib import Path as PathModule
                    tracking_uri = config.mlflow_tracking_uri if hasattr(config, 'mlflow_tracking_uri') and config.mlflow_tracking_uri else None
                    if not tracking_uri:
                        tracking_uri = os.getenv("MLFLOW_TRACKING_URI")
                        if not tracking_uri:
                            project_root = PathModule(__file__).resolve().parent.parent.parent.parent.parent
                            default_db_path = project_root / "mlflow" / "mlflow.db"
                            default_db_path.parent.mkdir(parents=True, exist_ok=True)
                            tracking_uri = f"sqlite:///{str(default_db_path).replace(os.sep, '/')}"
                            self.logger.info(f"✅ [MLFLOW] Using SQLite backend: {tracking_uri}")
                    
                    experiment_name = config.mlflow_experiment_name if hasattr(config, 'mlflow_experiment_name') else "faiss_index_versions"
                    self.mlflow_manager = MLflowFAISSManager(
                        experiment_name=experiment_name,
                        tracking_uri=tracking_uri
                    )
                    self._mlflow_initialized = True
                    self.logger.debug("✅ MLflow Manager initialized (lazy loading)")
                except ImportError as e:
                    self.logger.warning(f"MLflowFAISSManager not available: {e}")
                    self.use_mlflow_index = False
                    self._mlflow_initialized = True
                except Exception as e:
                    self.logger.warning(f"Failed to initialize MLflow manager (lazy loading): {e}")
                    self.use_mlflow_index = False
                    self._mlflow_initialized = True
            else:
                self._mlflow_initialized = True
        
        return self.mlflow_manager
    
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
            # 모델 이름 정리 (따옴표 및 공백 제거)
            if model_name:
                model_name = model_name.strip().strip('"').strip("'").strip()
            
            if not model_name:
                self.logger.error("Model name is empty after cleaning")
                return False
            
            self.embedder = SentenceEmbedder(model_name)
            self.dim = self.embedder.dim
            self.logger.info(f"Embedding model loaded: {model_name}, dim={self.dim}")
            
            # 메모리 정리: 모델 로딩 후 가비지 컬렉션
            collected = gc.collect()
            if collected > 0:
                self.logger.debug(f"Garbage collection after model loading: {collected} objects collected")
            
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
                    wait_time = min(0.1 * (2 ** retry_count), 0.5)
                    time.sleep(wait_time)
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
                    wait_time = min(0.1 * (2 ** retry_count), 0.5)
                    time.sleep(wait_time)
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
                wait_time = min(0.1 * (2 ** retry_count), 0.5)
                time.sleep(wait_time)
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
        # PostgreSQL을 사용하는 경우 db_path는 None일 수 있음
        if self.db_path and not Path(self.db_path).exists():
            return False
        elif not self.db_path:
            # PostgreSQL을 사용하는 경우 데이터베이스 연결 확인
            return self._db_adapter is not None and self._db_adapter.db_type == 'postgresql'
        
        if not self._ensure_embedder_initialized():
            return False
        
        try:
            with self._get_connection_context() as conn:
                cursor = conn.cursor()
                cursor.execute("SELECT COUNT(*) FROM embeddings LIMIT 1")
                row = cursor.fetchone()
                if row is None:
                    return False
                # PostgreSQL RealDictRow는 dict-like 접근 필요
                count = row.get('count') if hasattr(row, 'get') else (row[0] if isinstance(row, (tuple, list)) else 0)
                return count > 0
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
        
        # PostgreSQL을 사용하는 경우 db_path는 None일 수 있음
        if self.db_path:
            diagnosis["db_exists"] = Path(self.db_path).exists()
        else:
            # PostgreSQL을 사용하는 경우 데이터베이스 연결 확인
            diagnosis["db_exists"] = self._db_adapter is not None and self._db_adapter.db_type == 'postgresql'
        if not diagnosis["db_exists"]:
            diagnosis["issues"].append(f"Database not found: {self.db_path}")
            diagnosis["recommendations"].append("Check database path configuration")
            return diagnosis
        
        diagnosis["embedder_initialized"] = self._ensure_embedder_initialized()
        if not diagnosis["embedder_initialized"]:
            diagnosis["issues"].append("Embedder not initialized")
            diagnosis["recommendations"].append("Check embedding model availability")
            return diagnosis
        
        # pgvector를 사용하는 경우 FAISS 인덱스 체크 건너뛰기
        if PGVECTOR_AVAILABLE and self.vector_search_method == 'pgvector':
            diagnosis["faiss_index_exists"] = True  # pgvector는 FAISS가 필요 없음
            diagnosis["recommendations"].append("Using pgvector - FAISS index not required")
        else:
            diagnosis["faiss_index_exists"] = Path(self.index_path).exists()
            if not diagnosis["faiss_index_exists"]:
                diagnosis["issues"].append(f"FAISS index not found: {self.index_path}")
                diagnosis["recommendations"].append("FAISS index will be built on first search")
        
        try:
            with self._get_connection_context() as conn:
                cursor = conn.cursor()
                cursor.execute("SELECT COUNT(*) FROM embeddings")
                row = cursor.fetchone()
                if row:
                    # PostgreSQL RealDictRow는 dict-like 접근 필요
                    diagnosis["embeddings_count"] = row.get('count') if hasattr(row, 'get') else (row[0] if isinstance(row, (tuple, list)) else 0)
                else:
                    diagnosis["embeddings_count"] = 0
            
            if diagnosis["embeddings_count"] == 0:
                diagnosis["issues"].append("No embeddings found in database")
                diagnosis["recommendations"].append("Run embedding generation script")
        except Exception as e:
            # 예외 정보를 상세히 기록
            import traceback
            error_type = type(e).__name__
            error_message = str(e) if e else "Unknown error"
            error_repr = repr(e) if e else "Unknown error"
            
            # 예외 객체가 비정상적인 경우를 감지하고 상세 정보 기록
            if not error_message or error_message == "0" or error_repr == "0":
                error_detail = (
                    f"{error_type} (abnormal exception object: message='{error_message}', "
                    f"repr='{error_repr}')"
                )
                self.logger.warning(
                    f"Abnormal exception when checking embeddings: {error_detail}\n"
                    f"Traceback:\n{traceback.format_exc()}"
                )
                diagnosis["issues"].append(f"Error checking embeddings: {error_detail}")
            else:
                diagnosis["issues"].append(f"Error checking embeddings: {error_type}: {error_message}")
            diagnosis["recommendations"].append("Check database schema and connection")
        
        diagnosis["available"] = (
            diagnosis["db_exists"] and
            diagnosis["embedder_initialized"] and
            diagnosis["embeddings_count"] > 0
        )
        
        return diagnosis

    def _get_active_embedding_version_id(self, data_type: Optional[str] = None) -> Optional[int]:
        """
        활성 임베딩 버전 ID 조회 (data_type별)
        
        Args:
            data_type: 'statutes' 또는 'precedents' (None이면 첫 번째 활성 버전)
        
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
            
            # PostgreSQL을 사용하는 경우 db_path는 None일 수 있음
            if self.db_path and not Path(self.db_path).exists():
                self.logger.debug(f"Database file not found: {self.db_path}")
                return None
            elif not self.db_path:
                # PostgreSQL을 사용하는 경우 데이터베이스 연결 확인
                if not (self._db_adapter and self._db_adapter.db_type == 'postgresql'):
                    return None

            with self._get_connection_context() as conn:
                cursor = conn.cursor()

                # 🔥 개선: data_type별로 활성 버전 조회
                # 실제 스키마: version (integer), data_type (varchar) 컬럼 사용
                # 005_add_embedding_version_management_postgresql.sql 스키마 기준
                if data_type:
                    cursor.execute("""
                        SELECT id, version, data_type, is_active
                        FROM embedding_versions
                        WHERE is_active = TRUE AND data_type = %s
                        ORDER BY created_at DESC
                        LIMIT 1
                    """, (data_type,))
                else:
                    # data_type이 지정되지 않은 경우 첫 번째 활성 버전 반환 (하위 호환성)
                    cursor.execute("""
                        SELECT id, version, data_type, is_active
                        FROM embedding_versions
                        WHERE is_active = TRUE
                        ORDER BY created_at DESC
                        LIMIT 1
                    """)
                
                row = cursor.fetchone()

                if row:
                    version_id = row['id'] if hasattr(row, 'get') else row[0]
                    # PostgreSQL Row는 dict-like 또는 tuple로 반환됨
                    # 실제 스키마: version (integer), data_type (varchar) 컬럼 사용
                    version_num = row.get('version') if hasattr(row, 'get') else (row[1] if len(row) > 1 else version_id)
                    row_data_type = row.get('data_type') if hasattr(row, 'get') else (row[2] if len(row) > 2 else None)
                    version_name = f'v{version_num}' if version_num else f'v{version_id}'
                    if row_data_type:
                        version_name = f'{version_name}-{row_data_type}'
                    self.logger.info(
                        f"✅ Active embedding version detected: ID={version_id}, "
                        f"version={version_num}, data_type={row_data_type}, name={version_name}"
                    )
                    return version_id
                else:
                    if data_type:
                        self.logger.warning(f"⚠️  No active embedding version found for data_type={data_type} in database")
                    else:
                        self.logger.warning("⚠️  No active embedding version found in database")
                    return None

        except Exception as e:
            if "no such table" in str(e).lower():
                self.logger.debug(f"embedding_versions table not found: {e}")
            else:
                self.logger.warning(f"⚠️  Error getting active embedding version: {e}")
            return None

    def _determine_data_type_from_source_types(self, source_types: Optional[List[str]]) -> Optional[str]:
        """
        source_types에서 data_type 결정
        
        Args:
            source_types: source_type 목록
        
        Returns:
            'statutes', 'precedents', 또는 None (혼합된 경우)
        """
        if not source_types:
            return None
        
        # source_type -> data_type 매핑
        statute_types = {'statute_article', 'statute_articles'}
        precedent_types = {
            'case_paragraph', 'precedent_content', 'case',
            'decision_paragraph', 'interpretation_paragraph'
        }
        
        has_statutes = any(st in statute_types for st in source_types)
        has_precedents = any(st in precedent_types for st in source_types)
        
        if has_statutes and not has_precedents:
            return 'statutes'
        elif has_precedents and not has_statutes:
            return 'precedents'
        else:
            # 혼합된 경우 None 반환 (모든 버전 검색)
            return None

    def _get_model_name_for_data_type(self, data_type: Optional[str] = None) -> Optional[str]:
        """
        data_type에 따른 활성 버전의 모델명 조회
        
        Args:
            data_type: 'statutes' 또는 'precedents' (None이면 첫 번째 활성 버전)
        
        Returns:
            모델명 또는 None
        """
        try:
            active_version_id = self._get_active_embedding_version_id(data_type=data_type)
            if not active_version_id:
                return None
            
            with self._get_connection_context() as conn:
                cursor = conn.cursor()
                cursor.execute("""
                    SELECT model_name
                    FROM embedding_versions
                    WHERE id = %s
                """, (active_version_id,))
                row = cursor.fetchone()
                if row:
                    model_name = row.get('model_name') if hasattr(row, 'get') else (row[0] if len(row) > 0 else None)
                    if model_name:
                        model_name = model_name.strip().strip('"').strip("'")
                        self.logger.debug(
                            f"📋 [MODEL] Active version (ID={active_version_id}, data_type={data_type}) "
                            f"uses model: {model_name}"
                        )
                        return model_name
            return None
        except Exception as e:
            self.logger.warning(f"Failed to get model name for data_type={data_type}: {e}")
            return None

    def _get_embedding_dimension_for_version(self, version_id: Optional[int]) -> Optional[int]:
        """
        특정 버전의 임베딩 차원 조회
        
        Args:
            version_id: 임베딩 버전 ID
        
        Returns:
            차원 또는 None
        """
        if version_id is None:
            return None
        
        try:
            with self._get_connection_context() as conn:
                cursor = conn.cursor()
                cursor.execute("""
                    SELECT dim
                    FROM embedding_versions
                    WHERE id = %s
                """, (version_id,))
                row = cursor.fetchone()
                if row:
                    dim = row.get('dim') if hasattr(row, 'get') else (row[0] if len(row) > 0 else None)
                    if dim is not None:
                        return int(dim)
            return None
        except Exception as e:
            self.logger.debug(f"Failed to get dimension for version_id={version_id}: {e}")
            return None

    def _ensure_correct_embedding_model(self, data_type: Optional[str] = None) -> bool:
        """
        검색 시 저장된 임베딩 모델과 일치하는 모델 사용 보장
        
        Args:
            data_type: 'statutes' 또는 'precedents'
        
        Returns:
            모델이 올바르게 설정되었는지 여부
        """
        try:
            # data_type에 따른 활성 버전의 모델명 조회
            required_model = self._get_model_name_for_data_type(data_type=data_type)
            if not required_model:
                # 모델명을 찾을 수 없으면 현재 모델 유지 (메시지 출력하지 않음)
                return True
            
            # 현재 모델과 필요한 모델이 일치하는지 확인
            current_model = self.model_name
            if current_model and current_model.strip().strip('"').strip("'") == required_model:
                # 모델이 일치하면 그대로 사용
                self.logger.debug(
                    f"✅ [MODEL] Current model matches required model: {required_model}"
                )
                return True
            
            # 모델이 다르면 필요한 모델로 재초기화
            self.logger.info(
                f"🔄 [MODEL] Model mismatch detected. "
                f"Current: {current_model}, Required: {required_model} (data_type={data_type}). "
                f"Re-initializing embedder..."
            )
            
            # 모델 재초기화
            if self._initialize_embedder(required_model):
                self.model_name = required_model
                self.logger.info(
                    f"✅ [MODEL] Embedder re-initialized with correct model: {required_model}"
                )
                return True
            else:
                self.logger.warning(
                    f"⚠️ [MODEL] Failed to re-initialize embedder with model: {required_model}. "
                    f"Using current model: {current_model}"
                )
                return False
                
        except Exception as e:
            self.logger.warning(f"Error ensuring correct embedding model: {e}")
            return False

    def _get_version_chunk_count(self, version_id: int) -> int:
        """
        특정 버전의 청크 수 조회 (모든 벡터 테이블 확인)
        
        🔥 개선: precedent_chunks만 확인하던 것을 모든 벡터 테이블을 확인하도록 수정

        Args:
            version_id: 임베딩 버전 ID

        Returns:
            청크 수
        """
        try:
            # PostgreSQL을 사용하는 경우 db_path는 None일 수 있음
            if self.db_path and not Path(self.db_path).exists():
                return 0
            elif not self.db_path:
                # PostgreSQL을 사용하는 경우 데이터베이스 연결 확인
                if not (self._db_adapter and self._db_adapter.db_type == 'postgresql'):
                    return 0

            with self._get_connection_context() as conn:
                cursor = conn.cursor()
                
                # 🔥 개선: 모든 벡터 테이블 확인
                tables_to_check = [
                    'precedent_chunks',
                    'statute_embeddings',
                    'statute_articles',
                    'interpretation_paragraphs',
                    'decision_paragraphs',
                    'precedent_contents'
                ]
                
                total_count = 0
                
                for table_name in tables_to_check:
                    try:
                        # 테이블 존재 확인
                        cursor.execute("""
                            SELECT EXISTS (
                                SELECT 1 FROM information_schema.tables 
                                WHERE table_name = %s
                            )
                        """, (table_name,))
                        table_exists = cursor.fetchone()
                        if not table_exists or (isinstance(table_exists, tuple) and not table_exists[0]) or (hasattr(table_exists, 'get') and not table_exists.get('exists', False)):
                            continue
                        
                        # embedding_version 컬럼 존재 확인
                        cursor.execute("""
                            SELECT EXISTS (
                                SELECT 1 FROM information_schema.columns 
                                WHERE table_name = %s AND column_name = 'embedding_version'
                            )
                        """, (table_name,))
                        has_version_col = cursor.fetchone()
                        has_version = (isinstance(has_version_col, tuple) and has_version_col[0]) or (hasattr(has_version_col, 'get') and has_version_col.get('exists', False))
                        
                        if has_version:
                            # embedding_version으로 필터링
                            cursor.execute(f"""
                                SELECT COUNT(*) as count
                                FROM {table_name}
                                WHERE embedding_version = %s
                            """, (version_id,))
                        else:
                            # embedding_version 컬럼이 없으면 전체 카운트 (버전 구분 없음)
                            cursor.execute(f"SELECT COUNT(*) as count FROM {table_name}")
                        
                        row = cursor.fetchone()
                        if row:
                            count = row['count'] if hasattr(row, 'get') else row[0]
                            total_count += count
                            if count > 0:
                                self.logger.debug(f"Found {count} chunks in {table_name} for version {version_id}")
                    except Exception as table_error:
                        # 테이블이 없거나 오류가 발생해도 계속 진행
                        self.logger.debug(f"Error checking {table_name}: {table_error}")
                        continue
                
                return total_count

        except Exception as e:
            if "no such table" in str(e).lower():
                self.logger.debug(f"Table not found: {e}")
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
            # PostgreSQL을 사용하는 경우 db_path는 None일 수 있음
            if self.db_path and not Path(self.db_path).exists():
                self.logger.warning(f"Database {self.db_path} not found for model detection")
                return None
            elif not self.db_path:
                # PostgreSQL을 사용하는 경우 데이터베이스 연결 확인
                if not (self._db_adapter and self._db_adapter.db_type == 'postgresql'):
                    return None

            with self._get_connection_context() as conn:
                cursor = conn.cursor()

                # 먼저 활성 버전의 모델 조회 시도
                active_version_id = self._get_active_embedding_version_id()
                if active_version_id:
                    # embedding_versions 테이블에서 직접 모델명 조회 (더 정확함)
                    cursor.execute("""
                        SELECT model_name
                        FROM embedding_versions
                        WHERE id = %s
                    """, (active_version_id,))
                    row = cursor.fetchone()
                    if row:
                        model_name = row.get('model_name') if hasattr(row, 'get') else (row[0] if len(row) > 0 else None)
                        if model_name:
                            detected_model = model_name
                            # 따옴표 제거 (데이터베이스에서 따옴표가 포함된 경우)
                            if detected_model:
                                detected_model = detected_model.strip().strip('"').strip("'")
                            self.logger.info(
                                f"Detected embedding model from active version (ID={active_version_id}): "
                                f"{detected_model}"
                            )
                            return detected_model
                    
                    # embedding_versions에 모델명이 없는 경우, precedent_chunks에서 직접 조회 불가 (embedding_vector만 있음)
                    # 대신 embedding_versions의 model_name을 사용하거나, 다른 방법 사용
                    # 현재는 embedding_versions에서만 모델 정보를 가져옴
                    pass
                    row = cursor.fetchone()
                    if row:
                        detected_model = row.get('model') if hasattr(row, 'get') else row[0]
                        # 따옴표 제거 (데이터베이스에서 따옴표가 포함된 경우)
                        if detected_model:
                            detected_model = detected_model.strip().strip('"').strip("'")
                        count = row.get('count') if hasattr(row, 'get') else (row[1] if len(row) > 1 else 0)
                        self.logger.info(
                            f"Detected embedding model from active version embeddings (ID={active_version_id}): "
                            f"{detected_model} (count: {count})"
                        )
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

                if row:
                    detected_model = row.get('model') if hasattr(row, 'get') else row[0]
                    # 따옴표 제거 (데이터베이스에서 따옴표가 포함된 경우)
                    if detected_model:
                        detected_model = detected_model.strip().strip('"').strip("'")
                    count = row.get('count') if hasattr(row, 'get') else (row[1] if len(row) > 1 else 0)
                    self.logger.info(f"Detected embedding model from database: {detected_model} (count: {count})")
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

    def _get_model_name_for_data_type(self, data_type: Optional[str] = None) -> Optional[str]:
        """
        data_type에 따른 활성 버전의 모델명 조회
        
        Args:
            data_type: 'statutes' 또는 'precedents' (None이면 첫 번째 활성 버전)
        
        Returns:
            모델명 또는 None (에러 발생 시 현재 모델 반환)
        """
        # data_type 정규화 (오타 수정)
        if data_type:
            data_type = data_type.strip().lower()
            if data_type == 'precedentss':
                data_type = 'precedents'
            elif data_type == 'statutess':
                data_type = 'statutes'
        
        try:
            active_version_id = self._get_active_embedding_version_id(data_type=data_type)
            if not active_version_id:
                # 활성 버전을 찾을 수 없으면 현재 모델 반환 (fallback)
                if self.model_name:
                    self.logger.debug(
                        f"📋 [MODEL] No active version found for data_type={data_type}, "
                        f"using current model: {self.model_name}"
                    )
                    return self.model_name
                return None
            
            with self._get_connection_context() as conn:
                cursor = conn.cursor()
                cursor.execute("""
                    SELECT model_name
                    FROM embedding_versions
                    WHERE id = %s
                """, (active_version_id,))
                row = cursor.fetchone()
                if row:
                    model_name = row.get('model_name') if hasattr(row, 'get') else (row[0] if len(row) > 0 else None)
                    if model_name:
                        model_name = model_name.strip().strip('"').strip("'")
                        self.logger.debug(
                            f"📋 [MODEL] Active version (ID={active_version_id}, data_type={data_type}) "
                            f"uses model: {model_name}"
                        )
                        return model_name
            # 모델명을 찾을 수 없으면 현재 모델 반환 (fallback)
            if self.model_name:
                self.logger.debug(
                    f"📋 [MODEL] Model name not found in version (ID={active_version_id}), "
                    f"using current model: {self.model_name}"
                )
                return self.model_name
            return None
        except Exception as e:
            # 연결 풀 타임아웃 등 에러 발생 시 현재 모델 반환 (fallback)
            if self.model_name:
                self.logger.debug(
                    f"📋 [MODEL] Error getting model name for data_type={data_type} (fallback to current model): {e}"
                )
                return self.model_name
            # 현재 모델도 없으면 에러만 로깅하고 None 반환
            self.logger.debug(f"Failed to get model name for data_type={data_type}: {e}")
            return None

    def _ensure_correct_embedding_model(self, data_type: Optional[str] = None) -> bool:
        """
        검색 시 저장된 임베딩 모델과 일치하는 모델 사용 보장
        
        Args:
            data_type: 'statutes' 또는 'precedents'
        
        Returns:
            모델이 올바르게 설정되었는지 여부
        """
        try:
            # data_type에 따른 활성 버전의 모델명 조회
            required_model = self._get_model_name_for_data_type(data_type=data_type)
            if not required_model:
                # 모델명을 찾을 수 없으면 현재 모델 유지 (메시지 출력하지 않음)
                return True
            
            # 현재 모델과 필요한 모델이 일치하는지 확인
            current_model = self.model_name
            if current_model:
                current_model = current_model.strip().strip('"').strip("'")
            
            if current_model == required_model:
                # 모델이 일치하면 그대로 사용
                self.logger.debug(
                    f"✅ [MODEL] Current model matches required model: {required_model}"
                )
                return True
            
            # 모델이 다르면 필요한 모델로 재초기화
            self.logger.info(
                f"🔄 [MODEL] Model mismatch detected. "
                f"Current: {current_model}, Required: {required_model} (data_type={data_type}). "
                f"Re-initializing embedder..."
            )
            
            # 🔥 개선: 모델 변경 시 관련 캐시 무효화
            # 캐시 키에 모델명이 포함되어 있으므로 자동으로 다른 캐시를 사용하지만,
            # 성능 최적화를 위해 큰 캐시는 정리
            if len(self._query_vector_cache) > 100:
                self.logger.debug(
                    f"🔄 [CACHE] Clearing query vector cache due to model change "
                    f"({current_model} -> {required_model}, cache size: {len(self._query_vector_cache)})"
                )
                self._query_vector_cache.clear()
            
            # 모델 재초기화
            if self._initialize_embedder(required_model):
                self.model_name = required_model
                self.logger.info(
                    f"✅ [MODEL] Embedder re-initialized with correct model: {required_model}"
                )
                return True
            else:
                self.logger.warning(
                    f"⚠️ [MODEL] Failed to re-initialize embedder with model: {required_model}. "
                    f"Using current model: {current_model}"
                )
                return False
                
        except Exception as e:
            self.logger.warning(f"Error ensuring correct embedding model: {e}")
            return False

    def _get_connection(self):
        """Get database connection (PostgreSQL only, using DatabaseAdapter)"""
        if not self._db_adapter:
            raise RuntimeError("DatabaseAdapter is required. PostgreSQL database must be configured via DATABASE_URL.")
        return self._db_adapter.get_connection()
    
    def _get_connection_context(self):
        """Get database connection context manager (PostgreSQL only)"""
        if not self._db_adapter:
            raise RuntimeError("DatabaseAdapter is required. PostgreSQL database must be configured via DATABASE_URL.")
        return self._db_adapter.get_connection_context()
    
    def _safe_close_connection(self, conn):
        """연결을 안전하게 풀에 반환 (DatabaseAdapter 사용 시)"""
        if not self._db_adapter:
            raise RuntimeError("DatabaseAdapter is required. PostgreSQL database must be configured via DATABASE_URL.")
        
        # PostgreSQL 연결 풀에 반환
        if conn and hasattr(conn, 'conn'):
            try:
                self._db_adapter.connection_pool.putconn(conn.conn)
            except Exception as e:
                self.logger.warning(f"Error returning connection to pool: {e}")

    def _load_chunk_vectors(self,
                           source_types: Optional[List[str]] = None,
                           limit: Optional[int] = None,
                           embedding_version_id: Optional[int] = None,
                           chunk_ids: Optional[List[int]] = None) -> Dict[int, np.ndarray]:
        """
        source_types에 따라 올바른 테이블에서 벡터 로드
        
        - statute_article → statute_embeddings 테이블
        - case_paragraph, precedent_content → precedent_chunks 테이블

        Args:
            source_types: 필터링할 source_type 목록 (None이면 전체)
            limit: 최대 로드 개수 (None이면 전체)
            embedding_version_id: 임베딩 버전 ID 필터 (None이면 활성 버전만)
            chunk_ids: 로드할 특정 chunk_id 목록 (None이면 전체, 성능 최적화용)

        Returns:
            {chunk_id: vector} 딕셔너리
        """
        try:
            # data_type 결정
            data_type = self._determine_data_type_from_source_types(source_types)
            
            with self._get_connection_context() as conn:
                cursor = conn.cursor()
                chunk_vectors = {}
                chunk_metadata = {}

                # statutes 타입인 경우 statute_embeddings 테이블에서 로드
                # ⚠️ 중요: statute_embeddings는 article_id를 키로 사용 (id가 아님)
                if data_type == 'statutes' or (source_types and any(st in ['statute_article', 'statute_articles'] for st in source_types)):
                    query = """
                        SELECT
                            se.article_id,
                            se.embedding_vector,
                            se.embedding_version,
                            se.metadata
                        FROM statute_embeddings se
                        WHERE se.embedding_vector IS NOT NULL
                    """
                    params = []
                    
                    # 🔥 성능 최적화: chunk_ids가 제공되면 해당 ID만 로드
                    if chunk_ids:
                        # statute_embeddings는 article_id를 사용하므로 chunk_ids를 article_id로 사용
                        placeholders = ','.join(['%s'] * len(chunk_ids))
                        query += f" AND se.article_id IN ({placeholders})"
                        params.extend(chunk_ids)
                        self.logger.debug(f"📋 [OPTIMIZED] Loading only {len(chunk_ids)} specific chunk vectors from statute_embeddings")

                    # 버전 필터링
                    if PGVECTOR_AVAILABLE and self.vector_search_method == 'pgvector':
                        # pgvector 사용 시 버전 필터링 건너뛰고 모든 버전 로드
                        self.logger.debug(
                            f"📋 [PGVECTOR] Loading chunk vectors from statute_embeddings (all versions) "
                            f"(requested version_id={embedding_version_id}, but loading all for matching)"
                        )
                    elif embedding_version_id is not None:
                        query += " AND se.embedding_version = %s"
                        params.append(embedding_version_id)
                    else:
                        # 활성 버전만 조회
                        query += """
                            AND se.embedding_version IN (
                                SELECT version FROM embedding_versions WHERE is_active = TRUE AND data_type = 'statutes'
                            )
                        """

                    if limit:
                        query += " LIMIT %s"
                        params.append(limit)

                    cursor.execute(query, params)
                    rows = cursor.fetchall()

                    for row in rows:
                        if hasattr(row, 'keys'):
                            article_id = row['article_id']  # ⚠️ 키로 사용
                            embedding_vector = row['embedding_vector']
                            embedding_version = row.get('embedding_version')
                            metadata = row.get('metadata')
                        else:
                            article_id = row[0]  # ⚠️ 키로 사용
                            embedding_vector = row[1]
                            embedding_version = row[2] if len(row) > 2 else None
                            metadata = row[3] if len(row) > 3 else None

                        if embedding_vector is None:
                            continue

                        # 벡터 파싱
                        try:
                            if isinstance(embedding_vector, (list, tuple)):
                                vector = np.array(embedding_vector, dtype=np.float32)
                            elif hasattr(embedding_vector, 'tolist'):
                                vector = np.array(embedding_vector.tolist(), dtype=np.float32)
                            elif isinstance(embedding_vector, str):
                                if embedding_vector.startswith('[') and embedding_vector.endswith(']'):
                                    import json
                                    try:
                                        vector_list = json.loads(embedding_vector)
                                        vector = np.array(vector_list, dtype=np.float32)
                                    except json.JSONDecodeError:
                                        cleaned = embedding_vector.strip('[]')
                                        vector_list = [float(x.strip()) for x in cleaned.split(',') if x.strip()]
                                        vector = np.array(vector_list, dtype=np.float32)
                                else:
                                    vector_list = [float(x.strip()) for x in embedding_vector.split(',') if x.strip()]
                                    vector = np.array(vector_list, dtype=np.float32)
                            else:
                                vector = np.array(embedding_vector, dtype=np.float32)
                            
                            # 차원 검증
                            expected_dim = 768
                            if len(vector) != expected_dim:
                                self.logger.warning(f"Dimension mismatch for article_id {article_id}: expected {expected_dim}, got {len(vector)}")
                                continue
                        except (ValueError, TypeError) as parse_error:
                            self.logger.warning(
                                f"Failed to parse embedding_vector for article_id {article_id}: {parse_error}. "
                                f"Type: {type(embedding_vector).__name__}"
                            )
                            continue

                        # ⚠️ 중요: article_id를 키로 사용 (pgvector 검색 결과와 일치)
                        chunk_vectors[article_id] = vector
                        chunk_metadata[article_id] = {
                            'source_type': 'statute_article',
                            'source_id': article_id,
                            'metadata': metadata,
                            'embedding_version_id': embedding_version
                        }

                    self.logger.info(f"Loaded {len([k for k in chunk_vectors.keys() if chunk_metadata.get(k, {}).get('source_type') == 'statute_article'])} chunk vectors from statute_embeddings")

                # precedents 타입이거나 data_type이 None인 경우 precedent_chunks 테이블에서 로드
                if data_type != 'statutes':
                    # 🔥 성능 최적화: 벡터만 필요한 경우 chunk_content 제외 (큰 텍스트 필드)
                    # chunk_content는 필요시 별도 쿼리로 로드
                    query = """
                        SELECT
                            pc.id,
                            pc.embedding_vector,
                            pc.precedent_content_id,
                            pc.chunk_index,
                            pc.metadata,
                            pc.embedding_version
                        FROM precedent_chunks pc
                        WHERE pc.embedding_vector IS NOT NULL
                    """
                    params = []
                    
                    # 🔥 성능 최적화: chunk_ids가 제공되면 해당 ID만 로드
                    if chunk_ids:
                        placeholders = ','.join(['%s'] * len(chunk_ids))
                        query += f" AND pc.id IN ({placeholders})"
                        params.extend(chunk_ids)
                        self.logger.debug(f"📋 [OPTIMIZED] Loading only {len(chunk_ids)} specific chunk vectors from precedent_chunks")

                    # 버전 필터링
                    if PGVECTOR_AVAILABLE and self.vector_search_method == 'pgvector':
                        # 🔥 개선: precedent_chunks는 embedding_versions.version 값을 사용
                        # embedding_version_id가 지정된 경우, 해당 버전의 version 번호를 조회
                        if embedding_version_id is not None:
                            # embedding_version_id로 version 번호 조회
                            cursor.execute("""
                                SELECT version FROM embedding_versions WHERE id = %s
                            """, (embedding_version_id,))
                            version_row = cursor.fetchone()
                            if version_row:
                                version_num = version_row[0] if isinstance(version_row, (tuple, list)) else version_row.get('version', embedding_version_id)
                                query += " AND pc.embedding_version = %s"
                                params.append(version_num)
                                self.logger.debug(
                                    f"📋 [PGVECTOR] Loading chunk vectors from precedent_chunks "
                                    f"with version={version_num} (from version_id={embedding_version_id})"
                                )
                            else:
                                # version_id를 찾을 수 없으면 그대로 사용 (하위 호환성)
                                query += " AND pc.embedding_version = %s"
                                params.append(embedding_version_id)
                                self.logger.warning(
                                    f"⚠️ [PGVECTOR] version_id={embedding_version_id} not found, "
                                    f"using as-is for precedent_chunks"
                                )
                        else:
                            # 활성 버전 조회 후 version 번호 사용
                            active_version_id = self._get_active_embedding_version_id(data_type='precedents')
                            if active_version_id:
                                # 활성 버전의 version 번호 조회
                                cursor.execute("""
                                    SELECT version FROM embedding_versions WHERE id = %s
                                """, (active_version_id,))
                                version_row = cursor.fetchone()
                                if version_row:
                                    version_num = version_row[0] if isinstance(version_row, (tuple, list)) else version_row.get('version', 1)
                                    query += " AND pc.embedding_version = %s"
                                    params.append(version_num)
                                    self.logger.debug(
                                        f"📋 [PGVECTOR] Loading chunk vectors from precedent_chunks "
                                        f"with active version={version_num} (from version_id={active_version_id})"
                                    )
                                else:
                                    # 활성 버전이 없으면 버전 1 사용 (하위 호환성)
                                    query += " AND pc.embedding_version = 1"
                                    self.logger.warning(
                                        f"⚠️ [PGVECTOR] Active version_id={active_version_id} not found, "
                                        f"falling back to version=1 for precedent_chunks"
                                    )
                            else:
                                # 활성 버전이 없으면 버전 1 사용 (하위 호환성)
                                query += " AND pc.embedding_version = 1"
                                self.logger.warning(
                                    f"⚠️ [PGVECTOR] No active version found, "
                                    f"falling back to version=1 for precedent_chunks"
                                )
                    elif embedding_version_id is not None:
                        # 🔥 개선: embedding_version_id를 version 번호로 변환
                        cursor.execute("""
                            SELECT version FROM embedding_versions WHERE id = %s
                        """, (embedding_version_id,))
                        version_row = cursor.fetchone()
                        if version_row:
                            version_num = version_row[0] if isinstance(version_row, (tuple, list)) else version_row.get('version', embedding_version_id)
                            query += " AND pc.embedding_version = %s"
                            params.append(version_num)
                        else:
                            query += " AND pc.embedding_version = %s"
                            params.append(embedding_version_id)
                    else:
                        query += """
                            AND pc.embedding_version IN (
                                SELECT version FROM embedding_versions WHERE is_active = TRUE AND data_type = 'precedents'
                            )
                        """

                    if limit:
                        query += " LIMIT %s"
                        params.append(limit)

                    cursor.execute(query, params)
                    rows = cursor.fetchall()

                    for row in rows:
                        if hasattr(row, 'keys'):
                            chunk_id = row['id']
                            embedding_vector = row['embedding_vector']
                            row_dict = dict(row)
                        else:
                            chunk_id = row[0]
                            embedding_vector = row[1]
                            # 🔥 최적화: chunk_content 제거로 인덱스 조정
                            row_dict = {
                                'precedent_content_id': row[2] if len(row) > 2 else None,
                                'chunk_index': row[3] if len(row) > 3 else None,
                                'metadata': row[4] if len(row) > 4 else None,
                                'embedding_version': row[5] if len(row) > 5 else None
                            }

                        if embedding_vector is None:
                            continue

                        # 벡터 파싱 (강화된 파싱 로직)
                        try:
                            if isinstance(embedding_vector, (list, tuple)):
                                vector = np.array(embedding_vector, dtype=np.float32)
                            elif hasattr(embedding_vector, 'tolist'):
                                vector = np.array(embedding_vector.tolist(), dtype=np.float32)
                            elif isinstance(embedding_vector, str):
                                # 문자열 벡터 파싱 강화
                                embedding_str = embedding_vector.strip()
                                
                                # JSON 형식 시도
                                if embedding_str.startswith('[') and embedding_str.endswith(']'):
                                    import json
                                    try:
                                        vector_list = json.loads(embedding_str)
                                        vector = np.array(vector_list, dtype=np.float32)
                                    except json.JSONDecodeError:
                                        # JSON 파싱 실패 시 쉼표 분리 시도
                                        cleaned = embedding_str.strip('[]')
                                        # 빈 문자열 및 None 값 필터링
                                        parts = [x.strip() for x in cleaned.split(',') if x.strip() and x.strip().lower() != 'none']
                                        vector_list = []
                                        for part in parts:
                                            try:
                                                val = float(part)
                                                if not (np.isnan(val) or np.isinf(val)):
                                                    vector_list.append(val)
                                            except (ValueError, TypeError):
                                                continue
                                        if len(vector_list) > 0:
                                            vector = np.array(vector_list, dtype=np.float32)
                                        else:
                                            raise ValueError("No valid float values found in vector string")
                                else:
                                    # 쉼표로 분리된 문자열
                                    parts = [x.strip() for x in embedding_str.split(',') if x.strip() and x.strip().lower() != 'none']
                                    vector_list = []
                                    for part in parts:
                                        try:
                                            val = float(part)
                                            if not (np.isnan(val) or np.isinf(val)):
                                                vector_list.append(val)
                                        except (ValueError, TypeError):
                                            continue
                                    if len(vector_list) > 0:
                                        vector = np.array(vector_list, dtype=np.float32)
                                    else:
                                        raise ValueError("No valid float values found in vector string")
                            else:
                                # 기타 타입 시도
                                try:
                                    vector = np.array(embedding_vector, dtype=np.float32)
                                except (ValueError, TypeError):
                                    raise ValueError(f"Cannot convert {type(embedding_vector).__name__} to numpy array")
                            
                            # 차원 검증
                            expected_dim = 768
                            if len(vector) != expected_dim:
                                self.logger.warning(f"Dimension mismatch for chunk {chunk_id}: expected {expected_dim}, got {len(vector)}")
                                continue
                        except (ValueError, TypeError) as parse_error:
                            self.logger.warning(
                                f"Failed to parse embedding_vector for chunk {chunk_id}: {parse_error}. "
                                f"Type: {type(embedding_vector).__name__}, Value preview: {str(embedding_vector)[:100] if isinstance(embedding_vector, str) else 'N/A'}"
                            )
                            continue

                        chunk_vectors[chunk_id] = vector
                        chunk_metadata[chunk_id] = {
                            'source_type': 'precedent_content',
                            # 🔥 최적화: chunk_content는 큰 텍스트 필드이므로 벡터 로드 시 제외
                            # 필요시 별도 쿼리로 로드 (성능 향상)
                            'text': None,  # chunk_content는 필요시 별도 로드
                            'source_id': row_dict.get('precedent_content_id'),
                            'chunk_index': row_dict.get('chunk_index'),
                            'metadata': row_dict.get('metadata'),
                            'embedding_version_id': row_dict.get('embedding_version')
                        }

                    precedent_count = sum(1 for meta in chunk_metadata.values() if meta.get('source_type') == 'precedent_content')
                    if precedent_count > 0:
                        self.logger.info(f"Loaded {precedent_count} chunk vectors from precedent_chunks")

                # 메타데이터를 인스턴스 변수로 저장
                self._chunk_metadata = chunk_metadata

                return chunk_vectors

        except Exception as e:
            error_msg = str(e).lower()
            if "no such table" in error_msg:
                self.logger.error(
                    f"❌ Table not found in database: {e}. "
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
    
    def _adjust_threshold_dynamically(
        self,
        query: str,
        source_types: Optional[List[str]] = None,
        base_threshold: float = 0.4
    ) -> float:
        """
        쿼리 특성에 따라 동적으로 임계값 조정
        
        Args:
            query: 검색 쿼리
            source_types: 소스 타입 목록
            base_threshold: 기본 임계값
        
        Returns:
            조정된 임계값
        """
        try:
            adjusted_threshold = base_threshold
            
            # 1. 쿼리 길이 기반 조정
            query_length = len(query)
            if query_length < 10:
                # 짧은 쿼리는 낮은 임계값 (다양한 결과 필요)
                adjusted_threshold = max(0.2, adjusted_threshold - 0.05)
                self.logger.debug(f"📊 [THRESHOLD] Short query ({query_length} chars), lowering threshold")
            elif query_length > 100:
                # 긴 쿼리는 높은 임계값 (정확한 결과 필요)
                adjusted_threshold = min(0.7, adjusted_threshold + 0.05)
                self.logger.debug(f"📊 [THRESHOLD] Long query ({query_length} chars), raising threshold")
            
            # 2. 소스 타입 기반 조정
            if source_types:
                # 법령 조문은 조금 높은 임계값
                if 'statute_article' in source_types and len(source_types) == 1:
                    adjusted_threshold = max(0.35, min(0.6, adjusted_threshold + 0.05))
                    self.logger.debug(f"📊 [THRESHOLD] Statute article search, raising threshold")
                # 판례는 기본 임계값 유지
                elif 'precedent_content' in source_types:
                    # 판례는 기본값 유지하거나 약간 낮춤
                    adjusted_threshold = max(0.3, adjusted_threshold - 0.02)
                    self.logger.debug(f"📊 [THRESHOLD] Precedent search, slightly lowering threshold")
            
            # 3. 임계값 범위 제한
            adjusted_threshold = max(0.2, min(0.8, adjusted_threshold))
            
            return adjusted_threshold
            
        except Exception as e:
            self.logger.warning(f"⚠️ [THRESHOLD] Error adjusting threshold dynamically: {e}, using base threshold")
            return base_threshold
    
    def _encode_query(self, query: str, use_cache: bool = True, model_name: Optional[str] = None, version_id: Optional[int] = None) -> Optional[np.ndarray]:
        """쿼리 인코딩 (캐시 사용, 재정규화 포함)
        
        Args:
            query: 인코딩할 쿼리 텍스트
            use_cache: 캐시 사용 여부 (기본값: True)
            model_name: 모델명 (None이면 현재 모델 사용, 캐시 키에 포함)
            version_id: 임베딩 버전 ID (None이면 현재 활성 버전 사용, 캐시 키에 포함)
        
        Note:
            캐시 키에 모델명과 버전이 포함되어 있으므로, 모델이 변경되면 자동으로 다른 캐시를 사용합니다.
            use_cache=False는 더 이상 필요하지 않지만 하위 호환성을 위해 유지합니다.
        """
        normalized_query = self._normalize_query(query)
        
        # 🔥 개선: 캐시 키에 모델명과 버전이 포함되어 있으므로 모델 변경 시 자동으로 다른 캐시 사용
        if use_cache:
            query_vec = self._get_cached_query_vector(normalized_query, model_name=model_name, version_id=version_id)
            if query_vec is not None:
                self.logger.debug(f"🔍 [QUERY ENCODING] Cache hit for query: '{normalized_query[:80]}...' (model={model_name or self.model_name})")
                return query_vec
        
        if not self._ensure_embedder_initialized():
            self.logger.error("Cannot generate query embedding: embedder not initialized")
            return None
        
        try:
            self.logger.info(
                f"🔍 [QUERY ENCODING] Encoding new query: '{normalized_query[:100]}...' "
                f"(length={len(normalized_query)}, cache_miss=True)"
            )
            query_vec = self.embedder.encode([normalized_query], batch_size=1, normalize=True)[0]
            
            query_norm = np.linalg.norm(query_vec)
            if abs(query_norm - 1.0) > 0.01:
                self.logger.debug(f"Query vector not normalized (norm={query_norm:.4f}), re-normalizing")
                query_vec = query_vec / (query_norm + 1e-9)
            
            # 🔥 개선: 캐시 저장 시 모델명과 버전 포함
            self._cache_query_vector(query, query_vec, model_name=model_name, version_id=version_id)
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
                    # 🔥 개선: 캐시 저장 시 모델명과 버전 포함
                    self._cache_query_vector(query, query_vec, model_name=model_name, version_id=version_id)
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

    def _get_cached_query_vector(self, query: str, model_name: Optional[str] = None, version_id: Optional[int] = None) -> Optional[np.ndarray]:
        """캐시에서 쿼리 벡터 가져오기 (모델명 및 버전 포함)
        
        Args:
            query: 쿼리 텍스트
            model_name: 모델명 (None이면 현재 모델 사용)
            version_id: 임베딩 버전 ID (None이면 현재 활성 버전 사용)
        """
        normalized_query = self._normalize_query(query)
        
        # 🔥 개선: 캐시 키에 모델명과 버전 포함
        model = model_name or self.model_name or 'unknown'
        
        # 버전 정보 가져오기 (선택적)
        if version_id is None:
            try:
                # 현재 모델에 해당하는 활성 버전 조회 시도
                if model:
                    # 모델명으로 data_type 추정 (간단한 매핑)
                    if 'precedents' in model.lower() or 'ko-legal-sbert' in model.lower():
                        version_id = self._get_active_embedding_version_id(data_type='precedents')
                    elif 'statutes' in model.lower() or 'ko-sroberta' in model.lower():
                        version_id = self._get_active_embedding_version_id(data_type='statutes')
                    else:
                        version_id = self._get_active_embedding_version_id()
                else:
                    version_id = self._get_active_embedding_version_id()
            except Exception:
                version_id = None
        
        version_str = str(version_id) if version_id is not None else 'unknown'
        
        if self._use_hash_cache:
            import hashlib
            # 🔥 모델명과 버전을 캐시 키에 포함
            cache_key = f"{model}:v{version_str}:{normalized_query}"
            query_hash = hashlib.md5(cache_key.encode()).hexdigest()
            return self._query_vector_cache.get(query_hash)
        else:
            # 🔥 모델명과 버전을 캐시 키에 포함
            cache_key = f"{model}:v{version_str}:{normalized_query}"
            return self._query_vector_cache.get(cache_key)

    def _cache_query_vector(self, query: str, vector: np.ndarray, model_name: Optional[str] = None, version_id: Optional[int] = None):
        """쿼리 벡터를 캐시에 저장 (모델명 및 버전 포함)
        
        Args:
            query: 쿼리 텍스트
            vector: 인코딩된 벡터
            model_name: 모델명 (None이면 현재 모델 사용)
            version_id: 임베딩 버전 ID (None이면 현재 활성 버전 사용)
        """
        normalized_query = self._normalize_query(query)
        
        # 🔥 개선: 캐시 키에 모델명과 버전 포함
        model = model_name or self.model_name or 'unknown'
        
        # 버전 정보 가져오기 (선택적)
        if version_id is None:
            try:
                # 현재 모델에 해당하는 활성 버전 조회 시도
                if model:
                    # 모델명으로 data_type 추정 (간단한 매핑)
                    if 'precedents' in model.lower() or 'ko-legal-sbert' in model.lower():
                        version_id = self._get_active_embedding_version_id(data_type='precedents')
                    elif 'statutes' in model.lower() or 'ko-sroberta' in model.lower():
                        version_id = self._get_active_embedding_version_id(data_type='statutes')
                    else:
                        version_id = self._get_active_embedding_version_id()
                else:
                    version_id = self._get_active_embedding_version_id()
            except Exception:
                version_id = None
        
        version_str = str(version_id) if version_id is not None else 'unknown'
        
        if self._use_hash_cache:
            import hashlib
            # 🔥 모델명과 버전을 캐시 키에 포함
            cache_key = f"{model}:v{version_str}:{normalized_query}"
            query_hash = hashlib.md5(cache_key.encode()).hexdigest()
            # 캐시 크기 제한 (LRU: 오래된 항목 제거)
            if len(self._query_vector_cache) >= self._cache_max_size:
                # 가장 오래된 항목 제거 (단순 구현: 첫 번째 항목)
                oldest_key = next(iter(self._query_vector_cache))
                del self._query_vector_cache[oldest_key]
            self._query_vector_cache[query_hash] = vector
        else:
            # 🔥 모델명과 버전을 캐시 키에 포함
            cache_key = f"{model}:v{version_str}:{normalized_query}"
            # 캐시 크기 제한 (LRU: 오래된 항목 제거)
            if len(self._query_vector_cache) >= self._cache_max_size:
                # 가장 오래된 항목 제거 (단순 구현: 첫 번째 항목)
                oldest_key = next(iter(self._query_vector_cache))
                del self._query_vector_cache[oldest_key]
            self._query_vector_cache[cache_key] = vector

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
            with self._get_connection_context() as conn:
                cursor = conn.cursor()

                chunk_vectors = {}

                # 배치 단위로 처리
                for i in range(0, len(chunk_ids), batch_size):
                    batch = chunk_ids[i:i + batch_size]
                    placeholders = ','.join(['%s'] * len(batch))  # PostgreSQL은 %s 사용

                    query = f"""
                        SELECT
                            e.chunk_id,
                            e.vector,
                            e.dim
                        FROM embeddings e
                        WHERE e.model = %s AND e.chunk_id IN ({placeholders})
                    """
                    params = [self.model_name] + batch

                    cursor.execute(query, params)
                    rows = cursor.fetchall()

                    for row in rows:
                        # PostgreSQL의 경우 dict-like row 또는 tuple 반환
                        if hasattr(row, 'keys'):  # dict-like (RealDictRow)
                            chunk_id = row['chunk_id']
                            vector_blob = row['vector']
                            dim = row['dim']
                        else:  # tuple
                            chunk_id = row[0]
                            vector_blob = row[1]
                            dim = row[2]

                        # BLOB을 numpy 배열로 변환
                        vector = np.frombuffer(vector_blob, dtype=np.float32).reshape(-1)

                        if len(vector) == dim:
                            chunk_vectors[chunk_id] = vector

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
            import math
            
            if hasattr(self.index, 'metric_type'):
                if self.index.metric_type == faiss.METRIC_INNER_PRODUCT:
                    # Inner Product: 값이 클수록 유사도가 높음
                    # 정규화 개선: sigmoid 함수 사용으로 더 부드러운 변환
                    similarity = 1.0 / (1.0 + np.exp(-float(distance)))
                elif self.index.metric_type == faiss.METRIC_L2:
                    # L2 거리: 지수 감쇠 함수 사용 (개선된 정규화)
                    # IndexIVFPQ는 압축 인덱스이므로 더 큰 스케일 사용
                    index_type = type(self.index).__name__ if self.index else ''
                    if 'IndexIVFPQ' in index_type:
                        # IndexIVFPQ: 개선된 스케일링 (거리 값이 클 수 있음)
                        # 거리 분포를 고려한 동적 스케일링
                        scale = 12.0  # 스케일 증가로 유사도 점수 향상 (10.0 → 12.0)
                        # 지수 감쇠 + 최소값 보장
                        similarity = np.exp(-float(distance) / scale)
                        # 거리가 작을 때 더 높은 점수 부여 (보너스)
                        if distance < 1.0:
                            similarity = similarity * 1.1
                            # 강제 정규화: 반드시 0.0~1.0 범위로 제한
                            similarity = max(0.0, min(1.0, similarity))
                    else:
                        # 일반 L2: 개선된 스케일링
                        scale = 4.0  # 스케일 증가로 유사도 점수 향상 (3.0 → 4.0)
                        similarity = np.exp(-float(distance) / scale)
                        # 거리가 작을 때 더 높은 점수 부여 (보너스)
                        if distance < 0.5:
                            similarity = similarity * 1.15
                            # 강제 정규화: 반드시 0.0~1.0 범위로 제한
                            similarity = max(0.0, min(1.0, similarity))
                else:
                    # 기본: 개선된 역변환 (로그 스케일 적용)
                    similarity = 1.0 / (1.0 + math.log1p(float(distance)))
            else:
                # metric_type이 없는 경우: 개선된 기본 변환
                similarity = 1.0 / (1.0 + math.log1p(float(distance)))
        except Exception as e:
            self.logger.debug(f"Error calculating similarity: {e}, using default conversion")
            similarity = 1.0 / (1.0 + float(distance))
        
        # 정규화: 0-1 범위로 제한하되, 점수 차별화 강화
        similarity = max(0.0, min(1.0, similarity))
        
        # 점수 차별화: 모든 점수가 1.0이 되는 문제 방지
        # 거리가 매우 작거나 0인 경우에도 점수 차별화 유지
        if distance == 0.0:
            # 거리가 0이면 최대 점수이지만, 다른 결과와 차별화를 위해 약간 감소
            similarity = 0.99
        elif similarity >= 0.95:
            # 매우 높은 점수도 약간 차별화 (0.95-0.99 범위)
            similarity = min(0.99, similarity * 0.98)
            # 강제 정규화: 곱셈 후에도 1.0 초과 방지
            similarity = max(0.0, min(1.0, similarity))
        elif similarity >= 0.7:
            # 높은 유사도에 작은 보너스 (최대 0.95로 제한)
            similarity = min(0.95, similarity * 1.03)
            # 강제 정규화: 곱셈 후에도 1.0 초과 방지
            similarity = max(0.0, min(1.0, similarity))
        
        # 최종 강제 정규화 (차별화 로직 후에도)
        return float(max(0.0, min(1.0, similarity)))
    
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
        하이브리드 점수 계산 (유사도 + ML 신뢰도 + 품질 점수) - 개선된 가중치
        
        Args:
            similarity: 유사도 점수 (0-1)
            ml_confidence: ML 신뢰도 점수 (0-1)
            quality_score: 품질 점수 (0-1)
            weights: 가중치 딕셔너리 (None이면 기본값 사용)
        
        Returns:
            하이브리드 점수 (0-1)
        """
        if weights is None:
            # 개선된 가중치: 유사도 90%, ML 신뢰도 5%, 품질 점수 5%
            # similarity 가중치를 0.85 → 0.90으로 증가하여 실제 유사도가 더 반영되도록
            weights = {
                "similarity": 0.90,
                "ml_confidence": 0.05,
                "quality": 0.05
            }
        
        # 가중치 합이 1이 되도록 정규화
        total_weight = sum(weights.values())
        if total_weight > 0:
            weights = {k: v / total_weight for k, v in weights.items()}
        
        # 개선된 하이브리드 점수 계산
        # similarity가 높을 때 더 강한 가중치 적용 (비선형 가중치)
        similarity_weight = weights.get("similarity", 0.9)
        if similarity >= 0.7:
            # 높은 유사도에 추가 보너스 (비선형 가중치)
            similarity_weight = min(0.95, similarity_weight * 1.05)
        
        hybrid_score = (
            similarity_weight * similarity +
            weights.get("ml_confidence", 0.05) * ml_confidence +
            weights.get("quality", 0.05) * quality_score
        )
        
        # 점수 향상: 높은 similarity에 추가 보너스
        if similarity >= 0.75:
            # 보너스 적용 전 정규화 확인
            if hybrid_score < 0.97:  # 0.97 미만일 때만 보너스 적용
                hybrid_score = min(1.0, hybrid_score * 1.03)
            else:
                hybrid_score = 0.99  # 이미 높은 점수는 0.99로 제한
        
        # 강제 정규화: 반드시 0.0~1.0 범위로 제한
        return float(max(0.0, min(1.0, hybrid_score)))

    def search(self,
               query: str,
               k: int = 10,
               source_types: Optional[List[str]] = None,
               similarity_threshold: float = 0.4,
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
        
        # 확장된 쿼리 사용 여부 로깅
        self.logger.info(
            f"🔍 [SEARCH ENGINE] Received query: '{query[:100]}...' "
            f"(length={len(query)}, normalized will be applied)"
        )
        
        # 🔥 개선: source_types에서 data_type 결정하여 적절한 활성 버전 조회
        data_type = self._determine_data_type_from_source_types(source_types)
        
        # 🔥 개선: 저장된 임베딩 모델과 일치하는 모델 사용 보장
        if data_type:
            self._ensure_correct_embedding_model(data_type=data_type)
        else:
            # data_type이 없으면 모든 활성 버전 확인
            for dt in ['precedents', 'statutes']:
                required_model = self._get_model_name_for_data_type(data_type=dt)
                if required_model and self.model_name:
                    current_model = self.model_name.strip().strip('"').strip("'")
                    if current_model != required_model:
                        # 첫 번째로 발견된 모델로 설정 (우선순위: precedents > statutes)
                        self.logger.info(
                            f"🔄 [MODEL] Ensuring correct model for {dt}: {required_model}"
                        )
                        self._ensure_correct_embedding_model(data_type=dt)
                        break
        
        # 🔥 개선: pgvector를 사용하는 경우 버전 체크 및 폴백 로직
        # pgvector는 DB에서 직접 검색하므로 버전 필터링 실패 시 모든 버전 검색
        if PGVECTOR_AVAILABLE and self.vector_search_method == 'pgvector':
            # pgvector를 사용하는 경우 버전 체크는 선택적으로만 수행 (로깅용)
            # 🔥 개선: data_type별 활성 버전 조회
            active_version_id = self._get_active_embedding_version_id(data_type=data_type)
            if active_version_id:
                chunk_count = self._get_version_chunk_count(active_version_id)
                if chunk_count > 0:
                    self.logger.info(
                        f"🔍 Semantic search starting - Active embedding version ID: {active_version_id} "
                        f"({chunk_count} chunks, data_type={data_type or 'mixed'})"
                    )
                    # embedding_version_id가 지정되지 않은 경우 활성 버전 사용 (선택적)
                    if embedding_version_id is None:
                        embedding_version_id = active_version_id
                else:
                    # 🔥 개선: 활성 버전에 벡터가 없으면 모든 버전 검색
                    self.logger.warning(
                        f"⚠️ Active version (ID={active_version_id}, data_type={data_type}) has no chunks, "
                        f"falling back to search all versions"
                    )
                    # pgvector는 버전 필터 없이도 검색 가능하므로 None으로 설정
                    embedding_version_id = None
            else:
                if data_type:
                    self.logger.debug(
                        f"⚠️ No active embedding version found for data_type={data_type}, "
                        f"but pgvector will search all versions"
                    )
                else:
                    self.logger.debug("⚠️ No active embedding version found, but pgvector will search all versions")
                embedding_version_id = None
        else:
            # FAISS를 사용하는 경우 기존 버전 체크 로직 유지
            # 🔥 개선: data_type별 활성 버전 조회
            active_version_id = self._get_active_embedding_version_id(data_type=data_type)
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
                    # 🔥 개선: 지정된 버전에 벡터가 없으면 모든 버전 검색
                    self.logger.warning(
                        f"⚠️ Specified version (ID={embedding_version_id}) has no chunks! "
                        f"Falling back to search all versions."
                    )
                    embedding_version_id = None
        
        # 🔥 개선: pgvector를 사용하는 경우 FAISS 인덱스 로드 건너뛰기
        # pgvector는 DB에서 직접 검색하므로 FAISS 인덱스가 필요 없음
        if PGVECTOR_AVAILABLE and self.vector_search_method == 'pgvector':
            # pgvector를 사용하는 경우 인덱스 로드 건너뛰기
            self.index = None
        # 인덱스가 없으면 자동으로 로드 시도 (초기화 실패 시 재시도)
        elif self.index is None and FAISS_AVAILABLE and self.embedder:
            if self.use_mlflow_index and self.mlflow_manager:
                try:
                    self._load_mlflow_index()
                    self.logger.info("✅ MLflow 인덱스 로드 성공")
                except (RuntimeError, ImportError, Exception) as e:
                    self.logger.warning(f"⚠️ MLflow 인덱스 로드 실패: {e}")
                    self.logger.info("🔄 DB 기반 인덱스로 폴백 시도 중...")
                    # MLflow 인덱스 사용 불가 시 DB 기반 인덱스로 폴백
                    try:
                        self._load_faiss_index()
                        if self.index is not None:
                            self.logger.info("✅ DB 기반 인덱스 로드 성공 (MLflow 폴백)")
                            # MLflow 사용 비활성화하여 다음 검색에서도 DB 기반 인덱스 사용
                            self.use_mlflow_index = False
                        else:
                            raise RuntimeError("DB 기반 인덱스도 로드 실패")
                    except Exception as fallback_error:
                        self.logger.error(f"❌ DB 기반 인덱스 폴백도 실패: {fallback_error}")
                        raise RuntimeError(
                            f"MLflow 인덱스 로드 실패 ({e}) 및 DB 기반 인덱스 폴백 실패 ({fallback_error}). "
                            "인덱스 파일을 확인하거나 인덱스를 재생성하세요."
                        )
            elif self.use_mlflow_index and not self.mlflow_manager:
                # MLflow가 활성화되어 있지만 매니저가 없는 경우 DB 기반 인덱스로 폴백
                self.logger.warning("⚠️ MLflow 인덱스가 활성화되어 있지만 MLflow 매니저가 없습니다. DB 기반 인덱스로 폴백합니다.")
                try:
                    self._load_faiss_index()
                    if self.index is not None:
                        self.logger.info("✅ DB 기반 인덱스 로드 성공 (MLflow 매니저 없음으로 인한 폴백)")
                        self.use_mlflow_index = False
                    else:
                        raise RuntimeError("DB 기반 인덱스 로드 실패")
                except Exception as fallback_error:
                    self.logger.error(f"❌ DB 기반 인덱스 폴백 실패: {fallback_error}")
                    raise RuntimeError(
                        f"MLflow 매니저가 없고 DB 기반 인덱스 로드도 실패했습니다 ({fallback_error}). "
                        "인덱스 파일을 확인하거나 인덱스를 재생성하세요."
                    )
            else:
                # MLflow가 비활성화된 경우 DB 기반 인덱스 사용
                # 🔥 개선: pgvector를 사용하는 경우 FAISS 인덱스 로드 건너뛰기
                if PGVECTOR_AVAILABLE and self.vector_search_method == 'pgvector':
                    self.index = None
                else:
                    self._load_faiss_index()
                    if self.index is None:
                        # 🔥 개선: pgvector를 사용할 수 있으면 에러 대신 경고만 출력
                        # 단, pgvector 모드가 아닐 때만 경고 출력
                        if PGVECTOR_AVAILABLE and self.vector_search_method != 'pgvector':
                            self.logger.warning(
                                "⚠️ FAISS 인덱스 로드 실패. pgvector를 사용하여 검색을 계속합니다."
                            )
                            self.index = None
                        else:
                            raise RuntimeError(
                                "DB 기반 인덱스 로드 실패. 인덱스 파일을 확인하거나 인덱스를 재생성하세요."
                            )
        
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

        # min_results 기본값 검증
        if min_results <= 0:
            min_results = max(1, k // 2)  # 기본값: k의 절반
            self.logger.debug(f"min_results adjusted to {min_results} (was <= 0)")
        
        # 검색 파라미터 동적 조정 (쿼리 복잡도 기반)
        adjusted_threshold = similarity_threshold
        if not disable_retry:
            # 쿼리 복잡도 추정
            query_length = len(query)
            word_count = len(query.split())
            legal_terms = ["법", "조문", "판례", "계약", "손해배상", "불법행위", "해지", "해제", "시효"]
            legal_term_count = sum(1 for term in legal_terms if term in query)
            
            # 복잡도에 따른 threshold 조정
            if query_length < 15 and word_count < 4 and legal_term_count < 2:
                # 간단한 쿼리: 낮은 threshold (다양한 결과)
                adjusted_threshold = max(0.25, similarity_threshold - 0.1)
                self.logger.debug(f"Simple query detected, lowering threshold: {adjusted_threshold:.3f}")
            elif query_length > 80 or word_count > 12 or legal_term_count > 4:
                # 복잡한 쿼리: 높은 threshold (정확한 결과)
                adjusted_threshold = min(0.85, similarity_threshold + 0.1)
                self.logger.debug(f"Complex query detected, raising threshold: {adjusted_threshold:.3f}")
        
        # 재시도 로직 비활성화 옵션
        if disable_retry:
            # 높은 신뢰도만 원할 때는 첫 번째 임계값만 사용
            results = self._search_with_threshold(
                query, k, source_types, adjusted_threshold,
                min_ml_confidence=min_ml_confidence,
                min_quality_score=min_quality_score,
                filter_by_confidence=filter_by_confidence,
                chunk_size_category=chunk_size_category,
                deduplicate_by_group=deduplicate_by_group,
                embedding_version_id=embedding_version_id,
                search_k_multiplier=1.0
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

        # 재시도 로직 개선: 결과가 부족하면 search_k를 증가시켜 재시도
        # 재시도 횟수: 6회 → 3회로 감소
        # 전략 변경: 임계값 감소 → search_k 증가 (pgvector 최적화)
        max_retries = 3
        # search_k_multipliers를 환경 변수로 설정 가능하게
        search_k_multipliers_str = os.getenv("SEARCH_K_MULTIPLIERS", "1.0,2.0,4.0")
        search_k_multipliers = [float(x.strip()) for x in search_k_multipliers_str.split(",")]
        if len(search_k_multipliers) != max_retries:
            self.logger.warning(
                f"⚠️ SEARCH_K_MULTIPLIERS length ({len(search_k_multipliers)}) "
                f"does not match max_retries ({max_retries}), using default"
            )
            search_k_multipliers = [1.0, 2.0, 4.0]
        
        for attempt in range(max_retries):
            try:
                # search_k 배수 계산
                search_k_multiplier = search_k_multipliers[attempt]
                # k는 그대로 전달하고, search_k_multiplier만 pgvector 검색에 적용
                # 실제 search_k는 _search_with_pgvector_weighted에서 계산됨:
                # base_search_k = (k * 2) if max_results is None else max_results
                # search_k = int(base_search_k * search_k_multiplier)
                # 로깅용 effective_search_k 계산 (대략적인 값, 실제는 테이블별 설정에 따라 다를 수 있음)
                effective_search_k = int((k * 2) * search_k_multiplier)
                
                self.logger.debug(
                    f"🔍 [RETRY] Attempt {attempt + 1}/{max_retries}: "
                    f"k={k}, search_k_multiplier={search_k_multiplier:.1f}, "
                    f"effective_search_k≈{effective_search_k} (approximate, actual depends on table config), "
                    f"threshold={similarity_threshold:.3f}, min_results={min_results}"
                )
                
                results = self._search_with_threshold(
                    query, k, source_types, similarity_threshold,  # k를 그대로 전달
                    min_ml_confidence=min_ml_confidence,
                    min_quality_score=min_quality_score,
                    filter_by_confidence=filter_by_confidence,
                    chunk_size_category=chunk_size_category,
                    deduplicate_by_group=deduplicate_by_group,
                    embedding_version_id=embedding_version_id,
                    search_k_multiplier=search_k_multiplier
                )
                
                # 🔥 개선: 결과가 0개인 경우 재시도 로직
                if len(results) == 0:
                    if attempt == 0:
                        # 첫 번째 시도에서 0개 결과면 버전 필터링 실패 가능성 높음
                        # 버전 필터링 실패로 인한 0개 결과는 재시도해도 의미 없으므로 건너뛰기
                        self.logger.warning(
                            f"⚠️ [RETRY] First attempt returned 0 results. "
                            f"This may indicate version filtering failure or no matching data. "
                            f"Skipping retries to save time."
                        )
                        break  # 재시도 건너뛰기
                    elif attempt < max_retries - 1:
                        # 두 번째 이후 시도에서 0개 결과면 재시도
                        self.logger.debug(
                            f"⚠️ [RETRY] No results found (attempt {attempt + 1}/{max_retries}), "
                            f"retrying with increased search_k (multiplier={search_k_multipliers[attempt + 1]:.1f})..."
                        )
                        continue
                
                # 최소 결과 수를 만족하면 바로 반환 (조기 종료)
                if len(results) >= min_results:
                    if attempt > 0:
                        self.logger.info(
                            f"✅ [RETRY] Semantic search: Found {len(results)} results "
                            f"(attempt {attempt + 1}/{max_retries}, search_k_multiplier={search_k_multiplier:.1f})"
                        )
                    else:
                        self.logger.debug(
                            f"✅ [SEARCH] Found {len(results)} results on first attempt "
                            f"(no retry needed, search_k_multiplier={search_k_multiplier:.1f})"
                        )
                    # 조기 종료: 결과가 충분하면 재시도 중단
                    elapsed_time = time.time() - start_time
                    latency_ms = elapsed_time * 1000
                    avg_relevance = sum(r.get('score', 0.0) for r in results) / len(results) if results else 0.0
                    
                    self.logger.info(
                        f"⏱️  Search performance: {elapsed_time:.3f}s ({latency_ms:.1f}ms), "
                        f"results: {len(results)}/{k} requested, avg_score: {avg_relevance:.3f}, "
                        f"retries: {attempt}/{max_retries}, search_k_multiplier: {search_k_multiplier:.1f}"
                    )
                    
                    if self.enable_performance_monitoring and self.performance_monitor:
                        self.performance_monitor.log_search(
                            version=used_version,
                            query_id=query_id,
                            latency_ms=latency_ms,
                            relevance_score=avg_relevance
                        )
                    
                    return results
                
                # 마지막 시도인 경우 무조건 반환
                if attempt == max_retries - 1:
                    if len(results) < min_results:
                        self.logger.warning(
                            f"⚠️ [RETRY] Only {len(results)} results found after {max_retries} attempts "
                            f"(min_results={min_results}, search_k_multiplier={search_k_multiplier:.1f})"
                        )
                        # 🔥 개선: 결과가 부족하면 임계값을 더 낮춰서 재검색 시도
                        if len(results) == 0:
                            self.logger.warning(
                                f"⚠️ [FALLBACK] No results found, trying with lower threshold "
                                f"(current: {similarity_threshold:.3f} → 0.1)"
                            )
                            # 임계값을 0.1로 낮춰서 최소한의 결과라도 보장
                            fallback_results = self._search_with_threshold(
                                query, k, source_types, 0.1,  # 임계값을 0.1로 낮춤
                                min_ml_confidence=min_ml_confidence,
                                min_quality_score=min_quality_score,
                                filter_by_confidence=False,  # 신뢰도 필터도 비활성화
                                chunk_size_category=chunk_size_category,
                                deduplicate_by_group=deduplicate_by_group,
                                embedding_version_id=embedding_version_id,
                                search_k_multiplier=search_k_multiplier
                            )
                            if len(fallback_results) > 0:
                                self.logger.info(
                                    f"✅ [FALLBACK] Found {len(fallback_results)} results with lowered threshold"
                                )
                                results = fallback_results[:min_results]  # 최소 결과 수만큼만 반환
                    
                    # 마지막 시도 성능 모니터링 로깅
                    elapsed_time = time.time() - start_time
                    latency_ms = elapsed_time * 1000
                    avg_relevance = sum(r.get('score', 0.0) for r in results) / len(results) if results else 0.0
                    
                    self.logger.info(
                        f"⏱️  Search performance: {elapsed_time:.3f}s ({latency_ms:.1f}ms), "
                        f"results: {len(results)}/{k} requested, avg_score: {avg_relevance:.3f}, "
                        f"retries: {attempt}/{max_retries}, search_k_multiplier: {search_k_multiplier:.1f}"
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
                # 에러 타입에 따른 적절한 처리
                is_transient_error = isinstance(e, (ConnectionError, TimeoutError))
                is_value_error = isinstance(e, ValueError)
                
                if is_value_error:
                    # 잘못된 파라미터는 재시도하지 않음
                    self.logger.error(
                        f"❌ [RETRY] Invalid parameters, stopping retries: {e}",
                        exc_info=self.logger.isEnabledFor(logging.DEBUG)
                    )
                    return []
                
                self.logger.warning(
                    f"⚠️ [RETRY] Semantic search attempt {attempt + 1}/{max_retries} failed: {e}",
                    exc_info=self.logger.isEnabledFor(logging.DEBUG)
                )
                
                if is_transient_error and attempt < max_retries - 1:
                    # 일시적 오류는 재시도
                    self.logger.info(f"🔄 Retrying due to transient error: {type(e).__name__}")
                    continue
                
                if attempt == max_retries - 1:
                    # 마지막 시도 실패 시에도 빈 결과 반환
                    self.logger.error("❌ All semantic search attempts failed")
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
    
    def _search_batch_with_threshold(self,
                                     queries: List[str],
                                     k: int,
                                     source_types: Optional[List[str]],
                                     similarity_threshold: float,
                                     min_ml_confidence: Optional[float] = None,
                                     min_quality_score: Optional[float] = None,
                                     filter_by_confidence: bool = False,
                                     chunk_size_category: Optional[str] = None,
                                     deduplicate_by_group: bool = True,
                                     embedding_version_id: Optional[int] = None) -> List[List[Dict[str, Any]]]:
        """
        여러 쿼리를 배치로 검색 (배치 검색 최적화)
        
        Args:
            queries: 검색 쿼리 목록
            k: 각 쿼리당 반환할 최대 결과 수
            source_types: 필터링할 소스 타입 목록
            similarity_threshold: 최소 유사도 임계값
            min_ml_confidence: 최소 ML 신뢰도 점수
            min_quality_score: 최소 품질 점수
            filter_by_confidence: 신뢰도 기반 필터링 활성화
            chunk_size_category: 청크 크기 카테고리 필터
            deduplicate_by_group: 그룹별 중복 제거 활성화
            embedding_version_id: 임베딩 버전 ID 필터
        
        Returns:
            각 쿼리별 검색 결과 리스트의 리스트
        """
        if not queries:
            return []
        
        # 빈 쿼리 필터링
        valid_queries = [q for q in queries if q and q.strip()]
        if not valid_queries:
            return [[] for _ in queries]
        
        try:
            import time
            batch_start = time.time()
            
            # 1. 모든 쿼리 임베딩 생성 (배치)
            query_vectors = []
            query_to_idx = {}
            for i, query in enumerate(queries):
                if query and query.strip():
                    normalized_query = self._normalize_query(query)
                    query_vec = self._encode_query(query)
                    if query_vec is not None:
                        query_vectors.append(query_vec)
                        query_to_idx[len(query_vectors) - 1] = i
            
            if not query_vectors:
                return [[] for _ in queries]
            
            # 2. FAISS 배치 검색
            if FAISS_AVAILABLE and self.index is not None:
                query_vec_batch = np.array(query_vectors).astype('float32')
                
                # 차원 검증
                query_dim = query_vec_batch.shape[1]
                index_dim = self.index.d
                if query_dim != index_dim:
                    self.logger.error(
                        f"FAISS index dimension mismatch: query vector dimension ({query_dim}) "
                        f"does not match index dimension ({index_dim})"
                    )
                    return [[] for _ in queries]
                
                search_k = k * 3
                
                # nprobe 설정
                if hasattr(self.index, 'nprobe'):
                    optimal_nprobe = self._calculate_optimal_nprobe(k, self.index.ntotal)
                    if self.index.nprobe != optimal_nprobe:
                        self.index.nprobe = optimal_nprobe
                
                # 배치 검색 수행
                distances_batch, indices_batch = self.index.search(query_vec_batch, search_k)
                
                self.logger.info(
                    f"✅ [BATCH SEARCH] Processed {len(query_vectors)} queries in batch, "
                    f"returned {len(indices_batch[0])} results per query"
                )
                
                # 3. 각 쿼리별 결과 처리
                all_results = [[] for _ in queries]
                
                for batch_idx, query_idx in query_to_idx.items():
                    distances = distances_batch[batch_idx]
                    indices = indices_batch[batch_idx]
                    query = queries[query_idx]
                    
                    # 단일 쿼리 결과 처리 (기존 로직 재사용)
                    results = []
                    for i, (distance, idx) in enumerate(zip(distances, indices)):
                        if idx < 0 or idx >= len(self._chunk_ids):
                            continue
                        
                        chunk_id = self._chunk_ids[idx] if hasattr(self, '_chunk_ids') and self._chunk_ids else idx
                        
                        # 메타데이터 확인
                        chunk_meta = self._chunk_metadata.get(chunk_id, {})
                        
                        # source_types 필터링
                        if source_types:
                            chunk_source_type = chunk_meta.get('source_type')
                            if chunk_source_type and chunk_source_type not in source_types:
                                continue
                        
                        # similarity_threshold 필터링
                        similarity = 1.0 - distance  # FAISS distance를 similarity로 변환
                        if similarity < similarity_threshold:
                            continue
                        
                        # 결과 생성
                        result = {
                            'id': chunk_id,
                            'relevance_score': float(similarity),
                            'content': chunk_meta.get('text', ''),
                            'source': chunk_meta.get('source', ''),
                            'source_type': chunk_meta.get('source_type', ''),
                            'type': chunk_meta.get('type', ''),
                        }
                        results.append(result)
                        
                        if len(results) >= k:
                            break
                    
                    all_results[query_idx] = results
                
                batch_time = time.time() - batch_start
                self.logger.debug(f"⏱️  Batch search completed in {batch_time:.3f}s ({len(query_vectors)} queries)")
                
                return all_results
            else:
                # FAISS 없으면 단일 쿼리 검색으로 폴백
                self.logger.warning("FAISS not available, falling back to individual searches")
                return [
                    self._search_with_threshold(
                        q, k, source_types, similarity_threshold,
                        min_ml_confidence, min_quality_score,
                        filter_by_confidence, chunk_size_category,
                        deduplicate_by_group, embedding_version_id
                    ) if q and q.strip() else []
                    for q in queries
                ]
                
        except Exception as e:
            self.logger.error(f"Batch search failed: {e}", exc_info=True)
            return [[] for _ in queries]
    
    def _get_available_vector_tables(self, source_types: Optional[List[str]] = None, conn=None) -> List[Dict[str, Any]]:
        """
        사용 가능한 벡터 테이블 목록 조회 (동적 감지)
        
        Args:
            source_types: 필터링할 source_type 목록 (None이면 전체)
            conn: 기존 연결 (재사용, None이면 새로 생성)
        
        Returns:
            사용 가능한 테이블 설정 리스트
        """
        try:
            from core.config.vector_table_config import VECTOR_TABLE_MAPPING
        except ImportError:
            try:
                from lawfirm_langgraph.config.vector_table_config import VECTOR_TABLE_MAPPING
            except ImportError:
                self.logger.warning("⚠️ VECTOR_TABLE_MAPPING not found, using default")
                # 기본 설정 (하위 호환성)
                VECTOR_TABLE_MAPPING = {
                    'precedent_content': {
                        'table_name': 'precedent_chunks',
                        'id_column': 'id',
                        'vector_column': 'embedding_vector',
                        'version_column': 'embedding_version',
                        'source_type': 'precedent_content',
                        'enabled': True,
                        'priority': 1,
                        'weight': 1.0,
                        'min_results': 2,
                        'max_results': None
                    },
                    # 🔥 레거시 지원: case_paragraph는 precedent_content로 매핑
                    'case_paragraph': {
                        'table_name': 'precedent_chunks',
                        'id_column': 'id',
                        'vector_column': 'embedding_vector',
                        'version_column': 'embedding_version',
                        'source_type': 'precedent_content',
                        'enabled': True,
                        'priority': 1,
                        'weight': 1.0,
                        'min_results': 2,
                        'max_results': None
                    }
                }
        
        available_tables = []
        
        # 🔥 개선: 연결 재사용 (conn이 제공되면 재사용, 없으면 새로 생성)
        use_existing_conn = conn is not None
        if not use_existing_conn:
            conn_context = self._get_connection_context()
            conn = conn_context.__enter__()
        else:
            conn_context = None
        
        try:
            cursor = conn.cursor()
            
            for source_type, config in VECTOR_TABLE_MAPPING.items():
                # enabled 체크
                if not config.get('enabled', True):
                    continue
                
                # source_types 필터링 강화
                if source_types:
                    # 실제 source_type과 매핑된 source_type 모두 확인
                    mapped_source_type = config.get('source_type', source_type)
                    if source_type not in source_types and mapped_source_type not in source_types:
                        # 레거시 매핑 확인 (case_paragraph -> precedent_content)
                        if source_type == 'case_paragraph' and 'precedent_content' in source_types:
                            pass  # 허용
                        elif mapped_source_type == 'precedent_content' and 'precedent_content' in source_types:
                            pass  # 허용
                        else:
                            continue
                
                # 테이블 존재 여부 확인
                table_name = config['table_name']
                try:
                    cursor.execute("""
                        SELECT EXISTS (
                            SELECT FROM information_schema.tables 
                            WHERE table_schema = 'public' 
                            AND table_name = %s
                        )
                    """, (table_name,))
                    row = cursor.fetchone()
                    exists = row[0] if isinstance(row, tuple) else (row.get('exists', False) if isinstance(row, dict) else False)
                    
                    if exists:
                        # 벡터 컬럼 존재 여부 확인
                        cursor.execute("""
                            SELECT EXISTS (
                                SELECT FROM information_schema.columns 
                                WHERE table_schema = 'public' 
                                AND table_name = %s 
                                AND column_name = %s
                            )
                        """, (table_name, config['vector_column']))
                        row = cursor.fetchone()
                        has_vector = row[0] if isinstance(row, tuple) else (row.get('exists', False) if isinstance(row, dict) else False)
                        
                        if has_vector:
                            available_tables.append({
                                **config,
                                'source_type': source_type
                            })
                except Exception as e:
                    self.logger.warning(f"⚠️ Failed to check table {table_name}: {e}")
                    continue
        finally:
            # 🔥 개선: 새로 생성한 연결만 닫기 (재사용한 연결은 닫지 않음)
            if not use_existing_conn and conn_context:
                try:
                    conn_context.__exit__(None, None, None)
                except Exception:
                    pass
        
        # priority 기준 정렬
        available_tables.sort(key=lambda x: x.get('priority', 999))
        
        return available_tables
    
    def _search_with_pgvector_weighted(
        self,
        query_vec: np.ndarray,
        k: int,
        source_types: Optional[List[str]] = None,
        embedding_version_id: Optional[int] = None,
        similarity_threshold: float = 0.5,
        search_k_multiplier: float = 1.0,
        query_text: Optional[str] = None
    ) -> Tuple[List[Tuple[int, float, str]], Dict[str, Any]]:
        """
        가중치 기반 pgvector 검색 (각 테이블별 개별 검색 후 가중치 적용)
        
        Args:
            query_vec: 쿼리 벡터
            k: 반환할 최대 결과 수 (전체 검색 결과 기준)
            source_types: 필터링할 source_type 목록
            embedding_version_id: 임베딩 버전 ID
            similarity_threshold: 최소 유사도 임계값
            search_k_multiplier: search_k 배수 (재시도 시 증가)
        
        Returns:
            (검색 결과, 메타데이터) - 메타데이터에 table_version_map 포함
            검색 결과: [(chunk_id, weighted_similarity, source_type), ...]
            메타데이터: {'table_version_map': {source_type: [versions]}}
        
        Note:
            - 테이블별 min_results는 테이블별 최소 결과 수를 보장하기 위한 것이며,
              전체 검색 결과의 min_results와는 별개입니다.
            - 전체 검색 결과의 min_results는 재시도 로직(search 메서드)에서 처리됩니다.
            - 이 메서드는 각 테이블에서 충분한 결과를 확보하려고 시도하지만,
              최종 결과가 전체 min_results를 만족하지 않으면 재시도 로직이 작동합니다.
        """
        # 🔥 개선: 테이블별 버전 추적 맵 초기화
        table_version_map = {}  # source_type별 실제 사용된 버전 추적
        
        # 🔥 개선: 연결을 먼저 가져오고 재사용
        with self._get_connection_context() as conn:
            # 같은 연결을 재사용하여 테이블 목록 조회 (중첩 연결 제거)
            available_tables = self._get_available_vector_tables(source_types, conn=conn)
            
            if not available_tables:
                self.logger.warning("⚠️ No available vector tables found")
                return [], {'table_version_map': {}}
            
            all_results = []
            failed_tables_count = 0
            failed_table_names = []
            # 각 테이블별로 개별 검색
            for table_config in available_tables:
                table_name = table_config['table_name']
                id_column = table_config['id_column']
                vector_column = table_config['vector_column']
                source_type = table_config['source_type']
                # 🔥 레거시 정규화: case_paragraph는 precedent_content로 표시
                if source_type == 'case_paragraph':
                    source_type = 'precedent_content'
                weight = table_config.get('weight', 1.0)
                min_results = table_config.get('min_results', 0)
                max_results = table_config.get('max_results')
                
                try:
                    # PgVectorAdapter 생성
                    if VectorSearchFactory:
                        adapter = VectorSearchFactory.create(
                            method='pgvector',
                            connection=conn,
                            table_name=table_name,
                            id_column=id_column,
                            vector_column=vector_column
                        )
                    else:
                        from core.search.engines.vector_search_adapter import PgVectorAdapter
                        adapter = PgVectorAdapter(
                            connection=conn,
                            table_name=table_name,
                            id_column=id_column,
                            vector_column=vector_column
                        )
                    
                    # 필터 구성
                    filters = {}
                    # 🔥 개선: 테이블별 버전 전략 적용
                    # - precedent_chunks: embedding_version=1 고정
                    # - statute_embeddings: embedding_version IN (1, 2) (추후 변경 가능)
                    version_column = table_config.get('version_column', 'embedding_version')
                    
                    # 테이블에 버전 컬럼이 있는지 확인
                    try:
                        cursor = conn.cursor()
                        cursor.execute(f"""
                            SELECT EXISTS (
                                SELECT 1 FROM information_schema.columns 
                                WHERE table_name = %s AND column_name = %s
                            )
                        """, (table_name, version_column))
                        row = cursor.fetchone()
                        # 🔥 개선: fetchone() 결과 안전하게 처리
                        if row:
                            has_version_column = row[0] if isinstance(row, (tuple, list)) else row.get('exists', False)
                        else:
                            has_version_column = False
                        
                        if has_version_column:
                            # 테이블별 버전 전략 결정
                            if table_name == 'precedent_chunks':
                                # 🔥 개선: precedent_chunks는 embedding_versions.version 값을 사용
                                # precedent_chunks.embedding_version 컬럼에는 embedding_versions.version이 저장됨
                                # embedding_version_id가 지정된 경우, 해당 버전의 data_type을 확인하여 precedents 타입만 사용
                                if embedding_version_id is not None:
                                    # embedding_version_id의 data_type 확인
                                    cursor.execute("""
                                        SELECT version, data_type FROM embedding_versions WHERE id = %s
                                    """, (embedding_version_id,))
                                    version_row = cursor.fetchone()
                                    if version_row:
                                        version_num = version_row[0] if isinstance(version_row, (tuple, list)) else version_row.get('version', embedding_version_id)
                                        version_data_type = version_row[1] if isinstance(version_row, (tuple, list)) and len(version_row) > 1 else (version_row.get('data_type') if isinstance(version_row, dict) else None)
                                        
                                        # precedent_chunks는 precedents 타입만 사용
                                        if version_data_type == 'precedents':
                                            target_versions = [version_num]
                                            self.logger.debug(
                                                f"📋 [PGVECTOR] {source_type}: Using version={version_num} "
                                                f"(from version_id={embedding_version_id}, data_type=precedents) for precedent_chunks"
                                            )
                                        else:
                                            # statutes 타입이면 무시하고 precedents 활성 버전 사용
                                            self.logger.info(
                                                f"📋 [PGVECTOR] {source_type}: embedding_version_id={embedding_version_id} is {version_data_type} type, "
                                                f"ignoring and using precedents active version for precedent_chunks"
                                            )
                                            embedding_version_id = None  # 아래 else 블록으로 이동하여 활성 버전 사용
                                    else:
                                        # version_id를 찾을 수 없으면 활성 버전 사용
                                        self.logger.warning(
                                            f"⚠️ [PGVECTOR] {source_type}: version_id={embedding_version_id} not found, "
                                            f"using active precedents version for precedent_chunks"
                                        )
                                        embedding_version_id = None  # 아래 else 블록으로 이동하여 활성 버전 사용
                                
                                # embedding_version_id가 None이거나 precedents 타입이 아닌 경우 활성 버전 사용
                                if embedding_version_id is None:
                                    # 활성 버전 조회 후 version 번호 사용
                                    active_version_id = self._get_active_embedding_version_id(data_type='precedents')
                                    if active_version_id:
                                        # 활성 버전의 version 번호 조회
                                        cursor.execute("""
                                            SELECT version FROM embedding_versions WHERE id = %s
                                        """, (active_version_id,))
                                        version_row = cursor.fetchone()
                                        if version_row:
                                            version_num = version_row[0] if isinstance(version_row, (tuple, list)) else version_row.get('version', 1)
                                            target_versions = [version_num]
                                            self.logger.debug(
                                                f"📋 [PGVECTOR] {source_type}: Using active version={version_num} "
                                                f"(from version_id={active_version_id}) for precedent_chunks"
                                            )
                                        else:
                                            # 활성 버전이 없으면 버전 1 사용 (하위 호환성)
                                            target_versions = [1]
                                            self.logger.warning(
                                                f"⚠️ [PGVECTOR] {source_type}: Active version_id={active_version_id} not found, "
                                                f"falling back to version=1 for precedent_chunks"
                                            )
                                    else:
                                        # 활성 버전이 없으면 버전 1 사용 (하위 호환성)
                                        target_versions = [1]
                                        self.logger.warning(
                                            f"⚠️ [PGVECTOR] {source_type}: No active version found, "
                                            f"falling back to version=1 for precedent_chunks"
                                        )
                            elif table_name == 'statute_embeddings':
                                # statute_embeddings는 embedding_version IN (1, 2)
                                # 추후 변경 가능하도록 설정에서 가져올 수 있음
                                target_versions = [1, 2]
                                self.logger.debug(
                                    f"📋 [PGVECTOR] {source_type}: Using multi-version strategy "
                                    f"(statute_embeddings: version IN (1, 2))"
                                )
                            else:
                                # 기타 테이블은 embedding_version_id 사용 (지정된 경우)
                                if embedding_version_id is not None:
                                    target_versions = [embedding_version_id]
                                else:
                                    target_versions = None
                            
                            # 버전 필터 적용
                            if target_versions:
                                # 각 버전의 데이터 존재 여부 확인
                                available_versions = []
                                for version in target_versions:
                                    cursor.execute(f"""
                                        SELECT COUNT(*) FROM {table_name}
                                        WHERE {version_column} = %s AND {vector_column} IS NOT NULL
                                    """, (version,))
                                    row = cursor.fetchone()
                                    if row:
                                        version_data_count = row[0] if isinstance(row, (tuple, list)) else row.get('count', 0)
                                    else:
                                        version_data_count = 0
                                    
                                    if version_data_count > 0:
                                        available_versions.append(version)
                                
                                if available_versions:
                                    # 🔥 개선: 실제 사용된 버전을 table_version_map에 기록
                                    if source_type not in table_version_map:
                                        table_version_map[source_type] = []
                                    for version in available_versions:
                                        if version not in table_version_map[source_type]:
                                            table_version_map[source_type].append(version)
                                    
                                    if len(available_versions) == 1:
                                        # 단일 버전인 경우
                                        # 🔥 개선: 단일 값만 전달 (리스트가 아닌 정수 값)
                                        version_val = available_versions[0]
                                        filters[version_column] = int(version_val) if isinstance(version_val, (int, float)) else version_val
                                        # 해당 버전의 데이터 수 확인
                                        cursor.execute(f"""
                                            SELECT COUNT(*) FROM {table_name}
                                            WHERE {version_column} = %s AND {vector_column} IS NOT NULL
                                        """, (available_versions[0],))
                                        row = cursor.fetchone()
                                        version_count = row[0] if isinstance(row, (tuple, list)) else row.get('count', 0) if row else 0
                                        self.logger.debug(
                                            f"✅ [PGVECTOR] {source_type}: Using version filter "
                                            f"(version={available_versions[0]}, count={version_count})"
                                        )
                                    else:
                                        # 여러 버전인 경우 첫 번째 버전만 사용 (타입 오류 방지)
                                        # 🔥 개선: 단일 값만 전달 (리스트가 아닌 정수 값)
                                        if len(available_versions) > 0:
                                            version_val = available_versions[0]
                                            filters[version_column] = int(version_val) if isinstance(version_val, (int, float)) else version_val
                                        else:
                                            filters[version_column] = None
                                        if filters[version_column] is None:
                                            continue
                                        # 각 버전별 데이터 수 확인
                                        version_counts = {}
                                        for version in available_versions:
                                            cursor.execute(f"""
                                                SELECT COUNT(*) FROM {table_name}
                                                WHERE {version_column} = %s AND {vector_column} IS NOT NULL
                                            """, (version,))
                                            row = cursor.fetchone()
                                            version_counts[version] = row[0] if isinstance(row, (tuple, list)) else row.get('count', 0) if row else 0
                                        total_count = sum(version_counts.values())
                                        self.logger.debug(
                                            f"✅ [PGVECTOR] {source_type}: Using multi-version filter "
                                            f"(versions={available_versions}, counts={version_counts}, total={total_count})"
                                        )
                                else:
                                    # 🔥 개선: target_versions에 해당하는 버전이 없을 때, 사용 가능한 버전을 먼저 확인
                                    # 테이블에 실제로 존재하는 버전들을 조회
                                    cursor.execute(f"""
                                        SELECT DISTINCT {version_column} 
                                        FROM {table_name}
                                        WHERE {vector_column} IS NOT NULL
                                        ORDER BY {version_column} DESC
                                    """)
                                    version_rows = cursor.fetchall()
                                    all_available_versions = []
                                    for version_row in version_rows:
                                        if version_row:
                                            version_val = version_row[0] if isinstance(version_row, (tuple, list)) else version_row.get(version_column)
                                            if version_val is not None:
                                                all_available_versions.append(version_val)
                                    
                                    if all_available_versions:
                                        # 사용 가능한 버전이 있으면, target_versions에 가장 가까운 버전 선택
                                        # target_versions 중 가장 큰 값보다 작거나 같은 최대 버전 선택
                                        max_target_version = max(target_versions) if target_versions else None
                                        fallback_version = None
                                        
                                        if max_target_version:
                                            # target_versions보다 작거나 같은 최대 버전 찾기
                                            for version in sorted(all_available_versions, reverse=True):
                                                if version <= max_target_version:
                                                    fallback_version = version
                                                    break
                                        
                                        # 적절한 폴백 버전을 찾지 못한 경우, 가장 최신 버전 사용
                                        if fallback_version is None:
                                            fallback_version = max(all_available_versions)
                                        
                                        # 🔥 개선: 단일 값만 전달 (리스트가 아닌 정수 값)
                                        filters[version_column] = int(fallback_version) if isinstance(fallback_version, (int, float)) else fallback_version
                                        
                                        # 🔥 개선: 폴백된 버전을 table_version_map에 기록
                                        if source_type not in table_version_map:
                                            table_version_map[source_type] = []
                                        if fallback_version not in table_version_map[source_type]:
                                            table_version_map[source_type].append(fallback_version)
                                        
                                        # 선택된 버전의 데이터 수 확인
                                        cursor.execute(f"""
                                            SELECT COUNT(*) FROM {table_name}
                                            WHERE {version_column} = %s AND {vector_column} IS NOT NULL
                                        """, (fallback_version,))
                                        row = cursor.fetchone()
                                        version_count = row[0] if isinstance(row, (tuple, list)) else row.get('count', 0) if row else 0
                                        
                                        self.logger.warning(
                                            f"⚠️ [PGVECTOR] {source_type}: No vectors found for target versions {target_versions}. "
                                            f"Available versions: {sorted(all_available_versions)}. "
                                            f"Using fallback version {fallback_version} (count={version_count})"
                                        )
                                    else:
                                        # 사용 가능한 버전이 전혀 없으면 모든 버전 검색 (기존 동작)
                                        # 🔥 개선: 모든 버전 허용을 None으로 표시
                                        table_version_map[source_type] = None
                                        self.logger.warning(
                                            f"⚠️ [PGVECTOR] {source_type}: No vectors found for target versions {target_versions} "
                                            f"and no available versions in table. Searching all versions"
                                        )
                            elif embedding_version_id is not None:
                                # embedding_version_id가 지정된 경우 기존 로직 사용
                                cursor.execute(f"""
                                    SELECT COUNT(*) FROM {table_name}
                                    WHERE {version_column} = %s AND {vector_column} IS NOT NULL
                                """, (embedding_version_id,))
                                row = cursor.fetchone()
                                if row:
                                    version_data_count = row[0] if isinstance(row, (tuple, list)) else row.get('count', 0)
                                else:
                                    version_data_count = 0
                                
                                if version_data_count > 0:
                                    # 🔥 개선: 단일 값만 전달 (리스트가 아닌 정수 값)
                                    # psycopg2가 리스트를 배열로 자동 변환하는 문제 방지
                                    if isinstance(embedding_version_id, (list, tuple)):
                                        filters[version_column] = int(embedding_version_id[0]) if len(embedding_version_id) > 0 else None
                                    else:
                                        filters[version_column] = int(embedding_version_id) if isinstance(embedding_version_id, (int, float)) else embedding_version_id
                                    # 🔥 개선: 실제 사용된 버전을 table_version_map에 기록
                                    if source_type not in table_version_map:
                                        table_version_map[source_type] = []
                                    if embedding_version_id not in table_version_map[source_type]:
                                        table_version_map[source_type].append(embedding_version_id)
                                    self.logger.debug(
                                        f"✅ [PGVECTOR] {source_type}: Using version filter "
                                        f"(version_id={embedding_version_id}, count={version_data_count})"
                                    )
                                else:
                                    # 🔥 개선: embedding_version_id에 해당하는 버전이 없을 때, 사용 가능한 버전을 먼저 확인
                                    # 테이블에 실제로 존재하는 버전들을 조회
                                    cursor.execute(f"""
                                        SELECT DISTINCT {version_column} 
                                        FROM {table_name}
                                        WHERE {vector_column} IS NOT NULL
                                        ORDER BY {version_column} DESC
                                    """)
                                    version_rows = cursor.fetchall()
                                    all_available_versions = []
                                    for version_row in version_rows:
                                        if version_row:
                                            version_val = version_row[0] if isinstance(version_row, (tuple, list)) else version_row.get(version_column)
                                            if version_val is not None:
                                                all_available_versions.append(version_val)
                                    
                                    if all_available_versions:
                                        # 사용 가능한 버전이 있으면, embedding_version_id보다 작거나 같은 최대 버전 선택
                                        fallback_version = None
                                        for version in sorted(all_available_versions, reverse=True):
                                            if version <= embedding_version_id:
                                                fallback_version = version
                                                break
                                        
                                        # 적절한 폴백 버전을 찾지 못한 경우, 가장 최신 버전 사용
                                        if fallback_version is None:
                                            fallback_version = max(all_available_versions)
                                        
                                        # 🔥 개선: 단일 값만 전달 (리스트가 아닌 정수 값)
                                        filters[version_column] = int(fallback_version) if isinstance(fallback_version, (int, float)) else fallback_version
                                        
                                        # 🔥 개선: 폴백된 버전을 table_version_map에 기록
                                        if source_type not in table_version_map:
                                            table_version_map[source_type] = []
                                        if fallback_version not in table_version_map[source_type]:
                                            table_version_map[source_type].append(fallback_version)
                                        
                                        # 선택된 버전의 데이터 수 확인
                                        cursor.execute(f"""
                                            SELECT COUNT(*) FROM {table_name}
                                            WHERE {version_column} = %s AND {vector_column} IS NOT NULL
                                        """, (fallback_version,))
                                        row = cursor.fetchone()
                                        version_count = row[0] if isinstance(row, (tuple, list)) else row.get('count', 0) if row else 0
                                        
                                        self.logger.warning(
                                            f"⚠️ [PGVECTOR] {source_type}: No vectors found for version_id {embedding_version_id}. "
                                            f"Available versions: {sorted(all_available_versions)}. "
                                            f"Using fallback version {fallback_version} (count={version_count})"
                                        )
                                    else:
                                        # 사용 가능한 버전이 전혀 없으면 모든 버전 검색 (기존 동작)
                                        self.logger.warning(
                                            f"⚠️ [PGVECTOR] {source_type}: No vectors found for version_id {embedding_version_id} "
                                            f"and no available versions in table. Searching all versions"
                                        )
                            else:
                                # 버전 필터 없음 (모든 버전 검색)
                                # 🔥 개선: 모든 버전 허용을 None으로 표시
                                table_version_map[source_type] = None
                                self.logger.debug(
                                    f"📋 [PGVECTOR] {source_type}: No version filter, searching all versions"
                                )
                        else:
                            # 버전 컬럼이 없으면 필터 제거
                            self.logger.debug(
                                f"⚠️ [PGVECTOR] {source_type}: No version column found, "
                                f"searching all versions"
                            )
                    except Exception as e:
                        # 오류 발생 시 필터 제거하고 계속 진행
                        self.logger.warning(
                            f"⚠️ [PGVECTOR] {source_type}: Error checking version data: {e}, "
                            f"searching all versions"
                        )
                    
                    # 🔥 개선: 테이블별로 적절한 모델로 재인코딩된 벡터 사용
                    # precedent_chunks 테이블은 precedents 타입 모델 사용
                    if table_name == 'precedent_chunks':
                        if not query_text:
                            # 🔥 원본 벡터 사용 금지: query_text가 없으면 검색 불가
                            raise ValueError(
                                f"query_text is required for {source_type} search. "
                                f"Cannot use original vector due to potential dimension mismatch."
                            )
                        
                        # precedents 타입 모델로 재초기화
                        self._ensure_correct_embedding_model(data_type='precedents')
                        
                        # 🔥 간소화: 캐시 키에 모델명과 버전이 포함되어 있으므로 자동으로 올바른 캐시 사용
                        # 활성 버전 ID 가져오기
                        active_version_id = self._get_active_embedding_version_id(data_type='precedents')
                        
                        # 재인코딩 (캐시 키에 모델명과 버전이 포함되어 있어 자동으로 올바른 벡터 사용)
                        table_query_vec = self._encode_query(
                            query_text, 
                            use_cache=True,  # 캐시 사용 가능 (모델명과 버전이 키에 포함됨)
                            model_name=self.model_name,
                            version_id=active_version_id
                        )
                        
                        if table_query_vec is None:
                            raise ValueError(f"Failed to encode query for {source_type}")
                        
                        # 🔥 차원 검증 강화: 재인코딩 후 차원 확인, 틀리면 즉시 오류 발생
                        # 활성 버전의 차원 정보 가져오기
                        expected_dim = self._get_embedding_dimension_for_version(active_version_id)
                        if expected_dim is None:
                            # 차원 정보를 가져올 수 없으면 모델명으로 추정
                            if 'ko-legal-sbert' in (self.model_name or '').lower():
                                expected_dim = 768
                            elif 'ko-sroberta' in (self.model_name or '').lower():
                                expected_dim = 384
                            else:
                                # 기본값으로 768 사용 (precedents는 보통 768)
                                expected_dim = 768
                                self.logger.warning(
                                    f"⚠️ [DIMENSION] Could not determine expected dimension for version {active_version_id}, "
                                    f"using default: {expected_dim}"
                                )
                        
                        actual_dim = len(table_query_vec)
                        
                        if actual_dim != expected_dim:
                            raise ValueError(
                                f"Query vector dimension mismatch for {source_type}: "
                                f"expected {expected_dim}, got {actual_dim}. "
                                f"Model: {self.model_name}, Version: {active_version_id}"
                            )
                        
                        self.logger.info(
                            f"✅ [PGVECTOR] Using query vector for {source_type} "
                            f"(model={self.model_name}, version={active_version_id}, dim={actual_dim}, verified)"
                        )
                    else:
                        # 다른 테이블은 원본 벡터 사용
                        table_query_vec = query_vec
                    
                    # 각 테이블별 검색 (더 많은 후보 검색)
                    # search_k_multiplier를 적용하여 재시도 시 더 많은 후보 검색
                    # 재시도 시에는 max_results를 무시하고 search_k_multiplier 우선 적용
                    if search_k_multiplier > 1.0:
                        # 재시도 중이면 max_results 무시하고 더 많은 후보 검색
                        # 재시도 효과를 극대화하기 위해 max_results 제한을 우회
                        base_search_k = k * 2
                        self.logger.debug(
                            f"🔄 [RETRY] Ignoring max_results={max_results} for retry "
                            f"(search_k_multiplier={search_k_multiplier:.1f})"
                        )
                    else:
                        # 첫 번째 시도만 max_results 적용
                        base_search_k = (k * 2) if max_results is None else max_results
                    search_k = int(base_search_k * search_k_multiplier)
                    table_results = adapter.search(
                        query_vector=table_query_vec,
                        limit=search_k,
                        filters=filters
                    )
                    
                    self.logger.debug(f"🔍 [PGVECTOR] {source_type}: Found {len(table_results)} candidates")
                    
                    # 거리를 유사도로 변환하고 가중치 적용
                    # 성능 최적화: O(1) 중복 체크를 위한 set 사용
                    table_weighted_results = []
                    seen_chunk_ids = set()  # O(1) 중복 체크를 위한 set
                    distances_sample = []
                    similarities_sample = []
                    
                    for chunk_id, distance in table_results:
                        # 코사인 거리를 유사도로 변환
                        # pgvector의 <=> 연산자는 코사인 거리 (0~2 범위)
                        # similarity = 1.0 - distance (0~1 범위로 정규화)
                        similarity = 1.0 - float(distance)
                        
                        # 샘플링 (처음 5개만)
                        if len(distances_sample) < 5:
                            distances_sample.append(distance)
                            similarities_sample.append(similarity)
                        
                        if similarity >= similarity_threshold:
                            # 중복 체크 (O(1) 복잡도)
                            if chunk_id not in seen_chunk_ids:
                                # 가중치 적용: weighted_similarity = similarity * weight
                                weighted_similarity = similarity * weight
                                # 정규화: 가중치 적용 후 1.0 초과 방지
                                weighted_similarity = normalize_score(weighted_similarity)
                                table_weighted_results.append((chunk_id, weighted_similarity, similarity, source_type))
                                seen_chunk_ids.add(chunk_id)
                    
                    # 디버깅: 거리와 유사도 샘플 로깅
                    if distances_sample:
                        self.logger.debug(
                            f"📊 [PGVECTOR] {source_type} distance/similarity sample: "
                            f"distances={[f'{d:.4f}' for d in distances_sample[:3]]}, "
                            f"similarities={[f'{s:.4f}' for s in similarities_sample[:3]]}, "
                            f"threshold={similarity_threshold:.3f}"
                        )
                    
                    # 원본 유사도 기준 정렬 (가중치 적용 전)
                    table_weighted_results.sort(key=lambda x: x[2], reverse=True)
                    
                    # 최소 결과 수 보장 (테이블별 min_results)
                    # 주의: 이는 테이블별 최소 결과 수이며, 전체 검색 결과의 min_results와는 별개
                    # 전체 검색 결과의 min_results는 재시도 로직에서 처리됨
                    if min_results > 0 and len(table_weighted_results) < min_results:
                        # 임계값을 점진적으로 낮춰서 더 많은 결과 확보
                        # 1차: RELAXED_THRESHOLD_RATIO 비율로 임계값 완화
                        relaxed_threshold = similarity_threshold * self.RELAXED_THRESHOLD_RATIO
                        for chunk_id, distance in table_results:
                            # 중복 체크 (O(1) 복잡도)
                            if chunk_id in seen_chunk_ids:
                                continue
                            
                            similarity = 1.0 - float(distance)
                            if similarity >= relaxed_threshold:
                                weighted_similarity = similarity * weight
                                # 정규화: 가중치 적용 후 1.0 초과 방지
                                weighted_similarity = normalize_score(weighted_similarity)
                                table_weighted_results.append((chunk_id, weighted_similarity, similarity, source_type))
                                seen_chunk_ids.add(chunk_id)
                                if len(table_weighted_results) >= min_results:
                                    break
                        
                        # 2차: 여전히 부족하면 상위 N개 강제 포함 (임계값 무시)
                        if len(table_weighted_results) < min_results:
                            self.logger.warning(
                                f"⚠️ [PGVECTOR] {source_type}: Only {len(table_weighted_results)} results after relaxed threshold, "
                                f"forcing top {min_results} results (ignoring threshold)"
                            )
                            # 상위 min_results개를 강제로 포함 (임계값 무시)
                            for chunk_id, distance in table_results[:min_results * 2]:  # 더 많은 후보 확인
                                # 중복 체크 (O(1) 복잡도)
                                if chunk_id in seen_chunk_ids:
                                    continue
                                
                                similarity = 1.0 - float(distance)
                                weighted_similarity = similarity * weight
                                # 정규화: 가중치 적용 후 1.0 초과 방지
                                weighted_similarity = normalize_score(weighted_similarity)
                                table_weighted_results.append((chunk_id, weighted_similarity, similarity, source_type))
                                seen_chunk_ids.add(chunk_id)
                                if len(table_weighted_results) >= min_results:
                                    break
                    
                    # max_results 제한 적용
                    if max_results:
                        table_weighted_results = table_weighted_results[:max_results]
                    
                    all_results.extend(table_weighted_results)
                    self.logger.info(
                        f"✅ [PGVECTOR] {source_type}: {len(table_weighted_results)} results "
                        f"(weight={weight}, min={min_results})"
                    )
                    
                except Exception as e:
                    failed_tables_count += 1
                    failed_table_names.append(table_name)
                    self.logger.warning(
                        f"⚠️ [PGVECTOR] Failed to search {table_name}: {e}",
                        exc_info=self.logger.isEnabledFor(logging.DEBUG)
                    )
                    continue
        
        # 테이블별 검색 실패 통계 로깅
        if failed_tables_count > 0:
            self.logger.warning(
                f"⚠️ [PGVECTOR] {failed_tables_count}/{len(available_tables)} table(s) failed: {failed_table_names}"
            )
        
        # 가중 유사도 기준으로 상위 k개 선택 (성능 최적화: 부분 정렬)
        # heapq.nlargest를 사용하여 전체 정렬 대신 상위 k개만 선택
        # 시간 복잡도: O(n log k) vs O(n log n) (n=all_results 수, k=요청된 결과 수)
        if len(all_results) <= k:
            # 결과가 k개 이하면 정렬만 수행
            all_results.sort(key=lambda x: x[1], reverse=True)
            final_results = all_results
        else:
            # 결과가 k개보다 많으면 heapq.nlargest 사용 (더 효율적)
            final_results = heapq.nlargest(k, all_results, key=lambda x: x[1])
        
        self.logger.info(
            f"🔍 [PGVECTOR WEIGHTED] Total {len(final_results)} results from {len(available_tables)} tables"
        )
        
        # 타입별 분포 로깅
        type_counts = {}
        for _, _, _, source_type in final_results:
            type_counts[source_type] = type_counts.get(source_type, 0) + 1
        self.logger.debug(f"📊 [PGVECTOR] Type distribution: {type_counts}")
        
        # 🔥 개선: 테이블별 버전 맵 로깅
        if table_version_map:
            self.logger.debug(
                f"📝 [VERSION TRACKING] Table version map: {table_version_map}"
            )
        
        return (
            [(cid, weighted_sim, source_type) for cid, weighted_sim, _, source_type in final_results],
            {'table_version_map': table_version_map}
        )
    
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
                               embedding_version_id: Optional[int] = None,
                               search_k_multiplier: float = 1.0) -> List[Dict[str, Any]]:
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
            search_k_multiplier: search_k 배수 (재시도 시 증가)
        """
        # 빈 쿼리 검증 추가
        if not query or not query.strip():
            self.logger.warning("⚠️ [SEARCH] 빈 쿼리로 검색을 수행할 수 없습니다. 빈 결과를 반환합니다.")
            return []
        
        try:
            normalized_query = self._normalize_query(query)
            
            # 🔥 개선: 동적 임계값 조정 (쿼리 특성 기반)
            adjusted_threshold = self._adjust_threshold_dynamically(
                query=normalized_query,
                source_types=source_types,
                base_threshold=similarity_threshold
            )
            if adjusted_threshold != similarity_threshold:
                self.logger.debug(
                    f"📊 [DYNAMIC THRESHOLD] Adjusted threshold: {similarity_threshold:.3f} → {adjusted_threshold:.3f} "
                    f"(query_length={len(normalized_query)}, source_types={source_types})"
                )
                similarity_threshold = adjusted_threshold
            
            # 벡터 인덱스 검색 질의 로깅
            search_query_msg = (
                f"🔍 [VECTOR INDEX SEARCH] 질의: '{query}' "
                f"(normalized='{normalized_query}'), "
                f"k={k}, threshold={similarity_threshold:.3f}, "
                f"version_id={embedding_version_id}, source_types={source_types}"
            )
            print(search_query_msg, flush=True, file=sys.stdout)
            self.logger.info(search_query_msg)
            self.logger.info(
                f"🔍 [SEARCH WITH THRESHOLD] query: original='{query[:80]}...', "
                f"normalized='{normalized_query[:80]}...', "
                f"k={k}, threshold={similarity_threshold}, version_id={embedding_version_id}"
            )
            self.logger.debug(f"_search_with_threshold called: query='{query[:50]}...', k={k}, threshold={similarity_threshold}, version_id={embedding_version_id}")
            import time
            step_times = {}
            step_start = time.time()
            
            # 1. 쿼리 임베딩 생성 (캐시 사용)
            encode_start = time.time()
            self.logger.debug(f"🔍 [QUERY ENCODING] Encoding query: '{normalized_query[:80]}...'")
            query_vec = self._encode_query(query)
            if query_vec is None:
                return []
            step_times['query_encoding'] = time.time() - encode_start
            if step_times['query_encoding'] < 0.001:
                self.logger.debug("⏱️  Query encoding: 0.000s (cached)")
            else:
                self.logger.debug(f"⏱️  Query encoding: {step_times['query_encoding']:.3f}s")

            # 2. 벡터 검색 방법에 따라 분기 (pgvector 또는 FAISS)
            search_start = time.time()
            similarities = []
            
            # pgvector 검색 (가중치 기반 개별 검색) - pgvector만 사용
            if not PGVECTOR_AVAILABLE:
                self.logger.error("❌ [PGVECTOR] pgvector is not available. Cannot perform search.")
                return []
            
            try:
                self.logger.info(f"🔍 [PGVECTOR SEARCH] Using weighted pgvector search (pgvector only mode)")
                
                # 가중치 기반 검색 실행 (search_k_multiplier 적용)
                pgvector_results, pgvector_metadata = self._search_with_pgvector_weighted(
                    query_vec=query_vec,
                    k=k,
                    source_types=source_types,
                    embedding_version_id=embedding_version_id,
                    similarity_threshold=similarity_threshold,
                    search_k_multiplier=search_k_multiplier,
                    query_text=normalized_query
                )
                
                # 🔥 개선: 테이블별 버전 맵 저장 (검증 단계에서 사용)
                table_version_map = pgvector_metadata.get('table_version_map', {})
                if table_version_map:
                    # 인스턴스 변수에 저장 (검증 단계에서 사용)
                    self._pgvector_table_version_map = table_version_map
                    self.logger.debug(
                        f"📝 [VERSION TRACKING] Stored table_version_map: {table_version_map}"
                    )
                
                # 결과 변환 (가중 유사도 사용)
                for chunk_id, weighted_similarity, source_type in pgvector_results:
                    similarities.append((chunk_id, weighted_similarity))
                    # source_type은 나중에 메타데이터 조회 시 사용
                
                # 🔥 개선: pgvector 검색 후 결과 후처리를 위해 필요한 벡터만 로드
                # 결과 후처리 단계에서 벡터를 사용하므로 미리 로드
                if similarities and len(similarities) > 0:
                    # 🔥 성능 최적화: 검색 결과의 chunk_id만 로드
                    result_chunk_ids = [chunk_id for chunk_id, _ in similarities]
                    self.logger.debug(
                        f"📋 [OPTIMIZED] Loading only {len(result_chunk_ids)} chunk vectors "
                        f"for result processing (pgvector search completed)"
                    )
                    result_chunk_vectors = self._load_chunk_vectors(
                        source_types=source_types,
                        embedding_version_id=embedding_version_id,
                        chunk_ids=result_chunk_ids  # 🔥 검색 결과의 chunk_id만 전달
                    )
                    # 인스턴스 변수에 저장 (결과 후처리에서 사용)
                    if not hasattr(self, '_chunk_vectors'):
                        self._chunk_vectors = {}
                    self._chunk_vectors.update(result_chunk_vectors)
                
                step_times['pgvector_search'] = time.time() - search_start
                self.logger.info(
                    f"⏱️  PgVector weighted search: {step_times['pgvector_search']:.3f}s, "
                    f"{len(similarities)} results"
                )
                
            except Exception as e:
                self.logger.error(f"❌ [PGVECTOR] Search error: {e}", exc_info=True)
                # pgvector 전용 모드에서 실패하면 빈 결과 반환
                return []
            
            # 🔥 pgvector 검색 완료 후 similarities가 있으면 벡터 검색 단계 건너뛰기
            # (이미 pgvector 검색에서 필요한 벡터를 로드했으므로)
            if similarities and len(similarities) > 0:
                step_times['vector_search'] = time.time() - search_start
                self.logger.debug(f"⏱️  Vector search (pgvector): {step_times['vector_search']:.3f}s (already loaded)")
            # FAISS 검색은 건너뜀 (pgvector만 사용)
            # 기존 FAISS 검색 코드는 주석 처리
            elif False:  # FAISS 검색 비활성화 (pgvector만 사용)
                # nprobe 동적 튜닝 (k 값에 따라 조정)
                # FAISS 인덱스 검색 (빠른 근사 검색)
                query_vec_np = np.array([query_vec]).astype('float32')
                
                # 차원 검증: 쿼리 벡터 차원과 인덱스 차원이 일치하는지 확인
                query_dim = query_vec_np.shape[1]
                index_dim = self.index.d
                if query_dim != index_dim:
                    error_msg = (
                        f"FAISS index dimension mismatch: query vector dimension ({query_dim}) "
                        f"does not match index dimension ({index_dim}). "
                        f"This usually happens after re-embedding with a different model. "
                        f"Please rebuild the FAISS index."
                    )
                    self.logger.error(f"❌ {error_msg}")
                    raise ValueError(error_msg)
                
                # Phase 3 최적화: search_k 동적 조정 (인덱스 타입에 따라)
                index_type_name = type(self.index).__name__
                is_indexivfpq = 'IndexIVFPQ' in index_type_name
                
                if 'IndexIVFPQ' in index_type_name:
                    # 압축 인덱스는 더 많은 후보 필요
                    search_k = k * 4
                elif 'IndexIVF' in index_type_name:
                    # 일반 IVF 인덱스
                    search_k = k * 3
                else:
                    # 정확한 인덱스 (Flat 등)
                    search_k = k * 2
                
                # nprobe 설정 (IndexIVF 계열만 지원)
                if hasattr(self.index, 'nprobe'):
                    optimal_nprobe = self._calculate_optimal_nprobe(k, self.index.ntotal)
                    if self.index.nprobe != optimal_nprobe:
                        self.index.nprobe = optimal_nprobe
                        self.logger.debug(f"Adjusted nprobe to {optimal_nprobe} for k={k}")
                
                self.logger.debug(f"Using search_k={search_k} (k={k}, index_type={index_type_name})")
                
                distances, indices = self.index.search(query_vec_np, search_k)
                
                self.logger.info(f"🔍 FAISS search returned {len(indices[0])} results (index_type={type(self.index).__name__}, filtering with version_id={embedding_version_id})")
                
                # _chunk_ids 확인
                if not hasattr(self, '_chunk_ids') or not self._chunk_ids:
                    self.logger.error(f"❌ _chunk_ids is empty or not loaded! FAISS index has {self.index.ntotal} vectors but _chunk_ids has {len(self._chunk_ids) if hasattr(self, '_chunk_ids') else 0} entries")
                    self.logger.error("Attempting to reload _chunk_ids from database...")
                    try:
                        # DatabaseAdapter 사용
                        with self._get_connection_context() as conn:
                            cursor = conn.cursor()
                            cursor.execute(
                                "SELECT chunk_id FROM embeddings WHERE version_id = %s ORDER BY chunk_id",
                                (embedding_version_id or self._get_active_embedding_version_id(),)
                            )
                            rows = cursor.fetchall()
                            self._chunk_ids = [row[0] if isinstance(row, tuple) else row['chunk_id'] for row in rows]
                        self.logger.info(f"✅ Reloaded {len(self._chunk_ids)} chunk_ids from database")
                    except Exception as e:
                        self.logger.error(f"Failed to reload _chunk_ids: {e}")
                        return []

                similarities = []
                skipped_count = 0
                filtered_by_version = 0
                filtered_by_source = 0
                filtered_by_threshold = 0
                filtered_by_not_found = 0
                
                # source_type 필터 완화를 위한 플래그
                source_types_relaxed = False
                original_source_types = source_types.copy() if source_types else None
                
                # Phase 1 최적화: FAISS 검색 직후 모든 후보 chunk_id를 한 번에 수집
                candidate_chunk_ids = []
                distance_idx_map = {}  # chunk_id -> (distance, idx) 매핑
                
                for i, (distance, idx) in enumerate(zip(distances[0], indices[0])):
                    if idx < 0 or idx >= len(self._chunk_ids):
                        skipped_count += 1
                        continue
                    
                    # chunk_id 추출
                    if self.use_mlflow_index:
                        if hasattr(self, '_chunk_ids') and self._chunk_ids and idx < len(self._chunk_ids):
                            chunk_id = self._chunk_ids[idx]
                        else:
                            chunk_id = idx
                    else:
                        chunk_id = self._chunk_ids[idx] if hasattr(self, '_chunk_ids') and self._chunk_ids else idx
                    
                    candidate_chunk_ids.append(chunk_id)
                    distance_idx_map[chunk_id] = (distance, idx, i)
                
                # Phase 2 최적화: 메타데이터 사전 로딩 - 모든 후보 chunk_id의 메타데이터를 배치로 조회
                chunk_ids_to_fetch = []
                for chunk_id in candidate_chunk_ids:
                    if chunk_id not in self._chunk_metadata:
                        chunk_ids_to_fetch.append(chunk_id)
                    elif source_types or embedding_version_id is not None:
                        chunk_meta = self._chunk_metadata.get(chunk_id, {})
                        if source_types and not chunk_meta.get('source_type'):
                            chunk_ids_to_fetch.append(chunk_id)
                        if embedding_version_id is not None and 'embedding_version_id' not in chunk_meta:
                            chunk_ids_to_fetch.append(chunk_id)
                
                # 배치로 메타데이터 조회
                if chunk_ids_to_fetch:
                    try:
                        with self._get_connection_context() as conn_batch:
                            batch_metadata = self._batch_load_chunk_metadata(conn_batch, chunk_ids_to_fetch)
                            
                            # 조회된 메타데이터를 캐시에 저장
                            for chunk_id, metadata in batch_metadata.items():
                                if chunk_id not in self._chunk_metadata:
                                    self._chunk_metadata[chunk_id] = {}
                                self._chunk_metadata[chunk_id].update(metadata)
                                
                                # embedding_version_id가 None이면 활성 버전 사용
                                if self._chunk_metadata[chunk_id].get('embedding_version_id') is None:
                                    active_version_id = self._get_active_embedding_version_id()
                                    if active_version_id:
                                        self._chunk_metadata[chunk_id]['embedding_version_id'] = active_version_id
                    except Exception as e:
                        self.logger.debug(f"Batch metadata fetch failed: {e}")
                
                # 필터링 전 타입 분포 샘플링 (성능 최적화: 샘플 크기 감소 및 배치 처리)
                if source_types and len(candidate_chunk_ids) > 50:
                    sample_size = min(50, len(candidate_chunk_ids))
                    sample_types = {}
                    sample_checked = 0
                    
                    for chunk_id in candidate_chunk_ids[:sample_size]:
                        chunk_meta = self._chunk_metadata.get(chunk_id, {})
                        sample_type = chunk_meta.get('source_type')
                        
                        if sample_type:
                            sample_types[sample_type] = sample_types.get(sample_type, 0) + 1
                            sample_checked += 1
                    
                    # 샘플에서 요청한 타입이 있는지 확인
                    requested_types_found = sum(1 for st in source_types if sample_types.get(st, 0) > 0)
                    if requested_types_found == 0 and sample_checked > 0:
                        self.logger.warning(
                            f"⚠️  Requested source_types {source_types} not found in sample ({sample_checked} chunks). "
                            f"Sample distribution: {dict(sample_types)}. Relaxing source_type filter."
                        )
                        source_types = None
                        source_types_relaxed = True
                    elif requested_types_found > 0:
                        requested_ratio = sum(sample_types.get(st, 0) for st in source_types) / sample_checked
                        # 우선순위 2 개선: 필터 완화 임계값 완화 (5% → 10%)
                        if requested_ratio < 0.10:  # 10% 미만이면 필터 완화
                            self.logger.warning(
                                f"⚠️  Requested source_types {source_types} are very rare in sample ({requested_ratio:.1%}). "
                                f"Sample distribution: {dict(sample_types)}. Relaxing source_type filter."
                            )
                            source_types = None
                            source_types_relaxed = True
                
                # Phase 1 최적화: 단일 루프로 필터링 및 결과 생성 (이중 루프 제거)
                for chunk_id in candidate_chunk_ids:
                    distance, idx, original_i = distance_idx_map[chunk_id]
                    
                    # chunk_id가 데이터베이스에 존재하는지 확인 (재임베딩 후 버전 불일치 방지)
                    if chunk_id not in self._chunk_metadata:
                        filtered_by_not_found += 1
                        if filtered_by_not_found <= 5:
                            self.logger.warning(f"⚠️  chunk_id={chunk_id} not found in database (FAISS index may be built with different version)")
                        continue
                    
                    chunk_meta = self._chunk_metadata.get(chunk_id, {})
                    
                    # source_types 필터링 (FAISS 인덱스 사용 시 사전 필터링)
                    if source_types:
                        chunk_source_type = chunk_meta.get('source_type')
                        
                        # source_type이 source_types에 없으면 건너뛰기
                        # 단, 필터링 비율이 높으면 필터 완화
                        if chunk_source_type and chunk_source_type not in source_types:
                            filtered_by_source += 1
                            
                            # 우선순위 2 개선: 필터링 중간에 비율 확인 및 필터 완화 (더 적극적으로 완화)
                            processed_count = len(similarities) + filtered_by_source + filtered_by_version + filtered_by_not_found
                            if processed_count > 10 and not source_types_relaxed:  # 20 → 10으로 낮춤 (더 빠른 완화)
                                current_filter_ratio = filtered_by_source / processed_count if processed_count > 0 else 0
                                # 필터링 비율 임계값 완화: 70% → 50%
                                if current_filter_ratio >= 0.5:  # 50% 이상 필터링 시 완화
                                    self.logger.warning(
                                        f"⚠️  High source_type filtering rate detected: {current_filter_ratio:.1%} "
                                        f"({filtered_by_source}/{processed_count}). "
                                        f"Requested types: {original_source_types}, current chunk type: {chunk_source_type}. "
                                        f"Relaxing source_type filter to ensure minimum results."
                                    )
                                    source_types = None
                                    source_types_relaxed = True
                                    # 필터가 완화되었으므로 이 chunk는 통과
                                elif current_filter_ratio >= 0.3 and len(similarities) == 0:
                                    # 30% 이상 필터링되고 아직 결과가 없으면 경고 (50% → 30%로 완화)
                                    self.logger.warning(
                                        f"⚠️  Moderate source_type filtering rate: {current_filter_ratio:.1%} "
                                        f"({filtered_by_source}/{processed_count}). "
                                        f"Requested types: {original_source_types}, current chunk type: {chunk_source_type}."
                                    )
                                    # 결과가 없으면 필터 완화
                                    if len(similarities) == 0:
                                        self.logger.warning("   No results yet, relaxing source_type filter.")
                                        source_types = None
                                        source_types_relaxed = True
                            
                            # 필터가 완화되지 않았으면 건너뛰기
                            if source_types:
                                continue
                    
                    # 우선순위 2 개선: embedding_version_id 필터링 (더 완화된 로직)
                    if embedding_version_id is not None:
                        chunk_version_id = chunk_meta.get('embedding_version_id')
                        
                        # _chunk_metadata에 없으면 DB에서 조회 (이미 배치 조회했으므로 대부분 캐시에 있음)
                        if chunk_version_id is None:
                            try:
                                with self._get_connection_context() as conn_temp:
                                    cursor_temp = conn_temp.cursor()
                                    cursor_temp.execute(
                                        "SELECT embedding_version FROM precedent_chunks WHERE id = %s",
                                        (chunk_id,)
                                    )
                                    row_temp = cursor_temp.fetchone()
                                    if row_temp:
                                        chunk_version_id = row_temp.get('embedding_version_id') if hasattr(row_temp, 'get') else (row_temp[0] if len(row_temp) > 0 else None)
                                        # NULL인 경우 활성 버전 사용
                                        if chunk_version_id is None:
                                            active_version_id = self._get_active_embedding_version_id()
                                            if active_version_id:
                                                chunk_version_id = active_version_id
                                        # 메타데이터에 저장
                                        if chunk_id not in self._chunk_metadata:
                                            self._chunk_metadata[chunk_id] = {}
                                        self._chunk_metadata[chunk_id]['embedding_version_id'] = chunk_version_id
                            except Exception as e:
                                self.logger.debug(f"Failed to get embedding_version_id for chunk_id={chunk_id}: {e}")
                                # 예외 발생 시 활성 버전 사용
                                active_version_id = self._get_active_embedding_version_id()
                                if active_version_id:
                                    chunk_version_id = active_version_id
                        
                        # 우선순위 2 개선: 버전 필터링 완화 (버전 불일치 시에도 관련성 높은 문서는 포함)
                        if chunk_version_id != embedding_version_id:
                            # 활성 버전 확인
                            active_version_id = self._get_active_embedding_version_id()
                            
                            # 활성 버전이 요청 버전과 일치하면 허용 (버전이 다르거나 없어도)
                            if active_version_id == embedding_version_id:
                                # 활성 버전과 요청 버전이 일치하면 버전 필터링 완화
                                if chunk_version_id is None:
                                    self.logger.debug(f"chunk_id={chunk_id} has no version_id, allowing (active version {active_version_id} matches requested {embedding_version_id})")
                                elif is_indexivfpq:
                                    self.logger.debug(f"IndexIVFPQ: chunk_id={chunk_id} version_id={chunk_version_id} != {embedding_version_id}, allowing (active version matches)")
                                else:
                                    # 일반 인덱스도 활성 버전이 일치하면 허용
                                    self.logger.debug(f"chunk_id={chunk_id} version_id={chunk_version_id} != {embedding_version_id}, allowing (active version {active_version_id} matches requested)")
                            elif chunk_version_id is None:
                                # chunk_version_id가 None이고 활성 버전도 일치하지 않으면
                                if is_indexivfpq:
                                    # IndexIVFPQ 인덱스 사용 시: 버전이 없어도 허용
                                    self.logger.debug(f"IndexIVFPQ: chunk_id={chunk_id} has no version_id, allowing (filtering relaxed)")
                                else:
                                    # 일반 인덱스는 활성 버전 사용
                                    if active_version_id:
                                        chunk_version_id = active_version_id
                                        self.logger.debug(f"chunk_id={chunk_id} has no version_id, using active version {active_version_id}")
                                    else:
                                        filtered_by_version += 1
                                        self.logger.debug(f"Filtered out chunk_id={chunk_id}: no version_id and no active version")
                                        continue
                            elif is_indexivfpq:
                                # IndexIVFPQ 인덱스 사용 시: 버전이 매칭되지 않아도 허용
                                self.logger.debug(f"IndexIVFPQ: chunk_id={chunk_id} version_id={chunk_version_id} != {embedding_version_id}, allowing (filtering relaxed)")
                            else:
                                # 우선순위 2 개선: 일반 인덱스도 결과가 부족하면 버전 필터링 완화
                                # 결과가 없거나 매우 적으면 버전 불일치 문서도 포함
                                if len(similarities) < 3:  # 결과가 3개 미만이면 버전 필터링 완화
                                    self.logger.debug(
                                        f"⚠️ [VERSION FILTER] Results insufficient ({len(similarities)}), "
                                        f"allowing chunk_id={chunk_id} with version_id={chunk_version_id} != {embedding_version_id}"
                                    )
                                else:
                                    # 결과가 충분하면 엄격한 필터링
                                    filtered_by_version += 1
                                    self.logger.debug(f"Filtered out chunk_id={chunk_id}: version_id={chunk_version_id} != {embedding_version_id}")
                                    continue
                    
                    similarity = self._calculate_similarity_from_distance(distance)

                    if similarity >= similarity_threshold:
                        # hybrid 모드에서는 pgvector 결과와 중복 제거
                        if self.vector_search_method == 'hybrid':
                            # 이미 pgvector 결과에 있는지 확인
                            existing_chunk_ids = {cid for cid, _ in similarities}
                            if chunk_id not in existing_chunk_ids:
                                similarities.append((chunk_id, similarity))
                            else:
                                # 중복이면 더 높은 유사도로 업데이트
                                for i, (cid, sim) in enumerate(similarities):
                                    if cid == chunk_id and similarity > sim:
                                        similarities[i] = (chunk_id, similarity)
                                        break
                        else:
                            similarities.append((chunk_id, similarity))
                        self.logger.debug(f"Added to similarities: chunk_id={chunk_id}, similarity={similarity:.4f}, version_id={chunk_version_id if embedding_version_id is not None else 'N/A'}")
                    else:
                        filtered_by_threshold += 1

                # 유사도 기준 정렬
                similarities.sort(key=lambda x: x[1], reverse=True)  # similarity는 인덱스 1
                
                # hybrid 모드에서 pgvector와 FAISS 결과 병합
                if self.vector_search_method == 'hybrid' and len(similarities) > 0:
                    self.logger.info(f"🔍 [HYBRID] Merged {len(similarities)} results from FAISS with pgvector results")
                
                step_times['faiss_search'] = time.time() - search_start
                
                # 필터링 통계 로깅 및 source_type 필터 완화
                if source_types_relaxed:
                    self.logger.info(
                        f"✅ Source_type filter was relaxed during filtering. "
                        f"Original types: {original_source_types}, "
                        f"Final results: {len(similarities)} (from {len(indices[0])} FAISS results)"
                    )
                
                if len(similarities) == 0 and len(indices[0]) > 0:
                    self.logger.warning(f"⚠️  No results after filtering! FAISS returned {len(indices[0])} results but all were filtered out.")
                    self.logger.warning(f"   Filtering stats: skipped={skipped_count}, by_version={filtered_by_version}, by_source={filtered_by_source}, by_threshold={filtered_by_threshold}, not_found={filtered_by_not_found}")
                    
                    # source_type 필터링으로 인한 결과 손실이 큰 경우
                    if filtered_by_source > 0 and filtered_by_source >= len(indices[0]) * 0.8:
                        if source_types_relaxed:
                            self.logger.warning(
                                f"⚠️  Filter was relaxed but still no results. "
                                f"Most results filtered by source_type ({filtered_by_source}/{len(indices[0])}). "
                                f"Original requested types: {original_source_types}."
                            )
                        else:
                            self.logger.warning(
                                f"⚠️  Most results filtered by source_type ({filtered_by_source}/{len(indices[0])}). "
                                f"Requested types: {original_source_types if original_source_types else source_types}. "
                                f"Consider relaxing source_type filter at higher level."
                            )
                    
                    if filtered_by_not_found > 0:
                        self.logger.warning(f"   ⚠️  {filtered_by_not_found} chunk_ids not found in database - FAISS index may be built with different version. Please rebuild the index.")
                    self.logger.warning(f"   Possible causes: similarity_threshold too high ({similarity_threshold:.3f}) or embedding_version_id mismatch (requested={embedding_version_id})")
                
                # IndexIVFPQ 인덱스 사용 시 상세 로깅
                if is_indexivfpq:
                    self.logger.info(f"🔍 IndexIVFPQ search: {len(indices[0])} FAISS results → {len(similarities)} after filtering (threshold={similarity_threshold:.3f})")
                    if len(similarities) > 0:
                        avg_sim = sum(s[1] for s in similarities) / len(similarities)
                        max_sim = max(s[1] for s in similarities)
                        min_sim = min(s[1] for s in similarities)
                        self.logger.info(f"   Similarity scores: avg={avg_sim:.3f}, max={max_sim:.3f}, min={min_sim:.3f}")
                else:
                    self.logger.debug(f"⏱️  FAISS search: {step_times['faiss_search']:.3f}s, {len(similarities)} results")

            else:
                # 기존 방식 (전체 벡터 로드 및 선형 검색)
                # MLflow 인덱스 전용이므로 자동 빌드 제거

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
            
            # 🔥 개선: 불필요한 연결 제거 (로깅만 수행하므로 연결 불필요)
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
            if chunk_ids_for_batch:
                batch_start = time.time()
                # 연결 컨텍스트를 사용하여 메타데이터 조회
                with self._get_connection_context() as conn:
                    batch_chunk_metadata = self._batch_load_chunk_metadata(conn, chunk_ids_for_batch)
                batch_time = time.time() - batch_start
                self.logger.debug(f"Batch loaded metadata for {len(batch_chunk_metadata)} chunks in {batch_time:.3f}s")
                
                # 존재하지 않는 chunk_id 확인 및 재시도 로직
                missing_chunk_ids = [cid for cid in chunk_ids_for_batch if cid not in batch_chunk_metadata]
                if missing_chunk_ids:
                    self.logger.warning(f"⚠️  {len(missing_chunk_ids)} chunk_ids not found in database (out of {len(chunk_ids_for_batch)} requested)")
                    if len(missing_chunk_ids) <= 10:
                        self.logger.debug(f"Missing chunk_ids: {missing_chunk_ids}")
                    else:
                        self.logger.debug(f"Missing chunk_ids (first 10): {missing_chunk_ids[:10]}")
                    
                    # 참고: text_chunks 테이블은 PostgreSQL 환경에서 사용되지 않으므로 제거됨
            
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
            if source_items_for_batch:
                batch_start = time.time()
                # 연결 컨텍스트를 사용하여 소스 메타데이터 조회
                with self._get_connection_context() as conn:
                    batch_source_metadata = self._batch_load_source_metadata(conn, source_items_for_batch)
                batch_time = time.time() - batch_start
                self.logger.debug(f"Batch loaded source metadata for {len(batch_source_metadata)} source items in {batch_time:.3f}s")
            
            for similarity_item in similarities[:k]:
                chunk_id, score = similarity_item
                self.logger.debug(f"Processing result: chunk_id={chunk_id}, score={score:.4f}")
                
                # chunk_id가 배치 메타데이터에 없는 경우 건너뛰기 (데이터베이스에 존재하지 않음)
                if chunk_id not in batch_chunk_metadata and chunk_id not in self._chunk_metadata:
                    self.logger.debug(f"Skipping chunk_id={chunk_id}: not found in database")
                    continue
                
                # metadata 변수 초기화
                metadata = None
                
                # MLflow 인덱스 사용 시 메타데이터는 _chunk_metadata에 이미 로드됨
                if self.use_mlflow_index and chunk_id in self._chunk_metadata:
                    metadata = self._chunk_metadata[chunk_id]
                    self.logger.debug(f"Found metadata for chunk_id={chunk_id}")
                    text = metadata.get('content', '') or metadata.get('text', '')
                    # 🔥 source_type 제거: type 필드만 사용
                    doc_type = metadata.get('type') or metadata.get('source_type', '')
                    
                    # type이 없으면 메타데이터에서 추론
                    if not doc_type:
                        if metadata.get('case_id') or metadata.get('case_number') or metadata.get('doc_id'):
                            doc_type = 'precedent_content'  # 🔥 레거시: case_paragraph → precedent_content
                        elif metadata.get('law_id') or metadata.get('law_name') or metadata.get('article_number'):
                            doc_type = 'statute_article'
                        elif metadata.get('decision_id') or metadata.get('org'):
                            doc_type = 'decision_paragraph'
                        elif metadata.get('interpretation_id'):
                            doc_type = 'interpretation_paragraph'
                        
                        # source_meta에 모든 메타데이터 포함 (type으로 저장)
                        source_meta = metadata.copy()
                        source_meta['type'] = doc_type
                        # source_id 추출: 실제 DB ID를 찾기 위해 precedent_chunks에서 조회
                        # 외부 인덱스 메타데이터에는 case_number, doc_id 등이 있을 수 있지만 실제 DB ID는 다를 수 있음
                        potential_source_id = metadata.get('case_id') or metadata.get('law_id') or metadata.get('doc_id', '')
                        
                        # DB에서 실제 source_id 찾기 (precedent_chunks 테이블 사용)
                        actual_source_id = None
                        if doc_type and potential_source_id:
                            try:
                                # 연결 컨텍스트를 사용하여 DB 조회
                                with self._get_connection_context() as conn:
                                    # precedent_chunks에서 metadata로 조회
                                    if doc_type in ['case_paragraph', 'precedent_content']:  # 🔥 레거시 지원
                                        # case_number나 doc_id로 조회
                                        case_number = metadata.get('case_number') or metadata.get('doc_id')
                                        if case_number:
                                            cursor = conn.cursor()
                                            cursor.execute(
                                                "SELECT DISTINCT precedent_content_id FROM precedent_chunks WHERE metadata::text LIKE %s OR metadata::text LIKE %s LIMIT 1",
                                                (f'%{case_number}%', f'%doc_id%{case_number}%')
                                            )
                                            row = cursor.fetchone()
                                            if row:
                                                actual_source_id = row[0] if isinstance(row, tuple) else row['source_id']
                                    elif doc_type == 'statute_article':
                                        # law_id와 article_number로 조회
                                        law_id = metadata.get('law_id')
                                        article_no = metadata.get('article_number') or metadata.get('article_no')
                                        if law_id and article_no:
                                            cursor = conn.cursor()
                                            cursor.execute(
                                                # precedent_chunks는 판례만 저장하므로 statute_article 조회는 지원하지 않음
                                                # 필요시 별도 로직 구현
                                                None
                                            )
                                            row = cursor.fetchone()
                                            if row:
                                                actual_source_id = row[0] if isinstance(row, tuple) else row['source_id']
                                
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
                        try:
                            with self._get_connection_context() as conn:
                                cursor = conn.cursor()
                                cursor.execute(
                                    "SELECT chunk_index, metadata, embedding_version FROM precedent_chunks WHERE id = %s",
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
                            'type': doc_type,  # 🔥 source_type 제거: type으로 통일
                            'source_id': actual_source_id if actual_source_id is not None else potential_source_id,
                            'text': text,
                            **chunking_meta,
                            **metadata
                        }
                    else:
                        # MLflow 인덱스를 사용하지 않거나 _chunk_metadata에 없는 경우 - DB에서 직접 조회
                        self.logger.debug(f"Metadata not available in _chunk_metadata, loading from DB for chunk_id={chunk_id}")
                        if not chunk_id:
                            self.logger.warning(f"⚠️  Cannot load chunk_id={chunk_id} from DB (chunk_id={chunk_id})")
                            continue
                        
                        try:
                            with self._get_connection_context() as conn:
                                cursor = conn.cursor()
                                cursor.execute(
                                    "SELECT precedent_content_id, chunk_index, chunk_content, metadata, embedding_version FROM precedent_chunks WHERE id = %s",
                                    (chunk_id,)
                                )
                                row = cursor.fetchone()
                            if row:
                                # PostgreSQL의 경우 dict-like row 또는 tuple 반환
                                if hasattr(row, 'keys'):
                                    text = row.get('chunk_content') or ''
                                    precedent_content_id = row.get('precedent_content_id')
                                    chunk_index = row.get('chunk_index')
                                    metadata = row.get('metadata')
                                    embedding_version = row.get('embedding_version')
                                else:
                                    text = row[2] if len(row) > 2 else ''
                                    precedent_content_id = row[0] if len(row) > 0 else None
                                    chunk_index = row[1] if len(row) > 1 else None
                                    metadata = row[3] if len(row) > 3 else None
                                    embedding_version = row[4] if len(row) > 4 else None
                                
                                source_type = 'precedent_content'
                                source_id = precedent_content_id
                                self.logger.debug(f"Loaded from DB: chunk_id={chunk_id}, source_type={source_type}")
                                
                                # _chunk_metadata에 저장
                                version_id = embedding_version
                                if version_id is None:
                                    active_version_id = self._get_active_embedding_version_id()
                                    if active_version_id:
                                        version_id = active_version_id
                                
                                self._chunk_metadata[chunk_id] = {
                                    'source_type': source_type,
                                    'source_id': source_id,
                                    'text': text,
                                    'chunk_index': chunk_index,
                                    'metadata': metadata,
                                    'embedding_version_id': version_id
                                }
                            else:
                                self.logger.warning(f"⚠️  chunk_id={chunk_id} not found in database")
                                continue
                        except Exception as e:
                            self.logger.warning(f"⚠️  Failed to load chunk_id={chunk_id} from DB: {e}")
                            continue
                
                if chunk_id not in self._chunk_metadata:
                    # 메타데이터가 없으면 DB에서 직접 조회 (precedent_chunks에서 가져오기)
                    with self._get_connection_context() as conn:
                        cursor = conn.cursor()
                        cursor.execute(
                            "SELECT precedent_content_id, chunk_index, chunk_content, metadata, embedding_version FROM precedent_chunks WHERE id = %s",
                            (chunk_id,)
                        )
                        row = cursor.fetchone()
                        if row:
                            # PostgreSQL의 경우 dict-like row 또는 tuple 반환
                            if hasattr(row, 'keys'):
                                text_content = row.get('chunk_content') or ""
                                precedent_content_id = row.get('precedent_content_id')
                                chunk_index = row.get('chunk_index')
                                metadata = row.get('metadata')
                                embedding_version = row.get('embedding_version')
                            else:
                                text_content = row[2] if len(row) > 2 and row[2] else ""
                                precedent_content_id = row[0] if len(row) > 0 else None
                                chunk_index = row[1] if len(row) > 1 else None
                                metadata = row[3] if len(row) > 3 else None
                                embedding_version = row[4] if len(row) > 4 else None
                            
                            # text가 비어있거나 짧으면 원본 테이블에서 복원 시도
                            if not text_content or len(text_content.strip()) < 100:
                                restored_text = self._restore_text_from_precedent_content(conn, precedent_content_id)
                                if restored_text and len(restored_text.strip()) > len(text_content.strip()):
                                    text_content = restored_text
                                    self.logger.info(f"Restored longer text for chunk_id={chunk_id} (length: {len(text_content)} chars)")
                            
                            # embedding_version이 NULL인 경우 활성 버전 사용
                            version_id = embedding_version
                            if version_id is None:
                                active_version_id = self._get_active_embedding_version_id()
                                if active_version_id:
                                    version_id = active_version_id
                                    self.logger.debug(f"Using active version {version_id} for chunk_id={chunk_id} (precedent_chunks.embedding_version is NULL)")
                            
                            self._chunk_metadata[chunk_id] = {
                                'source_type': 'precedent_content',
                                'source_id': precedent_content_id,
                                'text': text_content,
                                'chunk_index': chunk_index,
                                'metadata': metadata,
                                'embedding_version_id': version_id
                            }

                chunk_metadata = self._chunk_metadata.get(chunk_id, {})
                # chunk_metadata가 비어있으면 DB에서 로드 (precedent_chunks)
                if not chunk_metadata:
                    try:
                        with self._get_connection_context() as conn:
                            cursor = conn.cursor()
                            cursor.execute(
                                "SELECT precedent_content_id, chunk_index, chunk_content, metadata, embedding_version FROM precedent_chunks WHERE id = %s",
                                (chunk_id,)
                            )
                            row = cursor.fetchone()
                            if row:
                                # PostgreSQL의 경우 dict-like row 또는 tuple 반환
                                if hasattr(row, 'keys'):
                                    version_id = row.get('embedding_version')
                                    metadata_val = row.get('metadata')
                                    precedent_content_id = row.get('precedent_content_id')
                                    chunk_index = row.get('chunk_index')
                                    chunk_content = row.get('chunk_content')
                                else:
                                    version_id = row[4] if len(row) > 4 else None
                                    metadata_val = row[3] if len(row) > 3 else None
                                    precedent_content_id = row[0] if len(row) > 0 else None
                                    chunk_index = row[1] if len(row) > 1 else None
                                    chunk_content = row[2] if len(row) > 2 else None
                                
                                if version_id is None:
                                    active_version_id = self._get_active_embedding_version_id()
                                    if active_version_id:
                                        version_id = active_version_id
                                
                                # precedent_chunks.metadata 컬럼에서 메타데이터 로드 (이미 JSONB)
                                chunk_meta_json = None
                                if metadata_val:
                                    if isinstance(metadata_val, dict):
                                        chunk_meta_json = metadata_val
                                    else:
                                        try:
                                            import json
                                            chunk_meta_json = json.loads(metadata_val) if isinstance(metadata_val, str) else metadata_val
                                        except Exception as e:
                                            self.logger.debug(f"Failed to parse metadata JSON for chunk_id={chunk_id}: {e}")
                                
                                chunk_metadata = {
                                    'source_type': 'precedent_content',
                                    'source_id': precedent_content_id,
                                    'text': chunk_content if chunk_content else '',
                                    'chunk_index': chunk_index,
                                    'embedding_version_id': version_id
                                }
                                
                                # precedent_chunks.metadata의 메타데이터를 chunk_metadata에 병합
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
                with self._get_connection_context() as conn:
                    text = self._ensure_text_content(
                        conn, chunk_id, text, source_type, source_id, full_metadata
                    )

                # 소스별 상세 메타데이터 조회 (배치 조회 결과 사용)
                if source_type and source_id:
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
                            with self._get_connection_context() as conn:
                                source_meta = self._get_source_metadata(conn, source_type, source_id_for_query)
                    
                    # precedent_chunks.metadata의 메타데이터를 우선적으로 사용 (소스 테이블 메타데이터와 병합)
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
                        elif source_type in ["case_paragraph", "precedent_content"]:  # 🔥 레거시 지원
                            # precedent_content: doc_id만 필수, casenames와 court는 선택적
                            if not source_meta.get("doc_id"):
                                needs_reload = True
                        elif source_type == "decision_paragraph":
                            if not source_meta.get("org") or not source_meta.get("doc_id"):
                                needs_reload = True
                        elif source_type == "interpretation_paragraph":
                            # interpretation_paragraph: interpretation_id만 필수, org와 doc_id는 선택적
                            if not source_meta.get("interpretation_id"):
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
                    
                    # 타입별 최소 길이 차등 적용 (텍스트 품질 개선: 최소 길이 보장)
                    if source_type == 'statute_article':
                        min_text_length = 30
                    elif source_type in ['case_paragraph', 'precedent_content', 'decision_paragraph']:  # 🔥 레거시 지원
                        min_text_length = 100  # 텍스트 품질 개선: 5자 → 100자로 증가
                    else:
                        min_text_length = 100  # 텍스트 품질 개선: 50자 → 100자로 증가
                    
                    # 텍스트 길이 보장 로직 강화 (추가 개선: 인접 청크에서 텍스트 복원)
                    original_text_length = len(text.strip()) if text else 0
                    if text and original_text_length < min_text_length:
                        # 1차: 원본 소스에서 복원 시도
                        restored_text = self._restore_text_from_source(conn, source_type, source_id)
                        if restored_text and len(restored_text.strip()) > original_text_length:
                            text = restored_text
                            self.logger.debug(f"Extended text for chunk_id={chunk_id} from {original_text_length} to {len(text)} chars")
                        
                        # 2차: 여전히 짧으면 인접 청크에서 텍스트 가져오기 시도
                        if text and len(text.strip()) < min_text_length:
                            # 인접한 청크에서 텍스트 복원 시도
                            try:
                                adjacent_text = self._restore_text_from_source(conn, source_type, source_id)
                                if adjacent_text and len(adjacent_text.strip()) > len(text.strip()):
                                    text = adjacent_text
                                    self.logger.debug(f"Extended text for chunk_id={chunk_id} from source to {len(text)} chars")
                            except Exception as e:
                                self.logger.debug(f"Could not restore text from adjacent chunks for chunk_id={chunk_id}: {e}")
                    
                    # 최소 길이 미만이면 건너뛰기 (텍스트 품질 개선 강화)
                    # 단, statute_article은 30자 이상이면 허용 (법령 조문은 짧을 수 있음)
                    final_text_length = len(text.strip()) if text else 0
                    if final_text_length < min_text_length:
                        # statute_article은 30자 이상이면 허용
                        if source_type == 'statute_article' and final_text_length >= 30:
                            pass  # 허용
                        else:
                            self.logger.debug(f"Skipping chunk {chunk_id}: text too short ({final_text_length} < {min_text_length})")
                            continue
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
                        cursor = conn.cursor()
                        cursor.execute(
                            "SELECT embedding_version FROM precedent_chunks WHERE id = %s",
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
                                    self.logger.debug(f"Using active version {active_version_id} for chunk_id={chunk_id} (precedent_chunks.embedding_version is NULL)")
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
                        # 🔥 개선: _chunk_vectors에서 먼저 확인 (이미 로드된 경우)
                        if hasattr(self, '_chunk_vectors') and chunk_id in self._chunk_vectors:
                            doc_vec = self._chunk_vectors[chunk_id]
                        else:
                            # 🔥 개선: pgvector 사용 시 올바른 테이블에서 벡터 로드
                            # source_type에 따라 올바른 테이블 선택
                            if self._db_adapter:
                                with self._db_adapter.get_connection_context() as conn:
                                    cursor = conn.cursor()
                                    
                                    # source_type에 따라 올바른 테이블과 컬럼 선택
                                    if source_type == 'statute_article':
                                        # statute_embeddings 테이블에서 article_id로 조회
                                        cursor.execute(
                                            """
                                            SELECT embedding_vector 
                                            FROM statute_embeddings 
                                            WHERE article_id = %s 
                                            AND embedding_vector IS NOT NULL 
                                            LIMIT 1
                                            """,
                                            (chunk_id,)
                                        )
                                    else:
                                        # precedent_chunks 테이블에서 id로 조회 (레거시)
                                        cursor.execute(
                                            """
                                            SELECT embedding_vector 
                                            FROM precedent_chunks 
                                            WHERE id = %s 
                                            AND embedding_vector IS NOT NULL 
                                            LIMIT 1
                                            """,
                                            (chunk_id,)
                                        )
                                    
                                    row = cursor.fetchone()
                                    if row:
                                        embedding_vector = row[0] if isinstance(row, tuple) else row.get('embedding_vector')
                                        if embedding_vector:
                                            # 벡터 파싱
                                            try:
                                                if isinstance(embedding_vector, (list, tuple)):
                                                    doc_vec = np.array(embedding_vector, dtype=np.float32)
                                                elif hasattr(embedding_vector, 'tolist'):
                                                    doc_vec = np.array(embedding_vector.tolist(), dtype=np.float32)
                                                else:
                                                    doc_vec = np.array(embedding_vector, dtype=np.float32)
                                                
                                                # _chunk_vectors에 캐시
                                                if not hasattr(self, '_chunk_vectors'):
                                                    self._chunk_vectors = {}
                                                self._chunk_vectors[chunk_id] = doc_vec
                                            except Exception as parse_error:
                                                self.logger.debug(f"Failed to parse vector for chunk_id {chunk_id}: {parse_error}")
                                                doc_vec = None
                                        else:
                                            doc_vec = None
                                    else:
                                        doc_vec = None
                            else:
                                raise RuntimeError("DatabaseAdapter is required. PostgreSQL database must be configured via DATABASE_URL.")
                        
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
                    if source_type in ["case_paragraph", "precedent_content"]:  # 🔥 레거시 지원
                        # precedent_content: doc_id만 필수, casenames와 court는 선택적
                        if not source_meta.get("doc_id"):
                            required_fields_missing = True
                    elif source_type == "decision_paragraph":
                        # decision_paragraph: doc_id만 필수, org는 선택적 (일부 결정례에 없을 수 있음, court 필드는 없음)
                        if not source_meta.get("doc_id"):
                            required_fields_missing = True
                    elif source_type == "statute_article":
                        if not source_meta.get("statute_name") or not source_meta.get("article_no"):
                            required_fields_missing = True
                    elif source_type == "interpretation_paragraph":
                        # interpretation_paragraph: interpretation_id만 필수, org와 doc_id는 선택적
                        if not source_meta.get("interpretation_id"):
                            required_fields_missing = True
                    
                    if required_fields_missing:
                        # 소스 테이블에서 메타데이터 재조회
                        additional_meta = self._get_source_metadata(conn, source_type, source_id)
                        if additional_meta:
                            # 누락된 필드만 보완
                            for key, value in additional_meta.items():
                                if key not in source_meta or not source_meta[key]:
                                    source_meta[key] = value
                
                # interpretation_paragraph 타입의 경우 오타 필드명 정규화 (검색 결과 생성 시)
                if source_type == "interpretation_paragraph":
                    # source_meta에서 오타 필드명 확인 및 정규화
                    typo_fields = ['interpretatiion_id', 'interpretattion_id']
                    for typo_field in typo_fields:
                        if typo_field in source_meta:
                            correct_value = source_meta[typo_field]
                            source_meta['interpretation_id'] = correct_value
                            del source_meta[typo_field]
                            self.logger.debug(f"✅ Normalized typo field {typo_field} → interpretation_id={correct_value} in source_meta for chunk_id={chunk_id}")
                
                # source_type 필드 보장 (메타데이터 완전성 개선)
                # unknown을 기본값으로 사용하지 않고, 여러 위치에서 확인
                if not source_type:
                    # 1단계: chunk_metadata에서 확인
                    source_type = chunk_metadata.get('source_type')
                    if source_type == "unknown":
                        source_type = None
                    
                    # 2단계: source_meta에서 확인
                    if not source_type:
                        source_type = source_meta.get('source_type') or source_meta.get('type')
                        if source_type == "unknown":
                            source_type = None
                    
                    # 3단계: DocumentType.from_metadata로 추론
                    if not source_type:
                        try:
                            from lawfirm_langgraph.core.workflow.constants.document_types import DocumentType
                            doc_for_inference = {
                                "metadata": {**chunk_metadata, **source_meta}
                            }
                            doc_type_enum = DocumentType.from_metadata(doc_for_inference)
                            if doc_type_enum != DocumentType.UNKNOWN:
                                source_type = doc_type_enum.value
                        except (ImportError, AttributeError):
                            pass
                    
                    # 최후의 수단으로만 "unknown" 사용
                    if not source_type:
                        source_type = 'unknown'
                
                # ccourt 필드 보장 (court 필드에서 가져오기) - 메타데이터 완전성 개선 강화
                court_value = source_meta.get("court")
                ccourt_value = source_meta.get("ccourt") or court_value
                # court가 없으면 DB에서 직접 조회 시도 (precedent_content의 경우)
                if not ccourt_value and source_type in ["case_paragraph", "precedent_content"] and source_id and conn:  # 🔥 레거시 지원
                    try:
                        # precedent_chunks를 통해 precedents 조회
                        cursor_court = conn.cursor()
                        cursor_court.execute(
                            """
                            SELECT p.court_name 
                            FROM precedent_chunks pc
                            JOIN precedent_contents pcc ON pc.precedent_content_id = pcc.id
                            JOIN precedents p ON pcc.precedent_id = p.id
                            WHERE pc.id = %s
                            """,
                            (source_id,)
                        )
                        row_court = cursor_court.fetchone()
                        if row_court:
                            court_val = row_court[0] if isinstance(row_court, tuple) else row_court.get("court_name")
                            if court_val:
                                ccourt_value = court_val
                                court_value = ccourt_value
                        else:
                            # precedent_chunks 조회 실패 시 precedents 테이블에서 직접 조회 (precedent_content_id로)
                            cursor_court = conn.cursor()
                            cursor_court.execute(
                                """
                                SELECT p.court_name 
                                FROM precedent_contents pcc
                                JOIN precedents p ON pcc.precedent_id = p.id
                                WHERE pcc.id = %s
                                """,
                                (source_id,)
                            )
                            row_court = cursor_court.fetchone()
                            if row_court:
                                court_val = row_court[0] if isinstance(row_court, tuple) else row_court.get("court_name")
                                if court_val:
                                    ccourt_value = court_val
                                    court_value = ccourt_value
                    except Exception as e:
                        self.logger.debug(f"Failed to fetch court for case_paragraph {source_id}: {e}")
                
                # 검색 결과 설명 개선: 출처 정보 명확화
                source_description = source_name
                if source_type == "statute_article":
                    statute_name = source_meta.get("statute_name") or source_meta.get("law_name")
                    article_no = source_meta.get("article_no") or source_meta.get("article_number")
                    if statute_name and article_no:
                        source_description = f"{statute_name} {article_no}"
                elif source_type in ["case_paragraph", "precedent_content"]:  # 🔥 레거시 지원
                    casenames = source_meta.get("casenames")
                    doc_id = source_meta.get("doc_id")
                    if casenames and doc_id:
                        source_description = f"{casenames} ({doc_id})"
                    elif casenames:
                        source_description = casenames
                elif source_type == "interpretation_paragraph":
                    title = source_meta.get("title")
                    org = source_meta.get("org")
                    if title and org:
                        source_description = f"{org} {title}"
                    elif title:
                        source_description = title
                
                result = {
                    "id": f"chunk_{chunk_id}",
                    "text": text,
                    "content": text,  # content 필드 보장
                    "score": normalize_score(float(score)),
                    "similarity": normalize_score(float(score)),
                    "direct_similarity": direct_similarity,  # 직접 계산된 유사도 추가
                    "type": source_type,
                    "source_type": source_type,  # source_type 필드 명시적 추가 (메타데이터 완전성 개선)
                    "source": source_name,
                    "source_description": source_description,  # 검색 결과 설명 개선: 명확한 출처 정보
                    "source_url": source_url,  # URL 필드 추가
                    "embedding_version_id": result_embedding_version_id,  # 버전 정보 추가
                    # 최상위 필드에 상세 정보 추가 (answer_formatter에서 쉽게 접근)
                    "statute_name": source_meta.get("statute_name") if source_type == "statute_article" else None,
                    "law_name": source_meta.get("statute_name") if source_type == "statute_article" else None,
                    "article_no": source_meta.get("article_no") if source_type == "statute_article" else None,
                    "article_number": source_meta.get("article_no") if source_type == "statute_article" else None,
                    "clause_no": source_meta.get("clause_no") if source_type == "statute_article" else None,
                    "item_no": source_meta.get("item_no") if source_type == "statute_article" else None,
                    "court": court_value if source_type in ["case_paragraph", "precedent_content"] else None,  # 🔥 레거시 지원
                    "ccourt": ccourt_value if source_type in ["case_paragraph", "precedent_content"] else None,  # ccourt 필드 추가 (메타데이터 완전성 개선 - 항상 포함)
                    "doc_id": source_meta.get("doc_id") if source_type in ["case_paragraph", "precedent_content", "decision_paragraph", "interpretation_paragraph"] else None,  # 🔥 레거시 지원
                    "casenames": source_meta.get("casenames") if source_type in ["case_paragraph", "precedent_content"] else None,  # 🔥 레거시 지원
                    "org": source_meta.get("org") if source_type in ["decision_paragraph", "interpretation_paragraph"] else None,
                    "title": source_meta.get("title") if source_type == "interpretation_paragraph" else None,
                    "interpretation_id": source_meta.get("interpretation_id") if source_type == "interpretation_paragraph" else None,  # 오타 필드명 정규화 후 올바른 필드명 사용
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
                    "relevance_score": normalize_score(float(score)),
                    "hybrid_score": hybrid_score,
                    "ml_confidence": ml_confidence,
                    "quality_score": quality_score,
                    "search_type": "semantic",
                    # 검색 결과 순위 개선: 소스 타입별 가중치 추가
                    "source_type_weight": {
                        "statute_article": 1.3,
                        "interpretation_paragraph": 1.2,
                        "decision_paragraph": 1.1,
                        "case_paragraph": 1.0,  # 🔥 레거시 지원
                        "precedent_content": 1.0
                    }.get(source_type, 1.0)
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

            # 그룹별 중복 제거 (하이브리드 청킹 지원) + 소스 ID 기반 중복 제거 강화
            if deduplicate_by_group:
                seen_groups = {}
                seen_source_ids = {}  # 소스 ID 기반 중복 제거 (추가 개선)
                deduplicated_results = []
                for result in results:
                    # 우선순위 1: 소스 ID 기반 중복 제거 (최우선)
                    source_type = result.get("source_type") or result.get("type", "")
                    source_id = result.get("metadata", {}).get("source_id") if isinstance(result.get("metadata"), dict) else None
                    if source_id and source_type:
                        source_key = f"{source_type}_{source_id}"
                        if source_key in seen_source_ids:
                            # 동일 소스에서 이미 결과가 있으면 건너뛰기
                            continue
                        else:
                            seen_source_ids[source_key] = result
                    
                    # 우선순위 2: 그룹 ID 기반 중복 제거
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
                # deduplicate_by_group이 False여도 소스 ID 기반 중복 제거는 수행
                seen_source_ids = {}
                deduplicated_results = []
                for result in results:
                    source_type = result.get("source_type") or result.get("type", "")
                    source_id = result.get("metadata", {}).get("source_id") if isinstance(result.get("metadata"), dict) else None
                    if source_id and source_type:
                        source_key = f"{source_type}_{source_id}"
                        if source_key in seen_source_ids:
                            continue
                        else:
                            seen_source_ids[source_key] = result
                    deduplicated_results.append(result)
                results = deduplicated_results[:k]
            
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
                                # TASK 7: Multi-version 전략은 의도된 동작이므로 DEBUG 레벨로 변경
                                self.logger.debug(
                                    f"ℹ️  [VERSION INFO] Multi-version strategy: "
                                    f"Requested version {embedding_version_id}, "
                                    f"found {mismatch_ratio:.1f}% from other versions (expected behavior)"
                                )
                            else:
                                self.logger.debug(f"✅ All results are from expected version {embedding_version_id}")
                    else:
                        self.logger.warning("⚠️  No embedding_version_id found in results")
                else:
                    self.logger.warning("⚠️  No scores found in results")
            else:
                self.logger.warning(f"⚠️  No results found for query: {query[:50]}")
                
                # Fallback: threshold를 낮춰서 재시도 (더 낮은 threshold로 시작)
                if similarity_threshold > 0.25:
                    new_threshold = max(0.25, similarity_threshold - 0.15)
                    self.logger.info(f"🔄 Retrying with lower threshold: {similarity_threshold:.3f} → {new_threshold:.3f}")
                    fallback_results = self._search_with_threshold(
                        query, k, source_types, new_threshold,
                        min_ml_confidence, min_quality_score, filter_by_confidence,
                        chunk_size_category, deduplicate_by_group, embedding_version_id
                    )
                    if fallback_results:
                        self.logger.info(f"✅ Fallback search found {len(fallback_results)} results")
                        results = fallback_results
                
                # 여전히 결과가 없으면 source_types 필터 제거 후 재시도
                # ⚠️ 주의: source_types 필터를 제거하면 질의 의도와 다른 타입의 문서가 반환될 수 있음
                if not results and source_types:
                    self.logger.warning(
                        f"⚠️ [FALLBACK] 검색 결과가 없어 source_types 필터를 제거하고 재시도합니다. "
                        f"원래 요청된 타입: {source_types}, 이는 질의 의도와 다른 타입의 문서가 반환될 수 있습니다."
                    )
                    fallback_results = self._search_with_threshold(
                        query, k, None, max(0.20, similarity_threshold - 0.10),
                        min_ml_confidence, min_quality_score, filter_by_confidence,
                        chunk_size_category, deduplicate_by_group, embedding_version_id
                    )
                    if fallback_results:
                        # 🔥 개선: 폴백 결과의 타입을 확인하고 원래 요청된 타입과 다르면 경고
                        fallback_types = {}
                        for doc in fallback_results:
                            doc_type = doc.get("type") or doc.get("metadata", {}).get("type", "unknown")
                            fallback_types[doc_type] = fallback_types.get(doc_type, 0) + 1
                        
                        requested_types = set(source_types)
                        returned_types = set(fallback_types.keys())
                        mismatched = returned_types - requested_types
                        
                        if mismatched:
                            self.logger.warning(
                                f"⚠️ [FALLBACK TYPE MISMATCH] 요청된 타입: {requested_types}, "
                                f"반환된 타입: {returned_types}, 불일치: {mismatched}, "
                                f"타입 분포: {fallback_types}"
                            )
                        
                        self.logger.info(
                            f"✅ Fallback search (no source filter) found {len(fallback_results)} results "
                            f"(타입 분포: {fallback_types})"
                        )
                        results = fallback_results
                
                # 원인 분석 (최종적으로 결과가 없을 때만)
                if not results:
                    self._analyze_no_results_cause(query, embedding_version_id, similarity_threshold, source_types)
            
            # 검색 결과 검증 및 복원 (개선 사항 #1, #2, #3)
            if results:
                # 🔥 개선: 테이블별 버전 맵 전달 (pgvector 검색에서 추적한 버전 정보)
                table_version_map = getattr(self, '_pgvector_table_version_map', None)
                results = self._validate_and_fix_search_results(
                    results, 
                    embedding_version_id,
                    table_version_map=table_version_map
                )
            
            # 벡터 인덱스 검색 결과 로깅
            result_msg = (
                f"✅ [VECTOR INDEX SEARCH RESULT] 질의: '{query}' → "
                f"{len(results)}개 결과 반환"
            )
            print(result_msg, flush=True, file=sys.stdout)
            self.logger.debug(result_msg)
            
            # 상위 결과 상세 로깅 (최대 10개)
            if results:
                top_results = results[:10]
                top_results_msg = f"📊 [VECTOR INDEX SEARCH TOP RESULTS] 상위 {len(top_results)}개 결과:"
                print(top_results_msg, flush=True, file=sys.stdout)
                self.logger.info(top_results_msg)
                for i, result in enumerate(top_results, 1):
                    score = result.get("similarity", result.get("score", 0.0))
                    chunk_id = result.get("chunk_id") or result.get("id") or "unknown"
                    source_type = result.get("type") or result.get("source_type", "unknown")
                    source = result.get("source", "")[:100] or "unknown"
                    content_preview = (result.get("content", result.get("text", ""))[:100] or "").replace("\n", " ")
                    result_detail = (
                        f"   {i}. score={score:.3f}, chunk_id={chunk_id}, "
                        f"type={source_type}, source={source}, "
                        f"content_preview={content_preview}"
                    )
                    print(result_detail, flush=True, file=sys.stdout)
                    self.logger.info(result_detail)
            
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
                with self._get_connection_context() as conn:
                    cursor = conn.cursor()
                    cursor.execute("SELECT COUNT(*) as count FROM precedent_chunks")
                    row = cursor.fetchone()
                    total_chunks = row.get('count') if hasattr(row, 'get') else (row[0] if row else 0)
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
                                        expected_version_id: Optional[int] = None,
                                        table_version_map: Optional[Dict[str, Optional[List[int]]]] = None) -> List[Dict[str, Any]]:
        """
        검색 결과 검증 및 복원 (개선 사항 #1, #2, #3)
        
        Args:
            results: 검색 결과 리스트
            expected_version_id: 예상되는 버전 ID (기본값, 테이블별로 다를 수 있음)
            table_version_map: 테이블별 허용 버전 맵
                예: {
                    'precedent_content': [1],  # 특정 버전만 허용
                    'statute_article': [1, 2],  # 여러 버전 허용
                    'interpretation': None      # 모든 버전 허용
                }
                None 값은 모든 버전 허용을 의미
            
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
                # 0. interpretation_paragraph 타입의 경우 오타 필드명 정규화 (검증 전)
                source_type = result.get('type') or result.get('metadata', {}).get('source_type')
                if source_type == 'interpretation_paragraph':
                    typo_fields = ['interpretatiion_id', 'interpretattion_id']
                    for typo_field in typo_fields:
                        if typo_field in result:
                            correct_value = result[typo_field]
                            result['interpretation_id'] = correct_value
                            if 'metadata' not in result:
                                result['metadata'] = {}
                            result['metadata']['interpretation_id'] = correct_value
                            del result[typo_field]
                            self.logger.debug(f"✅ Normalized typo field {typo_field} → interpretation_id={correct_value} in result")
                        if 'metadata' in result and isinstance(result['metadata'], dict) and typo_field in result['metadata']:
                            correct_value = result['metadata'][typo_field]
                            result['interpretation_id'] = correct_value
                            if 'metadata' not in result:
                                result['metadata'] = {}
                            result['metadata']['interpretation_id'] = correct_value
                            del result['metadata'][typo_field]
                            self.logger.debug(f"✅ Normalized typo field {typo_field} in metadata → interpretation_id={correct_value}")
                
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
                            with self._get_connection_context() as conn:
                                cursor = conn.cursor()
                                cursor.execute(
                                    "SELECT embedding_version FROM precedent_chunks WHERE id = %s",
                                    (chunk_id,)
                                )
                                row = cursor.fetchone()
                                if row:
                                    version_id = row[0] if isinstance(row, tuple) else row.get('embedding_version_id')
                                # NULL인 경우 활성 버전 사용
                                if version_id is None:
                                    active_version_id = self._get_active_embedding_version_id()
                                    if active_version_id:
                                        version_id = active_version_id
                                        self.logger.debug(f"Using active version {version_id} for chunk_id={chunk_id}")
                                
                                if version_id:
                                    result['embedding_version_id'] = version_id
                                    if 'metadata' not in result:
                                        result['metadata'] = {}
                                    result['metadata']['embedding_version_id'] = version_id
                                    self.logger.debug(f"Restored embedding_version_id={version_id} for chunk_id={chunk_id}")
                                    restoration_stats['version_id_restored'] += 1
                            self._safe_close_connection(conn)
                        except Exception as e:
                            self.logger.debug(f"Failed to restore embedding_version_id for chunk_id={chunk_id}: {e}")
                    
                    if version_id is None:
                        self.logger.warning(f"⚠️  Result {i+1}: embedding_version_id is missing and could not be restored")
                        restoration_stats['version_id_failed'] += 1
                elif version_id != original_version_id:
                    restoration_stats['version_id_restored'] += 1
                
                # 🔥 개선: 테이블별 허용 버전 확인
                source_type = result.get('type') or result.get('source_type') or result.get('metadata', {}).get('source_type')
                
                if version_id is not None and source_type:
                    # 테이블별 허용 버전 확인
                    allowed_versions = None
                    
                    if table_version_map and source_type in table_version_map:
                        # 테이블별 허용 버전이 명시된 경우
                        allowed_versions = table_version_map[source_type]
                    elif expected_version_id:
                        # 기본값: expected_version_id만 허용
                        allowed_versions = [expected_version_id]
                    
                    if allowed_versions is not None:
                        # None은 모든 버전 허용을 의미
                        if allowed_versions is None:
                            # 모든 버전 허용
                            self.logger.debug(
                                f"✅ [VERSION ACCEPTED] Result {i+1} ({source_type}): "
                                f"version {version_id} accepted (all versions allowed for this table)"
                            )
                        elif version_id in allowed_versions:
                            # 허용된 버전 목록에 있음
                            if expected_version_id and version_id != expected_version_id:
                                # 폴백된 버전이지만 허용됨
                                self.logger.debug(
                                    f"✅ [VERSION FALLBACK ACCEPTED] Result {i+1} ({source_type}): "
                                    f"using fallback version {version_id} (expected {expected_version_id}, "
                                    f"but allowed in {allowed_versions})"
                                )
                            else:
                                self.logger.debug(
                                    f"✅ [VERSION ACCEPTED] Result {i+1} ({source_type}): "
                                    f"version {version_id} matches expected {expected_version_id}"
                                )
                        else:
                            # 허용된 버전 목록에 없음
                            issues_found['version_mismatch'] += 1
                            self.logger.warning(
                                f"⚠️  [VERSION MISMATCH] Result {i+1} ({source_type}): "
                                f"version {version_id} not in allowed versions {allowed_versions}. "
                                f"Filtering out this result."
                            )
                            continue
                    elif expected_version_id:
                        # 허용 버전이 명시되지 않았지만 expected_version_id가 있는 경우
                        if version_id != expected_version_id:
                            issues_found['version_mismatch'] += 1
                            self.logger.warning(
                                f"⚠️  [VERSION MISMATCH] Result {i+1} ({source_type}): "
                                f"requested version {expected_version_id}, but found version {version_id}. "
                                f"Filtering out this result."
                            )
                            continue
                elif expected_version_id and version_id and version_id != expected_version_id:
                    # source_type이 없거나 version_id가 없는 경우 기존 로직 사용
                    issues_found['version_mismatch'] += 1
                    self.logger.warning(
                        f"⚠️  [VERSION MISMATCH] Result {i+1}: "
                        f"requested version {expected_version_id}, but found version {version_id}. "
                        f"Filtering out this result to ensure version consistency."
                    )
                    continue  # 버전이 다른 결과는 제외
                
                # 2. 메타데이터 완전성 검증 (개선 사항 #2)
                # source_type은 위에서 이미 추출했으므로 재추출 불필요 (없는 경우만 재추출)
                if not source_type:
                    source_type = result.get('type') or result.get('source_type') or result.get('metadata', {}).get('source_type')
                if not source_type:
                    # source_type 복원 시도
                    chunk_id = result.get('chunk_id') or result.get('metadata', {}).get('chunk_id')
                    if chunk_id:
                        try:
                            conn_temp = self._get_connection()
                            cursor_temp = conn_temp.execute(
                                "SELECT 'precedent_content' as source_type FROM precedent_chunks WHERE id = ?",
                                (chunk_id,)
                            )
                            row_temp = cursor_temp.fetchone()
                            if row_temp and row_temp.get('source_type'):
                                source_type = row_temp['source_type']
                                result['source_type'] = source_type
                                result['type'] = source_type
                                if 'metadata' not in result:
                                    result['metadata'] = {}
                                result['metadata']['source_type'] = source_type
                                self.logger.debug(f"✅ Restored source_type for result {i+1}: {source_type}")
                            conn_temp.close()
                        except Exception as e:
                            self.logger.debug(f"Failed to restore source_type for chunk_id={chunk_id}: {e}")
                    
                    # 복원 실패 시에만 경고 및 건너뛰기
                    if not source_type:
                        issues_found['missing_metadata'] += 1
                        self.logger.warning(f"⚠️  Result {i+1}: source_type is missing and could not be restored")
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
                        # 개선: 메타데이터가 일부 누락되어도 결과는 포함 (경고만 출력)
                        # 핵심 필드가 모두 누락된 경우에만 제외
                        critical_fields = self._get_critical_metadata_fields(source_type)
                        if critical_fields:
                            critical_missing = [f for f in still_missing if f in critical_fields]
                            if len(critical_missing) == len(critical_fields):
                                # 모든 핵심 필드가 누락된 경우에만 제외
                                self.logger.warning(
                                    f"⚠️  Result {i+1} ({source_type}): All critical fields missing: {critical_missing}, excluding result"
                                )
                                continue
                            elif critical_missing:
                                # 일부 핵심 필드가 누락되었지만 일부는 있으면 포함 (경고만)
                                self.logger.warning(
                                    f"⚠️  Result {i+1} ({source_type}): Some critical fields missing: {critical_missing}, but including result"
                                )
                        # 핵심 필드가 없거나 일부만 누락된 경우 결과 포함
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
                elif source_type in ['case_paragraph', 'precedent_content', 'decision_paragraph']:  # 🔥 레거시 지원
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
                            with self._get_connection_context() as conn:
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
                        except Exception as e:
                            self.logger.debug(f"Failed to restore text for result {i+1}: {e}")
                    
                    # 복원 후에도 짧으면 건너뛰기 (타입별 최소 길이 차등 적용)
                    final_text = result.get('text') or result.get('content', '')
                    source_type = result.get('type') or result.get('metadata', {}).get('source_type')
                    if source_type == 'statute_article':
                        effective_min_length = 50
                    elif source_type in ['case_paragraph', 'precedent_content', 'decision_paragraph']:  # 🔥 레거시 지원
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
            with self._get_connection_context() as conn:
                cursor = conn.cursor()
                
                # numpy 타입을 Python 정수형으로 변환
                import numpy as np
                chunk_ids_python = [int(cid) if isinstance(cid, (np.integer, np.int64, np.int32)) else cid for cid in chunk_ids]
                placeholders = ",".join(["%s"] * len(chunk_ids_python))  # PostgreSQL은 %s 사용
                query = f"SELECT id, embedding_version FROM precedent_chunks WHERE id IN ({placeholders})"
                self.logger.debug(f"Batch query: {query[:100]}... with {len(chunk_ids_python)} chunk_ids (sample: {chunk_ids_python[:3]})")
                cursor.execute(query, chunk_ids_python)
                rows = cursor.fetchall()
                
                # 실제로 쿼리가 실행되었는지 확인
                if len(rows) == 0:
                    # 샘플 chunk_id로 직접 조회 시도
                    sample_id = chunk_ids_python[0] if chunk_ids_python else None
                    if sample_id:
                        cursor.execute("SELECT id, embedding_version FROM precedent_chunks WHERE id = %s", (sample_id,))
                        test_row = cursor.fetchone()
                        if test_row:
                            self.logger.debug(f"Direct query for chunk_id={sample_id} succeeded, but batch query failed")
                        else:
                            self.logger.debug(f"Direct query for chunk_id={sample_id} also returned no rows - chunk may not exist")
                
                result = {}
                active_version_id = self._get_active_embedding_version_id()
                
                self.logger.debug(f"Batch query returned {len(rows)} rows")
                for row in rows:
                    # PostgreSQL의 경우 dict-like row 또는 tuple 반환
                    if hasattr(row, 'keys'):  # dict-like (RealDictRow)
                        version_id = row['embedding_version_id']
                        chunk_id = row['id']
                    else:  # tuple
                        chunk_id = row[0]
                        version_id = row[1] if len(row) > 1 else None
                    
                    if version_id is None and active_version_id:
                        version_id = active_version_id
                    result[chunk_id] = version_id
                    self.logger.debug(f"  chunk_id={chunk_id}, embedding_version_id={version_id}")
                
                if len(result) < len(chunk_ids):
                    missing = set(chunk_ids) - set(result.keys())
                    self.logger.debug(f"Missing chunks in batch result: {list(missing)[:10]}")
                
                return result
        except Exception as e:
            self.logger.error(f"Failed to batch load embedding_version_ids: {e}", exc_info=True)
            return {}
    
    def _get_critical_metadata_fields(self, source_type: str) -> List[str]:
        """
        필수 필드 중에서도 반드시 있어야 하는 핵심 필드 반환
        
        Args:
            source_type: 소스 타입
            
        Returns:
            핵심 필드 리스트
        """
        critical_fields_map = {
            'case_paragraph': ['doc_id'],  # 🔥 레거시: doc_id는 필수, casenames와 court는 선택적
            'precedent_content': ['doc_id'],  # doc_id는 필수, casenames와 court는 선택적
            'decision_paragraph': ['doc_id'],  # doc_id는 필수, org는 선택적
            'interpretation_paragraph': ['interpretation_id'],  # interpretation_id는 필수
            'statute_article': ['statute_name', 'article_no'],  # 법령명과 조문번호는 필수
        }
        return critical_fields_map.get(source_type, [])
    
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
            # 🔥 레거시: case_paragraph는 precedent_content로 대체
            'case_paragraph': ['doc_id'],  # doc_id만 필수, casenames와 court는 선택적 (일부 판례에 없을 수 있음)
            'precedent_content': ['doc_id'],  # doc_id만 필수, casenames와 court는 선택적 (일부 판례에 없을 수 있음)
            # decision_paragraph: doc_id만 필수, org는 선택적 (일부 결정례에 없을 수 있음, court 필드는 없음)
            'decision_paragraph': ['doc_id'],
            # interpretation_paragraph: interpretation_id만 필수, title은 선택적 (해석례에는 court 필드 없음)
            'interpretation_paragraph': ['interpretation_id']
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
                try:
                    with self._get_connection_context() as conn:
                        cursor = conn.cursor()
                        cursor.execute(
                            "SELECT precedent_content_id, 'precedent_content' as source_type FROM precedent_chunks WHERE id = %s",
                            (chunk_id,)
                        )
                        row = cursor.fetchone()
                        if row:
                            source_id_val = row[0] if isinstance(row, tuple) else row.get('precedent_content_id')
                            source_type_val = 'precedent_content'
                            if source_id_val:
                                source_id = source_id_val
                                if not source_type:
                                    source_type = source_type_val
                                self.logger.debug(f"✅ Restored source_id={source_id} for chunk_id={chunk_id} from precedent_chunks")
                            else:
                                # source_id가 None인 경우에도 계속 진행 (다른 방법으로 복원 시도)
                                self.logger.debug(f"⚠️  source_id is None for chunk_id={chunk_id}, will try alternative restoration methods")
                except Exception as e:
                    self.logger.debug(f"Failed to get source_id from precedent_chunks for chunk_id={chunk_id}: {e}")
            
            # source_id가 여전히 None인 경우에도 복원 시도 (chunk_id로 직접 조회)
            if not source_id and chunk_id:
                # chunk_id로 직접 메타데이터 조회 시도
                try:
                    with self._get_connection_context() as conn:
                        cursor = conn.cursor()
                        # precedent_chunks에서 직접 조회
                        cursor.execute(
                            "SELECT precedent_content_id, 'precedent_content' as source_type, chunk_content FROM precedent_chunks WHERE id = %s",
                            (chunk_id,)
                        )
                        row = cursor.fetchone()
                        if row:
                            # PostgreSQL의 경우 dict-like row 또는 tuple 반환
                            if hasattr(row, 'keys'):
                                source_id_val = row.get('source_id')
                                source_type_val = row.get('source_type')
                                text_val = row.get('text')
                            else:
                                source_id_val = row[0] if len(row) > 0 else None
                                source_type_val = row[1] if len(row) > 1 else None
                                text_val = row[2] if len(row) > 2 else None
                            
                            if not source_id and source_id_val:
                                source_id = source_id_val
                            if not source_type and source_type_val:
                                source_type = source_type_val
                            # text도 복원 시도 (Empty text content 해결)
                            if text_val:
                                restored_text = text_val
                                if len(restored_text.strip()) > 0:
                                    result['text'] = restored_text
                                    result['content'] = restored_text
                                    if 'metadata' not in result:
                                        result['metadata'] = {}
                                    result['metadata']['text'] = restored_text
                                    result['metadata']['content'] = restored_text
                                    self.logger.debug(f"✅ Restored text for chunk_id={chunk_id} (length: {len(restored_text)} chars)")
                except Exception as e:
                    self.logger.debug(f"Failed to get metadata for chunk_id={chunk_id}: {e}")
            
            # source_id가 여전히 None이면 복원 불가능하므로 경고만 출력하고 계속 진행
            if not source_id:
                self.logger.warning(f"⚠️  source_id is None for chunk_id={chunk_id}, metadata restoration may be incomplete")
                # source_id가 없어도 기본 메타데이터 복원은 시도
            
            # 컨텍스트 매니저를 사용하여 연결 자동 반환 보장
            with self._get_connection_context() as conn:
                # 먼저 precedent_chunks.metadata에서 확인
                chunk_metadata_json = None
                if chunk_id:
                    try:
                        cursor_meta = conn.cursor()
                        cursor_meta.execute(
                            "SELECT metadata FROM precedent_chunks WHERE id = %s",
                            (chunk_id,)
                        )
                        meta_row = cursor_meta.fetchone()
                        if meta_row:
                            meta_val = meta_row[0] if isinstance(meta_row, tuple) else meta_row.get('metadata')
                            if meta_val:
                                # precedent_chunks.metadata는 이미 JSONB이므로 파싱 불필요
                                if isinstance(meta_val, dict):
                                    chunk_metadata_json = meta_val
                                else:
                                    try:
                                        import json
                                        chunk_metadata_json = json.loads(meta_val) if isinstance(meta_val, str) else meta_val
                                    except Exception as e:
                                        self.logger.debug(f"Failed to parse metadata JSON for chunk_id={chunk_id}: {e}")
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
                        # precedent_chunks에서 source_id 복원 시도
                        cursor_source = conn.cursor()
                        cursor_source.execute(
                            "SELECT precedent_content_id, 'precedent_content' as source_type FROM precedent_chunks WHERE id = %s",
                            (chunk_id,)
                        )
                        source_row = cursor_source.fetchone()
                        if source_row:
                            restored_source_id = source_row[0] if isinstance(source_row, tuple) else source_row.get('source_id')
                            restored_source_type = source_row[1] if isinstance(source_row, tuple) else source_row.get('source_type')
                            if restored_source_id:
                                if not source_type and restored_source_type:
                                    source_type = restored_source_type
                                # 복원된 source_id로 메타데이터 조회
                                source_meta = self._get_source_metadata(conn, source_type, restored_source_id)
                                source_id = restored_source_id
                                self.logger.debug(f"✅ Restored source_id={source_id} for chunk_id={chunk_id} and retrieved metadata")
                    except Exception as e:
                        self.logger.debug(f"Failed to restore source_id and metadata for chunk_id={chunk_id}: {e}")
                
                # source_id가 None인 경우에도 chunk_id로 직접 복원 시도 (우선 처리)
                if not source_id and chunk_id:
                    try:
                        # precedent_content의 경우: doc_id, casenames, court 복원 (강화)
                        if source_type in ['case_paragraph', 'precedent_content']:  # 🔥 레거시 지원
                            # 먼저 source_id로 조회 시도
                            cursor_case = conn.cursor()
                            if source_id:
                                # precedent_chunks를 통해 precedents 조회
                                cursor_case.execute("""
                                    SELECT pc.precedent_content_id, p.case_name as casenames, p.case_number as doc_id, p.court_name as court
                                    FROM precedent_chunks pc
                                    JOIN precedent_contents pcc ON pc.precedent_content_id = pcc.id
                                    JOIN precedents p ON pcc.precedent_id = p.id
                                    WHERE pc.id = %s
                                """, (source_id,))
                            else:
                                # source_id가 없으면 chunk_id로 직접 조회
                                cursor_case.execute("""
                                    SELECT pc.precedent_content_id, p.case_name as casenames, p.case_number as doc_id, p.court_name as court
                                    FROM precedent_chunks pc
                                    JOIN precedent_contents pcc ON pc.precedent_content_id = pcc.id
                                    JOIN precedents p ON pcc.precedent_id = p.id
                                    WHERE pc.id = %s
                                """, (chunk_id,))
                            
                            case_row = cursor_case.fetchone()
                            if case_row:
                                # PostgreSQL의 경우 dict-like row 또는 tuple 반환
                                if hasattr(case_row, 'keys'):
                                    # 🔥 precedent_content_id는 사용하지 않음 (이미 조회된 결과에서 사용)
                                    casenames = case_row.get('casenames')
                                    doc_id = case_row.get('doc_id')
                                    court = case_row.get('court')
                                else:
                                    # 🔥 precedent_content_id는 사용하지 않음 (이미 조회된 결과에서 사용)
                                    casenames = case_row[1] if len(case_row) > 1 else None
                                    doc_id = case_row[2] if len(case_row) > 2 else None
                                    court = case_row[3] if len(case_row) > 3 else None
                                
                                restored_count = 0
                                if 'doc_id' in missing_fields and doc_id:
                                    result['doc_id'] = doc_id
                                    if 'metadata' not in result:
                                        result['metadata'] = {}
                                    result['metadata']['doc_id'] = doc_id
                                    restored_count += 1
                                if 'casenames' in missing_fields and casenames:
                                    result['casenames'] = casenames
                                    if 'metadata' not in result:
                                        result['metadata'] = {}
                                    result['metadata']['casenames'] = casenames
                                    restored_count += 1
                                if 'court' in missing_fields:
                                    if court:
                                        result['court'] = court
                                        if 'metadata' not in result:
                                            result['metadata'] = {}
                                        result['metadata']['court'] = court
                                        restored_count += 1
                                    else:
                                        # court가 NULL인 경우 기본값 설정
                                        result['court'] = "법원명 미상"
                                        if 'metadata' not in result:
                                            result['metadata'] = {}
                                        result['metadata']['court'] = "법원명 미상"
                                        self.logger.debug(f"⚠️  court is NULL for chunk_id={chunk_id}, set default value")
                                if restored_count > 0:
                                    self.logger.debug(f"✅ Restored {restored_count} case metadata fields for chunk_id={chunk_id} (doc_id, casenames, court)")
                            else:
                                # 조회 실패 시 기본값 설정
                                if 'court' in missing_fields:
                                    result['court'] = "법원명 미상"
                                    if 'metadata' not in result:
                                        result['metadata'] = {}
                                    result['metadata']['court'] = "법원명 미상"
                                    self.logger.debug(f"⚠️  Could not restore court for chunk_id={chunk_id}, set default value")
                        
                        # decision_paragraph의 경우: org, doc_id 복원
                        # 참고: text_chunks 테이블은 PostgreSQL 환경에서 사용되지 않으므로 제거됨
                        elif source_type == 'decision_paragraph':
                            # decision_paragraphs와 decisions를 직접 조회하도록 변경 필요
                            pass
                            decision_row = cursor_decision.fetchone()
                            if decision_row:
                                # PostgreSQL의 경우 dict-like row 또는 tuple 반환
                                if hasattr(decision_row, 'keys'):
                                    org_val = decision_row.get('org')
                                    doc_id_val = decision_row.get('doc_id')
                                else:
                                    org_val = decision_row[1] if len(decision_row) > 1 else None
                                    doc_id_val = decision_row[2] if len(decision_row) > 2 else None
                                
                                if 'org' in missing_fields and org_val:
                                    result['org'] = org_val
                                    if 'metadata' not in result:
                                        result['metadata'] = {}
                                    result['metadata']['org'] = org_val
                                if 'doc_id' in missing_fields and doc_id_val:
                                    result['doc_id'] = doc_id_val
                                    if 'metadata' not in result:
                                        result['metadata'] = {}
                                    result['metadata']['doc_id'] = doc_id_val
                                self.logger.debug(f"✅ Restored decision metadata for chunk_id={chunk_id} (org, doc_id)")
                        
                        # interpretation_paragraph의 경우: interpretation_id, doc_id 복원
                        # 참고: text_chunks 테이블은 PostgreSQL 환경에서 사용되지 않으므로 제거됨
                        elif source_type == 'interpretation_paragraph':
                            # interpretation_paragraphs와 interpretations를 직접 조회하도록 변경 필요
                            pass
                            interp_row = cursor_interp.fetchone()
                            if interp_row:
                                    # PostgreSQL의 경우 dict-like row 또는 tuple 반환
                                    if hasattr(interp_row, 'keys'):
                                        interpretation_id_val = interp_row.get('interpretation_id')
                                        doc_id_val = interp_row.get('doc_id')
                                    else:
                                        interpretation_id_val = interp_row[0] if len(interp_row) > 0 else None
                                        doc_id_val = interp_row[1] if len(interp_row) > 1 else None
                                    
                                    if 'interpretation_id' in missing_fields and interpretation_id_val:
                                        result['interpretation_id'] = interpretation_id_val
                                        if 'metadata' not in result:
                                            result['metadata'] = {}
                                        result['metadata']['interpretation_id'] = interpretation_id_val
                                    if 'doc_id' in missing_fields and doc_id_val:
                                        result['doc_id'] = doc_id_val
                                        if 'metadata' not in result:
                                            result['metadata'] = {}
                                        result['metadata']['doc_id'] = doc_id_val
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
                        # 참고: text_chunks 테이블은 PostgreSQL 환경에서 사용되지 않으므로 제거됨
                        if not field_value and source_id and conn:
                            try:
                                # text_chunks 대신 해당 소스 테이블에서 직접 조회하도록 변경 필요
                                pass
                                cursor_alt = None
                                if False:  # 비활성화
                                    cursor_alt = conn.cursor()
                                    cursor_alt.execute("""
                                        SELECT meta FROM text_chunks
                                        WHERE source_type = %s AND source_id = %s AND meta IS NOT NULL AND meta != ''
                                        LIMIT 1
                                    """, (source_type, source_id))
                                alt_row = cursor_alt.fetchone()
                                if alt_row:
                                    meta_val = alt_row[0] if isinstance(alt_row, tuple) else alt_row.get('meta')
                                    if meta_val:
                                        try:
                                            import json
                                            alt_metadata = json.loads(meta_val)
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
                                    if conn:
                                        cursor_direct = conn.cursor()
                                        # 🔥 개선: statutes_articles 테이블만 사용 (statute_articles는 레거시, 삭제됨)
                                        cursor_direct.execute("""
                                            SELECT sa.article_no, s.name as statute_name
                                            FROM statutes_articles sa
                                            JOIN statutes s ON sa.statute_id = s.id
                                            WHERE sa.id = %s
                                        """, (source_id,))
                                        direct_row = cursor_direct.fetchone()
                                        if direct_row:
                                            # PostgreSQL의 경우 dict-like row 또는 tuple 반환
                                            if hasattr(direct_row, 'keys'):
                                                article_no_val = direct_row.get('article_no')
                                                statute_name_val = direct_row.get('statute_name')
                                            else:
                                                article_no_val = direct_row[0] if len(direct_row) > 0 else None
                                                statute_name_val = direct_row[1] if len(direct_row) > 1 else None
                                            
                                            if field == 'statute_name' or field == 'law_name':
                                                field_value = statute_name_val
                                                if field_value:
                                                    result['statute_name'] = field_value
                                                    result['law_name'] = field_value
                                                    if 'metadata' not in result:
                                                        result['metadata'] = {}
                                                    result['metadata']['statute_name'] = field_value
                                                    result['metadata']['law_name'] = field_value
                                                    self.logger.debug(f"✅ Restored {field}={field_value} for chunk_id={chunk_id} from direct query")
                                            elif field == 'article_no' or field == 'article_number':
                                                field_value = article_no_val
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
                    if source_type in ['case_paragraph', 'precedent_content'] and field == 'court' and not field_value and conn:  # 🔥 레거시 지원
                        try:
                            # source_id가 None인 경우, chunk_id로 먼저 조회
                            actual_source_id = source_id
                            if not actual_source_id and chunk_id:
                                # precedent_chunks에서 precedent_content_id 조회
                                cursor_source = conn.cursor()
                                cursor_source.execute("""
                                    SELECT precedent_content_id FROM precedent_chunks WHERE id = %s
                                """, (chunk_id,))
                                source_row = cursor_source.fetchone()
                                if source_row:
                                    actual_source_id = source_row[0] if isinstance(source_row, tuple) else source_row.get('precedent_content_id')
                            
                            court_row = None
                            
                            # 방법 1: precedents 테이블에서 직접 조회 (source_id가 precedents.id인 경우)
                            if actual_source_id:
                                cursor_court = conn.cursor()
                                cursor_court.execute("""
                                    SELECT court_name FROM precedents WHERE id = %s
                                """, (actual_source_id,))
                                court_row = cursor_court.fetchone()
                            
                            # 방법 2: precedent_chunks를 통한 조회 (source_id가 precedent_chunks.id인 경우)
                            if not court_row and actual_source_id:
                                cursor_court = conn.cursor()
                                cursor_court.execute("""
                                    SELECT p.court
                                    FROM precedent_chunks pc
                                    JOIN precedent_contents pcc ON pc.precedent_content_id = pcc.id
                                    JOIN precedents p ON pcc.precedent_id = p.id
                                    WHERE pc.id = %s
                                """, (actual_source_id,))
                                court_row = cursor_court.fetchone()
                            
                            # 방법 3: precedent_chunks를 통해 precedent_id를 찾아서 조회 (source_id가 None이거나 실패한 경우)
                            if not court_row and chunk_id:
                                try:
                                    # precedent_chunks -> precedent_contents -> precedents 경로
                                    cursor_chunk = conn.cursor()
                                    cursor_chunk.execute("""
                                        SELECT pcc.precedent_id
                                        FROM precedent_chunks pc
                                        JOIN precedent_contents pcc ON pc.precedent_content_id = pcc.id
                                        WHERE pc.id = %s
                                    """, (chunk_id,))
                                    chunk_row = cursor_chunk.fetchone()
                                    if chunk_row:
                                        precedent_id = chunk_row[0] if isinstance(chunk_row, tuple) else chunk_row.get('precedent_id')
                                        if precedent_id:
                                            cursor_court = conn.cursor()
                                            cursor_court.execute("""
                                                SELECT court_name FROM precedents WHERE id = %s
                                            """, (precedent_id,))
                                            court_row = cursor_court.fetchone()
                                except Exception as e:
                                    self.logger.debug(f"Failed to get case_id for chunk_id={chunk_id}: {e}")
                            
                            # 방법 4: chunk_id로 직접 precedent_chunks 조회 (source_id가 없는 경우)
                            if not court_row and chunk_id and not actual_source_id:
                                try:
                                    cursor_chunk = conn.cursor()
                                    cursor_chunk.execute("""
                                        SELECT pcc.precedent_id
                                        FROM precedent_chunks pc
                                        JOIN precedent_contents pcc ON pc.precedent_content_id = pcc.id
                                        WHERE pc.id = %s
                                    """, (chunk_id,))
                                    chunk_row = cursor_chunk.fetchone()
                                    if chunk_row:
                                        precedent_id = chunk_row[0] if isinstance(chunk_row, tuple) else chunk_row.get('precedent_id')
                                        if precedent_id:
                                            cursor_court = conn.cursor()
                                            cursor_court.execute("""
                                                SELECT court_name FROM precedents WHERE id = %s
                                            """, (precedent_id,))
                                            court_row = cursor_court.fetchone()
                                except Exception as e:
                                    self.logger.debug(f"Failed to get case_id via chunk_id for chunk_id={chunk_id}: {e}")
                            
                            if court_row:
                                court_val = court_row[0] if isinstance(court_row, tuple) else court_row.get('court')
                                if court_val:
                                    field_value = court_val
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
                    if source_type in ['case_paragraph', 'precedent_content'] and field == 'casenames' and not field_value and conn:  # 🔥 레거시 지원
                        try:
                            actual_source_id = source_id
                            if not actual_source_id and chunk_id:
                                # precedent_chunks에서 precedent_content_id 조회
                                cursor_source = conn.cursor()
                                cursor_source.execute("""
                                    SELECT precedent_content_id FROM precedent_chunks WHERE id = %s
                                """, (chunk_id,))
                                source_row = cursor_source.fetchone()
                                if source_row:
                                    actual_source_id = source_row[0] if isinstance(source_row, tuple) else source_row.get('precedent_content_id')
                            
                            # source_id가 없으면 chunk_id로 직접 조회
                            if not actual_source_id and chunk_id:
                                try:
                                    # 방법 1: precedent_chunks -> precedent_contents -> precedents 경로
                                    cursor_direct = conn.cursor()
                                    cursor_direct.execute("""
                                        SELECT pcc.precedent_id, p.case_name as casenames, p.case_number as doc_id, p.court_name as court
                                        FROM precedent_chunks pc
                                        JOIN precedent_contents pcc ON pc.precedent_content_id = pcc.id
                                        JOIN precedents p ON pcc.precedent_id = p.id
                                        WHERE pc.id = %s
                                    """, (chunk_id,))
                                    direct_row = cursor_direct.fetchone()
                                    if direct_row:
                                        # PostgreSQL의 경우 dict-like row 또는 tuple 반환
                                        if hasattr(direct_row, 'keys'):
                                            casenames_val = direct_row.get('casenames')
                                            doc_id_val = direct_row.get('doc_id')
                                            court_val = direct_row.get('court')
                                        else:
                                            casenames_val = direct_row[1] if len(direct_row) > 1 else None
                                            doc_id_val = direct_row[2] if len(direct_row) > 2 else None
                                            court_val = direct_row[3] if len(direct_row) > 3 else None
                                        
                                        # casenames 복원
                                        if field == 'casenames' and casenames_val:
                                            field_value = casenames_val
                                            result['casenames'] = field_value
                                            if 'metadata' not in result:
                                                result['metadata'] = {}
                                            result['metadata']['casenames'] = field_value
                                            self.logger.debug(f"✅ Restored casenames={field_value} for chunk_id={chunk_id} (via chunk_id)")
                                        # doc_id 복원
                                        if 'doc_id' in missing_fields and doc_id_val:
                                            result['doc_id'] = doc_id_val
                                            if 'metadata' not in result:
                                                result['metadata'] = {}
                                            result['metadata']['doc_id'] = doc_id_val
                                            self.logger.debug(f"✅ Restored doc_id={doc_id_val} for chunk_id={chunk_id} (via chunk_id)")
                                        # court 복원
                                        if 'court' in missing_fields and court_val:
                                            result['court'] = court_val
                                            if 'metadata' not in result:
                                                result['metadata'] = {}
                                            result['metadata']['court'] = court_val
                                            self.logger.debug(f"✅ Restored court={court_val} for chunk_id={chunk_id} (via chunk_id)")
                                except Exception as e:
                                    self.logger.debug(f"Failed to restore case metadata via chunk_id for chunk_id={chunk_id}: {e}")
                            
                            if actual_source_id:
                                # precedent_chunks를 통해 precedents 조회
                                cursor_casenames = conn.cursor()
                                cursor_casenames.execute("""
                                    SELECT p.case_name as casenames
                                    FROM precedent_chunks pc
                                    JOIN precedent_contents pcc ON pc.precedent_content_id = pcc.id
                                    JOIN precedents p ON pcc.precedent_id = p.id
                                    WHERE pc.id = %s
                                """, (actual_source_id,))
                                casenames_row = cursor_casenames.fetchone()
                                if not casenames_row:
                                    # precedent_contents를 통해 precedents 조회 (source_id가 precedent_content_id인 경우)
                                    cursor_casenames = conn.cursor()
                                    cursor_casenames.execute("""
                                        SELECT p.case_name as casenames
                                        FROM precedent_contents pcc
                                        JOIN precedents p ON pcc.precedent_id = p.id
                                        WHERE pcc.id = %s
                                    """, (actual_source_id,))
                                    casenames_row = cursor_casenames.fetchone()
                                
                                if casenames_row:
                                    casenames_val = casenames_row[0] if isinstance(casenames_row, tuple) else casenames_row.get('casenames')
                                    if casenames_val:
                                        field_value = casenames_val
                                        result['casenames'] = field_value
                                        if 'metadata' not in result:
                                            result['metadata'] = {}
                                        result['metadata']['casenames'] = field_value
                                        self.logger.debug(f"✅ Restored casenames={field_value} for chunk_id={chunk_id}")
                        except Exception as e:
                            self.logger.debug(f"Failed to restore casenames for chunk_id={chunk_id}: {e}")
                    
                    # decision_paragraph의 org 복원 (source_id=None인 경우도 처리)
                    if source_type == 'decision_paragraph' and field == 'org' and not field_value and conn:
                        try:
                            actual_source_id = source_id
                            # 참고: text_chunks 테이블은 PostgreSQL 환경에서 사용되지 않으므로 제거됨
                            if not actual_source_id and chunk_id:
                                # text_chunks 대신 다른 방법으로 source_id 조회 필요
                                pass
                                cursor_source = None
                                if False:  # 비활성화
                                    cursor_source = conn.cursor()
                                    cursor_source.execute("""
                                        SELECT source_id FROM text_chunks WHERE id = %s AND source_type = 'decision_paragraph'
                                    """, (chunk_id,))
                                source_row = cursor_source.fetchone()
                                if source_row:
                                    actual_source_id = source_row[0] if isinstance(source_row, tuple) else source_row.get('source_id')
                            
                            # source_id가 없으면 chunk_id로 직접 조회
                            if not actual_source_id and chunk_id:
                                try:
                                    # 참고: text_chunks 테이블은 PostgreSQL 환경에서 사용되지 않으므로 제거됨
                                    # decision_paragraphs와 decisions를 직접 조회하도록 변경 필요
                                    cursor_direct = None
                                    if False:  # 비활성화
                                        cursor_direct = conn.cursor()
                                        cursor_direct.execute("""
                                            SELECT dp.decision_id, d.org, d.doc_id
                                            FROM text_chunks tc
                                            JOIN decision_paragraphs dp ON tc.source_id = dp.id
                                            JOIN decisions d ON dp.decision_id = d.id
                                            WHERE tc.id = %s AND tc.source_type = 'decision_paragraph'
                                        """, (chunk_id,))
                                    direct_row = cursor_direct.fetchone()
                                    if direct_row:
                                        # PostgreSQL의 경우 dict-like row 또는 tuple 반환
                                        if hasattr(direct_row, 'keys'):
                                            org_val = direct_row.get('org')
                                            doc_id_val = direct_row.get('doc_id')
                                        else:
                                            org_val = direct_row[1] if len(direct_row) > 1 else None
                                            doc_id_val = direct_row[2] if len(direct_row) > 2 else None
                                        
                                        # org 복원
                                        if field == 'org' and org_val:
                                            field_value = org_val
                                            result['org'] = field_value
                                            if 'metadata' not in result:
                                                result['metadata'] = {}
                                            result['metadata']['org'] = field_value
                                            self.logger.debug(f"✅ Restored org={field_value} for chunk_id={chunk_id} (via chunk_id)")
                                        # doc_id 복원
                                        if 'doc_id' in missing_fields and doc_id_val:
                                            result['doc_id'] = doc_id_val
                                            if 'metadata' not in result:
                                                result['metadata'] = {}
                                            result['metadata']['doc_id'] = doc_id_val
                                            self.logger.debug(f"✅ Restored doc_id={doc_id_val} for chunk_id={chunk_id} (via chunk_id)")
                                except Exception as e:
                                    self.logger.debug(f"Failed to restore decision metadata via chunk_id for chunk_id={chunk_id}: {e}")
                            
                            if actual_source_id:
                                # decision_paragraphs를 통해 decisions 조회
                                cursor_org = conn.cursor()
                                cursor_org.execute("""
                                    SELECT d.org
                                    FROM decision_paragraphs dp
                                    JOIN decisions d ON dp.decision_id = d.id
                                    WHERE dp.id = %s
                                """, (actual_source_id,))
                                org_row = cursor_org.fetchone()
                                if not org_row:
                                    # decisions 테이블에서 직접 조회
                                    cursor_org = conn.cursor()
                                    cursor_org.execute("""
                                        SELECT org FROM decisions WHERE id = %s
                                    """, (actual_source_id,))
                                    org_row = cursor_org.fetchone()
                                
                                if org_row:
                                    org_val = org_row[0] if isinstance(org_row, tuple) else org_row.get('org')
                                    if org_val:
                                        field_value = org_val
                                    result['org'] = field_value
                                    if 'metadata' not in result:
                                        result['metadata'] = {}
                                    result['metadata']['org'] = field_value
                                    self.logger.debug(f"✅ Restored org={field_value} for chunk_id={chunk_id}")
                        except Exception as e:
                            self.logger.debug(f"Failed to restore org for chunk_id={chunk_id}: {e}")
                    
                    # interpretation_paragraph의 interpretation_id 복원 (source_id=None인 경우도 처리)
                    # 개선: 오타 필드명도 확인 및 정규화 (interpretatiion_id, interpretattion_id)
                    if source_type == 'interpretation_paragraph' and field == 'interpretation_id' and not field_value:
                        # 오타 필드명도 확인
                        typo_fields = ['interpretatiion_id', 'interpretattion_id']
                        for typo_field in typo_fields:
                            typo_value = result.get(typo_field) or result.get('metadata', {}).get(typo_field)
                            if typo_value:
                                field_value = typo_value
                                result['interpretation_id'] = field_value
                                if 'metadata' not in result:
                                    result['metadata'] = {}
                                result['metadata']['interpretation_id'] = field_value
                                # 오타 필드명 제거
                                if typo_field in result:
                                    del result[typo_field]
                                if typo_field in result.get('metadata', {}):
                                    del result['metadata'][typo_field]
                                self.logger.debug(f"✅ Fixed typo field {typo_field} → interpretation_id={field_value} for chunk_id={chunk_id}")
                                break
                    
                    # 검색 결과 생성 시 오타 필드명 정규화 (추가 개선)
                    # interpretation_paragraph 타입의 경우 모든 오타 필드명을 올바른 필드명으로 정규화
                    if source_type == 'interpretation_paragraph':
                        typo_fields = ['interpretatiion_id', 'interpretattion_id']
                        for typo_field in typo_fields:
                            if typo_field in result:
                                correct_value = result[typo_field]
                                result['interpretation_id'] = correct_value
                                if 'metadata' not in result:
                                    result['metadata'] = {}
                                result['metadata']['interpretation_id'] = correct_value
                                del result[typo_field]
                                self.logger.debug(f"✅ Normalized typo field {typo_field} → interpretation_id={correct_value} for chunk_id={chunk_id}")
                            if 'metadata' in result and isinstance(result['metadata'], dict) and typo_field in result['metadata']:
                                correct_value = result['metadata'][typo_field]
                                result['interpretation_id'] = correct_value
                                if 'metadata' not in result:
                                    result['metadata'] = {}
                                result['metadata']['interpretation_id'] = correct_value
                                del result['metadata'][typo_field]
                                self.logger.debug(f"✅ Normalized typo field {typo_field} in metadata → interpretation_id={correct_value} for chunk_id={chunk_id}")
                    
                    if source_type == 'interpretation_paragraph' and field == 'interpretation_id' and not field_value:
                        try:
                            actual_source_id = source_id
                            # 참고: text_chunks 테이블은 PostgreSQL 환경에서 사용되지 않으므로 제거됨
                            if not actual_source_id and chunk_id and conn:
                                # text_chunks 대신 다른 방법으로 source_id 조회 필요
                                pass
                                cursor_source = None
                                if False:  # 비활성화
                                    cursor_source = conn.cursor()
                                    cursor_source.execute("""
                                        SELECT source_id FROM text_chunks WHERE id = %s AND source_type = 'interpretation_paragraph'
                                    """, (chunk_id,))
                                source_row = cursor_source.fetchone()
                                if source_row:
                                    actual_source_id = source_row[0] if isinstance(source_row, tuple) else source_row.get('source_id')
                            
                            # source_id가 없으면 chunk_id로 직접 조회
                            if not actual_source_id and chunk_id and conn:
                                cursor_direct = conn.cursor()
                                cursor_direct.execute("""
                                    SELECT ip.interpretation_id
                                    FROM text_chunks tc
                                    JOIN interpretation_paragraphs ip ON tc.source_id = ip.id
                                    WHERE tc.id = %s AND tc.source_type = 'interpretation_paragraph'
                                """, (chunk_id,))
                                direct_row = cursor_direct.fetchone()
                                if direct_row:
                                    interpretation_id_val = direct_row[0] if isinstance(direct_row, tuple) else direct_row.get('interpretation_id')
                                    if interpretation_id_val:
                                        field_value = interpretation_id_val
                                        result['interpretation_id'] = field_value
                                        if 'metadata' not in result:
                                            result['metadata'] = {}
                                        result['metadata']['interpretation_id'] = field_value
                                        self.logger.debug(f"✅ Restored interpretation_id={field_value} for chunk_id={chunk_id} (via chunk_id)")
                            
                            if actual_source_id and conn:
                                # interpretation_paragraphs에서 interpretation_id 조회
                                cursor_interp = conn.cursor()
                                cursor_interp.execute("""
                                    SELECT interpretation_id FROM interpretation_paragraphs WHERE id = %s
                                """, (actual_source_id,))
                                interp_row = cursor_interp.fetchone()
                                if interp_row:
                                    interpretation_id_val = interp_row[0] if isinstance(interp_row, tuple) else interp_row.get('interpretation_id')
                                    if interpretation_id_val:
                                        field_value = interpretation_id_val
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
            
            # 컨텍스트 매니저가 자동으로 연결을 반환하므로 _safe_close_connection 호출 불필요
            
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
                # PostgreSQL을 사용하는 경우 db_path는 None일 수 있음
                if self.db_path and Path(self.db_path).exists():
                    with self._get_connection_context() as conn:
                        cursor = conn.cursor()
                        cursor.execute("SELECT COUNT(*) as count FROM embeddings")
                        row = cursor.fetchone()
                        emb_count = row['count'] if row else 0
                        cursor.execute("SELECT COUNT(*) as count FROM precedent_chunks")
                        row = cursor.fetchone()
                        chunk_count = row['count'] if row else 0
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
        
        # 더 공격적인 nprobe 계산 (추가 개선: base_nprobe 계산 방식 최적화)
        # total_vectors에 따라 동적으로 조정하여 검색 속도와 정확도 균형
        if total_vectors < 10000:
            base_nprobe = max(16, int(np.sqrt(total_vectors) / 5))
        elif total_vectors < 100000:
            base_nprobe = max(32, int(np.sqrt(total_vectors) / 10))
        else:
            base_nprobe = max(64, int(np.sqrt(total_vectors) / 15))

        # k 값에 따라 nprobe 조정 (추가 개선: 더 정밀한 조정)
        if k <= 10:
            nprobe = min(base_nprobe * 2, 256)
        elif k <= 20:
            nprobe = min(base_nprobe * 3, 512)
        else:
            nprobe = min(base_nprobe * 4, 1024)

        # IndexIVFPQ는 압축 인덱스이므로 더 높은 nprobe 필요 (정확도 향상)
        if self.index and 'IndexIVFPQ' in type(self.index).__name__:
            nprobe = int(nprobe * 2.0)

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
    
    def _warmup_pgvector_connections(self):
        """pgvector 연결 풀 워밍업 (필요 시에만 실행)"""
        try:
            if not PGVECTOR_AVAILABLE or not self._db_adapter:
                return
            
            # 연결 풀 확인
            if not hasattr(self._db_adapter, 'connection_pool') or not self._db_adapter.connection_pool:
                self.logger.debug("Connection pool not available, skipping warmup")
                return
            
            # 환경 변수로 워밍업 활성화 여부 확인 (기본값: False, 필요 시에만 활성화)
            import os
            enable_warmup = os.getenv("PGVECTOR_ENABLE_WARMUP", "false").lower() == "true"
            if not enable_warmup:
                self.logger.debug("pgvector warmup disabled (set PGVECTOR_ENABLE_WARMUP=true to enable)")
                return
            
            # 초기화 시간 측정
            import time
            import logging
            warmup_start = time.time()
            
            # 연결 풀 크기에 따라 워밍업할 연결 수 결정 (최소화)
            max_conn = getattr(self._db_adapter.connection_pool, 'maxconn', 1)
            # 워밍업 연결 수 감소 (5개 → 2개)
            warmup_connections = min(2, max_conn)
            
            # 워밍업 중 연결 반환 로그 억제를 위해 db_adapter 로거 레벨 일시 조정
            # (요약 로그만 출력하기 위함)
            # 가능한 로거 이름들을 시도
            db_adapter_logger = None
            for logger_name in ['core.data.db_adapter', 'lawfirm_langgraph.core.data.db_adapter']:
                temp_logger = logging.getLogger(logger_name)
                if temp_logger.handlers:  # 핸들러가 있으면 실제 사용 중인 로거
                    db_adapter_logger = temp_logger
                    break
            
            # 로거를 찾지 못한 경우 기본 이름 사용
            if db_adapter_logger is None:
                db_adapter_logger = logging.getLogger('core.data.db_adapter')
            
            original_level = db_adapter_logger.level
            db_adapter_logger.setLevel(logging.WARNING)  # DEBUG/INFO 로그 억제
            
            warmed_count = 0
            try:
                for i in range(warmup_connections):
                    try:
                        with self._db_adapter.get_connection_context() as conn:
                            # 더미 쿼리로 연결 워밍업 및 pgvector 등록
                            cursor = conn.cursor()
                            cursor.execute("SELECT 1")
                            cursor.fetchone()
                            
                            # pgvector 등록 시도 (이미 등록되어 있을 수 있음)
                            try:
                                from pgvector.psycopg2 import register_vector
                                register_vector(conn)
                            except Exception:
                                # 이미 등록되었거나 오류 발생 시 무시
                                pass
                            
                            warmed_count += 1
                    except Exception as e:
                        self.logger.debug(f"Failed to warmup connection {i+1}: {e}")
            finally:
                # 로거 레벨 복원
                db_adapter_logger.setLevel(original_level)
            
            warmup_time = time.time() - warmup_start
            if warmed_count > 0:
                self.logger.info(f"✅ pgvector connections warmed up ({warmed_count}/{warmup_connections}, {warmup_time:.3f}초)")
            else:
                self.logger.warning(f"⚠️ No connections warmed up ({warmup_time:.3f}초)")
        except Exception as e:
            self.logger.debug(f"pgvector warmup failed: {e}")
    
    def _warmup_metadata_cache(self):
        """메타데이터 캐시 워밍업 (자주 사용되는 chunk_id 사전 로딩)"""
        try:
            if not self._db_adapter:
                return
            
            warmup_start = time.time()
            warmup_limit = int(os.getenv("METADATA_CACHE_WARMUP_LIMIT", "100"))  # 기본 100개
            
            with self._get_connection_context() as conn:
                cursor = conn.cursor()
                
                # 자주 사용되는 chunk_id 조회 (최근 검색된 chunk_id 또는 인기 chunk_id)
                # precedent_chunks에서 최근 업데이트된 chunk_id 우선
                warmup_query = """
                    SELECT id 
                    FROM precedent_chunks 
                    WHERE embedding_vector IS NOT NULL 
                    ORDER BY id DESC 
                    LIMIT %s
                """
                cursor.execute(warmup_query, (warmup_limit,))
                rows = cursor.fetchall()
                
                chunk_ids = [row[0] if isinstance(row, (tuple, list)) else row.get('id') for row in rows]
                
                if chunk_ids:
                    # 배치로 메타데이터 로드
                    metadata_map = self._batch_load_chunk_metadata(conn, chunk_ids)
                    warmed_count = len(metadata_map)
                    
                    warmup_time = time.time() - warmup_start
                    if warmed_count > 0:
                        self.logger.info(
                            f"✅ Metadata cache warmed up: {warmed_count}/{len(chunk_ids)} chunks "
                            f"({warmup_time:.3f}초)"
                        )
                    else:
                        self.logger.debug(f"Metadata cache warmup: no chunks loaded ({warmup_time:.3f}초)")
        except Exception as e:
            self.logger.debug(f"Metadata cache warmup failed (non-critical): {e}")

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
        """MLflow에서 FAISS 인덱스 로드 (필수)"""
        if not FAISS_AVAILABLE:
            raise RuntimeError("FAISS is not available")
        
        if not self.use_mlflow_index or not self.mlflow_manager:
            raise RuntimeError("MLflow index is required but not configured. Set USE_MLFLOW_INDEX=true")
        
        try:
            # run_id가 없으면 프로덕션 run 자동 조회
            run_id = self.mlflow_run_id
            if not run_id:
                mlflow_manager = self._get_mlflow_manager()
                if mlflow_manager:
                    run_id = mlflow_manager.get_production_run()
                else:
                    run_id = None
                if run_id:
                    self.logger.info(f"Auto-detected production run: {run_id}")
                    self.mlflow_run_id = run_id
                else:
                    raise RuntimeError("No production run found in MLflow. Please specify MLFLOW_RUN_ID or tag a run as 'production_ready'")
            
            # MLflow에서 인덱스 로드
            self.logger.info(f"Attempting to load index from MLflow run: {run_id}")
            mlflow_manager = self._get_mlflow_manager()
            if not mlflow_manager:
                raise RuntimeError("MLflow manager is not available")
            index_data = mlflow_manager.load_index(run_id)
            if not index_data:
                self.logger.error(f"MLflow manager returned None for run {run_id}. Check MLflow logs for details.")
                raise RuntimeError(f"Failed to load index from MLflow run: {run_id}")
            
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
                with self._get_connection_context() as conn:
                    cursor = conn.cursor()
                    cursor.execute(
                        "SELECT chunk_id FROM embeddings WHERE model = %s ORDER BY chunk_id",
                        (self.model_name,)
                    )
                    rows = cursor.fetchall()
                    self._chunk_ids = [row[0] if isinstance(row, tuple) else row.get('chunk_id') for row in rows]
            
            # 메타데이터 처리
            if metadata:
                self._chunk_metadata = {chunk_id: meta for chunk_id, meta in zip(self._chunk_ids, metadata)}
            
            # 인덱스 타입 감지 및 로깅
            index_type = type(self.index).__name__
            index_dimension = self.index.d
            self.logger.info(f"Loaded MLflow FAISS index: {index_type} ({self.index.ntotal:,} vectors, dimension: {index_dimension}) from run {run_id}")
            
            # Phase 4 최적화: FAISS 스레드 수 설정
            if FAISS_AVAILABLE:
                import os
                num_threads = min(4, os.cpu_count() or 1)
                faiss.omp_set_num_threads(num_threads)
                self.logger.info(f"Set FAISS threads to {num_threads} (CPU cores: {os.cpu_count() or 1})")
            
            # MLflow version_info에서 모델 정보 확인 및 차원 검증
            mlflow_model_name = None
            mlflow_dimension = None
            try:
                import mlflow
                self.logger.info(f"📖 MLflow version_info.json 로드 시도: run_id={run_id}")
                
                version_info = None
                mlflow_manager = self._get_mlflow_manager()
                if mlflow_manager and hasattr(mlflow_manager, 'load_version_info_from_local'):
                    version_info = mlflow_manager.load_version_info_from_local(run_id)
                    if version_info:
                        self.logger.info("✅ 로컬 파일 시스템에서 version_info.json 직접 로드 완료")
                
                if version_info is None:
                    self.logger.debug("로컬 경로에서 version_info.json을 찾을 수 없어 MLflow에서 다운로드 시도")
                    version_info = mlflow.artifacts.load_dict(f"runs:/{run_id}/version_info.json")
                
                self.logger.debug(f"version_info keys: {list(version_info.keys()) if isinstance(version_info, dict) else 'Not a dict'}")
                
                if isinstance(version_info, dict):
                    embedding_config = version_info.get('embedding_config', {})
                    self.logger.debug(f"embedding_config: {embedding_config}")
                    
                    mlflow_model_name = embedding_config.get('model')
                    mlflow_dimension = embedding_config.get('dimension')
                    
                    if mlflow_model_name:
                        self.logger.info(f"✅ MLflow version_info에서 모델 정보 확인: {mlflow_model_name}")
                        if mlflow_dimension:
                            self.logger.info(f"   MLflow 인덱스 차원: {mlflow_dimension}")
                    else:
                        self.logger.warning(f"⚠️  MLflow version_info에 모델 정보가 없습니다. embedding_config: {embedding_config}")
                        if index_dimension:
                            dimension_model_map = {
                                768: "woong0322/ko-legal-sbert-finetuned",
                                384: "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
                                512: "jhgan/ko-sroberta-multitask",
                            }
                            inferred_model = dimension_model_map.get(index_dimension)
                            if inferred_model:
                                self.logger.info(f"   💡 차원({index_dimension}) 기반 추론 모델: {inferred_model}")
                                self.logger.info("   💡 이 모델 정보를 MLflow version_info.json에 저장하려면 scripts/rag/update_mlflow_version_info.py를 실행하세요.")
                else:
                    self.logger.warning(f"⚠️  MLflow version_info가 dict가 아닙니다: {type(version_info)}")
            except Exception as e:
                self.logger.warning(f"⚠️  MLflow version_info.json 로드 실패: {e}", exc_info=True)
            
            # 차원 불일치 시 MLflow 모델로 재초기화 시도
            if self.dim is not None and index_dimension != self.dim:
                self.logger.warning(
                    f"⚠️  차원 불일치 감지:\n"
                    f"   - MLflow 인덱스 차원: {index_dimension}\n"
                    f"   - 현재 임베딩 모델 차원: {self.dim}\n"
                    f"   - 현재 사용 모델: {self.model_name}"
                )
                
                # MLflow에서 모델 정보를 가져왔고, 현재 모델과 다른 경우 재초기화 시도
                if mlflow_model_name and mlflow_model_name != self.model_name:
                    self.logger.info(
                        f"🔄 MLflow 인덱스에 사용된 모델({mlflow_model_name})로 재초기화 시도..."
                    )
                    try:
                        # 임베딩 모델 재초기화
                        self._initialize_embedder(mlflow_model_name)
                        self.model_name = mlflow_model_name
                        self.logger.info(f"✅ 모델 재초기화 완료: {mlflow_model_name} (차원: {self.dim})")
                        
                        # 차원 재검증
                        if self.dim is not None and index_dimension != self.dim:
                            error_msg = (
                                f"❌ 차원 불일치가 지속됩니다!\n"
                                f"   - MLflow 인덱스 차원: {index_dimension}\n"
                                f"   - 재초기화된 모델 차원: {self.dim}\n"
                                f"   - 재초기화된 모델: {self.model_name}\n"
                                f"해결 방법:\n"
                                f"   1. MLflow 인덱스를 재빌드하거나\n"
                                f"   2. 올바른 모델을 사용하세요."
                            )
                            self.logger.error(error_msg)
                            raise RuntimeError(
                                f"Dimension mismatch persists: MLflow index dimension ({index_dimension}) "
                                f"does not match embedding model dimension ({self.dim}) after reinitialization."
                            )
                        else:
                            self.logger.info(f"✅ 차원 일치 확인: 인덱스 차원({index_dimension}) = 모델 차원({self.dim})")
                    except Exception as e:
                        self.logger.error(f"❌ 모델 재초기화 실패: {e}")
                        error_msg = (
                            f"❌ 차원 불일치 및 모델 재초기화 실패!\n"
                            f"   - MLflow 인덱스 차원: {index_dimension}\n"
                            f"   - 현재 임베딩 모델 차원: {self.dim}\n"
                            f"   - 현재 사용 모델: {self.model_name}\n"
                            f"   - MLflow 인덱스 모델: {mlflow_model_name}\n"
                            f"해결 방법:\n"
                            f"   1. MLflow 인덱스에 사용된 모델({mlflow_model_name})을 설치하거나\n"
                            f"   2. 현재 모델({self.model_name})과 호환되는 MLflow 인덱스를 사용하거나\n"
                            f"   3. 새로운 MLflow run으로 인덱스를 재빌드하세요."
                        )
                        self.logger.error(error_msg)
                        raise RuntimeError(
                            f"Dimension mismatch: MLflow index dimension ({index_dimension}) "
                            f"does not match embedding model dimension ({self.dim}). "
                            f"Failed to reinitialize with MLflow model {mlflow_model_name}: {e}"
                        )
                elif not mlflow_model_name:
                    # MLflow 모델 정보가 없는 경우, 차원 기반으로 알려진 모델 시도
                    self.logger.warning(f"⚠️  MLflow version_info에 모델 정보가 없어 차원 기반으로 모델 추론 시도...")
                    
                    # 차원에 맞는 알려진 모델 매핑
                    dimension_model_map = {
                        768: "woong0322/ko-legal-sbert-finetuned",
                        384: "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
                        512: "jhgan/ko-sroberta-multitask",
                    }
                    
                    candidate_model = dimension_model_map.get(index_dimension)
                    if candidate_model and candidate_model != self.model_name:
                        self.logger.info(
                            f"🔄 차원({index_dimension})에 맞는 모델({candidate_model})로 재초기화 시도..."
                        )
                        try:
                            # 임베딩 모델 재초기화
                            self._initialize_embedder(candidate_model)
                            self.model_name = candidate_model
                            self.logger.info(f"✅ 모델 재초기화 완료: {candidate_model} (차원: {self.dim})")
                            
                            # 차원 재검증
                            if self.dim is not None and index_dimension != self.dim:
                                error_msg = (
                                    f"❌ 차원 불일치가 지속됩니다!\n"
                                    f"   - MLflow 인덱스 차원: {index_dimension}\n"
                                    f"   - 재초기화된 모델 차원: {self.dim}\n"
                                    f"   - 재초기화된 모델: {self.model_name}\n"
                                    f"해결 방법:\n"
                                    f"   1. MLflow 인덱스를 재빌드하거나\n"
                                    f"   2. 올바른 모델을 사용하세요."
                                )
                                self.logger.error(error_msg)
                                raise RuntimeError(
                                    f"Dimension mismatch persists: MLflow index dimension ({index_dimension}) "
                                    f"does not match embedding model dimension ({self.dim}) after reinitialization."
                                )
                            else:
                                self.logger.info(f"✅ 차원 일치 확인: 인덱스 차원({index_dimension}) = 모델 차원({self.dim})")
                        except Exception as e:
                            self.logger.error(f"❌ 모델 재초기화 실패: {e}")
                            error_msg = (
                                f"❌ 차원 불일치 및 모델 재초기화 실패!\n"
                                f"   - MLflow 인덱스 차원: {index_dimension}\n"
                                f"   - 현재 임베딩 모델 차원: {self.dim}\n"
                                f"   - 현재 사용 모델: {self.model_name}\n"
                                f"   - 시도한 모델: {candidate_model}\n"
                                f"해결 방법:\n"
                                f"   1. MLflow 인덱스에 사용된 모델을 확인하고 설치하거나\n"
                                f"   2. 현재 모델({self.model_name})과 호환되는 MLflow 인덱스를 사용하거나\n"
                                f"   3. 새로운 MLflow run으로 인덱스를 재빌드하세요."
                            )
                            self.logger.error(error_msg)
                            raise RuntimeError(
                                f"Dimension mismatch: MLflow index dimension ({index_dimension}) "
                                f"does not match embedding model dimension ({self.dim}). "
                                f"Failed to reinitialize with candidate model {candidate_model}: {e}"
                            )
                    else:
                        # 차원에 맞는 모델을 찾지 못한 경우
                        error_msg = (
                            f"❌ 차원 불일치 감지!\n"
                            f"   - MLflow 인덱스 차원: {index_dimension}\n"
                            f"   - 현재 임베딩 모델 차원: {self.dim}\n"
                            f"   - 현재 사용 모델: {self.model_name}\n"
                            f"   - MLflow version_info에 모델 정보가 없습니다.\n"
                            f"해결 방법:\n"
                            f"   1. MLflow 인덱스에 사용된 모델과 동일한 모델을 사용하거나\n"
                            f"   2. 현재 모델({self.model_name})과 호환되는 MLflow 인덱스를 사용하거나\n"
                            f"   3. 새로운 MLflow run으로 인덱스를 재빌드하세요."
                        )
                        self.logger.error(error_msg)
                        raise RuntimeError(
                            f"Dimension mismatch: MLflow index dimension ({index_dimension}) "
                            f"does not match embedding model dimension ({self.dim}). "
                            f"Please use the same model that was used to build the index, or rebuild the index."
                        )
                else:
                    # MLflow 모델 정보가 있지만 현재 모델과 동일한 경우
                    error_msg = (
                        f"❌ 차원 불일치 감지!\n"
                        f"   - MLflow 인덱스 차원: {index_dimension}\n"
                        f"   - 현재 임베딩 모델 차원: {self.dim}\n"
                        f"   - 현재 사용 모델: {self.model_name}\n"
                        f"   - MLflow 인덱스 모델: {mlflow_model_name}\n"
                        f"해결 방법:\n"
                        f"   1. MLflow 인덱스에 사용된 모델과 동일한 모델을 사용하거나\n"
                        f"   2. 현재 모델({self.model_name})과 호환되는 MLflow 인덱스를 사용하거나\n"
                        f"   3. 새로운 MLflow run으로 인덱스를 재빌드하세요."
                    )
                    self.logger.error(error_msg)
                    raise RuntimeError(
                        f"Dimension mismatch: MLflow index dimension ({index_dimension}) "
                        f"does not match embedding model dimension ({self.dim}). "
                        f"Please use the same model that was used to build the index, or rebuild the index."
                    )
            elif self.dim is not None:
                # 차원 일치 확인
                self.logger.info(f"✅ 차원 일치 확인: 인덱스 차원({index_dimension}) = 모델 차원({self.dim})")
                if mlflow_model_name and mlflow_model_name != self.model_name:
                    self.logger.warning(
                        f"⚠️  모델 이름 불일치 (차원은 일치):\n"
                        f"   - MLflow 인덱스 모델: {mlflow_model_name}\n"
                        f"   - 현재 사용 모델: {self.model_name}\n"
                        f"   차원이 일치하므로 검색은 가능하지만, 모델이 다를 수 있습니다."
                    )
            else:
                self.logger.warning(f"⚠️  임베딩 모델 차원을 확인할 수 없습니다. 검색 시 차원 불일치 오류가 발생할 수 있습니다.")
            
            # MLflow version_info의 차원 정보와 실제 인덱스 차원 비교
            if mlflow_dimension:
                if mlflow_dimension != index_dimension:
                    self.logger.warning(
                        f"⚠️  MLflow version_info의 차원 정보({mlflow_dimension})와 "
                        f"실제 인덱스 차원({index_dimension})이 일치하지 않습니다."
                    )
                elif self.dim and mlflow_dimension != self.dim:
                    self.logger.warning(
                        f"⚠️  MLflow version_info의 차원 정보({mlflow_dimension})와 "
                        f"현재 모델 차원({self.dim})이 일치하지 않습니다."
                    )
            
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
        
        except RuntimeError:
            raise
        except Exception as e:
            raise RuntimeError(f"Failed to load MLflow FAISS index: {e}") from e
    
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
                            with self._get_connection_context() as conn:
                                cursor = conn.cursor()
                                cursor.execute(
                                    "SELECT chunk_id FROM embeddings WHERE model = %s ORDER BY chunk_id",
                                    (self.model_name,)
                                )
                                rows = cursor.fetchall()
                                self._chunk_ids = [row[0] if isinstance(row, tuple) else row.get('chunk_id') for row in rows]
                        
                        metadata = index_data.get('metadata', [])
                        if metadata:
                            self._chunk_metadata = {chunk_id: meta for chunk_id, meta in zip(self._chunk_ids, metadata)}
                        
                        self.current_faiss_version = faiss_version_name
                        
                        # 인덱스 타입 감지 및 로깅
                        if self.index is not None:
                            index_type = type(self.index).__name__
                            self.logger.info(f"FAISS index loaded from version: {faiss_version_name} ({index_type}, {len(self._chunk_ids)} vectors)")
                            
                            # Phase 4 최적화: FAISS 스레드 수 설정
                            if FAISS_AVAILABLE:
                                import os
                                num_threads = min(4, os.cpu_count() or 1)
                                faiss.omp_set_num_threads(num_threads)
                                self.logger.info(f"Set FAISS threads to {num_threads} (CPU cores: {os.cpu_count() or 1})")
                            
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
            
            # Phase 4 최적화: FAISS 스레드 수 설정
            if FAISS_AVAILABLE:
                import os
                num_threads = min(4, os.cpu_count() or 1)
                faiss.omp_set_num_threads(num_threads)
                self.logger.info(f"Set FAISS threads to {num_threads} (CPU cores: {os.cpu_count() or 1})")
            
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
                    # DB에서 로드 (활성 버전 사용)
                    with self._get_connection_context() as conn:
                        active_version_id = self._get_active_embedding_version_id()
                        cursor = conn.cursor()
                        if active_version_id:
                            cursor.execute(
                                "SELECT DISTINCT chunk_id FROM embeddings WHERE version_id = %s ORDER BY chunk_id",
                                (active_version_id,)
                            )
                        else:
                            cursor.execute(
                                "SELECT DISTINCT chunk_id FROM embeddings WHERE model = %s ORDER BY chunk_id",
                                (self.model_name,)
                            )
                        rows = cursor.fetchall()
                        self._chunk_ids = [row[0] if isinstance(row, tuple) else row.get('chunk_id') for row in rows]
            else:
                # chunk_ids.json이 없으면 embeddings 테이블에서 로드 (활성 버전 사용)
                with self._get_connection_context() as conn:
                    active_version_id = self._get_active_embedding_version_id()
                    cursor = conn.cursor()
                    if active_version_id:
                        cursor.execute(
                            "SELECT DISTINCT chunk_id FROM embeddings WHERE version_id = %s ORDER BY chunk_id",
                            (active_version_id,)
                        )
                    else:
                        cursor.execute(
                            "SELECT DISTINCT chunk_id FROM embeddings WHERE model = %s ORDER BY chunk_id",
                            (self.model_name,)
                        )
                    rows = cursor.fetchall()
                    self._chunk_ids = [row[0] if isinstance(row, tuple) else row.get('chunk_id') for row in rows]
            
            # FAISS 인덱스 크기와 _chunk_ids 길이 일치 확인
            if self.index and hasattr(self.index, 'ntotal'):
                if len(self._chunk_ids) != self.index.ntotal:
                    diff = len(self._chunk_ids) - self.index.ntotal
                    self.logger.warning(
                        f"⚠️  _chunk_ids length ({len(self._chunk_ids)}) != FAISS index ntotal ({self.index.ntotal}). "
                        f"Difference: {diff} chunks. This may indicate the index was built with a different version."
                    )
                    
                    # 활성 버전의 chunk_id 범위 확인
                    active_version_id = self._get_active_embedding_version_id()
                    if active_version_id:
                        try:
                            conn_check = self._get_connection()
                            cursor_check = conn_check.execute(
                                "SELECT MIN(id) as min_id, MAX(id) as max_id, COUNT(*) as count FROM precedent_chunks WHERE embedding_version = ?",
                                (active_version_id,)
                            )
                            row_check = cursor_check.fetchone()
                            conn_check.close()
                            
                            if row_check and row_check['count'] > 0:
                                active_min_id = row_check['min_id']
                                active_max_id = row_check['max_id']
                                active_count = row_check['count']
                                
                                # _chunk_ids의 범위 확인
                                if self._chunk_ids:
                                    chunk_ids_min = min(self._chunk_ids)
                                    chunk_ids_max = max(self._chunk_ids)
                                    
                                    # 범위가 다르면 다른 버전으로 빌드된 것으로 판단
                                    if chunk_ids_min < active_min_id or chunk_ids_max > active_max_id:
                                        self.logger.warning(
                                            f"   ⚠️  FAISS index chunk_id range ({chunk_ids_min}-{chunk_ids_max}) "
                                            f"does not match active version {active_version_id} range ({active_min_id}-{active_max_id}). "
                                            f"Please rebuild the FAISS index for version {active_version_id}."
                                        )
                        except Exception as e:
                            self.logger.debug(f"Failed to check version consistency: {e}")
                    
                    if diff > 0:
                        self.logger.warning(f"   ⚠️  Truncating _chunk_ids to match index size. {diff} chunks will be excluded from search results.")
                    self._chunk_ids = self._chunk_ids[:self.index.ntotal]
            
            # _chunk_metadata 로드 (버전 정보 포함)
            if self._chunk_ids and not self._chunk_metadata:
                self.logger.info(f"Loading chunk metadata for {len(self._chunk_ids)} chunks...")
                try:
                    with self._get_connection_context() as conn:
                        # 배치로 메타데이터 로드
                        batch_size = 1000
                        for i in range(0, len(self._chunk_ids), batch_size):
                            batch_ids = self._chunk_ids[i:i + batch_size]
                            batch_metadata = self._batch_load_chunk_metadata(conn, batch_ids)
                            self._chunk_metadata.update(batch_metadata)
                    self.logger.info(f"Loaded metadata for {len(self._chunk_metadata)} chunks")
                except Exception as e:
                    self.logger.warning(f"Failed to load chunk metadata: {e}")

            self.logger.info(f"FAISS index loaded: {len(self._chunk_ids)} vectors from {self.index_path}")

        except Exception as e:
            self.logger.warning(f"Failed to load FAISS index: {e}, will rebuild")
            self.index = None
            # 인덱스 파일이 손상된 경우 삭제하여 재빌드 유도
            try:
                Path(self.index_path).unlink()
            except Exception:
                pass

    def _column_exists(self, conn, table_name: str, column_name: str) -> bool:
        """테이블에 컬럼이 존재하는지 확인"""
        try:
            cursor = conn.cursor()
            # PostgreSQL의 경우 information_schema 사용
            if self._db_adapter and self._db_adapter.db_type == 'postgresql':
                cursor.execute("""
                    SELECT column_name 
                    FROM information_schema.columns 
                    WHERE table_name = %s AND column_name = %s
                """, (table_name, column_name))
                row = cursor.fetchone()
                return row is not None
            else:
                # PostgreSQL 사용
                cursor.execute(f"PRAGMA table_info({table_name})")
                columns = [row[1] for row in cursor.fetchall()]
                return column_name in columns
        except Exception:
            return False

    def _batch_load_chunk_metadata(self, conn, chunk_ids: List[int]) -> Dict[int, Dict[str, Any]]:
        """
        여러 chunk_id의 text_chunks.meta를 배치로 조회 (캐싱 적용)
        
        Args:
            conn: 데이터베이스 연결 (DatabaseAdapter를 통해 가져온 연결)
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
            # 배치 크기 최적화: PostgreSQL도 충분히 큰 배치 처리 가능
            batch_size = min(1000, len(uncached_ids))
            
            # 연결 상태를 한 번만 확인 (최적화: 각 배치마다 확인하지 않음)
            connection_valid = True
            if hasattr(conn, '_is_closed') and conn._is_closed():
                self.logger.warning("Connection is closed, attempting to get new connection")
                try:
                    conn = self._get_connection()
                    connection_valid = True
                except Exception as e:
                    self.logger.error(f"Failed to get new connection: {e}")
                    connection_valid = False
            elif hasattr(conn, 'conn') and hasattr(conn.conn, 'closed') and conn.conn.closed != 0:
                self.logger.warning("Connection is closed, attempting to get new connection")
                try:
                    conn = self._get_connection()
                    connection_valid = True
                except Exception as e:
                    self.logger.error(f"Failed to get new connection: {e}")
                    connection_valid = False
            
            if not connection_valid:
                self.logger.error("Cannot proceed with batch load: connection is invalid")
                return metadata_map
            
            # 모든 배치를 하나의 연결로 처리 (연결 재사용)
            for i in range(0, len(uncached_ids), batch_size):
                batch = uncached_ids[i:i + batch_size]
                cursor = None
                try:
                    # DatabaseAdapter를 통한 연결은 cursor를 먼저 가져와야 함
                    cursor = conn.cursor()
                    placeholders = ','.join(['%s'] * len(batch))  # PostgreSQL은 %s 사용
                    query = f"""
                        SELECT id, metadata, 'precedent_content' as source_type, precedent_content_id as source_id, embedding_version
                        FROM precedent_chunks
                        WHERE id IN ({placeholders})
                    """
                    # SQL 변환
                    if self._db_adapter and SQLAdapter:
                        query = SQLAdapter.convert_sql(query, self._db_adapter.db_type)
                    cursor.execute(query, batch)
                    
                    rows = cursor.fetchall()
                    for row in rows:
                        # PostgreSQL의 경우 dict-like row 또는 tuple 반환
                        if hasattr(row, 'keys'):  # dict-like (RealDictRow)
                            chunk_id = row['id']
                            meta_json = {}
                            metadata_val = row.get('metadata')
                            if metadata_val:
                                # precedent_chunks.metadata는 이미 JSONB이므로 파싱 불필요
                                if isinstance(metadata_val, dict):
                                    meta_json = metadata_val
                                else:
                                    try:
                                        import json
                                        meta_json = json.loads(metadata_val) if isinstance(metadata_val, str) else metadata_val
                                    except Exception as e:
                                        self.logger.debug(f"Failed to parse metadata JSON for chunk_id={chunk_id}: {e}")
                            
                            chunk_meta = {
                                'meta': meta_json,
                                'source_type': row.get('source_type', 'precedent_content'),
                                'source_id': row.get('source_id'),
                                'embedding_version': row.get('embedding_version')
                            }
                        else:  # tuple
                            chunk_id = row[0]
                            meta_json = {}
                            metadata_val = row[1] if len(row) > 1 else None
                            if metadata_val:
                                # precedent_chunks.metadata는 이미 JSONB이므로 파싱 불필요
                                if isinstance(metadata_val, dict):
                                    meta_json = metadata_val
                                else:
                                    try:
                                        import json
                                        meta_json = json.loads(metadata_val) if isinstance(metadata_val, str) else metadata_val
                                    except Exception as e:
                                        self.logger.debug(f"Failed to parse metadata JSON for chunk_id={chunk_id}: {e}")
                            
                            chunk_meta = {
                                'meta': meta_json,
                                'source_type': row[2] if len(row) > 2 else 'precedent_content',
                                'source_id': row[3] if len(row) > 3 else None,
                                'embedding_version': row[4] if len(row) > 4 else None
                            }
                        
                        metadata_map[chunk_id] = chunk_meta
                        
                        # 캐시에 저장 (TTL 포함)
                        cache_key = f"chunk_{chunk_id}"
                        self._set_to_cache(cache_key, chunk_meta)
                except Exception as e:
                    self.logger.warning(f"Error in batch_load_chunk_metadata: {e}")
                    # 연결 오류인 경우 재시도하지 않고 다음 배치로 진행
                    if "connection" in str(e).lower() or "closed" in str(e).lower():
                        self.logger.error(f"Connection error in batch_load_chunk_metadata, skipping batch: {e}")
                finally:
                    # cursor 정리
                    if cursor:
                        try:
                            cursor.close()
                        except Exception:
                            pass
        
        if self._metadata_cache_hits + self._metadata_cache_misses > 0:
            hit_rate = self._metadata_cache_hits / (self._metadata_cache_hits + self._metadata_cache_misses) * 100
            if hit_rate > 0:
                self.logger.info(f"📊 Metadata cache hit rate: {hit_rate:.1f}% ({self._metadata_cache_hits} hits, {self._metadata_cache_misses} misses)")
        
        return metadata_map
    
    def _batch_load_embedding_versions(self, conn, chunk_ids: List[int]) -> Dict[int, Optional[int]]:
        """
        여러 chunk_id의 embedding_version을 배치로 조회
        
        Args:
            conn: 데이터베이스 연결
            chunk_ids: 조회할 chunk_id 리스트
            
        Returns:
            chunk_id -> embedding_version 딕셔너리
        """
        if not chunk_ids:
            return {}
        
        version_map = {}
        
        # 배치 크기 최적화
        batch_size = min(1000, len(chunk_ids))
        for i in range(0, len(chunk_ids), batch_size):
            batch = chunk_ids[i:i + batch_size]
            try:
                cursor = conn.cursor()
                placeholders = ','.join(['%s'] * len(batch))
                query = f"""
                    SELECT id, embedding_version
                    FROM precedent_chunks
                    WHERE id IN ({placeholders})
                """
                cursor.execute(query, batch)
                rows = cursor.fetchall()
                
                for row in rows:
                    if hasattr(row, 'keys'):
                        chunk_id = row['id']
                        embedding_version = row.get('embedding_version')
                    else:
                        chunk_id = row[0] if len(row) > 0 else None
                        embedding_version = row[1] if len(row) > 1 else None
                    
                    if chunk_id is not None:
                        version_map[chunk_id] = embedding_version
                
                cursor.close()
            except Exception as e:
                self.logger.warning(f"Error in batch_load_embedding_versions: {e}")
        
        return version_map
    
    def _batch_load_source_metadata(self, conn, source_items: List[Tuple[str, int]]) -> Dict[Tuple[str, int], Dict[str, Any]]:
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
            
            # 배치 크기 최적화
            batch_size = min(500, len(source_ids))
            for i in range(0, len(source_ids), batch_size):
                batch = source_ids[i:i + batch_size]
                try:
                    placeholders = ','.join(['%s'] * len(batch))  # PostgreSQL은 %s 사용
                    cursor = conn.cursor()
                    
                    if source_type == "statute_article":
                        # 🔥 개선: statutes_articles 테이블만 사용 (statute_articles는 레거시, 삭제됨)
                        cursor.execute(f"""
                            SELECT sa.id, sa.article_no, s.name as statute_name, s.abbrv, s.category, s.statute_type
                            FROM statutes_articles sa
                            JOIN statutes s ON sa.statute_id = s.id
                            WHERE sa.id IN ({placeholders})
                        """, batch)
                    elif source_type in ["case_paragraph", "precedent_content"]:  # 🔥 레거시 지원
                        # precedent_chunks를 통해 precedents 조회
                        cursor.execute(f"""
                            SELECT pc.id, p.case_number as doc_id, p.court_name as court, p.case_type_name, p.case_name as casenames, p.decision_date
                            FROM precedent_chunks pc
                            JOIN precedent_contents pcc ON pc.precedent_content_id = pcc.id
                            JOIN precedents p ON pcc.precedent_id = p.id
                            WHERE pc.id IN ({placeholders})
                        """, batch)
                        
                        # 조회된 결과의 source_id 추적
                        found_ids = set()
                        rows_from_paragraphs = []
                        for row in cursor.fetchall():
                            # PostgreSQL의 경우 dict-like row 또는 tuple 반환
                            if hasattr(row, 'keys'):
                                source_id = row.get('id')
                                metadata = dict(row)
                            else:
                                source_id = row[0] if len(row) > 0 else None
                                metadata = {
                                    'id': source_id,
                                    'doc_id': row[1] if len(row) > 1 else None,
                                    'court': row[2] if len(row) > 2 else None,
                                    'case_type_name': row[3] if len(row) > 3 else None,  # 🔥 case_type → case_type_name
                                    'casenames': row[4] if len(row) > 4 else None,
                                    'decision_date': row[5] if len(row) > 5 else None  # 🔥 announce_date → decision_date
                                }
                            
                            if source_id:
                                found_ids.add(source_id)
                                cache_key = (source_type, source_id)
                                metadata_map[cache_key] = metadata
                                self._set_to_cache(cache_key, metadata)
                        
                        # precedent_chunks를 통해 찾지 못한 ID는 precedent_contents를 통해 조회 시도
                        # (source_id가 precedent_content_id인 경우)
                        missing_ids = [sid for sid in batch if sid not in found_ids]
                        if missing_ids:
                            missing_placeholders = ','.join(['%s'] * len(missing_ids))
                            cursor_precedents = conn.cursor()
                            cursor_precedents.execute(f"""
                                SELECT pcc.id, p.case_number as doc_id, p.court_name as court, p.case_type_name, p.case_name as casenames, p.decision_date
                                FROM precedent_contents pcc
                                JOIN precedents p ON pcc.precedent_id = p.id
                                WHERE pcc.id IN ({missing_placeholders})
                            """, missing_ids)
                            
                            for row in cursor_precedents.fetchall():
                                # PostgreSQL의 경우 dict-like row 또는 tuple 반환
                                if hasattr(row, 'keys'):
                                    source_id = row.get('id')
                                    metadata = dict(row)
                                else:
                                    source_id = row[0] if len(row) > 0 else None
                                    metadata = {
                                        'id': source_id,
                                        'doc_id': row[1] if len(row) > 1 else None,
                                        'court': row[2] if len(row) > 2 else None,
                                        'case_type_name': row[3] if len(row) > 3 else None,  # 🔥 case_type → case_type_name
                                        'casenames': row[4] if len(row) > 4 else None,
                                        'decision_date': row[5] if len(row) > 5 else None  # 🔥 announce_date → decision_date
                                    }
                                
                                if source_id:
                                    cache_key = (source_type, source_id)
                                    metadata_map[cache_key] = metadata
                                    self._set_to_cache(cache_key, metadata)
                        
                        continue  # 이미 처리했으므로 아래 for 루프 건너뛰기
                    elif source_type == "decision_paragraph":
                        # decisions 테이블에서 직접 조회
                        cursor.execute(f"""
                            SELECT d.id, d.org, d.doc_id, d.decision_date, d.result
                            FROM decisions d
                            WHERE d.id IN ({placeholders})
                        """, batch)
                    elif source_type == "interpretation_paragraph":
                        cursor.execute(f"""
                            SELECT ip.id, i.org, i.doc_id, i.title, i.response_date
                            FROM interpretation_paragraphs ip
                            JOIN interpretations i ON ip.interpretation_id = i.id
                            WHERE ip.id IN ({placeholders})
                        """, batch)
                    else:
                        continue
                    
                    for row in cursor.fetchall():
                        # PostgreSQL의 경우 dict-like row 또는 tuple 반환
                        if hasattr(row, 'keys'):
                            source_id = row.get('id')
                            metadata = dict(row)
                        else:
                            source_id = row[0] if len(row) > 0 else None
                            metadata = {col: row[i] if len(row) > i else None for i, col in enumerate(['id', 'org', 'doc_id', 'decision_date', 'result'] if source_type == "decision_paragraph" else ['id', 'org', 'doc_id', 'title', 'response_date'])}
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

    def _get_source_metadata(self, conn, source_type: str, source_id: int) -> Dict[str, Any]:
        """
        소스 타입별 상세 메타데이터 조회
        source_id는 text_chunks.source_id로, 각 소스 테이블의 실제 id를 참조
        """
        try:
            if source_type == "statute_article":
                # 🔥 개선: statutes_articles 테이블만 사용 (statute_articles는 레거시, 삭제됨)
                # text_chunks.source_id가 statutes_articles.id를 참조
                base_columns = ["sa.article_no"]
                optional_columns = []
                
                # statute_name 컬럼 확인 (name 또는 law_name_kr 등 다양한 이름 가능)
                if self._column_exists(conn, "statutes", "name"):
                    base_columns.append("s.name as statute_name")
                elif self._column_exists(conn, "statutes", "law_name_kr"):
                    base_columns.append("s.law_name_kr as statute_name")
                elif self._column_exists(conn, "statutes", "law_name"):
                    base_columns.append("s.law_name as statute_name")
                else:
                    # 기본값으로 빈 문자열 사용
                    base_columns.append("'' as statute_name")
                
                # abbrv 컬럼 확인
                if self._column_exists(conn, "statutes", "abbrv"):
                    base_columns.append("s.abbrv")
                elif self._column_exists(conn, "statutes", "law_abbrv"):
                    base_columns.append("s.law_abbrv as abbrv")
                
                # category 컬럼 확인
                if self._column_exists(conn, "statutes", "category"):
                    base_columns.append("s.category")
                elif self._column_exists(conn, "statutes", "domain"):
                    base_columns.append("s.domain as category")
                
                # statute_type 컬럼 확인
                if self._column_exists(conn, "statutes", "statute_type"):
                    base_columns.append("s.statute_type")
                elif self._column_exists(conn, "statutes", "law_type"):
                    base_columns.append("s.law_type as statute_type")
                
                # 선택적 컬럼들
                if self._column_exists(conn, "statutes", "law_id"):
                    optional_columns.append("s.law_id")
                if self._column_exists(conn, "statutes", "mst"):
                    optional_columns.append("s.mst")
                if self._column_exists(conn, "statutes", "proclamation_number"):
                    optional_columns.append("s.proclamation_number")
                if self._column_exists(conn, "statutes", "effective_date"):
                    optional_columns.append("s.effective_date")
                
                select_clause = ", ".join(base_columns)
                if optional_columns:
                    select_clause += ", " + ", ".join(optional_columns)
                
                cursor = conn.cursor()
                cursor.execute(f"""
                    SELECT {select_clause}
                    FROM statutes_articles sa
                    JOIN statutes s ON sa.statute_id = s.id
                    WHERE sa.id = %s
                """, (source_id,))
            elif source_type in ["case_paragraph", "precedent_content"]:  # 🔥 레거시 지원
                # precedent_chunks를 통해 precedents 조회
                base_columns = "p.case_number as doc_id, p.court_name as court, p.case_type_name, p.case_name as casenames, p.decision_date"
                optional_columns = []
                
                if self._column_exists(conn, "precedents", "precedent_serial_number"):
                    optional_columns.append("p.precedent_serial_number")
                
                select_clause = base_columns
                if optional_columns:
                    select_clause += ", " + ", ".join(optional_columns)
                
                # 먼저 precedent_chunks를 통해 조회 시도 (source_id가 precedent_chunks.id인 경우)
                cursor = conn.cursor()
                cursor.execute(f"""
                    SELECT {select_clause}
                    FROM precedent_chunks pc
                    JOIN precedent_contents pcc ON pc.precedent_content_id = pcc.id
                    JOIN precedents p ON pcc.precedent_id = p.id
                    WHERE pc.id = %s
                """, (source_id,))
                row = cursor.fetchone()
                
                # precedent_contents를 통한 조회 시도 (source_id가 precedent_content_id인 경우)
                if not row:
                    cursor.execute(f"""
                        SELECT {select_clause}
                        FROM precedent_contents pcc
                        JOIN precedents p ON pcc.precedent_id = p.id
                        WHERE pcc.id = %s
                    """, (source_id,))
                    row = cursor.fetchone()
                
                if row:
                    return dict(row) if hasattr(row, 'keys') else {col: row[i] if len(row) > i else None for i, col in enumerate(['doc_id', 'court', 'case_type_name', 'casenames', 'decision_date'])}
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
                cursor = conn.cursor()
                cursor.execute(f"""
                    SELECT {select_clause}
                    FROM decisions d
                    WHERE d.id = %s
                """, (source_id,))
                row = cursor.fetchone()
                
                # decision_paragraphs를 통한 조회 시도 (text_chunks.source_id가 decision_paragraphs.id를 참조하는 경우)
                if not row:
                    cursor.execute(f"""
                        SELECT {select_clause}
                        FROM decision_paragraphs dp
                        JOIN decisions d ON dp.decision_id = d.id
                        WHERE dp.id = %s
                    """, (source_id,))
                    row = cursor.fetchone()
                
                if row:
                    return dict(row) if hasattr(row, 'keys') else {col: row[i] if len(row) > i else None for i, col in enumerate(['org', 'doc_id', 'decision_date', 'result'])}
                return {}
            elif source_type == "interpretation_paragraph":
                cursor = conn.cursor()
                cursor.execute("""
                    SELECT ip.*, i.org, i.doc_id, i.title, i.response_date
                    FROM interpretation_paragraphs ip
                    JOIN interpretations i ON ip.interpretation_id = i.id
                    WHERE ip.id = %s
                """, (source_id,))
            else:
                return {}

            row = cursor.fetchone()
            if row:
                metadata = dict(row) if hasattr(row, 'keys') else {col: row[i] if len(row) > i else None for i, col in enumerate(['org', 'doc_id', 'title', 'response_date'])}
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
            error_msg = str(e).lower()
            # 🔥 개선: PostgreSQL 트랜잭션 중단 오류 처리
            if "current transaction is aborted" in error_msg or "aborted" in error_msg:
                try:
                    # 트랜잭션 롤백 시도
                    if hasattr(conn, 'rollback'):
                        conn.rollback()
                    self.logger.warning(f"Transaction aborted for {source_type} {source_id}, rolled back")
                except Exception as rollback_error:
                    self.logger.warning(f"Failed to rollback transaction: {rollback_error}")
            self.logger.warning(f"Error getting source metadata for {source_type} {source_id}: {e}")
            return {}

    def _restore_text_from_source(self, conn, source_type: str, source_id: int, article_no: Optional[str] = None) -> str:
        """
        text_chunks 테이블의 text가 비어있을 때 원본 테이블에서 복원
        TASK 3: article_no를 사용하여 전체 조문 복원 지원
        
        Args:
            conn: 데이터베이스 연결
            source_type: 소스 타입 (statute_article, case_paragraph 등)
            source_id: 소스 ID
            article_no: 조문 번호 (전체 조문 복원용, 선택적)
            
        Returns:
            복원된 text 문자열 (없으면 빈 문자열)
        """
        try:
            # PostgreSQL 연결은 cursor를 직접 사용
            cursor = conn.cursor()
            
            if source_type == "statute_article":
                # TASK 3: article_no가 있으면 전체 조문 복원 시도 (statutes_articles 테이블 사용)
                if article_no:
                    try:
                        # statutes_articles 테이블에서 같은 조문의 모든 항목 조회
                        # article_no를 정규화하여 검색 (선행 0 제거)
                        article_no_normalized = str(article_no).strip().lstrip('0')
                        if not article_no_normalized:
                            article_no_normalized = "0"
                        
                        # 여러 형식으로 시도: 원본, 정규화된 값, 그리고 variants
                        article_no_variants = [
                            str(article_no).strip(),
                            article_no_normalized,
                            f"{article_no_normalized:06d}",  # 750 -> 000750
                            f"{article_no_normalized:06d}00",  # 750 -> 00075000
                        ]
                        
                        # 중복 제거
                        article_no_variants = list(dict.fromkeys(article_no_variants))
                        
                        where_conditions = " OR ".join(["sa.article_no = %s"] * len(article_no_variants))
                        params = article_no_variants
                        
                        cursor.execute(
                            f"""
                            SELECT sa.article_content, sa.clause_no, sa.item_no
                            FROM statutes_articles sa
                            WHERE ({where_conditions})
                            ORDER BY 
                                CASE WHEN sa.clause_no IS NULL THEN 1 ELSE 0 END,
                                sa.clause_no,
                                CASE WHEN sa.item_no IS NULL THEN 1 ELSE 0 END,
                                sa.item_no
                            """,
                            tuple(params)
                        )
                        rows = cursor.fetchall()
                        if rows:
                            # 전체 조문 내용 결합
                            full_article_parts = []
                            for row in rows:
                                if hasattr(row, 'keys'):
                                    content = row.get('article_content')
                                else:
                                    content = row[0] if len(row) > 0 else None
                                if content:
                                    full_article_parts.append(str(content))
                            
                            if full_article_parts:
                                full_article_content = "\n".join(full_article_parts)
                                if len(full_article_content.strip()) >= 200:  # TASK 3: 최소 200자
                                    self.logger.debug(f"✅ [TASK 3] Restored full article for article_no={article_no} (length: {len(full_article_content)} chars)")
                                    return full_article_content
                    except Exception as e:
                        self.logger.debug(f"Failed to restore full article from statutes_articles: {e}")
                
                # 🔥 개선: statutes_articles 테이블만 사용 (statute_articles는 레거시, 삭제됨)
                # statutes_articles 테이블에서 article_content 컬럼으로 조회
                cursor.execute(
                    """
                    SELECT article_content, article_no 
                    FROM statutes_articles 
                    WHERE id = %s
                    """,
                    (source_id,)
                )
                row = cursor.fetchone()
                
                # 결과 처리
                if row:
                    if hasattr(row, 'keys'):
                        text = row.get('article_content')
                        article_no_from_db = row.get('article_no')
                    else:
                        text = row[0] if len(row) > 0 else None
                        article_no_from_db = row[1] if len(row) > 1 else None
                    
                    # TASK 3: text가 짧으면 article_no로 전체 조문 복원 시도
                    if text and len(str(text).strip()) > 0:
                        text_str = str(text)
                        if len(text_str.strip()) < 200 and article_no_from_db:
                            # 전체 조문 복원 시도 (statutes_articles 테이블 사용)
                            try:
                                article_no_normalized = str(article_no_from_db).strip().lstrip('0')
                                if not article_no_normalized:
                                    article_no_normalized = "0"
                                
                                # 여러 형식으로 시도
                                article_no_variants = [
                                    str(article_no_from_db).strip(),
                                    article_no_normalized,
                                    f"{article_no_normalized:06d}",
                                    f"{article_no_normalized:06d}00",
                                ]
                                article_no_variants = list(dict.fromkeys(article_no_variants))
                                
                                where_conditions = " OR ".join(["sa.article_no = %s"] * len(article_no_variants))
                                params = article_no_variants
                                
                                cursor.execute(
                                    f"""
                                    SELECT sa.article_content, sa.clause_no, sa.item_no
                                    FROM statutes_articles sa
                                    WHERE ({where_conditions})
                                    ORDER BY 
                                        CASE WHEN sa.clause_no IS NULL THEN 1 ELSE 0 END,
                                        sa.clause_no,
                                        CASE WHEN sa.item_no IS NULL THEN 1 ELSE 0 END,
                                        sa.item_no
                                    """,
                                    tuple(params)
                                )
                                full_rows = cursor.fetchall()
                                if full_rows:
                                    full_parts = []
                                    for full_row in full_rows:
                                        if hasattr(full_row, 'keys'):
                                            content = full_row.get('article_content')
                                        else:
                                            content = full_row[0] if len(full_row) > 0 else None
                                        if content:
                                            full_parts.append(str(content))
                                    if full_parts:
                                        full_content = "\n".join(full_parts)
                                        if len(full_content.strip()) >= 200:
                                            self.logger.debug(f"✅ [TASK 3] Restored full article from statutes_articles for article_no={article_no_from_db} (length: {len(full_content)} chars)")
                                            return full_content
                            except Exception as e:
                                self.logger.debug(f"Failed to restore full article: {e}")
                        
                        self.logger.debug(f"Restored text for {source_type} id={source_id} (length: {len(text_str)} chars)")
                        return text_str
                
                # statutes_articles에서 찾지 못한 경우, statute_embeddings.metadata에서 시도
                if not row:
                    try:
                        cursor.execute(
                            """
                            SELECT metadata 
                            FROM statute_embeddings 
                            WHERE article_id = %s 
                            AND metadata IS NOT NULL
                            LIMIT 1
                            """,
                            (source_id,)
                        )
                        meta_row = cursor.fetchone()
                        if meta_row:
                            import json
                            metadata = meta_row[0] if isinstance(meta_row, tuple) else meta_row.get('metadata')
                            if metadata:
                                if isinstance(metadata, str):
                                    metadata = json.loads(metadata)
                                # metadata에서 텍스트 추출 시도
                                text = metadata.get('text') or metadata.get('article_content') or metadata.get('content')
                                if text and len(str(text).strip()) > 0:
                                    self.logger.debug(f"Restored text from statute_embeddings.metadata for article_id={source_id} (length: {len(str(text))} chars)")
                                    return str(text)
                    except Exception as e:
                        self.logger.debug(f"Failed to get text from statute_embeddings.metadata: {e}")
                
                # 모든 방법 실패
                self.logger.warning(f"No row found for {source_type} id={source_id}")
                return ""
            elif source_type == "case_paragraph":
                cursor.execute(
                    "SELECT text FROM case_paragraphs WHERE id = %s",
                    (source_id,)
                )
            elif source_type == "decision_paragraph":
                cursor.execute(
                    "SELECT text FROM decision_paragraphs WHERE id = %s",
                    (source_id,)
                )
            elif source_type == "interpretation_paragraph":
                cursor.execute(
                    "SELECT text FROM interpretation_paragraphs WHERE id = %s",
                    (source_id,)
                )
            elif source_type == "precedent_content":
                # precedent_contents 테이블에서 조회
                cursor.execute(
                    "SELECT section_content FROM precedent_contents WHERE id = %s",
                    (source_id,)
                )
            else:
                self.logger.warning(f"Unknown source_type for text restoration: {source_type}")
                return ""
            
            row = cursor.fetchone()
            if row:
                # PostgreSQL의 경우 dict-like row 또는 tuple 반환
                if hasattr(row, 'keys'):
                    text = row.get('text')
                else:
                    text = row[0] if len(row) > 0 else None
                
                if text and len(str(text).strip()) > 0:
                    self.logger.info(f"Successfully restored text for {source_type} id={source_id} (length: {len(str(text))} chars)")
                    return str(text)
                else:
                    self.logger.warning(f"Text field is empty or None for {source_type} id={source_id}")
                    # text가 비어있으면 다른 방법 시도: precedent_chunks에서 직접 조회
                    if source_type == "precedent_content":
                        return self._restore_text_from_precedent_content(conn, source_id)
                    return ""
            else:
                self.logger.warning(f"No row found for {source_type} id={source_id}")
                # 원본 테이블에 없으면 precedent_chunks에서 직접 조회
                if source_type == "precedent_content":
                    return self._restore_text_from_precedent_content(conn, source_id)
                return ""
        except Exception as e:
            self.logger.error(f"Error restoring text from source table ({source_type}, {source_id}): {e}")
            # 에러 발생 시 precedent_chunks에서 직접 조회 시도
            if source_type == "precedent_content":
                return self._restore_text_from_precedent_content(conn, source_id)
            return ""
    
    def _restore_text_from_precedent_content(self, conn, precedent_content_id: int) -> str:
        """
        precedent_chunks 테이블에서 직접 text 조회 (원본 테이블 조회 실패 시)
        """
        try:
            cursor = conn.cursor()
            # 같은 precedent_content_id를 가진 chunk에서 chunk_content 가져오기
            cursor.execute(
                "SELECT chunk_content FROM precedent_chunks WHERE precedent_content_id = %s AND chunk_content IS NOT NULL AND chunk_content != '' ORDER BY chunk_index LIMIT 1",
                (precedent_content_id,)
            )
            row = cursor.fetchone()
            if row:
                # PostgreSQL의 경우 dict-like row 또는 tuple 반환
                if hasattr(row, 'keys'):
                    text = row.get('chunk_content')
                else:
                    text = row[0] if len(row) > 0 else None
                
                if text and len(str(text).strip()) > 0:
                    self.logger.info(f"Restored text from precedent_chunks for precedent_content_id={precedent_content_id} (length: {len(str(text))} chars)")
                    return str(text)
            return ""
        except Exception as e:
            self.logger.error(f"Error restoring text from precedent_chunks (precedent_content_id={precedent_content_id}): {e}")
            return ""
    
    def _restore_text_from_chunks_by_metadata(self, conn, source_type: str, metadata: Dict[str, Any]) -> str:
        """
        메타데이터를 사용하여 text 조회 (source_id가 문자열인 경우)
        참고: text_chunks 테이블은 PostgreSQL 환경에서 사용되지 않으므로 비활성화됨
        """
        # 참고: text_chunks 테이블은 PostgreSQL 환경에서 사용되지 않으므로 비활성화됨
        # 필요시 해당 소스 테이블에서 직접 조회하도록 변경 필요
        return ""
    
    def _ensure_text_content(self,
                            conn,
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
        
        # 참고: text_chunks 테이블은 PostgreSQL 환경에서 사용되지 않으므로 비활성화됨
        # source_id가 None인 경우 chunk_id로 조회하는 로직은 제거됨
        
        # source_id가 없으면 텍스트만 반환 (메타데이터 복원은 불가능하지만 텍스트는 유지)
        if not source_type or not source_id:
            if text and len(text.strip()) >= min_length:
                return text
            return text or ""
        
        try:
            # TASK 3: article_no 추출 (전체 조문 복원용)
            article_no = None
            if full_metadata:
                article_no = full_metadata.get("article_no") or full_metadata.get("article_number")
            if not article_no and source_type == "statute_article":
                # metadata에서 직접 추출 시도
                if hasattr(self, '_extract_article_no_from_metadata'):
                    article_no = self._extract_article_no_from_metadata(full_metadata)
            
            if isinstance(source_id, str):
                if source_id.isdigit():
                    actual_source_id = int(source_id)
                    restored_text = self._restore_text_from_source(conn, source_type, actual_source_id, article_no)
                else:
                    self.logger.debug(f"source_id is string format: {source_id}, trying metadata lookup")
                    restored_text = self._restore_text_from_chunks_by_metadata(conn, source_type, full_metadata)
                    
                    if not restored_text:
                        extracted_id = self._extract_id_from_source_id_string(source_id, source_type)
                        if extracted_id:
                            self.logger.debug(f"Extracted ID from source_id string: {extracted_id}")
                            if isinstance(extracted_id, int):
                                restored_text = self._restore_text_from_source(conn, source_type, extracted_id, article_no)
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
            # 복원 실패 시 더 명확한 로깅
            self.logger.error(
                f"❌ [TEXT RESTORATION FAILED] chunk_id={chunk_id}, source_type={source_type}, source_id={source_id}"
            )
            # 복원 실패 원인 분석
            if not source_type or not source_id:
                self.logger.error(f"  - 원인: source_type 또는 source_id가 없음 (source_type={source_type}, source_id={source_id})")
            elif isinstance(source_id, str) and not source_id.isdigit():
                self.logger.error(f"  - 원인: source_id가 문자열 형식이며 숫자가 아님 (source_id={source_id})")
                self.logger.debug(f"  - metadata keys: {list(full_metadata.keys())[:10] if full_metadata else 'N/A'}")
            else:
                # 데이터베이스에서 실제로 존재하는지 확인
                try:
                    if conn:
                        cursor = conn.cursor()
                        # 참고: text_chunks 테이블은 PostgreSQL 환경에서 사용되지 않으므로 제거됨
                        self.logger.error(f"  - 원인: chunk_id={chunk_id}의 텍스트를 복원할 수 없음 (text_chunks 테이블 미사용)")
                except Exception as db_check_err:
                    self.logger.debug(f"  - 데이터베이스 확인 중 오류: {db_check_err}")
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
                k=int(k * 1.5),  # 여유 있게 검색 (개선: 2배 → 1.5배로 최적화)
                source_types=source_types,
                similarity_threshold=similarity_threshold,
                disable_retry=True  # 빠른 검색
            )
            
            # 중복 제거 및 가중치 적용
            for result in results:
                chunk_id = result.get("metadata", {}).get("chunk_id")
                if chunk_id and chunk_id not in seen_chunk_ids:
                    result["relevance_score"] *= var_weight
                    # 정규화: 가중치 적용 후 1.0 초과 방지
                    result["relevance_score"] = normalize_score(result["relevance_score"])
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
        
        # 2. 구조화된 정보 추출 (법령 조문 번호, 판례 사건번호 등)
        structured_info = self._extract_structured_info(query)
        
        # 3. 키워드 확장 쿼리
        if expanded_keywords and len(expanded_keywords) >= 2:
            # 상위 7-10개 키워드 추가 (개선: 5개 → 7-10개로 확대)
            # 키워드가 많으면 10개, 적으면 7개 사용
            num_keywords = min(10, max(7, len(expanded_keywords)))
            top_keywords = expanded_keywords[:num_keywords]
            
            # 구조화된 정보와 키워드 결합
            if structured_info:
                expanded_query = f"{query} {' '.join(top_keywords)} {' '.join(structured_info)}"
            else:
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
                if structured_info:
                    keyword_only_query = f"{keyword_only_query} {' '.join(structured_info)}"
                variations.append({
                    "query": keyword_only_query,
                    "type": "keyword_only",
                    "weight": 0.8,
                    "priority": 3
                })
        
        # 4. 구조화된 정보만으로 검색 (법령 조문 번호, 판례 사건번호 등)
        if structured_info:
            structured_query = " ".join(structured_info)
            variations.append({
                "query": structured_query,
                "type": "structured_info",
                "weight": 0.95,
                "priority": 2
            })
        
        # 5. 핵심 키워드 추출 쿼리
        core_keywords = self._extract_core_keywords_simple(query)
        if core_keywords and len(core_keywords) >= 2:
            core_query = " ".join(core_keywords)
            if structured_info:
                core_query = f"{core_query} {' '.join(structured_info)}"
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
    
    def _extract_structured_info(self, query: str) -> List[str]:
        """구조화된 정보 추출 (법령 조문 번호, 판례 사건번호 등)"""
        import re
        structured_info = []
        
        # 법령 조문 번호 추출 (예: "민법 제543조", "형법 제250조")
        law_pattern = r'([가-힣]+법)\s*제?\s*(\d+)\s*조'
        law_matches = re.findall(law_pattern, query)
        for law_name, article_num in law_matches:
            structured_info.append(f"{law_name} 제{article_num}조")
        
        # 판례 사건번호 추출 (예: "대법원 2020다12345", "2020나12345")
        precedent_pattern = r'(대법원\s*)?(\d{4}[다나마]\d+)'
        precedent_matches = re.findall(precedent_pattern, query)
        for court, case_num in precedent_matches:
            if court:
                structured_info.append(f"{court}{case_num}")
            else:
                structured_info.append(case_num)
        
        # 헌법 조문 추출 (예: "헌법 제10조")
        constitution_pattern = r'헌법\s*제?\s*(\d+)\s*조'
        constitution_matches = re.findall(constitution_pattern, query)
        for article_num in constitution_matches:
            structured_info.append(f"헌법 제{article_num}조")
        
        return structured_info

    def _extract_core_keywords_simple(self, query: str) -> List[str]:
        """핵심 키워드 추출 (간단한 구현 - KoreanStopwordProcessor 사용)"""
        import re
        
        # 한글 단어 추출
        words = re.findall(r'[가-힣]+', query)
        
        # 불용어 제거 및 길이 필터링 (KoreanStopwordProcessor 사용)
        core_keywords = []
        for w in words:
            if len(w) >= 2:
                if not self.stopword_processor or not self.stopword_processor.is_stopword(w):
                    core_keywords.append(w)
        
        return core_keywords[:5]  # 상위 5개
