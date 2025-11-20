from typing import List
import os
import threading

import numpy as np
import torch
from transformers import AutoModel, AutoTokenizer


class SentenceEmbedder:
    # 클래스 변수: 스레드 설정이 이미 완료되었는지 추적
    _threads_configured = False
    _threads_lock = threading.Lock()
    
    # 싱글톤 패턴: 모델 이름별 인스턴스 캐시
    _instances = {}
    _instances_lock = threading.Lock()
    
    def __new__(cls, model_name: str = None):
        # 모델 이름 결정
        if model_name is None:
            model_name = os.getenv("EMBEDDING_MODEL")
            if model_name is None:
                try:
                    import sys
                    from pathlib import Path
                    sys.path.insert(0, str(Path(__file__).parent.parent.parent / "lawfirm_langgraph"))
                    from core.utils.config import Config
                    config = Config()
                    model_name = config.embedding_model
                except Exception:
                    model_name = os.getenv("EMBEDDING_MODEL", None)
                    if model_name is None:
                        raise ValueError(
                            "Embedding model name is required. "
                            "Please set EMBEDDING_MODEL environment variable or configure it in .env file."
                        )
        
        # 모델 이름 정리
        if model_name:
            model_name = model_name.strip().strip('"').strip("'").strip()
        
        if not model_name:
            raise ValueError("Model name cannot be empty")
        
        # 싱글톤 패턴: 같은 모델 이름이면 기존 인스턴스 반환
        with cls._instances_lock:
            if model_name not in cls._instances:
                instance = super(SentenceEmbedder, cls).__new__(cls)
                cls._instances[model_name] = instance
                instance._initialized = False
            return cls._instances[model_name]
    
    def __init__(self, model_name: str = None):
        # 이미 초기화된 인스턴스는 재초기화하지 않음
        if hasattr(self, '_initialized') and self._initialized:
            return
        
        # 모델 이름 결정 (__new__에서 이미 처리했지만 여기서도 확인)
        if model_name is None:
            model_name = os.getenv("EMBEDDING_MODEL")
            if model_name is None:
                try:
                    import sys
                    from pathlib import Path
                    sys.path.insert(0, str(Path(__file__).parent.parent.parent / "lawfirm_langgraph"))
                    from core.utils.config import Config
                    config = Config()
                    model_name = config.embedding_model
                except Exception:
                    model_name = os.getenv("EMBEDDING_MODEL", None)
                    if model_name is None:
                        raise ValueError(
                            "Embedding model name is required. "
                            "Please set EMBEDDING_MODEL environment variable or configure it in .env file."
                        )
        
        # 모델 이름 정리
        if model_name:
            model_name = model_name.strip().strip('"').strip("'").strip()
        
        if not model_name:
            raise ValueError("Model name cannot be empty")
        # PyTorch 스레드 수 최대화 (시스템 사양에 맞게)
        # 이미 설정된 경우 재설정 시도하지 않음 (스레드 설정 오류 방지)
        cpu_count = os.cpu_count()
        if cpu_count:
            # 스레드 설정은 한 번만 수행 (이미 설정된 경우 무시)
            with SentenceEmbedder._threads_lock:
                if not SentenceEmbedder._threads_configured:
                    try:
                        torch.set_num_threads(cpu_count)
                        torch.set_num_interop_threads(cpu_count)
                        SentenceEmbedder._threads_configured = True
                    except RuntimeError as e:
                        # 이미 스레드가 설정된 경우 무시
                        if "cannot set number of interop threads" in str(e).lower():
                            SentenceEmbedder._threads_configured = True
                        else:
                            raise
                
                # 환경 변수는 항상 설정 가능 (이미 설정되어 있어도 덮어쓰기 가능)
                os.environ['MKL_NUM_THREADS'] = str(cpu_count)
                os.environ['OMP_NUM_THREADS'] = str(cpu_count)
                os.environ['NUMEXPR_NUM_THREADS'] = str(cpu_count)
        
        # GPU 디바이스 확인 (CUDA 또는 ROCm)
        if torch.cuda.is_available():
            self.device = "cuda"
        elif hasattr(torch, 'hip') and torch.hip.is_available():
            # AMD GPU (ROCm) - CUDA API 호환
            self.device = "cuda"
        else:
            self.device = "cpu"
        
        if model_name is None:
            model_name = os.getenv("EMBEDDING_MODEL")
            if model_name is None:
                try:
                    import sys
                    from pathlib import Path
                    sys.path.insert(0, str(Path(__file__).parent.parent.parent / "lawfirm_langgraph"))
                    from core.utils.config import Config
                    config = Config()
                    model_name = config.embedding_model
                except Exception:
                    # 환경 변수나 Config에서 모델을 가져올 수 없는 경우에만 기본값 사용
                    # 기본값도 환경 변수에서 가져오도록 시도
                    model_name = os.getenv("EMBEDDING_MODEL", None)
                    if model_name is None:
                        raise ValueError(
                            "Embedding model name is required. "
                            "Please set EMBEDDING_MODEL environment variable or configure it in .env file."
                        )
        
        # 모델 이름 정리 (따옴표 및 공백 제거)
        if model_name:
            model_name = model_name.strip().strip('"').strip("'").strip()
        
        if not model_name:
            raise ValueError("Model name cannot be empty")
        
        self.model_name = model_name
        
        # sentence-transformers를 우선 사용 (HuggingFace 모델 호환성 향상)
        # woong0322/ko-legal-sbert-finetuned 같은 모델은 sentence-transformers 형식으로 배포됨
        try:
            from sentence_transformers import SentenceTransformer
            import logging
            logger = logging.getLogger(__name__)
            
            logger.info(f"Loading embedding model using sentence-transformers: {model_name}")
            # sentence-transformers를 사용하여 모델 로딩
            # device_map=None, low_cpu_mem_usage=False로 meta tensor 문제 방지
            self.model = SentenceTransformer(
                model_name,
                device=self.device,
                model_kwargs={
                    "low_cpu_mem_usage": False,  # meta device 사용 방지
                    "device_map": None,  # device_map 사용 안 함
                    "torch_dtype": torch.float32,  # 명시적 dtype 설정
                    "trust_remote_code": True,  # 원격 코드 신뢰
                }
            )
            self.dim = self.model.get_sentence_embedding_dimension()
            # tokenizer는 sentence-transformers가 내부적으로 사용
            self.tokenizer = None
            logger.info(f"✅ Successfully loaded model {model_name} using sentence-transformers (dim={self.dim})")
            self._initialized = True
            
        except Exception as e:
            # sentence-transformers 로딩 실패 시 AutoModel 사용 (fallback)
            import warnings
            import logging
            logger = logging.getLogger(__name__)
            logger.warning(f"⚠️  sentence-transformers loading failed for {model_name}, falling back to AutoModel: {e}")
            warnings.warn(f"sentence-transformers loading failed, using AutoModel fallback: {e}")
            
            try:
                # 모델 로딩 시 meta tensor 문제 해결
                # CPU에 먼저 로드한 후 디바이스로 이동하는 방식 사용
                self.tokenizer = AutoTokenizer.from_pretrained(model_name)
                
                # CPU에 먼저 로드 (meta tensor 문제 방지)
                self.model = AutoModel.from_pretrained(
                    model_name,
                    torch_dtype=torch.float32,  # CPU에서는 float32 사용
                    low_cpu_mem_usage=False  # 안정성을 위해 False 사용
                )
                
                # CPU에서 디바이스로 이동 (단계별 이동으로 meta tensor 문제 방지)
                if self.device != "cpu":
                    # 먼저 CPU에서 메모리에 로드된 상태로 확인
                    self.model.eval()  # 평가 모드로 설정
                    # 디바이스로 이동
                    self.model = self.model.to(self.device)
                    # GPU인 경우 float16으로 변환 (선택적)
                    if self.device == "cuda":
                        try:
                            self.model = self.model.half()  # float16으로 변환
                        except Exception:
                            # 변환 실패 시 float32 유지
                            pass
                
                # try to infer dimension
                self.dim = self.model.config.hidden_size
                logger.info(f"✅ Successfully loaded model {model_name} using AutoModel (dim={self.dim})")
                self._initialized = True
                
            except Exception as e2:
                # 모든 방법 실패
                error_msg = (
                    f"Failed to load embedding model '{model_name}'. "
                    f"Tried sentence-transformers and AutoModel, both failed. "
                    f"Last error: {e2}. "
                    f"Please check if the model exists on HuggingFace: https://huggingface.co/{model_name}"
                )
                logger.error(error_msg)
                raise RuntimeError(error_msg) from e2

    @torch.no_grad()
    def encode(self, texts: List[str], batch_size: int = 16, normalize: bool = True) -> np.ndarray:
        # sentence-transformers를 사용하는 경우
        if self.tokenizer is None:
            # sentence-transformers의 encode 메서드 사용
            vectors = self.model.encode(texts, batch_size=batch_size, normalize_embeddings=normalize)
            return np.array(vectors, dtype=np.float32)
        
        # AutoModel을 사용하는 경우
        embeddings: List[np.ndarray] = []
        for i in range(0, len(texts), batch_size):
            batch = texts[i : i + batch_size]
            encoded = self.tokenizer(
                batch,
                padding=True,
                truncation=True,
                max_length=512,
                return_tensors="pt",
            ).to(self.device)
            outputs = self.model(**encoded)
            # mean pooling
            last_hidden = outputs.last_hidden_state  # (B, T, H)
            attention_mask = encoded["attention_mask"].unsqueeze(-1)  # (B, T, 1)
            masked = last_hidden * attention_mask
            sum_vec = masked.sum(dim=1)
            lengths = attention_mask.sum(dim=1).clamp(min=1)
            sent_vec = sum_vec / lengths
            vec = sent_vec.detach().cpu().numpy()
            if normalize:
                norms = np.linalg.norm(vec, axis=1, keepdims=True) + 1e-9
                vec = vec / norms
            embeddings.append(vec)
            
            # 메모리 정리: 중간 텐서 삭제
            del encoded
            del outputs
            del last_hidden
            del masked
            del sum_vec
            del lengths
            del sent_vec
            if self.device != "cpu":
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                elif hasattr(torch, 'hip') and torch.hip.is_available():
                    torch.hip.empty_cache()
        
        if not embeddings:
            return np.zeros((0, self.dim), dtype=np.float32)
        result = np.vstack(embeddings).astype(np.float32)
        del embeddings
        return result
