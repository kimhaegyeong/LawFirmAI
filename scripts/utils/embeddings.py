from typing import List
import os

import numpy as np
import torch
from transformers import AutoModel, AutoTokenizer


class SentenceEmbedder:
    def __init__(self, model_name: str = "snunlp/KR-SBERT-V40K-klueNLI-augSTS"):
        # PyTorch 스레드 수 최대화 (시스템 사양에 맞게)
        cpu_count = os.cpu_count()
        if cpu_count:
            torch.set_num_threads(cpu_count)
            torch.set_num_interop_threads(cpu_count)
            
            # BLAS 라이브러리 최적화
            os.environ['MKL_NUM_THREADS'] = str(cpu_count)
            os.environ['OMP_NUM_THREADS'] = str(cpu_count)
            os.environ['NUMEXPR_NUM_THREADS'] = str(cpu_count)
        
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model_name = model_name
        
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
            
        except Exception as e:
            # meta tensor 에러 발생 시 대체 방법 시도
            if "meta tensor" in str(e).lower() or "to_empty" in str(e).lower():
                import warnings
                warnings.warn(f"Meta tensor error detected, trying alternative loading method: {e}")
                
                # 대체 방법: CPU에 먼저 로드 후 디바이스로 이동
                try:
                    self.tokenizer = AutoTokenizer.from_pretrained(model_name)
                    # CPU에 먼저 로드
                    self.model = AutoModel.from_pretrained(
                        model_name,
                        torch_dtype=torch.float32,
                        low_cpu_mem_usage=False
                    )
                    # CPU에서 디바이스로 이동
                    if self.device != "cpu":
                        self.model.eval()
                        self.model = self.model.to(self.device)
                        if self.device == "cuda":
                            try:
                                self.model = self.model.half()
                            except Exception:
                                pass
                    self.dim = self.model.config.hidden_size
                except Exception as e2:
                    # 최후의 수단: sentence-transformers 사용
                    import warnings
                    warnings.warn(f"AutoModel loading failed, falling back to sentence-transformers: {e2}")
                    from sentence_transformers import SentenceTransformer
                    self.model = SentenceTransformer(model_name, device=self.device)
                    self.dim = self.model.get_sentence_embedding_dimension()
                    # tokenizer는 sentence-transformers가 내부적으로 사용
                    self.tokenizer = None
            else:
                # 다른 에러는 그대로 전파
                raise

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
            if self.device != "cpu" and torch.cuda.is_available():
                torch.cuda.empty_cache()
        
        if not embeddings:
            return np.zeros((0, self.dim), dtype=np.float32)
        result = np.vstack(embeddings).astype(np.float32)
        del embeddings
        return result
