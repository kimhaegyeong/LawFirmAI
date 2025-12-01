# -*- coding: utf-8 -*-
"""
공통 임베딩 생성기
SentenceTransformer를 사용한 벡터 임베딩 생성
"""

import logging
from typing import List, Optional
import numpy as np
from tqdm import tqdm

# lawfirm_langgraph 패키지 import 방지 (SQLite 오류 방지)
# 표준 logging 모듈 직접 사용
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

try:
    from scripts.utils.embeddings import SentenceEmbedder
except ImportError:
    try:
        from sentence_transformers import SentenceTransformer
        SentenceEmbedder = None
    except ImportError:
        SentenceTransformer = None
        SentenceEmbedder = None

logger = logging.getLogger(__name__)


class BaseEmbedder:
    """공통 임베딩 생성기"""
    
    def __init__(self, model_name: str = "jhgan/ko-sroberta-multitask"):
        """
        임베딩 생성기 초기화
        
        Args:
            model_name: SentenceTransformer 모델 이름
        """
        self.model_name = model_name
        # 표준 logging 모듈 직접 사용
        self.logger = logging.getLogger(__name__)
        
        # SentenceEmbedder 사용 (기존 시스템과 호환)
        if SentenceEmbedder:
            self.embedder = SentenceEmbedder(model_name)
            self.dimension = 768  # ko-sroberta-multitask는 768차원
        elif SentenceTransformer:
            self.model = SentenceTransformer(model_name)
            self.dimension = self.model.get_sentence_embedding_dimension()
            self.embedder = None
        else:
            raise ImportError(
                "sentence-transformers가 설치되지 않았습니다. "
                "pip install sentence-transformers 실행하세요."
            )
        
        try:
            self.logger.info(f"임베딩 모델 로드 완료: {model_name} (차원: {self.dimension})")
        except (ValueError, AttributeError):
            # Windows 환경에서 로깅 버퍼 오류 무시
            pass
    
    def encode(
        self,
        texts: List[str],
        batch_size: int = 100,
        show_progress: bool = True
    ) -> np.ndarray:
        """
        텍스트 리스트를 벡터 임베딩으로 변환
        
        Args:
            texts: 임베딩할 텍스트 리스트
            batch_size: 배치 크기
            show_progress: 진행 상황 표시 여부
        
        Returns:
            임베딩 벡터 배열 (n_samples, dimension)
        """
        if not texts:
            return np.array([])
        
        try:
            self.logger.info(f"임베딩 생성 시작: {len(texts)}개 텍스트, 배치 크기: {batch_size}")
        except (ValueError, AttributeError):
            # Windows 환경에서 로깅 버퍼 오류 무시
            pass
        
        try:
            if self.embedder:
                # 기존 SentenceEmbedder 사용
                embeddings = self.embedder.encode(
                    texts,
                    batch_size=batch_size,
                    normalize=True
                )
            else:
                # SentenceTransformer 직접 사용
                if show_progress:
                    embeddings = self.model.encode(
                        texts,
                        batch_size=batch_size,
                        show_progress_bar=True,
                        normalize_embeddings=True
                    )
                else:
                    embeddings = self.model.encode(
                        texts,
                        batch_size=batch_size,
                        show_progress_bar=False,
                        normalize_embeddings=True
                    )
            
            # numpy 배열로 변환
            if not isinstance(embeddings, np.ndarray):
                embeddings = np.array(embeddings)
            
            # float32로 변환 (메모리 절약)
            embeddings = embeddings.astype(np.float32)
            
            try:
                self.logger.info(
                    f"임베딩 생성 완료: {embeddings.shape[0]}개 벡터, "
                    f"차원: {embeddings.shape[1]}"
                )
            except (ValueError, AttributeError):
                # Windows 환경에서 로깅 버퍼 오류 무시
                pass
            
            return embeddings
        
        except Exception as e:
            self.logger.error(f"임베딩 생성 실패: {e}")
            raise
    
    def encode_batch(
        self,
        texts: List[str],
        batch_size: int = 100,
        show_progress: bool = True
    ) -> List[np.ndarray]:
        """
        배치 단위로 임베딩 생성 (메모리 효율적)
        
        Args:
            texts: 임베딩할 텍스트 리스트
            batch_size: 배치 크기
            show_progress: 진행 상황 표시 여부
        
        Returns:
            배치별 임베딩 벡터 리스트
        """
        if not texts:
            return []
        
        batches = []
        total_batches = (len(texts) + batch_size - 1) // batch_size
        
        iterator = range(0, len(texts), batch_size)
        if show_progress:
            iterator = tqdm(iterator, desc="임베딩 생성", total=total_batches)
        
        for i in iterator:
            batch_texts = texts[i:i + batch_size]
            batch_embeddings = self.encode(
                batch_texts,
                batch_size=len(batch_texts),
                show_progress=False
            )
            batches.append(batch_embeddings)
        
        return batches
    
    def get_dimension(self) -> int:
        """임베딩 차원 반환"""
        return self.dimension

