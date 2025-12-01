# -*- coding: utf-8 -*-
"""
FAISS 인덱스 생성기
FAISS 인덱스 빌드 및 최적화
"""

import logging
from pathlib import Path
from typing import Optional, List, Dict, Any
import json

import numpy as np
import faiss

try:
    from lawfirm_langgraph.core.utils.logger import get_logger
except ImportError:
    logging.basicConfig(level=logging.INFO)
    get_logger = lambda name: logging.getLogger(name)

logger = get_logger(__name__)


class FaissIndexer:
    """FAISS 인덱스 생성기"""
    
    def __init__(self, dimension: int = 768):
        """
        인덱스 생성기 초기화
        
        Args:
            dimension: 임베딩 차원
        """
        self.dimension = dimension
        try:
            from lawfirm_langgraph.core.utils.logger import get_logger
            self.logger = get_logger(__name__)
        except ImportError:
            self.logger = logging.getLogger(__name__)
    
    def build_index(
        self,
        embeddings: np.ndarray,
        index_type: str = "ivfflat",  # "ivfflat" or "ivfpq"
        nlist: Optional[int] = None
    ) -> faiss.Index:
        """
        FAISS 인덱스 빌드
        
        Args:
            embeddings: 임베딩 배열 (n_samples, dimension)
            index_type: 인덱스 타입 ("ivfflat" or "ivfpq")
            nlist: 클러스터 수 (ivfflat용, None이면 자동 계산)
        
        Returns:
            FAISS 인덱스
        """
        n_samples = embeddings.shape[0]
        
        if index_type == "ivfflat":
            # IndexIVFFlat: 빠른 검색, 정확도 높음
            if nlist is None:
                # 일반적으로 sqrt(n_samples) 또는 n_samples / 100
                nlist = min(int(np.sqrt(n_samples)), 10000)
                nlist = max(nlist, 1)
            
            quantizer = faiss.IndexFlatIP(self.dimension)
            index = faiss.IndexIVFFlat(quantizer, self.dimension, nlist)
            
            # 인덱스 학습
            self.logger.info(f"IndexIVFFlat 학습 시작 (nlist={nlist})")
            index.train(embeddings)
            
            # 인덱스에 벡터 추가
            self.logger.info(f"IndexIVFFlat에 벡터 추가 중...")
            index.add(embeddings)
            
            self.logger.info(
                f"IndexIVFFlat 빌드 완료: {n_samples}개 벡터, "
                f"nlist={nlist}, is_trained={index.is_trained}"
            )
        
        elif index_type == "ivfpq":
            # IndexIVFPQ: 메모리 효율적, 약간의 정확도 손실
            if nlist is None:
                nlist = min(int(np.sqrt(n_samples)), 10000)
                nlist = max(nlist, 1)
            
            # PQ 파라미터 (일반적으로 8, 16, 32, 64)
            m = 64  # sub-vector 수
            bits = 8  # 각 sub-vector의 비트 수
            
            quantizer = faiss.IndexFlatIP(self.dimension)
            index = faiss.IndexIVFPQ(quantizer, self.dimension, nlist, m, bits)
            
            # 인덱스 학습
            self.logger.info(f"IndexIVFPQ 학습 시작 (nlist={nlist}, m={m}, bits={bits})")
            index.train(embeddings)
            
            # 인덱스에 벡터 추가
            self.logger.info(f"IndexIVFPQ에 벡터 추가 중...")
            index.add(embeddings)
            
            self.logger.info(
                f"IndexIVFPQ 빌드 완료: {n_samples}개 벡터, "
                f"nlist={nlist}, m={m}, bits={bits}, is_trained={index.is_trained}"
            )
        
        else:
            raise ValueError(f"Unknown index_type: {index_type}")
        
        return index
    
    def save_index(
        self,
        index: faiss.Index,
        output_path: Path,
        chunk_ids: List[int],
        metadata: List[Dict[str, Any]],
        index_type: str = "ivfflat"
    ) -> bool:
        """
        인덱스 및 메타데이터 저장
        
        Args:
            index: FAISS 인덱스
            output_path: 출력 디렉토리
            chunk_ids: 청크 ID 리스트
            metadata: 메타데이터 리스트
            index_type: 인덱스 타입
        
        Returns:
            성공 여부
        """
        try:
            output_path = Path(output_path)
            output_path.mkdir(parents=True, exist_ok=True)
            
            # 인덱스 저장
            index_file = output_path / f"faiss_index_{index_type}.faiss"
            faiss.write_index(index, str(index_file))
            self.logger.info(f"FAISS 인덱스 저장: {index_file}")
            
            # chunk_ids 저장
            chunk_ids_file = output_path / "chunk_ids.json"
            with open(chunk_ids_file, 'w', encoding='utf-8') as f:
                json.dump(chunk_ids, f, ensure_ascii=False, indent=2)
            self.logger.info(f"chunk_ids 저장: {chunk_ids_file}")
            
            # metadata 저장
            metadata_file = output_path / "metadata.json"
            with open(metadata_file, 'w', encoding='utf-8') as f:
                json.dump(metadata, f, ensure_ascii=False, indent=2)
            self.logger.info(f"metadata 저장: {metadata_file}")
            
            # 통계 저장
            stats = {
                "index_type": index_type,
                "total_vectors": index.ntotal,
                "dimension": self.dimension,
                "is_trained": index.is_trained if hasattr(index, 'is_trained') else True
            }
            
            if hasattr(index, 'nlist'):
                stats["nlist"] = index.nlist
            
            stats_file = output_path / "stats.json"
            with open(stats_file, 'w', encoding='utf-8') as f:
                json.dump(stats, f, ensure_ascii=False, indent=2)
            self.logger.info(f"통계 저장: {stats_file}")
            
            return True
        
        except Exception as e:
            self.logger.error(f"인덱스 저장 실패: {e}")
            return False
    
    def load_index(
        self,
        index_path: Path
    ) -> faiss.Index:
        """
        인덱스 로드
        
        Args:
            index_path: 인덱스 파일 경로
        
        Returns:
            FAISS 인덱스
        """
        try:
            index = faiss.read_index(str(index_path))
            self.logger.info(f"FAISS 인덱스 로드: {index_path}")
            return index
        except Exception as e:
            self.logger.error(f"인덱스 로드 실패: {e}")
            raise

