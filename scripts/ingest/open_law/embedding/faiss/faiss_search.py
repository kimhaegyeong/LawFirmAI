# -*- coding: utf-8 -*-
"""
FAISS 검색 엔진
FAISS 인덱스를 사용한 벡터 유사도 검색
"""

import logging
from pathlib import Path
from typing import List, Dict, Any, Optional
import json

import numpy as np
import faiss
from sqlalchemy import create_engine, text

try:
    from lawfirm_langgraph.core.utils.logger import get_logger
except ImportError:
    logging.basicConfig(level=logging.INFO)
    get_logger = lambda name: logging.getLogger(name)

try:
    from scripts.ingest.open_law.embedding.base_embedder import BaseEmbedder
except ImportError:
    import sys
    from pathlib import Path
    _PROJECT_ROOT = Path(__file__).parent.parent.parent.parent
    sys.path.insert(0, str(_PROJECT_ROOT / "scripts"))
    from ingest.open_law.embedding.base_embedder import BaseEmbedder

logger = get_logger(__name__)


class FaissSearcher:
    """FAISS 검색 엔진"""
    
    def __init__(
        self,
        index_path: Path,
        db_url: str,
        model_name: str = "jhgan/ko-sroberta-multitask"
    ):
        """
        검색 엔진 초기화
        
        Args:
            index_path: FAISS 인덱스 파일 경로 또는 디렉토리
            db_url: PostgreSQL 데이터베이스 URL (메타데이터 조회용)
            model_name: 임베딩 모델 이름
        """
        self.index_path = Path(index_path)
        self.db_url = db_url
        
        try:
            from lawfirm_langgraph.core.utils.logger import get_logger
            self.logger = get_logger(__name__)
        except ImportError:
            self.logger = logging.getLogger(__name__)
        
        # 인덱스 로드
        if self.index_path.is_dir():
            # 디렉토리인 경우 ivfflat 인덱스 찾기
            index_file = self.index_path / "faiss_index_ivfflat.faiss"
            if not index_file.exists():
                # 다른 인덱스 타입 찾기
                for idx_file in self.index_path.glob("faiss_index_*.faiss"):
                    index_file = idx_file
                    break
        else:
            index_file = self.index_path
        
        self.index = faiss.read_index(str(index_file))
        self.logger.info(f"FAISS 인덱스 로드: {index_file} ({self.index.ntotal}개 벡터)")
        
        # chunk_ids 및 metadata 로드
        if self.index_path.is_dir():
            chunk_ids_file = self.index_path / "chunk_ids.json"
            metadata_file = self.index_path / "metadata.json"
        else:
            chunk_ids_file = self.index_path.parent / "chunk_ids.json"
            metadata_file = self.index_path.parent / "metadata.json"
        
        with open(chunk_ids_file, 'r', encoding='utf-8') as f:
            self.chunk_ids = json.load(f)
        
        with open(metadata_file, 'r', encoding='utf-8') as f:
            self.metadata_list = json.load(f)
        
        self.logger.info(f"메타데이터 로드: {len(self.chunk_ids)}개")
        
        # 임베딩 생성기
        self.embedder = BaseEmbedder(model_name)
        
        # 데이터베이스 연결 (메타데이터 필터링용)
        self.engine = create_engine(
            db_url,
            pool_pre_ping=True,
            echo=False
        )
    
    def search(
        self,
        query: str,
        top_k: int = 10,
        data_type: str = "precedents",
        domain: Optional[str] = None,
        similarity_threshold: Optional[float] = None
    ) -> List[Dict[str, Any]]:
        """
        벡터 유사도 검색
        
        Args:
            query: 검색 쿼리
            top_k: 반환할 결과 수
            data_type: 데이터 타입 ("precedents" or "statutes")
            domain: 도메인 필터
            similarity_threshold: 유사도 임계값
        
        Returns:
            검색 결과 리스트
        """
        # 쿼리 임베딩 생성
        query_embedding = self.embedder.encode([query], show_progress=False)[0]
        query_vector = query_embedding.reshape(1, -1).astype(np.float32)
        
        # FAISS 검색
        k = min(top_k * 2, self.index.ntotal)  # 더 많은 후보 수집 (필터링 전)
        distances, indices = self.index.search(query_vector, k)
        
        # 결과 구성
        results = []
        for i, (distance, idx) in enumerate(zip(distances[0], indices[0])):
            if idx == -1:  # FAISS에서 -1은 유효하지 않은 인덱스
                continue
            
            # 유사도 계산 (Inner Product → Cosine Similarity)
            similarity = float(distance)  # 이미 정규화된 벡터이므로 distance가 cosine similarity
            
            # 임계값 필터링
            if similarity_threshold and similarity < similarity_threshold:
                continue
            
            # chunk_id 및 메타데이터 가져오기
            chunk_id = self.chunk_ids[idx]
            metadata = self.metadata_list[idx]
            
            # 도메인 필터링
            if domain and metadata.get("domain") != domain:
                continue
            
            # 결과 추가
            result = {
                "id": chunk_id,
                "similarity": similarity,
                "distance": float(distance),
                "metadata": metadata
            }
            
            # 데이터 타입별 추가 정보
            if data_type == "precedents":
                result["chunk_content"] = metadata.get("chunk_content", "")
                result["section_type"] = metadata.get("section_type")
                result["case_name"] = metadata.get("case_name")
                result["case_number"] = metadata.get("case_number")
                result["decision_date"] = metadata.get("decision_date")
                result["court_name"] = metadata.get("court_name")
            else:  # statutes
                result["article_content"] = metadata.get("article_content", "")
                result["article_no"] = metadata.get("article_no")
                result["article_title"] = metadata.get("article_title")
                result["law_name_kr"] = metadata.get("law_name_kr")
                result["law_abbrv"] = metadata.get("law_abbrv")
            
            results.append(result)
            
            if len(results) >= top_k:
                break
        
        # 유사도 기준 정렬
        results.sort(key=lambda x: x["similarity"], reverse=True)
        
        return results
    
    def search_by_vector(
        self,
        query_vector: np.ndarray,
        top_k: int = 10
    ) -> List[Dict[str, Any]]:
        """
        벡터로 직접 검색
        
        Args:
            query_vector: 쿼리 벡터
            top_k: 반환할 결과 수
        
        Returns:
            검색 결과 리스트
        """
        query_vector = query_vector.reshape(1, -1).astype(np.float32)
        
        # FAISS 검색
        k = min(top_k, self.index.ntotal)
        distances, indices = self.index.search(query_vector, k)
        
        # 결과 구성
        results = []
        for distance, idx in zip(distances[0], indices[0]):
            if idx == -1:
                continue
            
            chunk_id = self.chunk_ids[idx]
            metadata = self.metadata_list[idx]
            
            results.append({
                "id": chunk_id,
                "similarity": float(distance),
                "distance": float(distance),
                "metadata": metadata
            })
        
        # 유사도 기준 정렬
        results.sort(key=lambda x: x["similarity"], reverse=True)
        
        return results

