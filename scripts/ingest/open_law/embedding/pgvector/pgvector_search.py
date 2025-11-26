# -*- coding: utf-8 -*-
"""
pgvector 검색 엔진
PostgreSQL pgvector를 사용한 벡터 유사도 검색
"""

import logging
from typing import List, Dict, Any, Optional
import numpy as np
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


class PgVectorSearcher:
    """pgvector 검색 엔진"""
    
    def __init__(
        self,
        db_url: str,
        model_name: str = "jhgan/ko-sroberta-multitask",
        data_type: str = "precedents",  # "precedents" or "statutes"
        version: Optional[int] = None
    ):
        """
        검색 엔진 초기화
        
        Args:
            db_url: PostgreSQL 데이터베이스 URL
            model_name: 임베딩 모델 이름
            data_type: 데이터 타입 ("precedents" or "statutes")
            version: 임베딩 버전 (None이면 활성 버전 사용)
        """
        self.db_url = db_url
        self.data_type = data_type
        try:
            from lawfirm_langgraph.core.utils.logger import get_logger
            self.logger = get_logger(__name__)
        except ImportError:
            self.logger = logging.getLogger(__name__)
        
        self.engine = create_engine(
            db_url,
            pool_pre_ping=True,
            echo=False
        )
        
        self.embedder = BaseEmbedder(model_name)
        
        # 버전 관리자 초기화
        try:
            from scripts.ingest.open_law.embedding.pgvector.version_manager import PgEmbeddingVersionManager
            self.version_manager = PgEmbeddingVersionManager(db_url)
        except ImportError:
            import sys
            from pathlib import Path
            _PROJECT_ROOT = Path(__file__).parent.parent.parent.parent
            sys.path.insert(0, str(_PROJECT_ROOT / "scripts"))
            from ingest.open_law.embedding.pgvector.version_manager import PgEmbeddingVersionManager
            self.version_manager = PgEmbeddingVersionManager(db_url)
        
        # 버전 설정
        if version is None:
            active_version = self.version_manager.get_active_version(data_type)
            if active_version:
                self.version = active_version['version']
                self.logger.info(f"활성 버전 사용: version={self.version}, data_type={data_type}")
            else:
                self.version = 1  # 기본값
                self.logger.warning(f"활성 버전 없음, 기본값 사용: version={self.version}")
        else:
            self.version = version
            self.logger.info(f"지정된 버전 사용: version={version}")
    
    def search(
        self,
        query: str,
        top_k: int = 10,
        data_type: Optional[str] = None,  # None이면 초기화 시 설정된 값 사용
        domain: Optional[str] = None,
        section_type: Optional[str] = None,
        similarity_threshold: Optional[float] = None,
        version: Optional[int] = None
    ) -> List[Dict[str, Any]]:
        """
        벡터 유사도 검색 (버전 필터링 포함)
        
        Args:
            query: 검색 쿼리
            top_k: 반환할 결과 수
            data_type: 데이터 타입 ("precedents" or "statutes", None이면 초기화 시 설정값 사용)
            domain: 도메인 필터
            section_type: 섹션 타입 필터 (precedents만)
            similarity_threshold: 유사도 임계값
            version: 임베딩 버전 (None이면 초기화 시 설정된 버전 사용)
        
        Returns:
            검색 결과 리스트
        """
        # 데이터 타입 및 버전 설정
        search_data_type = data_type or self.data_type
        search_version = version or self.version
        
        # 쿼리 임베딩 생성
        query_embedding = self.embedder.encode([query], show_progress=False)[0]
        query_vector_str = "[" + ",".join(map(str, query_embedding.tolist())) + "]"
        
        if search_data_type == "precedents":
            return self._search_precedents(
                query_vector_str,
                top_k,
                domain,
                section_type,
                similarity_threshold,
                search_version
            )
        elif search_data_type == "statutes":
            return self._search_statutes(
                query_vector_str,
                top_k,
                domain,
                similarity_threshold,
                search_version
            )
        else:
            raise ValueError(f"Unknown data_type: {search_data_type}")
    
    def _search_precedents(
        self,
        query_vector_str: str,
        top_k: int,
        domain: Optional[str],
        section_type: Optional[str],
        similarity_threshold: Optional[float],
        version: int
    ) -> List[Dict[str, Any]]:
        """판례 청크 검색 (버전 필터링 포함)"""
        query = """
            SELECT 
                pc.id,
                pc.chunk_content,
                1 - (pc.embedding_vector <-> :query_vector::vector) AS similarity,
                pc.metadata,
                pc.embedding_version,
                pcon.section_type,
                p.case_name,
                p.case_number,
                p.decision_date,
                p.court_name,
                p.domain
            FROM precedent_chunks pc
            JOIN precedent_contents pcon ON pc.precedent_content_id = pcon.id
            JOIN precedents p ON pcon.precedent_id = p.id
            WHERE pc.embedding_vector IS NOT NULL
              AND pc.embedding_version = :version
        """
        
        params = {"query_vector": query_vector_str, "version": version}
        
        if domain:
            query += " AND p.domain = :domain"
            params["domain"] = domain
        
        if section_type:
            query += " AND pcon.section_type = :section_type"
            params["section_type"] = section_type
        
        if similarity_threshold:
            query += " AND (1 - (pc.embedding_vector <-> :query_vector::vector)) >= :threshold"
            params["threshold"] = similarity_threshold
        
        query += """
            ORDER BY pc.embedding_vector <-> :query_vector::vector
            LIMIT :top_k
        """
        params["top_k"] = top_k
        
        results = []
        with self.engine.connect() as conn:
            result = conn.execute(text(query), params)
            for row in result:
                import json
                metadata = json.loads(row.metadata) if row.metadata else {}
                
                results.append({
                    "id": row.id,
                    "chunk_content": row.chunk_content,
                    "similarity": float(row.similarity),
                    "metadata": metadata,
                    "embedding_version": row.embedding_version,
                    "section_type": row.section_type,
                    "case_name": row.case_name,
                    "case_number": row.case_number,
                    "decision_date": str(row.decision_date) if row.decision_date else None,
                    "court_name": row.court_name,
                    "domain": row.domain
                })
        
        return results
    
    def _search_statutes(
        self,
        query_vector_str: str,
        top_k: int,
        domain: Optional[str],
        similarity_threshold: Optional[float],
        version: int
    ) -> List[Dict[str, Any]]:
        """법령 조문 검색 (버전 필터링 포함)"""
        query = """
            SELECT 
                se.id,
                sa.article_content,
                1 - (se.embedding_vector <-> :query_vector::vector) AS similarity,
                se.metadata,
                se.embedding_version,
                sa.article_no,
                sa.article_title,
                s.law_name_kr,
                s.law_abbrv,
                s.domain
            FROM statute_embeddings se
            JOIN statutes_articles sa ON se.article_id = sa.id
            JOIN statutes s ON sa.statute_id = s.id
            WHERE se.embedding_vector IS NOT NULL
              AND se.embedding_version = :version
        """
        
        params = {"query_vector": query_vector_str, "version": version}
        
        if domain:
            query += " AND s.domain = :domain"
            params["domain"] = domain
        
        if similarity_threshold:
            query += " AND (1 - (se.embedding_vector <-> :query_vector::vector)) >= :threshold"
            params["threshold"] = similarity_threshold
        
        query += """
            ORDER BY se.embedding_vector <-> :query_vector::vector
            LIMIT :top_k
        """
        params["top_k"] = top_k
        
        results = []
        with self.engine.connect() as conn:
            result = conn.execute(text(query), params)
            for row in result:
                import json
                metadata = json.loads(row.metadata) if row.metadata else {}
                
                results.append({
                    "id": row.id,
                    "article_content": row.article_content,
                    "similarity": float(row.similarity),
                    "metadata": metadata,
                    "embedding_version": row.embedding_version,
                    "article_no": row.article_no,
                    "article_title": row.article_title,
                    "law_name_kr": row.law_name_kr,
                    "law_abbrv": row.law_abbrv,
                    "domain": row.domain
                })
        
        return results
    
    def hybrid_search(
        self,
        query: str,
        top_k: int = 10,
        data_type: str = "precedents",
        vector_weight: float = 0.7,
        fts_weight: float = 0.3,
        domain: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """
        하이브리드 검색 (벡터 + FTS)
        
        Args:
            query: 검색 쿼리
            top_k: 반환할 결과 수
            data_type: 데이터 타입
            vector_weight: 벡터 검색 가중치
            fts_weight: FTS 검색 가중치
            domain: 도메인 필터
        
        Returns:
            검색 결과 리스트
        """
        # 벡터 검색
        vector_results = self.search(
            query,
            top_k=top_k * 2,  # 더 많은 후보 수집
            data_type=data_type,
            domain=domain
        )
        
        # FTS 검색 (간단한 구현)
        # 실제로는 더 정교한 FTS 검색이 필요할 수 있음
        fts_results = self._fts_search(
            query,
            top_k=top_k * 2,
            data_type=data_type,
            domain=domain
        )
        
        # 결과 통합 및 스코어링
        combined_results = self._combine_results(
            vector_results,
            fts_results,
            vector_weight,
            fts_weight
        )
        
        # 상위 k개 반환
        return combined_results[:top_k]
    
    def _fts_search(
        self,
        query: str,
        top_k: int,
        data_type: str,
        domain: Optional[str]
    ) -> List[Dict[str, Any]]:
        """FTS 검색 (간단한 구현)"""
        # TODO: 실제 FTS 검색 구현
        # 현재는 벡터 검색 결과를 반환 (플레이스홀더)
        return []
    
    def _combine_results(
        self,
        vector_results: List[Dict],
        fts_results: List[Dict],
        vector_weight: float,
        fts_weight: float
    ) -> List[Dict]:
        """검색 결과 통합"""
        # 결과 ID별로 그룹화
        result_map = {}
        
        # 벡터 검색 결과 추가
        for result in vector_results:
            result_id = result["id"]
            if result_id not in result_map:
                result_map[result_id] = result.copy()
                result_map[result_id]["combined_score"] = result["similarity"] * vector_weight
            else:
                result_map[result_id]["combined_score"] += result["similarity"] * vector_weight
        
        # FTS 검색 결과 추가
        for result in fts_results:
            result_id = result["id"]
            if result_id not in result_map:
                result_map[result_id] = result.copy()
                result_map[result_id]["combined_score"] = result.get("fts_score", 0) * fts_weight
            else:
                result_map[result_id]["combined_score"] += result.get("fts_score", 0) * fts_weight
        
        # 스코어 기준 정렬
        combined = list(result_map.values())
        combined.sort(key=lambda x: x["combined_score"], reverse=True)
        
        return combined

