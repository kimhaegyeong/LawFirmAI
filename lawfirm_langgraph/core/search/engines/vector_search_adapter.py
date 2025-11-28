# -*- coding: utf-8 -*-
"""
Vector Search Adapter
벡터 검색 추상화 레이어
pgvector와 FAISS를 통합 인터페이스로 제공
"""

import os
import sys
import threading
from abc import ABC, abstractmethod
from typing import List, Tuple, Optional, Dict, Any
from pathlib import Path

import numpy as np

try:
    from lawfirm_langgraph.core.utils.logger import get_logger
except ImportError:
    from core.utils.logger import get_logger

logger = get_logger(__name__)

# FAISS 지원 확인
try:
    import faiss
    FAISS_AVAILABLE = True
except ImportError:
    FAISS_AVAILABLE = False
    logger.warning("FAISS not available. Install with: pip install faiss-cpu")

# pgvector 지원 확인
try:
    from pgvector.psycopg2 import register_vector
    PGVECTOR_AVAILABLE = True
except ImportError:
    PGVECTOR_AVAILABLE = False
    logger.warning("pgvector not available. Install with: pip install pgvector")


class VectorSearchAdapter(ABC):
    """벡터 검색 어댑터 인터페이스"""
    
    @abstractmethod
    def search(
        self,
        query_vector: np.ndarray,
        limit: int,
        filters: Optional[Dict[str, Any]] = None
    ) -> List[Tuple[int, float]]:
        """
        벡터 유사도 검색
        
        Args:
            query_vector: 쿼리 벡터 (1D numpy array)
            limit: 반환할 결과 수
            filters: 필터 조건 (예: {'article_id': [1, 2, 3]})
        
        Returns:
            [(id, distance), ...] 리스트 (distance는 작을수록 유사함)
        """
        pass
    
    @abstractmethod
    def add_vectors(
        self,
        vectors: np.ndarray,
        ids: List[int],
        metadata: Optional[List[Dict[str, Any]]] = None
    ):
        """
        벡터 추가
        
        Args:
            vectors: 벡터 배열 (N x D)
            ids: 벡터 ID 리스트
            metadata: 메타데이터 리스트 (선택적)
        """
        pass
    
    @abstractmethod
    def get_vector(self, vector_id: int) -> Optional[np.ndarray]:
        """
        벡터 조회
        
        Args:
            vector_id: 벡터 ID
        
        Returns:
            벡터 (없으면 None)
        """
        pass


# 🔥 개선: 연결별 등록 상태 추적 (클래스 레벨, 스레드 안전)
_registered_connections = set()
_connection_lock = threading.Lock()


class PgVectorAdapter(VectorSearchAdapter):
    """pgvector 기반 벡터 검색"""
    
    def __init__(
        self,
        connection,
        table_name: str = 'statute_embeddings',
        id_column: str = 'article_id',
        vector_column: str = 'embedding_vector'
    ):
        """
        초기화
        
        Args:
            connection: PostgreSQL 연결 객체
            table_name: 임베딩 테이블명
            id_column: ID 컬럼명
            vector_column: 벡터 컬럼명
        """
        if not PGVECTOR_AVAILABLE:
            raise ImportError("pgvector is required. Install with: pip install pgvector")
        
        self.connection = connection
        self.table_name = table_name
        self.id_column = id_column
        self.vector_column = vector_column
        
        # 🔥 개선: 연결 ID 생성 (연결 객체의 고유 식별자)
        connection_id = id(connection)
        
        # pgvector 등록 (안전하게 처리)
        try:
            # 연결이 닫혔는지 확인
            if hasattr(connection, 'closed') and connection.closed:
                raise RuntimeError("Connection is closed")
            
            # pgvector 확장이 설치되어 있는지 확인
            cursor = connection.cursor()
            cursor.execute("SELECT EXISTS(SELECT 1 FROM pg_extension WHERE extname = 'vector')")
            row = cursor.fetchone()
            has_extension = row[0] if isinstance(row, tuple) else (row.get('exists', False) if isinstance(row, dict) else False)
            
            if not has_extension:
                raise RuntimeError("pgvector extension is not installed in the database. Run: CREATE EXTENSION IF NOT EXISTS vector;")
            
            # 🔥 개선: 이미 등록된 연결인지 확인 (스레드 안전)
            with _connection_lock:
                is_registered = connection_id in _registered_connections
            
            if is_registered:
                logger.debug(f"pgvector already registered for connection {connection_id}")
            else:
                # register_vector 호출 (이미 등록되었을 수 있으므로 예외 처리)
                try:
                    register_vector(connection)
                    # 등록 성공 시 추적에 추가
                    with _connection_lock:
                        _registered_connections.add(connection_id)
                    logger.info(f"✅ PgVectorAdapter initialized: table={table_name}")
                except Exception as reg_error:
                    # 이미 등록되었거나 다른 오류인 경우
                    error_str = str(reg_error).lower()
                    if "already" in error_str or "registered" in error_str:
                        # 이미 등록된 경우 추적에 추가
                        with _connection_lock:
                            _registered_connections.add(connection_id)
                        logger.debug(f"pgvector already registered: {reg_error}")
                    elif "vector type not found" in error_str:
                        # 🔥 개선: vector 타입을 찾을 수 없는 경우 - 확장은 설치되어 있지만 타입 인식 실패
                        # 실제로는 작동할 수 있으므로 디버그 레벨로 처리
                        with _connection_lock:
                            _registered_connections.add(connection_id)
                        logger.debug(
                            f"pgvector type not found (extension installed, may work anyway): {reg_error}. "
                            f"Continuing as pgvector extension is installed."
                        )
                    else:
                        # 기타 오류는 경고만 출력하고 계속 진행
                        logger.warning(f"⚠️ pgvector registration warning: {reg_error}, continuing anyway")
        except Exception as e:
            error_msg = str(e).lower()
            if "closed" in error_msg or "extension" in error_msg:
                logger.error(f"❌ Failed to register pgvector: {e}")
                raise
            # 기타 오류는 경고만 출력하고 계속 진행
            logger.warning(f"⚠️ pgvector registration warning: {e}, continuing anyway")
    
    def search(
        self,
        query_vector: np.ndarray,
        limit: int,
        filters: Optional[Dict[str, Any]] = None
    ) -> List[Tuple[int, float]]:
        """벡터 유사도 검색 (코사인 거리)"""
        # 쿼리 벡터를 1D로 변환
        if query_vector.ndim > 1:
            query_vector = query_vector.flatten()
        
        # 필터 조건 구성
        where_clauses = []
        params = []
        
        # embedding_vector IS NOT NULL 조건 추가 (precedent_chunks용)
        where_clauses.append(f"{self.vector_column} IS NOT NULL")
        
        if filters:
            for key, values in filters.items():
                if isinstance(values, list):
                    if len(values) > 0:
                        # PostgreSQL 배열로 변환하여 ANY 절 사용
                        where_clauses.append(f"{key} = ANY(%s::int[])")
                        params.append(values)
                else:
                    where_clauses.append(f"{key} = %s")
                    params.append(values)
        
        where_clause = ""
        if where_clauses:
            where_clause = "WHERE " + " AND ".join(where_clauses)
        
        # 코사인 거리 사용 (<=> 연산자)
        # pgvector의 <=> 연산자는 코사인 거리를 반환 (0에 가까울수록 유사)
        query = f"""
            SELECT {self.id_column}, 
                   {self.vector_column} <=> %s::vector AS distance
            FROM {self.table_name}
            {where_clause}
            ORDER BY distance
            LIMIT %s
        """
        
        # 파라미터 순서: query_vector, filters..., limit
        params = [query_vector.tolist()] + params + [limit]
        
        cursor = self.connection.cursor()
        try:
            cursor.execute(query, params)
            results = []
            for row in cursor.fetchall():
                # row가 dict인 경우와 tuple인 경우 모두 처리
                if isinstance(row, dict):
                    vector_id = row[self.id_column]
                    distance = float(row['distance'])
                else:
                    # tuple인 경우 컬럼 순서대로
                    vector_id = row[0]
                    distance = float(row[1])
                results.append((vector_id, distance))
            return results
        except Exception as e:
            logger.error(f"PgVector search error: {e}")
            raise
    
    def add_vectors(
        self,
        vectors: np.ndarray,
        ids: List[int],
        metadata: Optional[List[Dict[str, Any]]] = None
    ):
        """벡터 추가"""
        if vectors.ndim != 2:
            raise ValueError("vectors must be 2D array (N x D)")
        
        if len(vectors) != len(ids):
            raise ValueError("vectors and ids must have the same length")
        
        cursor = self.connection.cursor()
        try:
            for i, (vector, vector_id) in enumerate(zip(vectors, ids)):
                meta = metadata[i] if metadata else {}
                
                # 벡터를 리스트로 변환
                vector_list = vector.tolist()
                
                query = f"""
                    INSERT INTO {self.table_name} ({self.id_column}, {self.vector_column}, metadata)
                    VALUES (%s, %s, %s)
                    ON CONFLICT ({self.id_column}) DO UPDATE
                    SET {self.vector_column} = EXCLUDED.{self.vector_column},
                        metadata = EXCLUDED.metadata
                """
                
                import json
                cursor.execute(query, (vector_id, vector_list, json.dumps(meta)))
            
            self.connection.commit()
            logger.info(f"Added {len(vectors)} vectors to {self.table_name}")
        except Exception as e:
            self.connection.rollback()
            logger.error(f"Failed to add vectors: {e}")
            raise
    
    def get_vector(self, vector_id: int) -> Optional[np.ndarray]:
        """벡터 조회"""
        query = f"""
            SELECT {self.vector_column}
            FROM {self.table_name}
            WHERE {self.id_column} = %s
        """
        
        cursor = self.connection.cursor()
        try:
            cursor.execute(query, (vector_id,))
            row = cursor.fetchone()
            if row:
                vector = row[self.vector_column]
                if isinstance(vector, list):
                    return np.array(vector, dtype=np.float32)
                return vector
            return None
        except Exception as e:
            logger.error(f"Failed to get vector: {e}")
            return None


class FaissAdapter(VectorSearchAdapter):
    """FAISS 기반 벡터 검색"""
    
    def __init__(
        self,
        index_path: str,
        vector_loader: Optional[callable] = None,
        dimension: Optional[int] = None
    ):
        """
        초기화
        
        Args:
            index_path: FAISS 인덱스 파일 경로
            vector_loader: 벡터 로더 함수 (필요시)
            dimension: 벡터 차원 (인덱스가 없을 때 생성용)
        """
        if not FAISS_AVAILABLE:
            raise ImportError("FAISS is required. Install with: pip install faiss-cpu")
        
        self.index_path = Path(index_path)
        self.vector_loader = vector_loader
        self.dimension = dimension
        self.index = None
        self._load_index()
        
        logger.info(f"FaissAdapter initialized: index_path={index_path}")
    
    def _load_index(self):
        """FAISS 인덱스 로드"""
        if self.index_path.exists():
            try:
                self.index = faiss.read_index(str(self.index_path))
                logger.info(f"Loaded FAISS index: {self.index_path}")
            except Exception as e:
                logger.error(f"Failed to load FAISS index: {e}")
                raise
        else:
            # 인덱스가 없으면 생성
            if self.dimension is None:
                raise ValueError("dimension is required when creating new index")
            
            # 기본 인덱스 생성 (L2 거리)
            self.index = faiss.IndexFlatL2(self.dimension)
            logger.info(f"Created new FAISS index: dimension={self.dimension}")
    
    def search(
        self,
        query_vector: np.ndarray,
        limit: int,
        filters: Optional[Dict[str, Any]] = None
    ) -> List[Tuple[int, float]]:
        """벡터 유사도 검색 (L2 거리)"""
        if self.index is None:
            raise RuntimeError("FAISS index not loaded")
        
        # 쿼리 벡터를 2D로 변환 (1 x D)
        if query_vector.ndim == 1:
            query_vector = query_vector.reshape(1, -1)
        
        # float32로 변환
        query_vector = query_vector.astype('float32')
        
        # 검색
        distances, indices = self.index.search(query_vector, limit)
        
        results = []
        for idx, dist in zip(indices[0], distances[0]):
            if idx != -1:  # FAISS의 -1은 유효하지 않은 결과
                results.append((int(idx), float(dist)))
        
        # 필터 적용 (필요시)
        if filters:
            results = self._apply_filters(results, filters)
        
        return results
    
    def _apply_filters(
        self,
        results: List[Tuple[int, float]],
        filters: Dict[str, Any]
    ) -> List[Tuple[int, float]]:
        """필터 적용 (FAISS는 인덱스 기반이므로 사후 필터링)"""
        # FAISS는 인덱스 기반이므로 필터링이 제한적
        # 필요시 vector_loader를 통해 메타데이터 확인
        if self.vector_loader:
            filtered_results = []
            for vector_id, distance in results:
                # vector_loader를 통해 메타데이터 확인
                # 여기서는 간단히 ID만 확인
                if 'id' in filters:
                    if vector_id in filters['id']:
                        filtered_results.append((vector_id, distance))
                else:
                    filtered_results.append((vector_id, distance))
            return filtered_results
        return results
    
    def add_vectors(
        self,
        vectors: np.ndarray,
        ids: List[int],
        metadata: Optional[List[Dict[str, Any]]] = None
    ):
        """벡터 추가"""
        if self.index is None:
            raise RuntimeError("FAISS index not loaded")
        
        if vectors.ndim != 2:
            raise ValueError("vectors must be 2D array (N x D)")
        
        # float32로 변환
        vectors = vectors.astype('float32')
        
        # 인덱스에 추가
        self.index.add(vectors)
        
        # 인덱스 저장
        faiss.write_index(self.index, str(self.index_path))
        
        logger.info(f"Added {len(vectors)} vectors to FAISS index")
    
    def get_vector(self, vector_id: int) -> Optional[np.ndarray]:
        """벡터 조회 (FAISS는 인덱스 기반이므로 vector_loader 필요)"""
        if self.vector_loader:
            return self.vector_loader(vector_id)
        else:
            logger.warning("vector_loader not set, cannot get vector by ID")
            return None


class VectorSearchFactory:
    """벡터 검색 어댑터 팩토리"""
    
    @staticmethod
    def create(
        method: str,
        connection=None,
        table_name: str = 'statute_embeddings',
        index_path: str = None,
        vector_loader: Optional[callable] = None,
        dimension: Optional[int] = None,
        **kwargs
    ) -> VectorSearchAdapter:
        """
        벡터 검색 어댑터 생성
        
        Args:
            method: 'pgvector' 또는 'faiss'
            connection: PostgreSQL 연결 (pgvector용)
            table_name: 테이블명 (pgvector용)
            index_path: FAISS 인덱스 경로 (faiss용)
            vector_loader: 벡터 로더 함수 (faiss용)
            dimension: 벡터 차원 (faiss용, 인덱스 생성 시)
            **kwargs: 추가 파라미터
        
        Returns:
            VectorSearchAdapter 인스턴스
        """
        method = method.lower()
        
        if method == 'pgvector':
            if connection is None:
                raise ValueError("connection is required for pgvector")
            return PgVectorAdapter(
                connection=connection,
                table_name=table_name,
                **kwargs
            )
        
        elif method == 'faiss':
            if index_path is None:
                raise ValueError("index_path is required for faiss")
            return FaissAdapter(
                index_path=index_path,
                vector_loader=vector_loader,
                dimension=dimension
            )
        
        else:
            raise ValueError(f"Unknown vector search method: {method}. Use 'pgvector' or 'faiss'")

