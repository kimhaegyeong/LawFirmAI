# -*- coding: utf-8 -*-
"""
Semantic Search Engine V2
lawfirm_v2.db의 embeddings 테이블을 사용한 벡터 검색 엔진
"""

import logging
try:
    from lawfirm_langgraph.core.utils.logger import get_logger
except ImportError:
    from core.utils.logger import get_logger
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
        def __init__(self, model_name: Optional[str] = None):
            if model_name is None:
                import os
                model_name = os.getenv("EMBEDDING_MODEL")
                if model_name is None:
                    from ..utils.config import Config
                    config = Config()
                    model_name = config.embedding_model
            
            self.model_name = model_name
            self.model = SentenceTransformer(model_name)
            self.dim = self.model.get_sentence_embedding_dimension()

        def encode(self, texts, batch_size=16, normalize=True):
            import numpy as np
            vectors = self.model.encode(texts, batch_size=batch_size, normalize_embeddings=normalize)
            return np.array(vectors, dtype=np.float32)

logger = get_logger(__name__)


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
        self.logger = get_logger(__name__)

        # 모델명이 제공되지 않으면 데이터베이스에서 자동 감지
        if model_name is None:
            # 환경 변수에서 먼저 확인 (법률 특화 SBERT 모델 지원)
            import os
            model_name = os.getenv("EMBEDDING_MODEL")
            
            if model_name is None:
                model_name = self._detect_model_from_database()
                if model_name is None:
                    from ..utils.config import Config
                    config = Config()
                    model_name = config.embedding_model
                    self.logger.warning(f"Could not detect model from database or env, using config default: {model_name}")
            else:
                self.logger.info(f"Using embedding model from environment variable: {model_name}")

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
        try:
            self.embedder = SentenceEmbedder(model_name)
            self.dim = self.embedder.dim
            self.logger.info(f"Embedding model loaded: {model_name}, dim={self.dim}")
        except Exception as e:
            self.logger.error(f"Failed to load embedding model: {e}")
            self.embedder = None
            self.dim = None

        if not Path(db_path).exists():
            self.logger.warning(f"Database {db_path} not found")

        # FAISS 인덱스 로드 또는 빌드
        if FAISS_AVAILABLE and self.embedder:
            if Path(self.index_path).exists():
                self._load_faiss_index()
            else:
                self.logger.info("FAISS index not found, will build on first search")

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

    def search(self,
               query: str,
               k: int = 10,
               source_types: Optional[List[str]] = None,
               similarity_threshold: float = 0.5) -> List[Dict[str, Any]]:
        """
        의미적 벡터 검색 수행

        Args:
            query: 검색 쿼리
            k: 반환할 최대 결과 수
            source_types: 필터링할 소스 타입 목록
            similarity_threshold: 최소 유사도 임계값

        Returns:
            검색 결과 리스트 [{text, score, metadata, ...}]
        """
        if not self.embedder:
            self.logger.error("Embedder not initialized")
            return []

        try:
            # 1. 쿼리 임베딩 생성 (캐시 사용)
            query_vec = self._get_cached_query_vector(query)
            if query_vec is None:
                # 캐시에 없으면 생성
                query_vec = self.embedder.encode([query], batch_size=1, normalize=True)[0]
                self._cache_query_vector(query, query_vec)

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
                
                results.append({
                    "id": f"chunk_{chunk_id}",
                    "text": text,
                    "content": text,  # content 필드 보장
                    "score": float(score),
                    "similarity": float(score),
                    "type": source_type,
                    "source": self._format_source(source_type, source_meta),
                    "metadata": {
                        "chunk_id": chunk_id,
                        "source_type": source_type,
                        "source_id": source_id,
                        "text": text,  # metadata에도 text 포함
                        "content": text,  # metadata에도 content 저장
                        **source_meta
                    },
                    "relevance_score": float(score),
                    "search_type": "semantic"
                })

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
            except:
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
        """소스 정보 포맷팅"""
        if source_type == "statute_article":
            return metadata.get("statute_name", "법령")
        elif source_type == "case_paragraph":
            return f"{metadata.get('court', '')} {metadata.get('doc_id', '')}"
        elif source_type == "decision_paragraph":
            return f"{metadata.get('org', '')} {metadata.get('doc_id', '')}"
        elif source_type == "interpretation_paragraph":
            return f"{metadata.get('org', '')} {metadata.get('title', '')}"
        return "Unknown"
