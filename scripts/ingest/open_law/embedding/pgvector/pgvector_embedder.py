#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
pgvector 임베딩 생성기
PostgreSQL precedent_chunks 및 statute_embeddings 테이블에 임베딩 저장
"""

import argparse
import gc
import logging
import os
import sys
from pathlib import Path
from typing import List, Dict, Any, Optional
from datetime import datetime
import json

import numpy as np
from sqlalchemy import create_engine, text
from sqlalchemy.pool import QueuePool

# 프로젝트 루트를 sys.path에 추가
_CURRENT_FILE = Path(__file__).resolve()
# pgvector_embedder.py -> pgvector -> embedding -> open_law -> ingest -> scripts -> 프로젝트 루트
_PROJECT_ROOT = _CURRENT_FILE.parents[5]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

# .env 파일 로드
try:
    from utils.env_loader import ensure_env_loaded
    ensure_env_loaded(_PROJECT_ROOT)
except ImportError:
    try:
        from dotenv import load_dotenv
        scripts_env = _PROJECT_ROOT / "scripts" / ".env"
        if scripts_env.exists():
            load_dotenv(dotenv_path=str(scripts_env), override=True)
        root_env = _PROJECT_ROOT / ".env"
        if root_env.exists():
            load_dotenv(dotenv_path=str(root_env), override=False)
    except ImportError:
        pass

# 공통 모듈 임포트
try:
    from scripts.ingest.open_law.embedding.data_loader import PostgreSQLDataLoader
    from scripts.ingest.open_law.embedding.base_embedder import BaseEmbedder
except ImportError:
    # 폴백: scripts 디렉토리를 sys.path에 추가
    scripts_dir = _PROJECT_ROOT / "scripts"
    if str(scripts_dir) not in sys.path:
        sys.path.insert(0, str(scripts_dir))
    from ingest.open_law.embedding.data_loader import PostgreSQLDataLoader
    from ingest.open_law.embedding.base_embedder import BaseEmbedder

# 데이터베이스 URL 빌드
try:
    from scripts.ingest.open_law.utils import build_database_url
except ImportError:
    try:
        from ingest.open_law.utils import build_database_url
    except ImportError:
        from urllib.parse import quote_plus
        def build_database_url():
            db_url = os.getenv('DATABASE_URL')
            if db_url:
                return db_url
            host = os.getenv('POSTGRES_HOST', 'localhost')
            port = os.getenv('POSTGRES_PORT', '5432')
            db = os.getenv('POSTGRES_DB')
            user = os.getenv('POSTGRES_USER')
            password = os.getenv('POSTGRES_PASSWORD')
            if db and user and password:
                encoded_password = quote_plus(password)
                return f"postgresql://{user}:{encoded_password}@{host}:{port}/{db}"
            return None

# 로깅 설정
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class PgVectorEmbedder:
    """pgvector 임베딩 생성기"""
    
    def __init__(
        self, 
        db_url: str, 
        model_name: str = "jhgan/ko-sroberta-multitask",
        version: Optional[int] = None,
        chunking_strategy: Optional[str] = None
    ):
        """
        pgvector 임베딩 생성기 초기화
        
        Args:
            db_url: PostgreSQL 데이터베이스 URL
            model_name: 임베딩 모델 이름
            version: 임베딩 버전 (None이면 활성 버전 사용)
            chunking_strategy: 청킹 전략 (예: 'article', '512-token')
        """
        self.db_url = db_url
        # lawfirm_langgraph 패키지 import 방지 (SQLite 오류 방지)
        # 표준 logging 모듈 직접 사용
        self.logger = logging.getLogger(__name__)
        
        # 데이터 로더 및 임베딩 생성기 초기화
        self.data_loader = PostgreSQLDataLoader(db_url)
        self.embedder = BaseEmbedder(model_name)
        self.dimension = self.embedder.get_dimension()
        self.model_name = model_name
        self.chunking_strategy = chunking_strategy
        
        # 데이터베이스 엔진
        self.engine = create_engine(
            db_url,
            poolclass=QueuePool,
            pool_pre_ping=True,
            pool_size=5,
            max_overflow=10,
            echo=False
        )
        
        # 버전 관리자 초기화
        try:
            from scripts.ingest.open_law.embedding.pgvector.version_manager import PgEmbeddingVersionManager
            self.version_manager = PgEmbeddingVersionManager(db_url)
        except ImportError:
            sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent.parent))
            from ingest.open_law.embedding.pgvector.version_manager import PgEmbeddingVersionManager
            self.version_manager = PgEmbeddingVersionManager(db_url)
        
        # pgvector 확장 확인
        self._check_pgvector_extension()
    
    def _check_pgvector_extension(self):
        """pgvector 확장 설치 확인"""
        try:
            with self.engine.connect() as conn:
                result = conn.execute(
                    text("SELECT * FROM pg_extension WHERE extname = 'vector'")
                )
                if result.fetchone():
                    self.logger.info("pgvector 확장이 설치되어 있습니다.")
                else:
                    self.logger.warning(
                        "pgvector 확장이 설치되지 않았습니다. "
                        "CREATE EXTENSION vector; 실행이 필요합니다."
                    )
        except Exception as e:
            self.logger.warning(f"pgvector 확장 확인 실패: {e}")
    
    def generate_precedent_embeddings(
        self,
        batch_size: int = 100,
        limit: Optional[int] = None,
        domain: Optional[str] = None,
        version: Optional[int] = None,
        chunking_strategy: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        판례 청크 임베딩 생성
        
        Args:
            batch_size: 배치 크기
            limit: 최대 처리 개수
            domain: 도메인 필터
            version: 임베딩 버전 (None이면 활성 버전 또는 새 버전 생성)
            chunking_strategy: 청킹 전략 (None이면 기본값 사용)
        
        Returns:
            처리 결과 통계
        """
        self.logger.info("판례 청크 임베딩 생성 시작")
        
        # 버전 관리: 버전 조회 또는 생성
        if version is None:
            active_version = self.version_manager.get_active_version('precedents')
            if active_version:
                version = active_version['version']
                self.logger.info(f"활성 버전 사용: version={version}")
            else:
                version = self.version_manager.get_next_version('precedents')
                self.logger.info(f"새 버전 생성: version={version}")
        
        # 버전 등록 또는 조회
        version_id = self.version_manager.get_or_create_version(
            version=version,
            model_name=self.model_name,
            dim=self.dimension,
            data_type='precedents',
            chunking_strategy=chunking_strategy or self.chunking_strategy or '512-token',
            description=f"판례 청크 임베딩 - 모델: {self.model_name}",
            metadata={
                "model_name": self.model_name,
                "dimension": self.dimension,
                "chunking_strategy": chunking_strategy or self.chunking_strategy or '512-token',
                "created_at": datetime.now().isoformat()
            },
            set_active=True
        )
        
        self.current_version = version
        self.current_version_id = version_id
        
        stats = {
            "total_processed": 0,
            "total_embedded": 0,
            "total_skipped": 0,  # 이미 임베딩된 항목
            "total_failed": 0,
            "errors": []
        }
        
        offset = 0
        
        with self.engine.connect() as conn:
            trans = conn.begin()
            
            try:
                while True:
                    # 데이터 로드 (이미 임베딩된 항목 제외)
                    chunks = self.data_loader.load_precedent_chunks(
                        domain=domain,
                        limit=batch_size,
                        offset=offset,
                        skip_embedded=True
                    )
                    
                    if not chunks:
                        break
                    
                    # 저장 전 중복 체크 (이중 방어) - 현재 버전 기준
                    chunks_to_process = []
                    for chunk in chunks:
                        if self._check_precedent_embedding_exists(conn, chunk["id"], self.current_version):
                            stats["total_skipped"] += 1
                            self.logger.debug(f"이미 임베딩 존재 (chunk_id={chunk['id']}, version={self.current_version}), 건너뜀")
                        else:
                            chunks_to_process.append(chunk)
                    
                    if not chunks_to_process:
                        # 모두 이미 임베딩된 경우
                        self.logger.debug(f"배치의 모든 항목이 이미 임베딩됨, 다음 배치로 이동")
                        offset += batch_size
                        stats["total_processed"] += len(chunks)
                        continue
                    
                    # 텍스트 추출
                    texts = [chunk["chunk_content"] for chunk in chunks_to_process]
                    chunk_ids = [chunk["id"] for chunk in chunks_to_process]
                    
                    # 임베딩 생성
                    try:
                        embeddings = self.embedder.encode(
                            texts,
                            batch_size=len(texts),
                            show_progress=False
                        )
                        
                        # 임베딩 저장 (중복 방지 로직 포함)
                        for chunk_id, embedding in zip(chunk_ids, embeddings):
                            try:
                                # 저장 전 최종 중복 체크 (현재 버전 기준)
                                if self._check_precedent_embedding_exists(conn, chunk_id, self.current_version):
                                    stats["total_skipped"] += 1
                                    self.logger.debug(f"저장 직전 중복 발견 (chunk_id={chunk_id}, version={self.current_version}), 건너뜀")
                                    continue
                                
                                self._save_precedent_embedding(
                                    conn,
                                    chunk_id,
                                    embedding,
                                    version=self.current_version
                                )
                                stats["total_embedded"] += 1
                            except Exception as e:
                                # 중복 관련 에러 처리
                                error_str = str(e).lower()
                                if "duplicate" in error_str or "unique" in error_str or "already exists" in error_str:
                                    stats["total_skipped"] += 1
                                    self.logger.warning(f"중복 감지로 건너뜀 (chunk_id={chunk_id}): {e}")
                                else:
                                    self.logger.error(f"임베딩 저장 실패 (chunk_id={chunk_id}): {e}")
                                    stats["total_failed"] += 1
                                    stats["errors"].append(f"chunk_id={chunk_id}: {e}")
                        
                        stats["total_processed"] += len(chunks)
                        
                        # 배치 커밋
                        trans.commit()
                        trans = conn.begin()
                        
                        # 메모리 최적화: 큰 변수 명시적 삭제
                        del embeddings
                        del texts
                        del chunk_ids
                        del chunks_to_process
                        del chunks
                        gc.collect()
                        
                        self.logger.info(
                            f"진행 상황: {stats['total_processed']}개 처리, "
                            f"{stats['total_embedded']}개 임베딩 저장, "
                            f"{stats['total_skipped']}개 건너뜀, "
                            f"{stats['total_failed']}개 실패"
                        )
                        
                        offset += batch_size
                        
                        if limit and stats["total_processed"] >= limit:
                            break
                    
                    except Exception as e:
                        self.logger.error(f"배치 처리 실패: {e}")
                        stats["total_failed"] += len(chunks_to_process)
                        stats["errors"].append(f"배치 offset={offset}: {e}")
                        trans.rollback()
                        trans = conn.begin()
                        
                        # 메모리 최적화: 에러 발생 시에도 메모리 정리
                        if 'embeddings' in locals():
                            del embeddings
                        if 'texts' in locals():
                            del texts
                        if 'chunk_ids' in locals():
                            del chunk_ids
                        if 'chunks_to_process' in locals():
                            del chunks_to_process
                        if 'chunks' in locals():
                            del chunks
                        gc.collect()
                        
                        offset += batch_size
                        continue
                
                # 최종 커밋
                trans.commit()
                
                self.logger.info(
                    f"판례 청크 임베딩 생성 완료: "
                    f"{stats['total_embedded']}개 저장, "
                    f"{stats['total_skipped']}개 건너뜀, "
                    f"{stats['total_failed']}개 실패"
                )
                
                return stats
            
            except Exception as e:
                trans.rollback()
                self.logger.error(f"임베딩 생성 실패: {e}")
                raise
    
    def generate_statute_embeddings(
        self,
        batch_size: int = 100,
        limit: Optional[int] = None,
        domain: Optional[str] = None,
        version: Optional[int] = None,
        chunking_strategy: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        법령 조문 임베딩 생성
        
        Args:
            batch_size: 배치 크기
            limit: 최대 처리 개수
            domain: 도메인 필터
            version: 임베딩 버전 (None이면 활성 버전 또는 새 버전 생성)
            chunking_strategy: 청킹 전략 (None이면 기본값 사용)
        
        Returns:
            처리 결과 통계
        """
        self.logger.info("법령 조문 임베딩 생성 시작")
        
        # statute_embeddings 테이블 생성
        self._create_statute_embeddings_table()
        
        # 버전 관리: 버전 조회 또는 생성
        if version is None:
            active_version = self.version_manager.get_active_version('statutes')
            if active_version:
                version = active_version['version']
                self.logger.info(f"활성 버전 사용: version={version}")
            else:
                version = self.version_manager.get_next_version('statutes')
                self.logger.info(f"새 버전 생성: version={version}")
        
        # 버전 등록 또는 조회
        version_id = self.version_manager.get_or_create_version(
            version=version,
            model_name=self.model_name,
            dim=self.dimension,
            data_type='statutes',
            chunking_strategy=chunking_strategy or self.chunking_strategy or 'article',
            description=f"법령 조문 임베딩 - 모델: {self.model_name}",
            metadata={
                "model_name": self.model_name,
                "dimension": self.dimension,
                "chunking_strategy": chunking_strategy or self.chunking_strategy or 'article',
                "created_at": datetime.now().isoformat()
            },
            set_active=True
        )
        
        self.current_version = version
        self.current_version_id = version_id
        
        stats = {
            "total_processed": 0,
            "total_embedded": 0,
            "total_skipped": 0,  # 이미 임베딩된 항목
            "total_failed": 0,
            "errors": []
        }
        
        offset = 0
        
        with self.engine.connect() as conn:
            trans = conn.begin()
            
            try:
                while True:
                    # 데이터 로드 (이미 임베딩된 항목 제외)
                    articles = self.data_loader.load_statute_articles(
                        domain=domain,
                        limit=batch_size,
                        offset=offset,
                        skip_embedded=True
                    )
                    
                    if not articles:
                        break
                    
                    # 저장 전 중복 체크 (이중 방어) - 현재 버전 기준
                    articles_to_process = []
                    for article in articles:
                        if self._check_statute_embedding_exists(conn, article["id"], self.current_version):
                            stats["total_skipped"] += 1
                            self.logger.debug(f"이미 임베딩 존재 (article_id={article['id']}, version={self.current_version}), 건너뜀")
                        else:
                            articles_to_process.append(article)
                    
                    if not articles_to_process:
                        # 모두 이미 임베딩된 경우
                        self.logger.debug(f"배치의 모든 항목이 이미 임베딩됨, 다음 배치로 이동")
                        offset += batch_size
                        stats["total_processed"] += len(articles)
                        continue
                    
                    # 텍스트 추출 (조문 내용)
                    texts = [article["article_content"] for article in articles_to_process]
                    article_ids = [article["id"] for article in articles_to_process]
                    
                    # 임베딩 생성
                    try:
                        embeddings = self.embedder.encode(
                            texts,
                            batch_size=len(texts),
                            show_progress=False
                        )
                        
                        # 임베딩 저장 (중복 방지 로직 포함)
                        for article_id, embedding, article in zip(article_ids, embeddings, articles_to_process):
                            try:
                                # 저장 전 최종 중복 체크 (현재 버전 기준)
                                if self._check_statute_embedding_exists(conn, article_id, self.current_version):
                                    stats["total_skipped"] += 1
                                    self.logger.debug(f"저장 직전 중복 발견 (article_id={article_id}, version={self.current_version}), 건너뜀")
                                    continue
                                
                                self._save_statute_embedding(
                                    conn,
                                    article_id,
                                    embedding,
                                    article,
                                    version=self.current_version
                                )
                                stats["total_embedded"] += 1
                            except Exception as e:
                                # 트랜잭션 오류인 경우 롤백 후 계속
                                error_str = str(e).lower()
                                if "infailedsqltransaction" in error_str or "transaction is aborted" in error_str:
                                    try:
                                        trans.rollback()
                                        trans = conn.begin()
                                    except:
                                        pass
                                    stats["total_failed"] += 1
                                    self.logger.error(f"트랜잭션 오류로 건너뜀 (article_id={article_id}): {str(e)[:100]}")
                                # Unique constraint violation 등 중복 관련 에러 처리
                                elif "duplicate" in error_str or "unique" in error_str or "already exists" in error_str:
                                    stats["total_skipped"] += 1
                                    self.logger.warning(f"중복 감지로 건너뜀 (article_id={article_id}): {e}")
                                else:
                                    self.logger.error(f"임베딩 저장 실패 (article_id={article_id}): {e}")
                                    stats["total_failed"] += 1
                                    stats["errors"].append(f"article_id={article_id}: {e}")
                        
                        stats["total_processed"] += len(articles)
                        
                        # 배치 커밋
                        trans.commit()
                        trans = conn.begin()
                        
                        # 메모리 최적화: 큰 변수 명시적 삭제
                        del embeddings
                        del texts
                        del article_ids
                        del articles_to_process
                        del articles
                        gc.collect()
                        
                        self.logger.info(
                            f"진행 상황: {stats['total_processed']}개 처리, "
                            f"{stats['total_embedded']}개 임베딩 저장, "
                            f"{stats['total_skipped']}개 건너뜀, "
                            f"{stats['total_failed']}개 실패"
                        )
                        
                        offset += batch_size
                        
                        if limit and stats["total_processed"] >= limit:
                            break
                    
                    except Exception as e:
                        self.logger.error(f"배치 처리 실패: {e}")
                        stats["total_failed"] += len(articles_to_process)
                        stats["errors"].append(f"배치 offset={offset}: {e}")
                        trans.rollback()
                        trans = conn.begin()
                        
                        # 메모리 최적화: 에러 발생 시에도 메모리 정리
                        if 'embeddings' in locals():
                            del embeddings
                        if 'texts' in locals():
                            del texts
                        if 'article_ids' in locals():
                            del article_ids
                        if 'articles_to_process' in locals():
                            del articles_to_process
                        if 'articles' in locals():
                            del articles
                        gc.collect()
                        
                        offset += batch_size
                        continue
                
                # 최종 커밋
                trans.commit()
                
                self.logger.info(
                    f"법령 조문 임베딩 생성 완료: "
                    f"{stats['total_embedded']}개 저장, "
                    f"{stats['total_skipped']}개 건너뜀, "
                    f"{stats['total_failed']}개 실패"
                )
                
                return stats
            
            except Exception as e:
                trans.rollback()
                self.logger.error(f"임베딩 생성 실패: {e}")
                raise
    
    def _create_statute_embeddings_table(self):
        """statute_embeddings 테이블 생성"""
        create_table_sql = """
            CREATE TABLE IF NOT EXISTS statute_embeddings (
                id SERIAL PRIMARY KEY,
                article_id INTEGER NOT NULL REFERENCES statutes_articles(id) ON DELETE CASCADE,
                embedding_vector VECTOR(768),
                embedding_version INTEGER NOT NULL DEFAULT 1,
                metadata JSONB,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                UNIQUE(article_id, embedding_version)
            );
            
            CREATE INDEX IF NOT EXISTS idx_statute_embeddings_article_id 
                ON statute_embeddings(article_id);
            
            CREATE INDEX IF NOT EXISTS idx_statute_embeddings_version 
                ON statute_embeddings(embedding_version);
            
            CREATE INDEX IF NOT EXISTS idx_statute_embeddings_vector 
                ON statute_embeddings 
                USING ivfflat (embedding_vector vector_cosine_ops)
                WITH (lists = 100);
        """
        
        try:
            with self.engine.connect() as conn:
                trans = conn.begin()
                try:
                    conn.execute(text(create_table_sql))
                    trans.commit()
                    self.logger.info("statute_embeddings 테이블 생성 완료")
                except Exception as e:
                    trans.rollback()
                    # 테이블이 이미 존재하는 경우 무시
                    if "already exists" not in str(e).lower():
                        self.logger.warning(f"테이블 생성 중 오류 (무시 가능): {e}")
                
                # 기존 테이블에 UNIQUE 제약조건 추가 (없는 경우)
                try:
                    check_constraint_sql = """
                        SELECT 1 FROM pg_constraint 
                        WHERE conname = 'statute_embeddings_article_id_version_key'
                    """
                    result = conn.execute(text(check_constraint_sql))
                    if not result.fetchone():
                        # 제약조건 추가
                        add_constraint_sql = """
                            ALTER TABLE statute_embeddings 
                            ADD CONSTRAINT statute_embeddings_article_id_version_key 
                            UNIQUE(article_id, embedding_version)
                        """
                        trans = conn.begin()
                        try:
                            conn.execute(text(add_constraint_sql))
                            trans.commit()
                            self.logger.info("UNIQUE 제약조건 추가 완료: (article_id, embedding_version)")
                        except Exception as e:
                            trans.rollback()
                            # 이미 존재하거나 다른 오류
                            if "already exists" not in str(e).lower() and "duplicate" not in str(e).lower():
                                self.logger.warning(f"제약조건 추가 중 오류 (무시 가능): {e}")
                except Exception as e:
                    self.logger.warning(f"제약조건 확인 중 오류 (무시 가능): {e}")
        except Exception as e:
            self.logger.warning(f"테이블 생성 확인 실패: {e}")
    
    def _check_precedent_embedding_exists(
        self,
        conn,
        chunk_id: int,
        version: Optional[int] = None
    ) -> bool:
        """판례 청크 임베딩 존재 여부 확인"""
        if version is not None:
            check_sql = """
                SELECT embedding_vector IS NOT NULL
                FROM precedent_chunks
                WHERE id = :chunk_id
                  AND embedding_version = :version
            """
            result = conn.execute(
                text(check_sql), 
                {"chunk_id": chunk_id, "version": version}
            )
        else:
            check_sql = """
                SELECT embedding_vector IS NOT NULL
                FROM precedent_chunks
                WHERE id = :chunk_id
            """
            result = conn.execute(text(check_sql), {"chunk_id": chunk_id})
        row = result.fetchone()
        return row[0] if row else False
    
    def _save_precedent_embedding(
        self,
        conn,
        chunk_id: int,
        embedding: np.ndarray,
        version: int
    ):
        """판례 청크 임베딩 저장 (중복 방지 및 버전 관리 포함)"""
        # numpy 배열을 PostgreSQL VECTOR 형식으로 변환
        embedding_str = "[" + ",".join(map(str, embedding.tolist())) + "]"
        
        # 버전 정보 포함하여 업데이트
        # 같은 chunk_id에 다른 버전이 있는 경우도 처리
        update_sql = """
            UPDATE precedent_chunks
            SET embedding_vector = CAST(:embedding_vector AS vector),
                embedding_version = :embedding_version
            WHERE id = :chunk_id
              AND (embedding_vector IS NULL 
                   OR embedding_version != :embedding_version)
        """
        
        result = conn.execute(
            text(update_sql),
            {
                "chunk_id": chunk_id,
                "embedding_vector": embedding_str,
                "embedding_version": version
            }
        )
        
        # 실제로 업데이트되었는지 확인
        if result.rowcount == 0:
            # 같은 버전의 임베딩이 이미 존재하는 경우
            raise ValueError(f"이미 임베딩이 존재합니다 (chunk_id={chunk_id}, version={version})")
    
    def _check_statute_embedding_exists(
        self,
        conn,
        article_id: int,
        version: Optional[int] = None
    ) -> bool:
        """법령 조문 임베딩 존재 여부 확인"""
        if version is not None:
            check_sql = """
                SELECT EXISTS(
                    SELECT 1 FROM statute_embeddings
                    WHERE article_id = :article_id
                      AND embedding_version = :version
                )
            """
            result = conn.execute(
                text(check_sql), 
                {"article_id": article_id, "version": version}
            )
        else:
            check_sql = """
                SELECT EXISTS(
                    SELECT 1 FROM statute_embeddings
                    WHERE article_id = :article_id
                )
            """
            result = conn.execute(text(check_sql), {"article_id": article_id})
        return result.scalar()
    
    def _save_statute_embedding(
        self,
        conn,
        article_id: int,
        embedding: np.ndarray,
        article: Dict[str, Any],
        version: int
    ):
        """법령 조문 임베딩 저장 (중복 방지 및 버전 관리 포함)"""
        # numpy 배열을 PostgreSQL VECTOR 형식으로 변환
        embedding_str = "[" + ",".join(map(str, embedding.tolist())) + "]"
        
        # 메타데이터 구성 (버전 정보 포함)
        metadata = {
            "article_id": article_id,
            "statute_id": article["statute_id"],
            "article_no": article["article_no"],
            "law_name_kr": article["law_name_kr"],
            "law_abbrv": article["law_abbrv"],
            "domain": article["domain"],
            "embedding_version": version,
            "model_name": self.model_name
        }
        
        # ON CONFLICT를 사용하여 중복 방지 (이중 방어)
        # 같은 article_id에 다른 버전이 있는 경우도 처리
        insert_sql = """
            INSERT INTO statute_embeddings (
                article_id,
                embedding_vector,
                embedding_version,
                metadata
            ) VALUES (
                :article_id,
                CAST(:embedding_vector AS vector),
                :embedding_version,
                CAST(:metadata AS jsonb)
            )
            ON CONFLICT (article_id, embedding_version) DO UPDATE
            SET embedding_vector = EXCLUDED.embedding_vector,
                metadata = EXCLUDED.metadata
        """
        
        result = conn.execute(
            text(insert_sql),
            {
                "article_id": article_id,
                "embedding_vector": embedding_str,
                "embedding_version": version,
                "metadata": json.dumps(metadata, ensure_ascii=False)
            }
        )
        
        # 실제로 삽입/업데이트되었는지 확인
        if result.rowcount == 0:
            # 같은 버전의 임베딩이 이미 존재하는 경우
            raise ValueError(f"이미 임베딩이 존재합니다 (article_id={article_id}, version={version})")


def main():
    parser = argparse.ArgumentParser(description='pgvector 임베딩 생성')
    parser.add_argument(
        '--db',
        default=build_database_url() or os.getenv('DATABASE_URL'),
        help='PostgreSQL 데이터베이스 URL'
    )
    parser.add_argument(
        '--data-type',
        choices=['precedents', 'statutes', 'both'],
        default='precedents',
        help='임베딩 생성할 데이터 타입'
    )
    parser.add_argument(
        '--batch-size',
        type=int,
        default=100,
        help='배치 크기'
    )
    parser.add_argument(
        '--limit',
        type=int,
        default=None,
        help='최대 처리 개수'
    )
    parser.add_argument(
        '--domain',
        choices=['civil_law', 'criminal_law', 'administrative_law'],
        default=None,
        help='도메인 필터'
    )
    parser.add_argument(
        '--model',
        default='jhgan/ko-sroberta-multitask',
        help='임베딩 모델 이름'
    )
    
    args = parser.parse_args()
    
    if not args.db:
        logger.error("--db 인자 또는 DATABASE_URL 환경변수가 필요합니다.")
        return
    
    try:
        embedder = PgVectorEmbedder(args.db, model_name=args.model)
        
        if args.data_type in ['precedents', 'both']:
            logger.info("판례 청크 임베딩 생성 시작")
            results = embedder.generate_precedent_embeddings(
                batch_size=args.batch_size,
                limit=args.limit,
                domain=args.domain
            )
            logger.info(f"판례 청크 임베딩 생성 완료: {results}")
        
        if args.data_type in ['statutes', 'both']:
            logger.info("법령 조문 임베딩 생성 시작")
            results = embedder.generate_statute_embeddings(
                batch_size=args.batch_size,
                limit=args.limit,
                domain=args.domain
            )
            logger.info(f"법령 조문 임베딩 생성 완료: {results}")
    
    except Exception as e:
        logger.error(f"스크립트 실행 실패: {e}")
        sys.exit(1)


if __name__ == '__main__':
    main()

