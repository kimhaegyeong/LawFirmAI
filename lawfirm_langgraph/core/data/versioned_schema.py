# -*- coding: utf-8 -*-
"""
Versioned Schema Management
버전 분리(DB 파일/테이블) 및 체크포인트 테이블 제공
"""

import sqlite3
import os
from contextlib import contextmanager
from pathlib import Path
from typing import Optional

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


def get_versioned_db_path(base_dir: str, corpus_version: str) -> Path:
    base = Path(base_dir)
    base.mkdir(parents=True, exist_ok=True)
    return base / f"lawfirm_ai_{corpus_version}.db"


@contextmanager
def connect(db_path: Path, database_url: Optional[str] = None):
    """
    데이터베이스 연결 컨텍스트 매니저
    
    Args:
        db_path: 데이터베이스 파일 경로 (하위 호환성용, 사용 중단 예정)
        database_url: 데이터베이스 URL (필수, PostgreSQL만 지원)
    
    Note:
        SQLite 직접 연결은 더 이상 지원하지 않습니다. database_url을 필수로 제공해야 합니다.
    """
    # database_url 필수
    if not database_url:
        raise ValueError(
            "database_url is required. "
            "SQLite direct connections are not allowed per project rules. "
            "Please provide a PostgreSQL database URL."
        )
    
    # DatabaseAdapter 필수
    if not DatabaseAdapter:
        raise RuntimeError(
            "DatabaseAdapter is required. "
            "Please ensure db_adapter module is available. "
            "Direct SQLite connections are not allowed per project rules."
        )
    
    # SQLite URL 검증 (더 이상 지원하지 않음)
    if database_url.startswith('sqlite://'):
        raise ValueError(
            "SQLite is no longer supported. "
            "Please use PostgreSQL. "
            "Set database_url to a PostgreSQL URL (e.g., postgresql://user:password@host:port/database)"
        )
    
    # DatabaseAdapter 사용
    adapter = DatabaseAdapter(database_url)
    with adapter.get_connection_context() as conn:
        yield conn


def ensure_versioned_schema(db_path: Path, database_url: Optional[str] = None) -> None:
    """
    버전 관리 스키마 생성
    
    Args:
        db_path: 데이터베이스 파일 경로 (하위 호환성용, 사용 중단 예정)
        database_url: 데이터베이스 URL (필수, PostgreSQL만 지원)
    
    Note:
        SQLite 직접 연결은 더 이상 지원하지 않습니다. database_url을 필수로 제공해야 합니다.
    """
    # database_url 필수
    if not database_url:
        raise ValueError(
            "database_url is required. "
            "SQLite direct connections are not allowed per project rules. "
            "Please provide a PostgreSQL database URL."
        )
    
    # SQLite URL 검증 (더 이상 지원하지 않음)
    if database_url.startswith('sqlite://'):
        raise ValueError(
            "SQLite is no longer supported. "
            "Please use PostgreSQL. "
            "Set database_url to a PostgreSQL URL (e.g., postgresql://user:password@host:port/database)"
        )
    
    actual_db_url = database_url
    
    with connect(db_path, actual_db_url) as conn:
        # 커서 가져오기
        if hasattr(conn, 'cursor'):
            cur = conn.cursor()
        elif hasattr(conn, 'conn'):
            cur = conn.conn.cursor()
        else:
            cur = conn
        
        # SQL 문법 자동 변환
        # DatabaseAdapter를 사용하는 경우 db_type 가져오기
        if DatabaseAdapter:
            try:
                adapter = DatabaseAdapter(actual_db_url)
                db_type = adapter.db_type
            except Exception:
                # 폴백: URL 기반 감지
                db_type = 'postgresql' if actual_db_url and actual_db_url.startswith(('postgresql://', 'postgres://')) else 'sqlite'
        else:
            db_type = 'postgresql' if actual_db_url and actual_db_url.startswith(('postgresql://', 'postgres://')) else 'sqlite'
        
        def execute_sql(sql: str):
            """SQL 실행 (자동 변환)"""
            converted_sql = SQLAdapter.convert_sql(sql, db_type) if SQLAdapter else sql
            cur.execute(converted_sql)

        # minimal core tables with version column
        execute_sql(
            """
            CREATE TABLE IF NOT EXISTS laws (
                LID INTEGER PRIMARY KEY,
                title_ko TEXT,
                short_title TEXT,
                law_type TEXT,
                ministry_code TEXT,
                ministry_name TEXT,
                promulgation_date TEXT,
                promulgation_no TEXT,
                effective_date TEXT,
                status TEXT,
                rrClsCd TEXT,
                detail_url TEXT,
                content_hash TEXT,
                corpus_version TEXT,
                updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
            """
        )

        execute_sql(
            """
            CREATE TABLE IF NOT EXISTS provisions_meta (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                LID INTEGER NOT NULL,
                path_key TEXT NOT NULL,
                jo_no TEXT,
                title TEXT,
                order_index INTEGER,
                effective_date TEXT,
                content_hash TEXT,
                corpus_version TEXT,
                UNIQUE (LID, path_key)
            )
            """
        )

        execute_sql(
            """
            CREATE TABLE IF NOT EXISTS cases (
                case_no TEXT PRIMARY KEY,
                title TEXT,
                court_code TEXT,
                court_name TEXT,
                decision_date TEXT,
                case_type_code TEXT,
                case_type_name TEXT,
                issue TEXT,
                holding TEXT,
                source_url TEXT,
                content_hash TEXT,
                corpus_version TEXT
            )
            """
        )

        execute_sql(
            """
            CREATE TABLE IF NOT EXISTS interpretations (
                expc_id TEXT PRIMARY KEY,
                title TEXT,
                reply_date TEXT,
                org_code TEXT,
                org_name TEXT,
                question_summary TEXT,
                answer_text TEXT,
                source_url TEXT,
                content_hash TEXT,
                corpus_version TEXT
            )
            """
        )

        # PostgreSQL인 경우 JSONB 사용, SQLite인 경우 TEXT 사용
        if db_type == 'postgresql':
            execute_sql(
                """
                CREATE TABLE IF NOT EXISTS text_store (
                    id SERIAL PRIMARY KEY,
                    object_type VARCHAR(50) NOT NULL,
                    object_id VARCHAR(255) NOT NULL,
                    role VARCHAR(50) NOT NULL,
                    text TEXT NOT NULL,
                    corpus_version VARCHAR(50)
                )
                """
            )
        else:
            execute_sql(
                """
                CREATE TABLE IF NOT EXISTS text_store (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    object_type TEXT NOT NULL,
                    object_id TEXT NOT NULL,
                    role TEXT NOT NULL,
                    text TEXT NOT NULL,
                    corpus_version TEXT
                )
                """
            )

        # PostgreSQL인 경우 pgvector 사용, SQLite인 경우 BLOB 사용
        if db_type == 'postgresql':
            # pgvector 확장 활성화
            execute_sql("CREATE EXTENSION IF NOT EXISTS vector")
            
            # PostgreSQL: VECTOR 타입 사용 (차원은 동적으로 결정)
            # 기본값 768 차원 (ko-legal-sbert 모델)
            execute_sql(
                """
                CREATE TABLE IF NOT EXISTS embeddings (
                    id SERIAL PRIMARY KEY,
                    object_type VARCHAR(50) NOT NULL,
                    object_id VARCHAR(255) NOT NULL,
                    role VARCHAR(50) NOT NULL,
                    model_version VARCHAR(255) NOT NULL,
                    corpus_version VARCHAR(50) NOT NULL,
                    dim INTEGER NOT NULL,
                    vector VECTOR(768) NOT NULL,  -- pgvector 타입 (기본 768차원)
                    source_hash VARCHAR(64),
                    is_active BOOLEAN DEFAULT TRUE,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    UNIQUE(object_type, object_id, role, model_version, corpus_version)
                )
                """
            )
            
            # 벡터 검색 인덱스 (IVFFlat)
            execute_sql(
                """
                CREATE INDEX IF NOT EXISTS idx_embeddings_vector 
                ON embeddings USING ivfflat (vector vector_cosine_ops)
                WITH (lists = 100)
                """
            )
        else:
            # SQLite: BLOB 타입 사용
            execute_sql(
                """
                CREATE TABLE IF NOT EXISTS embeddings (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    object_type TEXT NOT NULL,
                    object_id TEXT NOT NULL,
                    role TEXT NOT NULL,
                    model_version TEXT NOT NULL,
                    corpus_version TEXT NOT NULL,
                    dim INTEGER NOT NULL,
                    vector BLOB NOT NULL,
                    source_hash TEXT,
                    is_active INTEGER DEFAULT 1,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    UNIQUE(object_type, object_id, role, model_version, corpus_version)
                )
                """
            )

        # embedding_versions 테이블 (PostgreSQL과 SQLite 모두 지원)
        if db_type == 'postgresql':
            execute_sql(
                """
                CREATE TABLE IF NOT EXISTS embedding_versions (
                    id SERIAL PRIMARY KEY,
                    version_name VARCHAR(255) NOT NULL UNIQUE,
                    chunking_strategy VARCHAR(50) NOT NULL,
                    model_name VARCHAR(255) NOT NULL,
                    description TEXT,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    is_active BOOLEAN DEFAULT FALSE,
                    metadata JSONB
                )
                """
            )
            # 활성 버전 조회 최적화 (부분 인덱스)
            execute_sql(
                """
                CREATE INDEX IF NOT EXISTS idx_embedding_versions_active 
                ON embedding_versions (id) WHERE is_active = TRUE
                """
            )
        else:
            execute_sql(
                """
                CREATE TABLE IF NOT EXISTS embedding_versions (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    version_name TEXT NOT NULL UNIQUE,
                    chunking_strategy TEXT NOT NULL,
                    model_name TEXT NOT NULL,
                    description TEXT,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    is_active INTEGER DEFAULT 0,
                    metadata TEXT
                )
                """
            )
        
        execute_sql(
            """
            CREATE TABLE IF NOT EXISTS relationships (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                from_type TEXT,
                from_id TEXT,
                to_type TEXT,
                to_id TEXT,
                relation_type TEXT,
                evidence TEXT,
                corpus_version TEXT
            )
            """
        )

        # reference code tables
        execute_sql(
            """
            CREATE TABLE IF NOT EXISTS ministry_codes (
                code TEXT PRIMARY KEY,
                name TEXT
            )
            """
        )

        execute_sql(
            """
            CREATE TABLE IF NOT EXISTS court_codes (
                code TEXT PRIMARY KEY,
                name TEXT
            )
            """
        )

        execute_sql(
            """
            CREATE TABLE IF NOT EXISTS case_type_codes (
                code TEXT PRIMARY KEY,
                name TEXT
            )
            """
        )

        # checkpoints for fault-tolerant resume
        execute_sql(
            """
            CREATE TABLE IF NOT EXISTS collection_jobs (
                job_id TEXT PRIMARY KEY,
                guide_id TEXT,
                params_json TEXT,
                status TEXT,
                started_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                finished_at TIMESTAMP,
                error TEXT
            )
            """
        )

        execute_sql(
            """
            CREATE TABLE IF NOT EXISTS collection_checkpoints (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                job_id TEXT NOT NULL,
                entity_type TEXT,
                last_page INTEGER,
                last_cursor TEXT,
                last_id TEXT,
                last_success_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                UNIQUE(job_id)
            )
            """
        )

        execute_sql(
            """
            CREATE TABLE IF NOT EXISTS schema_migrations (
                version TEXT PRIMARY KEY,
                applied_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
            """
        )

        conn.commit()

        # create helpful indexes
        index_sqls = [
            "CREATE INDEX IF NOT EXISTS idx_laws_effective ON laws(effective_date)",
            "CREATE INDEX IF NOT EXISTS idx_laws_ministry ON laws(ministry_code)",
            "CREATE INDEX IF NOT EXISTS idx_laws_updated_at ON laws(updated_at)",
            "CREATE UNIQUE INDEX IF NOT EXISTS idx_prov_unique ON provisions_meta(LID, path_key)",
            "CREATE INDEX IF NOT EXISTS idx_prov_lid ON provisions_meta(LID)",
            "CREATE INDEX IF NOT EXISTS idx_prov_order ON provisions_meta(order_index)",
            "CREATE INDEX IF NOT EXISTS idx_cases_date ON cases(decision_date)",
            "CREATE INDEX IF NOT EXISTS idx_cases_court ON cases(court_code)",
            "CREATE INDEX IF NOT EXISTS idx_cases_type ON cases(case_type_code)",
            "CREATE INDEX IF NOT EXISTS idx_interp_date ON interpretations(reply_date)",
            "CREATE INDEX IF NOT EXISTS idx_interp_org ON interpretations(org_code)",
            "CREATE INDEX IF NOT EXISTS idx_text_store_object ON text_store(object_type, object_id)",
            "CREATE INDEX IF NOT EXISTS idx_text_store_version ON text_store(corpus_version)",
            "CREATE UNIQUE INDEX IF NOT EXISTS idx_embed_unique ON embeddings(object_type, object_id, role, model_version, corpus_version)",
            "CREATE INDEX IF NOT EXISTS idx_embed_version ON embeddings(model_version, corpus_version)"
        ]
        
        for index_sql in index_sqls:
            execute_sql(index_sql)
        
        # 커밋 (DatabaseAdapter는 자동 커밋, 직접 연결은 수동 커밋)
        if not hasattr(conn, 'commit') or (hasattr(conn, 'conn') and hasattr(conn.conn, 'commit')):
            if hasattr(conn, 'conn') and hasattr(conn.conn, 'commit'):
                conn.conn.commit()
            elif hasattr(conn, 'commit'):
                conn.commit()


def ensure_versioned_db(versioned_dir: str, corpus_version: str, database_url: Optional[str] = None) -> Path:
    """
    버전 관리 데이터베이스 생성
    
    Args:
        versioned_dir: 버전 관리 디렉토리
        corpus_version: 코퍼스 버전
        database_url: 데이터베이스 URL (우선 사용, PostgreSQL 지원)
    
    Returns:
        데이터베이스 경로 (SQLite인 경우)
    """
    db_path = get_versioned_db_path(versioned_dir, corpus_version)
    ensure_versioned_schema(db_path, database_url)
    return db_path
