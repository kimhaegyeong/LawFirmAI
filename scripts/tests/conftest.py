#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
공통 pytest fixtures

모든 테스트에서 사용할 수 있는 공통 fixtures를 정의합니다.
"""

import pytest
import tempfile
import shutil
import sqlite3
import os
import logging
from pathlib import Path
import sys

# 프로젝트 경로 설정 (테스트 파일 import 전에 실행)
project_root = Path(__file__).parent.parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

# scripts/utils 경로 추가
scripts_utils_path = project_root / "scripts" / "utils"
if str(scripts_utils_path) not in sys.path:
    sys.path.insert(0, str(scripts_utils_path))


@pytest.fixture
def temp_dir():
    """임시 디렉토리 fixture"""
    temp_path = Path(tempfile.mkdtemp(prefix="test_"))
    yield temp_path
    if temp_path.exists():
        shutil.rmtree(temp_path)


@pytest.fixture
def project_root():
    """프로젝트 루트 경로 fixture"""
    return Path(__file__).parent.parent.parent


@pytest.fixture
def test_db_path(temp_dir):
    """테스트용 데이터베이스 경로 fixture"""
    db_path = temp_dir / "test.db"
    return str(db_path)


@pytest.fixture
def test_vector_store_path(temp_dir):
    """테스트용 벡터 스토어 경로 fixture"""
    vector_store_path = temp_dir / "vector_store"
    vector_store_path.mkdir(parents=True, exist_ok=True)
    return str(vector_store_path)


@pytest.fixture
def temp_db(temp_dir):
    """임시 데이터베이스 fixture (기본 스키마 포함)"""
    db_path = temp_dir / "test.db"
    conn = sqlite3.connect(str(db_path))
    
    # 기본 테이블 생성
    conn.execute("""
        CREATE TABLE IF NOT EXISTS embedding_versions (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            version_name TEXT UNIQUE,
            chunking_strategy TEXT,
            model_name TEXT,
            description TEXT,
            is_active INTEGER DEFAULT 0,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            metadata TEXT
        )
    """)
    
    conn.execute("""
        CREATE TABLE IF NOT EXISTS text_chunks (
            id INTEGER PRIMARY KEY,
            embedding_version_id INTEGER,
            source_type TEXT,
            source_id INTEGER,
            chunk_index INTEGER,
            content TEXT,
            metadata TEXT
        )
    """)
    
    conn.execute("""
        CREATE TABLE IF NOT EXISTS embeddings (
            id INTEGER PRIMARY KEY,
            chunk_id INTEGER,
            embedding BLOB
        )
    """)
    
    conn.execute("""
        CREATE TABLE IF NOT EXISTS statute_articles (
            id INTEGER PRIMARY KEY,
            statute_id INTEGER,
            article_number TEXT,
            content TEXT
        )
    """)
    
    conn.commit()
    conn.close()
    return str(db_path)


@pytest.fixture
def version_manager(temp_dir):
    """FAISSVersionManager 인스턴스 fixture"""
    from scripts.utils.faiss_version_manager import FAISSVersionManager
    return FAISSVersionManager(str(temp_dir))


@pytest.fixture
def embedding_version_manager(temp_db):
    """EmbeddingVersionManager 인스턴스 fixture"""
    from scripts.utils.embedding_version_manager import EmbeddingVersionManager
    return EmbeddingVersionManager(temp_db)


@pytest.fixture
def migration_manager(version_manager, embedding_version_manager, temp_db):
    """FAISSMigrationManager 인스턴스 fixture"""
    from scripts.utils.faiss_migration_manager import FAISSMigrationManager
    return FAISSMigrationManager(version_manager, embedding_version_manager, temp_db)


@pytest.fixture
def multi_search(version_manager):
    """MultiVersionSearch 인스턴스 fixture"""
    from scripts.utils.multi_version_search import MultiVersionSearch
    return MultiVersionSearch(version_manager)


@pytest.fixture
def load_env(project_root):
    """환경 변수 로드 fixture"""
    try:
        from dotenv import load_dotenv
        env_file = project_root / "api" / ".env"
        if env_file.exists():
            load_dotenv(env_file, override=False)
    except ImportError:
        pass


@pytest.fixture
def logger():
    """로거 fixture"""
    logger = logging.getLogger(__name__)
    if not logger.handlers:
        handler = logging.StreamHandler()
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        handler.setFormatter(formatter)
        logger.addHandler(handler)
        logger.setLevel(logging.INFO)
    return logger


@pytest.fixture
def workflow_service(load_env):
    """LangGraphWorkflowService 인스턴스 fixture"""
    try:
        from lawfirm_langgraph.config.langgraph_config import LangGraphConfig
        from lawfirm_langgraph.langgraph_core.workflow.workflow_service import LangGraphWorkflowService
        
        config = LangGraphConfig.from_env()
        config.enable_checkpoint = False
        service = LangGraphWorkflowService(config)
        return service
    except Exception as e:
        pytest.skip(f"워크플로우 서비스 초기화 실패: {e}")


@pytest.fixture
def search_engine(project_root):
    """SemanticSearchEngineV2 인스턴스 fixture"""
    try:
        from lawfirm_langgraph.core.search.engines.semantic_search_engine_v2 import SemanticSearchEngineV2
        
        db_path = project_root / "data" / "lawfirm_v2.db"
        if not db_path.exists():
            pytest.skip(f"데이터베이스 파일이 없습니다: {db_path}")
        
        engine = SemanticSearchEngineV2(db_path=str(db_path))
        return engine
    except Exception as e:
        pytest.skip(f"검색 엔진 초기화 실패: {e}")


@pytest.fixture
def real_db_path(project_root):
    """실제 데이터베이스 경로 fixture"""
    db_path = project_root / "data" / "lawfirm_v2.db"
    if not db_path.exists():
        pytest.skip(f"데이터베이스 파일이 없습니다: {db_path}")
    return str(db_path)


@pytest.fixture
def real_vector_store_path(project_root):
    """실제 벡터 스토어 경로 fixture"""
    vector_store_path = project_root / "data" / "vector_store"
    if not vector_store_path.exists():
        pytest.skip(f"벡터 스토어 디렉토리가 없습니다: {vector_store_path}")
    return str(vector_store_path)

