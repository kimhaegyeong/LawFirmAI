"""
FAISSMigrationManager 단위 테스트
"""
import pytest
import tempfile
import shutil
import sqlite3
from pathlib import Path

import sys
sys.path.insert(0, str(Path(__file__).parent.parent / "utils"))

from faiss_version_manager import FAISSVersionManager
from embedding_version_manager import EmbeddingVersionManager
from faiss_migration_manager import FAISSMigrationManager


@pytest.fixture
def temp_dir():
    """임시 디렉토리 생성"""
    temp_path = Path(tempfile.mkdtemp())
    yield temp_path
    shutil.rmtree(temp_path)


@pytest.fixture
def temp_db(temp_dir):
    """임시 데이터베이스 생성"""
    db_path = temp_dir / "test.db"
    conn = sqlite3.connect(str(db_path))
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
    conn.commit()
    conn.close()
    return str(db_path)


@pytest.fixture
def migration_manager(temp_dir, temp_db):
    """FAISSMigrationManager 인스턴스 생성"""
    faiss_manager = FAISSVersionManager(str(temp_dir))
    embedding_manager = EmbeddingVersionManager(temp_db)
    return FAISSMigrationManager(faiss_manager, embedding_manager, temp_db)


def test_get_original_document(migration_manager, temp_db):
    """원본 문서 가져오기 테스트"""
    conn = sqlite3.connect(temp_db)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS statute_articles (
            id INTEGER PRIMARY KEY,
            statute_id INTEGER,
            article_number TEXT,
            content TEXT
        )
    """)
    conn.execute("""
        INSERT INTO statute_articles (id, statute_id, article_number, content)
        VALUES (1, 1, '1', 'Test content')
    """)
    conn.commit()
    conn.close()
    
    doc = migration_manager.get_original_document("statute_article", 1)
    assert doc is not None
    assert doc["id"] == 1
    assert doc["content"] == "Test content"


def test_get_migration_status(migration_manager):
    """마이그레이션 상태 조회 테스트"""
    status = migration_manager.get_migration_status()
    assert isinstance(status, dict)
    assert "total" in status
    assert "success" in status
    assert "failed" in status


def test_clear_migration_log(migration_manager):
    """마이그레이션 로그 정리 테스트"""
    migration_manager.migration_log = {"test": {"status": "success"}}
    migration_manager.clear_migration_log()
    assert len(migration_manager.migration_log) == 0

