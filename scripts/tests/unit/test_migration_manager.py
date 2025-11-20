"""
FAISSMigrationManager 단위 테스트
"""
import pytest
import sqlite3

from scripts.utils.faiss_migration_manager import FAISSMigrationManager


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

