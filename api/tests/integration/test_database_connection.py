"""
데이터베이스 연결 테스트
"""
import pytest
import os
from unittest.mock import patch, MagicMock
from api.database.connection import (
    get_database_url,
    get_database_type,
    get_engine,
    get_session,
    init_database,
    close_database
)


class TestDatabaseConnection:
    """데이터베이스 연결 테스트"""
    
    def test_get_database_url_default(self):
        """기본 데이터베이스 URL 테스트"""
        with patch.dict(os.environ, {}, clear=True):
            url = get_database_url()
            assert "sqlite" in url.lower()
    
    def test_get_database_url_from_env(self):
        """환경 변수에서 데이터베이스 URL 가져오기 테스트"""
        with patch.dict(os.environ, {"DATABASE_URL": "sqlite:///./test.db"}):
            url = get_database_url()
            assert url == "sqlite:///./test.db"
    
    def test_get_database_type_sqlite(self):
        """SQLite 데이터베이스 타입 감지 테스트"""
        with patch('api.database.connection.get_database_url', return_value="sqlite:///./test.db"):
            db_type = get_database_type()
            assert db_type == "sqlite"
    
    def test_get_database_type_postgresql(self):
        """PostgreSQL 데이터베이스 타입 감지 테스트"""
        with patch('api.database.connection.get_database_url', return_value="postgresql://user:pass@localhost/db"):
            db_type = get_database_type()
            assert db_type == "postgresql"
    
    def test_get_database_type_unsupported(self):
        """지원하지 않는 데이터베이스 타입 테스트"""
        with patch('api.database.connection.get_database_url', return_value="mysql://user:pass@localhost/db"):
            with pytest.raises(ValueError):
                get_database_type()
    
    def test_get_engine_sqlite(self):
        """SQLite 엔진 생성 테스트"""
        with patch('api.database.connection.get_database_url', return_value="sqlite:///:memory:"):
            with patch('api.database.connection.get_database_type', return_value="sqlite"):
                with patch('sqlalchemy.create_engine') as mock_create:
                    engine = get_engine()
                    mock_create.assert_called_once()
    
    def test_get_engine_singleton(self):
        """엔진 싱글톤 패턴 테스트"""
        with patch('api.database.connection.get_database_url', return_value="sqlite:///:memory:"):
            with patch('api.database.connection.get_database_type', return_value="sqlite"):
                with patch('sqlalchemy.create_engine') as mock_create:
                    engine1 = get_engine()
                    engine2 = get_engine()
                    assert engine1 is engine2
                    assert mock_create.call_count == 1
    
    def test_close_database(self):
        """데이터베이스 연결 종료 테스트"""
        with patch('api.database.connection._engine') as mock_engine:
            mock_engine.dispose = MagicMock()
            close_database()
            mock_engine.dispose.assert_called_once()

