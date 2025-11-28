# -*- coding: utf-8 -*-
"""
Database 테스트
데이터베이스 모듈 단위 테스트
"""

import pytest
import sqlite3
import tempfile
import os
from pathlib import Path

from lawfirm_langgraph.core.data.database import DatabaseManager


class TestDatabaseManager:
    """DatabaseManager 테스트"""
    
    @pytest.fixture
    def temp_db(self):
        """임시 데이터베이스 픽스처"""
        with tempfile.NamedTemporaryFile(suffix='.db', delete=False) as f:
            db_path = f.name
        
        yield db_path
        
        if os.path.exists(db_path):
            try:
                os.remove(db_path)
            except Exception:
                pass
    
    @pytest.fixture
    def db_manager(self, temp_db):
        """DatabaseManager 인스턴스"""
        return DatabaseManager(temp_db)
    
    def test_database_initialization(self, temp_db):
        """데이터베이스 초기화 테스트"""
        manager = DatabaseManager(temp_db)
        
        assert manager.db_path == Path(temp_db)
        assert os.path.exists(temp_db)
    
    def test_get_connection(self, db_manager):
        """데이터베이스 연결 테스트"""
        with db_manager.get_connection() as conn:
            assert isinstance(conn, sqlite3.Connection)
            assert conn.row_factory == sqlite3.Row
    
    def test_create_tables(self, db_manager):
        """테이블 생성 테스트"""
        with db_manager.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT name FROM sqlite_master WHERE type='table'")
            tables = [row[0] for row in cursor.fetchall()]
            
            assert 'chat_history' in tables
            assert 'documents' in tables
    
    def test_execute_query_select(self, db_manager):
        """SELECT 쿼리 실행 테스트"""
        with db_manager.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT 1 as test_value")
            result = cursor.fetchone()
            
            assert result is not None
            assert result['test_value'] == 1
    
    def test_execute_query_insert(self, db_manager):
        """INSERT 쿼리 실행 테스트"""
        with db_manager.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                INSERT INTO chat_history (session_id, user_message, bot_response)
                VALUES (?, ?, ?)
            """, ("test_session", "테스트 메시지", "테스트 응답"))
            conn.commit()
            
            cursor.execute("""
                SELECT * FROM chat_history WHERE session_id = ?
            """, ("test_session",))
            result = cursor.fetchone()
            
            assert result is not None
            assert result['session_id'] == "test_session"
            assert result['user_message'] == "테스트 메시지"
            assert result['bot_response'] == "테스트 응답"
    
    def test_execute_query_update(self, db_manager):
        """UPDATE 쿼리 실행 테스트"""
        with db_manager.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                INSERT INTO chat_history (session_id, user_message, bot_response)
                VALUES (?, ?, ?)
            """, ("test_session", "테스트 메시지", "테스트 응답"))
            conn.commit()
            
            cursor.execute("""
                UPDATE chat_history SET bot_response = ? WHERE session_id = ?
            """, ("수정된 응답", "test_session"))
            conn.commit()
            
            cursor.execute("""
                SELECT bot_response FROM chat_history WHERE session_id = ?
            """, ("test_session",))
            result = cursor.fetchone()
            
            assert result['bot_response'] == "수정된 응답"
    
    def test_execute_query_delete(self, db_manager):
        """DELETE 쿼리 실행 테스트"""
        with db_manager.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                INSERT INTO chat_history (session_id, user_message, bot_response)
                VALUES (?, ?, ?)
            """, ("test_session", "테스트 메시지", "테스트 응답"))
            conn.commit()
            
            cursor.execute("""
                DELETE FROM chat_history WHERE session_id = ?
            """, ("test_session",))
            conn.commit()
            
            cursor.execute("""
                SELECT COUNT(*) as count FROM chat_history WHERE session_id = ?
            """, ("test_session",))
            result = cursor.fetchone()
            
            assert result['count'] == 0
    
    def test_transaction_rollback(self, db_manager):
        """트랜잭션 롤백 테스트"""
        with db_manager.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                INSERT INTO chat_history (session_id, user_message, bot_response)
                VALUES (?, ?, ?)
            """, ("test_session", "테스트 메시지", "테스트 응답"))
            conn.rollback()
            
            cursor.execute("""
                SELECT COUNT(*) as count FROM chat_history WHERE session_id = ?
            """, ("test_session",))
            result = cursor.fetchone()
            
            assert result['count'] == 0
    
    def test_error_handling(self, db_manager):
        """에러 핸들링 테스트"""
        with pytest.raises(Exception):
            with db_manager.get_connection() as conn:
                cursor = conn.cursor()
                cursor.execute("SELECT * FROM non_existent_table")
    
    def test_database_path_creation(self):
        """데이터베이스 경로 생성 테스트"""
        with tempfile.TemporaryDirectory() as temp_dir:
            db_path = os.path.join(temp_dir, "test.db")
            manager = DatabaseManager(db_path)
            
            assert os.path.exists(db_path)
            assert manager.db_path == Path(db_path)

