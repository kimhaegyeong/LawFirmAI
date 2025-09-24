"""
Database Management
데이터베이스 관리 모듈
"""

import sqlite3
import logging
from contextlib import contextmanager
from typing import List, Dict, Any, Optional
from pathlib import Path

logger = logging.getLogger(__name__)


class DatabaseManager:
    """데이터베이스 관리 클래스"""
    
    def __init__(self, db_path: str):
        """데이터베이스 관리자 초기화"""
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._create_tables()
        logger.info(f"DatabaseManager initialized with path: {self.db_path}")
    
    @contextmanager
    def get_connection(self):
        """데이터베이스 연결 컨텍스트 매니저"""
        conn = sqlite3.connect(str(self.db_path))
        conn.row_factory = sqlite3.Row
        try:
            yield conn
        except Exception as e:
            conn.rollback()
            logger.error(f"Database error: {e}")
            raise
        finally:
            conn.close()
    
    def _create_tables(self):
        """테이블 생성"""
        with self.get_connection() as conn:
            cursor = conn.cursor()
            
            # 채팅 기록 테이블
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS chat_history (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    session_id TEXT NOT NULL,
                    user_message TEXT NOT NULL,
                    bot_response TEXT NOT NULL,
                    confidence REAL DEFAULT 0.0,
                    processing_time REAL DEFAULT 0.0,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """)
            
            # 법률 문서 테이블
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS legal_documents (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    title TEXT NOT NULL,
                    content TEXT NOT NULL,
                    document_type TEXT NOT NULL,
                    source_url TEXT,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """)
            
            # 벡터 임베딩 테이블
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS document_embeddings (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    document_id INTEGER NOT NULL,
                    chunk_index INTEGER NOT NULL,
                    embedding BLOB NOT NULL,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    FOREIGN KEY (document_id) REFERENCES legal_documents (id)
                )
            """)
            
            conn.commit()
            logger.info("Database tables created successfully")
    
    def execute_query(self, query: str, params: tuple = ()) -> List[Dict[str, Any]]:
        """쿼리 실행"""
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(query, params)
            return [dict(row) for row in cursor.fetchall()]
    
    def execute_update(self, query: str, params: tuple = ()) -> int:
        """업데이트 쿼리 실행"""
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(query, params)
            conn.commit()
            return cursor.rowcount
    
    def insert_chat_message(self, session_id: str, user_message: str, 
                           bot_response: str, confidence: float = 0.0, 
                           processing_time: float = 0.0) -> int:
        """채팅 메시지 저장"""
        query = """
            INSERT INTO chat_history 
            (session_id, user_message, bot_response, confidence, processing_time)
            VALUES (?, ?, ?, ?, ?)
        """
        params = (session_id, user_message, bot_response, confidence, processing_time)
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(query, params)
            conn.commit()
            return cursor.lastrowid
    
    def get_chat_history(self, session_id: str, limit: int = 50) -> List[Dict[str, Any]]:
        """채팅 기록 조회"""
        query = """
            SELECT * FROM chat_history 
            WHERE session_id = ? 
            ORDER BY created_at DESC 
            LIMIT ?
        """
        return self.execute_query(query, (session_id, limit))
    
    def clear_chat_history(self, session_id: str) -> int:
        """채팅 기록 삭제"""
        query = "DELETE FROM chat_history WHERE session_id = ?"
        return self.execute_update(query, (session_id,))
