"""
SQLite에서 PostgreSQL로 데이터 마이그레이션 스크립트
"""
import os
import sys
import sqlite3
import psycopg2
from psycopg2.extras import execute_values
from typing import List, Dict, Any
import json
from datetime import datetime
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def migrate_sessions(sqlite_path: str, postgres_url: str):
    """세션 데이터 마이그레이션"""
    logger.info(f"Starting migration from {sqlite_path} to PostgreSQL...")
    
    # SQLite 연결
    sqlite_conn = sqlite3.connect(sqlite_path)
    sqlite_conn.row_factory = sqlite3.Row
    sqlite_cursor = sqlite_conn.cursor()
    
    # PostgreSQL 연결
    postgres_conn = psycopg2.connect(postgres_url)
    postgres_cursor = postgres_conn.cursor()
    
    try:
        # 세션 데이터 읽기
        logger.info("Reading sessions from SQLite...")
        sqlite_cursor.execute("SELECT * FROM sessions")
        sessions = sqlite_cursor.fetchall()
        logger.info(f"Found {len(sessions)} sessions")
        
        # PostgreSQL에 삽입
        logger.info("Inserting sessions into PostgreSQL...")
        for session in sessions:
            metadata = None
            if session.get('metadata'):
                try:
                    if isinstance(session['metadata'], str):
                        metadata = json.loads(session['metadata'])
                    else:
                        metadata = session['metadata']
                except:
                    metadata = None
            
            postgres_cursor.execute("""
                INSERT INTO sessions (session_id, title, category, created_at, updated_at, 
                                    message_count, user_id, ip_address, metadata)
                VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s)
                ON CONFLICT (session_id) DO UPDATE SET
                    title = EXCLUDED.title,
                    category = EXCLUDED.category,
                    updated_at = EXCLUDED.updated_at,
                    message_count = EXCLUDED.message_count,
                    user_id = EXCLUDED.user_id,
                    ip_address = EXCLUDED.ip_address,
                    metadata = EXCLUDED.metadata
            """, (
                session['session_id'],
                session.get('title'),
                session.get('category'),
                session.get('created_at'),
                session.get('updated_at'),
                session.get('message_count', 0),
                session.get('user_id'),
                session.get('ip_address'),
                json.dumps(metadata) if metadata else None
            ))
        
        # 메시지 데이터 읽기
        logger.info("Reading messages from SQLite...")
        sqlite_cursor.execute("SELECT * FROM messages")
        messages = sqlite_cursor.fetchall()
        logger.info(f"Found {len(messages)} messages")
        
        # PostgreSQL에 삽입
        logger.info("Inserting messages into PostgreSQL...")
        for message in messages:
            metadata = None
            if message.get('metadata'):
                try:
                    if isinstance(message['metadata'], str):
                        metadata = json.loads(message['metadata'])
                    else:
                        metadata = message['metadata']
                except:
                    metadata = None
            
            postgres_cursor.execute("""
                INSERT INTO messages (message_id, session_id, role, content, timestamp, metadata)
                VALUES (%s, %s, %s, %s, %s, %s)
                ON CONFLICT (message_id) DO UPDATE SET
                    session_id = EXCLUDED.session_id,
                    role = EXCLUDED.role,
                    content = EXCLUDED.content,
                    timestamp = EXCLUDED.timestamp,
                    metadata = EXCLUDED.metadata
            """, (
                message['message_id'],
                message['session_id'],
                message['role'],
                message['content'],
                message.get('timestamp'),
                json.dumps(metadata) if metadata else None
            ))
        
        postgres_conn.commit()
        logger.info(f"✅ Successfully migrated {len(sessions)} sessions and {len(messages)} messages")
        
    except Exception as e:
        postgres_conn.rollback()
        logger.error(f"❌ Migration failed: {e}")
        raise
    finally:
        sqlite_conn.close()
        postgres_conn.close()


def main():
    """메인 함수"""
    sqlite_path = os.getenv("SQLITE_PATH", "./data/api_sessions.db")
    postgres_url = os.getenv("POSTGRES_URL")
    
    if not postgres_url:
        logger.error("POSTGRES_URL environment variable is required")
        sys.exit(1)
    
    if not os.path.exists(sqlite_path):
        logger.error(f"SQLite database not found: {sqlite_path}")
        sys.exit(1)
    
    migrate_sessions(sqlite_path, postgres_url)


if __name__ == "__main__":
    main()

