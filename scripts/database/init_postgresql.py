"""
PostgreSQL 데이터베이스 초기화 스크립트
"""
import os
import sys
import psycopg2
from psycopg2.extensions import ISOLATION_LEVEL_AUTOCOMMIT
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def init_database(postgres_url: str):
    """PostgreSQL 데이터베이스 초기화"""
    logger.info("Initializing PostgreSQL database...")
    
    # 연결 URL에서 데이터베이스 이름 추출
    # postgresql://user:password@host:port/dbname
    from urllib.parse import urlparse
    parsed = urlparse(postgres_url)
    db_name = parsed.path[1:]  # / 제거
    
    # postgres 데이터베이스에 연결 (데이터베이스 생성 전)
    admin_url = postgres_url.rsplit('/', 1)[0] + '/postgres'
    admin_conn = psycopg2.connect(admin_url)
    admin_conn.set_isolation_level(ISOLATION_LEVEL_AUTOCOMMIT)
    admin_cursor = admin_conn.cursor()
    
    try:
        # 데이터베이스 존재 확인
        admin_cursor.execute(
            "SELECT 1 FROM pg_database WHERE datname = %s",
            (db_name,)
        )
        exists = admin_cursor.fetchone()
        
        if not exists:
            # 데이터베이스 생성
            logger.info(f"Creating database: {db_name}")
            admin_cursor.execute(f'CREATE DATABASE "{db_name}"')
            logger.info(f"✅ Database {db_name} created")
        else:
            logger.info(f"Database {db_name} already exists")
        
        admin_conn.close()
        
        # 실제 데이터베이스에 연결하여 테이블 생성
        conn = psycopg2.connect(postgres_url)
        cursor = conn.cursor()
        
        # 세션 테이블 생성
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS sessions (
                session_id VARCHAR(255) PRIMARY KEY,
                title TEXT,
                category TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                message_count INTEGER DEFAULT 0,
                user_id VARCHAR(255),
                ip_address VARCHAR(45),
                metadata JSONB
            )
        """)
        
        # 메시지 테이블 생성
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS messages (
                message_id VARCHAR(255) PRIMARY KEY,
                session_id VARCHAR(255) NOT NULL,
                role VARCHAR(50) NOT NULL,
                content TEXT NOT NULL,
                timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                metadata JSONB,
                FOREIGN KEY (session_id) REFERENCES sessions(session_id) ON DELETE CASCADE
            )
        """)
        
        # 인덱스 생성
        cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_messages_session_id 
            ON messages(session_id)
        """)
        
        cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_sessions_updated_at 
            ON sessions(updated_at)
        """)
        
        cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_sessions_user_id 
            ON sessions(user_id)
        """)
        
        conn.commit()
        conn.close()
        
        logger.info("✅ Database tables and indexes created successfully")
        
    except Exception as e:
        logger.error(f"❌ Database initialization failed: {e}")
        raise


def main():
    """메인 함수"""
    postgres_url = os.getenv("DATABASE_URL")
    
    if not postgres_url:
        logger.error("DATABASE_URL environment variable is required")
        sys.exit(1)
    
    if not postgres_url.startswith("postgresql"):
        logger.error("DATABASE_URL must be a PostgreSQL connection string")
        sys.exit(1)
    
    init_database(postgres_url)


if __name__ == "__main__":
    main()

