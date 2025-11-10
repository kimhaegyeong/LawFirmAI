"""
데이터베이스 연결 관리
"""
import os
import logging
from sqlalchemy import create_engine, event
from sqlalchemy.orm import sessionmaker, Session
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.pool import StaticPool
from typing import Optional

logger = logging.getLogger(__name__)

Base = declarative_base()

# 전역 엔진 및 세션 팩토리
_engine: Optional[object] = None
_SessionLocal: Optional[sessionmaker] = None


def get_database_url() -> str:
    """환경 변수에서 데이터베이스 URL 가져오기"""
    db_url = os.getenv("DATABASE_URL", "sqlite:///./data/lawfirm.db")
    return db_url


def get_database_type() -> str:
    """데이터베이스 타입 감지"""
    db_url = get_database_url()
    if db_url.startswith("sqlite"):
        return "sqlite"
    elif db_url.startswith("postgresql"):
        return "postgresql"
    else:
        raise ValueError(f"Unsupported database URL: {db_url}")


def get_engine():
    """데이터베이스 엔진 생성 (싱글톤)"""
    global _engine
    
    if _engine is not None:
        return _engine
    
    db_url = get_database_url()
    db_type = get_database_type()
    
    if db_type == "sqlite":
        engine = create_engine(
            db_url,
            connect_args={"check_same_thread": False},
            poolclass=StaticPool,
            echo=False
        )
        logger.info("SQLite database engine created")
    elif db_type == "postgresql":
        engine = create_engine(
            db_url,
            pool_size=10,
            max_overflow=20,
            pool_pre_ping=True,
            echo=False
        )
        logger.info("PostgreSQL database engine created")
    else:
        raise ValueError(f"Unsupported database type: {db_type}")
    
    _engine = engine
    return engine


def get_session() -> Session:
    """데이터베이스 세션 생성"""
    global _SessionLocal
    
    if _SessionLocal is None:
        engine = get_engine()
        _SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
    
    return _SessionLocal()


def init_database():
    """데이터베이스 초기화 (테이블 생성)"""
    engine = get_engine()
    Base.metadata.create_all(bind=engine)
    logger.info("Database tables created")


def close_database():
    """데이터베이스 연결 종료"""
    global _engine, _SessionLocal
    
    if _engine is not None:
        _engine.dispose()
        _engine = None
    
    _SessionLocal = None
    logger.info("Database connections closed")

