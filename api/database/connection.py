"""
데이터베이스 연결 관리
"""
import os
import logging
from pathlib import Path
from sqlalchemy import create_engine, event
from sqlalchemy.orm import sessionmaker, Session
from sqlalchemy.ext.declarative import declarative_base
from typing import Optional

logger = logging.getLogger(__name__)

Base = declarative_base()

# 전역 엔진 및 세션 팩토리
_engine: Optional[object] = None
_SessionLocal: Optional[sessionmaker] = None

# .env 파일 로드 (한 번만 실행)
_env_loaded = False


def _ensure_env_loaded():
    """.env 파일이 로드되었는지 확인하고, 로드되지 않았다면 로드"""
    global _env_loaded
    if _env_loaded:
        return
    
    try:
        # 프로젝트 루트 찾기 (api/database/connection.py -> api -> 프로젝트 루트)
        project_root = Path(__file__).parent.parent.parent
        
        # utils.env_loader 사용 (가능한 경우)
        try:
            from utils.env_loader import ensure_env_loaded
            ensure_env_loaded(project_root)
        except ImportError:
            # fallback: python-dotenv 직접 사용
            try:
                from dotenv import load_dotenv
                # api/.env 파일 로드 (최고 우선순위)
                api_env = project_root / "api" / ".env"
                if api_env.exists():
                    load_dotenv(dotenv_path=str(api_env), override=True)
                    logger.debug(f"Loaded .env file: {api_env}")
                # 프로젝트 루트 .env 파일 로드
                root_env = project_root / ".env"
                if root_env.exists():
                    load_dotenv(dotenv_path=str(root_env), override=False)
                    logger.debug(f"Loaded .env file: {root_env}")
            except ImportError:
                logger.warning("python-dotenv not installed. .env files will not be loaded.")
        
        _env_loaded = True
    except Exception as e:
        logger.warning(f"Failed to load .env files: {e}")


def get_database_url() -> str:
    """환경 변수에서 데이터베이스 URL 가져오기
    
    우선순위:
    1. DATABASE_URL이 명시적으로 설정되어 있으면 사용 (SQLite URL이면 에러)
    2. 없으면 POSTGRES_* 환경변수들을 조합하여 생성
       - 프로젝트 루트 .env 파일의 설정 우선 사용
       - api/.env 파일의 설정은 fallback
    3. 그것도 없으면 기본값 사용
    
    SQLite는 더 이상 지원하지 않습니다.
    """
    _ensure_env_loaded()
    
    # DATABASE_URL이 명시적으로 설정되어 있으면 사용
    db_url = os.getenv("DATABASE_URL")
    if db_url:
        # SQLite URL 검증
        if db_url.startswith("sqlite://"):
            raise ValueError(
                "SQLite is no longer supported. Please use PostgreSQL. "
                "Set DATABASE_URL to a PostgreSQL URL (e.g., postgresql://user:password@host:port/database) "
                "or configure POSTGRES_* environment variables."
            )
        return db_url
    
    # PostgreSQL 환경변수 조합
    # 프로젝트 루트 .env 파일의 설정을 우선 사용 (21-29줄)
    # 없으면 api/.env 파일의 설정 사용
    postgres_host = os.getenv("POSTGRES_HOST", "localhost")
    postgres_port = os.getenv("POSTGRES_PORT", "5432")
    postgres_db = os.getenv("POSTGRES_DB", "lawfirmai_local")
    postgres_user = os.getenv("POSTGRES_USER", "lawfirmai")
    postgres_password = os.getenv("POSTGRES_PASSWORD", "local_password")
    
    # URL 인코딩 (특수문자 처리)
    from urllib.parse import quote_plus
    encoded_password = quote_plus(postgres_password)
    
    # PostgreSQL URL 생성
    db_url = f"postgresql://{postgres_user}:{encoded_password}@{postgres_host}:{postgres_port}/{postgres_db}"
    logger.debug(f"Database URL generated from environment variables: postgresql://{postgres_user}:***@{postgres_host}:{postgres_port}/{postgres_db}")
    
    return db_url


def get_database_type() -> str:
    """데이터베이스 타입 반환 (PostgreSQL만 지원)"""
    return "postgresql"


def get_engine():
    """데이터베이스 엔진 생성 (싱글톤, PostgreSQL만 지원)"""
    global _engine
    
    if _engine is not None:
        # 연결 풀 상태 확인 및 로깅
        try:
            pool = _engine.pool
            logger.debug(
                f"Database connection pool status: "
                f"size={pool.size()}, checked_in={pool.checkedin()}, "
                f"checked_out={pool.checkedout()}, overflow={pool.overflow()}, "
                f"invalid={pool.invalid()}"
            )
        except Exception as e:
            logger.debug(f"Failed to get pool status: {e}")
        
        return _engine
    
    db_url = get_database_url()
    
    engine = create_engine(
        db_url,
        pool_size=10,
        max_overflow=20,
        pool_pre_ping=True,
        echo=False
    )
    logger.info(
        f"PostgreSQL database engine created: "
        f"pool_size=10, max_overflow=20, pool_pre_ping=True"
    )
    
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

