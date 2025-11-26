# -*- coding: utf-8 -*-
"""
Database Adapter
데이터베이스 타입에 독립적인 어댑터
PostgreSQL만 지원
"""

import os
from abc import ABC, abstractmethod
from contextlib import contextmanager
from typing import Optional, Any, Dict, List, Tuple
from urllib.parse import urlparse

try:
    from lawfirm_langgraph.core.utils.logger import get_logger
except ImportError:
    from core.utils.logger import get_logger

from .sql_adapter import SQLAdapter

logger = get_logger(__name__)

# PostgreSQL 지원
try:
    import psycopg2
    from psycopg2.extras import RealDictCursor, RealDictRow
    from psycopg2.pool import ThreadedConnectionPool
    POSTGRESQL_AVAILABLE = True
except ImportError:
    POSTGRESQL_AVAILABLE = False
    logger.warning("PostgreSQL (psycopg2) not available. Install with: pip install psycopg2-binary")


class DatabaseConnection(ABC):
    """데이터베이스 연결 추상 클래스"""
    
    @abstractmethod
    def cursor(self, *args, **kwargs):
        """커서 생성"""
        pass
    
    @abstractmethod
    def commit(self):
        """트랜잭션 커밋"""
        pass
    
    @abstractmethod
    def rollback(self):
        """트랜잭션 롤백"""
        pass
    
    @abstractmethod
    def close(self):
        """연결 닫기"""
        pass
    
    def __enter__(self):
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        if exc_type:
            self.rollback()
        else:
            self.commit()
        self.close()
        return False


class PostgreSQLConnection(DatabaseConnection):
    """PostgreSQL 연결 래퍼"""
    
    def __init__(self, conn):
        self.conn = conn
    
    def _is_closed(self) -> bool:
        """연결이 닫혀있는지 확인"""
        if not self.conn:
            return True
        # psycopg2 연결의 closed 속성 확인 (psycopg2 2.0.0+)
        if hasattr(self.conn, 'closed'):
            return self.conn.closed != 0  # closed는 정수 (0=열림, 1=닫힘)
        # closed 속성이 없는 경우 (구버전), 상태 확인 시도
        try:
            # 간단한 쿼리로 연결 상태 확인 (최후의 수단)
            cursor = self.conn.cursor()
            cursor.execute("SELECT 1")
            cursor.close()
            return False
        except (psycopg2.InterfaceError, psycopg2.OperationalError, AttributeError):
            return True
    
    def cursor(self, *args, **kwargs):
        # 연결이 닫혀있는지 확인
        if self._is_closed():
            # 연결이 닫혀있으면 새 연결을 가져오려고 시도 (연결 풀에서)
            # 하지만 이는 연결 풀 관리 문제이므로, 명확한 오류 메시지 제공
            raise psycopg2.InterfaceError(
                "Connection is closed. This may indicate a connection pool issue. "
                "Please check if the connection was properly returned to the pool."
            )
        # RealDictCursor 사용하여 dict-like 접근 가능
        try:
            return self.conn.cursor(cursor_factory=RealDictCursor)
        except psycopg2.InterfaceError as e:
            if "connection already closed" in str(e).lower():
                raise psycopg2.InterfaceError(
                    "Connection was closed during cursor creation. "
                    "This may indicate a connection pool issue."
                ) from e
            raise
    
    def commit(self):
        if self._is_closed():
            logger.warning("Attempted to commit on closed connection, skipping")
            return
        try:
            self.conn.commit()
        except psycopg2.InterfaceError as e:
            if "connection already closed" in str(e).lower():
                logger.warning(f"Connection already closed during commit: {e}")
            else:
                raise
    
    def rollback(self):
        if self._is_closed():
            logger.warning("Attempted to rollback on closed connection, skipping")
            return
        try:
            self.conn.rollback()
        except psycopg2.InterfaceError as e:
            if "connection already closed" in str(e).lower():
                logger.warning(f"Connection already closed during rollback: {e}")
            else:
                raise
    
    def close(self):
        if not self._is_closed():
            self.conn.close()


class DatabaseAdapter:
    """데이터베이스 타입에 독립적인 어댑터 (PostgreSQL만 지원)"""
    
    def __init__(self, database_url: str, minconn: Optional[int] = None, maxconn: Optional[int] = None):
        """
        초기화
        
        Args:
            database_url: 데이터베이스 연결 URL
                - PostgreSQL: postgresql://user:password@host:port/database
            minconn: 연결 풀 최소 크기 (기본값: 환경 변수 DB_POOL_MIN_SIZE 또는 1)
            maxconn: 연결 풀 최대 크기 (기본값: 환경 변수 DB_POOL_MAX_SIZE 또는 20)
        """
        self.database_url = database_url
        self.db_type = self._detect_db_type(database_url)
        self.connection_pool = None
        self._initialize_connection_pool(minconn=minconn, maxconn=maxconn)
        logger.info(f"DatabaseAdapter initialized: type={self.db_type}, url={self._mask_url(database_url)}")
    
    def _mask_url(self, url: str) -> str:
        """URL에서 비밀번호 마스킹"""
        try:
            parsed = urlparse(url)
            if parsed.password:
                return url.replace(parsed.password, "***")
        except Exception:
            pass
        return url
    
    def _detect_db_type(self, database_url: str) -> str:
        """
        데이터베이스 타입 자동 감지 (PostgreSQL만 지원)
        
        Args:
            database_url: 데이터베이스 연결 URL
        
        Returns:
            'postgresql'
        """
        if database_url.startswith('postgresql://') or database_url.startswith('postgres://'):
            return 'postgresql'
        else:
            raise ValueError(f"Unsupported database URL format: {database_url[:50]}... Only PostgreSQL is supported.")
    
    def _initialize_connection_pool(self, minconn: Optional[int] = None, maxconn: Optional[int] = None):
        """연결 풀 초기화"""
        if self.db_type == 'postgresql':
            if not POSTGRESQL_AVAILABLE:
                raise ImportError("psycopg2 is required for PostgreSQL support. Install with: pip install psycopg2-binary")
            
            # 환경 변수 또는 파라미터로 연결 풀 크기 설정
            min_size = minconn if minconn is not None else int(os.getenv("DB_POOL_MIN_SIZE", "1"))
            max_size = maxconn if maxconn is not None else int(os.getenv("DB_POOL_MAX_SIZE", "20"))
            
            # PostgreSQL 연결 풀 생성
            self.connection_pool = ThreadedConnectionPool(
                minconn=min_size,
                maxconn=max_size,
                dsn=self.database_url
            )
            logger.info(f"Connection pool initialized: min={min_size}, max={max_size}")
        else:
            raise ValueError(f"Unsupported database type: {self.db_type}. Only PostgreSQL is supported.")
    
    def get_connection(self) -> DatabaseConnection:
        """
        데이터베이스 연결 가져오기 (PostgreSQL만 지원)
        
        Returns:
            DatabaseConnection: 데이터베이스 연결 객체
        """
        if self.db_type == 'postgresql':
            if not self.connection_pool:
                raise RuntimeError("PostgreSQL connection pool not initialized")
            conn = self.connection_pool.getconn()
            return PostgreSQLConnection(conn)
        else:
            raise ValueError(f"Unsupported database type: {self.db_type}. Only PostgreSQL is supported.")
    
    @contextmanager
    def get_connection_context(self):
        """
        컨텍스트 매니저를 사용한 연결 가져오기
        
        사용 예:
            with adapter.get_connection_context() as conn:
                cursor = conn.cursor()
                cursor.execute("SELECT ...")
        """
        conn = self.get_connection()
        conn_returned = False
        try:
            yield conn
        except Exception as e:
            # rollback 시도 (연결이 닫혀있으면 무시)
            try:
                if hasattr(conn, '_is_closed') and not conn._is_closed():
                    conn.rollback()
            except (psycopg2.InterfaceError, AttributeError) as rollback_error:
                # 연결이 이미 닫혀있거나 rollback 실패 시 경고만 출력
                logger.debug(f"Rollback skipped (connection may be closed): {rollback_error}")
            logger.error(f"Database error in connection context: {e}")
            raise
        finally:
            # 연결 반환은 항상 시도 (conn_returned 플래그와 무관하게)
            if self.db_type == 'postgresql' and self.connection_pool and not conn_returned:
                # PostgreSQL 연결 풀에 반환
                if hasattr(conn, 'conn'):
                    try:
                        # 연결이 닫혀있지 않은 경우에만 풀에 반환
                        if hasattr(conn, '_is_closed') and not conn._is_closed():
                            self.connection_pool.putconn(conn.conn)
                            conn_returned = True
                            logger.debug("Connection returned to pool successfully")
                        else:
                            # 연결이 이미 닫혀있으면 풀에 반환하지 않음 (풀에서 제거됨)
                            logger.debug("Connection already closed, not returning to pool")
                    except Exception as put_error:
                        logger.warning(f"Error returning connection to pool: {put_error}")
                        # 연결이 손상된 경우 풀에 반환하지 않음
                        try:
                            if hasattr(conn, 'conn') and hasattr(conn, '_is_closed') and not conn._is_closed():
                                conn.conn.close()
                        except Exception:
                            pass
                else:
                    # conn 속성이 없는 경우 (예상치 못한 상황)
                    logger.warning("Connection wrapper missing 'conn' attribute, cannot return to pool")
    
    def execute_query(
        self,
        query: str,
        params: Optional[Tuple] = None,
        fetch: bool = True
    ) -> Optional[List[Dict[str, Any]]]:
        """
        쿼리 실행 (자동 변환)
        
        Args:
            query: SQL 쿼리
            params: 쿼리 파라미터
            fetch: 결과를 가져올지 여부
        
        Returns:
            쿼리 결과 (fetch=True인 경우)
        """
        # SQL 변환
        converted_query = SQLAdapter.convert_sql(query, self.db_type)
        
        # 파라미터 변환 (필요시)
        if params:
            # PostgreSQL의 경우 %s 사용, SQLite는 ? 사용
            # SQLAdapter에서 이미 변환했으므로 그대로 사용
            pass
        
        with self.get_connection_context() as conn:
            cursor = conn.cursor()
            try:
                cursor.execute(converted_query, params)
                
                if fetch:
                    rows = cursor.fetchall()
                    # Row 객체를 dict로 변환
                    return [SQLAdapter.convert_row_to_dict(row) for row in rows]
                else:
                    return None
            except Exception as e:
                logger.error(f"Query execution error: {e}")
                logger.error(f"Query: {converted_query[:200]}...")
                logger.error(f"Params: {params}")
                raise
    
    def convert_sql(self, sql: str) -> str:
        """
        SQL을 PostgreSQL SQL로 변환
        
        Args:
            sql: SQL 쿼리
        
        Returns:
            변환된 SQL 쿼리
        """
        return SQLAdapter.convert_sql(sql, self.db_type)
    
    def table_exists(self, table_name: str) -> bool:
        """
        테이블 존재 여부 확인
        
        Args:
            table_name: 테이블명
        
        Returns:
            테이블 존재 여부
        """
        query, params = SQLAdapter.convert_table_check_query(table_name, self.db_type)
        
        with self.get_connection_context() as conn:
            cursor = conn.cursor()
            cursor.execute(query, params)
            result = cursor.fetchone()
            return result is not None
    
    def close(self):
        """연결 풀 닫기"""
        if self.db_type == 'postgresql' and self.connection_pool:
            self.connection_pool.closeall()
            logger.info("PostgreSQL connection pool closed")

