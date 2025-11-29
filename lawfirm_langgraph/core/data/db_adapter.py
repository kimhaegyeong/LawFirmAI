# -*- coding: utf-8 -*-
"""
Database Adapter
데이터베이스 타입에 독립적인 어댑터
PostgreSQL만 지원
"""

import os
import time
import threading
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

# 싱글톤 인스턴스 캐시 (database_url별로 관리)
_database_adapter_cache: Dict[str, 'DatabaseAdapter'] = {}


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
    
    def __init__(self, database_url: str, minconn: Optional[int] = None, maxconn: Optional[int] = None, force_new: bool = False):
        """
        초기화
        
        Args:
            database_url: 데이터베이스 연결 URL
                - PostgreSQL: postgresql://user:password@host:port/database
            minconn: 연결 풀 최소 크기 (기본값: 환경 변수 DB_POOL_MIN_SIZE 또는 5)
            maxconn: 연결 풀 최대 크기 (기본값: 환경 변수 DB_POOL_MAX_SIZE 또는 50)
            force_new: True이면 새 인스턴스 생성 (기본값: False, 싱글톤 사용)
        """
        # 싱글톤 패턴: 동일한 database_url에 대해 기존 인스턴스 재사용
        global _database_adapter_cache
        if not force_new and database_url in _database_adapter_cache:
            existing = _database_adapter_cache[database_url]
            # 기존 인스턴스의 속성 복사
            self.database_url = existing.database_url
            self.db_type = existing.db_type
            self.connection_pool = existing.connection_pool
            # 캐시 재사용 시 DEBUG 레벨로 변경 (이미 INFO 레벨로 로그가 출력되었으므로)
            logger.debug(f"DatabaseAdapter reused from cache: type={self.db_type}, url={self._mask_url(database_url)}")
            return
        
        self.database_url = database_url
        self.db_type = self._detect_db_type(database_url)
        self.connection_pool = None
        # 연결 풀 통계 추적
        self._pool_stats = {
            'total_connections': 0,
            'active_connections': 0,
            'failed_connections': 0,
            'returned_connections': 0
        }
        self._pool_stats_lock = threading.Lock()
        
        # 초기화 시간 측정
        init_start = time.time()
        self._initialize_connection_pool(minconn=minconn, maxconn=maxconn)
        init_time = time.time() - init_start
        
        # 중복 로그 방지: force_new가 False이고 캐시에 이미 있으면 로그 출력 안 함
        if force_new or database_url not in _database_adapter_cache:
            logger.info(f"DatabaseAdapter initialized: type={self.db_type}, url={self._mask_url(database_url)} (초기화 시간: {init_time:.3f}초)")
        
        # 싱글톤 캐시에 저장 (force_new가 True여도 캐시에 저장하여 재사용 가능하게 함)
        _database_adapter_cache[database_url] = self
    
    @classmethod
    def get_instance(cls, database_url: str, minconn: Optional[int] = None, maxconn: Optional[int] = None) -> 'DatabaseAdapter':
        """
        싱글톤 인스턴스 가져오기
        
        Args:
            database_url: 데이터베이스 연결 URL
            minconn: 연결 풀 최소 크기
            maxconn: 연결 풀 최대 크기
            
        Returns:
            DatabaseAdapter 인스턴스
        """
        global _database_adapter_cache
        if database_url not in _database_adapter_cache:
            # force_new=False로 생성하여 캐시에서 재사용 가능하게 함
            _database_adapter_cache[database_url] = cls(database_url, minconn=minconn, maxconn=maxconn, force_new=False)
        return _database_adapter_cache[database_url]
    
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
            # 기본값 증가: 동시 검색 요청이 많을 때 연결 풀 고갈 방지
            min_size = minconn if minconn is not None else int(os.getenv("DB_POOL_MIN_SIZE", "5"))
            max_size = maxconn if maxconn is not None else int(os.getenv("DB_POOL_MAX_SIZE", "50"))
            
            # PostgreSQL 연결 풀 생성
            self.connection_pool = ThreadedConnectionPool(
                minconn=min_size,
                maxconn=max_size,
                dsn=self.database_url
            )
            # 연결 풀 초기화 로그는 DatabaseAdapter 초기화 로그에 포함되므로 별도 출력 안 함
            # (중복 방지)
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
            with self._pool_stats_lock:
                self._pool_stats['total_connections'] += 1
                self._pool_stats['active_connections'] += 1
            return PostgreSQLConnection(conn)
        else:
            raise ValueError(f"Unsupported database type: {self.db_type}. Only PostgreSQL is supported.")
    
    def _get_connection_with_timeout(self, timeout: Optional[int] = None) -> DatabaseConnection:
        """
        타임아웃이 있는 연결 가져오기
        
        Args:
            timeout: 타임아웃 시간 (초), None이면 환경 변수 사용
            
        Returns:
            DatabaseConnection: 데이터베이스 연결 객체
            
        Raises:
            TimeoutError: 타임아웃 발생 시
            RuntimeError: 재시도 실패 시
        """
        if timeout is None:
            timeout = int(os.getenv("DB_CONNECTION_TIMEOUT", "30"))  # 기본 30초
        
        start_time = time.time()
        retry_count = 0
        max_retries = 3
        
        while True:
            try:
                if self.db_type == 'postgresql':
                    if not self.connection_pool:
                        raise RuntimeError("PostgreSQL connection pool not initialized")
                    
                    # 연결 풀 상태 확인 (에러 발생해도 연결 가져오기 계속)
                    try:
                        stats = self.get_pool_status()
                        available = stats.get("available_connections", 0)
                        if available <= 0:
                            elapsed = time.time() - start_time
                            if elapsed >= timeout:
                                raise TimeoutError(
                                    f"Connection timeout after {timeout}s: "
                                    f"pool exhausted (active: {stats.get('active_connections')}/{stats.get('maxconn')})"
                                )
                            logger.debug(f"Connection pool exhausted, waiting... (elapsed: {elapsed:.1f}s)")
                            time.sleep(0.1)  # 짧은 대기 후 재시도
                            continue
                    except Exception as status_error:
                        # 상태 확인 실패해도 연결 가져오기 계속 (에러만 로깅)
                        logger.debug(f"Failed to get pool status (continuing anyway): {status_error}")
                        # 상태 확인 실패 시 바로 연결 시도
                    
                    conn = self.connection_pool.getconn()
                    # 통계 업데이트 (에러 발생해도 연결은 반환)
                    try:
                        if hasattr(self, '_pool_stats_lock') and hasattr(self, '_pool_stats'):
                            with self._pool_stats_lock:
                                self._pool_stats['total_connections'] += 1
                                self._pool_stats['active_connections'] += 1
                        else:
                            # 속성이 없으면 초기화 (하위 호환성)
                            if not hasattr(self, '_pool_stats_lock'):
                                self._pool_stats_lock = threading.Lock()
                            if not hasattr(self, '_pool_stats'):
                                self._pool_stats = {
                                    'total_connections': 0,
                                    'active_connections': 0,
                                    'failed_connections': 0,
                                    'returned_connections': 0
                                }
                            with self._pool_stats_lock:
                                self._pool_stats['total_connections'] += 1
                                self._pool_stats['active_connections'] += 1
                    except Exception as stats_error:
                        logger.debug(f"Failed to update pool stats (connection still returned): {stats_error}")
                    return PostgreSQLConnection(conn)
                else:
                    raise ValueError(f"Unsupported database type: {self.db_type}")
                    
            except psycopg2.pool.PoolError as e:
                elapsed = time.time() - start_time
                if elapsed >= timeout:
                    with self._pool_stats_lock:
                        self._pool_stats['failed_connections'] += 1
                    raise TimeoutError(f"Connection timeout after {timeout}s: {e}")
                
                retry_count += 1
                if retry_count >= max_retries:
                    with self._pool_stats_lock:
                        self._pool_stats['failed_connections'] += 1
                    raise RuntimeError(f"Failed to get connection after {max_retries} retries: {e}")
                
                wait_time = min(0.5 * retry_count, 2.0)  # 지수 백오프 (최대 2초)
                logger.debug(f"Retrying connection ({retry_count}/{max_retries}) after {wait_time:.1f}s...")
                time.sleep(wait_time)
    
    def _validate_connection(self, conn: DatabaseConnection):
        """연결 상태 검증"""
        if hasattr(conn, '_is_closed') and conn._is_closed():
            raise psycopg2.InterfaceError("Connection is closed")
        
        # 간단한 쿼리로 연결 상태 확인
        try:
            if hasattr(conn, 'conn'):
                cursor = conn.conn.cursor()
                cursor.execute("SELECT 1")
                cursor.close()
        except Exception as e:
            raise psycopg2.InterfaceError(f"Connection validation failed: {e}") from e
    
    def _safe_return_connection(self, conn_wrapper: DatabaseConnection):
        """안전하게 연결 반환 (에러 발생해도 연결은 반환)"""
        if not conn_wrapper:
            return
        
        if self.db_type == 'postgresql' and self.connection_pool:
            if hasattr(conn_wrapper, 'conn'):
                try:
                    # 연결 상태 확인
                    if hasattr(conn_wrapper, '_is_closed') and not conn_wrapper._is_closed():
                        # 연결이 유효한 경우에만 풀에 반환
                        self.connection_pool.putconn(conn_wrapper.conn)
                        # 통계 업데이트 (에러 발생해도 연결은 이미 반환됨)
                        try:
                            if hasattr(self, '_pool_stats_lock') and hasattr(self, '_pool_stats'):
                                with self._pool_stats_lock:
                                    self._pool_stats['active_connections'] = max(0, self._pool_stats['active_connections'] - 1)
                                    self._pool_stats['returned_connections'] += 1
                            else:
                                # 속성이 없으면 초기화 (하위 호환성)
                                if not hasattr(self, '_pool_stats_lock'):
                                    self._pool_stats_lock = threading.Lock()
                                if not hasattr(self, '_pool_stats'):
                                    self._pool_stats = {
                                        'total_connections': 0,
                                        'active_connections': 0,
                                        'failed_connections': 0,
                                        'returned_connections': 0
                                    }
                                with self._pool_stats_lock:
                                    self._pool_stats['active_connections'] = max(0, self._pool_stats['active_connections'] - 1)
                                    self._pool_stats['returned_connections'] += 1
                        except Exception as stats_error:
                            logger.debug(f"Failed to update pool stats (connection already returned): {stats_error}")
                        logger.debug("Connection returned to pool successfully")
                    else:
                        # 연결이 이미 닫혀있으면 풀에서 제거됨
                        try:
                            if hasattr(self, '_pool_stats_lock') and hasattr(self, '_pool_stats'):
                                with self._pool_stats_lock:
                                    self._pool_stats['active_connections'] = max(0, self._pool_stats['active_connections'] - 1)
                        except Exception:
                            pass
                        logger.debug("Connection already closed, not returning to pool")
                except Exception as e:
                    logger.warning(f"Error returning connection to pool: {e}")
                    # 연결이 손상된 경우 닫기 시도
                    try:
                        if hasattr(conn_wrapper, 'conn'):
                            conn_wrapper.conn.close()
                    except Exception:
                        pass
                    # 통계 업데이트 시도 (에러 발생해도 시도)
                    try:
                        if hasattr(self, '_pool_stats_lock') and hasattr(self, '_pool_stats'):
                            with self._pool_stats_lock:
                                self._pool_stats['active_connections'] = max(0, self._pool_stats['active_connections'] - 1)
                    except Exception:
                        pass
    
    @contextmanager
    def get_connection_context(self, timeout: Optional[int] = None):
        """
        개선된 컨텍스트 매니저를 사용한 연결 가져오기
        
        특징:
        - 항상 연결 반환 보장 (예외 발생 시에도)
        - 타임아웃 지원
        - 자동 재연결
        - 연결 상태 검증
        - 실행 시간 모니터링
        - 쿼리별 실행 시간 로깅 (느린 쿼리 자동 감지)
        
        사용 예:
            with adapter.get_connection_context() as conn:
                cursor = conn.cursor()
                cursor.execute("SELECT ...")
        
        Args:
            timeout: 연결 타임아웃 (초), None이면 환경 변수 DB_CONNECTION_TIMEOUT 사용 (기본 30초)
        """
        conn_wrapper = None
        start_time = time.time()
        query_times = []  # 쿼리별 실행 시간 저장
        query_count = 0
        
        try:
            # 타임아웃이 설정된 경우 연결 대기 시간 제한
            conn_wrapper = self._get_connection_with_timeout(timeout)
            
            # 연결 상태 검증
            self._validate_connection(conn_wrapper)
            
            # 쿼리 실행 추적을 위한 cursor 래퍼 추가
            if isinstance(conn_wrapper, PostgreSQLConnection):
                original_cursor = conn_wrapper.cursor
                
                def tracked_cursor(*args, **kwargs):
                    cursor = original_cursor(*args, **kwargs)
                    original_execute = cursor.execute
                    
                    def tracked_execute(query, *args, **kwargs):
                        nonlocal query_count
                        query_start_time = time.time()
                        query_count += 1
                        
                        # 쿼리 정리 (로깅용)
                        query_str = query[:200] if isinstance(query, str) else str(query)[:200]
                        
                        try:
                            result = original_execute(query, *args, **kwargs)
                            query_elapsed = time.time() - query_start_time
                            query_times.append(query_elapsed)
                            
                            # 느린 쿼리 감지 (0.5초 이상)
                            slow_query_threshold = float(os.getenv("DB_SLOW_QUERY_THRESHOLD", "0.5"))
                            if query_elapsed > slow_query_threshold:
                                logger.warning(
                                    f"🐌 Slow query detected ({query_elapsed:.3f}s): {query_str}..."
                                )
                            elif query_elapsed > 0.1:  # 0.1초 이상은 DEBUG 레벨로 로깅
                                logger.debug(
                                    f"⏱️  Query executed ({query_elapsed:.3f}s): {query_str}..."
                                )
                            
                            return result
                        except Exception as e:
                            query_elapsed = time.time() - query_start_time
                            query_times.append(query_elapsed)
                            logger.error(
                                f"❌ Query failed after {query_elapsed:.3f}s: {query_str}... Error: {e}"
                            )
                            raise
                    
                    cursor.execute = tracked_execute
                    return cursor
                
                # cursor 메서드를 래핑된 버전으로 교체
                conn_wrapper.cursor = tracked_cursor
            
            yield conn_wrapper
            
            # 정상 종료 시 commit (트랜잭션이 있는 경우)
            try:
                if hasattr(conn_wrapper, '_is_closed') and not conn_wrapper._is_closed():
                    conn_wrapper.commit()
            except (psycopg2.InterfaceError, AttributeError) as commit_error:
                logger.debug(f"Commit skipped (connection may be closed): {commit_error}")
                
        except psycopg2.pool.PoolError as e:
            # 연결 풀 고갈 시 재시도
            logger.warning(f"Connection pool exhausted, retrying with timeout={timeout}...")
            try:
                conn_wrapper = self._get_connection_with_timeout(timeout)
                self._validate_connection(conn_wrapper)
                yield conn_wrapper
                # 정상 종료 시 commit
                try:
                    if hasattr(conn_wrapper, '_is_closed') and not conn_wrapper._is_closed():
                        conn_wrapper.commit()
                except (psycopg2.InterfaceError, AttributeError):
                    pass
            except Exception as retry_error:
                # 재시도 실패 시 원래 예외 발생
                raise e from retry_error
                
        except Exception as e:
            # rollback 시도 (연결이 닫혀있으면 무시)
            if conn_wrapper:
                try:
                    if hasattr(conn_wrapper, '_is_closed') and not conn_wrapper._is_closed():
                        conn_wrapper.rollback()
                except (psycopg2.InterfaceError, AttributeError) as rollback_error:
                    logger.debug(f"Rollback skipped (connection may be closed): {rollback_error}")
            
            # 예외 정보를 상세히 로깅
            import traceback
            error_type = type(e).__name__
            error_message = str(e) if e else "Unknown error"
            error_repr = repr(e) if e else "Unknown error"
            
            # 예외 객체가 비정상적인 경우를 감지
            if not error_message or error_message == "0" or error_repr == "0":
                logger.error(
                    f"Database error in connection context: {error_type} - "
                    f"message='{error_message}', repr='{error_repr}'\n"
                    f"Traceback:\n{traceback.format_exc()}"
                )
            else:
                logger.error(
                    f"Database error in connection context: {error_type}: {error_message}"
                )
            raise
            
        finally:
            # 항상 연결 반환 (예외 발생 여부와 무관)
            if conn_wrapper:
                self._safe_return_connection(conn_wrapper)
            
            # 실행 시간 로깅
            elapsed = time.time() - start_time
            
            # 쿼리 통계 계산
            if query_count > 0:
                avg_query_time = sum(query_times) / len(query_times) if query_times else 0
                max_query_time = max(query_times) if query_times else 0
                total_query_time = sum(query_times)
                
                # 연결 유지 시간이 1초 이상이거나 쿼리가 여러 개인 경우 상세 로깅
                if elapsed > 1.0 or query_count > 1:
                    logger.warning(
                        f"🔗 Connection held for {elapsed:.2f}s "
                        f"(queries: {query_count}, "
                        f"total query time: {total_query_time:.3f}s, "
                        f"avg: {avg_query_time:.3f}s, "
                        f"max: {max_query_time:.3f}s)"
                    )
                elif elapsed > 0.5:  # 0.5초 이상은 DEBUG 레벨
                    logger.debug(
                        f"Connection held for {elapsed:.2f}s (queries: {query_count})"
                    )
            elif elapsed > 1.0:  # 쿼리 없이 1초 이상 유지
                logger.warning(f"Connection held for {elapsed:.2f}s without queries (longer than expected)")
    
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
    
    def analyze_query_performance(
        self, 
        query: str, 
        params: Optional[Tuple] = None
    ) -> Dict[str, Any]:
        """
        쿼리 성능 분석 (EXPLAIN ANALYZE 실행)
        
        Args:
            query: 분석할 쿼리
            params: 쿼리 파라미터
            
        Returns:
            성능 분석 결과 딕셔너리
        """
        explain_query = f"EXPLAIN (ANALYZE, BUFFERS, VERBOSE) {query}"
        
        with self.get_connection_context() as conn:
            cursor = conn.cursor()
            start_time = time.time()
            try:
                cursor.execute(explain_query, params)
                explain_result = cursor.fetchall()
                elapsed = time.time() - start_time
                
                # 결과 파싱
                explain_plan = '\n'.join([
                    str(row) if not hasattr(row, 'keys') else 
                    row.get('QUERY PLAN', str(row))
                    for row in explain_result
                ])
                
                return {
                    'query': query,
                    'params': params,
                    'execution_time': elapsed,
                    'explain_plan': explain_plan,
                    'raw_result': explain_result
                }
            except Exception as e:
                logger.error(f"Query performance analysis failed: {e}")
                logger.error(f"Query: {query[:200]}...")
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
    
    def get_pool_status(self) -> Dict[str, Any]:
        """
        연결 풀 상태 조회
        
        Returns:
            연결 풀 상태 정보 딕셔너리
        """
        if not self.connection_pool:
            return {"status": "not_initialized"}
        
        try:
            # _pool_stats_lock이 없으면 초기화 (하위 호환성)
            if not hasattr(self, '_pool_stats_lock'):
                self._pool_stats_lock = threading.Lock()
            if not hasattr(self, '_pool_stats'):
                self._pool_stats = {
                    'total_connections': 0,
                    'active_connections': 0,
                    'failed_connections': 0,
                    'returned_connections': 0
                }
            
            with self._pool_stats_lock:
                stats = self._pool_stats.copy()
            
            # ThreadedConnectionPool의 기본 정보
            minconn = self.connection_pool.minconn
            maxconn = self.connection_pool.maxconn
            active = stats.get('active_connections', 0)
            available = maxconn - active
            
            utilization = active / maxconn if maxconn > 0 else 0
            
            return {
                "status": "active",
                "minconn": minconn,
                "maxconn": maxconn,
                "active_connections": active,
                "available_connections": available,
                "utilization": utilization,
                "total_connections": stats.get('total_connections', 0),
                "returned_connections": stats.get('returned_connections', 0),
                "failed_connections": stats.get('failed_connections', 0)
            }
        except Exception as e:
            logger.debug(f"Failed to get pool status: {e}")
            # 에러 발생 시 기본값 반환 (연결 가져오기는 계속 가능하도록)
            try:
                minconn = self.connection_pool.minconn if self.connection_pool else 0
                maxconn = self.connection_pool.maxconn if self.connection_pool else 0
                return {
                    "status": "error",
                    "error": str(e),
                    "minconn": minconn,
                    "maxconn": maxconn,
                    "available_connections": maxconn  # 최악의 경우 전체 사용 가능하다고 가정
                }
            except Exception:
                return {"status": "error", "error": str(e), "available_connections": 50}  # 기본값
    
    def close(self):
        """연결 풀 닫기"""
        if self.db_type == 'postgresql' and self.connection_pool:
            self.connection_pool.closeall()
            logger.info("PostgreSQL connection pool closed")

