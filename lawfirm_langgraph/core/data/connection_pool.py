# -*- coding: utf-8 -*-
"""
Database Connection Pool
SQLite 연결 풀링을 위한 Thread-local Connection Manager
"""

import logging
try:
    from lawfirm_langgraph.core.utils.logger import get_logger
except ImportError:
    from core.utils.logger import get_logger
import sqlite3
import threading
from contextlib import contextmanager
from typing import Dict, Optional

logger = get_logger(__name__)


class ThreadLocalConnectionPool:
    """
    Thread-local SQLite 연결 풀
    
    각 스레드마다 독립적인 연결을 유지하여 스레드 안전성을 보장하면서
    연결 재사용을 통해 성능을 향상시킵니다.
    """
    
    def __init__(self, db_path: str):
        """
        연결 풀 초기화
        
        Args:
            db_path: 데이터베이스 파일 경로
        """
        self.db_path = db_path
        self._local = threading.local()
        self.logger = get_logger(__name__)
    
    def get_connection(self) -> sqlite3.Connection:
        """
        현재 스레드의 연결 가져오기 (없으면 생성)
        
        Returns:
            sqlite3.Connection: 데이터베이스 연결 객체
        """
        if not hasattr(self._local, 'connection'):
            self._local.connection = sqlite3.connect(
                self.db_path,
                check_same_thread=False
            )
            self._local.connection.row_factory = sqlite3.Row
            self.logger.debug(f"Created new connection for thread {threading.current_thread().ident}")
        return self._local.connection
    
    @contextmanager
    def get_connection_context(self):
        """
        컨텍스트 매니저를 사용한 연결 가져오기
        
        사용 예:
            with pool.get_connection_context() as conn:
                cursor = conn.cursor()
                cursor.execute("SELECT ...")
        """
        conn = self.get_connection()
        try:
            yield conn
        except Exception as e:
            conn.rollback()
            self.logger.error(f"Database error in connection context: {e}")
            raise
    
    def close_connection(self):
        """현재 스레드의 연결 닫기"""
        if hasattr(self._local, 'connection'):
            try:
                self._local.connection.close()
                delattr(self._local, 'connection')
                self.logger.debug(f"Closed connection for thread {threading.current_thread().ident}")
            except Exception as e:
                self.logger.warning(f"Error closing connection: {e}")
    
    def reset_connection(self):
        """현재 스레드의 연결을 닫고 새로 생성"""
        self.close_connection()
        return self.get_connection()


# 전역 연결 풀 저장소 (db_path별로 관리)
_connection_pools: Dict[str, ThreadLocalConnectionPool] = {}
_pools_lock = threading.Lock()


def get_connection_pool(db_path: str) -> ThreadLocalConnectionPool:
    """
    데이터베이스 경로별 연결 풀 가져오기 (싱글톤 패턴)
    
    Args:
        db_path: 데이터베이스 파일 경로
        
    Returns:
        ThreadLocalConnectionPool: 해당 경로의 연결 풀 인스턴스
    """
    with _pools_lock:
        if db_path not in _connection_pools:
            _connection_pools[db_path] = ThreadLocalConnectionPool(db_path)
        return _connection_pools[db_path]


def close_all_pools():
    """모든 연결 풀의 연결 닫기 (주로 테스트나 종료 시 사용)"""
    with _pools_lock:
        for pool in _connection_pools.values():
            pool.close_connection()
        _connection_pools.clear()

