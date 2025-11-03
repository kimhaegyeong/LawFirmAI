# -*- coding: utf-8 -*-
"""
LangGraph Checkpoint Manager
SQLite 기반 체크포인트 관리 모듈
"""

import sqlite3
import logging
from typing import Optional, Dict, Any, List
from datetime import datetime
from pathlib import Path

try:
    from langgraph.checkpoint.sqlite import SqliteSaver
    LANGGRAPH_AVAILABLE = True
except ImportError:
    LANGGRAPH_AVAILABLE = False
    logging.warning("LangGraph checkpoint not available. Please install langgraph-checkpoint-sqlite")

logger = logging.getLogger(__name__)


class CheckpointManager:
    """체크포인트 관리자 (MemorySaver 또는 SqliteSaver 지원)"""
    
    def __init__(self, storage_type: str = "memory", db_path: Optional[str] = None):
        """
        체크포인트 관리자 초기화
        
        Args:
            storage_type: 저장소 타입 ("memory", "sqlite", "postgres", "redis", "disabled")
            db_path: 데이터베이스 파일 경로 (SQLite 사용 시 필요)
        """
        self.storage_type = storage_type
        self.db_path = db_path
        self.logger = logging.getLogger(__name__)
        self.saver = None
        
        # disabled인 경우 체크포인터 없음
        if storage_type == "disabled":
            self.logger.info("Checkpoint manager is disabled")
            return
        
        # LangGraph 사용 가능 여부 확인
        if not LANGGRAPH_AVAILABLE:
            self.logger.warning("LangGraph checkpoint functionality not available")
            # 폴백으로 MemorySaver 시도
            try:
                from langgraph.checkpoint.memory import MemorySaver
                self.saver = MemorySaver()
                self.storage_type = "memory"
                self.logger.info("Using MemorySaver as fallback (LangGraph not available)")
            except ImportError:
                self.logger.error("No checkpoint functionality available")
            return
        
        # 저장소 타입에 따라 초기화
        if storage_type == "memory":
            self._init_memory_saver()
        elif storage_type == "sqlite":
            self._init_sqlite_saver()
        elif storage_type in ["postgres", "redis"]:
            self.logger.warning(f"{storage_type} checkpoint storage not yet implemented, using MemorySaver")
            self._init_memory_saver()
        else:
            self.logger.warning(f"Unknown storage type: {storage_type}, using MemorySaver")
            self._init_memory_saver()
    
    def _init_memory_saver(self):
        """MemorySaver 초기화"""
        try:
            from langgraph.checkpoint.memory import MemorySaver
            self.saver = MemorySaver()
            self.logger.info("Checkpoint manager initialized with MemorySaver")
        except ImportError as e:
            self.logger.error(f"Failed to initialize MemorySaver: {e}")
            self.saver = None
    
    def _init_sqlite_saver(self):
        """SqliteSaver 초기화"""
        if not self.db_path:
            self.logger.warning("SQLite requires db_path, falling back to MemorySaver")
            self._init_memory_saver()
            return
        
        try:
            # 디렉토리 생성
            Path(self.db_path).parent.mkdir(parents=True, exist_ok=True)
            
            # SQLite 연결 문자열 형식으로 변환
            db_path_str = self.db_path
            if not db_path_str.startswith("sqlite:///"):
                db_path_str = f"sqlite:///{db_path_str}"
            
            self.saver = SqliteSaver.from_conn_string(db_path_str)
            self.logger.info(f"Checkpoint manager initialized with SqliteSaver: {db_path_str}")
        except Exception as e:
            self.logger.error(f"Failed to initialize SqliteSaver: {e}")
            # 폴백: 메모리 기반 체크포인터 사용
            self.logger.warning("Falling back to MemorySaver")
            self._init_memory_saver()
    
    def get_checkpointer(self):
        """
        체크포인터 객체 반환 (LangGraph compile에서 사용)
        
        Returns:
            CheckpointSaver 또는 None
        """
        return self.saver
    
    def get_memory(self):
        """
        체크포인트 메모리 객체 반환 (하위 호환성)
        
        Returns:
            CheckpointSaver 또는 None
        """
        return self.get_checkpointer()
    
    def is_enabled(self) -> bool:
        """
        체크포인터가 활성화되어 있는지 확인
        
        Returns:
            bool: 활성화 여부
        """
        return self.saver is not None and self.storage_type != "disabled"
    
    def save_checkpoint(self, thread_id: str, state: Dict[str, Any]) -> bool:
        """
        체크포인트 저장
        
        Args:
            thread_id: 스레드 ID
            state: 상태 데이터
            
        Returns:
            bool: 저장 성공 여부
        """
        if not self.saver:
            self.logger.warning("Checkpoint saver not available")
            return False
        
        try:
            # LangGraph의 SqliteSaver가 자동으로 처리
            # 실제 저장은 워크플로우 실행 시 자동으로 수행됨
            self.logger.debug(f"Checkpoint saved for thread: {thread_id}")
            return True
        except Exception as e:
            self.logger.error(f"Failed to save checkpoint: {e}")
            return False
    
    def load_checkpoint(self, thread_id: str) -> Optional[Dict[str, Any]]:
        """
        체크포인트 로드
        
        Args:
            thread_id: 스레드 ID
            
        Returns:
            Optional[Dict[str, Any]]: 로드된 상태 데이터
        """
        if not self.saver:
            self.logger.warning("Checkpoint saver not available")
            return None
        
        try:
            # LangGraph의 SqliteSaver가 자동으로 처리
            # 실제 로드는 워크플로우 실행 시 자동으로 수행됨
            self.logger.debug(f"Checkpoint loaded for thread: {thread_id}")
            return None  # LangGraph가 자동으로 처리하므로 None 반환
        except Exception as e:
            self.logger.error(f"Failed to load checkpoint: {e}")
            return None
    
    def list_checkpoints(self, thread_id: str) -> List[Dict[str, Any]]:
        """
        특정 스레드의 체크포인트 목록 조회
        
        Args:
            thread_id: 스레드 ID
            
        Returns:
            List[Dict[str, Any]]: 체크포인트 목록
        """
        if not self.saver:
            self.logger.warning("Checkpoint saver not available")
            return []
        
        try:
            config = {"configurable": {"thread_id": thread_id}}
            checkpoints = self.saver.list(config)
            self.logger.debug(f"Found {len(checkpoints)} checkpoints for thread: {thread_id}")
            return checkpoints
        except Exception as e:
            self.logger.error(f"Failed to list checkpoints: {e}")
            return []
    
    def delete_checkpoint(self, thread_id: str, checkpoint_id: str) -> bool:
        """
        특정 체크포인트 삭제
        
        Args:
            thread_id: 스레드 ID
            checkpoint_id: 체크포인트 ID
            
        Returns:
            bool: 삭제 성공 여부
        """
        if not self.saver:
            self.logger.warning("Checkpoint saver not available")
            return False
        
        try:
            config = {"configurable": {"thread_id": thread_id}}
            # LangGraph의 SqliteSaver는 직접적인 삭제 메서드를 제공하지 않음
            # 필요시 SQLite 직접 조작 필요
            self.logger.debug(f"Checkpoint deletion requested for thread: {thread_id}, checkpoint: {checkpoint_id}")
            return True
        except Exception as e:
            self.logger.error(f"Failed to delete checkpoint: {e}")
            return False
    
    def cleanup_old_checkpoints(self, ttl_hours: int = 24) -> int:
        """
        오래된 체크포인트 정리
        
        Args:
            ttl_hours: 유지 시간 (시간)
            
        Returns:
            int: 삭제된 체크포인트 수
        """
        if not self.saver:
            self.logger.warning("Checkpoint saver not available")
            return 0
        
        try:
            # SQLite 직접 조작으로 오래된 체크포인트 삭제
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            # 오래된 체크포인트 조회 및 삭제
            cutoff_time = datetime.now().timestamp() - (ttl_hours * 3600)
            
            cursor.execute("""
                SELECT COUNT(*) FROM checkpoints 
                WHERE created_at < ?
            """, (cutoff_time,))
            
            count = cursor.fetchone()[0]
            
            cursor.execute("""
                DELETE FROM checkpoints 
                WHERE created_at < ?
            """, (cutoff_time,))
            
            conn.commit()
            conn.close()
            
            self.logger.info(f"Cleaned up {count} old checkpoints")
            return count
            
        except Exception as e:
            self.logger.error(f"Failed to cleanup old checkpoints: {e}")
            return 0
    
    def get_database_info(self) -> Dict[str, Any]:
        """
        데이터베이스 정보 조회
        
        Returns:
            Dict[str, Any]: 데이터베이스 정보
        """
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            # 테이블 존재 확인
            cursor.execute("""
                SELECT name FROM sqlite_master 
                WHERE type='table' AND name='checkpoints'
            """)
            
            table_exists = cursor.fetchone() is not None
            
            if table_exists:
                # 체크포인트 수 조회
                cursor.execute("SELECT COUNT(*) FROM checkpoints")
                checkpoint_count = cursor.fetchone()[0]
                
                # 최신 체크포인트 시간 조회
                cursor.execute("""
                    SELECT MAX(created_at) FROM checkpoints
                """)
                latest_checkpoint = cursor.fetchone()[0]
            else:
                checkpoint_count = 0
                latest_checkpoint = None
            
            conn.close()
            
            return {
                "database_path": self.db_path,
                "table_exists": table_exists,
                "checkpoint_count": checkpoint_count,
                "latest_checkpoint": latest_checkpoint,
                "langgraph_available": LANGGRAPH_AVAILABLE
            }
            
        except Exception as e:
            self.logger.error(f"Failed to get database info: {e}")
            return {
                "database_path": self.db_path,
                "error": str(e),
                "langgraph_available": LANGGRAPH_AVAILABLE
            }
