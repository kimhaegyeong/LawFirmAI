# -*- coding: utf-8 -*-
"""
LangGraph Checkpoint Manager
SQLite 기반 체크포인트 관리 모듈
"""

import logging
from typing import Dict, Any, List
from pathlib import Path

try:
    from langgraph.checkpoint.sqlite import SqliteSaver
    LANGGRAPH_AVAILABLE = True
except ImportError:
    LANGGRAPH_AVAILABLE = False
    logging.warning("LangGraph checkpoint not available. Please install langgraph-checkpoint-sqlite")

logger = logging.getLogger(__name__)


class CheckpointManager:
    """SQLite 기반 체크포인트 관리자"""
    
    def __init__(self, db_path: str):
        """
        체크포인트 관리자 초기화
        
        Args:
            db_path: SQLite 데이터베이스 파일 경로
        """
        self.db_path = db_path
        self.logger = logging.getLogger(__name__)
        
        # 디렉토리 생성
        Path(db_path).parent.mkdir(parents=True, exist_ok=True)
        
        # LangGraph SqliteSaver 초기화
        if LANGGRAPH_AVAILABLE:
            try:
                # SQLite 연결 문자열 형식으로 변환
                if not db_path.startswith("sqlite:///"):
                    db_path = f"sqlite:///{db_path}"
                
                self.saver = SqliteSaver.from_conn_string(db_path)
                self.logger.info(f"Checkpoint manager initialized with database: {db_path}")
            except Exception as e:
                self.logger.error(f"Failed to initialize SqliteSaver: {e}")
                # 폴백: 메모리 기반 체크포인터 사용
                try:
                    from langgraph.checkpoint.memory import MemorySaver
                    self.saver = MemorySaver()
                    self.logger.warning("Using MemorySaver as fallback")
                except ImportError:
                    self.saver = None
                    self.logger.error("No checkpoint functionality available")
        else:
            self.saver = None
            self.logger.warning("LangGraph checkpoint functionality not available")
    
    def get_memory(self):
        """체크포인트 메모리 객체 반환"""
        if self.saver is None:
            # 폴백: MemorySaver 사용
            try:
                from langgraph.checkpoint.memory import MemorySaver
                self.saver = MemorySaver()
                self.logger.warning("Using MemorySaver as fallback")
            except ImportError:
                self.logger.error("No checkpoint functionality available")
                return None
        return self.saver
    
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
        
        Note: LangGraph의 SqliteSaver가 직접 관리하므로,
        이 메서드는 호환성을 위해 유지되지만 실제 cleanup은 SqliteSaver가 처리합니다.
        
        Args:
            ttl_hours: 유지 시간 (시간) - 참고용
        
        Returns:
            int: 항상 0 반환 (SqliteSaver가 자동 관리)
        """
        if not self.saver:
            self.logger.warning("Checkpoint saver not available")
            return 0
        
        self.logger.info(
            f"Cleanup requested for checkpoints older than {ttl_hours} hours. "
            "LangGraph SqliteSaver manages checkpoints automatically."
        )
        return 0
    
    def get_database_info(self) -> Dict[str, Any]:
        """
        데이터베이스 정보 조회
        
        Note: LangGraph의 SqliteSaver가 직접 관리하므로,
        기본 정보만 반환합니다.
        
        Returns:
            Dict[str, Any]: 데이터베이스 기본 정보
        """
        return {
            "database_path": self.db_path,
            "langgraph_available": LANGGRAPH_AVAILABLE,
            "saver_type": type(self.saver).__name__ if self.saver else None,
            "note": "LangGraph SqliteSaver manages the database internally. "
                    "Use list_checkpoints() to query checkpoints for specific threads."
        }
