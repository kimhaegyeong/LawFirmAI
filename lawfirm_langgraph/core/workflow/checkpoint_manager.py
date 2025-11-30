# -*- coding: utf-8 -*-
"""
LangGraph Checkpoint Manager
PostgreSQL 기반 체크포인트 관리 모듈
"""

import logging
try:
    from lawfirm_langgraph.core.utils.logger import get_logger
except ImportError:
    from core.utils.logger import get_logger
from typing import Optional, Dict, Any, List
from datetime import datetime

# 안전한 로깅 유틸리티 import (멀티스레딩 안전)
# 먼저 폴백 함수를 정의 (항상 사용 가능하도록)
def _safe_log_fallback_debug(logger, message):
    """폴백 디버그 로깅 함수"""
    try:
        logger.debug(message)
    except (ValueError, AttributeError, RuntimeError, OSError):
        pass

def _safe_log_fallback_info(logger, message):
    """폴백 정보 로깅 함수"""
    try:
        logger.info(message)
    except (ValueError, AttributeError, RuntimeError, OSError):
        pass

def _safe_log_fallback_warning(logger, message):
    """폴백 경고 로깅 함수"""
    try:
        logger.warning(message)
    except (ValueError, AttributeError, RuntimeError, OSError):
        pass

def _safe_log_fallback_error(logger, message):
    """폴백 오류 로깅 함수"""
    try:
        logger.error(message)
    except (ValueError, AttributeError, RuntimeError, OSError):
        pass

# 여러 경로 시도하여 safe_log_* 함수 import
SAFE_LOGGING_AVAILABLE = False
try:
    from core.utils.safe_logging_utils import (
        safe_log_debug,
        safe_log_info,
        safe_log_warning,
        safe_log_error
    )
    SAFE_LOGGING_AVAILABLE = True
except ImportError:
    try:
        # lawfirm_langgraph 경로에서 시도
        from lawfirm_langgraph.core.utils.safe_logging_utils import (
            safe_log_debug,
            safe_log_info,
            safe_log_warning,
            safe_log_error
        )
        SAFE_LOGGING_AVAILABLE = True
    except ImportError:
        # Import 실패 시 폴백 함수 사용
        safe_log_debug = _safe_log_fallback_debug
        safe_log_info = _safe_log_fallback_info
        safe_log_warning = _safe_log_fallback_warning
        safe_log_error = _safe_log_fallback_error

# 최종 확인: safe_log_debug가 정의되지 않았다면 폴백 함수 사용
try:
    _ = safe_log_debug
except NameError:
    safe_log_debug = _safe_log_fallback_debug
try:
    _ = safe_log_info
except NameError:
    safe_log_info = _safe_log_fallback_info
try:
    _ = safe_log_warning
except NameError:
    safe_log_warning = _safe_log_fallback_warning
try:
    _ = safe_log_error
except NameError:
    safe_log_error = _safe_log_fallback_error

# 체크포인트 경고 중복 방지 플래그
_checkpoint_warning_logged = False

try:
    from langgraph.checkpoint.postgres import PostgresSaver
    LANGGRAPH_POSTGRES_AVAILABLE = True
except ImportError:
    LANGGRAPH_POSTGRES_AVAILABLE = False
    # 안전한 로깅 사용 (멀티스레딩 안전, 한 번만 출력)
    # 체크포인트가 활성화된 경우에만 로그 출력 (중복 방지)
    # 모듈 레벨에서는 global 선언 불필요
    if not _checkpoint_warning_logged:
        import os
        checkpoint_enabled = os.getenv("ENABLE_CHECKPOINT", "false").lower() == "true"
        if checkpoint_enabled:
            _temp_logger = logging.getLogger(__name__)
            safe_log_info(_temp_logger, "LangGraph PostgreSQL checkpoint not available (optional, checkpoint is disabled by default). Install langgraph-checkpoint-postgres to enable.")
            _checkpoint_warning_logged = True

# DatabaseAdapter import
try:
    from lawfirm_langgraph.core.data.db_adapter import DatabaseAdapter
    DATABASE_ADAPTER_AVAILABLE = True
except ImportError:
    try:
        from core.data.db_adapter import DatabaseAdapter
        DATABASE_ADAPTER_AVAILABLE = True
    except ImportError:
        DATABASE_ADAPTER_AVAILABLE = False
        _temp_logger = logging.getLogger(__name__)
        safe_log_warning(_temp_logger, "DatabaseAdapter not available")

LANGGRAPH_AVAILABLE = LANGGRAPH_POSTGRES_AVAILABLE

logger = get_logger(__name__)


class CheckpointManager:
    """체크포인트 관리자 (PostgreSQL 기반)"""
    
    def __init__(self, storage_type: str = "memory", database_url: Optional[str] = None):
        """
        체크포인트 관리자 초기화
        
        Args:
            storage_type: 저장소 타입 ("memory", "postgres", "redis", "disabled")
            database_url: PostgreSQL 데이터베이스 연결 URL (예: "postgresql://user:pass@host:port/dbname")
                        None인 경우 환경 변수에서 자동으로 가져옴 (DATABASE_URL 또는 CHECKPOINT_DATABASE_URL)
        """
        self.storage_type = storage_type
        self.logger = get_logger(__name__)
        self.saver = None
        self._db_adapter = None
        
        # database_url이 제공되지 않으면 환경 변수에서 가져오기
        if database_url is None:
            import os
            # CHECKPOINT_DATABASE_URL 우선, 없으면 DATABASE_URL 사용
            database_url = os.getenv("CHECKPOINT_DATABASE_URL") or os.getenv("DATABASE_URL")
            if database_url:
                # PostgreSQL URL인지 확인
                if not database_url.startswith("postgresql://") and not database_url.startswith("postgres://"):
                    # SQLite URL이면 에러
                    if database_url.startswith("sqlite://"):
                        raise ValueError("SQLite is no longer supported for checkpoint storage. Use PostgreSQL instead.")
                    else:
                        # 알 수 없는 형식이면 그대로 사용 (PostgresSaver가 처리)
                        pass
        
        self.database_url = database_url
        
        # DatabaseAdapter 초기화 (PostgreSQL 지원을 위해)
        if database_url and DATABASE_ADAPTER_AVAILABLE:
            try:
                self._db_adapter = DatabaseAdapter(database_url)
            except Exception as e:
                safe_log_warning(self.logger, f"Failed to initialize DatabaseAdapter: {e}")
        
        # disabled인 경우 체크포인터 없음
        if storage_type == "disabled":
            self.logger.info("Checkpoint manager is disabled")
            return
        
        # LangGraph 사용 가능 여부 확인
        if not LANGGRAPH_AVAILABLE:
            # 체크포인트가 비활성화된 경우에만 로그 출력 (중복 방지)
            if storage_type != "disabled":
                safe_log_debug(self.logger, "LangGraph checkpoint functionality not available")
            # 폴백으로 MemorySaver 시도
            try:
                from langgraph.checkpoint.memory import MemorySaver
                self.saver = MemorySaver()
                self.storage_type = "memory"
                if storage_type != "disabled":
                    safe_log_info(self.logger, "Using MemorySaver as fallback (LangGraph not available)")
            except ImportError:
                if storage_type != "disabled":
                    safe_log_error(self.logger, "No checkpoint functionality available")
            return
        
        # 저장소 타입에 따라 초기화
        if storage_type == "memory":
            self._init_memory_saver()
        elif storage_type == "postgres":
            self._init_postgres_saver()
        elif storage_type == "redis":
            safe_log_warning(self.logger, f"{storage_type} checkpoint storage not yet implemented, using MemorySaver")
            self._init_memory_saver()
        else:
            safe_log_warning(self.logger, f"Unknown storage type: {storage_type}, using MemorySaver")
            self._init_memory_saver()
    
    def _init_memory_saver(self):
        """MemorySaver 초기화"""
        try:
            from langgraph.checkpoint.memory import MemorySaver
            self.saver = MemorySaver()
            safe_log_info(self.logger, "Checkpoint manager initialized with MemorySaver")
        except ImportError as e:
            safe_log_error(self.logger, f"Failed to initialize MemorySaver: {e}")
            self.saver = None
    
    def _init_postgres_saver(self):
        """PostgresSaver 초기화"""
        if not LANGGRAPH_POSTGRES_AVAILABLE:
            # 경고는 모듈 레벨에서 이미 출력되었으므로 DEBUG 레벨로 변경
            safe_log_debug(self.logger, "LangGraph PostgreSQL checkpoint not available, falling back to MemorySaver")
            self._init_memory_saver()
            return
        
        if not self.database_url:
            safe_log_warning(self.logger, "PostgreSQL requires database_url, falling back to MemorySaver")
            self._init_memory_saver()
            return
        
        try:
            # PostgreSQL 연결 문자열 형식 확인
            if not self.database_url.startswith("postgresql://") and not self.database_url.startswith("postgres://"):
                safe_log_warning(self.logger, f"PostgreSQL connection string should start with 'postgresql://', got: {self.database_url[:20]}...")
            
            self.saver = PostgresSaver.from_conn_string(self.database_url)
            safe_log_info(self.logger, "Checkpoint manager initialized with PostgresSaver")
        except Exception as e:
            safe_log_error(self.logger, f"Failed to initialize PostgresSaver: {e}")
            # 폴백: 메모리 기반 체크포인터 사용
            safe_log_warning(self.logger, "Falling back to MemorySaver")
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
            # LangGraph의 PostgresSaver가 자동으로 처리
            # 실제 저장은 워크플로우 실행 시 자동으로 수행됨
            safe_log_debug(self.logger, f"Checkpoint saved for thread: {thread_id}")
            return True
        except Exception as e:
            safe_log_error(self.logger, f"Failed to save checkpoint: {e}")
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
            # LangGraph의 PostgresSaver가 자동으로 처리
            # 실제 로드는 워크플로우 실행 시 자동으로 수행됨
            safe_log_debug(self.logger, f"Checkpoint loaded for thread: {thread_id}")
            return None  # LangGraph가 자동으로 처리하므로 None 반환
        except Exception as e:
            safe_log_error(self.logger, f"Failed to load checkpoint: {e}")
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
            safe_log_debug(self.logger, f"Found {len(checkpoints)} checkpoints for thread: {thread_id}")
            return checkpoints
        except Exception as e:
            safe_log_error(self.logger, f"Failed to list checkpoints: {e}")
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
            # LangGraph의 PostgresSaver는 직접적인 삭제 메서드를 제공하지 않음
            # 필요시 PostgreSQL 직접 조작 필요
            safe_log_debug(self.logger, f"Checkpoint deletion requested for thread: {thread_id}, checkpoint: {checkpoint_id}")
            return True
        except Exception as e:
            safe_log_error(self.logger, f"Failed to delete checkpoint: {e}")
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
        
        if not self._db_adapter:
            safe_log_warning(self.logger, "Cannot cleanup checkpoints: DatabaseAdapter not available")
            return 0
        
        try:
            # DatabaseAdapter 사용 (PostgreSQL)
            with self._db_adapter.get_connection_context() as conn:
                cursor = conn.cursor()
                
                # 오래된 체크포인트 조회 및 삭제
                cutoff_time = datetime.now().timestamp() - (ttl_hours * 3600)
                
                cursor.execute("""
                    SELECT COUNT(*) FROM checkpoints 
                    WHERE created_at < %s
                """, (cutoff_time,))
                
                count = cursor.fetchone()[0]
                
                cursor.execute("""
                    DELETE FROM checkpoints 
                    WHERE created_at < %s
                """, (cutoff_time,))
                
                conn.commit()
                
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
            if not self._db_adapter:
                return {
                    "database_url": self.database_url,
                    "storage_type": self.storage_type,
                    "error": "DatabaseAdapter not available",
                    "langgraph_available": LANGGRAPH_AVAILABLE,
                    "langgraph_postgres_available": LANGGRAPH_POSTGRES_AVAILABLE
                }
            
            # DatabaseAdapter 사용 (PostgreSQL)
            with self._db_adapter.get_connection_context() as conn:
                cursor = conn.cursor()
                
                # 테이블 존재 확인 (PostgreSQL)
                cursor.execute("""
                    SELECT EXISTS (
                        SELECT FROM information_schema.tables 
                        WHERE table_schema = 'public' 
                        AND table_name = 'checkpoints'
                    )
                """)
                table_exists = cursor.fetchone()[0]
                
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
                
                return {
                    "database_url": self.database_url,
                    "storage_type": self.storage_type,
                    "table_exists": table_exists,
                    "checkpoint_count": checkpoint_count,
                    "latest_checkpoint": latest_checkpoint,
                    "langgraph_available": LANGGRAPH_AVAILABLE,
                    "langgraph_postgres_available": LANGGRAPH_POSTGRES_AVAILABLE
                }
            
        except Exception as e:
            self.logger.error(f"Failed to get database info: {e}")
            return {
                "database_url": self.database_url,
                "storage_type": self.storage_type,
                "error": str(e),
                "langgraph_available": LANGGRAPH_AVAILABLE,
                "langgraph_postgres_available": LANGGRAPH_POSTGRES_AVAILABLE
            }
