# -*- coding: utf-8 -*-
"""
LangGraph Workflow Service
워크플로우 서비스 통합 클래스
"""

import asyncio
import logging
import os
import sys
import time
import uuid
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

# 프로젝트 루트를 sys.path에 추가 (lawfirm_langgraph 구조에 맞게 수정)
# lawfirm_langgraph/langgraph_core/services/ 에서 프로젝트 루트까지의 경로
project_root = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(project_root))

# .env 파일 로드 (python-dotenv 패키지 필요)
try:
    from dotenv import load_dotenv
    # lawfirm_langgraph 디렉토리의 .env 파일 로드
    # langgraph_core/services/ 에서 lawfirm_langgraph/ 까지 상위 2단계
    langgraph_dir = Path(__file__).parent.parent.parent
    env_file = langgraph_dir / ".env"
    load_dotenv(dotenv_path=str(env_file))
except ImportError:
    # python-dotenv가 설치되지 않은 경우 경고만 출력하고 계속 진행
    logging.warning("python-dotenv not installed. .env file will not be loaded.")

# 워크플로우 import (상대 import 사용 - 같은 패키지 내부)
try:
    from .legal_workflow_enhanced import EnhancedLegalQuestionWorkflow
except ImportError:
    # Fallback: 프로젝트 루트 기준 import
    from core.workflow.legal_workflow_enhanced import EnhancedLegalQuestionWorkflow

# 콜백 핸들러 import
try:
    from .callbacks.streaming_callback_handler import StreamingCallbackHandler
except ImportError:
    try:
        from core.workflow.callbacks.streaming_callback_handler import StreamingCallbackHandler
    except ImportError:
        StreamingCallbackHandler = None

# ConversationFlowTracker import
try:
    from ..conversation.conversation_flow_tracker import ConversationFlowTracker
    from ..conversation.conversation_manager import ConversationContext, ConversationTurn
except ImportError:
    try:
        from core.conversation.conversation_flow_tracker import ConversationFlowTracker
        from core.conversation.conversation_manager import ConversationContext, ConversationTurn
    except ImportError:
        try:
            # 호환성을 위한 fallback
            from core.conversation.conversation_flow_tracker import ConversationFlowTracker
            from core.conversation.conversation_manager import ConversationContext, ConversationTurn
        except ImportError:
            ConversationFlowTracker = None
            ConversationContext = None
            ConversationTurn = None

# 설정 파일 import (lawfirm_langgraph 구조 우선 시도)
try:
    from lawfirm_langgraph.config.langgraph_config import LangGraphConfig
except ImportError:
    # Fallback: 기존 경로 (호환성 유지)
    try:
        from core.utils.langgraph_config import LangGraphConfig
    except ImportError:
        from core.utils.langgraph_config import LangGraphConfig

# Langfuse 클라이언트 통합 (지연 import - __init__에서 실제 사용)
LANGFUSE_CLIENT_AVAILABLE = False
LANGFUSE_TRACE_AVAILABLE = False

logger = logging.getLogger(__name__)

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

if not LANGFUSE_CLIENT_AVAILABLE:
    safe_log_warning(logger, "LangfuseClient not available for LangGraph workflow tracking")

# CheckpointManager import
try:
    from core.agents.checkpoint_manager import CheckpointManager
except ImportError:
    CheckpointManager = None
    safe_log_warning(logger, "CheckpointManager not available")

# state_definitions import (상대 import 사용 - 같은 패키지 내부)
try:
    from ..state.state_definitions import create_initial_legal_state
except ImportError:
    # Fallback: 프로젝트 루트 기준 import
    try:
        from lawfirm_langgraph.core.workflow.state.state_definitions import create_initial_legal_state
    except ImportError:
        # Fallback: 프로젝트 루트 기준 import
        from core.workflow.state.state_definitions import create_initial_legal_state


class LangGraphWorkflowService:
    """LangGraph 워크플로우 서비스"""
    
    # 상수 정의
    SLOW_NODE_THRESHOLD = 5.0  # 노드 실행 시간 임계값 (초)
    DEFAULT_RECURSION_LIMIT = 200  # 기본 재귀 제한
    MAX_PARAMETER_CHECK_COUNT = 5  # 모델 검증 시 확인할 파라미터 개수

    def __init__(self, config: Optional[LangGraphConfig] = None):
        """
        워크플로우 서비스 초기화

        Args:
            config: LangGraph 설정 객체
        """
        self.config = config or LangGraphConfig.from_env()
        self.logger = logging.getLogger(__name__)

        # 초기 input 보존을 위한 인스턴스 변수
        self._initial_input: Optional[Dict[str, str]] = None

        # 검색 결과 보존을 위한 캐시 (LangGraph reducer 문제 우회)
        self._search_results_cache: Optional[Dict[str, Any]] = None

        # ConversationFlowTracker 초기화 (추천 질문 생성용)
        self.conversation_flow_tracker = None
        if ConversationFlowTracker is not None:
            try:
                self.conversation_flow_tracker = ConversationFlowTracker()
                self.logger.info("ConversationFlowTracker initialized for suggested questions")
            except Exception as e:
                self.logger.warning(f"Failed to initialize ConversationFlowTracker: {e}, continuing without suggested questions")
                self.conversation_flow_tracker = None
        
        # 컴포넌트 초기화 - 체크포인터 설정에 따라 초기화
        self.checkpoint_manager = None
        if self.config.enable_checkpoint and CheckpointManager is not None:
            try:
                storage_type = self.config.checkpoint_storage.value
                db_path = self.config.checkpoint_db_path if storage_type == "sqlite" else None
                self.checkpoint_manager = CheckpointManager(
                    storage_type=storage_type,
                    db_path=db_path
                )
                if self.checkpoint_manager.is_enabled():
                    safe_log_info(self.logger, f"Checkpoint manager initialized with {storage_type} storage")
                else:
                    safe_log_warning(self.logger, "Checkpoint manager initialization failed, continuing without checkpoint")
                    self.checkpoint_manager = None
            except Exception as e:
                safe_log_warning(self.logger, f"Failed to initialize checkpoint manager: {e}, continuing without checkpoint")
                self.checkpoint_manager = None
        else:
            if not self.config.enable_checkpoint:
                safe_log_info(self.logger, "Checkpoint is disabled in configuration")
            else:
                safe_log_warning(self.logger, "CheckpointManager class not available")
        
        self.legal_workflow = EnhancedLegalQuestionWorkflow(self.config)

        # LangSmith 활성화 여부 확인 (환경 변수로 제어 가능)
        enable_langsmith = os.environ.get("ENABLE_LANGSMITH", "false").lower() == "true"

        if not enable_langsmith:
            # LangSmith 비활성화 모드 (기본값) - State Reduction으로 최적화된 후에도 기본은 비활성화
            # LangSmith 트레이싱 비활성화 (긴급) - 대용량 상태 로깅 방지
            original_tracing = os.environ.get("LANGSMITH_TRACING") or os.environ.get("LANGCHAIN_TRACING_V2")
            original_api_key = os.environ.get("LANGSMITH_API_KEY") or os.environ.get("LANGCHAIN_API_KEY")
            original_endpoint = os.environ.get("LANGSMITH_ENDPOINT") or os.environ.get("LANGCHAIN_ENDPOINT")
            original_project = os.environ.get("LANGSMITH_PROJECT") or os.environ.get("LANGCHAIN_PROJECT")

            # 임시로 LangSmith 비활성화
            os.environ["LANGSMITH_TRACING"] = "false"
            os.environ["LANGCHAIN_TRACING_V2"] = "false"
            if "LANGSMITH_API_KEY" in os.environ:
                del os.environ["LANGSMITH_API_KEY"]
            if "LANGCHAIN_API_KEY" in os.environ:
                del os.environ["LANGCHAIN_API_KEY"]

            try:
                # 체크포인터 설정 (활성화된 경우)
                checkpointer = None
                if self.checkpoint_manager and self.checkpoint_manager.is_enabled():
                    checkpointer = self.checkpoint_manager.get_checkpointer()
                    safe_log_info(self.logger, f"Using checkpoint: {self.config.checkpoint_storage.value}")
                else:
                    safe_log_info(self.logger, "Compiling workflow without checkpoint")
                
                # 워크플로우 컴파일
                self.app = self.legal_workflow.graph.compile(
                    checkpointer=checkpointer,
                    interrupt_before=None,
                    interrupt_after=None,
                    debug=False,
                )
                if checkpointer:
                    safe_log_info(self.logger, f"워크플로우가 체크포인트({self.config.checkpoint_storage.value})와 함께 컴파일되었습니다 (LangSmith 비활성화됨)")
                else:
                    safe_log_info(self.logger, "워크플로우가 체크포인트 없이 컴파일되었습니다 (LangSmith 비활성화됨)")
            finally:
                # 환경 변수 복원
                if original_tracing:
                    if isinstance(original_tracing, str) and "LANGSMITH" in original_tracing:
                        os.environ["LANGSMITH_TRACING"] = original_tracing
                    else:
                        os.environ["LANGCHAIN_TRACING_V2"] = original_tracing
                elif "LANGSMITH_TRACING" in os.environ:
                    del os.environ["LANGSMITH_TRACING"]
                elif "LANGCHAIN_TRACING_V2" in os.environ:
                    del os.environ["LANGCHAIN_TRACING_V2"]

                if original_api_key:
                    if isinstance(original_api_key, str) and original_api_key.startswith("ls-"):
                        os.environ["LANGSMITH_API_KEY"] = original_api_key
                    else:
                        os.environ["LANGCHAIN_API_KEY"] = original_api_key
                if original_endpoint:
                    os.environ["LANGSMITH_ENDPOINT"] = original_endpoint
                if original_project:
                    os.environ["LANGSMITH_PROJECT"] = original_project
        else:
            # LangSmith 활성화 모드 (ENABLE_LANGSMITH=true로 설정된 경우)
            # LangSmith 환경 변수 설정
            if self.config.langsmith_endpoint:
                os.environ["LANGSMITH_ENDPOINT"] = self.config.langsmith_endpoint
                os.environ["LANGCHAIN_ENDPOINT"] = self.config.langsmith_endpoint
            if self.config.langsmith_api_key:
                os.environ["LANGSMITH_API_KEY"] = self.config.langsmith_api_key
                os.environ["LANGCHAIN_API_KEY"] = self.config.langsmith_api_key
            if self.config.langsmith_project:
                os.environ["LANGSMITH_PROJECT"] = self.config.langsmith_project
                os.environ["LANGCHAIN_PROJECT"] = self.config.langsmith_project
            os.environ["LANGSMITH_TRACING"] = "true"
            os.environ["LANGCHAIN_TRACING_V2"] = "true"
            
            # 체크포인터 설정 (활성화된 경우)
            checkpointer = None
            if self.checkpoint_manager and self.checkpoint_manager.is_enabled():
                checkpointer = self.checkpoint_manager.get_checkpointer()
                safe_log_info(self.logger, f"Using checkpoint: {self.config.checkpoint_storage.value}")
            else:
                safe_log_info(self.logger, "Compiling workflow without checkpoint")
            
            # 워크플로우 컴파일
            self.app = self.legal_workflow.graph.compile(
                checkpointer=checkpointer,
                interrupt_before=None,
                interrupt_after=None,
                debug=False,
            )
            checkpoint_info = f" with checkpoint({self.config.checkpoint_storage.value})" if checkpointer else " without checkpoint"
            safe_log_info(self.logger, f"워크플로우가 LangSmith 추적으로 컴파일되었습니다{checkpoint_info} (State Reduction 적용됨)")

        if self.app is None:
            safe_log_error(self.logger, "Failed to compile workflow")
            raise RuntimeError("워크플로우 컴파일에 실패했습니다")

        # LangfuseClient 초기화 (답변 품질 추적)
        self.langfuse_client_service = None
        if LANGFUSE_CLIENT_AVAILABLE and self.config.langfuse_enabled:
            try:
                try:
                    from core.shared.clients.langfuse_client import LangfuseClient
                except ImportError:
                    # 호환성을 위한 fallback
                    from core.services.langfuse_client import LangfuseClient
                self.langfuse_client_service = LangfuseClient(self.config)
                if self.langfuse_client_service and self.langfuse_client_service.is_enabled():
                    safe_log_info(self.logger, "LangfuseClient initialized for answer quality tracking")
            except Exception as e:
                safe_log_warning(self.logger, f"Failed to initialize LangfuseClient: {e}")
                self.langfuse_client_service = None

        # A/B 테스트 관리자 초기화
        self.ab_test_manager = None
        enable_ab_testing = os.environ.get("ENABLE_AB_TESTING", "false").lower() == "true"
        if enable_ab_testing:
            try:
                from lawfirm_langgraph.core.services.ab_test_manager import ABTestManager
                self.ab_test_manager = ABTestManager()
                safe_log_info(self.logger, "ABTestManager initialized")
            except Exception as e:
                safe_log_warning(self.logger, f"Failed to initialize ABTestManager: {e}")
                self.ab_test_manager = None

        # 스트리밍 콜백 핸들러 초기화 (스트리밍 모드 활성화 시)
        self.streaming_callback_handler = None
        if StreamingCallbackHandler is not None:
            try:
                # 큐는 각 요청마다 생성되므로 여기서는 None으로 초기화
                # 실제 사용 시 ChatService에서 큐를 생성하여 전달
                self.streaming_callback_handler_class = StreamingCallbackHandler
                safe_log_info(self.logger, "StreamingCallbackHandler class available")
            except Exception as e:
                safe_log_warning(self.logger, f"Failed to initialize StreamingCallbackHandler: {e}")
                self.streaming_callback_handler_class = None
        else:
            safe_log_warning(self.logger, "StreamingCallbackHandler not available")

        safe_log_info(self.logger, "LangGraphWorkflowService initialized successfully")
    
    def create_streaming_callback_handler(self, queue: Optional[asyncio.Queue] = None) -> Optional[Any]:
        """
        스트리밍 콜백 핸들러 생성
        
        Args:
            queue: 청크를 저장할 asyncio.Queue. None이면 자동 생성
            
        Returns:
            StreamingCallbackHandler 인스턴스 또는 None
        """
        if self.streaming_callback_handler_class is None:
            return None
        
        try:
            if queue is None:
                queue = asyncio.Queue()
            handler = self.streaming_callback_handler_class(queue=queue)
            return handler
        except Exception as e:
            self.logger.warning(f"Failed to create StreamingCallbackHandler: {e}")
            return None
    
    def get_config_with_callbacks(self, session_id: Optional[str] = None, callbacks: Optional[List[Any]] = None) -> Dict[str, Any]:
        """
        콜백이 포함된 config 생성
        
        Args:
            session_id: 세션 ID
            callbacks: 콜백 핸들러 리스트
            
        Returns:
            LangGraph config 딕셔너리
        """
        config = {"configurable": {}}
        
        if session_id:
            config["configurable"]["thread_id"] = session_id
        
        if callbacks:
            config["callbacks"] = callbacks
        
        return config

    async def process_query(
        self,
        query: str,
        session_id: Optional[str] = None,
        enable_checkpoint: bool = True
    ) -> Dict[str, Any]:
        # 검색 결과 캐시 초기화 (각 쿼리마다 새로 시작)
        self._search_results_cache = None

        """
        질문 처리

        Args:
            query: 사용자 질문
            session_id: 세션 ID (없으면 자동 생성)
            enable_checkpoint: 체크포인트 사용 여부

        Returns:
            Dict[str, Any]: 처리 결과
        """
        try:
            start_time = time.time()

            # 세션 ID 생성
            if not session_id:
                session_id = str(uuid.uuid4())

            # A/B 테스트 변형 할당
            cache_variant = None
            parallel_variant = None
            
            if self.ab_test_manager:
                cache_variant = self.ab_test_manager.assign_variant(session_id, "cache_enabled")
                parallel_variant = self.ab_test_manager.assign_variant(session_id, "parallel_processing")
                
                # 변형에 따라 설정 적용
                cache_config = self.ab_test_manager.get_experiment_config("cache_enabled", cache_variant)
                if cache_config:
                    if not cache_config.get("cache", True):
                        # 캐시 비활성화
                        if hasattr(self.legal_workflow, 'cache_manager') and self.legal_workflow.cache_manager:
                            self.legal_workflow.cache_manager = None
                            self.logger.info(f"A/B Test: Cache disabled for variant {cache_variant}")
                
                parallel_config = self.ab_test_manager.get_experiment_config("parallel_processing", parallel_variant)
                if parallel_config:
                    max_workers = parallel_config.get("max_workers", 2)
                    # 병렬 처리 워커 수 설정 (향후 구현)
                    self.logger.info(f"A/B Test: Parallel processing max_workers={max_workers} for variant {parallel_variant}")

            self.logger.info(f"Processing query: {query[:100]}... (session: {session_id})")
            self.logger.debug(f"process_query: query length={len(query)}, query='{query[:50]}...'")

            # 초기 상태 설정 및 검증
            initial_state = self._prepare_and_validate_initial_state(query, session_id)

            # 워크플로우 실행 설정 (체크포인터 활성화 시 thread_id 설정)
            config = {}
            # 체크포인터가 워크플로우에 설정되어 있으면 thread_id가 필요함
            # enable_checkpoint가 False여도 이미 컴파일된 워크플로우에 체크포인터가 있으면 필요
            if (enable_checkpoint and self.checkpoint_manager and self.checkpoint_manager.is_enabled()) or \
               (self.app and hasattr(self.app, 'checkpointer') and self.app.checkpointer is not None):
                config = {"configurable": {"thread_id": session_id}}
                self.logger.debug(f"Using checkpoint with thread_id: {session_id}")

            # 워크플로우 실행 (스트리밍으로 진행상황 표시)
            if self.app:
                # Recursion limit 증가 (재시도 로직 개선으로 인해 더 높게 설정)
                # 재시도 최대 3회 + 각 단계별 노드 실행을 고려하여 여유있게 설정
                enhanced_config = {"recursion_limit": self.DEFAULT_RECURSION_LIMIT}
                enhanced_config.update(config)

                # 스트리밍으로 워크플로우 실행하여 진행상황 표시
                flat_result = None
                node_count = 0
                executed_nodes = []
                last_node_time = time.time()
                # processing_steps 추적 (state reduction으로 인해 손실될 수 있으므로)
                tracked_processing_steps = []

                self.logger.info("🔄 워크플로우 실행 시작...")
                print("🔄 워크플로우 실행 시작...", flush=True)

                # 초기 state 최종 검증
                if not self._validate_initial_state_before_execution(initial_state, query):
                    raise ValueError("Initial state validation failed before workflow execution")

                # 중요: 초기 input 보존 (모든 노드에서 복원 가능하도록)
                if initial_state.get("input") and isinstance(initial_state["input"], dict):
                    self._initial_input = initial_state["input"].copy()
                    # query가 비어있지 않은지 확인
                    if not self._initial_input.get("query"):
                        # 최상위에서 찾기
                        if initial_state.get("query"):
                            self._initial_input["query"] = initial_state["query"]
                elif initial_state.get("query"):
                    # nested 구조가 아니면 flat에서 추출
                    self._initial_input = {
                        "query": initial_state["query"],
                        "session_id": initial_state.get("session_id", "") if initial_state.get("session_id") else (initial_state.get("input", {}).get("session_id", "") if initial_state.get("input") else "")
                    }
                else:
                    self._initial_input = {"query": "", "session_id": ""}

                # 최종 확인: initial_input에 query가 있어야 함
                if not self._initial_input.get("query"):
                    self.logger.error("CRITICAL: _initial_input has no query! This should never happen.")
                else:
                    self.logger.debug(f"Preserved initial input: query length={len(self._initial_input.get('query', ''))}")

                # 성능 프로파일링을 위한 노드 실행 시간 추적
                node_start_times = {}  # 각 노드의 시작 시간 저장
                node_durations = {}  # 각 노드의 실행 시간 저장
                total_start_time = time.time()
                
                async for event in self.app.astream(initial_state, enhanced_config, stream_mode="updates"):
                    # 각 이벤트는 {node_name: updated_state} 형태
                    for node_name, node_state in event.items():
                        # 새로 실행된 노드인 경우에만 카운트
                        if node_name not in executed_nodes:
                            node_count += 1
                            executed_nodes.append(node_name)
                            
                            # 노드 시작 시간 기록 (이벤트가 발생하면 해당 노드가 완료된 것으로 간주)
                            current_time = time.time()
                            if node_name in node_start_times:
                                # 노드 실행 시간 계산 (시작 시간부터 현재까지)
                                node_duration = current_time - node_start_times[node_name]
                                node_durations[node_name] = node_duration
                            else:
                                # 첫 실행 시 이전 노드 완료 시간으로 계산
                                node_duration = current_time - last_node_time if node_count > 1 else 0
                                node_durations[node_name] = node_duration
                            
                            # 다음 노드 시작 시간 기록 (이벤트 발생 시점)
                            node_start_times[node_name] = current_time
                            last_node_time = current_time

                            # 진행상황 표시 (실행 시간 포함)
                            if node_count == 1:
                                progress_msg = f"  [{node_count}] 🔄 실행 중: {node_name}"
                            else:
                                progress_msg = f"  [{node_count}] 🔄 실행 중: {node_name} (실행 시간: {node_duration:.2f}초)"

                            self.logger.info(progress_msg)
                            print(progress_msg, flush=True)
                            
                            # 병목 지점 감지: 느린 노드에 대한 경고
                            if node_duration > self.SLOW_NODE_THRESHOLD:
                                self.logger.warning(
                                    f"⚠️ [PERFORMANCE] 느린 노드 감지: {node_name}가 {node_duration:.2f}초 소요되었습니다. "
                                    f"(임계값: {self.SLOW_NODE_THRESHOLD}초)"
                                )

                            # 노드 이름을 한국어로 변환하여 더 명확하게 표시
                            node_display_name = self._get_node_display_name(node_name)
                            if node_display_name != node_name:
                                detail_msg = f"      → {node_display_name}"
                                self.logger.info(detail_msg)
                                print(detail_msg, flush=True)

                            # 디버깅: node_state의 query 확인
                            # stream_mode="updates" 사용 시 변경된 필드만 포함되므로 직접 확인 가능
                            if node_name == "classify_query_and_complexity" and isinstance(node_state, dict):
                                # classification 그룹을 캐시에 저장 (stream_mode="updates" 사용 시 다음 노드로 전달 보장)
                                if "classification" in node_state and isinstance(node_state["classification"], dict):
                                    if not self._search_results_cache:
                                        self._search_results_cache = {}
                                    self._search_results_cache["classification"] = node_state["classification"].copy()
                                    # common 그룹에도 저장
                                    if "common" not in self._search_results_cache:
                                        self._search_results_cache["common"] = {}
                                    if "classification" not in self._search_results_cache["common"]:
                                        self._search_results_cache["common"]["classification"] = {}
                                    self._search_results_cache["common"]["classification"].update(node_state["classification"])
                                    self.logger.debug("astream: Cached classification group for future nodes")
                                
                                node_query = ""
                                # input 그룹이 변경되었는지 확인
                                if "input" in node_state and isinstance(node_state["input"], dict):
                                    node_query = node_state["input"].get("query", "")
                                # classification 그룹도 확인
                                elif "classification" in node_state and isinstance(node_state["classification"], dict):
                                    node_query = self._initial_input.get("query", "") if self._initial_input else ""
                                self.logger.debug(f"astream: event[{node_name}] query='{node_query[:50] if node_query else 'EMPTY'}...', keys={list(node_state.keys())}")

                        # processing_steps 추적 (state reduction으로 손실 방지, 개선)
                        # stream_mode="updates" 사용 시 변경된 필드만 포함되므로 직접 확인 가능
                        if isinstance(node_state, dict):
                            # 1. common 그룹에서 processing_steps 확인 (변경된 경우에만 포함)
                            if "common" in node_state and isinstance(node_state["common"], dict):
                                common_steps = node_state["common"].get("processing_steps", [])
                                if isinstance(common_steps, list) and len(common_steps) > 0:
                                    for step in common_steps:
                                        if isinstance(step, str) and step not in tracked_processing_steps:
                                            tracked_processing_steps.append(step)

                            # 2. 최상위 레벨에서도 확인 (변경된 경우에만 포함)
                            if "processing_steps" in node_state:
                                top_steps = node_state["processing_steps"]
                                if isinstance(top_steps, list) and len(top_steps) > 0:
                                    for step in top_steps:
                                        if isinstance(step, str) and step not in tracked_processing_steps:
                                            tracked_processing_steps.append(step)

                            # 3. metadata에서도 확인 (변경된 경우에만 포함)
                            if "metadata" in node_state and isinstance(node_state["metadata"], dict):
                                metadata_steps = node_state["metadata"].get("processing_steps", [])
                                if isinstance(metadata_steps, list) and len(metadata_steps) > 0:
                                    for step in metadata_steps:
                                        if isinstance(step, str) and step not in tracked_processing_steps:
                                            tracked_processing_steps.append(step)

                            # 4. 노드 실행 정보 추가 (추적 보강)
                            if node_name and len(node_name) < 50:
                                node_info = f"노드 실행: {node_name}"
                                if node_info not in tracked_processing_steps:
                                    tracked_processing_steps.append(node_info)

                        # 최종 결과 업데이트 (각 이벤트는 업데이트된 상태를 반환)
                        # 중요: 모든 노드의 결과에 input 그룹이 있도록 보장
                        # LangGraph는 state를 병합할 때 TypedDict의 각 필드를 병합하는데,
                        # input 필드가 없으면 이전 값이 사라질 수 있음
                        # 해결책: 보존된 초기 input을 항상 복원

                        # 디버깅: node_state의 query 확인 (모든 노드에 대해)
                        # stream_mode="updates" 사용 시 변경된 필드만 포함되므로 직접 확인 가능
                        if node_name in ["classify_query", "prepare_search_query", "execute_searches_parallel"] and isinstance(node_state, dict):
                            node_query = ""
                            # input 그룹이 변경되었는지 확인
                            if "input" in node_state and isinstance(node_state["input"], dict):
                                node_query = node_state["input"].get("query", "")
                            # query가 최상위 레벨에 직접 있는 경우 (legacy 호환)
                            elif "query" in node_state:
                                node_query = node_state["query"] if isinstance(node_state["query"], str) else ""
                            # input이 변경되지 않았으면 초기 input에서 가져오기
                            if not node_query and self._initial_input:
                                node_query = self._initial_input.get("query", "")
                            self.logger.debug(f"astream: event[{node_name}] - node_state query='{node_query[:50] if node_query else 'EMPTY'}...', keys={list(node_state.keys())}")

                            # execute_searches_parallel의 경우 search 그룹 확인 및 캐시
                            # stream_mode="updates" 사용 시 search 그룹이 변경된 경우에만 포함됨
                            if node_name == "execute_searches_parallel" and isinstance(node_state, dict):
                                # search 그룹이 변경되었는지 확인
                                search_group_for_cache = {}
                                semantic_results_for_cache = []
                                keyword_results_for_cache = []
                                
                                if "search" in node_state and isinstance(node_state["search"], dict):
                                    search_group_for_cache = node_state["search"]
                                    semantic_results_for_cache = search_group_for_cache.get("semantic_results", [])
                                    keyword_results_for_cache = search_group_for_cache.get("keyword_results", [])
                                
                                # 최상위 레벨에서도 확인 (legacy 호환)
                                if not semantic_results_for_cache and "semantic_results" in node_state:
                                    semantic_results_for_cache = node_state["semantic_results"] if isinstance(node_state["semantic_results"], list) else []
                                if not keyword_results_for_cache and "keyword_results" in node_state:
                                    keyword_results_for_cache = node_state["keyword_results"] if isinstance(node_state["keyword_results"], list) else []

                                # semantic_results와 keyword_results를 retrieved_docs로 변환
                                combined_docs = []
                                if isinstance(semantic_results_for_cache, list):
                                    combined_docs.extend(semantic_results_for_cache)
                                if isinstance(keyword_results_for_cache, list):
                                    combined_docs.extend(keyword_results_for_cache)

                                # 중복 제거 (id 기반)
                                seen_ids = set()
                                unique_docs = []
                                for doc in combined_docs:
                                    doc_id = doc.get("id") or doc.get("content_id") or str(doc.get("content", ""))[:100]
                                    if doc_id not in seen_ids:
                                        seen_ids.add(doc_id)
                                        unique_docs.append(doc)

                                # retrieved_docs를 캐시에 저장
                                if unique_docs:
                                    if not self._search_results_cache:
                                        self._search_results_cache = {}
                                    self._search_results_cache["retrieved_docs"] = unique_docs
                                    self._search_results_cache["merged_documents"] = unique_docs
                                    if "search" not in self._search_results_cache:
                                        self._search_results_cache["search"] = {}
                                    self._search_results_cache["search"]["retrieved_docs"] = unique_docs
                                    self._search_results_cache["search"]["merged_documents"] = unique_docs
                                    self.logger.debug(f"astream: Converted semantic_results to retrieved_docs: {len(unique_docs)} docs")
                                
                                # search 그룹에서 카운트 확인 (변경된 경우에만 포함)
                                search_group = search_group_for_cache if search_group_for_cache else {}
                                semantic_count = len(semantic_results_for_cache) if isinstance(semantic_results_for_cache, list) else 0
                                keyword_count = len(keyword_results_for_cache) if isinstance(keyword_results_for_cache, list) else 0

                                # 최상위 레벨에서도 확인 (legacy 호환)
                                top_semantic = node_state.get("semantic_results", []) if "semantic_results" in node_state else []
                                top_keyword = node_state.get("keyword_results", []) if "keyword_results" in node_state else []
                                if isinstance(top_semantic, list):
                                    semantic_count = max(semantic_count, len(top_semantic))
                                if isinstance(top_keyword, list):
                                    keyword_count = max(keyword_count, len(top_keyword))

                                self.logger.debug(f"astream: event[{node_name}] - search group: semantic_results={semantic_count}, keyword_results={keyword_count}, top_level_semantic={len(top_semantic) if isinstance(top_semantic, list) else 0}")

                                # search 그룹 또는 최상위 레벨에 결과가 있으면 캐시에 저장
                                if (semantic_count > 0 or keyword_count > 0):
                                    # search 그룹이 있으면 그걸 우선
                                    if search_group and (len(search_group.get("semantic_results", [])) > 0 or len(search_group.get("keyword_results", [])) > 0):
                                        self._search_results_cache = search_group.copy()
                                    # 최상위 레벨 값으로 구성 (legacy 호환)
                                    elif ("semantic_results" in node_state or "keyword_results" in node_state):
                                        self._search_results_cache = {
                                            "semantic_results": node_state.get("semantic_results", []) if isinstance(node_state.get("semantic_results"), list) else [],
                                            "keyword_results": node_state.get("keyword_results", []) if isinstance(node_state.get("keyword_results"), list) else [],
                                            "semantic_count": semantic_count,
                                            "keyword_count": keyword_count
                                        }
                                    self.logger.debug(f"astream: Cached search results - semantic={semantic_count}, keyword={keyword_count}")
                                # search 그룹이 없거나 비어있으면 캐시에서 복원 시도
                                elif self._search_results_cache:
                                    self.logger.debug("astream: Restoring search results from cache")
                                    if "search" not in node_state:
                                        node_state["search"] = {}
                                    node_state["search"].update(self._search_results_cache)
                                    # 최상위 레벨에도 추가
                                    if "semantic_results" not in node_state:
                                        node_state["semantic_results"] = self._search_results_cache.get("semantic_results", [])
                                    if "keyword_results" not in node_state:
                                        node_state["keyword_results"] = self._search_results_cache.get("keyword_results", [])
                                    semantic_restored = len(node_state["search"].get("semantic_results", []))
                                    keyword_restored = len(node_state["search"].get("keyword_results", []))
                                    self.logger.debug(f"astream: Restored search results - semantic={semantic_restored}, keyword={keyword_restored}")

                        # input 그룹 복원 (stream_mode="updates" 사용 시 input이 변경되지 않은 노드에는 포함되지 않을 수 있음)
                        self._restore_input_group(node_state, node_name)
                        
                        # classification 그룹 보존 (stream_mode="updates" 사용 시 direct_answer 노드 등에서 필요)
                        # direct_answer 노드는 required_state_groups={"input", "classification"}를 필요로 함
                        if isinstance(node_state, dict):
                            # classification 그룹이 필요한 노드들
                            nodes_requiring_classification = ["direct_answer", "generate_and_validate_answer"]
                            if node_name in nodes_requiring_classification:
                                # classification 그룹이 없으면 이전 노드에서 가져오기
                                if "classification" not in node_state:
                                    # 이전 노드에서 classification 정보를 찾아야 함
                                    # classify_query_and_complexity 노드에서 저장한 정보를 보존해야 함
                                    # 이 정보는 캐시나 초기 상태에서 가져올 수 있음
                                    if self._search_results_cache and isinstance(self._search_results_cache, dict):
                                        cached_classification = (
                                            self._search_results_cache.get("classification") or
                                            self._search_results_cache.get("common", {}).get("classification") or
                                            {}
                                        )
                                        if isinstance(cached_classification, dict) and cached_classification:
                                            node_state["classification"] = cached_classification.copy()
                                            self.logger.debug(f"astream: Restored classification group from cache for {node_name}")

                        # 중요: merge_and_rerank_with_keyword_weights 이후 retrieved_docs 캐시 업데이트
                        # stream_mode="updates" 사용 시 search 그룹이 변경된 경우에만 포함됨
                        if node_name == "merge_and_rerank_with_keyword_weights" and isinstance(node_state, dict):
                            search_group = {}
                            retrieved_docs = []
                            merged_documents = []
                            
                            # search 그룹이 변경되었는지 확인
                            if "search" in node_state and isinstance(node_state["search"], dict):
                                search_group = node_state["search"]
                                retrieved_docs = search_group.get("retrieved_docs", [])
                                merged_documents = search_group.get("merged_documents", [])

                            # 최상위 레벨에서도 확인 (legacy 호환)
                            top_retrieved_docs = node_state.get("retrieved_docs", []) if "retrieved_docs" in node_state else []
                            top_merged_docs = node_state.get("merged_documents", []) if "merged_documents" in node_state else []

                            # retrieved_docs 또는 merged_documents가 있으면 캐시 업데이트
                            final_retrieved_docs = (retrieved_docs if isinstance(retrieved_docs, list) and len(retrieved_docs) > 0 else
                                                   top_retrieved_docs if isinstance(top_retrieved_docs, list) and len(top_retrieved_docs) > 0 else
                                                   merged_documents if isinstance(merged_documents, list) and len(merged_documents) > 0 else
                                                   top_merged_docs if isinstance(top_merged_docs, list) and len(top_merged_docs) > 0 else [])

                            if isinstance(final_retrieved_docs, list) and len(final_retrieved_docs) > 0:
                                # 캐시 초기화 (없으면 생성)
                                if not self._search_results_cache:
                                    self._search_results_cache = {}

                                # retrieved_docs와 merged_documents를 캐시에 저장
                                self._search_results_cache["retrieved_docs"] = final_retrieved_docs
                                self._search_results_cache["merged_documents"] = final_retrieved_docs
                                # search 그룹 전체를 캐시에 저장 (나중에 복원할 때 사용)
                                if search_group:
                                    self._search_results_cache.update(search_group)
                                else:
                                    # search 그룹이 없으면 생성하여 저장
                                    if "search" not in self._search_results_cache:
                                        self._search_results_cache["search"] = {}
                                    self._search_results_cache["search"]["retrieved_docs"] = final_retrieved_docs
                                    self._search_results_cache["search"]["merged_documents"] = final_retrieved_docs

                                self.logger.debug(f"astream: Updated cache with retrieved_docs={len(final_retrieved_docs)}, cache has search group={bool(self._search_results_cache.get('search'))}")

                        # 중요: execute_searches_parallel 이후 노드들에 대해 캐시된 search 결과 복원
                        # stream_mode="updates" 사용 시 search 그룹이 변경되지 않은 노드에는 포함되지 않을 수 있음
                        if node_name in ["merge_and_rerank_with_keyword_weights", "filter_and_validate_results", "update_search_metadata", "prepare_document_context_for_prompt"]:
                            if self._search_results_cache and isinstance(node_state, dict):
                                # node_state에 search 그룹이 변경되었는지 확인
                                search_group = {}
                                if "search" in node_state and isinstance(node_state["search"], dict):
                                    search_group = node_state["search"]
                                
                                # 최상위 레벨에서도 확인 (legacy 호환)
                                top_semantic = node_state.get("semantic_results", []) if "semantic_results" in node_state else []
                                top_keyword = node_state.get("keyword_results", []) if "keyword_results" in node_state else []
                                
                                has_results = (len(search_group.get("semantic_results", [])) > 0 or
                                             len(search_group.get("keyword_results", [])) > 0 or
                                             (isinstance(top_semantic, list) and len(top_semantic) > 0) or
                                             (isinstance(top_keyword, list) and len(top_keyword) > 0))

                                if not has_results:
                                    self.logger.debug(f"astream: Restoring search results for {node_name} from cache")
                                    if "search" not in node_state:
                                        node_state["search"] = {}
                                    node_state["search"].update(self._search_results_cache)
                                    # 최상위 레벨에도 추가 (flat 구조 호환)
                                    if "semantic_results" not in node_state:
                                        node_state["semantic_results"] = self._search_results_cache.get("semantic_results", [])
                                    if "keyword_results" not in node_state:
                                        node_state["keyword_results"] = self._search_results_cache.get("keyword_results", [])
                                    if "semantic_count" not in node_state:
                                        node_state["semantic_count"] = self._search_results_cache.get("semantic_count", 0)
                                    if "keyword_count" not in node_state:
                                        node_state["keyword_count"] = self._search_results_cache.get("keyword_count", 0)
                                    semantic_restored = len(node_state["search"].get("semantic_results", []))
                                    keyword_restored = len(node_state["search"].get("keyword_results", []))
                                    self.logger.debug(f"astream: Restored for {node_name} - semantic={semantic_restored}, keyword={keyword_restored}, top_level_semantic={len(node_state.get('semantic_results', []))}")

                        # 중요: merge_and_rerank_with_keyword_weights 또는 process_search_results_combined 이후 retrieved_docs 캐시 업데이트
                        # stream_mode="updates" 사용 시 search 그룹이 변경된 경우에만 포함됨
                        if node_name in ["merge_and_rerank_with_keyword_weights", "process_search_results_combined"] and isinstance(node_state, dict):
                            self.logger.debug("astream: Checking merge_and_rerank node_state for retrieved_docs")
                            # search 그룹이 변경되었는지 확인
                            search_group_updated = {}
                            if "search" in node_state and isinstance(node_state["search"], dict):
                                search_group_updated = node_state["search"]
                            retrieved_docs_updated = search_group_updated.get("retrieved_docs", [])
                            merged_documents_updated = search_group_updated.get("merged_documents", [])

                            # 최상위 레벨에서도 확인 (legacy 호환)
                            top_retrieved_docs_updated = node_state.get("retrieved_docs", []) if "retrieved_docs" in node_state else []
                            top_merged_docs_updated = node_state.get("merged_documents", []) if "merged_documents" in node_state else []

                            self.logger.debug(f"astream: merge_and_rerank - search_group retrieved_docs={len(retrieved_docs_updated) if isinstance(retrieved_docs_updated, list) else 0}, merged_documents={len(merged_documents_updated) if isinstance(merged_documents_updated, list) else 0}, top_retrieved_docs={len(top_retrieved_docs_updated) if isinstance(top_retrieved_docs_updated, list) else 0}, top_merged_docs={len(top_merged_docs_updated) if isinstance(top_merged_docs_updated, list) else 0}")

                            # retrieved_docs 또는 merged_documents가 있으면 캐시 업데이트
                            final_retrieved_docs = (retrieved_docs_updated if isinstance(retrieved_docs_updated, list) and len(retrieved_docs_updated) > 0 else
                                                   top_retrieved_docs_updated if isinstance(top_retrieved_docs_updated, list) and len(top_retrieved_docs_updated) > 0 else
                                                   merged_documents_updated if isinstance(merged_documents_updated, list) and len(merged_documents_updated) > 0 else
                                                   top_merged_docs_updated if isinstance(top_merged_docs_updated, list) and len(top_merged_docs_updated) > 0 else [])

                            self.logger.debug(f"astream: merge_and_rerank - final_retrieved_docs={len(final_retrieved_docs) if isinstance(final_retrieved_docs, list) else 0}")

                            if isinstance(final_retrieved_docs, list) and len(final_retrieved_docs) > 0:
                                # 캐시 초기화 (없으면 생성)
                                if not self._search_results_cache:
                                    self._search_results_cache = {}

                                # retrieved_docs와 merged_documents를 캐시에 저장
                                self._search_results_cache["retrieved_docs"] = final_retrieved_docs
                                self._search_results_cache["merged_documents"] = final_retrieved_docs
                                # search 그룹 전체를 캐시에 저장 (나중에 복원할 때 사용)
                                if search_group_updated:
                                    self._search_results_cache.update(search_group_updated)
                                else:
                                    # search 그룹이 없으면 생성하여 저장
                                    if "search" not in self._search_results_cache:
                                        self._search_results_cache["search"] = {}
                                    self._search_results_cache["search"]["retrieved_docs"] = final_retrieved_docs
                                    self._search_results_cache["search"]["merged_documents"] = final_retrieved_docs

                                self.logger.debug(f"astream: Updated cache with retrieved_docs={len(final_retrieved_docs)}, cache has search group={bool(self._search_results_cache.get('search'))}, cache keys={list(self._search_results_cache.keys())}")
                            else:
                                self.logger.warning("astream: merge_and_rerank node_state has no retrieved_docs or merged_documents")

                        flat_result = node_state

                # 모든 노드 실행 완료 표시
                total_nodes = len(executed_nodes)
                total_execution_time = time.time() - total_start_time
                
                # A/B 테스트 메트릭 추적
                if self.ab_test_manager and cache_variant and parallel_variant:
                    self.ab_test_manager.track_metric(
                        session_id, "cache_enabled", cache_variant,
                        "execution_time", total_execution_time
                    )
                    self.ab_test_manager.track_metric(
                        session_id, "parallel_processing", parallel_variant,
                        "execution_time", total_execution_time
                    )
                    
                    # 캐시 히트율 추적 (캐시 관리자가 있는 경우)
                    if hasattr(self.legal_workflow, 'cache_manager') and self.legal_workflow.cache_manager:
                        cache_hit_rate = self.legal_workflow.cache_manager.get_hit_rate()
                        self.ab_test_manager.track_metric(
                            session_id, "cache_enabled", cache_variant,
                            "cache_hit_rate", cache_hit_rate
                        )
                
                if total_nodes > 0:
                    self.logger.info(f"✅ 워크플로우 실행 완료 (총 {total_nodes}개 노드 실행, 총 실행 시간: {total_execution_time:.2f}초)")
                    print(f"✅ 워크플로우 실행 완료 (총 {total_nodes}개 노드 실행, 총 실행 시간: {total_execution_time:.2f}초)", flush=True)
                    
                    # 성능 요약 출력 (느린 노드 순서대로 정렬)
                    if node_durations:
                        sorted_nodes = sorted(node_durations.items(), key=lambda x: x[1], reverse=True)
                        self.logger.info("📊 [PERFORMANCE] 노드 실행 시간 요약:")
                        print("📊 [PERFORMANCE] 노드 실행 시간 요약:", flush=True)
                        
                        for node_name, duration in sorted_nodes[:5]:  # 상위 5개만 표시
                            percentage = (duration / total_execution_time * 100) if total_execution_time > 0 else 0
                            node_display_name = self._get_node_display_name(node_name)
                            summary_msg = f"  - {node_display_name}: {duration:.2f}초 ({percentage:.1f}%)"
                            self.logger.info(summary_msg)
                            print(summary_msg, flush=True)
                        
                        # 가장 느린 노드 경고
                        if sorted_nodes and sorted_nodes[0][1] > self.SLOW_NODE_THRESHOLD:
                            slowest_node = sorted_nodes[0]
                            self.logger.warning(
                                f"⚠️ [PERFORMANCE] 가장 느린 노드: {slowest_node[0]} "
                                f"({slowest_node[1]:.2f}초, 전체의 {slowest_node[1]/total_execution_time*100:.1f}%)"
                            )
                    
                    # 실행된 노드 목록 표시
                    if total_nodes > 0:
                        nodes_list = ", ".join(executed_nodes[:5])
                        if total_nodes > 5:
                            nodes_list += f" 외 {total_nodes - 5}개"
                        self.logger.info(f"  실행된 노드: {nodes_list}")
                        print(f"  실행된 노드: {nodes_list}", flush=True)

                # 최종 결과가 없으면 초기 상태 사용
                if flat_result is None:
                    flat_result = initial_state

                # processing_steps를 flat_result에 명시적으로 저장 (state reduction 손실 방지, 개선)
                if isinstance(flat_result, dict):
                    if tracked_processing_steps and len(tracked_processing_steps) > 0:
                        # common 그룹에 저장
                        if "common" not in flat_result:
                            flat_result["common"] = {}
                        if not isinstance(flat_result["common"], dict):
                            flat_result["common"] = {}
                        flat_result["common"]["processing_steps"] = tracked_processing_steps
                        # 최상위 레벨에도 저장 (fallback)
                        flat_result["processing_steps"] = tracked_processing_steps
                    else:
                        # tracked_processing_steps가 비어있으면 기본값 추가 (개선)
                        default_steps = ["워크플로우 시작", "워크플로우 실행"]
                        if "common" not in flat_result:
                            flat_result["common"] = {}
                        if not isinstance(flat_result["common"], dict):
                            flat_result["common"] = {}
                        flat_result["common"]["processing_steps"] = default_steps
                        flat_result["processing_steps"] = default_steps

                # 중요: 최종 결과에 search 그룹 보존
                # LangGraph reducer가 search 그룹을 제거했을 수 있으므로 캐시에서 복원
                # 전역 캐시와 인스턴스 캐시 모두 확인
                if isinstance(flat_result, dict):
                    # 전역 캐시에서도 확인 (node_wrappers에서 저장한 것)
                    try:
                        import sys
                        node_wrappers_module = sys.modules.get('core.agents.node_wrappers')
                        if node_wrappers_module:
                            global_cache = getattr(node_wrappers_module, '_global_search_results_cache', None)
                        else:
                            global_cache = None
                    except (ImportError, AttributeError) as e:
                        global_cache = None
                        self.logger.debug(f"Failed to import global cache: {e}")

                    self.logger.debug(f"Final result check - has instance cache={self._search_results_cache is not None}, has global cache={global_cache is not None}")
                    if global_cache:
                        self.logger.debug(f"Global cache keys={list(global_cache.keys()) if isinstance(global_cache, dict) else 'N/A'}")
                        if isinstance(global_cache, dict):
                            if "search" in global_cache:
                                search_group_cache = global_cache["search"]
                                if isinstance(search_group_cache, dict):
                                    self.logger.debug(f"Global cache search group has retrieved_docs={len(search_group_cache.get('retrieved_docs', []))}, merged_documents={len(search_group_cache.get('merged_documents', []))}")
                            # 최상위 레벨에서도 확인
                            if "retrieved_docs" in global_cache:
                                self.logger.debug(f"Global cache top-level has retrieved_docs={len(global_cache.get('retrieved_docs', []))}")

                    # 전역 캐시 또는 인스턴스 캐시 사용 (전역 캐시 우선)
                    search_cache = global_cache if global_cache else self._search_results_cache

                    if search_cache:
                        # search 그룹이 없거나 비어있으면 캐시에서 복원
                        if "search" not in flat_result or not isinstance(flat_result.get("search"), dict):
                            self.logger.debug("Final result has no search group, creating from cache")
                            flat_result["search"] = {}

                        search_group = flat_result["search"]
                        has_results = (len(search_group.get("retrieved_docs", [])) > 0 or
                                     len(search_group.get("merged_documents", [])) > 0 or
                                     len(flat_result.get("retrieved_docs", [])) > 0)

                        if not has_results:
                            self.logger.debug("Restoring search group in final result from cache")
                            # 캐시에 search 그룹이 있으면 그걸 사용
                            if "search" in search_cache and isinstance(search_cache["search"], dict):
                                flat_result["search"].update(search_cache["search"])
                            else:
                                # 캐시의 최상위 retrieved_docs/merged_documents 사용
                                flat_result["search"].update({
                                    "retrieved_docs": search_cache.get("retrieved_docs", []),
                                    "merged_documents": search_cache.get("merged_documents", [])
                                })

                            # retrieved_docs가 없으면 merged_documents 사용
                            if "retrieved_docs" not in flat_result["search"] or len(flat_result["search"].get("retrieved_docs", [])) == 0:
                                if "merged_documents" in flat_result["search"] and len(flat_result["search"].get("merged_documents", [])) > 0:
                                    flat_result["search"]["retrieved_docs"] = flat_result["search"]["merged_documents"]

                            # 최상위 레벨에도 추가
                            if "retrieved_docs" not in flat_result:
                                flat_result["retrieved_docs"] = flat_result["search"].get("retrieved_docs", [])
                            if not flat_result["retrieved_docs"]:
                                flat_result["retrieved_docs"] = flat_result["search"].get("merged_documents", [])

                            restored_count = len(flat_result["search"].get("retrieved_docs", []))
                            merged_count = len(flat_result["search"].get("merged_documents", []))
                            self.logger.debug(f"Restored search group - retrieved_docs={restored_count}, merged_documents={merged_count}, top_level={len(flat_result.get('retrieved_docs', []))}")
                        else:
                            self.logger.debug(f"Final result already has search results - retrieved_docs={len(search_group.get('retrieved_docs', []))}, top_level={len(flat_result.get('retrieved_docs', []))}")
                    else:
                        self.logger.warning("No search cache available for final result restoration")

                # 중요: query_complexity와 needs_search 복원 (Adaptive RAG 정보)
                # prepare_final_response에서 보존했지만, reducer에 의해 사라질 수 있으므로 재확인
                if isinstance(flat_result, dict):
                    self.logger.debug(f"flat_result keys={list(flat_result.keys())[:20]}")

                    # 모든 가능한 경로에서 확인
                    query_complexity_found = None
                    needs_search_found = True

                    # 1. 최상위 레벨 직접 확인
                    query_complexity_found = flat_result.get("query_complexity")
                    if "needs_search" in flat_result:
                        needs_search_found = flat_result.get("needs_search", True)
                    self.logger.debug(f"[1] 최상위 레벨 - complexity={query_complexity_found}, needs_search={needs_search_found}")

                    # 2. common 그룹 확인
                    if not query_complexity_found:
                        has_common = "common" in flat_result
                        self.logger.debug(f"checking common - exists={has_common}")
                        if has_common:
                            common_value = flat_result["common"]
                            self.logger.debug(f"common type={type(common_value).__name__}")
                            if isinstance(common_value, dict):
                                query_complexity_found = common_value.get("query_complexity")
                                if "needs_search" in common_value:
                                    needs_search_found = common_value.get("needs_search", True)
                                self.logger.debug(f"[2] common 그룹 - complexity={query_complexity_found}, needs_search={needs_search_found}")
                            else:
                                self.logger.debug(f"common is not dict: {type(common_value)}")

                    # 3. metadata 확인 (여러 형태 지원)
                    if not query_complexity_found and "metadata" in flat_result:
                        metadata_value = flat_result["metadata"]
                        self.logger.debug(f"checking metadata - type={type(metadata_value).__name__}")
                        if isinstance(metadata_value, dict):
                            query_complexity_found = metadata_value.get("query_complexity")
                            if "needs_search" in metadata_value:
                                needs_search_found = metadata_value.get("needs_search", True)
                            self.logger.debug(f"[3] metadata (dict) - complexity={query_complexity_found}, needs_search={needs_search_found}")
                        # metadata가 다른 형태일 수도 있으므로 확인

                    # 4. classification 그룹 확인
                    if not query_complexity_found and "classification" in flat_result:
                        classification_value = flat_result["classification"]
                        if isinstance(classification_value, dict):
                            query_complexity_found = classification_value.get("query_complexity")
                            if "needs_search" in classification_value:
                                needs_search_found = classification_value.get("needs_search", True)
                            self.logger.debug(f"[4] classification 그룹 - complexity={query_complexity_found}, needs_search={needs_search_found}")

                    # 5. Global cache에서 확인 (classify_complexity에서 저장한 값)
                    if not query_complexity_found:
                        try:
                            from core.agents import node_wrappers
                            global_cache = getattr(node_wrappers, '_global_search_results_cache', None)
                            self.logger.debug(f"[5] Global cache 확인 - exists={global_cache is not None}, type={type(global_cache).__name__ if global_cache else 'None'}")
                            if global_cache and isinstance(global_cache, dict):
                                query_complexity_found = global_cache.get("query_complexity")
                                if "needs_search" in global_cache:
                                    needs_search_found = global_cache.get("needs_search", True)
                                self.logger.debug(f"[5] Global cache 내용 - complexity={query_complexity_found}, needs_search={needs_search_found}")
                                self.logger.debug(f"[5] Global cache 전체 keys={list(global_cache.keys())[:10]}")
                                if query_complexity_found:
                                    self.logger.debug(f"[5] ✅ Global cache에서 찾음 - complexity={query_complexity_found}, needs_search={needs_search_found}")
                            elif global_cache is None:
                                self.logger.debug("[5] Global cache is None")
                            else:
                                self.logger.debug(f"[5] Global cache is not dict: {type(global_cache)}")
                        except Exception as e:
                            self.logger.debug(f"[5] Global cache 확인 실패: {e}")
                            import traceback
                            self.logger.debug(f"[5] Exception details: {traceback.format_exc()}")

                    # 6. 전체 state 재귀 검색 (마지막 시도)
                    if not query_complexity_found:
                        self.logger.debug("[6] 재귀 검색 시작...")
                        def find_in_dict(d, depth=0):
                            if depth > 3:  # 최대 깊이 제한
                                return None, None
                            for k, v in d.items() if isinstance(d, dict) else []:
                                if k == "query_complexity":
                                    return v, d.get("needs_search", True)
                                elif isinstance(v, dict):
                                    found_c, found_n = find_in_dict(v, depth+1)
                                    if found_c:
                                        return found_c, found_n
                            return None, None

                        found_c, found_n = find_in_dict(flat_result)
                        if found_c:
                            query_complexity_found = found_c
                            needs_search_found = found_n if found_n is not None else True
                            self.logger.debug(f"[6] 재귀 검색으로 찾음 - complexity={query_complexity_found}, needs_search={needs_search_found}")

                    # 찾은 값을 최상위 레벨에 명시적으로 저장
                    if query_complexity_found:
                        flat_result["query_complexity"] = query_complexity_found
                        flat_result["needs_search"] = needs_search_found if needs_search_found is not None else True
                        # common과 metadata에도 저장 (다음 노드에서 사용)
                        if "common" not in flat_result:
                            flat_result["common"] = {}
                        flat_result["common"]["query_complexity"] = query_complexity_found
                        flat_result["common"]["needs_search"] = needs_search_found
                        if "metadata" not in flat_result:
                            flat_result["metadata"] = {}
                        if isinstance(flat_result["metadata"], dict):
                            flat_result["metadata"]["query_complexity"] = query_complexity_found
                            flat_result["metadata"]["needs_search"] = needs_search_found
                        self.logger.debug(f"✅ query_complexity 복원 완료 - {query_complexity_found}, needs_search={flat_result.get('needs_search')}")
                    else:
                        self.logger.debug("❌ query_complexity를 찾지 못함 (모든 경로 확인 완료)")

                # 최종 결과에 query가 없으면 보존된 초기 input에서 복원
                if isinstance(flat_result, dict) and self._initial_input:
                    # 중요: flat_result.get("input")이 None일 수 있으므로 안전하게 처리
                    flat_input = flat_result.get("input")
                    final_query = ""
                    if flat_input and isinstance(flat_input, dict):
                        final_query = flat_input.get("query", "")
                    elif not final_query:
                        final_query = flat_result.get("query", "")

                    if not final_query or not str(final_query).strip():
                        if self._initial_input.get("query"):
                            if "input" not in flat_result or not isinstance(flat_result.get("input"), dict):
                                flat_result["input"] = {}
                            flat_result["input"]["query"] = self._initial_input["query"]
                            if self._initial_input.get("session_id"):
                                flat_result["input"]["session_id"] = self._initial_input["session_id"]
                            self.logger.warning("Restored query from preserved initial_input in final result")

            else:
                raise RuntimeError("워크플로우가 컴파일되지 않았습니다")

            # 처리 시간 계산
            processing_time = time.time() - start_time

            # 결과 포맷팅
            # 중요: flat_result가 None이거나 dict가 아닐 수 있으므로 안전하게 처리
            if not isinstance(flat_result, dict):
                self.logger.error(f"flat_result is not a dict: {type(flat_result).__name__}, using empty dict")
                flat_result = {}

            # nested 구조에서 flat 구조로 변환 (필요한 경우)
            # 예: flat_result["input"]["query"] -> flat_result["query"]
            if isinstance(flat_result, dict) and "input" in flat_result and isinstance(flat_result["input"], dict):
                if "query" not in flat_result and flat_result["input"].get("query"):
                    flat_result["query"] = flat_result["input"]["query"]

            # query_complexity와 needs_search 추출 (Adaptive RAG 정보)
            query_complexity = None
            needs_search = True
            if isinstance(flat_result, dict):
                # 우선순위: 최상위 레벨 > common 그룹 > metadata > classification 그룹

                # 1. 최상위 레벨 직접 확인
                query_complexity = flat_result.get("query_complexity")
                if "needs_search" in flat_result:
                    needs_search = flat_result.get("needs_search", True)

                # 2. common 그룹 확인 (reducer가 보존하는 그룹)
                if not query_complexity and "common" in flat_result:
                    if isinstance(flat_result["common"], dict):
                        query_complexity = flat_result["common"].get("query_complexity")
                        if "needs_search" in flat_result["common"]:
                            needs_search = flat_result["common"].get("needs_search", True)

                # 3. metadata에서 확인
                if not query_complexity:
                    metadata = flat_result.get("metadata", {})
                    if isinstance(metadata, dict):
                        query_complexity = metadata.get("query_complexity")
                        if "needs_search" in metadata:
                            needs_search = metadata.get("needs_search", True)

                # 4. classification 그룹에서 확인
                if not query_complexity and "classification" in flat_result:
                    if isinstance(flat_result["classification"], dict):
                        query_complexity = flat_result["classification"].get("query_complexity")
                        if "needs_search" in flat_result["classification"]:
                            needs_search = flat_result["classification"].get("needs_search", True)

            # processing_steps 추출 (최상위 레벨, common 그룹, metadata, 또는 노드 실행 상태에서)
            processing_steps = []
            if isinstance(flat_result, dict):
                # 1. 최상위 레벨에서 확인
                processing_steps = flat_result.get("processing_steps", [])

                # 2. common 그룹에서 확인
                if (not processing_steps or (isinstance(processing_steps, list) and len(processing_steps) == 0)) and "common" in flat_result:
                    if isinstance(flat_result["common"], dict):
                        common_steps = flat_result["common"].get("processing_steps", [])
                        if isinstance(common_steps, list) and len(common_steps) > 0:
                            processing_steps = common_steps

                # 3. metadata에서 확인
                if (not processing_steps or (isinstance(processing_steps, list) and len(processing_steps) == 0)) and "metadata" in flat_result:
                    if isinstance(flat_result["metadata"], dict):
                        metadata_steps = flat_result["metadata"].get("processing_steps", [])
                        if isinstance(metadata_steps, list) and len(metadata_steps) > 0:
                            processing_steps = metadata_steps

                # 4. 전역 캐시에서 processing_steps 확인 (node_wrappers에서 저장한 것)
                if (not processing_steps or (isinstance(processing_steps, list) and len(processing_steps) == 0)):
                    try:
                        from core.shared.wrappers.node_wrappers import (
                            _global_search_results_cache,
                        )
                        if _global_search_results_cache and "processing_steps" in _global_search_results_cache:
                            cached_steps = _global_search_results_cache["processing_steps"]
                            if isinstance(cached_steps, list) and len(cached_steps) > 0:
                                processing_steps = cached_steps
                                self.logger.debug(f"Restored {len(processing_steps)} processing_steps from global cache")
                    except (ImportError, AttributeError, TypeError):
                        pass

                # 5. 추적된 processing_steps 사용 (최후의 수단)
                if (not processing_steps or (isinstance(processing_steps, list) and len(processing_steps) == 0)):
                    if tracked_processing_steps:
                        processing_steps = tracked_processing_steps
                        self.logger.debug(f"Using {len(processing_steps)} tracked processing_steps")

                if not isinstance(processing_steps, list):
                    processing_steps = []

            # retrieved_docs 추출
            retrieved_docs = self._extract_retrieved_docs_from_result(flat_result)
            
            # sources 추출: flat_result에서 직접 가져오거나 retrieved_docs에서 변환
            sources = flat_result.get("sources", []) if isinstance(flat_result, dict) else []
            
            # sources가 비어있고 retrieved_docs가 있으면 sources 생성
            if (not sources or len(sources) == 0) and retrieved_docs and len(retrieved_docs) > 0:
                sources = self._extract_sources_from_retrieved_docs(retrieved_docs)
                self.logger.debug(f"Extracted {len(sources)} sources from {len(retrieved_docs)} retrieved_docs")
            
            # metadata 추출 및 suggested_questions 변환
            metadata = flat_result.get("metadata", {}) if isinstance(flat_result, dict) else {}
            if not isinstance(metadata, dict):
                metadata = {}
            
            # phase_info에서 suggested_questions 추출하여 metadata.related_questions로 변환 (우선순위 1)
            if isinstance(flat_result, dict) and "phase_info" in flat_result:
                phase_info = flat_result.get("phase_info", {})
                if isinstance(phase_info, dict) and "phase2" in phase_info:
                    phase2_info = phase_info.get("phase2", {})
                    if isinstance(phase2_info, dict) and "flow_tracking_info" in phase2_info:
                        flow_tracking = phase2_info.get("flow_tracking_info", {})
                        if isinstance(flow_tracking, dict) and "suggested_questions" in flow_tracking:
                            suggested_questions = flow_tracking.get("suggested_questions", [])
                            if isinstance(suggested_questions, list) and len(suggested_questions) > 0:
                                # 각 항목이 딕셔너리인 경우 "question" 필드 추출
                                if isinstance(suggested_questions[0], dict):
                                    related_questions = [q.get("question", "") for q in suggested_questions if q.get("question")]
                                else:
                                    related_questions = [str(q) for q in suggested_questions if q]
                                
                                # metadata에 related_questions 추가
                                metadata["related_questions"] = related_questions
                                # flat_result에도 metadata 업데이트하여 다음 단계에서 사용 가능하도록
                                if isinstance(flat_result, dict):
                                    if "metadata" not in flat_result:
                                        flat_result["metadata"] = {}
                                    flat_result["metadata"]["related_questions"] = related_questions
                                self.logger.debug(f"Extracted {len(related_questions)} related questions from phase_info")
            
            # phase_info에 suggested_questions가 없으면 conversation_flow_tracker로 생성 (우선순위 2)
            if "related_questions" not in metadata or not metadata.get("related_questions"):
                self.logger.info(f"[workflow_service] Step 1: Checking related_questions in metadata: {len(metadata.get('related_questions', []))} questions")
                
                if self.conversation_flow_tracker and ConversationContext is not None and ConversationTurn is not None:
                    self.logger.info("[workflow_service] Step 2: ConversationFlowTracker is available, attempting to generate related_questions")
                    try:
                        # 현재 질문과 답변으로 ConversationContext 구성
                        # query 추출: flat_result > self._initial_input > 메서드 파라미터 순서
                        current_query = ""
                        if isinstance(flat_result, dict):
                            flat_input = flat_result.get("input", {})
                            if isinstance(flat_input, dict) and flat_input.get("query"):
                                current_query = flat_input.get("query", "")
                            elif flat_result.get("query"):
                                current_query = flat_result.get("query", "")
                        
                        if not current_query and self._initial_input and self._initial_input.get("query"):
                            current_query = self._initial_input.get("query", "")
                        
                        if not current_query:
                            current_query = query  # 메서드 파라미터 사용
                        
                        answer = flat_result.get("answer", "") if isinstance(flat_result, dict) else ""
                        query_type = flat_result.get("query_type", "general_question") if isinstance(flat_result, dict) else "general_question"
                        
                        self.logger.info(f"[workflow_service] Step 3: query={bool(current_query)}, query_length={len(current_query) if current_query else 0}, answer={bool(answer)}, answer_length={len(answer) if answer else 0}, query_type={query_type}")
                        
                        if not current_query or not answer:
                            self.logger.warning(f"[workflow_service] Step 3: Skipping suggested questions generation: query={bool(current_query)}, answer={bool(answer)}")
                        else:
                            # ConversationTurn 생성
                            turn = ConversationTurn(
                                user_query=current_query,
                                bot_response=answer,
                                timestamp=datetime.now(),
                                question_type=query_type
                            )
                            
                            # ConversationContext 생성 (올바른 필드 사용)
                            context = ConversationContext(
                                session_id=session_id or "default",
                                turns=[turn],
                                entities={},
                                topic_stack=[],
                                created_at=datetime.now(),
                                last_updated=datetime.now()
                            )
                            
                            # 추천 질문 생성
                            self.logger.info(f"[workflow_service] Step 4: Generating suggested questions for query_type={query_type}, session_id={session_id or 'default'}")
                            suggested_questions = self.conversation_flow_tracker.suggest_follow_up_questions(context)
                            
                            self.logger.info(f"[workflow_service] Step 5: suggest_follow_up_questions returned {len(suggested_questions) if suggested_questions else 0} questions")
                            
                            if suggested_questions and len(suggested_questions) > 0:
                                # 문자열 리스트로 변환 및 검증
                                related_questions = [str(q).strip() for q in suggested_questions if q and str(q).strip()]
                                if related_questions:
                                    metadata["related_questions"] = related_questions
                                    # flat_result에도 metadata 업데이트하여 다음 단계에서 사용 가능하도록
                                    if isinstance(flat_result, dict):
                                        if "metadata" not in flat_result:
                                            flat_result["metadata"] = {}
                                        flat_result["metadata"]["related_questions"] = related_questions
                                    self.logger.info(
                                        f"[workflow_service] Step 6: Successfully generated {len(related_questions)} related_questions "
                                        f"for query_type={query_type}, session_id={session_id or 'default'}"
                                    )
                                    self.logger.info(f"[workflow_service] Generated questions: {related_questions[:3]}")
                                else:
                                    self.logger.warning("[workflow_service] Step 6: All suggested questions were empty after filtering")
                            else:
                                self.logger.warning(f"[workflow_service] Step 6: No suggested questions generated for query_type={query_type}")
                    except Exception as e:
                        self.logger.error(
                            f"[workflow_service] Step 7: Failed to generate suggested questions using ConversationFlowTracker: {e}",
                            exc_info=True
                        )
                        # 에러가 발생해도 계속 진행 (기본 질문 추가하지 않음)
                else:
                    if not self.conversation_flow_tracker:
                        self.logger.warning("[workflow_service] ConversationFlowTracker not available, skipping suggested questions generation")
                    elif ConversationContext is None or ConversationTurn is None:
                        self.logger.warning("[workflow_service] ConversationContext or ConversationTurn not available, skipping suggested questions generation")
                    else:
                        self.logger.warning("[workflow_service] Unknown reason for skipping suggested questions generation")
            
            # 최종 확인
            final_related_questions = metadata.get("related_questions", [])
            self.logger.info(f"[workflow_service] Final check: related_questions count={len(final_related_questions)}")
            
            response = {
                "answer": flat_result.get("answer", "") if isinstance(flat_result, dict) else "",
                "sources": sources,
                "confidence": flat_result.get("confidence", 0.0) if isinstance(flat_result, dict) else 0.0,
                "legal_references": flat_result.get("legal_references", []) if isinstance(flat_result, dict) else [],
                "processing_steps": processing_steps,
                "session_id": session_id,
                "processing_time": processing_time,
                "query_type": flat_result.get("query_type", "") if isinstance(flat_result, dict) else "",
                "metadata": metadata,
                "errors": flat_result.get("errors", []) if isinstance(flat_result, dict) else [],
                # 새로 추가된 필드들
                "legal_field": flat_result.get("legal_field", "unknown") if isinstance(flat_result, dict) else "unknown",
                "legal_domain": flat_result.get("legal_domain", "unknown") if isinstance(flat_result, dict) else "unknown",
                "urgency_level": flat_result.get("urgency_level", "unknown") if isinstance(flat_result, dict) else "unknown",
                "urgency_reasoning": flat_result.get("urgency_reasoning", "") if isinstance(flat_result, dict) else "",
                "emergency_type": flat_result.get("emergency_type", None) if isinstance(flat_result, dict) else None,
                "complexity_level": flat_result.get("complexity_level", "unknown") if isinstance(flat_result, dict) else "unknown",
                "requires_expert": flat_result.get("requires_expert", False) if isinstance(flat_result, dict) else False,
                "expert_subgraph": flat_result.get("expert_subgraph", None) if isinstance(flat_result, dict) else None,
                "legal_validity_check": flat_result.get("legal_validity_check", True) if isinstance(flat_result, dict) else True,
                "document_type": flat_result.get("document_type", None) if isinstance(flat_result, dict) else None,
                "document_analysis": flat_result.get("document_analysis", None) if isinstance(flat_result, dict) else None,
                # ✨ Adaptive RAG 필드 추가
                "query_complexity": query_complexity if query_complexity else "unknown",
                "needs_search": needs_search,
                # retrieved_docs는 search 그룹 또는 최상위 레벨에 있을 수 있음
                "retrieved_docs": retrieved_docs
            }

            # Langfuse에 답변 품질 추적
            if self.langfuse_client_service and self.langfuse_client_service.is_enabled():
                self._track_answer_quality(query, response, processing_time)

            self.logger.info(f"Query processed successfully in {processing_time:.2f}s")
            return response

        except ValueError as e:
            import traceback
            error_msg = f"입력값 오류: {str(e)}"
            self.logger.warning(f"Value error in process_query: {error_msg}")
            self.logger.debug(traceback.format_exc())
            return {
                "answer": "입력값에 문제가 있습니다. 질문을 다시 확인해주세요.",
                "sources": [],
                "confidence": 0.0,
                "legal_references": [],
                "processing_steps": ["입력값 검증 실패"],
                "session_id": session_id or str(uuid.uuid4()),
                "processing_time": time.time() - start_time if 'start_time' in locals() else 0.0,
                "query_type": "error",
                "metadata": {"error": str(e), "error_type": "ValueError"},
                "errors": [error_msg]
            }
        except RuntimeError as e:
            import traceback
            error_msg = f"시스템 오류: {str(e)}"
            self.logger.error(f"Runtime error in process_query: {error_msg}")
            self.logger.error(traceback.format_exc())
            return {
                "answer": "시스템 오류가 발생했습니다. 잠시 후 다시 시도해주세요.",
                "sources": [],
                "confidence": 0.0,
                "legal_references": [],
                "processing_steps": ["시스템 오류 발생"],
                "session_id": session_id or str(uuid.uuid4()),
                "processing_time": time.time() - start_time if 'start_time' in locals() else 0.0,
                "query_type": "error",
                "metadata": {"error": str(e), "error_type": "RuntimeError"},
                "errors": [error_msg]
            }
        except Exception as e:
            import traceback
            error_msg = f"예상치 못한 오류: {str(e)}"
            self.logger.error(f"Unexpected error in process_query: {error_msg}")
            self.logger.error(traceback.format_exc())

            return {
                "answer": "죄송합니다. 질문 처리 중 오류가 발생했습니다. 잠시 후 다시 시도해주세요.",
                "sources": [],
                "confidence": 0.0,
                "legal_references": [],
                "processing_steps": ["처리 오류 발생"],
                "session_id": session_id or str(uuid.uuid4()),
                "processing_time": time.time() - start_time if 'start_time' in locals() else 0.0,
                "query_type": "error",
                "metadata": {"error": str(e), "error_type": type(e).__name__},
                "errors": [error_msg]
            }

    def _extract_retrieved_docs_from_result(self, flat_result: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        최종 결과에서 retrieved_docs 추출

        retrieved_docs는 search 그룹, 최상위 레벨, 또는 global cache에 있을 수 있음

        Args:
            flat_result: 최종 결과 딕셔너리

        Returns:
            retrieved_docs 리스트
        """
        if not isinstance(flat_result, dict):
            return []

        # 1. 최상위 레벨에서 확인
        if "retrieved_docs" in flat_result:
            docs = flat_result.get("retrieved_docs", [])
            if isinstance(docs, list) and len(docs) > 0:
                self.logger.debug(f"Found retrieved_docs in top level: {len(docs)}")
                return docs

        # 2. search 그룹에서 확인
        if "search" in flat_result and isinstance(flat_result["search"], dict):
            search_group = flat_result["search"]
            docs = search_group.get("retrieved_docs", [])
            if isinstance(docs, list) and len(docs) > 0:
                self.logger.debug(f"Found retrieved_docs in search group: {len(docs)}")
                return docs

        # 3. search.retrieved_docs가 없으면 search.merged_documents 확인
        if "search" in flat_result and isinstance(flat_result["search"], dict):
            search_group = flat_result["search"]
            merged_docs = search_group.get("merged_documents", [])
            if isinstance(merged_docs, list) and len(merged_docs) > 0:
                self.logger.debug(f"Found merged_documents in search group (using as retrieved_docs): {len(merged_docs)}")
                return merged_docs

        # 4. global cache에서 확인 (마지막 시도)
        try:
            from core.shared.wrappers.node_wrappers import _global_search_results_cache
            if _global_search_results_cache:
                # search 그룹에서 확인
                if "search" in _global_search_results_cache and isinstance(_global_search_results_cache["search"], dict):
                    cached_search = _global_search_results_cache["search"]
                    cached_docs = cached_search.get("retrieved_docs", [])
                    if isinstance(cached_docs, list) and len(cached_docs) > 0:
                        self.logger.debug(f"Found retrieved_docs in global cache search group: {len(cached_docs)}")
                        return cached_docs
                    cached_merged = cached_search.get("merged_documents", [])
                    if isinstance(cached_merged, list) and len(cached_merged) > 0:
                        self.logger.debug(f"Found merged_documents in global cache search group: {len(cached_merged)}")
                        return cached_merged

                    # semantic_results를 retrieved_docs로 변환 (retrieved_docs가 없는 경우)
                    cached_semantic = cached_search.get("semantic_results", [])
                    if isinstance(cached_semantic, list) and len(cached_semantic) > 0:
                        self.logger.debug(f"Converting semantic_results to retrieved_docs: {len(cached_semantic)}")
                        # semantic_results는 이미 문서 형태이므로 그대로 사용
                        return cached_semantic

                # 최상위 레벨에서 확인
                cached_docs = _global_search_results_cache.get("retrieved_docs", [])
                if isinstance(cached_docs, list) and len(cached_docs) > 0:
                    self.logger.debug(f"Found retrieved_docs in global cache top level: {len(cached_docs)}")
                    return cached_docs
                cached_merged = _global_search_results_cache.get("merged_documents", [])
                if isinstance(cached_merged, list) and len(cached_merged) > 0:
                    self.logger.debug(f"Found merged_documents in global cache top level: {len(cached_merged)}")
                    return cached_merged

                # semantic_results를 retrieved_docs로 변환 (최상위 레벨)
                cached_semantic = _global_search_results_cache.get("semantic_results", [])
                if isinstance(cached_semantic, list) and len(cached_semantic) > 0:
                    self.logger.debug(f"Converting semantic_results to retrieved_docs (top level): {len(cached_semantic)}")
                    return cached_semantic
        except (ImportError, AttributeError, TypeError):
            pass  # global cache를 사용할 수 없으면 무시

        # 5. flat_result에서 semantic_results를 retrieved_docs로 변환 (최후의 수단)
        if "search" in flat_result and isinstance(flat_result["search"], dict):
            search_group = flat_result["search"]
            semantic_results = search_group.get("semantic_results", [])
            if isinstance(semantic_results, list) and len(semantic_results) > 0:
                self.logger.debug(f"Converting semantic_results to retrieved_docs from search group: {len(semantic_results)}")
                return semantic_results

        # 최상위 레벨의 semantic_results 확인
        if "semantic_results" in flat_result:
            semantic_results = flat_result.get("semantic_results", [])
            if isinstance(semantic_results, list) and len(semantic_results) > 0:
                self.logger.debug(f"Converting semantic_results to retrieved_docs from top level: {len(semantic_results)}")
                return semantic_results

        self.logger.debug(f"No retrieved_docs found - keys={list(flat_result.keys())[:10]}")
        return []
    
    def _extract_sources_from_retrieved_docs(self, retrieved_docs: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        retrieved_docs에서 소스 정보 추출
        
        Args:
            retrieved_docs: 검색된 문서 리스트
            
        Returns:
            소스 정보 리스트
        """
        if not retrieved_docs or not isinstance(retrieved_docs, list):
            return []
        
        sources = []
        seen_sources = set()  # 중복 제거용
        
        # source_type 매핑 (데이터베이스 타입 → 일반 타입)
        source_type_mapping = {
            "statute_article": "law",
            "case_paragraph": "precedent",
            "decision_paragraph": "decision",
            "interpretation_paragraph": "interpretation"
        }
        
        for doc in retrieved_docs:
            if not isinstance(doc, dict):
                continue
            
            # 1. type 필드 확인 (우선순위 1)
            doc_type = doc.get("type", "")
            
            # 2. metadata.source_type 확인 (우선순위 2)
            if not doc_type and "metadata" in doc:
                metadata = doc.get("metadata", {})
                if isinstance(metadata, dict):
                    doc_type = metadata.get("source_type", "")
            
            # 3. source_type 필드 확인 (우선순위 3)
            if not doc_type:
                doc_type = doc.get("source_type", "")
            
            # 4. source_type 매핑 적용
            if doc_type in source_type_mapping:
                doc_type = source_type_mapping[doc_type]
            elif not doc_type:
                # 5. source_name에서 추론 (최후의 수단)
                source_name = (
                    doc.get("source") or
                    doc.get("title") or
                    doc.get("document_id") or
                    doc.get("law_name") or
                    doc.get("case_name") or
                    doc.get("precedent_name") or
                    ""
                )
                
                if "법령" in str(source_name) or "법" in str(source_name) or "조문" in str(source_name):
                    doc_type = "law"
                elif "판례" in str(source_name) or "판결" in str(source_name) or "사건" in str(source_name):
                    doc_type = "precedent"
                elif "해석" in str(source_name) or "의견" in str(source_name):
                    doc_type = "interpretation"
                elif "결정" in str(source_name) or "결정례" in str(source_name):
                    doc_type = "decision"
                else:
                    doc_type = "document"
            
            # 소스 정보 추출
            source_name = (
                doc.get("source") or
                doc.get("title") or
                doc.get("document_id") or
                doc.get("law_name") or
                doc.get("case_name") or
                doc.get("precedent_name") or
                doc.get("casenames") or  # 판례명
                ""
            )
            
            # relevance_score 추출
            relevance_score = (
                doc.get("relevance_score") or
                doc.get("score") or
                doc.get("final_weighted_score") or
                doc.get("similarity") or
                0.0
            )
            
            # 소스 정보 구성 (개선: 모든 문서를 소스로 포함)
            if source_name or doc_type != "document":  # 타입이 있으면 source_name이 없어도 포함
                source_key = f"{doc_type}:{source_name or 'unknown'}"
                if source_key not in seen_sources:
                    seen_sources.add(source_key)
                    
                    source_info = {
                        "type": doc_type,
                        "source": str(source_name) if source_name else f"{doc_type}_document",
                        "relevance_score": float(relevance_score) if relevance_score else 0.0
                    }
                    
                    # metadata 정보도 추가
                    if "metadata" in doc and isinstance(doc.get("metadata"), dict):
                        metadata = doc.get("metadata", {})
                        if "date" in metadata:
                            source_info["date"] = metadata.get("date")
                        if "url" in metadata:
                            source_info["url"] = metadata.get("url")
                    
                    # 추가 메타데이터
                    if "metadata" in doc and isinstance(doc["metadata"], dict):
                        source_info["metadata"] = doc["metadata"]
                    
                    # 법령 조문 정보
                    if doc_type == "law":
                        if "article" in doc or "article_no" in doc:
                            source_info["article"] = doc.get("article") or doc.get("article_no")
                        if "clause" in doc or "clause_no" in doc:
                            source_info["clause"] = doc.get("clause") or doc.get("clause_no")
                    
                    # 판례 정보
                    if doc_type == "precedent":
                        if "case_number" in doc or "doc_id" in doc:
                            source_info["case_number"] = doc.get("case_number") or doc.get("doc_id")
                        if "court" in doc:
                            source_info["court"] = doc.get("court")
                        if "casenames" in doc:
                            source_info["case_name"] = doc.get("casenames")
                    
                    # 결정례 정보
                    if doc_type == "decision":
                        if "doc_id" in doc:
                            source_info["doc_id"] = doc.get("doc_id")
                        if "org" in doc:
                            source_info["org"] = doc.get("org")
                    
                    # 해석례 정보
                    if doc_type == "interpretation":
                        if "doc_id" in doc:
                            source_info["doc_id"] = doc.get("doc_id")
                        if "org" in doc:
                            source_info["org"] = doc.get("org")
                        if "title" in doc:
                            source_info["title"] = doc.get("title")
                    
                    sources.append(source_info)
        
        # relevance_score 기준으로 정렬 (높은 순서대로)
        sources.sort(key=lambda x: x.get("relevance_score", 0.0), reverse=True)
        
        return sources

    async def resume_session(self, session_id: str, query: str) -> Dict[str, Any]:
        """
        세션 재개

        Args:
            session_id: 세션 ID
            query: 새로운 질문

        Returns:
            Dict[str, Any]: 처리 결과
        """
        try:
            self.logger.info(f"Resuming session: {session_id}")

            # 체크포인트에서 이전 상태 확인 (checkpoint_manager가 있는 경우)
            if self.checkpoint_manager:
                checkpoints = self.checkpoint_manager.list_checkpoints(session_id)

                if checkpoints:
                    self.logger.info(f"Found {len(checkpoints)} checkpoints for session {session_id}")
                    # 이전 상태에서 새로운 질문으로 계속
                    return await self.process_query(query, session_id, enable_checkpoint=True)
                else:
                    self.logger.info(f"No checkpoints found for session {session_id}, starting new session")
                    # 새로운 세션으로 시작
                    return await self.process_query(query, session_id, enable_checkpoint=True)
            else:
                # checkpoint_manager가 없으면 일반 프로세스
                return await self.process_query(query, session_id, enable_checkpoint=False)

        except Exception as e:
            error_msg = f"세션 재개 중 오류 발생: {str(e)}"
            self.logger.error(error_msg)

            # 새로운 세션으로 폴백
            return await self.process_query(query, session_id, enable_checkpoint=False)

    async def continue_answer(
        self,
        session_id: str,
        message_id: str,
        chunk_index: int
    ) -> Dict[str, Any]:
        """
        이전 답변의 마지막 부분부터 이어서 답변 생성
        
        Args:
            session_id: 세션 ID
            message_id: 메시지 ID
            chunk_index: 현재 청크 인덱스 (사용하지 않음, 하위 호환성)
        
        Returns:
            Dict[str, Any]: 이어서 생성된 답변
        """
        try:
            self.logger.info(f"Continuing answer for session: {session_id}, message: {message_id}")
            
            # 체크포인트에서 이전 상태 복원
            if not self.app:
                raise RuntimeError("워크플로우가 컴파일되지 않았습니다.")
            
            config = {"configurable": {"thread_id": session_id}}
            
            # 이전 상태 가져오기
            try:
                # LangGraph의 get_state를 사용하여 이전 상태 가져오기
                from langgraph.checkpoint.memory import MemorySaver
                
                # 체크포인터가 있으면 이전 상태 가져오기
                if self.checkpoint_manager and self.checkpoint_manager.is_enabled():
                    # 이전 상태에서 continue_answer_generation 노드만 실행
                    # 먼저 현재 상태 확인
                    current_state = await self.app.aget_state(config)
                    
                    if not current_state or not current_state.values:
                        raise ValueError(f"세션 {session_id}의 상태를 찾을 수 없습니다.")
                    
                    # continue_answer_generation 노드만 실행
                    result = await self.app.ainvoke(
                        {"continue": True},
                        config={"recursion_limit": 50, **config}
                    )
                    
                    # 결과에서 답변 추출
                    continued_answer = result.get("answer", "")
                    previous_answer = current_state.values.get("answer", "")
                    
                    if not continued_answer or continued_answer == previous_answer:
                        # 답변이 업데이트되지 않았으면 직접 노드 실행
                        from langgraph.graph import END
                        
                        # continue_answer_generation 노드 직접 호출
                        state_dict = dict(current_state.values)
                        state_dict["continue"] = True
                        
                        # 워크플로우의 continue_answer_generation 노드 실행
                        if hasattr(self.legal_workflow, "continue_answer_generation"):
                            updated_state = self.legal_workflow.continue_answer_generation(state_dict)
                            continued_answer = updated_state.get("answer", "")
                            
                            # 이전 답변과 비교하여 새로 추가된 부분만 추출
                            if continued_answer and previous_answer:
                                if continued_answer.startswith(previous_answer):
                                    new_content = continued_answer[len(previous_answer):].strip()
                                else:
                                    new_content = continued_answer
                            else:
                                new_content = continued_answer
                        else:
                            raise RuntimeError("continue_answer_generation 노드를 찾을 수 없습니다.")
                    else:
                        # 답변이 업데이트되었으면 새로 추가된 부분 추출
                        if continued_answer.startswith(previous_answer):
                            new_content = continued_answer[len(previous_answer):].strip()
                        else:
                            new_content = continued_answer
                    
                    # 메시지에서 전체 답변 가져오기
                    try:
                        from api.services.session_service import session_service
                        messages = session_service.get_messages(session_id)
                        full_answer = ""
                        for msg in messages:
                            if msg.get("message_id") == message_id and msg.get("role") == "assistant":
                                full_answer = msg.get("content", "")
                                break
                        
                        # 전체 답변에 새로 생성된 부분 추가
                        if full_answer and new_content:
                            full_answer = full_answer + "\n\n" + new_content
                        elif new_content:
                            full_answer = new_content
                        
                        # 메시지 업데이트 (save_full_answer 사용)
                        if full_answer:
                            # save_full_answer 메서드 사용
                            if hasattr(session_service, "save_full_answer"):
                                # 기존 메시지의 metadata 가져오기
                                existing_metadata = {}
                                for msg in messages:
                                    if msg.get("message_id") == message_id:
                                        existing_metadata = msg.get("metadata", {})
                                        break
                                
                                session_service.save_full_answer(
                                    session_id=session_id,
                                    message_id=message_id,
                                    full_answer=full_answer,
                                    metadata=existing_metadata
                                )
                            else:
                                self.logger.warning(f"save_full_answer 메서드가 없어 메시지를 업데이트할 수 없습니다. message_id: {message_id}")
                    except Exception as e:
                        self.logger.warning(f"메시지 업데이트 중 오류 (무시): {e}")
                    
                    return {
                        "content": new_content,
                        "chunk_index": chunk_index,
                        "total_chunks": 1,
                        "has_more": False
                    }
                else:
                    raise RuntimeError("체크포인트 관리자가 활성화되지 않았습니다.")
            
            except Exception as e:
                self.logger.error(f"Error continuing answer: {e}", exc_info=True)
                raise
        
        except Exception as e:
            error_msg = f"이어서 답변 생성 중 오류 발생: {str(e)}"
            self.logger.error(error_msg, exc_info=True)
            raise RuntimeError(error_msg) from e

    def get_session_info(self, session_id: str) -> Dict[str, Any]:
        """
        세션 정보 조회

        Args:
            session_id: 세션 ID

        Returns:
            Dict[str, Any]: 세션 정보
        """
        try:
            if self.checkpoint_manager:
                checkpoints = self.checkpoint_manager.list_checkpoints(session_id)

                return {
                    "session_id": session_id,
                    "checkpoint_count": len(checkpoints),
                    "latest_checkpoint": checkpoints[-1] if checkpoints else None,
                    "has_checkpoints": len(checkpoints) > 0
                }
            else:
                # checkpoint_manager가 없으면 기본 정보만 반환
                return {
                    "session_id": session_id,
                    "checkpoint_count": 0,
                    "has_checkpoints": False,
                    "note": "Checkpoint manager is disabled"
                }

        except Exception as e:
            self.logger.error(f"Failed to get session info: {e}")
            return {
                "session_id": session_id,
                "error": str(e),
                "checkpoint_count": 0,
                "has_checkpoints": False
            }

    def cleanup_old_sessions(self, ttl_hours: int = 24) -> int:
        """
        오래된 세션 정리

        Args:
            ttl_hours: 유지 시간 (시간)

        Returns:
            int: 정리된 체크포인트 수
        """
        try:
            if self.checkpoint_manager:
                cleaned_count = self.checkpoint_manager.cleanup_old_checkpoints(ttl_hours)
                self.logger.info(f"Cleaned up {cleaned_count} old checkpoints")
                return cleaned_count
            else:
                self.logger.info("Checkpoint manager is disabled")
                return 0

        except Exception as e:
            self.logger.error(f"Failed to cleanup old sessions: {e}")
            return 0

    def get_service_status(self) -> Dict[str, Any]:
        """
        서비스 상태 조회

        Returns:
            Dict[str, Any]: 서비스 상태 정보
        """
        try:
            if self.checkpoint_manager:
                db_info = self.checkpoint_manager.get_database_info()
            else:
                db_info = {"note": "Checkpoint manager is disabled"}

            return {
                "service_name": "LangGraphWorkflowService",
                "status": "running",
                "config": self.config.to_dict(),
                "database_info": db_info,
                "workflow_compiled": self.app is not None,
                "timestamp": datetime.now().isoformat()
            }

        except Exception as e:
            self.logger.error(f"Failed to get service status: {e}")
            return {
                "service_name": "LangGraphWorkflowService",
                "status": "error",
                "error": str(e),
                "timestamp": datetime.now().isoformat()
            }

    def validate_config(self) -> List[str]:
        """
        설정 유효성 검사

        Returns:
            List[str]: 오류 메시지 목록
        """
        return self.config.validate()

    async def test_workflow(self, test_query: str = "계약서 작성 시 주의사항은?") -> Dict[str, Any]:
        """
        워크플로우 테스트

        Args:
            test_query: 테스트 질문

        Returns:
            Dict[str, Any]: 테스트 결과
        """
        try:
            self.logger.info(f"Testing workflow with query: {test_query}")

            result = await self.process_query(test_query, enable_checkpoint=False)

            # 테스트 결과 검증
            test_passed = (
                "answer" in result and
                result["answer"] and
                len(result["processing_steps"]) > 0 and
                len(result["errors"]) == 0
            )

            return {
                "test_passed": test_passed,
                "test_query": test_query,
                "result": result,
                "timestamp": datetime.now().isoformat()
            }

        except Exception as e:
            self.logger.error(f"Workflow test failed: {e}")
            return {
                "test_passed": False,
                "test_query": test_query,
                "error": str(e),
                "timestamp": datetime.now().isoformat()
            }

    def _track_answer_quality(self, query: str, response: Dict[str, Any], processing_time: float):
        """
        Langfuse에 답변 품질 추적

        Args:
            query: 사용자 질문
            response: 응답 결과
            processing_time: 처리 시간
        """
        if not self.langfuse_client_service or not self.langfuse_client_service.is_enabled():
            return

        try:
            # 답변 품질 점수 계산 (종합 점수)
            quality_score = self._calculate_quality_score(response)

            # 답변 품질 메트릭 추적
            trace_id = self.langfuse_client_service.track_answer_quality_metrics(
                query=query,
                answer=response.get("answer", ""),
                confidence=response.get("confidence", 0.0),
                sources_count=len(response.get("sources", [])),
                legal_refs_count=len(response.get("legal_references", [])),
                processing_time=processing_time,
                has_errors=len(response.get("errors", [])) > 0,
                overall_quality=quality_score
            )

            self.logger.info(f"Answer quality tracked in Langfuse: quality_score={quality_score:.2f}, trace_id={trace_id}")

        except Exception as e:
            self.logger.error(f"Failed to track answer quality in Langfuse: {e}")

    def _calculate_quality_score(self, response: Dict[str, Any]) -> float:
        """
        답변 품질 점수 계산

        Args:
            response: 응답 결과

        Returns:
            float: 품질 점수 (0.0 ~ 1.0)
        """
        score = 0.0
        max_score = 0.0

        # 답변 길이 (20점)
        answer = response.get("answer", "")
        if len(answer) >= 50:
            score += 10
        if len(answer) >= 100:
            score += 10
        max_score += 20

        # 신뢰도 (30점)
        confidence = response.get("confidence", 0.0)
        score += confidence * 30
        max_score += 30

        # 소스 제공 (20점)
        sources_count = len(response.get("sources", []))
        if sources_count > 0:
            score += min(20, sources_count * 5)
        max_score += 20

        # 법률 참조 (20점)
        legal_refs_count = len(response.get("legal_references", []))
        if legal_refs_count > 0:
            score += min(20, legal_refs_count * 10)
        max_score += 20

        # 에러 없음 (10점)
        errors_count = len(response.get("errors", []))
        if errors_count == 0:
            score += 10
        max_score += 10

        # 정규화된 점수
        quality_score = score / max_score if max_score > 0 else 0.0
        return round(quality_score, 2)

    def _prepare_and_validate_initial_state(self, query: str, session_id: str) -> Dict[str, Any]:
        """초기 상태 준비 및 검증"""
        initial_state = create_initial_legal_state(query, session_id)
        
        if "input" not in initial_state:
            initial_state["input"] = {}
        if not initial_state["input"].get("query"):
            initial_state["input"]["query"] = query
        if not initial_state["input"].get("session_id"):
            initial_state["input"]["session_id"] = session_id
        
        if not initial_state.get("query"):
            initial_state["query"] = query
        if not initial_state.get("session_id"):
            initial_state["session_id"] = session_id
        
        initial_query = initial_state.get("input", {}).get("query", "") if initial_state.get("input") else initial_state.get("query", "")
        self.logger.debug(f"process_query: initial_state query length={len(initial_query)}, query='{initial_query[:50] if initial_query else 'EMPTY'}...'")
        
        if not initial_query or not str(initial_query).strip():
            self.logger.error(f"Initial state query is empty! Input query was: '{query[:50]}...'")
            self.logger.debug("process_query: ERROR - initial_state query is empty!")
            self.logger.debug(f"process_query: initial_state keys: {list(initial_state.keys())}")
            self.logger.debug(f"process_query: initial_state['input']: {initial_state.get('input')}")
        else:
            self.logger.debug(f"process_query: SUCCESS - initial_state has query with length={len(initial_query)}")
        
        return initial_state
    
    def _validate_initial_state_before_execution(self, initial_state: Dict[str, Any], query: str) -> bool:
        """워크플로우 실행 전 초기 상태 최종 검증"""
        initial_query_check = initial_state.get("input", {}).get("query", "") if initial_state.get("input") else initial_state.get("query", "")
        self.logger.debug(f"astream: initial_state before astream - query='{initial_query_check[:50] if initial_query_check else 'EMPTY'}...', keys={list(initial_state.keys())}")
        
        if not initial_query_check or not str(initial_query_check).strip():
            self.logger.error(f"Initial state query is empty before astream! Initial state keys: {list(initial_state.keys())}")
            if initial_state.get("input"):
                self.logger.error(f"Initial state input: {initial_state['input']}")
            
            if "input" not in initial_state:
                initial_state["input"] = {}
            if not initial_state["input"].get("query"):
                if initial_state.get("query"):
                    initial_state["input"]["query"] = initial_state["query"]
                else:
                    self.logger.error("CRITICAL: Cannot find query anywhere in initial_state!")
                    return False
        
        return True
    
    def _restore_input_group(self, node_state: Dict[str, Any], node_name: str) -> None:
        """input 그룹 복원 (State Reduction으로 인한 데이터 손실 방지)"""
        if not isinstance(node_state, dict) or not self._initial_input:
            return
        
        # stream_mode="updates" 사용 시 input 그룹이 변경된 경우에만 포함됨
        # input이 변경되지 않은 노드에서는 초기 input을 복원해야 함
        if "input" not in node_state:
            # input 그룹이 없으면 초기 input에서 복원
            node_state["input"] = self._initial_input.copy()
            if node_name == "classify_query":
                self.logger.debug(f"astream: Restored query from preserved initial_input for {node_name}: '{self._initial_input['query'][:50]}...'")
        elif isinstance(node_state.get("input"), dict):
            # input 그룹이 있지만 query가 없으면 복원
            node_input = node_state["input"]
            if not node_input.get("query") and self._initial_input.get("query"):
                node_state["input"]["query"] = self._initial_input["query"]
                if not node_input.get("session_id") and self._initial_input.get("session_id"):
                    node_state["input"]["session_id"] = self._initial_input["session_id"]
                self.logger.debug(f"astream: Restored query in input group for {node_name}")
    
    def _get_node_display_name(self, node_name: str) -> str:
        """
        노드 이름을 한국어로 변환하여 표시

        Args:
            node_name: 노드 이름

        Returns:
            str: 한국어 노드 이름
        """
        node_name_map = {
            # 워크플로우 주요 노드
            "classify_query": "질문 분류",
            "assess_urgency": "긴급도 평가",
            "analyze_document": "문서 분석",
            "resolve_multi_turn": "멀티턴 처리",
            "route_expert": "전문가 라우팅",
            "expand_keywords_ai": "AI 키워드 확장",
            "retrieve_documents": "문서 검색",
            "process_legal_terms": "법률 용어 처리",
            "generate_answer_enhanced": "답변 생성",
            "validate_answer_quality": "답변 품질 검증",
            "enhance_answer_structure": "답변 구조 향상",
            "apply_visual_formatting": "시각적 포맷팅",
            "prepare_final_response": "최종 응답 준비",
            # 기타 노드
            "validate_legal_basis": "법령 검증",
            "route_to_expert": "전문가 라우팅",
            "retrieve_context": "컨텍스트 검색",
            "generate_answer": "답변 생성",
            "enhance_answer": "답변 향상",
            "format_answer": "답변 포맷팅",
            "family_law_expert": "가족법 전문가",
            "corporate_law_expert": "기업법 전문가",
            "ip_law_expert": "지적재산권 전문가",
            "legal_term_extraction": "법률 용어 추출",
            "legal_domain_classification": "법률 분야 분류",
        }

        return node_name_map.get(node_name, node_name.replace("_", " ").title())
