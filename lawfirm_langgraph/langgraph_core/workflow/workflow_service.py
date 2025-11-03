# -*- coding: utf-8 -*-
"""
LangGraph Workflow Service
워크플로우 서비스 통합 클래스
"""

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
    try:
        from lawfirm_langgraph.langgraph_core.services.legal_workflow_enhanced import EnhancedLegalQuestionWorkflow
    except ImportError:
        # Fallback: 기존 경로 (호환성 유지)
        from core.agents.legal_workflow_enhanced import EnhancedLegalQuestionWorkflow

# 설정 파일 import (lawfirm_langgraph 구조 우선 시도)
try:
    from lawfirm_langgraph.config.langgraph_config import LangGraphConfig
except ImportError:
    # Fallback: 기존 경로 (호환성 유지)
    try:
        from core.utils.langgraph_config import LangGraphConfig
    except ImportError:
        from core.utils.langgraph_config import LangGraphConfig

# Langfuse 클라이언트 통합
try:
    from langfuse import Langfuse, trace

    from core.services.langfuse_client import LangfuseClient
    LANGFUSE_CLIENT_AVAILABLE = True
    LANGFUSE_TRACE_AVAILABLE = True
except ImportError:
    LANGFUSE_CLIENT_AVAILABLE = False
    LANGFUSE_TRACE_AVAILABLE = False

logger = logging.getLogger(__name__)

if not LANGFUSE_CLIENT_AVAILABLE:
    logger.warning("LangfuseClient not available for LangGraph workflow tracking")

# CheckpointManager import
try:
    from core.agents.checkpoint_manager import CheckpointManager
except ImportError:
    CheckpointManager = None
    logger.warning("CheckpointManager not available")

# state_definitions import (상대 import 사용 - 같은 패키지 내부)
try:
    from ..state.state_definitions import create_initial_legal_state
except ImportError:
    # Fallback: 프로젝트 루트 기준 import
    try:
        from lawfirm_langgraph.langgraph_core.utils.state_definitions import create_initial_legal_state
    except ImportError:
        # Fallback: 기존 경로 (호환성 유지)
        from core.agents.state_definitions import create_initial_legal_state


class LangGraphWorkflowService:
    """LangGraph 워크플로우 서비스"""

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
                    self.logger.info(f"Checkpoint manager initialized with {storage_type} storage")
                else:
                    self.logger.warning("Checkpoint manager initialization failed, continuing without checkpoint")
                    self.checkpoint_manager = None
            except Exception as e:
                self.logger.warning(f"Failed to initialize checkpoint manager: {e}, continuing without checkpoint")
                self.checkpoint_manager = None
        else:
            if not self.config.enable_checkpoint:
                self.logger.info("Checkpoint is disabled in configuration")
            else:
                self.logger.warning("CheckpointManager class not available")
        
        self.legal_workflow = EnhancedLegalQuestionWorkflow(self.config)

        # LangSmith 활성화 여부 확인 (환경 변수로 제어 가능)
        import os
        enable_langsmith = os.environ.get("ENABLE_LANGSMITH", "false").lower() == "true"

        if not enable_langsmith:
            # LangSmith 비활성화 모드 (기본값) - State Reduction으로 최적화된 후에도 기본은 비활성화
            # LangSmith 트레이싱 비활성화 (긴급) - 대용량 상태 로깅 방지
            original_tracing = os.environ.get("LANGCHAIN_TRACING_V2")
            original_api_key = os.environ.get("LANGCHAIN_API_KEY")

            # 임시로 LangSmith 비활성화
            os.environ["LANGCHAIN_TRACING_V2"] = "false"
            if "LANGCHAIN_API_KEY" in os.environ:
                del os.environ["LANGCHAIN_API_KEY"]

            try:
                # 체크포인터 설정 (활성화된 경우)
                checkpointer = None
                if self.checkpoint_manager and self.checkpoint_manager.is_enabled():
                    checkpointer = self.checkpoint_manager.get_checkpointer()
                    self.logger.info(f"Using checkpoint: {self.config.checkpoint_storage.value}")
                else:
                    self.logger.info("Compiling workflow without checkpoint")
                
                # 워크플로우 컴파일
                self.app = self.legal_workflow.graph.compile(
                    checkpointer=checkpointer,
                    interrupt_before=None,
                    interrupt_after=None,
                    debug=False,
                )
                if checkpointer:
                    self.logger.info(f"워크플로우가 체크포인트({self.config.checkpoint_storage.value})와 함께 컴파일되었습니다 (LangSmith 비활성화됨)")
                else:
                    self.logger.info("워크플로우가 체크포인트 없이 컴파일되었습니다 (LangSmith 비활성화됨)")
            finally:
                # 환경 변수 복원
                if original_tracing:
                    os.environ["LANGCHAIN_TRACING_V2"] = original_tracing
                elif "LANGCHAIN_TRACING_V2" in os.environ:
                    del os.environ["LANGCHAIN_TRACING_V2"]

                if original_api_key:
                    os.environ["LANGCHAIN_API_KEY"] = original_api_key
        else:
            # LangSmith 활성화 모드 (ENABLE_LANGSMITH=true로 설정된 경우)
            # 체크포인터 설정 (활성화된 경우)
            checkpointer = None
            if self.checkpoint_manager and self.checkpoint_manager.is_enabled():
                checkpointer = self.checkpoint_manager.get_checkpointer()
                self.logger.info(f"Using checkpoint: {self.config.checkpoint_storage.value}")
            
            self.app = self.legal_workflow.graph.compile(
                checkpointer=checkpointer,
                interrupt_before=None,
                interrupt_after=None,
                debug=False,
            )
            checkpoint_info = f" with checkpoint({self.config.checkpoint_storage.value})" if checkpointer else " without checkpoint"
            self.logger.info(f"워크플로우가 LangSmith 추적으로 컴파일되었습니다{checkpoint_info} (State Reduction 적용됨)")

        if self.app is None:
            self.logger.error("Failed to compile workflow")
            raise RuntimeError("워크플로우 컴파일에 실패했습니다")

        # LangfuseClient 초기화 (답변 품질 추적)
        self.langfuse_client_service = None
        if LANGFUSE_CLIENT_AVAILABLE and self.config.langfuse_enabled:
            try:
                from core.services.langfuse_client import LangfuseClient
                self.langfuse_client_service = LangfuseClient(self.config)
                if self.langfuse_client_service and self.langfuse_client_service.is_enabled():
                    self.logger.info("LangfuseClient initialized for answer quality tracking")
            except Exception as e:
                self.logger.warning(f"Failed to initialize LangfuseClient: {e}")
                self.langfuse_client_service = None

        self.logger.info("LangGraphWorkflowService initialized successfully")

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

            self.logger.info(f"Processing query: {query[:100]}... (session: {session_id})")
            self.logger.debug(f"process_query: query length={len(query)}, query='{query[:50]}...'")

            # 초기 상태 설정 (flat 구조 사용)
            initial_state = create_initial_legal_state(query, session_id)

            # 중요: initial_state에 query가 반드시 포함되도록 강제
            # LangGraph에 전달하기 전에 input 그룹에 query가 있어야 함
            if "input" not in initial_state:
                initial_state["input"] = {}
            if not initial_state["input"].get("query"):
                initial_state["input"]["query"] = query
            if not initial_state["input"].get("session_id"):
                initial_state["input"]["session_id"] = session_id

            # 최상위 레벨에도 query 포함 (이중 보장)
            if not initial_state.get("query"):
                initial_state["query"] = query
            if not initial_state.get("session_id"):
                initial_state["session_id"] = session_id

            # 초기 state 검증
            initial_query = initial_state.get("input", {}).get("query", "") if initial_state.get("input") else initial_state.get("query", "")
            self.logger.debug(f"process_query: initial_state query length={len(initial_query)}, query='{initial_query[:50] if initial_query else 'EMPTY'}...'")
            if not initial_query or not str(initial_query).strip():
                self.logger.error(f"Initial state query is empty! Input query was: '{query[:50]}...'")
                self.logger.debug(f"process_query: ERROR - initial_state query is empty!")
                self.logger.debug(f"process_query: initial_state keys: {list(initial_state.keys())}")
                self.logger.debug(f"process_query: initial_state['input']: {initial_state.get('input')}")
            else:
                self.logger.debug(f"process_query: SUCCESS - initial_state has query with length={len(initial_query)}")

            # 워크플로우 실행 설정 (체크포인터 활성화 시 thread_id 설정)
            config = {}
            if enable_checkpoint and self.checkpoint_manager and self.checkpoint_manager.is_enabled():
                config = {"configurable": {"thread_id": session_id}}
                self.logger.debug(f"Using checkpoint with thread_id: {session_id}")

            # 워크플로우 실행 (스트리밍으로 진행상황 표시)
            if self.app:
                # Recursion limit 증가 (재시도 로직 개선으로 인해 더 높게 설정)
                # 재시도 최대 3회 + 각 단계별 노드 실행을 고려하여 여유있게 설정
                enhanced_config = {"recursion_limit": 200}
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

                # 초기 state 검증: input 그룹과 query 확인
                initial_query_check = initial_state.get("input", {}).get("query", "") if initial_state.get("input") else initial_state.get("query", "")
                self.logger.debug(f"astream: initial_state before astream - query='{initial_query_check[:50] if initial_query_check else 'EMPTY'}...', keys={list(initial_state.keys())}")

                # 중요: initial_state에 input이 없거나 query가 비어있으면 복원
                # LangGraph에 전달하기 전에 반드시 query가 있어야 함
                if not initial_query_check or not str(initial_query_check).strip():
                    self.logger.error(f"Initial state query is empty before astream! Initial state keys: {list(initial_state.keys())}")
                    if initial_state.get("input"):
                        self.logger.error(f"Initial state input: {initial_state['input']}")

                    # query 파라미터에서 직접 복원 (process_query의 query 파라미터)
                    # 하지만 여기서는 이미 initial_state를 받았으므로, query를 다시 찾아야 함
                    # 대신 initial_state를 수정하여 query 포함 보장
                    if "input" not in initial_state:
                        initial_state["input"] = {}
                    # query 파라미터는 함수 인자에 있으므로 직접 접근 가능
                    # 하지만 이미 initial_state를 생성했으므로, 원본 query를 사용
                    # create_initial_legal_state에서 이미 설정했을 것이므로 문제 없어야 함
                    # 혹시 모르니 다시 확인
                    if not initial_state["input"].get("query"):
                        # 최상위 레벨 확인
                        if initial_state.get("query"):
                            initial_state["input"]["query"] = initial_state["query"]
                        else:
                            self.logger.error(f"CRITICAL: Cannot find query anywhere in initial_state!")

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
                    self.logger.error(f"CRITICAL: _initial_input has no query! This should never happen.")
                else:
                    self.logger.debug(f"Preserved initial input: query length={len(self._initial_input.get('query', ''))}")

                async for event in self.app.astream(initial_state, enhanced_config):
                    # 각 이벤트는 {node_name: updated_state} 형태
                    for node_name, node_state in event.items():
                        # 새로 실행된 노드인 경우에만 카운트
                        if node_name not in executed_nodes:
                            node_count += 1
                            executed_nodes.append(node_name)

                            # 노드 실행 시간 계산
                            node_duration = time.time() - last_node_time if node_count > 1 else 0
                            last_node_time = time.time()

                            # 진행상황 표시
                            if node_count == 1:
                                progress_msg = f"  [{node_count}] 🔄 실행 중: {node_name}"
                            else:
                                progress_msg = f"  [{node_count}] 🔄 실행 중: {node_name} (이전 노드 완료: {node_duration:.2f}초)"

                            self.logger.info(progress_msg)
                            print(progress_msg, flush=True)

                            # 노드 이름을 한국어로 변환하여 더 명확하게 표시
                            node_display_name = self._get_node_display_name(node_name)
                            if node_display_name != node_name:
                                detail_msg = f"      → {node_display_name}"
                                self.logger.info(detail_msg)
                                print(detail_msg, flush=True)

                            # 디버깅: node_state의 query 확인
                            if node_name == "classify_query":
                                # 중요: node_state.get("input")이 None일 수 있으므로 안전하게 처리
                                node_input = node_state.get("input") if isinstance(node_state, dict) else None
                                node_query = ""
                                if node_input and isinstance(node_input, dict):
                                    node_query = node_input.get("query", "")
                                elif isinstance(node_state, dict):
                                    node_query = node_state.get("query", "")
                                self.logger.debug(f"astream: event[{node_name}] query='{node_query[:50] if node_query else 'EMPTY'}...', keys={list(node_state.keys()) if isinstance(node_state, dict) else 'N/A'}")

                        # processing_steps 추적 (state reduction으로 손실 방지, 개선)
                        if isinstance(node_state, dict):
                            # 1. common 그룹에서 processing_steps 확인
                            node_common = node_state.get("common", {})
                            if isinstance(node_common, dict):
                                common_steps = node_common.get("processing_steps", [])
                                if isinstance(common_steps, list) and len(common_steps) > 0:
                                    for step in common_steps:
                                        if isinstance(step, str) and step not in tracked_processing_steps:
                                            tracked_processing_steps.append(step)

                            # 2. 최상위 레벨에서도 확인
                            top_steps = node_state.get("processing_steps", [])
                            if isinstance(top_steps, list) and len(top_steps) > 0:
                                for step in top_steps:
                                    if isinstance(step, str) and step not in tracked_processing_steps:
                                        tracked_processing_steps.append(step)

                            # 3. metadata에서도 확인 (개선)
                            metadata = node_state.get("metadata", {})
                            if isinstance(metadata, dict):
                                metadata_steps = metadata.get("processing_steps", [])
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
                        if node_name in ["classify_query", "prepare_search_query", "execute_searches_parallel"]:
                            # 중요: node_state.get("input")이 None일 수 있으므로 안전하게 처리
                            node_input = node_state.get("input") if isinstance(node_state, dict) else None
                            node_query = ""
                            if node_input and isinstance(node_input, dict):
                                node_query = node_input.get("query", "")
                            elif isinstance(node_state, dict):
                                node_query = node_state.get("query", "")
                            self.logger.debug(f"astream: event[{node_name}] - node_state query='{node_query[:50] if node_query else 'EMPTY'}...'")
                            self.logger.debug(f"astream: event[{node_name}] - node_state keys={list(node_state.keys()) if isinstance(node_state, dict) else 'N/A'}")

                            # execute_searches_parallel의 경우 search 그룹 확인 및 캐시
                            # semantic_results를 retrieved_docs로 변환하여 저장
                            if node_name == "execute_searches_parallel" and isinstance(node_state, dict):
                                # semantic_results를 retrieved_docs로 변환하여 저장
                                search_group_for_cache = node_state.get("search", {}) if isinstance(node_state.get("search"), dict) else {}
                                semantic_results_for_cache = search_group_for_cache.get("semantic_results", [])
                                keyword_results_for_cache = search_group_for_cache.get("keyword_results", [])

                                # 최상위 레벨에서도 확인
                                if not semantic_results_for_cache:
                                    semantic_results_for_cache = node_state.get("semantic_results", [])
                                if not keyword_results_for_cache:
                                    keyword_results_for_cache = node_state.get("keyword_results", [])

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
                                search_group = node_state.get("search", {}) if isinstance(node_state.get("search"), dict) else {}
                                semantic_count = len(search_group.get("semantic_results", []))
                                keyword_count = len(search_group.get("keyword_results", []))

                                # 최상위 레벨에서도 확인 (node_wrappers에서 추가했을 수 있음)
                                top_semantic = node_state.get("semantic_results", [])
                                top_keyword = node_state.get("keyword_results", [])
                                if isinstance(top_semantic, list):
                                    semantic_count = max(semantic_count, len(top_semantic))
                                if isinstance(top_keyword, list):
                                    keyword_count = max(keyword_count, len(top_keyword))

                                self.logger.debug(f"astream: event[{node_name}] - search group: semantic_results={semantic_count}, keyword_results={keyword_count}, top_level_semantic={len(top_semantic) if isinstance(top_semantic, list) else 0}")

                                # search 그룹 또는 최상위 레벨에 결과가 있으면 캐시에 저장
                                if (semantic_count > 0 or keyword_count > 0):
                                    # search 그룹이 있으면 그걸 우선, 없으면 최상위 레벨 값으로 구성
                                    if search_group and (len(search_group.get("semantic_results", [])) > 0 or len(search_group.get("keyword_results", [])) > 0):
                                        self._search_results_cache = search_group.copy()
                                    elif isinstance(top_semantic, list) or isinstance(top_keyword, list):
                                        # 최상위 레벨에서 캐시 구성
                                        self._search_results_cache = {
                                            "semantic_results": top_semantic if isinstance(top_semantic, list) else [],
                                            "keyword_results": top_keyword if isinstance(top_keyword, list) else [],
                                            "semantic_count": len(top_semantic) if isinstance(top_semantic, list) else 0,
                                            "keyword_count": len(top_keyword) if isinstance(top_keyword, list) else 0
                                        }
                                    self.logger.debug(f"astream: Cached search results - semantic={semantic_count}, keyword={keyword_count}")
                                # search 그룹이 없거나 비어있으면 캐시에서 복원 시도
                                elif self._search_results_cache:
                                    self.logger.debug(f"astream: Restoring search results from cache")
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

                        if isinstance(node_state, dict) and self._initial_input:
                            # 중요: node_state.get("input")이 None일 수 있으므로 안전하게 처리
                            node_input = node_state.get("input")
                            node_has_input = node_input is not None and isinstance(node_input, dict)
                            node_has_query = node_has_input and bool(node_input.get("query"))

                            # node_state에 input이 없거나 query가 없으면 보존된 초기 input에서 복원
                            if not node_has_input or not node_has_query:
                                if self._initial_input.get("query"):
                                    if "input" not in node_state or not isinstance(node_state.get("input"), dict):
                                        node_state["input"] = {}
                                    node_state["input"]["query"] = self._initial_input["query"]
                                    if self._initial_input.get("session_id"):
                                        node_state["input"]["session_id"] = self._initial_input["session_id"]
                                    if node_name == "classify_query":
                                        self.logger.debug(f"astream: Restored query from preserved initial_input for {node_name}: '{self._initial_input['query'][:50]}...'")

                            # 모든 노드 결과에 항상 input 그룹 포함 (LangGraph 병합 보장)
                            # 초기 input이 있으면 항상 포함
                            node_input_check = node_state.get("input")
                            if node_input_check is None or not isinstance(node_input_check, dict):
                                node_state["input"] = self._initial_input.copy()
                            elif not node_input_check.get("query") and self._initial_input.get("query"):
                                # query가 비어있으면 복원
                                node_state["input"]["query"] = self._initial_input["query"]
                                if self._initial_input.get("session_id"):
                                    node_state["input"]["session_id"] = self._initial_input["session_id"]

                        # 중요: merge_and_rerank_with_keyword_weights 이후 retrieved_docs 캐시 업데이트
                        if node_name == "merge_and_rerank_with_keyword_weights" and isinstance(node_state, dict):
                            search_group = node_state.get("search", {}) if isinstance(node_state.get("search"), dict) else {}
                            retrieved_docs = search_group.get("retrieved_docs", [])
                            merged_documents = search_group.get("merged_documents", [])

                            # 최상위 레벨에서도 확인
                            top_retrieved_docs = node_state.get("retrieved_docs", [])
                            top_merged_docs = node_state.get("merged_documents", [])

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
                        # LangGraph reducer가 search 그룹을 제거하는 문제를 우회하기 위해 캐시에서 복원
                        if node_name in ["merge_and_rerank_with_keyword_weights", "filter_and_validate_results", "update_search_metadata", "prepare_document_context_for_prompt"]:
                            if self._search_results_cache and isinstance(node_state, dict):
                                # node_state에 search 그룹이 없거나 비어있으면 캐시에서 복원
                                search_group = node_state.get("search", {}) if isinstance(node_state.get("search"), dict) else {}
                                top_semantic = node_state.get("semantic_results", [])
                                top_keyword = node_state.get("keyword_results", [])
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
                        # flat_result 업데이트 전에 캐시 업데이트 (node_state에서 직접 읽기)
                        if node_name in ["merge_and_rerank_with_keyword_weights", "process_search_results_combined"] and isinstance(node_state, dict):
                            self.logger.debug(f"astream: Checking merge_and_rerank node_state for retrieved_docs")
                            # node_state 업데이트 후 다시 읽기
                            search_group_updated = node_state.get("search", {}) if isinstance(node_state.get("search"), dict) else {}
                            retrieved_docs_updated = search_group_updated.get("retrieved_docs", [])
                            merged_documents_updated = search_group_updated.get("merged_documents", [])

                            # 최상위 레벨에서도 확인
                            top_retrieved_docs_updated = node_state.get("retrieved_docs", [])
                            top_merged_docs_updated = node_state.get("merged_documents", [])

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
                                self.logger.warning(f"astream: merge_and_rerank node_state has no retrieved_docs or merged_documents")

                        flat_result = node_state

                # 모든 노드 실행 완료 표시
                total_nodes = len(executed_nodes)
                self.logger.info(f"✅ 워크플로우 실행 완료 (총 {total_nodes}개 노드 실행)")
                print(f"✅ 워크플로우 실행 완료 (총 {total_nodes}개 노드 실행)", flush=True)

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
                            self.logger.debug(f"Final result has no search group, creating from cache")
                            flat_result["search"] = {}

                        search_group = flat_result["search"]
                        has_results = (len(search_group.get("retrieved_docs", [])) > 0 or
                                     len(search_group.get("merged_documents", [])) > 0 or
                                     len(flat_result.get("retrieved_docs", [])) > 0)

                        if not has_results:
                            self.logger.debug(f"Restoring search group in final result from cache")
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
                        self.logger.warning(f"No search cache available for final result restoration")

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
                                self.logger.debug(f"[5] Global cache is None")
                            else:
                                self.logger.debug(f"[5] Global cache is not dict: {type(global_cache)}")
                        except Exception as e:
                            self.logger.debug(f"[5] Global cache 확인 실패: {e}")
                            import traceback
                            self.logger.debug(f"[5] Exception details: {traceback.format_exc()}")

                    # 6. 전체 state 재귀 검색 (마지막 시도)
                    if not query_complexity_found:
                        self.logger.debug(f"[6] 재귀 검색 시작...")
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
                        self.logger.debug(f"❌ query_complexity를 찾지 못함 (모든 경로 확인 완료)")

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
                            self.logger.warning(f"Restored query from preserved initial_input in final result")

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
                        from core.agents.node_wrappers import (
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

            response = {
                "answer": flat_result.get("answer", "") if isinstance(flat_result, dict) else "",
                "sources": flat_result.get("sources", []) if isinstance(flat_result, dict) else [],
                "confidence": flat_result.get("confidence", 0.0) if isinstance(flat_result, dict) else 0.0,
                "legal_references": flat_result.get("legal_references", []) if isinstance(flat_result, dict) else [],
                "processing_steps": processing_steps,
                "session_id": session_id,
                "processing_time": processing_time,
                "query_type": flat_result.get("query_type", "") if isinstance(flat_result, dict) else "",
                "metadata": flat_result.get("metadata", {}) if isinstance(flat_result, dict) else {},
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
                "retrieved_docs": self._extract_retrieved_docs_from_result(flat_result)
            }

            # Langfuse에 답변 품질 추적
            if self.langfuse_client_service and self.langfuse_client_service.is_enabled():
                self._track_answer_quality(query, response, processing_time)

            self.logger.info(f"Query processed successfully in {processing_time:.2f}s")
            return response

        except Exception as e:
            import traceback
            error_msg = f"질문 처리 중 오류 발생: {str(e)}"
            self.logger.error(error_msg)
            self.logger.error(traceback.format_exc())

            return {
                "answer": "죄송합니다. 질문 처리 중 오류가 발생했습니다.",
                "sources": [],
                "confidence": 0.0,
                "legal_references": [],
                "processing_steps": [error_msg],
                "session_id": session_id or str(uuid.uuid4()),
                "processing_time": time.time() - start_time if 'start_time' in locals() else 0.0,
                "query_type": "error",
                "metadata": {"error": str(e)},
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
            from core.agents.node_wrappers import _global_search_results_cache
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
