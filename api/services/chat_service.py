"""
채팅 서비스 (lawfirm_langgraph 래퍼)
"""
import sys
import re
import json
import logging
from pathlib import Path
from datetime import datetime
from typing import Dict, Any, Optional, AsyncGenerator

# 프로젝트 루트를 sys.path에 추가
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

# lawfirm_langgraph 디렉토리를 sys.path에 추가 (core 모듈 import를 위해)
lawfirm_langgraph_path = project_root / "lawfirm_langgraph"
if lawfirm_langgraph_path.exists():
    sys.path.insert(0, str(lawfirm_langgraph_path))

# 환경 변수 로드 (ChatService 초기화 전에 반드시 로드)
try:
    from dotenv import load_dotenv
    
    # 1. lawfirm_langgraph/.env 로드 (LangGraphConfig가 사용)
    langgraph_env = lawfirm_langgraph_path / ".env"
    if langgraph_env.exists():
        load_dotenv(dotenv_path=str(langgraph_env), override=False)
        logging.info(f"✅ [ChatService] Loaded environment from: {langgraph_env}")
    else:
        logging.warning(f"⚠️  [ChatService] Environment file not found: {langgraph_env}")
    
    # 2. 프로젝트 루트 .env 로드 (공통 설정)
    root_env = project_root / ".env"
    if root_env.exists():
        load_dotenv(dotenv_path=str(root_env), override=False)
        logging.info(f"✅ [ChatService] Loaded environment from: {root_env}")
    
    # 3. api/.env 로드 (API 서버 전용 설정, 최우선)
    api_env = Path(__file__).parent.parent / ".env"
    if api_env.exists():
        load_dotenv(dotenv_path=str(api_env), override=True)
        logging.info(f"✅ [ChatService] Loaded environment from: {api_env}")
        
except ImportError:
    logging.warning("⚠️  python-dotenv not installed. Environment variables from .env files will not be loaded.")
except Exception as e:
    logging.warning(f"⚠️  Failed to load environment variables: {e}")

try:
    from lawfirm_langgraph.langgraph_core.workflow.workflow_service import LangGraphWorkflowService
    from lawfirm_langgraph.config.langgraph_config import LangGraphConfig
    LANGGRAPH_AVAILABLE = True
except ImportError as e:
    LANGGRAPH_AVAILABLE = False
    logging.warning(f"LangGraph not available: {e}")

logger = logging.getLogger(__name__)

# 로거 레벨을 명시적으로 설정 (루트 로거 레벨과 동기화)
# 환경 변수에서 로그 레벨 읽기
import os
log_level_str = os.getenv("LOG_LEVEL", "info").upper()
log_level_map = {
    "CRITICAL": logging.CRITICAL,
    "ERROR": logging.ERROR,
    "WARNING": logging.WARNING,
    "INFO": logging.INFO,
    "DEBUG": logging.DEBUG,
}
log_level = log_level_map.get(log_level_str, logging.INFO)
logger.setLevel(log_level)
logger.disabled = False  # 명시적으로 활성화
logger.propagate = True  # 루트 로거로 전파

# 로깅이 비활성화되지 않도록 보호
logging.disable(logging.NOTSET)  # 모든 로깅 활성화

# 루트 로거에 핸들러가 없으면 추가 (모듈 import 시점에 로깅이 설정되지 않았을 수 있음)
root_logger = logging.getLogger()
if not root_logger.handlers:
    handler = logging.StreamHandler()
    handler.setLevel(log_level)
    handler.setFormatter(
        logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    )
    root_logger.addHandler(handler)
    root_logger.setLevel(log_level)
    root_logger.disabled = False

# 테스트 로그 출력 (모듈 import 시점)
logger.info("✅ ChatService logger initialized and enabled")
logger.debug("✅ ChatService logger debug level enabled")


class ChatService:
    """채팅 서비스"""
    
    def __init__(self):
        """초기화"""
        logger.info("🚀 ChatService.__init__() called - Initializing ChatService...")
        self.workflow_service: Optional[LangGraphWorkflowService] = None
        self._initialize_workflow()
        logger.info("✅ ChatService.__init__() completed")
    
    def _initialize_workflow(self):
        """워크플로우 초기화"""
        if not LANGGRAPH_AVAILABLE:
            logger.warning("LangGraph is not available. Service will continue without LangGraph features.")
            return
        
        try:
            import os
            # 환경 변수 확인 (민감 정보는 로그에 노출하지 않음)
            google_api_key = os.getenv("GOOGLE_API_KEY", "")
            if not google_api_key:
                logger.warning("GOOGLE_API_KEY가 설정되지 않았습니다. 환경 변수를 확인하세요.")
                logger.warning("LangGraph는 Google API Key가 필요합니다.")
            else:
                logger.info("GOOGLE_API_KEY가 설정되었습니다.")
            
            logger.info("Loading LangGraphConfig from environment...")
            
            config = LangGraphConfig.from_env()
            logger.info(f"LangGraph Config loaded: langgraph_enabled={config.langgraph_enabled}, llm_provider={config.llm_provider}")
            
            if not config.langgraph_enabled:
                logger.warning("LangGraph is disabled in configuration")
                return
            
            logger.info("Initializing LangGraphWorkflowService...")
            
            self.workflow_service = LangGraphWorkflowService(config)
            logger.info("✅ ChatService initialized successfully with LangGraph workflow")
        except ImportError as e:
            logger.error(f"Import error during workflow initialization: {e}", exc_info=True)
            self.workflow_service = None
        except Exception as e:
            logger.error(f"Failed to initialize workflow service: {e}", exc_info=True)
            import traceback
            tb = traceback.format_exc()
            logger.error(f"Traceback:\n{tb}")
            self.workflow_service = None
    
    async def process_message(
        self,
        message: str,
        session_id: Optional[str] = None,
        enable_checkpoint: bool = True
    ) -> Dict[str, Any]:
        """
        메시지 처리
        
        Args:
            message: 사용자 메시지
            session_id: 세션 ID
            enable_checkpoint: 체크포인트 사용 여부
            
        Returns:
            처리 결과
        """
        if not self.workflow_service:
            import os
            error_details = []
            
            # 원인 분석
            if not LANGGRAPH_AVAILABLE:
                error_details.append("LangGraph 모듈을 import할 수 없습니다.")
            else:
                google_api_key = os.getenv("GOOGLE_API_KEY", "")
                if not google_api_key:
                    error_details.append("GOOGLE_API_KEY 환경 변수가 설정되지 않았습니다.")
                else:
                    error_details.append("워크플로우 서비스 초기화에 실패했습니다. API 서버 로그를 확인하세요.")
            
            error_msg = f"Workflow service unavailable. Details: {', '.join(error_details)}"
            logger.error(error_msg)
            
            return {
                "answer": f"죄송합니다. 서비스 초기화에 실패했습니다.\n\n원인:\n" + "\n".join(f"- {detail}" for detail in error_details) + "\n\nAPI 서버 로그를 확인하거나 환경 변수를 설정해주세요.",
                "sources": [],
                "confidence": 0.0,
                "legal_references": [],
                "processing_steps": ["서비스 초기화 실패"],
                "session_id": session_id or "error",
                "processing_time": 0.0,
                "query_type": "error",
                "metadata": {"error_details": error_details},
                "errors": error_details
            }
        
        try:
            result = await self.workflow_service.process_query(
                query=message,
                session_id=session_id,
                enable_checkpoint=enable_checkpoint
            )
            return result
        except Exception as e:
            logger.error(f"Error processing message: {e}", exc_info=True)
            import os
            debug_mode = os.getenv("DEBUG", "false").lower() == "true"
            error_detail = str(e) if debug_mode else "메시지 처리 중 오류가 발생했습니다"
            return {
                "answer": "죄송합니다. 메시지 처리 중 오류가 발생했습니다.",
                "sources": [],
                "confidence": 0.0,
                "legal_references": [],
                "processing_steps": [f"오류: {error_detail}"],
                "session_id": session_id or "error",
                "processing_time": 0.0,
                "query_type": "error",
                "metadata": {"error": error_detail} if debug_mode else {"error": True},
                "errors": [error_detail]
            }
    
    async def stream_message(
        self,
        message: str,
        session_id: Optional[str] = None
    ) -> AsyncGenerator[str, None]:
        """
        스트리밍 메시지 처리 (Server-Sent Events)
        실제 LLM 토큰 스트리밍을 지원합니다.
        
        Args:
            message: 사용자 메시지
            session_id: 세션 ID
            
        Yields:
            스트리밍 응답 청크 (토큰 단위)
        """
        # 디버그 모드 확인 (환경 변수로 제어) - 함수 시작 부분에서 정의
        import os
        DEBUG_STREAM = os.getenv("DEBUG_STREAM", "false").lower() == "true"
        
        has_yielded = False  # 최소한 하나의 yield가 있었는지 추적
        
        if not self.workflow_service:
            error_event = {
                "type": "final",
                "content": "[오류] 서비스 초기화에 실패했습니다.",
                "metadata": {"error": True},
                "timestamp": datetime.now().isoformat()
            }
            yield json.dumps(error_event, ensure_ascii=False) + "\n"
            has_yielded = True
            return
        
        try:
            import uuid
            
            # 세션 ID 생성
            if not session_id:
                session_id = str(uuid.uuid4())
            
            # 워크플로우 스트리밍 실행
            from lawfirm_langgraph.langgraph_core.state.state_definitions import create_initial_legal_state
            
            # 로깅: message 값 확인
            logger.info(f"stream_message: 받은 message='{message[:100] if message else 'EMPTY'}...', length={len(message) if message else 0}")
            
            # message를 query로 사용 (create_initial_legal_state의 첫 번째 파라미터는 query)
            initial_state = create_initial_legal_state(message, session_id)
            
            # 중요: initial_state에 query가 반드시 포함되도록 강제
            # LangGraph에 전달하기 전에 input 그룹에 query가 있어야 함
            if "input" not in initial_state:
                initial_state["input"] = {}
            if not initial_state["input"].get("query"):
                initial_state["input"]["query"] = message
                logger.debug(f"stream_message: input.query에 message 설정: '{message[:50]}...'")
            if not initial_state["input"].get("session_id"):
                initial_state["input"]["session_id"] = session_id
            
            # 최상위 레벨에도 query 포함 (이중 보장)
            if not initial_state.get("query"):
                initial_state["query"] = message
                logger.debug(f"stream_message: 최상위 query에 message 설정: '{message[:50]}...'")
            if not initial_state.get("session_id"):
                initial_state["session_id"] = session_id
            
            # 초기 state 검증
            initial_query = initial_state.get("input", {}).get("query", "") if initial_state.get("input") else initial_state.get("query", "")
            logger.info(f"stream_message: initial_state query length={len(initial_query)}, query='{initial_query[:100] if initial_query else 'EMPTY'}...'")
            logger.debug(f"stream_message: initial_state input.query='{initial_state.get('input', {}).get('query', 'NOT_FOUND')[:50] if initial_state.get('input', {}).get('query') else 'NOT_FOUND'}...'")
            logger.debug(f"stream_message: initial_state 최상위 query='{initial_state.get('query', 'NOT_FOUND')[:50] if initial_state.get('query') else 'NOT_FOUND'}...'")
            
            if not initial_query or not str(initial_query).strip():
                logger.error(f"Initial state query is empty! Input message was: '{message[:50]}...'")
                logger.debug(f"stream_message: ERROR - initial_state query is empty!")
                logger.debug(f"stream_message: initial_state keys: {list(initial_state.keys())}")
                error_event = {
                    "type": "final",
                    "content": "[오류] 질문이 제대로 전달되지 않았습니다. 다시 시도해주세요.",
                    "metadata": {"error": True},
                    "timestamp": datetime.now().isoformat()
                }
                yield json.dumps(error_event, ensure_ascii=False) + "\n"
                return
            
            config = {"configurable": {"thread_id": session_id}}
            
            # 실제 토큰 스트리밍을 위한 변수
            full_answer = ""
            answer_found = False
            tokens_received = 0
            last_node_name = None
            executed_nodes = set()  # 실행된 노드 추적
            answer_generation_started = False  # 답변 생성 노드 시작 플래그
            json_output_detected = False  # JSON 출력 감지 플래그
            
            # 노드 이름을 사용자 친화적인 메시지로 매핑
            node_name_mapping = {
                "classify_query_and_complexity": "질문 분석 중...",
                "classification_parallel": "질문 분류 중...",
                "route_expert": "전문가 라우팅 중...",
                "expand_keywords": "키워드 확장 중...",
                "prepare_search_query": "검색 쿼리 준비 중...",
                "execute_searches_parallel": "관련 법률 검색 중...",
                "process_search_results_combined": "검색 결과 분석 중...",
                "prepare_documents_and_terms": "문서 준비 중...",
                "generate_answer_enhanced": "답변 생성 중...",
                "generate_and_validate_answer": "답변 생성 중...",
                "validate_answer_quality": "답변 검증 중...",
                "prepare_final_response": "최종 답변 준비 중..."
            }
            
            # astream_events()를 사용하여 LLM 토큰 스트리밍 감지
            # 
            # 스트리밍 흐름:
            # 1. LangGraph의 astream_events()가 워크플로우 실행 중 모든 이벤트를 스트리밍
            # 2. LLM 호출 시 on_llm_stream 또는 on_chat_model_stream 이벤트 발생
            # 3. LangChain의 ChatGoogleGenerativeAI/Ollama는 invoke() 호출 시에도
            #    내부적으로 스트리밍을 사용하므로 astream_events()가 이를 캡처 가능
            # 4. 답변 생성 노드(generate_answer_enhanced)에서 발생한 이벤트만 필터링
            # 5. 각 토큰을 JSONL 형식으로 yield하여 HTTP 스트리밍으로 전달
            try:
                # 실제 스트리밍 이벤트 처리
                # LangGraph 버전별 호환성: version 파라미터가 없을 수도 있음
                # wrapper 함수로 버전 호환성 처리
                async def get_stream_events():
                    """버전 호환성을 위한 스트리밍 이벤트 래퍼
                    
                    LangGraph의 astream_events()는 워크플로우 실행 중 모든 이벤트를
                    스트리밍으로 제공합니다. LLM 호출 시 on_llm_stream 또는 
                    on_chat_model_stream 이벤트가 발생하며, LangChain의 LLM은 
                    invoke() 호출 시에도 내부적으로 스트리밍을 사용하므로 
                    이 이벤트들을 통해 실시간 토큰 스트리밍이 가능합니다.
                    """
                    try:
                        # version="v2" 시도 (LangGraph 최신 버전)
                        # include_names로 generate_and_validate_answer 노드의 이벤트만 필터링 시도
                        if DEBUG_STREAM:
                            logger.info("스트리밍 시작: astream_events(version='v2') 사용")
                        try:
                            # include_names 파라미터로 특정 노드만 필터링 시도
                            try:
                                async for event in self.workflow_service.app.astream_events(
                                    initial_state, 
                                    config,
                                    version="v2",
                                    include_names=["generate_and_validate_answer", "generate_answer_enhanced"]
                                ):
                                    yield event
                            except (TypeError, AttributeError):
                                # include_names가 지원되지 않는 경우 exclude_names 시도
                                try:
                                    async for event in self.workflow_service.app.astream_events(
                                        initial_state, 
                                        config,
                                        version="v2",
                                        exclude_names=["classify_query_and_complexity", "classification_parallel", 
                                                      "expand_keywords", "validate_answer_quality", "prepare_search_query",
                                                      "execute_searches_parallel", "process_search_results_combined",
                                                      "prepare_documents_and_terms", "prepare_final_response"]
                                    ):
                                        yield event
                                except (TypeError, AttributeError):
                                    # exclude_names도 지원되지 않는 경우 모든 이벤트 처리
                                    async for event in self.workflow_service.app.astream_events(
                                        initial_state, 
                                        config,
                                        version="v2"
                                    ):
                                        yield event
                        except (TypeError, AttributeError):
                            # include_names가 지원되지 않는 경우 모든 이벤트 처리
                            async for event in self.workflow_service.app.astream_events(
                                initial_state, 
                                config,
                                version="v2"
                            ):
                                yield event
                    except (TypeError, AttributeError) as ve:
                        # version 파라미터가 지원되지 않는 경우 (구버전)
                        logger.debug(f"astream_events에서 version 파라미터 미지원: {ve}, 기본 버전 사용")
                        if DEBUG_STREAM:
                            logger.info("스트리밍 시작: astream_events() 사용 (기본 버전)")
                        async for event in self.workflow_service.app.astream_events(
                            initial_state, 
                            config
                        ):
                            yield event
                
                # 스트리밍 이벤트 처리
                event_count = 0
                llm_stream_count = 0
                event_types_seen = set()  # 본 이벤트 타입 추적
                node_names_seen = set()  # 본 노드 이름 추적
                
                async for event in get_stream_events():
                    event_count += 1
                    # 이벤트 타입 확인
                    event_type = event.get("event", "")
                    event_name = event.get("name", "")
                    
                    # 모든 이벤트 타입 추적 (디버깅용)
                    event_types_seen.add(event_type)
                    if event_name:
                        node_names_seen.add(event_name)
                    
                    # 디버깅: 처음 100개 이벤트는 항상 로깅 (문제 진단용)
                    if event_count <= 100:
                        logger.info(f"이벤트 #{event_count}: type={event_type}, name={event_name}")
                    
                    # 관련 없는 이벤트는 즉시 건너뛰기 (성능 최적화)
                    # on_chat_model_end도 추가 (Google Gemini는 on_chat_model_stream 사용)
                    # on_chain_stream도 추가 (LangGraph에서 체인 레벨 스트리밍 이벤트)
                    if event_type not in ["on_llm_stream", "on_chat_model_stream", "on_chain_stream", "on_chain_start", "on_chain_end", "on_llm_end", "on_chat_model_end"]:
                        continue
                    
                    # 디버깅: 처리할 이벤트 로깅 (처음 20개만)
                    if DEBUG_STREAM and event_count <= 20:
                        logger.debug(f"처리할 이벤트 #{event_count}: type={event_type}, name={event_name}")
                    
                    # LLM 스트리밍 이벤트 감지 (답변 생성 노드에서만)
                    # 
                    # 중요: LangChain의 ChatGoogleGenerativeAI와 Ollama는
                    # invoke() 호출 시에도 내부적으로 스트리밍을 사용합니다.
                    # 따라서 astream_events()가 on_llm_stream 또는 on_chat_model_stream
                    # 이벤트를 발생시켜 실시간 토큰 스트리밍이 가능합니다.
                    # 
                    # LangGraph/LangChain 최신 버전에서는 on_chat_model_stream도 지원
                    # on_chain_stream은 체인 레벨의 스트리밍 이벤트로, 체인 전체의 출력을 포함할 수 있습니다.
                    if event_type in ["on_llm_stream", "on_chat_model_stream", "on_chain_stream"]:
                        # 답변 생성 노드가 시작되었는지 확인 (조기 종료 최적화)
                        if not answer_generation_started:
                            # 답변 생성 노드가 시작되지 않았으면 모든 LLM 출력 무시
                            llm_stream_count += 1
                            if DEBUG_STREAM and llm_stream_count <= 5:
                                logger.debug(f"답변 생성 노드가 시작되지 않음: {event_name} (모든 출력 무시)")
                            # JSON 출력이든 아니든 모두 무시
                            continue
                        
                        llm_stream_count += 1
                        if DEBUG_STREAM:
                            logger.debug(f"{event_type} 이벤트 발견: name={event_name}, 전체 이벤트 키: {list(event.keys())}")
                        
                        # 이벤트의 상위 노드 정보 확인
                        event_tags = event.get("tags", [])
                        event_metadata = event.get("metadata", {})
                        event_parent = event.get("parent", {})
                        
                        # 상위 노드 이름 확인
                        parent_node_name = None
                        if isinstance(event_parent, dict):
                            parent_node_name = event_parent.get("name", "")
                        elif isinstance(event_tags, list):
                            # tags에서 노드 이름 찾기
                            for tag in event_tags:
                                if isinstance(tag, str) and ("generate_answer" in tag.lower() or "generate_and_validate" in tag.lower()):
                                    parent_node_name = tag
                                    break
                        
                        # 답변 생성 노드 내부의 LLM 호출인지 확인
                        is_answer_node = False
                        
                        # 방법 1: 이벤트 이름으로 직접 판단
                        if "generate_answer" in event_name.lower() or \
                           "generate_and_validate" in event_name.lower() or \
                           event_name in ["generate_answer_enhanced", "generate_and_validate_answer", "direct_answer"]:
                            is_answer_node = True
                        
                        # 방법 2: 상위 노드가 답변 생성 노드인지 확인
                        elif parent_node_name and (
                            "generate_answer" in parent_node_name.lower() or 
                            "generate_and_validate" in parent_node_name.lower() or
                            parent_node_name in ["generate_answer_enhanced", "generate_and_validate_answer"]
                        ):
                            is_answer_node = True
                        
                        # 방법 3: ChatGoogleGenerativeAI인 경우, 마지막으로 실행된 노드가 답변 생성 노드인지 확인
                        elif event_name == "ChatGoogleGenerativeAI" and answer_generation_started:
                            # last_node_name이 정확히 generate_and_validate_answer인지 확인
                            if last_node_name == "generate_and_validate_answer":
                                is_answer_node = True
                            # generate_answer_enhanced도 허용 (generate_and_validate_answer 내부에서 호출)
                            elif last_node_name == "generate_answer_enhanced":
                                # generate_and_validate_answer 노드가 실행 중인지 확인
                                if "generate_and_validate_answer" in executed_nodes:
                                    is_answer_node = True
                            else:
                                # 다른 노드는 무시
                                is_answer_node = False
                        
                        # 디버깅: 모든 스트리밍 이벤트 로깅 (디버그 모드에서만, 처음 10개만)
                        if DEBUG_STREAM and llm_stream_count <= 10:
                            logger.debug(
                                f"{event_type} 이벤트 #{llm_stream_count}: "
                                f"name={event_name}, parent={parent_node_name}, "
                                f"is_answer_node={is_answer_node}, "
                                f"answer_generation_started={answer_generation_started}, "
                                f"last_node={last_node_name}, "
                                f"tags={event_tags}, metadata={event_metadata}"
                            )
                        
                        if not is_answer_node:
                            # 답변 생성 노드가 아니면 무시
                            if DEBUG_STREAM and llm_stream_count <= 5:
                                logger.debug(f"답변 생성 노드가 아님: {event_name}, parent={parent_node_name} (무시)")
                            continue
                        
                        if DEBUG_STREAM:
                            logger.debug(f"✅ 답변 생성 노드에서 {event_type} 이벤트 감지: {event_name}, parent={parent_node_name}")
                        
                        if is_answer_node:
                            if DEBUG_STREAM:
                                logger.debug(f"LLM 스트리밍 이벤트 감지: {event_name} (총 {llm_stream_count}개)")
                            # 토큰 추출 (다양한 이벤트 구조 지원)
                            chunk = None
                            event_data = event.get("data", {})
                            
                            try:
                                # on_chain_stream 이벤트의 경우 특별 처리
                                if event_type == "on_chain_stream":
                                    # on_chain_stream은 체인 레벨의 스트리밍 이벤트
                                    # data 필드에 체인의 출력이 포함될 수 있음
                                    # 주의: on_chain_stream은 체인 전체의 출력을 포함할 수 있으므로
                                    # 이전에 받은 답변과 비교하여 새로운 부분만 추출해야 함
                                    
                                    # 1. 이벤트 구조 확인 및 로깅 강화
                                    if DEBUG_STREAM:
                                        logger.debug(f"on_chain_stream 이벤트 수신: event_name={event_name}")
                                        logger.debug(f"on_chain_stream 이벤트 구조: event_data type={type(event_data)}")
                                        if isinstance(event_data, dict):
                                            logger.debug(f"on_chain_stream event_data keys: {list(event_data.keys())}")
                                            # event_data의 주요 키 값 로깅 (너무 길면 잘라서)
                                            for key in list(event_data.keys())[:5]:  # 처음 5개만
                                                value = event_data.get(key)
                                                if isinstance(value, str):
                                                    logger.debug(f"on_chain_stream event_data['{key}'] (str, {len(value)}자): {value[:100]}...")
                                                elif isinstance(value, dict):
                                                    logger.debug(f"on_chain_stream event_data['{key}'] (dict): keys={list(value.keys())[:10]}")
                                                else:
                                                    logger.debug(f"on_chain_stream event_data['{key}']: {type(value)}")
                                    
                                    try:
                                        if isinstance(event_data, dict):
                                            # 체인 출력에서 answer 필드 추출 시도
                                            chain_output = event_data.get("chunk") or event_data.get("output")
                                            
                                            # 로깅: chain_output 구조 확인
                                            if DEBUG_STREAM:
                                                logger.debug(f"on_chain_stream: chain_output type={type(chain_output)}")
                                                if chain_output:
                                                    if isinstance(chain_output, str):
                                                        logger.debug(f"on_chain_stream: chain_output (str, {len(chain_output)}자): {chain_output[:100]}...")
                                                    elif isinstance(chain_output, dict):
                                                        logger.debug(f"on_chain_stream: chain_output (dict): keys={list(chain_output.keys())[:10]}")
                                            
                                            if chain_output is not None:
                                                # chain_output이 딕셔너리인 경우 answer 필드 확인
                                                if isinstance(chain_output, dict):
                                                    # answer 그룹에서 추출
                                                    answer_group = chain_output.get("answer", {})
                                                    if isinstance(answer_group, dict):
                                                        full_answer_from_event = answer_group.get("answer", "") or answer_group.get("content", "")
                                                    elif isinstance(answer_group, str):
                                                        full_answer_from_event = answer_group
                                                    else:
                                                        full_answer_from_event = ""
                                                    
                                                    # 최상위 레벨에서 직접 추출
                                                    if not full_answer_from_event:
                                                        full_answer_from_event = chain_output.get("answer", "") or chain_output.get("content", "")
                                                    
                                                    # 이전에 받은 답변과 비교하여 새로운 부분만 추출
                                                    if full_answer_from_event and isinstance(full_answer_from_event, str):
                                                        if len(full_answer_from_event) > len(full_answer):
                                                            # 새로운 부분만 추출
                                                            new_part = full_answer_from_event[len(full_answer):]
                                                            
                                                            # 토큰 단위로 분할하여 전ㅇ
                                                            # 공백과 구두점을 기준으로 토큰 분할
                                                            # 공백, 구두점, 한글/영문/숫자를 기준으로 토큰 분할
                                                            tokens = re.findall(r'\S+|\s+', new_part)
                                                            for token in tokens:
                                                                # 줄바꿈을 포함한 모든 토큰 전송 (strip() 체크 제거)
                                                                if token:  # 빈 문자열이 아니면 전송 (줄바꿈 포함)
                                                                    try:
                                                                        # type: "stream"으로 전송
                                                                        stream_event = {
                                                                            "type": "stream",
                                                                            "content": token,
                                                                            "timestamp": datetime.now().isoformat()
                                                                        }
                                                                        event_json = json.dumps(stream_event, ensure_ascii=False)
                                                                        yield event_json + "\n"
                                                                        full_answer += token
                                                                        tokens_received += 1
                                                                        answer_found = True
                                                                        if DEBUG_STREAM:
                                                                            logger.debug(f"on_chain_stream: 토큰 전송 ({len(token)}자, 누적 {len(full_answer)}자)")
                                                                    except (TypeError, ValueError) as json_error:
                                                                        # JSON 직렬화 실패 시 로깅만 하고 계속 진행
                                                                        if DEBUG_STREAM:
                                                                            logger.warning(f"토큰 JSON 직렬화 실패: {json_error}, token={repr(token[:50])}")
                                                                        continue
                                                            # full_answer 업데이트
                                                            full_answer = full_answer_from_event
                                                            chunk = None  # 이미 토큰 단위로 전송했으므로 None으로 설정
                                                        else:
                                                            # 이미 받은 내용이면 스킵
                                                            chunk = None
                                                # chain_output이 문자열인 경우
                                                elif isinstance(chain_output, str):
                                                    # 이전에 받은 답변과 비교하여 새로운 부분만 추출
                                                    if len(chain_output) > len(full_answer):
                                                        new_part = chain_output[len(full_answer):]
                                                        
                                                        # 토큰 단위로 분할하여 전송 (타이핑 효과를 위해 개별 전송)
                                                        # 공백과 구두점을 기준으로 토큰 분할
                                                        tokens = re.findall(r'\S+|\s+', new_part)
                                                        for token in tokens:
                                                            # 줄바꿈을 포함한 모든 토큰 전송
                                                            if token:  # 빈 문자열이 아니면 전송 (줄바꿈 포함)
                                                                # type: "stream"으로 전송 (타이핑 효과를 위해 개별 전송)
                                                                stream_event = {
                                                                    "type": "stream",
                                                                    "content": token,
                                                                    "timestamp": datetime.now().isoformat()
                                                                }
                                                                yield json.dumps(stream_event, ensure_ascii=False) + "\n"
                                                                full_answer += token
                                                                tokens_received += 1
                                                                answer_found = True
                                                                if DEBUG_STREAM:
                                                                    logger.debug(f"on_chain_stream: 토큰 전송 ({len(token)}자, 누적 {len(full_answer)}자)")
                                                        # full_answer 업데이트
                                                        full_answer = chain_output
                                                        chunk = None  # 이미 토큰 단위로 전송했으므로 None으로 설정
                                                    else:
                                                        chunk = None
                                                # chain_output이 객체인 경우 content 속성 확인
                                                elif hasattr(chain_output, "content"):
                                                    content = chain_output.content
                                                    if isinstance(content, str):
                                                        if len(content) > len(full_answer):
                                                            new_part = content[len(full_answer):]
                                                            
                                                            # 토큰 단위로 분할하여 전송 (타이핑 효과를 위해 개별 전송)
                                                            # 공백과 구두점을 기준으로 토큰 분할
                                                            tokens = re.findall(r'\S+|\s+', new_part)
                                                            for token in tokens:
                                                                if token:  # 빈 문자열이 아닌 경우만 전송
                                                                    # type: "stream"으로 전송 (타이핑 효과를 위해 개별 전송)
                                                                    stream_event = {
                                                                        "type": "stream",
                                                                        "content": token,
                                                                        "timestamp": datetime.now().isoformat()
                                                                    }
                                                                    yield json.dumps(stream_event, ensure_ascii=False) + "\n"
                                                                    full_answer += token
                                                                    tokens_received += 1
                                                                    answer_found = True
                                                                    if DEBUG_STREAM:
                                                                        logger.debug(f"on_chain_stream: 토큰 전송 ({len(token)}자, 누적 {len(full_answer)}자)")
                                                            # full_answer 업데이트
                                                            full_answer = content
                                                            chunk = None  # 이미 토큰 단위로 전송했으므로 None으로 설정
                                                        else:
                                                            chunk = None
                                                    elif isinstance(content, dict):
                                                        answer_text = content.get("answer", "") or content.get("content", "")
                                                        if answer_text and isinstance(answer_text, str):
                                                            if len(answer_text) > len(full_answer):
                                                                new_part = answer_text[len(full_answer):]
                                                                
                                                                # 토큰 단위로 분할하여 전송 (타이핑 효과를 위해 개별 전송)
                                                                # 공백과 구두점을 기준으로 토큰 분할
                                                                tokens = re.findall(r'\S+|\s+', new_part)
                                                                for token in tokens:
                                                                    if token:  # 빈 문자열이 아닌 경우만 전송
                                                                        # type: "stream"으로 전송 (타이핑 효과를 위해 개별 전송)
                                                                        stream_event = {
                                                                            "type": "stream",
                                                                            "content": token,
                                                                            "timestamp": datetime.now().isoformat()
                                                                        }
                                                                        yield json.dumps(stream_event, ensure_ascii=False) + "\n"
                                                                        full_answer += token
                                                                        tokens_received += 1
                                                                        answer_found = True
                                                                        if DEBUG_STREAM:
                                                                            logger.debug(f"on_chain_stream: 토큰 전송 ({len(token)}자, 누적 {len(full_answer)}자)")
                                                                # full_answer 업데이트
                                                                full_answer = answer_text
                                                                chunk = None  # 이미 토큰 단위로 전송했으므로 None으로 설정
                                                            else:
                                                                chunk = None
                                                        else:
                                                            chunk = None
                                                    else:
                                                        chunk = None
                                            
                                            # 2. chunk 추출 실패 시 대체 경로 추가
                                            if not chunk:
                                                # 대체 경로 1: event_data에서 직접 추출
                                                text_content = event_data.get("text") or event_data.get("content")
                                                if text_content and isinstance(text_content, str):
                                                    if len(text_content) > len(full_answer):
                                                        new_part = text_content[len(full_answer):]
                                                        
                                                        # 토큰 단위로 분할하여 전송 (타이핑 효과를 위해 개별 전송)
                                                        # 공백과 구두점을 기준으로 토큰 분할
                                                        tokens = re.findall(r'\S+|\s+', new_part)
                                                        for token in tokens:
                                                            # 줄바꿈을 포함한 모든 토큰 전송
                                                            if token:  # 빈 문자열이 아니면 전송 (줄바꿈 포함)
                                                                # type: "stream"으로 전송 (타이핑 효과를 위해 개별 전송)
                                                                stream_event = {
                                                                    "type": "stream",
                                                                    "content": token,
                                                                    "timestamp": datetime.now().isoformat()
                                                                }
                                                                yield json.dumps(stream_event, ensure_ascii=False) + "\n"
                                                                full_answer += token
                                                                tokens_received += 1
                                                                answer_found = True
                                                                if DEBUG_STREAM:
                                                                    logger.debug(f"on_chain_stream: 대체 경로에서 토큰 전송 ({len(token)}자, 누적 {len(full_answer)}자)")
                                                        # full_answer 업데이트
                                                        full_answer = text_content
                                                        chunk = None  # 이미 토큰 단위로 전송했으므로 None으로 설정
                                                
                                                # 대체 경로 2: event 최상위 레벨에서 추출
                                                if not chunk:
                                                    top_level_content = event.get("chunk") or event.get("output") or event.get("text") or event.get("content")
                                                    if top_level_content and isinstance(top_level_content, str):
                                                        if len(top_level_content) > len(full_answer):
                                                            new_part = top_level_content[len(full_answer):]
                                                            
                                                            # 3. 전체 답변이 포함된 경우 작은 청크로 분할하여 전송
                                                            if len(new_part) > 200:  # 200자 이상이면 분할
                                                                chunk_size = 100  # 100자씩 분할
                                                                for i in range(0, len(new_part), chunk_size):
                                                                    chunk = new_part[i:i + chunk_size]
                                                                    if chunk:
                                                                        # type: "stream"으로 전송
                                                                        stream_event = {
                                                                            "type": "stream",
                                                                            "content": chunk,
                                                                            "timestamp": datetime.now().isoformat()
                                                                        }
                                                                        yield json.dumps(stream_event, ensure_ascii=False) + "\n"
                                                                        full_answer += chunk
                                                                        tokens_received += 1
                                                                        answer_found = True
                                                                        if DEBUG_STREAM:
                                                                            logger.debug(f"on_chain_stream: 최상위 레벨에서 분할 청크 전송 ({len(chunk)}자, 누적 {len(full_answer)}자)")
                                                                chunk = None  # 이미 전송했으므로 None으로 설정
                                                            else:
                                                                # 작은 청크는 그대로 전송
                                                                chunk = new_part
                                                                # full_answer 업데이트는 아래에서 처리
                                                                full_answer = top_level_content
                                                
                                                # 대체 경로 3: 로깅 및 경고
                                                if not chunk and DEBUG_STREAM:
                                                    logger.warning(f"on_chain_stream: chunk 추출 실패 - event_data keys: {list(event_data.keys()) if isinstance(event_data, dict) else 'N/A'}, event keys: {list(event.keys())[:10]}")
                                        else:
                                            # event_data가 dict가 아닌 경우
                                            if DEBUG_STREAM:
                                                logger.warning(f"on_chain_stream: event_data가 dict가 아님 - type={type(event_data)}, value={str(event_data)[:200]}")
                                    except Exception as chain_stream_error:
                                        # on_chain_stream 처리 중 예외 발생 시 로깅하고 계속 진행
                                        logger.error(f"on_chain_stream 처리 중 오류: {chain_stream_error}", exc_info=True)
                                        if DEBUG_STREAM:
                                            logger.debug(f"on_chain_stream 오류 상세: event_data={str(event_data)[:200] if event_data else 'None'}, event keys={list(event.keys())[:10] if isinstance(event, dict) else 'N/A'}")
                                        chunk = None
                                
                                # 경우 1: LangChain 표준 형식 - data.chunk.content
                                if not chunk and isinstance(event_data, dict):
                                    chunk_obj = event_data.get("chunk")
                                    if chunk_obj is not None:
                                        # AIMessageChunk 객체 처리
                                        if hasattr(chunk_obj, "content"):
                                            content = chunk_obj.content
                                            # content가 문자열이면 그대로 사용
                                            if isinstance(content, str):
                                                chunk = content
                                            # content가 리스트인 경우 (AIMessageChunk의 content는 리스트일 수 있음)
                                            elif isinstance(content, list) and len(content) > 0:
                                                # 리스트의 첫 번째 요소가 문자열이면 사용
                                                if isinstance(content[0], str):
                                                    chunk = content[0]
                                                else:
                                                    chunk = str(content[0])
                                            else:
                                                chunk = str(content)
                                        elif isinstance(chunk_obj, str):
                                            chunk = chunk_obj
                                        elif hasattr(chunk_obj, "text"):
                                            chunk = chunk_obj.text
                                        # AIMessageChunk 객체의 경우 직접 content 접근 시도
                                        elif hasattr(chunk_obj, "__class__") and "AIMessageChunk" in str(type(chunk_obj)):
                                            try:
                                                content = getattr(chunk_obj, "content", None)
                                                if isinstance(content, str):
                                                    chunk = content
                                                elif isinstance(content, list) and len(content) > 0:
                                                    if isinstance(content[0], str):
                                                        chunk = content[0]
                                                    else:
                                                        chunk = str(content[0])
                                                elif content is not None:
                                                    chunk = str(content)
                                            except Exception:
                                                pass
                                    
                                    # 경우 2: 직접 문자열 형식
                                    if not chunk:
                                        chunk = event_data.get("text") or event_data.get("content")
                                    
                                    # 경우 3: delta 형식 (LangGraph v2)
                                    if not chunk and "delta" in event_data:
                                        delta = event_data["delta"]
                                        if isinstance(delta, dict):
                                            chunk = delta.get("content") or delta.get("text")
                                        elif isinstance(delta, str):
                                            chunk = delta
                                
                                # 경우 4: 이벤트 최상위 레벨에 직접 포함
                                if not chunk:
                                    chunk = event.get("chunk") or event.get("text") or event.get("content")
                                
                                # 토큰이 있으면 즉시 전송
                                if chunk and isinstance(chunk, str):
                                    # JSON 형식 출력 감지 및 필터링 (중간 노드의 JSON 출력 제거)
                                    chunk_stripped = chunk.strip()
                                    
                                    # 이전에 JSON 출력이 감지되었으면 계속 무시
                                    if json_output_detected:
                                        if DEBUG_STREAM:
                                            logger.debug(f"이전에 JSON 출력이 감지되어 계속 무시: {chunk_stripped[:50]}...")
                                        continue
                                    
                                    # JSON 형식 시작 패턴 감지
                                    is_json_output = False
                                    
                                    # 방법 1: 청크 시작 부분이 JSON 형식인지 확인 (가장 강력한 필터)
                                    # {로 시작하는 모든 청크는 JSON으로 간주
                                    if chunk_stripped.startswith("{") or chunk.startswith("{"):
                                        is_json_output = True
                                        json_output_detected = True
                                    elif chunk_stripped.startswith("```json") or chunk.startswith("```json"):
                                        is_json_output = True
                                        json_output_detected = True
                                    elif chunk_stripped.startswith("```") and "json" in chunk_stripped[:20].lower():
                                        is_json_output = True
                                        json_output_detected = True
                                    # ```로 시작하는 경우도 JSON일 가능성 높음 (코드 블록)
                                    elif chunk_stripped.startswith("```") or chunk.startswith("```"):
                                        is_json_output = True
                                        json_output_detected = True
                                    # 청크 자체에 ```json이 포함되어 있으면 JSON
                                    elif "```json" in chunk or "``` json" in chunk:
                                        is_json_output = True
                                        json_output_detected = True
                                    
                                    # 방법 2: 누적된 답변과 현재 청크를 합쳐서 JSON 형식인지 확인
                                    if not is_json_output:
                                        # full_answer가 비어있지 않으면 누적 텍스트 확인
                                        if full_answer:
                                            combined_text = (full_answer + chunk).strip()
                                            # {로 시작하는 모든 텍스트는 JSON으로 간주
                                            if combined_text.startswith("{") or combined_text.startswith("```json"):
                                                is_json_output = True
                                                json_output_detected = True
                                            elif combined_text.startswith("```") and "json" in combined_text[:20].lower():
                                                is_json_output = True
                                                json_output_detected = True
                                            elif combined_text.startswith("```"):
                                                is_json_output = True
                                                json_output_detected = True
                                            elif "```json" in combined_text or "``` json" in combined_text:
                                                is_json_output = True
                                                json_output_detected = True
                                        # full_answer가 비어있고 현재 청크가 { 또는 ```로 시작하면 JSON
                                        elif chunk_stripped.startswith("{") or chunk.startswith("{") or chunk_stripped.startswith("```") or chunk.startswith("```"):
                                            is_json_output = True
                                            json_output_detected = True
                                    
                                    # 방법 3: JSON 키워드 패턴 감지 (중간 노드의 JSON 출력 특징)
                                    if not is_json_output:
                                        json_keywords = ['"complexity"', '"confidence"', '"reasoning"', '"core_keywords"', 
                                                         '"query_intent"', '"is_valid"', '"quality_score"', '"final_score"',
                                                         '"score"', '"issues"', '"strengths"', '"recommendations"',
                                                         '"needs_improvement"', '"improvement_instructions"', '"preserve_content"',
                                                         '"focus_areas"', '"meets_quality_threshold"', '"summary"']
                                        # 청크 자체에 키워드가 있거나, 누적 텍스트에 키워드가 있는지 확인
                                        if any(keyword in chunk for keyword in json_keywords):
                                            is_json_output = True
                                            json_output_detected = True
                                        elif full_answer:
                                            combined_text = full_answer + chunk
                                            if any(keyword in combined_text for keyword in json_keywords):
                                                is_json_output = True
                                                json_output_detected = True
                                    
                                    # 방법 4: JSON 구조 패턴 감지 (큰따옴표로 시작하는 키-값 쌍)
                                    if not is_json_output:
                                        # "key": 형태의 패턴이 있으면 JSON일 가능성 높음
                                        # re 모듈은 파일 상단에서 import됨
                                        json_pattern = r'^\s*"[^"]+"\s*:\s*'
                                        if re.match(json_pattern, chunk_stripped) or re.match(json_pattern, chunk):
                                            is_json_output = True
                                            json_output_detected = True
                                    
                                    # JSON 형식이면 무시 (중간 노드의 JSON 출력)
                                    if is_json_output:
                                        if DEBUG_STREAM:
                                            logger.debug(f"JSON 형식 출력 감지 및 무시: {chunk_stripped[:100]}...")
                                        # JSON 출력은 full_answer에 누적하지 않음
                                        continue
                                    
                                    # JSON 출력이 아닌 실제 답변이 시작되면 JSON 출력 플래그 리셋
                                    if not is_json_output and json_output_detected and len(chunk_stripped) > 0:
                                        # 실제 답변이 시작된 것으로 간주 (JSON이 아닌 텍스트)
                                        if not any(keyword in chunk for keyword in ['"complexity"', '"confidence"', '"reasoning"']):
                                            json_output_detected = False
                                            if DEBUG_STREAM:
                                                logger.debug("실제 답변이 시작되어 JSON 출력 플래그 리셋")
                                    
                                    # 공백 토큰도 포함 (실제 토큰 스트리밍)
                                    # 단, 완전히 빈 문자열은 제외
                                    if len(chunk) > 0:
                                        full_answer += chunk
                                        tokens_received += 1
                                        answer_found = True
                                        # 스트림 청크를 JSONL 형식으로 전송
                                        stream_event = {
                                            "type": "stream",
                                            "content": chunk,
                                            "timestamp": datetime.now().isoformat()
                                        }
                                        yield json.dumps(stream_event, ensure_ascii=False) + "\n"
                                        
                            except (AttributeError, TypeError, KeyError) as e:
                                # 이벤트 구조가 예상과 다를 경우 로깅만 하고 계속 진행
                                logger.debug(f"토큰 추출 실패 (이벤트 구조가 예상과 다름): {e}, event_keys={list(event.keys()) if isinstance(event, dict) else 'N/A'}")
                                # 디버깅: 이벤트 구조 상세 로깅 (처음 3개만)
                                if llm_stream_count <= 3:
                                    logger.debug(f"이벤트 구조 상세: event_data={event_data}, event_data type={type(event_data)}")
                                    if isinstance(event_data, dict):
                                        logger.debug(f"event_data keys: {list(event_data.keys())}")
                                continue
                    
                    # LLM 완료 이벤트 (on_llm_end 또는 on_chat_model_end)
                    elif event_type in ["on_llm_end", "on_chat_model_end"]:
                        # 최종 답변 확인 (누락된 부분이 있는지 체크)
                        try:
                            event_data = event.get("data", {})
                            if isinstance(event_data, dict):
                                output = event_data.get("output")
                                if output is not None:
                                    final_answer = None
                                    
                                    # 다양한 출력 형식 지원
                                    if hasattr(output, "content"):
                                        final_answer = output.content
                                    elif isinstance(output, str):
                                        final_answer = output
                                    elif isinstance(output, dict):
                                        final_answer = output.get("content") or output.get("text") or str(output)
                                    else:
                                        final_answer = str(output)
                                    
                                    # 누락된 부분이 있으면 전송 (스트리밍 중 일부 토큰이 누락된 경우)
                                    if final_answer and isinstance(final_answer, str):
                                        if len(final_answer) > len(full_answer):
                                            missing_part = final_answer[len(full_answer):]
                                            if missing_part:
                                                full_answer = final_answer
                                                # 스트림 청크를 JSONL 형식으로 전송
                                                stream_event = {
                                                    "type": "stream",
                                                    "content": missing_part,
                                                    "timestamp": datetime.now().isoformat()
                                                }
                                                yield json.dumps(stream_event, ensure_ascii=False) + "\n"
                                                logger.debug(f"누락된 부분 전송: {len(missing_part)}자")
                        except (AttributeError, TypeError, KeyError) as e:
                            logger.debug(f"on_llm_end 이벤트 처리 실패: {e}")
                            pass
                    
                    # 노드 실행 이벤트 (진행 상황 표시)
                    elif event_type == "on_chain_start":
                        node_name = event.get("name", "")
                        
                        # 주요 노드의 진행 상황 표시
                        if node_name in node_name_mapping:
                            if node_name not in executed_nodes:
                                progress_message = node_name_mapping.get(node_name, f"[{node_name} 실행 중...]")
                                # 진행 상황 메시지를 JSONL 형식으로 전송
                                step_number = len(executed_nodes) + 1
                                progress_event = {
                                    "type": "progress",
                                    "step": step_number,
                                    "message": progress_message,
                                    "node_name": node_name,
                                    "timestamp": datetime.now().isoformat()
                                }
                                yield json.dumps(progress_event, ensure_ascii=False) + "\n"
                                executed_nodes.add(node_name)
                                if DEBUG_STREAM:
                                    logger.debug(f"진행 상황 메시지 전송: {progress_message}")
                        
                        # 답변 생성 노드 시작 시 플래그 설정
                        if node_name in ["generate_answer_enhanced", "generate_and_validate_answer"]:
                            answer_generation_started = True
                            json_output_detected = False  # 답변 생성 노드 시작 시 JSON 출력 플래그 리셋
                            if not answer_found:
                                # 답변 생성 시작을 JSONL 형식으로 전송
                                step_number = len(executed_nodes) + 1
                                progress_event = {
                                    "type": "progress",
                                    "step": step_number,
                                    "message": "답변 생성 중...",
                                    "node_name": node_name,
                                    "timestamp": datetime.now().isoformat()
                                }
                                yield json.dumps(progress_event, ensure_ascii=False) + "\n"
                                last_node_name = node_name
                                if DEBUG_STREAM:
                                    logger.debug(f"답변 생성 노드 시작: {node_name}, answer_generation_started=True, json_output_detected=False")
                    
                    # 노드 완료 이벤트 (generate_and_validate_answer 노드의 answer 필드만 확인)
                    elif event_type == "on_chain_end":
                        node_name = event.get("name", "")
                        if node_name == "generate_and_validate_answer":
                            # 답변 생성 노드가 완료되면 플래그 해제
                            answer_generation_started = False
                            if DEBUG_STREAM:
                                logger.debug(f"답변 생성 노드 완료: {node_name}, answer_generation_started=False")
                            
                            # generate_and_validate_answer 노드의 output에서 answer 필드만 추출
                            try:
                                event_data = event.get("data", {})
                                if isinstance(event_data, dict):
                                    output = event_data.get("output")
                                    if output is not None:
                                        # answer 필드 추출 (다양한 구조 지원)
                                        answer_text = None
                                        
                                        if isinstance(output, dict):
                                            # 최상위 레벨
                                            answer_text = output.get("answer", "")
                                            
                                            # answer 그룹 (dict인 경우)
                                            if not answer_text and "answer" in output:
                                                answer_group = output.get("answer", {})
                                                if isinstance(answer_group, dict):
                                                    answer_text = answer_group.get("answer", "")
                                                elif isinstance(answer_group, str):
                                                    answer_text = answer_group
                                            
                                            # common 그룹
                                            if not answer_text and "common" in output:
                                                common = output.get("common", {})
                                                if isinstance(common, dict):
                                                    answer_text = common.get("answer", "")
                                        
                                        # answer 필드가 있고, JSON 형식이 아니면 확인
                                        if answer_text and isinstance(answer_text, str) and len(answer_text) > 0:
                                            # JSON 형식 필터링
                                            answer_stripped = answer_text.strip()
                                            is_json_answer = (
                                                answer_stripped.startswith("{") or 
                                                answer_stripped.startswith("```json") or
                                                (answer_stripped.startswith("```") and "json" in answer_stripped[:20].lower())
                                            )
                                            
                                            if not is_json_answer:
                                                # 스트리밍된 답변과 비교하여 누락된 부분만 전송
                                                if len(answer_text) > len(full_answer):
                                                    missing_part = answer_text[len(full_answer):]
                                                    if missing_part:
                                                        if DEBUG_STREAM:
                                                            logger.debug(f"누락된 부분 전송: {len(missing_part)}자")
                                                        # 스트림 청크를 JSONL 형식으로 전송
                                                        stream_event = {
                                                            "type": "stream",
                                                            "content": missing_part,
                                                            "timestamp": datetime.now().isoformat()
                                                        }
                                                        yield json.dumps(stream_event, ensure_ascii=False) + "\n"
                                                        full_answer = answer_text
                                                        answer_found = True
                                                elif not answer_found:
                                                    # 스트리밍이 없었던 경우 전체 답변 전송
                                                    if DEBUG_STREAM:
                                                        logger.debug(f"전체 답변 전송 (스트리밍 없음): {len(answer_text)}자")
                                                    stream_event = {
                                                        "type": "stream",
                                                        "content": answer_text,
                                                        "timestamp": datetime.now().isoformat()
                                                    }
                                                    yield json.dumps(stream_event, ensure_ascii=False) + "\n"
                                                    full_answer = answer_text
                                                    answer_found = True
                                                else:
                                                    if DEBUG_STREAM:
                                                        logger.debug("스트리밍된 답변이 이미 있습니다.")
                                            else:
                                                if DEBUG_STREAM:
                                                    logger.debug("answer 필드가 JSON 형식이므로 무시합니다.")
                                        else:
                                            if DEBUG_STREAM:
                                                logger.debug("answer 필드를 찾을 수 없거나 비어있습니다.")
                            except (AttributeError, TypeError, KeyError) as e:
                                if DEBUG_STREAM:
                                    logger.debug(f"on_chain_end에서 answer 추출 실패: {e}")
                            
                            if not answer_found:
                                # 스트리밍 이벤트가 전혀 발생하지 않은 경우
                                if DEBUG_STREAM:
                                    logger.warning("스트리밍 이벤트가 발생하지 않았습니다.")
                                error_event = {
                                    "type": "final",
                                    "content": "[오류] 스트리밍 응답을 생성할 수 없습니다. 다시 시도해주세요.",
                                    "metadata": {"error": True},
                                    "timestamp": datetime.now().isoformat()
                                }
                                yield json.dumps(error_event, ensure_ascii=False) + "\n"
                                answer_found = True
                        elif node_name == "generate_answer_enhanced":
                            # generate_answer_enhanced 노드는 generate_and_validate_answer 내부에서 호출되므로
                            # 여기서는 플래그만 확인하고 answer는 generate_and_validate_answer에서 처리
                            if DEBUG_STREAM:
                                logger.debug(f"generate_answer_enhanced 노드 완료: {node_name}")
                
                # 스트리밍 완료 후 최종 확인 (DEBUG_STREAM이 true일 때만)
                if DEBUG_STREAM:
                    logger.info(f"스트리밍 이벤트 처리 완료: 총 {event_count}개 이벤트, LLM 스트리밍 이벤트 {llm_stream_count}개, 토큰 수신 {tokens_received}개")
                    logger.info(f"발생한 이벤트 타입: {sorted(event_types_seen)}")
                    logger.info(f"발생한 노드 이름 (답변 생성 관련): {[n for n in sorted(node_names_seen) if 'answer' in n.lower() or 'generate' in n.lower()]}")
                
                # 디버깅: 발생한 모든 이벤트 타입과 노드 이름 로깅 (DEBUG_STREAM이 true일 때만)
                if llm_stream_count == 0:
                    if DEBUG_STREAM:
                        logger.warning("⚠️ LLM 스트리밍 이벤트가 발생하지 않았습니다.")
                        logger.debug(f"발생한 모든 이벤트 타입: {sorted(event_types_seen)}")
                        logger.debug(f"발생한 모든 노드 이름: {sorted(node_names_seen)}")
                    # 답변 생성 관련 노드가 실행되었는지 확인
                    answer_nodes_executed = [n for n in sorted(node_names_seen) if 'answer' in n.lower() or 'generate' in n.lower()]
                    if answer_nodes_executed:
                        if DEBUG_STREAM:
                            logger.info(f"답변 생성 관련 노드 실행됨: {answer_nodes_executed}")
                    else:
                        if DEBUG_STREAM:
                            logger.warning("답변 생성 관련 노드가 실행되지 않았습니다.")
                
                if not answer_found:
                    # 스트리밍 이벤트에서 답변을 찾지 못한 경우
                    # process_message를 호출하면 중복 실행이므로, 최종 결과만 가져오는 방법 사용
                    if DEBUG_STREAM:
                        logger.warning(f"LLM 스트리밍 이벤트에서 답변을 찾지 못했습니다. (이벤트 수: {event_count}, LLM 스트리밍: {llm_stream_count})")
                        logger.info("최종 결과를 가져오기 위해 워크플로우를 다시 실행합니다...")
                    # 최종 결과만 가져오기 (중복 실행 방지)
                    try:
                        result = await self.process_message(message, session_id)
                        final_answer = result.get("answer", "")
                        if final_answer and len(final_answer) > len(full_answer):
                            # 누락된 부분만 전송
                            missing_part = final_answer[len(full_answer):]
                            if missing_part:
                                full_answer = final_answer
                                # 스트림 청크를 JSONL 형식으로 전송
                                stream_event = {
                                    "type": "stream",
                                    "content": missing_part,
                                    "timestamp": datetime.now().isoformat()
                                }
                                yield json.dumps(stream_event, ensure_ascii=False) + "\n"
                                answer_found = True
                                if DEBUG_STREAM:
                                    logger.info(f"최종 답변에서 누락된 부분 전송: {len(missing_part)}자")
                        elif final_answer:
                            # 전체 답변 전송 (full_answer가 비어있는 경우)
                            full_answer = final_answer
                            stream_event = {
                                "type": "stream",
                                "content": final_answer,
                                "timestamp": datetime.now().isoformat()
                            }
                            yield json.dumps(stream_event, ensure_ascii=False) + "\n"
                            answer_found = True
                            if DEBUG_STREAM:
                                logger.info(f"전체 답변 전송: {len(final_answer)}자")
                    except Exception as e:
                        if DEBUG_STREAM:
                            logger.error(f"최종 결과 가져오기 실패: {e}", exc_info=True)
                        # 에러 발생 시에도 최소한 에러 메시지 yield (스트림이 비어있지 않도록)
                        if not answer_found:
                            error_event = {
                                "type": "final",
                                "content": f"[오류] 답변을 생성할 수 없습니다: {str(e)}",
                                "metadata": {"error": True},
                                "timestamp": datetime.now().isoformat()
                            }
                            yield json.dumps(error_event, ensure_ascii=False) + "\n"
                            answer_found = True
            
            except Exception as stream_error:
                # astream_events 실패 시 astream으로 폴백
                if DEBUG_STREAM:
                    logger.warning(f"astream_events 실패, astream으로 폴백: {stream_error}")
                # stream_mode="updates" 사용 시 변경된 필드만 포함되므로 직접 확인 가능
                async for event in self.workflow_service.app.astream(initial_state, config, stream_mode="updates"):
                    for node_name, node_state in event.items():
                        if isinstance(node_state, dict):
                            answer = None
                            # answer 그룹이 변경되었는지 확인
                            if "answer" in node_state:
                                answer = node_state.get("answer", "")
                            # common 그룹에서 answer 확인 (변경된 경우에만 포함)
                            elif "common" in node_state and isinstance(node_state["common"], dict):
                                common = node_state["common"]
                                if "answer" in common:
                                    answer = common.get("answer", "")
                            
                            if answer and isinstance(answer, str) and len(answer) > len(full_answer):
                                new_part = answer[len(full_answer):]
                                if new_part:
                                    full_answer = answer
                                    # 스트림 청크를 JSONL 형식으로 전송
                                    stream_event = {
                                        "type": "stream",
                                        "content": new_part,
                                        "timestamp": datetime.now().isoformat()
                                    }
                                    yield json.dumps(stream_event, ensure_ascii=False) + "\n"
                                    answer_found = True
            
            # 완료 메타데이터 (답변이 없어도 완료 신호 전송)
            if full_answer:
                if DEBUG_STREAM:
                    logger.info(f"스트리밍 완료: {len(full_answer)}자, {tokens_received}개 토큰 수신")
                has_yielded = True
                # 최종 완료 이벤트를 JSONL 형식으로 전송
                final_event = {
                    "type": "final",
                    "content": full_answer,
                    "metadata": {
                        "tokens_received": tokens_received,
                        "length": len(full_answer),
                        "answer_found": answer_found
                    },
                    "timestamp": datetime.now().isoformat()
                }
                yield json.dumps(final_event, ensure_ascii=False) + "\n"
            else:
                if DEBUG_STREAM:
                    logger.warning("스트리밍 완료: 답변이 생성되지 않았습니다.")
                # 답변이 없어도 최소한 빈 응답을 yield하여 스트림이 비어있지 않도록 보장
                if not answer_found:
                    error_event = {
                        "type": "final",
                        "content": "[오류] 답변을 생성할 수 없습니다. 다시 시도해주세요.",
                        "metadata": {
                            "error": True,
                            "tokens_received": tokens_received
                        },
                        "timestamp": datetime.now().isoformat()
                    }
                    yield json.dumps(error_event, ensure_ascii=False) + "\n"
                    has_yielded = True
            
        except Exception as e:
            logger.error(f"Error in stream_message: {e}", exc_info=True)
            # 에러 발생 시 에러 메시지를 JSONL 형식으로 전송
            try:
                error_event = {
                    "type": "final",
                    "content": f"[오류] 스트리밍 처리 중 오류 발생: {str(e)}",
                    "metadata": {
                        "error": True,
                        "error_type": type(e).__name__
                    },
                    "timestamp": datetime.now().isoformat()
                }
                yield json.dumps(error_event, ensure_ascii=False) + "\n"
                has_yielded = True
            except Exception as yield_error:
                logger.error(f"Error yielding error message: {yield_error}")
                # yield 자체가 실패한 경우에도 최소한 빈 문자열이라도 yield 시도
                try:
                    fallback_event = {
                        "type": "final",
                        "content": "[오류] 스트리밍 처리 중 오류가 발생했습니다.",
                        "metadata": {"error": True},
                        "timestamp": datetime.now().isoformat()
                    }
                    yield json.dumps(fallback_event, ensure_ascii=False) + "\n"
                    # 스트림 종료를 보장하기 위해 추가 빈 줄 yield
                    yield "\n"
                    has_yielded = True
                except Exception:
                    pass
            finally:
                # 예외 발생 시에도 스트림이 제대로 종료되도록 보장
                # (이미 위에서 yield했으므로 여기서는 추가 처리 불필요)
                # 하지만 스트림이 비어있지 않도록 보장
                if not has_yielded:
                    try:
                        fallback_event = {
                            "type": "final",
                            "content": "[오류] 스트리밍 응답을 생성할 수 없습니다.",
                            "metadata": {"error": True},
                            "timestamp": datetime.now().isoformat()
                        }
                        yield json.dumps(fallback_event, ensure_ascii=False) + "\n"
                        has_yielded = True
                    except Exception as e:
                        logger.error(f"Error yielding fallback message in finally: {e}")
                
                # HTTP chunked encoding이 제대로 종료되도록 보장
                # 스트림 종료 신호: 빈 줄 2개 yield
                # 이는 ERR_INCOMPLETE_CHUNKED_ENCODING 오류를 방지
                try:
                    yield "\n\n"  # 스트림 종료 신호 (빈 줄 2개)
                except Exception:
                    pass
        
        # 최소한 하나의 yield가 없었으면 에러 메시지 전송 (스트림이 비어있지 않도록 보장)
        # (finally 블록에서 이미 처리했을 수 있으므로 중복 방지)
        if not has_yielded:
            try:
                fallback_event = {
                    "type": "final",
                    "content": "[오류] 스트리밍 응답을 생성할 수 없습니다.",
                    "metadata": {"error": True},
                    "timestamp": datetime.now().isoformat()
                }
                yield json.dumps(fallback_event, ensure_ascii=False) + "\n"
            except Exception as e:
                logger.error(f"Error yielding fallback message: {e}")
    
    def is_available(self) -> bool:
        """서비스 사용 가능 여부"""
        return self.workflow_service is not None


# 전역 서비스 인스턴스 (지연 초기화)
chat_service: Optional[ChatService] = None

def get_chat_service() -> ChatService:
    """ChatService 인스턴스 가져오기 (싱글톤 패턴)"""
    global chat_service
    if chat_service is None:
        try:
            logger.info("Initializing ChatService...")
            chat_service = ChatService()
            if chat_service.is_available():
                logger.info("✅ ChatService initialized successfully with workflow service")
            else:
                logger.warning("⚠️  ChatService initialized but workflow service is not available")
                logger.warning("   Check API server logs for initialization errors")
        except Exception as e:
            logger.error(f"Failed to initialize ChatService: {e}", exc_info=True)
            import traceback
            tb = traceback.format_exc()
            logger.error(f"Traceback:\n{tb}")
            # 실패해도 ChatService 인스턴스는 생성 (workflow_service가 None일 수 있음)
            chat_service = ChatService()
    return chat_service

# 모듈 import 시점에는 초기화하지 않음 (지연 초기화)
# 첫 요청 시 get_chat_service()를 통해 초기화
# 이렇게 하면 api/main.py에서 환경 변수를 먼저 로드한 후 초기화 가능
logger.info("ChatService module loaded. Will initialize on first request via get_chat_service().")

