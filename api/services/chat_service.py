"""
채팅 서비스 (lawfirm_langgraph 래퍼)
"""
import sys
import logging
from pathlib import Path
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
            # 환경 변수 확인
            google_api_key = os.getenv("GOOGLE_API_KEY", "")
            if not google_api_key:
                logger.warning("GOOGLE_API_KEY가 설정되지 않았습니다. 환경 변수를 확인하세요.")
                logger.warning("LangGraph는 Google API Key가 필요합니다.")
            
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
            return {
                "answer": "죄송합니다. 메시지 처리 중 오류가 발생했습니다.",
                "sources": [],
                "confidence": 0.0,
                "legal_references": [],
                "processing_steps": [f"오류: {str(e)}"],
                "session_id": session_id or "error",
                "processing_time": 0.0,
                "query_type": "error",
                "metadata": {"error": str(e)},
                "errors": [str(e)]
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
        if not self.workflow_service:
            yield "[오류] 서비스 초기화에 실패했습니다."
            return
        
        try:
            import uuid
            import asyncio
            
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
                yield "[오류] 질문이 제대로 전달되지 않았습니다. 다시 시도해주세요."
                return
            
            config = {"configurable": {"thread_id": session_id}}
            
            # 실제 토큰 스트리밍을 위한 변수
            full_answer = ""
            answer_found = False
            tokens_received = 0
            last_node_name = None
            executed_nodes = set()  # 실행된 노드 추적
            answer_generation_started = False  # 답변 생성 노드 시작 플래그
            
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
            try:
                # 실제 스트리밍 이벤트 처리
                # LangGraph 버전별 호환성: version 파라미터가 없을 수도 있음
                # wrapper 함수로 버전 호환성 처리
                async def get_stream_events():
                    """버전 호환성을 위한 스트리밍 이벤트 래퍼"""
                    try:
                        # version="v2" 시도 (LangGraph 최신 버전)
                        logger.info("스트리밍 시작: astream_events(version='v2') 사용")
                        async for event in self.workflow_service.app.astream_events(
                            initial_state, 
                            config,
                            version="v2"
                        ):
                            yield event
                    except (TypeError, AttributeError) as ve:
                        # version 파라미터가 지원되지 않는 경우 (구버전)
                        logger.debug(f"astream_events에서 version 파라미터 미지원: {ve}, 기본 버전 사용")
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
                    
                    # 이벤트 타입과 노드 이름 추적
                    event_types_seen.add(event_type)
                    if event_name:
                        node_names_seen.add(event_name)
                    
                    # 디버깅: 이벤트 타입 로깅 (처음 20개만)
                    if event_count <= 20:
                        logger.debug(f"스트리밍 이벤트 #{event_count}: type={event_type}, name={event_name}")
                    
                    # LLM 스트리밍 이벤트 감지 (답변 생성 노드에서만)
                    # LangGraph/LangChain 최신 버전에서는 on_chat_model_stream도 지원
                    elif event_type in ["on_llm_stream", "on_chat_model_stream"]:
                        # 답변 생성 노드가 시작되었는지 확인
                        if not answer_generation_started:
                            # 답변 생성 노드가 시작되지 않았으면 무시
                            llm_stream_count += 1
                            if llm_stream_count <= 5:
                                logger.debug(f"답변 생성 노드가 시작되지 않음: {event_name} (무시)")
                            continue
                        
                        llm_stream_count += 1
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
                            # last_node_name이 답변 생성 노드인지 확인
                            if last_node_name in ["generate_answer_enhanced", "generate_and_validate_answer"]:
                                is_answer_node = True
                            # 또는 executed_nodes에 답변 생성 노드가 포함되어 있고, 아직 완료되지 않았는지 확인
                            elif "generate_answer_enhanced" in executed_nodes or "generate_and_validate_answer" in executed_nodes:
                                # 답변 생성 노드가 실행 중이면 스트리밍
                                is_answer_node = True
                        
                        # 디버깅: 모든 스트리밍 이벤트 로깅 (처음 10개만)
                        if llm_stream_count <= 10:
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
                            if llm_stream_count <= 5:
                                logger.debug(f"답변 생성 노드가 아님: {event_name}, parent={parent_node_name} (무시)")
                            continue
                        
                        logger.info(f"✅ 답변 생성 노드에서 {event_type} 이벤트 감지: {event_name}, parent={parent_node_name}")
                        
                        if is_answer_node:
                            logger.debug(f"LLM 스트리밍 이벤트 감지: {event_name} (총 {llm_stream_count}개)")
                            # 토큰 추출 (다양한 이벤트 구조 지원)
                            chunk = None
                            event_data = event.get("data", {})
                            
                            try:
                                # 경우 1: LangChain 표준 형식 - data.chunk.content
                                if isinstance(event_data, dict):
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
                                    
                                    # JSON 형식 시작 패턴 감지
                                    is_json_output = False
                                    if chunk_stripped.startswith("{") or chunk_stripped.startswith("```json"):
                                        is_json_output = True
                                    elif chunk_stripped.startswith("```") and "json" in chunk_stripped[:20].lower():
                                        is_json_output = True
                                    
                                    # JSON 형식이면 무시 (중간 노드의 JSON 출력)
                                    if is_json_output:
                                        logger.debug(f"JSON 형식 출력 감지 및 무시: {chunk_stripped[:100]}...")
                                        continue
                                    
                                    # 공백 토큰도 포함 (실제 토큰 스트리밍)
                                    # 단, 완전히 빈 문자열은 제외
                                    if len(chunk) > 0:
                                        full_answer += chunk
                                        tokens_received += 1
                                        answer_found = True
                                        yield chunk
                                        
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
                                                yield missing_part
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
                                # 진행 상황 메시지는 특별한 형식으로 전송 (프론트엔드에서 구분 가능하도록)
                                yield f"[진행상황]{progress_message}\n"
                                executed_nodes.add(node_name)
                                logger.debug(f"진행 상황 메시지 전송: {progress_message}")
                        
                        # 답변 생성 노드 시작 시 플래그 설정
                        if node_name in ["generate_answer_enhanced", "generate_and_validate_answer"]:
                            answer_generation_started = True
                            if not answer_found:
                                yield "[진행상황]답변 생성 중...\n"
                                last_node_name = node_name
                                logger.debug(f"답변 생성 노드 시작: {node_name}, answer_generation_started=True")
                    
                    # 노드 완료 이벤트 (포맷팅된 답변은 사용하지 않음)
                    elif event_type == "on_chain_end":
                        # 포맷팅된 답변은 원본이 변경될 수 있으므로 사용하지 않음
                        # 스트리밍된 원시 답변만 사용
                        node_name = event.get("name", "")
                        if node_name in ["generate_answer_enhanced", "generate_and_validate_answer"]:
                            # 답변 생성 노드가 완료되면 플래그 해제
                            answer_generation_started = False
                            logger.debug(f"답변 생성 노드 완료: {node_name}, answer_generation_started=False")
                            
                            if not answer_found:
                                # 스트리밍 이벤트가 전혀 발생하지 않은 경우
                                logger.warning("스트리밍 이벤트가 발생하지 않았습니다. 포맷팅된 답변은 사용하지 않습니다.")
                                yield "[오류] 스트리밍 응답을 생성할 수 없습니다. 다시 시도해주세요."
                                answer_found = True
                            else:
                                # 스트리밍이 있었을 때: 포맷팅된 답변은 완전히 무시
                                logger.debug("스트리밍된 답변이 있습니다. 포맷팅된 답변은 무시됩니다.")
                
                # 스트리밍 완료 후 최종 확인
                logger.info(f"스트리밍 이벤트 처리 완료: 총 {event_count}개 이벤트, LLM 스트리밍 이벤트 {llm_stream_count}개, 토큰 수신 {tokens_received}개")
                logger.info(f"발생한 이벤트 타입: {sorted(event_types_seen)}")
                logger.info(f"발생한 노드 이름 (답변 생성 관련): {[n for n in sorted(node_names_seen) if 'answer' in n.lower() or 'generate' in n.lower()]}")
                
                # 디버깅: 발생한 모든 이벤트 타입과 노드 이름 로깅
                if llm_stream_count == 0:
                    logger.warning("⚠️ LLM 스트리밍 이벤트가 발생하지 않았습니다.")
                    logger.debug(f"발생한 모든 이벤트 타입: {sorted(event_types_seen)}")
                    logger.debug(f"발생한 모든 노드 이름: {sorted(node_names_seen)}")
                    # 답변 생성 관련 노드가 실행되었는지 확인
                    answer_nodes_executed = [n for n in sorted(node_names_seen) if 'answer' in n.lower() or 'generate' in n.lower()]
                    if answer_nodes_executed:
                        logger.info(f"답변 생성 관련 노드 실행됨: {answer_nodes_executed}")
                    else:
                        logger.warning("답변 생성 관련 노드가 실행되지 않았습니다.")
                
                if not answer_found:
                    # 스트리밍 이벤트에서 답변을 찾지 못한 경우
                    # process_message를 호출하면 중복 실행이므로, 최종 결과만 가져오는 방법 사용
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
                                chunk_size = 10
                                for i in range(0, len(missing_part), chunk_size):
                                    chunk = missing_part[i:i + chunk_size]
                                    if chunk.strip():
                                        yield chunk
                                        await asyncio.sleep(0.03)
                                logger.info(f"최종 답변에서 누락된 부분 전송: {len(missing_part)}자")
                    except Exception as e:
                        logger.error(f"최종 결과 가져오기 실패: {e}", exc_info=True)
            
            except Exception as stream_error:
                # astream_events 실패 시 astream으로 폴백
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
                                    chunk_size = 10
                                    for i in range(0, len(new_part), chunk_size):
                                        chunk = new_part[i:i + chunk_size]
                                        if chunk.strip():
                                            yield chunk
                                            await asyncio.sleep(0.05)
                                    answer_found = True
            
            # 완료 메타데이터 (답변이 없어도 완료 신호 전송)
            if full_answer:
                logger.info(f"스트리밍 완료: {len(full_answer)}자, {tokens_received}개 토큰 수신")
            else:
                logger.warning("스트리밍 완료: 답변이 생성되지 않았습니다.")
            # 완료 신호는 chat.py에서 처리하므로 여기서는 전송하지 않음
            # (중복 전송 방지 및 SSE 형식 일관성 유지)
            
        except Exception as e:
            logger.error(f"Error in stream_message: {e}", exc_info=True)
            # 에러 발생 시 에러 메시지만 전송 (완료 신호는 chat.py에서 처리)
            try:
                yield f"[오류] 스트리밍 처리 중 오류 발생: {str(e)}"
            except Exception as yield_error:
                logger.error(f"Error yielding error message: {yield_error}")
    
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

