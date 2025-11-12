"""
채팅 서비스 (lawfirm_langgraph 래퍼)
"""
import sys
import json
import logging
import asyncio
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

# 환경 변수 로드 (중앙 집중식 로더 사용)
try:
    from utils.env_loader import ensure_env_loaded
    ensure_env_loaded(project_root)
except ImportError as e:
    logging.warning(f"⚠️  Failed to load environment variables: {e}")
    logging.warning("   Make sure utils/env_loader.py exists in the project root")
except Exception as e:
    logging.warning(f"⚠️  Failed to load environment variables: {e}")

try:
    from lawfirm_langgraph.core.workflow.workflow_service import LangGraphWorkflowService
    from lawfirm_langgraph.config.langgraph_config import LangGraphConfig
    LANGGRAPH_AVAILABLE = True
except ImportError as e:
    LANGGRAPH_AVAILABLE = False
    logging.warning(f"LangGraph not available: {e}")

logger = logging.getLogger(__name__)

# 로거 레벨을 명시적으로 설정 (루트 로거 레벨과 동기화)
# 환경 변수에서 로그 레벨 읽기
import os  # noqa: E402
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
        
        # 스트리밍 설정 및 서비스 초기화
        from api.services.stream_config import StreamConfig
        from api.services.stream_event_processor import StreamEventProcessor
        from api.services.sources_extractor import SourcesExtractor
        from api.services.session_service import session_service
        from api.services.streaming.stream_handler import StreamHandler
        
        self.stream_config = StreamConfig.from_env()
        self.event_processor = StreamEventProcessor(config=self.stream_config)
        self.sources_extractor = SourcesExtractor(
            workflow_service=self.workflow_service,
            session_service=session_service
        )
        self.stream_handler = StreamHandler(
            workflow_service=self.workflow_service,
            sources_extractor=self.sources_extractor,
            extract_related_questions_fn=self._extract_related_questions_from_state
        )
        
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
        # 이벤트 프로세서 초기화
        self.event_processor.reset()
        
        has_yielded = False  # 최소한 하나의 yield가 있었는지 추적
        
        if not self.workflow_service:
            error_event = self._create_error_event("[오류] 서비스 초기화에 실패했습니다.")
            yield json.dumps(error_event, ensure_ascii=False) + "\n"
            has_yielded = True
            return
        
        try:
            import uuid
            
            # 세션 ID 생성
            if not session_id:
                session_id = str(uuid.uuid4())
            
            # 워크플로우 스트리밍 실행
            from lawfirm_langgraph.core.workflow.state.state_definitions import create_initial_legal_state
            
            # 로깅: message 값 확인
            logger.info(f"stream_message: 받은 message='{message[:100] if message else 'EMPTY'}...', length={len(message) if message else 0}")
            
            # message를 query로 사용 (create_initial_legal_state의 첫 번째 파라미터는 query)
            initial_state = create_initial_legal_state(message, session_id)
            
            
            initial_query = self._validate_and_augment_state(initial_state, message, session_id)
            if not initial_query:
                error_event = self._create_error_event("[오류] 질문이 제대로 전달되지 않았습니다. 다시 시도해주세요.")
                yield json.dumps(error_event, ensure_ascii=False) + "\n"
                return
            
            config = {"configurable": {"thread_id": session_id}}
            
            # 디버그 모드 확인
            DEBUG_STREAM = self.stream_config.debug_stream
            
            try:
                # 스트리밍 이벤트 처리
                event_count = 0
                llm_stream_count = 0
                event_types_seen = set()  # 본 이벤트 타입 추적 (디버깅용, 제한적 사용)
                node_names_seen = set()  # 본 노드 이름 추적 (디버깅용, 제한적 사용)
                
                # 관련 이벤트 타입 집합 (성능 최적화)
                RELEVANT_EVENT_TYPES = self.stream_config.relevant_event_types
                
                # 메모리 최적화: 이벤트 히스토리 크기 제한
                MAX_EVENT_HISTORY = self.stream_config.max_event_history
                
                async for event in self._get_stream_events(initial_state, config):
                    event_count += 1
                    # 이벤트 타입 확인
                    event_type = event.get("event", "")
                    event_name = event.get("name", "")
                    
                    # 관련 없는 이벤트는 즉시 건너뛰기 (성능 최적화 - 조기 종료)
                    if event_type not in RELEVANT_EVENT_TYPES:
                        continue
                    
                    # 디버깅 모드에서만 이벤트 추적 (메모리 최적화: 제한적 추적)
                    if DEBUG_STREAM and event_count <= MAX_EVENT_HISTORY:
                        event_types_seen.add(event_type)
                        if event_name:
                            node_names_seen.add(event_name)
                        if event_count <= 20:
                            logger.debug(f"처리할 이벤트 #{event_count}: type={event_type}, name={event_name}")
                    
                    # StreamEventProcessor를 사용하여 이벤트 처리
                    stream_event = self.event_processor.process_stream_event(event)
                    if stream_event:
                        yield json.dumps(stream_event, ensure_ascii=False) + "\n"
                        has_yielded = True
                        if stream_event.get("type") == "stream":
                            llm_stream_count += 1
                
                # event_processor에서 상태 가져오기
                full_answer = self.event_processor.full_answer
                answer_found = self.event_processor.answer_found
                tokens_received = self.event_processor.tokens_received
                
                # 스트리밍 완료 후 최종 확인 (DEBUG_STREAM이 true일 때만)
                if DEBUG_STREAM:
                    logger.info(f"스트리밍 이벤트 처리 완료: 총 {event_count}개 이벤트, LLM 스트리밍 이벤트 {llm_stream_count}개, 토큰 수신 {tokens_received}개")
                
                # LLM 스트리밍 이벤트가 없을 때만 경고 (프로덕션에서도)
                if llm_stream_count == 0:
                    if DEBUG_STREAM:
                        logger.warning("⚠️ LLM 스트리밍 이벤트가 발생하지 않았습니다.")
                        logger.debug(f"발생한 모든 이벤트 타입: {sorted(event_types_seen)}")
                        logger.debug(f"발생한 모든 노드 이름: {sorted(node_names_seen)}")
                
                if not answer_found:
                    missing_event = await self._handle_missing_answer(message, session_id, full_answer)
                    if missing_event:
                        yield json.dumps(missing_event, ensure_ascii=False) + "\n"
                        has_yielded = True
                        if missing_event.get("type") == "stream":
                            self.event_processor.answer_found = True
            
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
                            
                            if answer and isinstance(answer, str):
                                current_full_answer = self.event_processor.full_answer
                                if len(answer) > len(current_full_answer):
                                    new_part = answer[len(current_full_answer):]
                                    if new_part:
                                        self.event_processor.full_answer = answer
                                        # 스트림 청크를 JSONL 형식으로 전송
                                        stream_event = {
                                            "type": "stream",
                                            "content": new_part,
                                            "timestamp": datetime.now().isoformat()
                                        }
                                        yield json.dumps(stream_event, ensure_ascii=False) + "\n"
                                        self.event_processor.answer_found = True
            
            sources_data = await self._extract_sources_from_state(session_id) if session_id else {}
            final_sources = sources_data.get("sources", [])
            final_legal_references = sources_data.get("legal_references", [])
            final_sources_detail = sources_data.get("sources_detail", [])
            final_related_questions = sources_data.get("related_questions", [])
            
            # event_processor에서 full_answer 가져오기
            full_answer = self.event_processor.full_answer
            answer_found = self.event_processor.answer_found
            tokens_received = self.event_processor.tokens_received
            
            if full_answer:
                has_yielded = True
                
                # 토큰 제한 확인
                MAX_OUTPUT_TOKENS = self.stream_config.max_output_tokens
                should_split = tokens_received >= MAX_OUTPUT_TOKENS
                
                import uuid
                
                # 메시지 ID 생성 (chat.py에서 저장 시 사용)
                message_id = str(uuid.uuid4())
                
                if not final_sources and not final_legal_references and not final_sources_detail:
                    re_extracted = await self._re_extract_sources_before_final(session_id, {})
                    if re_extracted.get("sources"):
                        final_sources = re_extracted["sources"]
                    if re_extracted.get("legal_references"):
                        final_legal_references = re_extracted["legal_references"]
                    if re_extracted.get("sources_detail"):
                        final_sources_detail = re_extracted["sources_detail"]
                    if re_extracted.get("related_questions"):
                        final_related_questions = re_extracted["related_questions"]
                
                if should_split:
                    from api.services.answer_splitter import AnswerSplitter
                    splitter = AnswerSplitter(chunk_size=self.stream_config.chunk_size)
                    chunks = splitter.split_answer(full_answer)
                    
                    content = chunks[0].content if chunks else full_answer
                    final_event = self._create_final_event(
                        content, tokens_received, answer_found,
                        final_sources, final_legal_references,
                        final_sources_detail, final_related_questions,
                        message_id, needs_continuation=bool(chunks)
                    )
                    yield json.dumps(final_event, ensure_ascii=False) + "\n"
                else:
                    final_event = self._create_final_event(
                        full_answer, tokens_received, answer_found,
                        final_sources, final_legal_references,
                        final_sources_detail, final_related_questions,
                        message_id
                    )
                    yield json.dumps(final_event, ensure_ascii=False) + "\n"
            else:
                if DEBUG_STREAM:
                    logger.warning("스트리밍 완료: 답변이 생성되지 않았습니다.")
                if not answer_found:
                    error_event = self._create_error_event("[오류] 답변을 생성할 수 없습니다. 다시 시도해주세요.")
                    error_event["metadata"]["tokens_received"] = tokens_received
                    yield json.dumps(error_event, ensure_ascii=False) + "\n"
                    has_yielded = True
            
        except Exception as e:
            logger.error(f"Error in stream_message: {e}", exc_info=True)
            try:
                error_event = self._create_error_event(
                    f"[오류] 스트리밍 처리 중 오류 발생: {str(e)}",
                    error_type=type(e).__name__
                )
                yield json.dumps(error_event, ensure_ascii=False) + "\n"
                has_yielded = True
            except Exception as yield_error:
                logger.error(f"Error yielding error message: {yield_error}")
                try:
                    fallback_event = self._create_error_event("[오류] 스트리밍 처리 중 오류가 발생했습니다.")
                    yield json.dumps(fallback_event, ensure_ascii=False) + "\n"
                    has_yielded = True
                except Exception:
                    pass
        # finally 블록 제거: finally에서 yield를 하면 제너레이터가 제대로 종료되지 않아
        # ERR_INCOMPLETE_CHUNKED_ENCODING 오류가 발생할 수 있음
        # 스트림 종료는 FastAPI StreamingResponse가 자동으로 처리
    
    async def get_sources_from_session(
        self,
        session_id: str,
        message_id: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        세션의 최종 state에서 sources 정보 가져오기
        
        Args:
            session_id: 세션 ID
            message_id: 메시지 ID (선택사항, 해당 메시지의 metadata에서 sources 가져오기)
        
        Returns:
            sources, legal_references, sources_detail 딕셔너리
        """
        # 먼저 메시지의 metadata에서 sources를 가져오기 시도
        result = await self.sources_extractor.extract_from_message_metadata(session_id, message_id)
        
        # 없으면 state에서 가져오기
        if not any(result.values()):
            result = await self.sources_extractor.extract_from_state(session_id)
        
        return result
    
    async def stream_final_answer(
        self,
        message: str,
        session_id: Optional[str] = None
    ) -> AsyncGenerator[str, None]:
        """
        LangGraph의 astream_events()를 사용하여 
        generate_and_validate_answer 노드의 LLM 응답만 스트림 형태로 전달
        
        예제 코드 참고:
        async for event in compiled_graph.astream_events({"topic": "AI"}):
            if event["event"] == "on_llm_stream" and event["name"] == "generate_response":
                yield f"data: {json.dumps({'token': data})}\n\n"
        """
        if not self.workflow_service:
            error_event = {
                "type": "error",
                "content": "[오류] 서비스 초기화에 실패했습니다.",
                "timestamp": datetime.now().isoformat()
            }
            yield f"data: {json.dumps(error_event, ensure_ascii=False)}\n\n"
            return
        
        try:
            import uuid
            
            # 세션 ID 생성
            if not session_id:
                session_id = str(uuid.uuid4())
            
            from lawfirm_langgraph.core.workflow.state.state_definitions import create_initial_legal_state
            initial_state = create_initial_legal_state(message, session_id)
            
            initial_query = self._validate_and_augment_state(initial_state, message, session_id)
            if not initial_query:
                error_event = {
                    "type": "error",
                    "content": "[오류] 질문이 제대로 전달되지 않았습니다.",
                    "timestamp": datetime.now().isoformat()
                }
                yield f"data: {json.dumps(error_event, ensure_ascii=False)}\n\n"
                return
            
            # 콜백 핸들러 및 큐 생성
            callback_queue = asyncio.Queue()
            callback_handler = None
            if self.workflow_service and hasattr(self.workflow_service, 'create_streaming_callback_handler'):
                callback_handler = self.workflow_service.create_streaming_callback_handler(queue=callback_queue)
                if callback_handler:
                    logger.info("[stream_final_answer] ✅ StreamingCallbackHandler created and ready")
                else:
                    logger.warning("[stream_final_answer] ⚠️ Failed to create StreamingCallbackHandler")
            
            # config에 콜백 포함
            config = {"configurable": {"thread_id": session_id}}
            if callback_handler:
                config = self.workflow_service.get_config_with_callbacks(
                    session_id=session_id,
                    callbacks=[callback_handler]
                )
                logger.info(f"[stream_final_answer] ✅ Callbacks added to config: {len(config.get('callbacks', []))} callback(s)")
            else:
                logger.warning("[stream_final_answer] ⚠️ No callback handler, streaming may not work optimally")
            
            # 진행 상황 표시
            progress_event = {
                "type": "progress",
                "content": "답변 생성 중...",
                "timestamp": datetime.now().isoformat()
            }
            yield f"data: {json.dumps(progress_event, ensure_ascii=False)}\n\n"
            
            # LangGraph의 astream_events() 사용
            # generate_and_validate_answer 노드의 LLM 스트림만 필터링
            # StreamEventProcessor의 로직을 참고하여 구현
            answer_generation_started = False
            last_node_name = None
            event_count = 0
            stream_event_count = 0
            on_llm_stream_count = 0
            on_chat_model_stream_count = 0
            on_chain_stream_count = 0
            full_answer = ""  # 스트리밍된 답변 추적
            callback_chunks_received = 0  # 콜백에서 받은 청크 수
            processed_callback_chunks = set()  # 이미 처리한 콜백 청크 추적
            
            # 콜백 큐의 청크를 저장할 출력 큐 생성
            chunk_output_queue = asyncio.Queue() if callback_queue else None
            
            # 콜백 큐 모니터링을 별도 태스크로 실행
            callback_monitoring_active = True
            
            async def monitor_callback_queue():
                """콜백 큐를 지속적으로 모니터링하여 청크를 출력 큐에 넣기"""
                nonlocal callback_monitoring_active
                while callback_monitoring_active:
                    try:
                        if callback_queue and chunk_output_queue:
                            try:
                                # 큐에서 청크 가져오기 (타임아웃 설정)
                                chunk_data = await asyncio.wait_for(
                                    callback_queue.get(),
                                    timeout=0.1
                                )
                                if chunk_data and chunk_data.get("type") == "chunk":
                                    # 출력 큐에 넣기
                                    await chunk_output_queue.put(chunk_data)
                            except asyncio.TimeoutError:
                                # 타임아웃은 정상 (큐가 비어있음)
                                await asyncio.sleep(0.01)  # 짧은 대기
                                continue
                            except asyncio.QueueEmpty:
                                await asyncio.sleep(0.01)  # 짧은 대기
                                continue
                        else:
                            await asyncio.sleep(0.1)
                    except Exception as e:
                        logger.debug(f"[stream_final_answer] Error in callback queue monitoring: {e}")
                        await asyncio.sleep(0.1)
            
            # 콜백 큐 모니터링 태스크 시작
            callback_task = None
            if callback_queue and chunk_output_queue:
                callback_task = asyncio.create_task(monitor_callback_queue())
                logger.info("[stream_final_answer] ✅ Callback queue monitoring task started")
            
            # astream_events와 콜백 큐 모니터링 병행
            try:
                async for event in self.workflow_service.app.astream_events(
                    initial_state,
                    config,
                    version="v2"
                ):
                    # 콜백 출력 큐에서 청크 확인 (non-blocking) - answer_generation_started와 무관하게 처리
                    if chunk_output_queue:
                        try:
                            while True:
                                try:
                                    chunk_data = chunk_output_queue.get_nowait()
                                    if chunk_data and chunk_data.get("type") == "chunk":
                                        content = chunk_data.get("content", "")
                                        chunk_index = chunk_data.get("chunk_index", 0)
                                        chunk_key = f"{chunk_index}_{content[:10]}"
                                        
                                        # 중복 처리 방지
                                        if chunk_key not in processed_callback_chunks and content:
                                            processed_callback_chunks.add(chunk_key)
                                            callback_chunks_received += 1
                                            full_answer += content
                                            
                                            # 콜백에서 받은 청크는 즉시 전달 (플래그와 무관)
                                            stream_event = {
                                                "type": "stream",
                                                "content": content,
                                                "source": "callback",
                                                "timestamp": datetime.now().isoformat()
                                            }
                                            yield f"data: {json.dumps(stream_event, ensure_ascii=False)}\n\n"
                                            
                                            # 처음 10개 청크만 상세 로깅
                                            if callback_chunks_received <= 10:
                                                logger.info(
                                                    f"[stream_final_answer] ✅ Callback chunk #{callback_chunks_received}: "
                                                    f"length={len(content)}, content={content[:50]}..."
                                                )
                                except asyncio.QueueEmpty:
                                    break
                        except Exception as e:
                            logger.debug(f"[stream_final_answer] Error checking callback output queue: {e}")
                    
                    event_count += 1
                    event_type = event.get("event", "")
                    event_name = event.get("name", "")
                    event_parent = event.get("parent", {})
                    
                    # on_llm_stream 이벤트 발생 추적
                    if event_type == "on_llm_stream":
                        on_llm_stream_count += 1
                        if on_llm_stream_count <= 10:  # 처음 10개만 상세 로깅
                            logger.debug(
                                f"[stream_final_answer] on_llm_stream 이벤트 #{on_llm_stream_count}: "
                                f"name={event_name}, "
                                f"parent={event_parent.get('name', '') if isinstance(event_parent, dict) else ''}"
                            )
                    elif event_type == "on_chat_model_stream":
                        on_chat_model_stream_count += 1
                        if on_chat_model_stream_count <= 10:
                            logger.debug(
                                f"[stream_final_answer] on_chat_model_stream 이벤트 #{on_chat_model_stream_count}: "
                                f"name={event_name}, "
                                f"parent={event_parent.get('name', '') if isinstance(event_parent, dict) else ''}"
                            )
                        
                        # 제거: on_chat_model_stream 이벤트를 콜백 큐에 추가하는 로직
                        # astream_events()에서 이미 직접 처리하므로(line 712-745) 중복 방지를 위해 제거
                        # 콜백 핸들러는 LLM의 stream() 호출 시 자동으로 on_llm_stream을 호출하므로
                        # 별도로 콜백 큐에 추가할 필요 없음
                    elif event_type == "on_chain_stream":
                        on_chain_stream_count += 1
                        if on_chain_stream_count <= 5:
                            logger.debug(
                                f"[stream_final_answer] on_chain_stream 이벤트 #{on_chain_stream_count}: "
                                f"name={event_name}, "
                                f"parent={event_parent.get('name', '') if isinstance(event_parent, dict) else ''}"
                            )
                    
                    # on_chain_start: 답변 생성 노드 시작 감지
                    if event_type == "on_chain_start":
                        node_name = event_name
                        if node_name in ["generate_answer_stream", "generate_answer_enhanced", "generate_and_validate_answer"]:
                            answer_generation_started = True
                            last_node_name = node_name
                            logger.debug(f"[stream_final_answer] 답변 생성 노드 시작: {node_name}")
                    
                    # on_chain_end: 답변 생성 노드 완료
                    elif event_type == "on_chain_end":
                        node_name = event_name
                        if node_name in ["generate_answer_stream", "generate_and_validate_answer"]:
                            answer_generation_started = False
                            logger.debug(f"[stream_final_answer] 답변 생성 노드 완료: {node_name}")
                    
                    # LLM 스트리밍 이벤트 처리 (on_llm_stream, on_chat_model_stream만 처리)
                    # on_chain_stream은 제외 (전체 답변을 한 번에 전달하므로 토큰 단위 스트리밍 불가)
                    elif event_type in ["on_llm_stream", "on_chat_model_stream"]:
                        # on_llm_stream 이벤트 발생 로깅
                        logger.debug(
                            f"[stream_final_answer] on_llm_stream 이벤트 발생: "
                            f"type={event_type}, name={event_name}, "
                            f"parent={event_parent.get('name', '') if isinstance(event_parent, dict) else ''}, "
                            f"last_node={last_node_name}, started={answer_generation_started}"
                        )
                        
                        # 답변 생성 노드가 시작되지 않았으면 건너뛰기
                        if not answer_generation_started:
                            logger.debug(f"[stream_final_answer] 답변 생성 노드가 시작되지 않아 건너뜀: {event_name}")
                            continue
                        
                        if self._is_target_node(event_name, event_parent, last_node_name):
                            logger.debug(f"[stream_final_answer] 타겟 노드 확인됨: {event_name}, 토큰 추출 시작")
                            event_data = event.get("data", {})
                            token = self._extract_token_from_event(event_data)
                            
                            if token:
                                stream_event_count += 1
                                full_answer += token  # full_answer 추적
                                logger.debug(
                                    f"[stream_final_answer] 토큰 전송: "
                                    f"token_length={len(token)}, "
                                    f"token_preview={token[:50]}..., "
                                    f"stream_event_count={stream_event_count}"
                                )
                                stream_event = {
                                    "type": "stream",
                                    "content": token,
                                    "timestamp": datetime.now().isoformat()
                                }
                                yield f"data: {json.dumps(stream_event, ensure_ascii=False)}\n\n"
                            else:
                                logger.debug(
                                    f"[stream_final_answer] 토큰 추출 실패: "
                                    f"token={token}, "
                                    f"event_data_keys={list(event_data.keys()) if isinstance(event_data, dict) else []}"
                                )
                        else:
                            # 디버깅: 필터링되지 않은 이벤트 로깅
                            logger.debug(
                                f"[stream_final_answer] 타겟 노드가 아님 (필터링됨): "
                                f"type={event_type}, name={event_name}, "
                                f"parent={event_parent.get('name', '') if isinstance(event_parent, dict) else ''}, "
                                f"last_node={last_node_name}, started={answer_generation_started}"
                            )
                    
                    # 답변 생성 노드 완료 시점 (generate_answer_stream 또는 generate_and_validate_answer)
                    if event_type == "on_chain_end" and event_name in ["generate_answer_stream", "generate_and_validate_answer"]:
                        # 콜백 큐에 남은 청크 처리
                        if chunk_output_queue:
                            try:
                                # 큐에 남은 모든 청크 처리
                                while True:
                                    try:
                                        chunk_data = chunk_output_queue.get_nowait()
                                        if chunk_data and chunk_data.get("type") == "chunk":
                                            content = chunk_data.get("content", "")
                                            chunk_index = chunk_data.get("chunk_index", 0)
                                            chunk_key = f"{chunk_index}_{content[:10]}"
                                            
                                            if chunk_key not in processed_callback_chunks and content:
                                                processed_callback_chunks.add(chunk_key)
                                                callback_chunks_received += 1
                                                full_answer += content
                                                
                                                stream_event = {
                                                    "type": "stream",
                                                    "content": content,
                                                    "source": "callback",
                                                    "timestamp": datetime.now().isoformat()
                                                }
                                                yield f"data: {json.dumps(stream_event, ensure_ascii=False)}\n\n"
                                    except asyncio.QueueEmpty:
                                        break
                            except Exception as e:
                                logger.debug(f"[stream_final_answer] Error processing remaining callback chunks: {e}")
                        
                        logger.info(
                            f"[stream_final_answer] 스트리밍 완료: "
                            f"총 {event_count}개 이벤트, "
                            f"스트림 이벤트 {stream_event_count}개, "
                            f"콜백 청크 {callback_chunks_received}개, "
                            f"on_llm_stream={on_llm_stream_count}개, "
                            f"on_chat_model_stream={on_chat_model_stream_count}개, "
                            f"on_chain_stream={on_chain_stream_count}개, "
                            f"full_answer_length={len(full_answer)}"
                        )
                        
                        # State가 완전히 저장될 때까지 약간의 지연 (타이밍 문제 해결)
                        await asyncio.sleep(0.2)
                        
                        # 최종 완료 이벤트 (metadata 포함)
                        try:
                            # State 가져오기 (재시도 최소화하여 스트림 블로킹 방지)
                            final_state = None
                            state_values = None
                            
                            try:
                                # 즉시 시도 (타임아웃 2초)
                                final_state = await asyncio.wait_for(
                                    self.workflow_service.app.aget_state(config),
                                    timeout=2.0
                                )
                                if final_state and final_state.values:
                                    state_values = final_state.values
                                    logger.debug(f"[stream_final_answer] State retrieved: answer_length={len(state_values.get('answer', ''))}, full_answer_length={len(full_answer)}")
                            except asyncio.TimeoutError:
                                logger.warning(f"[stream_final_answer] Timeout getting state, using empty metadata")
                            except Exception as e:
                                logger.warning(f"[stream_final_answer] Error getting state: {e}")
                            
                            if final_state and state_values:
                                sources = state_values.get("sources", [])
                                legal_references = state_values.get("legal_references", [])
                                sources_detail = state_values.get("sources_detail", [])
                                
                                if not sources and not sources_detail:
                                    retrieved_docs = state_values.get("retrieved_docs", [])
                                    if retrieved_docs and hasattr(self, 'sources_extractor') and self.sources_extractor:
                                        try:
                                            sources_data = self.sources_extractor._extract_sources(state_values)
                                            legal_references_data = self.sources_extractor._extract_legal_references(state_values)
                                            sources_detail_data = self.sources_extractor._extract_sources_detail(state_values)
                                            
                                            if sources_detail_data:
                                                sources_detail = sources_detail_data
                                                state_values["sources_detail"] = sources_detail_data
                                            if sources_data:
                                                sources = sources_data
                                                state_values["sources"] = sources_data
                                            if legal_references_data:
                                                legal_references = legal_references_data
                                        except Exception as e:
                                            logger.warning(f"[stream_final_answer] Failed to extract sources from retrieved_docs: {e}", exc_info=True)
                                
                                related_questions = await self._extract_related_questions_from_state(
                                    state_values, initial_state, message, full_answer, session_id
                                )
                                
                                llm_validation_result = state_values.get("metadata", {}).get("llm_validation_result", {})
                                
                                final_metadata = {
                                    "sources": sources,
                                    "legal_references": legal_references,
                                    "sources_detail": sources_detail,
                                    "related_questions": related_questions,
                                    "llm_validation": llm_validation_result if llm_validation_result else None
                                }
                            else:
                                final_metadata = {}
                        except Exception as e:
                            logger.error(f"Error getting final state: {e}", exc_info=True)
                            final_metadata = {}
                        
                        # 품질 검증 결과가 있으면 검증 이벤트 전송
                        if final_metadata.get("llm_validation"):
                            validation_result = final_metadata["llm_validation"]
                            validation_event = {
                                "type": "validation",
                                "content": "답변 품질 검증 완료",
                                "metadata": {
                                    "quality_score": validation_result.get("quality_score", 0.0),
                                    "is_valid": validation_result.get("is_valid", False),
                                    "needs_regeneration": validation_result.get("needs_regeneration", False),
                                    "regeneration_reason": validation_result.get("regeneration_reason"),
                                    "issues": validation_result.get("issues", []),
                                    "strengths": validation_result.get("strengths", [])
                                },
                                "timestamp": datetime.now().isoformat()
                            }
                            yield f"data: {json.dumps(validation_event, ensure_ascii=False)}\n\n"
                        
                        # 최종 답변 이벤트
                        final_event = {
                            "type": "final",
                            "content": "",  # 스트림으로 이미 전송됨
                            "metadata": final_metadata or {},
                            "timestamp": datetime.now().isoformat()
                        }
                        yield f"data: {json.dumps(final_event, ensure_ascii=False)}\n\n"
                        
                        # 완료 이벤트 (최종 답변과 메타데이터 포함)
                        done_event = {
                            "type": "done",
                            "content": full_answer,  # 최종 답변 포함
                            "metadata": final_metadata or {},
                            "timestamp": datetime.now().isoformat()
                        }
                        yield f"data: {json.dumps(done_event, ensure_ascii=False)}\n\n"
                        break  # 스트리밍 종료
                
            finally:
                # 콜백 모니터링 태스크 종료
                callback_monitoring_active = False
                if callback_task:
                    callback_task.cancel()
                    try:
                        await callback_task
                    except asyncio.CancelledError:
                        pass
                
                # 콜백 핸들러 통계 로깅
                if callback_handler and hasattr(callback_handler, 'get_stats'):
                    stats = callback_handler.get_stats()
                    logger.info(
                        f"[stream_final_answer] Callback stats: "
                        f"chunks={callback_chunks_received}, "
                        f"total_chunks={stats.get('total_chunks', 0)}, "
                        f"streaming_active={stats.get('streaming_active', False)}"
                    )
        
        except GeneratorExit:
            # 클라이언트가 연결을 끊은 경우 정상 종료
            logger.debug("[stream_final_answer] Client disconnected, closing stream")
            return
        except Exception as e:
            logger.error(f"Stream error: {e}", exc_info=True)
            try:
                error_event = {
                    "type": "error",
                    "content": f"[오류] 스트리밍 중 오류가 발생했습니다: {str(e)}",
                    "timestamp": datetime.now().isoformat()
                }
                yield f"data: {json.dumps(error_event, ensure_ascii=False)}\n\n"
            except GeneratorExit:
                # yield 중 클라이언트가 연결을 끊은 경우
                logger.debug("[stream_final_answer] Client disconnected during error handling")
                return
            except Exception as yield_error:
                # yield 자체가 실패한 경우 (스트림이 이미 닫힘)
                logger.error(f"Failed to yield error event: {yield_error}")
                return
    
    
    def _create_error_event(self, content: str, error_type: Optional[str] = None) -> Dict[str, Any]:
        """에러 이벤트 생성"""
        event = {
            "type": "final",
            "content": content,
            "metadata": {"error": True},
            "timestamp": datetime.now().isoformat()
        }
        if error_type:
            event["metadata"]["error_type"] = error_type
        return event
    
    def _validate_and_augment_state(self, initial_state: Dict[str, Any], message: str, session_id: str) -> Optional[str]:
        """상태 검증 및 보강"""
        if "input" not in initial_state:
            initial_state["input"] = {}
        if not initial_state["input"].get("query"):
            initial_state["input"]["query"] = message
        if not initial_state["input"].get("session_id"):
            initial_state["input"]["session_id"] = session_id
        
        if not initial_state.get("query"):
            initial_state["query"] = message
        if not initial_state.get("session_id"):
            initial_state["session_id"] = session_id
        
        initial_query = initial_state.get("input", {}).get("query") or initial_state.get("query")
        if not initial_query or not str(initial_query).strip():
            logger.error(f"Initial state query is empty! Input message was: '{message[:50]}...'")
            return None
        return initial_query
    
    async def _get_stream_events(self, initial_state: Dict[str, Any], config: Dict[str, Any]) -> AsyncGenerator[Dict[str, Any], None]:
        """스트리밍 이벤트 가져오기 (버전 호환성 처리)"""
        DEBUG_STREAM = self.stream_config.debug_stream
        
        try:
            if DEBUG_STREAM:
                logger.info("스트리밍 시작: astream_events(version='v2') 사용")
            try:
                try:
                    async for event in self.workflow_service.app.astream_events(
                        initial_state, 
                        config,
                        version="v2",
                        include_names=["generate_answer_stream", "generate_answer_enhanced", "generate_and_validate_answer"]
                    ):
                        yield event
                except (TypeError, AttributeError):
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
                        async for event in self.workflow_service.app.astream_events(
                            initial_state, 
                            config,
                            version="v2"
                        ):
                            yield event
            except (TypeError, AttributeError):
                async for event in self.workflow_service.app.astream_events(
                    initial_state, 
                    config,
                    version="v2"
                ):
                    yield event
        except (TypeError, AttributeError) as ve:
            logger.debug(f"astream_events에서 version 파라미터 미지원: {ve}, 기본 버전 사용")
            if DEBUG_STREAM:
                logger.info("스트리밍 시작: astream_events() 사용 (기본 버전)")
            async for event in self.workflow_service.app.astream_events(
                initial_state, 
                config
            ):
                yield event
    
    async def _extract_sources_from_state(self, session_id: str, timeout: float = 2.0) -> Dict[str, Any]:
        """State에서 sources 추출"""
        result = {
            "sources": [],
            "legal_references": [],
            "sources_detail": [],
            "related_questions": []
        }
        
        try:
            sources_data = await asyncio.wait_for(
                self.sources_extractor.extract_from_state(session_id),
                timeout=timeout
            )
            result["sources"] = sources_data.get("sources", [])
            result["legal_references"] = sources_data.get("legal_references", [])
            result["sources_detail"] = sources_data.get("sources_detail", [])
            result["related_questions"] = sources_data.get("related_questions", [])
        except asyncio.TimeoutError:
            logger.warning(f"Timeout getting sources from LangGraph state for session {session_id}")
        except Exception as e:
            logger.warning(f"Failed to get sources from LangGraph state: {e}")
        
        return result
    
    async def _re_extract_sources_before_final(self, session_id: str, state_values: Dict[str, Any]) -> Dict[str, Any]:
        """최종 이벤트 전 sources 재추출"""
        result = {
            "sources": [],
            "legal_references": [],
            "sources_detail": [],
            "related_questions": []
        }
        
        if not session_id or not self.workflow_service or not self.workflow_service.app:
            return result
        
        try:
            config = {"configurable": {"thread_id": session_id}}
            final_state = await asyncio.wait_for(
                self.workflow_service.app.aget_state(config),
                timeout=2.0
            )
            
            if final_state and final_state.values:
                state_values = final_state.values
                sources_data = self.sources_extractor._extract_sources(state_values)
                legal_references_data = self.sources_extractor._extract_legal_references(state_values)
                sources_detail_data = self.sources_extractor._extract_sources_detail(state_values)
                related_questions_data = self.sources_extractor._extract_related_questions(state_values)
                
                if sources_data:
                    result["sources"] = sources_data
                if legal_references_data:
                    result["legal_references"] = legal_references_data
                if sources_detail_data:
                    result["sources_detail"] = sources_detail_data
                if related_questions_data:
                    result["related_questions"] = related_questions_data
                
                logger.info(f"Re-extracted sources before final event: {len(result['sources'])} sources, {len(result['legal_references'])} legal_references, {len(result['sources_detail'])} sources_detail, {len(result['related_questions'])} related_questions")
        except asyncio.TimeoutError:
            logger.warning("Timeout re-getting sources before final event")
        except Exception as e:
            logger.warning(f"Failed to re-get sources before final event: {e}")
        
        return result
    
    def _create_final_event(
        self,
        content: str,
        tokens_received: int,
        answer_found: bool,
        sources: list,
        legal_references: list,
        sources_detail: list,
        related_questions: list,
        message_id: str,
        needs_continuation: bool = False
    ) -> Dict[str, Any]:
        """최종 이벤트 생성"""
        metadata = {
            "tokens_received": tokens_received,
            "length": len(content),
            "answer_found": answer_found,
            "sources": sources,
            "legal_references": legal_references,
            "sources_detail": sources_detail,
            "related_questions": related_questions,
            "message_id": message_id
        }
        if needs_continuation:
            metadata["needs_continuation"] = True
        
        return {
            "type": "final",
            "content": content,
            "metadata": metadata,
            "timestamp": datetime.now().isoformat()
        }
    
    async def _handle_missing_answer(self, message: str, session_id: str, full_answer: str) -> Optional[Dict[str, Any]]:
        """답변을 찾지 못한 경우 처리"""
        DEBUG_STREAM = self.stream_config.debug_stream
        
        if DEBUG_STREAM:
            logger.warning("LLM 스트리밍 이벤트에서 답변을 찾지 못했습니다.")
            logger.info("최종 결과를 가져오기 위해 워크플로우를 다시 실행합니다...")
        
        try:
            result = await self.process_message(message, session_id)
            final_answer = result.get("answer", "")
            if final_answer and len(final_answer) > len(full_answer):
                missing_part = final_answer[len(full_answer):]
                if missing_part:
                    self.event_processor.full_answer = final_answer
                    return {
                        "type": "stream",
                        "content": missing_part,
                        "timestamp": datetime.now().isoformat()
                    }
            elif final_answer:
                self.event_processor.full_answer = final_answer
                return {
                    "type": "stream",
                    "content": final_answer,
                    "timestamp": datetime.now().isoformat()
                }
        except Exception as e:
            if DEBUG_STREAM:
                logger.error(f"최종 결과 가져오기 실패: {e}", exc_info=True)
            if not self.event_processor.answer_found:
                return self._create_error_event(f"[오류] 답변을 생성할 수 없습니다: {str(e)}")
        
        return None
    
    def _extract_token_from_event(self, event_data: Dict[str, Any]) -> Optional[str]:
        """이벤트에서 토큰 추출"""
        chunk_obj = event_data.get("chunk")
        token = None
        
        if chunk_obj:
            if hasattr(chunk_obj, "content"):
                content = chunk_obj.content
                if isinstance(content, str):
                    token = content
                elif isinstance(content, list) and len(content) > 0:
                    token = content[0] if isinstance(content[0], str) else str(content[0])
                else:
                    token = str(content) if content else None
            elif isinstance(chunk_obj, str):
                token = chunk_obj
            elif isinstance(chunk_obj, dict):
                token = chunk_obj.get("content") or chunk_obj.get("text")
            elif hasattr(chunk_obj, "text"):
                token = chunk_obj.text
            elif hasattr(chunk_obj, "__class__") and "AIMessageChunk" in str(type(chunk_obj)):
                try:
                    content = getattr(chunk_obj, "content", None)
                    if isinstance(content, str):
                        token = content
                    elif isinstance(content, list) and len(content) > 0:
                        token = content[0] if isinstance(content[0], str) else str(content[0])
                    elif content is not None:
                        token = str(content)
                except Exception:
                    token = None
            else:
                token = str(chunk_obj) if chunk_obj else None
        
        if not token and "delta" in event_data:
            delta = event_data["delta"]
            if isinstance(delta, dict):
                token = delta.get("content") or delta.get("text")
            elif isinstance(delta, str):
                token = delta
        
        return token if isinstance(token, str) and len(token) > 0 else None
    
    def _is_target_node(self, event_name: str, event_parent: Any, last_node_name: Optional[str]) -> bool:
        """타겟 노드인지 확인"""
        target_nodes = ["generate_answer_stream", "generate_answer_enhanced", "generate_and_validate_answer", "direct_answer"]
        
        if "generate_answer" in event_name.lower() or \
           "generate_and_validate" in event_name.lower() or \
           event_name in target_nodes:
            return True
        
        if isinstance(event_parent, dict):
            parent_node_name = event_parent.get("name", "")
            if parent_node_name and (
                "generate_answer" in parent_node_name.lower() or 
                "generate_and_validate" in parent_node_name.lower() or
                parent_node_name in target_nodes
            ):
                return True
        
        if last_node_name in target_nodes:
            if "Chat" in event_name or "LLM" in event_name or "Model" in event_name:
                return True
        
        return False
    
    async def _extract_related_questions_from_state(
        self,
        state_values: Dict[str, Any],
        initial_state: Dict[str, Any],
        message: str,
        full_answer: str,
        session_id: str
    ) -> list:
        """State에서 related_questions 추출 및 생성"""
        related_questions = state_values.get("metadata", {}).get("related_questions", [])
        
        if related_questions:
            return related_questions
        
        sources = state_values.get("sources", [])
        sources_detail = state_values.get("sources_detail", [])
        
        if not sources and not sources_detail:
            retrieved_docs = state_values.get("retrieved_docs", [])
            if retrieved_docs and hasattr(self, 'sources_extractor') and self.sources_extractor:
                try:
                    sources_data = self.sources_extractor._extract_sources(state_values)
                    sources_detail_data = self.sources_extractor._extract_sources_detail(state_values)
                    
                    if sources_detail_data:
                        sources_detail = sources_detail_data
                        state_values["sources_detail"] = sources_detail_data
                    if sources_data:
                        sources = sources_data
                        state_values["sources"] = sources_data
                    
                    related_questions_data = self.sources_extractor._extract_related_questions(state_values)
                    if related_questions_data:
                        related_questions = related_questions_data
                except Exception as e:
                    logger.warning(f"[stream_final_answer] Failed to extract sources from retrieved_docs: {e}", exc_info=True)
        
        if not related_questions and hasattr(self, 'sources_extractor') and self.sources_extractor:
            try:
                if sources_detail and "sources_detail" not in state_values:
                    state_values["sources_detail"] = sources_detail
                if sources and "sources" not in state_values:
                    state_values["sources"] = sources
                
                related_questions_data = self.sources_extractor._extract_related_questions(state_values)
                if related_questions_data:
                    related_questions = related_questions_data
            except Exception as e:
                logger.warning(f"[stream_final_answer] Failed to extract related_questions from state: {e}", exc_info=True)
        
        if not related_questions and self.workflow_service and hasattr(self.workflow_service, 'conversation_flow_tracker') and self.workflow_service.conversation_flow_tracker:
            try:
                from lawfirm_langgraph.core.services.conversation_manager import ConversationContext, ConversationTurn
                
                query = state_values.get("query", "") or initial_state.get("query", "") or message
                answer = state_values.get("answer", "") or full_answer or ""
                query_type = state_values.get("metadata", {}).get("query_type", "general_question")
                
                if query and answer and len(answer.strip()) >= 10:
                    turn = ConversationTurn(
                        user_query=query,
                        bot_response=answer,
                        timestamp=datetime.now(),
                        question_type=query_type
                    )
                    
                    context = ConversationContext(
                        session_id=session_id or "default",
                        turns=[turn],
                        entities={},
                        topic_stack=[],
                        created_at=datetime.now(),
                        last_updated=datetime.now()
                    )
                    
                    suggested_questions = self.workflow_service.conversation_flow_tracker.suggest_follow_up_questions(context)
                    
                    if suggested_questions and len(suggested_questions) > 0:
                        related_questions = [str(q).strip() for q in suggested_questions if q and str(q).strip()]
            except Exception as e:
                logger.warning(f"[stream_final_answer] Failed to generate related_questions using ConversationFlowTracker: {e}", exc_info=True)
        
        return related_questions
    
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

