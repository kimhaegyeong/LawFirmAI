"""
채팅 서비스 (lawfirm_langgraph 래퍼)
"""
import sys
import json
import logging
import asyncio
import uuid
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


class ChatService:
    """채팅 서비스"""
    
    def __init__(self):
        """초기화"""
        self.workflow_service: Optional[LangGraphWorkflowService] = None
        self._initialize_workflow()
        
        # 스트리밍 설정 및 서비스 초기화
        from api.services.stream_config import StreamConfig
        from api.services.stream_event_processor import StreamEventProcessor
        from api.services.sources_extractor import SourcesExtractor
        from api.services.session_service import session_service
        from api.services.streaming.stream_handler import StreamHandler
        from api.utils.langgraph_config_helper import create_langgraph_config
        
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
                "answer": "죄송합니다. 서비스 초기화에 실패했습니다.\n\n원인:\n" + "\n".join(f"- {detail}" for detail in error_details) + "\n\nAPI 서버 로그를 확인하거나 환경 변수를 설정해주세요.",
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
        except asyncio.CancelledError:
            logger.warning(f"⚠️ [process_message] 워크플로우 실행이 취소되었습니다 (CancelledError)")
            import os
            debug_mode = os.getenv("DEBUG", "false").lower() == "true"
            error_detail = "작업이 취소되었습니다" if not debug_mode else "CancelledError: 작업이 취소되었습니다"
            return {
                "answer": "죄송합니다. 작업이 취소되었습니다. 다시 시도해주세요.",
                "sources": [],
                "confidence": 0.0,
                "legal_references": [],
                "processing_steps": [f"오류: {error_detail}"],
                "session_id": session_id or "error",
                "processing_time": 0.0,
                "query_type": "error",
                "metadata": {"error": error_detail, "cancelled": True} if debug_mode else {"error": True, "cancelled": True},
                "errors": [error_detail]
            }
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
        
        if not self.workflow_service:
            error_event = self._create_error_event("[오류] 서비스 초기화에 실패했습니다.")
            yield json.dumps(error_event, ensure_ascii=False) + "\n"
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
            
            # LangGraph config 생성 (유틸리티 함수 사용)
            config = create_langgraph_config(
                session_id=session_id,
                enable_checkpoint=enable_checkpoint
            )
            
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
                
                try:
                    async for event in self._get_stream_events(initial_state, config):
                        event_count += 1
                        # 이벤트 타입 확인
                        event_type = event.get("event", "")
                        event_name = event.get("name", "")
                        
                        # 이벤트 타입 추적 (디버깅용)
                        event_types_seen.add(event_type)
                        if event_name:
                            node_names_seen.add(event_name)
                        
                        # 관련 없는 이벤트는 로깅만 하고 건너뛰기 (성능 최적화 - 조기 종료)
                        if event_type not in RELEVANT_EVENT_TYPES:
                            if DEBUG_STREAM and event_count <= 20:
                                logger.debug(f"건너뛴 이벤트 #{event_count}: type={event_type}, name={event_name} (관련 이벤트 타입 아님)")
                            continue
                        
                        # 디버깅 모드에서만 이벤트 추적 (메모리 최적화: 제한적 추적)
                        if DEBUG_STREAM and event_count <= MAX_EVENT_HISTORY:
                            if event_count <= 20:
                                logger.debug(f"처리할 이벤트 #{event_count}: type={event_type}, name={event_name}")
                        
                        # StreamEventProcessor를 사용하여 이벤트 처리
                        try:
                            stream_event = self.event_processor.process_stream_event(event)
                            if stream_event:
                                yield json.dumps(stream_event, ensure_ascii=False) + "\n"
                                if stream_event.get("type") == "stream":
                                    llm_stream_count += 1
                                    if DEBUG_STREAM and llm_stream_count <= 10:
                                        logger.debug(f"✅ Stream 이벤트 생성: content_length={len(stream_event.get('content', ''))}")
                            elif DEBUG_STREAM and event_type in ["on_llm_stream", "on_chat_model_stream"]:
                                # on_llm_stream 이벤트가 처리되지 않은 경우 로깅
                                logger.warning(
                                    f"⚠️ on_llm_stream 이벤트가 stream_event로 변환되지 않음: "
                                    f"event_type={event_type}, name={event_name}, "
                                    f"data_keys={list(event.get('data', {}).keys()) if isinstance(event.get('data'), dict) else 'N/A'}"
                                )
                        except Exception as process_error:
                            logger.error(
                                f"이벤트 처리 중 오류: event_type={event_type}, name={event_name}, "
                                f"error={process_error}",
                                exc_info=True
                            )
                except asyncio.CancelledError:
                    logger.warning("⚠️ [stream_message] 워크플로우 스트리밍이 취소되었습니다 (CancelledError)")
                    # 취소된 경우 에러 이벤트 전송
                    error_event = self._create_error_event(
                        "[오류] 작업이 취소되었습니다. 다시 시도해주세요.",
                        error_type="cancelled"
                    )
                    yield json.dumps(error_event, ensure_ascii=False) + "\n"
                    return
                
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
                                
                                # 🔥 [END] 키워드가 이미 발견되었는지 확인
                                # 이미 [END] 이후라면 더 이상 전송하지 않음
                                if '[END]' in current_full_answer.upper():
                                    # [END] 키워드가 이미 발견되었으므로 추가 데이터 전송 안 함
                                    continue
                                
                                if len(answer) > len(current_full_answer):
                                    new_part = answer[len(current_full_answer):]
                                    if new_part:
                                        # 🔥 [END] 키워드 이후 내용 필터링
                                        # 현재까지의 전체 답변에서 [END] 위치 확인 (대소문자 구분 없이)
                                        full_answer_with_new = self.event_processor.full_answer + new_part
                                        end_keyword_pos = -1
                                        
                                        # 대소문자 구분 없이 [END] 키워드 찾기
                                        full_answer_upper = full_answer_with_new.upper()
                                        for keyword in ['[END]', '[END', 'END]']:
                                            pos = full_answer_upper.find(keyword.upper())
                                            if pos != -1:
                                                end_keyword_pos = pos
                                                break
                                        
                                        if end_keyword_pos != -1:
                                            # [END] 키워드가 발견되면 그 이후 내용은 제외
                                            # [END] 키워드까지의 내용만 전송
                                            answer_until_end = full_answer_with_new[:end_keyword_pos].rstrip()
                                            # 이미 전송한 부분 이후의 새로운 부분만 계산
                                            already_sent = len(self.event_processor.full_answer)
                                            if len(answer_until_end) > already_sent:
                                                new_part = answer_until_end[already_sent:]
                                                self.event_processor.full_answer = answer_until_end
                                            else:
                                                # [END] 이전 내용이 이미 모두 전송됨
                                                new_part = ""
                                        else:
                                            # [END] 키워드가 아직 없으면 정상적으로 전송
                                            self.event_processor.full_answer = full_answer_with_new
                                        
                                        if new_part:
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
            
        except Exception as e:
            logger.error(f"Error in stream_message: {e}", exc_info=True)
            try:
                error_event = self._create_error_event(
                    f"[오류] 스트리밍 처리 중 오류 발생: {str(e)}",
                    error_type=type(e).__name__
                )
                yield json.dumps(error_event, ensure_ascii=False) + "\n"
            except Exception as yield_error:
                logger.error(f"Error yielding error message: {yield_error}")
                try:
                    fallback_event = self._create_error_event("[오류] 스트리밍 처리 중 오류가 발생했습니다.")
                    yield json.dumps(fallback_event, ensure_ascii=False) + "\n"
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
        process_query(use_astream_events=True)를 사용하여 스트리밍 답변 생성
        
        workflow_service.process_query()의 내부 스트리밍 로직을 활용하여
        run_query_test.py와 동일한 방식으로 실행합니다.
        """
        if not self.workflow_service:
            error_event = self._create_error_event(
                "[오류] 서비스 초기화에 실패했습니다.",
                error_type="initialization_error"
            )
            error_chunk = f"data: {json.dumps(error_event, ensure_ascii=False)}\n\n"
            yield error_chunk
            return
        
        try:
            if not session_id:
                session_id = str(uuid.uuid4())
            
            # 스트리밍 이벤트를 처리하기 위한 큐 생성
            from api.services.streaming.event_builder import StreamEventBuilder
            from api.utils.sse_formatter import format_sse_event
            from datetime import datetime
            
            event_builder = StreamEventBuilder()
            stream_queue = asyncio.Queue()
            process_completed = asyncio.Event()
            process_error = None
            
            # [END] 키워드 필터링을 위한 변수
            end_keyword_found = False
            current_full_answer = ""
            
            async def process_query_task():
                """백그라운드에서 process_query 실행"""
                nonlocal process_error
                try:
                    # process_query 실행 (use_astream_events=True, stream_queue 전달)
                    result = await self.workflow_service.process_query(
                        query=message,
                        session_id=session_id,
                        enable_checkpoint=False,
                        use_astream_events=True,
                        stream_queue=stream_queue
                    )
                    
                    # 결과를 큐에 넣기
                    await stream_queue.put({
                        "type": "result",
                        "data": result
                    })
                except Exception as e:
                    process_error = e
                    await stream_queue.put({
                        "type": "error",
                        "data": str(e)
                    })
                finally:
                    process_completed.set()
            
            # 백그라운드 태스크 시작
            process_task = asyncio.create_task(process_query_task())
            
            # 진행 이벤트 전송
            progress_event = event_builder.create_progress_event("답변 생성 중...")
            yield format_sse_event(progress_event)
            
            # 스트리밍 이벤트 처리
            done_event_sent = False
            try:
                # 큐가 비어있지 않거나 프로세스가 완료되지 않았으면 계속 처리
                max_empty_iterations = 300  # 최대 30초 대기 (0.1초 * 300)
                empty_iterations = 0
                
                while True:
                    try:
                        # 큐에서 이벤트 가져오기 (타임아웃 설정)
                        try:
                            item = await asyncio.wait_for(
                                stream_queue.get(),
                                timeout=0.1
                            )
                            empty_iterations = 0  # 이벤트를 받았으면 카운터 리셋
                        except asyncio.TimeoutError:
                            # 타임아웃은 정상 (큐가 비어있을 수 있음)
                            empty_iterations += 1
                            
                            # 프로세스가 완료되었고 큐가 비어있으면 종료
                            if process_completed.is_set():
                                # 🔥 개선: 프로세스가 완료되었어도 큐에 이벤트가 있으면 계속 처리
                                # 큐가 비어있지 않으면 더 기다림 (이벤트가 큐에 들어오는 중일 수 있음)
                                if not stream_queue.empty():
                                    # 큐에 이벤트가 있으면 empty_iterations 리셋하고 계속 처리
                                    empty_iterations = 0
                                    continue
                                
                                # 큐가 비어있고 프로세스도 완료되었으면 종료
                                if stream_queue.empty():
                                    logger.debug("[stream_final_answer] 프로세스 완료 및 큐 비어있음, 종료")
                                    break
                                
                                # 큐가 비어있지 않은데 계속 타임아웃이 발생하면 종료
                                if empty_iterations >= max_empty_iterations:
                                    logger.warning("[stream_final_answer] 큐 대기 시간 초과, 종료")
                                    break
                            else:
                                # 프로세스가 아직 실행 중이면 계속 대기
                                # 🔥 개선: 프로세스가 실행 중일 때는 더 오래 기다림 (워크플로우가 오래 걸릴 수 있음)
                                if empty_iterations < max_empty_iterations * 3:  # 90초로 증가 (30초 * 3)
                                    continue
                                else:
                                    logger.warning("[stream_final_answer] 프로세스 대기 시간 초과")
                                    break
                        
                        if item["type"] == "stream":
                            # 실시간 스트리밍 이벤트 처리
                            chunk_content = item.get("content", "")
                            
                            if chunk_content and not end_keyword_found:
                                # [END] 키워드 확인
                                current_full_answer += chunk_content
                                
                                # [END] 키워드 찾기 (대소문자 무시)
                                end_pos = -1
                                for keyword in ["[END]", "[end]", "[End]"]:
                                    pos = current_full_answer.find(keyword)
                                    if pos != -1:
                                        end_pos = pos
                                        break
                                
                                if end_pos != -1:
                                    end_keyword_found = True
                                    # [END] 이전 내용만 전송
                                    content_to_send = current_full_answer[:end_pos].rstrip()
                                    if content_to_send:
                                        stream_event = event_builder.create_stream_event(
                                            content_to_send,
                                            source="process_query"
                                        )
                                        yield format_sse_event(stream_event)
                                    logger.debug(f"✅ [STREAM] [END] 키워드 이후 내용 제거됨 (위치: {end_pos})")
                                else:
                                    # [END] 키워드가 없으면 청크 전송
                                    stream_event = event_builder.create_stream_event(
                                        chunk_content,
                                        source="process_query"
                                    )
                                    yield format_sse_event(stream_event)
                        
                        elif item["type"] == "result":
                            # 결과 처리
                            result = item["data"]
                            answer = result.get("answer", "")
                            
                            # [END] 키워드 필터링
                            if not end_keyword_found:
                                # [END] 키워드 찾기 (대소문자 무시)
                                end_pos = -1
                                for keyword in ["[END]", "[end]", "[End]"]:
                                    pos = answer.find(keyword)
                                    if pos != -1:
                                        end_pos = pos
                                        break
                                
                                if end_pos != -1:
                                    end_keyword_found = True
                                    answer = answer[:end_pos].rstrip()
                                    logger.debug(f"✅ [STREAM] [END] 키워드 이후 내용 제거됨 (위치: {end_pos})")
                            
                            # 최종 이벤트 전송
                            final_event = event_builder.create_final_event(
                                content=answer,
                                metadata=result.get("metadata", {})
                            )
                            yield format_sse_event(final_event)
                            
                        elif item["type"] == "error":
                            # 에러 이벤트 전송
                            error_event = event_builder.create_error_event(
                                f"[오류] {item['data']}"
                            )
                            yield format_sse_event(error_event)
                            
                    except Exception as e:
                        logger.error(f"[stream_final_answer] 큐 처리 중 오류: {e}", exc_info=True)
                        # 에러 발생 시에도 done 이벤트 전송 보장
                        break
                
                # 프로세스 완료 대기 (타임아웃 설정)
                try:
                    await asyncio.wait_for(process_task, timeout=300.0)  # 최대 5분 대기
                except asyncio.TimeoutError:
                    logger.warning("[stream_final_answer] process_query 타임아웃")
                    process_task.cancel()
                    try:
                        await process_task
                    except asyncio.CancelledError:
                        pass
                
                # 에러가 발생한 경우 처리
                if process_error:
                    error_event = event_builder.create_error_event(
                        f"[오류] {str(process_error)}"
                    )
                    yield format_sse_event(error_event)
                
            except asyncio.CancelledError:
                # 태스크 취소
                logger.debug("[stream_final_answer] Stream cancelled (client disconnected)")
                process_task.cancel()
                try:
                    await process_task
                except asyncio.CancelledError:
                    pass
                # GeneratorExit가 아닌 경우에만 done 이벤트 전송 시도
                # 하지만 CancelledError는 클라이언트 연결이 끊어진 경우이므로 yield 시도하지 않음
                raise
            except GeneratorExit:
                # GeneratorExit는 제너레이터가 이미 종료된 상태이므로 yield를 시도하면 안 됨
                logger.debug("[stream_final_answer] Generator exit (client disconnected)")
                # 백그라운드 태스크 정리
                if not process_task.done():
                    process_task.cancel()
                    try:
                        await process_task
                    except (asyncio.CancelledError, GeneratorExit):
                        pass
                # GeneratorExit는 바로 raise (yield 시도하지 않음)
                raise
            except GeneratorExit:
                # GeneratorExit는 제너레이터가 이미 종료된 상태이므로 yield를 시도하면 안 됨
                logger.debug("[stream_final_answer] Generator exit during stream processing")
                # 백그라운드 태스크 정리
                if not process_task.done():
                    process_task.cancel()
                    try:
                        await process_task
                    except (asyncio.CancelledError, GeneratorExit):
                        pass
                raise
            except Exception as e:
                logger.error(f"[stream_final_answer] 스트리밍 처리 중 오류: {e}", exc_info=True)
                # ERR_INCOMPLETE_CHUNKED_ENCODING 방지: done 이벤트 전송 시도
                try:
                    error_event = event_builder.create_error_event(str(e))
                    yield format_sse_event(error_event)
                    done_event = {"type": "done", "timestamp": datetime.now().isoformat(), "error": str(e)}
                    yield format_sse_event(done_event)
                    done_event_sent = True
                except GeneratorExit:
                    # GeneratorExit는 제너레이터가 이미 종료된 상태이므로 바로 raise
                    raise
                except Exception:
                    pass  # 이미 연결이 끊어진 경우 무시
                raise
            finally:
                # 백그라운드 태스크 정리
                if 'process_task' in locals() and not process_task.done():
                    process_task.cancel()
                    try:
                        await process_task
                    except (asyncio.CancelledError, GeneratorExit):
                        pass
                    except Exception as cleanup_error:
                        logger.debug(f"[stream_final_answer] Error cleaning up process_task: {cleanup_error}")
                
                # GeneratorExit가 발생한 경우 yield를 시도하지 않음
                # GeneratorExit는 제너레이터가 이미 종료된 상태이므로 yield를 시도하면 RuntimeError 발생
                try:
                    # ERR_INCOMPLETE_CHUNKED_ENCODING 방지: done 이벤트가 아직 전송되지 않았으면 전송
                    if not done_event_sent:
                        done_event = {"type": "done", "timestamp": datetime.now().isoformat()}
                        yield format_sse_event(done_event)
                        done_event_sent = True
                except GeneratorExit:
                    # GeneratorExit는 제너레이터가 이미 종료된 상태이므로 바로 raise
                    raise
                except Exception:
                    # 다른 예외는 무시 (이미 연결이 끊어진 경우 등)
                    pass
            
        except asyncio.CancelledError:
            logger.warning("⚠️ [stream_final_answer] 스트리밍이 취소되었습니다 (CancelledError)")
            # CancelledError는 클라이언트 연결이 끊어진 경우이므로 yield 시도하지 않음
            raise  # 상위로 전파
        except GeneratorExit:
            # GeneratorExit는 제너레이터가 이미 종료된 상태이므로 yield를 시도하면 안 됨
            logger.debug("[stream_final_answer] Generator exit at outer level")
            raise
        except Exception as e:
            logger.error(f"[stream_final_answer] 스트리밍 처리 중 오류: {e}", exc_info=True)
            # 🔥 ERR_INCOMPLETE_CHUNKED_ENCODING 방지: done 이벤트 전송 시도
            try:
                error_event = self._create_error_event(
                    f"[오류] 스트리밍 처리 중 오류가 발생했습니다: {str(e)}"
                )
                error_chunk = f"data: {json.dumps(error_event, ensure_ascii=False)}\n\n"
                yield error_chunk
                done_event = {"type": "done", "timestamp": datetime.now().isoformat(), "error": str(e)}
                yield format_sse_event(done_event)
            except Exception:
                pass  # 이미 연결이 끊어진 경우 무시
    
    
    def _create_error_event(self, content: str, error_type: Optional[str] = None) -> Dict[str, Any]:
        """에러 이벤트 생성"""
        from api.services.streaming.event_builder import StreamEventBuilder
        error_event = StreamEventBuilder.create_error_event(content, error_type)
        return {
            "type": "final",
            "content": content,
            "metadata": {**error_event.get("metadata", {}), "error": True},
            "timestamp": error_event.get("timestamp")
        }
    
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
        """스트리밍 이벤트 가져오기 (최신 LangGraph API 호환, 하위 호환성 유지)"""
        DEBUG_STREAM = self.stream_config.debug_stream
        
        try:
            # 최신 API 시도: version 파라미터 없이, 필터링 없이 모든 이벤트 수신
            if DEBUG_STREAM:
                logger.info("스트리밍 시작: astream_events() 사용 (최신 API, 필터링 없음)")
            
            event_count = 0
            event_types_received = set()
            
            try:
                # 필터링 없이 모든 이벤트 수신 (문제 진단을 위해)
                async for event in self.workflow_service.app.astream_events(
                    initial_state, 
                    config
                ):
                    event_count += 1
                    event_type = event.get("event", "")
                    event_name = event.get("name", "")
                    event_types_received.add(event_type)
                    
                    # 상세 로깅 (처음 50개 이벤트만)
                    if DEBUG_STREAM and event_count <= 50:
                        logger.debug(
                            f"[_get_stream_events] 이벤트 #{event_count}: "
                            f"type={event_type}, name={event_name}, "
                            f"data_keys={list(event.get('data', {}).keys()) if isinstance(event.get('data'), dict) else 'N/A'}"
                        )
                    
                    # on_llm_stream 이벤트 상세 로깅
                    if event_type == "on_llm_stream" and DEBUG_STREAM:
                        event_data = event.get("data", {})
                        chunk = event_data.get("chunk") if isinstance(event_data, dict) else None
                        logger.info(
                            f"[_get_stream_events] ✅ on_llm_stream 이벤트 수신: "
                            f"name={event_name}, "
                            f"chunk_type={type(chunk).__name__ if chunk else 'None'}, "
                            f"has_chunk={chunk is not None}"
                        )
                    
                    yield event
                
                # 이벤트 수신 완료 후 통계 로깅
                if DEBUG_STREAM:
                    logger.info(
                        f"[_get_stream_events] 이벤트 수신 완료: "
                        f"총 {event_count}개, "
                        f"이벤트 타입={sorted(event_types_received)}, "
                        f"on_llm_stream 포함={'on_llm_stream' in event_types_received}"
                    )
                    
            except (TypeError, AttributeError) as ve:
                # 구버전 API 폴백: version="v2" 파라미터 사용
                logger.warning(f"최신 API 실패, 구버전 API 시도: {ve}")
                if DEBUG_STREAM:
                    logger.info("스트리밍 시작: astream_events(version='v2') 사용 (구버전 API)")
                try:
                    async for event in self.workflow_service.app.astream_events(
                        initial_state, 
                        config,
                        version="v2"
                    ):
                        event_type = event.get("event", "")
                        if DEBUG_STREAM and event_type == "on_llm_stream":
                            logger.info(f"[_get_stream_events] ✅ on_llm_stream 이벤트 수신 (구버전 API)")
                        yield event
                except (TypeError, AttributeError) as ve2:
                    # version 파라미터도 미지원 시 기본 호출
                    logger.warning(f"version 파라미터 미지원, 기본 호출: {ve2}")
                    if DEBUG_STREAM:
                        logger.info("스트리밍 시작: astream_events() 사용 (기본 버전)")
                    async for event in self.workflow_service.app.astream_events(
                        initial_state, 
                        config
                    ):
                        yield event
        except asyncio.CancelledError:
            logger.warning("⚠️ [_get_stream_events] astream_events가 취소되었습니다 (CancelledError)")
            raise  # 상위로 전파
        except Exception as e:
            logger.error(f"⚠️ [_get_stream_events] astream_events 실행 중 오류: {e}", exc_info=True)
            raise
    
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
            # LangGraph config 생성 (유틸리티 함수 사용)
            config = create_langgraph_config(session_id=session_id)
            final_state = await asyncio.wait_for(
                self.workflow_service.app.aget_state(config),
                timeout=2.0
            )
            
            if final_state and final_state.values:
                state_values = final_state.values
                # retrieved_docs에서 직접 sources_by_type 생성
                retrieved_docs = state_values.get("retrieved_docs", [])
                related_questions_data = self.sources_extractor._extract_related_questions(state_values)
                
                if retrieved_docs and isinstance(retrieved_docs, list):
                    sources_by_type = self.sources_extractor._generate_sources_by_type_from_retrieved_docs(retrieved_docs)
                    result["sources_by_type"] = sources_by_type
                if related_questions_data:
                    result["related_questions"] = related_questions_data
                
                logger.info(f"Re-extracted sources before final event: sources_by_type={bool(result.get('sources_by_type'))}, related_questions={len(result.get('related_questions', []))}")
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
        from api.services.streaming.token_extractor import TokenExtractor
        return TokenExtractor.extract_from_event(event_data)
    
    def _is_target_node(self, event_name: str, event_parent: Any, last_node_name: Optional[str]) -> bool:
        """타겟 노드인지 확인"""
        from api.services.streaming.node_filter import NodeFilter
        node_filter = NodeFilter()
        return node_filter.is_target_node(event_name, event_parent, last_node_name)
    
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
        
        # retrieved_docs에서 sources_by_type 생성
        retrieved_docs = state_values.get("retrieved_docs", [])
        sources_by_type = state_values.get("sources_by_type")
        if not sources_by_type and retrieved_docs and isinstance(retrieved_docs, list) and hasattr(self, 'sources_extractor') and self.sources_extractor:
            try:
                sources_by_type = self.sources_extractor._generate_sources_by_type_from_retrieved_docs(retrieved_docs)
                state_values["sources_by_type"] = sources_by_type
                
                related_questions_data = self.sources_extractor._extract_related_questions(state_values)
                if related_questions_data:
                    related_questions = related_questions_data
                    # state_values의 metadata에 저장
                    if isinstance(state_values, dict):
                        if "metadata" not in state_values:
                            state_values["metadata"] = {}
                        if isinstance(state_values["metadata"], dict):
                            state_values["metadata"]["related_questions"] = related_questions_data
                            logger.info(f"[chat_service] Saved {len(related_questions_data)} related_questions to state metadata")
            except Exception as e:
                logger.warning(f"[chat_service] Failed to generate sources_by_type from retrieved_docs: {e}", exc_info=True)
        
        if not related_questions and hasattr(self, 'sources_extractor') and self.sources_extractor:
            try:
                if sources_detail and "sources_detail" not in state_values:
                    state_values["sources_detail"] = sources_detail
                if sources and "sources" not in state_values:
                    state_values["sources"] = sources
                
                related_questions_data = self.sources_extractor._extract_related_questions(state_values)
                if related_questions_data:
                    related_questions = related_questions_data
                    # state_values의 metadata에 저장
                    if isinstance(state_values, dict):
                        if "metadata" not in state_values:
                            state_values["metadata"] = {}
                        if isinstance(state_values["metadata"], dict):
                            state_values["metadata"]["related_questions"] = related_questions_data
                            logger.info(f"[chat_service] Saved {len(related_questions_data)} related_questions to state metadata")
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
            chat_service = ChatService()
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

