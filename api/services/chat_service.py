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
        
        self.stream_config = StreamConfig.from_env()
        self.event_processor = StreamEventProcessor(config=self.stream_config)
        self.sources_extractor = SourcesExtractor(
            workflow_service=self.workflow_service,
            session_service=session_service
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
            from lawfirm_langgraph.core.workflow.state.state_definitions import create_initial_legal_state
            
            # 로깅: message 값 확인
            logger.info(f"stream_message: 받은 message='{message[:100] if message else 'EMPTY'}...', length={len(message) if message else 0}")
            
            # message를 query로 사용 (create_initial_legal_state의 첫 번째 파라미터는 query)
            initial_state = create_initial_legal_state(message, session_id)
            
            # 상태 검증 및 보강 (통합된 검증 로직)
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
            
            # 초기 state 검증 (한 번만 수행)
            initial_query = initial_state.get("input", {}).get("query") or initial_state.get("query")
            if not initial_query or not str(initial_query).strip():
                logger.error(f"Initial state query is empty! Input message was: '{message[:50]}...'")
                error_event = {
                    "type": "final",
                    "content": "[오류] 질문이 제대로 전달되지 않았습니다. 다시 시도해주세요.",
                    "metadata": {"error": True},
                    "timestamp": datetime.now().isoformat()
                }
                yield json.dumps(error_event, ensure_ascii=False) + "\n"
                return
            
            config = {"configurable": {"thread_id": session_id}}
            
            # 디버그 모드 확인
            DEBUG_STREAM = self.stream_config.debug_stream
            
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
                event_types_seen = set()  # 본 이벤트 타입 추적 (디버깅용, 제한적 사용)
                node_names_seen = set()  # 본 노드 이름 추적 (디버깅용, 제한적 사용)
                
                # 관련 이벤트 타입 집합 (성능 최적화)
                RELEVANT_EVENT_TYPES = self.stream_config.relevant_event_types
                
                # 메모리 최적화: 이벤트 히스토리 크기 제한
                MAX_EVENT_HISTORY = self.stream_config.max_event_history
                
                async for event in get_stream_events():
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
                                self.event_processor.full_answer = final_answer
                                # 스트림 청크를 JSONL 형식으로 전송
                                stream_event = {
                                    "type": "stream",
                                    "content": missing_part,
                                    "timestamp": datetime.now().isoformat()
                                }
                                yield json.dumps(stream_event, ensure_ascii=False) + "\n"
                                self.event_processor.answer_found = True
                                if DEBUG_STREAM:
                                    logger.info(f"최종 답변에서 누락된 부분 전송: {len(missing_part)}자")
                        elif final_answer:
                            # 전체 답변 전송 (full_answer가 비어있는 경우)
                            self.event_processor.full_answer = final_answer
                            stream_event = {
                                "type": "stream",
                                "content": final_answer,
                                "timestamp": datetime.now().isoformat()
                            }
                            yield json.dumps(stream_event, ensure_ascii=False) + "\n"
                            self.event_processor.answer_found = True
                            if DEBUG_STREAM:
                                logger.info(f"전체 답변 전송: {len(final_answer)}자")
                    except Exception as e:
                        if DEBUG_STREAM:
                            logger.error(f"최종 결과 가져오기 실패: {e}", exc_info=True)
                        # 에러 발생 시에도 최소한 에러 메시지 yield (스트림이 비어있지 않도록)
                        if not self.event_processor.answer_found:
                            error_event = {
                                "type": "final",
                                "content": f"[오류] 답변을 생성할 수 없습니다: {str(e)}",
                                "metadata": {"error": True},
                                "timestamp": datetime.now().isoformat()
                            }
                            yield json.dumps(error_event, ensure_ascii=False) + "\n"
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
            
            # 완료 메타데이터 (답변이 없어도 완료 신호 전송)
            # sources는 스트림 종료 직전에 SourcesExtractor를 사용하여 가져오기
            final_sources = []
            final_legal_references = []
            final_sources_detail = []
            final_related_questions = []
            
            # 스트림 종료 직전에 sources 가져오기 (빠르게 처리)
            if session_id:
                try:
                    try:
                        sources_data = await asyncio.wait_for(
                            self.sources_extractor.extract_from_state(session_id),
                            timeout=2.0  # 2초 타임아웃
                        )
                        final_sources = sources_data.get("sources", [])
                        final_legal_references = sources_data.get("legal_references", [])
                        final_sources_detail = sources_data.get("sources_detail", [])
                        final_related_questions = sources_data.get("related_questions", [])
                    except asyncio.TimeoutError:
                        logger.warning(f"Timeout getting sources from LangGraph state for session {session_id}")
                    except Exception as e:
                        logger.warning(f"Failed to get sources from LangGraph state: {e}")
                except Exception as e:
                    logger.warning(f"Error getting sources from LangGraph state: {e}")
            
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
                
                # final 이벤트 생성 직전에 sources를 다시 한 번 확인
                # LangGraph state에서 sources를 가져오는 시점이 너무 빠를 수 있으므로
                # final 이벤트 생성 직전에 다시 확인 (재시도 최소화하여 스트림 블로킹 방지)
                if not final_sources and not final_legal_references and not final_sources_detail:
                    if session_id and self.workflow_service and self.workflow_service.app:
                        try:
                            config = {"configurable": {"thread_id": session_id}}
                            
                            # 즉시 시도 (타임아웃 2초, 재시도 없음)
                            try:
                                final_state = await asyncio.wait_for(
                                    self.workflow_service.app.aget_state(config),
                                    timeout=2.0
                                )
                                
                                if final_state and final_state.values:
                                    state_values = final_state.values
                                    
                                    # sources 추출
                                    sources_data = self.sources_extractor._extract_sources(state_values)
                                    legal_references_data = self.sources_extractor._extract_legal_references(state_values)
                                    sources_detail_data = self.sources_extractor._extract_sources_detail(state_values)
                                    related_questions_data = self.sources_extractor._extract_related_questions(state_values)
                                    
                                    if sources_data or legal_references_data or sources_detail_data or related_questions_data:
                                        if sources_data:
                                            final_sources = sources_data
                                        if legal_references_data:
                                            final_legal_references = legal_references_data
                                        if sources_detail_data:
                                            final_sources_detail = sources_detail_data
                                        if related_questions_data:
                                            final_related_questions = related_questions_data
                                        
                                        logger.info(f"Re-extracted sources before final event: {len(final_sources)} sources, {len(final_legal_references)} legal_references, {len(final_sources_detail)} sources_detail, {len(final_related_questions)} related_questions")
                            except asyncio.TimeoutError:
                                logger.warning(f"Timeout re-getting sources before final event")
                            except Exception as e:
                                logger.warning(f"Failed to re-get sources before final event: {e}")
                        except Exception as e:
                            logger.warning(f"Error re-getting sources before final event: {e}")
                
                if should_split:
                    # 토큰 제한을 초과했을 때만 답변 분할 처리
                    from api.services.answer_splitter import AnswerSplitter
                    
                    splitter = AnswerSplitter(chunk_size=self.stream_config.chunk_size)
                    chunks = splitter.split_answer(full_answer)
                    
                    # 메시지 저장은 chat.py에서 처리하므로 여기서는 저장하지 않음
                    # message_id는 생성만 하고 final_event에 포함
                    
                    # 첫 번째 청크만 즉시 전송
                    if chunks:
                        first_chunk = chunks[0]
                        
                        # 최종 완료 이벤트에 첫 번째 청크만 포함
                        final_event = {
                            "type": "final",
                            "content": first_chunk.content,
                            "metadata": {
                                "tokens_received": tokens_received,
                                "length": len(full_answer),
                                "answer_found": answer_found,
                                "sources": final_sources,
                                "legal_references": final_legal_references,
                                "sources_detail": final_sources_detail,
                                "related_questions": final_related_questions,
                                "needs_continuation": True,
                                "message_id": message_id
                            },
                            "timestamp": datetime.now().isoformat()
                        }
                        yield json.dumps(final_event, ensure_ascii=False) + "\n"
                    else:
                        # 청크가 없는 경우 (짧은 답변) 기존 방식 유지
                        final_event = {
                            "type": "final",
                            "content": full_answer,
                            "metadata": {
                                "tokens_received": tokens_received,
                                "length": len(full_answer),
                                "answer_found": answer_found,
                                "sources": final_sources,
                                "legal_references": final_legal_references,
                                "sources_detail": final_sources_detail,
                                "related_questions": final_related_questions,
                                "message_id": message_id
                            },
                            "timestamp": datetime.now().isoformat()
                        }
                        yield json.dumps(final_event, ensure_ascii=False) + "\n"
                else:
                    # 메시지 저장은 chat.py에서 처리하므로 여기서는 저장하지 않음
                    # message_id는 생성만 하고 final_event에 포함
                    
                    # 전체 답변을 한 번에 전송 (needs_continuation 없음)
                    final_event = {
                        "type": "final",
                        "content": full_answer,
                        "metadata": {
                            "tokens_received": tokens_received,
                            "length": len(full_answer),
                            "answer_found": answer_found,
                            "sources": final_sources,
                            "legal_references": final_legal_references,
                            "sources_detail": final_sources_detail,
                            "related_questions": final_related_questions,
                            "message_id": message_id
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
                    has_yielded = True
                except Exception:
                    # yield 자체가 실패하면 스트림이 끊어질 수 있음
                    # 하지만 이 경우는 매우 드물고, FastAPI가 자동으로 처리
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
            
            # 초기 state 생성
            from lawfirm_langgraph.core.workflow.state.state_definitions import create_initial_legal_state
            initial_state = create_initial_legal_state(message, session_id)
            
            # 상태 검증 및 보강
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
            
            # 상태 검증
            initial_query = initial_state.get("input", {}).get("query") or initial_state.get("query")
            if not initial_query or not str(initial_query).strip():
                error_event = {
                    "type": "error",
                    "content": "[오류] 질문이 제대로 전달되지 않았습니다.",
                    "timestamp": datetime.now().isoformat()
                }
                yield f"data: {json.dumps(error_event, ensure_ascii=False)}\n\n"
                return
            
            config = {"configurable": {"thread_id": session_id}}
            
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
            
            async for event in self.workflow_service.app.astream_events(
                initial_state,
                config,
                version="v2"
            ):
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
                    if node_name in ["generate_and_validate_answer", "generate_answer_enhanced"]:
                        answer_generation_started = True
                        last_node_name = node_name
                        logger.debug(f"[stream_final_answer] 답변 생성 노드 시작: {node_name}")
                
                # on_chain_end: 답변 생성 노드 완료
                elif event_type == "on_chain_end":
                    node_name = event_name
                    if node_name == "generate_and_validate_answer":
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
                    
                    # 답변 생성 노드인지 확인 (StreamEventProcessor의 is_answer_node 로직 참고)
                    is_target_node = False
                    
                    # 방법 1: 이벤트 이름으로 직접 판단
                    if "generate_answer" in event_name.lower() or \
                       "generate_and_validate" in event_name.lower() or \
                       event_name in ["generate_answer_enhanced", "generate_and_validate_answer", "direct_answer"]:
                        is_target_node = True
                        logger.debug(f"[stream_final_answer] 방법 1: 이벤트 이름으로 타겟 노드 확인: {event_name}")
                    
                    # 방법 2: parent 필드에서 노드 이름 확인
                    if not is_target_node:
                        parent_node_name = None
                        if isinstance(event_parent, dict):
                            parent_node_name = event_parent.get("name", "")
                        
                        if parent_node_name and (
                            "generate_answer" in parent_node_name.lower() or 
                            "generate_and_validate" in parent_node_name.lower() or
                            parent_node_name in ["generate_answer_enhanced", "generate_and_validate_answer"]
                        ):
                            is_target_node = True
                            logger.debug(f"[stream_final_answer] 방법 2: parent 필드로 타겟 노드 확인: {parent_node_name}")
                    
                    # 방법 3: last_node_name으로 확인 (LLM 모델 이름인 경우)
                    if not is_target_node and last_node_name in ["generate_and_validate_answer", "generate_answer_enhanced"]:
                        if "Chat" in event_name or "LLM" in event_name or "Model" in event_name:
                            is_target_node = True
                            logger.debug(f"[stream_final_answer] 방법 3: last_node_name으로 타겟 노드 확인: {last_node_name}, event_name={event_name}")
                    
                    if is_target_node:
                        logger.debug(f"[stream_final_answer] 타겟 노드 확인됨: {event_name}, 토큰 추출 시작")
                        event_data = event.get("data", {})
                        
                        # on_llm_stream, on_chat_model_stream: StreamEventProcessor의 extract_chunk_from_llm_stream 로직 사용
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
                        
                        # delta 형식 처리 (LangGraph v2)
                        if not token and "delta" in event_data:
                            delta = event_data["delta"]
                            if isinstance(delta, dict):
                                token = delta.get("content") or delta.get("text")
                            elif isinstance(delta, str):
                                token = delta
                        
                        # 토큰이 있으면 SSE 형식으로 전달
                        if token and isinstance(token, str) and len(token) > 0:
                            stream_event_count += 1
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
                                f"chunk_obj={chunk_obj}, "
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
                
                # generate_and_validate_answer 노드 완료 시점
                elif event_type == "on_chain_end" and event_name == "generate_and_validate_answer":
                    logger.info(
                        f"[stream_final_answer] 스트리밍 완료: "
                        f"총 {event_count}개 이벤트, "
                        f"스트림 이벤트 {stream_event_count}개, "
                        f"on_llm_stream={on_llm_stream_count}개, "
                        f"on_chat_model_stream={on_chat_model_stream_count}개, "
                        f"on_chain_stream={on_chain_stream_count}개"
                    )
                    
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
                        except asyncio.TimeoutError:
                            logger.warning(f"[stream_final_answer] Timeout getting state, using empty metadata")
                        except Exception as e:
                            logger.warning(f"[stream_final_answer] Error getting state: {e}")
                        
                        if final_state and state_values:
                            # metadata 추출
                            sources = state_values.get("sources", [])
                            legal_references = state_values.get("legal_references", [])
                            sources_detail = state_values.get("sources_detail", [])
                            related_questions = state_values.get("metadata", {}).get("related_questions", [])
                            
                            # sources가 여전히 없으면 retrieved_docs에서 직접 추출 시도
                            if not sources and not sources_detail:
                                retrieved_docs = state_values.get("retrieved_docs", [])
                                if retrieved_docs:
                                    logger.info(f"[stream_final_answer] Sources not in state, extracting from {len(retrieved_docs)} retrieved_docs")
                                    # SourcesExtractor를 사용하여 sources 추출
                                    if hasattr(self, 'sources_extractor') and self.sources_extractor:
                                        try:
                                            sources_data = self.sources_extractor._extract_sources(state_values)
                                            legal_references_data = self.sources_extractor._extract_legal_references(state_values)
                                            sources_detail_data = self.sources_extractor._extract_sources_detail(state_values)
                                            related_questions_data = self.sources_extractor._extract_related_questions(state_values)
                                            
                                            if sources_data:
                                                sources = sources_data
                                            if legal_references_data:
                                                legal_references = legal_references_data
                                            if sources_detail_data:
                                                sources_detail = sources_detail_data
                                            if related_questions_data:
                                                related_questions = related_questions_data
                                            
                                            logger.info(f"[stream_final_answer] Extracted {len(sources)} sources from retrieved_docs")
                                        except Exception as e:
                                            logger.warning(f"[stream_final_answer] Failed to extract sources from retrieved_docs: {e}")
                            
                            # related_questions가 없으면 sources_extractor로 추출 시도
                            if not related_questions:
                                if hasattr(self, 'sources_extractor') and self.sources_extractor:
                                    try:
                                        related_questions_data = self.sources_extractor._extract_related_questions(state_values)
                                        if related_questions_data:
                                            related_questions = related_questions_data
                                            logger.info(f"[stream_final_answer] Extracted {len(related_questions)} related_questions from state")
                                    except Exception as e:
                                        logger.warning(f"[stream_final_answer] Failed to extract related_questions from state: {e}")
                            
                            # LLM 검증 결과 추출
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
                    
                    # 완료 이벤트
                    done_event = {
                        "type": "done",
                        "content": "[DONE]",
                        "timestamp": datetime.now().isoformat()
                    }
                    yield f"data: {json.dumps(done_event, ensure_ascii=False)}\n\n"
                    break  # 스트리밍 종료
                
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

