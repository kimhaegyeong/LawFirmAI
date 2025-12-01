"""
스트리밍 처리 전용 클래스
"""
import asyncio
import logging
import uuid
from datetime import datetime
from typing import Dict, Any, Optional, AsyncGenerator, List

from .constants import StreamingConstants
from .event_builder import StreamEventBuilder
from .token_extractor import TokenExtractor
from .node_filter import NodeFilter
from api.utils.sse_formatter import format_sse_event
from api.utils.source_type_mapper import get_default_sources_by_type
from api.utils.langgraph_config_helper import create_langgraph_config_with_callbacks

logger = logging.getLogger(__name__)

# 기본 sources_by_type 구조 (실제 PostgreSQL 테이블명 기반)
DEFAULT_SOURCES_BY_TYPE = get_default_sources_by_type()


class StreamHandler:
    """스트리밍 처리 전용 클래스"""
    
    def __init__(
        self,
        workflow_service,
        sources_extractor,
        extract_related_questions_fn=None
    ):
        self.workflow_service = workflow_service
        self.sources_extractor = sources_extractor
        self.extract_related_questions_fn = extract_related_questions_fn
        self.token_extractor = TokenExtractor()
        self.node_filter = NodeFilter()
        self.event_builder = StreamEventBuilder()
        # 로그 카운터 초기화
        self._skip_log_count = 0
        self._classification_skip_count = 0
    
    async def stream_final_answer(
        self,
        message: str,
        session_id: Optional[str] = None,
        validate_and_augment_state_fn=None
    ) -> AsyncGenerator[str, None]:
        """
        LangGraph의 astream_events()를 사용하여 
        generate_and_validate_answer 노드의 LLM 응답만 스트림 형태로 전달
        """
        if not self.workflow_service:
            error_event = self.event_builder.create_error_event(
                "[오류] 서비스 초기화에 실패했습니다."
            )
            yield format_sse_event(error_event)
            return
        
        try:
            if not session_id:
                session_id = str(uuid.uuid4())
            
            from lawfirm_langgraph.core.workflow.state.state_definitions import create_initial_legal_state
            initial_state = create_initial_legal_state(message, session_id)
            
            if validate_and_augment_state_fn:
                initial_query = validate_and_augment_state_fn(initial_state, message, session_id)
            else:
                initial_query = self._validate_and_augment_state(initial_state, message, session_id)
            
            if not initial_query:
                error_event = self.event_builder.create_error_event(
                    "[오류] 질문이 제대로 전달되지 않았습니다."
                )
                yield format_sse_event(error_event)
                return
            
            callback_handler, config = await self._setup_callback_handler(session_id)
            
            # ⚠️ 주의: state에 콜백을 저장하면 LangGraph 체크포인트 직렬화 시 오류 발생
            # config의 callbacks만 사용하고 state에는 저장하지 않음
            # LangGraph는 config의 callbacks를 자동으로 노드에 전달함
            if callback_handler:
                logger.debug(f"[stream_final_answer] ✅ 콜백이 config에 설정됨: {len(config.get('callbacks', []))}개")
            
            progress_event = self.event_builder.create_progress_event("답변 생성 중...")
            yield format_sse_event(progress_event)
            
            # 초기 State 로깅
            logger.info(
                f"[stream_final_answer] 초기 State 확인: "
                f"query={initial_state.get('query', '')[:50] if initial_state.get('query') else 'N/A'}..., "
                f"session_id={initial_state.get('session_id', 'N/A')}, "
                f"state_keys={list(initial_state.keys())[:20]}"
            )
            
            # Config 로깅
            logger.info(
                f"[stream_final_answer] Config 확인: "
                f"thread_id={config.get('configurable', {}).get('thread_id', 'N/A')}, "
                f"has_callbacks={bool(config.get('callbacks'))}, "
                f"config_keys={list(config.keys())}"
            )
            
            try:
                async for chunk in self._process_stream_events(
                    initial_state, config, callback_handler, message, session_id
                ):
                    yield chunk
            except Exception as process_error:
                logger.error(
                    f"[stream_final_answer] ⚠️ _process_stream_events 실행 중 오류: {process_error}",
                    exc_info=True
                )
                # 에러 이벤트 전송
                try:
                    error_event = self.event_builder.create_error_event(
                        f"[오류] 스트리밍 처리 중 오류가 발생했습니다: {str(process_error)}"
                    )
                    yield format_sse_event(error_event)
                except Exception:
                    pass
                
                # 🔥 ERR_INCOMPLETE_CHUNKED_ENCODING 방지: 예외 발생해도 done 이벤트 전송
                try:
                    minimal_done = {"type": "done", "timestamp": datetime.now().isoformat(), "error": str(process_error)}
                    yield format_sse_event(minimal_done)
                    logger.debug("[stream_final_answer] Minimal done event sent after process error")
                except Exception:
                    pass
                raise
                
        except (GeneratorExit, asyncio.CancelledError) as cancel_error:
            # 클라이언트가 연결을 끊은 경우 - 정상적인 종료
            logger.debug(f"[stream_final_answer] Stream cancelled or client disconnected: {cancel_error}")
            # GeneratorExit와 CancelledError는 상위로 전파하여 제너레이터 종료
            raise
        except Exception as e:
            logger.error(f"Stream error: {e}", exc_info=True)
            try:
                error_event = self.event_builder.create_error_event(
                    f"[오류] 스트리밍 중 오류가 발생했습니다: {str(e)}",
                    type(e).__name__
                )
                yield format_sse_event(error_event)
                
                # 에러 발생 시에도 done 이벤트 전송하여 스트림 종료를 명확히 함
                done_event = self.event_builder.create_done_event("", {})
                yield format_sse_event(done_event)
            except (GeneratorExit, asyncio.CancelledError):
                logger.debug("[stream_final_answer] Client disconnected or cancelled during error handling")
                raise
            except Exception as yield_error:
                logger.error(f"Failed to yield error event: {yield_error}")
                # 최종 폴백: 스트림 종료를 보장하기 위해 아무것도 yield하지 않고 종료
    
    async def _setup_callback_handler(self, session_id: str):
        """콜백 핸들러 설정"""
        callback_queue = asyncio.Queue()
        callback_handler = None
        
        if self.workflow_service and hasattr(self.workflow_service, 'create_streaming_callback_handler'):
            callback_handler = self.workflow_service.create_streaming_callback_handler(queue=callback_queue)
            if callback_handler:
                logger.info("[stream_final_answer] ✅ StreamingCallbackHandler created and ready")
            else:
                logger.warning("[stream_final_answer] ⚠️ Failed to create StreamingCallbackHandler")
        
        # LangGraph config 생성 (유틸리티 함수 사용)
        if callback_handler:
            config = create_langgraph_config_with_callbacks(
                session_id=session_id,
                callbacks=[callback_handler]
            )
            logger.info(f"[stream_final_answer] ✅ Callbacks added to config: {len(config.get('callbacks', []))} callback(s)")
        else:
            # 콜백이 없는 경우 기본 config 생성
            from api.utils.langgraph_config_helper import create_langgraph_config
            config = create_langgraph_config(session_id=session_id)
            logger.warning("[stream_final_answer] ⚠️ No callback handler, streaming may not work optimally")
        
        return callback_handler, config
    
    async def _process_stream_events(
        self,
        initial_state: Dict[str, Any],
        config: Dict[str, Any],
        callback_handler: Any,
        message: str,
        session_id: str
    ) -> AsyncGenerator[str, None]:
        """스트림 이벤트 처리"""
        # 로그 카운터 초기화 (각 스트림 세션마다)
        self._skip_log_count = 0
        self._classification_skip_count = 0
        
        answer_generation_started = False
        last_node_name = None
        event_count = 0
        stream_event_count = 0
        on_llm_stream_count = 0
        on_chat_model_stream_count = 0
        on_chain_stream_count = 0
        full_answer = ""
        callback_chunks_received = 0
        processed_callback_chunks = set()
        
        callback_queue = None
        if callback_handler and hasattr(callback_handler, 'queue'):
            callback_queue = callback_handler.queue
        chunk_output_queue = asyncio.Queue() if callback_queue else None
        
        callback_monitoring_active = True
        callback_task = None
        
        async def monitor_callback_queue():
            """콜백 큐 모니터링"""
            nonlocal callback_monitoring_active
            while callback_monitoring_active:
                try:
                    if callback_queue and chunk_output_queue:
                        try:
                            chunk_data = await asyncio.wait_for(
                                callback_queue.get(),
                                timeout=StreamingConstants.CALLBACK_QUEUE_TIMEOUT
                            )
                            if chunk_data and chunk_data.get("type") == StreamingConstants.CALLBACK_CHUNK_TYPE:
                                await chunk_output_queue.put(chunk_data)
                        except asyncio.TimeoutError:
                            await asyncio.sleep(StreamingConstants.CALLBACK_MONITORING_INTERVAL)
                            continue
                        except asyncio.QueueEmpty:
                            await asyncio.sleep(StreamingConstants.CALLBACK_MONITORING_INTERVAL)
                            continue
                    else:
                        await asyncio.sleep(0.1)
                except Exception as e:
                    logger.debug(f"[stream_final_answer] Error in callback queue monitoring: {e}")
                    await asyncio.sleep(0.1)
        
        if callback_queue and chunk_output_queue:
            callback_task = asyncio.create_task(monitor_callback_queue())
            logger.info("[stream_final_answer] ✅ Callback queue monitoring task started")
        
        try:
            # 최신 LangGraph API 호환: version 파라미터 없이 시도, 실패 시 구버전 폴백
            logger.info(
                f"[_process_stream_events] 스트리밍 시작: "
                f"session_id={session_id}, "
                f"message={message[:50]}..., "
                f"initial_state_keys={list(initial_state.keys())[:20]}, "
                f"config_thread_id={config.get('configurable', {}).get('thread_id', 'N/A')}"
            )
            
            try:
                logger.info(f"[_process_stream_events] astream_events() 호출 시작")
                try:
                    astream_events_iter = self.workflow_service.app.astream_events(
                        initial_state,
                        config
                    )
                    logger.info(f"[_process_stream_events] ✅ astream_events() 제너레이터 생성 완료, 이벤트 대기 시작")
                except Exception as iter_error:
                    logger.error(f"[_process_stream_events] ⚠️ astream_events() 제너레이터 생성 실패: {iter_error}", exc_info=True)
                    # 🔥 ERR_INCOMPLETE_CHUNKED_ENCODING 방지: 제너레이터 생성 실패해도 done 이벤트 전송
                    # done 이벤트를 보낸 후에는 raise하지 않고 정상 종료 (스트림은 완료됨)
                    try:
                        error_event = self.event_builder.create_error_event(str(iter_error))
                        yield format_sse_event(error_event)
                        minimal_done = {"type": "done", "timestamp": datetime.now().isoformat(), "error": str(iter_error)}
                        yield format_sse_event(minimal_done)
                        logger.debug("[_process_stream_events] Error and done event sent after generator creation error")
                    except (GeneratorExit, asyncio.CancelledError):
                        # 클라이언트가 연결을 끊은 경우
                        raise
                    except Exception as yield_error:
                        logger.error(f"[_process_stream_events] Failed to send error/done event: {yield_error}")
                    # 예외는 로깅만 하고 raise하지 않음 (done 이벤트를 보냈으므로 스트림은 정상 종료)
                
                try:
                    async for event in astream_events_iter:
                        logger.info(f"[_process_stream_events] ✅ 이벤트 수신 #{event_count + 1}: event_type={event.get('event', 'N/A')}, name={event.get('name', 'N/A')}")
                        
                        # 콜백 큐에서 청크 처리
                        if chunk_output_queue:
                            chunks_received, full_answer, chunks_to_yield = self._process_callback_queue_chunks(
                                chunk_output_queue, processed_callback_chunks, full_answer
                            )
                            callback_chunks_received += chunks_received
                            
                            # 콜백 청크를 스트림으로 전송
                            for content in chunks_to_yield:
                                stream_event = self.event_builder.create_stream_event(
                                    content, source="callback"
                                )
                                yield format_sse_event(stream_event)
                        
                        event_count += 1
                        event_type = event.get("event", "")
                        event_name = event.get("name", "")
                        event_parent = event.get("parent", {})
                        event_data = event.get("data", {})
                        
                        # 이벤트 타입별 카운터 및 로깅
                        if event_type == "on_llm_stream":
                            on_llm_stream_count += 1
                            if on_llm_stream_count <= StreamingConstants.MAX_DEBUG_LOGS:
                                logger.debug(
                                    f"[stream_final_answer] on_llm_stream 이벤트 #{on_llm_stream_count}: "
                                    f"name={event_name}, "
                                    f"parent={event_parent.get('name', '') if isinstance(event_parent, dict) else ''}"
                                )
                        elif event_type == "on_chat_model_stream":
                            on_chat_model_stream_count += 1
                            if on_chat_model_stream_count <= StreamingConstants.MAX_DEBUG_LOGS:
                                logger.debug(
                                    f"[stream_final_answer] on_chat_model_stream 이벤트 #{on_chat_model_stream_count}: "
                                    f"name={event_name}, "
                                    f"parent={event_parent.get('name', '') if isinstance(event_parent, dict) else ''}"
                                )
                        elif event_type == "on_chain_stream":
                            on_chain_stream_count += 1
                            if on_chain_stream_count <= StreamingConstants.MAX_DETAILED_LOGS:
                                logger.debug(
                                    f"[stream_final_answer] on_chain_stream 이벤트 #{on_chain_stream_count}: "
                                    f"name={event_name}, "
                                    f"parent={event_parent.get('name', '') if isinstance(event_parent, dict) else ''}"
                                )
                        
                        # 이벤트 타입별 처리
                        if event_type == "on_chain_start":
                            answer_generation_started, last_node_name = self._handle_on_chain_start_event(
                                event_name, answer_generation_started, last_node_name
                            )
                        
                        elif event_type == "on_chain_end":
                            answer_generation_started = self._handle_on_chain_end_event(
                                event_name, answer_generation_started
                            )
                            
                            # 답변 완료 노드인 경우 남은 콜백 청크 처리
                            if self.node_filter.is_answer_completion_node(event_name):
                                if chunk_output_queue:
                                    chunks_received, full_answer, chunks_to_yield = self._process_callback_queue_chunks(
                                        chunk_output_queue, processed_callback_chunks, full_answer
                                    )
                                    callback_chunks_received += chunks_received
                                    
                                    # 콜백 청크를 스트림으로 전송
                                    for content in chunks_to_yield:
                                        stream_event = self.event_builder.create_stream_event(
                                            content, source="callback"
                                        )
                                        yield format_sse_event(stream_event)
                        
                        elif event_type in ["on_llm_stream", "on_chat_model_stream"]:
                            should_continue, full_answer, stream_event_count, token = self._handle_streaming_event(
                                event_type, event_name, event_parent, event_data,
                                answer_generation_started, last_node_name,
                                full_answer, stream_event_count
                            )
                            
                            if not should_continue:
                                continue
                            
                            # 토큰이 있으면 스트림으로 전송
                            if token:
                                stream_event = self.event_builder.create_stream_event(token)
                                yield format_sse_event(stream_event)
                
                # async for 루프 종료 후 처리 (모든 이벤트 수신 완료)
                except StopAsyncIteration:
                    logger.info(f"[_process_stream_events] ✅ astream_events() 제너레이터 정상 종료 (StopAsyncIteration)")
                    # 정상 종료이므로 계속 진행
                except Exception as iter_error:
                    logger.error(f"[_process_stream_events] ⚠️ astream_events() 이터레이터 실행 중 오류: {iter_error}", exc_info=True)
                    # 🔥 ERR_INCOMPLETE_CHUNKED_ENCODING 방지: 예외 발생해도 done 이벤트 전송
                    # done 이벤트를 보낸 후에는 raise하지 않고 정상 종료 (스트림은 완료됨)
                    try:
                        error_event = self.event_builder.create_error_event(str(iter_error))
                        yield format_sse_event(error_event)
                        minimal_done = {"type": "done", "timestamp": datetime.now().isoformat(), "error": str(iter_error)}
                        yield format_sse_event(minimal_done)
                        logger.debug("[_process_stream_events] Error and done event sent after iterator error")
                    except (GeneratorExit, asyncio.CancelledError):
                        # 클라이언트가 연결을 끊은 경우
                        raise
                    except Exception as yield_error:
                        logger.error(f"[_process_stream_events] Failed to send error/done event: {yield_error}")
                    # 예외는 로깅만 하고 raise하지 않음 (done 이벤트를 보냈으므로 스트림은 정상 종료)
                
                logger.info(
                    f"[stream_final_answer] ✅ astream_events() 루프 완료: "
                    f"총 {event_count}개 이벤트, "
                    f"스트림 이벤트 {stream_event_count}개, "
                    f"콜백 청크 {callback_chunks_received}개, "
                    f"on_llm_stream={on_llm_stream_count}개, "
                    f"on_chat_model_stream={on_chat_model_stream_count}개, "
                    f"on_chain_stream={on_chain_stream_count}개, "
                    f"full_answer_length={len(full_answer)}"
                )
                
                # 이벤트가 너무 적으면 경고
                if event_count < 5:
                    logger.warning(
                        f"[stream_final_answer] ⚠️ 이벤트 수가 너무 적습니다 ({event_count}개). "
                        f"워크플로우가 제대로 실행되지 않았을 수 있습니다."
                    )
                
                await asyncio.sleep(StreamingConstants.STATE_RETRY_DELAY)
                
                # final_metadata 가져오기 (실패해도 계속 진행)
                final_metadata = {}
                try:
                    logger.info(f"[stream_final_answer] final_metadata 가져오기 시도: session_id={session_id}")
                    final_metadata = await self._get_final_metadata(
                        config, initial_state, message, full_answer, session_id
                    )
                    logger.info(
                        f"[stream_final_answer] ✅ final_metadata 가져오기 성공: "
                        f"sources_detail={len(final_metadata.get('sources_detail', []))}, "
                        f"sources_by_type={bool(final_metadata.get('sources_by_type'))}, "
                        f"sources={len(final_metadata.get('sources', []))}, "
                        f"legal_references={len(final_metadata.get('legal_references', []))}, "
                        f"metadata_keys={list(final_metadata.keys())[:20]}"
                    )
                except Exception as metadata_error:
                    logger.warning(
                        f"[stream_final_answer] ⚠️ Failed to get final metadata: {metadata_error}",
                        exc_info=True
                    )
                    # metadata 가져오기 실패해도 계속 진행
                
                # validation 이벤트 전송 (실패해도 계속 진행)
                if final_metadata.get("llm_validation"):
                    try:
                        validation_event = self.event_builder.create_validation_event(
                            final_metadata["llm_validation"]
                        )
                        yield format_sse_event(validation_event)
                    except Exception as validation_error:
                        logger.warning(f"[stream_final_answer] Failed to send validation event: {validation_error}")
                        # validation 이벤트 전송 실패해도 계속 진행
                
                # final_event 전송 (실패해도 done 이벤트는 전송)
                try:
                    final_event = self.event_builder.create_final_event(full_answer, final_metadata)
                    yield format_sse_event(final_event)
                except (GeneratorExit, asyncio.CancelledError):
                    # 클라이언트가 연결을 끊은 경우
                    raise
                except Exception as final_error:
                    logger.warning(f"[stream_final_answer] Failed to send final event: {final_error}")
                    # final 이벤트 전송 실패해도 done 이벤트는 전송
                
                # done_event는 스트림 종료를 알리는 용도로 반드시 전송 (ERR_INCOMPLETE_CHUNKED_ENCODING 방지)
                # 중요: done 이벤트는 예외가 발생해도 반드시 전송되어야 함
                done_event_sent = False
                try:
                    done_event = self.event_builder.create_done_event(full_answer, final_metadata)
                    yield format_sse_event(done_event)
                    done_event_sent = True
                    logger.debug("[stream_final_answer] Done event sent successfully")
                except (GeneratorExit, asyncio.CancelledError):
                    # 클라이언트가 연결을 끊은 경우
                    logger.debug("[stream_final_answer] Client disconnected while sending done event")
                    raise
                except Exception as done_error:
                    logger.error(f"[stream_final_answer] Failed to send done event: {done_error}", exc_info=True)
                    # done 이벤트 전송 실패 시 최소한의 done 이벤트라도 전송 시도
                    if not done_event_sent:
                        try:
                            minimal_done = {"type": "done", "timestamp": datetime.now().isoformat()}
                            yield format_sse_event(minimal_done)
                            logger.debug("[stream_final_answer] Minimal done event sent as fallback")
                        except Exception:
                            logger.error("[stream_final_answer] Failed to send minimal done event", exc_info=True)
            except (TypeError, AttributeError) as e:
                # 구버전 API 폴백: version="v2" 파라미터 사용
                logger.warning(f"[_process_stream_events] 최신 API 실패, 구버전 API(version='v2') 시도: {e}")
                try:
                    logger.info(f"[_process_stream_events] astream_events(version='v2') 호출 시작")
                    async for event in self.workflow_service.app.astream_events(
                        initial_state,
                        config,
                        version="v2"
                    ):
                        logger.info(f"[_process_stream_events] ✅ 이벤트 수신 (v2): event_type={event.get('event', 'N/A')}, name={event.get('name', 'N/A')}")
                        # 동일한 이벤트 처리 로직 적용 (위의 로직 재사용)
                        # TODO: 이벤트 처리 로직을 함수로 분리하여 재사용
                        event_count += 1
                        # 기본 이벤트 처리...
                except Exception as ve2:
                    logger.error(f"[_process_stream_events] 구버전 API도 실패: {ve2}", exc_info=True)
                    # 🔥 ERR_INCOMPLETE_CHUNKED_ENCODING 방지: 예외 발생해도 done 이벤트 전송
                    # done 이벤트를 보낸 후에는 raise하지 않고 정상 종료 (스트림은 완료됨)
                    try:
                        error_event = self.event_builder.create_error_event(str(ve2))
                        yield format_sse_event(error_event)
                        minimal_done = {"type": "done", "timestamp": datetime.now().isoformat(), "error": str(ve2)}
                        yield format_sse_event(minimal_done)
                        logger.debug("[_process_stream_events] Error and done event sent after v2 API error")
                    except (GeneratorExit, asyncio.CancelledError):
                        # 클라이언트가 연결을 끊은 경우
                        raise
                    except Exception as yield_error:
                        logger.error(f"[_process_stream_events] Failed to send error/done event: {yield_error}")
                    # 예외는 로깅만 하고 raise하지 않음 (done 이벤트를 보냈으므로 스트림은 정상 종료)
            except Exception as e:
                logger.error(f"[_process_stream_events] ⚠️ astream_events() 실행 중 예상치 못한 오류: {e}", exc_info=True)
                # 🔥 ERR_INCOMPLETE_CHUNKED_ENCODING 방지: 예외 발생해도 done 이벤트 전송
                # done 이벤트를 보낸 후에는 raise하지 않고 정상 종료 (스트림은 완료됨)
                try:
                    error_event = self.event_builder.create_error_event(str(e))
                    yield format_sse_event(error_event)
                    minimal_done = {"type": "done", "timestamp": datetime.now().isoformat(), "error": str(e)}
                    yield format_sse_event(minimal_done)
                    logger.debug("[_process_stream_events] Error and done event sent after unexpected error")
                except (GeneratorExit, asyncio.CancelledError):
                    # 클라이언트가 연결을 끊은 경우
                    raise
                except Exception as yield_error:
                    logger.error(f"[_process_stream_events] Failed to send error/done event: {yield_error}")
                # 예외는 로깅만 하고 raise하지 않음 (done 이벤트를 보냈으므로 스트림은 정상 종료)
                # 상위에서 예외를 처리할 필요가 있으면 로깅된 예외 정보를 사용
        
        except asyncio.CancelledError:
            logger.debug("[stream_final_answer] Stream cancelled (client disconnected)")
            # 🔥 개선: ERR_INCOMPLETE_CHUNKED_ENCODING 방지를 위해 done 이벤트 전송 시도
            try:
                done_event = self.event_builder.create_done_event("", {})
                yield format_sse_event(done_event)
            except Exception:
                pass  # 이미 연결이 끊어진 경우 무시
            raise
        except GeneratorExit:
            # GeneratorExit는 제너레이터가 이미 종료된 상태이므로 yield를 시도하면 안 됨
            logger.debug("[stream_final_answer] Generator exit (client disconnected)")
            raise
        except Exception as e:
            logger.error(f"[stream_final_answer] Unexpected error: {e}", exc_info=True)
            # 🔥 개선: ERR_INCOMPLETE_CHUNKED_ENCODING 방지를 위해 done 이벤트 전송 시도
            try:
                error_event = self.event_builder.create_error_event(str(e))
                yield format_sse_event(error_event)
                done_event = self.event_builder.create_done_event("", {})
                yield format_sse_event(done_event)
            except Exception:
                pass  # 이미 연결이 끊어진 경우 무시
            raise
        finally:
            callback_monitoring_active = False
            if callback_task:
                callback_task.cancel()
                try:
                    await callback_task
                except asyncio.CancelledError:
                    pass
            
            if callback_handler and hasattr(callback_handler, 'get_stats'):
                stats = callback_handler.get_stats()
                logger.info(
                    f"[stream_final_answer] Callback stats: "
                    f"chunks={callback_chunks_received}, "
                    f"total_chunks={stats.get('total_chunks', 0)}, "
                    f"streaming_active={stats.get('streaming_active', False)}"
                )
    
    
    async def _get_final_metadata(
        self,
        config: Dict[str, Any],
        initial_state: Dict[str, Any],
        message: str,
        full_answer: str,
        session_id: str
    ) -> Dict[str, Any]:
        """최종 메타데이터 가져오기"""
        try:
            final_state = await asyncio.wait_for(
                self.workflow_service.app.aget_state(config),
                timeout=StreamingConstants.STATE_TIMEOUT
            )
            if final_state and final_state.values:
                state_values = final_state.values
                logger.debug(
                    f"[stream_final_answer] State retrieved: "
                    f"answer_length={len(state_values.get('answer', ''))}, "
                    f"full_answer_length={len(full_answer)}"
                )
                
                # sources 추출 강화 (여러 위치에서 확인, 우선순위 적용)
                sources_from_top = state_values.get("sources", [])
                sources_from_common = (state_values.get("common", {}).get("sources") if isinstance(state_values.get("common"), dict) else None) or []
                sources_from_metadata = (state_values.get("metadata", {}).get("sources") if isinstance(state_values.get("metadata"), dict) else None) or []
                # 우선순위: top > common > metadata
                sources = sources_from_top if sources_from_top else (sources_from_common if sources_from_common else sources_from_metadata)
                if not sources:
                    sources = []
                
                # legal_references 추출 강화 (여러 위치에서 확인, 우선순위 적용)
                legal_refs_from_top = state_values.get("legal_references", [])
                legal_refs_from_common = (state_values.get("common", {}).get("legal_references") if isinstance(state_values.get("common"), dict) else None) or []
                legal_refs_from_metadata = (state_values.get("metadata", {}).get("legal_references") if isinstance(state_values.get("metadata"), dict) else None) or []
                # 우선순위: top > common > metadata
                legal_references = legal_refs_from_top if legal_refs_from_top else (legal_refs_from_common if legal_refs_from_common else legal_refs_from_metadata)
                if not legal_references:
                    legal_references = []
                
                # sources_detail 추출 강화 (여러 위치에서 확인, 우선순위 적용)
                sources_detail_from_top = state_values.get("sources_detail", [])
                sources_detail_from_common = (state_values.get("common", {}).get("sources_detail") if isinstance(state_values.get("common"), dict) else None) or []
                sources_detail_from_metadata = (state_values.get("metadata", {}).get("sources_detail") if isinstance(state_values.get("metadata"), dict) else None) or []
                
                # 상세 로깅: 각 위치에서 sources_detail 확인
                logger.info(
                    f"[_get_final_metadata] sources_detail 추출 시도: "
                    f"top={len(sources_detail_from_top) if isinstance(sources_detail_from_top, list) else 'not_list'}, "
                    f"common={len(sources_detail_from_common) if isinstance(sources_detail_from_common, list) else 'not_list'}, "
                    f"metadata={len(sources_detail_from_metadata) if isinstance(sources_detail_from_metadata, list) else 'not_list'}"
                )
                
                # state_values의 모든 키 확인 (디버깅용)
                state_keys = list(state_values.keys())
                logger.debug(
                    f"[_get_final_metadata] State keys: {state_keys[:30]}... "
                    f"(total: {len(state_keys)})"
                )
                
                # common과 metadata 구조 확인
                if isinstance(state_values.get("common"), dict):
                    common_keys = list(state_values["common"].keys())
                    logger.debug(f"[_get_final_metadata] Common keys: {common_keys[:20]}...")
                if isinstance(state_values.get("metadata"), dict):
                    metadata_keys = list(state_values["metadata"].keys())
                    logger.debug(f"[_get_final_metadata] Metadata keys: {metadata_keys[:20]}...")
                
                # 우선순위: top > common > metadata
                sources_detail = sources_detail_from_top if sources_detail_from_top else (sources_detail_from_common if sources_detail_from_common else sources_detail_from_metadata)
                if not sources_detail:
                    sources_detail = []
                
                sources_source = "top" if sources_from_top else ("common" if sources_from_common else ("metadata" if sources_from_metadata else "none"))
                sources_detail_source = "top" if sources_detail_from_top else ("common" if sources_detail_from_common else ("metadata" if sources_detail_from_metadata else "none"))
                
                logger.info(
                    f"[_get_final_metadata] Sources extraction result: "
                    f"state_sources={len(sources)} (from {sources_source}), "
                    f"state_legal_references={len(legal_references)}, "
                    f"state_sources_detail={len(sources_detail)} (from {sources_detail_source})"
                )
                
                # sources_detail이 비어있으면 상세 로깅
                if not sources_detail:
                    logger.warning(
                        f"[_get_final_metadata] ⚠️ sources_detail이 비어있습니다. "
                        f"state_values 구조 확인 필요. "
                        f"top_type={type(sources_detail_from_top).__name__}, "
                        f"common_type={type(sources_detail_from_common).__name__}, "
                        f"metadata_type={type(sources_detail_from_metadata).__name__}"
                    )
                
                if not sources_detail:
                    structured_docs_from_top = state_values.get("structured_documents")
                    structured_docs_from_search = (state_values.get("search", {}).get("structured_documents") if isinstance(state_values.get("search"), dict) else None)
                    structured_docs_from_common = (state_values.get("common", {}).get("search", {}).get("structured_documents") if isinstance(state_values.get("common"), dict) and isinstance(state_values["common"].get("search"), dict) else None)
                    
                    structured_docs = (
                        structured_docs_from_top or
                        structured_docs_from_search or
                        structured_docs_from_common
                    )
                    
                    prompt_used_docs = []
                    if structured_docs and isinstance(structured_docs, dict):
                        documents_in_prompt = structured_docs.get("documents", [])
                        if documents_in_prompt and isinstance(documents_in_prompt, list):
                            min_relevance_score = 0.80
                            filtered_docs = []
                            for doc in documents_in_prompt:
                                if not isinstance(doc, dict):
                                    continue
                                
                                relevance_score = (
                                    doc.get("relevance_score") or
                                    doc.get("score") or
                                    doc.get("final_weighted_score") or
                                    doc.get("similarity") or
                                    0.0
                                )
                                
                                if relevance_score >= min_relevance_score:
                                    filtered_docs.append(doc)
                                else:
                                    logger.debug(
                                        f"[stream_final_answer] Document filtered out due to low relevance: "
                                        f"score={relevance_score:.3f} < {min_relevance_score}, "
                                        f"doc_id={doc.get('doc_id') or doc.get('id') or 'unknown'}"
                                    )
                            
                            prompt_used_docs = filtered_docs
                            logger.info(
                                f"[stream_final_answer] Filtered documents by relevance (>= {min_relevance_score}): "
                                f"{len(prompt_used_docs)}/{len(documents_in_prompt)} documents passed"
                            )
                            
                            if not prompt_used_docs and documents_in_prompt:
                                logger.warning(
                                    f"[stream_final_answer] No documents with relevance >= {min_relevance_score} found. "
                                    f"All {len(documents_in_prompt)} documents were filtered out. "
                                    f"Consider lowering the threshold or checking document quality."
                                )
                    
                    retrieved_docs_from_top = state_values.get("retrieved_docs")
                    retrieved_docs_from_search = (state_values.get("search", {}).get("retrieved_docs") if isinstance(state_values.get("search"), dict) else None)
                    retrieved_docs_from_common = (state_values.get("common", {}).get("search", {}).get("retrieved_docs") if isinstance(state_values.get("common"), dict) and isinstance(state_values["common"].get("search"), dict) else None)
                    retrieved_docs_from_metadata = (state_values.get("metadata", {}).get("retrieved_docs") if isinstance(state_values.get("metadata"), dict) else None)
                    retrieved_docs_from_metadata_search = (state_values.get("metadata", {}).get("search", {}).get("retrieved_docs") if isinstance(state_values.get("metadata"), dict) and isinstance(state_values["metadata"].get("search"), dict) else None)
                    
                    all_retrieved_docs = (
                        retrieved_docs_from_top or
                        retrieved_docs_from_search or
                        retrieved_docs_from_common or
                        retrieved_docs_from_metadata or
                        retrieved_docs_from_metadata_search
                    )
                    
                    logger.info(
                        f"[_get_final_metadata] retrieved_docs 확인: "
                        f"top={len(retrieved_docs_from_top) if isinstance(retrieved_docs_from_top, list) else 0}, "
                        f"search={len(retrieved_docs_from_search) if isinstance(retrieved_docs_from_search, list) else 0}, "
                        f"common={len(retrieved_docs_from_common) if isinstance(retrieved_docs_from_common, list) else 0}, "
                        f"metadata={len(retrieved_docs_from_metadata) if isinstance(retrieved_docs_from_metadata, list) else 0}, "
                        f"all_retrieved_docs={len(all_retrieved_docs) if isinstance(all_retrieved_docs, list) else 0}"
                    )
                    
                    if prompt_used_docs:
                        retrieved_docs = prompt_used_docs
                        logger.info(
                            f"[stream_final_answer] Using {len(retrieved_docs)} documents from structured_documents "
                            f"(actual documents used in prompt) instead of all {len(all_retrieved_docs) if all_retrieved_docs else 0} retrieved_docs"
                        )
                    else:
                        retrieved_docs = all_retrieved_docs
                        if retrieved_docs:
                            logger.debug(
                                f"[stream_final_answer] structured_documents not found, "
                                f"using all {len(retrieved_docs)} retrieved_docs"
                            )
                    
                    if prompt_used_docs:
                        retrieved_docs_source = "structured_documents"
                    else:
                        retrieved_docs_source = (
                            "top" if retrieved_docs_from_top else
                            ("search" if retrieved_docs_from_search else
                            ("common.search" if retrieved_docs_from_common else
                            ("metadata" if retrieved_docs_from_metadata else
                            ("metadata.search" if retrieved_docs_from_metadata_search else "none"))))
                        )
                    
                    logger.debug(
                        f"[stream_final_answer] Retrieved docs check: "
                        f"count={len(retrieved_docs) if retrieved_docs else 0}, "
                        f"source={retrieved_docs_source}"
                    )
                    
                    # state에 retrieved_docs가 없으면 global cache에서 가져오기 시도
                    if not retrieved_docs:
                        logger.debug(f"[stream_final_answer] State has no retrieved_docs, attempting to restore from global cache")
                        try:
                            # 여러 경로 시도
                            _global_search_results_cache = None
                            import_errors = []
                            
                            # 경로 1: core.shared.wrappers.node_wrappers (가장 일반적)
                            try:
                                from core.shared.wrappers.node_wrappers import _global_search_results_cache
                                logger.debug(f"[stream_final_answer] Successfully imported _global_search_results_cache from core.shared.wrappers.node_wrappers")
                            except (ImportError, AttributeError) as e:
                                import_errors.append(f"core.shared.wrappers.node_wrappers: {e}")
                                
                                # 경로 2: lawfirm_langgraph.core.shared.wrappers.node_wrappers
                                try:
                                    from lawfirm_langgraph.core.shared.wrappers.node_wrappers import _global_search_results_cache
                                    logger.debug(f"[stream_final_answer] Successfully imported _global_search_results_cache from lawfirm_langgraph.core.shared.wrappers.node_wrappers")
                                except (ImportError, AttributeError) as e2:
                                    import_errors.append(f"lawfirm_langgraph.core.shared.wrappers.node_wrappers: {e2}")
                                    
                                    # 경로 3: core.agents.node_wrappers
                                    try:
                                        from core.agents.node_wrappers import _global_search_results_cache
                                        logger.debug(f"[stream_final_answer] Successfully imported _global_search_results_cache from core.agents.node_wrappers")
                                    except (ImportError, AttributeError) as e3:
                                        import_errors.append(f"core.agents.node_wrappers: {e3}")
                            
                            if _global_search_results_cache is not None:
                                logger.debug(f"[stream_final_answer] Global cache exists: {type(_global_search_results_cache).__name__}, keys: {list(_global_search_results_cache.keys()) if isinstance(_global_search_results_cache, dict) else 'N/A'}")
                                
                                cached_structured_docs = None
                                if isinstance(_global_search_results_cache, dict) and "search" in _global_search_results_cache:
                                    cached_search = _global_search_results_cache["search"]
                                    if isinstance(cached_search, dict):
                                        cached_structured_docs = cached_search.get("structured_documents")
                                
                                if cached_structured_docs and isinstance(cached_structured_docs, dict):
                                    cached_prompt_docs = cached_structured_docs.get("documents", [])
                                    if cached_prompt_docs and isinstance(cached_prompt_docs, list) and len(cached_prompt_docs) > 0:
                                        min_relevance_score = 0.80
                                        filtered_cached_docs = []
                                        for doc in cached_prompt_docs:
                                            if not isinstance(doc, dict):
                                                continue
                                            
                                            relevance_score = (
                                                doc.get("relevance_score") or
                                                doc.get("score") or
                                                doc.get("final_weighted_score") or
                                                doc.get("similarity") or
                                                0.0
                                            )
                                            
                                            if relevance_score >= min_relevance_score:
                                                filtered_cached_docs.append(doc)
                                        
                                        if filtered_cached_docs:
                                            retrieved_docs = filtered_cached_docs
                                            retrieved_docs_source = "global_cache.structured_documents"
                                            logger.info(
                                                f"[stream_final_answer] Restored {len(retrieved_docs)}/{len(cached_prompt_docs)} documents "
                                                f"from global cache structured_documents (filtered by relevance >= {min_relevance_score})"
                                            )
                                        else:
                                            logger.warning(
                                                f"[stream_final_answer] All {len(cached_prompt_docs)} documents from global cache "
                                                f"were filtered out (relevance < {min_relevance_score})"
                                            )
                                
                                if not retrieved_docs:
                                    if isinstance(_global_search_results_cache, dict) and "search" in _global_search_results_cache:
                                        cached_search = _global_search_results_cache["search"]
                                        if isinstance(cached_search, dict):
                                            cached_docs = cached_search.get("retrieved_docs", [])
                                            if isinstance(cached_docs, list) and len(cached_docs) > 0:
                                                retrieved_docs = cached_docs
                                                retrieved_docs_source = "global_cache.search.retrieved_docs"
                                                logger.debug(f"[stream_final_answer] Restored {len(retrieved_docs)} retrieved_docs from global cache search group")
                                            else:
                                                cached_merged = cached_search.get("merged_documents", [])
                                                if isinstance(cached_merged, list) and len(cached_merged) > 0:
                                                    retrieved_docs = cached_merged
                                                    retrieved_docs_source = "global_cache.search.merged_documents"
                                                    logger.debug(f"[stream_final_answer] Restored {len(retrieved_docs)} merged_documents from global cache search group")
                                
                                if not retrieved_docs and isinstance(_global_search_results_cache, dict):
                                    cached_docs = _global_search_results_cache.get("retrieved_docs", [])
                                    if isinstance(cached_docs, list) and len(cached_docs) > 0:
                                        retrieved_docs = cached_docs
                                        retrieved_docs_source = "global_cache.top"
                                        logger.debug(f"[stream_final_answer] Restored {len(retrieved_docs)} retrieved_docs from global cache top level")
                                
                                if not retrieved_docs:
                                    logger.debug(f"[stream_final_answer] Global cache exists but no retrieved_docs found in it")
                            else:
                                logger.debug(f"[stream_final_answer] Global cache is None after import attempts")
                                
                        except Exception as e:
                            logger.warning(f"[stream_final_answer] Failed to access global cache: {e}", exc_info=True)
                    
                    logger.debug(
                        f"[stream_final_answer] Attempting to extract sources: "
                        f"retrieved_docs_count={len(retrieved_docs) if retrieved_docs else 0}, "
                        f"retrieved_docs_source={retrieved_docs_source}, "
                        f"sources_extractor={self.sources_extractor is not None}"
                    )
                    
                    # 개선: sources, legal_references, sources_detail이 없으면 retrieved_docs에서 추출 시도
                    if retrieved_docs and self.sources_extractor:
                        try:
                            # 🔥 retrieved_docs 정규화 (type 통합) - 추출 전에 정규화
                            from lawfirm_langgraph.core.utils.document_type_normalizer import normalize_documents_type
                            retrieved_docs = normalize_documents_type(retrieved_docs) if retrieved_docs else []
                            
                            # retrieved_docs를 state_values에 임시로 추가하여 추출 함수가 사용할 수 있게 함
                            temp_state = {**state_values, "retrieved_docs": retrieved_docs}
                            
                            # sources 추출 시 예외가 발생해도 스트리밍이 중단되지 않도록 각각 try-except 처리
                            # sources가 없을 때만 추출 시도
                            if not sources:
                                try:
                                    sources_data = self.sources_extractor._extract_sources(temp_state)
                                    if sources_data:
                                        sources = sources_data
                                        state_values["sources"] = sources_data
                                        logger.info(f"[stream_final_answer] ✅ Extracted {len(sources)} sources from retrieved_docs")
                                except Exception as e:
                                    logger.warning(f"[stream_final_answer] Failed to extract sources: {e}", exc_info=True)
                            
                            # legal_references가 없을 때만 추출 시도
                            if not legal_references:
                                try:
                                    legal_references_data = self.sources_extractor._extract_legal_references(temp_state)
                                    if legal_references_data:
                                        legal_references = legal_references_data
                                        logger.info(f"[stream_final_answer] ✅ Extracted {len(legal_references)} legal_references from retrieved_docs")
                                except Exception as e:
                                    logger.warning(f"[stream_final_answer] Failed to extract legal_references: {e}", exc_info=True)
                            
                            # sources_by_type이 없을 때만 retrieved_docs에서 생성 시도
                            sources_by_type = temp_state.get("sources_by_type")
                            if not sources_by_type:
                                retrieved_docs = temp_state.get("retrieved_docs", [])
                                if retrieved_docs and isinstance(retrieved_docs, list):
                                    try:
                                        sources_by_type = self.sources_extractor._generate_sources_by_type_from_retrieved_docs(retrieved_docs)
                                        temp_state["sources_by_type"] = sources_by_type
                                        logger.info(f"[stream_final_answer] ✅ Generated sources_by_type from {len(retrieved_docs)} retrieved_docs")
                                    except Exception as e:
                                        logger.warning(f"[stream_final_answer] Failed to generate sources_by_type from retrieved_docs: {e}", exc_info=True)
                            
                            logger.debug(
                                f"[stream_final_answer] Sources extraction result: "
                                f"sources={len(sources) if sources else 0}, "
                                f"legal_references={len(legal_references) if legal_references else 0}, "
                                f"sources_detail={len(sources_detail) if sources_detail else 0}"
                            )
                        except Exception as e:
                            logger.warning(f"[stream_final_answer] Failed to extract sources from retrieved_docs: {e}", exc_info=True)
                    else:
                        logger.debug(
                            f"[stream_final_answer] Skipping sources extraction from state: "
                            f"retrieved_docs={retrieved_docs is not None and len(retrieved_docs) > 0}, "
                            f"sources_extractor={self.sources_extractor is not None}"
                        )
                        
                        # state에 retrieved_docs가 없으면 extract_from_message_metadata와 extract_from_state를 시도
                        if not retrieved_docs and self.sources_extractor and session_id:
                            try:
                                logger.debug(f"[stream_final_answer] Attempting to extract sources from session: session_id={session_id}")
                                
                                # 먼저 메시지 metadata에서 가져오기 시도
                                message_id = state_values.get("metadata", {}).get("message_id") if isinstance(state_values.get("metadata"), dict) else None
                                session_sources = await self.sources_extractor.extract_from_message_metadata(session_id, message_id)
                                
                                # 없으면 state에서 가져오기 시도
                                if not any(session_sources.values()):
                                    logger.debug(f"[stream_final_answer] No sources in message metadata, trying extract_from_state")
                                    session_sources = await self.sources_extractor.extract_from_state(session_id)
                                
                                if session_sources:
                                    session_sources_list = session_sources.get("sources", [])
                                    session_legal_refs = session_sources.get("legal_references", [])
                                    session_sources_detail = session_sources.get("sources_detail", [])
                                    
                                    logger.debug(
                                        f"[stream_final_answer] Sources extracted from session: "
                                        f"sources={len(session_sources_list)}, "
                                        f"legal_references={len(session_legal_refs)}, "
                                        f"sources_detail={len(session_sources_detail)}"
                                    )
                                    
                                    if session_sources_list:
                                        sources = session_sources_list
                                    if session_legal_refs:
                                        legal_references = session_legal_refs
                                    if session_sources_detail:
                                        sources_detail = session_sources_detail
                            except Exception as e:
                                logger.warning(f"[stream_final_answer] Failed to extract sources from session: {e}", exc_info=True)
                
                # related_questions 추출 강화
                related_questions = (
                    (state_values.get("metadata", {}).get("related_questions") if isinstance(state_values.get("metadata"), dict) else None) or
                    []
                )
                if not related_questions and self.extract_related_questions_fn:
                    try:
                        related_questions = await self.extract_related_questions_fn(
                            state_values, initial_state, message, full_answer, session_id
                        )
                    except Exception as e:
                        logger.warning(f"[stream_final_answer] Failed to extract related_questions: {e}", exc_info=True)
                        related_questions = []
                
                # llm_validation_result 추출 강화
                llm_validation_result = (
                    (state_values.get("metadata", {}).get("llm_validation_result", {}) if isinstance(state_values.get("metadata"), dict) else {}) or
                    {}
                )
                
                # message_id 추출 (프론트엔드에서 메시지 매칭에 사용)
                message_id = (
                    (state_values.get("metadata", {}).get("message_id") if isinstance(state_values.get("metadata"), dict) else None) or
                    None
                )
                
                # 타입별 그룹화 (새로운 기능) - 판례의 참조 법령 포함
                sources_by_type = self._generate_sources_by_type(sources_detail)
                
                final_metadata = {
                    "sources_by_type": sources_by_type,  # 유일한 필요한 필드
                    "related_questions": related_questions,
                    "llm_validation": llm_validation_result if llm_validation_result else None,
                    "message_id": message_id,  # 프론트엔드에서 메시지 매칭에 사용
                    # 하위 호환성을 위해 deprecated 필드도 포함 (점진적 제거)
                    "sources": sources,  # deprecated: sources_by_type에서 재구성 가능
                    "legal_references": legal_references,  # deprecated: sources_by_type에서 재구성 가능
                    "sources_detail": sources_detail,  # deprecated: sources_by_type에서 재구성 가능
                }
                
                logger.info(
                    f"[stream_final_answer] ✅ Final metadata extracted: "
                    f"sources={len(final_metadata['sources'])}, "
                    f"legal_references={len(final_metadata['legal_references'])}, "
                    f"sources_detail={len(final_metadata['sources_detail'])}, "
                    f"related_questions={len(final_metadata['related_questions'])}, "
                    f"sources_by_type={bool(final_metadata.get('sources_by_type'))}"
                )
                
                return final_metadata
        except asyncio.TimeoutError:
            logger.warning(f"[stream_final_answer] Timeout getting state, using empty metadata")
        except Exception as e:
            logger.warning(f"[stream_final_answer] Error getting state: {e}")
        
        return {}
    
    def _generate_sources_by_type(self, sources_detail: List[Dict[str, Any]]) -> Optional[Dict[str, List[Any]]]:
        """
        sources_by_type 생성 (판례의 참조 법령 포함)
        예외 발생 시에도 안전하게 기본 구조 반환
        """
        if not sources_detail or not self.sources_extractor:
            return None
        
        try:
            sources_by_type = self.sources_extractor._get_sources_by_type_with_reference_statutes(sources_detail)
            logger.debug(f"[stream_final_answer] Generated sources_by_type with reference statutes: {len(sources_by_type.get('statute_article', []))} statutes")
            return sources_by_type
        except Exception as e:
            logger.warning(f"[stream_final_answer] Failed to generate sources_by_type: {e}", exc_info=True)
            # 예외 발생 시 기본 sources_by_type 생성 (참조 법령 없이)
            try:
                sources_by_type = self.sources_extractor._get_sources_by_type(sources_detail) if sources_detail else DEFAULT_SOURCES_BY_TYPE.copy()
                return sources_by_type
            except Exception as fallback_error:
                logger.error(f"[stream_final_answer] Failed to generate fallback sources_by_type: {fallback_error}", exc_info=True)
                return DEFAULT_SOURCES_BY_TYPE.copy()
    
    def _process_callback_queue_chunks(
        self,
        chunk_output_queue: asyncio.Queue,
        processed_callback_chunks: set,
        full_answer: str
    ) -> tuple[int, str, list[str]]:
        """
        콜백 큐에서 청크를 가져와 처리
        
        Returns:
            (callback_chunks_received, updated_full_answer, chunks_to_yield)
            chunks_to_yield: 스트림으로 전송할 청크 리스트
        """
        callback_chunks_received = 0
        updated_full_answer = full_answer
        chunks_to_yield = []
        
        if not chunk_output_queue:
            return callback_chunks_received, updated_full_answer, chunks_to_yield
        
        try:
            while True:
                try:
                    chunk_data = chunk_output_queue.get_nowait()
                    if chunk_data and chunk_data.get("type") == StreamingConstants.CALLBACK_CHUNK_TYPE:
                        content = chunk_data.get("content", "")
                        chunk_index = chunk_data.get("chunk_index", 0)
                        chunk_key = f"{chunk_index}_{content[:10]}"
                        
                        if chunk_key not in processed_callback_chunks and content:
                            processed_callback_chunks.add(chunk_key)
                            callback_chunks_received += 1
                            updated_full_answer += content
                            chunks_to_yield.append(content)
                            
                            if callback_chunks_received <= StreamingConstants.MAX_DEBUG_LOGS:
                                logger.info(
                                    f"[stream_final_answer] ✅ Callback chunk #{callback_chunks_received}: "
                                    f"length={len(content)}, content={content[:50]}..."
                                )
                except asyncio.QueueEmpty:
                    break
        except Exception as e:
            logger.debug(f"[stream_final_answer] Error checking callback output queue: {e}")
        
        return callback_chunks_received, updated_full_answer, chunks_to_yield
    
    def _handle_on_chain_start_event(
        self,
        event_name: str,
        answer_generation_started: bool,
        last_node_name: Optional[str]
    ) -> tuple[bool, Optional[str]]:
        """
        on_chain_start 이벤트 처리
        
        Returns:
            (updated_answer_generation_started, updated_last_node_name)
        """
        node_name = event_name
        is_answer_node = self.node_filter.is_answer_generation_node(node_name)
        logger.info(
            f"[stream_final_answer] on_chain_start: "
            f"node_name={node_name}, "
            f"is_answer_generation_node={is_answer_node}"
        )
        
        if is_answer_node:
            answer_generation_started = True
            last_node_name = node_name
            logger.info(f"[stream_final_answer] ✅ 답변 생성 노드 시작: {node_name}, answer_generation_started=True")
        
        return answer_generation_started, last_node_name
    
    def _handle_on_chain_end_event(
        self,
        event_name: str,
        answer_generation_started: bool
    ) -> bool:
        """
        on_chain_end 이벤트 처리
        
        Returns:
            updated_answer_generation_started
        """
        node_name = event_name
        if self.node_filter.is_answer_completion_node(node_name):
            answer_generation_started = False
            logger.debug(f"[stream_final_answer] 답변 생성 노드 완료: {node_name}")
        
        return answer_generation_started
    
    def _handle_streaming_event(
        self,
        event_type: str,
        event_name: str,
        event_parent: Dict[str, Any],
        event_data: Dict[str, Any],
        answer_generation_started: bool,
        last_node_name: Optional[str],
        full_answer: str,
        stream_event_count: int
    ) -> tuple[bool, str, int, Optional[str]]:
        """
        스트리밍 이벤트 처리 (on_llm_stream, on_chat_model_stream)
        
        Returns:
            (should_continue, updated_full_answer, updated_stream_event_count, token_to_yield)
            should_continue: False면 continue, True면 계속 처리
            token_to_yield: None이 아니면 yield해야 할 토큰
        """
        # 분류 노드는 정상 동작이므로 조용히 건너뜀
        if self.node_filter.is_classification_node(event_name):
            self._classification_skip_count += 1
            return False, full_answer, stream_event_count, None
        
        # answer_generation_started가 False인 경우 로깅 후 건너뛰기
        if not answer_generation_started:
            # 로그 카운터를 사용하여 제한된 횟수만 로그 출력
            if self._skip_log_count < StreamingConstants.MAX_SKIP_LOGS:
                logger.warning(
                    f"[stream_final_answer] ⚠️ 답변 생성 노드가 시작되지 않아 건너뜀: "
                    f"event_name={event_name}, "
                    f"event_type={event_type}, "
                    f"last_node={last_node_name}, "
                    f"answer_generation_started={answer_generation_started}"
                )
                self._skip_log_count += 1
            return False, full_answer, stream_event_count, None
        
        # 타겟 노드 확인 및 토큰 추출
        if self.node_filter.is_target_node(event_name, event_parent, last_node_name):
            logger.info(f"[stream_final_answer] ✅ 타겟 노드 확인됨: {event_name}, 토큰 추출 시작")
            
            # 🔥 [END] 키워드가 이미 발견되었는지 확인
            # 이미 [END] 이후라면 더 이상 전송하지 않음
            if '[END]' in full_answer.upper():
                logger.debug(
                    f"[stream_final_answer] ⚠️ [END] 키워드가 이미 발견되어 추가 토큰 전송 중단"
                )
                return False, full_answer, stream_event_count, None
            
            token = self.token_extractor.extract_from_event(event_data)
            
            if token:
                stream_event_count += 1
                updated_full_answer = full_answer + token
                
                # 🔥 [END] 키워드 이후 내용 필터링
                # 대소문자 구분 없이 [END] 키워드 찾기
                end_keyword_pos = -1
                updated_full_answer_upper = updated_full_answer.upper()
                for keyword in ['[END]', '[END', 'END]']:
                    pos = updated_full_answer_upper.find(keyword.upper())
                    if pos != -1:
                        end_keyword_pos = pos
                        break
                
                if end_keyword_pos != -1:
                    # [END] 키워드가 발견되면 그 이후 내용은 제외
                    updated_full_answer = updated_full_answer[:end_keyword_pos].rstrip()
                    token = updated_full_answer[len(full_answer):] if len(updated_full_answer) > len(full_answer) else ""
                    logger.info(
                        f"[stream_final_answer] ✅ [END] 키워드 발견, 이후 내용 필터링 "
                        f"(위치: {end_keyword_pos}, 필터링된 토큰 길이: {len(token)})"
                    )
                
                if token:
                    logger.info(
                        f"[stream_final_answer] ✅ 토큰 전송 #{stream_event_count}: "
                        f"token_length={len(token)}, "
                        f"token_preview={token[:50]}..., "
                        f"full_answer_length={len(updated_full_answer)}"
                    )
                    return True, updated_full_answer, stream_event_count, token
                else:
                    # [END] 이후 내용만 있어서 필터링됨
                    logger.debug(
                        f"[stream_final_answer] ⚠️ [END] 이후 내용만 있어 토큰 전송 중단"
                    )
                    return False, updated_full_answer, stream_event_count, None
            else:
                logger.warning(
                    f"[stream_final_answer] ⚠️ 토큰 추출 실패: "
                    f"event_name={event_name}, "
                    f"event_data_keys={list(event_data.keys()) if isinstance(event_data, dict) else []}, "
                    f"event_data_type={type(event_data).__name__}"
                )
        else:
            logger.debug(
                f"[stream_final_answer] 타겟 노드가 아님 (필터링됨): "
                f"type={event_type}, name={event_name}, "
                f"parent={event_parent.get('name', '') if isinstance(event_parent, dict) else ''}, "
                f"last_node={last_node_name}, started={answer_generation_started}"
            )
        
        return True, full_answer, stream_event_count, None
    
    def _validate_and_augment_state(
        self,
        initial_state: Dict[str, Any],
        message: str,
        session_id: str
    ) -> Optional[str]:
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

