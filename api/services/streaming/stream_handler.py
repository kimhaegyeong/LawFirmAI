"""
스트리밍 처리 전용 클래스
"""
import json
import asyncio
import logging
import uuid
from typing import Dict, Any, Optional, AsyncGenerator, List

from .constants import StreamingConstants
from .event_builder import StreamEventBuilder
from .token_extractor import TokenExtractor
from .node_filter import NodeFilter
from api.utils.sse_formatter import format_sse_event

logger = logging.getLogger(__name__)

# 기본 sources_by_type 구조
DEFAULT_SOURCES_BY_TYPE = {
    "statute_article": [],
    "case_paragraph": [],
    "decision_paragraph": [],
    "interpretation_paragraph": []
}


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
            
            progress_event = self.event_builder.create_progress_event("답변 생성 중...")
            yield format_sse_event(progress_event)
            
            async for chunk in self._process_stream_events(
                initial_state, config, callback_handler, message, session_id
            ):
                yield chunk
                
        except GeneratorExit:
            logger.debug("[stream_final_answer] Client disconnected, closing stream")
            return
        except Exception as e:
            logger.error(f"Stream error: {e}", exc_info=True)
            try:
                error_event = self.event_builder.create_error_event(
                    f"[오류] 스트리밍 중 오류가 발생했습니다: {str(e)}",
                    type(e).__name__
                )
                yield format_sse_event(error_event)
            except GeneratorExit:
                logger.debug("[stream_final_answer] Client disconnected during error handling")
                return
            except Exception as yield_error:
                logger.error(f"Failed to yield error event: {yield_error}")
            # finally 블록에서 yield를 하면 제너레이터가 제대로 종료되지 않을 수 있으므로 제거
            # done 이벤트는 정상 종료 시에만 전송됨 (373줄)
    
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
        
        config = {"configurable": {"thread_id": session_id}}
        if callback_handler:
            config = self.workflow_service.get_config_with_callbacks(
                session_id=session_id,
                callbacks=[callback_handler]
            )
            logger.info(f"[stream_final_answer] ✅ Callbacks added to config: {len(config.get('callbacks', []))} callback(s)")
        else:
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
            async for event in self.workflow_service.app.astream_events(
                initial_state,
                config,
                version="v2"
            ):
                if chunk_output_queue:
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
                                        full_answer += content
                                        
                                        stream_event = self.event_builder.create_stream_event(
                                            content, source="callback"
                                        )
                                        yield format_sse_event(stream_event)
                                        
                                        if callback_chunks_received <= StreamingConstants.MAX_DEBUG_LOGS:
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
                
                if event_type == "on_chain_start":
                    node_name = event_name
                    if self.node_filter.is_answer_generation_node(node_name):
                        answer_generation_started = True
                        last_node_name = node_name
                        logger.debug(f"[stream_final_answer] 답변 생성 노드 시작: {node_name}")
                
                elif event_type == "on_chain_end":
                    node_name = event_name
                    if self.node_filter.is_answer_completion_node(node_name):
                        answer_generation_started = False
                        logger.debug(f"[stream_final_answer] 답변 생성 노드 완료: {node_name}")
                
                elif event_type in ["on_llm_stream", "on_chat_model_stream"]:
                    # 분류 노드는 정상 동작이므로 조용히 건너뜀
                    if self.node_filter.is_classification_node(event_name):
                        self._classification_skip_count += 1
                        continue
                    
                    if not answer_generation_started:
                        # 로그 카운터를 사용하여 제한된 횟수만 로그 출력
                        if self._skip_log_count < StreamingConstants.MAX_SKIP_LOGS:
                            logger.debug(f"[stream_final_answer] 답변 생성 노드가 시작되지 않아 건너뜀: {event_name}")
                            self._skip_log_count += 1
                        continue
                    
                    if self.node_filter.is_target_node(event_name, event_parent, last_node_name):
                        logger.debug(f"[stream_final_answer] 타겟 노드 확인됨: {event_name}, 토큰 추출 시작")
                        event_data = event.get("data", {})
                        token = self.token_extractor.extract_from_event(event_data)
                        
                        if token:
                            stream_event_count += 1
                            full_answer += token
                            logger.debug(
                                f"[stream_final_answer] 토큰 전송: "
                                f"token_length={len(token)}, "
                                f"token_preview={token[:50]}..., "
                                f"stream_event_count={stream_event_count}"
                            )
                            stream_event = self.event_builder.create_stream_event(token)
                            yield format_sse_event(stream_event)
                        else:
                            logger.debug(
                                f"[stream_final_answer] 토큰 추출 실패: "
                                f"token={token}, "
                                f"event_data_keys={list(event_data.keys()) if isinstance(event_data, dict) else []}"
                            )
                    else:
                        logger.debug(
                            f"[stream_final_answer] 타겟 노드가 아님 (필터링됨): "
                            f"type={event_type}, name={event_name}, "
                            f"parent={event_parent.get('name', '') if isinstance(event_parent, dict) else ''}, "
                            f"last_node={last_node_name}, started={answer_generation_started}"
                        )
                
                if event_type == "on_chain_end" and self.node_filter.is_answer_completion_node(event_name):
                    if chunk_output_queue:
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
                                            full_answer += content
                                            
                                            stream_event = self.event_builder.create_stream_event(
                                                content, source="callback"
                                            )
                                            yield format_sse_event(stream_event)
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
                    
                    await asyncio.sleep(StreamingConstants.STATE_RETRY_DELAY)
                    
                    final_metadata = await self._get_final_metadata(
                        config, initial_state, message, full_answer, session_id
                    )
                    
                    if final_metadata.get("llm_validation"):
                        validation_event = self.event_builder.create_validation_event(
                            final_metadata["llm_validation"]
                        )
                        yield format_sse_event(validation_event)
                    
                    final_event = self.event_builder.create_final_event("", final_metadata)
                    yield format_sse_event(final_event)
                    
                    done_event = self.event_builder.create_done_event(full_answer, final_metadata)
                    yield format_sse_event(done_event)
                    break
        
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
                
                sources = state_values.get("sources", [])
                legal_references = state_values.get("legal_references", [])
                sources_detail = state_values.get("sources_detail", [])
                
                logger.debug(
                    f"[stream_final_answer] Sources extraction check: "
                    f"state_sources={len(sources)}, "
                    f"state_legal_references={len(legal_references)}, "
                    f"state_sources_detail={len(sources_detail)}"
                )
                
                if not sources and not sources_detail:
                    retrieved_docs = state_values.get("retrieved_docs", [])
                    
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
                                
                                # search 그룹에서 확인
                                if isinstance(_global_search_results_cache, dict) and "search" in _global_search_results_cache:
                                    cached_search = _global_search_results_cache["search"]
                                    if isinstance(cached_search, dict):
                                        cached_docs = cached_search.get("retrieved_docs", [])
                                        if isinstance(cached_docs, list) and len(cached_docs) > 0:
                                            retrieved_docs = cached_docs
                                            logger.debug(f"[stream_final_answer] Restored {len(retrieved_docs)} retrieved_docs from global cache search group")
                                        else:
                                            cached_merged = cached_search.get("merged_documents", [])
                                            if isinstance(cached_merged, list) and len(cached_merged) > 0:
                                                retrieved_docs = cached_merged
                                                logger.debug(f"[stream_final_answer] Restored {len(retrieved_docs)} merged_documents from global cache search group")
                                
                                # 최상위 레벨에서 확인
                                if not retrieved_docs and isinstance(_global_search_results_cache, dict):
                                    cached_docs = _global_search_results_cache.get("retrieved_docs", [])
                                    if isinstance(cached_docs, list) and len(cached_docs) > 0:
                                        retrieved_docs = cached_docs
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
                        f"sources_extractor={self.sources_extractor is not None}"
                    )
                    
                    if retrieved_docs and self.sources_extractor:
                        try:
                            # retrieved_docs를 state_values에 임시로 추가하여 추출 함수가 사용할 수 있게 함
                            temp_state = {**state_values, "retrieved_docs": retrieved_docs}
                            
                            # sources 추출 시 예외가 발생해도 스트리밍이 중단되지 않도록 각각 try-except 처리
                            try:
                                sources_data = self.sources_extractor._extract_sources(temp_state)
                                if sources_data:
                                    sources = sources_data
                                    state_values["sources"] = sources_data
                            except Exception as e:
                                logger.warning(f"[stream_final_answer] Failed to extract sources: {e}", exc_info=True)
                            
                            try:
                                legal_references_data = self.sources_extractor._extract_legal_references(temp_state)
                                if legal_references_data:
                                    legal_references = legal_references_data
                            except Exception as e:
                                logger.warning(f"[stream_final_answer] Failed to extract legal_references: {e}", exc_info=True)
                            
                            try:
                                sources_detail_data = self.sources_extractor._extract_sources_detail(temp_state)
                                if sources_detail_data:
                                    sources_detail = sources_detail_data
                                    state_values["sources_detail"] = sources_detail_data
                            except Exception as e:
                                logger.warning(f"[stream_final_answer] Failed to extract sources_detail: {e}", exc_info=True)
                            
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
                
                related_questions = []
                if self.extract_related_questions_fn:
                    related_questions = await self.extract_related_questions_fn(
                        state_values, initial_state, message, full_answer, session_id
                    )
                
                llm_validation_result = state_values.get("metadata", {}).get("llm_validation_result", {})
                
                # 타입별 그룹화 (새로운 기능) - 판례의 참조 법령 포함
                sources_by_type = self._generate_sources_by_type(sources_detail)
                
                final_metadata = {
                    "sources_by_type": sources_by_type,  # 유일한 필요한 필드
                    "related_questions": related_questions,
                    "llm_validation": llm_validation_result if llm_validation_result else None,
                    # 하위 호환성을 위해 deprecated 필드도 포함 (점진적 제거)
                    "sources": sources,  # deprecated: sources_by_type에서 재구성 가능
                    "legal_references": legal_references,  # deprecated: sources_by_type에서 재구성 가능
                    "sources_detail": sources_detail,  # deprecated: sources_by_type에서 재구성 가능
                }
                
                logger.debug(
                    f"[stream_final_answer] Final metadata: "
                    f"sources={len(final_metadata['sources'])}, "
                    f"legal_references={len(final_metadata['legal_references'])}, "
                    f"sources_detail={len(final_metadata['sources_detail'])}, "
                    f"related_questions={len(final_metadata['related_questions'])}"
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

