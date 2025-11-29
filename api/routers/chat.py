"""
채팅 엔드포인트
"""
import json
import logging
import asyncio
import hashlib
import time
from collections import OrderedDict
from fastapi import APIRouter, HTTPException, Request, Depends
from fastapi.responses import StreamingResponse
from typing import AsyncGenerator, Optional, Dict, Any
from datetime import datetime

from api.schemas.chat import ChatRequest, ChatResponse, StreamingChatRequest, ContinueAnswerRequest, ContinueAnswerResponse
from api.services.chat_service import get_chat_service
from api.services.session_service import session_service
from api.services.ocr_service import extract_text_from_base64, is_ocr_available
from api.services.file_processor import process_file
from api.middleware.auth_middleware import require_auth
from api.middleware.rate_limit import limiter, is_rate_limit_enabled
from api.services.anonymous_quota_service import anonymous_quota_service
from api.routers.session import get_user_info
from api.utils.sse_formatter import format_sse_event
from api.config import get_api_config

router = APIRouter()
logger = logging.getLogger(__name__)


class StreamCache:
    """스트리밍 응답 캐시 관리 클래스"""
    
    def __init__(self, max_size: int = 100, ttl_seconds: int = 3600):
        self.cache: OrderedDict[str, Dict[str, Any]] = OrderedDict()
        self.max_size = max_size
        self.ttl_seconds = ttl_seconds
    
    def _generate_key(self, message: str, session_id: Optional[str] = None) -> str:
        """캐시 키 생성 - 메시지만 사용 (session_id 무시)"""
        # 메시지만을 기준으로 캐싱하여 LLM 비용 절감
        # 세션과 무관하게 같은 질문이면 같은 답변 반환
        key = hashlib.md5(message.encode('utf-8')).hexdigest()
        logger.debug(f"[StreamCache] Generated key for message: {message[:50]}... (session_id ignored) -> {key[:8]}...")
        return key
    
    def get(self, message: str, session_id: Optional[str] = None) -> Optional[Dict[str, Any]]:
        """캐시에서 응답 가져오기 - 메시지만 기준 (session_id 무시)"""
        # session_id는 무시하고 message만 사용
        key = self._generate_key(message)
        
        if key not in self.cache:
            logger.debug(f"[StreamCache] Cache MISS: key {key[:8]}... not found (cache size: {len(self.cache)})")
            return None
        
        entry = self.cache[key]
        
        # TTL 확인
        age = time.time() - entry['timestamp']
        if age > self.ttl_seconds:
            logger.debug(f"[StreamCache] Cache entry expired: key {key[:8]}... (age: {age:.1f}s, ttl: {self.ttl_seconds}s)")
            del self.cache[key]
            return None
        
        # LRU: 사용된 항목을 맨 뒤로 이동
        self.cache.move_to_end(key)
        
        logger.debug(f"[StreamCache] Cache HIT: key {key[:8]}... (age: {age:.1f}s, cache size: {len(self.cache)})")
        return entry['data']
    
    def set(self, message: str, content: str, metadata: Optional[Dict[str, Any]], 
            session_id: Optional[str] = None):
        """캐시에 응답 저장 - 메시지만 기준 (session_id 무시)"""
        # session_id는 무시하고 message만 사용
        key = self._generate_key(message)
        
        # 캐시 크기 제한
        if len(self.cache) >= self.max_size:
            # 가장 오래된 항목 제거 (FIFO)
            removed_key, _ = self.cache.popitem(last=False)
            logger.debug(f"[StreamCache] Cache full, removed oldest entry: {removed_key[:8]}...")
        
        self.cache[key] = {
            'data': {
                'content': content,
                'metadata': metadata or {}
            },
            'timestamp': time.time()
        }
        
        # LRU: 새 항목을 맨 뒤로 이동
        self.cache.move_to_end(key)
        
        logger.debug(
            f"[StreamCache] Cache SET: key {key[:8]}..., "
            f"content_length={len(content)}, "
            f"has_metadata={bool(metadata)}, "
            f"cache_size={len(self.cache)}"
        )
    
    def clear(self):
        """캐시 전체 삭제"""
        self.cache.clear()


# 전역 캐시 인스턴스 (지연 초기화)
_stream_cache_instance: Optional[StreamCache] = None


def get_stream_cache() -> Optional[StreamCache]:
    """스트리밍 캐시 인스턴스 가져오기"""
    global _stream_cache_instance
    config = get_api_config()
    
    if not config.enable_stream_cache:
        logger.debug("[StreamCache] Cache is disabled in config (enable_stream_cache=False)")
        return None
    
    if _stream_cache_instance is None:
        _stream_cache_instance = StreamCache(
            max_size=config.stream_cache_max_size,
            ttl_seconds=config.stream_cache_ttl_seconds
        )
        logger.info(
            f"[StreamCache] Cache initialized: max_size={config.stream_cache_max_size}, "
            f"ttl={config.stream_cache_ttl_seconds}s"
        )
    
    return _stream_cache_instance


def _process_file_and_image(
    message: str,
    file_base64: Optional[str] = None,
    filename: Optional[str] = None,
    image_base64: Optional[str] = None
) -> str:
    """파일 또는 이미지를 처리하여 최종 메시지 생성"""
    final_message = message
    
    if file_base64:
        try:
            logger.info("Processing file...")
            success, extracted_text, error_msg = process_file(file_base64, filename)
            if success and extracted_text:
                if final_message.strip():
                    final_message = f"{final_message}\n\n[참고 파일 텍스트]\n{extracted_text}"
                else:
                    final_message = f"다음 파일의 내용을 분석하고 질문에 답변해주세요:\n\n{extracted_text}"
                logger.info(f"File processing completed. Extracted {len(extracted_text)} characters")
            elif not success:
                logger.warning(f"File processing failed: {error_msg}")
                if final_message.strip():
                    final_message = f"{final_message}\n\n[경고: {error_msg}]"
                else:
                    final_message = f"[경고: {error_msg}]"
        except Exception as e:
            logger.error(f"File processing error: {e}", exc_info=True)
            if final_message.strip():
                final_message = f"{final_message}\n\n[경고: 파일 처리 중 오류가 발생했습니다: {str(e)}]"
            else:
                final_message = f"[경고: 파일 처리 중 오류가 발생했습니다: {str(e)}]"
    elif image_base64:
        try:
            if is_ocr_available():
                logger.info("Processing image with OCR...")
                extracted_text = extract_text_from_base64(image_base64)
                if extracted_text:
                    if final_message.strip():
                        final_message = f"{final_message}\n\n[참고 이미지 텍스트]\n{extracted_text}"
                    else:
                        final_message = f"다음 이미지의 내용을 분석하고 질문에 답변해주세요:\n\n{extracted_text}"
                    logger.info(f"OCR completed. Extracted {len(extracted_text)} characters")
                else:
                    logger.warning("OCR did not extract any text from image")
            else:
                logger.warning("OCR service is not available")
        except Exception as e:
            logger.error(f"OCR processing error: {e}", exc_info=True)
            if final_message.strip():
                final_message = f"{final_message}\n\n[경고: 이미지에서 텍스트를 추출하는 중 오류가 발생했습니다: {str(e)}]"
            else:
                final_message = f"[경고: 이미지에서 텍스트를 추출하는 중 오류가 발생했습니다: {str(e)}]"
    
    return final_message


def _maybe_generate_session_title(session_id: str):
    """세션에 제목이 없고 첫 번째 대화라면 제목 생성 (비동기)"""
    session = session_service.get_session(session_id)
    if session and not session.get("title"):
        messages = session_service.get_messages(session_id)
        if len(messages) == 2:
            asyncio.create_task(
                asyncio.to_thread(
                    session_service.generate_session_title,
                    session_id
                )
            )


def _create_sources_event(metadata: dict, message_id: Optional[str] = None) -> dict:
    """sources 이벤트 생성 (related_questions 제외, sources_by_type 포함)"""
    try:
        sources_detail = metadata.get("sources_detail", [])
        
        # sources_by_type이 없으면 생성 (판례의 참조 법령 포함)
        sources_by_type = metadata.get("sources_by_type")
        if not sources_by_type and sources_detail:
            try:
                from api.services.chat_service import get_chat_service
                chat_service = get_chat_service()
                if chat_service and hasattr(chat_service, 'sources_extractor') and chat_service.sources_extractor:
                    sources_by_type = chat_service.sources_extractor._get_sources_by_type_with_reference_statutes(sources_detail)
                    logger.debug(f"[_create_sources_event] Generated sources_by_type with reference statutes: {len(sources_by_type.get('statutes_articles', []))} statutes")
            except Exception as e:
                logger.warning(f"[_create_sources_event] Failed to generate sources_by_type: {e}", exc_info=True)
                # 예외 발생 시 기본 sources_by_type 생성 (참조 법령 없이)
                try:
                    chat_service = get_chat_service()
                    if chat_service and hasattr(chat_service, 'sources_extractor') and chat_service.sources_extractor:
                        from api.utils.source_type_mapper import get_default_sources_by_type
                        sources_by_type = chat_service.sources_extractor._get_sources_by_type(sources_detail) if sources_detail else get_default_sources_by_type()
                    else:
                        sources_by_type = None
                except Exception as fallback_error:
                    logger.error(f"[_create_sources_event] Failed to generate fallback sources_by_type: {fallback_error}", exc_info=True)
                    sources_by_type = None
        # sources_by_type이 이미 있는 경우에도 참조 법령 추가 (중복 체크)
        elif sources_by_type and sources_detail:
            try:
                from api.services.chat_service import get_chat_service
                chat_service = get_chat_service()
                if chat_service and hasattr(chat_service, 'sources_extractor') and chat_service.sources_extractor:
                    extracted_statutes = chat_service.sources_extractor._extract_statutes_from_reference_clauses(sources_detail)
                    
                    if extracted_statutes:
                        existing_statutes = sources_by_type.get("statutes_articles", [])
                        existing_keys = {
                            f"{s.get('statute_name', '')}_{s.get('article_no', '')}_{s.get('clause_no', '')}_{s.get('item_no', '')}"
                            for s in existing_statutes if isinstance(s, dict)
                        }
                        
                        for statute in extracted_statutes:
                            statute_key = f"{statute.get('statute_name', '')}_{statute.get('article_no', '')}_{statute.get('clause_no', '')}_{statute.get('item_no', '')}"
                            if statute_key not in existing_keys:
                                existing_statutes.append(statute)
                                existing_keys.add(statute_key)
                        
                        sources_by_type["statutes_articles"] = existing_statutes
                        logger.debug(f"[_create_sources_event] Added {len(extracted_statutes)} statutes from reference clauses to existing sources_by_type")
            except Exception as e:
                logger.warning(f"[_create_sources_event] Failed to add reference statutes to existing sources_by_type: {e}", exc_info=True)
                # 예외 발생 시 기존 sources_by_type 유지 (참조 법령 추가 실패해도 계속 진행)
        
        # sources_by_type의 각 항목 정리 (클라이언트용)
        cleaned_sources_by_type = None
        if sources_by_type and isinstance(sources_by_type, dict):
            try:
                from api.services.chat_service import get_chat_service
                chat_service = get_chat_service()
                if chat_service and hasattr(chat_service, 'sources_extractor') and chat_service.sources_extractor:
                    from api.utils.source_type_mapper import get_default_sources_by_type
                    cleaned_sources_by_type = get_default_sources_by_type()
                    
                    for source_type, items in sources_by_type.items():
                        # source_type이 실제 테이블명이거나 source_type 값일 수 있으므로 둘 다 처리
                        table_name = source_type
                        if source_type in ["statute_article", "case_paragraph", "decision_paragraph", 
                                          "interpretation_paragraph", "regulation_paragraph"]:
                            # source_type 값을 테이블명으로 변환
                            from api.utils.source_type_mapper import source_type_to_table
                            table_name = source_type_to_table(source_type) or source_type
                        
                        if table_name in cleaned_sources_by_type and isinstance(items, list):
                            for item in items:
                                if isinstance(item, dict):
                                    try:
                                        # 디버깅: precedent_contents의 경우 원본 구조 확인
                                        if table_name == "precedent_contents":
                                            logger.info(
                                                f"[_create_sources_event] Processing precedent_contents item: "
                                                f"keys={list(item.keys())[:15]}, "
                                                f"casenames={item.get('casenames')}, "
                                                f"case_name={item.get('case_name')}, "
                                                f"metadata keys={list(item.get('metadata', {}).keys())[:10] if isinstance(item.get('metadata'), dict) else []}, "
                                                f"metadata.casenames={item.get('metadata', {}).get('casenames') if isinstance(item.get('metadata'), dict) else None}"
                                            )
                                        cleaned = chat_service.sources_extractor._clean_source_for_client(item)
                                        if cleaned and isinstance(cleaned, dict):
                                            cleaned_sources_by_type[table_name].append(cleaned)
                                            # 디버깅: 정리 후 결과 확인
                                            if table_name == "precedent_contents":
                                                logger.info(
                                                    f"[_create_sources_event] After cleaning: "
                                                    f"case_name={cleaned.get('case_name')}, "
                                                    f"detail.case_name={cleaned.get('detail', {}).get('case_name') if isinstance(cleaned.get('detail'), dict) else None}"
                                                )
                                    except Exception as item_error:
                                        logger.warning(f"[_create_sources_event] Failed to clean item: {item_error}", exc_info=True)
                                        # 개별 항목 정리 실패해도 계속 진행
                                        continue
                else:
                    cleaned_sources_by_type = sources_by_type
            except Exception as e:
                logger.warning(f"[_create_sources_event] Failed to clean sources_by_type: {e}", exc_info=True)
                cleaned_sources_by_type = sources_by_type
        else:
            cleaned_sources_by_type = sources_by_type
        
        # cleaned_sources_by_type이 None이면 빈 구조로 설정
        if cleaned_sources_by_type is None:
            from api.utils.source_type_mapper import get_default_sources_by_type
            cleaned_sources_by_type = get_default_sources_by_type()
        
        return {
            "type": "sources",
            "metadata": {
                "message_id": message_id or metadata.get("message_id"),
                "sources_by_type": cleaned_sources_by_type,  # 유일한 필요한 필드 (정리됨)
                # sources_detail도 포함 (done 이벤트와 일관성 유지, 프론트엔드에서 즉시 사용 가능)
                "sources_detail": sources_detail,  # 실제 값 사용 (빈 배열 아님)
                # 하위 호환성을 위해 deprecated 필드도 포함 (점진적 제거)
                "sources": [],  # deprecated: sources_by_type에서 재구성 가능
                "legal_references": [],  # deprecated: sources_by_type에서 재구성 가능
            },
            "timestamp": datetime.now().isoformat()
        }
    except Exception as e:
        logger.error(f"[_create_sources_event] Critical error creating sources event: {e}", exc_info=True)
        # 최종 예외 발생 시에도 기본 이벤트 반환 (스트림 중단 방지)
        return {
            "type": "sources",
            "metadata": {
                "message_id": message_id or metadata.get("message_id"),
                "sources_by_type": None,
                "sources": metadata.get("sources", []),
                "legal_references": metadata.get("legal_references", []),
                "sources_detail": metadata.get("sources_detail", []),
            },
            "timestamp": datetime.now().isoformat()
        }


def _has_sources_data(metadata: dict) -> bool:
    """metadata에 sources 데이터가 있는지 확인"""
    return bool(
        metadata.get("sources") or
        metadata.get("legal_references") or
        metadata.get("sources_detail") or
        metadata.get("related_questions")
    )


def _has_actual_sources(metadata: dict) -> bool:
    """metadata에 실제 참고자료(sources, legal_references (deprecated), sources_detail)가 있는지 확인
    related_questions는 제외"""
    return bool(
        metadata.get("sources") or
        metadata.get("legal_references") or  # deprecated: Phase 4에서 제거 예정
        metadata.get("sources_detail")
    )


def _add_quota_headers(response: ChatResponse, current_user: dict):
    """익명 사용자의 경우 응답 헤더에 남은 질의 횟수 추가"""
    if not current_user.get("authenticated") and anonymous_quota_service.is_enabled():
        from fastapi.responses import JSONResponse
        remaining = current_user.get("quota_remaining", 0)
        return JSONResponse(
            content=response.model_dump(),
            headers={
                "X-Quota-Remaining": str(remaining),
                "X-Quota-Limit": str(anonymous_quota_service.quota_limit)
            }
        )
    return response


async def _generate_stream_response(
    chat_service,
    message: str,
    session_id: str
) -> AsyncGenerator[str, None]:
    """스트리밍 응답 생성 - chunk 단위로 실시간 전달"""
    full_answer = ""
    final_metadata = None
    stream_completed = False
    done_event_sent = False  # stream_final_answer가 이미 done 이벤트를 보냈는지 추적
    
    try:
        # stream_final_answer를 직접 호출하여 chunk 단위 스트리밍
        try:
            async for chunk in chat_service.stream_final_answer(
                message=message,
                session_id=session_id
            ):
                if chunk:
                    # chunk는 이미 "data: {...}\n\n" 형식이므로 파싱하여 내용 추출
                    try:
                        # SSE 형식에서 JSON 추출
                        if chunk.startswith("data: "):
                            json_str = chunk[6:].strip()  # "data: " 제거
                            if json_str:
                                try:
                                    event_data = json.loads(json_str)
                                    event_type = event_data.get("type", "")
                                    
                                    # stream 이벤트에서 content 추출하여 full_answer 업데이트
                                    if event_type == "stream":
                                        content = event_data.get("content", "")
                                        if content:
                                            full_answer += content
                                    
                                    # final 이벤트에서 content와 metadata 추출
                                    elif event_type == "final":
                                        final_content = event_data.get("content", "")
                                        if final_content:
                                            full_answer = final_content
                                        final_metadata = event_data.get("metadata", {})
                                    
                                    # done 이벤트 처리
                                    elif event_type == "done":
                                        # done 이벤트의 content가 있으면 사용
                                        done_content = event_data.get("content", "")
                                        if done_content:
                                            full_answer = done_content
                                        done_metadata = event_data.get("metadata", {})
                                        if done_metadata:
                                            final_metadata = done_metadata
                                            logger.debug(
                                                f"[_generate_stream_response] Done event metadata received: "
                                                f"sources_detail={len(done_metadata.get('sources_detail', []))}, "
                                                f"sources={len(done_metadata.get('sources', []))}, "
                                                f"metadata_keys={list(done_metadata.keys())[:20]}"
                                            )
                                        else:
                                            logger.warning("[_generate_stream_response] Done event received but metadata is empty")
                                        # done 이벤트가 이미 전송되었음을 표시
                                        done_event_sent = True
                                        stream_completed = True
                                        logger.debug("[_generate_stream_response] Done event received from stream_final_answer")
                                    
                                except json.JSONDecodeError as json_error:
                                    logger.debug(f"[_generate_stream_response] Failed to parse chunk JSON: {json_error}, chunk: {chunk[:100]}")
                        
                        # chunk를 그대로 yield (클라이언트로 전송)
                        yield chunk
                    except (GeneratorExit, asyncio.CancelledError) as cancel_error:
                        # 클라이언트가 연결을 끊은 경우
                        logger.debug(f"[_generate_stream_response] Client disconnected, stopping stream: {cancel_error}")
                        stream_completed = True
                        raise  # 상위로 전파하여 제너레이터 종료
                    except Exception as yield_error:
                        logger.warning(f"[_generate_stream_response] Error yielding chunk: {yield_error}")
                        # yield 오류는 무시하고 계속 진행
        except asyncio.CancelledError:
            logger.warning("⚠️ [_generate_stream_response] 워크플로우 스트리밍이 취소되었습니다 (CancelledError)")
            # 에러 이벤트 전송
            try:
                error_event = {
                    "type": "error",
                    "content": "[오류] 작업이 취소되었습니다. 다시 시도해주세요.",
                    "metadata": {"error": True, "cancelled": True},
                    "timestamp": datetime.now().isoformat()
                }
                yield format_sse_event(error_event)
                
                # ERR_INCOMPLETE_CHUNKED_ENCODING 방지를 위해 done 이벤트도 전송
                if not done_event_sent:
                    done_event = {
                        "type": "done",
                        "timestamp": datetime.now().isoformat()
                    }
                    yield format_sse_event(done_event)
                    done_event_sent = True
                stream_completed = True
            except (GeneratorExit, asyncio.CancelledError):
                # 이미 연결이 끊어진 경우 무시
                stream_completed = True
            except Exception as cancel_error:
                logger.warning(f"Error sending error/done event after cancellation: {cancel_error}")
                stream_completed = True
            raise  # 상위로 전파
        
        # full_answer가 비어있으면 경고 로그
        if not full_answer:
            logger.warning(f"[_generate_stream_response] full_answer is empty for message: {message[:50]}...")
            logger.debug(f"[_generate_stream_response] Stream completed but no content was accumulated. This may indicate that stream events were not properly processed.")
        
        # done_event_sent는 이미 위에서 설정되었으므로 재설정하지 않음
        # stream_final_answer가 이미 done 이벤트를 보냈는지 확인 (위에서 설정된 값 사용)
        
        if full_answer:
            metadata = final_metadata if final_metadata else {}
            
            if not _has_sources_data(metadata):
                try:
                    sources_data = await chat_service.get_sources_from_session(
                        session_id=session_id,
                        message_id=None
                    )
                    if sources_data:
                        metadata["sources"] = sources_data.get("sources", [])
                        metadata["legal_references"] = sources_data.get("legal_references", [])
                        metadata["sources_detail"] = sources_data.get("sources_detail", [])
                except Exception as e:
                    logger.warning(f"Failed to get sources after stream: {e}")
            
            expected_message_id = metadata.get("message_id")
            saved_message_id = session_service.add_message(
                session_id=session_id,
                role="assistant",
                content=full_answer,
                metadata=metadata,
                message_id=expected_message_id
            )
            
            # 캐시 저장
            stream_cache = get_stream_cache()
            if stream_cache:
                try:
                    stream_cache.set(message, full_answer, metadata, session_id)
                    logger.info(
                        f"[_generate_stream_response] Response cached: "
                        f"message='{message[:50]}...', "
                        f"content_length={len(full_answer)}, "
                        f"has_metadata={bool(metadata)}"
                    )
                except Exception as e:
                    logger.error(f"[_generate_stream_response] Failed to cache response: {e}", exc_info=True)
            else:
                logger.debug("[_generate_stream_response] Stream cache is disabled or not available")
            
            # sources, legal_references, sources_detail 중 하나라도 있으면 sources 이벤트 전송
            # related_questions가 없어도 실제 참고자료가 있으면 전송
            # 중요: sources 이벤트는 done 이벤트 전에 전송되어야 함
            sources_event_sent = False
            
            # metadata 상세 로깅
            logger.info(
                f"[_generate_stream_response] Sources 이벤트 생성 시도: "
                f"sources={len(metadata.get('sources', []))}, "
                f"legal_references={len(metadata.get('legal_references', []))}, "
                f"sources_detail={len(metadata.get('sources_detail', []))}, "
                f"sources_by_type={bool(metadata.get('sources_by_type'))}, "
                f"has_actual_sources={_has_actual_sources(metadata)}"
            )
            
            if _has_actual_sources(metadata):
                try:
                    sources_event = _create_sources_event(metadata, saved_message_id)
                    sources_by_type = sources_event.get("metadata", {}).get("sources_by_type", {})
                    sources_detail = sources_event.get("metadata", {}).get("sources_detail", [])
                    
                    logger.info(
                        f"[_generate_stream_response] ✅ Sources 이벤트 생성 성공: "
                        f"sources_by_type_keys={list(sources_by_type.keys())}, "
                        f"sources_detail_count={len(sources_detail)}, "
                        f"total_sources={sum(len(items) for items in sources_by_type.values() if isinstance(items, list))}"
                    )
                    
                    yield format_sse_event(sources_event)
                    sources_event_sent = True
                except (GeneratorExit, asyncio.CancelledError):
                    # 클라이언트가 연결을 끊은 경우
                    logger.debug("[_generate_stream_response] Client disconnected while sending sources event")
                    raise
                except Exception as sources_error:
                    logger.error(f"Failed to create or send sources event: {sources_error}", exc_info=True)
                    # sources 이벤트 생성 실패해도 스트림은 계속 진행
                    # done 이벤트는 아래에서 반드시 전송됨
            else:
                logger.warning(
                    f"[_generate_stream_response] ⚠️ Sources 이벤트를 전송하지 않음: "
                    f"metadata에 sources 데이터가 없음. "
                    f"metadata_keys={list(metadata.keys())}"
                )
            
            # 세션 제목 생성 (실패해도 스트림은 계속 진행)
            try:
                _maybe_generate_session_title(session_id)
            except Exception as title_error:
                logger.debug(f"Failed to generate session title: {title_error}")
                # 세션 제목 생성 실패해도 스트림은 계속 진행
        
        # 정상 종료 시 done 이벤트 전송 (stream_handler에서 보내지 않았을 수 있으므로)
        # ERR_INCOMPLETE_CHUNKED_ENCODING 오류를 방지하기 위해 반드시 done 이벤트 전송
        # 중요: done 이벤트는 마지막에 전송되어야 함 (sources 이벤트 이후)
        # 단, stream_final_answer가 이미 done 이벤트를 보냈다면 중복 전송하지 않음
        if not done_event_sent:
            try:
                done_event = {
                    "type": "done",
                    "content": full_answer if full_answer else "",
                    "metadata": final_metadata if final_metadata else {},
                    "timestamp": datetime.now().isoformat()
                }
                yield format_sse_event(done_event)
                done_event_sent = True
                stream_completed = True
                logger.debug("[_generate_stream_response] Done event sent at end of stream")
            except (GeneratorExit, asyncio.CancelledError):
                logger.debug("[_generate_stream_response] Client disconnected while sending done event")
                stream_completed = True
                raise
            except Exception as done_error:
                logger.error(f"[_generate_stream_response] Error sending done event: {done_error}", exc_info=True)
                stream_completed = True
                # done 이벤트 전송 실패는 심각한 문제이지만, 제너레이터는 종료됨
        else:
            logger.debug("[_generate_stream_response] Done event already sent by stream_final_answer")
            stream_completed = True
        
    except (GeneratorExit, asyncio.CancelledError) as cancel_error:
        # 클라이언트가 연결을 끊은 경우 - 정상적인 종료
        logger.debug(f"[_generate_stream_response] Client disconnected or cancelled: {cancel_error}")
        stream_completed = True
        # GeneratorExit와 CancelledError는 상위로 전파하여 제너레이터 종료
        raise
    except Exception as e:
        logger.error(f"Error in _generate_stream_response: {e}", exc_info=True)
        error_msg = f"[오류] {str(e)}"
        try:
            error_event = {
                "type": "final",
                "content": error_msg,
                "metadata": {
                    "error": True,
                    "error_type": type(e).__name__
                },
                "timestamp": datetime.now().isoformat()
            }
            yield format_sse_event(error_event)
            
            # 에러 발생 시에도 done 이벤트 전송 (stream_handler에서 보내지 않을 수 있으므로)
            # ERR_INCOMPLETE_CHUNKED_ENCODING 방지를 위해 반드시 done 이벤트 전송
            if not done_event_sent:
                try:
                    done_event = {
                        "type": "done",
                        "timestamp": datetime.now().isoformat()
                    }
                    yield format_sse_event(done_event)
                    done_event_sent = True
                    stream_completed = True
                except (GeneratorExit, asyncio.CancelledError):
                    logger.debug("[_generate_stream_response] Client disconnected while sending error done event")
                    stream_completed = True
                    raise
                except Exception as error_done_error:
                    logger.warning(f"Error sending done event after error: {error_done_error}")
                    stream_completed = True
            else:
                stream_completed = True
        except (GeneratorExit, asyncio.CancelledError):
            logger.debug("[_generate_stream_response] Client disconnected or cancelled during error handling")
            stream_completed = True
            raise
        except Exception as yield_error:
            logger.error(f"Error yielding error message: {yield_error}")
            stream_completed = True
            # 최종 폴백: 스트림 종료를 보장하기 위해 done 이벤트 전송 시도
            # 하지만 yield_error가 발생했으므로 제너레이터가 이미 종료되었을 수 있음
            # 이 경우는 로그만 남기고 종료
    finally:
        # Generator 종료 시 로그만 남기고 yield는 하지 않음
        # finally 블록에서 yield를 사용하면 제너레이터가 이미 종료된 후에 실행될 수 있어 문제가 발생할 수 있음
        # done 이벤트는 정상 종료 경로와 예외 처리 경로에서 이미 전송되므로 finally에서는 로그만 남김
        try:
            if not stream_completed:
                logger.warning("[_generate_stream_response] Stream not properly completed (done event may not have been sent)")
            else:
                logger.debug("[_generate_stream_response] Generator completed successfully")
        except Exception as final_error:
            # finally 블록에서 발생한 오류는 로그만 남기고 무시
            logger.debug(f"[_generate_stream_response] Error in finally block (ignored): {final_error}")


@router.post("/chat", response_model=ChatResponse)
@limiter.limit("10/minute") if is_rate_limit_enabled() else lambda f: f
async def chat(request: Request, chat_request: ChatRequest, current_user: dict = Depends(require_auth)):
    """채팅 메시지 처리"""
    try:
        # ChatService 가져오기 (지연 초기화)
        chat_service = get_chat_service()
        
        # 사용자 정보 가져오기
        user_id, client_ip = get_user_info(request, current_user)
        
        # 세션이 없으면 생성
        if not chat_request.session_id:
            chat_request.session_id = session_service.create_session(
                user_id=user_id,
                ip_address=client_ip
            )
        else:
            # 기존 세션의 user_id 확인 및 업데이트 (user_id가 None인 경우에만)
            session = session_service.get_session(chat_request.session_id)
            if session:
                session_user_id = session.get("user_id")
                if session_user_id is None and user_id:
                    # 세션의 user_id가 없고 현재 user_id가 있으면 업데이트
                    session_service.update_session(chat_request.session_id, user_id=user_id)
                    logger.info(f"Updated session {chat_request.session_id} with user_id: {user_id}")
        
        # 이미지 또는 파일 처리
        final_message = _process_file_and_image(
            message=chat_request.message,
            file_base64=chat_request.file_base64,
            filename=chat_request.filename,
            image_base64=chat_request.image_base64
        )
        
        # 사용자 메시지 저장 (파일/이미지 처리 결과 포함)
        session_service.add_message(
            session_id=chat_request.session_id,
            role="user",
            content=final_message
        )
        
        # 쿼터 증가 플래그 확인 (익명 사용자이고 쿼터 증가가 필요한 경우)
        should_increment_quota = (
            not current_user.get("authenticated") and 
            current_user.get("_should_increment_quota", False) and
            anonymous_quota_service.is_enabled()
        )
        
        # AI 답변 생성
        try:
            result = await chat_service.process_message(
                message=final_message,
                session_id=chat_request.session_id,
                enable_checkpoint=chat_request.enable_checkpoint
            )
            
            # 성공적으로 답변을 받은 경우에만 쿼터 증가
            if should_increment_quota:
                remaining = anonymous_quota_service.increment_quota(client_ip)
                logger.debug(f"[chat] Quota incremented after successful response, ip={client_ip}, remaining={remaining}")
                # current_user에 업데이트된 쿼터 정보 반영
                current_user["quota_remaining"] = remaining
        except asyncio.CancelledError:
            logger.warning("⚠️ [chat] 워크플로우 실행이 취소되었습니다 (CancelledError)")
            # 취소된 경우 쿼터 증가하지 않음
            raise HTTPException(
                status_code=500,
                detail="작업이 취소되었습니다. 다시 시도해주세요."
            )
        except Exception as e:
            # 에러 발생 시 쿼터 증가하지 않음
            logger.error(f"[chat] Error processing message, quota not incremented: {e}")
            raise
        
        # AI 답변 저장
        session_service.add_message(
            session_id=chat_request.session_id,
            role="assistant",
            content=result.get("answer", ""),
            metadata=result.get("metadata", {})
        )
        
        # 세션에 제목이 없고 첫 번째 대화라면 제목 생성 (비동기)
        _maybe_generate_session_title(chat_request.session_id)
        
        # 익명 사용자의 경우 응답 헤더에 남은 질의 횟수 추가
        response = ChatResponse(**result)
        return _add_quota_headers(response, current_user)
    except HTTPException:
        raise
    except ValueError as e:
        logger.warning(f"Validation error in chat endpoint: {e}")
        raise HTTPException(status_code=400, detail="입력 데이터가 올바르지 않습니다")
    except Exception as e:
        logger.error(f"Error in chat endpoint: {e}", exc_info=True)
        raise HTTPException(
            status_code=500, 
            detail="메시지 처리 중 오류가 발생했습니다"
        )


@router.post("/chat/stream")
@limiter.limit("10/minute") if is_rate_limit_enabled() else lambda f: f
async def chat_stream(
    request: Request,
    stream_request: StreamingChatRequest,
    current_user: dict = Depends(require_auth)
):
    """
    스트리밍 채팅 응답
    
    HTTP 스트리밍 흐름:
    1. 클라이언트: fetch() API로 POST 요청 (Accept: text/event-stream)
    2. FastAPI: StreamingResponse로 AsyncGenerator 처리
    3. ChatService: LangGraph의 astream_events()로 LLM 스트리밍 이벤트 캡처
    4. LangGraph: 워크플로우 실행 중 on_llm_stream/on_chat_model_stream 이벤트 발생
    5. LLM: invoke() 호출 시에도 내부적으로 스트리밍 사용 (ChatGoogleGenerativeAI)
    6. 각 토큰을 JSONL 형식으로 yield → SSE 형식으로 변환 → HTTP 스트리밍 전송
    """
    try:
        # 이미지 또는 파일 처리
        final_message = _process_file_and_image(
            message=stream_request.message,
            file_base64=stream_request.file_base64,
            filename=stream_request.filename,
            image_base64=stream_request.image_base64
        )
        
        # 캐시 확인 (메시지만 기준, session_id 무시)
        stream_cache = get_stream_cache()
        if stream_cache:
            logger.debug(f"[chat_stream] Checking cache for message: {final_message[:50]}... (session_id ignored)")
            cached_response = stream_cache.get(final_message, stream_request.session_id)
            if cached_response:
                logger.info(
                    f"[chat_stream] Cache HIT: message='{final_message[:50]}...', "
                    f"content_length={len(cached_response.get('content', ''))}, "
                    f"has_metadata={bool(cached_response.get('metadata'))}"
                )
                
                # 사용자 정보 가져오기
                user_id, client_ip = get_user_info(request, current_user)
                
                # 세션이 없으면 생성
                if not stream_request.session_id:
                    stream_request.session_id = session_service.create_session(
                        user_id=user_id,
                        ip_address=client_ip
                    )
                else:
                    # 기존 세션의 user_id 확인 및 업데이트 (user_id가 None인 경우에만)
                    session = session_service.get_session(stream_request.session_id)
                    if session:
                        session_user_id = session.get("user_id")
                        if session_user_id is None and user_id:
                            # 세션의 user_id가 없고 현재 user_id가 있으면 업데이트
                            session_service.update_session(stream_request.session_id, user_id=user_id)
                            logger.info(f"Updated session {stream_request.session_id} with user_id: {user_id}")
                
                # 사용자 메시지 저장
                session_service.add_message(
                    session_id=stream_request.session_id,
                    role="user",
                    content=final_message
                )
                
                # 캐시된 응답을 스트리밍 형식으로 반환
                async def cached_stream():
                    try:
                        # 쿼터 정보 (캐시된 응답이므로 쿼터 소모 없음)
                        quota_event = {
                            "type": "quota",
                            "remaining": 999,
                            "limit": 1000
                        }
                        yield format_sse_event(quota_event)
                        
                        # 캐시된 내용을 청크 단위로 전송 (타이핑 효과 시뮬레이션)
                        content = cached_response['content']
                        metadata = cached_response.get('metadata', {})
                        chunk_size = 10
                        
                        for i in range(0, len(content), chunk_size):
                            chunk = content[i:i+chunk_size]
                            stream_event = {
                                "type": "stream",
                                "content": chunk,
                                "timestamp": datetime.now().isoformat()
                            }
                            yield format_sse_event(stream_event)
                            # 작은 딜레이로 스트리밍 효과 향상
                            await asyncio.sleep(0.01)
                        
                        # 최종 이벤트
                        final_event = {
                            "type": "final",
                            "content": content,
                            "metadata": metadata,
                            "timestamp": datetime.now().isoformat()
                        }
                        yield format_sse_event(final_event)
                        
                        # sources 이벤트 (메타데이터에 sources가 있는 경우)
                        if _has_actual_sources(metadata):
                            try:
                                saved_message_id = session_service.add_message(
                                    session_id=stream_request.session_id,
                                    role="assistant",
                                    content=content,
                                    metadata=metadata
                                )
                                sources_event = _create_sources_event(metadata, saved_message_id)
                                yield format_sse_event(sources_event)
                            except Exception as sources_error:
                                logger.error(f"Failed to create or send sources event in cached stream: {sources_error}", exc_info=True)
                                # sources 이벤트 생성 실패해도 스트림은 계속 진행
                        else:
                            # sources가 없어도 메시지 저장
                            session_service.add_message(
                                session_id=stream_request.session_id,
                                role="assistant",
                                content=content,
                                metadata=metadata
                            )
                        
                        # done 이벤트 (event_builder 형식과 동일)
                        done_event = {
                            "type": "done",
                            "content": content,
                            "metadata": metadata,
                            "timestamp": datetime.now().isoformat()
                        }
                        yield format_sse_event(done_event)
                        
                        _maybe_generate_session_title(stream_request.session_id)
                    except (GeneratorExit, asyncio.CancelledError):
                        logger.debug("[cached_stream] Client disconnected or cancelled")
                        return
                    except Exception as e:
                        logger.error(f"Error in cached_stream: {e}", exc_info=True)
                        error_event = {
                            "type": "final",
                            "content": f"[오류] {str(e)}",
                            "metadata": {
                                "error": True,
                                "error_type": type(e).__name__
                            },
                            "timestamp": datetime.now().isoformat()
                        }
                        yield format_sse_event(error_event)
                        done_event = {
                            "type": "done",
                            "timestamp": datetime.now().isoformat()
                        }
                        yield format_sse_event(done_event)
                
                return StreamingResponse(
                    cached_stream(),
                    media_type="text/event-stream",
                    headers={
                        "Cache-Control": "no-cache, no-transform",
                        "Connection": "keep-alive",
                        "X-Accel-Buffering": "no",
                        "Content-Type": "text/event-stream; charset=utf-8",
                        "X-Content-Type-Options": "nosniff",
                        "X-Cache": "HIT",
                        "X-Stream-Status": "active",  # 스트림 상태 추적용 (디버깅)
                    }
                )
            else:
                logger.debug(f"[chat_stream] Cache MISS for message: {final_message[:50]}...")
        else:
            logger.debug("[chat_stream] Stream cache is disabled or not available")
        
        # ChatService 가져오기 (지연 초기화)
        chat_service = get_chat_service()
        
        # 사용자 정보 가져오기
        user_id, client_ip = get_user_info(request, current_user)
        
        # 세션이 없으면 생성
        if not stream_request.session_id:
            stream_request.session_id = session_service.create_session(
                user_id=user_id,
                ip_address=client_ip
            )
        else:
            # 기존 세션의 user_id 확인 및 업데이트 (user_id가 None인 경우에만)
            session = session_service.get_session(stream_request.session_id)
            if session:
                session_user_id = session.get("user_id")
                if session_user_id is None and user_id:
                    # 세션의 user_id가 없고 현재 user_id가 있으면 업데이트
                    session_service.update_session(stream_request.session_id, user_id=user_id)
                    logger.info(f"Updated session {stream_request.session_id} with user_id: {user_id}")
        
        # 사용자 메시지 저장 (파일/이미지 처리 결과 포함)
        session_service.add_message(
            session_id=stream_request.session_id,
            role="user",
            content=final_message
        )
        
        # 쿼터 증가 플래그 확인 (익명 사용자이고 쿼터 증가가 필요한 경우)
        should_increment_quota = (
            not current_user.get("authenticated") and 
            current_user.get("_should_increment_quota", False) and
            anonymous_quota_service.is_enabled()
        )
        
        # 쿼터 증가 여부를 추적하기 위한 변수
        quota_incremented = False
        
        try:
            # 스트리밍 응답 생성 (성공적으로 완료된 경우에만 쿼터 증가)
            async def stream_with_quota_management():
                nonlocal quota_incremented
                done_event_sent = False
                last_chunk = None
                try:
                    async for chunk in _generate_stream_response(
                        chat_service=chat_service,
                        message=final_message,
                        session_id=stream_request.session_id
                    ):
                        yield chunk
                        last_chunk = chunk
                        # done 이벤트가 전송되었는지 확인 (SSE 형식: "data: {...}\n\n")
                        # format_sse_event는 "data: {json}\n\n" 형식이므로 JSON 부분에서 확인
                        if chunk:
                            # SSE 형식에서 JSON 부분 추출하여 확인
                            if chunk.startswith("data: "):
                                try:
                                    json_str = chunk[6:].strip()  # "data: " 제거
                                    if json_str:
                                        event_data = json.loads(json_str)
                                        if event_data.get("type") == "done":
                                            done_event_sent = True
                                            logger.debug("[stream_with_quota_management] Done event detected in chunk")
                                except (json.JSONDecodeError, ValueError):
                                    # JSON 파싱 실패 시 문자열 검색으로 폴백
                                    if '"type":"done"' in chunk or "'type':'done'" in chunk:
                                        done_event_sent = True
                            else:
                                # SSE 형식이 아닌 경우 문자열 검색
                                if '"type":"done"' in chunk or "'type':'done'" in chunk:
                                    done_event_sent = True
                    
                    # 스트리밍이 성공적으로 완료된 경우에만 쿼터 증가
                    if should_increment_quota and not quota_incremented:
                        remaining = anonymous_quota_service.increment_quota(client_ip)
                        quota_incremented = True
                        logger.debug(f"[chat_stream] Quota incremented after successful stream, ip={client_ip}, remaining={remaining}")
                    
                    # 🔥 개선: 정상 종료 시에도 done 이벤트가 전송되었는지 확인
                    # _generate_stream_response가 완전히 소비되었지만 done 이벤트가 없을 수 있음
                    if not done_event_sent:
                        try:
                            done_event = {
                                "type": "done",
                                "timestamp": datetime.now().isoformat()
                            }
                            yield format_sse_event(done_event)
                            done_event_sent = True
                            logger.debug("[stream_with_quota_management] Sent done event after stream completion")
                        except (GeneratorExit, asyncio.CancelledError):
                            # 이미 연결이 끊어진 경우 무시
                            pass
                        except Exception as done_error:
                            logger.warning(f"[stream_with_quota_management] Failed to send done event after completion: {done_error}")
                            
                except (GeneratorExit, asyncio.CancelledError):
                    # 클라이언트가 연결을 끊은 경우 - 정상적인 종료
                    logger.debug("[stream_with_quota_management] Client disconnected or cancelled")
                    # GeneratorExit나 CancelledError는 제너레이터가 이미 종료된 상태이므로 yield 불가
                    # done 이벤트는 전송하지 않음 (이미 연결이 끊어짐)
                    raise
                except Exception as e:
                    # 에러 발생 시 쿼터 증가하지 않음
                    logger.warning(f"[chat_stream] Stream error occurred, quota not incremented: {e}", exc_info=True)
                    # 에러 발생 시에도 done 이벤트 전송 (ERR_INCOMPLETE_CHUNKED_ENCODING 방지)
                    if not done_event_sent:
                        try:
                            error_event = {
                                "type": "error",
                                "content": f"[오류] {str(e)}",
                                "metadata": {
                                    "error": True,
                                    "error_type": type(e).__name__
                                },
                                "timestamp": datetime.now().isoformat()
                            }
                            yield format_sse_event(error_event)
                            
                            done_event = {
                                "type": "done",
                                "timestamp": datetime.now().isoformat()
                            }
                            yield format_sse_event(done_event)
                            done_event_sent = True
                        except (GeneratorExit, asyncio.CancelledError):
                            # 이미 연결이 끊어진 경우 무시
                            pass
                        except Exception as yield_error:
                            logger.error(f"[stream_with_quota_management] Failed to send error/done event: {yield_error}")
                    raise
            
            # 초기 쿼터 정보 가져오기 (성공 시 업데이트됨)
            initial_remaining = current_user.get("quota_remaining", anonymous_quota_service.get_remaining_quota(client_ip))
            
            return StreamingResponse(
                stream_with_quota_management(),
                media_type="text/event-stream",
                headers={
                    "Cache-Control": "no-cache, no-transform",
                    "Connection": "keep-alive",
                    "X-Accel-Buffering": "no",  # Nginx 버퍼링 비활성화
                    "Content-Type": "text/event-stream; charset=utf-8",
                    # Transfer-Encoding: chunked는 FastAPI가 자동으로 처리하므로 명시하지 않음
                    # 명시하면 ERR_INCOMPLETE_CHUNKED_ENCODING 오류가 발생할 수 있음
                    "X-Content-Type-Options": "nosniff",
                    "X-Cache": "MISS",
                    "X-Stream-Status": "active",  # 스트림 상태 추적용 (디버깅)
                    # 쿼터 정보 (성공 시 업데이트됨)
                    "X-Quota-Remaining": str(initial_remaining),
                    "X-Quota-Limit": str(anonymous_quota_service.quota_limit),
                }
            )
        except asyncio.CancelledError:
            logger.warning("⚠️ [chat_stream] 워크플로우 스트리밍이 취소되었습니다 (CancelledError)")
            # 에러 응답 생성
            async def error_stream():
                error_event = {
                    "type": "error",
                    "content": "[오류] 작업이 취소되었습니다. 다시 시도해주세요.",
                    "metadata": {"error": True, "cancelled": True},
                    "timestamp": datetime.now().isoformat()
                }
                yield format_sse_event(error_event)
                done_event = {
                    "type": "done",
                    "timestamp": datetime.now().isoformat()
                }
                yield format_sse_event(done_event)
            
            return StreamingResponse(
                error_stream(),
                media_type="text/event-stream",
                headers={
                    "Cache-Control": "no-cache, no-transform",
                    "Connection": "keep-alive",
                    "Content-Type": "text/event-stream; charset=utf-8",
                    "X-Content-Type-Options": "nosniff",
                    "X-Stream-Status": "cancelled",
                }
            )
        except (GeneratorExit, asyncio.CancelledError):
            logger.debug("[chat_stream] Client disconnected or cancelled")
            # 취소된 경우 쿼터 증가하지 않음
            return
    except ValueError as e:
        logger.warning(f"Validation error in chat_stream endpoint: {e}")
        # 검증 오류 시 쿼터 증가하지 않음
        raise HTTPException(status_code=400, detail="입력 데이터가 올바르지 않습니다")
    except Exception as e:
        logger.error(f"Error in chat_stream endpoint: {e}", exc_info=True)
        # 에러 발생 시 쿼터 증가하지 않음
        from api.config import api_config
        if api_config.debug:
            detail = f"스트리밍 처리 중 오류가 발생했습니다: {str(e)}"
        else:
            detail = "스트리밍 처리 중 오류가 발생했습니다"
        raise HTTPException(status_code=500, detail=detail)


@router.get("/chat/{session_id}/sources")
async def get_chat_sources(
    session_id: str,
    message_id: Optional[str] = None
):
    """
    스트림 완료 후 sources 정보를 가져오는 API
    
    Args:
        session_id: 세션 ID
        message_id: 특정 메시지 ID (선택사항, 현재는 사용하지 않음)
    
    Returns:
        sources, legal_references, sources_detail 정보
    """
    try:
        chat_service = get_chat_service()
        
        if not chat_service.is_available():
            raise HTTPException(
                status_code=503,
                detail="Chat service is not available"
            )
        
        # LangGraph의 최종 state에서 sources 추출
        sources_data = await chat_service.get_sources_from_session(
            session_id=session_id,
            message_id=message_id
        )
        
        return {
            "session_id": session_id,
            "sources": sources_data.get("sources", []),
            "legal_references": sources_data.get("legal_references", []),
            "sources_detail": sources_data.get("sources_detail", [])
        }
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting sources for session {session_id}: {e}", exc_info=True)
        from api.config import api_config
        if api_config.debug:
            detail = f"sources 가져오기 중 오류가 발생했습니다: {str(e)}"
        else:
            detail = "sources 가져오기 중 오류가 발생했습니다"
        raise HTTPException(status_code=500, detail=detail)


@router.post("/chat/continue", response_model=ContinueAnswerResponse)
@limiter.limit("30/minute") if is_rate_limit_enabled() else lambda f: f
async def continue_answer(
    request: Request,
    continue_request: ContinueAnswerRequest,
    current_user: dict = Depends(require_auth)
):
    """이전 답변의 마지막 부분부터 이어서 답변 생성 (워크플로우 재개 방식)"""
    try:
        from api.services.chat_service import chat_service
        
        # LangGraph 워크플로우 서비스를 사용하여 이어서 답변 생성
        if not chat_service.workflow_service:
            raise HTTPException(
                status_code=503,
                detail="워크플로우 서비스가 초기화되지 않았습니다."
            )
        
        # 워크플로우에서 이어서 답변 생성
        try:
            result = await chat_service.workflow_service.continue_answer(
                session_id=continue_request.session_id,
                message_id=continue_request.message_id,
                chunk_index=continue_request.chunk_index
            )
        except asyncio.CancelledError:
            logger.warning("⚠️ [continue_answer] 워크플로우 실행이 취소되었습니다 (CancelledError)")
            raise HTTPException(
                status_code=500,
                detail="작업이 취소되었습니다. 다시 시도해주세요."
            )
        
        if result:
            return ContinueAnswerResponse(
                content=result.get("content", ""),
                chunk_index=result.get("chunk_index", 0),
                total_chunks=result.get("total_chunks", 1),
                has_more=result.get("has_more", False)
            )
        else:
            raise HTTPException(
                status_code=404,
                detail="이어서 답변을 생성할 수 없습니다."
            )
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error continuing answer: {e}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail=f"답변을 이어서 생성하는 중 오류가 발생했습니다: {str(e)}"
        )

