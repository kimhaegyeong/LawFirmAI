"""
채팅 엔드포인트
"""
import json
import logging
import asyncio
from fastapi import APIRouter, HTTPException, Request, Depends
from fastapi.responses import StreamingResponse
from typing import AsyncGenerator, Optional
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

router = APIRouter()
logger = logging.getLogger(__name__)


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
    """sources 이벤트 생성"""
    return {
        "type": "sources",
        "metadata": {
            "message_id": message_id or metadata.get("message_id"),
            "sources": metadata.get("sources", []),
            "legal_references": metadata.get("legal_references", []),
            "sources_detail": metadata.get("sources_detail", []),
            "related_questions": metadata.get("related_questions", [])
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
    
    try:
        # stream_final_answer를 직접 호출하여 chunk 단위 스트리밍
        async for chunk in chat_service.stream_final_answer(
            message=message,
            session_id=session_id
        ):
            if chunk:
                # chunk는 이미 "data: {...}\n\n" 형식이므로 그대로 yield
                yield chunk
                
                # final 이벤트에서 메타데이터 추출
                try:
                    # "data: " 접두사 제거
                    if chunk.startswith("data: "):
                        json_str = chunk[6:].strip()
                        event_data = json.loads(json_str)
                        event_type = event_data.get("type", "")
                        
                        if event_type == "stream":
                            content = event_data.get("content", "")
                            if content:
                                full_answer += content
                        
                        elif event_type == "final":
                            content = event_data.get("content", "")
                            if content and not content.startswith("[오류]"):
                                full_answer = content
                            final_metadata = event_data.get("metadata", {})
                            
                            logger.debug(
                                f"[_generate_stream_response] Final event received: "
                                f"sources={len(final_metadata.get('sources', []))}, "
                                f"legal_references={len(final_metadata.get('legal_references', []))}, "
                                f"sources_detail={len(final_metadata.get('sources_detail', []))}, "
                                f"related_questions={len(final_metadata.get('related_questions', []))}, "
                                f"has_sources_data={_has_sources_data(final_metadata)}"
                            )
                            
                            # sources 데이터가 없으면 세션에서 가져오기
                            if not _has_sources_data(final_metadata):
                                logger.debug(f"[_generate_stream_response] No sources data in final metadata, attempting to fetch from session")
                                try:
                                    sources_data = await chat_service.get_sources_from_session(
                                        session_id=session_id,
                                        message_id=final_metadata.get("message_id")
                                    )
                                    
                                    logger.debug(
                                        f"[_generate_stream_response] Sources fetched from session: "
                                        f"sources={len(sources_data.get('sources', []))}, "
                                        f"legal_references={len(sources_data.get('legal_references', []))}, "
                                        f"sources_detail={len(sources_data.get('sources_detail', []))}"
                                    )
                                    
                                    if sources_data:
                                        final_metadata["sources"] = sources_data.get("sources", [])
                                        final_metadata["legal_references"] = sources_data.get("legal_references", [])
                                        final_metadata["sources_detail"] = sources_data.get("sources_detail", [])
                                        
                                        if _has_sources_data(final_metadata):
                                            sources_event = _create_sources_event(final_metadata)
                                            logger.debug(f"[_generate_stream_response] Sending sources event with data")
                                            yield f"data: {json.dumps(sources_event, ensure_ascii=False)}\n\n"
                                except Exception as e:
                                    logger.warning(f"Failed to get sources after final event: {e}")
                            
                            # related_questions만 있는 경우 sources 이벤트 전송
                            if final_metadata.get("related_questions") and not _has_sources_data(final_metadata):
                                logger.debug(f"[_generate_stream_response] Sending sources event with related_questions only")
                                sources_event = _create_sources_event(final_metadata)
                                yield f"data: {json.dumps(sources_event, ensure_ascii=False)}\n\n"
                except (json.JSONDecodeError, ValueError) as e:
                    # JSON 파싱 실패는 무시 (일부 청크는 JSON이 아닐 수 있음)
                    if '[스트리밍 완료]' not in chunk and '[완료]' not in chunk:
                        logger.debug(f"Failed to parse chunk as JSON: {e}")
        
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
            
            if _has_sources_data(metadata):
                sources_event = _create_sources_event(metadata, saved_message_id)
                logger.debug(f"Sending sources_event with related_questions: {len(metadata.get('related_questions', []))} questions")
                yield f"data: {json.dumps(sources_event, ensure_ascii=False)}\n\n"
            
            _maybe_generate_session_title(session_id)
    
    except GeneratorExit:
        logger.debug("[chat_stream] Client disconnected, closing stream")
        return
    except Exception as e:
        logger.error(f"Error in stream_message: {e}", exc_info=True)
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
            yield f"data: {json.dumps(error_event, ensure_ascii=False)}\n\n"
        except GeneratorExit:
            logger.debug("[chat_stream] Client disconnected during error handling")
            return
        except Exception as yield_error:
            logger.error(f"Error yielding error message: {yield_error}")


@router.post("/chat", response_model=ChatResponse)
@limiter.limit("10/minute") if is_rate_limit_enabled() else lambda f: f
async def chat(request: Request, chat_request: ChatRequest, current_user: dict = Depends(require_auth)):
    """채팅 메시지 처리"""
    try:
        # ChatService 가져오기 (지연 초기화)
        chat_service = get_chat_service()
        
        # 세션이 없으면 생성
        if not chat_request.session_id:
            user_id, client_ip = get_user_info(request, current_user)
            chat_request.session_id = session_service.create_session(
                user_id=user_id,
                ip_address=client_ip
            )
        
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
        
        # AI 답변 생성
        result = await chat_service.process_message(
            message=final_message,
            session_id=chat_request.session_id,
            enable_checkpoint=chat_request.enable_checkpoint
        )
        
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
    5. LLM: invoke() 호출 시에도 내부적으로 스트리밍 사용 (ChatGoogleGenerativeAI/Ollama)
    6. 각 토큰을 JSONL 형식으로 yield → SSE 형식으로 변환 → HTTP 스트리밍 전송
    """
    try:
        # ChatService 가져오기 (지연 초기화)
        chat_service = get_chat_service()
        
        # 세션이 없으면 생성
        if not stream_request.session_id:
            user_id, client_ip = get_user_info(request, current_user)
            stream_request.session_id = session_service.create_session(
                user_id=user_id,
                ip_address=client_ip
            )
        
        # 이미지 또는 파일 처리
        final_message = _process_file_and_image(
            message=stream_request.message,
            file_base64=stream_request.file_base64,
            filename=stream_request.filename,
            image_base64=stream_request.image_base64
        )
        
        # 사용자 메시지 저장 (파일/이미지 처리 결과 포함)
        session_service.add_message(
            session_id=stream_request.session_id,
            role="user",
            content=final_message
        )
        
        return StreamingResponse(
            _generate_stream_response(
                chat_service=chat_service,
                message=final_message,
                session_id=stream_request.session_id
            ),
            media_type="text/event-stream",
            headers={
                "Cache-Control": "no-cache, no-transform",
                "Connection": "keep-alive",
                "X-Accel-Buffering": "no",  # Nginx 버퍼링 비활성화
                "Content-Type": "text/event-stream; charset=utf-8",
                # Transfer-Encoding: chunked는 FastAPI가 자동으로 처리하므로 명시하지 않음
                # 명시하면 ERR_INCOMPLETE_CHUNKED_ENCODING 오류가 발생할 수 있음
                "X-Content-Type-Options": "nosniff",
            }
        )
    except ValueError as e:
        logger.warning(f"Validation error in chat_stream endpoint: {e}")
        raise HTTPException(status_code=400, detail="입력 데이터가 올바르지 않습니다")
    except Exception as e:
        logger.error(f"Error in chat_stream endpoint: {e}", exc_info=True)
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
        result = await chat_service.workflow_service.continue_answer(
            session_id=continue_request.session_id,
            message_id=continue_request.message_id,
            chunk_index=continue_request.chunk_index
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

