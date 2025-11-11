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

router = APIRouter()
logger = logging.getLogger(__name__)


@router.post("/chat", response_model=ChatResponse)
@limiter.limit("10/minute") if is_rate_limit_enabled() else lambda f: f
async def chat(request: Request, chat_request: ChatRequest, current_user: dict = Depends(require_auth)):
    """채팅 메시지 처리"""
    try:
        # ChatService 가져오기 (지연 초기화)
        chat_service = get_chat_service()
        
        # 세션이 없으면 생성
        if not chat_request.session_id:
            chat_request.session_id = session_service.create_session()
        
        # 이미지 또는 파일 처리
        final_message = chat_request.message
        extracted_text = None
        
        # 우선순위: file_base64 > image_base64
        if chat_request.file_base64:
            try:
                logger.info("Processing file...")
                success, extracted_text, error_msg = process_file(
                    chat_request.file_base64,
                    chat_request.filename
                )
                if success and extracted_text:
                    # 질의 유형에 따라 동적 구성
                    if final_message.strip():
                        # 질의문이 있는 경우: 질의문을 먼저, 파일 텍스트를 참고 자료로
                        final_message = f"{final_message}\n\n[참고 파일 텍스트]\n{extracted_text}"
                    else:
                        # 질의문이 없는 경우: 파일 텍스트를 질의로 사용
                        final_message = f"다음 파일의 내용을 분석하고 질문에 답변해주세요:\n\n{extracted_text}"
                    logger.info(f"File processing completed. Extracted {len(extracted_text)} characters")
                elif not success:
                    logger.warning(f"File processing failed: {error_msg}")
                    # 파일 처리 실패해도 메시지는 처리 (경고만 추가)
                    if final_message.strip():
                        final_message = f"{final_message}\n\n[경고: {error_msg}]"
                    else:
                        final_message = f"[경고: {error_msg}]"
            except Exception as e:
                logger.error(f"File processing error: {e}", exc_info=True)
                # 파일 처리 실패해도 메시지는 처리 (경고만 추가)
                if final_message.strip():
                    final_message = f"{final_message}\n\n[경고: 파일 처리 중 오류가 발생했습니다: {str(e)}]"
                else:
                    final_message = f"[경고: 파일 처리 중 오류가 발생했습니다: {str(e)}]"
        elif chat_request.image_base64:
            try:
                if is_ocr_available():
                    logger.info("Processing image with OCR...")
                    extracted_text = extract_text_from_base64(chat_request.image_base64)
                    if extracted_text:
                        # 질의 유형에 따라 동적 구성
                        if final_message.strip():
                            # 질의문이 있는 경우: 질의문을 먼저, OCR 결과를 참고 자료로
                            final_message = f"{final_message}\n\n[참고 이미지 텍스트]\n{extracted_text}"
                        else:
                            # 질의문이 없는 경우: OCR 결과를 질의로 사용
                            final_message = f"다음 이미지의 내용을 분석하고 질문에 답변해주세요:\n\n{extracted_text}"
                        logger.info(f"OCR completed. Extracted {len(extracted_text)} characters")
                    else:
                        logger.warning("OCR did not extract any text from image")
                else:
                    logger.warning("OCR service is not available")
            except Exception as e:
                logger.error(f"OCR processing error: {e}", exc_info=True)
                # OCR 실패해도 메시지는 처리 (경고만 추가)
                if final_message.strip():
                    final_message = f"{final_message}\n\n[경고: 이미지에서 텍스트를 추출하는 중 오류가 발생했습니다: {str(e)}]"
                else:
                    final_message = f"[경고: 이미지에서 텍스트를 추출하는 중 오류가 발생했습니다: {str(e)}]"
        
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
        session = session_service.get_session(chat_request.session_id)
        if session and not session.get("title"):
            messages = session_service.get_messages(chat_request.session_id)
            if len(messages) == 2:  # user + assistant
                # 백그라운드에서 제목 생성 (응답 지연 방지)
                import asyncio
                asyncio.create_task(
                    asyncio.to_thread(
                        session_service.generate_session_title,
                        chat_request.session_id
                    )
                )
        
        # 익명 사용자의 경우 응답 헤더에 남은 질의 횟수 추가
        response = ChatResponse(**result)
        if not current_user.get("authenticated") and anonymous_quota_service.is_enabled():
            # current_user에 이미 quota_remaining이 포함되어 있음
            remaining = current_user.get("quota_remaining", 0)
            # Response 객체로 변환하여 헤더 추가
            from fastapi.responses import JSONResponse
            return JSONResponse(
                content=response.model_dump(),
                headers={
                    "X-Quota-Remaining": str(remaining),
                    "X-Quota-Limit": str(anonymous_quota_service.quota_limit)
                }
            )
        
        return response
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
            stream_request.session_id = session_service.create_session()
        
        # 이미지 또는 파일 처리
        final_message = stream_request.message
        extracted_text = None
        
        # 우선순위: file_base64 > image_base64
        if stream_request.file_base64:
            try:
                logger.info("Processing file...")
                success, extracted_text, error_msg = process_file(
                    stream_request.file_base64,
                    stream_request.filename
                )
                if success and extracted_text:
                    # 질의 유형에 따라 동적 구성
                    if final_message.strip():
                        # 질의문이 있는 경우: 질의문을 먼저, 파일 텍스트를 참고 자료로
                        final_message = f"{final_message}\n\n[참고 파일 텍스트]\n{extracted_text}"
                    else:
                        # 질의문이 없는 경우: 파일 텍스트를 질의로 사용
                        final_message = f"다음 파일의 내용을 분석하고 질문에 답변해주세요:\n\n{extracted_text}"
                    logger.info(f"File processing completed. Extracted {len(extracted_text)} characters")
                elif not success:
                    logger.warning(f"File processing failed: {error_msg}")
                    # 파일 처리 실패해도 메시지는 처리 (경고만 추가)
                    if final_message.strip():
                        final_message = f"{final_message}\n\n[경고: {error_msg}]"
                    else:
                        final_message = f"[경고: {error_msg}]"
            except Exception as e:
                logger.error(f"File processing error: {e}", exc_info=True)
                # 파일 처리 실패해도 메시지는 처리 (경고만 추가)
                if final_message.strip():
                    final_message = f"{final_message}\n\n[경고: 파일 처리 중 오류가 발생했습니다: {str(e)}]"
                else:
                    final_message = f"[경고: 파일 처리 중 오류가 발생했습니다: {str(e)}]"
        elif stream_request.image_base64:
            try:
                if is_ocr_available():
                    logger.info("Processing image with OCR...")
                    extracted_text = extract_text_from_base64(stream_request.image_base64)
                    if extracted_text:
                        # 질의 유형에 따라 동적 구성
                        if final_message.strip():
                            # 질의문이 있는 경우: 질의문을 먼저, OCR 결과를 참고 자료로
                            final_message = f"{final_message}\n\n[참고 이미지 텍스트]\n{extracted_text}"
                        else:
                            # 질의문이 없는 경우: OCR 결과를 질의로 사용
                            final_message = f"다음 이미지의 내용을 분석하고 질문에 답변해주세요:\n\n{extracted_text}"
                        logger.info(f"OCR completed. Extracted {len(extracted_text)} characters")
                    else:
                        logger.warning("OCR did not extract any text from image")
                else:
                    logger.warning("OCR service is not available")
            except Exception as e:
                logger.error(f"OCR processing error: {e}", exc_info=True)
                # OCR 실패해도 메시지는 처리 (경고만 추가)
                if final_message.strip():
                    final_message = f"{final_message}\n\n[경고: 이미지에서 텍스트를 추출하는 중 오류가 발생했습니다: {str(e)}]"
                else:
                    final_message = f"[경고: 이미지에서 텍스트를 추출하는 중 오류가 발생했습니다: {str(e)}]"
        
        # 사용자 메시지 저장 (파일/이미지 처리 결과 포함)
        session_service.add_message(
            session_id=stream_request.session_id,
            role="user",
            content=final_message
        )
        
        async def generate() -> AsyncGenerator[str, None]:
            """
            스트리밍 응답 생성
            
            ChatService.stream_message()에서 받은 JSONL 형식의 이벤트를
            SSE(Server-Sent Events) 형식으로 변환하여 HTTP 스트리밍으로 전송합니다.
            """
            full_answer = ""
            has_yielded = False  # 최소한 하나의 yield가 있었는지 추적
            final_metadata = None  # final 이벤트의 metadata 저장
            stream_closed = False  # 스트림이 정상적으로 종료되었는지 추적
            
            try:
                async for chunk in chat_service.stream_message(
                    message=final_message,
                    session_id=stream_request.session_id
                ):
                    if chunk:
                        has_yielded = True
                        chunk_stripped = chunk.strip()
                        
                        # chat_service.stream_message()에서 이미 JSONL 형식으로 yield하므로
                        # JSON 파싱 없이 직접 SSE 형식으로 변환 (성능 최적화)
                        if chunk_stripped:
                            # JSONL 형식을 SSE 형식으로 직접 변환 (불필요한 JSON 파싱/직렬화 제거)
                            yield f"data: {chunk_stripped}\n\n"
                            
                            # full_answer와 final_metadata 추적을 위해 선택적으로 JSON 파싱
                            try:
                                event_data = json.loads(chunk_stripped)
                                event_type = event_data.get("type", "")
                                
                                # "stream" 타입인 경우에만 full_answer에 추가
                                if event_type == "stream":
                                    content = event_data.get("content", "")
                                    if content:
                                        full_answer += content
                                
                                # "final" 타입인 경우 full_answer 업데이트 및 metadata 저장
                                elif event_type == "final":
                                    content = event_data.get("content", "")
                                    if content and not content.startswith("[오류]"):
                                        full_answer = content
                                    # final 이벤트의 metadata 저장 (sources 포함)
                                    # workflow_service에서 이미 related_questions로 변환되어 있음
                                    final_metadata = event_data.get("metadata", {})
                                    
                                    # related_questions 포함 여부 확인 (디버깅용)
                                    if final_metadata.get("related_questions"):
                                        logger.debug(
                                            f"Final event metadata contains {len(final_metadata.get('related_questions', []))} related_questions"
                                        )
                                    else:
                                        logger.debug("Final event metadata does not contain related_questions")
                                    
                                    # final 이벤트의 metadata에 sources가 없으면 즉시 가져오기
                                    if not final_metadata.get("sources") and not final_metadata.get("legal_references") and not final_metadata.get("sources_detail"):
                                        try:
                                            sources_data = await chat_service.get_sources_from_session(
                                                session_id=stream_request.session_id,
                                                message_id=final_metadata.get("message_id")
                                            )
                                            if sources_data:
                                                final_metadata["sources"] = sources_data.get("sources", [])
                                                final_metadata["legal_references"] = sources_data.get("legal_references", [])
                                                final_metadata["sources_detail"] = sources_data.get("sources_detail", [])
                                                
                                                # sources_event를 즉시 전송
                                                if final_metadata.get("sources") or final_metadata.get("legal_references") or final_metadata.get("sources_detail") or final_metadata.get("related_questions"):
                                                    sources_event = {
                                                        "type": "sources",
                                                        "metadata": {
                                                            "message_id": final_metadata.get("message_id"),
                                                            "sources": final_metadata.get("sources", []),
                                                            "legal_references": final_metadata.get("legal_references", []),
                                                            "sources_detail": final_metadata.get("sources_detail", []),
                                                            "related_questions": final_metadata.get("related_questions", [])
                                                        },
                                                        "timestamp": datetime.now().isoformat()
                                                    }
                                                    logger.debug(f"Sending sources_event after fetching sources: {len(final_metadata.get('related_questions', []))} related_questions")
                                                    yield f"data: {json.dumps(sources_event, ensure_ascii=False)}\n\n"
                                        except Exception as e:
                                            logger.warning(f"Failed to get sources after final event: {e}")
                                    
                                    # related_questions만 있어도 sources_event 전송 (sources가 없는 경우)
                                    if final_metadata.get("related_questions") and not (final_metadata.get("sources") or final_metadata.get("legal_references") or final_metadata.get("sources_detail")):
                                        sources_event = {
                                            "type": "sources",
                                            "metadata": {
                                                "message_id": final_metadata.get("message_id"),
                                                "sources": final_metadata.get("sources", []),
                                                "legal_references": final_metadata.get("legal_references", []),
                                                "sources_detail": final_metadata.get("sources_detail", []),
                                                "related_questions": final_metadata.get("related_questions", [])
                                            },
                                            "timestamp": datetime.now().isoformat()
                                        }
                                        logger.debug(f"Sending sources_event with related_questions only: {len(final_metadata.get('related_questions', []))} questions")
                                        yield f"data: {json.dumps(sources_event, ensure_ascii=False)}\n\n"
                            except (json.JSONDecodeError, ValueError):
                                # JSON 파싱 실패 시 기존 형식으로 처리 (하위 호환성)
                                if '[스트리밍 완료]' not in chunk and '[완료]' not in chunk:
                                    full_answer += chunk
                                    if '\n' in chunk:
                                        lines = chunk.split('\n')
                                        for line in lines:
                                            yield f"data: {line}\n"
                                        yield "\n"
                                    else:
                                        yield f"data: {chunk}\n\n"
                
                # 스트리밍이 정상적으로 완료되었음을 보장
                # chat_service.stream_message에서 이미 final 이벤트를 보냈으므로
                # 여기서는 추가 완료 신호는 불필요
                # FastAPI StreamingResponse가 자동으로 스트림 종료를 처리하므로
                # 빈 줄을 yield하면 ERR_INCOMPLETE_CHUNKED_ENCODING 오류가 발생할 수 있음
                
                # 완료 후 메시지 저장 (metadata 포함)
                if full_answer:
                    # final_metadata가 있으면 포함, 없으면 빈 dict
                    metadata = final_metadata if final_metadata else {}
                    
                    # sources가 비어있으면 별도로 가져오기
                    if not metadata.get("sources") and not metadata.get("legal_references") and not metadata.get("sources_detail"):
                        try:
                            sources_data = await chat_service.get_sources_from_session(
                                session_id=stream_request.session_id,
                                message_id=None  # 아직 message_id가 없으므로 None
                            )
                            if sources_data:
                                metadata["sources"] = sources_data.get("sources", [])
                                metadata["legal_references"] = sources_data.get("legal_references", [])
                                metadata["sources_detail"] = sources_data.get("sources_detail", [])
                        except Exception as e:
                            logger.warning(f"Failed to get sources after stream: {e}")
                    
                    # final_metadata의 message_id를 사용하여 메시지 저장 (일관성 유지)
                    # final_event의 message_id와 실제 저장된 message_id가 일치하도록 함
                    expected_message_id = metadata.get("message_id")
                    saved_message_id = session_service.add_message(
                        session_id=stream_request.session_id,
                        role="assistant",
                        content=full_answer,
                        metadata=metadata,
                        message_id=expected_message_id  # final_event의 message_id 사용
                    )
                    
                    # sources가 포함된 metadata를 별도 이벤트로 전송 (프론트엔드에서 사용)
                    # final_event를 이미 보냈으므로, sources가 포함된 metadata를 별도 이벤트로 전송
                    if metadata.get("sources") or metadata.get("legal_references") or metadata.get("sources_detail") or metadata.get("related_questions"):
                        sources_event = {
                            "type": "sources",
                            "metadata": {
                                "message_id": saved_message_id,
                                "sources": metadata.get("sources", []),
                                "legal_references": metadata.get("legal_references", []),
                                "sources_detail": metadata.get("sources_detail", []),
                                "related_questions": metadata.get("related_questions", [])
                            },
                            "timestamp": datetime.now().isoformat()
                        }
                        logger.debug(f"Sending sources_event with related_questions: {len(metadata.get('related_questions', []))} questions")
                        yield f"data: {json.dumps(sources_event, ensure_ascii=False)}\n\n"
                    
                    # 세션에 제목이 없고 첫 번째 대화라면 제목 생성 (비동기)
                    session = session_service.get_session(stream_request.session_id)
                    if session and not session.get("title"):
                        messages = session_service.get_messages(stream_request.session_id)
                        if len(messages) == 2:  # user + assistant
                            asyncio.create_task(
                                asyncio.to_thread(
                                    session_service.generate_session_title,
                                    stream_request.session_id
                                )
                            )
                
                # 스트리밍이 정상적으로 완료됨
                stream_closed = True
                
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
                    has_yielded = True
                    stream_closed = True
                except Exception as yield_error:
                    logger.error(f"Error yielding error message: {yield_error}")
                    # yield 자체가 실패한 경우에도 스트림이 정상 종료되도록 보장
                    stream_closed = True
        
        return StreamingResponse(
            generate(),
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

