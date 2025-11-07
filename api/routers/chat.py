"""
채팅 엔드포인트
"""
import logging
from fastapi import APIRouter, HTTPException, Request, Depends
from fastapi.responses import StreamingResponse
from typing import AsyncGenerator
from datetime import datetime

from api.schemas.chat import ChatRequest, ChatResponse, StreamingChatRequest
from api.services.chat_service import get_chat_service
from api.services.session_service import session_service
from api.services.ocr_service import extract_text_from_base64, is_ocr_available
from api.services.file_processor import process_file
from api.middleware.auth_middleware import require_auth
from api.middleware.rate_limit import limiter, is_rate_limit_enabled

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
        
        return ChatResponse(**result)
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
            
            try:
                async for chunk in chat_service.stream_message(
                    message=final_message,
                    session_id=stream_request.session_id
                ):
                    if chunk:
                        has_yielded = True
                        chunk_stripped = chunk.strip()
                        
                        # JSONL 형식 파싱 시도
                        import json
                        try:
                            event_data = json.loads(chunk_stripped)
                            event_type = event_data.get("type", "")
                            
                            # "stream" 타입인 경우에만 full_answer에 추가
                            if event_type == "stream":
                                content = event_data.get("content", "")
                                if content:
                                    full_answer += content
                            
                            # "final" 타입인 경우 full_answer 업데이트
                            elif event_type == "final":
                                content = event_data.get("content", "")
                                if content and not content.startswith("[오류]"):
                                    full_answer = content
                            
                            # JSONL 이벤트를 SSE 형식으로 변환
                            # JSON 이스케이프는 json.dumps가 자동으로 처리
                            json_str = json.dumps(event_data, ensure_ascii=False)
                            # SSE 형식으로 전송
                            yield f"data: {json_str}\n\n"
                            
                        except (json.JSONDecodeError, ValueError):
                            # JSON 파싱 실패 시 기존 형식으로 처리 (하위 호환성)
                            # 완료 메타데이터는 제외
                            if '[스트리밍 완료]' not in chunk and '[완료]' not in chunk:
                                full_answer += chunk
                                # SSE 형식으로 전송 (줄바꿈 보존)
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
                # FastAPI StreamingResponse가 자동으로 스트림 종료를 처리
                
                # 완료 후 메시지 저장
                if full_answer:
                    session_service.add_message(
                        session_id=stream_request.session_id,
                        role="assistant",
                        content=full_answer
                    )
                    
                    # 세션에 제목이 없고 첫 번째 대화라면 제목 생성 (비동기)
                    session = session_service.get_session(stream_request.session_id)
                    if session and not session.get("title"):
                        messages = session_service.get_messages(stream_request.session_id)
                        if len(messages) == 2:  # user + assistant
                            import asyncio
                            asyncio.create_task(
                                asyncio.to_thread(
                                    session_service.generate_session_title,
                                    stream_request.session_id
                                )
                            )
            except Exception as e:
                import logging
                logger = logging.getLogger(__name__)
                logger.error(f"Error in stream_message: {e}", exc_info=True)
                error_msg = f"[오류] {str(e)}"
                try:
                    # 에러 메시지를 JSON 형식으로 전송
                    import json
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
                except Exception as yield_error:
                    logger.error(f"Error yielding error message: {yield_error}")
            finally:
                # 최소한 하나의 yield가 없었으면 에러 메시지 전송
                if not has_yielded:
                    try:
                        import json
                        error_event = {
                            "type": "final",
                            "content": "[오류] 스트리밍 응답을 생성할 수 없습니다.",
                            "metadata": {"error": True},
                            "timestamp": datetime.now().isoformat()
                        }
                        yield f"data: {json.dumps(error_event, ensure_ascii=False)}\n\n"
                    except Exception as e:
                        import logging
                        logger = logging.getLogger(__name__)
                        logger.error(f"Error yielding fallback message: {e}")
                
                # HTTP chunked encoding이 제대로 종료되도록 보장
                # SSE 이벤트 종료 신호: 빈 줄 2개 yield
                # 이는 ERR_INCOMPLETE_CHUNKED_ENCODING 오류를 방지
                try:
                    yield "\n\n"  # SSE 이벤트 종료 신호 (빈 줄 2개)
                except Exception:
                    pass
        
        return StreamingResponse(
            generate(),
            media_type="text/event-stream",
            headers={
                "Cache-Control": "no-cache",
                "Connection": "keep-alive",
                "X-Accel-Buffering": "no",  # Nginx 버퍼링 비활성화
                "Content-Type": "text/event-stream; charset=utf-8",
                "Transfer-Encoding": "chunked",  # 명시적으로 chunked encoding 지정
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

