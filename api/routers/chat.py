"""
채팅 엔드포인트
"""
from fastapi import APIRouter, HTTPException
from fastapi.responses import StreamingResponse
from typing import AsyncGenerator

from api.schemas.chat import ChatRequest, ChatResponse, StreamingChatRequest
from api.services.chat_service import get_chat_service
from api.services.session_service import session_service

router = APIRouter()


@router.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest):
    """채팅 메시지 처리"""
    try:
        # ChatService 가져오기 (지연 초기화)
        chat_service = get_chat_service()
        
        # 세션이 없으면 생성
        if not request.session_id:
            request.session_id = session_service.create_session()
        
        # 사용자 메시지 저장
        session_service.add_message(
            session_id=request.session_id,
            role="user",
            content=request.message
        )
        
        # AI 답변 생성
        result = await chat_service.process_message(
            message=request.message,
            session_id=request.session_id,
            enable_checkpoint=request.enable_checkpoint
        )
        
        # AI 답변 저장
        session_service.add_message(
            session_id=request.session_id,
            role="assistant",
            content=result.get("answer", ""),
            metadata=result.get("metadata", {})
        )
        
        # 세션에 제목이 없고 첫 번째 대화라면 제목 생성 (비동기)
        session = session_service.get_session(request.session_id)
        if session and not session.get("title"):
            messages = session_service.get_messages(request.session_id)
            if len(messages) == 2:  # user + assistant
                # 백그라운드에서 제목 생성 (응답 지연 방지)
                import asyncio
                asyncio.create_task(
                    asyncio.to_thread(
                        session_service.generate_session_title,
                        request.session_id
                    )
                )
        
        return ChatResponse(**result)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/chat/stream")
async def chat_stream(request: StreamingChatRequest):
    """스트리밍 채팅 응답"""
    try:
        # ChatService 가져오기 (지연 초기화)
        chat_service = get_chat_service()
        
        # 세션이 없으면 생성
        if not request.session_id:
            request.session_id = session_service.create_session()
        
        # 사용자 메시지 저장
        session_service.add_message(
            session_id=request.session_id,
            role="user",
            content=request.message
        )
        
        async def generate() -> AsyncGenerator[str, None]:
            """스트리밍 응답 생성"""
            full_answer = ""
            stream_completed = False
            try:
                async for chunk in chat_service.stream_message(
                    message=request.message,
                    session_id=request.session_id
                ):
                    if chunk:
                        # 완료 메타데이터는 제외 (줄바꿈이 앞에 있을 수 있으므로 포함 여부로 체크)
                        if '[스트리밍 완료]' not in chunk and '[완료]' not in chunk:
                            full_answer += chunk
                            # SSE 형식으로 전송 (줄바꿈 보존)
                            # 줄바꿈이 포함된 경우 SSE 다중 라인 형식으로 처리
                            # SSE 표준: 각 줄을 "data: "로 시작하면 줄바꿈으로 결합됨
                            if '\n' in chunk:
                                # 줄바꿈이 포함된 경우 다중 라인으로 전송
                                # 각 줄을 "data: "로 시작하는 별도 라인으로 전송하여 줄바꿈 보존
                                lines = chunk.split('\n')
                                for line in lines:
                                    # 빈 줄도 보존 (SSE 표준에 따르면 "data: \n"로 전송)
                                    yield f"data: {line}\n"
                                yield "\n"  # SSE 이벤트 종료 (빈 줄)
                            else:
                                # 줄바꿈이 없는 경우 일반 형식
                                yield f"data: {chunk}\n\n"
                
                stream_completed = True
                
                # 완료 후 메시지 저장
                if full_answer:
                    session_service.add_message(
                        session_id=request.session_id,
                        role="assistant",
                        content=full_answer
                    )
                    
                    # 세션에 제목이 없고 첫 번째 대화라면 제목 생성 (비동기)
                    session = session_service.get_session(request.session_id)
                    if session and not session.get("title"):
                        messages = session_service.get_messages(request.session_id)
                        if len(messages) == 2:  # user + assistant
                            import asyncio
                            asyncio.create_task(
                                asyncio.to_thread(
                                    session_service.generate_session_title,
                                    request.session_id
                                )
                            )
            except Exception as e:
                import logging
                logger = logging.getLogger(__name__)
                logger.error(f"Error in stream_message: {e}", exc_info=True)
                error_msg = f"[오류] {str(e)}"
                try:
                    yield f"data: {error_msg}\n\n"
                except Exception as yield_error:
                    logger.error(f"Error yielding error message: {yield_error}")
            finally:
                # 스트리밍 완료 보장 (항상 실행)
                # 완료 신호를 명시적으로 전송하여 스트림 종료를 보장
                try:
                    yield f"data: [스트리밍 완료]\n\n"
                    # 스트림 종료를 명확히 하기 위해 추가 빈 줄 전송
                    yield "\n"
                except Exception as e:
                    import logging
                    logger = logging.getLogger(__name__)
                    logger.error(f"Error sending completion signal: {e}")
        
        return StreamingResponse(
            generate(),
            media_type="text/event-stream",
            headers={
                "Cache-Control": "no-cache",
                "Connection": "keep-alive",
                "X-Accel-Buffering": "no",  # Nginx 버퍼링 비활성화
                "Content-Type": "text/event-stream; charset=utf-8",
            }
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

