"""
히스토리 관리 엔드포인트
"""
import logging
from fastapi import APIRouter, HTTPException, Query, Depends, Request
from fastapi.responses import Response
from typing import Optional
import json

from api.schemas.history import HistoryQuery, HistoryResponse, MessageResponse, ExportRequest
from api.services.session_service import session_service
from api.services.history_service import history_service
from api.middleware.auth_middleware import require_auth

router = APIRouter()
logger = logging.getLogger(__name__)


@router.get("/history", response_model=HistoryResponse)
async def get_history(
    request: Request,
    current_user: dict = Depends(require_auth),
    session_id: Optional[str] = Query(None, description="세션 ID 필터"),
    category: Optional[str] = Query(None, description="카테고리 필터"),
    search: Optional[str] = Query(None, description="검색어"),
    page: int = Query(1, ge=1, description="페이지 번호"),
    page_size: int = Query(10, ge=1, le=1000, description="페이지 크기"),
    sort_by: str = Query("updated_at", description="정렬 기준"),
    sort_order: str = Query("desc", regex="^(asc|desc)$", description="정렬 순서 (asc 또는 desc)")
):
    """히스토리 조회"""
    from datetime import datetime
    
    try:
        # 사용자별 필터링
        user_id = None
        
        if current_user.get("authenticated"):
            # 로그인 사용자: user_id로 필터링
            user_id = current_user.get("user_id")
        else:
            # 비회원: anonymous_session_id로 필터링
            anonymous_session_id = request.headers.get("X-Anonymous-Session-Id")
            if anonymous_session_id:
                user_id = f"anonymous_{anonymous_session_id}"
        
        result = history_service.get_history(
            session_id=session_id,
            category=category,
            search=search,
            page=page,
            page_size=page_size,
            user_id=user_id
        )
        
        # 메시지를 MessageResponse로 변환
        message_responses = []
        for msg in result.get("messages", []):
            try:
                # 필수 필드 검증
                if not msg.get("message_id"):
                    logger.warning(f"Message missing message_id: {msg}")
                    continue
                if not msg.get("session_id"):
                    logger.warning(f"Message missing session_id: {msg}")
                    continue
                if not msg.get("role"):
                    logger.warning(f"Message missing role: {msg}")
                    continue
                if not msg.get("content"):
                    msg["content"] = ""  # 빈 내용은 허용
                
                # timestamp 변환 처리
                timestamp = msg.get("timestamp")
                if timestamp is None:
                    # timestamp가 None이면 현재 시간 사용
                    timestamp = datetime.now()
                elif isinstance(timestamp, str):
                    try:
                        # ISO 형식 문자열을 datetime으로 변환
                        timestamp_str = timestamp.replace("Z", "+00:00")
                        try:
                            timestamp = datetime.fromisoformat(timestamp_str)
                        except ValueError:
                            # 다른 형식 시도
                            try:
                                timestamp = datetime.strptime(timestamp, "%Y-%m-%d %H:%M:%S")
                            except ValueError:
                                try:
                                    timestamp = datetime.strptime(timestamp, "%Y-%m-%d %H:%M:%S.%f")
                                except ValueError:
                                    # 모두 실패하면 현재 시간 사용
                                    logger.warning(f"Failed to parse timestamp: {timestamp}, using current time")
                                    timestamp = datetime.now()
                    except Exception as e:
                        logger.warning(f"Error parsing timestamp '{timestamp}': {e}, using current time")
                        timestamp = datetime.now()
                elif not isinstance(timestamp, datetime):
                    # datetime 객체가 아니면 현재 시간 사용
                    logger.warning(f"Invalid timestamp type: {type(timestamp)}, using current time")
                    timestamp = datetime.now()
                
                msg["timestamp"] = timestamp
                
                # metadata 기본값 설정
                if not msg.get("metadata"):
                    msg["metadata"] = {}
                elif isinstance(msg["metadata"], str):
                    # 문자열인 경우 JSON 파싱 시도
                    try:
                        import json
                        msg["metadata"] = json.loads(msg["metadata"])
                    except:
                        msg["metadata"] = {}
                
                message_responses.append(MessageResponse(**msg))
            except Exception as e:
                logger.error(f"Failed to convert message to MessageResponse: {e}, message: {msg}", exc_info=True)
                # 개별 메시지 변환 실패는 건너뛰고 계속 진행
                continue
        
        return HistoryResponse(
            messages=message_responses,
            total=result.get("total", 0),
            page=page,
            page_size=page_size
        )
    except ValueError as e:
        logger.warning(f"Validation error in get_history: {e}")
        raise HTTPException(status_code=400, detail="입력 데이터가 올바르지 않습니다")
    except Exception as e:
        logger.error(f"Error in get_history: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="히스토리 조회 중 오류가 발생했습니다")


@router.post("/history/export")
async def export_history(
    request: Request,
    export_request: ExportRequest,
    current_user: dict = Depends(require_auth)
):
    """히스토리 내보내기"""
    try:
        # 사용자별 필터링
        user_id = None
        
        if current_user.get("authenticated"):
            # 로그인 사용자: user_id로 필터링
            user_id = current_user.get("user_id")
        else:
            # 비회원: anonymous_session_id로 필터링
            anonymous_session_id = request.headers.get("X-Anonymous-Session-Id")
            if anonymous_session_id:
                user_id = f"anonymous_{anonymous_session_id}"
        
        export_data = []
        
        for session_id in export_request.session_ids:
            session = session_service.get_session(session_id)
            if session:
                # 세션 소유권 확인
                if user_id:
                    session_user_id = session.get("user_id")
                    if session_user_id != user_id:
                        # 소유권이 없으면 건너뛰기
                        continue
                
                messages = session_service.get_messages(session_id, user_id=user_id)
                export_data.append({
                    "session": session,
                    "messages": messages
                })
        
        if export_request.format == "json":
            content = json.dumps(export_data, ensure_ascii=False, indent=2)
            return Response(
                content=content,
                media_type="application/json",
                headers={
                    "Content-Disposition": "attachment; filename=history.json"
                }
            )
        elif export_request.format == "txt":
            # 텍스트 형식으로 변환
            lines = []
            for item in export_data:
                session = item["session"]
                lines.append(f"=== 세션: {session.get('title', session['session_id'])} ===")
                lines.append(f"생성일: {session.get('created_at')}")
                lines.append("")
                
                for msg in item["messages"]:
                    role = "사용자" if msg["role"] == "user" else "AI"
                    lines.append(f"[{role}] {msg['timestamp']}")
                    lines.append(msg["content"])
                    lines.append("")
                
                lines.append("=" * 50)
                lines.append("")
            
            content = "\n".join(lines)
            return Response(
                content=content,
                media_type="text/plain",
                headers={
                    "Content-Disposition": "attachment; filename=history.txt"
                }
            )
        else:
            raise HTTPException(status_code=400, detail="Invalid format. Use 'json' or 'txt'")
    except HTTPException:
        raise
    except ValueError as e:
        logger.warning(f"Validation error in export_history: {e}")
        raise HTTPException(status_code=400, detail="입력 데이터가 올바르지 않습니다")
    except Exception as e:
        logger.error(f"Error in export_history: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="히스토리 내보내기 중 오류가 발생했습니다")

