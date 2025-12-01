"""
세션 관리 엔드포인트
"""
from fastapi import APIRouter, HTTPException, Query, Depends
from fastapi import Request
from typing import Optional
from datetime import datetime, timedelta, timezone
import logging

from api.schemas.session import (
    SessionCreate,
    SessionUpdate,
    SessionResponse,
    SessionListResponse
)
from api.services.session_service import session_service
from api.middleware.auth_middleware import require_auth
from api.config import api_config

router = APIRouter()
logger = logging.getLogger(__name__)

# KST 시간대 (UTC+9)
KST = timezone(timedelta(hours=9))


def get_kst_now() -> datetime:
    """KST 기준 현재 시간 반환"""
    return datetime.now(KST)


def get_kst_date() -> datetime.date:
    """KST 기준 오늘 날짜 반환"""
    return get_kst_now().date()


def get_user_info(request: Request, current_user: dict) -> tuple[Optional[str], str]:
    """
    요청에서 user_id와 ip_address 추출
    
    Returns:
        tuple: (user_id, ip_address)
    """
    # 클라이언트 IP 주소 가져오기
    client_ip = request.client.host if request.client else None
    if not client_ip:
        forwarded_for = request.headers.get("X-Forwarded-For")
        if forwarded_for:
            client_ip = forwarded_for.split(",")[0].strip()
        else:
            client_ip = "unknown"
    
    # 사용자 ID 가져오기
    user_id = None
    if current_user and current_user.get("authenticated"):
        user_id = current_user.get("user_id")
    else:
        anonymous_session_id = request.headers.get("X-Anonymous-Session-Id")
        if anonymous_session_id:
            user_id = f"anonymous_{anonymous_session_id}"
    
    return user_id, client_ip


@router.get("/sessions", response_model=SessionListResponse)
async def list_sessions(
    request: Request,
    current_user: dict = Depends(require_auth),
    search: Optional[str] = Query(None, description="검색어"),
    page: int = Query(1, ge=1, description="페이지 번호"),
    page_size: int = Query(10, ge=1, le=1000, description="페이지 크기"),
    sort_by: str = Query("updated_at", description="정렬 기준"),
    sort_order: str = Query("desc", regex="^(asc|desc)$", description="정렬 순서 (asc 또는 desc)"),
    date_from: Optional[str] = Query(None, description="시작 날짜 (YYYY-MM-DD)"),
    date_to: Optional[str] = Query(None, description="종료 날짜 (YYYY-MM-DD)")
):
    """세션 목록 조회"""
    try:
        # 사용자별 필터링
        user_id = None
        ip_address = None
        
        if current_user.get("authenticated"):
            user_id = current_user.get("user_id")
        else:
            # 비로그인 사용자의 경우 익명 세션 ID 또는 IP 주소로 필터링
            anonymous_session_id = request.headers.get("X-Anonymous-Session-Id")
            if anonymous_session_id:
                # 익명 세션 ID를 user_id처럼 사용 (prefix로 구분)
                user_id = f"anonymous_{anonymous_session_id}"
            else:
                # 익명 세션 ID가 없으면 IP 주소로 필터링 (하위 호환성)
                client_ip = request.client.host if request.client else None
                if not client_ip:
                    forwarded_for = request.headers.get("X-Forwarded-For")
                    if forwarded_for:
                        client_ip = forwarded_for.split(",")[0].strip()
                    else:
                        client_ip = "unknown"
                ip_address = client_ip
        
        sessions, total = session_service.list_sessions(
            search=search,
            page=page,
            page_size=page_size,
            sort_by=sort_by,
            sort_order=sort_order,
            date_from=date_from,
            date_to=date_to,
            user_id=user_id,
            ip_address=ip_address
        )
        
        # 세션 목록을 SessionResponse로 변환
        session_responses = []
        for session in sessions:
            try:
                # datetime 객체를 문자열로 변환
                if isinstance(session.get("created_at"), datetime):
                    session["created_at"] = session["created_at"].isoformat()
                if isinstance(session.get("updated_at"), datetime):
                    session["updated_at"] = session["updated_at"].isoformat()
                session_responses.append(SessionResponse(**session))
            except Exception as e:
                import logging
                logging.error(f"Failed to convert session to response: {e}, session: {session}")
                continue
        
        return SessionListResponse(
            sessions=session_responses,
            total=total,
            page=page,
            page_size=page_size
        )
    except ValueError as e:
        logger.warning(f"Validation error in list_sessions: {e}")
        raise HTTPException(status_code=400, detail="입력 데이터가 올바르지 않습니다")
    except Exception as e:
        logger.error(f"Error in list_sessions: {e}", exc_info=True)
        import traceback
        error_traceback = traceback.format_exc()
        logger.error(f"Traceback: {error_traceback}")
        error_detail = f"{str(e)}\n\n{error_traceback}"
        print(f"[ERROR] list_sessions failed: {error_detail}", flush=True)
        raise HTTPException(status_code=500, detail=error_detail)


@router.get("/sessions/by-date", response_model=SessionListResponse)
async def get_sessions_by_date(
    request: Request,
    current_user: dict = Depends(require_auth),
    date_group: str = Query(..., description="날짜 그룹: today, yesterday, week, month, older"),
    page: int = Query(1, ge=1, description="페이지 번호"),
    page_size: int = Query(20, ge=1, le=100, description="페이지 크기"),
    search: Optional[str] = Query(None, description="검색어")
):
    """날짜별 그룹 세션 조회 (KST 기준)"""
    try:
        now_kst = get_kst_now()
        date_from = None
        date_to = None
        
        # 날짜 그룹에 따라 날짜 범위 설정 (KST 기준)
        if date_group == "today":
            # 오늘 (KST 기준)
            today_kst = now_kst.date()
            date_from = today_kst.isoformat()
            date_to = today_kst.isoformat()
        elif date_group == "yesterday":
            # 어제 (KST 기준)
            yesterday_kst = (now_kst - timedelta(days=1)).date()
            date_from = yesterday_kst.isoformat()
            date_to = yesterday_kst.isoformat()
        elif date_group == "week":
            # 지난 7일 (어제까지, 오늘 제외) (KST 기준)
            week_start_kst = (now_kst - timedelta(days=7)).date()
            yesterday_kst = (now_kst - timedelta(days=1)).date()
            date_from = week_start_kst.isoformat()
            date_to = yesterday_kst.isoformat()
        elif date_group == "month":
            # 지난 30일 (어제까지, 오늘 제외) (KST 기준)
            month_start_kst = (now_kst - timedelta(days=30)).date()
            yesterday_kst = (now_kst - timedelta(days=1)).date()
            date_from = month_start_kst.isoformat()
            date_to = yesterday_kst.isoformat()
        elif date_group == "older":
            # 30일 이전 (KST 기준)
            older_date_kst = (now_kst - timedelta(days=30)).date()
            date_to = older_date_kst.isoformat()
        else:
            raise HTTPException(
                status_code=400, 
                detail=f"Invalid date_group: {date_group}. Must be one of: today, yesterday, week, month, older"
            )
        
        # 사용자별 필터링
        user_id = None
        ip_address = None
        
        if current_user.get("authenticated"):
            user_id = current_user.get("user_id")
        else:
            # 비로그인 사용자의 경우 익명 세션 ID 또는 IP 주소로 필터링
            anonymous_session_id = request.headers.get("X-Anonymous-Session-Id")
            if anonymous_session_id:
                # 익명 세션 ID를 user_id처럼 사용 (prefix로 구분)
                user_id = f"anonymous_{anonymous_session_id}"
            else:
                # 익명 세션 ID가 없으면 IP 주소로 필터링 (하위 호환성)
                client_ip = request.client.host if request.client else None
                if not client_ip:
                    forwarded_for = request.headers.get("X-Forwarded-For")
                    if forwarded_for:
                        client_ip = forwarded_for.split(",")[0].strip()
                    else:
                        client_ip = "unknown"
                ip_address = client_ip
        
        sessions, total = session_service.list_sessions(
            search=search,
            page=page,
            page_size=page_size,
            sort_by="updated_at",
            sort_order="desc",
            date_from=date_from,
            date_to=date_to,
            user_id=user_id,
            ip_address=ip_address
        )
        
        # 세션 목록을 SessionResponse로 변환
        session_responses = []
        for session in sessions:
            try:
                # datetime 객체를 문자열로 변환
                if isinstance(session.get("created_at"), datetime):
                    session["created_at"] = session["created_at"].isoformat()
                if isinstance(session.get("updated_at"), datetime):
                    session["updated_at"] = session["updated_at"].isoformat()
                session_responses.append(SessionResponse(**session))
            except Exception as e:
                logging.error(f"Failed to convert session to response: {e}, session: {session}")
                continue
        
        return SessionListResponse(
            sessions=session_responses,
            total=total,
            page=page,
            page_size=page_size
        )
    except HTTPException:
        raise
    except ValueError as e:
        logger.warning(f"Validation error in get_sessions_by_date: {e}")
        raise HTTPException(status_code=400, detail="입력 데이터가 올바르지 않습니다")
    except Exception as e:
        logger.error(f"Error in get_sessions_by_date: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="세션 조회 중 오류가 발생했습니다")


@router.post("/sessions", response_model=SessionResponse)
async def create_session(
    session: SessionCreate,
    request: Request,
    current_user: dict = Depends(require_auth)
):
    """새 세션 생성"""
    try:
        user_id, client_ip = get_user_info(request, current_user)
        logger.info(f"[create_session] Request received: title={session.title}, user_id={user_id}, ip={client_ip}")
        
        session_id = session_service.create_session(
            title=session.title,
            user_id=user_id,
            ip_address=client_ip
        )
        
        logger.info(f"[create_session] Session created: session_id={session_id}")
        
        created_session = session_service.get_session(session_id, check_expiry=False)
        if not created_session:
            logger.error(f"[create_session] Session created but not found immediately: session_id={session_id}")
            raise HTTPException(status_code=404, detail="Session not found")
        
        logger.info(f"[create_session] Session retrieved successfully: session_id={session_id}")
        
        # datetime 객체를 문자열로 변환
        if isinstance(created_session.get("created_at"), datetime):
            created_session["created_at"] = created_session["created_at"].isoformat()
        if isinstance(created_session.get("updated_at"), datetime):
            created_session["updated_at"] = created_session["updated_at"].isoformat()
        
        return SessionResponse(**created_session)
    except HTTPException:
        raise
    except ValueError as e:
        logger.warning(f"Validation error in create_session: {e}")
        raise HTTPException(status_code=400, detail=f"입력 데이터가 올바르지 않습니다: {str(e)}")
    except Exception as e:
        logger.error(f"Error in create_session: {e}", exc_info=True)
        import traceback
        error_traceback = traceback.format_exc()
        logger.error(f"Traceback: {error_traceback}")
        # 항상 상세한 오류 메시지 반환 (개발 중이므로)
        error_detail = f"{str(e)}\n\n{error_traceback}"
        print(f"[ERROR] create_session failed: {error_detail}", flush=True)
        raise HTTPException(status_code=500, detail=error_detail)


@router.get("/sessions/{session_id}", response_model=SessionResponse)
async def get_session(
    request: Request,
    session_id: str,
    current_user: dict = Depends(require_auth)
):
    """세션 상세 조회"""
    try:
        logger.info(f"[get_session] Request received: session_id={session_id}, user_id={current_user.get('user_id') if current_user else None}")
        
        # 세션 ID 형식 검증 (UUID 형식인지 확인)
        import re
        uuid_pattern = re.compile(r'^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$', re.IGNORECASE)
        
        # UUID 형식이 아닌 경우, OAuth2 토큰 교환 세션 ID일 가능성 확인
        if not uuid_pattern.match(session_id):
            # OAuth2 토큰 교환 세션 ID는 base64url 형식 (약 43자, - 또는 _ 포함 가능)
            # secrets.token_urlsafe(32)로 생성됨
            base64url_pattern = re.compile(r'^[A-Za-z0-9_-]{30,50}$')
            
            if base64url_pattern.match(session_id):
                # OAuth2 토큰 스토어에서 확인 (간접적으로 확인)
                # 실제로는 데이터베이스 조회 후 없으면 OAuth2 세션 ID일 가능성 높음
                logger.warning(f"[get_session] Non-UUID session ID format detected: {session_id[:20]}... (might be OAuth2 token exchange session ID)")
            else:
                logger.warning(f"[get_session] Invalid session ID format: {session_id} (expected UUID format)")
                raise HTTPException(
                    status_code=400, 
                    detail=f"Invalid session ID format. Expected UUID format for chat sessions, got: {session_id[:50]}..."
                )
        
        session = session_service.get_session(session_id, check_expiry=False)  # 만료 확인 비활성화하여 디버깅
        if not session:
            logger.warning(f"[get_session] Session not found in database: {session_id}")
            # 데이터베이스에 직접 확인
            from api.database.connection import get_session as get_db_session
            from api.database.models import Session as SessionModel
            db = get_db_session()
            try:
                direct_check = db.query(SessionModel).filter(SessionModel.session_id == session_id).first()
                if direct_check:
                    logger.error(f"[get_session] Session exists in DB but get_session() returned None! session_id={session_id}, title={direct_check.title}")
                else:
                    logger.info(f"[get_session] Session does not exist in database: {session_id}")
                    
                    # UUID 형식이 아니고 데이터베이스에 없으면 OAuth2 토큰 교환 세션 ID일 가능성
                    if not uuid_pattern.match(session_id):
                        base64url_pattern = re.compile(r'^[A-Za-z0-9_-]{30,50}$')
                        if base64url_pattern.match(session_id):
                            logger.warning(f"[get_session] This appears to be an OAuth2 token exchange session ID, not a chat session ID: {session_id[:20]}...")
                            raise HTTPException(
                                status_code=400,
                                detail="This appears to be an OAuth2 token exchange session ID, not a chat session ID. Please create a new chat session or use a valid chat session ID."
                            )
                    
                    # 유사한 세션 ID 검색 (디버깅용)
                    similar_sessions = db.query(SessionModel).filter(
                        SessionModel.session_id.like(f"%{session_id[:10]}%")
                    ).limit(5).all()
                    if similar_sessions:
                        logger.debug(f"[get_session] Found {len(similar_sessions)} similar session IDs (first 10 chars match)")
            except HTTPException:
                raise
            except Exception as db_error:
                logger.error(f"[get_session] Error checking database directly: {db_error}", exc_info=True)
            finally:
                db.close()
            raise HTTPException(
                status_code=404, 
                detail=f"Session not found. The session ID '{session_id[:50]}...' does not exist in the database."
            )
        
        # 소유자 확인
        if current_user.get("authenticated"):
            user_id = current_user.get("user_id")
            session_user_id = session.get("user_id")
            if session_user_id and session_user_id != user_id:
                raise HTTPException(status_code=403, detail="이 세션에 접근할 권한이 없습니다.")
        else:
            # 비로그인 사용자의 경우 익명 세션 ID 또는 IP 주소로 확인
            anonymous_session_id = request.headers.get("X-Anonymous-Session-Id")
            session_user_id = session.get("user_id")
            
            if anonymous_session_id:
                # 익명 세션 ID로 확인
                expected_user_id = f"anonymous_{anonymous_session_id}"
                if session_user_id and session_user_id != expected_user_id:
                    raise HTTPException(status_code=403, detail="이 세션에 접근할 권한이 없습니다.")
            else:
                # IP 주소로 확인 (하위 호환성)
                client_ip = request.client.host if request.client else None
                if not client_ip:
                    forwarded_for = request.headers.get("X-Forwarded-For")
                    if forwarded_for:
                        client_ip = forwarded_for.split(",")[0].strip()
                    else:
                        client_ip = "unknown"
                session_ip = session.get("ip_address")
                if session_user_id or (session_ip and session_ip != client_ip):
                    raise HTTPException(status_code=403, detail="이 세션에 접근할 권한이 없습니다.")
        
        # datetime 객체를 문자열로 변환
        if isinstance(session.get("created_at"), datetime):
            session["created_at"] = session["created_at"].isoformat()
        if isinstance(session.get("updated_at"), datetime):
            session["updated_at"] = session["updated_at"].isoformat()
        
        return SessionResponse(**session)
    except HTTPException:
        raise
    except ValueError as e:
        logger.warning(f"Validation error in get_session: {e}")
        raise HTTPException(status_code=400, detail="입력 데이터가 올바르지 않습니다")
    except Exception as e:
        logger.error(f"Error in get_session: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="세션 조회 중 오류가 발생했습니다")


@router.put("/sessions/{session_id}", response_model=SessionResponse)
async def update_session(
    request: Request,
    session_id: str,
    session: SessionUpdate,
    current_user: dict = Depends(require_auth)
):
    """세션 업데이트"""
    try:
        # 소유자 확인
        existing_session = session_service.get_session(session_id)
        if not existing_session:
            raise HTTPException(status_code=404, detail="Session not found")
        
        if current_user.get("authenticated"):
            user_id = current_user.get("user_id")
            session_user_id = existing_session.get("user_id")
            if session_user_id and session_user_id != user_id:
                raise HTTPException(status_code=403, detail="이 세션을 수정할 권한이 없습니다.")
        else:
            # 비로그인 사용자의 경우 익명 세션 ID 또는 IP 주소로 확인
            anonymous_session_id = request.headers.get("X-Anonymous-Session-Id")
            session_user_id = existing_session.get("user_id")
            
            if anonymous_session_id:
                # 익명 세션 ID로 확인
                expected_user_id = f"anonymous_{anonymous_session_id}"
                if session_user_id and session_user_id != expected_user_id:
                    raise HTTPException(status_code=403, detail="이 세션을 수정할 권한이 없습니다.")
                user_id = expected_user_id
            else:
                # IP 주소로 확인 (하위 호환성)
                client_ip = request.client.host if request.client else None
                if not client_ip:
                    forwarded_for = request.headers.get("X-Forwarded-For")
                    if forwarded_for:
                        client_ip = forwarded_for.split(",")[0].strip()
                    else:
                        client_ip = "unknown"
                session_ip = existing_session.get("ip_address")
                if session_user_id or (session_ip and session_ip != client_ip):
                    raise HTTPException(status_code=403, detail="이 세션을 수정할 권한이 없습니다.")
                user_id = None
        
        success = session_service.update_session(
            session_id=session_id,
            title=session.title
        )
        
        if not success:
            raise HTTPException(status_code=404, detail="Session not found")
        
        updated_session = session_service.get_session(session_id)
        if not updated_session:
            raise HTTPException(status_code=404, detail="Session not found")
        
        # datetime 객체를 문자열로 변환
        if isinstance(updated_session.get("created_at"), datetime):
            updated_session["created_at"] = updated_session["created_at"].isoformat()
        if isinstance(updated_session.get("updated_at"), datetime):
            updated_session["updated_at"] = updated_session["updated_at"].isoformat()
        
        return SessionResponse(**updated_session)
    except HTTPException:
        raise
    except ValueError as e:
        logger.warning(f"Validation error in update_session: {e}")
        raise HTTPException(status_code=400, detail="입력 데이터가 올바르지 않습니다")
    except Exception as e:
        logger.error(f"Error in update_session: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="세션 업데이트 중 오류가 발생했습니다")


@router.delete("/sessions/{session_id}")
async def delete_session(
    request: Request,
    session_id: str,
    current_user: dict = Depends(require_auth)
):
    """세션 삭제"""
    try:
        # 소유자 확인
        existing_session = session_service.get_session(session_id)
        if not existing_session:
            raise HTTPException(status_code=404, detail="Session not found")
        
        user_id = None
        if current_user.get("authenticated"):
            user_id = current_user.get("user_id")
            session_user_id = existing_session.get("user_id")
            if session_user_id and session_user_id != user_id:
                raise HTTPException(status_code=403, detail="이 세션을 삭제할 권한이 없습니다.")
        else:
            # 비로그인 사용자의 경우 익명 세션 ID 또는 IP 주소로 확인
            anonymous_session_id = request.headers.get("X-Anonymous-Session-Id")
            session_user_id = existing_session.get("user_id")
            
            if anonymous_session_id:
                # 익명 세션 ID로 확인
                expected_user_id = f"anonymous_{anonymous_session_id}"
                if session_user_id and session_user_id != expected_user_id:
                    raise HTTPException(status_code=403, detail="이 세션을 삭제할 권한이 없습니다.")
                user_id = expected_user_id
            else:
                # IP 주소로 확인 (하위 호환성)
                client_ip = request.client.host if request.client else None
                if not client_ip:
                    forwarded_for = request.headers.get("X-Forwarded-For")
                    if forwarded_for:
                        client_ip = forwarded_for.split(",")[0].strip()
                    else:
                        client_ip = "unknown"
                session_ip = existing_session.get("ip_address")
                if session_user_id or (session_ip and session_ip != client_ip):
                    raise HTTPException(status_code=403, detail="이 세션을 삭제할 권한이 없습니다.")
        
        success = session_service.delete_session(session_id, user_id=user_id)
        if not success:
            raise HTTPException(status_code=404, detail="Session not found")
        
        return {"message": "Session deleted successfully"}
    except HTTPException:
        raise
    except ValueError as e:
        logger.warning(f"Validation error in delete_session: {e}")
        raise HTTPException(status_code=400, detail="입력 데이터가 올바르지 않습니다")
    except Exception as e:
        logger.error(f"Error in delete_session: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="세션 삭제 중 오류가 발생했습니다")


@router.post("/sessions/{session_id}/generate-title", response_model=SessionResponse)
async def generate_session_title(
    session_id: str,
    current_user: dict = Depends(require_auth)
):
    """
    세션의 첫 번째 질문과 답변을 기반으로 Gemini를 사용하여 제목 생성
    
    Args:
        session_id: 세션 ID
        
    Returns:
        생성된 제목이 포함된 세션 정보
    """
    try:
        # 세션 존재 확인
        session = session_service.get_session(session_id)
        if not session:
            raise HTTPException(status_code=404, detail="Session not found")
        
        # 제목 생성
        generated_title = session_service.generate_session_title(session_id)
        
        if not generated_title:
            raise HTTPException(
                status_code=400, 
                detail="Failed to generate title. Please ensure there are at least 2 messages (user + assistant) in the session."
            )
        
        # 업데이트된 세션 정보 조회
        updated_session = session_service.get_session(session_id)
        if not updated_session:
            raise HTTPException(status_code=404, detail="Session not found after title generation")
        
        # datetime 객체를 문자열로 변환
        if isinstance(updated_session.get("created_at"), datetime):
            updated_session["created_at"] = updated_session["created_at"].isoformat()
        if isinstance(updated_session.get("updated_at"), datetime):
            updated_session["updated_at"] = updated_session["updated_at"].isoformat()
        
        return SessionResponse(**updated_session)
        
    except HTTPException:
        raise
    except ValueError as e:
        logger.warning(f"Validation error in generate_session_title: {e}")
        raise HTTPException(status_code=400, detail="입력 데이터가 올바르지 않습니다")
    except Exception as e:
        logger.error(f"Error in generate_session_title: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="제목 생성 중 오류가 발생했습니다")

