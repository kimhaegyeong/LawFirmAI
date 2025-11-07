"""
세션 관리 엔드포인트
"""
from fastapi import APIRouter, HTTPException, Query
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

router = APIRouter()

# KST 시간대 (UTC+9)
KST = timezone(timedelta(hours=9))


def get_kst_now() -> datetime:
    """KST 기준 현재 시간 반환"""
    return datetime.now(KST)


def get_kst_date() -> datetime.date:
    """KST 기준 오늘 날짜 반환"""
    return get_kst_now().date()


@router.get("/sessions", response_model=SessionListResponse)
async def list_sessions(
    category: Optional[str] = Query(None, description="카테고리 필터"),
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
        sessions, total = session_service.list_sessions(
            category=category,
            search=search,
            page=page,
            page_size=page_size,
            sort_by=sort_by,
            sort_order=sort_order,
            date_from=date_from,
            date_to=date_to
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
    except Exception as e:
        import logging
        logging.error(f"Error in list_sessions: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/sessions/by-date", response_model=SessionListResponse)
async def get_sessions_by_date(
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
        
        sessions, total = session_service.list_sessions(
            search=search,
            page=page,
            page_size=page_size,
            sort_by="updated_at",
            sort_order="desc",
            date_from=date_from,
            date_to=date_to
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
    except Exception as e:
        logging.error(f"Error in get_sessions_by_date: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/sessions", response_model=SessionResponse)
async def create_session(session: SessionCreate):
    """새 세션 생성"""
    try:
        session_id = session_service.create_session(
            title=session.title,
            category=session.category
        )
        
        created_session = session_service.get_session(session_id)
        if not created_session:
            raise HTTPException(status_code=404, detail="Session not found")
        
        # datetime 객체를 문자열로 변환
        if isinstance(created_session.get("created_at"), datetime):
            created_session["created_at"] = created_session["created_at"].isoformat()
        if isinstance(created_session.get("updated_at"), datetime):
            created_session["updated_at"] = created_session["updated_at"].isoformat()
        
        return SessionResponse(**created_session)
    except HTTPException:
        raise
    except Exception as e:
        logging.error(f"Error in create_session: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/sessions/{session_id}", response_model=SessionResponse)
async def get_session(session_id: str):
    """세션 상세 조회"""
    try:
        session = session_service.get_session(session_id)
        if not session:
            raise HTTPException(status_code=404, detail="Session not found")
        
        # datetime 객체를 문자열로 변환
        if isinstance(session.get("created_at"), datetime):
            session["created_at"] = session["created_at"].isoformat()
        if isinstance(session.get("updated_at"), datetime):
            session["updated_at"] = session["updated_at"].isoformat()
        
        return SessionResponse(**session)
    except HTTPException:
        raise
    except Exception as e:
        logging.error(f"Error in get_session: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@router.put("/sessions/{session_id}", response_model=SessionResponse)
async def update_session(session_id: str, session: SessionUpdate):
    """세션 업데이트"""
    try:
        success = session_service.update_session(
            session_id=session_id,
            title=session.title,
            category=session.category
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
    except Exception as e:
        logging.error(f"Error in update_session: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@router.delete("/sessions/{session_id}")
async def delete_session(session_id: str):
    """세션 삭제"""
    try:
        success = session_service.delete_session(session_id)
        if not success:
            raise HTTPException(status_code=404, detail="Session not found")
        
        return {"message": "Session deleted successfully"}
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/sessions/{session_id}/generate-title", response_model=SessionResponse)
async def generate_session_title(session_id: str):
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
    except Exception as e:
        logging.error(f"Error in generate_session_title: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))

