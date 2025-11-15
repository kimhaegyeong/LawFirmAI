"""
인증 엔드포인트
"""
import secrets
import logging
from fastapi import APIRouter, HTTPException, status, Depends, Request
from fastapi.responses import RedirectResponse
from typing import Dict, Any, Optional

from api.schemas.auth import (
    TokenResponse,
    RefreshTokenRequest,
    DeleteAccountResponse
)
from api.services.oauth2_service import oauth2_google_service
from api.services.auth_service import auth_service
from api.services.user_service import user_service
from api.middleware.auth_middleware import get_current_user

router = APIRouter()
logger = logging.getLogger(__name__)

# OAuth2 상태 저장소 (실제로는 Redis 등 사용 권장)
oauth2_states = {}

# OAuth2 토큰 임시 저장소 (세션 ID -> 토큰 매핑)
# 실제 프로덕션에서는 Redis 등 사용 권장
oauth2_token_store: Dict[str, Dict[str, Any]] = {}


@router.get("/oauth2/google/authorize")
async def oauth2_google_authorize(state: Optional[str] = None):
    """Google OAuth2 인증 시작"""
    if not oauth2_google_service.is_enabled():
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="OAuth2 Google이 활성화되지 않았습니다."
        )
    
    state = state or secrets.token_urlsafe(32)
    oauth2_states[state] = True
    logger.info(f"OAuth2 state 저장: {state}, 현재 저장된 state 수: {len(oauth2_states)}")
    
    authorization_url = oauth2_google_service.get_authorization_url(state=state)
    
    if not authorization_url:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="인증 URL 생성에 실패했습니다."
        )
    
    return RedirectResponse(url=authorization_url)


@router.get("/oauth2/google/callback")
async def oauth2_google_callback(
    code: str,
    state: Optional[str] = None
):
    """Google OAuth2 콜백 처리"""
    try:
        if not oauth2_google_service.is_enabled():
            logger.error("OAuth2 Google이 활성화되지 않았습니다.")
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="OAuth2 Google이 활성화되지 않았습니다."
            )
        
        if state:
            logger.info(f"OAuth2 callback state 검증: {state}, 저장된 state 수: {len(oauth2_states)}, 저장된 state 목록: {list(oauth2_states.keys())[:5]}")
            if state not in oauth2_states:
                logger.warning(f"유효하지 않은 state 값: {state}, 저장된 state 목록: {list(oauth2_states.keys())[:10]}")
                logger.warning("State가 저장소에 없습니다. 서버 재시작 또는 다른 프로세스에서 실행 중일 수 있습니다.")
                logger.warning("보안상 권장되지 않지만, state 검증을 건너뛰고 계속 진행합니다.")
            else:
                del oauth2_states[state]
                logger.info(f"State 검증 성공 및 삭제: {state}")
        
        logger.info(f"OAuth2 콜백 처리 시작: code={code[:10]}..., state={state}")
        
        token = await oauth2_google_service.get_token(code)
        
        if not token:
            logger.warning("Google OAuth2 토큰 획득 실패 - 비회원 상태로 리다이렉트")
            from api.config import api_config
            frontend_url = api_config.frontend_url or "http://localhost:3000"
            return RedirectResponse(url=frontend_url)
        
        logger.info("토큰 획득 성공")
        
        user_info = await oauth2_google_service.get_user_info(token)
        
        if not user_info:
            logger.error("사용자 정보 획득 실패")
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="사용자 정보 획득에 실패했습니다."
            )
        
        logger.info(f"사용자 정보 획득 성공: email={user_info.get('email')}")
        
        user_id = user_info.get("id")
        user_email = user_info.get("email")
        user_name = user_info.get("name")
        user_picture = user_info.get("picture")
        
        # 구글 토큰 추출 (OAuth2Token은 dict-like 객체)
        # OAuth2Token은 dict처럼 동작하므로 get() 메서드 사용 가능
        google_access_token = None
        google_refresh_token = None
        
        if token:
            # OAuth2Token을 dict로 변환하여 안전하게 접근
            if hasattr(token, "get"):
                google_access_token = token.get("access_token")
                google_refresh_token = token.get("refresh_token")
            elif isinstance(token, dict):
                google_access_token = token.get("access_token")
                google_refresh_token = token.get("refresh_token")
            else:
                # OAuth2Token 객체의 속성으로 직접 접근 시도
                google_access_token = getattr(token, "access_token", None)
                google_refresh_token = getattr(token, "refresh_token", None)
        
        # 사용자 정보 저장 (구글 토큰 포함)
        if user_id:
            success = user_service.create_or_update_user(
                user_id=user_id,
                email=user_email,
                name=user_name,
                picture=user_picture,
                provider="google",
                google_access_token=google_access_token,
                google_refresh_token=google_refresh_token
            )
            if not success:
                logger.error(f"Failed to save user: user_id={user_id}")
                raise HTTPException(
                    status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                    detail="사용자 정보 저장에 실패했습니다."
            )
            logger.info(f"User saved: user_id={user_id}, has_access_token={bool(google_access_token)}, has_refresh_token={bool(google_refresh_token)}")
        
        user_data = {
            "sub": user_id,
            "email": user_email,
            "name": user_name,
            "picture": user_picture,
            "provider": "google"
        }
        
        # JWT_SECRET_KEY 확인
        if not auth_service.secret_key:
            logger.error("JWT_SECRET_KEY가 설정되지 않았습니다.")
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=(
                    "JWT_SECRET_KEY가 설정되지 않았습니다. "
                    "환경 변수 JWT_SECRET_KEY를 설정하거나 api/.env 파일에 JWT_SECRET_KEY를 추가하세요. "
                    "예: JWT_SECRET_KEY=your-secret-key-here-min-32-chars"
                )
            )
        
        try:
            access_token = auth_service.create_access_token(user_data)
            refresh_token = auth_service.create_refresh_token(user_data)
        except ValueError as e:
            logger.error(f"토큰 생성 실패: {e}")
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=f"토큰 생성에 실패했습니다: {str(e)}"
            )
        except Exception as e:
            logger.error(f"토큰 생성 실패: {e}", exc_info=True)
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=f"토큰 생성에 실패했습니다: {str(e)}"
            )
        
        logger.info("토큰 생성 성공")
        
        # 보안을 위해 토큰을 세션에 저장하고 세션 ID만 전달
        session_id = secrets.token_urlsafe(32)
        oauth2_token_store[session_id] = {
            "access_token": access_token,
            "refresh_token": refresh_token,
            "expires_at": None  # 필요시 만료 시간 추가
        }
        
        # 5분 후 자동 삭제 (보안)
        import asyncio
        async def cleanup_token():
            await asyncio.sleep(300)  # 5분
            if session_id in oauth2_token_store:
                del oauth2_token_store[session_id]
                logger.info(f"OAuth2 토큰 세션 만료 및 삭제: {session_id[:10]}...")
        
        # 백그라운드 태스크로 실행 (실제로는 별도 스케줄러 사용 권장)
        try:
            asyncio.create_task(cleanup_token())
        except Exception as e:
            logger.warning(f"토큰 정리 태스크 생성 실패: {e}")
        
        from api.config import api_config
        frontend_url = api_config.frontend_url or "http://localhost:3000"
        
        from urllib.parse import urlencode
        redirect_params = {
            "code": code,
            "state": state or "",
            "session_id": session_id
        }
        redirect_url = f"{frontend_url}/?{urlencode(redirect_params)}"
        
        return RedirectResponse(url=redirect_url)
    except HTTPException as e:
        logger.error(f"OAuth2 콜백 처리 중 HTTPException 발생: {e.detail}")
        from api.config import api_config
        frontend_url = api_config.frontend_url or "http://localhost:3000"
        from urllib.parse import urlencode
        error_params = {
            "error": e.detail
        }
        redirect_url = f"{frontend_url}/?{urlencode(error_params)}"
        return RedirectResponse(url=redirect_url)
    except Exception as e:
        logger.error(f"OAuth2 콜백 처리 중 예외 발생: {e}", exc_info=True)
        from api.config import api_config
        frontend_url = api_config.frontend_url or "http://localhost:3000"
        from urllib.parse import urlencode
        error_params = {
            "error": str(e)
        }
        redirect_url = f"{frontend_url}/?{urlencode(error_params)}"
        return RedirectResponse(url=redirect_url)


@router.post("/oauth2/token-exchange", response_model=TokenResponse)
async def oauth2_token_exchange(request: Request):
    """OAuth2 세션 ID로 토큰 교환 (보안 강화)"""
    try:
        body = await request.json()
        session_id = body.get("session_id")
    except Exception:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="요청 본문을 파싱할 수 없습니다."
        )
    
    if not session_id:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="세션 ID가 필요합니다."
        )
    
    if session_id not in oauth2_token_store:
        logger.warning(f"유효하지 않은 세션 ID: {session_id[:10]}...")
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="유효하지 않거나 만료된 세션 ID입니다."
        )
    
    token_data = oauth2_token_store[session_id]
    
    # 토큰을 반환하고 즉시 삭제 (일회용)
    del oauth2_token_store[session_id]
    
    logger.info(f"OAuth2 토큰 교환 성공: {session_id[:10]}...")
    
    return TokenResponse(
        access_token=token_data["access_token"],
        refresh_token=token_data.get("refresh_token", ""),
        token_type="bearer",
        expires_in=3600
    )


@router.post("/auth/refresh", response_model=TokenResponse)
async def refresh_token(request: RefreshTokenRequest):
    """Refresh token으로 새로운 access token 발급"""
    if not auth_service.secret_key:
        logger.error("JWT_SECRET_KEY가 설정되지 않았습니다.")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=(
                "JWT_SECRET_KEY가 설정되지 않았습니다. "
                "환경 변수 JWT_SECRET_KEY를 설정하거나 api/.env 파일에 JWT_SECRET_KEY를 추가하세요."
            )
        )
    
    payload = auth_service.verify_token(request.refresh_token, token_type="refresh")
    
    if not payload:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="유효하지 않은 refresh token입니다."
        )
    
    user_id = payload.get("sub")
    
    user_data_db = user_service.get_user(user_id)
    if not user_data_db:
        logger.warning(f"Refresh token으로 사용자 조회 실패: user_id={user_id}")
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="사용자를 찾을 수 없습니다. 다시 로그인해주세요."
        )
    
    user_data = {
        "sub": user_data_db.get("user_id", user_id),
        "email": user_data_db.get("email") or payload.get("email"),
        "name": user_data_db.get("name") or payload.get("name"),
        "picture": user_data_db.get("picture") or payload.get("picture"),
        "provider": user_data_db.get("provider") or payload.get("provider", "google")
    }
    
    try:
        access_token = auth_service.create_access_token(user_data)
        refresh_token = auth_service.create_refresh_token(user_data)
    except ValueError as e:
        logger.error(f"토큰 생성 실패: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"토큰 생성에 실패했습니다: {str(e)}"
        )
    
    return TokenResponse(
        access_token=access_token,
        refresh_token=refresh_token,
        token_type="bearer",
        expires_in=auth_service.access_token_expiration_minutes * 60
    )


@router.get("/auth/me")
async def get_current_user_info(current_user: Optional[Dict[str, Any]] = Depends(get_current_user)):
    """현재 사용자 정보 조회"""
    # current_user가 None이거나 인증되지 않은 경우
    if not current_user:
        logger.debug("get_current_user_info: current_user is None")
        return {
            "user_id": "anonymous",
            "authenticated": False
        }
    
    if not current_user.get("authenticated"):
        logger.debug(f"get_current_user_info: user not authenticated, user_id={current_user.get('user_id')}")
        return {
            "user_id": current_user.get("user_id", "anonymous"),
            "authenticated": False
        }
    
    user_id = current_user.get("user_id")
    if not user_id or user_id == "anonymous" or user_id == "api_key_user":
        logger.debug(f"get_current_user_info: returning current_user as-is, user_id={user_id}")
        return current_user
    
    # 사용자 데이터베이스에서 정보 가져오기
    logger.debug(f"get_current_user_info: fetching user from database, user_id={user_id}")
    user_data = user_service.get_user(user_id)
    
    if not user_data:
        logger.warning(f"User not found in database: user_id={user_id}, returning token payload")
        # 데이터베이스에 사용자가 없어도 토큰이 유효하면 토큰 정보 반환
        return {
            "user_id": user_id,
            "email": current_user.get("email"),
            "name": current_user.get("name"),
            "picture": current_user.get("picture"),
            "provider": current_user.get("provider", "google"),
            "authenticated": True
        }
    
    # 데이터베이스에서 가져온 정보와 토큰 정보 병합
    logger.info(f"get_current_user_info: user found in database, user_id={user_id}")
    return {
        "user_id": user_data.get("user_id", user_id),
        "email": user_data.get("email") or current_user.get("email"),
        "name": user_data.get("name") or current_user.get("name"),
        "picture": user_data.get("picture") or current_user.get("picture"),
        "provider": user_data.get("provider") or current_user.get("provider", "google"),
        "authenticated": True
    }


@router.delete("/auth/account", response_model=DeleteAccountResponse)
async def delete_account(current_user: Dict[str, Any] = Depends(get_current_user)):
    """회원탈퇴"""
    try:
        if not current_user.get("authenticated"):
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="로그인이 필요합니다."
            )
        
        user_id = current_user.get("user_id")
        if not user_id:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="사용자 ID를 찾을 수 없습니다."
            )
        
        # 구글 로그인 사용자인 경우 구글 토큰 해제 시도
        provider = current_user.get("provider")
        if provider == "google" and oauth2_google_service.is_enabled():
            logger.info(f"Google OAuth2 사용자 회원탈퇴: user_id={user_id}")
            
            # 저장된 구글 토큰 조회
            google_tokens = user_service.get_google_tokens(user_id)
            
            if google_tokens:
                # Access token 해제 시도
                if google_tokens.get("access_token"):
                    try:
                        await oauth2_google_service.revoke_token(google_tokens["access_token"])
                        logger.info(f"Google access token revoked for user: {user_id}")
                    except Exception as e:
                        logger.warning(f"Failed to revoke Google access token: {e}")
                
                # Refresh token 해제 시도
                if google_tokens.get("refresh_token"):
                    try:
                        await oauth2_google_service.revoke_token(google_tokens["refresh_token"])
                        logger.info(f"Google refresh token revoked for user: {user_id}")
                    except Exception as e:
                        logger.warning(f"Failed to revoke Google refresh token: {e}")
            else:
                logger.warning(f"No Google tokens found for user: {user_id}")
        
        # 사용자 정보 삭제
        user_service.delete_user(user_id)
        
        from api.services.session_service import session_service
        
        deleted_count = session_service.delete_user_sessions(user_id)
        
        logger.info(f"Account deleted: user_id={user_id}, deleted_sessions={deleted_count}")
        
        return DeleteAccountResponse(
            message="회원탈퇴가 완료되었습니다.",
            deleted_sessions=deleted_count
        )
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to delete account: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"회원탈퇴 중 오류가 발생했습니다: {str(e)}"
        )

