"""
인증 미들웨어
"""
from fastapi import Request, HTTPException, status
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from typing import Optional
import logging
from slowapi.util import get_remote_address

from api.services.auth_service import auth_service
from api.services.anonymous_quota_service import anonymous_quota_service
from api.config import api_config

logger = logging.getLogger(__name__)

security = HTTPBearer(auto_error=False)


async def get_current_user(request: Request) -> Optional[dict]:
    """현재 사용자 정보 가져오기"""
    if not auth_service.is_auth_enabled():
        return {"user_id": "anonymous", "authenticated": False}
    
    credentials: Optional[HTTPAuthorizationCredentials] = await security(request)
    
    if credentials:
        token = credentials.credentials
        payload = auth_service.verify_token(token, token_type="access")
        if payload:
            return {"user_id": payload.get("sub", "unknown"), "authenticated": True, **payload}
    
    api_key = request.headers.get(api_config.api_key_header)
    if api_key:
        if auth_service.verify_api_key(api_key):
            return {"user_id": "api_key_user", "authenticated": True}
    
    return None


async def require_auth(request: Request) -> dict:
    """인증 필수 데코레이터"""
    if not auth_service.is_auth_enabled():
        return {"user_id": "anonymous", "authenticated": False}
    
    user = await get_current_user(request)
    if user and user.get("authenticated"):
        return user
    
    # 익명 사용자 제한 확인
    if anonymous_quota_service.is_enabled():
        ip_address = get_remote_address(request)
        
        # 질의 가능 여부 확인
        if not anonymous_quota_service.check_quota(ip_address):
            remaining = anonymous_quota_service.get_remaining_quota(ip_address)
            raise HTTPException(
                status_code=status.HTTP_429_TOO_MANY_REQUESTS,
                detail="무료 질의 3회를 모두 사용하셨습니다. 계속 사용하려면 로그인이 필요합니다.",
                headers={
                    "X-Quota-Remaining": "0",
                    "X-Quota-Limit": str(anonymous_quota_service.quota_limit),
                    "Retry-After": "86400"
                }
            )
        
        # 질의 횟수 증가 (실제 질의 처리 전에 증가)
        remaining = anonymous_quota_service.increment_quota(ip_address)
        return {
            "user_id": "anonymous",
            "authenticated": False,
            "quota_remaining": remaining
        }
    
    # 익명 사용자 제한이 비활성화된 경우 기존 동작
    raise HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="인증이 필요합니다",
        headers={"WWW-Authenticate": "Bearer"},
    )

