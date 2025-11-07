"""
인증 미들웨어
"""
from fastapi import Request, HTTPException, status
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from typing import Optional
import logging

from api.services.auth_service import auth_service
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
        payload = auth_service.verify_token(token)
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
    if not user or not user.get("authenticated"):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="인증이 필요합니다",
            headers={"WWW-Authenticate": "Bearer"},
        )
    return user

