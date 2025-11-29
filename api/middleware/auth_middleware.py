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
        logger.debug("get_current_user: Auth not enabled, returning anonymous")
        return {"user_id": "anonymous", "authenticated": False}
    
    credentials: Optional[HTTPAuthorizationCredentials] = await security(request)
    
    if credentials:
        token = credentials.credentials
        logger.debug(f"get_current_user: Token found, verifying... (token length: {len(token)})")
        payload = auth_service.verify_token(token, token_type="access")
        if payload:
            user_id = payload.get("sub", "unknown")
            logger.info(f"get_current_user: Token verified successfully, user_id={user_id}")
            return {"user_id": user_id, "authenticated": True, **payload}
        else:
            logger.warning("get_current_user: Token verification failed")
    else:
        logger.debug("get_current_user: No credentials found in request")
    
    api_key = request.headers.get(api_config.api_key_header)
    if api_key:
        if auth_service.verify_api_key(api_key):
            logger.info("get_current_user: API key verified")
            return {"user_id": "api_key_user", "authenticated": True}
    
    logger.debug("get_current_user: No valid authentication found, returning None")
    return None


async def require_auth(request: Request) -> dict:
    """인증 필수 데코레이터"""
    if not auth_service.is_auth_enabled():
        return {"user_id": "anonymous", "authenticated": False}
    
    user = await get_current_user(request)
    
    # 로그인한 사용자는 익명 사용자 쿼터 제한을 건너뜀 (질의 제한 없음)
    if user and user.get("authenticated"):
        logger.info(f"require_auth: Authenticated user found, user_id={user.get('user_id')}, skipping quota check (unlimited queries)")
        return user
    
    # 토큰이 전달되었지만 검증이 실패한 경우
    auth_header = request.headers.get("Authorization")
    if auth_header and auth_header.startswith("Bearer "):
        logger.warning("require_auth: Token provided but verification failed")
        # 토큰이 있는데 검증이 실패한 경우, 익명 사용자로 처리하지 않고 인증 오류 반환
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="인증 토큰이 유효하지 않습니다. 다시 로그인해주세요.",
            headers={"WWW-Authenticate": "Bearer"},
        )
    
    # 무료 질의 횟수 체크는 /chat/stream 또는 /stream 엔드포인트에서만 수행
    request_path = request.url.path
    is_stream_endpoint = "/chat/stream" in request_path or request_path.endswith("/stream")
    
    if is_stream_endpoint and anonymous_quota_service.is_enabled():
        ip_address = get_remote_address(request)
        
        # 질의 가능 여부 확인
        if not anonymous_quota_service.check_quota(ip_address):
            remaining = anonymous_quota_service.get_remaining_quota(ip_address)
            logger.warning(f"require_auth: Anonymous user quota exceeded, ip={ip_address}, remaining={remaining}")
            raise HTTPException(
                status_code=status.HTTP_429_TOO_MANY_REQUESTS,
                detail="무료 질의 3회를 모두 사용하셨습니다. 계속 사용하려면 로그인이 필요합니다.",
                headers={
                    "X-Quota-Remaining": "0",
                    "X-Quota-Limit": str(anonymous_quota_service.quota_limit),
                    "Retry-After": "86400"
                }
            )
        
        # 질의 횟수는 응답이 성공적으로 완료된 후에만 증가하도록 변경
        # 여기서는 증가하지 않고, 응답 완료 후 증가하도록 플래그만 설정
        # 쿼터 증가는 chat_stream 엔드포인트에서 성공적으로 완료된 후에 수행
        remaining = anonymous_quota_service.get_remaining_quota(ip_address)
        logger.debug(f"require_auth: Anonymous user quota check passed, ip={ip_address}, remaining={remaining}")
        return {
            "user_id": "anonymous",
            "authenticated": False,
            "quota_remaining": remaining,
            "_should_increment_quota": True  # 쿼터 증가 플래그
        }
    
    # /stream 엔드포인트가 아닌 경우 또는 익명 사용자 제한이 비활성화된 경우
    # 인증 없이도 접근 가능하도록 처리
    return {
        "user_id": "anonymous",
        "authenticated": False
    }

