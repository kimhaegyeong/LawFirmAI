"""
CSRF 보호 미들웨어
"""
import os
import secrets
from fastapi import Request, HTTPException, status
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.responses import Response
import logging

from api.config import api_config

logger = logging.getLogger(__name__)

# CSRF 토큰 저장소 (실제로는 세션이나 Redis 사용 권장)
csrf_tokens = {}


class CSRFProtectionMiddleware(BaseHTTPMiddleware):
    """CSRF 보호 미들웨어"""
    
    async def dispatch(self, request: Request, call_next):
        # GET, HEAD, OPTIONS 요청은 CSRF 검증 제외
        if request.method in ["GET", "HEAD", "OPTIONS"]:
            response = await call_next(request)
            # GET 요청 시 CSRF 토큰 생성 및 응답 헤더에 추가
            if request.method == "GET":
                csrf_token = secrets.token_urlsafe(32)
                # 클라이언트 IP를 키로 사용 (실제로는 세션 ID 사용 권장)
                client_ip = request.client.host if request.client else "unknown"
                csrf_tokens[client_ip] = csrf_token
                response.headers["X-CSRF-Token"] = csrf_token
            return response
        
        # POST, PUT, DELETE, PATCH 요청은 CSRF 검증
        if request.method in ["POST", "PUT", "DELETE", "PATCH"]:
            # CSRF 토큰 확인
            csrf_token_header = request.headers.get("X-CSRF-Token")
            client_ip = request.client.host if request.client else "unknown"
            stored_token = csrf_tokens.get(client_ip)
            
            if not csrf_token_header or csrf_token_header != stored_token:
                logger.warning(f"CSRF token validation failed: {client_ip}, header: {csrf_token_header}, stored: {stored_token}")
                error_detail = f"CSRF 토큰이 유효하지 않습니다. 요청 헤더: {csrf_token_header}, 저장된 토큰: {stored_token}"
                raise HTTPException(
                    status_code=status.HTTP_403_FORBIDDEN,
                    detail=error_detail
                )
            
            # 토큰 사용 후 삭제 (일회용 토큰)
            if client_ip in csrf_tokens:
                del csrf_tokens[client_ip]
        
        response = await call_next(request)
        return response


def setup_csrf_protection(app):
    """CSRF 보호 설정"""
    # 개발 환경에서는 CSRF 보호 비활성화 (localhost에서 테스트하기 쉽도록)
    if not api_config.debug and os.getenv("ENABLE_CSRF", "false").lower() == "true":
        app.add_middleware(CSRFProtectionMiddleware)
        logger.info("CSRF 보호가 활성화되었습니다.")
    else:
        logger.info("개발 환경에서는 CSRF 보호가 비활성화되었습니다.")

