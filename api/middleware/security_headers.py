"""
보안 헤더 미들웨어
"""
import os
from fastapi import Request
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.responses import RedirectResponse

class SecurityHeadersMiddleware(BaseHTTPMiddleware):
    """보안 헤더 추가 미들웨어"""
    
    async def dispatch(self, request: Request, call_next):
        from api.config import api_config
        
        # 프로덕션 환경에서 HTTPS 강제 (localhost 제외)
        # 환경 변수 FORCE_HTTPS로 강제 설정 가능
        force_https = os.getenv("FORCE_HTTPS", "").lower() in ("true", "1", "yes")
        
        if not api_config.debug or force_https:
            # localhost나 127.0.0.1은 HTTPS 강제하지 않음 (개발 환경)
            host = request.url.hostname
            is_localhost = host in ("localhost", "127.0.0.1", "0.0.0.0", "::1")
            
            # X-Forwarded-Proto 헤더 확인 (프록시 뒤에서 실행 중인 경우)
            forwarded_proto = request.headers.get("X-Forwarded-Proto", "").lower()
            is_https = request.url.scheme == "https" or forwarded_proto == "https"
            
            if not is_localhost and not is_https:
                https_url = request.url.replace(scheme="https")
                return RedirectResponse(url=str(https_url), status_code=301)
        
        response = await call_next(request)
        
        # CORS 헤더가 이미 있는지 확인하고 보존
        cors_headers = {}
        for header in ["Access-Control-Allow-Origin", "Access-Control-Allow-Credentials", 
                       "Access-Control-Allow-Methods", "Access-Control-Allow-Headers",
                       "Access-Control-Expose-Headers", "Access-Control-Max-Age"]:
            if header in response.headers:
                cors_headers[header] = response.headers[header]
        
        # 보안 헤더 추가
        response.headers["X-Content-Type-Options"] = "nosniff"
        response.headers["X-Frame-Options"] = "DENY"
        response.headers["X-XSS-Protection"] = "1; mode=block"
        response.headers["Referrer-Policy"] = "strict-origin-when-cross-origin"
        
        # CORS 헤더 복원 (보안 헤더 추가 후에도 유지)
        for header, value in cors_headers.items():
            response.headers[header] = value
        
        # 프로덕션 환경에서만 HSTS 추가 (HTTPS 사용 시)
        forwarded_proto = request.headers.get("X-Forwarded-Proto", "").lower()
        is_https = request.url.scheme == "https" or forwarded_proto == "https"
        
        if (not api_config.debug or force_https) and is_https:
            # HSTS 헤더: 1년간 HTTPS 강제, 서브도메인 포함, preload 허용
            response.headers["Strict-Transport-Security"] = "max-age=31536000; includeSubDomains; preload"
        
        # Content Security Policy (필요에 따라 조정)
        host = request.url.hostname
        is_localhost = host in ("localhost", "127.0.0.1", "0.0.0.0", "::1")
        
        if api_config.debug or is_localhost:
            # 개발 환경 또는 localhost에서는 CSP 완화
            response.headers["Content-Security-Policy"] = (
                "default-src 'self' *; "
                "script-src 'self' 'unsafe-inline' *; "
                "style-src 'self' 'unsafe-inline' *; "
                "img-src 'self' data: https: *; "
                "font-src 'self' data: *; "
                "connect-src 'self' *"
            )
        else:
            # 프로덕션 환경에서는 엄격한 CSP
            response.headers["Content-Security-Policy"] = (
                "default-src 'self'; "
                "script-src 'self' 'unsafe-inline'; "
                "style-src 'self' 'unsafe-inline'; "
                "img-src 'self' data: https:; "
                "font-src 'self' data:; "
                "connect-src 'self'"
            )
        
        return response

