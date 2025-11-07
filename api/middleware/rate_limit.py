"""
Rate Limiting 미들웨어
"""
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.util import get_remote_address
from slowapi.errors import RateLimitExceeded
from fastapi import Request, HTTPException, status
from starlette.responses import JSONResponse
import logging

from api.config import api_config

logger = logging.getLogger(__name__)

limiter = Limiter(key_func=get_remote_address)


def create_rate_limit_response(request: Request, exc: RateLimitExceeded):
    """Rate limit 초과 시 응답 생성"""
    response = JSONResponse(
        status_code=status.HTTP_429_TOO_MANY_REQUESTS,
        content={
            "detail": "요청 한도를 초과했습니다. 잠시 후 다시 시도해주세요.",
            "retry_after": exc.retry_after
        }
    )
    response = request.app.state.limiter._inject_headers(
        response, request.state.view_rate_limit
    )
    return response


def get_rate_limit() -> Limiter:
    """Rate Limiter 인스턴스 반환"""
    return limiter


def is_rate_limit_enabled() -> bool:
    """Rate Limiting 활성화 여부"""
    return api_config.rate_limit_enabled

