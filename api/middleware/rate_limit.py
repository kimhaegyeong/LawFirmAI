"""
Rate Limiting 미들웨어
"""
import os
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.util import get_remote_address
from slowapi.errors import RateLimitExceeded
from fastapi import Request, HTTPException, status
from starlette.responses import JSONResponse
from starlette.config import Config
import logging

from api.config import api_config

logger = logging.getLogger(__name__)

# Starlette Config의 .env 파일 읽기 인코딩 문제 해결
# Windows에서 cp949 인코딩 오류를 방지하기 위해 UTF-8 인코딩을 강제
# 이 패치는 애플리케이션 전체에 적용되어 다른 Starlette Config 사용에도 도움이 됨
if not hasattr(Config, "_read_file_patched"):
    _original_read_file = Config._read_file

    def _read_file_utf8(self, env_file, encoding=None):
        """UTF-8 인코딩으로 .env 파일 읽기"""
        if env_file is None:
            return {}
        if not os.path.exists(env_file):
            return {}
        
        file_values = {}
        try:
            # UTF-8 인코딩으로 파일 읽기 (encoding 인자는 무시하고 항상 UTF-8 사용)
            with open(env_file, "r", encoding="utf-8") as input_file:
                for line in input_file.readlines():
                    line = line.strip()
                    if not line or line.startswith("#") or "=" not in line:
                        continue
                    key, value = line.split("=", 1)
                    key = key.strip()
                    value = value.strip().strip('"').strip("'")
                    file_values[key] = value
        except UnicodeDecodeError:
            # UTF-8로 읽기 실패 시 원래 메서드 사용 (fallback)
            logger.warning(f"Failed to read {env_file} with UTF-8, trying original method")
            # 원래 메서드가 encoding 인자를 받는 경우 대비
            try:
                return _original_read_file(self, env_file, encoding)
            except TypeError:
                return _original_read_file(self, env_file)
        except Exception as e:
            logger.warning(f"Error reading {env_file}: {e}")
            return {}
        
        return file_values

    # Starlette Config의 _read_file 메서드를 UTF-8 버전으로 교체
    Config._read_file = _read_file_utf8
    Config._read_file_patched = True

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

