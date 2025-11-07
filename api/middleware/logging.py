"""
로깅 미들웨어
"""
import logging
import sys
import time
import os
from fastapi import FastAPI, Request
from starlette.middleware.base import BaseHTTPMiddleware

# 로그 레벨 환경 변수 읽기 (기본값: INFO)
log_level_str = os.getenv("LOG_LEVEL", "info").upper()
log_level_map = {
    "CRITICAL": logging.CRITICAL,
    "ERROR": logging.ERROR,
    "WARNING": logging.WARNING,
    "INFO": logging.INFO,
    "DEBUG": logging.DEBUG,
}
log_level = log_level_map.get(log_level_str, logging.INFO)

# Windows multiprocessing과 호환되는 로깅 설정
if sys.platform == "win32":
    # Windows에서 multiprocessing 사용 시 로깅 에러 방지
    # uvicorn의 reload 기능이 multiprocessing을 사용할 때 로깅 스트림이 분리되는 문제 해결
    try:
        # 로깅 예외를 무시하도록 설정 (multiprocessing에서 발생하는 버퍼 분리 에러 방지)
        logging.raiseExceptions = False
        
        # 기존 로깅 핸들러가 있는지 확인
        root_logger = logging.getLogger()
        
        # 기존 핸들러 중 분리된 스트림을 가진 핸들러 제거
        for handler in root_logger.handlers[:]:
            try:
                if hasattr(handler, 'stream') and hasattr(handler.stream, 'closed'):
                    if handler.stream.closed:
                        root_logger.removeHandler(handler)
            except (AttributeError, ValueError):
                # 핸들러가 이미 분리된 경우 제거 시도
                try:
                    root_logger.removeHandler(handler)
                except:
                    pass
        
        # 핸들러가 없거나 모두 제거된 경우 새로 추가
        if not root_logger.handlers:
            handler = logging.StreamHandler(sys.stdout)
            handler.setFormatter(
                logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
            )
            handler.setLevel(log_level)
            root_logger.addHandler(handler)
            root_logger.setLevel(log_level)
        else:
            # 기존 핸들러가 있으면 레벨만 업데이트
            root_logger.setLevel(log_level)
            for handler in root_logger.handlers:
                handler.setLevel(log_level)
    except Exception:
        # 로깅 설정 실패 시 기본 설정 사용 (에러 무시)
        try:
            logging.basicConfig(
                level=log_level,
                format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
                handlers=[logging.StreamHandler(sys.stdout)],
                force=True
            )
            logging.raiseExceptions = False
        except:
            pass

logger = logging.getLogger(__name__)


class LoggingMiddleware(BaseHTTPMiddleware):
    """요청 로깅 미들웨어"""
    
    async def dispatch(self, request: Request, call_next):
        start_time = time.time()
        
        # 요청 로깅
        logger.info(f"{request.method} {request.url.path}")
        
        # 응답 처리
        response = await call_next(request)
        
        # 처리 시간 계산
        process_time = time.time() - start_time
        
        # 응답 로깅
        logger.info(
            f"{request.method} {request.url.path} - "
            f"Status: {response.status_code} - "
            f"Time: {process_time:.3f}s"
        )
        
        return response


def setup_logging(app: FastAPI):
    """로깅 미들웨어 설정"""
    app.add_middleware(LoggingMiddleware)

