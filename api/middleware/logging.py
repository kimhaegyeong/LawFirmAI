"""
로깅 미들웨어
"""
import logging
import sys
import time
import os
from pathlib import Path
from datetime import datetime
from logging.handlers import RotatingFileHandler
from fastapi import FastAPI, Request
from starlette.middleware.base import BaseHTTPMiddleware
from api.utils.logging_security import setup_secure_logging

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

# 로그 디렉토리 설정
LOG_DIR = Path(__file__).parent.parent.parent / "logs" / "api"
LOG_DIR.mkdir(parents=True, exist_ok=True)

# 서버 시작 시 타임스탬프 (서버 재시작마다 새 파일 생성)
SERVER_START_TIME = datetime.now().strftime("%Y%m%d_%H%M%S")

# 롤링 로그 설정 (환경 변수에서 읽기)
try:
    LOG_FILE_MAX_BYTES = int(os.getenv("LOG_FILE_MAX_BYTES", "10485760"))
except (ValueError, TypeError):
    LOG_FILE_MAX_BYTES = 10485760

try:
    LOG_FILE_BACKUP_COUNT = int(os.getenv("LOG_FILE_BACKUP_COUNT", "10"))
except (ValueError, TypeError):
    LOG_FILE_BACKUP_COUNT = 10

# Windows multiprocessing과 호환되는 로깅 설정
if sys.platform == "win32":
    # Windows에서 multiprocessing 사용 시 로깅 에러 방지
    # uvicorn의 reload 기능이 multiprocessing을 사용할 때 로깅 스트림이 분리되는 문제 해결
    try:
        # 로깅 예외를 무시하도록 설정 (multiprocessing에서 발생하는 버퍼 분리 에러 방지)
        logging.raiseExceptions = False
        
        root_logger = logging.getLogger()
        
        # 기존 핸들러 중 분리된 스트림을 가진 핸들러 제거
        for handler in root_logger.handlers[:]:
            try:
                if hasattr(handler, 'stream') and hasattr(handler.stream, 'closed'):
                    if handler.stream.closed:
                        root_logger.removeHandler(handler)
            except (AttributeError, ValueError):
                try:
                    root_logger.removeHandler(handler)
                except (ValueError, AttributeError):
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
        except Exception:
            pass

logger = logging.getLogger(__name__)


def setup_file_logging():
    """서버 재시작마다 새 파일로 롤링 파일 로깅 설정"""
    try:
        root_logger = logging.getLogger()
        
        # 기존 파일 핸들러 제거 (중복 방지)
        for handler in root_logger.handlers[:]:
            if isinstance(handler, (RotatingFileHandler, logging.FileHandler)):
                root_logger.removeHandler(handler)
        
        # 서버 시작 타임스탬프가 포함된 파일명 생성
        log_file = LOG_DIR / f"api_{SERVER_START_TIME}.log"
        
        # 크기 기반 롤링 파일 핸들러 생성
        file_handler = RotatingFileHandler(
            filename=str(log_file),
            maxBytes=LOG_FILE_MAX_BYTES,
            backupCount=LOG_FILE_BACKUP_COUNT,
            encoding='utf-8',
            delay=False
        )
        
        # 로그 포맷 설정
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        file_handler.setFormatter(formatter)
        file_handler.setLevel(log_level)
        
        root_logger.addHandler(file_handler)
        
        # 에러 로그는 별도 파일로 분리
        error_log_file = LOG_DIR / f"api_error_{SERVER_START_TIME}.log"
        error_handler = RotatingFileHandler(
            filename=str(error_log_file),
            maxBytes=LOG_FILE_MAX_BYTES,
            backupCount=LOG_FILE_BACKUP_COUNT,
            encoding='utf-8',
            delay=False
        )
        error_handler.setFormatter(formatter)
        error_handler.setLevel(logging.ERROR)
        
        # ERROR 레벨만 통과하는 필터 추가
        class ErrorLevelFilter(logging.Filter):
            def filter(self, record):
                return record.levelno >= logging.ERROR
        
        error_handler.addFilter(ErrorLevelFilter())
        
        root_logger.addHandler(error_handler)
        
        logger.info(f"롤링 파일 로깅 설정 완료: {log_file}")
        logger.info(f"에러 로그 파일: {error_log_file}")
        logger.info(f"서버 시작 시간: {SERVER_START_TIME}")
    except (OSError, PermissionError) as e:
        logger.error(f"파일 로깅 설정 실패: {e}", exc_info=True)
    except Exception as e:
        logger.error(f"로깅 설정 중 예상치 못한 오류: {e}", exc_info=True)


class LoggingMiddleware(BaseHTTPMiddleware):
    """요청 로깅 미들웨어"""
    
    async def dispatch(self, request: Request, call_next):
        start_time = time.time()
        
        try:
            logger.info(f"{request.method} {request.url.path}")
        except Exception:
            pass
        
        try:
            response = await call_next(request)
            process_time = time.time() - start_time
            
            try:
                logger.info(
                    f"{request.method} {request.url.path} - "
                    f"Status: {response.status_code} - "
                    f"Time: {process_time:.3f}s"
                )
            except Exception:
                pass
            
            return response
        except Exception as e:
            process_time = time.time() - start_time
            try:
                logger.error(
                    f"{request.method} {request.url.path} - "
                    f"Error: {type(e).__name__} - "
                    f"Time: {process_time:.3f}s"
                )
            except Exception:
                pass
            raise


def setup_logging(app: FastAPI):
    """로깅 미들웨어 설정"""
    setup_file_logging()
    app.add_middleware(LoggingMiddleware)
    setup_secure_logging()

