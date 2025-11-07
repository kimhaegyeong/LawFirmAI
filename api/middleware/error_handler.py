"""
에러 처리 미들웨어
"""
import logging
from fastapi import Request, status
from fastapi.responses import JSONResponse
from fastapi.exceptions import RequestValidationError
from starlette.exceptions import HTTPException as StarletteHTTPException

logger = logging.getLogger(__name__)


async def error_handler(request: Request, call_next):
    """전역 에러 핸들러"""
    try:
        response = await call_next(request)
        return response
    except RequestValidationError as e:
        logger.warning(f"Validation error: {e.errors()}")
        return JSONResponse(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            content={
                "error": "입력 검증 실패",
                "detail": "요청 데이터가 올바르지 않습니다"
            }
        )
    except StarletteHTTPException as e:
        logger.warning(f"HTTP exception: {e.status_code} - {e.detail}")
        return JSONResponse(
            status_code=e.status_code,
            content={
                "error": "요청 처리 실패",
                "detail": str(e.detail) if e.status_code < 500 else "서버 오류가 발생했습니다"
            }
        )
    except Exception as e:
        logger.error(f"Unexpected error: {e}", exc_info=True)
        return JSONResponse(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            content={
                "error": "서버 오류",
                "detail": "서버에서 오류가 발생했습니다. 잠시 후 다시 시도해주세요."
            }
        )

