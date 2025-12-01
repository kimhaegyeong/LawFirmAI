"""
에러 핸들러 미들웨어 테스트
"""
import pytest
import json
from fastapi import Request, status
from fastapi.exceptions import RequestValidationError
from starlette.exceptions import HTTPException as StarletteHTTPException
from unittest.mock import MagicMock
from api.middleware.error_handler import error_handler


class TestErrorHandler:
    """에러 핸들러 테스트"""
    
    @pytest.mark.asyncio
    async def test_successful_request(self):
        """정상 요청 처리 테스트"""
        request = MagicMock(spec=Request)
        request.url.path = "/test"
        
        response = MagicMock()
        response.status_code = 200
        
        async def call_next(req):
            return response
        
        result = await error_handler(request, call_next)
        assert result == response
    
    @pytest.mark.asyncio
    async def test_validation_error(self):
        """요청 검증 에러 테스트"""
        request = MagicMock(spec=Request)
        request.url.path = "/test"
        
        validation_error = RequestValidationError(
            errors=[
                {"loc": ("body", "message"), "msg": "field required", "type": "value_error.missing"}
            ]
        )
        
        async def call_next(req):
            raise validation_error
        
        result = await error_handler(request, call_next)
        assert result.status_code == status.HTTP_422_UNPROCESSABLE_ENTITY
        
        content = json.loads(result.body.decode())
        assert "error" in content
        assert "detail" in content
        assert "errors" in content
    
    @pytest.mark.asyncio
    async def test_http_exception(self):
        """HTTP 예외 테스트"""
        request = MagicMock(spec=Request)
        request.url.path = "/test"
        
        http_exception = StarletteHTTPException(
            status_code=404,
            detail="Not found"
        )
        
        async def call_next(req):
            raise http_exception
        
        result = await error_handler(request, call_next)
        assert result.status_code == 404
        
        content = json.loads(result.body.decode())
        assert "error" in content
        assert "detail" in content
    
    @pytest.mark.asyncio
    async def test_unexpected_error(self):
        """예상치 못한 에러 테스트"""
        request = MagicMock(spec=Request)
        request.url.path = "/test"
        
        async def call_next(req):
            raise ValueError("Unexpected error")
        
        result = await error_handler(request, call_next)
        assert result.status_code == status.HTTP_500_INTERNAL_SERVER_ERROR
        
        content = json.loads(result.body.decode())
        assert "error" in content
        assert "detail" in content

