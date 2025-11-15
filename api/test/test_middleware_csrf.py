"""
CSRF 보호 미들웨어 테스트
"""
import pytest
import os
import sys
from pathlib import Path
from fastapi import Request, status
from unittest.mock import AsyncMock, MagicMock, patch

# 프로젝트 루트를 sys.path에 추가
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from api.middleware.csrf import CSRFProtectionMiddleware, setup_csrf_protection, csrf_tokens


class TestCSRFProtection:
    """CSRF 보호 테스트"""
    
    def setup_method(self):
        """테스트 전 초기화"""
        csrf_tokens.clear()
    
    @pytest.mark.asyncio
    async def test_get_request_generates_token(self):
        """GET 요청 시 CSRF 토큰 생성 테스트"""
        middleware = CSRFProtectionMiddleware(app=MagicMock())
        
        request = MagicMock(spec=Request)
        request.method = "GET"
        request.client = MagicMock()
        request.client.host = "127.0.0.1"
        
        response = MagicMock()
        response.headers = {}
        
        async def call_next(req):
            return response
        
        result = await middleware.dispatch(request, call_next)
        
        assert "X-CSRF-Token" in result.headers
        assert result.headers["X-CSRF-Token"] in csrf_tokens.values()
    
    @pytest.mark.asyncio
    async def test_post_request_with_valid_token(self):
        """유효한 토큰으로 POST 요청 테스트"""
        middleware = CSRFProtectionMiddleware(app=MagicMock())
        
        csrf_token = "test_token_123"
        client_ip = "127.0.0.1"
        csrf_tokens[client_ip] = csrf_token
        
        request = MagicMock(spec=Request)
        request.method = "POST"
        request.client = MagicMock()
        request.client.host = client_ip
        request.headers.get = MagicMock(return_value=csrf_token)
        
        response = MagicMock()
        
        async def call_next(req):
            return response
        
        result = await middleware.dispatch(request, call_next)
        assert result == response
        assert client_ip not in csrf_tokens
    
    @pytest.mark.asyncio
    async def test_post_request_with_invalid_token(self):
        """유효하지 않은 토큰으로 POST 요청 테스트"""
        middleware = CSRFProtectionMiddleware(app=MagicMock())
        
        client_ip = "127.0.0.1"
        csrf_tokens[client_ip] = "valid_token"
        
        request = MagicMock(spec=Request)
        request.method = "POST"
        request.client = MagicMock()
        request.client.host = client_ip
        request.headers.get = MagicMock(return_value="invalid_token")
        
        async def call_next(req):
            return MagicMock()
        
        with pytest.raises(Exception):
            await middleware.dispatch(request, call_next)
    
    @pytest.mark.asyncio
    async def test_options_request_bypass(self):
        """OPTIONS 요청은 CSRF 검증 제외 테스트"""
        middleware = CSRFProtectionMiddleware(app=MagicMock())
        
        request = MagicMock(spec=Request)
        request.method = "OPTIONS"
        
        response = MagicMock()
        
        async def call_next(req):
            return response
        
        result = await middleware.dispatch(request, call_next)
        assert result == response
    
    def test_setup_csrf_protection_debug_mode(self):
        """개발 모드에서 CSRF 보호 비활성화 테스트"""
        app = MagicMock()
        
        with patch('api.middleware.csrf.api_config') as mock_config:
            mock_config.debug = True
            with patch.dict(os.environ, {"ENABLE_CSRF": "false"}):
                setup_csrf_protection(app)
                app.add_middleware.assert_not_called()

