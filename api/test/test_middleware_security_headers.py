"""
보안 헤더 미들웨어 테스트
"""
import pytest
import sys
from pathlib import Path
from fastapi import Request
from unittest.mock import AsyncMock, MagicMock, patch

# 프로젝트 루트를 sys.path에 추가
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from api.middleware.security_headers import SecurityHeadersMiddleware


class TestSecurityHeadersMiddleware:
    """보안 헤더 미들웨어 테스트"""
    
    @pytest.mark.asyncio
    async def test_security_headers_added(self):
        """보안 헤더 추가 테스트"""
        middleware = SecurityHeadersMiddleware(app=MagicMock())
        
        request = MagicMock(spec=Request)
        request.url.scheme = "http"
        request.url.hostname = "localhost"
        request.headers.get = MagicMock(return_value="")
        
        response = MagicMock()
        response.headers = {}
        
        async def call_next(req):
            return response
        
        with patch('api.middleware.security_headers.api_config') as mock_config:
            mock_config.debug = True
            result = await middleware.dispatch(request, call_next)
            
            assert "X-Content-Type-Options" in result.headers
            assert result.headers["X-Content-Type-Options"] == "nosniff"
            assert "X-Frame-Options" in result.headers
            assert result.headers["X-Frame-Options"] == "DENY"
            assert "X-XSS-Protection" in result.headers
            assert "Referrer-Policy" in result.headers
    
    @pytest.mark.asyncio
    async def test_hsts_header_production_https(self):
        """프로덕션 환경에서 HTTPS 시 HSTS 헤더 추가 테스트"""
        middleware = SecurityHeadersMiddleware(app=MagicMock())
        
        request = MagicMock(spec=Request)
        request.url.scheme = "https"
        request.url.hostname = "example.com"
        request.headers.get = MagicMock(return_value="")
        
        response = MagicMock()
        response.headers = {}
        
        async def call_next(req):
            return response
        
        with patch('api.middleware.security_headers.api_config') as mock_config:
            mock_config.debug = False
            result = await middleware.dispatch(request, call_next)
            
            assert "Strict-Transport-Security" in result.headers
    
    @pytest.mark.asyncio
    async def test_csp_header_debug_mode(self):
        """개발 모드에서 CSP 헤더 테스트"""
        middleware = SecurityHeadersMiddleware(app=MagicMock())
        
        request = MagicMock(spec=Request)
        request.url.scheme = "http"
        request.url.hostname = "localhost"
        request.headers.get = MagicMock(return_value="")
        
        response = MagicMock()
        response.headers = {}
        
        async def call_next(req):
            return response
        
        with patch('api.middleware.security_headers.api_config') as mock_config:
            mock_config.debug = True
            result = await middleware.dispatch(request, call_next)
            
            assert "Content-Security-Policy" in result.headers
            csp = result.headers["Content-Security-Policy"]
            assert "'unsafe-inline'" in csp
    
    @pytest.mark.asyncio
    async def test_cors_headers_preserved(self):
        """CORS 헤더 보존 테스트"""
        middleware = SecurityHeadersMiddleware(app=MagicMock())
        
        request = MagicMock(spec=Request)
        request.url.scheme = "http"
        request.url.hostname = "localhost"
        request.headers.get = MagicMock(return_value="")
        
        response = MagicMock()
        response.headers = {
            "Access-Control-Allow-Origin": "http://localhost:3000",
            "Access-Control-Allow-Credentials": "true"
        }
        
        async def call_next(req):
            return response
        
        with patch('api.middleware.security_headers.api_config') as mock_config:
            mock_config.debug = True
            result = await middleware.dispatch(request, call_next)
            
            assert "Access-Control-Allow-Origin" in result.headers
            assert result.headers["Access-Control-Allow-Origin"] == "http://localhost:3000"

