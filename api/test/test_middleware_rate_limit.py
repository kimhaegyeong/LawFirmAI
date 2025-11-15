"""
Rate Limiting 미들웨어 테스트
"""
import pytest
import sys
from pathlib import Path
from unittest.mock import patch, MagicMock
from slowapi.errors import RateLimitExceeded

# 프로젝트 루트를 sys.path에 추가
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from api.middleware.rate_limit import (
    limiter,
    create_rate_limit_response,
    get_rate_limit,
    is_rate_limit_enabled
)


class TestRateLimit:
    """Rate Limiting 테스트"""
    
    def test_get_rate_limit(self):
        """Rate Limiter 인스턴스 반환 테스트"""
        result = get_rate_limit()
        assert result is limiter
    
    def test_is_rate_limit_enabled(self):
        """Rate Limiting 활성화 여부 테스트"""
        with patch('api.middleware.rate_limit.api_config') as mock_config:
            mock_config.rate_limit_enabled = True
            assert is_rate_limit_enabled() is True
            
            mock_config.rate_limit_enabled = False
            assert is_rate_limit_enabled() is False
    
    def test_create_rate_limit_response(self):
        """Rate limit 초과 응답 생성 테스트"""
        request = MagicMock()
        request.app = MagicMock()
        request.app.state = MagicMock()
        request.app.state.limiter = MagicMock()
        request.state = MagicMock()
        request.state.view_rate_limit = MagicMock()
        
        exc = RateLimitExceeded("Rate limit exceeded", retry_after=60)
        exc.retry_after = 60
        
        mock_response = MagicMock()
        mock_response.headers = {}
        request.app.state.limiter._inject_headers = MagicMock(return_value=mock_response)
        
        result = create_rate_limit_response(request, exc)
        
        assert result is not None
        assert hasattr(result, 'status_code') or hasattr(result, 'headers')

