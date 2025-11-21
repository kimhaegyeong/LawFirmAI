"""
API 설정 테스트
"""
import pytest
import os
from unittest.mock import patch
from api.config import APIConfig, get_api_config, api_config


class TestAPIConfig:
    """APIConfig 테스트"""
    
    def test_default_values(self):
        """기본값 테스트"""
        with patch.dict(os.environ, {}, clear=True):
            config = APIConfig()
            assert config.api_host == "0.0.0.0"
            assert config.api_port == 8000
            assert config.cors_origins == "http://localhost:3000,http://127.0.0.1:3000"
    
    def test_debug_auto_detection_from_env(self):
        """환경 변수로부터 debug 자동 감지 테스트"""
        with patch.dict(os.environ, {"DEBUG": "true"}):
            config = APIConfig()
            assert config.debug is True
        
        with patch.dict(os.environ, {"ENVIRONMENT": "development"}):
            config = APIConfig()
            assert config.debug is True
    
    def test_get_cors_origins_string(self):
        """문자열 형태의 CORS origins 파싱 테스트"""
        config = APIConfig(cors_origins="http://localhost:3000,http://localhost:5173")
        origins = config.get_cors_origins()
        assert isinstance(origins, list)
        assert "http://localhost:3000" in origins
        assert "http://localhost:5173" in origins
    
    def test_get_cors_origins_list(self):
        """리스트 형태의 CORS origins 테스트"""
        # Pydantic은 문자열만 받으므로, get_cors_origins 내부에서 리스트로 변환되는 경우를 테스트
        config = APIConfig(cors_origins="http://localhost:3000,http://localhost:5173")
        origins = config.get_cors_origins()
        assert isinstance(origins, list)
        assert "http://localhost:3000" in origins
        assert "http://localhost:5173" in origins
    
    def test_get_cors_origins_json(self):
        """JSON 형태의 CORS origins 파싱 테스트"""
        config = APIConfig(cors_origins='["http://localhost:3000", "http://localhost:5173"]')
        origins = config.get_cors_origins()
        assert isinstance(origins, list)
        assert "http://localhost:3000" in origins
        assert "http://localhost:5173" in origins
    
    def test_get_cors_origins_wildcard_debug(self):
        """개발 환경에서 와일드카드 허용 테스트"""
        config = APIConfig(cors_origins="*", debug=True)
        origins = config.get_cors_origins()
        assert origins == ["*"]
    
    def test_get_cors_origins_wildcard_production(self):
        """프로덕션 환경에서 와일드카드 제거 테스트"""
        config = APIConfig(cors_origins="*", debug=False)
        origins = config.get_cors_origins()
        assert "*" not in origins
        assert "http://localhost:3000" in origins
    
    def test_get_api_config_singleton(self):
        """get_api_config 싱글톤 패턴 테스트"""
        config1 = get_api_config()
        config2 = get_api_config()
        assert config1 is config2
    
    def test_api_config_proxy(self):
        """api_config 프록시 테스트"""
        assert api_config.api_host == "0.0.0.0"
        assert api_config.api_port == 8000

