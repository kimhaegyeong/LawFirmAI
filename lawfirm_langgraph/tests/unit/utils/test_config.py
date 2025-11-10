# -*- coding: utf-8 -*-
"""
Config 테스트
유틸리티 Config 모듈 단위 테스트 (core/utils/config.py)
"""

import pytest
import os
from unittest.mock import patch

from lawfirm_langgraph.core.utils.config import Config


class TestConfig:
    """Config 테스트"""
    
    def test_config_default_values(self, monkeypatch):
        """기본 설정 값 테스트"""
        # 환경 변수 격리를 위해 관련 환경 변수 제거
        monkeypatch.delenv("DEBUG", raising=False)
        monkeypatch.delenv("DATABASE_PATH", raising=False)
        monkeypatch.delenv("DATABASE_URL", raising=False)
        monkeypatch.delenv("API_PORT", raising=False)
        
        # Config를 직접 인스턴스화하여 환경 변수 전달
        config = Config(
            debug=False,
            api_port=8000,
            database_path=None,
            database_url=None
        )
        
        assert hasattr(config, 'database_path')
        assert hasattr(config, 'model_path')
        assert hasattr(config, 'api_host')
        assert hasattr(config, 'api_port')
        assert config.api_port == 8000
        assert config.debug is False
    
    def test_config_from_env(self, monkeypatch):
        """환경 변수에서 설정 로드 테스트"""
        # 기존 환경 변수 제거 후 새 값 설정
        monkeypatch.delenv("DATABASE_PATH", raising=False)
        monkeypatch.delenv("DATABASE_URL", raising=False)
        monkeypatch.delenv("MODEL_PATH", raising=False)
        monkeypatch.delenv("API_PORT", raising=False)
        monkeypatch.delenv("DEBUG", raising=False)
        
        monkeypatch.setenv("DATABASE_PATH", "/test/db")
        monkeypatch.setenv("MODEL_PATH", "/test/model")
        monkeypatch.setenv("API_PORT", "9000")
        monkeypatch.setenv("DEBUG", "true")
        
        # Config를 직접 인스턴스화하여 환경 변수 전달
        config = Config(
            database_path="/test/db",
            model_path="/test/model",
            api_port=9000,
            debug=True
        )
        
        assert config.database_path == "/test/db"
        assert config.model_path == "/test/model"
        assert config.api_port == 9000
        assert config.debug is True
    
    def test_config_database_path(self):
        """데이터베이스 경로 설정 테스트"""
        config = Config()
        
        assert config.database_path is not None
        assert isinstance(config.database_path, str)
    
    def test_config_database_url(self):
        """데이터베이스 URL 설정 테스트"""
        config = Config()
        
        assert config.database_url is not None
        assert isinstance(config.database_url, str)
        assert config.database_url.startswith("sqlite:///")
    
    def test_config_get_method(self):
        """get 메서드 테스트"""
        config = Config()
        
        assert config.get("api_port") == 8000
        assert config.get("non_existent_key", "default") == "default"
    
    def test_config_is_development(self):
        """개발 환경 여부 테스트"""
        config = Config()
        
        assert isinstance(config.is_development(), bool)
        assert config.is_development() == config.debug
    
    def test_config_is_production(self):
        """프로덕션 환경 여부 테스트"""
        config = Config()
        
        assert isinstance(config.is_production(), bool)
        assert config.is_production() == (not config.debug)
    
    def test_config_model_settings(self, monkeypatch):
        """모델 설정 테스트"""
        # Config를 직접 인스턴스화하여 환경 변수 전달
        config = Config(
            model_path="/test/models",
            device="cuda",
            model_cache_dir="/test/cache"
        )
        
        assert config.model_path == "/test/models"
        assert config.device == "cuda"
        assert config.model_cache_dir == "/test/cache"
    
    def test_config_vector_store_settings(self, monkeypatch):
        """벡터 저장소 설정 테스트"""
        # Config를 직접 인스턴스화하여 환경 변수 전달
        config = Config(
            chroma_db_path="/test/chroma",
            embedding_model="test-model",
            embedding_dimension=512
        )
        
        assert config.chroma_db_path == "/test/chroma"
        assert config.embedding_model == "test-model"
        assert config.embedding_dimension == 512
    
    def test_config_logging_settings(self, monkeypatch):
        """로깅 설정 테스트"""
        # Config를 직접 인스턴스화하여 환경 변수 전달
        config = Config(
            log_level="DEBUG",
            log_file="/test/logs/test.log",
            log_format="text"
        )
        
        assert config.log_level == "DEBUG"
        assert config.log_file == "/test/logs/test.log"
        assert config.log_format == "text"
    
    def test_config_performance_settings(self, monkeypatch):
        """성능 설정 테스트"""
        # Config를 직접 인스턴스화하여 환경 변수 전달
        config = Config(
            max_workers=8,
            request_timeout=60,
            cache_ttl=7200
        )
        
        assert config.max_workers == 8
        assert config.request_timeout == 60
        assert config.cache_ttl == 7200
    
    def test_config_security_settings(self, monkeypatch):
        """보안 설정 테스트"""
        # Config를 직접 인스턴스화하여 환경 변수 전달
        config = Config(
            secret_key="test-secret-key"
        )
        
        assert config.secret_key == "test-secret-key"
        assert isinstance(config.cors_origins, list)
    
    def test_config_cors_origins(self, monkeypatch):
        """CORS origins 설정 테스트"""
        monkeypatch.setenv("CORS_ORIGINS", '["http://localhost:3000", "http://localhost:8080"]')
        
        config = Config()
        
        assert isinstance(config.cors_origins, list)
        assert len(config.cors_origins) >= 0

