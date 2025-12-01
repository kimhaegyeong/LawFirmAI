# -*- coding: utf-8 -*-
"""
AppConfig 테스트
config/app_config.py 단위 테스트
"""

import os
import pytest
from unittest.mock import patch, mock_open
from pathlib import Path

from lawfirm_langgraph.config.app_config import Config


class TestAppConfig:
    """AppConfig 테스트"""
    
    def test_config_default_values(self, monkeypatch):
        """기본 설정 값 테스트"""
        monkeypatch.delenv("DEBUG", raising=False)
        monkeypatch.delenv("API_PORT", raising=False)
        monkeypatch.delenv("DATABASE_PATH", raising=False)
        
        config = Config()
        
        assert config.api_host == "0.0.0.0"
        assert config.api_port == 8000
        assert config.debug is False
        assert config.database_path == "./data/lawfirm_v2.db"
        assert config.model_path == "./models"
        assert config.device == "cpu"
    
    def test_config_from_env(self, monkeypatch):
        """환경 변수에서 설정 로드 테스트"""
        monkeypatch.setenv("API_HOST", "127.0.0.1")
        monkeypatch.setenv("API_PORT", "9000")
        monkeypatch.setenv("DEBUG", "true")
        monkeypatch.setenv("DATABASE_PATH", "/test/db")
        monkeypatch.setenv("MODEL_PATH", "/test/model")
        
        config = Config()
        
        assert config.api_host == "127.0.0.1"
        assert config.api_port == 9000
        assert config.debug is True
        assert config.database_path == "/test/db"
        assert config.model_path == "/test/model"
    
    def test_config_get_method(self):
        """get 메서드 테스트"""
        config = Config()
        
        assert config.get("api_port") == 8000
        assert config.get("non_existent_key", "default") == "default"
        assert config.get("api_host") == "0.0.0.0"
    
    def test_config_is_development(self, monkeypatch):
        """개발 환경 여부 테스트"""
        monkeypatch.setenv("DEBUG", "true")
        config = Config()
        assert config.is_development() is True
        
        monkeypatch.setenv("DEBUG", "false")
        config = Config()
        assert config.is_development() is False
    
    def test_config_is_production(self, monkeypatch):
        """프로덕션 환경 여부 테스트"""
        monkeypatch.setenv("DEBUG", "false")
        config = Config()
        assert config.is_production() is True
        
        monkeypatch.setenv("DEBUG", "true")
        config = Config()
        assert config.is_production() is False
    
    def test_config_database_settings(self, monkeypatch):
        """데이터베이스 설정 테스트"""
        monkeypatch.setenv("DATABASE_URL", "sqlite:///test.db")
        monkeypatch.setenv("DATABASE_PATH", "/test/path.db")
        
        config = Config()
        
        assert config.database_url == "sqlite:///test.db"
        assert config.database_path == "/test/path.db"
    
    def test_config_model_settings(self, monkeypatch):
        """모델 설정 테스트"""
        monkeypatch.setenv("MODEL_PATH", "/test/models")
        monkeypatch.setenv("DEVICE", "cuda")
        monkeypatch.setenv("MODEL_CACHE_DIR", "/test/cache")
        
        config = Config()
        
        assert config.model_path == "/test/models"
        assert config.device == "cuda"
        assert config.model_cache_dir == "/test/cache"
    
    def test_config_vector_store_settings(self, monkeypatch):
        """벡터 저장소 설정 테스트"""
        monkeypatch.setenv("CHROMA_DB_PATH", "/test/chroma")
        monkeypatch.setenv("EMBEDDING_MODEL", "test-model")
        monkeypatch.setenv("EMBEDDING_DIMENSION", "512")
        
        config = Config()
        
        assert config.chroma_db_path == "/test/chroma"
        assert config.embedding_model == "test-model"
        assert config.embedding_dimension == 512
    
    def test_config_logging_settings(self, monkeypatch):
        """로깅 설정 테스트"""
        monkeypatch.setenv("LOG_LEVEL", "DEBUG")
        monkeypatch.setenv("LOG_FILE", "/test/logs/test.log")
        monkeypatch.setenv("LOG_FORMAT", "text")
        
        config = Config()
        
        assert config.log_level == "DEBUG"
        assert config.log_file == "/test/logs/test.log"
        assert config.log_format == "text"
    
    def test_config_performance_settings(self, monkeypatch):
        """성능 설정 테스트"""
        monkeypatch.setenv("MAX_WORKERS", "8")
        monkeypatch.setenv("REQUEST_TIMEOUT", "60")
        monkeypatch.setenv("CACHE_TTL", "7200")
        
        config = Config()
        
        assert config.max_workers == 8
        assert config.request_timeout == 60
        assert config.cache_ttl == 7200
    
    def test_config_security_settings(self, monkeypatch):
        """보안 설정 테스트"""
        monkeypatch.setenv("SECRET_KEY", "test-secret-key")
        
        config = Config()
        
        assert config.secret_key == "test-secret-key"
        assert isinstance(config.cors_origins, list)
    
    def test_config_env_file_loading(self, monkeypatch, tmp_path):
        """환경 변수 파일 로딩 테스트"""
        env_file = tmp_path / ".env"
        env_file.write_text("API_PORT=9000\nDEBUG=true\nDATABASE_PATH=/test/db")
        
        with patch('lawfirm_langgraph.config.app_config.Path') as mock_path:
            mock_path.return_value.exists.return_value = True
            mock_path.return_value.__str__ = lambda x: str(env_file)
            
            with patch('builtins.open', mock_open(read_data="API_PORT=9000\nDEBUG=true\nDATABASE_PATH=/test/db")):
                config = Config()
                assert config.api_port == 9000
    
    def test_config_env_file_not_exists(self, monkeypatch):
        """환경 변수 파일이 없는 경우 테스트"""
        with patch('lawfirm_langgraph.config.app_config.Path') as mock_path:
            mock_path.return_value.exists.return_value = False
            config = Config()
            assert config.api_port == 8000
    
    def test_config_law_open_api_settings(self, monkeypatch):
        """법제처 API 설정 테스트"""
        monkeypatch.setenv("LAW_OPEN_API_OC", "test@example.com")
        monkeypatch.setenv("LAW_FIRM_AI_API_KEY", "test-api-key")
        
        config = Config()
        
        assert config.law_open_api_oc == "test@example.com"
        assert config.law_firm_ai_api_key == "test-api-key"
    
    def test_config_versioned_settings(self, monkeypatch):
        """버전 관리 설정 테스트"""
        monkeypatch.setenv("ACTIVE_CORPUS_VERSION", "v2")
        monkeypatch.setenv("ACTIVE_MODEL_VERSION", "default@2.0")
        monkeypatch.setenv("EMBEDDINGS_BASE_DIR", "/test/embeddings")
        monkeypatch.setenv("VERSIONED_DATABASE_DIR", "/test/database")
        
        config = Config()
        
        assert config.active_corpus_version == "v2"
        assert config.active_model_version == "default@2.0"
        assert config.embeddings_base_dir == "/test/embeddings"
        assert config.versioned_database_dir == "/test/database"
    
    def test_config_monitoring_settings(self, monkeypatch):
        """모니터링 설정 테스트"""
        monkeypatch.setenv("ENABLE_METRICS", "true")
        monkeypatch.setenv("METRICS_PORT", "9090")
        
        config = Config()
        
        assert config.enable_metrics is True
        assert config.metrics_port == 9090
    
    def test_config_huggingface_spaces_settings(self, monkeypatch):
        """HuggingFace Spaces 설정 테스트"""
        monkeypatch.setenv("HF_SPACE_ID", "test-space")
        
        config = Config()
        
        assert config.hf_space_id == "test-space"
    
    def test_config_retry_settings(self, monkeypatch):
        """재시도 설정 테스트"""
        monkeypatch.setenv("REQUEST_MAX_RETRIES", "5")
        monkeypatch.setenv("REQUEST_BACKOFF_BASE", "1.0")
        monkeypatch.setenv("REQUEST_BACKOFF_MAX", "10.0")
        
        config = Config()
        
        assert config.request_max_retries == 5
        assert config.request_backoff_base == 1.0
        assert config.request_backoff_max == 10.0

