# -*- coding: utf-8 -*-
"""
LangGraph Config Utils 테스트
core/utils/langgraph_config.py 단위 테스트
"""

import os
import pytest
from unittest.mock import patch

from lawfirm_langgraph.core.utils.langgraph_config import (
    LangGraphConfig,
    CheckpointStorageType
)


class TestLangGraphConfigUtils:
    """LangGraphConfig Utils 테스트"""
    
    def test_config_default_values(self):
        """기본 설정 값 테스트"""
        config = LangGraphConfig()
        
        assert config.checkpoint_storage == CheckpointStorageType.SQLITE
        assert config.checkpoint_db_path == "./data/checkpoints/langgraph.db"
        assert config.checkpoint_ttl == 3600
        assert config.max_iterations == 10
        assert config.recursion_limit == 25
        assert config.enable_streaming is True
        assert config.llm_provider == "google"
        assert config.google_model == "gemini-2.5-flash-lite"
        assert config.langgraph_enabled is True
        assert config.use_agentic_mode is False
    
    def test_config_from_env(self, monkeypatch):
        """환경 변수에서 설정 로드 테스트"""
        monkeypatch.setenv("LANGGRAPH_ENABLED", "true")
        monkeypatch.setenv("CHECKPOINT_STORAGE", "postgres")
        monkeypatch.setenv("LANGGRAPH_CHECKPOINT_DB", "/test/db")
        monkeypatch.setenv("CHECKPOINT_TTL", "7200")
        monkeypatch.setenv("MAX_ITERATIONS", "20")
        monkeypatch.setenv("RECURSION_LIMIT", "50")
        monkeypatch.setenv("ENABLE_STREAMING", "false")
        monkeypatch.setenv("LLM_PROVIDER", "google")
        monkeypatch.setenv("GOOGLE_MODEL", "gemini-pro")
        monkeypatch.setenv("GOOGLE_API_KEY", "test-key")
        monkeypatch.setenv("USE_AGENTIC_MODE", "true")
        
        config = LangGraphConfig.from_env()
        
        assert config.langgraph_enabled is True
        assert config.checkpoint_storage == CheckpointStorageType.POSTGRES
        assert config.checkpoint_db_path == "/test/db"
        assert config.checkpoint_ttl == 7200
        assert config.max_iterations == 20
        assert config.recursion_limit == 50
        assert config.enable_streaming is False
        assert config.llm_provider == "google"
        assert config.google_model == "gemini-pro"
        assert config.google_api_key == "test-key"
        assert config.use_agentic_mode is True
    
    def test_config_from_env_sqlite(self, monkeypatch):
        """SQLite 저장소 설정 테스트"""
        monkeypatch.setenv("CHECKPOINT_STORAGE", "sqlite")
        
        config = LangGraphConfig.from_env()
        
        assert config.checkpoint_storage == CheckpointStorageType.SQLITE
    
    def test_config_from_env_redis(self, monkeypatch):
        """Redis 저장소 설정 테스트"""
        monkeypatch.setenv("CHECKPOINT_STORAGE", "redis")
        
        config = LangGraphConfig.from_env()
        
        assert config.checkpoint_storage == CheckpointStorageType.REDIS
    
    def test_config_validate_success(self):
        """설정 유효성 검사 성공 테스트"""
        config = LangGraphConfig(
            langgraph_enabled=True,
            checkpoint_db_path="./test.db",
            checkpoint_ttl=3600,
            max_iterations=10,
            recursion_limit=25,
            ollama_timeout=15
        )
        
        errors = config.validate()
        
        assert len(errors) == 0
    
    def test_config_validate_failures(self):
        """설정 유효성 검사 실패 테스트"""
        config = LangGraphConfig(
            langgraph_enabled=True,
            checkpoint_db_path="",
            checkpoint_ttl=-1,
            max_iterations=0,
            recursion_limit=-1,
            ollama_timeout=0
        )
        
        errors = config.validate()
        
        assert len(errors) > 0
        assert any("LANGGRAPH_CHECKPOINT_DB" in error for error in errors)
        assert any("CHECKPOINT_TTL" in error for error in errors)
        assert any("MAX_ITERATIONS" in error for error in errors)
        assert any("RECURSION_LIMIT" in error for error in errors)
        assert any("OLLAMA_TIMEOUT" in error for error in errors)
    
    def test_config_to_dict(self):
        """설정을 딕셔너리로 변환 테스트"""
        config = LangGraphConfig(
            checkpoint_storage=CheckpointStorageType.SQLITE,
            max_iterations=15
        )
        
        config_dict = config.to_dict()
        
        assert isinstance(config_dict, dict)
        assert config_dict["checkpoint_storage"] == "sqlite"
        assert config_dict["max_iterations"] == 15
        assert "checkpoint_db_path" in config_dict
        assert "enable_streaming" in config_dict
        assert "langgraph_enabled" in config_dict
    
    def test_config_ollama_settings(self, monkeypatch):
        """Ollama 설정 테스트"""
        monkeypatch.setenv("OLLAMA_BASE_URL", "http://localhost:11434")
        monkeypatch.setenv("OLLAMA_MODEL", "qwen2.5:3b")
        monkeypatch.setenv("OLLAMA_TIMEOUT", "30")
        
        config = LangGraphConfig.from_env()
        
        assert config.ollama_base_url == "http://localhost:11434"
        assert config.ollama_model == "qwen2.5:3b"
        assert config.ollama_timeout == 30
    
    def test_config_rag_settings(self, monkeypatch):
        """RAG 설정 테스트"""
        monkeypatch.setenv("SIMILARITY_THRESHOLD", "0.5")
        monkeypatch.setenv("MAX_CONTEXT_LENGTH", "8000")
        monkeypatch.setenv("MAX_TOKENS", "4000")
        
        config = LangGraphConfig.from_env()
        
        assert config.similarity_threshold == 0.5
        assert config.max_context_length == 8000
        assert config.max_tokens == 4000
    
    def test_config_state_optimization_settings(self, monkeypatch):
        """State 최적화 설정 테스트"""
        monkeypatch.setenv("MAX_RETRIEVED_DOCS", "20")
        monkeypatch.setenv("MAX_DOCUMENT_CONTENT_LENGTH", "1000")
        monkeypatch.setenv("MAX_CONVERSATION_HISTORY", "10")
        monkeypatch.setenv("MAX_PROCESSING_STEPS", "30")
        monkeypatch.setenv("ENABLE_STATE_PRUNING", "false")
        
        config = LangGraphConfig.from_env()
        
        assert config.max_retrieved_docs == 20
        assert config.max_document_content_length == 1000
        assert config.max_conversation_history == 10
        assert config.max_processing_steps == 30
        assert config.enable_state_pruning is False
    
    def test_config_langfuse_settings(self, monkeypatch):
        """Langfuse 설정 테스트"""
        monkeypatch.setenv("LANGFUSE_ENABLED", "true")
        monkeypatch.setenv("LANGFUSE_SECRET_KEY", "test_secret")
        monkeypatch.setenv("LANGFUSE_PUBLIC_KEY", "test_public")
        monkeypatch.setenv("LANGFUSE_HOST", "https://test.langfuse.com")
        monkeypatch.setenv("LANGFUSE_DEBUG", "true")
        
        config = LangGraphConfig.from_env()
        
        assert config.langfuse_enabled is True
        assert config.langfuse_secret_key == "test_secret"
        assert config.langfuse_public_key == "test_public"
        assert config.langfuse_host == "https://test.langfuse.com"
        assert config.langfuse_debug is True
    
    def test_config_langsmith_settings(self, monkeypatch):
        """LangSmith 설정 테스트"""
        monkeypatch.setenv("LANGSMITH_TRACING", "true")
        monkeypatch.setenv("LANGSMITH_API_KEY", "test_langsmith_key")
        monkeypatch.setenv("LANGSMITH_PROJECT", "TestProject")
        monkeypatch.setenv("LANGSMITH_ENDPOINT", "https://test.smith.langchain.com")
        
        config = LangGraphConfig.from_env()
        
        assert config.langsmith_enabled is True
        assert config.langsmith_api_key == "test_langsmith_key"
        assert config.langsmith_project == "TestProject"
        assert config.langsmith_endpoint == "https://test.smith.langchain.com"
    
    def test_config_langsmith_backward_compatibility(self, monkeypatch):
        """LangSmith 하위 호환성 테스트"""
        monkeypatch.setenv("LANGCHAIN_TRACING_V2", "true")
        monkeypatch.setenv("LANGCHAIN_API_KEY", "test_key")
        monkeypatch.setenv("LANGCHAIN_PROJECT", "TestProject")
        monkeypatch.setenv("LANGCHAIN_ENDPOINT", "https://test.endpoint.com")
        
        config = LangGraphConfig.from_env()
        
        assert config.langsmith_enabled is True
        assert config.langsmith_api_key == "test_key"
        assert config.langsmith_project == "TestProject"
        assert config.langsmith_endpoint == "https://test.endpoint.com"
    
    def test_config_statistics_settings(self, monkeypatch):
        """통계 관리 설정 테스트"""
        monkeypatch.setenv("ENABLE_STATISTICS", "false")
        monkeypatch.setenv("STATS_UPDATE_ALPHA", "0.2")
        
        config = LangGraphConfig.from_env()
        
        assert config.enable_statistics is False
        assert config.stats_update_alpha == 0.2
    
    def test_config_complexity_settings(self, monkeypatch):
        """복잡도 분류 설정 테스트"""
        monkeypatch.setenv("USE_LLM_FOR_COMPLEXITY", "false")
        monkeypatch.setenv("COMPLEXITY_LLM_MODEL", "main")
        
        config = LangGraphConfig.from_env()
        
        assert config.use_llm_for_complexity is False
        assert config.complexity_llm_model == "main"

