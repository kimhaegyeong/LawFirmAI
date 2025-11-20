# -*- coding: utf-8 -*-
"""
LangGraphConfig 테스트
설정 관리 모듈 테스트
"""

import os
import pytest
from unittest.mock import patch, MagicMock

from lawfirm_langgraph.config.langgraph_config import (
    LangGraphConfig,
    CheckpointStorageType,
)


class TestCheckpointStorageType:
    """CheckpointStorageType Enum 테스트"""
    
    def test_enum_values(self):
        """Enum 값 확인"""
        assert CheckpointStorageType.MEMORY.value == "memory"
        assert CheckpointStorageType.SQLITE.value == "sqlite"
        assert CheckpointStorageType.POSTGRES.value == "postgres"
        assert CheckpointStorageType.REDIS.value == "redis"
        assert CheckpointStorageType.DISABLED.value == "disabled"


class TestLangGraphConfig:
    """LangGraphConfig 테스트"""
    
    def test_config_default_values(self):
        """기본 설정 값 테스트"""
        config = LangGraphConfig()
        
        assert config.enable_checkpoint is True
        assert config.checkpoint_storage == CheckpointStorageType.MEMORY
        assert config.max_iterations == 10
        assert config.recursion_limit == 25
        assert config.enable_streaming is True
        assert config.langgraph_enabled is True
        assert config.use_agentic_mode is False
    
    def test_config_from_env(self, monkeypatch):
        """환경 변수에서 설정 로드 테스트"""
        monkeypatch.setenv("LANGGRAPH_ENABLED", "true")
        monkeypatch.setenv("CHECKPOINT_STORAGE", "sqlite")
        monkeypatch.setenv("MAX_ITERATIONS", "20")
        monkeypatch.setenv("USE_AGENTIC_MODE", "true")
        
        config = LangGraphConfig.from_env()
        
        assert config.langgraph_enabled is True
        assert config.checkpoint_storage == CheckpointStorageType.SQLITE
        assert config.max_iterations == 20
        assert config.use_agentic_mode is True
    
    def test_config_from_env_memory_storage(self, monkeypatch):
        """메모리 저장소 설정 테스트"""
        monkeypatch.setenv("CHECKPOINT_STORAGE", "memory")
        
        config = LangGraphConfig.from_env()
        
        assert config.checkpoint_storage == CheckpointStorageType.MEMORY
    
    def test_config_from_env_disabled_checkpoint(self, monkeypatch):
        """체크포인트 비활성화 테스트"""
        monkeypatch.setenv("CHECKPOINT_STORAGE", "disabled")
        
        config = LangGraphConfig.from_env()
        
        assert config.checkpoint_storage == CheckpointStorageType.DISABLED
        assert config.enable_checkpoint is False
    
    def test_config_validate_success(self):
        """설정 유효성 검사 성공 테스트"""
        config = LangGraphConfig(
            enable_checkpoint=True,
            checkpoint_db_path="./test.db",
            checkpoint_ttl=3600,
            max_iterations=10,
            recursion_limit=25,
        )
        
        errors = config.validate()
        
        assert len(errors) == 0
    
    def test_config_validate_failures(self):
        """설정 유효성 검사 실패 테스트"""
        config = LangGraphConfig(
            enable_checkpoint=True,
            checkpoint_db_path="",  # 빈 경로
            checkpoint_ttl=-1,  # 음수
            max_iterations=0,  # 0 또는 음수
            recursion_limit=-1,  # 음수
        )
        
        errors = config.validate()
        
        assert len(errors) > 0
        assert any("LANGGRAPH_CHECKPOINT_DB" in error for error in errors)
        assert any("CHECKPOINT_TTL" in error for error in errors)
        assert any("MAX_ITERATIONS" in error for error in errors)
        assert any("RECURSION_LIMIT" in error for error in errors)
    
    def test_config_to_dict(self):
        """설정을 딕셔너리로 변환 테스트"""
        config = LangGraphConfig(
            enable_checkpoint=True,
            checkpoint_storage=CheckpointStorageType.MEMORY,
            max_iterations=15,
        )
        
        config_dict = config.to_dict()
        
        assert isinstance(config_dict, dict)
        assert config_dict["enable_checkpoint"] is True
        assert config_dict["checkpoint_storage"] == "memory"
        assert config_dict["max_iterations"] == 15
    
    def test_config_llm_provider_settings(self, monkeypatch):
        """LLM 프로바이더 설정 테스트"""
        monkeypatch.setenv("LLM_PROVIDER", "google")
        monkeypatch.setenv("GOOGLE_MODEL", "gemini-pro")
        monkeypatch.setenv("GOOGLE_API_KEY", "test_key")
        
        config = LangGraphConfig.from_env()
        
        assert config.llm_provider == "google"
        assert config.google_model == "gemini-pro"
        assert config.google_api_key == "test_key"
    
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
        
        config = LangGraphConfig.from_env()
        
        assert config.langfuse_enabled is True
        assert config.langfuse_secret_key == "test_secret"
        assert config.langfuse_public_key == "test_public"
        assert config.langfuse_host == "https://test.langfuse.com"
    
    def test_config_langsmith_settings(self, monkeypatch):
        """LangSmith 설정 테스트"""
        monkeypatch.setenv("LANGCHAIN_TRACING_V2", "true")
        monkeypatch.setenv("LANGCHAIN_API_KEY", "test_langsmith_key")
        monkeypatch.setenv("LANGCHAIN_PROJECT", "TestProject")
        
        config = LangGraphConfig.from_env()
        
        assert config.langsmith_enabled is True
        assert config.langsmith_api_key == "test_langsmith_key"
        assert config.langsmith_project == "TestProject"

