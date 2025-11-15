# -*- coding: utf-8 -*-
"""
LangChain Config 테스트
core/utils/langchain_config.py 단위 테스트
"""

import os
import pytest
from unittest.mock import patch

from lawfirm_langgraph.core.utils.langchain_config import (
    LangChainConfig,
    VectorStoreType,
    LLMProvider,
    PromptTemplates
)


class TestVectorStoreType:
    """VectorStoreType Enum 테스트"""
    
    def test_enum_values(self):
        """Enum 값 확인"""
        assert VectorStoreType.FAISS.value == "faiss"
        assert VectorStoreType.CHROMA.value == "chroma"
        assert VectorStoreType.PINECONE.value == "pinecone"


class TestLLMProvider:
    """LLMProvider Enum 테스트"""
    
    def test_enum_values(self):
        """Enum 값 확인"""
        assert LLMProvider.OPENAI.value == "openai"
        assert LLMProvider.LOCAL.value == "local"
        assert LLMProvider.ANTHROPIC.value == "anthropic"
        assert LLMProvider.GOOGLE.value == "google"


class TestLangChainConfig:
    """LangChainConfig 테스트"""
    
    def test_config_default_values(self):
        """기본 설정 값 테스트"""
        config = LangChainConfig()
        
        assert config.vector_store_type == VectorStoreType.FAISS
        assert config.vector_store_path == "./data/embeddings/faiss_index"
        assert config.embedding_model == "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
        assert config.chunk_size == 1000
        assert config.chunk_overlap == 200
        assert config.max_context_length == 4000
        assert config.search_k == 5
        assert config.similarity_threshold == 0.7
        assert config.llm_provider == LLMProvider.OPENAI
        assert config.llm_model == "gpt-3.5-turbo"
        assert config.llm_temperature == 0.7
        assert config.llm_max_tokens == 1000
        assert config.enable_caching is True
        assert config.cache_ttl == 3600
        assert config.enable_async is True
    
    def test_config_from_env(self, monkeypatch):
        """환경 변수에서 설정 로드 테스트"""
        monkeypatch.setenv("VECTOR_STORE_TYPE", "chroma")
        monkeypatch.setenv("VECTOR_STORE_PATH", "/test/path")
        monkeypatch.setenv("EMBEDDING_MODEL", "test-model")
        monkeypatch.setenv("CHUNK_SIZE", "2000")
        monkeypatch.setenv("CHUNK_OVERLAP", "400")
        monkeypatch.setenv("MAX_CONTEXT_LENGTH", "8000")
        monkeypatch.setenv("SEARCH_K", "10")
        monkeypatch.setenv("SIMILARITY_THRESHOLD", "0.8")
        monkeypatch.setenv("LLM_PROVIDER", "google")
        monkeypatch.setenv("LLM_MODEL", "gemini-pro")
        monkeypatch.setenv("LLM_TEMPERATURE", "0.9")
        monkeypatch.setenv("LLM_MAX_TOKENS", "2000")
        monkeypatch.setenv("GOOGLE_API_KEY", "test-key")
        monkeypatch.setenv("ENABLE_CACHING", "false")
        monkeypatch.setenv("CACHE_TTL", "7200")
        monkeypatch.setenv("ENABLE_ASYNC", "false")
        
        config = LangChainConfig.from_env()
        
        assert config.vector_store_type == VectorStoreType.CHROMA
        assert config.vector_store_path == "/test/path"
        assert config.embedding_model == "test-model"
        assert config.chunk_size == 2000
        assert config.chunk_overlap == 400
        assert config.max_context_length == 8000
        assert config.search_k == 10
        assert config.similarity_threshold == 0.8
        assert config.llm_provider == LLMProvider.GOOGLE
        assert config.llm_model == "gemini-pro"
        assert config.llm_temperature == 0.9
        assert config.llm_max_tokens == 2000
        assert config.google_api_key == "test-key"
        assert config.enable_caching is False
        assert config.cache_ttl == 7200
        assert config.enable_async is False
    
    def test_config_from_env_faiss(self, monkeypatch):
        """FAISS 벡터 저장소 설정 테스트"""
        monkeypatch.setenv("VECTOR_STORE_TYPE", "faiss")
        
        config = LangChainConfig.from_env()
        
        assert config.vector_store_type == VectorStoreType.FAISS
    
    def test_config_from_env_pinecone(self, monkeypatch):
        """Pinecone 벡터 저장소 설정 테스트"""
        monkeypatch.setenv("VECTOR_STORE_TYPE", "pinecone")
        
        config = LangChainConfig.from_env()
        
        assert config.vector_store_type == VectorStoreType.PINECONE
    
    def test_config_from_env_llm_providers(self, monkeypatch):
        """LLM 프로바이더 설정 테스트"""
        providers = ["openai", "local", "anthropic", "google"]
        
        for provider in providers:
            monkeypatch.setenv("LLM_PROVIDER", provider)
            config = LangChainConfig.from_env()
            assert config.llm_provider.value == provider
    
    def test_config_validate_success(self):
        """설정 유효성 검사 성공 테스트"""
        config = LangChainConfig(
            langfuse_enabled=False,
            llm_provider=LLMProvider.OPENAI,
            chunk_size=1000,
            chunk_overlap=200,
            similarity_threshold=0.7,
            llm_temperature=0.7
        )
        
        errors = config.validate()
        
        assert len(errors) == 0
    
    def test_config_validate_langfuse_errors(self):
        """Langfuse 설정 유효성 검사 테스트"""
        config = LangChainConfig(
            langfuse_enabled=True,
            langfuse_secret_key=None,
            langfuse_public_key=None
        )
        
        errors = config.validate()
        
        assert len(errors) > 0
        assert any("LANGFUSE_SECRET_KEY" in error for error in errors)
        assert any("LANGFUSE_PUBLIC_KEY" in error for error in errors)
    
    def test_config_validate_local_llm_error(self):
        """로컬 LLM 설정 유효성 검사 테스트"""
        config = LangChainConfig(
            llm_provider=LLMProvider.LOCAL,
            local_model_path=None
        )
        
        errors = config.validate()
        
        assert len(errors) > 0
        assert any("LOCAL_MODEL_PATH" in error for error in errors)
    
    def test_config_validate_google_llm_error(self):
        """Google LLM 설정 유효성 검사 테스트"""
        config = LangChainConfig(
            llm_provider=LLMProvider.GOOGLE,
            google_api_key=None
        )
        
        errors = config.validate()
        
        assert len(errors) > 0
        assert any("GOOGLE_API_KEY" in error for error in errors)
    
    def test_config_validate_chunk_size_error(self):
        """chunk_size 유효성 검사 테스트"""
        config = LangChainConfig(
            chunk_size=0
        )
        
        errors = config.validate()
        
        assert len(errors) > 0
        assert any("chunk_size" in error for error in errors)
    
    def test_config_validate_chunk_overlap_error(self):
        """chunk_overlap 유효성 검사 테스트"""
        config = LangChainConfig(
            chunk_size=1000,
            chunk_overlap=1000
        )
        
        errors = config.validate()
        
        assert len(errors) > 0
        assert any("chunk_overlap" in error for error in errors)
    
    def test_config_validate_similarity_threshold_error(self):
        """similarity_threshold 유효성 검사 테스트"""
        config = LangChainConfig(
            similarity_threshold=1.5
        )
        
        errors = config.validate()
        
        assert len(errors) > 0
        assert any("similarity_threshold" in error for error in errors)
    
    def test_config_validate_temperature_error(self):
        """llm_temperature 유효성 검사 테스트"""
        config = LangChainConfig(
            llm_temperature=3.0
        )
        
        errors = config.validate()
        
        assert len(errors) > 0
        assert any("llm_temperature" in error for error in errors)
    
    def test_config_to_dict(self):
        """설정을 딕셔너리로 변환 테스트"""
        config = LangChainConfig(
            vector_store_type=VectorStoreType.CHROMA,
            llm_provider=LLMProvider.GOOGLE,
            chunk_size=2000
        )
        
        config_dict = config.to_dict()
        
        assert isinstance(config_dict, dict)
        assert config_dict["vector_store_type"] == "chroma"
        assert config_dict["llm_provider"] == "google"
        assert config_dict["chunk_size"] == 2000
        assert "vector_store_path" in config_dict
        assert "embedding_model" in config_dict
        assert "llm_model" in config_dict
    
    def test_config_langfuse_settings(self, monkeypatch):
        """Langfuse 설정 테스트"""
        monkeypatch.setenv("LANGFUSE_ENABLED", "true")
        monkeypatch.setenv("LANGFUSE_SECRET_KEY", "test_secret")
        monkeypatch.setenv("LANGFUSE_PUBLIC_KEY", "test_public")
        monkeypatch.setenv("LANGFUSE_HOST", "https://test.langfuse.com")
        monkeypatch.setenv("LANGFUSE_DEBUG", "true")
        
        config = LangChainConfig.from_env()
        
        assert config.langfuse_enabled is True
        assert config.langfuse_secret_key == "test_secret"
        assert config.langfuse_public_key == "test_public"
        assert config.langfuse_host == "https://test.langfuse.com"
        assert config.langfuse_debug is True
    
    def test_config_local_model_settings(self, monkeypatch):
        """로컬 모델 설정 테스트"""
        monkeypatch.setenv("LOCAL_MODEL_PATH", "/test/model")
        monkeypatch.setenv("LOCAL_MODEL_DEVICE", "cuda")
        
        config = LangChainConfig.from_env()
        
        assert config.local_model_path == "/test/model"
        assert config.local_model_device == "cuda"


class TestPromptTemplates:
    """PromptTemplates 테스트"""
    
    def test_get_template_legal_qa(self):
        """법률 Q&A 템플릿 테스트"""
        template = PromptTemplates.get_template("legal_qa")
        
        assert isinstance(template, str)
        assert "{context}" in template
        assert "{question}" in template
    
    def test_get_template_legal_analysis(self):
        """법률 분석 템플릿 테스트"""
        template = PromptTemplates.get_template("legal_analysis")
        
        assert isinstance(template, str)
        assert "{context}" in template
        assert "{question}" in template
    
    def test_get_template_contract_review(self):
        """계약서 검토 템플릿 테스트"""
        template = PromptTemplates.get_template("contract_review")
        
        assert isinstance(template, str)
        assert "{context}" in template
        assert "{question}" in template
    
    def test_get_template_default(self):
        """기본 템플릿 테스트"""
        template = PromptTemplates.get_template("unknown")
        
        assert isinstance(template, str)
        assert template == PromptTemplates.LEGAL_QA_TEMPLATE
    
    def test_template_constants(self):
        """템플릿 상수 테스트"""
        assert hasattr(PromptTemplates, "LEGAL_QA_TEMPLATE")
        assert hasattr(PromptTemplates, "LEGAL_ANALYSIS_TEMPLATE")
        assert hasattr(PromptTemplates, "CONTRACT_REVIEW_TEMPLATE")
        
        assert isinstance(PromptTemplates.LEGAL_QA_TEMPLATE, str)
        assert isinstance(PromptTemplates.LEGAL_ANALYSIS_TEMPLATE, str)
        assert isinstance(PromptTemplates.CONTRACT_REVIEW_TEMPLATE, str)

