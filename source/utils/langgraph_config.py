# -*- coding: utf-8 -*-
"""
LangGraph Configuration
LangGraph 설정 관리 모듈
"""

import logging
import os
from dataclasses import dataclass
from enum import Enum
from typing import List

from dotenv import load_dotenv

# .env 파일 로드
load_dotenv()

logger = logging.getLogger(__name__)


class CheckpointStorageType(Enum):
    """체크포인트 저장소 타입"""
    SQLITE = "sqlite"
    POSTGRES = "postgres"
    REDIS = "redis"


@dataclass
class LangGraphConfig:
    """LangGraph 설정 클래스"""

    # 체크포인트 설정
    checkpoint_storage: CheckpointStorageType = CheckpointStorageType.SQLITE
    checkpoint_db_path: str = "./data/checkpoints/langgraph.db"
    checkpoint_ttl: int = 3600  # 1시간

    # 워크플로우 설정
    max_iterations: int = 10
    recursion_limit: int = 25
    enable_streaming: bool = True

    # LLM 설정 (Google Gemini 우선)
    llm_provider: str = "google"
    google_model: str = "gemini-2.5-flash-lite"  # .env 파일의 설정에 맞게 변경
    google_api_key: str = ""

    # 기존 Ollama 설정 (백업용)
    ollama_base_url: str = "http://localhost:11434"
    ollama_model: str = "qwen2.5:3b"
    ollama_timeout: int = 15

    # LangGraph 활성화 설정
    langgraph_enabled: bool = True

    @classmethod
    def from_env(cls) -> 'LangGraphConfig':
        """환경 변수에서 설정 로드"""
        config = cls()

        # LangGraph 활성화 설정
        config.langgraph_enabled = os.getenv("LANGGRAPH_ENABLED", "true").lower() == "true"

        # 체크포인트 설정
        checkpoint_storage = os.getenv("CHECKPOINT_STORAGE", "sqlite")
        if checkpoint_storage == "sqlite":
            config.checkpoint_storage = CheckpointStorageType.SQLITE
        elif checkpoint_storage == "postgres":
            config.checkpoint_storage = CheckpointStorageType.POSTGRES
        elif checkpoint_storage == "redis":
            config.checkpoint_storage = CheckpointStorageType.REDIS
        else:
            logger.warning(f"Unknown checkpoint storage type: {checkpoint_storage}, defaulting to SQLITE")

        config.checkpoint_db_path = os.getenv("LANGGRAPH_CHECKPOINT_DB", config.checkpoint_db_path)
        try:
            config.checkpoint_ttl = int(os.getenv("CHECKPOINT_TTL", config.checkpoint_ttl))
        except ValueError:
            logger.warning(f"Invalid CHECKPOINT_TTL value, using default: {config.checkpoint_ttl}")

        # 워크플로우 설정
        try:
            config.max_iterations = int(os.getenv("MAX_ITERATIONS", config.max_iterations))
        except ValueError:
            logger.warning(f"Invalid MAX_ITERATIONS value, using default: {config.max_iterations}")

        try:
            config.recursion_limit = int(os.getenv("RECURSION_LIMIT", config.recursion_limit))
        except ValueError:
            logger.warning(f"Invalid RECURSION_LIMIT value, using default: {config.recursion_limit}")

        config.enable_streaming = os.getenv("ENABLE_STREAMING", "true").lower() == "true"

        # LLM 설정
        config.llm_provider = os.getenv("LLM_PROVIDER", config.llm_provider)
        config.google_model = os.getenv("GOOGLE_MODEL", config.google_model)
        config.google_api_key = os.getenv("GOOGLE_API_KEY", config.google_api_key)

        # Ollama 설정 (백업용)
        config.ollama_base_url = os.getenv("OLLAMA_BASE_URL", config.ollama_base_url)
        config.ollama_model = os.getenv("OLLAMA_MODEL", config.ollama_model)
        try:
            config.ollama_timeout = int(os.getenv("OLLAMA_TIMEOUT", config.ollama_timeout))
        except ValueError:
            logger.warning(f"Invalid OLLAMA_TIMEOUT value, using default: {config.ollama_timeout}")

        logger.info(f"LangGraph configuration loaded: enabled={config.langgraph_enabled}")
        return config

    def validate(self) -> List[str]:
        """설정 유효성 검사"""
        errors = []

        # 필수 설정 검사
        if self.langgraph_enabled:
            if not self.checkpoint_db_path:
                errors.append("LANGGRAPH_CHECKPOINT_DB is required when LangGraph is enabled")

            if self.checkpoint_ttl <= 0:
                errors.append("CHECKPOINT_TTL must be positive")

            if self.max_iterations <= 0:
                errors.append("MAX_ITERATIONS must be positive")

            if self.recursion_limit <= 0:
                errors.append("RECURSION_LIMIT must be positive")

            if self.ollama_timeout <= 0:
                errors.append("OLLAMA_TIMEOUT must be positive")

            # LLM Provider 설정 검사
            if self.llm_provider == "google" and not self.google_api_key:
                errors.append("GOOGLE_API_KEY is required when using Google provider")

            if self.llm_provider == "ollama" and not self.ollama_model:
                errors.append("OLLAMA_MODEL is required when using Ollama provider")

        return errors

    def to_dict(self) -> dict:
        """설정을 딕셔너리로 변환"""
        return {
            "checkpoint_storage": self.checkpoint_storage.value,
            "checkpoint_db_path": self.checkpoint_db_path,
            "checkpoint_ttl": self.checkpoint_ttl,
            "max_iterations": self.max_iterations,
            "recursion_limit": self.recursion_limit,
            "enable_streaming": self.enable_streaming,
            "llm_provider": self.llm_provider,
            "google_model": self.google_model,
            "google_api_key": "***" if self.google_api_key else "",
            "ollama_base_url": self.ollama_base_url,
            "ollama_model": self.ollama_model,
            "ollama_timeout": self.ollama_timeout,
            "langgraph_enabled": self.langgraph_enabled
        }


# 전역 설정 인스턴스
try:
    langgraph_config = LangGraphConfig.from_env()

    # 설정 유효성 검사
    validation_errors = langgraph_config.validate()
    if validation_errors:
        logger.warning(f"LangGraph configuration validation errors: {validation_errors}")
    else:
        logger.info("LangGraph configuration is valid")

except Exception as e:
    logger.error(f"Failed to load LangGraph configuration: {e}")
    # 기본 설정으로 폴백
    langgraph_config = LangGraphConfig()
    logger.info("Using default LangGraph configuration")
