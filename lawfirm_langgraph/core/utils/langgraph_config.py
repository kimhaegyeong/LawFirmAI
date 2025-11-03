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

    # Langfuse 설정 (답변 품질 추적)
    langfuse_enabled: bool = False
    langfuse_secret_key: str = ""
    langfuse_public_key: str = ""
    langfuse_host: str = "https://cloud.langfuse.com"
    langfuse_debug: bool = False

    # LangSmith 설정 (LangChain 모니터링)
    langsmith_enabled: bool = False
    langsmith_api_key: str = ""
    langsmith_project: str = "LawFirmAI"
    langsmith_tracing: bool = True

    # RAG 품질 제어 설정
    similarity_threshold: float = 0.5  # 문서 유사도 임계값 (V2 벡터 검색에 최적화, 0.9는 너무 높아 0.5로 조정)
    max_context_length: int = 4000  # 최대 컨텍스트 길이 (문자)
    max_tokens: int = 2000  # 최대 토큰 수

    # 통계 관리 설정
    enable_statistics: bool = True
    stats_update_alpha: float = 0.1  # 이동 평균 업데이트 계수

    # State optimization settings (for reducing memory usage)
    max_retrieved_docs: int = 10
    max_document_content_length: int = 500
    max_conversation_history: int = 5
    max_processing_steps: int = 20
    enable_state_pruning: bool = True

    # LLM 기반 복잡도 분류 설정
    use_llm_for_complexity: bool = True  # LLM 기반 복잡도 분류 활성화
    complexity_llm_model: str = "fast"  # "fast" | "main" | "disabled"
    
    # Agentic AI 모드 설정
    use_agentic_mode: bool = False  # Agentic AI 모드 활성화 (Tool Use/Function Calling)

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

        config.checkpoint_db_path = os.getenv("LANGGRAPH_CHECKPOINT_DB", config.checkpoint_db_path)
        config.checkpoint_ttl = int(os.getenv("CHECKPOINT_TTL", config.checkpoint_ttl))

        # 워크플로우 설정
        config.max_iterations = int(os.getenv("MAX_ITERATIONS", config.max_iterations))
        config.recursion_limit = int(os.getenv("RECURSION_LIMIT", config.recursion_limit))
        config.enable_streaming = os.getenv("ENABLE_STREAMING", "true").lower() == "true"

        # LLM 설정
        config.llm_provider = os.getenv("LLM_PROVIDER", config.llm_provider)
        config.google_model = os.getenv("GOOGLE_MODEL", config.google_model)
        config.google_api_key = os.getenv("GOOGLE_API_KEY", config.google_api_key)

        # Ollama 설정 (백업용)
        config.ollama_base_url = os.getenv("OLLAMA_BASE_URL", config.ollama_base_url)
        config.ollama_model = os.getenv("OLLAMA_MODEL", config.ollama_model)
        config.ollama_timeout = int(os.getenv("OLLAMA_TIMEOUT", config.ollama_timeout))

        # Langfuse 설정
        config.langfuse_enabled = os.getenv("LANGFUSE_ENABLED", "false").lower() == "true"
        config.langfuse_secret_key = os.getenv("LANGFUSE_SECRET_KEY", config.langfuse_secret_key)
        config.langfuse_public_key = os.getenv("LANGFUSE_PUBLIC_KEY", config.langfuse_public_key)
        config.langfuse_host = os.getenv("LANGFUSE_HOST", config.langfuse_host)
        config.langfuse_debug = os.getenv("LANGFUSE_DEBUG", "false").lower() == "true"

        # LangSmith 설정 (LangChain 환경 변수 사용)
        config.langsmith_enabled = os.getenv("LANGCHAIN_TRACING_V2", "false").lower() == "true"
        config.langsmith_api_key = os.getenv("LANGCHAIN_API_KEY", config.langsmith_api_key)
        config.langsmith_project = os.getenv("LANGCHAIN_PROJECT", config.langsmith_project)
        config.langsmith_tracing = os.getenv("LANGCHAIN_TRACING", "true").lower() == "true"

        # RAG 품질 제어 설정
        config.similarity_threshold = float(os.getenv("SIMILARITY_THRESHOLD", config.similarity_threshold))
        config.max_context_length = int(os.getenv("MAX_CONTEXT_LENGTH", config.max_context_length))
        config.max_tokens = int(os.getenv("MAX_TOKENS", config.max_tokens))

        # 통계 관리 설정
        config.enable_statistics = os.getenv("ENABLE_STATISTICS", "true").lower() == "true"
        config.stats_update_alpha = float(os.getenv("STATS_UPDATE_ALPHA", config.stats_update_alpha))

        # State optimization settings
        config.max_retrieved_docs = int(os.getenv("MAX_RETRIEVED_DOCS", config.max_retrieved_docs))
        config.max_document_content_length = int(os.getenv("MAX_DOCUMENT_CONTENT_LENGTH", config.max_document_content_length))
        config.max_conversation_history = int(os.getenv("MAX_CONVERSATION_HISTORY", config.max_conversation_history))
        config.max_processing_steps = int(os.getenv("MAX_PROCESSING_STEPS", config.max_processing_steps))
        config.enable_state_pruning = os.getenv("ENABLE_STATE_PRUNING", "true").lower() == "true"

        # LLM 기반 복잡도 분류 설정
        config.use_llm_for_complexity = os.getenv("USE_LLM_FOR_COMPLEXITY", "true").lower() == "true"
        config.complexity_llm_model = os.getenv("COMPLEXITY_LLM_MODEL", "fast")
        
        # Agentic AI 모드 설정
        config.use_agentic_mode = os.getenv("USE_AGENTIC_MODE", "false").lower() == "true"

        logger.info(f"LangGraph configuration loaded: enabled={config.langgraph_enabled}, langfuse_enabled={config.langfuse_enabled}, langsmith_enabled={config.langsmith_enabled}, use_agentic_mode={config.use_agentic_mode}")
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
            "ollama_base_url": self.ollama_base_url,
            "ollama_model": self.ollama_model,
            "ollama_timeout": self.ollama_timeout,
            "langgraph_enabled": self.langgraph_enabled,
            "langfuse_enabled": self.langfuse_enabled,
            "langsmith_enabled": self.langsmith_enabled,
            "similarity_threshold": self.similarity_threshold,
            "max_context_length": self.max_context_length,
            "max_tokens": self.max_tokens,
            "enable_statistics": self.enable_statistics,
            "max_retrieved_docs": self.max_retrieved_docs,
            "max_document_content_length": self.max_document_content_length,
            "max_conversation_history": self.max_conversation_history,
            "max_processing_steps": self.max_processing_steps,
            "enable_state_pruning": self.enable_state_pruning
        }


# 전역 설정 인스턴스
langgraph_config = LangGraphConfig.from_env()
