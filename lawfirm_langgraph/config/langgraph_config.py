# -*- coding: utf-8 -*-
"""
LangGraph Configuration
LangGraph 설정 관리 모듈
"""

import logging
try:
    from lawfirm_langgraph.core.utils.logger import get_logger
except ImportError:
    from core.utils.logger import get_logger
import os
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import List

from dotenv import load_dotenv

# 환경 변수 로드 (중앙 집중식 로더 사용)
# 프로젝트 루트의 .env 파일을 우선적으로 로드
try:
    # 프로젝트 루트 찾기: lawfirm_langgraph/config/ -> lawfirm_langgraph/ -> 프로젝트 루트
    _langgraph_dir = Path(__file__).parent.parent
    _project_root = _langgraph_dir.parent
    
    # 공통 로더 사용 (프로젝트 루트 .env 파일 우선 로드)
    try:
        import sys
        if str(_project_root) not in sys.path:
            sys.path.insert(0, str(_project_root))
        from utils.env_loader import ensure_env_loaded, load_all_env_files
        
        # 프로젝트 루트 .env 파일 명시적으로 로드
        ensure_env_loaded(_project_root)
        loaded_files = load_all_env_files(_project_root)
        # 환경 변수 로드 완료 메시지는 출력하지 않음 (run_query_test.py에서 불필요한 메시지 방지)
    except ImportError:
        # 공통 로더가 없으면 직접 로드 (프로젝트 루트 .env 우선)
        root_env = _project_root / ".env"
        langgraph_env = _langgraph_dir / ".env"
        
        # 프로젝트 루트 .env 먼저 로드
        if root_env.exists():
            load_dotenv(dotenv_path=str(root_env), override=False)
        
        # lawfirm_langgraph/.env 로드 (덮어쓰기)
        if langgraph_env.exists():
            load_dotenv(dotenv_path=str(langgraph_env), override=True)
except Exception as e:
    # 모든 방법이 실패하면 기존 방식으로 fallback
    _langgraph_dir = Path(__file__).parent.parent
    _project_root = _langgraph_dir.parent
    
    # 프로젝트 루트 .env 시도
    root_env = _project_root / ".env"
    if root_env.exists():
        try:
            load_dotenv(dotenv_path=str(root_env), override=False)
        except Exception:
            pass
    
    # lawfirm_langgraph/.env 시도
    _env_file = _langgraph_dir / ".env"
    if _env_file.exists():
        try:
            load_dotenv(dotenv_path=str(_env_file), override=True)
        except Exception:
            pass

logger = get_logger(__name__)


class CheckpointStorageType(Enum):
    """체크포인트 저장소 타입"""
    MEMORY = "memory"  # MemorySaver 사용 (기본값, 개발용)
    POSTGRES = "postgres"  # PostgresSaver 사용 (프로덕션용)
    REDIS = "redis"
    DISABLED = "disabled"  # 체크포인터 비활성화


@dataclass
class LangGraphConfig:
    """LangGraph 설정 클래스"""

    # 체크포인트 설정
    enable_checkpoint: bool = True  # 체크포인터 활성화 여부
    checkpoint_storage: CheckpointStorageType = CheckpointStorageType.POSTGRES  # 기본값: PostgresSaver (로컬/개발/프로덕션 모두 지원)
    checkpoint_ttl: int = 3600  # 1시간

    # 워크플로우 설정
    max_iterations: int = 10
    recursion_limit: int = 25
    enable_streaming: bool = True

    # LLM 설정 (Google Gemini 우선)
    llm_provider: str = "google"
    google_model: str = "gemini-2.5-flash-lite"  # .env 파일의 설정에 맞게 변경
    google_api_key: str = ""


    # LangGraph 활성화 설정
    langgraph_enabled: bool = True

    # LangSmith 설정 (LangChain 모니터링)
    langsmith_enabled: bool = False
    langsmith_endpoint: str = "https://api.smith.langchain.com"
    langsmith_api_key: str = ""
    langsmith_project: str = "LawFirmAI"
    langsmith_tracing: bool = True

    # RAG 품질 제어 설정
    similarity_threshold: float = 0.75  # 문서 유사도 임계값 (높은 관련성 결과만 반환)
    max_context_length: int = 4000  # 최대 컨텍스트 길이 (문자)
    max_tokens: int = 2000  # 최대 토큰 수
    
    # 성능 모니터링 설정
    slow_node_threshold: float = 10.0  # 느린 노드 경고 임계값 (초) - LLM 호출 노드는 더 오래 걸릴 수 있음
    slow_llm_node_threshold: float = 15.0  # LLM 호출 노드 경고 임계값 (초)
    slow_search_node_threshold: float = 30.0  # 검색/처리 노드 경고 임계값 (초)
    
    # 캐시 설정
    disable_query_cache: bool = False  # 쿼리 최적화 캐시 비활성화 (환경 변수 DISABLE_QUERY_CACHE로 설정 가능)
    disable_llm_enhancement_cache: bool = False  # LLM 쿼리 확장 캐시 비활성화 (환경 변수 DISABLE_LLM_ENHANCEMENT_CACHE로 설정 가능)
    
    # 재랭킹 가중치 설정 (Cross-Encoder 재랭킹)
    rerank_original_weight: float = 0.6  # 기존 점수 가중치 (환경 변수 RERANK_ORIGINAL_WEIGHT로 설정 가능, 기본값: 0.6)
    rerank_cross_encoder_weight: float = 0.4  # Cross-Encoder 점수 가중치 (환경 변수 RERANK_CROSS_ENCODER_WEIGHT로 설정 가능, 기본값: 0.4)
    
    # Embedding Model 설정 (환경 변수 EMBEDDING_MODEL에서 읽음, 없으면 기본값)
    embedding_model: str = ""  # from_env()에서 환경 변수로 설정됨
    # 사용 가능한 법률 특화 모델:
    # - "woong0322/ko-legal-sbert-finetuned" (법률 특화 모델)
    # - "snunlp/KR-SBERT-V40K-klueNLI-augSTS" (기본 한국어 SBERT)
    # - "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2" (다국어 모델)

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
        config.enable_checkpoint = os.getenv("ENABLE_CHECKPOINT", "true").lower() == "true"
        
        checkpoint_storage = os.getenv("CHECKPOINT_STORAGE", "postgres")
        if checkpoint_storage == "memory":
            config.checkpoint_storage = CheckpointStorageType.MEMORY
        elif checkpoint_storage == "postgres":
            config.checkpoint_storage = CheckpointStorageType.POSTGRES
        elif checkpoint_storage == "redis":
            config.checkpoint_storage = CheckpointStorageType.REDIS
        elif checkpoint_storage == "disabled":
            config.checkpoint_storage = CheckpointStorageType.DISABLED
            config.enable_checkpoint = False
        else:
            # 알 수 없는 타입은 memory로 폴백
            logger.warning(f"Unknown checkpoint storage type: {checkpoint_storage}, using memory")
            config.checkpoint_storage = CheckpointStorageType.MEMORY

        # PostgreSQL은 DATABASE_URL 또는 CHECKPOINT_DATABASE_URL을 사용
        config.checkpoint_ttl = int(os.getenv("CHECKPOINT_TTL", config.checkpoint_ttl))

        # 워크플로우 설정
        config.max_iterations = int(os.getenv("MAX_ITERATIONS", config.max_iterations))
        config.recursion_limit = int(os.getenv("RECURSION_LIMIT", config.recursion_limit))
        config.enable_streaming = os.getenv("ENABLE_STREAMING", "true").lower() == "true"

        # LLM 설정
        config.llm_provider = os.getenv("LLM_PROVIDER", config.llm_provider)
        config.google_model = os.getenv("GOOGLE_MODEL", config.google_model)
        config.google_api_key = os.getenv("GOOGLE_API_KEY", config.google_api_key)


        # Embedding Model 설정 (환경 변수에서 읽음)
        embedding_model_env = os.getenv("EMBEDDING_MODEL")
        if embedding_model_env:
            config.embedding_model = embedding_model_env.strip().strip('"').strip("'")
        else:
            # 환경 변수가 없으면 Config 클래스에서 가져오기
            try:
                from lawfirm_langgraph.core.utils.config import Config
                core_config = Config()
                config.embedding_model = core_config.embedding_model
            except Exception:
                # 최후의 수단: 기본값 사용
                config.embedding_model = "snunlp/KR-SBERT-V40K-klueNLI-augSTS"
        
        # LangSmith 설정 (LangSmith 환경 변수 사용, 하위 호환성을 위해 LANGCHAIN_*도 지원)
        config.langsmith_enabled = (
            os.getenv("LANGSMITH_TRACING", "false").lower() == "true" or
            os.getenv("LANGCHAIN_TRACING_V2", "false").lower() == "true"
        )
        config.langsmith_endpoint = (
            os.getenv("LANGSMITH_ENDPOINT") or 
            os.getenv("LANGCHAIN_ENDPOINT") or 
            "https://api.smith.langchain.com"
        )
        config.langsmith_api_key = (
            os.getenv("LANGSMITH_API_KEY") or 
            os.getenv("LANGCHAIN_API_KEY") or 
            config.langsmith_api_key
        )
        config.langsmith_project = (
            os.getenv("LANGSMITH_PROJECT") or 
            os.getenv("LANGCHAIN_PROJECT") or 
            config.langsmith_project
        )
        config.langsmith_tracing = (
            os.getenv("LANGSMITH_TRACING", "true").lower() == "true" or
            os.getenv("LANGCHAIN_TRACING", "true").lower() == "true"
        )

        # RAG 품질 제어 설정
        config.similarity_threshold = float(os.getenv("SIMILARITY_THRESHOLD", config.similarity_threshold))
        config.max_context_length = int(os.getenv("MAX_CONTEXT_LENGTH", config.max_context_length))
        config.max_tokens = int(os.getenv("MAX_TOKENS", config.max_tokens))
        
        # 성능 모니터링 설정
        config.slow_node_threshold = float(os.getenv("SLOW_NODE_THRESHOLD", config.slow_node_threshold))
        config.slow_llm_node_threshold = float(os.getenv("SLOW_LLM_NODE_THRESHOLD", config.slow_llm_node_threshold))
        config.slow_search_node_threshold = float(os.getenv("SLOW_SEARCH_NODE_THRESHOLD", config.slow_search_node_threshold))
        
        # 캐시 설정
        config.disable_query_cache = os.getenv("DISABLE_QUERY_CACHE", "false").lower() == "true"
        config.disable_llm_enhancement_cache = os.getenv("DISABLE_LLM_ENHANCEMENT_CACHE", "false").lower() == "true"
        
        # Embedding Model 설정은 이미 from_env() 시작 부분에서 처리됨

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
        
        # 재랭킹 가중치 설정
        config.rerank_original_weight = float(os.getenv("RERANK_ORIGINAL_WEIGHT", str(config.rerank_original_weight)))
        config.rerank_cross_encoder_weight = float(os.getenv("RERANK_CROSS_ENCODER_WEIGHT", str(config.rerank_cross_encoder_weight)))
        
        # 가중치 합이 1.0이 되도록 정규화
        total_weight = config.rerank_original_weight + config.rerank_cross_encoder_weight
        if total_weight > 0:
            config.rerank_original_weight = config.rerank_original_weight / total_weight
            config.rerank_cross_encoder_weight = config.rerank_cross_encoder_weight / total_weight
        else:
            # 잘못된 값이면 기본값으로 재설정
            config.rerank_original_weight = 0.6
            config.rerank_cross_encoder_weight = 0.4

        logger.info(f"LangGraph configuration loaded: enabled={config.langgraph_enabled}, langsmith_enabled={config.langsmith_enabled}, use_agentic_mode={config.use_agentic_mode}, rerank_weights: original={config.rerank_original_weight:.2f}, cross_encoder={config.rerank_cross_encoder_weight:.2f}")
        return config

    def validate(self) -> List[str]:
        """설정 유효성 검사"""
        errors = []

        # 필수 설정 검사
        if self.langgraph_enabled:
            # PostgreSQL checkpoint는 DATABASE_URL 또는 CHECKPOINT_DATABASE_URL을 사용
            # checkpoint_db_path는 더 이상 사용하지 않음

            if self.checkpoint_ttl <= 0:
                errors.append("CHECKPOINT_TTL must be positive")

            if self.max_iterations <= 0:
                errors.append("MAX_ITERATIONS must be positive")

            if self.recursion_limit <= 0:
                errors.append("RECURSION_LIMIT must be positive")


        return errors

    def to_dict(self) -> dict:
        """설정을 딕셔너리로 변환"""
        return {
            "enable_checkpoint": self.enable_checkpoint,
            "checkpoint_storage": self.checkpoint_storage.value,
            "checkpoint_ttl": self.checkpoint_ttl,
            "max_iterations": self.max_iterations,
            "recursion_limit": self.recursion_limit,
            "enable_streaming": self.enable_streaming,
            "langgraph_enabled": self.langgraph_enabled,
            "langfuse_enabled": self.langfuse_enabled,
            "langsmith_enabled": self.langsmith_enabled,
            "langsmith_endpoint": self.langsmith_endpoint,
            "langsmith_project": self.langsmith_project,
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

