# -*- coding: utf-8 -*-
"""
Configuration Management
환경 변수 및 설정 관리
"""

import os
import sys
from pathlib import Path
from typing import Any, Optional

from pydantic import Field, ConfigDict
from pydantic_settings import BaseSettings

# 한글 출력을 위한 인코딩 설정
if sys.platform == "win32":
    # Windows에서 한글 출력을 위한 설정
    try:
        import codecs
        # Windows 콘솔에서 UTF-8 지원 확인 후 설정
        if hasattr(sys.stdout, 'detach'):
            sys.stdout = codecs.getwriter('utf-8')(sys.stdout.detach())
            sys.stderr = codecs.getwriter('utf-8')(sys.stderr.detach())
        else:
            # detach()가 지원되지 않는 경우 다른 방법 사용
            import io
            sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
            sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8')
    except Exception:
        # 인코딩 설정 실패 시 기본 설정 유지
        pass

    # 환경변수 인코딩 설정
    os.environ['PYTHONIOENCODING'] = 'utf-8'


class Config(BaseSettings):
    """설정 관리 클래스"""
    
    model_config = ConfigDict(protected_namespaces=('settings_',))

    # LAW OPEN API Configuration
    law_open_api_oc: str = Field(default="{OC}", env="LAW_OPEN_API_OC")
    law_firm_ai_api_key: str = Field(default="your-api-key-here", env="LAW_FIRM_AI_API_KEY")

    # API Configuration
    api_host: str = Field(default="0.0.0.0", env="API_HOST")
    api_port: int = Field(default=8000, env="API_PORT")
    debug: bool = Field(default=False, env="DEBUG")

    # Database Configuration
    database_url: str = Field(default="sqlite:///./data/lawfirm_v2.db", env="DATABASE_URL")
    database_path: str = Field(default="./data/lawfirm_v2.db", env="DATABASE_PATH")

    # Model Configuration
    model_path: str = Field(default="./models", env="MODEL_PATH")
    device: str = Field(default="cpu", env="DEVICE")
    model_cache_dir: str = Field(default="./model_cache", env="MODEL_CACHE_DIR")

    # Vector Store Configuration
    chroma_db_path: str = Field(default="./data/chroma_db", env="CHROMA_DB_PATH")
    embedding_model: str = Field(
        default="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
        env="EMBEDDING_MODEL"
    )
    embedding_dimension: int = Field(default=768, env="EMBEDDING_DIMENSION")
    # Versioned corpus/model routing
    active_corpus_version: str = Field(default="v1", env="ACTIVE_CORPUS_VERSION")
    active_model_version: str = Field(default="default@1.0", env="ACTIVE_MODEL_VERSION")
    embeddings_base_dir: str = Field(default="./data/embeddings", env="EMBEDDINGS_BASE_DIR")
    versioned_database_dir: str = Field(default="./data/database", env="VERSIONED_DATABASE_DIR")

    # Logging Configuration
    log_level: str = Field(default="INFO", env="LOG_LEVEL")
    log_file: str = Field(default="./logs/lawfirm_ai.log", env="LOG_FILE")
    log_format: str = Field(default="json", env="LOG_FORMAT")

    # HuggingFace Spaces Configuration
    hf_space_id: Optional[str] = Field(default=None, env="HF_SPACE_ID")
    hf_token: Optional[str] = Field(default=None, env="HF_TOKEN")

    # Performance Configuration
    max_workers: int = Field(default=4, env="MAX_WORKERS")
    request_timeout: int = Field(default=30, env="REQUEST_TIMEOUT")
    cache_ttl: int = Field(default=3600, env="CACHE_TTL")
    request_max_retries: int = Field(default=3, env="REQUEST_MAX_RETRIES")
    request_backoff_base: float = Field(default=0.8, env="REQUEST_BACKOFF_BASE")
    request_backoff_max: float = Field(default=8.0, env="REQUEST_BACKOFF_MAX")

    # Security Configuration
    secret_key: str = Field(default="your-secret-key-here", env="SECRET_KEY")
    cors_origins: list = Field(default=["http://localhost:3000", "http://localhost:7860"], env="CORS_ORIGINS")

    # Monitoring Configuration
    enable_metrics: bool = Field(default=True, env="ENABLE_METRICS")
    metrics_port: int = Field(default=9090, env="METRICS_PORT")

    class Settings:
        env_file = ".env"
        env_file_encoding = "utf-8"
        case_sensitive = False
        extra = "ignore"  # 추가 필드 무시

        @classmethod
        def _load_env_file(cls, env_file: str) -> None:
            """환경변수 파일 로딩"""
            env_path = Path(env_file)
            if env_path.exists():
                try:
                    with open(env_path, 'r', encoding='utf-8') as f:
                        for line in f:
                            line = line.strip()
                            if line and not line.startswith('#') and '=' in line:
                                key, value = line.split('=', 1)
                                os.environ[key.strip()] = value.strip()
                except Exception as e:
                    print(f"환경변수 파일 로딩 실패: {e}")
            else:
                # .env 파일이 없으면 환경변수에서 직접 설정
                if not os.getenv("LAW_OPEN_API_OC"):
                    print("⚠️ LAW_OPEN_API_OC 환경변수가 설정되지 않았습니다.")
                    print("다음 중 하나의 방법으로 설정하세요:")
                    print("1. .env 파일 생성: LAW_OPEN_API_OC=your_email@example.com")
                    print("2. 환경변수 직접 설정: set LAW_OPEN_API_OC=your_email@example.com")
                    print("3. PowerShell에서: $env:LAW_OPEN_API_OC='your_email@example.com'")

    def get(self, key: str, default: Any = None) -> Any:
        """설정 값 조회"""
        return getattr(self, key, default)

    def is_development(self) -> bool:
        """개발 환경 여부"""
        return self.debug

    def is_production(self) -> bool:
        """프로덕션 환경 여부"""
        return not self.debug

    def __init__(self, **kwargs):
        """초기화 시 환경변수 파일 로딩"""
        # 환경변수 파일 로딩
        self.Settings._load_env_file(".env")
        super().__init__(**kwargs)
