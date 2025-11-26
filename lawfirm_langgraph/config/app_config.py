# -*- coding: utf-8 -*-
"""
Configuration Management
환경 변수 및 설정 관리
"""

import os
import sys
from pathlib import Path
from typing import Any, Optional

from pydantic import Field, ConfigDict, field_validator
from pydantic_settings import BaseSettings
from lawfirm_langgraph.core.shared.utils.environment import Environment

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

    # Environment Configuration
    environment: Environment = Field(
        default=Environment.DEVELOPMENT,
        env="ENVIRONMENT"
    )

    # LAW OPEN API Configuration
    law_open_api_oc: str = Field(default="{OC}", env="LAW_OPEN_API_OC")
    law_firm_ai_api_key: str = Field(default="your-api-key-here", env="LAW_FIRM_AI_API_KEY")

    # API Configuration
    api_host: str = Field(default="0.0.0.0", env="API_HOST")
    api_port: int = Field(default=8000, env="API_PORT")
    debug: bool = Field(default=False, env="DEBUG")

    # Database Configuration
    database_url: str = Field(
        default="",
        env="DATABASE_URL",
        description="PostgreSQL 연결 URL (예: postgresql://user:password@host:port/database). DATABASE_URL이 없으면 POSTGRES_* 환경 변수로 자동 구성"
    )
    # PostgreSQL 개별 설정 (DATABASE_URL이 없을 때 사용)
    postgres_host: str = Field(default="localhost", env="POSTGRES_HOST")
    postgres_port: int = Field(default=5432, env="POSTGRES_PORT")
    postgres_db: str = Field(default="lawfirmai_local", env="POSTGRES_DB")
    postgres_user: str = Field(default="lawfirmai", env="POSTGRES_USER")
    postgres_password: str = Field(default="local_password", env="POSTGRES_PASSWORD")
    # 하위 호환성을 위한 database_path (사용 안 함, deprecated)
    # DATABASE_PATH는 더 이상 사용하지 않습니다. PostgreSQL을 사용하세요.
    database_path: Optional[str] = Field(default=None, env="DATABASE_PATH")
    
    # Database Connection Pool Configuration
    db_pool_min_size: int = Field(
        default=1,
        env="DB_POOL_MIN_SIZE",
        description="데이터베이스 연결 풀 최소 크기"
    )
    db_pool_max_size: int = Field(
        default=20,
        env="DB_POOL_MAX_SIZE",
        description="데이터베이스 연결 풀 최대 크기 (기본값: 20, 연결 풀 고갈 방지)"
    )
    
    # Vector Search Configuration
    vector_search_method: str = Field(
        default="pgvector",
        env="VECTOR_SEARCH_METHOD",
        description="벡터 검색 방법: 'pgvector' (PostgreSQL pgvector), 'faiss' (FAISS 인덱스), 'hybrid' (하이브리드)"
    )
    faiss_index_path: Optional[str] = Field(
        default=None,
        env="FAISS_INDEX_PATH",
        description="FAISS 인덱스 파일 경로 (faiss 방법 사용 시)"
    )

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

    # Performance Configuration
    max_workers: int = Field(default=4, env="MAX_WORKERS")
    request_timeout: int = Field(default=30, env="REQUEST_TIMEOUT")
    cache_ttl: int = Field(default=3600, env="CACHE_TTL")
    request_max_retries: int = Field(default=3, env="REQUEST_MAX_RETRIES")
    request_backoff_base: float = Field(default=0.8, env="REQUEST_BACKOFF_BASE")
    request_backoff_max: float = Field(default=8.0, env="REQUEST_BACKOFF_MAX")

    # Security Configuration
    secret_key: str = Field(default="your-secret-key-here", env="SECRET_KEY")
    cors_origins: list = Field(
        default=["http://localhost:3000", "http://localhost:7860"]
    )

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

    def is_local(self) -> bool:
        """로컬 환경 여부"""
        return self.environment.is_local()
    
    def is_development(self) -> bool:
        """개발 환경 여부"""
        return self.environment.is_development()

    def is_production(self) -> bool:
        """프로덕션 환경 여부"""
        return self.environment.is_production()

    def __init__(self, **kwargs):
        """초기화 시 환경변수 파일 로딩"""
        # 환경변수 파일 로딩
        self.Settings._load_env_file(".env")
        
        # cors_origins 환경 변수 미리 처리 (Pydantic이 파싱하기 전에 JSON으로 변환)
        cors_origins_env = os.getenv("CORS_ORIGINS", "").strip()
        if cors_origins_env:
            try:
                import json
                # JSON 배열 형식이 아니면 JSON 배열로 변환
                if not (cors_origins_env.startswith('[') and cors_origins_env.endswith(']')):
                    # 쉼표로 구분된 문자열인 경우 JSON 배열로 변환
                    origins = [origin.strip() for origin in cors_origins_env.split(',') if origin.strip()]
                    if origins:
                        os.environ["CORS_ORIGINS"] = json.dumps(origins)
            except (json.JSONDecodeError, ValueError):
                # 파싱 실패 시 환경 변수 제거하여 기본값 사용
                os.environ.pop("CORS_ORIGINS", None)
        else:
            # 환경 변수가 없으면 제거하여 기본값 사용
            os.environ.pop("CORS_ORIGINS", None)
        
        # ENVIRONMENT 환경 변수에서 자동 설정
        if "environment" not in kwargs:
            kwargs["environment"] = Environment.get_current()
        
        # 환경에 따른 자동 설정
        if kwargs.get("environment", Environment.DEVELOPMENT).is_debug_enabled():
            if "debug" not in kwargs:
                kwargs["debug"] = True
        
        # cors_origins 환경 변수 미리 처리 (Pydantic V2 호환)
        # 환경 변수를 JSON 문자열로 변환하여 Pydantic이 파싱할 수 있도록 함
        if "cors_origins" not in kwargs:
            cors_origins_env = os.getenv("CORS_ORIGINS", "").strip()
            if cors_origins_env:
                try:
                    import json
                    # JSON 배열 형식인 경우
                    if cors_origins_env.startswith('[') and cors_origins_env.endswith(']'):
                        # 이미 JSON 형식이면 그대로 사용
                        parsed = json.loads(cors_origins_env)
                        if isinstance(parsed, list):
                            kwargs["cors_origins"] = [str(origin).strip() for origin in parsed if origin]
                    else:
                        # 쉼표로 구분된 문자열인 경우 JSON 배열로 변환
                        origins = [origin.strip() for origin in cors_origins_env.split(',') if origin.strip()]
                        if origins:
                            kwargs["cors_origins"] = origins
                except (json.JSONDecodeError, ValueError):
                    # 파싱 실패 시 기본값 사용 (kwargs에 설정하지 않음)
                    pass
            else:
                # 환경 변수가 없으면 환경 변수에서 제거하여 기본값 사용
                os.environ.pop("CORS_ORIGINS", None)
        
        super().__init__(**kwargs)
        
        # DATABASE_PATH가 설정되어 있으면 경고 (더 이상 사용하지 않음)
        if self.database_path:
            import warnings
            import logging
            logger = logging.getLogger(__name__)
            warnings.warn(
                "DATABASE_PATH is deprecated and will be ignored. "
                "Use DATABASE_URL or POSTGRES_* environment variables instead. "
                "SQLite is no longer supported. Please use PostgreSQL.",
                DeprecationWarning,
                stacklevel=2
            )
            logger.warning(
                "DATABASE_PATH is deprecated. It will be ignored. "
                "Use DATABASE_URL or POSTGRES_* environment variables instead."
            )
            # DATABASE_PATH 무시
            self.database_path = None
        
        # SQLite URL이 설정되어 있으면 무시하고 PostgreSQL 환경변수로 조합
        if self.database_url and self.database_url.startswith("sqlite://"):
            import warnings
            import logging
            logger = logging.getLogger(__name__)
            warnings.warn(
                f"SQLite URL detected and will be ignored: {self.database_url}. "
                "SQLite is no longer supported. Using PostgreSQL configuration from POSTGRES_* environment variables.",
                DeprecationWarning,
                stacklevel=2
            )
            logger.warning(
                f"SQLite URL detected and will be ignored: {self.database_url}. "
                "Using PostgreSQL configuration from POSTGRES_* environment variables."
            )
            # SQLite URL 무시
            self.database_url = ""
        
        # database_url이 없거나 비어있는 경우 환경 변수로 조합
        if not self.database_url or self.database_url.strip() == "":
            from urllib.parse import quote_plus
            # POSTGRES_* 환경 변수로부터 database_url 구성
            # 프로젝트 루트 .env 파일의 설정을 우선 사용 (21-29줄)
            host = self.postgres_host
            port = self.postgres_port
            db = self.postgres_db
            user = self.postgres_user
            password = quote_plus(self.postgres_password)  # URL 인코딩
            
            self.database_url = f"postgresql://{user}:{password}@{host}:{port}/{db}"
            import logging
            logger = logging.getLogger(__name__)
            logger.info(f"Database URL generated from POSTGRES_* environment variables: postgresql://{user}:***@{host}:{port}/{db}")
        
        # 최종 검증: SQLite URL이 여전히 있으면 에러
        if self.database_url and self.database_url.startswith("sqlite://"):
            import logging
            logger = logging.getLogger(__name__)
            logger.error(f"SQLite URL detected: {self.database_url}")
            logger.error(f"   Current DATABASE_URL: {self.database_url}")
            logger.error(f"   Current DATABASE_PATH: {self.database_path}")
            logger.error(f"   POSTGRES_HOST: {os.getenv('POSTGRES_HOST', 'NOT SET')}")
            logger.error(f"   POSTGRES_DB: {os.getenv('POSTGRES_DB', 'NOT SET')}")
            raise ValueError(
                "SQLite is no longer supported. Please use PostgreSQL. "
                "Set DATABASE_URL to a PostgreSQL URL (e.g., postgresql://user:password@host:port/database) "
                "or configure POSTGRES_* environment variables in .env file (lines 21-29)."
            )
