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

# 환경 변수 로드 (프로젝트 루트 .env 파일 사용)
# core/utils/config.py -> core/utils/ -> core/ -> lawfirm_langgraph/ -> 프로젝트 루트
try:
    _config_file_dir = Path(__file__).parent
    _core_dir = _config_file_dir.parent
    _langgraph_dir = _core_dir.parent
    _project_root = _langgraph_dir.parent
    
    # 공통 로더 사용 (프로젝트 루트 .env 파일 우선 로드)
    try:
        if str(_project_root) not in sys.path:
            sys.path.insert(0, str(_project_root))
        from utils.env_loader import ensure_env_loaded, load_all_env_files
        
        # 프로젝트 루트 .env 파일 명시적으로 로드
        ensure_env_loaded(_project_root)
        loaded_files = load_all_env_files(_project_root)
        if loaded_files:
            import logging
            logger = logging.getLogger(__name__)
            logger.debug(f"✅ Config: 환경 변수 로드 완료 ({len(loaded_files)}개 .env 파일)")
    except ImportError:
        # 공통 로더가 없으면 직접 로드 (프로젝트 루트 .env 우선)
        try:
            from dotenv import load_dotenv
            
            # 프로젝트 루트 .env 먼저 로드
            root_env = _project_root / ".env"
            if root_env.exists():
                load_dotenv(dotenv_path=str(root_env), override=False)
            
            # lawfirm_langgraph/.env 로드 (덮어쓰기)
            langgraph_env = _langgraph_dir / ".env"
            if langgraph_env.exists():
                load_dotenv(dotenv_path=str(langgraph_env), override=True)
        except ImportError:
            pass  # python-dotenv가 없으면 환경 변수만 사용
except Exception:
    pass  # 환경 변수 로드 실패 시 기본값 사용

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


# 환경 변수 경고 메시지 중복 출력 방지
_warned_env_vars = set()


class Config(BaseSettings):
    """
    설정 관리 클래스 (레거시)
    
    ⚠️ DEPRECATED: 이 클래스는 레거시입니다.
    새로운 코드에서는 `lawfirm_langgraph.config.app_config.Config`를 사용하세요.
    
    이 클래스는 하위 호환성을 위해 유지되지만, SQLite는 더 이상 지원하지 않습니다.
    PostgreSQL만 지원합니다.
    """
    
    model_config = ConfigDict(protected_namespaces=('settings_',))

    # LAW OPEN API Configuration
    law_open_api_oc: str = Field(default="{OC}", env="LAW_OPEN_API_OC")
    law_firm_ai_api_key: str = Field(default="your-api-key-here", env="LAW_FIRM_AI_API_KEY")

    # API Configuration
    api_host: str = Field(default="0.0.0.0", env="API_HOST")
    api_port: int = Field(default=8000, env="API_PORT")
    debug: bool = Field(default=False, env="DEBUG")

    # Database Configuration
    # 환경변수 DATABASE_PATH와 DATABASE_URL에서 읽음 (.env 파일 또는 시스템 환경변수)
    database_url: Optional[str] = Field(default=None, env="DATABASE_URL")
    database_path: Optional[str] = Field(default=None, env="DATABASE_PATH")

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
    # MLflow Configuration
    mlflow_tracking_uri: Optional[str] = Field(default=None, env="MLFLOW_TRACKING_URI")
    mlflow_run_id: Optional[str] = Field(default=None, env="MLFLOW_RUN_ID")
    mlflow_experiment_name: str = Field(default="faiss_index_versions", env="MLFLOW_EXPERIMENT_NAME")
    use_mlflow_index: bool = Field(default=True, env="USE_MLFLOW_INDEX")
    
    # Optimized Search Parameters Configuration
    optimized_search_params_path: Optional[str] = Field(
        default="data/ml_config/optimized_search_params.json",
        env="OPTIMIZED_SEARCH_PARAMS_PATH"
    )
    
    # Search Quality Improvement Features
    enable_search_improvements: bool = Field(default=True, env="ENABLE_SEARCH_IMPROVEMENTS")
    use_adaptive_weights: bool = Field(default=True, env="USE_ADAPTIVE_WEIGHTS")
    use_adaptive_threshold: bool = Field(default=True, env="USE_ADAPTIVE_THRESHOLD")
    use_diversity_ranking: bool = Field(default=True, env="USE_DIVERSITY_RANKING")
    use_metadata_enhancement: bool = Field(default=True, env="USE_METADATA_ENHANCEMENT")
    use_quality_scoring: bool = Field(default=True, env="USE_QUALITY_SCORING")
    use_advanced_reranker: bool = Field(default=False, env="USE_ADVANCED_RERANKER")

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
        default=["http://localhost:3000", "http://localhost:7860"], 
        env="CORS_ORIGINS",
        json_schema_extra={
            "description": "CORS allowed origins (JSON array or comma-separated string)"
        }
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

    def is_development(self) -> bool:
        """개발 환경 여부"""
        return self.debug

    def is_production(self) -> bool:
        """프로덕션 환경 여부"""
        return not self.debug

    def __init__(self, **kwargs):
        """
        초기화 시 환경변수 파일 로딩
        
        ⚠️ DEPRECATED: 이 클래스는 레거시입니다.
        새로운 코드에서는 `lawfirm_langgraph.config.app_config.Config`를 사용하세요.
        """
        import warnings
        warnings.warn(
            "core.utils.config.Config is deprecated. "
            "Use lawfirm_langgraph.config.app_config.Config instead. "
            "This class will be removed in a future version.",
            DeprecationWarning,
            stacklevel=2
        )
        
        # 환경변수 파일 로딩
        self.Settings._load_env_file(".env")
        
        # CORS_ORIGINS 환경 변수 처리 (빈 문자열이거나 잘못된 형식인 경우 기본값 사용)
        cors_origins_env = os.getenv("CORS_ORIGINS", "").strip()
        if cors_origins_env:
            try:
                import json
                # JSON 형식으로 파싱 시도
                parsed = json.loads(cors_origins_env)
                if isinstance(parsed, list):
                    # 유효한 리스트인 경우 환경 변수 유지
                    pass
                else:
                    # 리스트가 아니면 환경 변수 제거하여 기본값 사용
                    os.environ.pop("CORS_ORIGINS", None)
            except (json.JSONDecodeError, ValueError):
                # JSON 파싱 실패 시 쉼표로 구분된 문자열로 처리
                try:
                    origins = [origin.strip() for origin in cors_origins_env.split(",") if origin.strip()]
                    if origins:
                        # 쉼표로 구분된 문자열을 JSON 배열로 변환
                        os.environ["CORS_ORIGINS"] = json.dumps(origins)
                    else:
                        # 빈 리스트인 경우 환경 변수 제거하여 기본값 사용
                        os.environ.pop("CORS_ORIGINS", None)
                except Exception:
                    # 모든 파싱 실패 시 환경 변수 제거하여 기본값 사용
                    os.environ.pop("CORS_ORIGINS", None)
        else:
            # 환경 변수가 없으면 기본값 사용 (이미 설정되지 않았으므로 아무것도 하지 않음)
            pass
        
        super().__init__(**kwargs)

        # 데이터베이스 URL 설정 (PostgreSQL 전용, SQLite 지원 제거)
        # 1. DATABASE_URL이 환경변수에 있으면 우선 사용
        if not self.database_url:
            # 2. PostgreSQL 환경변수 조합 (DATABASE_URL이 없을 때 사용)
            # 프로젝트 루트 .env 파일의 설정을 우선 사용 (21-29줄)
            postgres_host = os.getenv("POSTGRES_HOST", "localhost")
            postgres_port = os.getenv("POSTGRES_PORT", "5432")
            postgres_db = os.getenv("POSTGRES_DB", "lawfirmai_local")
            postgres_user = os.getenv("POSTGRES_USER", "lawfirmai")
            postgres_password = os.getenv("POSTGRES_PASSWORD", "local_password")
            
            # URL 인코딩 (특수문자 처리)
            from urllib.parse import quote_plus
            encoded_password = quote_plus(postgres_password)
            
            # PostgreSQL URL 생성
            self.database_url = f"postgresql://{postgres_user}:{encoded_password}@{postgres_host}:{postgres_port}/{postgres_db}"
        
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
        
        # database_url이 없거나 비어있는 경우 PostgreSQL 환경변수로 조합
        if not self.database_url or self.database_url.strip() == "":
            from urllib.parse import quote_plus
            postgres_host = os.getenv("POSTGRES_HOST", "localhost")
            postgres_port = os.getenv("POSTGRES_PORT", "5432")
            postgres_db = os.getenv("POSTGRES_DB", "lawfirmai_local")
            postgres_user = os.getenv("POSTGRES_USER", "lawfirmai")
            postgres_password = os.getenv("POSTGRES_PASSWORD", "local_password")
            
            encoded_password = quote_plus(postgres_password)
            self.database_url = f"postgresql://{postgres_user}:{encoded_password}@{postgres_host}:{postgres_port}/{postgres_db}"
            import logging
            logger = logging.getLogger(__name__)
            logger.info(f"Database URL generated from POSTGRES_* environment variables: postgresql://{postgres_user}:***@{postgres_host}:{postgres_port}/{postgres_db}")
        
        # 최종 검증: SQLite URL이 여전히 있으면 에러
        if self.database_url and self.database_url.startswith("sqlite://"):
            import logging
            logger = logging.getLogger(__name__)
            logger.error(f"SQLite URL detected: {self.database_url}")
            raise ValueError(
                "SQLite is no longer supported. Please use PostgreSQL. "
                "Set DATABASE_URL to a PostgreSQL URL (e.g., postgresql://user:password@host:port/database) "
                "or configure POSTGRES_* environment variables in .env file (lines 21-29)."
            )
        
        # DATABASE_PATH는 더 이상 사용하지 않음 (레거시 호환성 유지)
        # DATABASE_PATH가 설정되어 있으면 무시하고 경고만 출력
        if self.database_path:
            import warnings
            warnings.warn(
                "DATABASE_PATH is deprecated. Use DATABASE_URL or POSTGRES_* environment variables instead. "
                "SQLite is no longer supported. Please use PostgreSQL.",
                DeprecationWarning,
                stacklevel=2
            )
    
    def configure_logging(self) -> None:
        """
        Config의 log_level을 사용하여 전역 로깅 설정
        
        이 메서드는 Config 인스턴스의 log_level 속성을 사용하여
        전역 로깅 레벨을 설정합니다.
        
        사용 예시:
            config = Config()
            config.configure_logging()  # config.log_level 사용
        """
        try:
            from .logger import configure_global_logging
            configure_global_logging(self.log_level)
        except ImportError:
            # logger 모듈이 없으면 기본 설정
            import logging
            log_level_map = {
                "CRITICAL": logging.CRITICAL,
                "ERROR": logging.ERROR,
                "WARNING": logging.WARNING,
                "INFO": logging.INFO,
                "DEBUG": logging.DEBUG,
            }
            log_level = log_level_map.get(self.log_level.upper(), logging.INFO)
            logging.basicConfig(
                level=log_level,
                format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
            )