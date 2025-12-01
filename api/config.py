"""
API 서버 설정 관리
"""
import os
import logging
from typing import List, Optional
from pydantic_settings import BaseSettings
from pydantic import Field

# 환경 변수 로더를 먼저 로드
try:
    from utils.env_loader import ensure_env_loaded
    from pathlib import Path
    ensure_env_loaded(Path(__file__).parent.parent)
except ImportError:
    pass

from lawfirm_langgraph.core.shared.utils.environment import Environment

logger = logging.getLogger(__name__)


class APIConfig(BaseSettings):
    """API 서버 설정"""
    
    # 환경 설정
    environment: Environment = Field(
        default=Environment.DEVELOPMENT,
        env="ENVIRONMENT"
    )
    
    # 서버 설정
    api_host: str = "0.0.0.0"
    api_port: int = 8000
    debug: bool = False
    
    # CORS 설정
    # 개발 환경: "http://localhost:3000,http://127.0.0.1:3000"
    # 프로덕션: 실제 도메인을 지정
    cors_origins: str = "http://localhost:3000,http://127.0.0.1:3000"
    
    # 데이터베이스 설정
    # DATABASE_URL이 명시적으로 설정되어 있으면 사용
    # 없으면 POSTGRES_* 환경변수들을 조합하여 생성
    database_url: Optional[str] = None
    
    # PostgreSQL 설정 (DATABASE_URL이 없을 때 사용)
    # 프로젝트 루트 .env 파일의 설정을 우선 사용 (21-29줄)
    # 없으면 api/.env 파일의 설정 사용
    postgres_host: str = "localhost"
    postgres_port: int = 5432
    postgres_db: str = "lawfirmai_local"
    postgres_user: str = "lawfirmai"
    postgres_password: str = "local_password"
    
    def __init__(self, **kwargs):
        # ENVIRONMENT 환경 변수에서 자동 설정
        if "environment" not in kwargs:
            kwargs["environment"] = Environment.get_current()
        
        super().__init__(**kwargs)
        
        # database_url이 설정되지 않았으면 환경변수 조합으로 생성
        # 프로젝트 루트 .env 파일의 설정을 우선 사용 (21-29줄)
        # 없으면 api/.env 파일의 설정 사용
        if not self.database_url:
            from urllib.parse import quote_plus
            encoded_password = quote_plus(self.postgres_password)
            self.database_url = f"postgresql://{self.postgres_user}:{encoded_password}@{self.postgres_host}:{self.postgres_port}/{self.postgres_db}"
        
        # SQLite URL 검증
        if self.database_url and self.database_url.startswith("sqlite://"):
            raise ValueError(
                "SQLite is no longer supported. Please use PostgreSQL. "
                "Set DATABASE_URL to a PostgreSQL URL (e.g., postgresql://user:password@host:port/database) "
                "or configure POSTGRES_* environment variables in .env file (lines 21-29)."
            )
        
        # 환경에 따른 자동 설정
        if self.environment.is_debug_enabled():
            self.debug = True
        elif "DEBUG" in os.environ:
            # DEBUG 환경 변수가 명시적으로 설정된 경우 우선
            debug_env = os.getenv("DEBUG", "").lower()
            if debug_env in ("true", "1", "yes"):
                self.debug = True
            elif debug_env in ("false", "0", "no"):
                self.debug = False
        
        # 환경별 기본값 설정
        if self.environment == Environment.PRODUCTION:
            # 프로덕션 환경 기본값
            if not self.cors_origins or self.cors_origins == "*":
                logger.warning("프로덕션 환경에서 CORS_ORIGINS를 명시적으로 설정해야 합니다.")
            # 프로덕션에서는 인증 권장 (기본값은 False 유지, 명시적으로 설정 필요)
    
    # lawfirm_langgraph 설정
    langgraph_enabled: bool = True
    
    # 세션 관리
    session_ttl_hours: int = 24
    
    # 인증 설정
    auth_enabled: bool = False
    jwt_secret_key: str = ""
    jwt_algorithm: str = "HS256"
    jwt_access_token_expiration_minutes: int = 30
    jwt_refresh_token_expiration_days: int = 7
    api_key_header: str = "X-API-Key"
    
    # OAuth2 Google 설정
    google_client_id: str = ""
    google_client_secret: str = ""
    google_redirect_uri: str = ""
    frontend_url: str = "http://localhost:3000"
    
    # Rate Limiting 설정
    rate_limit_enabled: bool = False
    rate_limit_per_minute: int = 10
    
    # 익명 사용자 질의 제한 설정
    anonymous_quota_enabled: bool = True
    anonymous_quota_limit: int = 3
    anonymous_quota_reset_hour: int = 0
    
    # 스트리밍 캐싱 설정
    enable_stream_cache: bool = False
    stream_cache_ttl_seconds: int = 3600
    stream_cache_max_size: int = 100
    
    def get_cors_origins(self) -> List[str]:
        """CORS origins 리스트 반환"""
        import json
        import ast
        
        # 이미 리스트인 경우
        if isinstance(self.cors_origins, list):
            return [str(origin).strip() for origin in self.cors_origins]
        
        # 문자열 처리
        cors_origins_str = str(self.cors_origins).strip()
        
        # 프로덕션 환경에서 "*" 사용 금지
        if cors_origins_str == "*":
            if self.debug:
                return ["*"]
            else:
                logger.warning("프로덕션 환경에서 CORS 와일드카드(*) 사용은 보안상 위험합니다. 기본값을 사용합니다.")
                return ["http://localhost:3000", "http://127.0.0.1:3000"]
        
        # JSON 리스트 형태인 경우 (예: '["http://localhost:3000", "http://localhost:7860"]')
        if cors_origins_str.startswith("[") and cors_origins_str.endswith("]"):
            try:
                # JSON으로 파싱 시도
                parsed = json.loads(cors_origins_str)
                if isinstance(parsed, list):
                    return [str(origin).strip() for origin in parsed]
            except (json.JSONDecodeError, ValueError):
                # JSON 파싱 실패 시 ast.literal_eval 시도
                try:
                    parsed = ast.literal_eval(cors_origins_str)
                    if isinstance(parsed, list):
                        return [str(origin).strip() for origin in parsed]
                except (ValueError, SyntaxError):
                    pass
        
        # 콤마로 구분된 문자열인 경우 (기본 처리)
        return [origin.strip() for origin in cors_origins_str.split(",") if origin.strip()]
    
    def is_local(self) -> bool:
        """로컬 환경 여부"""
        return self.environment.is_local()
    
    def is_development(self) -> bool:
        """개발 환경 여부"""
        return self.environment.is_development()
    
    def is_production(self) -> bool:
        """프로덕션 환경 여부"""
        return self.environment.is_production()
    
    class Config:
        env_file = ".env"
        case_sensitive = False
        env_prefix = ""
        # Pydantic v2 protected namespaces 설정
        protected_namespaces = ('settings_',)
        # 환경 변수에서 값을 가져올 때 우선순위 설정
        # api/.env > root/.env > lawfirm_langgraph/.env
        # 추가 환경 변수 허용 (extra='ignore')
        extra = "ignore"


# 전역 설정 인스턴스 (지연 초기화)
_api_config_instance: Optional[APIConfig] = None


def get_api_config() -> APIConfig:
    """API 설정 인스턴스 가져오기 (지연 초기화)"""
    global _api_config_instance
    if _api_config_instance is None:
        _api_config_instance = APIConfig()
    return _api_config_instance


# 하위 호환성을 위한 전역 변수 (속성 접근 시 지연 초기화)
class _APIConfigProxy:
    """APIConfig 프록시 클래스 (지연 초기화 지원)"""
    
    def __getattr__(self, name: str):
        return getattr(get_api_config(), name)
    
    def __setattr__(self, name: str, value):
        config = get_api_config()
        setattr(config, name, value)


# 전역 설정 인스턴스 (프록시를 통해 지연 초기화)
api_config = _APIConfigProxy()

