"""
API 서버 설정 관리
"""
import os
from typing import List, Optional
from pydantic_settings import BaseSettings


class APIConfig(BaseSettings):
    """API 서버 설정"""
    
    # 서버 설정
    api_host: str = "0.0.0.0"
    api_port: int = 8000
    debug: bool = False
    
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        # 개발 모드 자동 감지: DEBUG 환경 변수 또는 ENVIRONMENT=development
        if self.debug is False:
            debug_env = os.getenv("DEBUG", "").lower()
            environment = os.getenv("ENVIRONMENT", "").lower()
            if debug_env in ("true", "1", "yes") or environment == "development":
                self.debug = True
    
    # CORS 설정
    # 개발 환경: "http://localhost:3000,http://127.0.0.1:3000"
    # 프로덕션: 실제 도메인을 지정
    cors_origins: str = "http://localhost:3000,http://127.0.0.1:3000"
    
    # 데이터베이스 설정
    # 로컬: sqlite:///./data/api_sessions.db
    # 개발/운영: postgresql://user:password@host:port/dbname
    database_url: str = "sqlite:///./data/api_sessions.db"
    
    # PostgreSQL 설정 (PostgreSQL 사용 시)
    postgres_host: str = "postgres"
    postgres_port: int = 5432
    postgres_db: str = "lawfirmai"
    postgres_user: str = "lawfirmai"
    postgres_password: str = ""
    
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

