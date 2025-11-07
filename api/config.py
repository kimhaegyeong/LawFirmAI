"""
API 서버 설정 관리
"""
import os
from typing import List
from pydantic_settings import BaseSettings


class APIConfig(BaseSettings):
    """API 서버 설정"""
    
    # 서버 설정
    api_host: str = "0.0.0.0"
    api_port: int = 8000
    debug: bool = False
    
    # CORS 설정
    # 개발 환경: "http://localhost:3000,http://127.0.0.1:3000"
    # 프로덕션: 실제 도메인을 지정
    cors_origins: str = "http://localhost:3000,http://127.0.0.1:3000"
    
    # 데이터베이스 설정
    database_url: str = "sqlite:///./data/api_sessions.db"
    
    # lawfirm_langgraph 설정
    langgraph_enabled: bool = True
    
    # 세션 관리
    session_ttl_hours: int = 24
    
    def get_cors_origins(self) -> List[str]:
        """CORS origins 리스트 반환"""
        import json
        import ast
        
        # 이미 리스트인 경우
        if isinstance(self.cors_origins, list):
            return [str(origin).strip() for origin in self.cors_origins]
        
        # 문자열 처리
        cors_origins_str = str(self.cors_origins).strip()
        
        # "*"인 경우
        if cors_origins_str == "*":
            return ["*"]
        
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


# 전역 설정 인스턴스
api_config = APIConfig()

