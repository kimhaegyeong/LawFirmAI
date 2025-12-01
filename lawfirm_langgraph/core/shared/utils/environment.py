"""
Environment Management
환경 변수 관리 유틸리티
"""
import os
from enum import Enum
from typing import Optional


class Environment(str, Enum):
    """환경 타입 Enum"""
    LOCAL = "local"
    DEVELOPMENT = "development"
    PRODUCTION = "production"
    
    @classmethod
    def from_string(cls, value: Optional[str]) -> "Environment":
        """문자열로부터 Environment 생성"""
        if value is None:
            return cls.DEVELOPMENT  # 기본값
        
        value = value.lower().strip()
        
        # 유사한 값 매핑
        mapping = {
            "dev": cls.DEVELOPMENT,
            "prod": cls.PRODUCTION,
            "local": cls.LOCAL,
            "development": cls.DEVELOPMENT,
            "production": cls.PRODUCTION,
        }
        
        if value in mapping:
            return mapping[value]
        
        # 직접 매칭 시도
        try:
            return cls(value)
        except ValueError:
            # 잘못된 값인 경우 기본값 반환
            print(f"⚠️  Invalid ENVIRONMENT value: {value}. Using default: {cls.DEVELOPMENT}")
            return cls.DEVELOPMENT
    
    @classmethod
    def get_current(cls) -> "Environment":
        """현재 환경 변수에서 Environment 가져오기"""
        env_value = os.getenv("ENVIRONMENT", "").strip()
        return cls.from_string(env_value)
    
    def is_local(self) -> bool:
        """로컬 환경 여부"""
        return self == Environment.LOCAL
    
    def is_development(self) -> bool:
        """개발 환경 여부"""
        return self == Environment.DEVELOPMENT
    
    def is_production(self) -> bool:
        """프로덕션 환경 여부"""
        return self == Environment.PRODUCTION
    
    def is_debug_enabled(self) -> bool:
        """디버그 모드 활성화 여부"""
        return self in (Environment.LOCAL, Environment.DEVELOPMENT)

