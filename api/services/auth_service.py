"""
인증 서비스
"""
import os
import secrets
from datetime import datetime, timedelta, timezone
from typing import Optional, Dict, Any
from jose import JWTError, jwt
from passlib.context import CryptContext
import logging

from api.config import api_config

logger = logging.getLogger(__name__)

pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")


class AuthService:
    """인증 서비스"""
    
    def __init__(self):
        """초기화"""
        self.secret_key = api_config.jwt_secret_key or os.getenv("JWT_SECRET_KEY", "")
        self.algorithm = api_config.jwt_algorithm
        self.access_token_expiration_minutes = api_config.jwt_access_token_expiration_minutes
        self.refresh_token_expiration_days = api_config.jwt_refresh_token_expiration_days
        
        # 개발 환경에서 JWT_SECRET_KEY가 없으면 자동 생성 (보안상 프로덕션에서는 사용 금지)
        if not self.secret_key and api_config.debug:
            self.secret_key = secrets.token_urlsafe(32)
            logger.warning(
                f"⚠️  개발 환경: JWT_SECRET_KEY가 설정되지 않아 자동으로 생성되었습니다. "
                f"프로덕션 환경에서는 반드시 JWT_SECRET_KEY를 설정하세요. "
                f"생성된 키: {self.secret_key[:10]}..."
            )
        elif not self.secret_key:
            logger.warning("JWT_SECRET_KEY가 설정되지 않았습니다. 인증이 비활성화됩니다.")
    
    def verify_password(self, plain_password: str, hashed_password: str) -> bool:
        """비밀번호 검증"""
        return pwd_context.verify(plain_password, hashed_password)
    
    def get_password_hash(self, password: str) -> str:
        """비밀번호 해시"""
        return pwd_context.hash(password)
    
    def create_access_token(self, data: Dict[str, Any], expires_delta: Optional[timedelta] = None) -> str:
        """JWT access token 생성"""
        if not self.secret_key:
            raise ValueError("JWT_SECRET_KEY가 설정되지 않았습니다.")
        
        to_encode = data.copy()
        to_encode.update({"type": "access"})
        
        if expires_delta:
            expire = datetime.now(timezone.utc) + expires_delta
        else:
            expire = datetime.now(timezone.utc) + timedelta(minutes=self.access_token_expiration_minutes)
        
        to_encode.update({"exp": expire})
        encoded_jwt = jwt.encode(to_encode, self.secret_key, algorithm=self.algorithm)
        return encoded_jwt
    
    def create_refresh_token(self, data: Dict[str, Any]) -> str:
        """JWT refresh token 생성"""
        if not self.secret_key:
            raise ValueError("JWT_SECRET_KEY가 설정되지 않았습니다.")
        
        to_encode = data.copy()
        to_encode.update({"type": "refresh"})
        expire = datetime.now(timezone.utc) + timedelta(days=self.refresh_token_expiration_days)
        
        to_encode.update({"exp": expire})
        encoded_jwt = jwt.encode(to_encode, self.secret_key, algorithm=self.algorithm)
        return encoded_jwt
    
    def verify_token(self, token: str, token_type: str = "access") -> Optional[Dict[str, Any]]:
        """JWT 토큰 검증"""
        if not self.secret_key:
            return None
        
        try:
            payload = jwt.decode(token, self.secret_key, algorithms=[self.algorithm])
            if payload.get("type") != token_type:
                logger.debug(f"JWT 토큰 타입 불일치: 기대={token_type}, 실제={payload.get('type')}")
                return None
            return payload
        except JWTError as e:
            logger.debug(f"JWT 토큰 검증 실패: {e}")
            return None
    
    def verify_api_key(self, api_key: str) -> bool:
        """API 키 검증"""
        expected_key = os.getenv("API_KEY", "")
        if not expected_key:
            return False
        return api_key == expected_key
    
    def is_auth_enabled(self) -> bool:
        """인증 활성화 여부"""
        return api_config.auth_enabled and bool(self.secret_key or os.getenv("API_KEY"))


auth_service = AuthService()

