"""
인증 서비스
"""
import os
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
        self.expiration_hours = api_config.jwt_expiration_hours
        
        if not self.secret_key:
            logger.warning("JWT_SECRET_KEY가 설정되지 않았습니다. 인증이 비활성화됩니다.")
    
    def verify_password(self, plain_password: str, hashed_password: str) -> bool:
        """비밀번호 검증"""
        return pwd_context.verify(plain_password, hashed_password)
    
    def get_password_hash(self, password: str) -> str:
        """비밀번호 해시"""
        return pwd_context.hash(password)
    
    def create_access_token(self, data: Dict[str, Any], expires_delta: Optional[timedelta] = None) -> str:
        """JWT 토큰 생성"""
        if not self.secret_key:
            raise ValueError("JWT_SECRET_KEY가 설정되지 않았습니다.")
        
        to_encode = data.copy()
        
        if expires_delta:
            expire = datetime.now(timezone.utc) + expires_delta
        else:
            expire = datetime.now(timezone.utc) + timedelta(hours=self.expiration_hours)
        
        to_encode.update({"exp": expire})
        encoded_jwt = jwt.encode(to_encode, self.secret_key, algorithm=self.algorithm)
        return encoded_jwt
    
    def verify_token(self, token: str) -> Optional[Dict[str, Any]]:
        """JWT 토큰 검증"""
        if not self.secret_key:
            return None
        
        try:
            payload = jwt.decode(token, self.secret_key, algorithms=[self.algorithm])
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

