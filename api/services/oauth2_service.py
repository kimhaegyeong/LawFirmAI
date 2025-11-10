"""
OAuth2 Google 인증 서비스
"""
import os
import logging
from typing import Optional, Dict, Any
from authlib.integrations.httpx_client import AsyncOAuth2Client
from authlib.oauth2.rfc6749 import OAuth2Token

from api.config import api_config

logger = logging.getLogger(__name__)


class OAuth2GoogleService:
    """OAuth2 Google 인증 서비스"""
    
    def __init__(self):
        """초기화"""
        self.client_id = api_config.google_client_id or os.getenv("GOOGLE_CLIENT_ID", "")
        self.client_secret = api_config.google_client_secret or os.getenv("GOOGLE_CLIENT_SECRET", "")
        self.redirect_uri = api_config.google_redirect_uri or os.getenv("GOOGLE_REDIRECT_URI", "")
        
        if not self.client_id or not self.client_secret:
            logger.warning("Google OAuth2 설정이 완료되지 않았습니다.")
    
    def is_enabled(self) -> bool:
        """OAuth2 Google 활성화 여부"""
        return bool(self.client_id and self.client_secret)
    
    def get_authorization_url(self, state: Optional[str] = None) -> Optional[str]:
        """Google OAuth2 인증 URL 생성"""
        if not self.is_enabled():
            return None
        
        client = AsyncOAuth2Client(
            client_id=self.client_id,
            client_secret=self.client_secret,
            redirect_uri=self.redirect_uri
        )
        
        authorization_url, _ = client.create_authorization_url(
            "https://accounts.google.com/o/oauth2/v2/auth",
            state=state,
            scope="openid email profile"
        )
        
        return authorization_url
    
    async def get_token(self, code: str) -> Optional[OAuth2Token]:
        """인증 코드로 토큰 획득"""
        if not self.is_enabled():
            return None
        
        try:
            client = AsyncOAuth2Client(
                client_id=self.client_id,
                client_secret=self.client_secret,
                redirect_uri=self.redirect_uri
            )
            
            token = await client.fetch_token(
                "https://oauth2.googleapis.com/token",
                code=code,
                grant_type="authorization_code"
            )
            
            return token
        except Exception as e:
            logger.error(f"Google OAuth2 토큰 획득 실패: {e}")
            return None
    
    async def get_user_info(self, token: OAuth2Token) -> Optional[Dict[str, Any]]:
        """토큰으로 사용자 정보 획득"""
        if not self.is_enabled():
            return None
        
        try:
            client = AsyncOAuth2Client(
                client_id=self.client_id,
                client_secret=self.client_secret,
                token=token
            )
            
            resp = await client.get("https://www.googleapis.com/oauth2/v2/userinfo")
            resp.raise_for_status()
            
            return resp.json()
        except Exception as e:
            logger.error(f"Google 사용자 정보 획득 실패: {e}")
            return None
    
    async def revoke_token(self, token: str) -> bool:
        """
        Google OAuth2 토큰 해제
        
        Args:
            token: 해제할 토큰 (access_token 또는 refresh_token)
        
        Returns:
            성공 여부
        """
        if not self.is_enabled():
            logger.warning("OAuth2 Google이 활성화되지 않아 토큰 해제를 건너뜁니다.")
            return False
        
        try:
            import httpx
            
            async with httpx.AsyncClient() as client:
                response = await client.post(
                    "https://oauth2.googleapis.com/revoke",
                    params={"token": token},
                    headers={"Content-Type": "application/x-www-form-urlencoded"}
                )
                
                if response.status_code == 200:
                    logger.info("Google OAuth2 토큰 해제 성공")
                    return True
                else:
                    logger.warning(f"Google OAuth2 토큰 해제 실패: {response.status_code} - {response.text}")
                    return False
        except Exception as e:
            logger.error(f"Google OAuth2 토큰 해제 중 오류 발생: {e}")
            return False


oauth2_google_service = OAuth2GoogleService()

