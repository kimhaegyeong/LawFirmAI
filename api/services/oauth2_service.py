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
            logger.info(f"토큰 획득 시도: code={code[:20]}..., redirect_uri={self.redirect_uri}, client_id={self.client_id[:20] if self.client_id else None}...")
            
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
            
            logger.info("토큰 획득 성공")
            return token
        except Exception as e:
            error_msg = str(e)
            error_type = type(e).__name__
            logger.error(f"Google OAuth2 토큰 획득 실패: {error_type}: {error_msg}", exc_info=True)
            
            error_code = None
            error_description = None
            error_uri = None
            
            if hasattr(e, 'error'):
                error_code = getattr(e, 'error', None)
                logger.error(f"OAuth2 오류 코드: {error_code}")
            
            if hasattr(e, 'error_description'):
                error_description = getattr(e, 'error_description', None)
                if error_description:
                    logger.error(f"OAuth2 오류 설명: {error_description}")
            
            if hasattr(e, 'error_uri'):
                error_uri = getattr(e, 'error_uri', None)
                if error_uri:
                    logger.error(f"OAuth2 오류 URI: {error_uri}")
            
            if hasattr(e, 'response'):
                response = getattr(e, 'response', None)
                if response:
                    try:
                        if hasattr(response, 'text'):
                            response_text = getattr(response, 'text', None)
                            if response_text:
                                logger.error(f"OAuth2 응답 본문: {response_text}")
                        if hasattr(response, 'status_code'):
                            status_code = getattr(response, 'status_code', None)
                            if status_code:
                                logger.error(f"OAuth2 HTTP 상태 코드: {status_code}")
                        if hasattr(response, 'headers'):
                            headers = getattr(response, 'headers', None)
                            if headers:
                                logger.error(f"OAuth2 응답 헤더: {headers}")
                    except Exception as ex:
                        logger.debug(f"응답 정보 추출 실패: {ex}")
            
            logger.error(f"현재 설정: redirect_uri={self.redirect_uri}, client_id={self.client_id[:20] if self.client_id else None}...")
            
            logger.error("=== OAuth2 오류 상세 정보 ===")
            logger.error(f"오류 코드: {error_code}")
            logger.error(f"오류 설명: {error_description}")
            logger.error(f"오류 URI: {error_uri}")
            logger.error(f"리다이렉트 URI: {self.redirect_uri}")
            logger.error(f"클라이언트 ID: {self.client_id[:30] if self.client_id else None}...")
            logger.error(f"오류 메시지: {error_msg}")
            logger.error(f"오류 타입: {error_type}")
            
            if error_code == "invalid_grant" or "invalid_grant" in error_msg.lower():
                logger.error("=== invalid_grant 오류 원인 분석 ===")
                logger.error("1. 인증 코드가 이미 사용되었거나 만료되었을 수 있습니다 (보통 10분 이내 만료)")
                logger.error("2. 인증 코드가 다른 클라이언트 ID로 생성되었을 수 있습니다")
                logger.error("3. 리다이렉트 URI가 인증 코드 생성 시와 다를 수 있습니다")
                logger.error(f"   현재 redirect_uri: {self.redirect_uri}")
                logger.error("4. Google OAuth2 콘솔에서 리다이렉트 URI를 확인하세요:")
                logger.error("   https://console.cloud.google.com/apis/credentials")
                logger.error("5. 인증 코드를 여러 번 사용하려고 시도했을 수 있습니다")
                logger.error("6. 시스템 시간이 잘못 설정되어 있을 수 있습니다")
            elif error_code == "redirect_uri_mismatch" or "redirect_uri_mismatch" in error_msg.lower():
                logger.error(f"리다이렉트 URI가 일치하지 않습니다. 설정된 URI: {self.redirect_uri}")
                logger.error("Google OAuth2 콘솔에서 리다이렉트 URI를 확인하세요:")
                logger.error("https://console.cloud.google.com/apis/credentials")
            elif error_code == "invalid_client" or "invalid_client" in error_msg.lower():
                logger.error("클라이언트 ID 또는 Secret이 잘못되었습니다.")
                logger.error("Google OAuth2 콘솔에서 클라이언트 ID와 Secret을 확인하세요:")
                logger.error("https://console.cloud.google.com/apis/credentials")
            
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

