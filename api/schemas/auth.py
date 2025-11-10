"""
인증 관련 스키마
"""
from typing import Optional
from pydantic import BaseModel, Field


class OAuth2GoogleAuthRequest(BaseModel):
    """OAuth2 Google 인증 요청"""
    state: Optional[str] = Field(None, description="상태 값 (CSRF 보호용)")


class OAuth2GoogleCallbackRequest(BaseModel):
    """OAuth2 Google 콜백 요청"""
    code: str = Field(..., description="인증 코드")
    state: Optional[str] = Field(None, description="상태 값")


class TokenResponse(BaseModel):
    """토큰 응답"""
    access_token: str = Field(..., description="Access token")
    refresh_token: Optional[str] = Field(None, description="Refresh token")
    token_type: str = Field(default="bearer", description="토큰 타입")
    expires_in: int = Field(..., description="만료 시간 (초)")


class RefreshTokenRequest(BaseModel):
    """Refresh token 요청"""
    refresh_token: str = Field(..., description="Refresh token")


class DeleteAccountResponse(BaseModel):
    """회원탈퇴 응답"""
    message: str = Field(..., description="응답 메시지")
    deleted_sessions: int = Field(..., description="삭제된 세션 수")
