/**
 * 인증 서비스
 */
import { api, extractApiError } from './api';
import logger from '../utils/logger';
import type { TokenResponse, UserInfo, DeleteAccountResponse } from '../types/auth';

const ACCESS_TOKEN_KEY = 'access_token';
const REFRESH_TOKEN_KEY = 'refresh_token';

/**
 * Google OAuth2 로그인 시작
 */
export function startGoogleLogin(): void {
  try {
    const state = generateState();
    sessionStorage.setItem('oauth2_state', state);
    
    const authorizeUrl = api.defaults.baseURL 
      ? `${api.defaults.baseURL}/oauth2/google/authorize?state=${encodeURIComponent(state)}`
      : `/api/v1/oauth2/google/authorize?state=${encodeURIComponent(state)}`;
    
    window.location.href = authorizeUrl;
  } catch (error) {
    logger.error('Failed to start Google login:', error);
    throw extractApiError(error);
  }
}

/**
 * OAuth2 콜백 처리
 */
export async function handleOAuthCallback(code: string, state: string): Promise<TokenResponse> {
  try {
    const savedState = sessionStorage.getItem('oauth2_state');
    if (savedState && savedState !== state) {
      throw new Error('Invalid state parameter');
    }
    sessionStorage.removeItem('oauth2_state');
    
    const callbackUrl = api.defaults.baseURL 
      ? `${api.defaults.baseURL}/oauth2/google/callback?code=${code}&state=${state}`
      : `/api/v1/oauth2/google/callback?code=${code}&state=${state}`;
    
    const response = await fetch(callbackUrl, {
      method: 'GET',
      headers: {
        'Accept': 'application/json',
      },
    });
    
    if (!response.ok) {
      const errorData = await response.json().catch(() => ({ detail: 'OAuth callback failed' }));
      throw new Error(errorData.detail || 'OAuth callback failed');
    }
    
    const data: TokenResponse = await response.json();
    
    setAccessToken(data.access_token);
    if (data.refresh_token) {
      setRefreshToken(data.refresh_token);
    }
    
    return data;
  } catch (error) {
    logger.error('Failed to handle OAuth callback:', error);
    throw extractApiError(error);
  }
}

/**
 * 토큰 갱신
 */
export async function refreshAccessToken(): Promise<TokenResponse> {
  try {
    const refreshToken = getRefreshToken();
    if (!refreshToken) {
      throw new Error('No refresh token available');
    }
    
    const response = await api.post<TokenResponse>('/auth/refresh', {
      refresh_token: refreshToken,
    });
    
    const data = response.data;
    setAccessToken(data.access_token);
    if (data.refresh_token) {
      setRefreshToken(data.refresh_token);
    }
    
    return data;
  } catch (error) {
    logger.error('Failed to refresh access token:', error);
    removeTokens();
    throw extractApiError(error);
  }
}

/**
 * 현재 사용자 정보 조회
 */
export async function getCurrentUser(): Promise<UserInfo> {
  try {
    const response = await api.get<UserInfo>('/auth/me');
    return response.data;
  } catch (error) {
    logger.error('Failed to get current user:', error);
    throw extractApiError(error);
  }
}

/**
 * 로그아웃
 */
export function logout(): void {
  removeTokens();
}

/**
 * Access token 조회
 */
export function getAccessToken(): string | null {
  try {
    return localStorage.getItem(ACCESS_TOKEN_KEY);
  } catch (error) {
    logger.error('Failed to get access token:', error);
    return null;
  }
}

/**
 * Access token 저장
 */
export function setAccessToken(token: string): void {
  try {
    localStorage.setItem(ACCESS_TOKEN_KEY, token);
  } catch (error) {
    logger.error('Failed to set access token:', error);
  }
}

/**
 * Refresh token 조회
 */
export function getRefreshToken(): string | null {
  try {
    return localStorage.getItem(REFRESH_TOKEN_KEY);
  } catch (error) {
    logger.error('Failed to get refresh token:', error);
    return null;
  }
}

/**
 * Refresh token 저장
 */
export function setRefreshToken(token: string): void {
  try {
    localStorage.setItem(REFRESH_TOKEN_KEY, token);
  } catch (error) {
    logger.error('Failed to set refresh token:', error);
  }
}

/**
 * 토큰 삭제
 */
export function removeTokens(): void {
  try {
    localStorage.removeItem(ACCESS_TOKEN_KEY);
    localStorage.removeItem(REFRESH_TOKEN_KEY);
  } catch (error) {
    logger.error('Failed to remove tokens:', error);
  }
}

/**
 * 인증 상태 확인
 */
export function isAuthenticated(): boolean {
  return getAccessToken() !== null;
}

/**
 * 회원탈퇴
 */
export async function deleteAccount(): Promise<DeleteAccountResponse> {
  try {
    const response = await api.delete<DeleteAccountResponse>('/auth/account');
    removeTokens();
    return response.data;
  } catch (error) {
    logger.error('Failed to delete account:', error);
    throw extractApiError(error);
  }
}

/**
 * OAuth2 state 생성
 */
function generateState(): string {
  const array = new Uint8Array(32);
  crypto.getRandomValues(array);
  return Array.from(array, byte => byte.toString(16).padStart(2, '0')).join('');
}

