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
    const processedCodeKey = `oauth2_processed_code_${code}`;
    const isProcessed = sessionStorage.getItem(processedCodeKey);
    
    if (isProcessed === 'true') {
      logger.warn('OAuth callback: 인증 코드가 이미 처리되었습니다. 중복 호출을 방지합니다.', { code: code.substring(0, 10) });
      throw new Error('인증 코드가 이미 사용되었습니다. 다시 로그인해주세요.');
    }
    
    sessionStorage.setItem(processedCodeKey, 'true');
    
    const savedState = sessionStorage.getItem('oauth2_state');
    
    if (savedState) {
      if (savedState !== state) {
        logger.error('State mismatch!', { savedState, urlState: state });
        sessionStorage.removeItem(processedCodeKey);
        throw new Error('Invalid state parameter');
      }
      sessionStorage.removeItem('oauth2_state');
    }
    
    logger.info('OAuth callback: 인증 코드 처리 시작', { code: code.substring(0, 10), state: state?.substring(0, 10) });
    
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
      sessionStorage.removeItem(processedCodeKey);
      throw new Error(errorData.detail || 'OAuth callback failed');
    }
    
    const data: TokenResponse = await response.json();
    
    setAccessToken(data.access_token);
    
    const savedToken = getAccessToken();
    if (!savedToken) {
      logger.error('Failed to save access token!');
      sessionStorage.removeItem(processedCodeKey);
      throw new Error('토큰 저장에 실패했습니다.');
    }
    
    if (data.refresh_token) {
      setRefreshToken(data.refresh_token);
    }
    
    logger.info('OAuth callback: 인증 코드 처리 완료', { code: code.substring(0, 10) });
    
    setTimeout(() => {
      sessionStorage.removeItem(processedCodeKey);
    }, 60000);
    
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
    const verify = localStorage.getItem(ACCESS_TOKEN_KEY);
    if (!verify || verify !== token) {
      logger.error('Token verification failed!');
      throw new Error('토큰 저장 후 검증에 실패했습니다.');
    }
  } catch (error) {
    logger.error('Failed to set access token:', error);
    throw error;
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

