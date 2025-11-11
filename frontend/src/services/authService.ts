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
  const processedCodeKey = `oauth2_processed_code_${code}`;
  
  try {
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
        throw new Error('Invalid state parameter');
      }
      sessionStorage.removeItem('oauth2_state');
    }
    
    logger.info('OAuth callback: 인증 코드 처리 시작', { code: code.substring(0, 10), state: state?.substring(0, 10) });
    
    // URL 파라미터에서 토큰 확인 (백엔드가 리다이렉트로 전달)
    const urlParams = new URLSearchParams(window.location.search);
    const accessToken = urlParams.get('access_token');
    const refreshToken = urlParams.get('refresh_token');
    
    if (accessToken) {
      // 백엔드가 토큰을 쿼리 파라미터로 전달한 경우
      logger.info('OAuth callback: URL 파라미터에서 토큰 확인');
      
      setAccessToken(accessToken);
      
      const savedToken = getAccessToken();
      if (!savedToken) {
        logger.error('Failed to save access token!');
        throw new Error('토큰 저장에 실패했습니다.');
      }
      
      if (refreshToken) {
        setRefreshToken(refreshToken);
      }
      
      logger.info('OAuth callback: 인증 코드 처리 완료', { code: code.substring(0, 10) });
      
      setTimeout(() => {
        sessionStorage.removeItem(processedCodeKey);
      }, 60000);
      
      return {
        access_token: accessToken,
        refresh_token: refreshToken || '',
        token_type: 'bearer',
        expires_in: 3600
      };
    }
    
    // 토큰이 URL 파라미터에 없는 경우 백엔드 API 호출 (fallback)
    logger.info('OAuth callback: 백엔드 API 호출로 토큰 교환');
    
    const callbackUrl = api.defaults.baseURL 
      ? `${api.defaults.baseURL}/oauth2/google/callback?code=${code}&state=${state}`
      : `/api/v1/oauth2/google/callback?code=${code}&state=${state}`;
    
    const response = await fetch(callbackUrl, {
      method: 'GET',
      headers: {
        'Accept': 'application/json',
      },
      redirect: 'manual'
    });
    
    if (response.status === 307 || response.status === 302) {
      // 리다이렉트 응답인 경우, Location 헤더에서 토큰 추출 시도
      const location = response.headers.get('Location');
      if (location) {
        const redirectUrl = new URL(location);
        const redirectAccessToken = redirectUrl.searchParams.get('access_token');
        const redirectRefreshToken = redirectUrl.searchParams.get('refresh_token');
        
        if (redirectAccessToken) {
          setAccessToken(redirectAccessToken);
          if (redirectRefreshToken) {
            setRefreshToken(redirectRefreshToken);
          }
          
          logger.info('OAuth callback: 리다이렉트 URL에서 토큰 추출 완료');
          
          setTimeout(() => {
            sessionStorage.removeItem(processedCodeKey);
          }, 60000);
          
          return {
            access_token: redirectAccessToken,
            refresh_token: redirectRefreshToken || '',
            token_type: 'bearer',
            expires_in: 3600
          };
        }
      }
    }
    
    if (!response.ok) {
      const errorData = await response.json().catch(() => ({ detail: 'OAuth callback failed' }));
      throw new Error(errorData.detail || 'OAuth callback failed');
    }
    
    const data: TokenResponse = await response.json();
    
    setAccessToken(data.access_token);
    
    const savedToken = getAccessToken();
    if (!savedToken) {
      logger.error('Failed to save access token!');
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
    const errorMessage = error instanceof Error ? error.message : String(error);
    
    if (errorMessage.includes('이미 사용되었습니다') || errorMessage.includes('invalid_grant')) {
      logger.warn('인증 코드가 이미 사용되었거나 만료되었습니다. sessionStorage에 유지합니다.');
    } else {
      sessionStorage.removeItem(processedCodeKey);
    }
    
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

/**
 * 오래된 OAuth 키들을 sessionStorage에서 정리
 */
export function cleanupOldOAuthKeys(): void {
  try {
    const keysToRemove: string[] = [];
    
    for (let i = 0; i < sessionStorage.length; i++) {
      const key = sessionStorage.key(i);
      if (key && key.startsWith('oauth2_processed_code_')) {
        keysToRemove.push(key);
      }
    }
    
    if (keysToRemove.length > 0) {
      keysToRemove.forEach(key => sessionStorage.removeItem(key));
      logger.info(`Cleaned up ${keysToRemove.length} old OAuth keys from sessionStorage`);
    }
  } catch (error) {
    logger.error('Failed to cleanup old OAuth keys:', error);
  }
}

