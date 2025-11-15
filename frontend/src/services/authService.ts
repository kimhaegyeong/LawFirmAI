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
    
    // URL 파라미터에서 세션 ID 확인 (보안 강화: 토큰을 URL에 포함하지 않음)
    const urlParams = new URLSearchParams(window.location.search);
    const sessionId = urlParams.get('session_id');
    
    if (sessionId) {
      // 세션 ID로 토큰 교환 (보안 강화)
      logger.info('OAuth callback: 세션 ID로 토큰 교환');
      
      try {
        const exchangeUrl = api.defaults.baseURL 
          ? `${api.defaults.baseURL}/oauth2/token-exchange`
          : `/api/v1/oauth2/token-exchange`;
        
        const response = await fetch(exchangeUrl, {
          method: 'POST',
          headers: {
            'Content-Type': 'application/json',
            'Accept': 'application/json',
          },
          body: JSON.stringify({ session_id: sessionId }),
          credentials: 'include'
        });
        
        if (!response.ok) {
          const errorData = await response.json().catch(() => ({ detail: '토큰 교환 실패' }));
          throw new Error(errorData.detail || '토큰 교환에 실패했습니다.');
        }
        
        const tokenResponse: TokenResponse = await response.json();
        
        if (!tokenResponse.access_token) {
          throw new Error('토큰 응답이 올바르지 않습니다.');
        }
        
        setAccessToken(tokenResponse.access_token);
        
        const savedToken = getAccessToken();
        if (!savedToken || savedToken !== tokenResponse.access_token) {
          logger.error('Failed to save access token!', { saved: !!savedToken, matches: savedToken === tokenResponse.access_token });
          throw new Error('토큰 저장에 실패했습니다.');
        }
        
        if (tokenResponse.refresh_token) {
          setRefreshToken(tokenResponse.refresh_token);
        }
        
        // 보안을 위해 URL에서 세션 ID 제거
        urlParams.delete('session_id');
        const newUrl = window.location.pathname + (urlParams.toString() ? `?${urlParams.toString()}` : '');
        window.history.replaceState({}, document.title, newUrl);
        
        logger.info('OAuth callback: 토큰 교환 완료', { code: code.substring(0, 10) });
        
        setTimeout(() => {
          sessionStorage.removeItem(processedCodeKey);
        }, 60000);
        
        return tokenResponse;
      } catch (tokenError) {
        logger.error('OAuth callback: 토큰 교환 중 오류 발생', { error: tokenError });
        throw tokenError;
      }
    }
    
    // 하위 호환성을 위해 기존 방식도 지원 (URL 파라미터에서 토큰 확인)
    const accessToken = urlParams.get('access_token');
    const refreshToken = urlParams.get('refresh_token');
    
    if (accessToken) {
      logger.warn('OAuth callback: URL 파라미터에서 토큰 확인 (하위 호환성, 보안상 권장되지 않음)');
      
      try {
        setAccessToken(accessToken);
        
        const savedToken = getAccessToken();
        if (!savedToken || savedToken !== accessToken) {
          logger.error('Failed to save access token!', { saved: !!savedToken, matches: savedToken === accessToken });
          throw new Error('토큰 저장에 실패했습니다.');
        }
        
        if (refreshToken) {
          setRefreshToken(refreshToken);
        }
        
        // 보안을 위해 URL에서 토큰 제거
        urlParams.delete('access_token');
        urlParams.delete('refresh_token');
        const newUrl = window.location.pathname + (urlParams.toString() ? `?${urlParams.toString()}` : '');
        window.history.replaceState({}, document.title, newUrl);
        
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
      } catch (tokenError) {
        logger.error('OAuth callback: 토큰 저장 중 오류 발생', { error: tokenError });
        throw tokenError;
      }
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
        try {
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
          } else {
            // 리다이렉트 응답이지만 토큰이 없는 경우
            // 백엔드가 이미 처리했을 수 있으므로, 에러 파라미터 확인
            const errorParam = redirectUrl.searchParams.get('error');
            if (errorParam) {
              logger.error('OAuth callback: 리다이렉트 URL에 에러 파라미터 발견', { error: errorParam, location });
              throw new Error(errorParam);
            }
            
            // 토큰이 없고 에러도 없는 경우, 백엔드가 이미 처리했을 수 있음
            // 하지만 프론트엔드에서는 토큰이 필요하므로 오류로 처리
            logger.warn('OAuth callback: 리다이렉트 응답을 받았지만 토큰이 없습니다', { location });
            throw new Error('OAuth 콜백 처리 중 토큰을 받지 못했습니다. 다시 로그인해주세요.');
          }
        } catch (urlError) {
          if (urlError instanceof Error && urlError.message.includes('OAuth')) {
            throw urlError;
          }
          logger.error('OAuth callback: 리다이렉트 URL 파싱 실패', { location, error: urlError });
          throw new Error('OAuth 콜백 리다이렉트 URL 처리에 실패했습니다.');
        }
      } else {
        logger.error('OAuth callback: 리다이렉트 응답이지만 Location 헤더가 없습니다', { status: response.status });
        throw new Error('OAuth 콜백 리다이렉트 응답이 올바르지 않습니다.');
      }
    }
    
    if (!response.ok) {
      let errorMessage = 'OAuth callback failed';
      try {
        const errorData = await response.json();
        errorMessage = errorData.detail || errorData.message || errorMessage;
        logger.error('OAuth callback: API 오류 응답', { status: response.status, error: errorData });
      } catch (jsonError) {
        const text = await response.text().catch(() => '');
        logger.error('OAuth callback: API 오류 응답 (JSON 파싱 실패)', { status: response.status, text });
        errorMessage = text || errorMessage;
      }
      throw new Error(errorMessage);
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

