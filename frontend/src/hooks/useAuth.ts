/**
 * 인증 관련 훅
 */
import { useState, useCallback, useEffect, useRef } from 'react';
import {
  startGoogleLogin,
  handleOAuthCallback,
  refreshAccessToken,
  getCurrentUser,
  logout as logoutService,
  isAuthenticated as checkAuthenticated,
} from '../services/authService';
import logger from '../utils/logger';
import type { UserInfo } from '../types/auth';

export function useAuth() {
  const [user, setUser] = useState<UserInfo | null>(null);
  const [isAuthenticated, setIsAuthenticated] = useState(() => {
    return checkAuthenticated();
  });
  const [isLoading, setIsLoading] = useState(true);
  const [error, setError] = useState<Error | null>(null);
  const isInitialMount = useRef(true);
  const isLoadingUser = useRef(false);
  const isHandlingCallback = useRef(false);

  /**
   * 사용자 정보 조회
   */
  const loadUser = useCallback(async () => {
    if (isLoadingUser.current) {
      logger.debug('loadUser: Already loading, skipping duplicate call');
      return;
    }

    // handleCallback이 실행 중이면 토큰을 삭제하지 않음
    if (isHandlingCallback.current) {
      logger.debug('loadUser: Callback handling in progress, skipping to avoid token deletion');
      return;
    }

    if (!checkAuthenticated()) {
      setIsLoading(false);
      setIsAuthenticated(false);
      setUser(null);
      return;
    }

    isLoadingUser.current = true;
    setIsLoading(true);
    setError(null);

    try {
      const userInfo = await getCurrentUser();
      setUser(userInfo);
      setIsAuthenticated(userInfo.authenticated);
      
      // handleCallback이 실행 중이면 토큰을 삭제하지 않음
      if (!userInfo.authenticated && !isHandlingCallback.current) {
        logger.warn('User info indicates not authenticated, clearing tokens');
        logoutService();
      } else if (!userInfo.authenticated && isHandlingCallback.current) {
        logger.debug('loadUser: Callback handling in progress, not clearing tokens');
      }
    } catch (err) {
      const error = err instanceof Error ? err : new Error('사용자 정보를 불러올 수 없습니다.');
      logger.error('Failed to load user:', error);
      
      const axiosError = err as any;
      if (axiosError?.response?.status === 401 && !isHandlingCallback.current) {
        setError(error);
        setIsAuthenticated(false);
        setUser(null);
        logoutService();
      } else {
        setError(error);
        setIsAuthenticated(false);
        setUser(null);
        if (axiosError?.response?.status === 401 && isHandlingCallback.current) {
          logger.debug('loadUser: Callback handling in progress, not clearing tokens on 401');
        }
      }
    } finally {
      setIsLoading(false);
      isLoadingUser.current = false;
    }
  }, []);

  /**
   * Google 로그인 시작
   */
  const login = useCallback(() => {
    try {
      startGoogleLogin();
    } catch (err) {
      const error = err instanceof Error ? err : new Error('로그인을 시작할 수 없습니다.');
      setError(error);
      logger.error('Failed to start login:', error);
    }
  }, []);

  /**
   * OAuth2 콜백 처리
   */
  const handleCallback = useCallback(async (code: string, state: string) => {
    if (isLoadingUser.current) {
      logger.debug('handleCallback: User loading in progress, waiting...');
      await new Promise(resolve => {
        const checkInterval = setInterval(() => {
          if (!isLoadingUser.current) {
            clearInterval(checkInterval);
            resolve(undefined);
          }
        }, 100);
      });
    }

    isHandlingCallback.current = true;
    isLoadingUser.current = true;
    setIsLoading(true);
    setError(null);

    try {
      const tokenResponse = await handleOAuthCallback(code, state);
      
      // 토큰이 응답에 포함되어 있는지 확인
      if (!tokenResponse || !tokenResponse.access_token) {
        throw new Error('토큰 응답이 올바르지 않습니다.');
      }
      
      // 토큰을 다시 한 번 저장 (확실하게)
      const { setAccessToken: saveToken } = await import('../services/authService');
      saveToken(tokenResponse.access_token);
      
      if (tokenResponse.refresh_token) {
        const { setRefreshToken: saveRefreshToken } = await import('../services/authService');
        saveRefreshToken(tokenResponse.refresh_token);
      }
      
      // 토큰이 localStorage에 제대로 저장되었는지 확인
      const savedToken = checkAuthenticated();
      logger.info('Token saved, verifying:', { hasToken: !!savedToken });
      
      if (!savedToken) {
        throw new Error('토큰이 저장되지 않았습니다.');
      }
      
      // localStorage에서 직접 토큰 확인
      const directToken = localStorage.getItem('access_token');
      logger.info('Direct token check:', { hasToken: !!directToken, tokenLength: directToken?.length });
      
      if (!directToken) {
        throw new Error('토큰이 localStorage에 저장되지 않았습니다.');
      }
      
      // 토큰이 저장된 후 충분히 대기 (다른 useEffect가 실행되지 않도록)
      await new Promise(resolve => setTimeout(resolve, 100));
      
      // 다시 한 번 토큰 확인
      const tokenBeforeRequest = localStorage.getItem('access_token');
      logger.info('Token before /auth/me request:', { hasToken: !!tokenBeforeRequest });
      
      if (!tokenBeforeRequest) {
        throw new Error('토큰이 요청 전에 사라졌습니다.');
      }
      
      const userInfo = await getCurrentUser();
      
      logger.info('User info from /auth/me:', { authenticated: userInfo.authenticated, userId: userInfo.user_id });
      
      if (!userInfo.authenticated) {
        logger.error('Authentication failed:', { userInfo });
        throw new Error('사용자 인증에 실패했습니다.');
      }
      
      // 사용자 정보 설정 (retryLoadUser useEffect가 실행되지 않도록)
      setUser(userInfo);
      setIsAuthenticated(userInfo.authenticated);
      logger.info('Login successful');
      
      // handleCallback에서 이미 사용자 정보를 가져왔으므로, 
      // retryLoadUser useEffect가 실행되지 않도록 user 상태를 먼저 설정
    } catch (err) {
      const error = err instanceof Error ? err : new Error('로그인에 실패했습니다.');
      setError(error);
      logger.error('Failed to handle OAuth callback:', error);
      setIsAuthenticated(false);
      setUser(null);
      throw error;
    } finally {
      // handleCallback 완료 후 상태 업데이트
      // user 상태가 설정되면 retryLoadUser useEffect가 실행되지 않음
      setIsLoading(false);
      isLoadingUser.current = false;
      isHandlingCallback.current = false;
      
      // handleCallback이 완료되었음을 로깅
      logger.debug('handleCallback: Completed, user state updated');
    }
  }, []);

  /**
   * 로그아웃
   */
  const logout = useCallback(() => {
    logoutService();
    setUser(null);
    setIsAuthenticated(false);
    setError(null);
    // 로그아웃 후 페이지 새로고침하여 상태 초기화
    window.location.href = '/';
  }, []);

  /**
   * 토큰 갱신
   */
  const refresh = useCallback(async () => {
    if (!checkAuthenticated()) {
      return;
    }

    setIsLoading(true);
    setError(null);

    try {
      await refreshAccessToken();
      await loadUser();
    } catch (err) {
      const error = err instanceof Error ? err : new Error('토큰 갱신에 실패했습니다.');
      setError(error);
      setIsAuthenticated(false);
      setUser(null);
      logoutService();
    } finally {
      setIsLoading(false);
    }
  }, [loadUser]);

  /**
   * 초기 로드 시 사용자 정보 조회
   */
  useEffect(() => {
    if (isInitialMount.current) {
      isInitialMount.current = false;
      if (checkAuthenticated()) {
        loadUser().catch((err) => {
          logger.error('Failed to load user on mount:', err);
        });
      } else {
        setIsLoading(false);
        setIsAuthenticated(false);
        setUser(null);
      }
    }
  }, [loadUser]);

  /**
   * 토큰이 있지만 사용자 정보가 로드되지 않은 경우 다시 조회
   */
  useEffect(() => {
    // handleCallback이 실행 중이면 retryLoadUser를 실행하지 않음
    if (isHandlingCallback.current) {
      logger.debug('retryLoadUser: Callback handling in progress, skipping');
      return;
    }

    // handleCallback에서 이미 사용자 정보를 가져왔다면 실행하지 않음
    if (isLoading) {
      logger.debug('retryLoadUser: Still loading, skipping');
      return;
    }

    if (checkAuthenticated() && !user && !isInitialMount.current) {
      logger.debug('retryLoadUser: Token found but no user, loading user info');
      const retryLoadUser = async () => {
        try {
          await loadUser();
        } catch (err) {
          logger.error('Failed to reload user:', err);
          setTimeout(() => {
            // handleCallback이 실행 중이거나 로딩 중이면 재시도하지 않음
            if (checkAuthenticated() && !user && !isHandlingCallback.current && !isLoading) {
              loadUser().catch((err) => {
                logger.error('Retry failed to reload user:', err);
              });
            }
          }, 1000);
        }
      };
      retryLoadUser();
    }
  }, [isLoading, user, loadUser]);

  return {
    user,
    isAuthenticated,
    isLoading,
    error,
    login,
    logout,
    handleCallback,
    refresh,
    loadUser,
  };
}

