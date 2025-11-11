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

  /**
   * 사용자 정보 조회
   */
  const loadUser = useCallback(async () => {
    if (!checkAuthenticated()) {
      setIsLoading(false);
      setIsAuthenticated(false);
      setUser(null);
      return;
    }

    setIsLoading(true);
    setError(null);

    try {
      const userInfo = await getCurrentUser();
      setUser(userInfo);
      setIsAuthenticated(userInfo.authenticated);
      if (!userInfo.authenticated) {
        logger.warn('User info indicates not authenticated, clearing tokens');
        logoutService();
      }
    } catch (err) {
      const error = err instanceof Error ? err : new Error('사용자 정보를 불러올 수 없습니다.');
      logger.error('Failed to load user:', error);
      
      const axiosError = err as any;
      if (axiosError?.response?.status === 401) {
        setError(error);
        setIsAuthenticated(false);
        setUser(null);
        logoutService();
      } else {
        setError(error);
        setIsAuthenticated(false);
        setUser(null);
      }
    } finally {
      setIsLoading(false);
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
    setIsLoading(true);
    setError(null);

    try {
      await handleOAuthCallback(code, state);
      
      if (!checkAuthenticated()) {
        throw new Error('토큰이 저장되지 않았습니다.');
      }
      
      await new Promise(resolve => setTimeout(resolve, 200));
      
      const userInfo = await getCurrentUser();
      
      if (!userInfo.authenticated) {
        throw new Error('사용자 인증에 실패했습니다.');
      }
      
      setUser(userInfo);
      setIsAuthenticated(userInfo.authenticated);
      logger.info('Login successful');
    } catch (err) {
      const error = err instanceof Error ? err : new Error('로그인에 실패했습니다.');
      setError(error);
      logger.error('Failed to handle OAuth callback:', error);
      setIsAuthenticated(false);
      setUser(null);
      throw error;
    } finally {
      setIsLoading(false);
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
    if (!isLoading && checkAuthenticated() && !user && !isInitialMount.current) {
      const retryLoadUser = async () => {
        try {
          await loadUser();
        } catch (err) {
          logger.error('Failed to reload user:', err);
          setTimeout(() => {
            if (checkAuthenticated() && !user) {
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

