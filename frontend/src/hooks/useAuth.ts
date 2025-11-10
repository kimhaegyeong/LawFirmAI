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
  const [isAuthenticated, setIsAuthenticated] = useState(false);
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
    } catch (err) {
      const error = err instanceof Error ? err : new Error('사용자 정보를 불러올 수 없습니다.');
      setError(error);
      setIsAuthenticated(false);
      setUser(null);
      logoutService();
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
      await new Promise(resolve => setTimeout(resolve, 100));
      const userInfo = await getCurrentUser();
      setUser(userInfo);
      setIsAuthenticated(userInfo.authenticated);
      logger.info('Login successful, user info loaded:', userInfo);
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
      loadUser();
    }
  }, [loadUser]);

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

