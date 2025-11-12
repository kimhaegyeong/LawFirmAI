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

// 모듈 레벨에서 전역 호출 추적 (React StrictMode 대응)
let globalLoadingPromise: Promise<UserInfo | null> | null = null;
let globalHasLoaded = false;

export function useAuth() {
  const [user, setUser] = useState<UserInfo | null>(null);
  const [isAuthenticated, setIsAuthenticated] = useState(() => {
    return checkAuthenticated();
  });
  const [isLoading, setIsLoading] = useState(true);
  const [error, setError] = useState<Error | null>(null);
  const isInitialMount = useRef(true);
  const isInitialMountComplete = useRef(false); // 초기 마운트 완료 여부 추적
  const isLoadingUser = useRef(false);
  const isHandlingCallback = useRef(false);
  const hasLoadedUser = useRef(false); // 사용자 정보가 이미 로드되었는지 추적
  const userRef = useRef<UserInfo | null>(null); // 최신 user 상태 추적
  const isAuthenticatedRef = useRef<boolean>(false); // 최신 isAuthenticated 상태 추적

  // user와 isAuthenticated 상태 변경 시 ref 업데이트
  useEffect(() => {
    userRef.current = user;
    isAuthenticatedRef.current = isAuthenticated;
  }, [user, isAuthenticated]);

  /**
   * 사용자 정보 조회
   */
  const loadUser = useCallback(async () => {
    // 전역 로딩 중이면 기존 Promise 반환
    if (globalLoadingPromise) {
      logger.debug('loadUser: Global loading in progress, waiting for existing call');
      try {
        const userInfo = await globalLoadingPromise;
        if (userInfo) {
          setUser(userInfo);
          setIsAuthenticated(userInfo.authenticated);
          hasLoadedUser.current = true;
        }
      } catch (err) {
        logger.error('loadUser: Error from global loading promise:', err);
      }
      return;
    }

    if (isLoadingUser.current) {
      logger.debug('loadUser: Already loading, skipping duplicate call');
      return;
    }

    // handleCallback이 실행 중이면 토큰을 삭제하지 않음
    if (isHandlingCallback.current) {
      logger.debug('loadUser: Callback handling in progress, skipping to avoid token deletion');
      return;
    }

    // 사용자 정보가 이미 로드되었고 인증 상태가 유지되면 다시 호출하지 않음
    if (globalHasLoaded && hasLoadedUser.current && userRef.current && isAuthenticatedRef.current) {
      logger.debug('loadUser: User already loaded globally, skipping duplicate call');
      return;
    }

    if (!checkAuthenticated()) {
      setIsLoading(false);
      setIsAuthenticated(false);
      setUser(null);
      hasLoadedUser.current = false;
      return;
    }

    isLoadingUser.current = true;
    setIsLoading(true);
    setError(null);

    // 전역 Promise 생성
    globalLoadingPromise = (async () => {
      try {
        const userInfo = await getCurrentUser();
        globalHasLoaded = true;
        return userInfo;
      } catch (err) {
        globalHasLoaded = false;
        throw err;
      } finally {
        globalLoadingPromise = null;
      }
    })();

    try {
      const userInfo = await globalLoadingPromise;
      setUser(userInfo);
      setIsAuthenticated(userInfo.authenticated);
      hasLoadedUser.current = true;
      
      // handleCallback이 실행 중이면 토큰을 삭제하지 않음
      if (!userInfo.authenticated && !isHandlingCallback.current) {
        logger.warn('User info indicates not authenticated, clearing tokens');
        logoutService();
        hasLoadedUser.current = false;
        globalHasLoaded = false;
      } else if (!userInfo.authenticated && isHandlingCallback.current) {
        logger.debug('loadUser: Callback handling in progress, not clearing tokens');
      }
    } catch (err) {
      const error = err instanceof Error ? err : new Error('사용자 정보를 불러올 수 없습니다.');
      logger.error('Failed to load user:', error);
      
      const axiosError = err as any;
      hasLoadedUser.current = false;
      globalHasLoaded = false;
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
      
      // 전역 로딩 중이면 기다림
      let userInfo: UserInfo;
      if (globalLoadingPromise) {
        logger.debug('handleCallback: Global loading in progress, waiting for existing call');
        const result = await globalLoadingPromise;
        if (!result) {
          throw new Error('사용자 정보를 불러올 수 없습니다.');
        }
        userInfo = result;
      } else {
        // 전역 Promise 생성
        globalLoadingPromise = (async () => {
          try {
            const info = await getCurrentUser();
            globalHasLoaded = true;
            return info;
          } catch (err) {
            globalHasLoaded = false;
            throw err;
          } finally {
            globalLoadingPromise = null;
          }
        })();
        userInfo = await globalLoadingPromise;
      }
      
      logger.info('User info from /auth/me:', { authenticated: userInfo.authenticated, userId: userInfo.user_id });
      
      if (!userInfo.authenticated) {
        logger.error('Authentication failed:', { userInfo });
        throw new Error('사용자 인증에 실패했습니다.');
      }
      
      // 사용자 정보 설정 (retryLoadUser useEffect가 실행되지 않도록)
      setUser(userInfo);
      setIsAuthenticated(userInfo.authenticated);
      hasLoadedUser.current = true;
      globalHasLoaded = true;
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
    hasLoadedUser.current = false;
    globalHasLoaded = false;
    globalLoadingPromise = null;
    isInitialMountComplete.current = false;
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
      // 전역적으로 이미 로드되었으면 스킵
      if (globalHasLoaded && hasLoadedUser.current) {
        logger.debug('Initial mount: User already loaded globally, skipping');
        setIsLoading(false);
        isInitialMountComplete.current = true;
        return;
      }
      if (checkAuthenticated()) {
        loadUser().catch((err) => {
          logger.error('Failed to load user on mount:', err);
        }).finally(() => {
          isInitialMountComplete.current = true;
        });
      } else {
        setIsLoading(false);
        setIsAuthenticated(false);
        setUser(null);
        isInitialMountComplete.current = true;
      }
    }
  }, [loadUser]);

  /**
   * 토큰이 있지만 사용자 정보가 로드되지 않은 경우 다시 조회
   * 초기 마운트는 별도 useEffect에서 처리하므로 이 useEffect는 실행하지 않음
   */
  useEffect(() => {
    // 초기 마운트가 완료되지 않았으면 스킵
    if (!isInitialMountComplete.current) {
      logger.debug('retryLoadUser: Initial mount not complete, skipping');
      return;
    }

    // 전역적으로 이미 로드되었으면 스킵
    if (globalHasLoaded) {
      logger.debug('retryLoadUser: User already loaded globally, skipping');
      return;
    }

    // handleCallback이 실행 중이면 retryLoadUser를 실행하지 않음
    if (isHandlingCallback.current) {
      logger.debug('retryLoadUser: Callback handling in progress, skipping');
      return;
    }

    // 이미 사용자 정보가 로드되었으면 실행하지 않음
    if (hasLoadedUser.current && user) {
      logger.debug('retryLoadUser: User already loaded, skipping');
      return;
    }

    // 로딩 중이면 실행하지 않음
    if (isLoading || isLoadingUser.current || globalLoadingPromise) {
      logger.debug('retryLoadUser: Still loading, skipping');
      return;
    }

    // 토큰이 있고 사용자 정보가 없을 때만 실행
    if (checkAuthenticated() && !user) {
      logger.debug('retryLoadUser: Token found but no user, loading user info');
      loadUser().catch((err) => {
        logger.error('Failed to reload user:', err);
      });
    }
  }, [user, loadUser, isLoading]);

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

