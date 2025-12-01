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
let globalInitialMountStarted = false; // 초기 마운트 시작 여부 추적
let globalUserInfo: UserInfo | null = null; // 전역 사용자 정보 저장

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
    // 전역적으로 이미 로드되었으면 즉시 스킵 (가장 먼저 체크)
    if (globalHasLoaded && globalUserInfo) {
      logger.debug('loadUser: User already loaded globally, syncing state', {
        userId: globalUserInfo.user_id,
        authenticated: globalUserInfo.authenticated
      });
      // 전역 사용자 정보로 상태 동기화
      setUser(globalUserInfo);
      setIsAuthenticated(globalUserInfo.authenticated);
      hasLoadedUser.current = true;
      userRef.current = globalUserInfo;
      isAuthenticatedRef.current = globalUserInfo.authenticated;
      return;
    }

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
    if (hasLoadedUser.current && userRef.current && isAuthenticatedRef.current) {
      logger.debug('loadUser: User already loaded, skipping duplicate call');
      return;
    }

    if (!checkAuthenticated()) {
      setIsLoading(false);
      setIsAuthenticated(false);
      setUser(null);
      hasLoadedUser.current = false;
      return;
    }

    // 전역 로딩 플래그를 먼저 설정하여 중복 호출 방지
    isLoadingUser.current = true;
    setIsLoading(true);
    setError(null);

    // 전역 Promise 생성 (동기적으로 설정하여 경쟁 조건 방지)
    if (!globalLoadingPromise) {
      globalLoadingPromise = (async () => {
        try {
          const userInfo = await getCurrentUser();
          globalHasLoaded = true;
          globalUserInfo = userInfo; // 전역 사용자 정보 저장
          return userInfo;
        } catch (err) {
          globalHasLoaded = false;
          globalUserInfo = null;
          throw err;
        } finally {
          globalLoadingPromise = null;
        }
      })();
    }

    try {
      const userInfo = await globalLoadingPromise;
      if (userInfo) {
        logger.info('loadUser: User info received', { 
          authenticated: userInfo.authenticated, 
          userId: userInfo.user_id,
          email: userInfo.email,
          name: userInfo.name 
        });
        
        // 전역 사용자 정보 업데이트
        globalUserInfo = userInfo;
        
        setUser(userInfo);
        setIsAuthenticated(userInfo.authenticated);
        hasLoadedUser.current = true;
        userRef.current = userInfo;
        isAuthenticatedRef.current = userInfo.authenticated;
        
        // 상태 업데이트 확인을 위한 로그
        logger.info('loadUser: State updated', { 
          authenticated: userInfo.authenticated,
          hasUser: !!userInfo,
          userId: userInfo.user_id 
        });
        
        // handleCallback이 실행 중이면 토큰을 삭제하지 않음
        if (!userInfo.authenticated && !isHandlingCallback.current) {
          logger.warn('User info indicates not authenticated, clearing tokens');
          logoutService();
          hasLoadedUser.current = false;
          globalHasLoaded = false;
          globalUserInfo = null;
        } else if (!userInfo.authenticated && isHandlingCallback.current) {
          logger.debug('loadUser: Callback handling in progress, not clearing tokens');
        }
      } else {
        logger.warn('loadUser: User info is null or undefined');
        globalUserInfo = null;
        setIsAuthenticated(false);
        hasLoadedUser.current = true;
      }
    } catch (err) {
      const error = err instanceof Error ? err : new Error('사용자 정보를 불러올 수 없습니다.');
      logger.error('Failed to load user:', error);
      
      // eslint-disable-next-line @typescript-eslint/no-explicit-any
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
      let userInfo: UserInfo | null;
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
            globalUserInfo = info; // 전역 사용자 정보 저장
            return info;
          } catch (err) {
            globalHasLoaded = false;
            globalUserInfo = null;
            throw err;
          } finally {
            globalLoadingPromise = null;
          }
        })();
        userInfo = await globalLoadingPromise;
      }
      
      if (!userInfo) {
        throw new Error('사용자 정보를 불러올 수 없습니다.');
      }
      
      logger.info('User info from /auth/me:', { authenticated: userInfo.authenticated, userId: userInfo.user_id });
      
      if (!userInfo.authenticated) {
        logger.error('Authentication failed:', { userInfo });
        throw new Error('사용자 인증에 실패했습니다.');
      }
      
      // 사용자 정보 설정 (retryLoadUser useEffect가 실행되지 않도록)
      globalUserInfo = userInfo; // 전역 사용자 정보 저장
      setUser(userInfo);
      setIsAuthenticated(userInfo.authenticated);
      hasLoadedUser.current = true;
      globalHasLoaded = true;
      userRef.current = userInfo;
      isAuthenticatedRef.current = userInfo.authenticated;
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
    globalUserInfo = null;
    globalLoadingPromise = null;
    globalInitialMountStarted = false;
    isInitialMountComplete.current = false;
    userRef.current = null;
    isAuthenticatedRef.current = false;
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
      
      // 전역적으로 이미 초기 마운트가 시작되었으면 스킵 (StrictMode 대응)
      if (globalInitialMountStarted) {
        logger.debug('Initial mount: Already started globally, skipping duplicate mount');
        // 이미 로딩 중이거나 로드 완료된 경우 상태 동기화
        if (globalLoadingPromise) {
          globalLoadingPromise.then((userInfo) => {
            if (userInfo) {
              setUser(userInfo);
              setIsAuthenticated(userInfo.authenticated);
              hasLoadedUser.current = true;
            }
            setIsLoading(false);
            isInitialMountComplete.current = true;
          }).catch(() => {
            setIsLoading(false);
            isInitialMountComplete.current = true;
          });
        } else if (globalHasLoaded && globalUserInfo) {
          setIsLoading(false);
          isInitialMountComplete.current = true;
          setUser(globalUserInfo);
          setIsAuthenticated(globalUserInfo.authenticated);
          hasLoadedUser.current = true;
          userRef.current = globalUserInfo;
          isAuthenticatedRef.current = globalUserInfo.authenticated;
        } else {
          setIsLoading(false);
          isInitialMountComplete.current = true;
        }
        return;
      }
      
      // 전역 초기 마운트 플래그 설정
      globalInitialMountStarted = true;
      
      // 전역적으로 이미 로드되었으면 스킵 (가장 먼저 체크)
      if (globalHasLoaded && globalUserInfo) {
        logger.debug('Initial mount: User already loaded globally, syncing state', {
          userId: globalUserInfo.user_id,
          authenticated: globalUserInfo.authenticated
        });
        setIsLoading(false);
        isInitialMountComplete.current = true;
        // 전역 사용자 정보로 상태 동기화
        setUser(globalUserInfo);
        setIsAuthenticated(globalUserInfo.authenticated);
        hasLoadedUser.current = true;
        userRef.current = globalUserInfo;
        isAuthenticatedRef.current = globalUserInfo.authenticated;
        return;
      }
      // 이미 로딩 중이면 스킵
      if (globalLoadingPromise) {
        logger.debug('Initial mount: Already loading, waiting for existing call');
        globalLoadingPromise.then((userInfo) => {
          if (userInfo) {
            globalUserInfo = userInfo;
            setUser(userInfo);
            setIsAuthenticated(userInfo.authenticated);
            hasLoadedUser.current = true;
            userRef.current = userInfo;
            isAuthenticatedRef.current = userInfo.authenticated;
          }
          setIsLoading(false);
          isInitialMountComplete.current = true;
        }).catch(() => {
          setIsLoading(false);
          isInitialMountComplete.current = true;
        });
        return;
      }
      if (checkAuthenticated()) {
        logger.info('Initial mount: Token found, loading user info');
        loadUser().catch((err) => {
          logger.error('Failed to load user on mount:', err);
        }).finally(() => {
          isInitialMountComplete.current = true;
        });
      } else {
        logger.debug('Initial mount: No token found, user not authenticated');
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
    if (globalHasLoaded && globalUserInfo) {
      logger.debug('retryLoadUser: User already loaded globally, syncing state', {
        userId: globalUserInfo.user_id,
        authenticated: globalUserInfo.authenticated
      });
      // 전역 사용자 정보로 상태 동기화
      if (!user || user.user_id !== globalUserInfo.user_id) {
        setUser(globalUserInfo);
        setIsAuthenticated(globalUserInfo.authenticated);
        hasLoadedUser.current = true;
        userRef.current = globalUserInfo;
        isAuthenticatedRef.current = globalUserInfo.authenticated;
      }
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
      logger.info('retryLoadUser: Token found but no user, loading user info', {
        hasToken: checkAuthenticated(),
        hasUser: !!user,
        isAuthenticated,
        isLoading
      });
      loadUser().catch((err) => {
        logger.error('Failed to reload user:', err);
      });
    } else if (checkAuthenticated() && user && !isAuthenticated) {
      // 토큰이 있고 사용자 정보가 있지만 인증 상태가 false인 경우
      // user.authenticated를 확인하여 상태 동기화
      logger.warn('retryLoadUser: User exists but isAuthenticated is false, syncing state', {
        userAuthenticated: user.authenticated,
        isAuthenticated,
        userId: user.user_id
      });
      if (user.authenticated) {
        setIsAuthenticated(true);
      }
    }
  }, [user, loadUser, isLoading, isAuthenticated]);

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

