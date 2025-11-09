/**
 * 세션 관련 훅
 */
import { useState, useCallback, useEffect } from 'react';
import {
  getSessions,
  createSession,
  getSession,
  updateSession,
  deleteSession,
  generateSessionTitle,
  getSessionsByDate,
} from '../services/sessionService';
import logger from '../utils/logger';
import type {
  Session,
  SessionCreate,
  SessionUpdate,
  SessionListQuery,
} from '../types/session';

export function useSession() {
  const [currentSession, setCurrentSession] = useState<Session | null>(null);
  const [sessions, setSessions] = useState<Session[]>([]);
  const [isLoading, setIsLoading] = useState(false);
  const [error, setError] = useState<Error | null>(null);

  /**
   * 세션 목록 조회
   */
  const loadSessions = useCallback(async (query?: SessionListQuery) => {
    setIsLoading(true);
    setError(null);

    try {
      const response = await getSessions(query);
      setSessions(response.sessions);
      return response;
    } catch (err) {
      const error = err instanceof Error ? err : new Error('세션 목록을 불러올 수 없습니다.');
      setError(error);
      throw error;
    } finally {
      setIsLoading(false);
    }
  }, []);

  /**
   * 새 세션 생성
   */
  const newSession = useCallback(async (data?: SessionCreate, skipLoadSessions: boolean = false) => {
    setIsLoading(true);
    setError(null);

    try {
      const session = await createSession(data || {});
      setCurrentSession(session);
      
      // skipLoadSessions가 true이면 세션 목록 새로고침을 완전히 건너뜀
      // 대화형 인터페이스로 넘어갈 때는 세션 목록이 필요 없으므로
      // 세션 목록은 사이드바에서 필요할 때만 로드됨
      if (!skipLoadSessions) {
        await loadSessions();
      }
      // skipLoadSessions가 true인 경우는 아무것도 하지 않음
      // 세션 목록은 사이드바에서 필요할 때만 로드됨
      
      return session;
    } catch (err) {
      const error = err instanceof Error ? err : new Error('세션을 생성할 수 없습니다.');
      setError(error);
      throw error;
    } finally {
      setIsLoading(false);
    }
  }, [loadSessions]);

  /**
   * 세션 불러오기
   */
  const loadSession = useCallback(async (sessionId: string) => {
    setIsLoading(true);
    setError(null);

    try {
      const session = await getSession(sessionId);
      setCurrentSession(session);
      return session;
    } catch (err) {
      const error = err instanceof Error ? err : new Error('세션을 불러올 수 없습니다.');
      setError(error);
      throw error;
    } finally {
      setIsLoading(false);
    }
  }, []);

  /**
   * 세션 업데이트
   */
  const updateCurrentSession = useCallback(
    async (sessionId: string, data: SessionUpdate) => {
      setIsLoading(true);
      setError(null);

      try {
        const session = await updateSession(sessionId, data);
        if (currentSession?.session_id === sessionId) {
          setCurrentSession(session);
        }
        await loadSessions(); // 세션 목록 새로고침
        return session;
      } catch (err) {
        const error = err instanceof Error ? err : new Error('세션을 업데이트할 수 없습니다.');
        setError(error);
        throw error;
      } finally {
        setIsLoading(false);
      }
    },
    [currentSession, loadSessions]
  );

  /**
   * 세션 삭제
   */
  const removeSession = useCallback(
    async (sessionId: string) => {
      setIsLoading(true);
      setError(null);

      try {
        await deleteSession(sessionId);
        if (currentSession?.session_id === sessionId) {
          setCurrentSession(null);
        }
        await loadSessions(); // 세션 목록 새로고침
      } catch (err) {
        const error = err instanceof Error ? err : new Error('세션을 삭제할 수 없습니다.');
        setError(error);
        throw error;
      } finally {
        setIsLoading(false);
      }
    },
    [currentSession, loadSessions]
  );

  /**
   * 세션 제목 생성
   */
  const generateTitle = useCallback(
    async (sessionId: string) => {
      setIsLoading(true);
      setError(null);

      try {
        const session = await generateSessionTitle(sessionId);
        if (currentSession?.session_id === sessionId) {
          setCurrentSession(session);
        }
        await loadSessions(); // 세션 목록 새로고침
        return session;
      } catch (err) {
        const error = err instanceof Error ? err : new Error('제목을 생성할 수 없습니다.');
        setError(error);
        throw error;
      } finally {
        setIsLoading(false);
      }
    },
    [currentSession, loadSessions]
  );

  /**
   * 세션 초기화
   */
  const clearSession = useCallback(() => {
    setCurrentSession(null);
  }, []);

  /**
   * 날짜별 그룹 세션 조회
   */
  const loadSessionsByDate = useCallback(async (
    dateGroup: 'today' | 'yesterday' | 'week' | 'month' | 'older',
    page: number = 1,
    pageSize: number = 20,
    search?: string
  ) => {
    setIsLoading(true);
    setError(null);

    try {
      const response = await getSessionsByDate(dateGroup, page, pageSize, search);
      return response;
    } catch (err) {
      const error = err instanceof Error ? err : new Error('세션 목록을 불러올 수 없습니다.');
      setError(error);
      throw error;
    } finally {
      setIsLoading(false);
    }
  }, []);

  return {
    currentSession,
    sessions,
    isLoading,
    error,
    loadSessions,
    newSession,
    loadSession,
    updateSession: updateCurrentSession,
    removeSession,
    generateTitle,
    clearSession,
    loadSessionsByDate,
  };
}

