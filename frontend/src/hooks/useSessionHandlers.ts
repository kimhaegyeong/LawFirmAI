/**
 * 세션 핸들러 훅
 */
import { useCallback } from 'react';
import { exportHistory, downloadHistory } from '../services/historyService';
import { createSession } from '../services/sessionService';
import logger from '../utils/logger';
import type { Session, SessionListQuery, SessionListResponse } from '../types/session';

interface UseSessionHandlersOptions {
  loadSession: (sessionId: string) => Promise<Session>;
  updateSession: (sessionId: string, data: { title?: string }) => Promise<Session>;
  removeSession: (sessionId: string) => Promise<void>;
  loadSessions: (query?: SessionListQuery) => Promise<SessionListResponse>;
  clearSession: () => void;
  clearMessages: () => void;
  currentSession: Session | null;
  setInputResetTrigger: (updater: (prev: number) => number) => void;
}

export function useSessionHandlers({
  loadSession,
  updateSession,
  removeSession,
  loadSessions,
  clearSession,
  clearMessages,
  currentSession,
  setInputResetTrigger,
}: UseSessionHandlersOptions) {
  const handleNewSession = useCallback(async () => {
    logger.debug('[DEBUG] handleNewSession called');
    try {
      clearSession();
      clearMessages();
      setInputResetTrigger((prev) => prev + 1);
      logger.debug('[DEBUG] Session and messages cleared for welcome screen');
    } catch (error) {
      logger.error('[ERROR] Failed to reset session:', error);
      const errorMessage = error instanceof Error 
        ? error.message 
        : '새 상담을 시작할 수 없습니다. 다시 시도해주세요.';
      alert(errorMessage);
    }
  }, [clearSession, clearMessages, setInputResetTrigger]);

  const handleSelectSession = useCallback(async (session: Session) => {
    try {
      await loadSession(session.session_id);
    } catch (error) {
      logger.error('Failed to load session:', error);
    }
  }, [loadSession]);

  const handleRenameSession = useCallback(async (session: Session) => {
    const newTitle = prompt('새 제목을 입력하세요:', session.title || '');
    if (newTitle && newTitle !== session.title) {
      try {
        await updateSession(session.session_id, { title: newTitle });
      } catch (error) {
        logger.error('Failed to rename session:', error);
      }
    }
  }, [updateSession]);

  const handleCopySession = useCallback(async (session: Session) => {
    try {
      const newSession = await createSession({
        title: `${session.title || '제목 없음'} (복사본)`,
      });
      
      await loadSessions();
      await loadSession(newSession.session_id);
      
      alert('세션이 복사되었습니다. 메시지는 새 대화에서 다시 전송해야 합니다.');
    } catch (error) {
      logger.error('Failed to copy session:', error);
      alert('세션 복사에 실패했습니다.');
    }
  }, [loadSessions, loadSession]);

  const handleExportSession = useCallback(async (session: Session) => {
    try {
      const blob = await exportHistory({
        session_ids: [session.session_id],
        format: 'txt',
      });
      
      const filename = `세션_${session.title || session.session_id}_${new Date().toISOString().split('T')[0]}.txt`;
      downloadHistory(blob, filename);
    } catch (error) {
      logger.error('Failed to export session:', error);
      alert('세션 내보내기에 실패했습니다.');
    }
  }, []);

  const handleDeleteSession = useCallback(async (session: Session) => {
    if (confirm(`정말 이 세션을 삭제하시겠습니까?\n"${session.title || '제목 없음'}"`)) {
      try {
        await removeSession(session.session_id);
        if (currentSession?.session_id === session.session_id) {
          clearMessages();
        }
      } catch (error) {
        logger.error('Failed to delete session:', error);
      }
    }
  }, [removeSession, currentSession, clearMessages]);

  return {
    handleNewSession,
    handleSelectSession,
    handleRenameSession,
    handleCopySession,
    handleExportSession,
    handleDeleteSession,
  };
}

