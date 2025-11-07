/**
 * 히스토리 관련 훅
 */
import { useState, useCallback } from 'react';
import { getHistory, exportHistory, downloadHistory } from '../services/historyService';
import type { HistoryQuery, Message } from '../types/history';
import { groupSessionsByDate } from '../utils/dateUtils';

export function useHistory() {
  const [messages, setMessages] = useState<Message[]>([]);
  const [isLoading, setIsLoading] = useState(false);
  const [error, setError] = useState<Error | null>(null);

  /**
   * 히스토리 조회
   */
  const loadHistory = useCallback(async (query: HistoryQuery) => {
    setIsLoading(true);
    setError(null);

    try {
      const response = await getHistory(query);
      setMessages(response.messages);
      return response;
    } catch (err) {
      const error = err instanceof Error ? err : new Error('히스토리를 불러올 수 없습니다.');
      setError(error);
      throw error;
    } finally {
      setIsLoading(false);
    }
  }, []);

  /**
   * 히스토리 내보내기
   */
  const exportHistoryData = useCallback(
    async (sessionIds: string[], format: 'json' | 'txt' = 'json') => {
      setIsLoading(true);
      setError(null);

      try {
        const blob = await exportHistory({ session_ids: sessionIds, format });
        const filename = `history_${new Date().toISOString().split('T')[0]}.${format}`;
        downloadHistory(blob, filename);
      } catch (err) {
        const error = err instanceof Error ? err : new Error('히스토리를 내보낼 수 없습니다.');
        setError(error);
        throw error;
      } finally {
        setIsLoading(false);
      }
    },
    []
  );

  return {
    messages,
    isLoading,
    error,
    loadHistory,
    exportHistory: exportHistoryData,
  };
}

