/**
 * 스트리밍 관련 훅
 */
import { useState, useCallback, useRef } from 'react';
import { sendStreamingChatMessage } from '../services/chatService';
import type { StreamingChatRequest } from '../types/chat';

export function useStreaming() {
  const [isStreaming, setIsStreaming] = useState(false);
  const [streamingContent, setStreamingContent] = useState('');
  const [error, setError] = useState<Error | null>(null);
  const abortControllerRef = useRef<AbortController | null>(null);

  /**
   * 스트리밍 시작
   */
  const startStreaming = useCallback(
    async (
      request: StreamingChatRequest,
      onChunk?: (chunk: string, fullContent: string) => void
    ) => {
      setIsStreaming(true);
      setStreamingContent('');
      setError(null);

      abortControllerRef.current = new AbortController();

      try {
        let fullContent = '';

        for await (const chunk of sendStreamingChatMessage(request)) {
          if (abortControllerRef.current?.signal.aborted) {
            break;
          }

          fullContent += chunk;
          setStreamingContent(fullContent);
          onChunk?.(chunk, fullContent);
        }

        return fullContent;
      } catch (err) {
        const error = err instanceof Error ? err : new Error('스트리밍 중 에러가 발생했습니다.');
        setError(error);
        throw error;
      } finally {
        setIsStreaming(false);
        abortControllerRef.current = null;
      }
    },
    []
  );

  /**
   * 스트리밍 중지
   */
  const stopStreaming = useCallback(() => {
    if (abortControllerRef.current) {
      abortControllerRef.current.abort();
      abortControllerRef.current = null;
    }
    setIsStreaming(false);
  }, []);

  /**
   * 스트리밍 내용 초기화
   */
  const clearStreamingContent = useCallback(() => {
    setStreamingContent('');
  }, []);

  return {
    isStreaming,
    streamingContent,
    error,
    startStreaming,
    stopStreaming,
    clearStreamingContent,
  };
}

