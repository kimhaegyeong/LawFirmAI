/**
 * 채팅 관련 훅
 */
import { useState, useCallback } from 'react';
import { sendChatMessage, sendStreamingChatMessage } from '../services/chatService';
import logger from '../utils/logger';
import type { ChatRequest, ChatResponse, ChatMessage, FileAttachment } from '../types/chat';

interface UseChatOptions {
  onMessage?: (message: ChatMessage) => void;
  onError?: (error: Error) => void;
}

export function useChat(options?: UseChatOptions) {
  const [isLoading, setIsLoading] = useState(false);
  const [isStreaming, setIsStreaming] = useState(false);
  const [error, setError] = useState<Error | null>(null);

  /**
   * 일반 채팅 메시지 전송
   */
  const sendMessage = useCallback(
    async (
      message: string,
      sessionId?: string,
      attachments?: FileAttachment[]
    ): Promise<ChatResponse | null> => {
      setIsLoading(true);
      setError(null);

      try {
        const request: ChatRequest = {
          message,
          session_id: sessionId,
          enable_checkpoint: true,
        };

        const response = await sendChatMessage(request);

        // 사용자 메시지 생성
        const userMessage: ChatMessage = {
          id: `user-${Date.now()}`,
          role: 'user',
          content: message,
          timestamp: new Date(),
          attachments,
        };

        // AI 답변 메시지 생성
        const assistantMessage: ChatMessage = {
          id: `assistant-${Date.now()}`,
          role: 'assistant',
          content: response.answer,
          timestamp: new Date(),
          metadata: {
            sources: response.sources,
            legal_references: response.legal_references,
            confidence: response.confidence,
            processing_steps: response.processing_steps,
            query_type: response.query_type,
          },
        };

        options?.onMessage?.(userMessage);
        options?.onMessage?.(assistantMessage);

        return response;
      } catch (err) {
        const error = err instanceof Error ? err : new Error('알 수 없는 에러가 발생했습니다.');
        setError(error);
        options?.onError?.(error);
        return null;
      } finally {
        setIsLoading(false);
      }
    },
    [options]
  );

  /**
   * 스트리밍 채팅 메시지 전송
   */
  const sendStreamingMessage = useCallback(
    async (
      message: string,
      sessionId: string,
      onChunk: (chunk: string) => void,
      imageBase64?: string,
      fileBase64?: string,
      filename?: string
    ): Promise<void> => {
      setIsStreaming(true);
      setError(null);

      try {
        const request = {
          message,
          session_id: sessionId,
          ...(imageBase64 && { image_base64: imageBase64 }),
          ...(fileBase64 && { file_base64: fileBase64 }),
          ...(filename && { filename: filename }),
        };

        let chunkCount = 0;
        for await (const chunk of sendStreamingChatMessage(request)) {
          chunkCount++;
          onChunk(chunk);
        }
        
        if (chunkCount === 0) {
          logger.warn('[Stream] No chunks received from stream');
        }

        // 스트리밍 완료 - 메시지는 App.tsx에서 이미 추가하고 있으므로 여기서는 추가하지 않음
      } catch (err) {
        logger.error('[Stream] Error in sendStreamingMessage:', err);
        const error = err instanceof Error 
          ? err 
          : new Error(`스트리밍 중 오류가 발생했습니다: ${String(err)}`);
        setError(error);
        options?.onError?.(error);
      } finally {
        setIsStreaming(false);
      }
    },
    [options]
  );

  return {
    sendMessage,
    sendStreamingMessage,
    isLoading,
    isStreaming,
    error,
  };
}

