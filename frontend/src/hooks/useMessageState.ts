/**
 * 메시지 상태 관리 훅
 */
import { useState, useCallback } from 'react';
import type { ChatMessage } from '../types/chat';
import type { StreamError } from '../types/error';

export function useMessageState() {
  const [messages, setMessages] = useState<ChatMessage[]>([]);
  const [streamingMessageId, setStreamingMessageId] = useState<string | null>(null);
  const [streamErrors, setStreamErrors] = useState<Map<string, StreamError>>(new Map());

  const addMessage = useCallback((message: ChatMessage) => {
    setMessages((prev) => [...prev, message]);
  }, []);

  const updateMessage = useCallback((messageId: string, updater: (message: ChatMessage) => ChatMessage) => {
    setMessages((prev) => {
      const index = prev.findIndex((msg) => msg.id === messageId);
      if (index === -1) return prev;
      
      const updated = [...prev];
      // eslint-disable-next-line security/detect-object-injection
      const existingMsg = updated[index];
      if (existingMsg) {
        // eslint-disable-next-line security/detect-object-injection
        updated[index] = updater(existingMsg);
      }
      return updated;
    });
  }, []);

  const updateMessages = useCallback((updater: (messages: ChatMessage[]) => ChatMessage[]) => {
    setMessages(updater);
  }, []);

  const removeMessage = useCallback((messageId: string) => {
    setMessages((prev) => prev.filter((msg) => msg.id !== messageId));
  }, []);

  const clearMessages = useCallback(() => {
    setMessages([]);
  }, []);

  const setStreamingId = useCallback((messageId: string | null) => {
    setStreamingMessageId(messageId);
  }, []);

  const addError = useCallback((messageId: string, error: StreamError) => {
    setStreamErrors((prev) => {
      const newMap = new Map(prev);
      newMap.set(messageId, error);
      return newMap;
    });
  }, []);

  const removeError = useCallback((messageId: string) => {
    setStreamErrors((prev) => {
      const newMap = new Map(prev);
      newMap.delete(messageId);
      return newMap;
    });
  }, []);

  const clearErrors = useCallback(() => {
    setStreamErrors(new Map());
  }, []);

  return {
    messages,
    streamingMessageId,
    streamErrors,
    addMessage,
    updateMessage,
    updateMessages,
    removeMessage,
    clearMessages,
    setStreamingId,
    addError,
    removeError,
    clearErrors,
  };
}

