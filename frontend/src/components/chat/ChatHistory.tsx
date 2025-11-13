/**
 * 채팅 히스토리 컴포넌트
 */
import { useEffect, useRef, useState } from 'react';
import { ChatMessage } from './ChatMessage';
import { ErrorBoundary } from '../common/ErrorBoundary';
import type { ChatMessage as ChatMessageType } from '../../types/chat';
import type { StreamError } from '../../types/error';
import { ChevronDown } from 'lucide-react';

interface ChatHistoryProps {
  messages: ChatMessageType[];
  sessionId?: string;
  isLoading?: boolean;
  currentProgress?: string | null; // 기존 코드와의 호환성을 위해 유지 (나중에 제거 예정)
  progressHistory?: string[]; // 기존 코드와의 호환성을 위해 유지 (나중에 제거 예정)
  onQuestionClick?: (question: string) => void;
  streamingMessageId?: string | null; // 현재 스트리밍 중인 메시지 ID
  streamErrors?: Map<string, StreamError>; // 에러 상태 맵
  onRetryMessage?: (messageId: string) => void; // 재시도 핸들러
  onDocumentClick?: (message: ChatMessageType, documentIndex: number) => void; // 문서 클릭 핸들러
  onOpenReferencesSidebar?: (message: ChatMessageType, selectedType: 'all' | 'law' | 'precedent' | 'decision' | 'interpretation' | 'regulation') => void; // 참고자료 사이드바 열기 핸들러
}

export function ChatHistory({ 
  messages, 
  sessionId, 
  isLoading = false, 
  currentProgress, 
  progressHistory: _progressHistory = [], 
  onQuestionClick, 
  streamingMessageId = null,
  streamErrors = new Map(),
  onRetryMessage,
  onDocumentClick,
  onOpenReferencesSidebar
}: ChatHistoryProps) {
  const messagesEndRef = useRef<HTMLDivElement>(null);
  const containerRef = useRef<HTMLDivElement>(null);
  const [showScrollToBottom, setShowScrollToBottom] = useState(false);
  const autoScrollEnabledRef = useRef(true);
  const previousStreamingMessageIdRef = useRef<string | null>(null);

  const checkIfNearBottom = (container: HTMLElement): boolean => {
    const threshold = 100;
    const scrollTop = container.scrollTop;
    const scrollHeight = container.scrollHeight;
    const clientHeight = container.clientHeight;
    return scrollHeight - scrollTop - clientHeight < threshold;
  };

  const scrollToBottom = (behavior: ScrollBehavior = 'smooth') => {
    const container = containerRef.current?.closest('.overflow-y-auto');
    if (container && container instanceof HTMLElement) {
      container.scrollTo({
        top: container.scrollHeight,
        behavior
      });
      autoScrollEnabledRef.current = true;
      setShowScrollToBottom(false);
    } else if (messagesEndRef.current) {
      messagesEndRef.current.scrollIntoView({ behavior });
    }
  };

  useEffect(() => {
    const container = containerRef.current?.closest('.overflow-y-auto');
    if (!container || !(container instanceof HTMLElement)) return;

    const handleScroll = () => {
      const isNearBottom = checkIfNearBottom(container);
      
      if (isNearBottom) {
        setShowScrollToBottom(false);
        autoScrollEnabledRef.current = true;
      } else {
        setShowScrollToBottom(true);
        autoScrollEnabledRef.current = false;
      }
    };

    container.addEventListener('scroll', handleScroll);
    return () => container.removeEventListener('scroll', handleScroll);
  }, []);

  useEffect(() => {
    if (messagesEndRef.current && containerRef.current) {
      const container = containerRef.current.closest('.overflow-y-auto');
      if (container && container instanceof HTMLElement) {
        const lastMessage = messages[messages.length - 1];
        const isStreaming = streamingMessageId !== null && 
          lastMessage?.role === 'assistant' && 
          lastMessage?.id === streamingMessageId;
        
        const isFirstChunk = streamingMessageId !== null && 
          previousStreamingMessageIdRef.current !== streamingMessageId;
        
        if (isFirstChunk) {
          previousStreamingMessageIdRef.current = streamingMessageId;
          container.scrollTo({
            top: container.scrollHeight,
            behavior: 'auto'
          });
          autoScrollEnabledRef.current = true;
          setShowScrollToBottom(false);
        } else if (isStreaming) {
          const isNearBottom = checkIfNearBottom(container);
          if (isNearBottom) {
            container.scrollTo({
              top: container.scrollHeight,
              behavior: 'auto'
            });
          }
        } else {
          if (autoScrollEnabledRef.current) {
            container.scrollTo({
              top: container.scrollHeight,
              behavior: 'smooth'
            });
          }
        }
      } else if (messagesEndRef.current) {
        messagesEndRef.current.scrollIntoView({ behavior: 'auto' });
      }
    }
    
    if (streamingMessageId === null) {
      previousStreamingMessageIdRef.current = null;
    }
  }, [messages, streamingMessageId]);

  if (messages.length === 0 && !isLoading) {
    return null;
  }

  return (
    <div ref={containerRef} className="h-full bg-gradient-to-b from-slate-50 to-white relative">
      <div className="max-w-4xl mx-auto px-6 py-8">
        <div className="space-y-4">
          {messages.map((message) => {
            const isStreaming = message.role === 'assistant' && 
              streamingMessageId !== null &&
              message.id === streamingMessageId;
            
            const error = streamErrors.get(message.id);
            
            return (
              <ErrorBoundary
                key={message.id}
                fallback={
                  <div className="p-4 bg-red-50 border border-red-200 rounded-lg">
                    <div className="text-sm text-red-800">
                      메시지를 표시하는 중 오류가 발생했습니다.
                    </div>
                  </div>
                }
              >
                <ChatMessage
                  message={message}
                  sessionId={sessionId}
                  onQuestionClick={onQuestionClick}
                  onDocumentClick={onDocumentClick}
                  onOpenReferencesSidebar={onOpenReferencesSidebar}
                  isStreaming={isStreaming}
                  error={error}
                  onRetry={error && error.canRetry && onRetryMessage ? () => onRetryMessage(message.id) : undefined}
                />
              </ErrorBoundary>
            );
          })}
          <div ref={messagesEndRef} />
        </div>
      </div>

      {showScrollToBottom && (
        <button
          onClick={() => scrollToBottom('smooth')}
          className="fixed bottom-24 right-8 z-50 bg-white border border-slate-300 rounded-full p-3 shadow-lg hover:bg-slate-50 transition-all duration-200 flex items-center justify-center group"
          aria-label="맨 아래로 스크롤"
        >
          <ChevronDown className="w-5 h-5 text-slate-600 group-hover:text-slate-900" />
        </button>
      )}
    </div>
  );
}

