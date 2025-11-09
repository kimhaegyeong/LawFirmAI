/**
 * 채팅 히스토리 컴포넌트
 */
import { useEffect, useRef } from 'react';
import { ChatMessage } from './ChatMessage';
import { ErrorBoundary } from '../common/ErrorBoundary';
import type { ChatMessage as ChatMessageType } from '../../types/chat';
import type { StreamError } from '../../types/error';

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
  onContinueReading?: (sessionId: string, messageId: string, chunkIndex: number) => Promise<void>; // 계속 읽기 핸들러
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
  onOpenReferencesSidebar,
  onContinueReading
}: ChatHistoryProps) {
  const messagesEndRef = useRef<HTMLDivElement>(null);
  const containerRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    // 새 메시지가 추가되거나 업데이트될 때 스크롤 (스트리밍 중에도 실시간 스크롤)
    if (messagesEndRef.current && containerRef.current) {
      const container = containerRef.current.closest('.overflow-y-auto');
      if (container) {
        // 스트리밍 중에는 즉시 스크롤 (smooth 없이)
        // 마지막 메시지가 progress이거나 isLoading이면 스트리밍 중으로 간주
        const lastMessage = messages[messages.length - 1];
        const isStreaming = isLoading || lastMessage?.role === 'progress' || currentProgress !== null;
        container.scrollTo({
          top: container.scrollHeight,
          behavior: isStreaming ? 'auto' : 'smooth'
        });
      } else {
        messagesEndRef.current.scrollIntoView({ behavior: 'auto' });
      }
    }
  }, [messages, currentProgress, isLoading]);

  if (messages.length === 0 && !isLoading) {
    return null;
  }

  return (
    <div ref={containerRef} className="h-full bg-gradient-to-b from-slate-50 to-white">
      <div className="max-w-4xl mx-auto px-6 py-8">
        <div className="space-y-4">
          {messages.map((message) => {
            // 특정 메시지가 스트리밍 중이면 isStreaming prop 전달
            // streamingMessageId가 null이 아니고 메시지 ID와 일치하면 스트리밍 중
            // streamingMessageId가 null이면 스트리밍 완료로 간주하여 isStreaming: false
            // 이렇게 하면 ChatMessage에서 Markdown 렌더링이 활성화됨
            const isStreaming = message.role === 'assistant' && 
              streamingMessageId !== null &&
              message.id === streamingMessageId;
            
            // 에러 상태 확인
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
                  onContinueReading={onContinueReading}
                />
              </ErrorBoundary>
            );
          })}
          <div ref={messagesEndRef} />
        </div>
      </div>
    </div>
  );
}

