/**
 * 채팅 히스토리 컴포넌트
 */
import { useEffect, useRef } from 'react';
import { ChatMessage } from './ChatMessage';
import { LoadingSpinner } from '../common/LoadingSpinner';
import type { ChatMessage as ChatMessageType } from '../../types/chat';

interface ChatHistoryProps {
  messages: ChatMessageType[];
  sessionId?: string;
  isLoading?: boolean;
  onQuestionClick?: (question: string) => void;
}

export function ChatHistory({ messages, sessionId, isLoading = false, onQuestionClick }: ChatHistoryProps) {
  const messagesEndRef = useRef<HTMLDivElement>(null);
  const containerRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    // 새 메시지가 추가될 때 스크롤
    if (messagesEndRef.current && containerRef.current) {
      const container = containerRef.current.closest('.overflow-y-auto');
      if (container) {
        container.scrollTo({
          top: container.scrollHeight,
          behavior: 'smooth'
        });
      } else {
        messagesEndRef.current.scrollIntoView({ behavior: 'smooth' });
      }
    }
  }, [messages]);

  if (messages.length === 0 && !isLoading) {
    return null;
  }

  return (
    <div ref={containerRef} className="h-full bg-gradient-to-b from-slate-50 to-white">
      <div className="max-w-4xl mx-auto px-6 py-8">
        <div className="space-y-4">
          {messages.map((message) => (
            <ChatMessage
              key={message.id}
              message={message}
              sessionId={sessionId}
              onQuestionClick={onQuestionClick}
            />
          ))}
          {isLoading && (
            <div className="flex items-center justify-center py-4">
              <LoadingSpinner size="md" />
              <span className="ml-2 text-sm text-slate-600">답변을 생성하고 있습니다...</span>
            </div>
          )}
          <div ref={messagesEndRef} />
        </div>
      </div>
    </div>
  );
}

