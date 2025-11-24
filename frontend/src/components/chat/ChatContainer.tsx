/**
 * 채팅 컨테이너 컴포넌트
 * 채팅 메시지 표시 및 입력 처리
 */
import { ChatHistory } from './ChatHistory';
import { ChatInput } from './ChatInput';
import { WelcomeScreen } from '../welcome/WelcomeScreen';
import type { ChatMessage, FileAttachment } from '../../types/chat';
import type { StreamError } from '../../types/error';

interface ChatContainerProps {
  messages: ChatMessage[];
  sessionId?: string;
  isSending: boolean;
  isStreaming: boolean;
  currentProgress: string | null;
  progressHistory: string[];
  streamingMessageId: string | null;
  streamErrors: Map<string, StreamError>;
  quotaInfo: { remaining: number; limit: number } | null;
  isAuthenticated: boolean;
  inputResetTrigger: number;
  onQuestionClick: (question: string) => void;
  onSendMessage: (message: string, attachments?: FileAttachment[]) => void;
  onDocumentClick: (message: ChatMessage, documentIndex: number) => void;
  onOpenReferencesSidebar: (message: ChatMessage, selectedType: 'all' | 'law' | 'precedent' | 'decision' | 'interpretation' | 'regulation') => void;
  onRetryMessage: (messageId: string) => void;
  onLoginClick: () => void;
  onRetryQuestion: (question: string, messageId: string) => void;
}

export function ChatContainer({
  messages,
  sessionId,
  isSending,
  isStreaming,
  currentProgress,
  progressHistory,
  streamingMessageId,
  streamErrors,
  quotaInfo,
  isAuthenticated,
  inputResetTrigger,
  onQuestionClick,
  onSendMessage,
  onDocumentClick,
  onOpenReferencesSidebar,
  onRetryMessage,
  onLoginClick,
  onRetryQuestion,
}: ChatContainerProps) {
  return (
    <div className="flex-1 flex flex-col overflow-hidden min-h-0">
      {/* 콘텐츠 영역 - 스크롤 가능 */}
      <div className="flex-1 overflow-y-auto min-h-0">
        {messages.length === 0 && !isSending && !sessionId ? (
          <WelcomeScreen 
            onQuestionClick={onQuestionClick} 
            isAuthenticated={isAuthenticated} 
            quotaInfo={quotaInfo} 
            onLoginClick={onLoginClick} 
          />
        ) : (
          <ChatHistory
            messages={messages}
            sessionId={sessionId}
            isLoading={isSending || isStreaming}
            currentProgress={currentProgress}
            progressHistory={progressHistory}
            onQuestionClick={onQuestionClick}
            streamingMessageId={streamingMessageId}
            streamErrors={streamErrors}
            onDocumentClick={onDocumentClick}
            onOpenReferencesSidebar={onOpenReferencesSidebar}
            onRetryMessage={onRetryMessage}
            onRetryQuestion={onRetryQuestion}
          />
        )}
      </div>
      
      {/* 입력창 - 하단 고정 */}
      <div className="flex-shrink-0">
        <ChatInput 
          onSend={onSendMessage} 
          disabled={isSending || isStreaming || (!isAuthenticated && quotaInfo?.remaining === 0)} 
          isLoading={isSending || isStreaming}
          resetTrigger={inputResetTrigger}
        />
      </div>
    </div>
  );
}

