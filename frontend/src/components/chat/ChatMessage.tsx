/**
 * 채팅 메시지 컴포넌트
 */
import { Copy, Check, ThumbsUp, ThumbsDown, Loader2, RefreshCw } from 'lucide-react';
import { useState } from 'react';
import ReactMarkdown from 'react-markdown';
import { Prism as SyntaxHighlighter } from 'react-syntax-highlighter';
import { vscDarkPlus } from 'react-syntax-highlighter/dist/esm/styles/prism';
import { formatRelativeTime } from '../../utils/dateUtils';
import { FileAttachment } from '../common/FileAttachment';
import { CompactReferencesBadge } from './CompactReferencesBadge';
import { RelatedQuestions } from './RelatedQuestions';
import { ErrorMessage } from './ErrorMessage';
import { sendFeedback, ratingToNumber } from '../../services/feedbackService';
import { useTypingEffect } from '../../hooks/useTypingEffect';
import logger from '../../utils/logger';
import type { ChatMessage as ChatMessageType } from '../../types/chat';
import type { StreamError } from '../../types/error';

function escapeHtml(text: string): string {
  const div = document.createElement('div');
  div.textContent = text;
  return div.innerHTML;
}

interface ChatMessageProps {
  message: ChatMessageType;
  sessionId?: string;
  onQuestionClick?: (question: string) => void;
  isStreaming?: boolean; // 스트리밍 중인지 여부
  error?: StreamError; // 에러 상태
  onRetry?: () => void; // 재시도 핸들러
}

export function ChatMessage({ 
  message, 
  sessionId, 
  onQuestionClick, 
  isStreaming = false,
  error,
  onRetry
}: ChatMessageProps) {
  const [copied, setCopied] = useState(false);
  const [feedback, setFeedback] = useState<'positive' | 'negative' | null>(null);
  
  // 안전성 검사: message가 없거나 필수 필드가 없으면 렌더링하지 않음
  if (!message || !message.id) {
    return null;
  }
  
  // content가 없으면 빈 문자열로 처리
  const content = message.content || '';
  
  // 타이핑 효과 적용 (스트리밍 중일 때만 활성화)
  const { displayed: displayedContent, isComplete: isTypingComplete } = useTypingEffect(
    content,
    {
      speed: 50, // 50ms마다 한 글자씩 표시 (더 느리게)
      enabled: isStreaming // 스트리밍 중일 때만 타이핑 효과 활성화
    }
  );

  // 타이핑 효과가 완료되고 스트리밍이 종료되었을 때만 마크다운 렌더링
  const shouldRenderMarkdown = !isStreaming && isTypingComplete;

  if (import.meta.env.DEV) {
    if (isStreaming) {
      logger.debug('[ChatMessage] Streaming:', {
        messageId: message.id,
        contentLength: content.length,
        displayedLength: displayedContent.length,
        isStreaming,
        isTypingComplete,
        displayedContent: displayedContent.substring(0, 50) + '...'
      });
    } else if (message.role === 'assistant' && content.length > 0) {
      logger.debug('[ChatMessage] Not streaming:', {
        messageId: message.id,
        contentLength: content.length,
        displayedLength: displayedContent.length,
        isStreaming,
        isTypingComplete,
        shouldRenderMarkdown
      });
    }
  }

  const handleCopy = async () => {
    try {
      await navigator.clipboard.writeText(content);
      setCopied(true);
      setTimeout(() => setCopied(false), 2000);
    } catch (err) {
      logger.error('Failed to copy:', err);
    }
  };

  const handleFeedback = async (rating: 'positive' | 'negative') => {
    if (feedback === rating) {
      // 이미 선택한 피드백이면 취소
      setFeedback(null);
      return;
    }

    setFeedback(rating);

    try {
      // 세션 ID는 메시지 메타데이터에서 가져오거나, 부모 컴포넌트에서 전달받아야 함
      // 현재는 임시로 빈 문자열 사용
      await sendFeedback({
        message_id: message.id,
        session_id: sessionId || message.metadata?.session_id || '',
        rating: ratingToNumber(rating),
      });
    } catch (error) {
      logger.error('Failed to send feedback:', error);
      setFeedback(null);
    }
  };

  const isUser = message.role === 'user';
  const isProgress = message.role === 'progress';
  const metadata = message.metadata || {};

  // assistant 메시지이고 content가 비어있으면 렌더링하지 않음
  if (!isUser && !isProgress && !content) {
    return null;
  }

  // progress 타입 메시지 처리
  if (isProgress) {
    return (
      <div className="flex items-center justify-start py-2 px-4">
        <div className="flex items-center gap-2 text-slate-400 text-sm">
          <Loader2 className="w-4 h-4 animate-spin" />
          <span>{escapeHtml(content)}</span>
          {metadata.step && (
            <span className="text-xs text-slate-300">
              ({metadata.step})
            </span>
          )}
        </div>
      </div>
    );
  }

  return (
    <div
      className={`flex gap-3 mb-4 ${
        isUser ? 'justify-end' : 'justify-start'
      }`}
    >
      {!isUser && (
        <div className="w-8 h-8 bg-blue-100 rounded-full flex items-center justify-center flex-shrink-0">
          <span className="text-blue-600 text-sm">AI</span>
        </div>
      )}
      <div
        className={`max-w-[80%] rounded-xl p-5 shadow-sm border ${
          isUser
            ? 'bg-blue-50 border-blue-200'
            : 'bg-white border-slate-200'
        }`}
      >
        {message.attachments && message.attachments.length > 0 && (
          <div className="mb-3 flex flex-wrap gap-2">
            {message.attachments.map((att) => (
              <FileAttachment
                key={att.id}
                attachment={att}
                showRemove={false}
              />
            ))}
          </div>
        )}
        <div 
          className={`leading-relaxed ${isUser ? 'text-slate-800 whitespace-pre-wrap' : 'prose prose-slate max-w-none text-slate-800'}`}
        >
          {isUser ? (
            // 사용자 메시지는 일반 텍스트로 표시 (XSS 방지를 위해 이스케이프)
            <div className="whitespace-pre-wrap">{escapeHtml(content)}</div>
          ) : !shouldRenderMarkdown ? (
            // 타이핑 효과가 완료되기 전까지는 타이핑 효과로 표시
            <div className="whitespace-pre-wrap">{displayedContent}</div>
          ) : (
            // 타이핑 효과 완료 후 마크다운으로 렌더링
            <ReactMarkdown
              components={{
                // 코드 블록 스타일링
                code: ({ className, children, ...props }: any) => {
                  const match = /language-(\w+)/.exec(className || '');
                  const isInline = !match;
                  
                  if (!isInline && match) {
                    return (
                      <SyntaxHighlighter
                        style={vscDarkPlus as any}
                        language={match[1]}
                        PreTag="div"
                        className="rounded-lg my-2"
                      >
                        {String(children).replace(/\n$/, '')}
                      </SyntaxHighlighter>
                    );
                  }
                  
                  return (
                    <code className="bg-slate-100 px-1.5 py-0.5 rounded text-sm font-mono" {...props}>
                      {children}
                    </code>
                  );
                },
                // 링크 스타일링
                a: ({ node, ...props }) => (
                  <a className="text-blue-600 hover:text-blue-800 underline" target="_blank" rel="noopener noreferrer" {...props} />
                ),
                // 리스트 스타일링
                ul: ({ node, ...props }) => (
                  <ul className="list-disc list-inside my-2 space-y-1" {...props} />
                ),
                ol: ({ node, ...props }) => (
                  <ol className="list-decimal list-inside my-2 space-y-1" {...props} />
                ),
                // 제목 스타일링
                h1: ({ node, ...props }) => (
                  <h1 className="text-2xl font-bold mt-4 mb-2" {...props} />
                ),
                h2: ({ node, ...props }) => (
                  <h2 className="text-xl font-bold mt-3 mb-2" {...props} />
                ),
                h3: ({ node, ...props }) => (
                  <h3 className="text-lg font-semibold mt-2 mb-1" {...props} />
                ),
                // 강조 스타일링
                strong: ({ node, ...props }) => (
                  <strong className="font-semibold" {...props} />
                ),
                em: ({ node, ...props }) => (
                  <em className="italic" {...props} />
                ),
                // 인용구 스타일링
                blockquote: ({ node, ...props }) => (
                  <blockquote className="border-l-4 border-slate-300 pl-4 italic my-2 text-slate-600" {...props} />
                ),
                // 테이블 스타일링
                table: ({ node, ...props }) => (
                  <div className="overflow-x-auto my-2">
                    <table className="min-w-full border-collapse border border-slate-300" {...props} />
                  </div>
                ),
                th: ({ node, ...props }) => (
                  <th className="border border-slate-300 px-4 py-2 bg-slate-100 font-semibold" {...props} />
                ),
                td: ({ node, ...props }) => (
                  <td className="border border-slate-300 px-4 py-2" {...props} />
                ),
                // 수평선 스타일링
                hr: ({ node, ...props }) => (
                  <hr className="my-4 border-slate-300" {...props} />
                ),
                // 단락 스타일링
                p: ({ node, ...props }) => (
                  <p className="my-2" {...props} />
                ),
              }}
            >
              {content}
            </ReactMarkdown>
          )}
        </div>

        {/* 에러 메시지 표시 */}
        {error && !isUser && (
          <div className="mt-3">
            <ErrorMessage error={error} onRetry={onRetry} />
          </div>
        )}

        {!isUser && (
          <>
            <CompactReferencesBadge
              references={metadata.sources}
              legalReferences={metadata.legal_references}
              sources={metadata.sources}
            />
            <RelatedQuestions
              questions={metadata.related_questions}
              onQuestionClick={onQuestionClick}
            />
          </>
        )}

        <div className="flex items-center justify-between mt-2">
          <span className="text-xs text-slate-500">
            {message.timestamp ? formatRelativeTime(message.timestamp) : '방금 전'}
          </span>
          <div className="flex items-center gap-2">
            {/* 에러가 발생한 assistant 메시지에 재시도 버튼 표시 */}
            {!isUser && error && error.canRetry && onRetry && (
              <button
                onClick={onRetry}
                className="p-1 hover:bg-blue-100 rounded transition-colors text-blue-600"
                title="다시 시도"
              >
                <RefreshCw className="w-4 h-4" />
              </button>
            )}
            
            {!isUser && (
              <>
                <button
                  onClick={() => handleFeedback('positive')}
                  className={`p-1 rounded transition-colors ${
                    feedback === 'positive'
                      ? 'bg-green-100 text-green-600'
                      : 'hover:bg-slate-100 text-slate-400'
                  }`}
                  title="좋아요"
                >
                  <ThumbsUp className="w-4 h-4" />
                </button>
                <button
                  onClick={() => handleFeedback('negative')}
                  className={`p-1 rounded transition-colors ${
                    feedback === 'negative'
                      ? 'bg-red-100 text-red-600'
                      : 'hover:bg-slate-100 text-slate-400'
                  }`}
                  title="싫어요"
                >
                  <ThumbsDown className="w-4 h-4" />
                </button>
              </>
            )}
            {!isUser && (
              <button
                onClick={handleCopy}
                className="ml-2 p-1 hover:bg-slate-100 rounded transition-colors"
                title="복사"
              >
                {copied ? (
                  <Check className="w-4 h-4 text-green-600" />
                ) : (
                  <Copy className="w-4 h-4 text-slate-400" />
                )}
              </button>
            )}
          </div>
        </div>
      </div>
      {isUser && (
        <div className="w-8 h-8 bg-slate-200 rounded-full flex items-center justify-center flex-shrink-0">
          <span className="text-slate-600 text-sm">U</span>
        </div>
      )}
    </div>
  );
}

