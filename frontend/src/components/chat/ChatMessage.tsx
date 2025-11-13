/**
 * 채팅 메시지 컴포넌트
 */
import { Copy, Check, ThumbsUp, ThumbsDown, Loader2, RefreshCw } from 'lucide-react';
import React, { useState, useMemo } from 'react';
import ReactMarkdown from 'react-markdown';
import { Prism as SyntaxHighlighter } from 'react-syntax-highlighter';
import { vscDarkPlus } from 'react-syntax-highlighter/dist/esm/styles/prism';
import { formatRelativeTime } from '../../utils/dateUtils';
import { FileAttachment } from '../common/FileAttachment';
import { CompactReferencesBadge } from './CompactReferencesBadge';
import { RelatedQuestions } from './RelatedQuestions';
import { ErrorMessage } from './ErrorMessage';
import { DocumentReference } from './DocumentReference';
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
  onDocumentClick?: (message: ChatMessageType, documentIndex: number) => void; // 문서 클릭 핸들러
  onOpenReferencesSidebar?: (message: ChatMessageType, selectedType: 'all' | 'law' | 'precedent' | 'decision' | 'interpretation' | 'regulation') => void; // 참고자료 사이드바 열기 핸들러
  isStreaming?: boolean; // 스트리밍 중인지 여부
  error?: StreamError; // 에러 상태
  onRetry?: () => void; // 재시도 핸들러
}

export function ChatMessage({ 
  message, 
  sessionId, 
  onQuestionClick,
  onDocumentClick,
  onOpenReferencesSidebar,
  isStreaming = false,
  error,
  onRetry
}: ChatMessageProps) {
  const [copied, setCopied] = useState(false);
  const [feedback, setFeedback] = useState<'positive' | 'negative' | null>(null);
  
  // 안전성 검사: message가 없거나 필수 필드가 없으면 기본값 설정
  const safeMessage = message || { id: '', content: '', role: 'user' as const, timestamp: new Date() };
  const safeId = safeMessage.id || '';
  
  // content가 없으면 빈 문자열로 처리
  const content = safeMessage.content || '';
  
  // 메시지 타입 확인 (먼저 정의)
  const isUser = safeMessage.role === 'user';
  const isProgress = safeMessage.role === 'progress';
  const metadata = safeMessage.metadata || {};
  
  // 스트리밍 완료 여부 확인 (isStreaming prop이 false이거나 undefined인 경우)
  const isStreamingComplete = !isStreaming;
  
  if (import.meta.env.DEV && !isUser && !isProgress) {
    logger.debug('[ChatMessage] Component render:', {
      messageId: safeId,
      isStreaming,
      isStreamingComplete,
      hasMetadata: !!metadata,
      relatedQuestions: metadata.related_questions?.length || 0,
      sources: metadata.sources?.length || 0,
      legalReferences: metadata.legal_references?.length || 0,
      sourcesDetail: metadata.sources_detail?.length || 0,
      metadataKeys: Object.keys(metadata),
      relatedQuestionsValue: metadata.related_questions,
    });
  }
  
  // 타이핑 효과 적용 (스트리밍 중일 때만 활성화)
  const { displayed: displayedContent, isComplete: isTypingComplete } = useTypingEffect(
    content,
    {
      speed: 1, // ms마다 한 글자씩 표시
      enabled: isStreaming // 스트리밍 중일 때만 타이핑 효과 활성화
    }
  );

  // 실시간 마크다운 렌더링을 위한 콘텐츠 결정
  // 스트리밍 중이면 displayedContent 사용, 아니면 전체 content 사용
  const markdownContent = useMemo(() => {
    return isStreaming ? displayedContent : content;
  }, [isStreaming, displayedContent, content]);
  
  // 마크다운 렌더링 여부: assistant 메시지이고 콘텐츠가 있으면 렌더링
  const shouldRenderMarkdown = !isUser && markdownContent.length > 0;

  // 마크다운 렌더링 전에 "문서 N" 패턴을 특별한 링크로 변환
  const processedMarkdown = useMemo(() => {
    if (!markdownContent) return markdownContent;
    if (!onDocumentClick) return markdownContent;
    
    // 이미 링크로 변환된 텍스트는 건너뛰기 (#doc-로 시작하는 링크가 있으면 이미 변환된 것)
    if (markdownContent.includes('#doc-')) {
      return markdownContent;
    }
    
    let processed = markdownContent;
    const sources = metadata.sources || [];
    const sourcesDetail = metadata.sources_detail || [];
    
    // sources가 있으면 기존 로직 사용
    if (sources.length > 0) {
      sources.forEach((source, index) => {
        const docNum = index + 1;
        // "문서 1", "문서1", "문서 1번" 등의 패턴 매칭
        // 이미 링크로 변환된 부분은 제외하기 위해 링크 패턴이 아닌 경우만 매칭
        // eslint-disable-next-line security/detect-non-literal-regexp
        const pattern = new RegExp(`문서\\s*${docNum}(?!\\d)`, 'g');
        
        processed = processed.replace(pattern, (match, offset) => {
          // 이전 문자 확인: [로 시작하는 링크가 아니어야 함
          const beforeMatch = processed.substring(Math.max(0, offset - 10), offset);
          if (beforeMatch.includes('[') && beforeMatch.includes('](')) {
            // 이미 링크로 변환된 부분
            return match;
          }
          // 링크로 변환
          // eslint-disable-next-line security/detect-object-injection
          return `[문서 ${docNum}](#doc-${index} "${source || sourcesDetail[index]?.name || `문서 ${docNum}`}")`;
        });
      });
    } else {
      // sources가 없으면 답변에서 "문서 N" 패턴을 찾아서 최대 문서 개수 추정
      const documentPattern = /문서\s*(\d+)/g;
      const matches = Array.from(markdownContent.matchAll(documentPattern));
      const maxDocNum = matches.length > 0 
        ? Math.max(...matches.map(m => parseInt(m[1] || '0', 10)))
        : 0;
      
      // 추정된 문서 개수만큼 링크 생성
      if (maxDocNum > 0) {
        for (let i = 0; i < maxDocNum; i++) {
          const docNum = i + 1;
          // "문서 1", "문서1", "문서 1번" 등의 패턴 매칭
          // 이미 링크로 변환된 부분은 제외하기 위해 링크 패턴이 아닌 경우만 매칭
          // eslint-disable-next-line security/detect-non-literal-regexp
          const pattern = new RegExp(`문서\\s*${docNum}(?!\\d)`, 'g');
          
          processed = processed.replace(pattern, (match, offset) => {
            // 이전 문자 확인: [로 시작하는 링크가 아니어야 함
            const beforeMatch = processed.substring(Math.max(0, offset - 10), offset);
            if (beforeMatch.includes('[') && beforeMatch.includes('](')) {
              // 이미 링크로 변환된 부분
              return match;
            }
            // 링크로 변환
            // eslint-disable-next-line security/detect-object-injection
            return `[문서 ${docNum}](#doc-${i} "문서 ${docNum}")`;
          });
        }
      }
    }
    
    return processed;
  }, [markdownContent, metadata.sources, metadata.sources_detail, onDocumentClick]);

  // 마크다운 컴포넌트 메모이제이션
  const markdownComponents = useMemo(() => ({
    // 코드 블록 스타일링
    code: ({ className, children, ...props }: React.ComponentPropsWithoutRef<'code'>) => {
      const match = /language-(\w+)/.exec(className || '');
      const isInline = !match;
      
      if (!isInline && match) {
        return (
          <SyntaxHighlighter
            style={vscDarkPlus}
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
    // 링크 스타일링 (문서 참조 링크 처리)
    a: ({ href, title, children, ...props }: React.ComponentPropsWithoutRef<'a'> & { href?: string; title?: string }) => {
      // 문서 참조 링크인지 확인 (#doc-로 시작)
      if (href?.startsWith('#doc-')) {
        const docIndex = parseInt(href.replace('#doc-', ''), 10);
        if (!isNaN(docIndex) && onDocumentClick) {
          return (
            <DocumentReference
              documentIndex={docIndex}
              onClick={() => onDocumentClick(safeMessage, docIndex)}
            />
          );
        }
      }
      
      // 일반 링크
      return (
        <a 
          className="text-blue-600 hover:text-blue-800 underline" 
          target="_blank" 
          rel="noopener noreferrer" 
          href={href}
          title={title}
          {...props}
        >
          {children}
        </a>
      );
    },
    // 리스트 스타일링
    ul: ({ ...props }: React.ComponentPropsWithoutRef<'ul'>) => (
      <ul className="list-disc list-inside my-2 space-y-1" {...props} />
    ),
    ol: ({ ...props }: React.ComponentPropsWithoutRef<'ol'>) => (
      <ol className="list-decimal list-inside my-2 space-y-1" {...props} />
    ),
    // 제목 스타일링
    h1: ({ children, ...props }: React.ComponentPropsWithoutRef<'h1'>) => (
      <h1 className="text-2xl font-bold mt-4 mb-2" {...props}>{children}</h1>
    ),
    h2: ({ children, ...props }: React.ComponentPropsWithoutRef<'h2'>) => (
      <h2 className="text-xl font-bold mt-3 mb-2" {...props}>{children}</h2>
    ),
    h3: ({ children, ...props }: React.ComponentPropsWithoutRef<'h3'>) => (
      <h3 className="text-lg font-semibold mt-2 mb-1" {...props}>{children}</h3>
    ),
    // 강조 스타일링
    strong: ({ ...props }: React.ComponentPropsWithoutRef<'strong'>) => (
      <strong className="font-semibold" {...props} />
    ),
    em: ({ ...props }: React.ComponentPropsWithoutRef<'em'>) => (
      <em className="italic" {...props} />
    ),
    // 인용구 스타일링
    blockquote: ({ ...props }: React.ComponentPropsWithoutRef<'blockquote'>) => (
      <blockquote className="border-l-4 border-slate-300 pl-4 italic my-2 text-slate-600" {...props} />
    ),
    // 테이블 스타일링
    table: ({ ...props }: React.ComponentPropsWithoutRef<'table'>) => (
      <div className="overflow-x-auto my-2">
        <table className="min-w-full border-collapse border border-slate-300" {...props} />
      </div>
    ),
    th: ({ ...props }: React.ComponentPropsWithoutRef<'th'>) => (
      <th className="border border-slate-300 px-4 py-2 bg-slate-100 font-semibold" {...props} />
    ),
    td: ({ ...props }: React.ComponentPropsWithoutRef<'td'>) => (
      <td className="border border-slate-300 px-4 py-2" {...props} />
    ),
    // 수평선 스타일링
    hr: ({ ...props }: React.ComponentPropsWithoutRef<'hr'>) => (
      <hr className="my-4 border-slate-300" {...props} />
    ),
    // 단락 스타일링
    p: ({ ...props }: React.ComponentPropsWithoutRef<'p'>) => (
      <p className="my-2" {...props} />
    ),
  }), [safeMessage, onDocumentClick]);

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
      const sessionIdValue = sessionId || (typeof safeMessage.metadata?.session_id === 'string' ? safeMessage.metadata.session_id : '') || '';
      await sendFeedback({
        message_id: safeMessage.id,
        session_id: sessionIdValue,
        rating: ratingToNumber(rating),
      });
    } catch (error) {
      logger.error('Failed to send feedback:', error);
      setFeedback(null);
    }
  };


  // 안전성 검사: message가 없거나 필수 필드가 없으면 렌더링하지 않음
  if (!safeId) {
    return null;
  }
  
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
        {safeMessage.attachments && safeMessage.attachments.length > 0 && (
          <div className="mb-3 flex flex-wrap gap-2">
            {safeMessage.attachments.map((att) => (
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
          ) : shouldRenderMarkdown ? (
            // 실시간 마크다운 렌더링 (스트리밍 중에도 적용)
            <div className="relative">
              <ReactMarkdown
                components={markdownComponents}
              >
                {processedMarkdown}
              </ReactMarkdown>
              
              {/* 스트리밍 중일 때 커서 표시 */}
              {isStreaming && !isTypingComplete && (
                <span className="inline-block w-0.5 h-4 bg-slate-600 ml-1 animate-blink">
                  |
                </span>
              )}
            </div>
          ) : (
            // 콘텐츠가 없을 때 기본 표시
            <div className="whitespace-pre-wrap">{displayedContent}</div>
          )}
        </div>

        {/* 에러 메시지 표시 */}
        {error && !isUser && (
          <div className="mt-3">
            <ErrorMessage error={error} onRetry={onRetry} />
          </div>
        )}

        {!isUser && !isStreaming && (() => {
          const hasReferences = (metadata.sources && metadata.sources.length > 0) ||
            (metadata.legal_references && metadata.legal_references.length > 0) ||
            (metadata.sources_detail && metadata.sources_detail.length > 0);
          
          if (import.meta.env.DEV) {
            logger.debug('[ChatMessage] References check:', {
              messageId: safeMessage.id,
              isStreaming,
              isStreamingComplete,
              hasReferences,
              sources: metadata.sources?.length || 0,
              legalReferences: metadata.legal_references?.length || 0,
              sourcesDetail: metadata.sources_detail?.length || 0,
              metadataKeys: Object.keys(metadata || {}),
            });
          }
          
          if (hasReferences) {
            return (
              <CompactReferencesBadge
                references={metadata.sources}
                legalReferences={metadata.legal_references}
                sources={metadata.sources}
                sourcesDetail={metadata.sources_detail}
                onOpenSidebar={(selectedType) => onOpenReferencesSidebar?.(safeMessage, selectedType)}
              />
            );
          }
          
          return null;
        })()}

        {!isUser && !isStreaming && (() => {
          const relatedQuestions = metadata.related_questions;
          const questionsArray = Array.isArray(relatedQuestions) 
            ? relatedQuestions.filter((q): q is string => typeof q === 'string' && q.trim().length > 0)
            : undefined;
          
          if (import.meta.env.DEV) {
            logger.debug('[ChatMessage] Related questions check:', {
              messageId: safeMessage.id,
              isStreaming,
              isStreamingComplete,
              hasQuestions: !!questionsArray,
              questionsCount: questionsArray?.length || 0,
              questions: questionsArray,
              metadataKeys: Object.keys(metadata || {}),
            });
          }
          
          const shouldShowRelatedQuestions = questionsArray && questionsArray.length > 0;
          
          if (shouldShowRelatedQuestions) {
            return (
              <div className="mt-4">
                <RelatedQuestions
                  questions={questionsArray}
                  onQuestionClick={onQuestionClick}
                />
              </div>
            );
          }
          
          return null;
        })()}

        <div className="flex items-center justify-between mt-2">
          <span className="text-xs text-slate-500">
            {safeMessage.timestamp ? formatRelativeTime(safeMessage.timestamp) : '방금 전'}
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

