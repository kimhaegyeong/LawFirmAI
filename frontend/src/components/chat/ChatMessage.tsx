/**
 * 채팅 메시지 컴포넌트
 */
import { Copy, Check, ThumbsUp, ThumbsDown, Loader2, RefreshCw } from 'lucide-react';
import React, { useState, useMemo, useRef, useEffect } from 'react';
import ReactMarkdown from 'react-markdown';
import { Prism as SyntaxHighlighter } from 'react-syntax-highlighter';
import { vscDarkPlus } from 'react-syntax-highlighter/dist/esm/styles/prism';
import { formatRelativeTime } from '../../utils/dateUtils';
import { FileAttachment } from '../common/FileAttachment';
import { MessageReferencesSection } from './MessageReferencesSection';
import { MessageRelatedQuestionsSection } from './MessageRelatedQuestionsSection';
import { ErrorMessage } from './ErrorMessage';
import { DocumentReference } from './DocumentReference';
import { sendFeedback, ratingToNumber } from '../../services/feedbackService';
import { useTypingEffect } from '../../hooks/useTypingEffect';
import logger from '../../utils/logger';
import type { ChatMessage as ChatMessageType } from '../../types/chat';
import type { StreamError } from '../../types/error';
import { getSourcesDetailFromSourcesByType } from '../../utils/sourcesParser';

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
  onOpenReferencesSidebar?: (message: ChatMessageType, selectedType: 'all' | 'law' | 'precedent' | 'decision' | 'interpretation' | 'regulation', referenceId?: string | null) => void; // 참고자료 사이드바 열기 핸들러
  isStreaming?: boolean; // 스트리밍 중인지 여부
  error?: StreamError; // 에러 상태
  onRetry?: () => void; // 재시도 핸들러
  onRetryQuestion?: () => void; // 질문 재전송 핸들러
  hasEmptyOrErrorResponse?: boolean; // 답변이 비어있거나 오류가 있는지 여부
}

export function ChatMessage({ 
  message, 
  sessionId, 
  onQuestionClick,
  onDocumentClick,
  onOpenReferencesSidebar,
  isStreaming = false,
  error,
  onRetry,
  onRetryQuestion,
  hasEmptyOrErrorResponse = false
}: ChatMessageProps) {
  const [copied, setCopied] = useState(false);
  const [feedback, setFeedback] = useState<'positive' | 'negative' | null>(null);
  
  // 안전성 검사: message가 없거나 필수 필드가 없으면 기본값 설정 (useMemo로 최적화)
  const safeMessage = useMemo(() => 
    message || { id: '', content: '', role: 'user' as const, timestamp: new Date() },
    [message]
  );
  const safeId = safeMessage.id || '';
  
  // content가 없으면 빈 문자열로 처리
  const content = safeMessage.content || '';
  
  // 메시지 타입 확인 (먼저 정의)
  const isUser = safeMessage.role === 'user';
  const isProgress = safeMessage.role === 'progress';
  
  // metadata를 명시적으로 추출하여 변경 감지 보장 (useMemo로 최적화)
  const metadata = useMemo(() => safeMessage.metadata || {}, [safeMessage.metadata]);
  
  // done 이벤트를 받았는지 확인 (직접 계산하여 변경 감지 보장)
  const isDone = metadata._isDone === true;
  
  // 스트리밍 완료 여부 확인 (done 이벤트를 받았거나 isStreaming prop이 false인 경우)
  const isStreamingComplete = isDone || !isStreaming;
  
  if (import.meta.env.DEV && !isUser && !isProgress) {
    logger.debug('[ChatMessage] Component render:', {
      messageId: safeId,
      isStreaming,
      isDone,
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
  
  // 타이핑 효과 적용 (done 이벤트를 받지 않았을 때만 활성화)
  // isDone이 true이면 isStreaming과 관계없이 타이핑 효과 비활성화
  const typingEnabled = useMemo(() => !isDone && isStreaming, [isDone, isStreaming]);
  
  // isDone이 true이면 타이핑 효과를 완전히 우회하고 전체 content 사용
  const typingEffectResult = useTypingEffect(
    content,
    {
      speed: 1, // ms마다 한 글자씩 표시
      enabled: typingEnabled
    }
  );
  
  // isDone이 true이면 즉시 전체 content 표시 (useMemo로 메모이제이션하여 즉시 반영)
  const displayedContent = useMemo(() => {
    if (isDone) {
      return content;
    }
    return typingEffectResult.displayed;
  }, [isDone, content, typingEffectResult.displayed]);
  
  const isTypingComplete = useMemo(() => {
    return isDone ? true : typingEffectResult.isComplete;
  }, [isDone, typingEffectResult.isComplete]);

  // 실시간 마크다운 렌더링을 위한 콘텐츠 결정
  // displayedContent는 이미 isDone일 때 전체 content로 설정됨
  const markdownContent = useMemo(() => {
    return displayedContent;
  }, [displayedContent]);
  
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
    
    // sources_by_type 우선 사용
    const sourcesByType = metadata.sources_by_type;
    const sourcesDetail = sourcesByType
      ? getSourcesDetailFromSourcesByType(sourcesByType)
      : (metadata.sources_detail || []);
    const sources = metadata.sources || [];
    
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
            return `[문서 ${docNum}](#doc-${i} "문서 ${docNum}")`;
          });
        }
      }
    }
    
    return processed;
  }, [markdownContent, metadata.sources, metadata.sources_by_type, metadata.sources_detail, onDocumentClick]);

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
          // sources 이벤트를 받았거나 done 이벤트를 받았는지 확인
          const hasSourcesEvent = metadata._hasSourcesEvent === true;
          const hasDoneEvent = metadata._isDone === true;
          // sources 이벤트 또는 done 이벤트를 받았으면 활성화
          const isEnabled = hasSourcesEvent || hasDoneEvent;
          return (
            <DocumentReference
              documentIndex={docIndex}
              onClick={() => onDocumentClick(message || safeMessage, docIndex)}
              disabled={!isEnabled}
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
  }), [message, safeMessage, onDocumentClick, metadata._hasSourcesEvent, metadata._isDone]);

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

  // 참고자료 및 연관질문 확인 (메타데이터가 있으면 스트리밍 중이어도 표시)
  // sources_by_type 우선 사용, 없으면 sources_detail 사용
  const sourcesByType = metadata.sources_by_type;
  const sourcesDetailFromByType = (sourcesByType && typeof sourcesByType === 'object' && !Array.isArray(sourcesByType))
    ? getSourcesDetailFromSourcesByType(sourcesByType as Parameters<typeof getSourcesDetailFromSourcesByType>[0])
    : [];
  
  const sourcesArray = Array.isArray(metadata.sources) ? metadata.sources : [];
  const legalReferencesArray = Array.isArray(metadata.legal_references) ? metadata.legal_references : [];
  const sourcesDetailArray = sourcesDetailFromByType.length > 0
    ? sourcesDetailFromByType
    : (Array.isArray(metadata.sources_detail) ? metadata.sources_detail : []);
  
  const hasReferences = !isUser && (
    sourcesArray.length > 0 ||
    legalReferencesArray.length > 0 ||
    sourcesDetailArray.length > 0 ||
    (sourcesByType && Object.keys(sourcesByType).length > 0)
  );

  const relatedQuestions = !isUser ? metadata.related_questions : undefined;
  const questionsArray = Array.isArray(relatedQuestions)
    ? relatedQuestions.filter((q): q is string => typeof q === 'string' && q.trim().length > 0)
    : [];
  const hasRelatedQuestions = !isUser && questionsArray.length > 0;
  
  // 메타데이터 변경 감지를 위한 ref (디버깅용) - hooks는 early return 전에 호출되어야 함
  const prevMetadataRef = useRef<string>('');
  
  // 메타데이터 변경 감지 및 디버깅 로그
  useEffect(() => {
    if (import.meta.env.DEV && !isUser) {
      const metadataString = JSON.stringify(metadata);
      const metadataChanged = prevMetadataRef.current !== metadataString;
      if (metadataChanged) {
        prevMetadataRef.current = metadataString;
        logger.debug('[ChatMessage] Metadata changed, component should re-render:', {
          messageId: safeId,
          hasReferences,
          hasRelatedQuestions,
          sourcesCount: sourcesArray.length,
          sourcesDetailCount: sourcesDetailArray.length,
          questionsCount: questionsArray.length,
          isStreaming,
          metadataKeys: Object.keys(metadata),
        });
      }
    }
  }, [metadata, safeId, hasReferences, hasRelatedQuestions, sourcesArray.length, sourcesDetailArray.length, questionsArray.length, isStreaming, isUser]);

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

  if (import.meta.env.DEV && !isUser) {
    logger.debug('[ChatMessage] References and questions check:', {
      messageId: safeId,
      hasReferences,
      hasRelatedQuestions,
      sourcesCount: sourcesArray.length,
      legalReferencesCount: legalReferencesArray.length,
      sourcesDetailCount: sourcesDetailArray.length,
      questionsCount: questionsArray.length,
      metadataKeys: Object.keys(metadata),
      metadataSources: metadata.sources,
      metadataSourcesDetail: metadata.sources_detail,
      metadataRelatedQuestions: metadata.related_questions,
      relatedQuestionsRaw: relatedQuestions,
      questionsArrayFiltered: questionsArray,
    });
    
    // related_questions가 있는데 표시되지 않는 경우 경고
    if (metadata.related_questions && Array.isArray(metadata.related_questions) && metadata.related_questions.length > 0 && !hasRelatedQuestions) {
      logger.warn('[ChatMessage] related_questions exists but not displayed:', {
        messageId: safeId,
        relatedQuestionsRaw: metadata.related_questions,
        questionsArrayFiltered: questionsArray,
        filterResult: metadata.related_questions.map(q => ({
          original: q,
          isString: typeof q === 'string',
          trimmed: typeof q === 'string' ? q.trim() : q,
          hasLength: typeof q === 'string' ? q.trim().length > 0 : false,
        })),
      });
    }
  }

  return (
    <div className="flex flex-col gap-3 mb-4">
      <div
        className={`flex gap-3 ${
          isUser ? 'justify-end' : 'justify-start'
        }`}
      >
        {!isUser && (
          <div className="w-8 h-8 bg-blue-100 rounded-full flex items-center justify-center flex-shrink-0">
            <span className="text-blue-600 text-sm">AI</span>
          </div>
        )}
        <div
          className={`w-full max-w-[80%] rounded-xl p-5 shadow-sm border ${
            isUser
              ? 'bg-blue-50 border-blue-200'
              : 'bg-white border-slate-200'
          } ${!isUser ? 'min-h-[100px]' : ''}`}
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
              
              {/* 타이핑 효과가 활성화되어 있고 완료되지 않았을 때 커서 표시 */}
              {typingEnabled && !isTypingComplete && (
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
      
      {/* 사용자 메시지 밑에 새로고침 아이콘 표시 (답변에 오류가 있는 경우만) */}
      {isUser && hasEmptyOrErrorResponse && onRetryQuestion && (
        <div className="flex justify-end mt-1">
          <button
            onClick={onRetryQuestion}
            className="p-1.5 hover:bg-blue-50 rounded-full transition-colors text-blue-600 group"
            title="질문 다시 보내기"
          >
            <RefreshCw className="w-4 h-4 group-hover:rotate-180 transition-transform duration-300" />
          </button>
        </div>
      )}

      {/* 메타데이터 섹션 (참고자료 및 연관질문) - 답변 표시 완료 후에만 표시 */}
      {/* isDone이 false여도 isStreaming이 false이고 content가 있으면 표시 (done 이벤트가 오지 않았을 수 있음) */}
      {!isUser && (isDone || (!isStreaming && content.trim().length > 0)) && (hasReferences || hasRelatedQuestions) && (
        <div
          className={`flex gap-3 ${
            isUser ? 'justify-end' : 'justify-start'
          }`}
        >
          {!isUser && (
            <div className="w-8 h-8 flex-shrink-0" />
          )}
          <div className="w-full max-w-[80%] flex-shrink-0 rounded-xl p-5 shadow-sm border bg-white border-slate-200 min-h-[80px]">
            {hasReferences && (
              <MessageReferencesSection
                references={sourcesArray}
                legalReferences={legalReferencesArray}
                sources={sourcesArray}
                sourcesDetail={sourcesDetailArray}
                sourcesByType={sourcesByType}
                onOpenSidebar={(selectedType) => onOpenReferencesSidebar?.(safeMessage, selectedType)}
                onReferenceClick={(ref, sourceDetail) => {
                  const referenceId = sourceDetail?.case_number || 
                                    sourceDetail?.article_no || 
                                    sourceDetail?.decision_number || 
                                    sourceDetail?.interpretation_number ||
                                    sourceDetail?.metadata?.doc_id ||
                                    ref.metadata?.doc_id ||
                                    ref.metadata?.case_number ||
                                    ref.metadata?.article_no ||
                                    ref.id;
                  onOpenReferencesSidebar?.(safeMessage, ref.type, referenceId);
                }}
                defaultExpanded={false}
              />
            )}
            {hasRelatedQuestions && (
              <MessageRelatedQuestionsSection
                questions={questionsArray}
                onQuestionClick={onQuestionClick}
                defaultExpanded={true}
              />
            )}
          </div>
        </div>
      )}
    </div>
  );
}

