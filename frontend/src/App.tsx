/**
 * 메인 App 컴포넌트
 */
import { useState, useEffect, useRef } from 'react';
import { MainLayout } from './components/layout/MainLayout';
import { SidebarContent } from './components/sidebar/SidebarContent';
import { ChatContainer } from './components/chat/ChatContainer';
import { Toast } from './components/common/Toast';
import { DocumentSidebar } from './components/chat/DocumentSidebar';
import { ReferencesSidebar } from './components/chat/ReferencesSidebar';
import { LoginPage } from './components/auth/LoginPage';
import { useChat } from './hooks/useChat';
import { useSession } from './hooks/useSession';
import { useAuth } from './hooks/useAuth';
import { useToast } from './hooks/useToast';
import { useSidebar } from './hooks/useSidebar';
import { useMessageState } from './hooks/useMessageState';
import { useSessionHandlers } from './hooks/useSessionHandlers';
import { useStreamingMessage } from './hooks/useStreamingMessage';
import { getHistory } from './services/historyService';
import { cleanupOldOAuthKeys } from './services/authService';
import { toStreamError, extractQuotaInfo } from './utils/errorHandler';
import { parseSourcesMetadata, mergeSourcesMetadata } from './utils/sourcesParser';
import logger from './utils/logger';
import type { AxiosError } from 'axios';
import { convertImageToBase64, convertFileToBase64, isImageFile, isDocumentFile } from './utils/fileUtils';
import type { ChatMessage, FileAttachment } from './types/chat';

function App() {
  const { isAuthenticated, isLoading: isAuthLoading, login } = useAuth();
  const { toasts, showToast, hideToast } = useToast();
  const {
    documentSidebarOpen,
    selectedDocumentIndex,
    selectedMessageForDocument,
    openDocumentSidebar,
    closeDocumentSidebar,
    updateSelectedMessageForDocument,
    referencesSidebarOpen,
    selectedMessageForReferences,
    selectedReferenceType,
    selectedReferenceId,
    openReferencesSidebar,
    closeReferencesSidebar,
  } = useSidebar();
  
  const {
    messages,
    streamingMessageId,
    streamErrors,
    addMessage,
    updateMessages,
    clearMessages,
    setStreamingId,
    addError,
    removeError,
  } = useMessageState();
  
  const [searchQuery, setSearchQuery] = useState('');
  const [locationKey, setLocationKey] = useState(0);
  const [currentProgress, setCurrentProgress] = useState<string | null>(null);
  const [progressHistory, setProgressHistory] = useState<string[]>([]);
  const [inputResetTrigger, setInputResetTrigger] = useState(0);
  const [quotaInfo, setQuotaInfo] = useState<{ remaining: number; limit: number } | null>(null);
  const [sessionListRefreshTrigger, setSessionListRefreshTrigger] = useState(0);
  
  // 초기 마운트 여부 추적 (StrictMode 대응)
  const isInitialMount = useRef(true);

  const { currentSession, isLoading, loadSessions, newSession, loadSession, updateSession, removeSession, clearSession } = useSession();
  
  const {
    handleNewSession,
    handleSelectSession,
    handleRenameSession,
    handleCopySession,
    handleExportSession,
    handleDeleteSession,
  } = useSessionHandlers({
    loadSession,
    updateSession,
    removeSession,
    loadSessions,
    clearSession,
    clearMessages,
    currentSession,
    setInputResetTrigger,
  });
  const { isLoading: isSending, isStreaming } = useChat({
    onMessage: (message) => {
      addMessage(message);
      // 에러 메시지가 있으면 제거
      if (streamErrors.has(message.id)) {
        removeError(message.id);
      }
    },
    onError: (error) => {
      logger.error('Chat error:', error);
      const streamError = toStreamError(error, isAuthenticated);
      
      // Toast 표시 (로그인한 사용자에게는 익명 사용자 쿼터 메시지 표시 안 함)
      if (!isAuthenticated || !streamError.message.includes('무료 질의 3회를 모두 사용하셨습니다')) {
        showToast({
          message: streamError.message,
          type: 'error',
          action: streamError.canRetry ? {
            label: '다시 시도',
            onClick: () => {
              // 재시도 로직은 필요시 구현
            }
          } : undefined,
        });
      }
    },
  });

  const { handleStreamingMessage } = useStreamingMessage({
    isAuthenticated,
    streamingMessageId,
    setStreamingId,
    updateMessages,
    addMessage,
    addError,
    setQuotaInfo,
    setCurrentProgress,
    setProgressHistory,
    setSessionListRefreshTrigger,
    showToast,
    streamErrors,
  });

  // URL 변경 감지를 위한 popstate 이벤트 리스너
  useEffect(() => {
    const handlePopState = () => {
      setLocationKey(prev => prev + 1);
    };
    window.addEventListener('popstate', handlePopState);
    return () => {
      window.removeEventListener('popstate', handlePopState);
    };
  }, []);

  // 초기 세션 로드 (StrictMode 대응)
  // 세션 목록은 SessionList 컴포넌트에서 로드하므로 여기서는 URL 파라미터의 세션 ID만 처리
  useEffect(() => {
    if (isInitialMount.current) {
      isInitialMount.current = false;
      
      cleanupOldOAuthKeys();
      
      // URL 파라미터에서 세션 ID 확인
      const urlParams = new URLSearchParams(window.location.search);
      const sessionIdParam = urlParams.get('session_id');
      
      if (sessionIdParam) {
        // URL 파라미터에서 세션 ID가 있으면 해당 세션 로드
        loadSession(sessionIdParam)
          .then(() => {
            // URL에서 세션 ID 파라미터 제거
            urlParams.delete('session_id');
            const newUrl = window.location.pathname + (urlParams.toString() ? `?${urlParams.toString()}` : '');
            window.history.replaceState({}, document.title, newUrl);
          })
          .catch((error) => {
            // 429 에러 처리 - 쿼터 정보 업데이트 (익명 사용자만)
            if (error && typeof error === 'object' && 'status' in error) {
              const apiError = error as AxiosError<{ quotaInfo?: { remaining: number; limit: number } }> & { status?: number; quotaInfo?: { remaining: number; limit: number } };
              if (apiError.status === 429 && !isAuthenticated) {
                const quotaInfo = extractQuotaInfo(error) || { remaining: 0, limit: 3 };
                setQuotaInfo(quotaInfo);
                
                // Toast 메시지 표시
                showToast({
                  message: '무료 질의 3회를 모두 사용하셨습니다. 계속 사용하려면 로그인이 필요합니다.',
                  type: 'warning',
                  action: {
                    label: '로그인',
                    onClick: () => login(),
                  },
                });
              }
            }
            
            // 404 오류는 세션이 존재하지 않는 정상적인 상황일 수 있음
            const isNotFound = error?.status === 404 || error?.message?.includes('Session not found');
            if (isNotFound) {
              logger.warn(`Session not found in URL parameter: ${sessionIdParam}. The session may have been deleted or does not exist.`);
              // URL에서 세션 ID 파라미터 제거 (존재하지 않는 세션 ID는 URL에 남겨두지 않음)
              urlParams.delete('session_id');
              const newUrl = window.location.pathname + (urlParams.toString() ? `?${urlParams.toString()}` : '');
              window.history.replaceState({}, document.title, newUrl);
            } else {
              logger.error('Failed to load session from URL parameter:', error);
            }
          });
      }
    }
  }, [loadSession, isAuthenticated, login, showToast]);

  // 세션 변경 시 메시지 로드
  useEffect(() => {
    const loadSessionMessages = async () => {
      if (currentSession) {
        try {
          const history = await getHistory({
            session_id: currentSession.session_id,
          });
          
          // Message 타입을 ChatMessage 타입으로 변환
          const chatMessages: ChatMessage[] = history.messages.map((msg) => ({
            id: msg.message_id,
            role: msg.role,
            content: msg.content,
            timestamp: new Date(msg.timestamp),
            metadata: msg.metadata || {},
          }));
          
          // 메시지가 있을 때만 설정 (빈 배열로 덮어쓰지 않도록)
          if (chatMessages.length > 0) {
            updateMessages(() => chatMessages);
          }
        } catch (error) {
          logger.error('Failed to load session messages:', error);
          // 에러 발생 시에도 현재 메시지를 유지 (빈 배열로 덮어쓰지 않음)
        }
      } else {
        clearMessages();
      }
    };
    
    loadSessionMessages();
  }, [currentSession, updateMessages, clearMessages]);


  const handleSendMessage = async (message: string, attachments?: FileAttachment[]) => {
    // 파일이 있으면 Base64로 변환
    let imageBase64: string | undefined;
    let fileBase64: string | undefined;
    let filename: string | undefined;
    
    if (attachments && attachments.length > 0) {
      // 첫 번째 파일만 처리
      const attachment = attachments[0];
      if (attachment?.file) {
        try {
          if (isImageFile(attachment.type)) {
            // 이미지 파일: image_base64 사용 (하위 호환성)
            imageBase64 = await convertImageToBase64(attachment.file);
            logger.debug('Image converted to Base64:', imageBase64.substring(0, 50) + '...');
          } else if (isDocumentFile(attachment.type) || attachment.type === 'text/plain') {
            // 텍스트, PDF, DOCX 파일: file_base64 사용
            fileBase64 = await convertFileToBase64(attachment.file);
            filename = attachment.name;
            logger.debug('File converted to Base64:', fileBase64.substring(0, 50) + '...', 'filename:', filename);
          } else {
            logger.warn('Unsupported file type:', attachment.type);
          }
        } catch (error) {
          logger.error('Failed to convert file to Base64:', error);
        }
      }
    }

    if (!currentSession) {
      // 세션이 없으면 새로 생성
      const session = await newSession();
      if (session) {
        await handleStreamingMessage(message, session.session_id, attachments, imageBase64, fileBase64, filename);
      }
    } else {
      await handleStreamingMessage(message, currentSession.session_id, attachments, imageBase64, fileBase64, filename);
    }
  };


  const handleQuestionClick = async (question: string) => {
    if (!currentSession) {
      // 1. 낙관적 업데이트: 메시지를 먼저 추가하여 UI를 즉시 전환
      const userMessage: ChatMessage = {
        id: `user-${Date.now()}`,
        role: 'user',
        content: question,
        timestamp: new Date(),
      };
      updateMessages(() => [userMessage]);
      
      // 2. 세션 생성 (세션 목록 새로고침은 백그라운드에서 처리)
      try {
        const session = await newSession(undefined, true); // skipLoadSessions: true
        if (session) {
          // 스트리밍 메시지 처리 (사용자 메시지는 이미 추가했으므로 skip)
          await handleStreamingMessage(question, session.session_id, undefined, undefined, undefined, undefined, true);
        }
      } catch (error) {
        logger.error('[ERROR] Failed to create session:', error);
        // 에러 발생 시 메시지 제거하고 웰컴 화면으로 복귀
        clearMessages();
        const errorMessage = error instanceof Error 
          ? error.message 
          : '세션 생성에 실패했습니다. 다시 시도해주세요.';
        alert(errorMessage);
      }
    } else {
      await handleStreamingMessage(question, currentSession.session_id, undefined, undefined, undefined, undefined);
    }
  };

  const handleRelatedQuestionClick = async (question: string) => {
    await handleQuestionClick(question);
  };

  const handleRetryQuestion = async (question: string, _messageId: string) => {
    if (!currentSession) {
      // 세션이 없으면 새로 생성
      const session = await newSession();
      if (session) {
        await handleStreamingMessage(question, session.session_id);
      }
    } else {
      await handleStreamingMessage(question, currentSession.session_id);
    }
  };

  const handleDocumentClick = async (message: ChatMessage, documentIndex: number) => {
    openDocumentSidebar(message, documentIndex);
    
    // sources가 없으면 API로 가져오기
    if (!message.metadata?.sources?.length && 
        !message.metadata?.sources_detail?.length && 
        currentSession?.session_id) {
      try {
        const { getChatSources } = await import('./services/chatService');
        const sourcesData = await getChatSources(
          currentSession.session_id,
          message.id
        );
        
        // 메시지 metadata 업데이트
        const parsedSources = parseSourcesMetadata({
          sources: sourcesData.sources,
          legal_references: sourcesData.legal_references,
          sources_detail: sourcesData.sources_detail,
        });
        
        updateMessages((prev) => {
          const messageIndex = prev.findIndex((msg) => msg.id === message.id);
          if (messageIndex !== -1) {
            const updated = [...prev];
            // eslint-disable-next-line security/detect-object-injection
            const existingMsg = updated[messageIndex];
            if (existingMsg) {
              // eslint-disable-next-line security/detect-object-injection
              updated[messageIndex] = {
                ...existingMsg,
                metadata: mergeSourcesMetadata(existingMsg.metadata, parsedSources),
              };
            }
            return updated;
          }
          return prev;
        });
        
        // 사이드바에 표시할 메시지도 업데이트
        updateSelectedMessageForDocument((prev) => {
          if (prev?.id === message.id) {
            return {
              ...prev,
              metadata: mergeSourcesMetadata(prev.metadata, parsedSources),
            };
          }
          return prev;
        });
      } catch (error) {
        logger.error('[App] Error fetching sources for sidebar:', error);
      }
    }
  };

  if (isAuthLoading) {
    return (
      <div className="flex items-center justify-center min-h-screen">
        <div className="text-center">
          <div className="inline-block animate-spin rounded-full h-8 w-8 border-b-2 border-blue-600 mb-4"></div>
          <p className="text-slate-600">로딩 중...</p>
        </div>
      </div>
    );
  }

  // OAuth 콜백 처리 또는 로그인 페이지 표시
  // locationKey를 의존성에 추가하여 URL 변경 시 리렌더링 보장
  const urlParams = new URLSearchParams(window.location.search);
  const hasOAuthCallback = urlParams.get('code') && urlParams.get('state');
  const hasOAuthError = urlParams.get('error');
  const currentPath = window.location.pathname;
  
  // locationKey를 사용하여 URL 변경 시 리렌더링 보장
  // locationKey가 변경되면 이 컴포넌트가 리렌더링되어 URL 파라미터를 다시 확인
  void locationKey; // locationKey를 사용하여 리렌더링 보장
  
  // 이미 로그인되어 있고 OAuth 콜백이 아닌 경우 메인 화면으로 리다이렉트
  if (isAuthenticated && !hasOAuthCallback && !hasOAuthError && currentPath === '/login') {
    logger.info('App: Already authenticated, redirecting to main screen');
    window.history.replaceState({}, document.title, '/');
    setLocationKey(prev => prev + 1);
    return null;
  }
  
  if (hasOAuthCallback || hasOAuthError || currentPath === '/login') {
    logger.info('App: Showing LoginPage, hasOAuthCallback:', hasOAuthCallback, 'hasOAuthError:', hasOAuthError, 'currentPath:', currentPath);
    return <LoginPage />;
  }

  // 로그인하지 않은 사용자도 챗봇 사용 가능 (백엔드에서 익명 사용자 쿼터로 제한)

  return (
    <MainLayout
      sidebarContent={
        <SidebarContent
          searchQuery={searchQuery}
          onNewSession={handleNewSession}
          onSearch={setSearchQuery}
          onSelectSession={handleSelectSession}
          onRenameSession={handleRenameSession}
          onCopySession={handleCopySession}
          onExportSession={handleExportSession}
          onDeleteSession={handleDeleteSession}
          isLoading={isLoading}
          refreshTrigger={sessionListRefreshTrigger}
        />
      }
    >
      {/* Toast 컨테이너 */}
      <div className="fixed top-4 right-4 z-50 space-y-2 pointer-events-none">
        {toasts.map((toast) => (
          <div key={toast.id} className="pointer-events-auto">
            <Toast
              message={toast.message}
              type={toast.type}
              onClose={() => hideToast(toast.id)}
              action={toast.action}
            />
          </div>
        ))}
      </div>

      <div className={`flex-1 flex flex-col overflow-hidden min-h-0 transition-all duration-300 ease-in-out ${
        (documentSidebarOpen || referencesSidebarOpen) ? 'mr-[28rem]' : 'mr-0'
      }`}>
        <ChatContainer
          messages={messages}
          sessionId={currentSession?.session_id}
          isSending={isSending}
          isStreaming={isStreaming}
          currentProgress={currentProgress}
          progressHistory={progressHistory}
          streamingMessageId={streamingMessageId}
          streamErrors={streamErrors}
          quotaInfo={quotaInfo}
          isAuthenticated={isAuthenticated}
          inputResetTrigger={inputResetTrigger}
          onQuestionClick={handleRelatedQuestionClick}
          onSendMessage={handleSendMessage}
          onDocumentClick={handleDocumentClick}
          onOpenReferencesSidebar={(message, selectedType, referenceId) => {
            openReferencesSidebar(message, selectedType, referenceId);
          }}
          onRetryMessage={(messageId) => {
            const error = streamErrors.get(messageId);
            if (error && error.canRetry && currentSession) {
              const message = messages.find(m => m.id === messageId);
              if (message && message.role === 'assistant') {
                const messageIndex = messages.findIndex(m => m.id === messageId);
                if (messageIndex > 0) {
                  const userMessage = messages[messageIndex - 1];
                  if (userMessage && userMessage.role === 'user') {
                    handleStreamingMessage(userMessage.content, currentSession.session_id, userMessage.attachments, undefined, undefined, undefined, true);
                  }
                }
              }
            }
          }}
          onRetryQuestion={handleRetryQuestion}
          onLoginClick={login}
        />
      </div>

      {/* 문서 사이드바 */}
      {selectedMessageForDocument && (
        <DocumentSidebar
          isOpen={documentSidebarOpen}
          onClose={closeDocumentSidebar}
          documentIndex={selectedDocumentIndex}
          sources={selectedMessageForDocument.metadata?.sources}
          sourcesDetail={selectedMessageForDocument.metadata?.sources_detail}
          sourcesByType={selectedMessageForDocument.metadata?.sources_by_type}
          metadata={selectedMessageForDocument.metadata}
          sessionId={currentSession?.session_id}
          messageId={selectedMessageForDocument.id}
        />
      )}

      {/* 참고자료 사이드바 */}
      {selectedMessageForReferences && (
        <ReferencesSidebar
          isOpen={referencesSidebarOpen}
          onClose={closeReferencesSidebar}
          references={selectedMessageForReferences.metadata?.sources}
          legalReferences={selectedMessageForReferences.metadata?.legal_references}
          sources={selectedMessageForReferences.metadata?.sources}
          sourcesDetail={selectedMessageForReferences.metadata?.sources_detail}
          sourcesByType={selectedMessageForReferences.metadata?.sources_by_type}
          initialSelectedType={selectedReferenceType}
          initialSelectedReferenceId={selectedReferenceId}
        />
      )}
    </MainLayout>
  );
}

export default App;

