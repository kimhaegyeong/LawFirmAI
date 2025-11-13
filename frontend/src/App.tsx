/**
 * 메인 App 컴포넌트
 */
import { useState, useEffect, useRef } from 'react';
import { MainLayout } from './components/layout/MainLayout';
import { SidebarContent } from './components/sidebar/SidebarContent';
import { WelcomeScreen } from './components/welcome/WelcomeScreen';
import { ChatHistory } from './components/chat/ChatHistory';
import { ChatInput } from './components/chat/ChatInput';
import { Toast } from './components/common/Toast';
import { DocumentSidebar } from './components/chat/DocumentSidebar';
import { ReferencesSidebar } from './components/chat/ReferencesSidebar';
import { QuotaIndicator } from './components/chat/QuotaIndicator';
import { LoginPage } from './components/auth/LoginPage';
import { useChat } from './hooks/useChat';
import { useSession } from './hooks/useSession';
import { useAuth } from './hooks/useAuth';
import { getHistory, exportHistory, downloadHistory } from './services/historyService';
import { createSession } from './services/sessionService';
import { cleanupOldOAuthKeys } from './services/authService';
import { parseStreamChunk, type ParsedChunk } from './utils/streamParser';
import { classifyStreamError, StreamError } from './types/error';
import logger from './utils/logger';
import { convertImageToBase64, convertFileToBase64, isImageFile, isDocumentFile } from './utils/fileUtils';
import type { ChatMessage, FileAttachment, SourceInfo } from './types/chat';
import type { Session } from './types/session';

interface ToastItem {
  id: string;
  message: string;
  type: 'error' | 'success' | 'info' | 'warning';
  action?: {
    label: string;
    onClick: () => void;
  };
}

function App() {
  const { isAuthenticated, isLoading: isAuthLoading, login } = useAuth();
  const [messages, setMessages] = useState<ChatMessage[]>([]);
  const [searchQuery, setSearchQuery] = useState('');
  const [locationKey, setLocationKey] = useState(0);
  const [currentProgress, setCurrentProgress] = useState<string | null>(null);
  const [progressHistory, setProgressHistory] = useState<string[]>([]);
  const [streamingMessageId, setStreamingMessageId] = useState<string | null>(null); // 현재 스트리밍 중인 메시지 ID
  const [, setMessageBuffers] = useState<Map<string, string>>(new Map()); // 메시지 버퍼
  const [inputResetTrigger, setInputResetTrigger] = useState(0); // 입력창 초기화 트리거
  const [toasts, setToasts] = useState<ToastItem[]>([]);
  const [streamErrors, setStreamErrors] = useState<Map<string, StreamError>>(new Map());
  const [quotaInfo, setQuotaInfo] = useState<{ remaining: number; limit: number } | null>(null); // 익명 사용자 쿼터 정보
  const [documentSidebarOpen, setDocumentSidebarOpen] = useState(false);
  const [selectedDocumentIndex, setSelectedDocumentIndex] = useState<number | null>(null);
  const [selectedMessageForDocument, setSelectedMessageForDocument] = useState<ChatMessage | null>(null);
  const [referencesSidebarOpen, setReferencesSidebarOpen] = useState(false);
  const [selectedMessageForReferences, setSelectedMessageForReferences] = useState<ChatMessage | null>(null);
  const [selectedReferenceType, setSelectedReferenceType] = useState<'all' | 'law' | 'precedent' | 'decision' | 'interpretation' | 'regulation'>('all');
  
  // 토큰 배치 업데이트를 위한 ref
  const tokenBufferRef = useRef<Map<string, string>>(new Map()); // 메시지별 토큰 버퍼
  const tokenBufferTimeoutRef = useRef<Map<string, number>>(new Map()); // 메시지별 타이머 (setTimeout 반환값)
  
  // 초기 마운트 여부 추적 (StrictMode 대응)
  const isInitialMount = useRef(true);

  const { currentSession, isLoading, loadSessions, newSession, loadSession, updateSession, removeSession, clearSession } = useSession();
  const { sendStreamingMessage, isLoading: isSending, isStreaming } = useChat({
    onMessage: (message) => {
      setMessages((prev) => [...prev, message]);
      // 에러 메시지가 있으면 제거
      if (streamErrors.has(message.id)) {
        setStreamErrors((prev) => {
          const newMap = new Map(prev);
          newMap.delete(message.id);
          return newMap;
        });
      }
    },
    onError: (error) => {
      logger.error('Chat error:', error);
      
      let errorMessage = error instanceof Error ? error.message : String(error);
      
      // Axios 에러의 경우 detail 필드 확인
      if (error && typeof error === 'object' && 'response' in error) {
        const axiosError = error as any;
        if (axiosError.response?.data?.detail) {
          errorMessage = axiosError.response.data.detail;
        }
      }
      
      // 로그인한 사용자에게는 익명 사용자 쿼터 메시지를 필터링
      if (isAuthenticated && (
        errorMessage.includes('무료 질의 3회를 모두 사용하셨습니다') ||
        errorMessage.includes('무료로 3회 체험')
      )) {
        errorMessage = '요청이 너무 많습니다. 잠시 후 다시 시도하세요.';
      }
      
      const streamError = classifyStreamError(new Error(errorMessage));
      
      // Toast 표시 (로그인한 사용자에게는 익명 사용자 쿼터 메시지 표시 안 함)
      if (!isAuthenticated || !streamError.message.includes('무료 질의 3회를 모두 사용하셨습니다')) {
        const toastId = `toast-${Date.now()}`;
        setToasts((prev) => [...prev, {
          id: toastId,
          message: streamError.message,
          type: 'error',
          action: streamError.canRetry ? {
            label: '다시 시도',
            onClick: () => {
              setToasts((prev) => prev.filter(t => t.id !== toastId));
              // 재시도 로직은 필요시 구현
            }
          } : undefined,
        }]);
      }
    },
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
              const apiError = error as any;
              if (apiError.status === 429 && !isAuthenticated) {
                // 에러 객체에서 쿼터 정보 추출
                if (apiError.quotaInfo) {
                  setQuotaInfo(apiError.quotaInfo);
                } else if (apiError.response?.headers) {
                  // 응답 헤더에서 쿼터 정보 추출 (fallback)
                  const quotaRemaining = apiError.response.headers['x-quota-remaining'];
                  const quotaLimit = apiError.response.headers['x-quota-limit'];
                  
                  if (quotaRemaining !== undefined && quotaLimit !== undefined) {
                    setQuotaInfo({ 
                      remaining: parseInt(quotaRemaining, 10), 
                      limit: parseInt(quotaLimit, 10) 
                    });
                  } else {
                    // 헤더 정보가 없으면 기본값 설정
                    setQuotaInfo({ remaining: 0, limit: 3 });
                  }
                } else {
                  // 기본값 설정
                  setQuotaInfo({ remaining: 0, limit: 3 });
                }
                
                // Toast 메시지 표시
                setToasts((prev) => [
                  ...prev,
                  {
                    id: `quota-exceeded-${Date.now()}`,
                    message: '무료 질의 3회를 모두 사용하셨습니다. 계속 사용하려면 로그인이 필요합니다.',
                    type: 'warning',
                    action: {
                      label: '로그인',
                      onClick: () => login(),
                    },
                  },
                ]);
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
  }, [loadSession, isAuthenticated, login]);

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
            setMessages(chatMessages);
          }
        } catch (error) {
          logger.error('Failed to load session messages:', error);
          // 에러 발생 시에도 현재 메시지를 유지 (빈 배열로 덮어쓰지 않음)
        }
      } else {
        setMessages([]);
      }
    };
    
    loadSessionMessages();
  }, [currentSession]);

  const handleNewSession = async () => {
    logger.debug('[DEBUG] handleNewSession called');
    try {
      // 웰컴 화면을 표시하기 위해 세션을 초기화하고 메시지도 초기화
      clearSession();
      setMessages([]);
      // 입력창 초기화
      setInputResetTrigger((prev) => prev + 1);
      logger.debug('[DEBUG] Session and messages cleared for welcome screen');
    } catch (error) {
      logger.error('[ERROR] Failed to reset session:', error);
      const errorMessage = error instanceof Error 
        ? error.message 
        : '새 상담을 시작할 수 없습니다. 다시 시도해주세요.';
      alert(errorMessage);
    }
  };

  const handleSelectSession = async (session: Session) => {
    try {
      await loadSession(session.session_id);
    } catch (error) {
      logger.error('Failed to load session:', error);
    }
  };

  const handleRenameSession = async (session: Session) => {
    const newTitle = prompt('새 제목을 입력하세요:', session.title || '');
    if (newTitle && newTitle !== session.title) {
      try {
        await updateSession(session.session_id, { title: newTitle });
      } catch (error) {
        logger.error('Failed to rename session:', error);
      }
    }
  };

  const handleCopySession = async (session: Session) => {
    try {
      // 새 세션 생성
      const newSession = await createSession({
        title: `${session.title || '제목 없음'} (복사본)`,
      });
      
      // 기존 세션의 메시지 가져오기 (현재는 사용하지 않음)
      // const history = await getHistory({ session_id: session.session_id });
      
      // 새 세션에 메시지 복사 (백엔드 API를 통해 메시지 추가)
      // 주의: 현재 백엔드 API는 메시지 추가를 POST /api/chat을 통해서만 지원
      // 따라서 실제로는 메시지를 다시 전송하는 방식으로 구현
      // 또는 백엔드에 세션 복사 전용 엔드포인트 추가 필요
      
      // 임시: 세션 목록 새로고침
      await loadSessions();
      
      // 새 세션 불러오기
      await loadSession(newSession.session_id);
      
      alert('세션이 복사되었습니다. 메시지는 새 대화에서 다시 전송해야 합니다.');
    } catch (error) {
      logger.error('Failed to copy session:', error);
      alert('세션 복사에 실패했습니다.');
    }
  };

  const handleExportSession = async (session: Session) => {
    try {
      const blob = await exportHistory({
        session_ids: [session.session_id],
        format: 'txt',
      });
      
      const filename = `세션_${session.title || session.session_id}_${new Date().toISOString().split('T')[0]}.txt`;
      downloadHistory(blob, filename);
    } catch (error) {
      logger.error('Failed to export session:', error);
      alert('세션 내보내기에 실패했습니다.');
    }
  };

  const handleDeleteSession = async (session: Session) => {
    if (confirm(`정말 이 세션을 삭제하시겠습니까?\n"${session.title || '제목 없음'}"`)) {
      try {
        await removeSession(session.session_id);
        if (currentSession?.session_id === session.session_id) {
          setMessages([]);
        }
      } catch (error) {
        logger.error('Failed to delete session:', error);
      }
    }
  };

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

  const handleStreamingMessage = async (
    message: string,
    sessionId: string,
    attachments?: FileAttachment[],
    imageBase64?: string,
    fileBase64?: string,
    filename?: string,
    skipUserMessage: boolean = false
  ) => {
    // 사용자 메시지 추가 (skipUserMessage가 false일 때만)
    if (!skipUserMessage) {
      const userMessage: ChatMessage = {
        id: `user-${Date.now()}`,
        role: 'user',
        content: message,
        timestamp: new Date(),
        attachments,
      };
      setMessages((prev) => [...prev, userMessage]);
    }

    // AI 답변 메시지 ID 생성 (첫 번째 chunk를 받았을 때 추가)
    const assistantMessageId = `assistant-${Date.now()}`;
    const assistantMessage: ChatMessage = {
      id: assistantMessageId,
      role: 'assistant',
      content: '',
      timestamp: new Date(),
    };
    
    // 스트리밍 시작 시 메시지 ID는 첫 번째 chunk를 받았을 때 설정
    // (type: "stream" 이벤트를 받았을 때 설정하여 안전성 보장)
    
    // 첫 번째 chunk를 받았는지 추적
    let isFirstChunk = true;

    // 스트리밍 메시지 전송
    let fullContent = '';
    try {
      // 진행 상황 초기화 (기존 코드와의 호환성을 위해 유지, 나중에 제거 예정)
      setCurrentProgress(null);
      setProgressHistory([]);
      
      await sendStreamingMessage(
        message,
        sessionId,
        (chunk) => {
          if (import.meta.env.DEV) {
            logger.debug('[Stream] Chunk received:', JSON.stringify(chunk));
          }
          
          // 스트림 청크 파싱
          const parsed = parseStreamChunk(chunk);
          
          // 쿼터 정보 처리
          if (parsed.type === 'quota' && parsed.metadata) {
            const quotaRemaining = parsed.metadata.remaining;
            const quotaLimit = parsed.metadata.limit;
            if (quotaRemaining !== undefined && quotaLimit !== undefined) {
              setQuotaInfo({ remaining: quotaRemaining, limit: quotaLimit });
            }
            return; // 쿼터 정보는 메시지로 표시하지 않음
          }
          
          if (parsed.type === 'progress') {
            // 진행 상황을 메시지 배열에 추가
            setMessages((prev) => {
              // 마지막 진행 상황 메시지 찾기 (역순으로 검색)
              let lastProgressIndex = -1;
              for (let i = prev.length - 1; i >= 0; i--) {
                // eslint-disable-next-line security/detect-object-injection
                const msg = prev[i];
                if (msg && msg.role === 'progress') {
                  lastProgressIndex = i;
                  break;
                }
              }
              
              if (lastProgressIndex !== -1) {
                // 기존 진행 상황 메시지 업데이트
                const updated = [...prev];
                // eslint-disable-next-line security/detect-object-injection
                const existingMsg = updated[lastProgressIndex];
                if (existingMsg) {
                  // eslint-disable-next-line security/detect-object-injection
                  updated[lastProgressIndex] = {
                    ...existingMsg,
                    content: parsed.content,
                    timestamp: new Date(),
                    metadata: parsed.metadata ? {
                      ...existingMsg.metadata,
                      ...parsed.metadata,
                    } : existingMsg.metadata,
                  };
                }
                if (import.meta.env.DEV) {
                  logger.debug('[Stream] Progress message updated:', parsed.content);
                }
                return updated;
              } else {
                // 새로운 진행 상황 메시지 추가
                const progressMessage: ChatMessage = {
                  id: `progress-${Date.now()}`,
                  role: 'progress',
                  content: parsed.content,
                  timestamp: new Date(),
                  metadata: parsed.metadata ? {
                    ...parsed.metadata,
                  } : undefined,
                };
                if (import.meta.env.DEV) {
                  logger.debug('[Stream] Progress message added:', parsed.content);
                }
                return [...prev, progressMessage];
              }
            });
            
            // 기존 코드와의 호환성을 위해 currentProgress도 업데이트 (나중에 제거 예정)
            setCurrentProgress(parsed.content);
          } else if (parsed.type === 'stream') {
            // 실시간 스트리밍 (토큰 단위로 도착하는 데이터를 즉시 업데이트)
            // 받은 청크를 fullContent에 누적
            fullContent += parsed.content;
            if (import.meta.env.DEV) {
              logger.debug('[Stream] Token received, full content so far length:', fullContent.length);
            }
            
            // streamingMessageId를 첫 번째 chunk를 받기 전에 미리 설정
            if (isFirstChunk) {
              setStreamingMessageId(assistantMessageId);
              if (import.meta.env.DEV) {
                logger.debug('[Stream] First chunk received, setting streamingMessageId:', assistantMessageId);
              }
            }
            
            // 배치 업데이트 제거, 즉시 업데이트
            setMessageBuffers(prev => {
              const newMap = new Map(prev);
              const currentBuffer = newMap.get(assistantMessageId) || '';
              const updatedBuffer = currentBuffer + parsed.content;
              newMap.set(assistantMessageId, updatedBuffer);
              
              // 버퍼 업데이트 후 메시지도 즉시 업데이트
              setMessages((prevMessages) => {
                const messageIndex = prevMessages.findIndex((msg) => msg.id === assistantMessageId);
                
                // 첫 번째 chunk를 받았을 때만 메시지 추가
                if (isFirstChunk && messageIndex === -1) {
                  if (import.meta.env.DEV) {
                    logger.debug('[Stream] First chunk received, adding assistant message:', assistantMessageId, 'Content length:', updatedBuffer.length);
                  }
                  return [...prevMessages, { ...assistantMessage, content: updatedBuffer }];
                }
                
                // 이미 메시지가 있으면 실시간 업데이트
                if (messageIndex !== -1) {
                  const updated = [...prevMessages];
                  // eslint-disable-next-line security/detect-object-injection
                  const existingMsg = updated[messageIndex];
                  if (existingMsg) {
                    // eslint-disable-next-line security/detect-object-injection
                    updated[messageIndex] = { 
                      ...existingMsg, 
                      content: updatedBuffer 
                    };
                  }
                  if (import.meta.env.DEV && updatedBuffer.length % 50 === 0) {
                    logger.debug('[Stream] Message updated at index:', messageIndex, 'Content length:', updatedBuffer.length);
                  }
                  return updated;
                }
                
                // 메시지가 없으면 추가 (fallback)
                if (import.meta.env.DEV) {
                  logger.warn('[Stream] Message not found, adding as fallback:', assistantMessageId);
                }
                return [...prevMessages, { ...assistantMessage, content: updatedBuffer }];
              });
              
              return newMap;
            });
            
            // 첫 번째 chunk 플래그 업데이트
            if (isFirstChunk) {
              isFirstChunk = false;
            }
          } else if (parsed.type === 'chunk') {
            // chunk 타입 이벤트 처리 (첫 번째 청크)
            fullContent = parsed.content;
            
            if (isFirstChunk) {
              setStreamingMessageId(assistantMessageId);
              if (import.meta.env.DEV) {
                logger.debug('[Stream] First chunk received, setting streamingMessageId:', assistantMessageId);
              }
            }
            
            // 메시지 업데이트
            setMessageBuffers(prev => {
              const newMap = new Map(prev);
              newMap.set(assistantMessageId, parsed.content);
              return newMap;
            });
            
            setMessages((prevMessages) => {
              const messageIndex = prevMessages.findIndex((msg) => msg.id === assistantMessageId);
              
              if (isFirstChunk && messageIndex === -1) {
                // 첫 번째 청크를 받았을 때 메시지 추가
                return [...prevMessages, {
                  ...assistantMessage,
                  content: parsed.content,
                  metadata: {
                    message_id: parsed.metadata?.message_id,
                  },
                }];
              } else if (messageIndex !== -1) {
                // 기존 메시지 업데이트
                const updated = [...prevMessages];
                // eslint-disable-next-line security/detect-object-injection
                const existingMsg = updated[messageIndex];
                if (existingMsg) {
                  // eslint-disable-next-line security/detect-object-injection
                  updated[messageIndex] = {
                    ...existingMsg,
                    content: parsed.content,
                    metadata: {
                      ...existingMsg.metadata,
                      message_id: parsed.metadata?.message_id || existingMsg.metadata?.message_id,
                    },
                  };
                }
                return updated;
              }
              return prevMessages;
            });
            
            if (isFirstChunk) {
              isFirstChunk = false;
            }
          } else if (parsed.type === 'sources') {
            // sources 이벤트 처리 (별도 이벤트)
            if (parsed.metadata) {
              const sourcesMetadata = parsed.metadata;
              const sourcesMessageId = sourcesMetadata.message_id;
              const relatedQuestions = sourcesMetadata.related_questions as string[] | undefined;
              
              if (import.meta.env.DEV) {
                logger.debug('[Stream] Sources event received:', {
                  messageId: sourcesMessageId,
                  assistantMessageId,
                  sources: sourcesMetadata.sources,
                  legalReferences: sourcesMetadata.legal_references,
                  sourcesDetail: sourcesMetadata.sources_detail,
                  relatedQuestions: relatedQuestions,
                });
              }
              
              // message_id로 메시지 찾기 (여러 방법 시도)
              setMessages((prev) => {
                let messageIndex = -1;
                let foundMessageId: string | null = null;
                
                // 1. sourcesMessageId로 metadata.message_id와 일치하는 메시지 찾기
                if (sourcesMessageId) {
                  messageIndex = prev.findIndex((msg) => 
                    msg.metadata?.message_id === sourcesMessageId
                  );
                  if (messageIndex !== -1) {
                    // eslint-disable-next-line security/detect-object-injection
                    foundMessageId = prev[messageIndex]?.id || null;
                    if (import.meta.env.DEV) {
                      logger.debug('[Stream] Found message by metadata.message_id:', foundMessageId);
                    }
                  }
                }
                
                // 2. assistantMessageId로 메시지 찾기
                if (messageIndex === -1 && assistantMessageId) {
                  messageIndex = prev.findIndex((msg) => 
                    msg.id === assistantMessageId
                  );
                  if (messageIndex !== -1) {
                    foundMessageId = assistantMessageId;
                    if (import.meta.env.DEV) {
                      logger.debug('[Stream] Found message by assistantMessageId:', foundMessageId);
                    }
                  }
                }
                
                // 3. sourcesMessageId를 id로 사용하는 메시지 찾기
                if (messageIndex === -1 && sourcesMessageId) {
                  messageIndex = prev.findIndex((msg) => 
                    msg.id === sourcesMessageId
                  );
                  if (messageIndex !== -1) {
                    foundMessageId = sourcesMessageId;
                    if (import.meta.env.DEV) {
                      logger.debug('[Stream] Found message by sourcesMessageId as id:', foundMessageId);
                    }
                  }
                }
                
                // 4. 마지막 assistant 메시지 찾기 (fallback)
                if (messageIndex === -1) {
                  for (let i = prev.length - 1; i >= 0; i--) {
                    // eslint-disable-next-line security/detect-object-injection
                    const msg = prev[i];
                    if (msg && msg.role === 'assistant') {
                      messageIndex = i;
                      foundMessageId = msg.id;
                      if (import.meta.env.DEV) {
                        logger.debug('[Stream] Found message by last assistant message (fallback):', foundMessageId);
                      }
                      break;
                    }
                  }
                }
                
                if (messageIndex !== -1) {
                  const updated = [...prev];
                  // eslint-disable-next-line security/detect-object-injection
                  const existingMsg = updated[messageIndex];
                  if (existingMsg) {
                    const sources = (sourcesMetadata.sources as string[] | undefined) || [];
                    const legalReferences = (sourcesMetadata.legal_references as string[] | undefined) || [];
                    const sourcesDetail = (sourcesMetadata.sources_detail as SourceInfo[] | undefined) || [];
                    const finalRelatedQuestions = Array.isArray(relatedQuestions) ? relatedQuestions : (existingMsg.metadata?.related_questions || []);
                    
                    // eslint-disable-next-line security/detect-object-injection
                    updated[messageIndex] = {
                      ...existingMsg,
                      metadata: {
                        ...existingMsg.metadata,
                        sources: sources,
                        legal_references: legalReferences,
                        sources_detail: sourcesDetail,
                        message_id: sourcesMessageId || existingMsg.metadata?.message_id,
                        related_questions: finalRelatedQuestions,
                      },
                    };
                    
                    if (import.meta.env.DEV) {
                      logger.debug('[Stream] Message metadata updated with sources:', {
                        messageIndex,
                        messageId: existingMsg?.id,
                        metadataMessageId: existingMsg?.metadata?.message_id,
                        sourcesMessageId,
                        sourcesCount: sources.length,
                        legalReferencesCount: legalReferences.length,
                        sourcesDetailCount: sourcesDetail.length,
                        relatedQuestionsCount: finalRelatedQuestions.length,
                        hasRelatedQuestions: finalRelatedQuestions.length > 0,
                      });
                    }
                  }
                  
                  return updated;
                } else {
                  if (import.meta.env.DEV) {
                    logger.warn('[Stream] Sources event: Message not found, attempting to add new message', {
                      sourcesMessageId,
                      assistantMessageId,
                      messageIds: prev.map(msg => ({ id: msg.id, role: msg.role, metadataMessageId: msg.metadata?.message_id })),
                    });
                  }
                  
                  // 메시지를 찾지 못했으면 마지막 assistant 메시지에 추가하거나 새로 생성
                  const lastAssistantIndex = prev.length - 1;
                  if (lastAssistantIndex >= 0 && prev[lastAssistantIndex]?.role === 'assistant') {
                    const updated = [...prev];
                    // eslint-disable-next-line security/detect-object-injection
                    const lastMsg = updated[lastAssistantIndex];
                    if (lastMsg) {
                      const sources = (sourcesMetadata.sources as string[] | undefined) || [];
                      const legalReferences = (sourcesMetadata.legal_references as string[] | undefined) || [];
                      const sourcesDetail = (sourcesMetadata.sources_detail as SourceInfo[] | undefined) || [];
                      const finalRelatedQuestions = Array.isArray(relatedQuestions) ? relatedQuestions : [];
                      
                      // eslint-disable-next-line security/detect-object-injection
                      updated[lastAssistantIndex] = {
                        ...lastMsg,
                        metadata: {
                          ...lastMsg.metadata,
                          sources: sources,
                          legal_references: legalReferences,
                          sources_detail: sourcesDetail,
                          message_id: sourcesMessageId || lastMsg.metadata?.message_id,
                          related_questions: finalRelatedQuestions,
                        },
                      };
                      
                      if (import.meta.env.DEV) {
                        logger.debug('[Stream] Updated last assistant message with sources (fallback):', {
                          messageId: lastMsg.id,
                          sourcesCount: sources.length,
                          relatedQuestionsCount: finalRelatedQuestions.length,
                        });
                      }
                      
                      return updated;
                    }
                  }
                }
                return prev;
              });
              
              // sources 이벤트 수신 시 스트리밍 완료 처리 (done 이벤트가 오지 않은 경우 대비)
              // sources 이벤트가 오면 무조건 스트리밍 완료로 간주
              setStreamingMessageId((currentStreamingId) => {
                if (currentStreamingId !== null) {
                  if (import.meta.env.DEV) {
                    logger.debug('[Stream] StreamingMessageId set to null after sources event, was:', currentStreamingId, 'sourcesMessageId:', sourcesMessageId, 'assistantMessageId:', assistantMessageId);
                  }
                  return null;
                }
                return currentStreamingId;
              });
            }
          } else if (parsed.metadata && 'sources' in parsed.metadata) {
            // final 이벤트의 metadata에 sources가 있는 경우 (기존 로직 유지)
            if (parsed.metadata) {
              const sourcesMetadata = parsed.metadata;
              const sourcesMessageId = sourcesMetadata.message_id;
              const relatedQuestions = sourcesMetadata.related_questions as string[] | undefined;
              
              if (import.meta.env.DEV) {
                logger.debug('[Stream] Sources in final metadata:', {
                  messageId: sourcesMessageId,
                  sources: sourcesMetadata.sources,
                  legalReferences: sourcesMetadata.legal_references,
                  sourcesDetail: sourcesMetadata.sources_detail,
                  relatedQuestions: relatedQuestions,
                });
              }
              
              // message_id로 메시지 찾기 (assistantMessageId 또는 sourcesMessageId 사용)
              const targetMessageId = sourcesMessageId || assistantMessageId;
              
              setMessages((prev) => {
                const messageIndex = prev.findIndex((msg) => 
                  msg.id === targetMessageId || 
                  msg.metadata?.message_id === sourcesMessageId
                );
                
                if (messageIndex !== -1) {
                  const updated = [...prev];
                  // eslint-disable-next-line security/detect-object-injection
                  const existingMsg = updated[messageIndex];
                  if (existingMsg) {
                    // eslint-disable-next-line security/detect-object-injection
                    updated[messageIndex] = {
                      ...existingMsg,
                      metadata: {
                        ...existingMsg.metadata,
                        sources: (sourcesMetadata.sources as string[] | undefined) || [],
                        legal_references: (sourcesMetadata.legal_references as string[] | undefined) || [],
                        sources_detail: (sourcesMetadata.sources_detail as SourceInfo[] | undefined) || [],
                        message_id: sourcesMessageId || existingMsg.metadata?.message_id,
                        related_questions: Array.isArray(relatedQuestions) ? relatedQuestions : existingMsg.metadata?.related_questions,
                      },
                    };
                  }
                  
                  if (import.meta.env.DEV && existingMsg) {
                    logger.debug('[Stream] Message metadata updated with sources from final:', {
                      messageId: targetMessageId,
                      updatedMetadata: updated[messageIndex]?.metadata,
                    });
                  }
                  
                  return updated;
                }
                return prev;
              });
            }
          } else if (parsed.type === 'final') {
            // final 이벤트: 이전 stream 데이터 삭제하고 final content로 전체 답변 교체
            // final content가 비어있으면 fullContent 사용 (네트워크 에러 등으로 인한 fallback)
            const finalContent = parsed.content && parsed.content.trim() ? parsed.content : fullContent;
            
            if (import.meta.env.DEV) {
              logger.debug('[Stream] Final event received, replacing content with final:', {
                finalContentLength: finalContent.length,
                parsedContentLength: parsed.content?.length || 0,
                previousContentLength: fullContent.length,
                hasMetadata: !!parsed.metadata,
                usingFallback: !parsed.content || !parsed.content.trim(),
              });
              if (parsed.metadata) {
                logger.debug('[Stream] Final metadata:', parsed.metadata);
              }
            }
            
            // 남아있는 토큰 버퍼 정리
            tokenBufferRef.current.delete(assistantMessageId);
            if (tokenBufferTimeoutRef.current.has(assistantMessageId)) {
              clearTimeout(tokenBufferTimeoutRef.current.get(assistantMessageId)!);
              tokenBufferTimeoutRef.current.delete(assistantMessageId);
            }
            
            // 버퍼에 final content 저장 (이전 stream 데이터 대체)
            setMessageBuffers(prev => {
              const newMap = new Map(prev);
              newMap.set(assistantMessageId, finalContent);
              return newMap;
            });
            
            // fullContent를 final content로 업데이트
            fullContent = finalContent;
            
            // 스트리밍은 계속 진행 중이므로 streamingMessageId는 유지
            // (done 이벤트에서 null로 설정)
            
            // 메시지 content를 final content로 교체 (이전 stream 데이터 삭제)
            const metadata = parsed.metadata || {};
            const sources = metadata.sources || [];
            const legalReferences = metadata.legal_references || [];
            const sourcesDetail = metadata.sources_detail || [];
            const messageId = metadata.message_id;
            const relatedQuestions = metadata.related_questions as string[] | undefined;
            
            setMessages((prev) => {
              const messageIndex = prev.findIndex((msg) => msg.id === assistantMessageId);
              
              if (messageIndex !== -1) {
                const updated = [...prev];
                // eslint-disable-next-line security/detect-object-injection
                const existingMsg = updated[messageIndex];
                if (existingMsg) {
                  // eslint-disable-next-line security/detect-object-injection
                  updated[messageIndex] = {
                    ...existingMsg,
                    content: finalContent,  // final content로 교체 (이전 stream 데이터 삭제)
                    metadata: {
                      ...existingMsg.metadata,
                      ...metadata,
                      sources: Array.isArray(sources) ? sources : [],
                      legal_references: Array.isArray(legalReferences) ? legalReferences : [],
                      sources_detail: Array.isArray(sourcesDetail) ? sourcesDetail : [],
                      message_id: messageId || existingMsg.metadata?.message_id,
                      related_questions: Array.isArray(relatedQuestions) ? relatedQuestions : existingMsg.metadata?.related_questions,
                    },
                  };
                }
                return updated;
              } else {
                return [...prev, {
                  ...assistantMessage,
                  content: finalContent,
                  metadata: {
                    ...metadata,
                    sources: sources,
                    legal_references: legalReferences,
                    sources_detail: sourcesDetail,
                    message_id: messageId,
                    related_questions: Array.isArray(relatedQuestions) ? relatedQuestions : undefined,
                  },
                }];
              }
            });
            
          } else if (parsed.type === 'done') {
            // done 이벤트: 서버에서 보낸 최종 답변으로 교체 (타이핑 효과 없이)
            // done content가 비어있으면 fullContent 사용 (네트워크 에러 등으로 인한 fallback)
            const finalContent = parsed.content && parsed.content.trim() ? parsed.content : fullContent;
            
            if (import.meta.env.DEV) {
              logger.debug('[Stream] Done event received, final content length:', finalContent.length);
              if (parsed.metadata) {
                logger.debug('[Stream] Done metadata:', parsed.metadata);
              }
            }
            
            // 버퍼에 최종 내용 저장
            setMessageBuffers(prev => {
              const newMap = new Map(prev);
              newMap.set(assistantMessageId, finalContent);
              return newMap;
            });
            
            // 스트리밍 완료 (타이핑 효과 비활성화)
            setStreamingMessageId(null);
            
            if (import.meta.env.DEV) {
              logger.debug('[Stream] StreamingMessageId set to null after done event');
            }
            
            // 최종 메시지 업데이트 (서버에서 보낸 최종 답변으로 교체)
            const metadata = parsed.metadata || {};
            const sources = metadata.sources || [];
            const legalReferences = metadata.legal_references || [];
            const sourcesDetail = metadata.sources_detail || [];
            const messageId = metadata.message_id;
            const relatedQuestions = metadata.related_questions as string[] | undefined;
            
            setMessages((prev) => {
              const messageIndex = prev.findIndex((msg) => msg.id === assistantMessageId);
              
              if (messageIndex !== -1) {
                const updated = [...prev];
                const existingMsg = updated[messageIndex];
                if (existingMsg) {
                  updated[messageIndex] = {
                    ...existingMsg,
                    content: finalContent,  // 서버에서 보낸 최종 답변으로 교체
                    metadata: {
                      ...existingMsg.metadata,
                      ...metadata,
                      sources: Array.isArray(sources) ? sources : [],
                      legal_references: Array.isArray(legalReferences) ? legalReferences : [],
                      sources_detail: Array.isArray(sourcesDetail) ? sourcesDetail : [],
                      message_id: messageId || existingMsg.metadata?.message_id,
                      related_questions: Array.isArray(relatedQuestions) ? relatedQuestions : existingMsg.metadata?.related_questions,
                    },
                  };
                }
                return updated;
              } else {
                if (import.meta.env.DEV) {
                  logger.warn('[Stream] Done event received but message not found. It should have been added by final event.');
                }
                return prev;
              }
            });
            
            // sources가 비어있으면 별도 API로 가져오기
            // message_id가 있으면 사용하고, 없으면 assistantMessageId 사용
            const actualMessageId = messageId || assistantMessageId;
            
            if (import.meta.env.DEV) {
              logger.debug('[App] Done event sources check:', {
                sessionId,
                messageId: actualMessageId,
                sourcesLength: sources.length,
                legalReferencesLength: legalReferences.length,
                sourcesDetailLength: sourcesDetail.length,
                metadata: metadata,
              });
            }
            
            // sources가 비어있으면 API로 가져오기
            // (백엔드에서 sources가 제대로 전달되지 않을 수 있으므로)
            if (sources.length === 0 && legalReferences.length === 0 && sourcesDetail.length === 0) {
              if (import.meta.env.DEV) {
                logger.debug('[App] Sources are empty, fetching from API...', {
                  sessionId,
                  messageId: actualMessageId,
                });
              }
              
              // 비동기로 sources 가져오기 (UI 블로킹 방지)
              import('./services/chatService').then(({ getChatSources }) => {
                // message_id가 있으면 사용, 없으면 sessionId만 사용
                getChatSources(sessionId, actualMessageId || undefined)
                  .then((sourcesData) => {
                    if (import.meta.env.DEV) {
                      logger.debug('[App] Sources fetched from API:', {
                        sessionId,
                        messageId: actualMessageId,
                        sourcesData,
                      });
                    }
                    
                    // sources가 있는 경우에만 업데이트
                    if (sourcesData.sources.length > 0 || sourcesData.legal_references.length > 0 || sourcesData.sources_detail.length > 0) {
                      // 메시지 metadata 업데이트
                      setMessages((prev) => {
                        const messageIndex = prev.findIndex((msg) => msg.id === assistantMessageId);
                        
                        if (messageIndex !== -1) {
                          const updated = [...prev];
                          // eslint-disable-next-line security/detect-object-injection
                          updated[messageIndex] = {
                            // eslint-disable-next-line security/detect-object-injection
                            ...updated[messageIndex],
                            metadata: {
                              // eslint-disable-next-line security/detect-object-injection
                              ...updated[messageIndex].metadata,
                              sources: sourcesData.sources,
                              legal_references: sourcesData.legal_references,
                              sources_detail: sourcesData.sources_detail,
                            },
                          };
                          
                          if (import.meta.env.DEV) {
                            logger.debug('[App] Message metadata updated with sources:', {
                              messageId: assistantMessageId,
                              // eslint-disable-next-line security/detect-object-injection
                              updatedMetadata: updated[messageIndex].metadata,
                            });
                          }
                          
                          return updated;
                        }
                        return prev;
                      });
                    } else {
                      if (import.meta.env.DEV) {
                        logger.warn('[App] Sources fetched but empty:', {
                          sessionId,
                          messageId: actualMessageId,
                          sourcesData,
                        });
                      }
                    }
                  })
                  .catch((error) => {
                    if (import.meta.env.DEV) {
                      logger.error('[App] Error fetching sources:', error);
                    }
                    // 에러가 발생해도 계속 진행 (sources 없이)
                  });
              });
            } else {
              if (import.meta.env.DEV) {
                logger.debug('[App] Sources already present in metadata:', {
                  sessionId,
                  messageId: actualMessageId,
                  sources,
                  legalReferences,
                  sourcesDetail,
                });
              }
            }
          } else if ((parsed as ParsedChunk & { type: 'answer' }).type === 'answer') {
            // 기존 'answer' 타입 지원 (하위 호환성)
            fullContent += parsed.content;
            if (import.meta.env.DEV) {
              logger.debug('[Stream] Full content so far length:', fullContent.length);
            }
            
            // 실시간으로 메시지 업데이트
            setMessages((prev) => {
              const messageIndex = prev.findIndex((msg) => msg.id === assistantMessageId);
              
              if (isFirstChunk && messageIndex === -1) {
                isFirstChunk = false;
                return [...prev, { ...assistantMessage, content: fullContent }];
              }
              
              if (messageIndex !== -1) {
                const updated = [...prev];
                updated[messageIndex] = { ...updated[messageIndex], content: fullContent };
                return updated;
              }
              
              return [...prev, { ...assistantMessage, content: fullContent }];
            });
          }
        },
        imageBase64,
        fileBase64,
        filename
      );
      
      // 스트리밍 완료 시 현재 진행 상황만 초기화 (히스토리는 유지)
      setCurrentProgress(null);
      if (import.meta.env.DEV) {
        logger.debug('[Stream] Streaming completed. Final content length:', fullContent.length);
      }
    } catch (error) {
      logger.error('[Stream] Streaming error:', error);
      
      // 에러 발생 시에도 스트리밍 상태 해제 및 받은 데이터 저장
      // streamingMessageId가 설정되어 있으면 해제 (에러 발생 시 항상 해제)
      setStreamingMessageId((currentId) => {
        if (currentId === assistantMessageId || currentId !== null) {
          if (import.meta.env.DEV) {
            logger.debug('[Stream] StreamingMessageId set to null due to error, was:', currentId);
          }
          return null;
        }
        return currentId;
      });
      
      // 받은 데이터가 있으면 최종 메시지로 저장
      if (fullContent.trim()) {
        setMessages((prev) => {
          const messageIndex = prev.findIndex((msg) => msg.id === assistantMessageId);
          
          if (messageIndex !== -1) {
            const updated = [...prev];
            const existingMsg = updated[messageIndex];
            if (existingMsg) {
              updated[messageIndex] = {
                ...existingMsg,
                content: fullContent,
              };
            }
            return updated;
          }
          return prev;
        });
        
        // 버퍼에도 저장
        setMessageBuffers(prev => {
          const newMap = new Map(prev);
          newMap.set(assistantMessageId, fullContent);
          return newMap;
        });
        
        if (import.meta.env.DEV) {
          logger.debug('[Stream] Saved partial content due to error, length:', fullContent.length);
        }
      }
      
      // 429 에러 처리 - 쿼터 정보 업데이트 (익명 사용자만)
      if (error && typeof error === 'object' && 'status' in error) {
        const apiError = error as any;
        if (apiError.status === 429 && !isAuthenticated) {
          // 에러 객체에서 쿼터 정보 추출
          if (apiError.quotaInfo) {
            setQuotaInfo(apiError.quotaInfo);
            logger.debug('[Stream] Quota info updated from error:', apiError.quotaInfo);
          } else if (apiError.response?.headers) {
            // 응답 헤더에서 쿼터 정보 추출
            const quotaRemaining = apiError.response.headers['x-quota-remaining'];
            const quotaLimit = apiError.response.headers['x-quota-limit'];
            if (quotaRemaining !== undefined && quotaLimit !== undefined) {
              setQuotaInfo({ 
                remaining: parseInt(quotaRemaining, 10), 
                limit: parseInt(quotaLimit, 10) 
              });
              logger.debug('[Stream] Quota info updated from headers:', { remaining: quotaRemaining, limit: quotaLimit });
            } else {
              // 기본값 설정
              setQuotaInfo({ remaining: 0, limit: 3 });
              logger.debug('[Stream] Quota info set to default (0, 3)');
            }
          } else {
            // 기본값 설정
            setQuotaInfo({ remaining: 0, limit: 3 });
            logger.debug('[Stream] Quota info set to default (0, 3)');
          }
        }
      } else if (error instanceof Error && error.message.includes('429') && !isAuthenticated) {
        // fallback: 메시지에 429가 포함된 경우
        setQuotaInfo({ remaining: 0, limit: 3 });
        logger.debug('[Stream] Quota info set to default (0, 3) from error message');
      }
      
      // 남아있는 토큰 버퍼 처리
      const remainingTokens = tokenBufferRef.current.get(assistantMessageId) || '';
      if (remainingTokens) {
        fullContent += remainingTokens;
        tokenBufferRef.current.delete(assistantMessageId);
      }
      
      // 기존 타이머 취소
      if (tokenBufferTimeoutRef.current.has(assistantMessageId)) {
        clearTimeout(tokenBufferTimeoutRef.current.get(assistantMessageId)!);
        tokenBufferTimeoutRef.current.delete(assistantMessageId);
      }
      
      // 에러 발생 시 현재 진행 상황만 초기화 (히스토리는 유지)
      setCurrentProgress(null);
      
      let errorMessage = error instanceof Error ? error.message : String(error);
      
      // Axios 에러의 경우 detail 필드 확인
      if (error && typeof error === 'object' && 'response' in error) {
        const axiosError = error as any;
        if (axiosError.response?.data?.detail) {
          errorMessage = axiosError.response.data.detail;
        }
      }
      
      // 로그인한 사용자에게는 익명 사용자 쿼터 메시지를 필터링
      if (isAuthenticated && (
        errorMessage.includes('무료 질의 3회를 모두 사용하셨습니다') ||
        errorMessage.includes('무료로 3회 체험')
      )) {
        errorMessage = '요청이 너무 많습니다. 잠시 후 다시 시도하세요.';
      }
      
      const streamError = classifyStreamError(
        new Error(errorMessage)
      );
      
      // 에러 발생 시 메시지 업데이트 (내용이 있거나 에러 메시지가 있는 경우에만)
      if (fullContent || error) {
        setMessages((prev) => {
          const messageIndex = prev.findIndex((msg) => msg.id === assistantMessageId);
          const errorContent = fullContent || streamError.message;
          
          if (messageIndex === -1) {
            return [...prev, { 
              ...assistantMessage, 
              content: errorContent,
              metadata: {
                ...assistantMessage.metadata,
                error: true,
                error_type: streamError.type,
              }
            }];
          }
          const updated = [...prev];
          updated[messageIndex] = { 
            ...updated[messageIndex], 
            content: errorContent,
            metadata: {
              ...updated[messageIndex].metadata,
              error: true,
              error_type: streamError.type,
            }
          };
          return updated;
        });
      }
      
      // 에러 상태 저장 (재시도용)
      setStreamErrors((prev) => {
        const newMap = new Map(prev);
        newMap.set(assistantMessageId, streamError);
        return newMap;
      });
      
      // Toast 표시 (로그인한 사용자에게는 익명 사용자 쿼터 메시지 표시 안 함)
      if (!isAuthenticated || !streamError.message.includes('무료 질의 3회를 모두 사용하셨습니다')) {
        const toastId = `toast-${Date.now()}`;
        setToasts((prev) => [...prev, {
          id: toastId,
          message: streamError.message,
          type: 'error',
          action: streamError.canRetry ? {
            label: '다시 시도',
            onClick: () => {
              // 재시도 로직
              setStreamErrors((prev) => {
                const newMap = new Map(prev);
                const error = newMap.get(assistantMessageId);
                if (error) {
                  error.retryCount = (error.retryCount || 0) + 1;
                  newMap.set(assistantMessageId, error);
                }
                return newMap;
              });
              // 메시지 재전송
              handleStreamingMessage(message, sessionId, attachments, undefined, undefined, undefined, true);
              setToasts((prev) => prev.filter(t => t.id !== toastId));
            }
          } : undefined,
        }]);
      }
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
      setMessages([userMessage]);
      
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
        setMessages([]);
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

  const handleDocumentClick = async (message: ChatMessage, documentIndex: number) => {
    setSelectedMessageForDocument(message);
    setSelectedDocumentIndex(documentIndex);
    setDocumentSidebarOpen(true);
    
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
        setMessages((prev) => {
          const messageIndex = prev.findIndex((msg) => msg.id === message.id);
          if (messageIndex !== -1) {
            const updated = [...prev];
            // eslint-disable-next-line security/detect-object-injection
            updated[messageIndex] = {
              // eslint-disable-next-line security/detect-object-injection
              ...updated[messageIndex],
              metadata: {
                // eslint-disable-next-line security/detect-object-injection
                ...updated[messageIndex].metadata,
                sources: sourcesData.sources,
                legal_references: sourcesData.legal_references,
                sources_detail: sourcesData.sources_detail,
              },
            };
            return updated;
          }
          return prev;
        });
        
        // selectedMessageForDocument도 업데이트 (setMessages 외부에서)
        setSelectedMessageForDocument((prev) => {
          if (prev?.id === message.id) {
            return {
              ...prev,
              metadata: {
                ...prev.metadata,
                sources: sourcesData.sources,
                legal_references: sourcesData.legal_references,
                sources_detail: sourcesData.sources_detail,
              },
            };
          }
          return prev;
        });
      } catch (error) {
        logger.error('[App] Error fetching sources for sidebar:', error);
      }
    }
  };

  const handleCloseDocumentSidebar = () => {
    setDocumentSidebarOpen(false);
    setSelectedDocumentIndex(null);
    setSelectedMessageForDocument(null);
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
  const _ = locationKey; // locationKey를 사용하여 리렌더링 보장
  
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
              onClose={() => setToasts((prev) => prev.filter(t => t.id !== toast.id))}
              action={toast.action}
            />
          </div>
        ))}
      </div>

      <div className={`flex-1 flex flex-col overflow-hidden min-h-0 transition-all duration-300 ease-in-out ${
        (documentSidebarOpen || referencesSidebarOpen) ? 'mr-[28rem]' : 'mr-0'
      }`}>
        {/* 콘텐츠 영역 - 스크롤 가능 */}
        <div className="flex-1 overflow-y-auto min-h-0">
          {messages.length === 0 && !isSending && !currentSession ? (
            <WelcomeScreen onQuestionClick={handleQuestionClick} isAuthenticated={isAuthenticated} quotaInfo={quotaInfo} onLoginClick={login} />
          ) : (
            <ChatHistory
              messages={messages}
              sessionId={currentSession?.session_id}
              isLoading={isSending || isStreaming}
              currentProgress={currentProgress}
              progressHistory={progressHistory}
              onQuestionClick={handleRelatedQuestionClick}
              streamingMessageId={streamingMessageId}
              streamErrors={streamErrors}
              onDocumentClick={handleDocumentClick}
              onOpenReferencesSidebar={(message, selectedType) => {
                setSelectedMessageForReferences(message);
                setSelectedReferenceType(selectedType);
                setReferencesSidebarOpen(true);
              }}
              onRetryMessage={(messageId) => {
                const error = streamErrors.get(messageId);
                if (error && error.canRetry && currentSession) {
                  // 재시도 로직
                  const message = messages.find(m => m.id === messageId);
                  if (message && message.role === 'assistant') {
                    // 이전 사용자 메시지 찾기
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
            />
          )}
        </div>
        {/* 쿼터 정보 표시 (익명 사용자만) */}
        {!isAuthenticated && quotaInfo && (
          <QuotaIndicator
            remaining={quotaInfo.remaining}
            limit={quotaInfo.limit}
            isAuthenticated={isAuthenticated}
            onLoginClick={login}
          />
        )}
        {/* 입력창 - 하단 고정 */}
        <div className="flex-shrink-0">
          <ChatInput 
            onSend={handleSendMessage} 
            disabled={isSending || isStreaming || (!isAuthenticated && quotaInfo?.remaining === 0)} 
            isLoading={isSending || isStreaming}
            resetTrigger={inputResetTrigger}
          />
        </div>
      </div>

      {/* 문서 사이드바 */}
      {selectedMessageForDocument && (
        <DocumentSidebar
          isOpen={documentSidebarOpen}
          onClose={handleCloseDocumentSidebar}
          documentIndex={selectedDocumentIndex}
          sources={selectedMessageForDocument.metadata?.sources}
          sourcesDetail={selectedMessageForDocument.metadata?.sources_detail}
          metadata={selectedMessageForDocument.metadata}
          sessionId={currentSession?.session_id}
          messageId={selectedMessageForDocument.id}
        />
      )}

      {/* 참고자료 사이드바 */}
      {selectedMessageForReferences && (
        <ReferencesSidebar
          isOpen={referencesSidebarOpen}
          onClose={() => {
            setReferencesSidebarOpen(false);
            setSelectedMessageForReferences(null);
            setSelectedReferenceType('all');
          }}
          references={selectedMessageForReferences.metadata?.sources}
          legalReferences={selectedMessageForReferences.metadata?.legal_references}
          sources={selectedMessageForReferences.metadata?.sources}
          sourcesDetail={selectedMessageForReferences.metadata?.sources_detail}
          initialSelectedType={selectedReferenceType}
        />
      )}
    </MainLayout>
  );
}

export default App;

