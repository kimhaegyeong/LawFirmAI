/**
 * 메인 App 컴포넌트
 */
import { useState, useEffect } from 'react';
import { MainLayout } from './components/layout/MainLayout';
import { SidebarContent } from './components/sidebar/SidebarContent';
import { WelcomeScreen } from './components/welcome/WelcomeScreen';
import { ChatHistory } from './components/chat/ChatHistory';
import { ChatInput } from './components/chat/ChatInput';
import { LoadingSpinner } from './components/common/LoadingSpinner';
import { useChat } from './hooks/useChat';
import { useSession } from './hooks/useSession';
import { getHistory, exportHistory, downloadHistory } from './services/historyService';
import { createSession } from './services/sessionService';
import type { ChatMessage, FileAttachment } from './types/chat';
import type { Session } from './types/session';

function App() {
  const [messages, setMessages] = useState<ChatMessage[]>([]);
  const [searchQuery, setSearchQuery] = useState('');

  const { currentSession, sessions, isLoading, loadSessions, newSession, loadSession, updateSession, removeSession, clearSession } = useSession();
  const { sendMessage, sendStreamingMessage, isLoading: isSending, isStreaming } = useChat({
    onMessage: (message) => {
      setMessages((prev) => [...prev, message]);
    },
    onError: (error) => {
      console.error('Chat error:', error);
      alert(`에러가 발생했습니다: ${error.message}`);
    },
  });

  // 초기 세션 목록 로드
  useEffect(() => {
    loadSessions();
  }, []);

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
          console.error('Failed to load session messages:', error);
          // 에러 발생 시에도 현재 메시지를 유지 (빈 배열로 덮어쓰지 않음)
        }
      } else {
        setMessages([]);
      }
    };
    
    loadSessionMessages();
  }, [currentSession]);

  const handleNewSession = async () => {
    console.log('[DEBUG] handleNewSession called');
    try {
      // 웰컴 화면을 표시하기 위해 세션을 초기화하고 메시지도 초기화
      clearSession();
      setMessages([]);
      console.log('[DEBUG] Session and messages cleared for welcome screen');
    } catch (error) {
      console.error('[ERROR] Failed to reset session:', error);
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
      console.error('Failed to load session:', error);
    }
  };

  const handleRenameSession = async (session: Session) => {
    const newTitle = prompt('새 제목을 입력하세요:', session.title || '');
    if (newTitle && newTitle !== session.title) {
      try {
        await updateSession(session.session_id, { title: newTitle });
      } catch (error) {
        console.error('Failed to rename session:', error);
      }
    }
  };

  const handleCopySession = async (session: Session) => {
    try {
      // 새 세션 생성
      const newSession = await createSession({
        title: `${session.title || '제목 없음'} (복사본)`,
        category: session.category,
      });
      
      // 기존 세션의 메시지 가져오기
      const history = await getHistory({ session_id: session.session_id });
      
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
      console.error('Failed to copy session:', error);
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
      console.error('Failed to export session:', error);
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
        console.error('Failed to delete session:', error);
      }
    }
  };

  const handleSendMessage = async (message: string, attachments?: FileAttachment[]) => {
    if (!currentSession) {
      // 세션이 없으면 새로 생성
      const session = await newSession();
      if (session) {
        await handleStreamingMessage(message, session.session_id, attachments);
      }
    } else {
      await handleStreamingMessage(message, currentSession.session_id, attachments);
    }
  };

  const handleStreamingMessage = async (
    message: string,
    sessionId: string,
    attachments?: FileAttachment[],
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
    
    // 첫 번째 chunk를 받았는지 추적
    let isFirstChunk = true;

    // 스트리밍 메시지 전송
    let fullContent = '';
    try {
      await sendStreamingMessage(
        message,
        sessionId,
        (chunk) => {
          if (import.meta.env.DEV) {
            console.log('[Stream] Chunk received:', JSON.stringify(chunk));
          }
          fullContent += chunk;
          if (import.meta.env.DEV) {
            console.log('[Stream] Full content so far length:', fullContent.length);
          }
          // 함수형 업데이트로 항상 최신 상태 참조
          setMessages((prev) => {
            const messageIndex = prev.findIndex((msg) => msg.id === assistantMessageId);
            
            // 첫 번째 chunk를 받았을 때만 메시지 추가
            if (isFirstChunk && messageIndex === -1) {
              isFirstChunk = false;
              if (import.meta.env.DEV) {
                console.log('[Stream] First chunk received, adding assistant message:', assistantMessageId);
              }
              // 첫 번째 chunk와 함께 메시지 추가
              return [...prev, { ...assistantMessage, content: fullContent }];
            }
            
            // 이미 메시지가 있으면 업데이트
            if (messageIndex !== -1) {
              const updated = [...prev];
              updated[messageIndex] = { ...updated[messageIndex], content: fullContent };
              if (import.meta.env.DEV) {
                console.log('[Stream] Message updated at index:', messageIndex, 'Content length:', fullContent.length);
              }
              return updated;
            }
            
            // 메시지가 없으면 추가 (fallback - 첫 번째 chunk가 아닌 경우)
            if (import.meta.env.DEV) {
              console.warn('[Stream] Message not found, adding as fallback:', assistantMessageId);
            }
            return [...prev, { ...assistantMessage, content: fullContent }];
          });
        }
      );
      if (import.meta.env.DEV) {
        console.log('[Stream] Streaming completed. Final content length:', fullContent.length);
      }
    } catch (error) {
      console.error('[Stream] Streaming error:', error);
      // 에러 발생 시 메시지 업데이트 (내용이 있거나 에러 메시지가 있는 경우에만)
      if (fullContent || error) {
        setMessages((prev) => {
          const messageIndex = prev.findIndex((msg) => msg.id === assistantMessageId);
          const errorContent = fullContent || '스트리밍 중 오류가 발생했습니다.';
          
          if (messageIndex === -1) {
            return [...prev, { ...assistantMessage, content: errorContent }];
          }
          const updated = [...prev];
          updated[messageIndex] = { ...updated[messageIndex], content: errorContent };
          return updated;
        });
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
          await handleStreamingMessage(question, session.session_id, undefined, true);
        }
      } catch (error) {
        console.error('[ERROR] Failed to create session:', error);
        // 에러 발생 시 메시지 제거하고 웰컴 화면으로 복귀
        setMessages([]);
        const errorMessage = error instanceof Error 
          ? error.message 
          : '세션 생성에 실패했습니다. 다시 시도해주세요.';
        alert(errorMessage);
      }
    } else {
      await handleStreamingMessage(question, currentSession.session_id);
    }
  };

  const handleRelatedQuestionClick = async (question: string) => {
    await handleQuestionClick(question);
  };

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
      <div className="flex-1 flex flex-col overflow-hidden min-h-0">
        {/* 콘텐츠 영역 - 스크롤 가능 */}
        <div className="flex-1 overflow-y-auto min-h-0">
          {messages.length === 0 && !isSending && !currentSession ? (
            <WelcomeScreen onQuestionClick={handleQuestionClick} />
          ) : (
            <ChatHistory
              messages={messages}
              sessionId={currentSession?.session_id}
              isLoading={isSending || isStreaming}
              onQuestionClick={handleRelatedQuestionClick}
            />
          )}
        </div>
        {/* 입력창 - 하단 고정 */}
        <div className="flex-shrink-0">
          <ChatInput onSend={handleSendMessage} disabled={isSending || isStreaming} isLoading={isSending || isStreaming} />
        </div>
      </div>
    </MainLayout>
  );
}

export default App;

