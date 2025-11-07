/**
 * 채팅 서비스
 */
import { api, extractApiError } from './api';
import type { ChatRequest, ChatResponse, StreamingChatRequest } from '../types/chat';

/**
 * 일반 채팅 메시지 전송
 */
export async function sendChatMessage(request: ChatRequest): Promise<ChatResponse> {
  try {
    const response = await api.post<ChatResponse>('/chat', request);
    return response.data;
  } catch (error) {
    throw extractApiError(error);
  }
}

/**
 * 스트리밍 채팅 메시지 전송
 */
export async function* sendStreamingChatMessage(
  request: StreamingChatRequest
): AsyncGenerator<string, void, unknown> {
  try {
    // 개발 환경에서는 상대 경로 사용 (Vite 프록시 활용)
    // 프로덕션 환경에서는 절대 URL 사용
    const streamUrl = api.defaults.baseURL 
      ? `${api.defaults.baseURL}/chat/stream`
      : '/api/v1/chat/stream';
    
    const response = await fetch(streamUrl, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
      },
      body: JSON.stringify(request),
    });

    if (!response.ok) {
      throw new Error(`HTTP error! status: ${response.status}`);
    }

    const reader = response.body?.getReader();
    if (!reader) {
      throw new Error('Response body is not readable');
    }

    const decoder = new TextDecoder('utf-8');
    let buffer = '';
    let currentEvent: string[] = []; // 현재 SSE 이벤트의 data 라인들
    let streamCompleted = false;
    let hasReceivedData = false;

    try {
      while (true) {
        let readResult;
        try {
          readResult = await reader.read();
        } catch (readError) {
          // 읽기 오류 발생 시 처리
          console.error('[SSE] Error reading from stream:', readError);
          // 이미 받은 데이터가 있으면 처리
          if (hasReceivedData && currentEvent.length > 0) {
            const content = currentEvent.join('\n');
            if (content !== '[스트리밍 완료]' && content !== '[완료]') {
              yield content;
            }
          }
          break;
        }
        
        const { done, value } = readResult;
        
        if (done) {
          // 스트림 종료 시 남은 이벤트 처리
          if (currentEvent.length > 0) {
            const content = currentEvent.join('\n');
            if (content !== '[스트리밍 완료]' && content !== '[완료]') {
              if (import.meta.env.DEV) {
                console.log('[SSE] Final event:', content);
              }
              yield content;
            }
          }
          // 버퍼에 남은 데이터 처리
          if (buffer.trim()) {
            const lines = buffer.split('\n');
            for (const line of lines) {
              if (line.startsWith('data: ')) {
                const content = line.slice(6);
                if (content !== '[스트리밍 완료]' && content !== '[완료]') {
                  if (import.meta.env.DEV) {
                    console.log('[SSE] Final buffer content:', content);
                  }
                  yield content;
                }
              }
            }
          }
          break;
        }

        hasReceivedData = true;
        const chunk = decoder.decode(value, { stream: true });
        buffer += chunk;
        
        // 디버깅: 원시 데이터 확인
        if (import.meta.env.DEV) {
          console.log('[SSE] Raw chunk received:', JSON.stringify(chunk));
        }

        // 버퍼에서 완전한 라인들만 처리
        const lines = buffer.split('\n');
        buffer = lines.pop() || ''; // 마지막 불완전한 라인은 버퍼에 보관

        for (const line of lines) {
          if (line.startsWith('data: ')) {
            // SSE data 라인: "data: {content}" 형식
            const content = line.slice(6); // "data: " 제거
            
            // 완료 신호 확인
            if (content === '[스트리밍 완료]' || content === '[완료]') {
              streamCompleted = true;
              if (import.meta.env.DEV) {
                console.log('[SSE] Stream completion signal received');
              }
              // 완료 신호는 yield하지 않음
              continue;
            }
            
            currentEvent.push(content);
            if (import.meta.env.DEV) {
              console.log('[SSE] Data line received:', JSON.stringify(content));
            }
          } else if (line.trim() === '') {
            // 빈 줄: SSE 이벤트 종료 신호
            if (currentEvent.length > 0) {
              // 연속된 data 라인들을 줄바꿈으로 결합
              const content = currentEvent.join('\n');
              // 완료 신호 및 진행 상황 메시지는 제외
              if (
                content !== '[스트리밍 완료]' && 
                content !== '[완료]' &&
                !content.includes('[답변 생성 중...]')
              ) {
                if (import.meta.env.DEV) {
                  console.log('[SSE] Event complete, yielding:', JSON.stringify(content));
                }
                yield content;
              }
              currentEvent = []; // 이벤트 초기화
            }
          } else if (line.trim() !== '') {
            // 다른 라인 (주석, 이벤트 타입 등) - 디버깅용
            if (import.meta.env.DEV) {
              console.log('[SSE] Other line:', JSON.stringify(line));
            }
          }
        }
      }
    } catch (error) {
      // 네트워크 오류 처리
      if (error instanceof TypeError && (error.message.includes('network') || error.message.includes('ERR_INCOMPLETE'))) {
        console.error('[SSE] Network error during streaming:', error);
        // 이미 받은 데이터가 있으면 처리
        if (hasReceivedData && currentEvent.length > 0) {
          const content = currentEvent.join('\n');
          if (content !== '[스트리밍 완료]' && content !== '[완료]') {
            yield content;
          }
        }
        // 스트림이 완료되지 않았을 수 있으므로 경고만 출력
        if (!streamCompleted) {
          console.warn('[SSE] Stream may have been interrupted, but partial data was received');
        }
        // 네트워크 오류는 정상적으로 처리되었으므로 예외를 다시 던지지 않음
        return;
      } else {
        throw error;
      }
    }
  } catch (error) {
    console.error('[SSE] Streaming error:', error);
    // 더 자세한 에러 정보 제공
    if (error instanceof Error) {
      if (error.message.includes('ERR_INCOMPLETE_CHUNKED_ENCODING') || 
          error.message.includes('incomplete')) {
        throw new Error('스트리밍 연결이 완료되지 않았습니다. 서버에서 스트리밍이 제대로 종료되지 않았을 수 있습니다.');
      }
      throw error;
    }
    throw extractApiError(error);
  }
}

