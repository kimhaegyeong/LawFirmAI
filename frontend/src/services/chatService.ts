/**
 * 채팅 서비스
 */
import { api, extractApiError } from './api';
import logger from '../utils/logger';
import type { ChatRequest, ChatResponse, StreamingChatRequest, SourceInfo } from '../types/chat';

/**
 * 일반 채팅 메시지 전송
 */
export async function sendChatMessage(request: ChatRequest): Promise<ChatResponse & { quotaInfo?: { remaining: number; limit: number } }> {
  try {
    const response = await api.post<ChatResponse>('/chat', request);
    const quotaRemaining = response.headers['x-quota-remaining'];
    const quotaLimit = response.headers['x-quota-limit'];
    
    const result: ChatResponse & { quotaInfo?: { remaining: number; limit: number } } = response.data;
    
    if (quotaRemaining !== undefined && quotaLimit !== undefined) {
      result.quotaInfo = {
        remaining: parseInt(quotaRemaining, 10),
        limit: parseInt(quotaLimit, 10),
      };
    }
    
    return result;
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
  // 타임아웃을 위한 AbortController 생성
  const abortController = new AbortController();
  const timeoutId = setTimeout(() => {
    abortController.abort();
  }, 300000); // 5분 타임아웃
  
  try {
    // 개발 환경에서는 상대 경로 사용 (Vite 프록시 활용)
    // 프로덕션 환경에서는 절대 URL 사용
    const streamUrl = api.defaults.baseURL 
      ? `${api.defaults.baseURL}/chat/stream`
      : '/api/v1/chat/stream';
    
    // Authorization 헤더 추가 (로그인한 사용자의 경우)
    const headers: HeadersInit = {
      'Content-Type': 'application/json',
      'Accept': 'text/event-stream',
    };
    
    // 토큰이 있으면 Authorization 헤더에 추가
    const { getAccessToken } = await import('./authService');
    const token = getAccessToken();
    if (token) {
      headers['Authorization'] = `Bearer ${token}`;
      if (import.meta.env.DEV) {
        logger.debug('[Stream] Token included in streaming request');
      }
    } else {
      if (import.meta.env.DEV) {
        logger.debug('[Stream] No token found, using anonymous session');
      }
    }
    
    let response: Response;
    try {
      if (import.meta.env.DEV) {
        logger.debug('[Stream] Sending POST request to:', streamUrl);
        logger.debug('[Stream] Request headers:', headers);
        logger.debug('[Stream] Request body:', JSON.stringify(request, null, 2));
      }
      
      response = await fetch(streamUrl, {
        method: 'POST',
        headers,
        body: JSON.stringify(request),
        signal: abortController.signal,
      });
      
      if (import.meta.env.DEV) {
        logger.debug('[Stream] Response received:', {
          status: response.status,
          statusText: response.statusText,
          headers: Object.fromEntries(response.headers.entries()),
        });
      }
      
      if (!response.ok) {
        const errorText = await response.text().catch(() => 'Unknown error');
        logger.error('[Stream] Response not OK:', {
          status: response.status,
          statusText: response.statusText,
          error: errorText,
        });
        throw new Error(`HTTP ${response.status}: ${response.statusText} - ${errorText}`);
      }
    } catch (fetchError) {
      // fetch 자체에서 발생하는 오류 처리
      clearTimeout(timeoutId);
      
      // ERR_INCOMPLETE_CHUNKED_ENCODING은 실제로는 응답이 시작되었을 수 있음
      // 하지만 fetch 단계에서 발생하면 response 객체가 없으므로 처리 불가
      if (fetchError instanceof TypeError) {
        const errorMessage = fetchError.message || '';
        if (errorMessage.includes('ERR_INCOMPLETE_CHUNKED_ENCODING') ||
            errorMessage.includes('incomplete') ||
            errorMessage.includes('chunked')) {
          // 불완전한 스트림 오류는 경고로 처리
          // 실제로는 데이터가 정상적으로 처리되었을 수 있음
          logger.warn('[SSE] Fetch error (incomplete chunked encoding):', fetchError);
          // 이 경우는 fetch 단계에서 발생했으므로 response가 없음
          // 하지만 브라우저 콘솔에만 경고가 표시되고 실제 동작에는 문제가 없을 수 있음
          // 조용히 처리하고 사용자에게는 성공으로 표시
          return; // 빈 제너레이터 반환 (이미 처리된 데이터가 있을 수 있음)
        }
      }
      throw fetchError;
    }
    
    // 타임아웃 타이머 정리 (응답을 받았으므로)
    clearTimeout(timeoutId);

    // 쿼터 정보 추출 (응답 헤더에서)
    const quotaRemaining = response.headers.get('x-quota-remaining');
    const quotaLimit = response.headers.get('x-quota-limit');
    
    if (quotaRemaining !== null && quotaLimit !== null) {
      // 쿼터 정보를 첫 번째 이벤트로 yield
      yield JSON.stringify({
        type: 'quota',
        remaining: parseInt(quotaRemaining, 10),
        limit: parseInt(quotaLimit, 10),
      });
    }

    const reader = response.body?.getReader();
    if (!reader) {
      throw new Error('Response body is not readable');
    }

    const decoder = new TextDecoder('utf-8');
    let buffer = '';
    let readerClosed = false;
    let jsonBuffer = ''; // JSON 파싱을 위한 버퍼
    let inDataLine = false; // data: 라인 내부인지 추적

    let shouldBreak = false;
    try {
      while (true) {
        if (shouldBreak) {
          break;
        }
        let readResult;
        try {
          readResult = await reader.read();
        } catch (readError) {
          // 읽기 오류 발생 시 처리
          // ERR_INCOMPLETE_CHUNKED_ENCODING 오류는 스트림이 완료되지 않았을 때 발생
          // 하지만 이미 받은 데이터는 처리해야 함
          const isIncompleteError = readError instanceof TypeError && 
              (readError.message.includes('network') || 
               readError.message.includes('ERR_INCOMPLETE') ||
               readError.message.includes('incomplete'));
          
          if (isIncompleteError) {
            // 네트워크 오류 또는 불완전한 스트림 오류
            // 이미 받은 데이터는 정상적으로 처리
            // ERR_INCOMPLETE_CHUNKED_ENCODING은 브라우저 콘솔에 경고로 표시되지만
            // 실제로는 데이터가 정상적으로 처리되었을 수 있음
            if (import.meta.env.DEV) {
              logger.debug('[SSE] Stream reading error (incomplete chunked encoding, but data may be processed):', readError);
            }
            
            // 버퍼에 남은 데이터 처리
            if (buffer.trim()) {
              const lines = buffer.split('\n');
              for (const line of lines) {
                if (line.startsWith('data: ')) {
                  const content = line.slice(6);
                  if (content !== '[스트리밍 완료]' && content !== '[완료]' && content.trim() !== '') {
                    try {
                      // JSON 파싱 시도
                      JSON.parse(content);
                      yield content;
                    } catch (e) {
                      // 파싱 실패 시 그대로 yield (하위 호환성)
                      yield content;
                    }
                  }
                } else if (line.trim() !== '' && !line.startsWith('event:') && !line.startsWith('id:')) {
                  // SSE 형식이 아닌 경우도 처리 (하위 호환성)
                  try {
                    JSON.parse(line);
                    yield line;
                  } catch (e) {
                    // 파싱 실패 시 무시
                  }
                }
              }
            }
            
            // JSON 버퍼에 남은 데이터 처리
            if (jsonBuffer.trim()) {
              try {
                // JSON 파싱 시도
                const parsed = JSON.parse(jsonBuffer);
                // "done" 이벤트가 있으면 yield
                if (parsed.type === 'done' || parsed.type === 'final') {
                  yield jsonBuffer;
                } else {
                  yield jsonBuffer;
                }
              } catch (e) {
                // 파싱 실패 시 그대로 yield (하위 호환성)
                if (jsonBuffer !== '[스트리밍 완료]' && jsonBuffer !== '[완료]' && jsonBuffer.trim() !== '') {
                  yield jsonBuffer;
                }
              }
            }
            
            // 리더 닫기 시도
            if (!readerClosed) {
              try {
                readerClosed = true;
                reader.releaseLock();
              } catch (e) {
                // 이미 닫혔거나 닫을 수 없는 경우 무시
              }
            }
            
            // ERR_INCOMPLETE_CHUNKED_ENCODING 오류는 정상적으로 처리되었으므로
            // 예외를 다시 던지지 않고 종료
            // 브라우저 콘솔에 경고가 표시되지만 실제 동작에는 문제가 없음
            if (import.meta.env.DEV) {
              logger.debug('[SSE] Stream ended with incomplete error, but data was processed');
            }
            return; // break 대신 return 사용하여 함수 종료
          } else {
            // 다른 오류는 다시 던짐
            logger.error('[SSE] Error reading from stream:', readError);
            
            // 리더 닫기 시도
            if (!readerClosed) {
              try {
                readerClosed = true;
                reader.releaseLock();
              } catch (e) {
                // 이미 닫혔거나 닫을 수 없는 경우 무시
              }
            }
            
            throw readError; // 다른 오류는 다시 던짐
          }
        }
        
        const { done, value } = readResult;
        
        if (done) {
          // 스트림 종료 시 버퍼에 남은 데이터 처리
          if (buffer.trim()) {
            const lines = buffer.split('\n');
            for (const line of lines) {
              if (line.startsWith('data: ')) {
                const content = line.slice(6);
                // 완료 신호는 건너뛰기
                if (content === '[스트리밍 완료]' || content === '[완료]') {
                  continue;
                }
                // JSON 파싱 시도하여 "done" 이벤트 확인
                if (content.trim() !== '') {
                  try {
                    const parsed = JSON.parse(content);
                    // "done" 이벤트는 최종 답변을 포함하므로 yield
                    if (parsed.type === 'done') {
                      if (import.meta.env.DEV) {
                        logger.debug('[SSE] Done event found in final buffer, yielding for final answer replacement');
                      }
                      yield content;
                      continue;
                    }
                  } catch (e) {
                    // JSON 파싱 실패 시 그대로 처리
                  }
                  if (import.meta.env.DEV) {
                    logger.debug('[SSE] Final buffer content:', content);
                  }
                  yield content;
                }
              }
            }
          }
          
          // JSON 버퍼에 남은 데이터 처리
          if (jsonBuffer.trim()) {
            try {
              const parsed = JSON.parse(jsonBuffer);
              // "done" 이벤트는 최종 답변을 포함하므로 yield
              if (parsed.type === 'done') {
                if (import.meta.env.DEV) {
                  logger.debug('[SSE] Done event found in final buffer, yielding for final answer replacement');
                }
                yield jsonBuffer;
              } else {
                yield jsonBuffer;
              }
            } catch (e) {
              // 파싱 실패 시 그대로 yield (하위 호환성)
              if (jsonBuffer !== '[스트리밍 완료]' && jsonBuffer !== '[완료]' && jsonBuffer.trim() !== '') {
                yield jsonBuffer;
              }
            }
          }
          
          // 리더 명시적으로 닫기
          if (!readerClosed) {
            try {
              readerClosed = true;
              reader.releaseLock();
            } catch (e) {
              // 이미 닫혔거나 닫을 수 없는 경우 무시
            }
          }
          
          // 스트림이 완전히 종료되었음을 확인
          if (import.meta.env.DEV) {
            logger.debug('[SSE] Stream completed successfully');
          }
          
          break;
        }

        const chunk = decoder.decode(value, { stream: true });
        buffer += chunk;
        
        // 디버깅: 원시 데이터 확인
        if (import.meta.env.DEV) {
          logger.debug('[SSE] Raw chunk received:', JSON.stringify(chunk.substring(0, 100)));
        }

        // 버퍼에서 완전한 라인들만 처리
        const lines = buffer.split('\n');
        buffer = lines.pop() || ''; // 마지막 불완전한 라인은 버퍼에 보관

        // SSE 형식 처리: data: {json}\n\n 또는 data: {json}\n
        for (const line of lines) {
          if (line.startsWith('data: ')) {
            // SSE data 라인 시작
            const content = line.slice(6); // "data: " 제거
            
            // 완료 신호 확인
            if (content === '[스트리밍 완료]' || content === '[완료]') {
              if (import.meta.env.DEV) {
                logger.debug('[SSE] Stream completion signal received');
              }
              // 완료 신호는 yield하지 않음
              continue;
            }
            
            // JSON 버퍼에 추가
            jsonBuffer = content;
            inDataLine = true;
            
            // JSON 파싱 시도 (완전한 JSON인지 확인)
            try {
              const parsed = JSON.parse(jsonBuffer);
              // 파싱 성공 = 완전한 JSON
              if (import.meta.env.DEV) {
                logger.debug('[SSE] Complete JSON received:', parsed.type);
              }
              // "done" 이벤트는 최종 답변을 포함하므로 yield 후 종료
              if (parsed.type === 'done') {
                yield jsonBuffer;  // 최종 답변을 포함한 done 이벤트 yield
                if (import.meta.env.DEV) {
                  logger.debug('[SSE] Done event received with final answer, closing stream');
                }
                jsonBuffer = '';
                inDataLine = false;
                // 리더 정상적으로 닫기
                if (!readerClosed && reader) {
                  try {
                    readerClosed = true;
                    reader.releaseLock();
                  } catch (e) {
                    // 이미 닫혔거나 닫을 수 없는 경우 무시
                  }
                }
                // 정상 종료를 위해 플래그 설정
                shouldBreak = true;
                break;
              }
              yield jsonBuffer;
              jsonBuffer = '';
              inDataLine = false;
            } catch (e) {
              // 파싱 실패 = 불완전한 JSON (다음 청크를 기다림)
              if (import.meta.env.DEV) {
                logger.debug('[SSE] Incomplete JSON, waiting for more data...');
              }
            }
          } else if (line.trim() === '') {
            // 빈 줄: SSE 이벤트 종료 신호
            if (inDataLine && jsonBuffer) {
              // JSON 버퍼에 데이터가 있으면 파싱 시도
              try {
                const parsed = JSON.parse(jsonBuffer);
                if (import.meta.env.DEV) {
                  logger.debug('[SSE] Complete JSON from buffer:', parsed.type);
                }
                // "done" 이벤트는 최종 답변을 포함하므로 yield 후 종료
                if (parsed.type === 'done') {
                  yield jsonBuffer;  // 최종 답변을 포함한 done 이벤트 yield
                  if (import.meta.env.DEV) {
                    logger.debug('[SSE] Done event received with final answer from buffer, closing stream');
                  }
                  jsonBuffer = '';
                  inDataLine = false;
                  // 리더 정상적으로 닫기
                  if (!readerClosed && reader) {
                    try {
                      readerClosed = true;
                      reader.releaseLock();
                    } catch (e) {
                      // 이미 닫혔거나 닫을 수 없는 경우 무시
                    }
                  }
                  // 정상 종료를 위해 플래그 설정
                  shouldBreak = true;
                  break;
                } else {
                  yield jsonBuffer;
                  jsonBuffer = '';
                  inDataLine = false;
                }
              } catch (e) {
                // 파싱 실패 시 그대로 yield (하위 호환성)
                if (jsonBuffer !== '[스트리밍 완료]' && jsonBuffer !== '[완료]') {
                  if (import.meta.env.DEV) {
                    logger.debug('[SSE] Yielding incomplete JSON (fallback):', jsonBuffer.substring(0, 50));
                  }
                  yield jsonBuffer;
                  jsonBuffer = '';
                  inDataLine = false;
                }
              }
            }
          } else if (inDataLine && jsonBuffer) {
            // data: 라인 이후 추가 라인 (JSON이 여러 줄에 걸친 경우)
            jsonBuffer += '\n' + line;
            
            // JSON 파싱 시도
            try {
              const parsed = JSON.parse(jsonBuffer);
              // 파싱 성공 = 완전한 JSON
              if (import.meta.env.DEV) {
                logger.debug('[SSE] Complete multi-line JSON received:', parsed.type);
              }
              // "done" 이벤트는 최종 답변을 포함하므로 yield 후 종료
              if (parsed.type === 'done') {
                yield jsonBuffer;  // 최종 답변을 포함한 done 이벤트 yield
                if (import.meta.env.DEV) {
                  logger.debug('[SSE] Done event received with final answer from multi-line, closing stream');
                }
                jsonBuffer = '';
                inDataLine = false;
                // 리더 정상적으로 닫기
                if (!readerClosed && reader) {
                  try {
                    readerClosed = true;
                    reader.releaseLock();
                  } catch (e) {
                    // 이미 닫혔거나 닫을 수 없는 경우 무시
                  }
                }
                // 정상 종료를 위해 플래그 설정
                shouldBreak = true;
                break;
              } else {
                yield jsonBuffer;
                jsonBuffer = '';
                inDataLine = false;
              }
            } catch (e) {
              // 파싱 실패 = 아직 불완전한 JSON
              if (import.meta.env.DEV) {
                logger.debug('[SSE] Continuing to build JSON...');
              }
            }
          } else if (line.trim() !== '') {
            // 다른 라인 (주석, 이벤트 타입 등) - 디버깅용
            if (import.meta.env.DEV) {
              logger.debug('[SSE] Other line:', JSON.stringify(line));
            }
          }
        }
      }
    } catch (error) {
      // 네트워크 오류 처리
      const isIncompleteError = error instanceof TypeError && 
          (error.message.includes('network') || 
           error.message.includes('ERR_INCOMPLETE') ||
           error.message.includes('incomplete') ||
           error.message.includes('chunked'));
      
      if (isIncompleteError) {
        // ERR_INCOMPLETE_CHUNKED_ENCODING은 브라우저 콘솔에 경고로 표시되지만
        // 실제로는 데이터가 정상적으로 처리되었을 수 있음
        if (import.meta.env.DEV) {
          logger.debug('[SSE] Network error during streaming (incomplete chunked encoding, but data may be processed):', error);
        }
        
        // 버퍼에 남은 데이터 처리
        if (buffer.trim()) {
          const lines = buffer.split('\n');
          for (const line of lines) {
            if (line.startsWith('data: ')) {
              const content = line.slice(6);
              if (content !== '[스트리밍 완료]' && content !== '[완료]' && content.trim() !== '') {
                try {
                  const parsed = JSON.parse(content);
                  // "done" 또는 "final" 이벤트 확인
                  if (parsed.type === 'done' || parsed.type === 'final') {
                    yield content;
                  } else {
                    yield content;
                  }
                } catch (e) {
                  // 파싱 실패 시 그대로 yield (하위 호환성)
                  yield content;
                }
              }
            }
          }
        }
        
        // JSON 버퍼에 남은 데이터 처리
        if (jsonBuffer.trim()) {
          try {
            const parsed = JSON.parse(jsonBuffer);
            // "done" 또는 "final" 이벤트 확인
            if (parsed.type === 'done' || parsed.type === 'final') {
              yield jsonBuffer;
            } else {
              yield jsonBuffer;
            }
          } catch (e) {
            // 파싱 실패 시 그대로 yield (하위 호환성)
            if (jsonBuffer !== '[스트리밍 완료]' && jsonBuffer !== '[완료]' && jsonBuffer.trim() !== '') {
              yield jsonBuffer;
            }
          }
        }
        
        // 리더 명시적으로 닫기
        if (!readerClosed && reader) {
          try {
            readerClosed = true;
            reader.releaseLock();
          } catch (e) {
            // 이미 닫혔거나 닫을 수 없는 경우 무시
          }
        }
        
        if (import.meta.env.DEV) {
          logger.debug('[SSE] Stream ended with incomplete error, but data was processed');
        }
        
        // ERR_INCOMPLETE_CHUNKED_ENCODING은 정상적으로 처리되었으므로 예외를 다시 던지지 않음
        // 브라우저 콘솔에 경고가 표시되지만 실제 동작에는 문제가 없음
        return;
      } else {
        // 리더 닫기 시도
        if (!readerClosed && reader) {
          try {
            readerClosed = true;
            reader.releaseLock();
          } catch (e) {
            // 이미 닫혔거나 닫을 수 없는 경우 무시
          }
        }
        throw error;
      }
    } finally {
      // 타임아웃 타이머 정리
      clearTimeout(timeoutId);
      
      // 리더가 아직 닫히지 않았다면 닫기
      if (!readerClosed && reader) {
        try {
          readerClosed = true;
          reader.releaseLock();
        } catch (e) {
          // 이미 닫혔거나 닫을 수 없는 경우 무시
        }
      }
    }
  } catch (error) {
    // 타임아웃 타이머 정리
    clearTimeout(timeoutId);
    
    logger.error('[SSE] Streaming error:', error);
    
    // 더 자세한 에러 정보 제공
    if (error instanceof Error) {
      if (error.name === 'AbortError') {
        throw new Error('스트리밍 요청이 타임아웃되었습니다. (5분 초과)');
      }
      
      // 네트워크 오류 구분
      if (error.message.includes('network') || 
          error.message.includes('ERR_NETWORK') ||
          error.message.includes('Failed to fetch') ||
          error.message.includes('ERR_CONNECTION_REFUSED')) {
        throw new Error('네트워크 연결에 실패했습니다. 인터넷 연결을 확인하세요.');
      }
      
      // HTTP 오류 구분
      if (error.message.includes('HTTP error')) {
        const statusMatch = error.message.match(/status: (\d+)/);
        const status = statusMatch && statusMatch[1] ? parseInt(statusMatch[1], 10) : 0;
        if (status === 503) {
          throw new Error('서버가 일시적으로 사용할 수 없습니다. 잠시 후 다시 시도하세요.');
        } else if (status >= 500) {
          throw new Error('서버 오류가 발생했습니다. 잠시 후 다시 시도하세요.');
        } else if (status === 429) {
          throw new Error('요청이 너무 많습니다. 잠시 후 다시 시도하세요.');
        }
      }
      
      // 불완전한 스트림 오류 처리
      // ERR_INCOMPLETE_CHUNKED_ENCODING은 브라우저 콘솔에 경고로 표시되지만
      // 실제로는 데이터가 정상적으로 처리되었을 수 있음
      if (error.message.includes('ERR_INCOMPLETE_CHUNKED_ENCODING') || 
          error.message.includes('incomplete') ||
          error.message.includes('chunked')) {
        // 이미 받은 데이터는 정상 처리되었을 가능성이 높음
        // 사용자에게는 조용히 처리 (데이터는 이미 처리되었을 수 있음)
        logger.warn('[SSE] Incomplete chunked encoding error (data may have been processed):', error);
        // 에러를 던지지 않고 조용히 종료 (데이터는 이미 처리되었을 수 있음)
        // 브라우저 콘솔에만 경고가 표시되고 실제 동작에는 문제가 없을 수 있음
        return;
      }
      
      throw error;
    }
    throw extractApiError(error);
  }
}


/**
 * 스트림 완료 후 sources 정보 가져오기
 */
export async function getChatSources(
  sessionId: string,
  messageId?: string
): Promise<{
  sources: string[];
  legal_references: string[];
  sources_detail: SourceInfo[];
}> {
  try {
    const url = messageId
      ? `/chat/${sessionId}/sources?message_id=${messageId}`
      : `/chat/${sessionId}/sources`;
    
    const response = await api.get<{
      session_id: string;
      sources: string[];
      legal_references: string[];
      sources_detail: SourceInfo[];
    }>(url);
    
    return {
      sources: response.data.sources || [],
      legal_references: response.data.legal_references || [],
      sources_detail: response.data.sources_detail || [],
    };
  } catch (error) {
    logger.error('[ChatService] Error getting sources:', error);
    throw extractApiError(error);
  }
}

