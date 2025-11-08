/**
 * 스트림 데이터 파서
 * JSONL 형식과 기존 [진행상황] 형식을 모두 지원
 */

export interface ParsedChunk {
  type: 'progress' | 'stream' | 'final';
  content: string;
  metadata?: {
    step?: number;
    message?: string;
    timestamp?: string;
    node_name?: string;
    tokens_received?: number;
    length?: number;
    answer_found?: boolean;
    error?: boolean;
    error_type?: string;
    [key: string]: any;
  };
}

/**
 * 스트림 청크를 파싱하여 진행 상황, 스트림, 최종 결과를 분리
 * 
 * @param chunk - 스트림에서 받은 원시 청크
 * @returns 파싱된 청크 정보 (타입, 내용, 메타데이터)
 */
export function parseStreamChunk(chunk: string): ParsedChunk {
  const trimmed = chunk.trim();
  
  // JSONL 형식 파싱 시도
  try {
    const parsed = JSON.parse(trimmed);
    if (parsed.type && ['progress', 'stream', 'final'].includes(parsed.type)) {
      return {
        type: parsed.type,
        content: parsed.content || parsed.message || '',
        metadata: {
          step: parsed.step,
          message: parsed.message,
          timestamp: parsed.timestamp,
          node_name: parsed.node_name,
          ...parsed.metadata
        }
      };
    }
  } catch (e) {
    // JSON 파싱 실패 시 기존 형식으로 처리 (하위 호환성)
  }
  
  // 기존 [진행상황] 형식 지원 (하위 호환성)
  if (trimmed.startsWith('[진행상황]')) {
    const message = trimmed.replace(/^\[진행상황\]/, '').trim();
    return {
      type: 'progress',
      content: message
    };
  }
  
  // 기본 답변 내용 (기존 형식)
  return {
    type: 'stream',
    content: trimmed
  };
}

