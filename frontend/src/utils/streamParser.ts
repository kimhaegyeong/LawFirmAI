/**
 * 스트림 데이터 파서
 * JSONL 형식과 기존 [진행상황] 형식을 모두 지원
 */

export interface ParsedChunk {
  type: 'progress' | 'stream' | 'final' | 'chunk' | 'quota' | 'sources' | 'validation' | 'validation_start' | 'regeneration_start' | 'done' | 'error';
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
    cancelled?: boolean;
    error_type?: string;
    chunk_index?: number;
    total_chunks?: number;
    has_more?: boolean;
    message_id?: string;
    remaining?: number;
    limit?: number;
    quality_score?: number;
    is_valid?: boolean;
    needs_regeneration?: boolean;
    regeneration_reason?: string;
    issues?: string[];
    strengths?: string[];
    [key: string]: unknown;
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
    if (parsed.type && ['progress', 'stream', 'final', 'chunk', 'quota', 'sources', 'validation', 'validation_start', 'regeneration_start', 'done', 'error'].includes(parsed.type)) {
      // error 이벤트 처리
      if (parsed.type === 'error') {
        return {
          type: 'error',
          content: parsed.content || parsed.message || '오류가 발생했습니다.',
          metadata: {
            error: true,
            cancelled: parsed.metadata?.cancelled || false,
            error_type: parsed.metadata?.error_type || 'unknown',
            ...parsed.metadata
          }
        };
      }
      
      // sources 이벤트의 경우 metadata를 그대로 보존
      if (parsed.type === 'sources') {
        return {
          type: parsed.type,
          content: parsed.content || parsed.message || '',
          metadata: parsed.metadata || {}
        };
      }
      
      return {
        type: parsed.type,
        content: parsed.content || parsed.message || '',
        metadata: {
          step: parsed.step,
          message: parsed.message,
          timestamp: parsed.timestamp,
          node_name: parsed.node_name,
          chunk_index: parsed.chunk_index,
          total_chunks: parsed.total_chunks,
          has_more: parsed.has_more,
          message_id: parsed.message_id,
          remaining: parsed.remaining,
          limit: parsed.limit,
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

