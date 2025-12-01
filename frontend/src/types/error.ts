/**
 * 에러 관련 타입 정의
 */

export enum ErrorType {
  TIMEOUT = 'timeout',
  NETWORK = 'network',
  SERVER = 'server',
  ABORTED = 'aborted',
  UNKNOWN = 'unknown',
}

export interface StreamError {
  type: ErrorType;
  message: string;
  originalError?: Error;
  canRetry?: boolean;
  retryCount?: number;
}

/**
 * 스트림 에러를 분류합니다.
 */
export function classifyStreamError(error: Error): StreamError {
  if (error.name === 'AbortError' || error.message.includes('타임아웃')) {
    return {
      type: ErrorType.TIMEOUT,
      message: '스트리밍 요청이 타임아웃되었습니다. (5분 초과)',
      originalError: error,
      canRetry: true,
    };
  }
  
  if (error.message.includes('network') || 
      error.message.includes('ERR_NETWORK') ||
      error.message.includes('Failed to fetch') ||
      error.message.includes('ERR_CONNECTION_REFUSED')) {
    return {
      type: ErrorType.NETWORK,
      message: '네트워크 연결에 실패했습니다. 인터넷 연결을 확인하세요.',
      originalError: error,
      canRetry: true,
    };
  }
  
  if (error.message.includes('HTTP error') || 
      error.message.includes('status:')) {
    const statusMatch = error.message.match(/status: (\d+)/);
    const status = statusMatch && statusMatch[1] ? parseInt(statusMatch[1], 10) : 0;
    
    if (status === 503) {
      return {
        type: ErrorType.SERVER,
        message: '서버가 일시적으로 사용할 수 없습니다. 잠시 후 다시 시도하세요.',
        originalError: error,
        canRetry: true,
      };
    } else if (status === 429) {
      return {
        type: ErrorType.SERVER,
        message: '요청이 너무 많습니다. 잠시 후 다시 시도하세요.',
        originalError: error,
        canRetry: true,
      };
    } else if (status >= 500) {
      return {
        type: ErrorType.SERVER,
        message: '서버 오류가 발생했습니다. 잠시 후 다시 시도하세요.',
        originalError: error,
        canRetry: true,
      };
    } else if (status >= 400) {
      return {
        type: ErrorType.SERVER,
        message: '요청 처리 중 오류가 발생했습니다.',
        originalError: error,
        canRetry: false,
      };
    }
    
    return {
      type: ErrorType.SERVER,
      message: '서버에서 오류가 발생했습니다.',
      originalError: error,
      canRetry: true,
    };
  }
  
  if (error.message.includes('ERR_INCOMPLETE_CHUNKED_ENCODING') || 
      error.message.includes('incomplete')) {
    return {
      type: ErrorType.NETWORK,
      message: '스트리밍 연결이 완료되지 않았습니다. 잠시 후 다시 시도해주세요.',
      originalError: error,
      canRetry: true,
    };
  }
  
  return {
    type: ErrorType.UNKNOWN,
    message: error.message || '알 수 없는 오류가 발생했습니다.',
    originalError: error,
    canRetry: true,
  };
}

