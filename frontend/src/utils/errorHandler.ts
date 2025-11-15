/**
 * 에러 처리 유틸리티
 * 에러 분류 및 처리 로직 통합
 */
import type { AxiosError } from 'axios';
import { classifyStreamError, type StreamError } from '../types/error';

/**
 * 에러 메시지 추출 (Axios 에러 지원)
 */
export function extractErrorMessage(error: unknown, isAuthenticated: boolean = false): string {
  let errorMessage = error instanceof Error ? error.message : String(error);
  
  // Axios 에러의 경우 detail 필드 확인
  if (error && typeof error === 'object' && 'response' in error) {
    const axiosError = error as AxiosError<{ detail?: string }>;
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
  
  return errorMessage;
}

/**
 * 에러를 StreamError로 변환
 */
export function toStreamError(error: unknown, isAuthenticated: boolean = false): StreamError {
  const errorMessage = extractErrorMessage(error, isAuthenticated);
  return classifyStreamError(new Error(errorMessage));
}

/**
 * 쿼터 정보 추출 (Axios 에러에서)
 */
export function extractQuotaInfo(
  error: unknown
): { remaining: number; limit: number } | null {
  if (error && typeof error === 'object' && 'status' in error) {
    const apiError = error as AxiosError<{ quotaInfo?: { remaining: number; limit: number } }> & { 
      status?: number; 
      quotaInfo?: { remaining: number; limit: number } 
    };
    
    if (apiError.quotaInfo) {
      return apiError.quotaInfo;
    }
    
    if (apiError.response?.headers) {
      const quotaRemaining = apiError.response.headers['x-quota-remaining'];
      const quotaLimit = apiError.response.headers['x-quota-limit'];
      
      if (quotaRemaining !== undefined && quotaLimit !== undefined) {
        return {
          remaining: parseInt(quotaRemaining, 10),
          limit: parseInt(quotaLimit, 10),
        };
      }
    }
  }
  
  return null;
}

