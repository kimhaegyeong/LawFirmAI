/**
 * API 기본 설정 및 Axios 인스턴스
 */
import axios, { AxiosInstance, AxiosError, AxiosRequestConfig, AxiosResponse } from 'axios';

// 개발 환경에서는 Vite 프록시를 통해 요청 (상대 경로 사용)
// 프로덕션 환경에서는 절대 URL 사용
const isDev = import.meta.env.DEV;
const API_BASE_URL = import.meta.env.VITE_API_BASE_URL || (isDev ? '' : 'http://localhost:8000');
const API_VERSION = '/api/v1'; // API 버전 prefix

/**
 * Axios 인스턴스 생성
 */
export const api: AxiosInstance = axios.create({
  baseURL: `${API_BASE_URL}${API_VERSION}`,
  timeout: 30000,
  headers: {
    'Content-Type': 'application/json',
  },
});

/**
 * 요청 인터셉터
 */
api.interceptors.request.use(
  (config) => {
    // 요청 전 처리 (인증 토큰 등)
    // 디버깅: 요청 정보 로깅
    if (import.meta.env.DEV) {
      console.log(`[API Request] ${config.method?.toUpperCase()} ${config.url}`);
      console.log('[API Request] Full URL:', config.baseURL + config.url);
    }
    return config;
  },
  (error) => {
    return Promise.reject(error);
  }
);

/**
 * 응답 인터셉터
 */
api.interceptors.response.use(
  (response: AxiosResponse) => {
    // 디버깅: 응답 정보 로깅
    if (import.meta.env.DEV) {
      console.log(`[API Response] ${response.config.method?.toUpperCase()} ${response.config.url} - ${response.status}`);
      // CORS 헤더 확인
      const corsHeaders = {
        'Access-Control-Allow-Origin': response.headers['access-control-allow-origin'],
        'Access-Control-Allow-Credentials': response.headers['access-control-allow-credentials'],
      };
      console.log('[API Response] CORS Headers:', corsHeaders);
    }
    return response;
  },
  (error: AxiosError) => {
    // 에러 처리
    if (error.response) {
      // 서버 응답 에러
      const status = error.response.status;
      const data = error.response.data as any;
      
      // CORS 에러 체크
      if (error.code === 'ERR_NETWORK' || error.message.includes('CORS')) {
        console.error('CORS 에러: 서버에서 CORS 헤더가 올바르게 설정되지 않았습니다.');
        console.error('응답 헤더:', error.response.headers);
      }
      
      switch (status) {
        case 401:
          // 인증 에러
          console.error('인증이 필요합니다.');
          break;
        case 403:
          // 권한 에러
          console.error('접근 권한이 없습니다.');
          break;
        case 404:
          // 리소스 없음
          console.error('요청한 리소스를 찾을 수 없습니다.');
          break;
        case 500:
          // 서버 에러
          console.error('서버 오류가 발생했습니다.');
          break;
        default:
          console.error('에러가 발생했습니다:', data?.detail || error.message);
      }
    } else if (error.request) {
      // 요청은 보냈지만 응답이 없음
      const baseURL = error.config?.baseURL || API_BASE_URL + API_VERSION;
      const url = error.config?.url || '';
      const fullURL = baseURL + url;
      
      console.error('서버에 연결할 수 없습니다.');
      console.error('요청 URL:', url);
      console.error('요청 BaseURL:', baseURL);
      console.error('전체 URL:', fullURL);
      
      // 연결 거부 에러인 경우 더 자세한 안내 제공
      if (error.code === 'ERR_NETWORK' || error.message.includes('ERR_CONNECTION_REFUSED')) {
        console.error('');
        console.error('========================================');
        console.error('🔴 API 서버 연결 실패');
        console.error('========================================');
        console.error('가능한 원인:');
        console.error('1. API 서버가 실행되지 않았습니다.');
        console.error('   → API 서버를 시작하세요:');
        console.error('      cd api');
        console.error('      python -m uvicorn api.main:app --host 0.0.0.0 --port 8000 --reload');
        console.error('   또는:');
        console.error('      cd api');
        console.error('      start_server.bat');
        console.error('');
        console.error('2. API 서버가 다른 포트에서 실행 중입니다.');
        console.error('   → .env 파일에서 VITE_API_BASE_URL을 확인하세요.');
        console.error('');
        console.error('3. 방화벽이 연결을 차단하고 있습니다.');
        console.error('   → 방화벽 설정을 확인하세요.');
        console.error('========================================');
      }
    } else {
      // 요청 설정 중 에러
      console.error('요청 설정 중 에러가 발생했습니다:', error.message);
    }
    
    return Promise.reject(error);
  }
);

/**
 * API 에러 타입
 */
export interface ApiError {
  message: string;
  status?: number;
  detail?: string;
}

/**
 * API 에러 추출
 */
export function extractApiError(error: any): ApiError {
  if (axios.isAxiosError(error)) {
    const axiosError = error as AxiosError;
    if (axiosError.response) {
      const data = axiosError.response.data as any;
      return {
        message: data?.detail || data?.message || '에러가 발생했습니다.',
        status: axiosError.response.status,
        detail: data?.detail,
      };
    } else if (axiosError.request) {
      // 연결 거부 에러인 경우 더 자세한 메시지 제공
      if (axiosError.code === 'ERR_NETWORK' || 
          axiosError.message.includes('ERR_CONNECTION_REFUSED') ||
          axiosError.message.includes('Failed to fetch')) {
        return {
          message: 'API 서버에 연결할 수 없습니다. 서버가 실행 중인지 확인하세요.',
          detail: `연결 시도: ${axiosError.config?.baseURL || ''}${axiosError.config?.url || ''}`,
        };
      }
      return {
        message: '서버에 연결할 수 없습니다.',
        detail: `연결 시도: ${axiosError.config?.baseURL || ''}${axiosError.config?.url || ''}`,
      };
    }
  }
  
  return {
    message: error?.message || '알 수 없는 에러가 발생했습니다.',
  };
}

