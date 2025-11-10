/**
 * API 기본 설정 및 Axios 인스턴스
 */
import axios, { AxiosInstance, AxiosError, AxiosResponse } from 'axios';
import logger from '../utils/logger';

// 개발 환경에서는 명시적으로 localhost:8000 사용
// 프로덕션 환경에서는 환경 변수 또는 기본값 사용
const isDev = import.meta.env.DEV;
const API_BASE_URL = import.meta.env.VITE_API_BASE_URL || (isDev ? 'http://localhost:8000' : 'http://localhost:8000');
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
  async (config) => {
    // 요청 전 처리 (인증 토큰 등)
    const token = getAccessToken();
    if (token) {
      config.headers.Authorization = `Bearer ${token}`;
    } else {
      // 비로그인 사용자의 경우 익명 세션 ID 헤더 추가
      const { getOrCreateAnonymousSessionId } = await import('../utils/anonymousSession');
      const anonymousSessionId = await getOrCreateAnonymousSessionId();
      if (anonymousSessionId) {
        config.headers['X-Anonymous-Session-Id'] = anonymousSessionId;
      }
    }
    
    // 디버깅: 요청 정보 로깅
    const fullURL = `${config.baseURL || ''}${config.url || ''}`;
    if (import.meta.env.DEV) {
      logger.debug(`[API Request] ${config.method?.toUpperCase()} ${config.url}`);
      logger.debug('[API Request] Base URL:', config.baseURL);
      logger.debug('[API Request] Full URL:', fullURL);
      console.log('[API] Request URL:', fullURL);
    }
    return config;
  },
  (error) => {
    logger.error('[API] Request interceptor error:', error);
    return Promise.reject(error);
  }
);

/**
 * Access token 조회 (순환 참조 방지를 위해 로컬 함수로 구현)
 */
function getAccessToken(): string | null {
  try {
    return localStorage.getItem('access_token');
  } catch (error) {
    return null;
  }
}

/**
 * 응답 인터셉터
 */
api.interceptors.response.use(
  (response: AxiosResponse) => {
    // 디버깅: 응답 정보 로깅
    if (import.meta.env.DEV) {
      logger.debug(`[API Response] ${response.config.method?.toUpperCase()} ${response.config.url} - ${response.status}`);
      // CORS 헤더 확인
      const corsHeaders = {
        'Access-Control-Allow-Origin': response.headers['access-control-allow-origin'],
        'Access-Control-Allow-Credentials': response.headers['access-control-allow-credentials'],
      };
      logger.debug('[API Response] CORS Headers:', corsHeaders);
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
        logger.error('CORS 에러: 서버에서 CORS 헤더가 올바르게 설정되지 않았습니다.');
        logger.error('응답 헤더:', error.response.headers);
      }
      
      switch (status) {
        case 401:
          // 인증 에러
          logger.error('인증이 필요합니다.');
          // 토큰이 있는 경우 토큰 갱신 시도 (useAuth hook에서 처리)
          // 여기서는 에러만 로깅하고 그대로 전달
          break;
        case 403:
          // 권한 에러
          logger.error('접근 권한이 없습니다.');
          break;
        case 404:
          // 리소스 없음
          logger.error('요청한 리소스를 찾을 수 없습니다.');
          break;
        case 429:
          // Rate Limit 또는 익명 사용자 제한 초과
          const quotaRemaining = error.response.headers['x-quota-remaining'];
          const quotaLimit = error.response.headers['x-quota-limit'];
          if (quotaRemaining === '0' && quotaLimit === '3') {
            logger.error('무료 질의 3회를 모두 사용하셨습니다. 계속 사용하려면 로그인이 필요합니다.');
          } else {
            logger.error('요청 한도를 초과했습니다. 잠시 후 다시 시도해주세요.');
          }
          break;
        case 500:
          // 서버 에러
          logger.error('서버 오류가 발생했습니다.');
          break;
        default:
          logger.error('에러가 발생했습니다:', data?.detail || error.message);
      }
    } else if (error.request) {
      // 요청은 보냈지만 응답이 없음
      const baseURL = error.config?.baseURL || API_BASE_URL + API_VERSION;
      const url = error.config?.url || '';
      const fullURL = baseURL + url;
      
      console.error('[API] Connection failed:', {
        baseURL,
        url,
        fullURL,
        errorCode: error.code,
        errorMessage: error.message,
        timeout: error.config?.timeout,
      });
      
      logger.error('서버에 연결할 수 없습니다.');
      logger.error('요청 URL:', url);
      logger.error('요청 BaseURL:', baseURL);
      logger.error('전체 URL:', fullURL);
      logger.error('에러 코드:', error.code || 'N/A');
      logger.error('에러 메시지:', error.message);
      
      // 연결 거부 에러인 경우 더 자세한 안내 제공
      if (error.code === 'ERR_NETWORK' || 
          error.code === 'ECONNREFUSED' ||
          error.message.includes('ERR_CONNECTION_REFUSED') || 
          error.message.includes('Failed to fetch') ||
          error.message.includes('Network Error')) {
        logger.error('');
        logger.error('========================================');
        logger.error('🔴 API 서버 연결 실패');
        logger.error('========================================');
        logger.error('요청 정보:');
        logger.error(`  - Base URL: ${baseURL}`);
        logger.error(`  - 요청 URL: ${url}`);
        logger.error(`  - 전체 URL: ${fullURL}`);
        logger.error(`  - 에러 코드: ${error.code || 'N/A'}`);
        logger.error(`  - 에러 메시지: ${error.message}`);
        logger.error(`  - 타임아웃: ${error.config?.timeout || 'N/A'}ms`);
        logger.error('');
        logger.error('가능한 원인:');
        logger.error('1. API 서버가 실행되지 않았습니다.');
        logger.error('   → API 서버를 시작하세요:');
        logger.error('      cd api');
        logger.error('      python -m uvicorn api.main:app --host 0.0.0.0 --port 8000 --reload');
        logger.error('   또는:');
        logger.error('      cd api');
        logger.error('      start_server.bat');
        logger.error('');
        logger.error('2. API 서버가 다른 포트에서 실행 중입니다.');
        logger.error(`   → 현재 설정: ${API_BASE_URL}`);
        logger.error('   → .env 파일에서 VITE_API_BASE_URL을 확인하세요.');
        logger.error('');
        logger.error('3. CORS 또는 CSP 문제일 수 있습니다.');
        logger.error('   → 브라우저 개발자 도구의 Network 탭에서 응답 헤더를 확인하세요.');
        logger.error('');
        logger.error('4. 방화벽이 연결을 차단하고 있습니다.');
        logger.error('   → 방화벽 설정을 확인하세요.');
        logger.error('========================================');
      }
    } else {
      // 요청 설정 중 에러
      logger.error('요청 설정 중 에러가 발생했습니다:', error.message);
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
export function extractApiError(error: any): Error {
  if (axios.isAxiosError(error)) {
    const axiosError = error as AxiosError;
    if (axiosError.response) {
      const data = axiosError.response.data as any;
      let message = data?.detail || data?.message || '에러가 발생했습니다.';
      
      // 500 오류인 경우 더 자세한 정보 표시
      if (axiosError.response.status === 500) {
        console.error('[API] 500 Error Details:', {
          status: axiosError.response.status,
          data: data,
          detail: data?.detail,
          message: data?.message,
        });
        
        // detail이 있으면 그대로 사용, 없으면 기본 메시지
        if (data?.detail) {
          message = data.detail;
        } else {
          message = '서버 오류가 발생했습니다. API 서버 로그를 확인하세요.';
        }
      }
      
      const apiError = new Error(message);
      (apiError as any).status = axiosError.response.status;
      (apiError as any).detail = data?.detail;
      return apiError;
    } else if (axiosError.request) {
      // 연결 거부 에러인 경우 더 자세한 메시지 제공
      if (axiosError.code === 'ERR_NETWORK' || 
          axiosError.message.includes('ERR_CONNECTION_REFUSED') ||
          axiosError.message.includes('Failed to fetch')) {
        const message = 'API 서버에 연결할 수 없습니다. 서버가 실행 중인지 확인하세요.';
        const apiError = new Error(message);
        (apiError as any).detail = `연결 시도: ${axiosError.config?.baseURL || ''}${axiosError.config?.url || ''}`;
        return apiError;
      }
      const message = '서버에 연결할 수 없습니다.';
      const apiError = new Error(message);
      (apiError as any).detail = `연결 시도: ${axiosError.config?.baseURL || ''}${axiosError.config?.url || ''}`;
      return apiError;
    }
  }
  
  if (error instanceof Error) {
    return error;
  }
  
  return new Error(error?.message || '알 수 없는 에러가 발생했습니다.');
}


