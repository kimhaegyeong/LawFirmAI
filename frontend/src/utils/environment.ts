/**
 * Environment Management
 * 환경 변수 관리 유틸리티
 */

export enum Environment {
  LOCAL = 'local',
  DEVELOPMENT = 'development',
  PRODUCTION = 'production',
}

/**
 * 현재 환경 가져오기
 */
export function getCurrentEnvironment(): Environment {
  const envValue = import.meta.env.VITE_ENVIRONMENT?.toLowerCase().trim() || '';
  
  // 유사한 값 매핑
  const mapping: Record<string, Environment> = {
    'dev': Environment.DEVELOPMENT,
    'prod': Environment.PRODUCTION,
    'local': Environment.LOCAL,
    'development': Environment.DEVELOPMENT,
    'production': Environment.PRODUCTION,
  };
  
  if (envValue in mapping) {
    const result = mapping[envValue as keyof typeof mapping];
    if (result) {
      return result;
    }
  }
  
  // 직접 매칭 시도
  if (Object.values(Environment).includes(envValue as Environment)) {
    return envValue as Environment;
  }
  
  // 기본값: 개발 모드면 development, 아니면 production
  return import.meta.env.DEV ? Environment.DEVELOPMENT : Environment.PRODUCTION;
}

/**
 * 환경 체크 헬퍼 함수
 */
export const isLocal = () => getCurrentEnvironment() === Environment.LOCAL;
export const isDevelopment = () => getCurrentEnvironment() === Environment.DEVELOPMENT;
export const isProduction = () => getCurrentEnvironment() === Environment.PRODUCTION;
export const isDebugEnabled = () => 
  getCurrentEnvironment() === Environment.LOCAL || 
  getCurrentEnvironment() === Environment.DEVELOPMENT;

/**
 * 환경별 기본값
 */
export const getEnvironmentDefaults = () => {
  const env = getCurrentEnvironment();
  
  const defaults = {
    [Environment.LOCAL]: {
      apiBaseUrl: 'http://localhost:8000',
      logLevel: 'debug',
      enableDebugTools: true,
    },
    [Environment.DEVELOPMENT]: {
      apiBaseUrl: import.meta.env.VITE_API_BASE_URL || 'http://localhost:8000',
      logLevel: 'info',
      enableDebugTools: true,
    },
    [Environment.PRODUCTION]: {
      apiBaseUrl: import.meta.env.VITE_API_BASE_URL || window.location.origin,
      logLevel: 'warn',
      enableDebugTools: false,
    },
  };
  
  return defaults[env];
};

