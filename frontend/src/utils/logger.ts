import log from 'loglevel';
import { getEnvironmentDefaults, isDebugEnabled } from './environment';

// 환경별 기본값 가져오기
const envDefaults = getEnvironmentDefaults();
const logLevel = import.meta.env.VITE_LOG_LEVEL || envDefaults.logLevel;

// 로그 레벨 설정
if (logLevel === 'debug') {
  log.setLevel('debug');
} else if (logLevel === 'info') {
  log.setLevel('info');
} else if (logLevel === 'warn') {
  log.setLevel('warn');
} else if (logLevel === 'error') {
  log.setLevel('error');
} else {
  // 기본값: 환경에 따라 설정
  log.setLevel(isDebugEnabled() ? 'debug' : 'warn');
}

// 원본 메서드 저장
const originalMethods = {
  trace: log.trace.bind(log),
  debug: log.debug.bind(log),
  info: log.info.bind(log),
  warn: log.warn.bind(log),
  error: log.error.bind(log),
};

// 스타일 설정
const styles = {
  trace: 'color: #9E9E9E; font-weight: bold;',
  debug: 'color: #2196F3; font-weight: bold;',
  info: 'color: #4CAF50; font-weight: bold;',
  warn: 'color: #FF9800; font-weight: bold;',
  error: 'color: #F44336; font-weight: bold;',
};

const emojis = {
  trace: '🔍',
  debug: '🐛',
  info: 'ℹ️',
  warn: '⚠️',
  error: '❌',
};

// 스타일이 적용된 로거 생성
const createStyledLogger = (level: keyof typeof originalMethods) => {
  return (...args: unknown[]) => {
    if (isDebugEnabled()) {
      const timestamp = new Date().toISOString();
      // eslint-disable-next-line security/detect-object-injection
      const prefix = `%c[${timestamp}] ${emojis[level]} [${level.toUpperCase()}]`;
      const consoleMethod = console[level === 'trace' ? 'log' : level];
      consoleMethod(
        prefix,
        // eslint-disable-next-line security/detect-object-injection
        styles[level],
        ...args
      );
    } else {
      // eslint-disable-next-line security/detect-object-injection
      originalMethods[level](...args);
    }
  };
};

// 메서드 오버라이드
log.trace = createStyledLogger('trace');
log.debug = createStyledLogger('debug');
log.info = createStyledLogger('info');
log.warn = createStyledLogger('warn');
log.error = createStyledLogger('error');

// 카테고리별 로거 헬퍼
export const createCategoryLogger = (category: string, color: string) => {
  return {
    trace: (...args: unknown[]) => {
      if (isDebugEnabled()) {
        console.trace(`%c[${category}]`, `color: ${color}; font-weight: bold;`, ...args);
      }
    },
    debug: (...args: unknown[]) => {
      if (isDebugEnabled()) {
        console.debug(`%c[${category}]`, `color: ${color}; font-weight: bold;`, ...args);
      }
    },
    info: (...args: unknown[]) => {
      if (isDebugEnabled()) {
        console.info(`%c[${category}]`, `color: ${color}; font-weight: bold;`, ...args);
      }
    },
    warn: (...args: unknown[]) => {
      console.warn(`%c[${category}]`, `color: ${color}; font-weight: bold;`, ...args);
    },
    error: (...args: unknown[]) => {
      console.error(`%c[${category}]`, `color: ${color}; font-weight: bold;`, ...args);
    },
  };
};

export const apiLogger = createCategoryLogger('API', '#4CAF50');
export const chatLogger = createCategoryLogger('Chat', '#2196F3');
export const authLogger = createCategoryLogger('Auth', '#FF9800');
export const errorLogger = createCategoryLogger('Error', '#F44336');

export default log;

