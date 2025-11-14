/**
 * Vitest 테스트 설정 파일
 */
import { expect, afterEach } from 'vitest';
import { cleanup } from '@testing-library/react';
import * as matchers from '@testing-library/jest-dom/matchers';

// @testing-library/jest-dom 매처 확장
expect.extend(matchers);

// 각 테스트 후 cleanup
afterEach(() => {
  cleanup();
});

// 전역 모킹 설정
global.console = {
  ...console,
  // 테스트 중 불필요한 로그 억제
  log: () => {},
  debug: () => {},
  info: () => {},
  warn: console.warn,
  error: console.error,
};

