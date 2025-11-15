/**
 * 익명 세션 관리 유틸리티
 */
import logger from './logger';

const ANONYMOUS_SESSION_ID_KEY = 'anonymous_session_id';
const ANONYMOUS_FINGERPRINT_KEY = 'anonymous_fingerprint';

/**
 * 익명 세션 ID 생성
 */
function generateAnonymousSessionId(): string {
  const array = new Uint8Array(32);
  crypto.getRandomValues(array);
  return Array.from(array, byte => byte.toString(16).padStart(2, '0')).join('');
}

/**
 * 브라우저 fingerprint 생성
 */
async function generateFingerprint(): Promise<string> {
  const components: string[] = [];
  
  // User Agent
  components.push(navigator.userAgent);
  
  // Language
  components.push(navigator.language);
  
  // Screen resolution
  components.push(`${screen.width}x${screen.height}`);
  
  // Color depth
  components.push(screen.colorDepth.toString());
  
  // Timezone
  components.push(Intl.DateTimeFormat().resolvedOptions().timeZone);
  
  // Platform
  components.push(navigator.platform);
  
  // Hardware concurrency
  components.push(navigator.hardwareConcurrency?.toString() || '0');
  
  // Canvas fingerprint (간단한 버전)
  try {
    const canvas = document.createElement('canvas');
    const ctx = canvas.getContext('2d');
    if (ctx) {
      ctx.textBaseline = 'top';
      ctx.font = '14px Arial';
      ctx.fillText('Anonymous session fingerprint', 2, 2);
      const canvasData = canvas.toDataURL();
      components.push(canvasData.substring(0, 100));
    }
  } catch (e) {
    logger.warn('Failed to generate canvas fingerprint:', e);
  }
  
  // 모든 컴포넌트를 결합하여 해시 생성
  const combined = components.join('|');
  
  // 간단한 해시 함수 (SHA-256 사용)
  const encoder = new TextEncoder();
  const data = encoder.encode(combined);
  const hashBuffer = await crypto.subtle.digest('SHA-256', data);
  const hashArray = Array.from(new Uint8Array(hashBuffer));
  const hashHex = hashArray.map(b => b.toString(16).padStart(2, '0')).join('');
  
  return hashHex.substring(0, 32);
}

/**
 * 익명 세션 ID 가져오기 또는 생성
 */
export async function getOrCreateAnonymousSessionId(): Promise<string> {
  try {
    // 로컬스토리지에서 기존 ID 확인
    let sessionId = localStorage.getItem(ANONYMOUS_SESSION_ID_KEY);
    
    if (!sessionId) {
      // 새 ID 생성
      sessionId = generateAnonymousSessionId();
      localStorage.setItem(ANONYMOUS_SESSION_ID_KEY, sessionId);
      logger.info('Created new anonymous session ID:', sessionId);
    }
    
    return sessionId;
  } catch (error) {
    logger.error('Failed to get or create anonymous session ID:', error);
    // 에러 발생 시 임시 ID 반환
    return `temp_${Date.now()}_${Math.random().toString(36).substring(2, 11)}`;
  }
}

/**
 * 브라우저 fingerprint 가져오기 또는 생성
 */
export async function getOrCreateFingerprint(): Promise<string> {
  try {
    // 로컬스토리지에서 기존 fingerprint 확인
    let fingerprint = localStorage.getItem(ANONYMOUS_FINGERPRINT_KEY);
    
    if (!fingerprint) {
      // 새 fingerprint 생성
      fingerprint = await generateFingerprint();
      localStorage.setItem(ANONYMOUS_FINGERPRINT_KEY, fingerprint);
      logger.info('Created new browser fingerprint:', fingerprint);
    }
    
    return fingerprint;
  } catch (error) {
    logger.error('Failed to get or create fingerprint:', error);
    // 에러 발생 시 기본 fingerprint 반환
    return `default_${navigator.userAgent.substring(0, 20)}`;
  }
}

/**
 * 익명 세션 ID 삭제
 */
export function clearAnonymousSessionId(): void {
  try {
    localStorage.removeItem(ANONYMOUS_SESSION_ID_KEY);
    logger.info('Cleared anonymous session ID');
  } catch (error) {
    logger.error('Failed to clear anonymous session ID:', error);
  }
}

/**
 * 익명 세션 ID 가져오기 (생성하지 않음)
 */
export function getAnonymousSessionId(): string | null {
  try {
    return localStorage.getItem(ANONYMOUS_SESSION_ID_KEY);
  } catch (error) {
    logger.error('Failed to get anonymous session ID:', error);
    return null;
  }
}

