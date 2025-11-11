/**
 * 날짜 유틸리티 함수
 */
import logger from './logger';
import type { Session } from '../types/session';

export type DateGroup = '오늘' | '어제' | '지난 7일' | '지난 30일' | '이전';

export interface DateGrouped {
  group: DateGroup;
  sessions: Session[];
}

/**
 * 안전한 날짜 파싱
 */
function parseDate(date: string | Date): Date {
  if (date instanceof Date) {
    return date;
  }
  
  if (typeof date === 'string') {
    // 빈 문자열 처리
    if (!date.trim()) {
      return new Date(0); // 유효하지 않은 날짜로 처리
    }
    
    // ISO 형식 또는 SQLite 형식 파싱
    const normalized = date.replace(' ', 'T').replace(/Z$/, '');
    const parsed = new Date(normalized);
    
    // 유효성 검증
    if (isNaN(parsed.getTime())) {
      logger.warn(`Invalid date string: ${date}`);
      return new Date(0);
    }
    
    return parsed;
  }
  
  return new Date(0);
}

/**
 * 한국 시간대(KST, UTC+9) 기준으로 날짜만 추출 (시간 제외)
 * KST 기준 자정(00:00:00)의 타임스탬프를 반환
 */
function getKSTDateTimestamp(date: Date): number {
  // 한국 시간대 오프셋 (UTC+9, 9시간 = 32400000 밀리초)
  const KST_OFFSET_MS = 9 * 60 * 60 * 1000;
  
  // 현재 날짜를 UTC 시간으로 변환
  // date.getTime()은 UTC 기준 밀리초를 반환하므로
  // 로컬 시간대 오프셋을 더하면 UTC 시간을 얻을 수 있음
  const utcTime = date.getTime() + (date.getTimezoneOffset() * 60000);
  
  // KST 시간으로 변환
  const kstTime = utcTime + KST_OFFSET_MS;
  const kstDate = new Date(kstTime);
  
  // KST 기준으로 년/월/일 추출
  const kstYear = kstDate.getUTCFullYear();
  const kstMonth = kstDate.getUTCMonth();
  const kstDay = kstDate.getUTCDate();
  
  // KST 기준 자정(00:00:00)의 UTC 타임스탬프 계산
  // Date.UTC는 UTC 기준으로 해당 날짜의 자정을 계산
  // 하지만 우리가 원하는 것은 KST 기준 날짜의 자정이므로
  // KST 오프셋을 빼서 KST 자정이 UTC에서 어떤 시점인지 계산
  const kstMidnightUTC = Date.UTC(kstYear, kstMonth, kstDay, 0, 0, 0, 0) - KST_OFFSET_MS;
  
  return kstMidnightUTC;
}

/**
 * 날짜를 그룹으로 분류 (KST 기반)
 */
export function getDateGroup(date: string | Date): DateGroup {
  const now = new Date();
  const targetDate = parseDate(date);
  
  // 유효하지 않은 날짜 처리
  if (isNaN(targetDate.getTime()) || targetDate.getTime() === 0) {
    return '이전';
  }
  
  // KST 기준으로 날짜만 추출 (자정 타임스탬프)
  const nowKST = getKSTDateTimestamp(now);
  const targetKST = getKSTDateTimestamp(targetDate);
  
  // 날짜 차이 계산 (밀리초 단위)
  const diffTime = nowKST - targetKST;
  const diffDays = Math.floor(diffTime / (1000 * 60 * 60 * 24));
  
  if (diffDays === 0) {
    return '오늘';
  } else if (diffDays === 1) {
    return '어제';
  } else if (diffDays <= 7) {
    return '지난 7일';
  } else if (diffDays <= 30) {
    return '지난 30일';
  } else {
    return '이전';
  }
}

/**
 * 세션 목록을 날짜별로 그룹화
 */
export function groupSessionsByDate(sessions: Session[]): DateGrouped[] {
  const grouped: Record<DateGroup, Session[]> = {
    '오늘': [],
    '어제': [],
    '지난 7일': [],
    '지난 30일': [],
    '이전': [],
  };
  
  sessions.forEach(session => {
    const dateStr = session.updated_at || session.created_at;
    if (!dateStr) {
      grouped['이전'].push(session);
      return;
    }
    
    const group = getDateGroup(dateStr);
    grouped[group].push(session);
  });
  
  // 순서대로 반환
  const result: DateGrouped[] = [
    { group: '오늘' as DateGroup, sessions: grouped['오늘'] },
    { group: '어제' as DateGroup, sessions: grouped['어제'] },
    { group: '지난 7일' as DateGroup, sessions: grouped['지난 7일'] },
    { group: '지난 30일' as DateGroup, sessions: grouped['지난 30일'] },
    { group: '이전' as DateGroup, sessions: grouped['이전'] },
  ];
  
  return result.filter(item => item.sessions.length > 0);
}

/**
 * 상대 시간 포맷팅 (예: "2시간 전", "3일 전")
 */
export function formatRelativeTime(date: string | Date): string {
  const now = new Date();
  const targetDate = typeof date === 'string' ? new Date(date) : date;
  
  const diffTime = now.getTime() - targetDate.getTime();
  const diffSeconds = Math.floor(diffTime / 1000);
  const diffMinutes = Math.floor(diffSeconds / 60);
  const diffHours = Math.floor(diffMinutes / 60);
  const diffDays = Math.floor(diffHours / 24);
  
  if (diffSeconds < 60) {
    return '방금 전';
  } else if (diffMinutes < 60) {
    return `${diffMinutes}분 전`;
  } else if (diffHours < 24) {
    return `${diffHours}시간 전`;
  } else if (diffDays < 7) {
    return `${diffDays}일 전`;
  } else {
    return formatDate(targetDate);
  }
}

/**
 * 날짜 포맷팅
 */
export function formatDate(date: string | Date): string {
  const targetDate = typeof date === 'string' ? new Date(date) : date;
  return targetDate.toLocaleDateString('ko-KR', {
    year: 'numeric',
    month: 'long',
    day: 'numeric',
  });
}

