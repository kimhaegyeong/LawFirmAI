/**
 * 메타데이터 처리 유틸리티
 */

/**
 * unknown 타입을 string | number | undefined로 안전하게 변환하는 헬퍼 함수
 */
export function getMetadataValue(value: unknown): string | number | undefined {
  if (value === null || value === undefined) {
    return undefined;
  }
  if (typeof value === 'string' || typeof value === 'number') {
    return value;
  }
  return String(value);
}

