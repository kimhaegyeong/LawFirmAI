/**
 * 파일 유틸리티 함수
 */

export interface FileInfo {
  id: string;
  name: string;
  size: number;
  type: string;
  url?: string;
  file?: File;
}

/**
 * 파일 크기 포맷팅
 */
export function formatFileSize(bytes: number): string {
  if (bytes < 1024) return bytes + ' B';
  if (bytes < 1024 * 1024) return (bytes / 1024).toFixed(1) + ' KB';
  return (bytes / (1024 * 1024)).toFixed(1) + ' MB';
}

/**
 * 파일 타입 확인
 */
export function isImageFile(type: string): boolean {
  return type.startsWith('image/');
}

export function isDocumentFile(type: string): boolean {
  return type.includes('pdf') || 
         type.includes('msword') || 
         type.includes('document') ||
         type.includes('text');
}

/**
 * 파일 ID 생성
 */
export function generateFileId(): string {
  return Math.random().toString(36).substr(2, 9);
}

/**
 * 파일을 FileInfo 객체로 변환
 */
export function fileToFileInfo(file: File): FileInfo {
  return {
    id: generateFileId(),
    name: file.name,
    size: file.size,
    type: file.type,
    file: file,
    url: URL.createObjectURL(file),
  };
}

