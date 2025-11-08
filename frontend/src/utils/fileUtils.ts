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

/**
 * 이미지 파일을 Base64로 변환
 */
export function convertImageToBase64(file: File): Promise<string> {
  return new Promise((resolve, reject) => {
    const reader = new FileReader();
    reader.onload = () => {
      const result = reader.result as string;
      // data:image/png;base64, 형태에서 base64 부분만 추출
      if (result.startsWith('data:')) {
        resolve(result);
      } else {
        resolve(result);
      }
    };
    reader.onerror = (error) => {
      reject(error);
    };
    reader.readAsDataURL(file);
  });
}

/**
 * 파일을 Base64로 변환 (이미지, 텍스트, PDF, DOCX 등)
 */
export function convertFileToBase64(file: File): Promise<string> {
  return new Promise((resolve, reject) => {
    const reader = new FileReader();
    reader.onload = () => {
      const result = reader.result as string;
      // data:image/png;base64, 형태
      if (result.startsWith('data:')) {
        resolve(result);
      } else {
        resolve(result);
      }
    };
    reader.onerror = (error) => {
      reject(error);
    };
    reader.readAsDataURL(file);
  });
}

