/**
 * 클립보드 복사 유틸리티
 */

/**
 * 텍스트를 클립보드에 복사
 * @param text 복사할 텍스트
 * @returns 복사 성공 여부
 */
export async function copyToClipboard(text: string): Promise<boolean> {
  try {
    if (navigator.clipboard && navigator.clipboard.writeText) {
      await navigator.clipboard.writeText(text);
      return true;
    } else {
      const textArea = document.createElement('textarea');
      textArea.value = text;
      textArea.style.position = 'fixed';
      textArea.style.left = '-999999px';
      textArea.style.top = '-999999px';
      document.body.appendChild(textArea);
      textArea.focus();
      textArea.select();
      
      try {
        const successful = document.execCommand('copy');
        document.body.removeChild(textArea);
        return successful;
      } catch (err) {
        document.body.removeChild(textArea);
        return false;
      }
    }
  } catch (err) {
    console.error('Failed to copy text to clipboard:', err);
    return false;
  }
}

/**
 * 텍스트를 클립보드에 복사하고 피드백 표시
 * @param text 복사할 텍스트
 * @param onSuccess 성공 시 콜백
 * @param onError 실패 시 콜백
 */
export async function copyToClipboardWithFeedback(
  text: string,
  onSuccess?: () => void,
  onError?: () => void
): Promise<void> {
  const success = await copyToClipboard(text);
  if (success) {
    onSuccess?.();
  } else {
    onError?.();
  }
}

