/**
 * 문서 참조 링크 컴포넌트
 * "문서 1", "문서 2" 등의 텍스트를 클릭 가능한 링크로 변환합니다.
 */
import { FileText } from 'lucide-react';

interface DocumentReferenceProps {
  documentIndex: number;
  onClick: () => void;
  disabled?: boolean;
}

export function DocumentReference({
  documentIndex,
  onClick,
  disabled = false,
}: DocumentReferenceProps) {
  return (
    <button
      onClick={(e) => {
        if (disabled) {
          e.preventDefault();
          e.stopPropagation();
          return;
        }
        e.preventDefault();
        e.stopPropagation();
        onClick();
      }}
      disabled={disabled}
      className={`inline-flex items-center gap-1 px-1.5 py-0.5 rounded text-xs font-medium transition-colors border ${
        disabled
          ? 'bg-slate-100 text-slate-400 border-slate-200 cursor-not-allowed'
          : 'bg-blue-50 text-blue-600 hover:bg-blue-100 cursor-pointer border-blue-200 hover:border-blue-300'
      }`}
      title={disabled ? '참고자료 정보를 기다리는 중...' : `문서 ${documentIndex + 1} 정보 보기`}
      aria-label={disabled ? '참고자료 정보를 기다리는 중...' : `문서 ${documentIndex + 1} 정보 보기`}
    >
      <FileText className="w-3 h-3 flex-shrink-0" />
      <span>문서 {documentIndex + 1}</span>
    </button>
  );
}

