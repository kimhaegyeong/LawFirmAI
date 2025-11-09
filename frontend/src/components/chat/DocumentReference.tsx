/**
 * 문서 참조 링크 컴포넌트
 * "문서 1", "문서 2" 등의 텍스트를 클릭 가능한 링크로 변환합니다.
 */
import { FileText } from 'lucide-react';

interface DocumentReferenceProps {
  documentIndex: number;
  onClick: () => void;
}

export function DocumentReference({
  documentIndex,
  onClick,
}: DocumentReferenceProps) {
  return (
    <button
      onClick={(e) => {
        e.preventDefault();
        e.stopPropagation();
        onClick();
      }}
      className="inline-flex items-center gap-1 px-1.5 py-0.5 bg-blue-50 text-blue-600 rounded text-xs font-medium hover:bg-blue-100 transition-colors cursor-pointer border border-blue-200 hover:border-blue-300"
      title={`문서 ${documentIndex + 1} 정보 보기`}
      aria-label={`문서 ${documentIndex + 1} 정보 보기`}
    >
      <FileText className="w-3 h-3 flex-shrink-0" />
      <span>문서 {documentIndex + 1}</span>
    </button>
  );
}

