/**
 * 진행 상황 표시 컴포넌트
 * 연한 색으로 진행 상황만 표시
 */
import { Loader2 } from 'lucide-react';

interface ProgressIndicatorProps {
  message: string;
}

export function ProgressIndicator({ message }: ProgressIndicatorProps) {
  return (
    <div className="flex items-center justify-start py-2 px-4">
      <div className="flex items-center gap-2 text-slate-400 text-sm">
        <Loader2 className="w-4 h-4 animate-spin" />
        <span>{message}</span>
      </div>
    </div>
  );
}

