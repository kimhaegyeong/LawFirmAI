/**
 * 헤더 컴포넌트
 */
import { Menu, Scale } from 'lucide-react';

interface HeaderProps {
  onMenuClick?: () => void;
}

export function Header({ onMenuClick }: HeaderProps) {
  return (
    <div className="bg-white border-b border-slate-200 px-6 py-4 flex items-center justify-between">
      <div className="flex items-center gap-3">
        <button
          onClick={onMenuClick}
          className="p-2 hover:bg-slate-100 rounded-lg transition-colors"
        >
          <Menu className="w-5 h-5 text-slate-600" />
        </button>
        <div className="flex items-center gap-2">
          <Scale className="w-6 h-6 text-blue-600" />
          <h1 className="text-xl font-bold text-slate-800">법률 AI 어시스턴트</h1>
        </div>
      </div>
    </div>
  );
}

