/**
 * 사이드바 콘텐츠 컴포넌트
 */
import { Plus } from 'lucide-react';
import { SearchBar } from './SearchBar';
import { SessionList } from './SessionList';
import { LoadingSpinner } from '../common/LoadingSpinner';
import type { Session } from '../../types/session';

interface SidebarContentProps {
  searchQuery?: string;
  onNewSession?: () => void;
  onSearch?: (query: string) => void;
  onSelectSession?: (session: Session) => void;
  onRenameSession?: (session: Session) => void;
  onCopySession?: (session: Session) => void;
  onExportSession?: (session: Session) => void;
  onDeleteSession?: (session: Session) => void;
  isLoading?: boolean;
  refreshTrigger?: number;
}

export function SidebarContent({
  searchQuery,
  onNewSession,
  onSearch,
  onSelectSession,
  onRenameSession,
  onCopySession,
  onExportSession,
  onDeleteSession,
  isLoading = false,
  refreshTrigger,
}: SidebarContentProps) {

  return (
    <>
      {/* 새 상담 시작 버튼 */}
      <div className="p-4 border-b border-slate-200">
        <button
          onClick={onNewSession}
          disabled={isLoading}
          className={`w-full bg-red-500 hover:bg-red-600 text-white rounded-lg py-3 px-4 font-medium transition-colors flex items-center justify-center gap-2 ${
            isLoading ? 'opacity-50 cursor-not-allowed' : ''
          }`}
        >
          {isLoading ? (
            <>
              <LoadingSpinner size="sm" />
              생성 중...
            </>
          ) : (
            <>
              <Plus className="w-5 h-5" />
              새 상담 시작
            </>
          )}
        </button>
      </div>

      {/* 검색 바 */}
      <SearchBar onSearch={onSearch} />

      {/* 세션 리스트 */}
      <SessionList
        searchQuery={searchQuery}
        onSelect={onSelectSession}
        onRename={onRenameSession}
        onCopy={onCopySession}
        onExport={onExportSession}
        onDelete={onDeleteSession}
        refreshTrigger={refreshTrigger}
      />
    </>
  );
}

