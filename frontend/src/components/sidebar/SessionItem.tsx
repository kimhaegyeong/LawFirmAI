/**
 * 세션 아이템 컴포넌트
 */
import { MoreVertical } from 'lucide-react';
import { useState, useEffect, useRef } from 'react';
import { formatRelativeTime } from '../../utils/dateUtils';
import type { Session } from '../../types/session';

interface SessionItemProps {
  session: Session;
  onSelect?: (session: Session) => void;
  onRename?: (session: Session) => void;
  onCopy?: (session: Session) => void;
  onExport?: (session: Session) => void;
  onDelete?: (session: Session) => void;
}

export function SessionItem({
  session,
  onSelect,
  onRename,
  onCopy,
  onExport,
  onDelete,
}: SessionItemProps) {
  const [showMenu, setShowMenu] = useState(false);
  const menuRef = useRef<HTMLDivElement>(null);
  const itemRef = useRef<HTMLDivElement>(null);

  // 외부 클릭 시 메뉴 닫기
  useEffect(() => {
    const handleClickOutside = (event: MouseEvent) => {
      if (
        showMenu &&
        menuRef.current &&
        itemRef.current &&
        !menuRef.current.contains(event.target as Node) &&
        !itemRef.current.querySelector('button')?.contains(event.target as Node)
      ) {
        setShowMenu(false);
      }
    };

    if (showMenu) {
      // 약간의 지연을 두어 현재 클릭 이벤트가 완료된 후 리스너 추가
      setTimeout(() => {
        document.addEventListener('click', handleClickOutside);
      }, 0);
      
      return () => {
        document.removeEventListener('click', handleClickOutside);
      };
    }
  }, [showMenu]);

  const handleClick = (e: React.MouseEvent) => {
    if (showMenu) {
      e.stopPropagation();
      return;
    }
    onSelect?.(session);
  };

  const handleMenuClick = (e: React.MouseEvent) => {
    e.stopPropagation();
    setShowMenu(!showMenu);
  };

  const handleAction = (action: (session: Session) => void) => {
    action(session);
    setShowMenu(false);
  };

  return (
    <div
      ref={itemRef}
      className="group pl-6 pr-2 py-2 hover:bg-slate-100 rounded-lg cursor-pointer transition-colors relative"
      onClick={handleClick}
      onKeyDown={(e) => {
        if (e.key === 'Enter' || e.key === ' ') {
          e.preventDefault();
          handleClick();
        }
      }}
      role="button"
      tabIndex={0}
    >
      <div className="flex items-start justify-between gap-2">
        <div className="flex-1 min-w-0">
          <div className="text-sm font-medium text-slate-800 truncate">
            {session.title || '제목 없음'}
          </div>
          <div className="text-xs text-slate-500 mt-0.5">
            {session.message_count}개 메시지 •{' '}
            {session.updated_at
              ? formatRelativeTime(session.updated_at)
              : session.created_at
              ? formatRelativeTime(session.created_at)
              : ''}
          </div>
        </div>
        <button
          onClick={handleMenuClick}
          className="opacity-0 group-hover:opacity-100 transition-opacity p-1 hover:bg-slate-200 rounded"
        >
          <MoreVertical className="w-4 h-4 text-slate-400" />
        </button>
      </div>

      {showMenu && (
        <div
          ref={menuRef}
          className="absolute right-2 top-8 bg-white border border-slate-200 rounded-lg shadow-lg z-10 min-w-[150px]"
        >
          {onRename && (
            <button
              onClick={() => handleAction(onRename)}
              className="w-full px-3 py-2 text-sm text-left hover:bg-slate-100 rounded-t-lg transition-colors"
            >
              이름 변경
            </button>
          )}
          {onCopy && (
            <button
              onClick={() => handleAction(onCopy)}
              className="w-full px-3 py-2 text-sm text-left hover:bg-slate-100 transition-colors"
            >
              복사본 만들기
            </button>
          )}
          {onExport && (
            <button
              onClick={() => handleAction(onExport)}
              className="w-full px-3 py-2 text-sm text-left hover:bg-slate-100 transition-colors"
            >
              내보내기
            </button>
          )}
          {onDelete && (
            <button
              onClick={() => handleAction(onDelete)}
              className="w-full px-3 py-2 text-sm text-left hover:bg-red-50 text-red-600 rounded-b-lg transition-colors"
            >
              삭제
            </button>
          )}
        </div>
      )}
    </div>
  );
}

