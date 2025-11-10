/**
 * 헤더 컴포넌트
 */
import { useState, useEffect, useRef } from 'react';
import { Menu, Scale, LogOut, User, LogIn, Settings, Trash2 } from 'lucide-react';
import { useAuth } from '../../hooks/useAuth';
import { DeleteAccountModal } from '../auth/DeleteAccountModal';

interface HeaderProps {
  onMenuClick?: () => void;
}

export function Header({ onMenuClick }: HeaderProps) {
  const { user, isAuthenticated, login, logout } = useAuth();
  const [showDeleteModal, setShowDeleteModal] = useState(false);
  const [showUserMenu, setShowUserMenu] = useState(false);
  const userMenuRef = useRef<HTMLDivElement>(null);

  const handleLogout = () => {
    if (confirm('로그아웃하시겠습니까?')) {
      logout();
    }
  };

  const handleDeleteSuccess = () => {
    setShowDeleteModal(false);
  };

  useEffect(() => {
    const handleClickOutside = (event: MouseEvent) => {
      if (userMenuRef.current && !userMenuRef.current.contains(event.target as Node)) {
        setShowUserMenu(false);
      }
    };

    if (showUserMenu) {
      document.addEventListener('mousedown', handleClickOutside);
    }

    return () => {
      document.removeEventListener('mousedown', handleClickOutside);
    };
  }, [showUserMenu]);

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
      {isAuthenticated && user ? (
        <div className="flex items-center gap-3">
          <div className="flex items-center gap-2 text-slate-700">
            {user.picture ? (
              <img
                src={user.picture}
                alt={user.name || user.email || 'User'}
                className="w-8 h-8 rounded-full"
              />
            ) : (
              <div className="w-8 h-8 rounded-full bg-blue-100 flex items-center justify-center">
                <User className="w-4 h-4 text-blue-600" />
              </div>
            )}
            <div className="flex flex-col items-start">
              {user.name && (
                <span className="text-sm font-medium text-slate-800">{user.name}</span>
              )}
              {user.email && (
                <span className="text-xs text-slate-500">{user.email}</span>
              )}
              {user.user_id && (
                <span className="text-xs text-slate-400">ID: {user.user_id}</span>
              )}
            </div>
          </div>
          <div className="relative" ref={userMenuRef}>
            <button
              onClick={() => setShowUserMenu(!showUserMenu)}
              className="flex items-center gap-2 px-3 py-2 text-slate-700 hover:bg-slate-100 rounded-lg transition-colors"
              title="설정"
            >
              <Settings className="w-4 h-4 text-slate-500" />
            </button>
            {showUserMenu && (
              <div className="absolute right-0 mt-2 w-48 bg-white rounded-lg shadow-lg border border-slate-200 py-1 z-50">
                <button
                  onClick={() => {
                    setShowDeleteModal(true);
                    setShowUserMenu(false);
                  }}
                  className="w-full flex items-center gap-2 px-4 py-2 text-sm text-red-600 hover:bg-red-50 transition-colors"
                >
                  <Trash2 className="w-4 h-4" />
                  회원탈퇴
                </button>
              </div>
            )}
          </div>
          <button
            onClick={handleLogout}
            className="flex items-center gap-2 px-3 py-2 text-sm text-slate-700 hover:bg-slate-100 rounded-lg transition-colors"
            title="로그아웃"
          >
            <LogOut className="w-4 h-4" />
            <span>로그아웃</span>
          </button>
        </div>
      ) : (
        <div className="flex items-center gap-3">
          <button
            onClick={login}
            className="flex items-center gap-2 px-4 py-2 bg-blue-600 hover:bg-blue-700 text-white text-sm font-medium rounded-lg transition-colors"
            title="로그인"
          >
            <LogIn className="w-4 h-4" />
            <span>로그인</span>
          </button>
        </div>
      )}
      <DeleteAccountModal
        isOpen={showDeleteModal}
        onClose={() => setShowDeleteModal(false)}
        onSuccess={handleDeleteSuccess}
      />
    </div>
  );
}

