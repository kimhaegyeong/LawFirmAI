/**
 * 회원탈퇴 확인 모달 컴포넌트
 */
import { useState } from 'react';
import { AlertTriangle, X, Loader2 } from 'lucide-react';
import { deleteAccount } from '../../services/authService';
import logger from '../../utils/logger';

interface DeleteAccountModalProps {
  isOpen: boolean;
  onClose: () => void;
  onSuccess: () => void;
}

export function DeleteAccountModal({ isOpen, onClose, onSuccess }: DeleteAccountModalProps) {
  const [isDeleting, setIsDeleting] = useState(false);
  const [confirmText, setConfirmText] = useState('');
  const [error, setError] = useState<string | null>(null);

  const requiredText = '회원탈퇴';

  const handleDelete = async () => {
    if (confirmText !== requiredText) {
      setError('확인 문구를 정확히 입력해주세요.');
      return;
    }

    setIsDeleting(true);
    setError(null);

    try {
      const result = await deleteAccount();
      logger.info('Account deleted:', result);
      onSuccess();
      onClose();
      window.location.href = '/';
    } catch (err) {
      const errorMessage = err instanceof Error ? err.message : '회원탈퇴에 실패했습니다.';
      setError(errorMessage);
      logger.error('Failed to delete account:', err);
    } finally {
      setIsDeleting(false);
    }
  };

  const handleClose = () => {
    if (!isDeleting) {
      setConfirmText('');
      setError(null);
      onClose();
    }
  };

  if (!isOpen) return null;

  return (
    <div className="fixed inset-0 z-50 flex items-center justify-center bg-black bg-opacity-50">
      <div className="bg-white rounded-lg shadow-xl max-w-md w-full mx-4">
        <div className="p-6">
          <div className="flex items-center justify-between mb-4">
            <div className="flex items-center gap-3">
              <div className="w-10 h-10 bg-red-100 rounded-full flex items-center justify-center">
                <AlertTriangle className="w-5 h-5 text-red-600" />
              </div>
              <h2 className="text-xl font-bold text-slate-800">회원탈퇴</h2>
            </div>
            <button
              onClick={handleClose}
              disabled={isDeleting}
              className="p-2 hover:bg-slate-100 rounded-lg transition-colors disabled:opacity-50 disabled:cursor-not-allowed"
            >
              <X className="w-5 h-5 text-slate-600" />
            </button>
          </div>

          <div className="mb-6">
            <p className="text-sm text-slate-600 mb-4">
              회원탈퇴를 진행하시겠습니까? 이 작업은 되돌릴 수 없으며, 모든 세션과 메시지가 삭제됩니다.
            </p>
            <div className="bg-red-50 border border-red-200 rounded-lg p-4 mb-4">
              <p className="text-sm font-medium text-red-800 mb-2">주의사항:</p>
              <ul className="text-sm text-red-700 space-y-1 list-disc list-inside">
                <li>모든 대화 히스토리가 삭제됩니다</li>
                <li>저장된 세션 정보가 모두 삭제됩니다</li>
                <li>이 작업은 되돌릴 수 없습니다</li>
              </ul>
            </div>
            <div className="mb-4">
              <label className="block text-sm font-medium text-slate-700 mb-2">
                확인을 위해 <span className="font-bold text-red-600">"{requiredText}"</span>를 입력하세요:
              </label>
              <input
                type="text"
                value={confirmText}
                onChange={(e) => {
                  setConfirmText(e.target.value);
                  setError(null);
                }}
                disabled={isDeleting}
                className="w-full px-3 py-2 border border-slate-300 rounded-lg focus:outline-none focus:ring-2 focus:ring-red-500 disabled:opacity-50 disabled:cursor-not-allowed"
                placeholder={requiredText}
              />
            </div>
            {error && (
              <div className="mb-4 p-3 bg-red-50 border border-red-200 rounded-lg">
                <p className="text-sm text-red-600">{error}</p>
              </div>
            )}
          </div>

          <div className="flex items-center gap-3">
            <button
              onClick={handleClose}
              disabled={isDeleting}
              className="flex-1 px-4 py-2 text-sm font-medium text-slate-700 bg-slate-100 hover:bg-slate-200 rounded-lg transition-colors disabled:opacity-50 disabled:cursor-not-allowed"
            >
              취소
            </button>
            <button
              onClick={handleDelete}
              disabled={isDeleting || confirmText !== requiredText}
              className="flex-1 px-4 py-2 text-sm font-medium text-white bg-red-600 hover:bg-red-700 rounded-lg transition-colors disabled:opacity-50 disabled:cursor-not-allowed flex items-center justify-center gap-2"
            >
              {isDeleting ? (
                <>
                  <Loader2 className="w-4 h-4 animate-spin" />
                  처리 중...
                </>
              ) : (
                '회원탈퇴'
              )}
            </button>
          </div>
        </div>
      </div>
    </div>
  );
}

