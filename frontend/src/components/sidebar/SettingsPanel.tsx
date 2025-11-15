/**
 * 설정 패널 컴포넌트
 */
import { useState } from 'react';

interface SettingsPanelProps {
  theme?: 'light' | 'dark' | 'auto';
  fontSize?: 'small' | 'medium' | 'large';
  responseSpeed?: 'instant' | 'typing';
  onThemeChange?: (theme: 'light' | 'dark' | 'auto') => void;
  onFontSizeChange?: (size: 'small' | 'medium' | 'large') => void;
  onResponseSpeedChange?: (speed: 'instant' | 'typing') => void;
  onExportAll?: () => void;
  onDeleteAll?: () => void;
  onClearCache?: () => void;
}

export function SettingsPanel({
  theme = 'light',
  fontSize = 'medium',
  responseSpeed = 'instant',
  onThemeChange,
  onFontSizeChange,
  onResponseSpeedChange,
  onExportAll,
  onDeleteAll,
  onClearCache,
}: SettingsPanelProps) {
  const [showConfirm, setShowConfirm] = useState(false);

  const handleDeleteAll = () => {
    if (showConfirm) {
      onDeleteAll?.();
      setShowConfirm(false);
    } else {
      setShowConfirm(true);
    }
  };

  return (
    <div className="p-4 border-t border-slate-200">
      <div className="text-xs font-semibold text-slate-500 mb-3">설정</div>

      <div className="space-y-3 mb-4">
        <div>
          <div className="text-xs text-slate-600 mb-1">테마</div>
          <div className="flex gap-2">
            {(['light', 'dark', 'auto'] as const).map((t) => (
              <button
                key={t}
                onClick={() => onThemeChange?.(t)}
                className={`px-3 py-1 text-xs rounded-lg transition-colors ${
                  theme === t
                    ? 'bg-blue-100 text-blue-700 font-medium'
                    : 'bg-slate-100 text-slate-700 hover:bg-slate-200'
                }`}
              >
                {t === 'light' ? '라이트' : t === 'dark' ? '다크' : '자동'}
              </button>
            ))}
          </div>
        </div>

        <div>
          <div className="text-xs text-slate-600 mb-1">글꼴 크기</div>
          <div className="flex gap-2">
            {(['small', 'medium', 'large'] as const).map((size) => (
              <button
                key={size}
                onClick={() => onFontSizeChange?.(size)}
                className={`px-3 py-1 text-xs rounded-lg transition-colors ${
                  fontSize === size
                    ? 'bg-blue-100 text-blue-700 font-medium'
                    : 'bg-slate-100 text-slate-700 hover:bg-slate-200'
                }`}
              >
                {size === 'small' ? '작게' : size === 'medium' ? '보통' : '크게'}
              </button>
            ))}
          </div>
        </div>

        <div>
          <div className="text-xs text-slate-600 mb-1">답변 속도</div>
          <div className="flex gap-2">
            {(['instant', 'typing'] as const).map((speed) => (
              <button
                key={speed}
                onClick={() => onResponseSpeedChange?.(speed)}
                className={`px-3 py-1 text-xs rounded-lg transition-colors ${
                  responseSpeed === speed
                    ? 'bg-blue-100 text-blue-700 font-medium'
                    : 'bg-slate-100 text-slate-700 hover:bg-slate-200'
                }`}
              >
                {speed === 'instant' ? '즉시 표시' : '타이핑 효과'}
              </button>
            ))}
          </div>
        </div>
      </div>

      <div className="space-y-2">
        <button
          onClick={onExportAll}
          className="w-full px-3 py-2 text-sm text-slate-700 bg-slate-100 hover:bg-slate-200 rounded-lg transition-colors"
        >
          전체 히스토리 내보내기
        </button>
        <button
          onClick={handleDeleteAll}
          className={`w-full px-3 py-2 text-sm rounded-lg transition-colors ${
            showConfirm
              ? 'bg-red-600 text-white hover:bg-red-700'
              : 'text-red-600 bg-red-50 hover:bg-red-100'
          }`}
        >
          {showConfirm ? '정말 삭제하시겠습니까?' : '전체 대화 삭제'}
        </button>
        <button
          onClick={onClearCache}
          className="w-full px-3 py-2 text-sm text-slate-700 bg-slate-100 hover:bg-slate-200 rounded-lg transition-colors"
        >
          로컬 캐시 지우기
        </button>
      </div>
    </div>
  );
}

