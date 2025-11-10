/**
 * 익명 사용자 쿼터 표시 컴포넌트
 */
import { AlertCircle, LogIn } from 'lucide-react';

interface QuotaIndicatorProps {
  remaining: number;
  limit: number;
  isAuthenticated: boolean;
  onLoginClick: () => void;
}

export function QuotaIndicator({ remaining, limit, isAuthenticated, onLoginClick }: QuotaIndicatorProps) {
  if (isAuthenticated) {
    return null;
  }

  const isExhausted = remaining === 0;
  const percentage = (remaining / limit) * 100;

  return (
    <div className="bg-white border-t border-slate-200 px-6 py-3 flex-shrink-0">
      <div className="max-w-4xl mx-auto">
        {isExhausted ? (
          <div className="flex items-center justify-between p-3 bg-amber-50 border border-amber-200 rounded-lg">
            <div className="flex items-center gap-2 text-amber-800">
              <AlertCircle className="w-5 h-5" />
              <span className="text-sm font-medium">무료 질의 3회를 모두 사용하셨습니다.</span>
            </div>
            <button
              onClick={onLoginClick}
              className="flex items-center gap-2 px-4 py-2 bg-blue-600 hover:bg-blue-700 text-white text-sm font-medium rounded-lg transition-colors"
            >
              <LogIn className="w-4 h-4" />
              로그인하여 무제한 사용
            </button>
          </div>
        ) : (
          <div className="flex items-center justify-between p-3 bg-blue-50 border border-blue-200 rounded-lg">
            <div className="flex items-center gap-3 flex-1">
              <div className="flex-1">
                <div className="flex items-center justify-between mb-1">
                  <span className="text-sm font-medium text-blue-800">무료 질의 남은 횟수</span>
                  <span className="text-sm font-semibold text-blue-900">{remaining} / {limit}</span>
                </div>
                <div className="w-full bg-blue-200 rounded-full h-2">
                  <div
                    className="bg-blue-600 h-2 rounded-full transition-all duration-300"
                    style={{ width: `${percentage}%` }}
                  />
                </div>
              </div>
            </div>
            <button
              onClick={onLoginClick}
              className="ml-4 px-4 py-2 bg-blue-600 hover:bg-blue-700 text-white text-sm font-medium rounded-lg transition-colors whitespace-nowrap"
            >
              로그인
            </button>
          </div>
        )}
      </div>
    </div>
  );
}

