/**
 * 에러 메시지 컴포넌트
 */
import { AlertCircle, RefreshCw, XCircle } from 'lucide-react';
import { StreamError } from '../../types/error';

interface ErrorMessageProps {
  error: StreamError;
  onRetry?: () => void;
  onDismiss?: () => void;
}

export function ErrorMessage({ error, onRetry, onDismiss }: ErrorMessageProps) {
  return (
    <div className="rounded-lg border border-red-200 bg-red-50 p-4 my-2">
      <div className="flex items-start gap-3">
        <AlertCircle className="w-5 h-5 text-red-600 flex-shrink-0 mt-0.5" />
        <div className="flex-1">
          <p className="text-sm font-medium text-red-800">{error.message}</p>
          {error.originalError && import.meta.env.DEV && (
            <p className="text-xs text-red-600 mt-1 font-mono break-all">
              {error.originalError.message}
            </p>
          )}
          {error.canRetry && onRetry && (
            <button
              onClick={onRetry}
              className="mt-2 flex items-center gap-1 text-sm font-semibold text-red-700 hover:text-red-900 transition-colors"
            >
              <RefreshCw className="w-4 h-4" />
              다시 시도
            </button>
          )}
        </div>
        {onDismiss && (
          <button
            onClick={onDismiss}
            className="flex-shrink-0 text-red-400 hover:text-red-600 transition-colors"
          >
            <XCircle className="w-4 h-4" />
          </button>
        )}
      </div>
    </div>
  );
}

