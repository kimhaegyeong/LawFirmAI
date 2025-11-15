/**
 * Toast 알림 컴포넌트
 */
import { useEffect, useState } from 'react';
import { X, AlertCircle, CheckCircle, Info, AlertTriangle } from 'lucide-react';

export interface ToastProps {
  message: string;
  type: 'error' | 'success' | 'info' | 'warning';
  onClose: () => void;
  duration?: number;
  action?: {
    label: string;
    onClick: () => void;
  };
}

export function Toast({ 
  message, 
  type, 
  onClose, 
  duration = 5000,
  action 
}: ToastProps) {
  const [isVisible, setIsVisible] = useState(true);

  useEffect(() => {
    const timer = setTimeout(() => {
      setIsVisible(false);
      setTimeout(onClose, 300); // 애니메이션 시간
    }, duration);

    return () => clearTimeout(timer);
  }, [duration, onClose]);

  const icons = {
    error: AlertCircle,
    success: CheckCircle,
    info: Info,
    warning: AlertTriangle,
  };

  const colors = {
    error: 'bg-red-50 border-red-200 text-red-800',
    success: 'bg-green-50 border-green-200 text-green-800',
    info: 'bg-blue-50 border-blue-200 text-blue-800',
    warning: 'bg-yellow-50 border-yellow-200 text-yellow-800',
  };

  const iconColors = {
    error: 'text-red-600',
    success: 'text-green-600',
    info: 'text-blue-600',
    warning: 'text-yellow-600',
  };

  // eslint-disable-next-line security/detect-object-injection
  const Icon = icons[type];

  if (!isVisible) return null;

  return (
    <div 
      className={`fixed top-4 right-4 z-50 min-w-[300px] max-w-[500px] rounded-lg border p-4 shadow-lg transition-all ${
        isVisible ? 'opacity-100 translate-y-0' : 'opacity-0 -translate-y-2'
        // eslint-disable-next-line security/detect-object-injection
      } ${colors[type]}`}
    >
      <div className="flex items-start gap-3">
        <Icon className={`w-5 h-5 mt-0.5 flex-shrink-0 ${
          // eslint-disable-next-line security/detect-object-injection
          iconColors[type]
        }`} />
        <div className="flex-1">
          <p className="text-sm font-medium">{message}</p>
          {action && (
            <button
              onClick={action.onClick}
              className="mt-2 text-sm font-semibold underline hover:no-underline"
            >
              {action.label}
            </button>
          )}
        </div>
        <button
          onClick={() => {
            setIsVisible(false);
            setTimeout(onClose, 300);
          }}
          className="flex-shrink-0 text-gray-400 hover:text-gray-600 transition-colors"
        >
          <X className="w-4 h-4" />
        </button>
      </div>
    </div>
  );
}

