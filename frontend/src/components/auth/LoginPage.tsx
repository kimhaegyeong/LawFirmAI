/**
 * 로그인 페이지 컴포넌트
 */
import { useEffect, useState, useRef } from 'react';
import { Scale, LogIn, AlertCircle } from 'lucide-react';
import { useAuth } from '../../hooks/useAuth';
import { LoadingSpinner } from '../common/LoadingSpinner';
import { isAuthenticated as checkAuthenticated } from '../../services/authService';
import logger from '../../utils/logger';

export function LoginPage() {
  const { login, handleCallback, isLoading, error, isAuthenticated } = useAuth();
  const [isProcessing, setIsProcessing] = useState(false);
  const [callbackCompleted, setCallbackCompleted] = useState(false);
  const [urlError, setUrlError] = useState<string | null>(null);
  const hasProcessedCallback = useRef(false);

  useEffect(() => {
    if (hasProcessedCallback.current) {
      logger.debug('LoginPage: Callback already processed, skipping...');
      return;
    }

    const urlParams = new URLSearchParams(window.location.search);
    const code = urlParams.get('code');
    const state = urlParams.get('state');
    const errorParam = urlParams.get('error');

    if (errorParam) {
      logger.error('OAuth error in URL:', errorParam);
      setUrlError(errorParam);
      window.history.replaceState({}, document.title, window.location.pathname);
      return;
    }

    if (code && state) {
      const processedCodeKey = `oauth2_processed_code_${code}`;
      const isProcessed = sessionStorage.getItem(processedCodeKey);
      
      if (isProcessed === 'true') {
        logger.warn('LoginPage: 인증 코드가 이미 처리되었습니다. 중복 호출을 방지합니다.');
        hasProcessedCallback.current = true;
        return;
      }
      
      hasProcessedCallback.current = true;
      setIsProcessing(true);
      
      logger.info('LoginPage: OAuth callback 처리 시작', { code: code.substring(0, 10), state: state.substring(0, 10) });
      
      handleCallback(code, state)
        .then(() => {
          logger.info('LoginPage: OAuth callback 처리 성공');
          window.history.replaceState({}, document.title, window.location.pathname);
          setCallbackCompleted(true);
        })
        .catch((err) => {
          logger.warn('LoginPage: OAuth callback failed, continuing as guest:', err);
          setCallbackCompleted(false);
          hasProcessedCallback.current = false;
          window.history.replaceState({}, document.title, window.location.pathname);
          setTimeout(() => {
            window.location.replace('/');
          }, 300);
        })
        .finally(() => {
          setIsProcessing(false);
        });
    }
  }, [handleCallback]);

  useEffect(() => {
    if (callbackCompleted && isAuthenticated && !isLoading && !isProcessing) {
      if (checkAuthenticated()) {
        setTimeout(() => {
          window.location.replace('/');
        }, 300);
      } else {
        logger.error('Token not found before redirect!');
      }
    }
  }, [callbackCompleted, isAuthenticated, isLoading, isProcessing]);

  const handleLogin = () => {
    login();
  };

  return (
    <div className="flex flex-col items-center justify-center min-h-screen px-6 py-16 bg-gradient-to-br from-blue-50 to-slate-50">
      <div className="w-full max-w-md">
        <div className="bg-white rounded-2xl shadow-lg p-8">
          <div className="text-center mb-8">
            <div className="inline-flex items-center justify-center w-16 h-16 bg-blue-100 rounded-full mb-4">
              <Scale className="w-8 h-8 text-blue-600" />
            </div>
            <h2 className="text-3xl font-bold text-slate-800 mb-2">법률 AI 어시스턴트</h2>
            <p className="text-slate-600">로그인하여 서비스를 이용하세요</p>
          </div>

          {(error || urlError) && (
            <div className="mb-6 p-4 bg-red-50 border border-red-200 rounded-lg flex items-start gap-3">
              <AlertCircle className="w-5 h-5 text-red-600 flex-shrink-0 mt-0.5" />
              <div className="flex-1">
                <p className="text-sm font-medium text-red-800">로그인 실패</p>
                <p className="text-sm text-red-600 mt-1">{error?.message || urlError}</p>
              </div>
            </div>
          )}

          {(isLoading || isProcessing) && (
            <div className="mb-6 flex items-center justify-center gap-3 text-slate-600">
              <LoadingSpinner size="sm" />
              <span className="text-sm">로그인 처리 중...</span>
            </div>
          )}

          <button
            onClick={handleLogin}
            disabled={isLoading || isProcessing}
            className="w-full flex items-center justify-center gap-3 px-6 py-3 bg-white border-2 border-slate-300 hover:border-blue-500 hover:bg-blue-50 rounded-lg transition-all duration-200 disabled:opacity-50 disabled:cursor-not-allowed group"
          >
            <svg className="w-5 h-5" viewBox="0 0 24 24">
              <path
                fill="#4285F4"
                d="M22.56 12.25c0-.78-.07-1.53-.2-2.25H12v4.26h5.92c-.26 1.37-1.04 2.53-2.21 3.31v2.77h3.57c2.08-1.92 3.28-4.74 3.28-8.09z"
              />
              <path
                fill="#34A853"
                d="M12 23c2.97 0 5.46-.98 7.28-2.66l-3.57-2.77c-.98.66-2.23 1.06-3.71 1.06-2.86 0-5.29-1.93-6.16-4.53H2.18v2.84C3.99 20.53 7.7 23 12 23z"
              />
              <path
                fill="#FBBC05"
                d="M5.84 14.09c-.22-.66-.35-1.36-.35-2.09s.13-1.43.35-2.09V7.07H2.18C1.43 8.55 1 10.22 1 12s.43 3.45 1.18 4.93l2.85-2.22.81-.62z"
              />
              <path
                fill="#EA4335"
                d="M12 5.38c1.62 0 3.06.56 4.21 1.64l3.15-3.15C17.45 2.09 14.97 1 12 1 7.7 1 3.99 3.47 2.18 7.07l3.66 2.84c.87-2.6 3.3-4.53 6.16-4.53z"
              />
            </svg>
            <LogIn className="w-5 h-5 text-slate-600 group-hover:text-blue-600 transition-colors" />
            <span className="text-base font-medium text-slate-700 group-hover:text-blue-600 transition-colors">
              Google로 로그인
            </span>
          </button>

          <div className="mt-6 text-center">
            <p className="text-sm text-slate-500">
              로그인하면 무료 질의 제한 없이 서비스를 이용할 수 있습니다
            </p>
          </div>
        </div>
      </div>
    </div>
  );
}

