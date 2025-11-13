/**
 * 웰컴 스크린 컴포넌트
 */
import { Scale, MessageSquare, ChevronLeft, AlertCircle } from 'lucide-react';
import { useState, useEffect, useMemo } from 'react';
import { QUICK_QUESTIONS } from '../../utils/constants';

interface WelcomeScreenProps {
  onQuestionClick?: (question: string) => void;
  isAuthenticated?: boolean;
  quotaInfo?: { remaining: number; limit: number } | null;
  onLoginClick?: () => void;
}

const DEFAULT_QUOTA_LIMIT = 3;

export function WelcomeScreen({ onQuestionClick, isAuthenticated = false, quotaInfo = null, onLoginClick }: WelcomeScreenProps) {
  const [selectedCategory, setSelectedCategory] = useState<typeof QUICK_QUESTIONS[number] | null>(null);

  const isQuotaExhausted = useMemo(() => {
    return !isAuthenticated && quotaInfo !== null && quotaInfo.remaining <= 0;
  }, [isAuthenticated, quotaInfo]);

  useEffect(() => {
    if (import.meta.env.DEV) {
      console.log('[WelcomeScreen] Debug:', {
        isAuthenticated,
        quotaInfo,
        isQuotaExhausted,
        remaining: quotaInfo?.remaining,
        limit: quotaInfo?.limit
      });
    }
  }, [isAuthenticated, quotaInfo, isQuotaExhausted]);

  const handleCategoryClick = (category: typeof QUICK_QUESTIONS[number]) => {
    if (!isQuotaExhausted) {
      setSelectedCategory(category);
    }
  };

  const handleQuestionClick = (question: string) => {
    if (!isQuotaExhausted && onQuestionClick) {
      onQuestionClick(question);
    }
  };

  const handleBack = () => {
    setSelectedCategory(null);
  };

  const renderQuotaBanner = () => {
    if (isAuthenticated) return null;

    if (isQuotaExhausted) {
      return (
        <div className="mt-4 px-6 py-4 bg-amber-50 border-2 border-amber-300 rounded-lg inline-block max-w-2xl">
          <div className="flex items-center justify-between gap-3">
            <div className="flex items-center gap-3 flex-1">
              <AlertCircle className="w-5 h-5 text-amber-600 flex-shrink-0" />
              <div className="text-left flex-1">
                <p className="text-sm font-semibold text-amber-900 mb-1">
                  무료 질의 {quotaInfo?.limit || DEFAULT_QUOTA_LIMIT}회를 모두 사용하셨습니다
                </p>
                <p className="text-xs text-amber-700">
                  계속 사용하려면 로그인이 필요합니다. 로그인하면 무제한으로 사용할 수 있습니다.
                </p>
              </div>
            </div>
            {onLoginClick && (
              <button
                onClick={onLoginClick}
                className="px-4 py-2 bg-blue-600 hover:bg-blue-700 text-white text-sm font-medium rounded-lg transition-colors whitespace-nowrap flex-shrink-0"
              >
                로그인
              </button>
            )}
          </div>
        </div>
      );
    }

    if (quotaInfo !== null && quotaInfo.remaining > 0) {
      return (
        <div className="mt-4 px-4 py-2 bg-blue-50 border border-blue-200 rounded-lg inline-block">
          <p className="text-sm text-blue-800">
            <span className="font-semibold">무료로 {quotaInfo.limit}회 체험</span> 가능합니다. 로그인하면 무제한으로 사용할 수 있습니다.
          </p>
        </div>
      );
    }

    return (
      <div className="mt-4 px-4 py-2 bg-blue-50 border border-blue-200 rounded-lg inline-block">
        <p className="text-sm text-blue-800">
          <span className="font-semibold">무료로 {DEFAULT_QUOTA_LIMIT}회 체험</span> 가능합니다. 로그인하면 무제한으로 사용할 수 있습니다.
        </p>
      </div>
    );
  };

  const getButtonStyles = (disabled: boolean) => {
    const baseStyles = 'flex items-center gap-3 p-4 bg-white border-2 rounded-xl transition-all duration-200 text-left group';
    return disabled
      ? `${baseStyles} border-slate-100 opacity-50 cursor-not-allowed`
      : `${baseStyles} border-slate-200 hover:border-blue-400 hover:shadow-md`;
  };

  const getTextStyles = (disabled: boolean, variant: 'title' | 'subtitle' | 'arrow' = 'title') => {
    if (disabled) {
      return variant === 'arrow' ? 'text-slate-300' : 'text-slate-400';
    }
    if (variant === 'title') {
      return 'text-slate-700 group-hover:text-blue-600';
    }
    if (variant === 'subtitle') {
      return 'text-slate-500';
    }
    return 'text-slate-400 group-hover:text-blue-500';
  };

  // 세부 질문 선택 화면
  if (selectedCategory) {
    return (
      <div className="flex flex-col items-center justify-center min-h-full px-6 py-16">
        <div className="w-full max-w-4xl">
          {/* 뒤로가기 버튼 */}
          <button
            onClick={handleBack}
            className="flex items-center gap-2 mb-6 text-slate-600 hover:text-slate-800 transition-colors"
          >
            <ChevronLeft className="w-5 h-5" />
            <span className="text-sm font-medium">뒤로</span>
          </button>

          {/* 카테고리 제목 */}
          <div className="text-center mb-8">
            <div className="inline-flex items-center justify-center w-16 h-16 bg-blue-100 rounded-full mb-4">
              <span className="text-3xl">{selectedCategory.icon}</span>
            </div>
            <h2 className="text-2xl font-bold text-slate-800 mb-2">{selectedCategory.title}</h2>
            <p className="text-slate-600">관련 질문을 선택하세요</p>
          </div>

          {/* 세부 질문 목록 */}
          <div className="space-y-3">
            {selectedCategory.questions.map((item, idx) => (
              <button
                key={idx}
                onClick={() => handleQuestionClick(item.question)}
                disabled={isQuotaExhausted}
                className={`w-full ${getButtonStyles(isQuotaExhausted)}`}
              >
                <div className="flex items-center justify-between">
                  <span className={`text-base font-medium transition-colors ${getTextStyles(isQuotaExhausted, 'title')}`}>
                    {item.title}
                  </span>
                  <span className={`text-sm transition-colors ${getTextStyles(isQuotaExhausted, 'arrow')}`}>
                    →
                  </span>
                </div>
                <p className={`text-sm mt-1 ${getTextStyles(isQuotaExhausted, 'subtitle')}`}>
                  {item.question}
                </p>
              </button>
            ))}
          </div>
        </div>
      </div>
    );
  }

  // 초기 화면: 카테고리 선택
  return (
    <div className="flex flex-col items-center justify-center min-h-full px-6 py-8 py-16">
      <div className="text-center mb-12">
        <div className="inline-flex items-center justify-center w-16 h-16 bg-blue-100 rounded-full mb-4">
          <Scale className="w-8 h-8 text-blue-600" />
        </div>
        <h2 className="text-3xl font-bold text-slate-800 mb-3">법률 AI 어시스턴트</h2>
        <p className="text-slate-600 text-lg">법률 관련 질문에 답변해드립니다</p>
        {renderQuotaBanner()}
      </div>

      <div className="w-full max-w-4xl">
        <div className="flex items-center gap-2 mb-4">
          <MessageSquare className="w-5 h-5 text-amber-500" />
          <h3 className="text-lg font-semibold text-slate-800">빠른 질문</h3>
        </div>
        
        {/* 쿼터 초과 시 안내 메시지 */}
        {isQuotaExhausted && (
          <div className="mb-4 p-3 bg-slate-50 border border-slate-200 rounded-lg">
            <p className="text-sm text-slate-600 text-center">
              빠른 질문을 사용하려면 로그인이 필요합니다.
            </p>
          </div>
        )}
        
        <div className="grid grid-cols-2 gap-3">
          {QUICK_QUESTIONS.map((category, idx) => (
            <button
              key={idx}
              onClick={() => handleCategoryClick(category)}
              disabled={isQuotaExhausted}
              className={getButtonStyles(isQuotaExhausted)}
            >
              <span className="text-2xl">{category.icon}</span>
              <span className={`text-sm font-medium transition-colors ${getTextStyles(isQuotaExhausted, 'title')}`}>
                {category.title}
              </span>
            </button>
          ))}
        </div>
      </div>
    </div>
  );
}

