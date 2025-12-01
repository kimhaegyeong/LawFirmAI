/**
 * 메시지 연관질문 섹션 컴포넌트
 * 답변과 별도로 표시되는 연관질문 섹션
 */
import { Lightbulb, ChevronDown, ChevronUp } from 'lucide-react';
import { useState } from 'react';
import { RelatedQuestions } from './RelatedQuestions';

interface MessageRelatedQuestionsSectionProps {
  questions?: string[];
  onQuestionClick?: (question: string) => void;
  defaultExpanded?: boolean;
}

export function MessageRelatedQuestionsSection({
  questions = [],
  onQuestionClick,
  defaultExpanded = false,
}: MessageRelatedQuestionsSectionProps) {
  const [isExpanded, setIsExpanded] = useState(defaultExpanded);

  const questionsArray = Array.isArray(questions)
    ? questions.filter((q): q is string => typeof q === 'string' && q.trim().length > 0)
    : [];

  if (questionsArray.length === 0) {
    return null;
  }

  return (
    <div className="border-t-2 border-amber-200 pt-4">
      <div className="flex items-center justify-between mb-3">
        <div className="flex items-center gap-2">
          <Lightbulb className="w-5 h-5 text-amber-600 flex-shrink-0" />
          <h3 className="text-sm font-semibold text-slate-700">연관 질문</h3>
          <span className="text-xs text-slate-500 bg-amber-50 px-2 py-0.5 rounded-full">
            {questionsArray.length}개
          </span>
        </div>
        <button
          onClick={() => setIsExpanded(!isExpanded)}
          className="flex items-center gap-1 text-xs text-slate-500 hover:text-slate-700 transition-colors p-1 rounded hover:bg-slate-100"
          aria-label={isExpanded ? '연관질문 접기' : '연관질문 펼치기'}
        >
          {isExpanded ? (
            <>
              <ChevronUp className="w-4 h-4" />
              <span>접기</span>
            </>
          ) : (
            <>
              <ChevronDown className="w-4 h-4" />
              <span>펼치기</span>
            </>
          )}
        </button>
      </div>

      <div className={`overflow-hidden transition-all duration-300 ${isExpanded ? 'max-h-[5000px] opacity-100' : 'max-h-0 opacity-0'}`}>
        {isExpanded && (
          <div className="animate-fadeIn">
            <RelatedQuestions
              questions={questionsArray}
              onQuestionClick={onQuestionClick}
            />
          </div>
        )}
      </div>
    </div>
  );
}

