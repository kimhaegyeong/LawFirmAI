/**
 * 연관 질문 컴포넌트
 */
import { Lightbulb } from 'lucide-react';

interface RelatedQuestionsProps {
  questions?: string[];
  onQuestionClick?: (question: string) => void;
}

export function RelatedQuestions({
  questions = [],
  onQuestionClick,
}: RelatedQuestionsProps) {
  if (questions.length === 0) {
    return null;
  }

  return (
    <div className="mt-4 border-t border-slate-200 pt-4">
      <div className="flex items-center gap-2 mb-3">
        <Lightbulb className="w-4 h-4 text-amber-500" />
        <span className="text-sm font-semibold text-slate-700">연관 질문</span>
      </div>
      <div className="space-y-2">
        {questions.map((question, idx) => (
          <button
            key={idx}
            onClick={() => onQuestionClick?.(question)}
            className="w-full text-left px-3 py-2 text-sm bg-slate-50 hover:bg-blue-50 border border-slate-200 hover:border-blue-300 rounded-lg transition-colors text-slate-700 hover:text-blue-700"
          >
            {question}
          </button>
        ))}
      </div>
    </div>
  );
}

