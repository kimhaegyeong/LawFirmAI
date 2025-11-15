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
    <div className="bg-amber-50 border border-amber-200 rounded-xl p-4 shadow-sm">
      <div className="flex items-center gap-2 mb-3">
        <Lightbulb className="w-4 h-4 text-amber-500" />
        <span className="text-sm font-semibold text-slate-700">연관 질문</span>
      </div>
      <div className="space-y-2">
        {questions.map((question, idx) => (
          <button
            key={idx}
            onClick={() => onQuestionClick?.(question)}
            className="w-full text-left px-3 py-2 text-sm bg-white hover:bg-amber-100 border border-amber-200 hover:border-amber-300 rounded-lg transition-colors text-slate-700 hover:text-amber-700"
          >
            {question}
          </button>
        ))}
      </div>
    </div>
  );
}

