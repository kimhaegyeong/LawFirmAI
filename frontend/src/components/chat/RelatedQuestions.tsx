/**
 * 연관 질문 컴포넌트
 */

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
    <div>
      <div className="space-y-2">
        {questions.map((question, idx) => (
          <button
            key={idx}
            onClick={() => onQuestionClick?.(question)}
            className="w-full text-left px-3 py-2 text-sm bg-amber-50 hover:bg-amber-100 border border-amber-200 hover:border-amber-300 rounded-lg transition-colors text-slate-700 hover:text-amber-700"
          >
            {question}
          </button>
        ))}
      </div>
    </div>
  );
}

