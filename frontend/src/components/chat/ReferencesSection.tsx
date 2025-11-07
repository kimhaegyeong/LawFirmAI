/**
 * 참고자료 섹션 컴포넌트
 */
import { ChevronDown, ChevronUp, ExternalLink } from 'lucide-react';
import { useState } from 'react';
import { Modal } from '../common/Modal';

interface Reference {
  id: string;
  type: 'law' | 'case' | 'regulation';
  title: string;
  content: string;
  link?: string;
}

interface ReferencesSectionProps {
  references?: string[];
  legalReferences?: string[];
  sources?: string[];
}

export function ReferencesSection({
  references = [],
  legalReferences = [],
  sources = [],
}: ReferencesSectionProps) {
  const [isExpanded, setIsExpanded] = useState(false);
  const [selectedReference, setSelectedReference] = useState<Reference | null>(null);

  const allReferences = [
    ...legalReferences.map((ref, idx) => ({
      id: `legal-${idx}`,
      type: 'law' as const,
      title: `법령: ${ref}`,
      content: ref,
    })),
    ...sources.map((src, idx) => ({
      id: `source-${idx}`,
      type: 'case' as const,
      title: `출처: ${src}`,
      content: src,
    })),
    ...references.map((ref, idx) => ({
      id: `ref-${idx}`,
      type: 'regulation' as const,
      title: `참고: ${ref}`,
      content: ref,
    })),
  ];

  if (allReferences.length === 0) {
    return null;
  }

  return (
    <>
      <div className="mt-4 border-t border-slate-200 pt-4">
        <button
          onClick={() => setIsExpanded(!isExpanded)}
          className="flex items-center gap-2 text-sm text-blue-600 hover:text-blue-700 transition-colors"
        >
          {isExpanded ? (
            <ChevronUp className="w-4 h-4" />
          ) : (
            <ChevronDown className="w-4 h-4" />
          )}
          <span>참고 법령/판례 보기 ({allReferences.length}개)</span>
        </button>

        {isExpanded && (
          <div className="mt-3 space-y-2">
            {allReferences.map((ref, idx) => (
              <div
                key={ref.id}
                className="p-3 bg-slate-50 rounded-lg border border-slate-200"
              >
                <div className="flex items-start justify-between gap-2">
                  <div className="flex-1">
                    <div className="text-xs font-semibold text-blue-600 mb-1">
                      [{idx + 1}] {ref.title}
                    </div>
                    <div className="text-sm text-slate-700">{ref.content}</div>
                  </div>
                  <div className="flex items-center gap-2">
                    {ref.link && (
                      <a
                        href={ref.link}
                        target="_blank"
                        rel="noopener noreferrer"
                        className="text-blue-600 hover:text-blue-700"
                      >
                        <ExternalLink className="w-4 h-4" />
                      </a>
                    )}
                    <button
                      onClick={() => setSelectedReference(ref)}
                      className="text-xs text-blue-600 hover:text-blue-700 px-2 py-1 rounded hover:bg-blue-50"
                    >
                      원문 보기
                    </button>
                  </div>
                </div>
              </div>
            ))}
          </div>
        )}
      </div>

      <Modal
        isOpen={selectedReference !== null}
        onClose={() => setSelectedReference(null)}
        title={selectedReference?.title}
        size="lg"
      >
        {selectedReference && (
          <div className="whitespace-pre-wrap text-sm text-slate-700">
            {selectedReference.content}
          </div>
        )}
      </Modal>
    </>
  );
}

