/**
 * 참고자료 섹션 컴포넌트
 */
import { ChevronDown, ChevronUp, ExternalLink, FileText } from 'lucide-react';
import { useState, useMemo } from 'react';
import { Modal } from '../common/Modal';
import type { SourceInfo } from '../../types/chat';
import { extractLegalReferencesFromSourcesDetail, getSourcesDetailFromSourcesByType } from '../../utils/sourcesParser';
import type { SourcesByType } from '../../utils/sourcesParser';
import { OriginalDocumentViewer } from './OriginalDocumentViewer';

interface Reference {
  id: string;
  type: 'law' | 'case' | 'regulation';
  title: string;
  content: string;
  link?: string;
  originalUrl?: string;
  source?: SourceInfo;
}

interface ReferencesSectionProps {
  references?: string[];
  legalReferences?: string[];  // deprecated
  sources?: string[];  // deprecated
  sourcesDetail?: SourceInfo[];  // deprecated, 하위 호환성
  sourcesByType?: SourcesByType;  // 우선 사용
}

export function ReferencesSection({
  references = [],
  legalReferences = [],  // deprecated
  sources = [],
  sourcesDetail = [],
  sourcesByType: propSourcesByType,
}: ReferencesSectionProps) {
  const [isExpanded, setIsExpanded] = useState(false);
  const [selectedReference, setSelectedReference] = useState<Reference | null>(null);
  const [selectedSourceForOriginal, setSelectedSourceForOriginal] = useState<SourceInfo | null>(null);

  // sourcesByType 우선 사용, 없으면 sourcesDetail에서 재구성
  const effectiveSourcesDetail = useMemo(() => {
    if (propSourcesByType) {
      return getSourcesDetailFromSourcesByType(propSourcesByType);
    }
    return sourcesDetail;
  }, [propSourcesByType, sourcesDetail]);

  // sourcesDetail에서 legal_references 추출 (deprecated용)
  const extractedLegalRefs = useMemo(() => {
    return extractLegalReferencesFromSourcesDetail(effectiveSourcesDetail);
  }, [effectiveSourcesDetail]);
  
  // legalReferences와 extractedLegalRefs 병합 (하위 호환성)
  const allLegalRefs = useMemo(() => {
    return [...new Set([...legalReferences, ...extractedLegalRefs])];
  }, [legalReferences, extractedLegalRefs]);
  
  // sourcesDetail을 Reference로 변환
  const sourcesDetailReferences = useMemo(() => {
    return effectiveSourcesDetail.map((detail, idx): Reference => {
      let type: 'law' | 'case' | 'regulation' = 'regulation';
      let title = '';
      const content = detail.content || detail.name || '';
      
      if (detail.type === 'statute_article') {
        type = 'law';
        const lawName = detail.statute_name || detail.metadata?.statute_name || detail.name || '';
        const articleNo = detail.article_no || detail.metadata?.article_no || '';
        title = `법령: ${lawName}${articleNo ? ` 제${articleNo}조` : ''}`;
      } else if (detail.type === 'case_paragraph') {
        type = 'case';
        const caseName = detail.case_name || detail.metadata?.case_name || detail.name || '';
        const caseNumber = detail.case_number || detail.metadata?.case_number || '';
        title = `판례: ${caseName}${caseNumber ? ` (${caseNumber})` : ''}`;
      } else if (detail.type === 'decision_paragraph') {
        type = 'case';
        const org = detail.org || detail.metadata?.org || '';
        const decisionNumber = detail.decision_number || detail.metadata?.decision_number || '';
        title = `결정례: ${org}${decisionNumber ? ` (${decisionNumber})` : ''}`;
      } else if (detail.type === 'interpretation_paragraph') {
        type = 'case';
        const titleText = detail.title || detail.metadata?.title || detail.name || '';
        const org = detail.org || detail.metadata?.org || '';
        title = `해석례: ${titleText}${org ? ` (${org})` : ''}`;
      } else {
        title = `참고: ${detail.name || '참고자료'}`;
      }
      
      return {
        id: `detail-${idx}`,
        type,
        title,
        content,
        link: detail.url,
        originalUrl: detail.original_url,
        source: detail,
      };
    });
  }, [effectiveSourcesDetail]);

  const allReferences: Reference[] = useMemo(() => {
    const seen = new Set<string>();
    const result: Reference[] = [];
    
    // sourcesDetail 우선 처리
    sourcesDetailReferences.forEach(ref => {
      const key = ref.id;
      if (!seen.has(key)) {
        seen.add(key);
        result.push(ref);
      }
    });
    
    // legal_references 처리 (sources_detail과 중복 제외)
    allLegalRefs.forEach((ref, idx) => {
      const matched = sourcesDetail.find(detail => {
        const detailName = detail.name || detail.content || '';
        return detailName.includes(ref) || ref.includes(detailName);
      });
      
      if (!matched && !seen.has(ref)) {
        seen.add(ref);
        result.push({
          id: `legal-${idx}`,
          type: 'law' as const,
          title: `법령: ${ref}`,
          content: ref,
          link: undefined,
        });
      }
    });
    
    // sources 처리 (sources_detail과 중복 제외)
    sources.forEach((src, idx) => {
      const matched = sourcesDetail.find(detail => {
        const detailName = detail.name || detail.content || '';
        return detailName.includes(src) || src.includes(detailName);
      });
      
      if (!matched && !seen.has(src)) {
        seen.add(src);
        result.push({
          id: `source-${idx}`,
          type: 'case' as const,
          title: `출처: ${src}`,
          content: src,
          link: undefined,
        });
      }
    });
    
    // references 처리
    references.forEach((ref, idx) => {
      if (!seen.has(ref)) {
        seen.add(ref);
        result.push({
          id: `ref-${idx}`,
          type: 'regulation' as const,
          title: `참고: ${ref}`,
          content: ref,
          link: undefined,
        });
      }
    });
    
    return result;
  }, [sourcesDetailReferences, allLegalRefs, sources, references, sourcesDetail]);

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
                    {ref.originalUrl && ref.source ? (
                      <button
                        onClick={() => setSelectedSourceForOriginal(ref.source!)}
                        className="text-xs text-blue-600 hover:text-blue-700 px-2 py-1 rounded hover:bg-blue-50 flex items-center gap-1"
                      >
                        <FileText className="w-3 h-3" />
                        원문 보기
                      </button>
                    ) : (
                      <button
                        onClick={() => setSelectedReference(ref)}
                        className="text-xs text-blue-600 hover:text-blue-700 px-2 py-1 rounded hover:bg-blue-50"
                      >
                        상세 보기
                      </button>
                    )}
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

      {selectedSourceForOriginal && (
        <OriginalDocumentViewer
          source={selectedSourceForOriginal}
          onClose={() => setSelectedSourceForOriginal(null)}
        />
      )}
    </>
  );
}

