/**
 * 메시지 참고자료 섹션 컴포넌트
 * 답변과 별도로 표시되는 참고자료 섹션
 */
import { FileText, ChevronDown, ChevronUp } from 'lucide-react';
import { useState, useMemo } from 'react';
import { CompactReferencesBadge } from './CompactReferencesBadge';
import type { SourceInfo } from '../../types/chat';
import { extractLegalReferencesFromSourcesDetail, getSourcesDetailFromSourcesByType, type SourcesByType } from '../../utils/sourcesParser';

type ReferenceType = 'all' | 'law' | 'precedent' | 'decision' | 'interpretation' | 'regulation';

interface MessageReferencesSectionProps {
  references?: string[];
  legalReferences?: string[];  // deprecated
  sources?: string[];  // deprecated
  sourcesDetail?: SourceInfo[];  // deprecated, 하위 호환성
  sourcesByType?: SourcesByType;  // 우선 사용
  onOpenSidebar?: (selectedType: ReferenceType) => void;
  defaultExpanded?: boolean;
}

export function MessageReferencesSection({
  references = [],
  legalReferences = [],  // deprecated
  sources = [],
  sourcesDetail = [],
  sourcesByType: propSourcesByType,
  onOpenSidebar,
  defaultExpanded = false,
}: MessageReferencesSectionProps) {
  const [isExpanded, setIsExpanded] = useState(defaultExpanded);

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

  // totalCount 계산은 CompactReferencesBadge의 parsedReferences를 기반으로 계산
  // 실제 표시되는 항목 수와 일치하도록 하기 위해 CompactReferencesBadge와 동일한 로직 사용
  const totalCount = useMemo(() => {
    const seen = new Set<string>();
    const all: string[] = [];
    
    // effectiveSourcesDetail 우선 처리 (CompactReferencesBadge와 동일한 키 생성 로직)
    effectiveSourcesDetail.forEach((detail, idx) => {
      let title = '';
      
      if (detail.type === 'statute_article') {
        const lawName = detail.statute_name || detail.metadata?.statute_name || detail.name || '';
        title = lawName || '법령';
      } else if (detail.type === 'case_paragraph') {
        const caseName = detail.case_name || 
                        detail.metadata?.case_name ||
                        detail.metadata?.casenames || 
                        detail.name || '';
        const caseNumber = detail.case_number || 
                          detail.metadata?.case_number ||
                          detail.metadata?.doc_id || '';
        title = (caseName || caseNumber || '판례') as string;
      } else if (detail.type === 'decision_paragraph') {
        const org = detail.org || detail.metadata?.org || '';
        const decisionNumber = detail.decision_number || 
                              detail.metadata?.decision_number ||
                              detail.metadata?.doc_id || '';
        title = (org || decisionNumber || '결정례') as string;
      } else if (detail.type === 'interpretation_paragraph') {
        const titleText = detail.title || detail.metadata?.title || detail.name || '';
        title = titleText || '해석례';
      } else {
        title = detail.name || detail.content || '참고자료';
      }
      
      // CompactReferencesBadge와 동일한 키 생성 로직
      const key = detail.case_number || 
                 detail.article_no ||
                 detail.decision_number ||
                 detail.interpretation_number ||
                 detail.metadata?.doc_id || 
                 title || 
                 `detail-${idx}`;
      
      if (!seen.has(key)) {
        seen.add(key);
        all.push(key);
      }
    });
    
    // legalReferences 처리 (sources_detail과 중복 제외, CompactReferencesBadge와 동일한 로직)
    allLegalRefs.forEach(ref => {
      const matched = effectiveSourcesDetail.find(detail => {
        const detailName = detail.name || detail.content || '';
        return detailName.includes(ref) || ref.includes(detailName);
      });
      
      if (!matched && !seen.has(ref)) {
        seen.add(ref);
        all.push(ref);
      }
    });
    
    // sources 처리 (sources_detail과 중복 제외, CompactReferencesBadge와 동일한 로직)
    sources.forEach(src => {
      const matched = effectiveSourcesDetail.find(detail => {
        const detailName = detail.name || detail.content || '';
        return detailName.includes(src) || src.includes(detailName);
      });
      
      if (!matched && !seen.has(src)) {
        seen.add(src);
        all.push(src);
      }
    });
    
    // references 처리
    references.forEach(ref => {
      if (!seen.has(ref)) {
        seen.add(ref);
        all.push(ref);
      }
    });
    
    return all.length;
  }, [references, allLegalRefs, sources, effectiveSourcesDetail]);

  if (totalCount === 0) {
    return null;
  }

  return (
    <div className="border-t-2 border-blue-200 pt-4">
      <div className="flex items-center justify-between mb-3">
        <div className="flex items-center gap-2">
          <FileText className="w-5 h-5 text-blue-600 flex-shrink-0" />
          <h3 className="text-sm font-semibold text-slate-700">참고자료</h3>
          <span className="text-xs text-slate-500 bg-blue-50 px-2 py-0.5 rounded-full">
            {totalCount}개
          </span>
        </div>
        <button
          onClick={() => setIsExpanded(!isExpanded)}
          className="flex items-center gap-1 text-xs text-slate-500 hover:text-slate-700 transition-colors p-1 rounded hover:bg-slate-100"
          aria-label={isExpanded ? '참고자료 접기' : '참고자료 펼치기'}
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

      {isExpanded && (
        <div className="animate-fadeIn">
          <CompactReferencesBadge
            references={references}
            legalReferences={allLegalRefs}  // 병합된 legalReferences 사용
            sources={sources}
            sourcesDetail={effectiveSourcesDetail}
            sourcesByType={propSourcesByType}
            onOpenSidebar={onOpenSidebar}
          />
        </div>
      )}
    </div>
  );
}

