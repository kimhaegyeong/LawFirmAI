/**
 * 참고자료 사이드바 컴포넌트
 * 오른쪽에서 슬라이드되는 사이드바로 참고자료 목록을 표시합니다.
 */
import { useState, useMemo, useEffect } from 'react';
import { X } from 'lucide-react';
import { ReferencesModalContent } from './ReferencesModalContent';
import { ReferenceDetailView } from './ReferenceDetailView';
import type { SourceInfo, LegalReferenceDetail } from '../../types/chat';
import { extractLegalReferencesFromSourcesDetail, getSourcesDetailFromSourcesByType, type SourcesByType } from '../../utils/sourcesParser';

interface ReferencesSidebarProps {
  isOpen: boolean;
  onClose: () => void;
  references?: string[];
  legalReferences?: string[];
  sources?: string[];
  sourcesDetail?: SourceInfo[];
  sourcesByType?: SourcesByType;
  initialSelectedType?: 'all' | 'law' | 'precedent' | 'decision' | 'interpretation' | 'regulation';
  initialSelectedReferenceId?: string | null;
}

export function ReferencesSidebar({
  isOpen,
  onClose,
  references = [],
  legalReferences = [],  // deprecated
  sources = [],
  sourcesDetail = [],
  sourcesByType: propSourcesByType,
  initialSelectedType = 'all',
  initialSelectedReferenceId,
}: ReferencesSidebarProps) {
  const [selectedReference, setSelectedReference] = useState<LegalReferenceDetail | null>(null);
  const [selectedSourceDetail, setSelectedSourceDetail] = useState<SourceInfo | null>(null);

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

  // totalCount 계산 - CompactReferencesBadge와 동일한 로직 사용
  // 실제 표시되는 parsedReferences 개수와 일치하도록 함
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

  // initialSelectedReferenceId가 있으면 해당 참고자료를 찾아서 자동으로 선택
  useEffect(() => {
    if (initialSelectedReferenceId && effectiveSourcesDetail.length > 0 && !selectedReference) {
      const foundDetail = effectiveSourcesDetail.find(detail => {
        const detailId = detail.case_number || 
                        detail.article_no ||
                        detail.decision_number ||
                        detail.interpretation_number ||
                        detail.metadata?.doc_id || 
                        '';
        return detailId === initialSelectedReferenceId || 
               detail.metadata?.doc_id === initialSelectedReferenceId ||
               (detail.case_number && detail.case_number === initialSelectedReferenceId) ||
               (detail.article_no && detail.article_no === initialSelectedReferenceId);
      });
      
      if (foundDetail) {
        const legalRef: LegalReferenceDetail = {
          id: foundDetail.metadata?.doc_id || foundDetail.case_number || foundDetail.article_no || '',
          type: foundDetail.type === 'statute_article' ? 'law' :
                foundDetail.type === 'case_paragraph' ? 'precedent' :
                foundDetail.type === 'decision_paragraph' ? 'decision' :
                foundDetail.type === 'interpretation_paragraph' ? 'interpretation' : 'regulation',
          law_name: foundDetail.statute_name || foundDetail.metadata?.statute_name,
          case_name: foundDetail.case_name || foundDetail.metadata?.casenames,
          case_number: foundDetail.case_number || foundDetail.metadata?.case_number || foundDetail.metadata?.doc_id,
          article_number: foundDetail.article_no ? `제${foundDetail.article_no}조` : undefined,
          decision_number: foundDetail.decision_number || foundDetail.metadata?.decision_number,
          interpretation_number: foundDetail.interpretation_number || foundDetail.metadata?.interpretation_number,
          content: foundDetail.content,
          article_content: foundDetail.content,
          summary: foundDetail.content,
        };
        setSelectedReference(legalRef);
        setSelectedSourceDetail(foundDetail);
      }
    }
  }, [initialSelectedReferenceId, effectiveSourcesDetail, selectedReference]);

  if (!isOpen) {
    return null;
  }

  // 참고자료 클릭 핸들러
  const handleReferenceClick = (reference: LegalReferenceDetail, sourceDetail?: SourceInfo) => {
    setSelectedReference(reference);
    setSelectedSourceDetail(sourceDetail || null);
  };

  // 뒤로 가기 핸들러
  const handleBack = () => {
    setSelectedReference(null);
    setSelectedSourceDetail(null);
  };

  return (
    <>
      {/* 사이드바 */}
      <div
        className={`fixed right-0 top-0 h-full w-full sm:max-w-md bg-white shadow-xl z-50 transform transition-transform duration-300 ease-in-out ${
          isOpen ? 'translate-x-0' : 'translate-x-full'
        }`}
        role="dialog"
        aria-modal="true"
        aria-labelledby="references-sidebar-title"
      >
        <div className="flex flex-col h-full">
          {/* 선택된 참고자료가 있으면 상세 정보 표시, 없으면 목록 표시 */}
          {selectedReference ? (
            <ReferenceDetailView
              reference={selectedReference}
              sourceDetail={selectedSourceDetail || undefined}
              onBack={handleBack}
              onClose={onClose}
            />
          ) : (
            <>
              {/* 헤더 */}
              <div className="flex items-center justify-between p-4 border-b border-slate-200 bg-slate-50">
                <h2 id="references-sidebar-title" className="text-lg font-semibold text-slate-800">
                  참고자료 ({totalCount}개)
                </h2>
                <button
                  onClick={onClose}
                  className="p-1.5 rounded-md hover:bg-slate-200 transition-colors text-slate-600"
                  aria-label="사이드바 닫기"
                >
                  <X className="w-5 h-5" />
                </button>
              </div>

              {/* 본문 */}
              <div className="flex-1 overflow-y-auto p-4">
                <ReferencesModalContent
                  references={references}
                  legalReferences={allLegalRefs}  // 병합된 legalReferences 사용
                  sources={sources}
                  sourcesDetail={effectiveSourcesDetail}
                  sourcesByType={propSourcesByType}
                  initialSelectedType={initialSelectedType}
                  onReferenceClick={handleReferenceClick}
                />
              </div>
            </>
          )}
        </div>
      </div>
    </>
  );
}

