/**
 * 컴팩트 참고자료 배지 컴포넌트
 * 하이브리드 방식: 인라인 미리보기 + 확장 섹션 + 사이드바
 */
import { FileText, Scale, Bookmark } from 'lucide-react';
import { useMemo } from 'react';

import type { SourceInfo } from '../../types/chat';
import { getSourcesDetailFromSourcesByType, extractLegalReferencesFromSourcesDetail, type SourcesByType } from '../../utils/sourcesParser';

type ReferenceType = 'all' | 'law' | 'precedent' | 'decision' | 'interpretation' | 'regulation';

interface CompactReferencesBadgeProps {
  references?: string[];
  legalReferences?: string[];  // deprecated
  sources?: string[];  // deprecated
  sourcesDetail?: SourceInfo[];  // deprecated, 하위 호환성
  sourcesByType?: SourcesByType;  // 우선 사용
  onOpenSidebar?: (selectedType: ReferenceType) => void;
  onReferenceClick?: (reference: { id: string; type: ReferenceType; title: string; subtitle?: string; metadata?: Record<string, unknown>; sourceDetail?: SourceInfo }, sourceDetail?: SourceInfo) => void;
}

export function CompactReferencesBadge({
  references = [],
  legalReferences = [],
  sources = [],
  sourcesDetail = [],
  sourcesByType: propSourcesByType,
  onOpenSidebar,
  onReferenceClick,
}: CompactReferencesBadgeProps) {

  // sources_by_type을 직접 사용하여 참고자료 표시 (sources 이벤트 데이터와 일치)
  const effectiveSourcesDetail = useMemo(() => {
    if (propSourcesByType) {
      // sources_by_type을 직접 사용 (sources 이벤트 데이터와 일치)
      return getSourcesDetailFromSourcesByType(propSourcesByType);
    }
    return sourcesDetail;
  }, [propSourcesByType, sourcesDetail]);

  // sourcesDetail에서 legal_references 추출 (deprecated용)
  const extractedLegalRefStrings = useMemo(() => {
    return extractLegalReferencesFromSourcesDetail(effectiveSourcesDetail);
  }, [effectiveSourcesDetail]);
  
  // content에서 법령 참조 추출 제거 (sources_by_type에 이미 포함되어 있으므로 불필요)
  // sources 이벤트의 sources_by_type을 직접 사용하므로 추가 추출 불필요
  
  // legalReferences와 extractedLegalRefStrings 병합 (하위 호환성)
  const allLegalRefs = useMemo(() => {
    return [...new Set([...legalReferences, ...extractedLegalRefStrings])];
  }, [legalReferences, extractedLegalRefStrings]);

  // 참고자료 파싱 - sources_by_type을 직접 사용 (sources 이벤트 데이터와 일치)
  const parsedReferences = useMemo(() => {
    const all: Array<{ 
      id: string; 
      type: 'law' | 'precedent' | 'decision' | 'interpretation' | 'regulation'; 
      title: string;
      subtitle?: string;
      metadata?: Record<string, unknown>;
      sourceDetail?: SourceInfo;
    }> = [];
    const seen = new Set<string>();
    
    // effectiveSourcesDetail 우선 처리 (sources_by_type에서 재구성된 데이터)
    effectiveSourcesDetail.forEach((detail, idx) => {
      let type: 'law' | 'precedent' | 'decision' | 'interpretation' | 'regulation' = 'regulation';
      let title: string = '';
      let subtitle: string = '';
      let metadata: Record<string, unknown> = detail.metadata || {};
      
      if (detail.type === 'statute_article') {
        type = 'law';
        const lawName = detail.statute_name || detail.metadata?.statute_name || detail.name || '';
        const articleNo = detail.article_no || detail.metadata?.article_no;
        title = lawName || '법령';
        subtitle = articleNo ? `제${articleNo}조` : '';
        
        // 판례에서 추출된 법령인지 확인
        const sourceFrom = detail.source_from || detail.metadata?.source_from;
        if (sourceFrom === 'case_paragraph' || sourceFrom === 'decision_paragraph' || sourceFrom === 'interpretation_paragraph') {
          const updatedMetadata = {
            ...metadata,
            isFromPrecedent: true,
            sourceFrom: sourceFrom,
            sourceDocId: detail.source_doc_id || detail.metadata?.source_doc_id
          };
          metadata = updatedMetadata;
        }
      } else if (detail.type === 'case_paragraph') {
        type = 'precedent';
        // 사건명 추출 (다양한 필드명 지원)
        const caseName = detail.case_name || 
                        detail.metadata?.case_name ||
                        detail.metadata?.casenames || 
                        detail.name || '';
        // 사건번호 추출
        const caseNumber = detail.case_number || 
                          detail.metadata?.case_number ||
                          detail.metadata?.doc_id || '';
        // 법원 추출
        const court = detail.court || detail.metadata?.court || '';
        // 판결일 추출 (다양한 필드명 지원)
        const decisionDate = detail.decision_date || 
                            detail.metadata?.decision_date ||
                            detail.metadata?.announce_date || '';
        
        title = (caseName || caseNumber || '판례') as string;
        subtitle = [caseNumber, court, decisionDate].filter(Boolean).join(' · ');
      } else if (detail.type === 'decision_paragraph') {
        type = 'decision';
        const org = detail.org || detail.metadata?.org || '';
        const decisionNumber = detail.decision_number || 
                              detail.metadata?.decision_number ||
                              detail.metadata?.doc_id || '';
        const decisionDate = detail.decision_date || 
                           detail.metadata?.decision_date || '';
        
        title = (org || decisionNumber || '결정례') as string;
        subtitle = [decisionNumber, decisionDate].filter(Boolean).join(' · ');
      } else if (detail.type === 'interpretation_paragraph') {
        type = 'interpretation';
        const titleText = detail.title || detail.metadata?.title || detail.name || '';
        const org = detail.org || detail.metadata?.org || '';
        const responseDate = detail.response_date || 
                           detail.metadata?.response_date || '';
        
        title = titleText || '해석례';
        subtitle = [org, responseDate].filter(Boolean).join(' · ');
      } else if (detail.type === 'regulation_paragraph') {
        type = 'regulation';
        title = detail.name || detail.title || detail.content || '기타 참고자료';
        subtitle = detail.metadata?.doc_id || '';
      } else {
        title = detail.name || detail.content || '참고자료';
      }
      
      // 중복 체크용 키 생성 (ReferencesModalContent와 동일한 로직)
      // 판례: case_number 우선 사용 (같은 사건번호는 하나로 통합)
      // 법령: law_name + article_number 조합
      // 결정례: decision_number 우선
      // 해석례: interpretation_number 우선
      let key: string;
      
      if (type === 'precedent' && detail.case_number) {
        // 판례는 case_number로 중복 제거 (같은 사건번호는 하나로 통합)
        key = `precedent-${detail.case_number}`;
      } else if (type === 'law' && detail.statute_name && detail.article_no) {
        // 법령은 statute_name + article_no 조합
        key = `law-${detail.statute_name}-${detail.article_no}`;
      } else if (type === 'decision' && detail.decision_number) {
        // 결정례는 decision_number로 중복 제거
        key = `decision-${detail.decision_number}`;
      } else if (type === 'interpretation' && detail.interpretation_number) {
        // 해석례는 interpretation_number로 중복 제거
        key = `interpretation-${detail.interpretation_number}`;
      } else {
        // 기타: doc_id > case_number > article_number > law_name > content 순서
        const docId = detail.metadata?.doc_id || detail.case_number;
        key = docId || 
             detail.case_number || 
             (detail.article_no && detail.statute_name ? `${detail.statute_name}-${detail.article_no}` : undefined) ||
             detail.article_no ||
             detail.decision_number ||
             detail.interpretation_number ||
             detail.statute_name ||
             title || 
             `detail-${idx}`;
      }
      
      if (!seen.has(key)) {
        seen.add(key);
        all.push({
          id: `detail-${idx}`,
          type,
          title,
          subtitle,
          metadata: detail.metadata,
          sourceDetail: detail,
        });
      }
    });

    // sources_by_type이 있으면 sources와 references 배열은 무시 (중복 방지)
    const shouldUseStringRefs = !propSourcesByType || effectiveSourcesDetail.length === 0;
    
    if (shouldUseStringRefs) {
      // legal_references 처리 (sources_detail과 중복 제외, deprecated)
      allLegalRefs.forEach((ref, idx) => {
        const matched = effectiveSourcesDetail.find(detail => {
          const detailName = detail.name || detail.content || '';
          return detailName.includes(ref) || ref.includes(detailName);
        });
        
        if (!matched && !seen.has(ref)) {
          seen.add(ref);
          all.push({
            id: `legal-${idx}`,
            type: 'law',
            title: ref,
          });
        }
      });

      // sources 처리 (sources_detail과 중복 제외)
      sources.forEach((src, idx) => {
        const matched = effectiveSourcesDetail.find(detail => {
          const detailName = detail.name || detail.content || '';
          return detailName.includes(src) || src.includes(detailName);
        });
        
        if (!matched && !seen.has(src)) {
          seen.add(src);
          const isPrecedent = /대법원|고등법원|지방법원|판결|선고|\d{4}[가-힣]\d+/.test(src);
          all.push({
            id: `source-${idx}`,
            type: isPrecedent ? 'precedent' : 'law',
            title: src,
          });
        }
      });

      // references 처리
      references.forEach((ref, idx) => {
        if (!seen.has(ref)) {
          seen.add(ref);
          all.push({
            id: `ref-${idx}`,
            type: 'regulation',
            title: ref,
          });
        }
      });
    }

    // 타입별 정렬: 법령, 판례, 결정례, 해석례, 기타 순서
    const typeOrder: Record<string, number> = {
      'law': 1,
      'precedent': 2,
      'decision': 3,
      'interpretation': 4,
      'regulation': 5,
    };
    
    return all.sort((a, b) => {
      const orderA = typeOrder[a.type] || 999;
      const orderB = typeOrder[b.type] || 999;
      return orderA - orderB;
    });
  }, [references, allLegalRefs, sources, effectiveSourcesDetail, propSourcesByType]);

  // 전체 참고자료 개수 계산 (중복 제거된 parsedReferences 사용)
  const totalCount = parsedReferences.length;

  // 타입별 개수 계산 (parsedReferences 기반으로 정확하게 계산)
  const counts = useMemo(() => {
    return {
      law: parsedReferences.filter(r => r.type === 'law').length,
      precedent: parsedReferences.filter(r => r.type === 'precedent').length,
      decision: parsedReferences.filter(r => r.type === 'decision').length,
      interpretation: parsedReferences.filter(r => r.type === 'interpretation').length,
      regulation: parsedReferences.filter(r => r.type === 'regulation').length,
    };
  }, [parsedReferences]);

  // 첫 3개만 미리보기
  const previewReferences = parsedReferences.slice(0, 3);

  const handleReferenceClick = (ref: typeof parsedReferences[0]) => {
    if (onReferenceClick) {
      onReferenceClick(ref, ref.sourceDetail);
    } else if (onOpenSidebar) {
      onOpenSidebar(ref.type);
    }
  };

  // 참고자료가 없으면 렌더링하지 않음 (모든 hooks 호출 후에 체크)
  if (totalCount === 0) {
    return null;
  }

  return (
    <>
      <div>
        {/* 타입별 배지 (항상 표시, 클릭 가능) */}
        <div className="flex flex-wrap gap-2 mb-3">
          {counts.law > 0 && (
            <button
              onClick={() => onOpenSidebar?.('law')}
              className="text-xs px-2 py-1 bg-blue-50 text-blue-700 rounded-full border border-blue-200 hover:bg-blue-100 hover:border-blue-300 transition-colors cursor-pointer"
            >
              법령 {counts.law}
            </button>
          )}
          {counts.precedent > 0 && (
            <button
              onClick={() => onOpenSidebar?.('precedent')}
              className="text-xs px-2 py-1 bg-green-50 text-green-700 rounded-full border border-green-200 hover:bg-green-100 hover:border-green-300 transition-colors cursor-pointer"
            >
              판례 {counts.precedent}
            </button>
          )}
          {counts.decision > 0 && (
            <button
              onClick={() => onOpenSidebar?.('decision')}
              className="text-xs px-2 py-1 bg-orange-50 text-orange-700 rounded-full border border-orange-200 hover:bg-orange-100 hover:border-orange-300 transition-colors cursor-pointer"
            >
              결정례 {counts.decision}
            </button>
          )}
          {counts.interpretation > 0 && (
            <button
              onClick={() => onOpenSidebar?.('interpretation')}
              className="text-xs px-2 py-1 bg-indigo-50 text-indigo-700 rounded-full border border-indigo-200 hover:bg-indigo-100 hover:border-indigo-300 transition-colors cursor-pointer"
            >
              해석례 {counts.interpretation}
            </button>
          )}
          {counts.regulation > 0 && (
            <button
              onClick={() => onOpenSidebar?.('regulation')}
              className="text-xs px-2 py-1 bg-purple-50 text-purple-700 rounded-full border border-purple-200 hover:bg-purple-100 hover:border-purple-300 transition-colors cursor-pointer"
            >
              기타 {counts.regulation}
            </button>
          )}
        </div>

        {/* 인라인 미리보기 (첫 3개) - 개선된 형식 */}
        {previewReferences.length > 0 && (
          <div className="space-y-2 mb-3">
            {previewReferences.map((ref) => (
              <div
                key={ref.id}
                onClick={() => handleReferenceClick(ref)}
                className="text-xs p-2 bg-slate-50 rounded border border-slate-200 hover:bg-slate-100 transition-colors cursor-pointer"
              >
                <div className="flex items-start gap-2">
                  {ref.type === 'law' && <FileText className="w-3 h-3 text-blue-600 mt-0.5 flex-shrink-0" />}
                  {ref.type === 'precedent' && <Scale className="w-3 h-3 text-green-600 mt-0.5 flex-shrink-0" />}
                  {ref.type === 'decision' && <Bookmark className="w-3 h-3 text-orange-600 mt-0.5 flex-shrink-0" />}
                  {ref.type === 'interpretation' && <FileText className="w-3 h-3 text-indigo-600 mt-0.5 flex-shrink-0" />}
                  {ref.type === 'regulation' && <Bookmark className="w-3 h-3 text-purple-600 mt-0.5 flex-shrink-0" />}
                  <div className="flex-1 min-w-0">
                    <div className="text-slate-700 font-medium line-clamp-1">
                      {ref.title}
                    </div>
                    {ref.subtitle && (
                      <div className="text-slate-500 text-[10px] mt-0.5 line-clamp-1">
                        {ref.subtitle}
                      </div>
                    )}
                  </div>
                </div>
              </div>
            ))}
          </div>
        )}

        {/* 확장 섹션 - 개선된 형식 */}
        {parsedReferences.length > 3 && (
          <div className="space-y-2 mb-3 max-h-60 overflow-y-auto">
            {parsedReferences.slice(3).map((ref) => (
              <div
                key={ref.id}
                onClick={() => handleReferenceClick(ref)}
                className="text-xs p-2 bg-slate-50 rounded border border-slate-200 hover:bg-slate-100 transition-colors cursor-pointer"
              >
                <div className="flex items-start gap-2">
                  {ref.type === 'law' && <FileText className="w-3 h-3 text-blue-600 mt-0.5 flex-shrink-0" />}
                  {ref.type === 'precedent' && <Scale className="w-3 h-3 text-green-600 mt-0.5 flex-shrink-0" />}
                  {ref.type === 'decision' && <Bookmark className="w-3 h-3 text-orange-600 mt-0.5 flex-shrink-0" />}
                  {ref.type === 'interpretation' && <FileText className="w-3 h-3 text-indigo-600 mt-0.5 flex-shrink-0" />}
                  {ref.type === 'regulation' && <Bookmark className="w-3 h-3 text-purple-600 mt-0.5 flex-shrink-0" />}
                  <div className="flex-1 min-w-0">
                    <div className="text-slate-700 font-medium line-clamp-1">
                      {ref.title}
                    </div>
                    {ref.subtitle && (
                      <div className="text-slate-500 text-[10px] mt-0.5 line-clamp-1">
                        {ref.subtitle}
                      </div>
                    )}
                  </div>
                </div>
              </div>
            ))}
          </div>
        )}

        {/* 전체 보기 버튼 */}
        {totalCount > 0 && (
          <button
            onClick={() => onOpenSidebar?.('all')}
            className="text-xs text-blue-600 hover:text-blue-700 mt-1 mb-3 font-medium"
          >
            전체 상세 보기 →
          </button>
        )}
      </div>
    </>
  );
}

