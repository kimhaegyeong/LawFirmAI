/**
 * 컴팩트 참고자료 배지 컴포넌트
 * 하이브리드 방식: 인라인 미리보기 + 확장 섹션 + 사이드바
 */
import { FileText, ChevronDown, ChevronUp, Scale, Bookmark } from 'lucide-react';
import { useState, useMemo } from 'react';

import type { SourceInfo } from '../../types/chat';
import { getSourcesByType, extractLegalReferencesFromSourcesDetail } from '../../utils/sourcesParser';

type ReferenceType = 'all' | 'law' | 'precedent' | 'decision' | 'interpretation' | 'regulation';

interface CompactReferencesBadgeProps {
  references?: string[];
  legalReferences?: string[];
  sources?: string[];
  sourcesDetail?: SourceInfo[];
  onOpenSidebar?: (selectedType: ReferenceType) => void;
}

export function CompactReferencesBadge({
  references = [],
  legalReferences = [],
  sources = [],
  sourcesDetail = [],
  onOpenSidebar,
}: CompactReferencesBadgeProps) {
  const [isExpanded, setIsExpanded] = useState(false);

  // 전체 참고자료 개수 계산
  // sourcesDetail에서 legal_references 추출 (deprecated용)
  const extractedLegalRefs = useMemo(() => {
    return extractLegalReferencesFromSourcesDetail(sourcesDetail);
  }, [sourcesDetail]);
  
  // legalReferences와 extractedLegalRefs 병합 (하위 호환성)
  const allLegalRefs = useMemo(() => {
    return [...new Set([...legalReferences, ...extractedLegalRefs])];
  }, [legalReferences, extractedLegalRefs]);
  
  // sources_by_type 사용
  const sourcesByType = useMemo(() => {
    return getSourcesByType(sourcesDetail);
  }, [sourcesDetail]);

  const totalCount = references.length + allLegalRefs.length + sources.length + sourcesDetail.length;

  // 참고자료 파싱 개선 - 제목과 부제목 구조로 정보 표시
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
    
    // sources_detail 우선 처리 (상세 정보 활용)
    sourcesDetail.forEach((detail, idx) => {
      let type: 'law' | 'precedent' | 'decision' | 'interpretation' | 'regulation' = 'regulation';
      let title = '';
      let subtitle = '';
      
      if (detail.type === 'statute_article') {
        type = 'law';
        const lawName = detail.statute_name || detail.metadata?.statute_name || detail.name || '';
        const articleNo = detail.article_no || detail.metadata?.article_no;
        title = lawName;
        subtitle = articleNo ? `제${articleNo}조` : '';
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
        
        title = caseName || caseNumber || '판례';
        subtitle = [caseNumber, court, decisionDate].filter(Boolean).join(' · ');
      } else if (detail.type === 'decision_paragraph') {
        type = 'decision';
        const org = detail.org || detail.metadata?.org || '';
        const decisionNumber = detail.decision_number || 
                              detail.metadata?.decision_number ||
                              detail.metadata?.doc_id || '';
        const decisionDate = detail.decision_date || 
                           detail.metadata?.decision_date || '';
        
        title = org || decisionNumber || '결정례';
        subtitle = [decisionNumber, decisionDate].filter(Boolean).join(' · ');
      } else if (detail.type === 'interpretation_paragraph') {
        type = 'interpretation';
        const titleText = detail.title || detail.metadata?.title || detail.name || '';
        const org = detail.org || detail.metadata?.org || '';
        const responseDate = detail.response_date || 
                           detail.metadata?.response_date || '';
        
        title = titleText || '해석례';
        subtitle = [org, responseDate].filter(Boolean).join(' · ');
      } else {
        title = detail.name || detail.content || '';
      }
      
      // 중복 체크용 키 생성
      const key = detail.case_number || 
                 detail.article_number ||
                 detail.decision_number ||
                 detail.interpretation_number ||
                 detail.metadata?.doc_id || 
                 title || 
                 `detail-${idx}`;
      
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

    // legal_references 처리 (sources_detail과 중복 제외, deprecated)
    allLegalRefs.forEach((ref, idx) => {
      const matched = sourcesDetail.find(detail => {
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
      const matched = sourcesDetail.find(detail => {
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

    return all;
  }, [references, allLegalRefs, sources, sourcesDetail]);

  // 타입별 개수 계산 (sourcesByType 사용)
  const counts = useMemo(() => ({
    law: sourcesByType.statute_article.length,
    precedent: sourcesByType.case_paragraph.length,
    decision: sourcesByType.decision_paragraph.length,
    interpretation: sourcesByType.interpretation_paragraph.length,
    regulation: parsedReferences.filter(r => r.type === 'regulation').length,
  }), [sourcesByType, parsedReferences]);

  // 첫 3개만 미리보기
  const previewReferences = parsedReferences.slice(0, 3);
  const remainingCount = totalCount - previewReferences.length;

  // 참고자료가 없으면 렌더링하지 않음 (모든 hooks 호출 후에 체크)
  if (totalCount === 0) {
    return null;
  }

  return (
    <>
      <div className="mt-4 pt-4 border-t border-slate-200">
        {/* 헤더 */}
        <div className="flex items-center justify-between mb-2">
          <div className="flex items-center gap-2">
            <FileText className="w-4 h-4 text-slate-500" />
            <span className="text-sm font-medium text-slate-700">참고자료</span>
            <span className="text-xs text-slate-500">({totalCount}개)</span>
          </div>
          <button
            onClick={() => setIsExpanded(!isExpanded)}
            className="text-xs text-slate-500 hover:text-slate-700 flex items-center gap-1"
          >
            {isExpanded ? (
              <>
                <ChevronUp className="w-3 h-3" />
                <span>접기</span>
              </>
            ) : (
              <>
                <ChevronDown className="w-3 h-3" />
                <span>펼치기</span>
              </>
            )}
          </button>
        </div>

        {/* 타입별 배지 (항상 표시, 클릭 가능) */}
        <div className="flex flex-wrap gap-2 mb-2">
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
          <div className="space-y-1.5 mb-2">
            {previewReferences.map((ref) => (
              <div
                key={ref.id}
                className="text-xs p-2 bg-slate-50 rounded border border-slate-200 hover:bg-slate-100 transition-colors"
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
        {isExpanded && parsedReferences.length > 3 && (
          <div className="space-y-1.5 mb-2 max-h-60 overflow-y-auto">
            {parsedReferences.slice(3).map((ref) => (
              <div
                key={ref.id}
                className="text-xs p-2 bg-slate-50 rounded border border-slate-200 hover:bg-slate-100 transition-colors"
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
        {remainingCount > 0 && !isExpanded && (
          <button
            onClick={() => onOpenSidebar?.('all')}
            className="text-xs text-blue-600 hover:text-blue-700 mt-2 font-medium"
          >
            +{remainingCount}개 더 보기 →
          </button>
        )}
        {isExpanded && (
          <button
            onClick={() => onOpenSidebar?.('all')}
            className="text-xs text-blue-600 hover:text-blue-700 mt-2 font-medium"
          >
            전체 상세 보기 →
          </button>
        )}
      </div>
    </>
  );
}

