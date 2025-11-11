/**
 * 참고자료 모달 컨텐츠 컴포넌트
 * 법령/판례를 카드 형태로 분류하여 표시합니다.
 */
import { FileText, Scale, Bookmark, ExternalLink } from 'lucide-react';
import { useState, useMemo, useEffect } from 'react';
import logger from '../../utils/logger';
import type { LegalReferenceDetail, SourceInfo } from '../../types/chat';
import { generateLawUrl, generateSearchUrl, type LawUrlType } from '../../utils/lawUrlGenerator';

interface ReferencesModalContentProps {
  references?: string[];
  legalReferences?: string[];
  sources?: string[];
  sourcesDetail?: SourceInfo[];
  initialSelectedType?: ReferenceType;
  onReferenceClick?: (reference: LegalReferenceDetail, sourceDetail?: SourceInfo) => void;
}

type ReferenceType = 'all' | 'law' | 'precedent' | 'decision' | 'interpretation' | 'regulation';

/**
 * 참고자료 문자열을 구조화된 데이터로 변환
 */
function parseReferences(
  references: string[],
  legalReferences: string[],
  sources: string[]
): LegalReferenceDetail[] {
  const allReferences: LegalReferenceDetail[] = [];
  const seenRefs = new Set<string>(); // 중복 제거용

  // 입력값 검증 및 정규화
  const normalizeArray = (arr: string[] | undefined | null): string[] => {
    if (!Array.isArray(arr)) return [];
    return arr.filter(item => typeof item === 'string' && item.trim().length > 0);
  };

  const normalizedLegalRefs = normalizeArray(legalReferences);
  const normalizedSources = normalizeArray(sources);
  const normalizedRefs = normalizeArray(references);

  // 법령 참조 파싱
  normalizedLegalRefs.forEach((ref, idx) => {
    try {
      const trimmedRef = ref.trim();
      if (!trimmedRef || seenRefs.has(trimmedRef)) return;
      seenRefs.add(trimmedRef);
    
      const parsed = parseLegalReference(trimmedRef);
      // 판례 패턴이 포함되어 있으면 판례로 분류
      if (isPrecedentPattern(trimmedRef)) {
        const precedentParsed = parsePrecedentReference(trimmedRef);
        allReferences.push({
          id: `precedent-${idx}`,
          type: 'precedent',
          content: trimmedRef,
          ...precedentParsed,
        });
      } else {
        allReferences.push({
          id: `legal-${idx}`,
          type: 'law',
          content: trimmedRef,
          ...parsed,
        });
      }
    } catch (error) {
      logger.warn(`Failed to parse legal reference: ${ref}`, error);
      // 에러가 발생해도 기본 정보는 추가
      allReferences.push({
        id: `legal-error-${idx}`,
        type: 'law',
        content: ref,
        law_name: '법령',
      });
    }
  });

  // 판례/출처 파싱
  normalizedSources.forEach((src, idx) => {
    try {
      const trimmedSrc = src.trim();
      if (!trimmedSrc || seenRefs.has(trimmedSrc)) return;
      seenRefs.add(trimmedSrc);
    
      // 판례 패턴인지 확인
      if (isPrecedentPattern(trimmedSrc)) {
        const parsed = parsePrecedentReference(trimmedSrc);
        allReferences.push({
          id: `precedent-src-${idx}`,
          type: 'precedent',
          content: trimmedSrc,
          ...parsed,
        });
      } else if (isLegalPattern(trimmedSrc)) {
        // 법령 패턴이면 법령으로 분류
        const parsed = parseLegalReference(trimmedSrc);
        allReferences.push({
          id: `legal-src-${idx}`,
          type: 'law',
          content: trimmedSrc,
          ...parsed,
        });
      } else {
        // 기타는 판례로 분류 (기본값)
        const parsed = parsePrecedentReference(trimmedSrc);
        allReferences.push({
          id: `precedent-src-${idx}`,
          type: 'precedent',
          content: trimmedSrc,
          ...parsed,
        });
      }
    } catch (error) {
      logger.warn(`Failed to parse source: ${src}`, error);
      // 에러가 발생해도 기본 정보는 추가
      allReferences.push({
        id: `source-error-${idx}`,
        type: 'precedent',
        content: src,
      });
    }
  });

  // 기타 참조 파싱
  normalizedRefs.forEach((ref, idx) => {
    try {
      const trimmedRef = ref.trim();
      if (!trimmedRef || seenRefs.has(trimmedRef)) return;
      seenRefs.add(trimmedRef);
      
      // 이미 처리된 경우 건너뛰기
      if (normalizedLegalRefs.includes(trimmedRef) || normalizedSources.includes(trimmedRef)) {
        return;
      }

      // 패턴 기반 자동 분류
      if (isPrecedentPattern(trimmedRef)) {
        const parsed = parsePrecedentReference(trimmedRef);
        allReferences.push({
          id: `precedent-ref-${idx}`,
          type: 'precedent',
          content: trimmedRef,
          ...parsed,
        });
      } else if (isLegalPattern(trimmedRef)) {
        const parsed = parseLegalReference(trimmedRef);
        allReferences.push({
          id: `legal-ref-${idx}`,
          type: 'law',
          content: trimmedRef,
          ...parsed,
        });
      } else {
        const parsed = parseOtherReference(trimmedRef);
        allReferences.push({
          id: `reference-${idx}`,
          type: 'regulation',
          content: trimmedRef,
          ...parsed,
        });
      }
    } catch (error) {
      logger.warn(`Failed to parse reference: ${ref}`, error);
      // 에러가 발생해도 기본 정보는 추가
      allReferences.push({
        id: `ref-error-${idx}`,
        type: 'regulation',
        content: ref,
      });
    }
  });

  return allReferences;
}

/**
 * 판례 패턴인지 확인
 */
function isPrecedentPattern(text: string): boolean {
  if (!text) return false;
  
  const precedentPatterns = [
    /대법원|고등법원|지방법원|특허법원|가정법원|행정법원|법원\s*\d{4}/,
    /\d{4}[가-힣]\d+.*판결/,
    /\d{4}[가-힣]\d+.*선고/,
    /판례|사건번호|판결일/,
    /서울|부산|대구|인천|광주|대전|울산|수원|춘천|청주|전주|창원|제주.*지법/,
  ];
  
  return precedentPatterns.some(pattern => pattern.test(text));
}

/**
 * 법령 패턴인지 확인
 */
function isLegalPattern(text: string): boolean {
  if (!text) return false;
  
  const legalPatterns = [
    /[가-힣]+법(?:률)?\s*제\d+조/,
    /제\d+조.*항/,
    /부칙/,
    /시행령|시행규칙/,
  ];
  
  return legalPatterns.some(pattern => pattern.test(text));
}

/**
 * 법령 참조 파싱
 * 예: "민법 제123조", "형법 제250조 제1항", "민사소송법 제1조 제1항 제1호"
 */
function parseLegalReference(ref: string): Partial<LegalReferenceDetail> {
  const result: Partial<LegalReferenceDetail> = {};

  // 법률명 추출 (예: "민법", "형법", "상법", "민사소송법", "형사소송법")
  // 다양한 법률명 패턴 지원
  const lawNamePatterns = [
    /([가-힣]+법률)/,  // "민법률", "형법률" 등
    /([가-힣]+법)/,     // "민법", "형법", "민사소송법" 등
  ];
  
  for (const pattern of lawNamePatterns) {
    const match = ref.match(pattern);
    if (match) {
      result.law_name = match[1];
      break;
    }
  }

  // 조문 추출 (복잡한 형식 지원)
  // 예: "제123조", "제250조 제1항", "제1조 제1항 제1호", "제1조 제1항 제1호 제1목"
  // eslint-disable-next-line security/detect-unsafe-regex
  const articlePattern = /제(\d+)조(?:\s+제(\d+)항)?(?:\s+제(\d+)호)?(?:\s+제(\d+)목)?/;
  const articleMatch = ref.match(articlePattern);
  
  if (articleMatch) {
    let articleNumber = `제${articleMatch[1]}조`;
    if (articleMatch[2]) articleNumber += ` 제${articleMatch[2]}항`;
    if (articleMatch[3]) articleNumber += ` 제${articleMatch[3]}호`;
    if (articleMatch[4]) articleNumber += ` 제${articleMatch[4]}목`;
    result.article_number = articleNumber;
  }

  // 조문 제목 추출 (있는 경우)
  // "제1조 (목적)" 형식 또는 "제1조 목적" 형식
  const titlePatterns = [
    /제\d+조\s*[(（](.+?)[)）]/,  // 괄호 형식
    /제\d+조\s+(.+?)(?:\s+제\d+항|$)/,  // 일반 형식
  ];
  
  for (const pattern of titlePatterns) {
    const match = ref.match(pattern);
    if (match && match[1]) {
      result.article_title = match[1].trim();
      break;
    }
  }

  return result;
}

/**
 * 판례 참조 파싱
 * 예: "대법원 2020다12345 판결", "서울중앙지법 2019나12345 판결", "대법원 2020.12.25. 선고 2020다12345 판결"
 */
function parsePrecedentReference(ref: string): Partial<LegalReferenceDetail> {
  const result: Partial<LegalReferenceDetail> = {};

  // 법원 추출 (더 많은 법원 패턴 지원)
  const courtPatterns = [
    /(대법원|고등법원|특허법원|가정법원|행정법원)/,
    /(서울|부산|대구|인천|광주|대전|울산|수원|춘천|청주|전주|창원|제주)(?:중앙|서부|동부|북부)?지법/,
    /(서울|부산|대구|인천|광주|대전|울산|수원|춘천|청주|전주|창원|제주)고등법원/,
    /(서울|부산|대구|인천|광주|대전|울산|수원|춘천|청주|전주|창원|제주)지방법원/,
  ];

  for (const pattern of courtPatterns) {
    const match = ref.match(pattern);
    if (match) {
      result.court = match[0];
      break;
    }
  }

  // 사건번호 추출 (예: "2020다12345", "2019나12345", "2020다12345, 2020가12345")
  // 사건번호는 4자리 연도 + 한글 1자 + 숫자 형식
  const caseNumberPattern = /(\d{4}[가-힣]\d{4,})/;
  const caseNumberMatch = ref.match(caseNumberPattern);
  if (caseNumberMatch) {
    result.case_number = caseNumberMatch[1];
  }

  // 판결일 추출 (다양한 형식 지원)
  // 예: "2020.12.25", "2020-12-25", "2020년 12월 25일", "20201225"
  const datePatterns = [
    /(\d{4}[.-]\d{1,2}[.-]\d{1,2})/,  // 2020.12.25, 2020-12-25
    /(\d{4}년\s*\d{1,2}월\s*\d{1,2}일)/,  // 2020년 12월 25일
    /(\d{4}\.\s*\d{1,2}\.\s*\d{1,2}\.)/,  // 2020. 12. 25.
  ];

  for (const pattern of datePatterns) {
    const match = ref.match(pattern);
    if (match) {
      result.decision_date = match[1];
      break;
    }
  }

  // 판례명 추출
  // 전체 텍스트에서 법원, 사건번호, 판결일, 판결 등을 제외한 부분
  let caseName = ref;
  
  if (result.court) {
    caseName = caseName.replace(result.court, '').trim();
  }
  if (result.case_number) {
    caseName = caseName.replace(result.case_number, '').trim();
  }
  if (result.decision_date) {
    caseName = caseName.replace(result.decision_date, '').trim();
  }
  
  // 판결 관련 키워드 제거
  caseName = caseName
    .replace(/판결|선고|선고일|등|,|\./g, '')
    .replace(/\s+/g, ' ')
    .trim();

  if (caseName && caseName.length > 2) {
    result.case_name = caseName;
  } else if (!result.case_number && !result.court) {
    // 판례 정보가 없으면 전체 텍스트를 판례명으로
    result.case_name = ref;
  }

  // 판례 요약 정보가 있는 경우 (있는 경우)
  if (ref.includes('요지') || ref.includes('판시사항')) {
    const summaryMatch = ref.match(/(요지|판시사항)[:\s]+(.+?)(?:\n|$)/);
    if (summaryMatch && summaryMatch[2]) {
      result.summary = summaryMatch[2].trim();
    }
  }

  return result;
}

/**
 * 기타 참조 파싱
 */
function parseOtherReference(ref: string): Partial<LegalReferenceDetail> {
  const result: Partial<LegalReferenceDetail> = {};

  // 규칙, 시행령, 시행규칙 등
  const regulationPatterns = [
    /([가-힣]+시행규칙)/,
    /([가-힣]+시행령)/,
    /([가-힣]+규칙)/,
    /([가-힣]+고시)/,
    /([가-힣]+예규)/,
    /([가-힣]+훈령)/,
  ];

  for (const pattern of regulationPatterns) {
    const match = ref.match(pattern);
    if (match) {
      result.law_name = match[1];
      break;
    }
  }

  // 조문 추출 (있는 경우)
  const articleMatch = ref.match(/제\d+조/);
  if (articleMatch) {
    result.article_number = articleMatch[0];
  }

  return result;
}

/**
 * 법령 카드 컴포넌트
 */
function LawCard({ law, onClick, sourceDetail }: { law: LegalReferenceDetail & { url?: string }; onClick?: () => void; sourceDetail?: SourceInfo }) {
  const [isExpanded, setIsExpanded] = useState(false);
  
  const cardUrl = useMemo(() => {
    if (law.url) return law.url;
    
    const metadata = {
      ...sourceDetail?.metadata,
      ...law,
      law_id: sourceDetail?.metadata?.law_id,
      mst: sourceDetail?.metadata?.mst,
      article_no: law.article_number || sourceDetail?.article_no,
      effective_date: sourceDetail?.metadata?.effective_date,
      statute_name: law.law_name || sourceDetail?.statute_name,
    };
    
    return generateLawUrl('statute', metadata);
  }, [law, sourceDetail]);
  
  const searchUrl = useMemo(() => {
    if (cardUrl) return null;
    
    const metadata = {
      statute_name: law.law_name || sourceDetail?.statute_name || sourceDetail?.metadata?.statute_name,
    };
    
    return generateSearchUrl('statute', metadata);
  }, [cardUrl, law, sourceDetail]);
  
  return (
    <div 
      className={`p-4 bg-blue-50 border border-blue-200 rounded-lg hover:shadow-md transition-shadow ${onClick ? 'cursor-pointer' : ''}`}
      onClick={onClick}
      onKeyDown={onClick ? (e) => {
        if (e.key === 'Enter' || e.key === ' ') {
          e.preventDefault();
          onClick();
        }
      } : undefined}
      role={onClick ? 'button' : undefined}
      tabIndex={onClick ? 0 : undefined}
    >
      <div className="flex items-start justify-between gap-3">
        <div className="flex-1">
          <div className="flex items-center gap-2 mb-2">
            <FileText className="w-4 h-4 text-blue-600 flex-shrink-0" />
            <span className="text-xs font-semibold text-blue-600">법령</span>
            {law.relevance_score !== undefined && (
              <span className="text-xs text-slate-500">
                관련도: {Math.round(law.relevance_score * 100)}%
              </span>
            )}
            {law.similarity !== undefined && (
              <span className="text-xs text-slate-500">
                유사도: {Math.round(law.similarity * 100)}%
              </span>
            )}
          </div>
          <h4 className="font-semibold text-slate-800 mb-1">
            {law.law_name || '법령'}
          </h4>
          {law.article_number && (
            <p className="text-sm text-blue-700 mb-1 font-medium">
              {law.article_number} {law.article_title || ''}
            </p>
          )}
          {law.article_content ? (
            <div className="mt-2">
              <p className={`text-sm text-slate-600 ${!isExpanded ? 'line-clamp-3' : ''}`}>
                {law.article_content}
              </p>
              {law.article_content.length > 200 && (
                <button
                  onClick={() => setIsExpanded(!isExpanded)}
                  className="text-xs text-blue-600 hover:text-blue-700 mt-1"
                >
                  {isExpanded ? '접기' : '더 보기'}
                </button>
              )}
            </div>
          ) : law.content ? (
            <p className="text-sm text-slate-600 mt-2">{law.content}</p>
          ) : null}
        </div>
        {cardUrl ? (
          <a
            href={cardUrl}
            target="_blank"
            rel="noopener noreferrer"
            className="text-blue-600 hover:text-blue-700 p-1 rounded hover:bg-blue-100 transition-colors"
            title="원문 보기"
            onClick={(e) => e.stopPropagation()}
          >
            <ExternalLink className="w-4 h-4" />
          </a>
        ) : searchUrl ? (
          <a
            href={searchUrl}
            target="_blank"
            rel="noopener noreferrer"
            className="text-blue-600 hover:text-blue-700 p-1 rounded hover:bg-blue-100 transition-colors"
            title="검색하기"
            onClick={(e) => e.stopPropagation()}
          >
            <ExternalLink className="w-4 h-4" />
          </a>
        ) : (
          <button
            className="text-blue-600 hover:text-blue-700 p-1 rounded hover:bg-blue-100 transition-colors"
            title="상세 정보 보기"
            onClick={(e) => {
              e.stopPropagation();
              onClick?.();
            }}
          >
            <ExternalLink className="w-4 h-4" />
          </button>
        )}
      </div>
    </div>
  );
}

/**
 * 판례 카드 컴포넌트
 */
function PrecedentCard({ precedent, onClick, sourceDetail }: { precedent: LegalReferenceDetail & { url?: string }; onClick?: () => void; sourceDetail?: SourceInfo }) {
  const [isExpanded, setIsExpanded] = useState(false);
  
  const cardUrl = useMemo(() => {
    if (precedent.url) return precedent.url;
    
    const metadata = {
      ...sourceDetail?.metadata,
      ...precedent,
      precedent_serial_number: sourceDetail?.metadata?.precedent_serial_number,
      doc_id: precedent.case_number || sourceDetail?.doc_id,
      casenames: precedent.case_name || sourceDetail?.case_name,
    };
    
    return generateLawUrl('case', metadata);
  }, [precedent, sourceDetail]);
  
  const searchUrl = useMemo(() => {
    if (cardUrl) return null;
    
    const metadata = {
      casenames: precedent.case_name || sourceDetail?.case_name || sourceDetail?.metadata?.casenames,
    };
    
    return generateSearchUrl('case', metadata);
  }, [cardUrl, precedent, sourceDetail]);
  
  return (
    <div 
      className={`p-4 bg-green-50 border border-green-200 rounded-lg hover:shadow-md transition-shadow ${onClick ? 'cursor-pointer' : ''}`}
      onClick={onClick}
      onKeyDown={onClick ? (e) => {
        if (e.key === 'Enter' || e.key === ' ') {
          e.preventDefault();
          onClick();
        }
      } : undefined}
      role={onClick ? 'button' : undefined}
      tabIndex={onClick ? 0 : undefined}
    >
      <div className="flex items-start justify-between gap-3">
        <div className="flex-1">
          <div className="flex items-center gap-2 mb-2">
            <Scale className="w-4 h-4 text-green-600 flex-shrink-0" />
            <span className="text-xs font-semibold text-green-600">판례</span>
            {precedent.relevance_score !== undefined && (
              <span className="text-xs text-slate-500">
                관련도: {Math.round(precedent.relevance_score * 100)}%
              </span>
            )}
            {precedent.similarity !== undefined && (
              <span className="text-xs text-slate-500">
                유사도: {Math.round(precedent.similarity * 100)}%
              </span>
            )}
          </div>
          <h4 className="font-semibold text-slate-800 mb-1">
            {precedent.case_name || precedent.case_number || '판례'}
          </h4>
          <div className="flex flex-wrap gap-2 text-xs text-slate-600 mb-2">
            {precedent.case_number && (
              <span className="bg-white px-2 py-1 rounded">사건번호: {precedent.case_number}</span>
            )}
            {precedent.court && (
              <span className="bg-white px-2 py-1 rounded">법원: {precedent.court}</span>
            )}
            {precedent.decision_date && (
              <span className="bg-white px-2 py-1 rounded">판결일: {precedent.decision_date}</span>
            )}
          </div>
          {precedent.summary ? (
            <div className="mt-2">
              <p className={`text-sm text-slate-700 ${!isExpanded ? 'line-clamp-3' : ''}`}>
                {precedent.summary}
              </p>
              {precedent.summary.length > 200 && (
                <button
                  onClick={() => setIsExpanded(!isExpanded)}
                  className="text-xs text-green-600 hover:text-green-700 mt-1"
                >
                  {isExpanded ? '접기' : '더 보기'}
                </button>
              )}
            </div>
          ) : precedent.content ? (
            <p className="text-sm text-slate-700 mt-2">{precedent.content}</p>
          ) : null}
        </div>
        {cardUrl ? (
          <a
            href={cardUrl}
            target="_blank"
            rel="noopener noreferrer"
            className="text-green-600 hover:text-green-700 p-1 rounded hover:bg-green-100 transition-colors"
            title="원문 보기"
            onClick={(e) => e.stopPropagation()}
          >
            <ExternalLink className="w-4 h-4" />
          </a>
        ) : searchUrl ? (
          <a
            href={searchUrl}
            target="_blank"
            rel="noopener noreferrer"
            className="text-green-600 hover:text-green-700 p-1 rounded hover:bg-green-100 transition-colors"
            title="검색하기"
            onClick={(e) => e.stopPropagation()}
          >
            <ExternalLink className="w-4 h-4" />
          </a>
        ) : (
          <button
            className="text-green-600 hover:text-green-700 p-1 rounded hover:bg-green-100 transition-colors"
            title="상세 정보 보기"
            onClick={(e) => {
              e.stopPropagation();
              onClick?.();
            }}
          >
            <ExternalLink className="w-4 h-4" />
          </button>
        )}
      </div>
    </div>
  );
}

/**
 * 결정례 카드 컴포넌트
 */
function DecisionCard({ decision, onClick, sourceDetail }: { decision: LegalReferenceDetail & { url?: string }; onClick?: () => void; sourceDetail?: SourceInfo }) {
  const [isExpanded, setIsExpanded] = useState(false);
  
  const cardUrl = useMemo(() => {
    if (decision.url) return decision.url;
    
    const metadata = {
      ...sourceDetail?.metadata,
      ...decision,
      decision_serial_number: sourceDetail?.metadata?.decision_serial_number,
      doc_id: decision.decision_number || sourceDetail?.doc_id,
      org: decision.org || sourceDetail?.org,
    };
    
    return generateLawUrl('decision', metadata);
  }, [decision, sourceDetail]);
  
  const searchUrl = useMemo(() => {
    if (cardUrl) return null;
    
    const metadata = {
      org: decision.org || sourceDetail?.org || sourceDetail?.metadata?.org,
    };
    
    return generateSearchUrl('decision', metadata);
  }, [cardUrl, decision, sourceDetail]);
  
  return (
    <div 
      className={`p-4 bg-orange-50 border border-orange-200 rounded-lg hover:shadow-md transition-shadow ${onClick ? 'cursor-pointer' : ''}`}
      onClick={onClick}
      onKeyDown={onClick ? (e) => {
        if (e.key === 'Enter' || e.key === ' ') {
          e.preventDefault();
          onClick();
        }
      } : undefined}
      role={onClick ? 'button' : undefined}
      tabIndex={onClick ? 0 : undefined}
    >
      <div className="flex items-start justify-between gap-3">
        <div className="flex-1">
          <div className="flex items-center gap-2 mb-2">
            <Bookmark className="w-4 h-4 text-orange-600 flex-shrink-0" />
            <span className="text-xs font-semibold text-orange-600">결정례</span>
            {decision.relevance_score !== undefined && (
              <span className="text-xs text-slate-500">
                관련도: {Math.round(decision.relevance_score * 100)}%
              </span>
            )}
          </div>
          <h4 className="font-semibold text-slate-800 mb-1">
            {decision.org || '결정례'}
          </h4>
          <div className="flex flex-wrap gap-2 text-xs text-slate-600 mb-2">
            {decision.decision_number && (
              <span className="bg-white px-2 py-1 rounded">일련번호: {decision.decision_number}</span>
            )}
            {decision.decision_date && (
              <span className="bg-white px-2 py-1 rounded">결정일: {decision.decision_date}</span>
            )}
            {decision.result && (
              <span className="bg-white px-2 py-1 rounded">결과: {decision.result}</span>
            )}
          </div>
          {decision.content ? (
            <div className="mt-2">
              <p className={`text-sm text-slate-700 ${!isExpanded ? 'line-clamp-3' : ''}`}>
                {decision.content}
              </p>
              {decision.content.length > 200 && (
                <button
                  onClick={() => setIsExpanded(!isExpanded)}
                  className="text-xs text-orange-600 hover:text-orange-700 mt-1"
                >
                  {isExpanded ? '접기' : '더 보기'}
                </button>
              )}
            </div>
          ) : null}
        </div>
        {cardUrl ? (
          <a
            href={cardUrl}
            target="_blank"
            rel="noopener noreferrer"
            className="text-orange-600 hover:text-orange-700 p-1 rounded hover:bg-orange-100 transition-colors"
            title="원문 보기"
            onClick={(e) => e.stopPropagation()}
          >
            <ExternalLink className="w-4 h-4" />
          </a>
        ) : searchUrl ? (
          <a
            href={searchUrl}
            target="_blank"
            rel="noopener noreferrer"
            className="text-orange-600 hover:text-orange-700 p-1 rounded hover:bg-orange-100 transition-colors"
            title="검색하기"
            onClick={(e) => e.stopPropagation()}
          >
            <ExternalLink className="w-4 h-4" />
          </a>
        ) : (
          <button
            className="text-orange-600 hover:text-orange-700 p-1 rounded hover:bg-orange-100 transition-colors"
            title="상세 정보 보기"
            onClick={(e) => {
              e.stopPropagation();
              onClick?.();
            }}
          >
            <ExternalLink className="w-4 h-4" />
          </button>
        )}
      </div>
    </div>
  );
}

/**
 * 해석례 카드 컴포넌트
 */
function InterpretationCard({ interpretation, onClick, sourceDetail }: { interpretation: LegalReferenceDetail & { url?: string }; onClick?: () => void; sourceDetail?: SourceInfo }) {
  const [isExpanded, setIsExpanded] = useState(false);
  
  const cardUrl = useMemo(() => {
    if (interpretation.url) return interpretation.url;
    
    const metadata = {
      ...sourceDetail?.metadata,
      ...interpretation,
      interpretation_serial_number: sourceDetail?.metadata?.interpretation_serial_number,
      doc_id: interpretation.interpretation_number || sourceDetail?.doc_id,
      title: interpretation.title || sourceDetail?.title,
    };
    
    return generateLawUrl('interpretation', metadata);
  }, [interpretation, sourceDetail]);
  
  const searchUrl = useMemo(() => {
    if (cardUrl) return null;
    
    const metadata = {
      title: interpretation.title || sourceDetail?.title || sourceDetail?.metadata?.title,
    };
    
    return generateSearchUrl('interpretation', metadata);
  }, [cardUrl, interpretation, sourceDetail]);
  
  return (
    <div 
      className={`p-4 bg-indigo-50 border border-indigo-200 rounded-lg hover:shadow-md transition-shadow ${onClick ? 'cursor-pointer' : ''}`}
      onClick={onClick}
      onKeyDown={onClick ? (e) => {
        if (e.key === 'Enter' || e.key === ' ') {
          e.preventDefault();
          onClick();
        }
      } : undefined}
      role={onClick ? 'button' : undefined}
      tabIndex={onClick ? 0 : undefined}
    >
      <div className="flex items-start justify-between gap-3">
        <div className="flex-1">
          <div className="flex items-center gap-2 mb-2">
            <FileText className="w-4 h-4 text-indigo-600 flex-shrink-0" />
            <span className="text-xs font-semibold text-indigo-600">해석례</span>
            {interpretation.relevance_score !== undefined && (
              <span className="text-xs text-slate-500">
                관련도: {Math.round(interpretation.relevance_score * 100)}%
              </span>
            )}
          </div>
          <h4 className="font-semibold text-slate-800 mb-1">
            {interpretation.title || interpretation.org || '해석례'}
          </h4>
          <div className="flex flex-wrap gap-2 text-xs text-slate-600 mb-2">
            {interpretation.interpretation_number && (
              <span className="bg-white px-2 py-1 rounded">일련번호: {interpretation.interpretation_number}</span>
            )}
            {interpretation.org && (
              <span className="bg-white px-2 py-1 rounded">기관: {interpretation.org}</span>
            )}
            {interpretation.response_date && (
              <span className="bg-white px-2 py-1 rounded">회신일: {interpretation.response_date}</span>
            )}
          </div>
          {interpretation.content ? (
            <div className="mt-2">
              <p className={`text-sm text-slate-700 ${!isExpanded ? 'line-clamp-3' : ''}`}>
                {interpretation.content}
              </p>
              {interpretation.content.length > 200 && (
                <button
                  onClick={() => setIsExpanded(!isExpanded)}
                  className="text-xs text-indigo-600 hover:text-indigo-700 mt-1"
                >
                  {isExpanded ? '접기' : '더 보기'}
                </button>
              )}
            </div>
          ) : null}
        </div>
        {cardUrl ? (
          <a
            href={cardUrl}
            target="_blank"
            rel="noopener noreferrer"
            className="text-indigo-600 hover:text-indigo-700 p-1 rounded hover:bg-indigo-100 transition-colors"
            title="원문 보기"
            onClick={(e) => e.stopPropagation()}
          >
            <ExternalLink className="w-4 h-4" />
          </a>
        ) : searchUrl ? (
          <a
            href={searchUrl}
            target="_blank"
            rel="noopener noreferrer"
            className="text-indigo-600 hover:text-indigo-700 p-1 rounded hover:bg-indigo-100 transition-colors"
            title="검색하기"
            onClick={(e) => e.stopPropagation()}
          >
            <ExternalLink className="w-4 h-4" />
          </a>
        ) : (
          <button
            className="text-indigo-600 hover:text-indigo-700 p-1 rounded hover:bg-indigo-100 transition-colors"
            title="상세 정보 보기"
            onClick={(e) => {
              e.stopPropagation();
              onClick?.();
            }}
          >
            <ExternalLink className="w-4 h-4" />
          </button>
        )}
      </div>
    </div>
  );
}

/**
 * 기타 참조 카드 컴포넌트
 */
function RegulationCard({ regulation, onClick }: { regulation: LegalReferenceDetail; onClick?: () => void }) {
  return (
    <div 
      className={`p-4 bg-purple-50 border border-purple-200 rounded-lg hover:shadow-md transition-shadow ${onClick ? 'cursor-pointer' : ''}`}
      onClick={onClick}
      onKeyDown={onClick ? (e) => {
        if (e.key === 'Enter' || e.key === ' ') {
          e.preventDefault();
          onClick();
        }
      } : undefined}
      role={onClick ? 'button' : undefined}
      tabIndex={onClick ? 0 : undefined}
    >
      <div className="flex items-start justify-between gap-3">
        <div className="flex-1">
          <div className="flex items-center gap-2 mb-2">
            <Bookmark className="w-4 h-4 text-purple-600 flex-shrink-0" />
            <span className="text-xs font-semibold text-purple-600">기타</span>
          </div>
          <h4 className="font-semibold text-slate-800 mb-1">
            {regulation.law_name || '기타 참조'}
          </h4>
          {regulation.content && (
            <p className="text-sm text-slate-600 mt-2">{regulation.content}</p>
          )}
        </div>
        <button
          className="text-purple-600 hover:text-purple-700 p-1 rounded hover:bg-purple-100 transition-colors"
          title="상세 정보 보기"
        >
          <ExternalLink className="w-4 h-4" />
        </button>
      </div>
    </div>
  );
}

/**
 * 참고자료 모달 컨텐츠 메인 컴포넌트
 */
export function ReferencesModalContent({
  references = [],
  legalReferences = [],
  sources = [],
  sourcesDetail = [],
  initialSelectedType = 'all',
  onReferenceClick,
}: ReferencesModalContentProps) {
  const [selectedType, setSelectedType] = useState<ReferenceType>(initialSelectedType);
  
  // initialSelectedType이 변경되면 selectedType 업데이트
  useEffect(() => {
    if (initialSelectedType) {
      setSelectedType(initialSelectedType);
    }
  }, [initialSelectedType]);

  // sourcesDetail을 LegalReferenceDetail로 변환 (sourceDetail 정보도 함께 저장)
  const sourcesDetailReferences = useMemo(() => {
    return sourcesDetail.map((detail, idx): LegalReferenceDetail & { sourceDetail?: SourceInfo } => {
      const baseRef: LegalReferenceDetail & { sourceDetail?: SourceInfo } = {
        id: `source-detail-${idx}`,
        type: 'regulation',
        content: detail.content || detail.name,
        sourceDetail: detail,
      };

      // 법령 정보
      if (detail.type === 'statute_article') {
        return {
          ...baseRef,
          type: 'law',
          law_name: detail.statute_name || detail.metadata?.statute_name,
          article_number: detail.article_no 
            ? `제${detail.article_no}조${detail.clause_no ? ` 제${detail.clause_no}항` : ''}${detail.item_no ? ` 제${detail.item_no}호` : ''}`
            : detail.metadata?.article_no,
          article_content: detail.content,
          sourceDetail: detail,
        };
      }

      // 판례 정보
      if (detail.type === 'case_paragraph') {
        return {
          ...baseRef,
          type: 'precedent',
          case_name: detail.case_name || detail.metadata?.casenames,
          case_number: detail.case_number || detail.metadata?.doc_id,
          court: detail.court || detail.metadata?.court,
          summary: detail.content,
          sourceDetail: detail,
        };
      }

      // 결정례 정보
      if (detail.type === 'decision_paragraph') {
        return {
          ...baseRef,
          type: 'decision',
          decision_number: detail.decision_number || detail.metadata?.doc_id,
          org: detail.org || detail.metadata?.org,
          decision_date: detail.decision_date || detail.metadata?.decision_date,
          result: detail.result || detail.metadata?.result,
          content: detail.content || detail.name,
          sourceDetail: detail,
        };
      }

      // 해석례 정보
      if (detail.type === 'interpretation_paragraph') {
        return {
          ...baseRef,
          type: 'interpretation',
          interpretation_number: detail.interpretation_number || detail.metadata?.doc_id,
          org: detail.org || detail.metadata?.org,
          title: detail.title || detail.metadata?.title,
          response_date: detail.response_date || detail.metadata?.response_date,
          content: detail.content || detail.name,
          sourceDetail: detail,
        };
      }

      return baseRef;
    });
  }, [sourcesDetail]);

  // 참고자료 파싱 및 분류 (sourcesDetail 우선)
  const parsedReferences = useMemo(() => {
    const detailRefs = sourcesDetailReferences;
    const stringRefs = parseReferences(references, legalReferences, sources);
    
    // sourcesDetail을 우선하고, 중복 제거
    const seen = new Set<string>();
    const result: LegalReferenceDetail[] = [];
    
    // sourcesDetail 먼저 추가
    detailRefs.forEach(ref => {
      // key 생성: case_number, article_number, law_name, content 순서로 시도
      const key = ref.case_number || ref.article_number || ref.law_name || ref.content || ref.id || `detail-${result.length}`;
      if (!seen.has(key)) {
        seen.add(key);
        result.push(ref);
      }
    });
    
    // 문자열 참조 추가 (중복 제외)
    stringRefs.forEach(ref => {
      const key = ref.case_number || ref.article_number || ref.law_name || ref.content || ref.id || `string-${result.length}`;
      if (!seen.has(key)) {
        seen.add(key);
        result.push(ref);
      }
    });
    
    return result;
  }, [sourcesDetailReferences, references, legalReferences, sources]);

  // 타입별 필터링 및 상위 3개만 선택
  const filteredReferences = useMemo(() => {
    let filtered: LegalReferenceDetail[] = [];
    
    if (selectedType === 'all') {
      filtered = parsedReferences;
    } else {
      filtered = parsedReferences.filter((ref) => ref.type === selectedType);
    }
    
    // 관련도 점수로 정렬 (relevance_score 우선, 없으면 similarity 사용)
    filtered.sort((a, b) => {
      const scoreA = a.relevance_score ?? a.similarity ?? 0;
      const scoreB = b.relevance_score ?? b.similarity ?? 0;
      return scoreB - scoreA; // 내림차순 정렬
    });
    
    // 상위 3개만 반환 (전체가 아닌 경우에만)
    if (selectedType !== 'all') {
      return filtered.slice(0, 3);
    }
    
    return filtered;
  }, [parsedReferences, selectedType]);

  // 타입별 개수 계산
  const counts = useMemo(() => {
    return {
      all: parsedReferences.length,
      law: parsedReferences.filter((r) => r.type === 'law').length,
      precedent: parsedReferences.filter((r) => r.type === 'precedent').length,
      decision: parsedReferences.filter((r) => r.type === 'decision').length,
      interpretation: parsedReferences.filter((r) => r.type === 'interpretation').length,
      regulation: parsedReferences.filter((r) => r.type === 'regulation').length,
    };
  }, [parsedReferences]);

  if (parsedReferences.length === 0) {
    return (
      <div className="text-center py-8 text-slate-500">
        참고자료가 없습니다.
      </div>
    );
  }

  return (
    <div className="flex flex-col h-full space-y-4">
      {/* 탭/필터 UI */}
      <div className="flex gap-2 border-b border-slate-200 pb-2 overflow-x-auto flex-shrink-0">
        <button
          onClick={() => setSelectedType('all')}
          className={`px-3 py-1 rounded-md text-sm font-medium transition-colors whitespace-nowrap ${
            selectedType === 'all'
              ? 'bg-blue-100 text-blue-700 border border-blue-300'
              : 'bg-slate-100 text-slate-600 hover:bg-slate-200'
          }`}
        >
          전체 ({counts.all})
        </button>
        {counts.law > 0 && (
          <button
            onClick={() => setSelectedType('law')}
            className={`px-3 py-1 rounded-md text-sm font-medium transition-colors whitespace-nowrap ${
              selectedType === 'law'
                ? 'bg-blue-100 text-blue-700 border border-blue-300'
                : 'bg-slate-100 text-slate-600 hover:bg-slate-200'
            }`}
          >
            법령 ({counts.law})
          </button>
        )}
        {counts.precedent > 0 && (
          <button
            onClick={() => setSelectedType('precedent')}
            className={`px-3 py-1 rounded-md text-sm font-medium transition-colors whitespace-nowrap ${
              selectedType === 'precedent'
                ? 'bg-green-100 text-green-700 border border-green-300'
                : 'bg-slate-100 text-slate-600 hover:bg-slate-200'
            }`}
          >
            판례 ({counts.precedent})
          </button>
        )}
        {counts.decision > 0 && (
          <button
            onClick={() => setSelectedType('decision')}
            className={`px-3 py-1 rounded-md text-sm font-medium transition-colors whitespace-nowrap ${
              selectedType === 'decision'
                ? 'bg-orange-100 text-orange-700 border border-orange-300'
                : 'bg-slate-100 text-slate-600 hover:bg-slate-200'
            }`}
          >
            결정례 ({counts.decision})
          </button>
        )}
        {counts.interpretation > 0 && (
          <button
            onClick={() => setSelectedType('interpretation')}
            className={`px-3 py-1 rounded-md text-sm font-medium transition-colors whitespace-nowrap ${
              selectedType === 'interpretation'
                ? 'bg-indigo-100 text-indigo-700 border border-indigo-300'
                : 'bg-slate-100 text-slate-600 hover:bg-slate-200'
            }`}
          >
            해석례 ({counts.interpretation})
          </button>
        )}
        {counts.regulation > 0 && (
          <button
            onClick={() => setSelectedType('regulation')}
            className={`px-3 py-1 rounded-md text-sm font-medium transition-colors whitespace-nowrap ${
              selectedType === 'regulation'
                ? 'bg-purple-100 text-purple-700 border border-purple-300'
                : 'bg-slate-100 text-slate-600 hover:bg-slate-200'
            }`}
          >
            기타 ({counts.regulation})
          </button>
        )}
      </div>

      {/* 참고자료 목록 */}
      <div className="space-y-3 flex-1 overflow-y-auto pr-2 -mr-2 min-h-0">
        {filteredReferences.length === 0 ? (
          <div className="text-center py-8 text-slate-500">
            선택한 타입의 참고자료가 없습니다.
          </div>
        ) : (
          <>
            {/* 상위 3개 제한 안내 (전체가 아닌 경우) */}
            {/* eslint-disable-next-line security/detect-object-injection */}
            {selectedType !== 'all' && counts[selectedType] > 3 && (
              <div className="text-xs text-slate-500 bg-slate-50 p-2 rounded border border-slate-200">
                관련도가 높은 상위 3개만 표시됩니다. (전체 {counts[selectedType]}개)
              </div>
            )}
            
            {filteredReferences.map((ref) => {
              const sourceDetail = (ref as LegalReferenceDetail & { sourceDetail?: SourceInfo }).sourceDetail;
              
              const handleClick = () => {
                if (onReferenceClick) {
                  onReferenceClick(ref, sourceDetail);
                }
              };

              switch (ref.type) {
                case 'law':
                  return <LawCard key={ref.id} law={ref} onClick={handleClick} sourceDetail={sourceDetail} />;
                case 'precedent':
                  return <PrecedentCard key={ref.id} precedent={ref} onClick={handleClick} sourceDetail={sourceDetail} />;
                case 'decision':
                  return <DecisionCard key={ref.id} decision={ref} onClick={handleClick} sourceDetail={sourceDetail} />;
                case 'interpretation':
                  return <InterpretationCard key={ref.id} interpretation={ref} onClick={handleClick} sourceDetail={sourceDetail} />;
                case 'regulation':
                  return <RegulationCard key={ref.id} regulation={ref} onClick={handleClick} />;
                default:
                  return null;
              }
            })}
          </>
        )}
      </div>
    </div>
  );
}

