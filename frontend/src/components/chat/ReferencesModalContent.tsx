/**
 * 참고자료 모달 컨텐츠 컴포넌트
 * 법령/판례를 카드 형태로 분류하여 표시합니다.
 */
import { FileText, Scale, Bookmark, ExternalLink } from 'lucide-react';
import { useState, useMemo } from 'react';
import logger from '../../utils/logger';
import type { LegalReferenceDetail, SourceInfo } from '../../types/chat';

interface ReferencesModalContentProps {
  references?: string[];
  legalReferences?: string[];
  sources?: string[];
  sourcesDetail?: SourceInfo[];
}

type ReferenceType = 'all' | 'law' | 'precedent' | 'regulation';

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
    /제\d+조\s*[\(（](.+?)[\)）]/,  // 괄호 형식
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
    /(\d{4}[.\-]\d{1,2}[.\-]\d{1,2})/,  // 2020.12.25, 2020-12-25
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
function LawCard({ law }: { law: LegalReferenceDetail & { url?: string } }) {
  return (
    <div className="p-4 bg-blue-50 border border-blue-200 rounded-lg hover:shadow-md transition-shadow">
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
            <p className="text-sm text-slate-600 line-clamp-3 mt-2">
              {law.article_content}
            </p>
          ) : law.content ? (
            <p className="text-sm text-slate-600 mt-2">{law.content}</p>
          ) : null}
        </div>
        {law.url ? (
          <a
            href={law.url}
            target="_blank"
            rel="noopener noreferrer"
            className="text-blue-600 hover:text-blue-700 p-1 rounded hover:bg-blue-100 transition-colors"
            title="원문 보기"
          >
            <ExternalLink className="w-4 h-4" />
          </a>
        ) : (
          <button
            className="text-blue-600 hover:text-blue-700 p-1 rounded hover:bg-blue-100 transition-colors"
            title="상세 정보 보기"
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
function PrecedentCard({ precedent }: { precedent: LegalReferenceDetail & { url?: string } }) {
  return (
    <div className="p-4 bg-green-50 border border-green-200 rounded-lg hover:shadow-md transition-shadow">
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
            <p className="text-sm text-slate-700 line-clamp-3 mt-2">
              {precedent.summary}
            </p>
          ) : precedent.content ? (
            <p className="text-sm text-slate-700 mt-2">{precedent.content}</p>
          ) : null}
        </div>
        {precedent.url ? (
          <a
            href={precedent.url}
            target="_blank"
            rel="noopener noreferrer"
            className="text-green-600 hover:text-green-700 p-1 rounded hover:bg-green-100 transition-colors"
            title="원문 보기"
          >
            <ExternalLink className="w-4 h-4" />
          </a>
        ) : (
          <button
            className="text-green-600 hover:text-green-700 p-1 rounded hover:bg-green-100 transition-colors"
            title="상세 정보 보기"
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
function RegulationCard({ regulation }: { regulation: LegalReferenceDetail }) {
  return (
    <div className="p-4 bg-purple-50 border border-purple-200 rounded-lg hover:shadow-md transition-shadow">
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
}: ReferencesModalContentProps) {
  const [selectedType, setSelectedType] = useState<ReferenceType>('all');

  // 참고자료 파싱 및 분류
  const parsedReferences = useMemo(
    () => parseReferences(references, legalReferences, sources),
    [references, legalReferences, sources]
  );

  // 타입별 필터링
  const filteredReferences = useMemo(() => {
    if (selectedType === 'all') {
      return parsedReferences;
    }
    return parsedReferences.filter((ref) => ref.type === selectedType);
  }, [parsedReferences, selectedType]);

  // 타입별 개수 계산
  const counts = useMemo(() => {
    return {
      all: parsedReferences.length,
      law: parsedReferences.filter((r) => r.type === 'law').length,
      precedent: parsedReferences.filter((r) => r.type === 'precedent').length,
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
    <div className="space-y-4">
      {/* 탭/필터 UI */}
      <div className="flex gap-2 border-b border-slate-200 pb-2 overflow-x-auto">
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
      <div className="space-y-3 max-h-[60vh] overflow-y-auto pr-2 -mr-2">
        {filteredReferences.length === 0 ? (
          <div className="text-center py-8 text-slate-500">
            선택한 타입의 참고자료가 없습니다.
          </div>
        ) : (
          filteredReferences.map((ref) => {
            switch (ref.type) {
              case 'law':
                return <LawCard key={ref.id} law={ref} />;
              case 'precedent':
                return <PrecedentCard key={ref.id} precedent={ref} />;
              case 'regulation':
                return <RegulationCard key={ref.id} regulation={ref} />;
              default:
                return null;
            }
          })
        )}
      </div>
    </div>
  );
}

