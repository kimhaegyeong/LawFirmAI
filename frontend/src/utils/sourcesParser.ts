/**
 * 참고자료 파싱 유틸리티
 * sources, legal_references, sources_detail 파싱 로직 통합
 */
import type { SourceInfo } from '../types/chat';

export interface SourcesMetadata {
  sources?: string[] | unknown;  // deprecated: sources_by_type에서 재구성 가능
  legal_references?: string[] | unknown;  // deprecated: sources_by_type에서 재구성 가능
  sources_detail?: SourceInfo[] | unknown;  // deprecated: sources_by_type에서 재구성 가능
  sources_by_type?: SourcesByType | unknown;  // 유일한 필요한 필드
  message_id?: string;
  related_questions?: string[] | unknown;
  [key: string]: unknown;
}

export interface SourcesByType {
  statute_article: SourceInfo[];
  case_paragraph: SourceInfo[];
  decision_paragraph: SourceInfo[];
  interpretation_paragraph: SourceInfo[];
}

export interface ParseSourcesOptions {
  // 옵션은 현재 사용되지 않음 (향후 확장을 위해 유지)
}

export interface ParsedSources {
  sources: string[];
  legalReferences: string[];  // deprecated: sources_detail에서 자동 추출됨, Phase 4에서 제거 예정
  sourcesDetail: SourceInfo[];
  sourcesByType: SourcesByType;  // 우선 사용 필드
  messageId?: string;
  relatedQuestions: string[];
}

/**
 * 빈 필드를 정리하는 유틸리티 함수
 */
/* eslint-disable security/detect-object-injection */
function cleanEmptyFields(obj: Record<string, unknown>): Record<string, unknown> {
  const cleaned: Record<string, unknown> = {};
  Object.keys(obj).forEach((key) => {
    const value = obj[key];
    if (value !== '' && value !== null && value !== undefined) {
      cleaned[key] = value;
    }
  });
  return cleaned;
}
/* eslint-enable security/detect-object-injection */

/**
 * 판례 URL 생성 (doc_id 형식 처리 개선)
 */
function generateCaseUrl(docId?: string, caseNumber?: string): string | undefined {
  if (!docId && !caseNumber) {
    return undefined;
  }
  
  const identifier = docId || caseNumber;
  if (!identifier) {
    return undefined;
  }
  
  // doc_id가 이미 URL 형식인 경우
  if (identifier.startsWith('http://') || identifier.startsWith('https://')) {
    return identifier;
  }
  
  // case_ 접두사가 있는 경우 제거
  const cleanId = identifier.replace(/^case_/, '');
  
  return `http://www.law.go.kr/DRF/lawService.do?target=prec&ID=${encodeURIComponent(cleanId)}&type=HTML`;
}

/**
 * SourceInfo 데이터 정규화
 * metadata 내부 정보를 최상위 레벨로 이동하고 빈 필드를 정리
 */
function normalizeSourceInfo(source: unknown): SourceInfo | null {
  try {
    if (!source || typeof source !== 'object') {
      return null;
    }
    
    const src = source as Record<string, unknown>;
    
    // 최소 필수 필드 확인 (name 또는 type이 없어도 기본값으로 처리)
    const name = src.name ? String(src.name) : '';
    const type = src.type ? String(src.type) : 'regulation';
    
    // name과 type이 모두 없으면 null 반환
    if (!name && !type) {
      return null;
    }
    
    const normalized: SourceInfo = {
      name: name || '참고자료',
      type: type,
    };
    
    // 기본 필드 복사 (빈 값 제외)
    if (src.url && typeof src.url === 'string' && src.url.trim()) {
      normalized.url = src.url.trim();
    }
    
    if (src.content && typeof src.content === 'string' && src.content.trim()) {
      normalized.content = src.content.trim();
    }
    
    // 타입별 필드 정규화 (metadata에서 최상위 레벨로 이동)
    const meta = (src.metadata && typeof src.metadata === 'object') 
      ? src.metadata as Record<string, unknown>
      : {};
    
    // 판례 정보 정규화
    if (src.type === 'case_paragraph' || String(src.type).includes('case')) {
      normalized.case_number = (src.case_number as string) || 
                                 (meta.doc_id as string) || 
                                 (meta.case_number as string) ||
                                 undefined;
      normalized.case_name = (src.case_name as string) || 
                             (meta.casenames as string) || 
                             (meta.case_name as string) ||
                             undefined;
      normalized.court = (src.court as string) || 
                         (meta.court as string) || 
                         undefined;
      
      // announce_date 처리 (판결일)
      if (src.decision_date && typeof src.decision_date === 'string') {
        normalized.decision_date = src.decision_date;
      } else if (meta.announce_date && typeof meta.announce_date === 'string') {
        normalized.decision_date = meta.announce_date;
      } else if (meta.decision_date && typeof meta.decision_date === 'string') {
        normalized.decision_date = meta.decision_date;
      }
      
      // URL 생성 (url이 없고 doc_id나 case_number가 있는 경우)
      if (!normalized.url) {
        const docId = (meta.doc_id as string) || normalized.case_number;
        normalized.url = generateCaseUrl(docId, normalized.case_number);
      }
    }
    
    // 법령 정보 정규화
    if (src.type === 'statute_article' || String(src.type).includes('statute')) {
      normalized.statute_name = (src.statute_name as string) || 
                                 (meta.statute_name as string) || 
                                 (meta.law_name as string) ||
                                 undefined;
      normalized.article_no = (src.article_no as string) || 
                              (meta.article_no as string) || 
                              (meta.article_number as string) ||
                              undefined;
      if (src.clause_no || meta.clause_no) {
        normalized.clause_no = (src.clause_no as string) || (meta.clause_no as string);
      }
      if (src.item_no || meta.item_no) {
        normalized.item_no = (src.item_no as string) || (meta.item_no as string);
      }
    }
    
    // 결정례 정보 정규화
    if (src.type === 'decision_paragraph' || String(src.type).includes('decision')) {
      normalized.decision_number = (src.decision_number as string) || 
                                    (meta.doc_id as string) || 
                                    (meta.decision_id as string) ||
                                    undefined;
      normalized.org = (src.org as string) || 
                       (meta.org as string) || 
                       undefined;
      if (src.decision_date || meta.decision_date || meta.announce_date) {
        normalized.decision_date = (src.decision_date as string) || 
                                    (meta.decision_date as string) || 
                                    (meta.announce_date as string) ||
                                    undefined;
      }
      if (src.result || meta.result) {
        normalized.result = (src.result as string) || (meta.result as string);
      }
    }
    
    // 해석례 정보 정규화
    if (src.type === 'interpretation_paragraph' || String(src.type).includes('interpretation')) {
      normalized.interpretation_number = (src.interpretation_number as string) || 
                                          (meta.doc_id as string) || 
                                          (meta.interpretation_id as string) ||
                                          undefined;
      normalized.title = (src.title as string) || 
                         (meta.title as string) || 
                         undefined;
      normalized.org = (src.org as string) || 
                       (meta.org as string) || 
                       undefined;
      if (src.response_date || meta.response_date || meta.announce_date) {
        normalized.response_date = (src.response_date as string) || 
                                    (meta.response_date as string) || 
                                    (meta.announce_date as string) ||
                                    undefined;
      }
    }
    
    // metadata 객체는 하위 호환성을 위해 최소한만 유지 (빈 필드 제거)
    const cleanedMetadata = cleanEmptyFields(meta);
    if (Object.keys(cleanedMetadata).length > 0) {
      normalized.metadata = cleanedMetadata as SourceInfo['metadata'];
    }
    
    // metadata가 없어도 최상위 레벨 정보는 복사
    if (!src.metadata || typeof src.metadata !== 'object') {
      // metadata가 없어도 최상위 레벨 정보는 복사
      if (src.case_number && typeof src.case_number === 'string') {
        normalized.case_number = src.case_number;
      }
      if (src.case_name && typeof src.case_name === 'string') {
        normalized.case_name = src.case_name;
      }
      if (src.court && typeof src.court === 'string') {
        normalized.court = src.court;
      }
      if (src.statute_name && typeof src.statute_name === 'string') {
        normalized.statute_name = src.statute_name;
      }
      if (src.article_no && typeof src.article_no === 'string') {
        normalized.article_no = src.article_no;
      }
      
      // URL이 없고 case_number가 있으면 생성
      if (!normalized.url && normalized.case_number) {
        normalized.url = generateCaseUrl(undefined, normalized.case_number);
      }
    }
    
    // 빈 문자열 필드 제거
    const normalizedRecord = normalized as unknown as Record<string, unknown>;
    const cleaned = cleanEmptyFields(normalizedRecord);
    
    // 필수 필드는 항상 포함
    return {
      name: normalized.name,
      type: normalized.type,
      ...cleaned,
    } as SourceInfo;
  } catch (error) {
    // 에러 발생 시에도 기본 정보는 반환
    const src = source as Record<string, unknown>;
    return {
      name: (src.name as string) || '참고자료',
      type: (src.type as string) || 'regulation',
      content: (src.content as string) || '',
    } as SourceInfo;
  }
}

/**
 * sources_detail을 타입별로 그룹화
 */
export function getSourcesByType(
  sourcesDetail: SourceInfo[]
): SourcesByType {
  const grouped: SourcesByType = {
    statute_article: [],
    case_paragraph: [],
    decision_paragraph: [],
    interpretation_paragraph: [],
  };
  
  for (const detail of sourcesDetail) {
    const type = detail.type;
    if (type in grouped) {
      grouped[type as keyof SourcesByType].push(detail);
    }
  }
  
  return grouped;
}

/**
 * sources_by_type에서 sources_detail 재구성
 */
export function getSourcesDetailFromSourcesByType(sourcesByType: SourcesByType): SourceInfo[] {
  return [
    ...sourcesByType.statute_article,
    ...sourcesByType.case_paragraph,
    ...sourcesByType.decision_paragraph,
    ...sourcesByType.interpretation_paragraph,
  ];
}

/**
 * sources_by_type에서 sources (문자열 배열) 재구성
 */
export function getSourcesFromSourcesByType(sourcesByType: SourcesByType): string[] {
  const allDetails = getSourcesDetailFromSourcesByType(sourcesByType);
  return allDetails
    .map((detail) => detail.name || '')
    .filter((name) => name.trim().length > 0);
}

/**
 * sources_detail에서 legal_references 추출 (deprecated용)
 */
export function extractLegalReferencesFromSourcesDetail(
  sourcesDetail: SourceInfo[]
): string[] {
  const legalRefs: string[] = [];
  const seen = new Set<string>();
  
  for (const detail of sourcesDetail) {
    if (detail.type !== 'statute_article') {
      continue;
    }
    
    const statuteName = detail.statute_name || '';
    const articleNo = detail.article_no || '';
    const clauseNo = detail.clause_no || '';
    const itemNo = detail.item_no || '';
    
    if (!statuteName && !articleNo) {
      continue;
    }
    
    const parts: string[] = [];
    if (statuteName) parts.push(statuteName);
    if (articleNo) parts.push(articleNo);
    if (clauseNo) parts.push(`제${clauseNo}항`);
    if (itemNo) parts.push(`제${itemNo}호`);
    
    const legalRef = parts.join(' ');
    if (legalRef && !seen.has(legalRef)) {
      legalRefs.push(legalRef);
      seen.add(legalRef);
    }
  }
  
  return legalRefs;
}

/**
 * 참고자료 메타데이터를 안전하게 파싱
 */
/**
 * sources_by_type을 우선 사용하고, 없으면 sources_detail에서 생성
 * sources_detail도 없으면 빈 구조 반환
 * sources_by_type이 있으면 항상 sourcesDetail을 재구성하여 반환
 * 
 * @param metadata - 소스 메타데이터
 * @param options - 파싱 옵션 (현재 사용되지 않음)
 */
export function parseSourcesMetadata(
  metadata: SourcesMetadata | undefined,
  _options?: ParseSourcesOptions
): ParsedSources {
  if (!metadata) {
    return {
      sources: [],
      legalReferences: [],
      sourcesDetail: [],
      sourcesByType: {
        statute_article: [],
        case_paragraph: [],
        decision_paragraph: [],
        interpretation_paragraph: [],
      },
      relatedQuestions: [],
    };
  }

  // sources_by_type을 우선 사용 (유일한 필요한 필드)
  let sourcesByType: SourcesByType;
  const sourcesDetailFromMetadata = Array.isArray(metadata.sources_detail)
    ? metadata.sources_detail
        .map(normalizeSourceInfo)
        .filter((s): s is SourceInfo => s !== null)
    : [];
  
  if (metadata.sources_by_type && typeof metadata.sources_by_type === 'object') {
    const byType = metadata.sources_by_type as Record<string, unknown>;
    const parsedByType = {
      statute_article: Array.isArray(byType.statute_article)
        ? byType.statute_article.map(normalizeSourceInfo).filter((s): s is SourceInfo => s !== null)
        : [],
      case_paragraph: Array.isArray(byType.case_paragraph)
        ? byType.case_paragraph.map(normalizeSourceInfo).filter((s): s is SourceInfo => s !== null)
        : [],
      decision_paragraph: Array.isArray(byType.decision_paragraph)
        ? byType.decision_paragraph.map(normalizeSourceInfo).filter((s): s is SourceInfo => s !== null)
        : [],
      interpretation_paragraph: Array.isArray(byType.interpretation_paragraph)
        ? byType.interpretation_paragraph.map(normalizeSourceInfo).filter((s): s is SourceInfo => s !== null)
        : [],
    };
    
    // sources_by_type이 유효한지 확인 (최소한 하나의 타입이라도 데이터가 있어야 함)
    const hasValidData = 
      parsedByType.statute_article.length > 0 ||
      parsedByType.case_paragraph.length > 0 ||
      parsedByType.decision_paragraph.length > 0 ||
      parsedByType.interpretation_paragraph.length > 0;
    
    if (hasValidData) {
      sourcesByType = parsedByType;
    } else {
      // sources_by_type이 비어있거나 invalid하면 sources_detail에서 생성 (하위 호환성)
      sourcesByType = getSourcesByType(sourcesDetailFromMetadata);
    }
  } else {
    // sources_by_type이 없으면 sources_detail에서 생성 (하위 호환성)
    sourcesByType = getSourcesByType(sourcesDetailFromMetadata);
  }
  
  // sourcesDetail은 항상 sourcesByType에서 재구성 (sources_by_type이 우선이므로)
  const sourcesDetail = getSourcesDetailFromSourcesByType(sourcesByType);
  
  // sources 재구성 (하위 호환성)
  const sources = Array.isArray(metadata.sources) 
    ? metadata.sources.filter((s): s is string => typeof s === 'string' && s.trim().length > 0)
    : getSourcesFromSourcesByType(sourcesByType);
  
  // legal_references는 sources_detail에서 추출 (deprecated)
  const extractedLegalRefs = extractLegalReferencesFromSourcesDetail(sourcesDetail);
  const existingLegalRefs = Array.isArray(metadata.legal_references)
    ? metadata.legal_references.filter((r): r is string => typeof r === 'string' && r.trim().length > 0)
    : [];
  const legalReferences = [...new Set([...existingLegalRefs, ...extractedLegalRefs])];
  
  const relatedQuestions = Array.isArray(metadata.related_questions)
    ? metadata.related_questions.filter((q): q is string => typeof q === 'string' && q.trim().length > 0)
    : [];
  
  const messageId = typeof metadata.message_id === 'string' ? metadata.message_id : undefined;

  // 데이터 일관성 검증: sources와 sources_detail 개수 불일치 시 경고
  if (sources.length > 0 && sourcesDetail.length > 0 && sources.length !== sourcesDetail.length) {
    if (typeof console !== 'undefined' && console.warn) {
      console.warn('[parseSourcesMetadata] Sources count mismatch:', {
        sourcesCount: sources.length,
        sourcesDetailCount: sourcesDetail.length,
        messageId,
      });
    }
  }

  return {
    sources,
    legalReferences,  // deprecated: sources_detail에서 자동 추출됨, Phase 4에서 제거 예정
    sourcesDetail,
    sourcesByType,  // 새로운 필드
    messageId,
    relatedQuestions,
  };
}

/**
 * 기존 메타데이터와 새 참고자료를 병합
 * - 새 메타데이터가 있으면 우선 사용
 * - 새 메타데이터가 비어있으면 기존 값 유지
 * - related_questions는 중복 제거 후 병합
 */
export function mergeSourcesMetadata(
  existingMetadata: Record<string, unknown> | undefined,
  newSources: ParsedSources
): Record<string, unknown> {
  const existingRelatedQuestions = Array.isArray(existingMetadata?.related_questions)
    ? existingMetadata.related_questions.filter((q): q is string => typeof q === 'string' && q.trim().length > 0)
    : [];
  
  // related_questions 중복 제거 및 병합
  const mergedRelatedQuestions = newSources.relatedQuestions.length > 0
    ? (() => {
        const newSet = new Set(newSources.relatedQuestions.map(q => q.trim().toLowerCase()));
        const existingFiltered = existingRelatedQuestions.filter(q => !newSet.has(q.trim().toLowerCase()));
        return [...newSources.relatedQuestions, ...existingFiltered];
      })()
    : existingRelatedQuestions;
  
  return {
    ...existingMetadata,
    sources_by_type: newSources.sourcesByType,  // sources_by_type 직접 저장
    sources: newSources.sources.length > 0 ? newSources.sources : (Array.isArray(existingMetadata?.sources) && existingMetadata.sources.length > 0 ? existingMetadata.sources : []),
    legal_references: newSources.legalReferences.length > 0 ? newSources.legalReferences : (Array.isArray(existingMetadata?.legal_references) && existingMetadata.legal_references.length > 0 ? existingMetadata.legal_references : []),
    sources_detail: newSources.sourcesDetail.length > 0 ? newSources.sourcesDetail : (Array.isArray(existingMetadata?.sources_detail) && existingMetadata.sources_detail.length > 0 ? existingMetadata.sources_detail : []),
    message_id: newSources.messageId || existingMetadata?.message_id,
    related_questions: mergedRelatedQuestions.length > 0 ? mergedRelatedQuestions : [],
  };
}

