/**
 * 법률 정보 URL 생성 유틸리티
 * Open Law API 형식의 URL을 생성합니다.
 */

export type LawUrlType = 'statute' | 'case' | 'decision' | 'interpretation';

/**
 * 조문번호를 6자리 형식으로 변환
 * 예: 제2조 -> 000200, 제10조의2 -> 001002
 */
export function formatArticleNo(articleNo: string | number): string {
  const str = String(articleNo);
  const numbers = str.match(/\d+/g);
  if (!numbers || numbers.length === 0) return '';
  
  const mainNo = parseInt(numbers[0], 10);
  const subNo = numbers.length > 1 ? parseInt(numbers[1], 10) : 0;
  
  return `${String(mainNo).padStart(4, '0')}${String(subNo).padStart(2, '0')}`;
}

/**
 * 법령 조문 URL 생성
 */
export function generateStatuteUrl(metadata: Record<string, unknown>): string | null {
  const baseUrl = 'http://www.law.go.kr/DRF/lawService.do';
  
  const lawId = metadata.law_id || metadata.법령ID || metadata.ID;
  if (lawId) {
    let url = `${baseUrl}?target=eflaw&ID=${lawId}&type=HTML`;
    const articleNo = metadata.article_no || metadata.article_number;
    if (articleNo) {
      const joNo = formatArticleNo(articleNo);
      if (joNo) {
        url += `&JO=${joNo}`;
      }
    }
    return url;
  }
  
  const mst = metadata.mst || metadata.MST || metadata.lsi_seq;
  const effectiveDate = metadata.effective_date || metadata.efYd || metadata.시행일자;
  if (mst && effectiveDate) {
    const efYd = String(effectiveDate).replace(/-/g, '');
    let url = `${baseUrl}?target=eflaw&MST=${mst}&efYd=${efYd}&type=HTML`;
    const articleNo = metadata.article_no || metadata.article_number;
    if (articleNo) {
      const joNo = formatArticleNo(articleNo);
      if (joNo) {
        url += `&JO=${joNo}`;
      }
    }
    return url;
  }
  
  return null;
}

/**
 * 판례 URL 생성
 */
export function generateCaseUrl(metadata: Record<string, unknown>): string | null {
  const baseUrl = 'http://www.law.go.kr/DRF/lawService.do';
  
  const precedentId = 
    metadata.precedent_serial_number || 
    metadata.판례일련번호 || 
    metadata.판례정보일련번호 ||
    metadata.doc_id;
  
  if (precedentId) {
    return `${baseUrl}?target=prec&ID=${precedentId}&type=HTML`;
  }
  
  return null;
}

/**
 * 헌재결정례 URL 생성
 */
export function generateDecisionUrl(metadata: Record<string, unknown>): string | null {
  const baseUrl = 'http://www.law.go.kr/DRF/lawService.do';
  
  const decisionId = 
    metadata.decision_serial_number || 
    metadata.헌재결정례일련번호 || 
    metadata.결정ID ||
    metadata.doc_id;
  
  if (decisionId) {
    return `${baseUrl}?target=detc&ID=${decisionId}&type=HTML`;
  }
  
  return null;
}

/**
 * 법령해석례 URL 생성
 */
export function generateInterpretationUrl(metadata: Record<string, unknown>): string | null {
  const baseUrl = 'http://www.law.go.kr/DRF/lawService.do';
  
  const interpretationId = 
    metadata.interpretation_serial_number || 
    metadata.법령해석례일련번호 || 
    metadata.해석ID ||
    metadata.expcId ||
    metadata.doc_id;
  
  if (interpretationId) {
    return `${baseUrl}?target=expc&ID=${interpretationId}&type=HTML`;
  }
  
  return null;
}

/**
 * 타입별 법률 정보 URL 생성
 */
export function generateLawUrl(
  type: LawUrlType,
  metadata: Record<string, unknown>
): string | null {
  switch (type) {
    case 'statute':
      return generateStatuteUrl(metadata);
    case 'case':
      return generateCaseUrl(metadata);
    case 'decision':
      return generateDecisionUrl(metadata);
    case 'interpretation':
      return generateInterpretationUrl(metadata);
    default:
      return null;
  }
}

/**
 * 검색 링크 생성 (URL이 없을 때 사용)
 */
export function generateSearchUrl(
  type: LawUrlType,
  metadata: Record<string, unknown>
): string | null {
  const baseUrl = 'https://www.law.go.kr';
  
  switch (type) {
    case 'statute': {
      const statuteName = 
        metadata.statute_name || 
        metadata.법령명 || 
        metadata.law_name ||
        metadata.name;
      if (statuteName) {
        return `${baseUrl}/LSW/lsSc.do?query=${encodeURIComponent(String(statuteName))}`;
      }
      break;
    }
    case 'case': {
      const caseName = 
        metadata.casenames || 
        metadata.case_name ||
        metadata.name;
      if (caseName) {
        return `${baseUrl}/DRF/precSearch.do?target=prec&query=${encodeURIComponent(String(caseName))}`;
      }
      break;
    }
    case 'decision': {
      const org = metadata.org || metadata.기관;
      if (org) {
        return `${baseUrl}/LSW/lsSc.do?query=${encodeURIComponent(String(org))}`;
      }
      break;
    }
    case 'interpretation': {
      const title = metadata.title || metadata.안건명;
      if (title) {
        return `${baseUrl}/DRF/expcSearch.do?target=expc&query=${encodeURIComponent(String(title))}`;
      }
      break;
    }
  }
  
  return null;
}

