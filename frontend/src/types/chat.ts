/**
 * 채팅 관련 타입 정의
 */

export interface ChatRequest {
  message: string;
  session_id?: string;
  context?: string;
  enable_checkpoint?: boolean;
  image_base64?: string;
  file_base64?: string;
  filename?: string;
}

export interface SourceInfo {
  name: string;
  type: 'statute_article' | 'case_paragraph' | 'decision_paragraph' | 'interpretation_paragraph' | string;
  url?: string;
  
  // 법령 정보 (최상위 레벨)
  statute_name?: string;
  article_no?: string;
  clause_no?: string;
  item_no?: string;
  
  // 판례 정보 (최상위 레벨)
  case_number?: string;
  case_name?: string;
  court?: string;
  
  // 결정례 정보 (최상위 레벨)
  decision_number?: string;
  org?: string;
  decision_date?: string;
  result?: string;
  
  // 해석례 정보 (최상위 레벨)
  interpretation_number?: string;
  title?: string;
  response_date?: string;
  
  // 상세본문
  content?: string;
  
  // 기존 metadata (하위 호환성 유지)
  metadata?: {
    statute_name?: string;
    article_no?: string;
    clause_no?: string;
    item_no?: string;
    court?: string;
    doc_id?: string;
    casenames?: string;
    org?: string;
    title?: string;
    decision_date?: string;
    response_date?: string;
    result?: string;
    [key: string]: any;
  };
}


export interface ChatResponse {
  answer: string;
  sources: string[];  // 기존 호환성 유지
  sources_detail?: SourceInfo[];  // 신규 필드
  confidence: number;
  legal_references: string[];
  processing_steps: string[];
  session_id: string;
  processing_time: number;
  query_type: string;
  metadata: Record<string, any>;
  errors: string[];
  message_id?: string;
}

export interface StreamingChatRequest {
  message: string;
  session_id?: string;
  context?: string;
  image_base64?: string;
  file_base64?: string;
  filename?: string;
}

export interface ChatMessage {
  id: string;
  role: 'user' | 'assistant' | 'progress';
  content: string;
  timestamp: Date;
  attachments?: FileAttachment[];
  metadata?: {
    sources?: string[];
    sources_detail?: SourceInfo[];  // 신규 필드
    legal_references?: string[];
    confidence?: number;
    processing_steps?: string[];
    query_type?: string;
    // Progress 메시지용 추가 필드
    step?: number;
    message?: string;
    timestamp?: string;
    node_name?: string;
    tokens_received?: number;
    length?: number;
    answer_found?: boolean;
    error?: boolean;
    error_type?: string;
    message_id?: string;
    // 기타 필드 허용
    [key: string]: any;
  };
}


export interface FileAttachment {
  id: string;
  name: string;
  size: number;
  type: string;
  url?: string;
  file?: File;
}

/**
 * 참고자료 상세 정보 타입
 */
export interface LegalReferenceDetail {
  id: string;
  type: 'law' | 'precedent' | 'decision' | 'interpretation' | 'regulation';
  
  // 법령 정보
  law_name?: string;
  article_number?: string;
  article_title?: string;
  article_content?: string;
  
  // 판례 정보
  case_name?: string;
  case_number?: string;
  court?: string;
  decision_date?: string;
  summary?: string;
  
  // 결정례 정보
  decision_number?: string;  // 일련번호 (doc_id)
  org?: string;              // 기관
  result?: string;           // 결과
  
  // 해석례 정보
  interpretation_number?: string;  // 일련번호 (doc_id)
  title?: string;                  // 제목
  response_date?: string;          // 회신일
  
  // 공통
  similarity?: number;
  confidence?: number;
  relevance_score?: number;
  content?: string; // 원본 문자열 (기존 호환용)
}

