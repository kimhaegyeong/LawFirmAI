/**
 * 채팅 관련 타입 정의
 */

export interface ChatRequest {
  message: string;
  session_id?: string;
  context?: string;
  enable_checkpoint?: boolean;
}

export interface ChatResponse {
  answer: string;
  sources: string[];
  confidence: number;
  legal_references: string[];
  processing_steps: string[];
  session_id: string;
  processing_time: number;
  query_type: string;
  metadata: Record<string, any>;
  errors: string[];
}

export interface StreamingChatRequest {
  message: string;
  session_id?: string;
  context?: string;
}

export interface ChatMessage {
  id: string;
  role: 'user' | 'assistant';
  content: string;
  timestamp: Date;
  attachments?: FileAttachment[];
  metadata?: {
    sources?: string[];
    legal_references?: string[];
    confidence?: number;
    processing_steps?: string[];
    query_type?: string;
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
  type: 'law' | 'precedent' | 'regulation';
  
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
  
  // 공통
  similarity?: number;
  confidence?: number;
  relevance_score?: number;
  content?: string; // 원본 문자열 (기존 호환용)
}

