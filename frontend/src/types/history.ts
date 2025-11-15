/**
 * 히스토리 관련 타입 정의
 */

export interface Message {
  message_id: string;
  session_id: string;
  role: 'user' | 'assistant';
  content: string;
  timestamp: string;
  metadata?: Record<string, unknown>;
}

export interface HistoryQuery {
  session_id?: string;
  category?: string;
  search?: string;
  page?: number;
  page_size?: number;
  sort_by?: string;
  sort_order?: 'asc' | 'desc';
}

export interface HistoryResponse {
  messages: Message[];
  total: number;
  page: number;
  page_size: number;
}

export interface ExportRequest {
  session_ids: string[];
  format: 'json' | 'txt';
}

