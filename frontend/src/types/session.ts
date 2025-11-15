/**
 * 세션 관련 타입 정의
 */

export interface Session {
  session_id: string;
  title?: string;
  created_at?: string;
  updated_at?: string;
  message_count: number;
  metadata?: Record<string, unknown>;
}

export interface SessionCreate {
  title?: string;
}

export interface SessionUpdate {
  title?: string;
}

export interface SessionListResponse {
  sessions: Session[];
  total: number;
  page: number;
  page_size: number;
}

export type SortBy = 'updated_at' | 'created_at' | 'title' | 'message_count';
export type SortOrder = 'asc' | 'desc';

export interface SessionListQuery {
  search?: string;
  page?: number;
  page_size?: number;
  sort_by?: SortBy;
  sort_order?: SortOrder;
}

