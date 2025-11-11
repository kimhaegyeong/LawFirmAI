/**
 * 세션 관련 타입 정의
 */

export interface Session {
  session_id: string;
  title?: string;
  category?: string;
  created_at?: string;
  updated_at?: string;
  message_count: number;
  metadata?: Record<string, unknown>;
}

export interface SessionCreate {
  title?: string;
  category?: string;
}

export interface SessionUpdate {
  title?: string;
  category?: string;
}

export interface SessionListResponse {
  sessions: Session[];
  total: number;
  page: number;
  page_size: number;
}

export type SessionCategory =
  | '전체'
  | '계약/합의'
  | '분쟁/소송'
  | '노동/근로'
  | '부동산'
  | '형사'
  | '기타';

export type SortBy = 'updated_at' | 'created_at' | 'title' | 'message_count';
export type SortOrder = 'asc' | 'desc';

export interface SessionListQuery {
  category?: SessionCategory;
  search?: string;
  page?: number;
  page_size?: number;
  sort_by?: SortBy;
  sort_order?: SortOrder;
}

