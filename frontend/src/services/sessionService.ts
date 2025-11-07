/**
 * 세션 서비스
 */
import { api, extractApiError } from './api';
import type {
  Session,
  SessionCreate,
  SessionUpdate,
  SessionListResponse,
  SessionListQuery,
} from '../types/session';

/**
 * 세션 목록 조회
 */
export async function getSessions(query?: SessionListQuery): Promise<SessionListResponse> {
  try {
    const params = new URLSearchParams();
    
    if (query?.category && query.category !== '전체') {
      params.append('category', query.category);
    }
    if (query?.search) {
      params.append('search', query.search);
    }
    if (query?.page) {
      params.append('page', query.page.toString());
    }
    if (query?.page_size) {
      params.append('page_size', query.page_size.toString());
    }
    if (query?.sort_by) {
      params.append('sort_by', query.sort_by);
    }
    if (query?.sort_order) {
      params.append('sort_order', query.sort_order);
    }
    
    const response = await api.get<SessionListResponse>(`/sessions?${params.toString()}`);
    return response.data;
  } catch (error) {
    throw extractApiError(error);
  }
}

/**
 * 세션 생성
 */
export async function createSession(data: SessionCreate): Promise<Session> {
  try {
    const response = await api.post<Session>('/sessions', data);
    return response.data;
  } catch (error) {
    throw extractApiError(error);
  }
}

/**
 * 세션 조회
 */
export async function getSession(sessionId: string): Promise<Session> {
  try {
    const response = await api.get<Session>(`/sessions/${sessionId}`);
    return response.data;
  } catch (error) {
    throw extractApiError(error);
  }
}

/**
 * 세션 업데이트
 */
export async function updateSession(
  sessionId: string,
  data: SessionUpdate
): Promise<Session> {
  try {
    const response = await api.put<Session>(`/sessions/${sessionId}`, data);
    return response.data;
  } catch (error) {
    throw extractApiError(error);
  }
}

/**
 * 세션 삭제
 */
export async function deleteSession(sessionId: string): Promise<void> {
  try {
    await api.delete(`/sessions/${sessionId}`);
  } catch (error) {
    throw extractApiError(error);
  }
}

/**
 * 세션 제목 생성
 */
export async function generateSessionTitle(sessionId: string): Promise<Session> {
  try {
    const response = await api.post<Session>(`/sessions/${sessionId}/generate-title`);
    return response.data;
  } catch (error) {
    throw extractApiError(error);
  }
}

/**
 * 날짜별 그룹 세션 조회
 */
export async function getSessionsByDate(
  dateGroup: 'today' | 'yesterday' | 'week' | 'month' | 'older',
  page: number = 1,
  pageSize: number = 20,
  search?: string
): Promise<SessionListResponse> {
  try {
    const params = new URLSearchParams();
    params.append('date_group', dateGroup);
    params.append('page', page.toString());
    params.append('page_size', pageSize.toString());
    
    if (search) {
      params.append('search', search);
    }
    
    const response = await api.get<SessionListResponse>(`/sessions/by-date?${params.toString()}`);
    return response.data;
  } catch (error) {
    throw extractApiError(error);
  }
}

