/**
 * 히스토리 서비스
 */
import { api, extractApiError } from './api';
import type { HistoryQuery, HistoryResponse, ExportRequest } from '../types/history';

/**
 * 히스토리 조회
 */
export async function getHistory(query: HistoryQuery): Promise<HistoryResponse> {
  try {
    const params = new URLSearchParams();
    
    if (query.session_id) {
      params.append('session_id', query.session_id);
    }
    if (query.category) {
      params.append('category', query.category);
    }
    if (query.search) {
      params.append('search', query.search);
    }
    if (query.page) {
      params.append('page', query.page.toString());
    }
    if (query.page_size) {
      params.append('page_size', query.page_size.toString());
    }
    if (query.sort_by) {
      params.append('sort_by', query.sort_by);
    }
    if (query.sort_order) {
      params.append('sort_order', query.sort_order);
    }
    
    const response = await api.get<HistoryResponse>(`/history?${params.toString()}`);
    return response.data;
  } catch (error) {
    throw extractApiError(error);
  }
}

/**
 * 히스토리 내보내기
 */
export async function exportHistory(request: ExportRequest): Promise<Blob> {
  try {
    const response = await api.post('/history/export', request, {
      responseType: 'blob',
    });
    
    return response.data;
  } catch (error) {
    throw extractApiError(error);
  }
}

/**
 * 내보내기 파일 다운로드
 */
export function downloadHistory(data: Blob, filename: string): void {
  const url = window.URL.createObjectURL(data);
  const link = document.createElement('a');
  link.href = url;
  link.download = filename;
  document.body.appendChild(link);
  link.click();
  document.body.removeChild(link);
  window.URL.revokeObjectURL(url);
}

