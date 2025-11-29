/**
 * 원본 문서 조회 서비스
 */
import axios from 'axios';
import { getEnvironmentDefaults } from '../utils/environment';

// API 기본 URL (api.ts와 동일한 설정 사용)
const envDefaults = getEnvironmentDefaults();
const API_BASE_URL = import.meta.env.VITE_API_BASE_URL || envDefaults.apiBaseUrl;

// documents API는 /api/v1 prefix 없이 직접 사용
const documentsApi = axios.create({
  baseURL: API_BASE_URL,
  timeout: 30000,
  headers: {
    'Content-Type': 'application/json',
  },
});

export interface OriginalDocument {
  id: number;
  title: string;
  text: string;
  source_type: string;
  [key: string]: unknown;
}

export interface ChunkInfo {
  id: number;
  source_type: string;
  source_id: number;
  chunk_index: number;
  text: string;
  chunk_size_category?: string;
  chunking_strategy?: string;
}

export interface ChunksByGroup {
  chunk_group_id: string;
  chunks: ChunkInfo[];
  count: number;
}

/**
 * 원본 문서 조회
 */
export async function fetchOriginalDocument(
  sourceType: string,
  sourceId: number
): Promise<OriginalDocument> {
  const response = await documentsApi.get<OriginalDocument>(
    `/api/documents/original/${sourceType}/${sourceId}`
  );
  return response.data;
}

/**
 * 청크 그룹 ID로 관련 청크 조회
 */
export function fetchChunksByGroup(
  chunkGroupId: string
): Promise<ChunksByGroup> {
  return documentsApi
    .get<ChunksByGroup>(`/api/documents/chunks/${chunkGroupId}`)
    .then((response) => response.data);
}

/**
 * 청크 정보 조회
 */
export function fetchChunkInfo(chunkId: number): Promise<ChunkInfo> {
  return documentsApi
    .get<ChunkInfo>(`/api/documents/chunk/${chunkId}`)
    .then((response) => response.data);
}

