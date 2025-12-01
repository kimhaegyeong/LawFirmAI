/**
 * 원본 문서 뷰어 컴포넌트
 */
import { useState, useEffect } from 'react';
import { X, FileText, Loader2 } from 'lucide-react';
import { fetchOriginalDocument, fetchChunksByGroup, OriginalDocument, ChunksByGroup } from '../../services/documentService';
import { SourceInfo } from '../../types/chat';

interface OriginalDocumentViewerProps {
  source: SourceInfo;
  onClose: () => void;
}

export function OriginalDocumentViewer({ source, onClose }: OriginalDocumentViewerProps) {
  const [document, setDocument] = useState<OriginalDocument | null>(null);
  const [chunks, setChunks] = useState<ChunksByGroup | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [selectedChunkIndex, setSelectedChunkIndex] = useState<number | null>(null);

  useEffect(() => {
    const loadDocument = async () => {
      if (!source.original_url) {
        setError('원본 문서 URL이 없습니다.');
        setLoading(false);
        return;
      }

      try {
        setLoading(true);
        setError(null);

        // URL에서 source_type과 source_id 추출
        const urlMatch = source.original_url.match(/\/api\/documents\/original\/([^/]+)\/(\d+)/);
        if (!urlMatch) {
          throw new Error('잘못된 URL 형식입니다.');
        }

        const [, sourceType, sourceIdStr] = urlMatch;
        if (!sourceIdStr) {
          throw new Error('잘못된 URL 형식입니다.');
        }
        const sourceId = parseInt(sourceIdStr, 10);

        if (isNaN(sourceId)) {
          throw new Error('잘못된 문서 ID입니다.');
        }

        if (!sourceType) {
          throw new Error('잘못된 문서 타입입니다.');
        }

        // 원본 문서 조회
        const doc = await fetchOriginalDocument(sourceType, sourceId);
        setDocument(doc);

        // 하이브리드 청킹인 경우 청크 그룹 조회
        // metadata에서 chunk_group_id 확인
        const chunkGroupId = source.metadata?.chunk_group_id;
        if (chunkGroupId && typeof chunkGroupId === 'string') {
          try {
            const chunksData = await fetchChunksByGroup(chunkGroupId);
            setChunks(chunksData);
          } catch (err) {
            // 청크 그룹 조회 실패는 무시 (선택적 기능)
            console.warn('Failed to load chunks:', err);
          }
        }
      } catch (err) {
        setError(err instanceof Error ? err.message : '문서를 불러오는 중 오류가 발생했습니다.');
      } finally {
        setLoading(false);
      }
    };

    loadDocument();
  }, [source]);

  const handleChunkSelect = (index: number) => {
    setSelectedChunkIndex(index === selectedChunkIndex ? null : index);
  };

  if (loading) {
    return (
      <div className="fixed inset-0 bg-black bg-opacity-50 flex items-center justify-center z-50">
        <div className="bg-white rounded-lg p-8 max-w-md w-full mx-4">
          <div className="flex items-center justify-center gap-3">
            <Loader2 className="w-6 h-6 animate-spin text-blue-600" />
            <span className="text-lg">문서를 불러오는 중...</span>
          </div>
        </div>
      </div>
    );
  }

  if (error) {
    return (
      <div className="fixed inset-0 bg-black bg-opacity-50 flex items-center justify-center z-50">
        <div className="bg-white rounded-lg p-6 max-w-md w-full mx-4">
          <div className="flex items-center justify-between mb-4">
            <h2 className="text-xl font-bold">오류</h2>
            <button
              onClick={onClose}
              className="text-gray-500 hover:text-gray-700"
            >
              <X className="w-5 h-5" />
            </button>
          </div>
          <p className="text-red-600">{error}</p>
          <button
            onClick={onClose}
            className="mt-4 px-4 py-2 bg-blue-600 text-white rounded hover:bg-blue-700"
          >
            닫기
          </button>
        </div>
      </div>
    );
  }

  return (
    <div className="fixed inset-0 bg-black bg-opacity-50 flex items-center justify-center z-50 p-4">
      <div className="bg-white rounded-lg shadow-xl max-w-4xl w-full max-h-[90vh] flex flex-col">
        {/* 헤더 */}
        <div className="flex items-center justify-between p-4 border-b">
          <div className="flex items-center gap-2">
            <FileText className="w-5 h-5 text-blue-600" />
            <h2 className="text-xl font-bold">{document?.title || source.name}</h2>
          </div>
          <button
            onClick={onClose}
            className="text-gray-500 hover:text-gray-700"
          >
            <X className="w-5 h-5" />
          </button>
        </div>

        {/* 내용 */}
        <div className="flex-1 overflow-y-auto p-6">
          {chunks && chunks.chunks.length > 0 && (
            <div className="mb-6">
              <h3 className="text-lg font-semibold mb-3">청크 선택 (하이브리드 청킹)</h3>
              <div className="flex flex-wrap gap-2 mb-4">
                {chunks.chunks.map((chunk, index) => (
                  <button
                    key={chunk.id}
                    onClick={() => handleChunkSelect(index)}
                    className={`px-3 py-1 rounded text-sm border transition-colors ${
                      selectedChunkIndex === index
                        ? 'bg-blue-600 text-white border-blue-600'
                        : 'bg-white text-gray-700 border-gray-300 hover:bg-gray-50'
                    }`}
                  >
                    {chunk.chunk_size_category || '청크'} {index + 1}
                  </button>
                ))}
              </div>
            </div>
          )}

          <div className="prose max-w-none">
            {selectedChunkIndex !== null && chunks?.chunks && 
             selectedChunkIndex >= 0 && selectedChunkIndex < chunks.chunks.length ? (
              <div>
                {(() => {
                  // 안전한 배열 접근: 인덱스 범위가 이미 검증됨
                  const chunksArray = chunks.chunks;
                  const validIndex = selectedChunkIndex;
                  // 배열 접근은 이미 범위 검증이 완료되었으므로 안전함
                  let selectedChunk;
                  if (validIndex >= 0 && validIndex < chunksArray.length) {
                    // eslint-disable-next-line security/detect-object-injection
                    selectedChunk = chunksArray[validIndex];
                  } else {
                    selectedChunk = null;
                  }
                  return (
                    <>
                      <h4 className="text-md font-semibold mb-2">
                        선택된 청크 ({selectedChunk?.chunk_size_category || '청크'})
                      </h4>
                      <div className="bg-gray-50 p-4 rounded border">
                        <pre className="whitespace-pre-wrap text-sm">{selectedChunk?.text || ''}</pre>
                      </div>
                    </>
                  );
                })()}
              </div>
            ) : (
              <div>
                <h4 className="text-md font-semibold mb-2">원본 문서</h4>
                <div className="bg-gray-50 p-4 rounded border">
                  <pre className="whitespace-pre-wrap text-sm">{document?.text || '내용이 없습니다.'}</pre>
                </div>
              </div>
            )}
          </div>
        </div>

        {/* 푸터 */}
        <div className="p-4 border-t flex justify-end">
          <button
            onClick={onClose}
            className="px-4 py-2 bg-blue-600 text-white rounded hover:bg-blue-700"
          >
            닫기
          </button>
        </div>
      </div>
    </div>
  );
}

