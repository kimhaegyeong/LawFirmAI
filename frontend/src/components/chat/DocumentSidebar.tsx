/**
 * 문서 상세 정보 사이드바 컴포넌트
 * 오른쪽에서 슬라이드되는 사이드바로 문서 상세 정보를 표시합니다.
 */
import { X, FileText, Scale, Bookmark, ExternalLink, Loader2 } from 'lucide-react';
import { useEffect, useState } from 'react';
import { getChatSources } from '../../services/chatService';
import logger from '../../utils/logger';
import type { SourceInfo } from '../../types/chat';

interface DocumentSidebarProps {
  isOpen: boolean;
  onClose: () => void;
  documentIndex: number | null;
  sources?: string[];
  sourcesDetail?: SourceInfo[];
  metadata?: any;
  sessionId?: string;
  messageId?: string;
}

export function DocumentSidebar({
  isOpen,
  onClose,
  documentIndex,
  sources = [],
  sourcesDetail = [],
  metadata,
  sessionId,
  messageId
}: DocumentSidebarProps) {
  const [loadingSources, setLoadingSources] = useState(false);
  const [loadedSources, setLoadedSources] = useState<string[]>(sources);
  const [loadedSourcesDetail, setLoadedSourcesDetail] = useState<SourceInfo[]>(sourcesDetail);
  
  // sources가 없으면 API로 가져오기
  useEffect(() => {
    if (isOpen && (!sources.length && !sourcesDetail.length) && sessionId) {
      setLoadingSources(true);
      getChatSources(sessionId, messageId)
        .then((sourcesData) => {
          setLoadedSources(sourcesData.sources);
          setLoadedSourcesDetail(sourcesData.sources_detail);
        })
        .catch((error) => {
          logger.error('[DocumentSidebar] Error fetching sources:', error);
        })
        .finally(() => {
          setLoadingSources(false);
        });
    } else {
      setLoadedSources(sources);
      setLoadedSourcesDetail(sourcesDetail);
    }
  }, [isOpen, sources, sourcesDetail, sessionId, messageId]);
  
  // ESC 키로 닫기
  useEffect(() => {
    const handleEscape = (e: KeyboardEvent) => {
      if (e.key === 'Escape' && isOpen) {
        onClose();
      }
    };
    
    if (isOpen) {
      document.addEventListener('keydown', handleEscape);
      document.body.style.overflow = 'hidden';
    }
    
    return () => {
      document.removeEventListener('keydown', handleEscape);
      document.body.style.overflow = 'unset';
    };
  }, [isOpen, onClose]);

  if (!isOpen || documentIndex === null) {
    return null;
  }

  const source = loadedSources[documentIndex];
  const sourceDetail = loadedSourcesDetail[documentIndex];

  // 로딩 중일 때
  if (loadingSources) {
    return (
      <>
        {/* 오버레이 제거 (dim 처리 안함) */}
        <div
          className={`fixed right-0 top-0 h-full w-full max-w-md bg-white shadow-xl z-50 transform transition-transform duration-300 ease-in-out ${
            isOpen ? 'translate-x-0' : 'translate-x-full'
          }`}
          role="dialog"
          aria-modal="true"
          aria-labelledby="document-sidebar-title"
        >
          <div className="flex flex-col h-full">
            <div className="flex items-center justify-between p-4 border-b border-slate-200 bg-slate-50">
              <h2 id="document-sidebar-title" className="text-lg font-semibold text-slate-800">
                문서 {documentIndex + 1}
              </h2>
              <button
                onClick={onClose}
                className="p-1.5 rounded-md hover:bg-slate-200 transition-colors text-slate-600"
                aria-label="사이드바 닫기"
              >
                <X className="w-5 h-5" />
              </button>
            </div>
            <div className="flex-1 flex items-center justify-center">
              <div className="flex items-center gap-2 text-slate-500">
                <Loader2 className="w-5 h-5 animate-spin" />
                <span>참조자료를 불러오는 중...</span>
              </div>
            </div>
          </div>
        </div>
      </>
    );
  }

  // 문서 정보가 없으면 표시하지 않음
  if (!source && !sourceDetail) {
    return (
      <>
        {/* 오버레이 제거 (dim 처리 안함) */}
        <div
          className={`fixed right-0 top-0 h-full w-full max-w-md bg-white shadow-xl z-50 transform transition-transform duration-300 ease-in-out ${
            isOpen ? 'translate-x-0' : 'translate-x-full'
          }`}
          role="dialog"
          aria-modal="true"
          aria-labelledby="document-sidebar-title"
        >
          <div className="flex flex-col h-full">
            <div className="flex items-center justify-between p-4 border-b border-slate-200 bg-slate-50">
              <h2 id="document-sidebar-title" className="text-lg font-semibold text-slate-800">
                문서 {documentIndex + 1}
              </h2>
              <button
                onClick={onClose}
                className="p-1.5 rounded-md hover:bg-slate-200 transition-colors text-slate-600"
                aria-label="사이드바 닫기"
              >
                <X className="w-5 h-5" />
              </button>
            </div>
            <div className="flex-1 flex items-center justify-center">
              <div className="text-slate-500 text-sm">
                참조자료 정보를 찾을 수 없습니다.
              </div>
            </div>
          </div>
        </div>
      </>
    );
  }

  // 문서 타입 결정
  const getDocumentType = (): 'law' | 'precedent' | 'decision' | 'interpretation' | 'regulation' => {
    if (sourceDetail) {
      if (sourceDetail.type === 'statute_article') return 'law';
      if (sourceDetail.type === 'case_paragraph') return 'precedent';
      if (sourceDetail.type === 'decision_paragraph') return 'decision';
      if (sourceDetail.type === 'interpretation_paragraph') return 'interpretation';
      return 'regulation';
    }
    
    if (source) {
      const isPrecedent = /대법원|고등법원|지방법원|판결|선고|\d{4}[가-힣]\d+/.test(source);
      if (isPrecedent) return 'precedent';
      return 'law';
    }
    
    return 'regulation';
  };

  const documentType = getDocumentType();

  // 문서 제목/이름
  const documentName = sourceDetail?.name || source || `문서 ${documentIndex + 1}`;

  // 문서 메타데이터
  const docMetadata = sourceDetail?.metadata || {};

  return (
    <>
      {/* 오버레이 제거 (dim 처리 안함) */}

      {/* 사이드바 */}
      <div
        className={`fixed right-0 top-0 h-full w-full max-w-md bg-white shadow-xl z-50 transform transition-transform duration-300 ease-in-out ${
          isOpen ? 'translate-x-0' : 'translate-x-full'
        }`}
        role="dialog"
        aria-modal="true"
        aria-labelledby="document-sidebar-title"
      >
        <div className="flex flex-col h-full">
          {/* 헤더 */}
          <div className="flex items-center justify-between p-4 border-b border-slate-200 bg-slate-50">
            <div className="flex items-center gap-2">
              {documentType === 'law' && <FileText className="w-5 h-5 text-blue-600" />}
              {documentType === 'precedent' && <Scale className="w-5 h-5 text-green-600" />}
              {documentType === 'decision' && <Bookmark className="w-5 h-5 text-orange-600" />}
              {documentType === 'interpretation' && <FileText className="w-5 h-5 text-indigo-600" />}
              {documentType === 'regulation' && <Bookmark className="w-5 h-5 text-purple-600" />}
              <h2 id="document-sidebar-title" className="text-lg font-semibold text-slate-800">
                문서 {documentIndex + 1}
              </h2>
            </div>
            <button
              onClick={onClose}
              className="p-1.5 rounded-md hover:bg-slate-200 transition-colors text-slate-600"
              aria-label="사이드바 닫기"
            >
              <X className="w-5 h-5" />
            </button>
          </div>

          {/* 본문 */}
          <div className="flex-1 overflow-y-auto p-4">
            <div className="space-y-4">
              {/* 문서 타입 배지 */}
              <div className="flex items-center gap-2">
                <span
                  className={`text-xs font-semibold px-2 py-1 rounded ${
                    documentType === 'law'
                      ? 'bg-blue-100 text-blue-700'
                      : documentType === 'precedent'
                      ? 'bg-green-100 text-green-700'
                      : documentType === 'decision'
                      ? 'bg-orange-100 text-orange-700'
                      : documentType === 'interpretation'
                      ? 'bg-indigo-100 text-indigo-700'
                      : 'bg-purple-100 text-purple-700'
                  }`}
                >
                  {documentType === 'law' ? '법령' : 
                   documentType === 'precedent' ? '판례' : 
                   documentType === 'decision' ? '결정례' : 
                   documentType === 'interpretation' ? '해석례' : 
                   '기타'}
                </span>
              </div>

              {/* 문서 제목/이름 */}
              <div>
                <h3 className="text-base font-semibold text-slate-800 mb-2">{documentName}</h3>
              </div>

              {/* 법령 정보 */}
              {documentType === 'law' && (
                <div className="space-y-3">
                  {(sourceDetail?.statute_name || docMetadata.statute_name) && (
                    <div>
                      <span className="text-xs font-medium text-slate-500">법령명</span>
                      <p className="text-sm text-slate-800 mt-1">
                        {sourceDetail?.statute_name || docMetadata.statute_name}
                      </p>
                    </div>
                  )}
                  {(sourceDetail?.article_no || docMetadata.article_no) && (
                    <div>
                      <span className="text-xs font-medium text-slate-500">조문</span>
                      <p className="text-sm text-slate-800 mt-1">
                        제{sourceDetail?.article_no || docMetadata.article_no}조
                      </p>
                    </div>
                  )}
                  {(sourceDetail?.clause_no || docMetadata.clause_no) && (
                    <div>
                      <span className="text-xs font-medium text-slate-500">항</span>
                      <p className="text-sm text-slate-800 mt-1">
                        제{sourceDetail?.clause_no || docMetadata.clause_no}항
                      </p>
                    </div>
                  )}
                  {(sourceDetail?.item_no || docMetadata.item_no) && (
                    <div>
                      <span className="text-xs font-medium text-slate-500">호</span>
                      <p className="text-sm text-slate-800 mt-1">
                        제{sourceDetail?.item_no || docMetadata.item_no}호
                      </p>
                    </div>
                  )}
                  {docMetadata.title && (
                    <div>
                      <span className="text-xs font-medium text-slate-500">제목</span>
                      <p className="text-sm text-slate-800 mt-1">{docMetadata.title}</p>
                    </div>
                  )}
                </div>
              )}

              {/* 판례 정보 */}
              {documentType === 'precedent' && (
                <div className="space-y-3">
                  {(sourceDetail?.case_name || docMetadata.casenames) && (
                    <div>
                      <span className="text-xs font-medium text-slate-500">사건명</span>
                      <p className="text-sm text-slate-800 mt-1">
                        {sourceDetail?.case_name || docMetadata.casenames}
                      </p>
                    </div>
                  )}
                  {(sourceDetail?.court || docMetadata.court) && (
                    <div>
                      <span className="text-xs font-medium text-slate-500">법원</span>
                      <p className="text-sm text-slate-800 mt-1">
                        {sourceDetail?.court || docMetadata.court}
                      </p>
                    </div>
                  )}
                  {(sourceDetail?.case_number || docMetadata.doc_id) && (
                    <div>
                      <span className="text-xs font-medium text-slate-500">사건번호</span>
                      <p className="text-sm text-slate-800 mt-1">
                        {sourceDetail?.case_number || docMetadata.doc_id}
                      </p>
                    </div>
                  )}
                </div>
              )}

              {/* 결정례 정보 */}
              {documentType === 'decision' && (
                <div className="space-y-3">
                  {(sourceDetail?.org || docMetadata.org) && (
                    <div>
                      <span className="text-xs font-medium text-slate-500">기관</span>
                      <p className="text-sm text-slate-800 mt-1">
                        {sourceDetail?.org || docMetadata.org}
                      </p>
                    </div>
                  )}
                  {(sourceDetail?.decision_number || docMetadata.doc_id) && (
                    <div>
                      <span className="text-xs font-medium text-slate-500">일련번호</span>
                      <p className="text-sm text-slate-800 mt-1">
                        {sourceDetail?.decision_number || docMetadata.doc_id}
                      </p>
                    </div>
                  )}
                  {(sourceDetail?.decision_date || docMetadata.decision_date) && (
                    <div>
                      <span className="text-xs font-medium text-slate-500">결정일</span>
                      <p className="text-sm text-slate-800 mt-1">
                        {sourceDetail?.decision_date || docMetadata.decision_date}
                      </p>
                    </div>
                  )}
                  {(sourceDetail?.result || docMetadata.result) && (
                    <div>
                      <span className="text-xs font-medium text-slate-500">결과</span>
                      <p className="text-sm text-slate-800 mt-1">
                        {sourceDetail?.result || docMetadata.result}
                      </p>
                    </div>
                  )}
                </div>
              )}

              {/* 해석례 정보 */}
              {documentType === 'interpretation' && (
                <div className="space-y-3">
                  {(sourceDetail?.title || docMetadata.title) && (
                    <div>
                      <span className="text-xs font-medium text-slate-500">제목</span>
                      <p className="text-sm text-slate-800 mt-1">
                        {sourceDetail?.title || docMetadata.title}
                      </p>
                    </div>
                  )}
                  {(sourceDetail?.org || docMetadata.org) && (
                    <div>
                      <span className="text-xs font-medium text-slate-500">기관</span>
                      <p className="text-sm text-slate-800 mt-1">
                        {sourceDetail?.org || docMetadata.org}
                      </p>
                    </div>
                  )}
                  {(sourceDetail?.interpretation_number || docMetadata.doc_id) && (
                    <div>
                      <span className="text-xs font-medium text-slate-500">일련번호</span>
                      <p className="text-sm text-slate-800 mt-1">
                        {sourceDetail?.interpretation_number || docMetadata.doc_id}
                      </p>
                    </div>
                  )}
                  {(sourceDetail?.response_date || docMetadata.response_date) && (
                    <div>
                      <span className="text-xs font-medium text-slate-500">회신일</span>
                      <p className="text-sm text-slate-800 mt-1">
                        {sourceDetail?.response_date || docMetadata.response_date}
                      </p>
                    </div>
                  )}
                </div>
              )}

              {/* 기타 정보 */}
              {documentType === 'regulation' && docMetadata.org && (
                <div>
                  <span className="text-xs font-medium text-slate-500">기관</span>
                  <p className="text-sm text-slate-800 mt-1">{docMetadata.org}</p>
                </div>
              )}

              {/* 원문 링크 */}
              {sourceDetail?.url && (
                <div className="pt-3 border-t border-slate-200">
                  <a
                    href={sourceDetail.url}
                    target="_blank"
                    rel="noopener noreferrer"
                    className="inline-flex items-center gap-2 text-sm text-blue-600 hover:text-blue-700 font-medium"
                  >
                    <ExternalLink className="w-4 h-4" />
                    원문 보기
                  </a>
                </div>
              )}

              {/* 상세본문 */}
              {sourceDetail?.content && (
                <div className="pt-3 border-t border-slate-200">
                  <span className="text-xs font-medium text-slate-500 mb-2 block">상세본문</span>
                  <div className="bg-slate-50 p-3 rounded-md max-h-96 overflow-y-auto">
                    <p className="text-sm text-slate-700 whitespace-pre-wrap break-words">
                      {sourceDetail.content}
                    </p>
                  </div>
                </div>
              )}

              {/* 원본 텍스트 (상세본문이 없는 경우에만 표시) */}
              {!sourceDetail?.content && source && (
                <div className="pt-3 border-t border-slate-200">
                  <span className="text-xs font-medium text-slate-500">원본 참조</span>
                  <p className="text-sm text-slate-600 mt-1 break-words">{source}</p>
                </div>
              )}
            </div>
          </div>

          {/* 푸터 */}
          <div className="p-4 border-t border-slate-200 bg-slate-50">
            <button
              onClick={onClose}
              className="w-full px-4 py-2 bg-slate-200 hover:bg-slate-300 text-slate-800 rounded-md transition-colors font-medium"
            >
              닫기
            </button>
          </div>
        </div>
      </div>
    </>
  );
}

