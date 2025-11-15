/**
 * 문서 상세 정보 사이드바 컴포넌트
 * 오른쪽에서 슬라이드되는 사이드바로 문서 상세 정보를 표시합니다.
 */
import { X, FileText, Scale, Bookmark, ExternalLink, Copy, Check } from 'lucide-react';
import { useEffect, useState, useMemo } from 'react';
import type { SourceInfo } from '../../types/chat';
import { generateLawUrl, generateSearchUrl, type LawUrlType } from '../../utils/lawUrlGenerator';
import { copyToClipboardWithFeedback } from '../../utils/copyToClipboard';
import { getMetadataValue } from '../../utils/metadataUtils';
import { getSourcesByType, getSourcesDetailFromSourcesByType, type SourcesByType } from '../../utils/sourcesParser';

interface DocumentSidebarProps {
  isOpen: boolean;
  onClose: () => void;
  documentIndex: number | null;
  sources?: string[];  // deprecated
  sourcesDetail?: SourceInfo[];  // deprecated, 하위 호환성
  sourcesByType?: SourcesByType;  // 우선 사용
  metadata?: Record<string, unknown>;
  sessionId?: string;
  messageId?: string;
}

export function DocumentSidebar({
  isOpen,
  onClose,
  documentIndex,
  sources = [],
  sourcesDetail = [],
  sourcesByType: propSourcesByType,
  metadata: _metadata,
  sessionId: _sessionId,
  messageId: _messageId
}: DocumentSidebarProps) {
  const [copiedField, setCopiedField] = useState<string | null>(null);
  
  // sourcesByType 우선 사용, 없으면 sourcesDetail에서 재구성
  const effectiveSourcesDetail = useMemo(() => {
    if (propSourcesByType) {
      return getSourcesDetailFromSourcesByType(propSourcesByType);
    }
    return sourcesDetail;
  }, [propSourcesByType, sourcesDetail]);
  
  useEffect(() => {
    if (copiedField) {
      const timer = setTimeout(() => setCopiedField(null), 2000);
      return () => clearTimeout(timer);
    }
    return undefined;
  }, [copiedField]);
  
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

  // 안전한 배열 접근 헬퍼 함수 (타입 안전, 인덱스 검증 완료)
  const getSafeArrayItem = <T,>(array: T[], index: number | null): T | undefined => {
    if (index === null || index < 0 || index >= array.length) {
      return undefined;
    }
    // eslint-disable-next-line security/detect-object-injection
    return array[index];
  };

  const source = getSafeArrayItem(sources, documentIndex);
  const sourceDetail = getSafeArrayItem(effectiveSourcesDetail, documentIndex);
  
  // sources_by_type 사용 (타입별 그룹화)
  const sourcesByType = useMemo(() => {
    if (propSourcesByType) {
      return propSourcesByType;
    }
    return getSourcesByType(effectiveSourcesDetail);
  }, [propSourcesByType, effectiveSourcesDetail]);

  // 문서 타입 결정 (useMemo로 최적화)
  const documentType = useMemo((): 'law' | 'precedent' | 'decision' | 'interpretation' | 'regulation' => {
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
  }, [sourceDetail, source]);

  // 문서 제목/이름
  const documentName = sourceDetail?.name || source || (documentIndex !== null ? `문서 ${documentIndex + 1}` : '');

  // 문서 메타데이터 (useMemo로 최적화)
  const docMetadata = useMemo(() => sourceDetail?.metadata || {}, [sourceDetail?.metadata]);
  
  // URL 생성 (useMemo로 최적화)
  const generatedUrl = useMemo(() => {
    if (sourceDetail?.url) return sourceDetail.url;
    
    const metadata = {
      ...docMetadata,
      ...sourceDetail,
      law_id: docMetadata.law_id || sourceDetail?.metadata?.law_id,
      mst: docMetadata.mst || sourceDetail?.metadata?.mst,
      article_no: sourceDetail?.article_no || docMetadata.article_no,
      effective_date: docMetadata.effective_date || sourceDetail?.metadata?.effective_date,
      precedent_serial_number: docMetadata.precedent_serial_number || sourceDetail?.metadata?.precedent_serial_number,
      decision_serial_number: docMetadata.decision_serial_number || sourceDetail?.metadata?.decision_serial_number,
      interpretation_serial_number: docMetadata.interpretation_serial_number || sourceDetail?.metadata?.interpretation_serial_number,
      doc_id: getMetadataValue(sourceDetail?.metadata?.doc_id) || getMetadataValue(docMetadata.doc_id),
    };
    
    const urlType: LawUrlType = 
      documentType === 'law' ? 'statute' :
      documentType === 'precedent' ? 'case' :
      documentType === 'decision' ? 'decision' :
      documentType === 'interpretation' ? 'interpretation' : 'statute';
    
    return generateLawUrl(urlType, metadata);
  }, [sourceDetail, docMetadata, documentType]);
  
  // 검색 링크 생성 (URL이 없을 때)
  const searchUrl = useMemo(() => {
    if (generatedUrl) return null;
    
    const metadata = {
      ...docMetadata,
      ...sourceDetail,
      statute_name: sourceDetail?.statute_name || docMetadata.statute_name,
      casenames: sourceDetail?.case_name || docMetadata.casenames,
      org: sourceDetail?.org || docMetadata.org,
      title: sourceDetail?.title || docMetadata.title,
    };
    
    const urlType: LawUrlType = 
      documentType === 'law' ? 'statute' :
      documentType === 'precedent' ? 'case' :
      documentType === 'decision' ? 'decision' :
      documentType === 'interpretation' ? 'interpretation' : 'statute';
    
    return generateSearchUrl(urlType, metadata);
  }, [generatedUrl, sourceDetail, docMetadata, documentType]);
  
  // 정보 복사 핸들러
  const handleCopy = async (text: string, fieldName: string) => {
    await copyToClipboardWithFeedback(
      text,
      () => setCopiedField(fieldName),
      () => {}
    );
  };
  
  // 정보 필드 컴포넌트 (복사 기능 포함)
  const InfoField = ({ label, value, fieldName, isMonospace = false }: { 
    label: string; 
    value: string | number | undefined; 
    fieldName: string;
    isMonospace?: boolean;
  }) => {
    if (!value) return null;
    
    const valueStr = String(value);
    const isCopied = copiedField === fieldName;
    
    return (
      <div>
        <div className="flex items-center justify-between mb-1">
          <span className="text-xs font-medium text-slate-500">{label}</span>
          <button
            onClick={() => handleCopy(valueStr, fieldName)}
            className="p-1 hover:bg-slate-100 rounded transition-colors text-slate-400 hover:text-slate-600"
            title="복사"
          >
            {isCopied ? (
              <Check className="w-3 h-3 text-green-600" />
            ) : (
              <Copy className="w-3 h-3" />
            )}
          </button>
        </div>
        <p className={`text-sm text-slate-800 mt-1 ${isMonospace ? 'font-mono' : ''}`}>
          {valueStr}
        </p>
      </div>
    );
  };

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
                문서 {documentIndex !== null ? documentIndex + 1 : '?'}
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
                  <InfoField
                    label="법령명"
                    value={sourceDetail?.statute_name || docMetadata.statute_name}
                    fieldName="statute_name"
                  />
                  <InfoField
                    label="조문"
                    value={sourceDetail?.article_no || docMetadata.article_no ? `제${sourceDetail?.article_no || docMetadata.article_no}조` : undefined}
                    fieldName="article_no"
                  />
                  <InfoField
                    label="항"
                    value={sourceDetail?.clause_no || docMetadata.clause_no ? `제${sourceDetail?.clause_no || docMetadata.clause_no}항` : undefined}
                    fieldName="clause_no"
                  />
                  <InfoField
                    label="호"
                    value={sourceDetail?.item_no || docMetadata.item_no ? `제${sourceDetail?.item_no || docMetadata.item_no}호` : undefined}
                    fieldName="item_no"
                  />
                  <InfoField
                    label="법령ID"
                    value={getMetadataValue(docMetadata.law_id) || getMetadataValue(docMetadata.법령ID) || getMetadataValue(docMetadata.ID)}
                    fieldName="law_id"
                    isMonospace
                  />
                  <InfoField
                    label="법령 마스터번호 (MST)"
                    value={getMetadataValue(docMetadata.mst) || getMetadataValue(docMetadata.MST) || getMetadataValue(docMetadata.lsi_seq)}
                    fieldName="mst"
                    isMonospace
                  />
                  <InfoField
                    label="공포번호"
                    value={getMetadataValue(docMetadata.proclamation_number) || getMetadataValue(docMetadata.공포번호)}
                    fieldName="proclamation_number"
                  />
                  <InfoField
                    label="시행일자"
                    value={getMetadataValue(docMetadata.effective_date) || getMetadataValue(docMetadata.efYd) || getMetadataValue(docMetadata.시행일자)}
                    fieldName="effective_date"
                  />
                  {docMetadata.title && (
                    <div>
                      <span className="text-xs font-medium text-slate-500">제목</span>
                      <p className="text-sm text-slate-800 mt-1">{String(docMetadata.title)}</p>
                    </div>
                  )}
                </div>
              )}

              {/* 판례 정보 */}
              {documentType === 'precedent' && (
                <div className="space-y-3">
                  <InfoField
                    label="사건명"
                    value={sourceDetail?.case_name || docMetadata.casenames}
                    fieldName="case_name"
                  />
                  <InfoField
                    label="법원"
                    value={sourceDetail?.court || docMetadata.court}
                    fieldName="court"
                  />
                  <InfoField
                    label="사건번호"
                    value={sourceDetail?.case_number || getMetadataValue(docMetadata.doc_id)}
                    fieldName="case_number"
                  />
                  <InfoField
                    label="판례일련번호"
                    value={getMetadataValue(docMetadata.precedent_serial_number) || getMetadataValue(docMetadata.판례일련번호) || getMetadataValue(docMetadata.판례정보일련번호)}
                    fieldName="precedent_serial_number"
                    isMonospace
                  />
                  {(() => {
                    const announceDate = getMetadataValue(docMetadata.announce_date);
                    return announceDate ? (
                      <InfoField
                        label="선고일자"
                        value={announceDate}
                        fieldName="announce_date"
                      />
                    ) : null;
                  })()}
                </div>
              )}

              {/* 결정례 정보 */}
              {documentType === 'decision' && (
                <div className="space-y-3">
                  <InfoField
                    label="기관"
                    value={sourceDetail?.org || docMetadata.org}
                    fieldName="org"
                  />
                  <InfoField
                    label="일련번호"
                    value={sourceDetail?.decision_number || getMetadataValue(docMetadata.doc_id)}
                    fieldName="decision_number"
                  />
                  <InfoField
                    label="헌재결정례일련번호"
                    value={getMetadataValue(docMetadata.decision_serial_number) || getMetadataValue(docMetadata.헌재결정례일련번호)}
                    fieldName="decision_serial_number"
                    isMonospace
                  />
                  <InfoField
                    label="결정일"
                    value={sourceDetail?.decision_date || docMetadata.decision_date}
                    fieldName="decision_date"
                  />
                  <InfoField
                    label="결과"
                    value={sourceDetail?.result || docMetadata.result}
                    fieldName="result"
                  />
                </div>
              )}

              {/* 해석례 정보 */}
              {documentType === 'interpretation' && (
                <div className="space-y-3">
                  <InfoField
                    label="제목"
                    value={sourceDetail?.title || docMetadata.title}
                    fieldName="title"
                  />
                  <InfoField
                    label="기관"
                    value={sourceDetail?.org || docMetadata.org}
                    fieldName="org"
                  />
                  <InfoField
                    label="일련번호"
                    value={sourceDetail?.interpretation_number || getMetadataValue(docMetadata.doc_id)}
                    fieldName="interpretation_number"
                  />
                  <InfoField
                    label="법령해석례일련번호"
                    value={getMetadataValue(docMetadata.interpretation_serial_number) || getMetadataValue(docMetadata.법령해석례일련번호) || getMetadataValue(docMetadata.해석ID) || getMetadataValue(docMetadata.expcId)}
                    fieldName="interpretation_serial_number"
                    isMonospace
                  />
                  <InfoField
                    label="회신일"
                    value={sourceDetail?.response_date || docMetadata.response_date}
                    fieldName="response_date"
                  />
                </div>
              )}

              {/* 기타 정보 */}
              {documentType === 'regulation' && docMetadata.org && (
                <div>
                  <span className="text-xs font-medium text-slate-500">기관</span>
                  <p className="text-sm text-slate-800 mt-1">{docMetadata.org}</p>
                </div>
              )}

              {/* 원문 링크 및 검색 링크 */}
              <div className="pt-3 border-t border-slate-200 space-y-2">
                {generatedUrl && (
                  <div>
                    <a
                      href={generatedUrl}
                      target="_blank"
                      rel="noopener noreferrer"
                      className="inline-flex items-center gap-2 px-4 py-2 bg-blue-50 hover:bg-blue-100 text-blue-700 rounded-lg transition-colors font-medium w-full justify-center"
                      title="국가법령정보센터에서 원문 보기"
                    >
                      <ExternalLink className="w-4 h-4" />
                      국가법령정보센터에서 원문 보기
                    </a>
                    {import.meta.env.DEV && (
                      <div className="mt-2 text-xs text-slate-500 break-all">
                        {generatedUrl}
                      </div>
                    )}
                  </div>
                )}
                {!generatedUrl && searchUrl && (
                  <div>
                    <a
                      href={searchUrl}
                      target="_blank"
                      rel="noopener noreferrer"
                      className="inline-flex items-center gap-2 px-4 py-2 bg-slate-50 hover:bg-slate-100 text-slate-700 rounded-lg transition-colors font-medium w-full justify-center"
                      title="국가법령정보센터에서 검색하기"
                    >
                      <ExternalLink className="w-4 h-4" />
                      국가법령정보센터에서 검색하기
                    </a>
                  </div>
                )}
                {!generatedUrl && !searchUrl && (
                  <div className="text-xs text-slate-500 text-center py-2">
                    원문 링크를 생성할 수 없습니다. 필요한 정보가 부족합니다.
                  </div>
                )}
              </div>

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

