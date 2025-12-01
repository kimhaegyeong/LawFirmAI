/**
 * 참고자료 상세 정보 뷰 컴포넌트
 * DocumentSidebar와 동일한 디자인으로 참고자료 상세 정보를 표시합니다.
 */
import { FileText, Scale, Bookmark, ExternalLink, ArrowLeft, Copy, Check } from 'lucide-react';
import { useMemo, useState, useEffect } from 'react';
import type { LegalReferenceDetail, SourceInfo } from '../../types/chat';
import { generateLawUrl, generateSearchUrl, type LawUrlType } from '../../utils/lawUrlGenerator';
import { copyToClipboardWithFeedback } from '../../utils/copyToClipboard';
import { getMetadataValue } from '../../utils/metadataUtils';

interface ReferenceDetailViewProps {
  reference: LegalReferenceDetail;
  sourceDetail?: SourceInfo;
  onBack: () => void;
  onClose?: () => void;
}

export function ReferenceDetailView({
  reference,
  sourceDetail,
  onBack,
  onClose,
}: ReferenceDetailViewProps) {
  const [copiedField, setCopiedField] = useState<string | null>(null);
  
  useEffect(() => {
    if (copiedField) {
      const timer = setTimeout(() => setCopiedField(null), 2000);
      return () => clearTimeout(timer);
    }
    return undefined;
  }, [copiedField]);
  
  // 문서 타입 결정
  const documentType: 'law' | 'precedent' | 'decision' | 'interpretation' | 'regulation' = reference.type;

  // 문서 제목/이름
  const documentName = 
    reference.law_name || 
    reference.case_name || 
    reference.title || 
    reference.content || 
    '참고자료';

  // sourceDetail 또는 reference에서 정보 추출 (useMemo로 최적화)
  const docMetadata = useMemo(() => sourceDetail?.metadata || {}, [sourceDetail?.metadata]);
  
  // 해석례 여부 확인 (타입 안전성을 위해)
  const isInterpretation: boolean = documentType === 'interpretation';
  
  // URL 생성 (useMemo로 최적화)
  const generatedUrl = useMemo(() => {
    if (sourceDetail?.url) {
      return sourceDetail.url;
    }
    
    const metadata = {
      ...docMetadata,
      ...reference,
      ...sourceDetail,
      law_id: docMetadata.law_id || sourceDetail?.metadata?.law_id,
      mst: docMetadata.mst || sourceDetail?.metadata?.mst,
      article_no: reference.article_number || sourceDetail?.article_no || docMetadata.article_no,
      effective_date: docMetadata.effective_date || sourceDetail?.metadata?.effective_date,
      precedent_serial_number: docMetadata.precedent_serial_number || sourceDetail?.metadata?.precedent_serial_number,
      decision_serial_number: docMetadata.decision_serial_number || sourceDetail?.metadata?.decision_serial_number,
      interpretation_serial_number: docMetadata.interpretation_serial_number || sourceDetail?.metadata?.interpretation_serial_number,
      doc_id: reference.case_number || reference.decision_number || reference.interpretation_number || docMetadata.doc_id,
      statute_name: reference.law_name || sourceDetail?.statute_name || docMetadata.statute_name,
      casenames: reference.case_name || sourceDetail?.case_name || docMetadata.casenames,
    };
    
    const urlType: LawUrlType = 
      documentType === 'law' ? 'statute' :
      documentType === 'precedent' ? 'case' :
      documentType === 'decision' ? 'decision' :
      documentType === 'interpretation' ? 'interpretation' : 'statute';
    
    return generateLawUrl(urlType, metadata);
  }, [reference, sourceDetail, docMetadata, documentType]);
  
  // 검색 링크 생성 (URL이 없을 때)
  const searchUrl = useMemo(() => {
    if (generatedUrl) return null;
    
    const metadata = {
      ...docMetadata,
      ...reference,
      ...sourceDetail,
      statute_name: reference.law_name || sourceDetail?.statute_name || docMetadata.statute_name,
      casenames: reference.case_name || sourceDetail?.case_name || docMetadata.casenames,
      org: reference.org || sourceDetail?.org || docMetadata.org,
      title: reference.title || sourceDetail?.title || docMetadata.title,
    };
    
    const urlType: LawUrlType = 
      documentType === 'law' ? 'statute' :
      documentType === 'precedent' ? 'case' :
      documentType === 'decision' ? 'decision' :
      documentType === 'interpretation' ? 'interpretation' : 'statute';
    
    return generateSearchUrl(urlType, metadata);
  }, [generatedUrl, reference, sourceDetail, docMetadata, documentType]);
  
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
    <div className="flex flex-col h-full">
      {/* 헤더 */}
      <div className="flex items-center justify-between p-4 border-b border-slate-200 bg-slate-50">
        <div className="flex items-center gap-2">
          <button
            onClick={onBack}
            className="p-1.5 rounded-md hover:bg-slate-200 transition-colors text-slate-600"
            aria-label="뒤로 가기"
          >
            <ArrowLeft className="w-5 h-5" />
          </button>
          {documentType === 'law' && <FileText className="w-5 h-5 text-blue-600" />}
          {documentType === 'precedent' && <Scale className="w-5 h-5 text-green-600" />}
          {documentType === 'decision' && <Bookmark className="w-5 h-5 text-orange-600" />}
          {documentType === 'interpretation' && <FileText className="w-5 h-5 text-indigo-600" />}
          {documentType === 'regulation' && <Bookmark className="w-5 h-5 text-purple-600" />}
          <h2 className="text-lg font-semibold text-slate-800">
            {documentName}
          </h2>
        </div>
        {onClose && (
          <button
            onClick={onClose}
            className="p-1.5 rounded-md hover:bg-slate-200 transition-colors text-slate-600"
            aria-label="사이드바 닫기"
          >
            <FileText className="w-5 h-5" />
          </button>
        )}
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
                value={reference.law_name || sourceDetail?.statute_name || docMetadata.statute_name}
                fieldName="statute_name"
              />
              <InfoField
                label="조문"
                value={
                  reference.article_number || 
                  (sourceDetail?.article_no ? `제${sourceDetail.article_no}조` : undefined) ||
                  (docMetadata.article_no ? `제${docMetadata.article_no}조` : undefined)
                }
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
              {getMetadataValue(docMetadata.title) && (
                <div>
                  <span className="text-xs font-medium text-slate-500">제목</span>
                  <p className="text-sm text-slate-800 mt-1">{String(getMetadataValue(docMetadata.title))}</p>
                </div>
              )}
            </div>
          )}

          {/* 판례 정보 */}
          {documentType === 'precedent' && (
            <div className="space-y-3">
              <InfoField
                label="사건명"
                value={reference.case_name || sourceDetail?.case_name || docMetadata.casenames}
                fieldName="case_name"
              />
              <InfoField
                label="법원"
                value={reference.court || sourceDetail?.court || docMetadata.court}
                fieldName="court"
              />
              <InfoField
                label="사건번호"
                value={reference.case_number || sourceDetail?.case_number || docMetadata.doc_id}
                fieldName="case_number"
              />
              <InfoField
                label="판례일련번호"
                value={getMetadataValue(docMetadata.precedent_serial_number) || getMetadataValue(docMetadata.판례일련번호) || getMetadataValue(docMetadata.판례정보일련번호)}
                fieldName="precedent_serial_number"
                isMonospace
              />
              <InfoField
                label="판결일"
                value={reference.decision_date || getMetadataValue(docMetadata.announce_date)}
                fieldName="decision_date"
              />
            </div>
          )}

          {/* 결정례 정보 */}
          {documentType === 'decision' && (
            <div className="space-y-3">
              <InfoField
                label="기관"
                value={reference.org || sourceDetail?.org || docMetadata.org}
                fieldName="org"
              />
              <InfoField
                label="일련번호"
                value={reference.decision_number || sourceDetail?.decision_number || docMetadata.doc_id}
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
                value={sourceDetail?.decision_date || getMetadataValue(docMetadata.decision_date)}
                fieldName="decision_date"
              />
              <InfoField
                label="결과"
                value={reference.result || sourceDetail?.result || docMetadata.result}
                fieldName="result"
              />
            </div>
          )}

          {(documentType as string) === 'interpretation' ? (
            <div className="space-y-3">
              <InfoField
                label="제목"
                value={reference.title || sourceDetail?.title || docMetadata.title}
                fieldName="title"
              />
              <InfoField
                label="기관"
                value={reference.org || sourceDetail?.org || docMetadata.org}
                fieldName="org"
              />
              <InfoField
                label="일련번호"
                value={reference.interpretation_number || sourceDetail?.interpretation_number || docMetadata.doc_id}
                fieldName="interpretation_number"
              />
              <InfoField
                label="법령해석례일련번호"
                value={
                  getMetadataValue(docMetadata.interpretation_serial_number) ||
                  getMetadataValue(docMetadata.법령해석례일련번호) ||
                  getMetadataValue(docMetadata.해석ID) ||
                  getMetadataValue(docMetadata.expcId)
                }
                fieldName="interpretation_serial_number"
                isMonospace
              />
              <InfoField
                label="회신일"
                value={sourceDetail?.response_date || getMetadataValue(docMetadata.response_date)}
                fieldName="response_date"
              />
            </div>
          ) : null}

          {/* 기타 정보 */}
          {documentType === 'regulation' && (reference.org || sourceDetail?.org || docMetadata.org) && (
            <div>
              <span className="text-xs font-medium text-slate-500">기관</span>
              <p className="text-sm text-slate-800 mt-1">
                {reference.org || sourceDetail?.org || docMetadata.org}
              </p>
            </div>
          )}

          {/* 해석례 정보 */}
          {isInterpretation ? (
            <div className="space-y-3">
              <InfoField
                label="제목"
                value={reference.title || sourceDetail?.title || docMetadata.title}
                fieldName="title"
              />
              <InfoField
                label="기관"
                value={reference.org || sourceDetail?.org || docMetadata.org}
                fieldName="org"
              />
              <InfoField
                label="일련번호"
                value={reference.interpretation_number || sourceDetail?.interpretation_number || docMetadata.doc_id}
                fieldName="interpretation_number"
              />
              <InfoField
                label="법령해석례일련번호"
                value={
                  getMetadataValue(docMetadata.interpretation_serial_number) ||
                  getMetadataValue(docMetadata.법령해석례일련번호) ||
                  getMetadataValue(docMetadata.해석ID) ||
                  getMetadataValue(docMetadata.expcId)
                }
                fieldName="interpretation_serial_number"
                isMonospace
              />
              <InfoField
                label="회신일"
                value={sourceDetail?.response_date || getMetadataValue(docMetadata.response_date)}
                fieldName="response_date"
              />
            </div>
          ) : null}

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
          {(reference.article_content || reference.summary || reference.content || sourceDetail?.content) && (
            <div className="pt-3 border-t border-slate-200">
              <span className="text-xs font-medium text-slate-500 mb-2 block">상세본문</span>
              <div className="bg-slate-50 p-3 rounded-md max-h-96 overflow-y-auto">
                <p className="text-sm text-slate-700 whitespace-pre-wrap break-words">
                  {reference.article_content || 
                   reference.summary || 
                   reference.content || 
                   sourceDetail?.content}
                </p>
              </div>
            </div>
          )}

          {/* 원본 텍스트 (상세본문이 없는 경우에만 표시) */}
          {!reference.article_content && 
           !reference.summary && 
           !reference.content && 
           !sourceDetail?.content && 
           reference.content && (
            <div className="pt-3 border-t border-slate-200">
              <span className="text-xs font-medium text-slate-500">원본 참조</span>
              <p className="text-sm text-slate-600 mt-1 break-words">{reference.content}</p>
            </div>
          )}

          {/* 원천 정보 섹션 (접을 수 있는 섹션) */}
          {(sourceDetail?.chunk_id || sourceDetail?.source_id || sourceDetail?.original_url || 
            sourceDetail?.metadata?.chunk_id || sourceDetail?.metadata?.source_id || sourceDetail?.metadata?.original_url) && (
            <details className="border-t border-slate-200 pt-3 mt-3">
              <summary className="text-sm font-medium text-slate-600 cursor-pointer hover:text-slate-800">
                원천 정보
              </summary>
              <div className="mt-2 space-y-2 text-xs">
                <InfoField
                  label="Source ID"
                  value={(sourceDetail?.source_id ? String(sourceDetail.source_id) : undefined) || (sourceDetail?.metadata?.source_id ? String(sourceDetail.metadata.source_id) : undefined)}
                  fieldName="source_id"
                  isMonospace
                />
                <InfoField
                  label="Chunk ID"
                  value={(sourceDetail?.chunk_id ? String(sourceDetail.chunk_id) : undefined) || (sourceDetail?.metadata?.chunk_id ? String(sourceDetail.metadata.chunk_id) : undefined)}
                  fieldName="chunk_id"
                  isMonospace
                />
                {sourceDetail?.original_url || sourceDetail?.metadata?.original_url ? (
                  <div>
                    <div className="flex items-center justify-between mb-1">
                      <span className="text-xs font-medium text-slate-500">Original URL</span>
                      <button
                        onClick={() => {
                          const url = sourceDetail?.original_url || sourceDetail?.metadata?.original_url;
                          if (url && typeof url === 'string') {
                            window.open(url, '_blank', 'noopener,noreferrer');
                          }
                        }}
                        className="p-1 hover:bg-slate-100 rounded transition-colors text-slate-400 hover:text-slate-600"
                        title="새 창에서 열기"
                      >
                        <ExternalLink className="w-3 h-3" />
                      </button>
                    </div>
                    <a
                      href={String(sourceDetail?.original_url || sourceDetail?.metadata?.original_url)}
                      target="_blank"
                      rel="noopener noreferrer"
                      className="text-xs text-blue-600 hover:underline break-all"
                    >
                      {String(sourceDetail?.original_url || sourceDetail?.metadata?.original_url)}
                    </a>
                  </div>
                    ) : null}
                  </div>
                </details>
              )}
        </div>
      </div>
    </div>
  );
}

