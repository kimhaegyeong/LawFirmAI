/**
 * 참고자료 상세 정보 뷰 컴포넌트
 * DocumentSidebar와 동일한 디자인으로 참고자료 상세 정보를 표시합니다.
 */
import { FileText, Scale, Bookmark, ExternalLink, ArrowLeft } from 'lucide-react';
import type { LegalReferenceDetail, SourceInfo } from '../../types/chat';

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
  // 문서 타입 결정
  const documentType = reference.type;

  // 문서 제목/이름
  const documentName = 
    reference.law_name || 
    reference.case_name || 
    reference.title || 
    reference.content || 
    '참고자료';

  // sourceDetail 또는 reference에서 정보 추출
  const docMetadata = sourceDetail?.metadata || {};

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
              {(reference.law_name || sourceDetail?.statute_name || docMetadata.statute_name) && (
                <div>
                  <span className="text-xs font-medium text-slate-500">법령명</span>
                  <p className="text-sm text-slate-800 mt-1">
                    {reference.law_name || sourceDetail?.statute_name || docMetadata.statute_name}
                  </p>
                </div>
              )}
              {(reference.article_number || sourceDetail?.article_no || docMetadata.article_no) && (
                <div>
                  <span className="text-xs font-medium text-slate-500">조문</span>
                  <p className="text-sm text-slate-800 mt-1">
                    {reference.article_number || 
                     (sourceDetail?.article_no ? `제${sourceDetail.article_no}조` : '') ||
                     (docMetadata.article_no ? `제${docMetadata.article_no}조` : '')}
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
              {(reference.case_name || sourceDetail?.case_name || docMetadata.casenames) && (
                <div>
                  <span className="text-xs font-medium text-slate-500">사건명</span>
                  <p className="text-sm text-slate-800 mt-1">
                    {reference.case_name || sourceDetail?.case_name || docMetadata.casenames}
                  </p>
                </div>
              )}
              {(reference.court || sourceDetail?.court || docMetadata.court) && (
                <div>
                  <span className="text-xs font-medium text-slate-500">법원</span>
                  <p className="text-sm text-slate-800 mt-1">
                    {reference.court || sourceDetail?.court || docMetadata.court}
                  </p>
                </div>
              )}
              {(reference.case_number || sourceDetail?.case_number || docMetadata.doc_id) && (
                <div>
                  <span className="text-xs font-medium text-slate-500">사건번호</span>
                  <p className="text-sm text-slate-800 mt-1">
                    {reference.case_number || sourceDetail?.case_number || docMetadata.doc_id}
                  </p>
                </div>
              )}
              {reference.decision_date && (
                <div>
                  <span className="text-xs font-medium text-slate-500">판결일</span>
                  <p className="text-sm text-slate-800 mt-1">{reference.decision_date}</p>
                </div>
              )}
            </div>
          )}

          {/* 결정례 정보 */}
          {documentType === 'decision' && (
            <div className="space-y-3">
              {(reference.org || sourceDetail?.org || docMetadata.org) && (
                <div>
                  <span className="text-xs font-medium text-slate-500">기관</span>
                  <p className="text-sm text-slate-800 mt-1">
                    {reference.org || sourceDetail?.org || docMetadata.org}
                  </p>
                </div>
              )}
              {(reference.decision_number || sourceDetail?.decision_number || docMetadata.doc_id) && (
                <div>
                  <span className="text-xs font-medium text-slate-500">일련번호</span>
                  <p className="text-sm text-slate-800 mt-1">
                    {reference.decision_number || sourceDetail?.decision_number || docMetadata.doc_id}
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
              {(reference.result || sourceDetail?.result || docMetadata.result) && (
                <div>
                  <span className="text-xs font-medium text-slate-500">결과</span>
                  <p className="text-sm text-slate-800 mt-1">
                    {reference.result || sourceDetail?.result || docMetadata.result}
                  </p>
                </div>
              )}
            </div>
          )}

          {/* 해석례 정보 */}
          {documentType === 'interpretation' && (
            <div className="space-y-3">
              {(reference.title || sourceDetail?.title || docMetadata.title) && (
                <div>
                  <span className="text-xs font-medium text-slate-500">제목</span>
                  <p className="text-sm text-slate-800 mt-1">
                    {reference.title || sourceDetail?.title || docMetadata.title}
                  </p>
                </div>
              )}
              {(reference.org || sourceDetail?.org || docMetadata.org) && (
                <div>
                  <span className="text-xs font-medium text-slate-500">기관</span>
                  <p className="text-sm text-slate-800 mt-1">
                    {reference.org || sourceDetail?.org || docMetadata.org}
                  </p>
                </div>
              )}
              {(reference.interpretation_number || sourceDetail?.interpretation_number || docMetadata.doc_id) && (
                <div>
                  <span className="text-xs font-medium text-slate-500">일련번호</span>
                  <p className="text-sm text-slate-800 mt-1">
                    {reference.interpretation_number || sourceDetail?.interpretation_number || docMetadata.doc_id}
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
          {documentType === 'regulation' && (reference.org || sourceDetail?.org || docMetadata.org) && (
            <div>
              <span className="text-xs font-medium text-slate-500">기관</span>
              <p className="text-sm text-slate-800 mt-1">
                {reference.org || sourceDetail?.org || docMetadata.org}
              </p>
            </div>
          )}

          {/* 원문 링크 */}
          {(reference.url || sourceDetail?.url) && (
            <div className="pt-3 border-t border-slate-200">
              <a
                href={reference.url || sourceDetail?.url}
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
        </div>
      </div>
    </div>
  );
}

