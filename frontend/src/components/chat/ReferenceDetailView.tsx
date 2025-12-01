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

// ============================================================================
// 타입 정의
// ============================================================================

interface ReferenceDetailViewProps {
  reference: LegalReferenceDetail;
  sourceDetail?: SourceInfo;
  onBack: () => void;
  onClose?: () => void;
}

type DocumentType = 'law' | 'precedent' | 'decision' | 'interpretation' | 'regulation';

interface InfoFieldProps {
  label: string;
  value: string | number | undefined;
  fieldName: string;
  isMonospace?: boolean;
}

// ============================================================================
// 상수 정의
// ============================================================================

const DOCUMENT_TYPE_CONFIG: Record<DocumentType, { label: string; icon: typeof FileText; color: string; badgeColor: string }> = {
  law: {
    label: '법령',
    icon: FileText,
    color: 'text-blue-600',
    badgeColor: 'bg-blue-100 text-blue-700',
  },
  precedent: {
    label: '판례',
    icon: Scale,
    color: 'text-green-600',
    badgeColor: 'bg-green-100 text-green-700',
  },
  decision: {
    label: '결정례',
    icon: Bookmark,
    color: 'text-orange-600',
    badgeColor: 'bg-orange-100 text-orange-700',
  },
  interpretation: {
    label: '해석례',
    icon: FileText,
    color: 'text-indigo-600',
    badgeColor: 'bg-indigo-100 text-indigo-700',
  },
  regulation: {
    label: '기타',
    icon: Bookmark,
    color: 'text-purple-600',
    badgeColor: 'bg-purple-100 text-purple-700',
  },
};

// ============================================================================
// 유틸리티 함수
// ============================================================================

function getDocumentName(reference: LegalReferenceDetail): string {
  return reference.law_name || 
         reference.case_name || 
         reference.title || 
         reference.content || 
         '참고자료';
}

function getDocumentTypeUrlType(documentType: DocumentType): LawUrlType {
  switch (documentType) {
    case 'law':
      return 'statute';
    case 'precedent':
      return 'case';
    case 'decision':
      return 'decision';
    case 'interpretation':
      return 'interpretation';
    case 'regulation':
      return 'statute';
    default:
      return 'statute';
  }
}

function buildMetadataForUrl(
  reference: LegalReferenceDetail,
  sourceDetail: SourceInfo | undefined,
  docMetadata: Record<string, unknown>
): Record<string, unknown> {
  return {
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
}

function buildMetadataForSearch(
  reference: LegalReferenceDetail,
  sourceDetail: SourceInfo | undefined,
  docMetadata: Record<string, unknown>
): Record<string, unknown> {
  return {
    ...docMetadata,
    ...reference,
    ...sourceDetail,
    statute_name: reference.law_name || sourceDetail?.statute_name || getMetadataValue(docMetadata.statute_name),
    casenames: reference.case_name || sourceDetail?.case_name || getMetadataValue(docMetadata.casenames),
    org: reference.org || sourceDetail?.org || getMetadataValue(docMetadata.org),
    title: reference.title || sourceDetail?.title || getMetadataValue(docMetadata.title),
  };
}

// ============================================================================
// 서브 컴포넌트
// ============================================================================

function InfoField({ label, value, fieldName, isMonospace = false, onCopy, copiedField }: InfoFieldProps & { onCopy: (text: string, fieldName: string) => void; copiedField: string | null }) {
  if (!value) return null;
  
  const valueStr = String(value);
  const isCopied = copiedField === fieldName;
  
  return (
    <div>
      <div className="flex items-center justify-between mb-1">
        <span className="text-xs font-medium text-slate-500">{label}</span>
        <button
          onClick={() => onCopy(valueStr, fieldName)}
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
}

function Header({
  documentType,
  documentName,
  onBack,
  onClose,
}: {
  documentType: DocumentType;
  documentName: string;
  onBack: () => void;
  onClose?: () => void;
}) {
  let config = DOCUMENT_TYPE_CONFIG.law;
  switch (documentType) {
    case 'law':
      config = DOCUMENT_TYPE_CONFIG.law;
      break;
    case 'precedent':
      config = DOCUMENT_TYPE_CONFIG.precedent;
      break;
    case 'decision':
      config = DOCUMENT_TYPE_CONFIG.decision;
      break;
    case 'interpretation':
      config = DOCUMENT_TYPE_CONFIG.interpretation;
      break;
    case 'regulation':
      config = DOCUMENT_TYPE_CONFIG.regulation;
      break;
    default:
      config = DOCUMENT_TYPE_CONFIG.law;
  }
  const Icon = config.icon;
  
  return (
    <div className="flex items-center justify-between p-4 border-b border-slate-200 bg-slate-50">
      <div className="flex items-center gap-2">
        <button
          onClick={onBack}
          className="p-1.5 rounded-md hover:bg-slate-200 transition-colors text-slate-600"
          aria-label="뒤로 가기"
        >
          <ArrowLeft className="w-5 h-5" />
        </button>
        <Icon className={`w-5 h-5 ${config.color}`} />
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
  );
}

function DocumentTypeBadge({ documentType }: { documentType: DocumentType }) {
  let config = DOCUMENT_TYPE_CONFIG.law;
  switch (documentType) {
    case 'law':
      config = DOCUMENT_TYPE_CONFIG.law;
      break;
    case 'precedent':
      config = DOCUMENT_TYPE_CONFIG.precedent;
      break;
    case 'decision':
      config = DOCUMENT_TYPE_CONFIG.decision;
      break;
    case 'interpretation':
      config = DOCUMENT_TYPE_CONFIG.interpretation;
      break;
    case 'regulation':
      config = DOCUMENT_TYPE_CONFIG.regulation;
      break;
    default:
      config = DOCUMENT_TYPE_CONFIG.law;
  }
  
  return (
    <div className="flex items-center gap-2">
      <span className={`text-xs font-semibold px-2 py-1 rounded ${config.badgeColor}`}>
        {config.label}
      </span>
    </div>
  );
}

function LawInfoSection({
  reference,
  sourceDetail,
  docMetadata,
  onCopy,
  copiedField,
}: {
  reference: LegalReferenceDetail;
  sourceDetail?: SourceInfo;
  docMetadata: Record<string, unknown>;
  onCopy: (text: string, fieldName: string) => void;
  copiedField: string | null;
}) {
  return (
    <div className="space-y-3">
      <InfoField
        label="법령명"
        value={reference.law_name || sourceDetail?.statute_name || (docMetadata.statute_name as string | undefined)}
        fieldName="statute_name"
        onCopy={onCopy}
        copiedField={copiedField}
      />
      <InfoField
        label="조문"
        value={
          reference.article_number || 
          (sourceDetail?.article_no ? `제${sourceDetail.article_no}조` : undefined) ||
          (docMetadata.article_no ? `제${docMetadata.article_no}조` : undefined)
        }
        fieldName="article_no"
        onCopy={onCopy}
        copiedField={copiedField}
      />
      <InfoField
        label="항"
        value={sourceDetail?.clause_no || docMetadata.clause_no ? `제${sourceDetail?.clause_no || docMetadata.clause_no}항` : undefined}
        fieldName="clause_no"
        onCopy={onCopy}
        copiedField={copiedField}
      />
      <InfoField
        label="호"
        value={sourceDetail?.item_no || docMetadata.item_no ? `제${sourceDetail?.item_no || docMetadata.item_no}호` : undefined}
        fieldName="item_no"
        onCopy={onCopy}
        copiedField={copiedField}
      />
      <InfoField
        label="법령ID"
        value={getMetadataValue(docMetadata.law_id) || getMetadataValue(docMetadata.법령ID) || getMetadataValue(docMetadata.ID)}
        fieldName="law_id"
        isMonospace
        onCopy={onCopy}
        copiedField={copiedField}
      />
      <InfoField
        label="법령 마스터번호 (MST)"
        value={getMetadataValue(docMetadata.mst) || getMetadataValue(docMetadata.MST) || getMetadataValue(docMetadata.lsi_seq)}
        fieldName="mst"
        isMonospace
        onCopy={onCopy}
        copiedField={copiedField}
      />
      <InfoField
        label="공포번호"
        value={getMetadataValue(docMetadata.proclamation_number) || getMetadataValue(docMetadata.공포번호)}
        fieldName="proclamation_number"
        onCopy={onCopy}
        copiedField={copiedField}
      />
      <InfoField
        label="시행일자"
        value={getMetadataValue(docMetadata.effective_date) || getMetadataValue(docMetadata.efYd) || getMetadataValue(docMetadata.시행일자)}
        fieldName="effective_date"
        onCopy={onCopy}
        copiedField={copiedField}
      />
      {getMetadataValue(docMetadata.title) && (
        <div>
          <span className="text-xs font-medium text-slate-500">제목</span>
          <p className="text-sm text-slate-800 mt-1">{String(getMetadataValue(docMetadata.title))}</p>
        </div>
      )}
    </div>
  );
}

function PrecedentInfoSection({
  reference,
  sourceDetail,
  docMetadata,
  onCopy,
  copiedField,
}: {
  reference: LegalReferenceDetail;
  sourceDetail?: SourceInfo;
  docMetadata: Record<string, unknown>;
  onCopy: (text: string, fieldName: string) => void;
  copiedField: string | null;
}) {
  return (
    <div className="space-y-3">
      <InfoField
        label="사건명"
        value={reference.case_name || sourceDetail?.case_name || (docMetadata.casenames as string | undefined)}
        fieldName="case_name"
        onCopy={onCopy}
        copiedField={copiedField}
      />
      <InfoField
        label="법원"
        value={reference.court || sourceDetail?.court || (docMetadata.court as string | undefined)}
        fieldName="court"
        onCopy={onCopy}
        copiedField={copiedField}
      />
      <InfoField
        label="사건번호"
        value={reference.case_number || sourceDetail?.case_number || (docMetadata.doc_id as string | undefined)}
        fieldName="case_number"
        onCopy={onCopy}
        copiedField={copiedField}
      />
      <InfoField
        label="판례일련번호"
        value={getMetadataValue(docMetadata.precedent_serial_number) || getMetadataValue(docMetadata.판례일련번호) || getMetadataValue(docMetadata.판례정보일련번호)}
        fieldName="precedent_serial_number"
        isMonospace
        onCopy={onCopy}
        copiedField={copiedField}
      />
      <InfoField
        label="판결일"
        value={reference.decision_date || getMetadataValue(docMetadata.announce_date)}
        fieldName="decision_date"
        onCopy={onCopy}
        copiedField={copiedField}
      />
    </div>
  );
}

function DecisionInfoSection({
  reference,
  sourceDetail,
  docMetadata,
  onCopy,
  copiedField,
}: {
  reference: LegalReferenceDetail;
  sourceDetail?: SourceInfo;
  docMetadata: Record<string, unknown>;
  onCopy: (text: string, fieldName: string) => void;
  copiedField: string | null;
}) {
  return (
    <div className="space-y-3">
      <InfoField
        label="기관"
        value={reference.org || sourceDetail?.org || getMetadataValue(docMetadata.org)}
        fieldName="org"
        onCopy={onCopy}
        copiedField={copiedField}
      />
      <InfoField
        label="일련번호"
        value={reference.decision_number || sourceDetail?.decision_number || getMetadataValue(docMetadata.doc_id)}
        fieldName="decision_number"
        onCopy={onCopy}
        copiedField={copiedField}
      />
      <InfoField
        label="헌재결정례일련번호"
        value={getMetadataValue(docMetadata.decision_serial_number) || getMetadataValue(docMetadata.헌재결정례일련번호)}
        fieldName="decision_serial_number"
        isMonospace
        onCopy={onCopy}
        copiedField={copiedField}
      />
      <InfoField
        label="결정일"
        value={sourceDetail?.decision_date || getMetadataValue(docMetadata.decision_date)}
        fieldName="decision_date"
        onCopy={onCopy}
        copiedField={copiedField}
      />
      <InfoField
        label="결과"
        value={reference.result || sourceDetail?.result || getMetadataValue(docMetadata.result)}
        fieldName="result"
        onCopy={onCopy}
        copiedField={copiedField}
      />
    </div>
  );
}

function InterpretationInfoSection({
  reference,
  sourceDetail,
  docMetadata,
  onCopy,
  copiedField,
}: {
  reference: LegalReferenceDetail;
  sourceDetail?: SourceInfo;
  docMetadata: Record<string, unknown>;
  onCopy: (text: string, fieldName: string) => void;
  copiedField: string | null;
}) {
  return (
    <div className="space-y-3">
      <InfoField
        label="제목"
        value={reference.title || sourceDetail?.title || getMetadataValue(docMetadata.title)}
        fieldName="title"
        onCopy={onCopy}
        copiedField={copiedField}
      />
      <InfoField
        label="기관"
        value={reference.org || sourceDetail?.org || getMetadataValue(docMetadata.org)}
        fieldName="org"
        onCopy={onCopy}
        copiedField={copiedField}
      />
      <InfoField
        label="일련번호"
        value={reference.interpretation_number || sourceDetail?.interpretation_number || getMetadataValue(docMetadata.doc_id)}
        fieldName="interpretation_number"
        onCopy={onCopy}
        copiedField={copiedField}
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
        onCopy={onCopy}
        copiedField={copiedField}
      />
      <InfoField
        label="회신일"
        value={sourceDetail?.response_date || getMetadataValue(docMetadata.response_date)}
        fieldName="response_date"
        onCopy={onCopy}
        copiedField={copiedField}
      />
    </div>
  );
}

function RegulationInfoSection({ orgValue }: { orgValue: string | number | undefined }) {
  if (!orgValue) return null;
  
  return (
    <div>
      <span className="text-xs font-medium text-slate-500">기관</span>
      <p className="text-sm text-slate-800 mt-1">
        {String(orgValue)}
      </p>
    </div>
  );
}

function LinkSection({
  generatedUrl,
  searchUrl,
}: {
  generatedUrl: string | null;
  searchUrl: string | null;
}) {
  return (
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
  );
}

function ContentSection({
  reference,
  sourceDetail,
}: {
  reference: LegalReferenceDetail;
  sourceDetail?: SourceInfo;
}) {
  const content = reference.article_content || 
                  reference.summary || 
                  reference.content || 
                  sourceDetail?.content;
  
  if (!content) return null;
  
  return (
    <div className="pt-3 border-t border-slate-200">
      <span className="text-xs font-medium text-slate-500 mb-2 block">상세본문</span>
      <div className="bg-slate-50 p-3 rounded-md max-h-96 overflow-y-auto">
        <p className="text-sm text-slate-700 whitespace-pre-wrap break-words">
          {content}
        </p>
      </div>
    </div>
  );
}

function SourceInfoSection({
  sourceDetail,
  onCopy,
  copiedField,
}: {
  sourceDetail?: SourceInfo;
  onCopy: (text: string, fieldName: string) => void;
  copiedField: string | null;
}) {
  const hasSourceInfo = sourceDetail?.chunk_id || 
                        sourceDetail?.source_id || 
                        sourceDetail?.original_url || 
                        sourceDetail?.metadata?.chunk_id || 
                        sourceDetail?.metadata?.source_id || 
                        sourceDetail?.metadata?.original_url;
  
  if (!hasSourceInfo) return null;
  
  return (
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
          onCopy={onCopy}
          copiedField={copiedField}
        />
        <InfoField
          label="Chunk ID"
          value={(sourceDetail?.chunk_id ? String(sourceDetail.chunk_id) : undefined) || (sourceDetail?.metadata?.chunk_id ? String(sourceDetail.metadata.chunk_id) : undefined)}
          fieldName="chunk_id"
          isMonospace
          onCopy={onCopy}
          copiedField={copiedField}
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
  );
}

// ============================================================================
// 메인 컴포넌트
// ============================================================================

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
  
  const documentType = reference.type as DocumentType;
  const documentName = useMemo(() => getDocumentName(reference), [reference]);
  const docMetadata = useMemo(() => sourceDetail?.metadata || {}, [sourceDetail?.metadata]);
  
  const generatedUrl = useMemo(() => {
    if (sourceDetail?.url) {
      return sourceDetail.url;
    }
    
    const metadata = buildMetadataForUrl(reference, sourceDetail, docMetadata);
    const urlType = getDocumentTypeUrlType(documentType);
    return generateLawUrl(urlType, metadata);
  }, [reference, sourceDetail, docMetadata, documentType]);
  
  const searchUrl = useMemo(() => {
    if (generatedUrl) return null;
    
    const metadata = buildMetadataForSearch(reference, sourceDetail, docMetadata);
    const urlType = getDocumentTypeUrlType(documentType);
    return generateSearchUrl(urlType, metadata);
  }, [generatedUrl, reference, sourceDetail, docMetadata, documentType]);
  
  const orgValue = useMemo(() => {
    return reference.org || sourceDetail?.org || getMetadataValue(docMetadata.org);
  }, [reference.org, sourceDetail?.org, docMetadata]);
  
  const handleCopy = async (text: string, fieldName: string) => {
    await copyToClipboardWithFeedback(
      text,
      () => setCopiedField(fieldName),
      () => {}
    );
  };
  
  const renderDocumentInfo = () => {
    switch (documentType) {
      case 'law':
        return (
          <LawInfoSection
            reference={reference}
            sourceDetail={sourceDetail}
            docMetadata={docMetadata}
            onCopy={handleCopy}
            copiedField={copiedField}
          />
        );
      case 'precedent':
        return (
          <PrecedentInfoSection
            reference={reference}
            sourceDetail={sourceDetail}
            docMetadata={docMetadata}
            onCopy={handleCopy}
            copiedField={copiedField}
          />
        );
      case 'decision':
        return (
          <DecisionInfoSection
            reference={reference}
            sourceDetail={sourceDetail}
            docMetadata={docMetadata}
            onCopy={handleCopy}
            copiedField={copiedField}
          />
        );
      case 'interpretation':
        return (
          <InterpretationInfoSection
            reference={reference}
            sourceDetail={sourceDetail}
            docMetadata={docMetadata}
            onCopy={handleCopy}
            copiedField={copiedField}
          />
        );
      case 'regulation':
        return <RegulationInfoSection orgValue={orgValue} />;
      default:
        return null;
    }
  };
  
  return (
    <div className="flex flex-col h-full">
      <Header
        documentType={documentType}
        documentName={documentName}
        onBack={onBack}
        onClose={onClose}
      />
      
      <div className="flex-1 overflow-y-auto p-4">
        <div className="space-y-4">
          <DocumentTypeBadge documentType={documentType} />
          
          <div>
            <h3 className="text-base font-semibold text-slate-800 mb-2">{documentName}</h3>
          </div>
          
          {renderDocumentInfo()}
          
          <LinkSection generatedUrl={generatedUrl} searchUrl={searchUrl} />
          
          <ContentSection reference={reference} sourceDetail={sourceDetail} />
          
          <SourceInfoSection
            sourceDetail={sourceDetail}
            onCopy={handleCopy}
            copiedField={copiedField}
          />
        </div>
      </div>
    </div>
  );
}
