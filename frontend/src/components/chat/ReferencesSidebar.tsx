/**
 * 참고자료 사이드바 컴포넌트
 * 오른쪽에서 슬라이드되는 사이드바로 참고자료 목록을 표시합니다.
 */
import { useState } from 'react';
import { X } from 'lucide-react';
import { ReferencesModalContent } from './ReferencesModalContent';
import { ReferenceDetailView } from './ReferenceDetailView';
import type { SourceInfo, LegalReferenceDetail } from '../../types/chat';

interface ReferencesSidebarProps {
  isOpen: boolean;
  onClose: () => void;
  references?: string[];
  legalReferences?: string[];
  sources?: string[];
  sourcesDetail?: SourceInfo[];
  initialSelectedType?: 'all' | 'law' | 'precedent' | 'decision' | 'interpretation' | 'regulation';
}

export function ReferencesSidebar({
  isOpen,
  onClose,
  references = [],
  legalReferences = [],
  sources = [],
  sourcesDetail = [],
  initialSelectedType = 'all',
}: ReferencesSidebarProps) {
  const [selectedReference, setSelectedReference] = useState<LegalReferenceDetail | null>(null);
  const [selectedSourceDetail, setSelectedSourceDetail] = useState<SourceInfo | null>(null);

  if (!isOpen) {
    return null;
  }

  const totalCount = references.length + legalReferences.length + sources.length + sourcesDetail.length;

  // 참고자료 클릭 핸들러
  const handleReferenceClick = (reference: LegalReferenceDetail, sourceDetail?: SourceInfo) => {
    setSelectedReference(reference);
    setSelectedSourceDetail(sourceDetail || null);
  };

  // 뒤로 가기 핸들러
  const handleBack = () => {
    setSelectedReference(null);
    setSelectedSourceDetail(null);
  };

  return (
    <>
      {/* 사이드바 */}
      <div
        className={`fixed right-0 top-0 h-full w-full max-w-md bg-white shadow-xl z-50 transform transition-transform duration-300 ease-in-out ${
          isOpen ? 'translate-x-0' : 'translate-x-full'
        }`}
        role="dialog"
        aria-modal="true"
        aria-labelledby="references-sidebar-title"
      >
        <div className="flex flex-col h-full">
          {/* 선택된 참고자료가 있으면 상세 정보 표시, 없으면 목록 표시 */}
          {selectedReference ? (
            <ReferenceDetailView
              reference={selectedReference}
              sourceDetail={selectedSourceDetail || undefined}
              onBack={handleBack}
              onClose={onClose}
            />
          ) : (
            <>
              {/* 헤더 */}
              <div className="flex items-center justify-between p-4 border-b border-slate-200 bg-slate-50">
                <h2 id="references-sidebar-title" className="text-lg font-semibold text-slate-800">
                  참고자료 ({totalCount}개)
                </h2>
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
                <ReferencesModalContent
                  references={references}
                  legalReferences={legalReferences}
                  sources={sources}
                  sourcesDetail={sourcesDetail}
                  initialSelectedType={initialSelectedType}
                  onReferenceClick={handleReferenceClick}
                />
              </div>
            </>
          )}
        </div>
      </div>
    </>
  );
}

