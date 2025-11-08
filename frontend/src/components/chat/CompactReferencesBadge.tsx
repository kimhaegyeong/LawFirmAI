/**
 * 컴팩트 참고자료 배지 컴포넌트
 * 플로팅 버튼 형태로 참고자료 개수를 표시하고, 클릭 시 모달을 열어 상세 정보를 보여줍니다.
 */
import { FileText } from 'lucide-react';
import { useState } from 'react';
import { Modal } from '../common/Modal';
import { ReferencesModalContent } from './ReferencesModalContent';

import type { SourceInfo } from '../../types/chat';

interface CompactReferencesBadgeProps {
  references?: string[];
  legalReferences?: string[];
  sources?: string[];
  sourcesDetail?: SourceInfo[];
}

export function CompactReferencesBadge({
  references = [],
  legalReferences = [],
  sources = [],
  sourcesDetail = [],
}: CompactReferencesBadgeProps) {
  const [isModalOpen, setIsModalOpen] = useState(false);

  // 전체 참고자료 개수 계산
  const totalCount = references.length + legalReferences.length + sources.length + sourcesDetail.length;

  // 참고자료가 없으면 렌더링하지 않음
  if (totalCount === 0) {
    return null;
  }

  return (
    <>
      <button
        onClick={() => setIsModalOpen(true)}
        className="mt-3 inline-flex items-center gap-1.5 px-3 py-1.5 
                   bg-blue-50 text-blue-600 text-xs font-medium rounded-full 
                   hover:bg-blue-100 transition-colors
                   border border-blue-200 shadow-sm
                   active:scale-95 transition-transform"
        title="참고 법령/판례 보기"
      >
        <FileText className="w-3.5 h-3.5 flex-shrink-0" />
        <span>참고자료 {totalCount}개</span>
      </button>

      <Modal
        isOpen={isModalOpen}
        onClose={() => setIsModalOpen(false)}
        title={`참고 법령/판례 (${totalCount}개)`}
        size="xl"
      >
        <ReferencesModalContent
          references={references}
          legalReferences={legalReferences}
          sources={sources}
          sourcesDetail={sourcesDetail}
        />
      </Modal>
    </>
  );
}

