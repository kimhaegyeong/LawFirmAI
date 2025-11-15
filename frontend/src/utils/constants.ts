/**
 * 상수 정의
 */

export const QUICK_QUESTIONS = [
  { title: '계약 관련', icon: '📄', questions: [
    { title: '계약서 검토', question: '계약서 작성 시 주의사항은 무엇인가요?' },
    { title: '위약금', question: '계약 위약금에 대해 설명해주세요' },
    { title: '계약 해제', question: '계약 해제 조건은 무엇인가요?' },
  ]},
  { title: '임대차 분쟁', icon: '🏠', questions: [
    { title: '임대차 분쟁', question: '임대차 분쟁 시 해결 방법은?' },
    { title: '전세금 반환', question: '전세금 반환 보증에 대해 알려주세요' },
    { title: '임대차 보호법', question: '임대차 보호법의 주요 내용은?' },
  ]},
  { title: '근로 관련', icon: '💼', questions: [
    { title: '근로계약', question: '근로계약서 작성 시 필수 항목은?' },
    { title: '퇴직금', question: '퇴직금 계산 방법은?' },
    { title: '근로시간', question: '근로시간 제한에 대해 설명해주세요' },
  ]},
  { title: '손해배상', icon: '⚖️', questions: [
    { title: '손해배상 청구', question: '손해배상 청구 요건은 무엇인가요?' },
    { title: '과실 상계', question: '과실 상계에 대해 설명해주세요' },
    { title: '배상 범위', question: '손해배상의 범위는 어떻게 결정되나요?' },
  ]},
] as const;

export const SESSION_CATEGORIES = [
  '전체',
  '계약/합의',
  '분쟁/소송',
  '노동/근로',
  '부동산',
  '형사',
  '기타',
] as const;

export const SORT_OPTIONS = [
  { value: 'updated_at', label: '최근 활동순' },
  { value: 'created_at', label: '생성일순' },
  { value: 'title', label: '제목순' },
  { value: 'message_count', label: '메시지 많은 순' },
] as const;

