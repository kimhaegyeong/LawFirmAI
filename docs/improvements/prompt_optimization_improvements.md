# 프롬프트 최적화 개선사항

## 개요

`generate_answer_final` 노드에서 LLM에 전달되는 프롬프트의 비효율성을 제거하고, 문서 정보를 더 간결하고 명확하게 표시하기 위한 개선사항입니다.

## 개선 목표

1. **토큰 사용량 감소**: 불필요한 메타데이터 및 중복 정보 제거
2. **가독성 향상**: 문서 정보를 간결하고 명확하게 표시
3. **일관성 확보**: 문서 포맷팅 및 관련도 점수 표기 통일
4. **정확성 향상**: 문서 출처 정보 명확화

## 개선사항 상세

### 1. 문서 메타데이터 제거 강화

**문제점**:
- 프롬프트에 `'query'`, `'cross_encoder_score'`, `'original_score'`, `'keyword_bonus'` 등 불필요한 메타데이터가 포함됨
- 중첩된 딕셔너리 구조의 메타데이터가 완전히 제거되지 않음

**개선 방안**:
- `_clean_content` 메서드의 정규식 강화
- 중첩된 딕셔너리 패턴 처리 추가
- 문서 정규화 단계에서 메타데이터 필터링 강화

**수정 파일**: `lawfirm_langgraph/core/services/unified_prompt_manager.py`
- `_clean_content` 메서드 개선
- `_normalize_document_fields` 메서드에서 메타데이터 필터링 강화

### 2. 문서 중복 표시 제거

**문제점**:
- 같은 문서가 여러 섹션에 중복 표시됨
- "### 문서 1", "**[문서 1]**", "[문서 출처: 1]" 등 여러 형식으로 표시

**개선 방안**:
- 문서 섹션 생성 전 중복 체크 로직 추가
- 문서 ID 기반 중복 제거
- 단일 문서 섹션만 유지

**수정 파일**: `lawfirm_langgraph/core/services/unified_prompt_manager.py`
- `_build_final_prompt` 메서드에서 중복 체크 추가
- `_build_documents_section` 메서드 개선

### 3. 불필요한 섹션 제목 제거

**문제점**:
- "문서 정보", "핵심 내용", "관련 문장", "키워드 관련 문맥" 등 불필요한 제목이 많음
- LLM이 문서 내용을 이해하는 데 방해가 됨

**개선 방안**:
- 프롬프트용 문서 포맷팅에서는 제목 제거 또는 간소화
- 문서 제목과 내용만 포함

**수정 파일**: `lawfirm_langgraph/core/services/unified_prompt_manager.py`
- `_format_document_for_prompt` 메서드 개선
- `_build_documents_section` 메서드에서 간소화된 형식 사용

### 4. 딕셔너리 형태의 문서 내용 제거

**문제점**:
- `{'text': '...', 'content': '...'}` 형태로 표시됨
- LLM이 이해하기 어려운 형식

**개선 방안**:
- 딕셔너리 키를 제거하고 실제 내용만 추출
- `_normalize_document_fields`에서 content 필드만 사용

**수정 파일**: `lawfirm_langgraph/core/services/unified_prompt_manager.py`
- `_normalize_document_fields` 메서드 개선
- `_clean_content` 메서드에서 딕셔너리 패턴 제거 강화

### 5. 문서 출처 정보 명확화

**문제점**:
- "문서 1", "문서 2" 등으로만 표시되고 법령명/판례명이 명확하지 않음
- 일부 문서는 제목이 없음

**개선 방안**:
- 법령명/조문번호 또는 판례명/법원명을 우선 표시
- 제목이 없는 경우 source 필드 활용

**수정 파일**: `lawfirm_langgraph/core/services/unified_prompt_manager.py`
- `_get_document_title_and_max_length` 메서드 개선
- `_build_documents_section` 메서드에서 제목 생성 로직 강화

### 6. 관련도 점수 표기 통일

**문제점**:
- 관련도 점수가 `.2f`, `.3f` 등으로 혼재
- 여러 메서드에서 서로 다른 포맷 사용

**개선 방안**:
- 관련도 점수 표기 형식을 `.2f`로 통일
- 모든 문서 포맷팅 메서드에서 일관된 형식 사용

**수정 파일**: `lawfirm_langgraph/core/services/unified_prompt_manager.py`
- 모든 관련도 점수 표기를 `.2f`로 통일
- 상수 정의 추가 (선택사항)

### 7. 문서 정보 섹션 구조 간소화

**문제점**:
- "문서 정보", "전체 문서 길이", "추출된 핵심 내용", "축약 비율" 등 불필요한 정보 포함
- LLM이 이해하기 어려운 구조

**개선 방안**:
- 문서 제목과 내용만 포함하도록 간소화
- 불필요한 메타데이터 제거

**수정 파일**: `lawfirm_langgraph/core/services/unified_prompt_manager.py`
- `_build_fallback_documents_section` 메서드 개선
- `_format_document_for_prompt` 메서드 간소화

### 8. 검색 쿼리 정보 제거 강화

**문제점**:
- 프롬프트에 `'query': '계약서 작성 시 주의할 사항은 무엇인가요'` 등이 남아있음
- 정규식이 모든 쿼리 정보를 제거하지 못함

**개선 방안**:
- 쿼리 정보 제거 정규식 강화
- 다양한 패턴 처리 추가

**수정 파일**: `lawfirm_langgraph/core/services/unified_prompt_manager.py`
- `_clean_content` 메서드에서 쿼리 정보 제거 강화

### 9. 문서 내용 축약 로직 적용

**문제점**:
- 일부 문서가 너무 길게 표시됨
- `_smart_truncate_document`가 모든 경로에서 호출되지 않음

**개선 방안**:
- 모든 문서 포맷팅 경로에서 축약 로직 적용
- 토큰 제한 내에서 최적화

**수정 파일**: `lawfirm_langgraph/core/services/unified_prompt_manager.py`
- `_build_documents_section` 메서드에서 축약 로직 적용
- `_format_document_for_prompt` 메서드에서도 축약 로직 적용

### 10. 문서 번호 매핑 일관성 확보

**문제점**:
- 같은 문서가 다른 번호로 표시될 수 있음
- 여러 섹션에서 각각 번호를 부여함

**개선 방안**:
- 문서 ID 기반으로 일관된 번호 매핑
- 전역 문서 번호 매핑 사용

**수정 파일**: `lawfirm_langgraph/core/services/unified_prompt_manager.py`
- `_build_final_prompt` 메서드에서 문서 번호 매핑 생성
- 모든 문서 섹션에서 동일한 번호 사용

### 11. 불필요한 예시 섹션 제거

**문제점**:
- 프롬프트 하단에 "📚 답변 형식 가이드" 등 예시가 중복 포함될 수 있음
- base_prompt에 이미 예시가 포함되어 있는 경우 중복

**개선 방안**:
- 예시 섹션 중복 체크 및 통합
- base_prompt에 예시가 있으면 추가 예시 제거

**수정 파일**: `lawfirm_langgraph/core/services/unified_prompt_manager.py`
- `_build_final_prompt` 메서드에서 예시 섹션 중복 체크
- `_remove_duplicate_citation_requirements` 메서드 확장

### 12. 관련도 점수 표기 위치 통일

**문제점**:
- 관련도 점수가 제목 옆, 내용 옆 등 여러 위치에 표시됨
- 일관성이 없음

**개선 방안**:
- 관련도 점수 표기 위치를 제목 옆으로 통일
- 형식: `**[문서 N]** 제목 (관련도: 0.XX)`

**수정 파일**: `lawfirm_langgraph/core/services/unified_prompt_manager.py`
- `_build_documents_section` 메서드에서 위치 통일
- `_format_document_for_prompt` 메서드에서도 위치 통일

## 구현 완료 사항

### ✅ 완료된 개선사항

1. **메타데이터 제거 강화** (`_clean_content` 메서드)
   - 중첩된 딕셔너리 패턴 처리 추가
   - 쿼리 정보 제거 강화 (다양한 패턴 처리)
   - 점수 정보 제거 강화 (추가 필드 포함)
   - 딕셔너리 키 패턴 제거 추가

2. **문서 중복 제거** (`_build_final_prompt` 메서드)
   - `_generate_document_id` 메서드 추가
   - 문서 정규화 단계에서 중복 체크
   - 중복 문서 자동 제거

3. **문서 포맷팅 간소화** (`_format_document_for_prompt` 메서드)
   - 불필요한 섹션 제목 제거
   - 간결한 형식으로 변경
   - 관련도 점수 표기 통일 (.2f)

4. **문서 출처 정보 명확화** (`_get_document_title_and_max_length` 메서드)
   - 법령 조문: "민법 제543조" 형식
   - 판례: "대법원 판례명" 또는 "서울고등법원 판례명" 형식
   - 판례 번호 포함 지원

5. **관련도 점수 표기 통일** (`_build_documents_section` 메서드)
   - 모든 관련도 점수를 `.2f` 형식으로 통일
   - 제목 옆에 일관된 위치로 표시

6. **메타데이터 필터링 강화** (`_normalize_document_fields` 메서드)
   - EXCLUDED_METADATA_FIELDS에 있는 필드 제거
   - 정규화된 문서에서 불필요한 메타데이터 제외

## 예상 효과

- **토큰 사용량**: 약 20-30% 감소 예상
- **가독성**: 문서 정보가 더 명확하고 간결하게 표시
- **일관성**: 모든 문서가 동일한 형식으로 표시
- **정확성**: 문서 출처 정보가 명확해져 LLM이 더 정확한 인용 가능

## 테스트 계획

1. 기존 프롬프트와 개선된 프롬프트 비교
2. 토큰 사용량 측정
3. 답변 품질 검증 (인용 정확도, 문서 참조 정확도)
4. 성능 테스트 (응답 시간, 메모리 사용량)

