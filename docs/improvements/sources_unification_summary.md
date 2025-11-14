# Sources 통일 구현 완료 요약

## 구현 완료 사항

### Backend (Python)

#### 1. `api/services/sources_extractor.py`
- ✅ `_get_sources_by_type()` 추가: sources_detail을 타입별로 그룹화
- ✅ `_format_legal_reference_from_detail()` 추가: sources_detail 항목에서 legal_reference 문자열 생성
- ✅ `_extract_legal_references_from_sources_detail_only()` 추가: sources_detail에서만 legal_references 추출
- ✅ `extract_from_state()` 수정: sources_by_type 추가, legal_references를 sources_detail에서 추출
- ✅ `_extract_from_message()` 수정: sources_by_type 추가, legal_references를 sources_detail에서 추출

#### 2. `api/routers/chat.py`
- ✅ `_create_sources_event()` 수정: sources_by_type 필드 추가

#### 3. `api/services/streaming/stream_handler.py`
- ✅ `_get_final_metadata()` 수정: sources_by_type 필드 추가

### Frontend (TypeScript/React)

#### 1. `frontend/src/utils/sourcesParser.ts`
- ✅ `SourcesByType` 인터페이스 추가
- ✅ `getSourcesByType()` 함수 추가: sources_detail을 타입별로 그룹화
- ✅ `extractLegalReferencesFromSourcesDetail()` 함수 추가: sources_detail에서 legal_references 추출
- ✅ `parseSourcesMetadata()` 수정: sourcesByType 필드 추가, legal_references를 sources_detail에서 추출

#### 2. `frontend/src/components/chat/CompactReferencesBadge.tsx`
- ✅ `getSourcesByType()`, `extractLegalReferencesFromSourcesDetail()` import 추가
- ✅ sourcesByType 사용하여 타입별 개수 계산
- ✅ legal_references를 sources_detail에서 추출하여 사용

## 주요 변경 사항

### 1. 새로운 필드: `sources_by_type`

**Backend 응답 구조:**
```json
{
  "sources": [...],
  "legal_references": [...],  // deprecated
  "sources_detail": [...],
  "sources_by_type": {
    "statute_article": [...],
    "case_paragraph": [...],
    "decision_paragraph": [...],
    "interpretation_paragraph": [...]
  }
}
```

**Frontend 타입:**
```typescript
interface SourcesByType {
  statute_article: SourceInfo[];
  case_paragraph: SourceInfo[];
  decision_paragraph: SourceInfo[];
  interpretation_paragraph: SourceInfo[];
}
```

### 2. legal_references 처리 변경

**이전:**
- `legal_references`는 별도로 추출하여 저장
- 법령만 `legal_references`에 포함

**현재:**
- `legal_references`는 `sources_detail`에서 자동 추출
- 모든 문서 타입이 `sources_detail`에 포함
- `legal_references`는 하위 호환성을 위해 유지 (deprecated)

### 3. 타입별 그룹화 헬퍼

**Backend:**
```python
sources_by_type = self._get_sources_by_type(sources_detail)
```

**Frontend:**
```typescript
const sourcesByType = getSourcesByType(sourcesDetail);
// 또는
const { sourcesByType } = parseSourcesMetadata(metadata);
```

## 하위 호환성

1. **legal_references 필드 유지**
   - 기존 클라이언트와의 호환성을 위해 계속 제공
   - `sources_detail`에서 자동 추출하여 병합

2. **기존 API 응답 구조 유지**
   - 기존 필드(`sources`, `legal_references`, `sources_detail`) 모두 유지
   - 새로운 필드(`sources_by_type`) 추가

3. **점진적 마이그레이션**
   - 프론트엔드에서 `sourcesByType` 사용 가능
   - 기존 `legalReferences` 사용도 계속 작동

## 사용 예시

### Backend에서 sources_by_type 생성

```python
# sources_extractor.py
sources_by_type = self._get_sources_by_type(sources_detail)

# 결과:
# {
#   "statute_article": [{"type": "statute_article", "statute_name": "민법", ...}, ...],
#   "case_paragraph": [{"type": "case_paragraph", "case_number": "2021다123", ...}, ...],
#   ...
# }
```

### Frontend에서 sourcesByType 사용

```typescript
// 방법 1: parseSourcesMetadata 사용
const { sourcesByType } = parseSourcesMetadata(metadata);
const statutes = sourcesByType.statute_article;
const cases = sourcesByType.case_paragraph;

// 방법 2: 직접 필터링
const statutes = sourcesDetail.filter(d => d.type === 'statute_article');
const cases = sourcesDetail.filter(d => d.type === 'case_paragraph');

// 방법 3: getSourcesByType 사용
const sourcesByType = getSourcesByType(sourcesDetail);
const statutes = sourcesByType.statute_article;
```

## 다음 단계

### 권장 사항

1. **프론트엔드 마이그레이션**
   - `ReferencesModalContent.tsx`에서 `sourcesByType` 사용
   - `ReferencesSection.tsx`에서 `sourcesByType` 사용
   - `DocumentSidebar.tsx`에서 `sourcesByType` 사용

2. **테스트**
   - 단위 테스트: `_get_sources_by_type()`, `getSourcesByType()` 등
   - 통합 테스트: 전체 스트리밍 플로우에서 `sources_by_type` 포함 확인
   - 하위 호환성 테스트: `legal_references` 여전히 작동하는지 확인

3. **문서화**
   - API 문서에 `sources_by_type` 필드 추가
   - 마이그레이션 가이드 작성

### 장기 계획

1. **legal_references 제거 (Phase 4)**
   - 모든 프론트엔드에서 `sources_detail` 기반으로 변경 완료 후
   - 백엔드에서 `legal_references` 필드 제거
   - 관련 코드 정리

## 참고 문서

- [구현 계획](./sources_unification_implementation_plan.md)
- [Stream API 응답 개선 방안](./stream_api_response_improvements.md)

