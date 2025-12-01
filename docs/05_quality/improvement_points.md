# 테스트 결과 개선사항 (번호순)

## 테스트 결과 통계
- **Text too short**: 102회
- **Missing required fields**: 213회
- **Empty text content**: 46회
- **No results found**: 43회
- **Valid docs 0개**: 0회 ✅ (해결됨)
- **최대 semantic results**: 117개

---

## 개선사항 (우선순위별)

### 🔴 P0 (CRITICAL) - 즉시 해결 필요

#### 1. FAISS Index와 Chunk IDs 불일치
- **문제**: `_chunk_ids length (32583) != FAISS index ntotal (26630)`
- **원인**: 인덱스 빌드 시 chunk_ids와 실제 FAISS 인덱스 크기가 불일치
- **영향**: 약 5,953개의 chunk가 인덱스에서 누락되어 검색 결과 손실
- **해결 방안**:
  - 인덱스 빌드 시 chunk_ids와 FAISS 인덱스 크기 일치 검증 추가
  - 불일치 시 경고 및 자동 조정 로직 구현
  - 인덱스 재빌드 시 일관성 보장

#### 2. Metadata 필드 복원 실패 (Missing required fields: 213회)
- **문제**: 
  - `case_paragraph`: `doc_id`, `casenames`, `court` 누락
  - `decision_paragraph`: `org`, `doc_id` 누락
  - `interpretation_paragraph`: `interpretation_id` 누락
- **원인**: 
  - `source_id=None`인 경우 메타데이터 복원 실패
  - 복원 로직이 일부 케이스를 커버하지 못함
- **영향**: 검색 결과의 메타데이터 부족으로 인한 품질 저하
- **해결 방안**:
  - `source_id=None`인 경우 `chunk_id`로 직접 조회하는 로직 강화
  - `case_paragraph`의 `casenames` 복원 로직 추가
  - `decision_paragraph`의 `org` 복원 로직 추가
  - `interpretation_paragraph`의 `interpretation_id` 복원 로직 추가

#### 3. Empty Text Content (46회)
- **문제**: `source_id=None`인 경우 텍스트 복원 실패
- **원인**: `_ensure_text_content`에서 `source_id=None` 처리 미흡
- **영향**: 검색 결과의 텍스트 내용 부족
- **해결 방안**:
  - `source_id=None`인 경우 `chunk_id`로 직접 `text_chunks` 테이블 조회
  - 복원 실패 시 기본값 설정 또는 스킵 로직 개선

#### 3-1. os 변수 오류 (여전히 발생)
- **문제**: `Failed to get full text from database: cannot access local variable 'os' where it is not associated with a value`
- **원인**: `answer_formatter.py` 외 다른 파일에서도 `os` import 누락 가능
- **영향**: 텍스트 복원 실패로 인한 소스 정보 손실
- **해결 방안**:
  - 모든 관련 파일에서 `import os` 확인 및 추가
  - `grep`으로 `os.` 사용하는 모든 파일 검색 및 수정

---

### 🟠 P1 (HIGH) - 빠른 개선 필요

#### 4. Text Too Short 문제 (102회)
- **문제**: 최소 길이 10자 기준으로 인한 문서 필터링
- **현재 상태**: 
  - `case_paragraph`/`decision_paragraph`: 최소 10자
  - `statute_article`: 최소 30자
  - 기타: 최소 50자
- **원인**: 일부 문서가 실제로 매우 짧음 (4~9자)
- **영향**: 유효한 문서가 필터링되어 검색 결과 손실
- **해결 방안**:
  - 문서 타입별 최소 길이 기준 재검토
  - `case_paragraph`/`decision_paragraph`: 10자 → 5자로 완화
  - 텍스트 복원 후에도 짧은 경우, 인접 chunk와 병합 고려

#### 5. No Results Found (43회)
- **문제**: 일부 Multi-Query에서 검색 결과 없음
- **원인**:
  - 쿼리가 너무 길거나 복잡함
  - `similarity_threshold`가 너무 높음
  - `source_types` 필터가 너무 제한적
- **영향**: Multi-Query의 효과 감소
- **해결 방안**:
  - Fallback 검색 로직 강화 (이미 구현됨, 추가 개선 필요)
  - 쿼리 길이에 따른 동적 `similarity_threshold` 조정
  - `source_types` 필터 완화 또는 제거

#### 6. Performance 이슈: expand_keywords 느림 (6.17초)
- **문제**: `expand_keywords` 노드가 5초 임계값 초과
- **원인**: LLM 호출 또는 키워드 확장 로직 비효율
- **영향**: 전체 워크플로우 처리 시간 증가
- **해결 방안**:
  - 키워드 확장 로직 최적화
  - LLM 호출 캐싱 강화
  - 병렬 처리 고려

---

### 🟡 P2 (MEDIUM) - 점진적 개선

#### 7. Metadata 복원 로직 개선
- **문제**: `court`, `casenames`, `org` 등 일부 필드 복원 실패
- **해결 방안**:
  - `_restore_missing_metadata` 메서드의 fallback 로직 강화
  - 여러 테이블 조인을 통한 복원 시도
  - 복원 실패 시 기본값 설정

#### 8. 검색 결과 검증 개선
- **문제**: 검증 후에도 일부 문제가 남아있음
- **해결 방안**:
  - 검증 로직을 더 엄격하게 적용
  - 검증 실패 시 자동 복원 시도
  - 검증 결과 로깅 강화

#### 9. Multi-Query 최적화
- **문제**: 일부 Multi-Query가 중복되거나 비효율적
- **해결 방안**:
  - Multi-Query 생성 로직 개선
  - 중복 쿼리 제거
  - 쿼리 다양성 향상

---

### 🟢 P3 (LOW) - 장기 개선

#### 10. 로깅 및 모니터링 개선
- **문제**: 일부 경고가 과도하게 출력됨
- **해결 방안**:
  - 로그 레벨 조정
  - 중요한 경고만 출력
  - 통계 집계 및 리포트 생성

#### 11. 인덱스 빌드 프로세스 개선
- **문제**: 인덱스 빌드 시 chunk_ids 불일치
- **해결 방안**:
  - 빌드 전 검증 강화
  - 빌드 후 일관성 검증
  - 자동 재빌드 로직

---

## 우선순위별 작업 계획

### 즉시 작업 (P0)
1. ✅ FAISS Index와 Chunk IDs 불일치 해결
2. ✅ Metadata 필드 복원 로직 강화
3. ✅ Empty Text Content 복원 로직 개선

### 단기 작업 (P1)
4. ✅ Text Too Short 기준 완화
5. ✅ No Results Found Fallback 강화
6. ⏳ Performance 이슈 조사 및 개선

### 중기 작업 (P2)
7. ⏳ Metadata 복원 로직 추가 개선
8. ⏳ 검색 결과 검증 개선
9. ⏳ Multi-Query 최적화

### 장기 작업 (P3)
10. ⏳ 로깅 및 모니터링 개선
11. ⏳ 인덱스 빌드 프로세스 개선

