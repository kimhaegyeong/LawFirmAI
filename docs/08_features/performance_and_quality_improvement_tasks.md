# 성능 및 품질 개선 작업 계획서

## 📋 목차

1. [개요](#개요)
2. [우선순위별 작업 목록](#우선순위별-작업-목록)
3. [상세 작업 계획](#상세-작업-계획)
4. [예상 효과](#예상-효과)
5. [검증 방법](#검증-방법)

---

## 개요

본 문서는 `test_langgraph_query_20251129_160731.log` 분석 결과를 바탕으로 한 성능 및 품질 개선 작업 계획서입니다.

**분석 기준**: `logs/langgraph/test_langgraph_query_20251129_160731.log`  
**작성일**: 2025-11-29  
**목표**: 전체 처리 시간 50% 단축, 검색 성공률 90% 이상 달성

---

## 우선순위별 작업 목록

### 🔴 높은 우선순위 (즉시 개선 필요)

1. **TASK-001**: 데이터베이스 커서 생명주기 관리 개선
2. **TASK-002**: 검색 결과 없음 문제 해결
3. **TASK-003**: 초기화 시간 최적화
4. **TASK-004**: 검색 타입 불일치 문제 해결

### 🟡 중간 우선순위 (단기 개선 필요)

5. **TASK-005**: Metadata 캐시 효율성 개선
6. **TASK-006**: 검색 임계값 동적 조정
7. **TASK-007**: 프롬프트 검증 로직 수정
8. **TASK-008**: 검색 성능 편차 최소화

### 🟢 낮은 우선순위 (중장기 개선)

9. **TASK-009**: LangChain 통합 개선
10. **TASK-010**: 동의어 데이터베이스 설정
11. **TASK-011**: MLflow 백엔드 전환
12. **TASK-012**: 모델 캐시 로드 오류 수정

---

## 상세 작업 계획

### TASK-001: 데이터베이스 커서 생명주기 관리 개선

**문제점**:
- `⚠️ Failed to load chunk_id=... from DB: cursor already closed` 경고 15회 이상 발생
- 연결 보유 시간 경고: `Connection held for 1.17s (queries: 27)`

**목표**:
- 커서 관련 경고 0회 달성
- 연결 보유 시간 0.5초 이하로 단축

**작업 내용**:

1. **커서 생명주기 관리 개선**
   - 파일: `lawfirm_langgraph/core/search/engines/semantic_search_engine_v2.py`
   - `_load_chunk_metadata` 메서드에서 커서 생명주기 명시적 관리
   - Context manager 사용 강화
   - 예외 처리 시 커서 정리 보장

2. **연결 풀 최적화**
   - 파일: `lawfirm_langgraph/core/data/db_adapter.py`
   - 연결 보유 시간 모니터링 강화
   - 쿼리 배치 처리 최적화
   - 불필요한 연결 보유 시간 단축

3. **재시도 로직 개선**
   - 커서 닫힘 오류 발생 시 자동 재시도
   - 재시도 횟수 제한 및 백오프 전략

**예상 소요 시간**: 4시간  
**담당자**: Backend Developer  
**우선순위**: 🔴 높음

---

### TASK-002: 검색 결과 없음 문제 해결

**문제점**:
- `⚠️ No results found for query` 경고 다수 발생
- Threshold 미달로 인한 강제 결과 반환 빈번
- FALLBACK 발생 빈도 높음

**목표**:
- 검색 성공률 90% 이상 달성
- FALLBACK 발생률 10% 이하로 감소

**작업 내용**:

1. **쿼리 확장 개선**
   - 파일: `lawfirm_langgraph/core/search/processors/query_expander.py`
   - 동의어 확장 강화
   - 법률 용어 사전 활용
   - 쿼리 재작성 로직 개선

2. **임계값 동적 조정**
   - 파일: `lawfirm_langgraph/core/search/engines/semantic_search_engine_v2.py`
   - 초기 임계값을 0.4로 낮춤 (현재 0.5)
   - 검색 결과에 따라 동적 조정
   - 테이블별 최적 임계값 설정

3. **검색 전략 다양화**
   - Hybrid 검색 (벡터 + 키워드) 강화
   - Multi-query 검색 품질 향상
   - 검색 결과 부족 시 자동 확장

4. **FALLBACK 로직 개선**
   - FALLBACK 발생 전 더 많은 시도
   - FALLBACK 시 로깅 강화
   - 사용자 피드백 수집

**예상 소요 시간**: 8시간  
**담당자**: Search Engine Developer  
**우선순위**: 🔴 높음

---

### TASK-003: 초기화 시간 최적화

**문제점**:
- 설정 로드 시간: 15.480초
- 서비스 초기화 총 시간: 31.201초

**목표**:
- 초기화 시간 50% 단축 (15초 이하)
- 설정 로드 시간 5초 이하

**작업 내용**:

1. **지연 로딩 적용**
   - 파일: `lawfirm_langgraph/core/workflow/workflow_service.py`
   - 불필요한 초기화 지연
   - 첫 사용 시 로딩 (Lazy Loading)
   - 모델 로딩 최적화

2. **병렬 초기화**
   - 독립적인 컴포넌트 병렬 초기화
   - 비동기 초기화 활용
   - 초기화 순서 최적화

3. **캐싱 강화**
   - 설정 파일 캐싱
   - 모델 로딩 캐싱
   - 데이터베이스 연결 풀 사전 생성

4. **불필요한 초기화 제거**
   - 사용하지 않는 컴포넌트 초기화 제거
   - 선택적 초기화 옵션 추가
   - 초기화 로깅 최적화

**예상 소요 시간**: 6시간  
**담당자**: Backend Developer  
**우선순위**: 🔴 높음

---

### TASK-004: 검색 타입 불일치 문제 해결

**문제점**:
- `⚠️ [FALLBACK TYPE MISMATCH] 요청된 타입: {'statute_article'}, 반환된 타입: {'precedent_content'}` 다수 발생
- `Filtered chunk ...: source_type precedent_content not in ['statute_article']` 다수 발생

**목표**:
- 타입 불일치 발생률 5% 이하로 감소
- 검색 결과 타입 정확도 95% 이상

**작업 내용**:

1. **타입 필터링 로직 개선**
   - 파일: `lawfirm_langgraph/core/search/engines/semantic_search_engine_v2.py`
   - 초기 검색 시 타입 필터링 강화
   - 타입별 검색 전략 분리
   - 혼합 검색 시 타입 가중치 적용

2. **검색 전략 조정**
   - 타입별 최적 검색 파라미터 설정
   - 타입 불일치 시 재검색 로직 개선
   - 검색 결과 타입 검증 강화

3. **로깅 및 모니터링**
   - 타입 불일치 발생 시 상세 로깅
   - 타입별 검색 성공률 모니터링
   - 사용자 피드백 수집

**예상 소요 시간**: 6시간  
**담당자**: Search Engine Developer  
**우선순위**: 🔴 높음

---

### TASK-005: Metadata 캐시 효율성 개선

**문제점**:
- 초기 Metadata 캐시 히트율: 21.4% (3 hits, 11 misses)
- 점진적으로 개선되지만 초기 성능 저하

**목표**:
- 초기 캐시 히트율 50% 이상
- 전체 캐시 히트율 80% 이상

**작업 내용**:

1. **캐시 워밍업**
   - 파일: `lawfirm_langgraph/core/search/engines/semantic_search_engine_v2.py`
   - 자주 사용되는 메타데이터 사전 로딩
   - 초기화 시 인기 문서 메타데이터 캐싱
   - 배치 메타데이터 로딩

2. **캐시 크기 조정**
   - 캐시 크기 동적 조정
   - LRU 캐시 전략 최적화
   - 메모리 사용량 모니터링

3. **캐시 키 최적화**
   - 캐시 키 생성 로직 개선
   - 중복 캐시 키 제거
   - 캐시 무효화 전략 개선

**예상 소요 시간**: 4시간  
**담당자**: Backend Developer  
**우선순위**: 🟡 중간

---

### TASK-006: 검색 임계값 동적 조정

**문제점**:
- `🔄 Retrying with lower threshold: 0.500 → 0.30` 다수 발생
- Threshold 미달로 인한 강제 결과 반환 빈번

**목표**:
- Threshold 재시도 발생률 20% 이하
- 초기 검색 성공률 80% 이상

**작업 내용**:

1. **동적 임계값 전략**
   - 파일: `lawfirm_langgraph/core/search/engines/semantic_search_engine_v2.py`
   - 쿼리 유형별 최적 임계값 설정
   - 검색 결과 분포에 따른 동적 조정
   - 테이블별 최적 임계값 설정

2. **초기 임계값 조정**
   - 기본 임계값을 0.4로 낮춤
   - 검색 결과 품질 모니터링
   - 사용자 피드백 기반 조정

3. **재시도 로직 개선**
   - 재시도 횟수 제한
   - 재시도 간격 조정
   - 재시도 로깅 강화

**예상 소요 시간**: 4시간  
**담당자**: Search Engine Developer  
**우선순위**: 🟡 중간

---

### TASK-007: 프롬프트 검증 로직 수정

**문제점**:
- `❌ [PROMPT VALIDATION FAILED] No documents from structured_documents found in final prompt!`
- 실제로는 문서가 포함되어 있지만 검증 로직 오류

**목표**:
- 프롬프트 검증 오류 0회 달성
- 검증 로직 정확도 100%

**작업 내용**:

1. **검증 로직 수정**
   - 파일: `lawfirm_langgraph/core/workflow/legal_workflow_enhanced.py`
   - 문서 매칭 로직 개선
   - 검증 조건 명확화
   - 오류 원인 분석 및 수정

2. **로깅 개선**
   - 검증 실패 시 상세 로깅
   - 문서 포함 여부 명확히 확인
   - 디버깅 정보 추가

3. **테스트 강화**
   - 검증 로직 단위 테스트 추가
   - 다양한 시나리오 테스트
   - 통합 테스트 추가

**예상 소요 시간**: 3시간  
**담당자**: Backend Developer  
**우선순위**: 🟡 중간

---

### TASK-008: 검색 성능 편차 최소화

**문제점**:
- pgvector_search: 0.094s ~ 1.208s (최대 12.8배 차이)
- 검색 성능 편차가 큼

**목표**:
- 검색 성능 편차 3배 이하로 감소
- 평균 검색 시간 0.3초 이하

**작업 내용**:

1. **인덱스 최적화**
   - 파일: `lawfirm_langgraph/core/search/engines/semantic_search_engine_v2.py`
   - pgvector 인덱스 최적화
   - 인덱스 통계 정보 업데이트
   - 인덱스 재구성

2. **쿼리 최적화**
   - 쿼리 실행 계획 분석
   - 비효율적인 쿼리 개선
   - 쿼리 캐싱 강화

3. **성능 모니터링**
   - 검색 성능 메트릭 수집
   - 느린 쿼리 식별 및 개선
   - 성능 대시보드 구축

**예상 소요 시간**: 6시간  
**담당자**: Database/Backend Developer  
**우선순위**: 🟡 중간

---

### TASK-009: LangChain 통합 개선

**문제점**:
- `❌ LangChain not available. Cannot initialize agent.`
- 폴백 처리로 동작하지만 기능 제한

**목표**:
- LangChain 통합 완료 또는 폴백 로직 강화

**작업 내용**:

1. **LangChain 설치 확인**
   - LangChain 의존성 확인
   - 설치 가이드 문서화
   - 환경 설정 검증

2. **폴백 로직 강화**
   - 파일: `lawfirm_langgraph/core/workflow/legal_workflow_enhanced.py`
   - LangChain 없이도 동작하도록 개선
   - 폴백 기능 강화
   - 사용자 경험 개선

**예상 소요 시간**: 4시간  
**담당자**: Backend Developer  
**우선순위**: 🟢 낮음

---

### TASK-010: 동의어 데이터베이스 설정

**문제점**:
- `Synonym database not available or error` (3회 발생)
- 동의어 검색 기능 제한

**목표**:
- 동의어 데이터베이스 정상 작동
- 동의어 검색 기능 활성화

**작업 내용**:

1. **동의어 데이터베이스 설정**
   - PostgreSQL URL 설정 확인
   - 동의어 데이터베이스 생성
   - 초기 데이터 로딩

2. **폴백 처리 개선**
   - 파일: `lawfirm_langgraph/core/processing/integration/term_integration_system.py`
   - 동의어 DB 없이도 동작하도록 개선
   - 폴백 동의어 사전 활용

**예상 소요 시간**: 3시간  
**담당자**: Backend Developer  
**우선순위**: 🟢 낮음

---

### TASK-011: MLflow 백엔드 전환

**문제점**:
- `Filesystem tracking backend (e.g., './mlruns') is deprecated`
- 파일시스템 백엔드 사용 중

**목표**:
- 데이터베이스 백엔드로 전환 완료

**작업 내용**:

1. **MLflow 백엔드 전환**
   - 파일: `lawfirm_langgraph/core/search/engines/semantic_search_engine_v2.py`
   - SQLite 백엔드 설정 (`sqlite:///mlflow.db`)
   - 기존 데이터 마이그레이션
   - 설정 파일 업데이트

**예상 소요 시간**: 2시간  
**담당자**: Backend Developer  
**우선순위**: 🟢 낮음

---

### TASK-012: 모델 캐시 로드 오류 수정

**문제점**:
- `⚠️ [MODEL CACHE] Failed to load "woong0322/ko-legal-sbert-finetuned": Repo id must use alphanumeric chars`

**목표**:
- 모델 캐시 로드 오류 해결

**작업 내용**:

1. **모델명 검증 로직 수정**
   - 파일: `lawfirm_langgraph/core/shared/utils/model_cache_manager.py`
   - 모델명 검증 로직 개선
   - 허용되는 문자 확인
   - 오류 처리 개선

**예상 소요 시간**: 2시간  
**담당자**: Backend Developer  
**우선순위**: 🟢 낮음

---

## 예상 효과

### 성능 개선

- **초기화 시간**: 31초 → 15초 (52% 단축)
- **전체 처리 시간**: 52초 → 26초 (50% 단축)
- **검색 성능**: 평균 0.3초 이하 달성
- **캐시 히트율**: 80% 이상 달성

### 품질 개선

- **검색 성공률**: 90% 이상 달성
- **타입 불일치 발생률**: 5% 이하로 감소
- **FALLBACK 발생률**: 10% 이하로 감소
- **커서 관련 경고**: 0회 달성

### 안정성 개선

- **데이터베이스 연결 안정성**: 향상
- **오류 발생률**: 감소
- **시스템 안정성**: 향상

---

## 검증 방법

### 1. 성능 테스트

```bash
# 테스트 실행
python lawfirm_langgraph/tests/runners/run_query_test.py "계약 해지 사유에 대해 알려주세요"

# 성능 메트릭 확인
- 초기화 시간
- 전체 처리 시간
- 검색 성능
- 캐시 히트율
```

### 2. 로그 분석

```bash
# 로그 파일 분석
- 경고/오류 발생 횟수 확인
- 성능 메트릭 추출
- 문제 패턴 분석
```

### 3. 통합 테스트

```bash
# 다양한 쿼리로 테스트
- 법령 검색 쿼리
- 판례 검색 쿼리
- 복합 검색 쿼리
```

### 4. 모니터링

- 실시간 성능 모니터링
- 오류 발생률 모니터링
- 사용자 피드백 수집

---

## 작업 일정

### Phase 1: 긴급 개선 (1주)
- TASK-001: 데이터베이스 커서 생명주기 관리 개선
- TASK-002: 검색 결과 없음 문제 해결
- TASK-003: 초기화 시간 최적화
- TASK-004: 검색 타입 불일치 문제 해결

### Phase 2: 단기 개선 (2주)
- TASK-005: Metadata 캐시 효율성 개선
- TASK-006: 검색 임계값 동적 조정
- TASK-007: 프롬프트 검증 로직 수정
- TASK-008: 검색 성능 편차 최소화

### Phase 3: 중장기 개선 (3주)
- TASK-009: LangChain 통합 개선
- TASK-010: 동의어 데이터베이스 설정
- TASK-011: MLflow 백엔드 전환
- TASK-012: 모델 캐시 로드 오류 수정

---

## 참고 자료

- 분석 로그: `logs/langgraph/test_langgraph_query_20251129_160731.log`
- 관련 문서: `docs/08_features/quality_improvement_implementation_plan.md`
- 코드베이스: `lawfirm_langgraph/core/search/engines/semantic_search_engine_v2.py`

---

**작성일**: 2025-11-29  
**최종 수정일**: 2025-11-29  
**버전**: 1.0

