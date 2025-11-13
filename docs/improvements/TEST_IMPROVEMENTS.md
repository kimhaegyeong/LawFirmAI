# 테스트 결과 기반 개선 사항

## 테스트 실행 정보
- **테스트 질의**: "전세금 반환 보증에 대해 설명해주세요"
- **실행 시간**: 2025-11-13
- **총 실행 시간**: 19.57초

## 발견된 개선 사항 (번호 순)

### 1. retrieved_docs 복구 로직 개선 필요
**문제점**:
- `⚠️ [GENERATE_ANSWER] No retrieved_docs available at start. Attempting to recover...`
- `generate_answer_enhanced` 노드 시작 시점에 `retrieved_docs`가 없어 복구 시도가 발생

**원인 분석**:
- `process_search_results_combined` 노드에서 `retrieved_docs`가 생성되었지만, `generate_answer_enhanced` 노드로 전달되지 않음
- State 전달 과정에서 `retrieved_docs`가 손실됨

**개선 방안**:
- `_recover_retrieved_docs_at_start` 메서드의 복구 로직 강화
- `process_search_results_combined` 노드에서 `retrieved_docs`를 state의 여러 위치에 저장 (top-level, `common`, `metadata`)
- Global cache에서 `retrieved_docs` 복구 로직 추가

### 2. 검색 품질 점수 0.00 문제 해결
**문제점**:
- `⚠️ [SEARCH QUALITY] Very low quality detected: overall_score=0.00 (relevance=0.00, coverage=0.00, sufficiency=0.00)`
- `⚠️ [SEARCH QUALITY] CRITICAL: Search quality is 0.00. This may cause answer generation failure.`

**원인 분석**:
- 검색 품질 평가 로직에서 relevance, coverage, sufficiency가 모두 0.00으로 계산됨
- 검색 결과는 있지만 (semantic_results=15, keyword_results=9) 품질 평가가 실패

**개선 방안**:
- `evaluate_search_quality` 메서드의 품질 평가 로직 검증
- 검색 결과가 있는 경우 최소 품질 점수 보장
- 품질 평가 실패 시 fallback 로직 추가

### 3. 컨텍스트 사용률(Coverage) 개선
**문제점**:
- `⚠️ [ANSWER QUALITY] Moderate coverage: 0.42 (expected >= 0.6). Keyword: 0.20, Citation: 0.40`
- `⚠️ [VALIDATION] Context usage low (coverage: 0.42), but regeneration is disabled.`

**원인 분석**:
- 키워드 커버리지가 0.20으로 낮음
- 인용 커버리지가 0.40으로 목표(0.5) 미만
- 전체 커버리지가 0.42로 목표(0.6) 미만

**개선 방안**:
- 프롬프트 템플릿에서 컨텍스트 활용 강조
- 검색 결과의 키워드 매칭 개선
- 인용 추출 로직 강화

### 4. 인용 수 부족 문제 해결
**문제점**:
- `⚠️ [VALIDATION] Low citation count: 1 (expected >= 2) for 6 documents.`

**원인 분석**:
- 6개의 문서가 검색되었지만 인용이 1개만 추출됨
- 법률 참조 추출 로직이 제대로 작동하지 않음

**개선 방안**:
- `legal_references` 추출 로직 강화
- 문서 내용에서 법률 참조 패턴 추출 개선
- `sources_detail`에서 법률 참조 추출 로직 추가

### 5. 판례/결정례 문서 복원 실패 개선
**문제점**:
- `🔀 [DIVERSITY] ⚠️ Failed to restore precedent/decision documents from weighted_docs (precedent_candidates: 0, decision_candidates: 0)`

**원인 분석**:
- `process_search_results_combined` 노드에서 판례/결정례 문서 복원 시도가 실패
- `weighted_docs`에 판례/결정례 후보가 없음

**개선 방안**:
- 검색 단계에서 판례/결정례 문서 검색 강화
- `weighted_docs`에 판례/결정례 문서가 포함되도록 가중치 조정
- 복원 실패 시 대체 검색 로직 추가

### 6. query_type 복원 실패 개선
**문제점**:
- `⚠️ [QUESTION TYPE] query_type not found in state or global cache, using default: general_question`
- `[QUERY_TYPE] ⚠️ query_type not found, using default: general_question`

**원인 분석**:
- `classify_query_and_complexity` 노드에서 `query_type`이 생성되었지만 전달되지 않음
- State 전달 과정에서 `query_type`이 손실됨

**개선 방안**:
- `query_type`을 state의 여러 위치에 저장 (top-level, `common`, `metadata`)
- Global cache에서 `query_type` 복구 로직 추가
- `_restore_query_type_enhanced` 메서드 강화

### 7. 답변 시작 검증 실패 개선
**문제점**:
- `⚠️ [IMMEDIATE VALIDATION] Answer start validation failed: has_specific_case_in_start: None, has_general_principle_in_start: None`
- `🔄 [AUTO RETRY] Regeneration needed: general_principle_not_in_start. Retrying answer generation (retry count: 1/4)`

**원인 분석**:
- 답변 시작 부분 검증 로직이 제대로 작동하지 않음
- 검증 실패로 인해 자동 재시도가 발생하여 성능 저하

**개선 방안**:
- 답변 시작 검증 로직 개선
- 검증 실패 시 재시도 대신 프롬프트 조정
- 검증 기준 완화 또는 제거 검토

### 8. 성능 최적화 (느린 노드 개선)
**문제점**:
- `⚠️ [PERFORMANCE] 느린 노드 감지: generate_answer_stream가 13.13초 소요되었습니다. (임계값: 5.0초)`
- `Generate Answer Stream: 4.92초 (25.1%)`
- `Prepare Search Query: 4.91초 (25.1%)`
- `Generate Answer Final: 4.24초 (21.7%)`

**원인 분석**:
- `generate_answer_stream` 노드가 13.13초 소요 (임계값 5.0초 초과)
- `prepare_search_query` 노드가 4.91초 소요
- `generate_answer_final` 노드가 4.24초 소요

**개선 방안**:
- LLM 호출 최적화 (배치 처리, 캐싱 강화)
- 불필요한 재시도 로직 제거
- 검색 쿼리 준비 시간 단축

### 9. Sources 변환률 개선
**문제점**:
- 테스트 결과에서 `sources_conversion_rate: 10.0%` (10개 문서 중 1개만 변환)
- `sources_detail_count: 0` (sources_detail이 생성되지 않음)

**원인 분석**:
- `retrieved_docs`에서 `sources`로의 변환률이 매우 낮음
- `sources_detail` 생성 로직이 실행되지 않음

**개선 방안**:
- `prepare_final_response_part`에서 `sources` 생성 로직 강화
- `sources_detail` 생성 fallback 로직 추가
- 모든 `retrieved_docs`가 `sources`로 변환되도록 보장

### 10. Legal References 생성률 개선
**문제점**:
- 테스트 결과에서 `legal_references_generation_rate: 0.0%`
- `legal_references_count: 0`

**원인 분석**:
- 법률 참조 추출 로직이 실행되지 않음
- 검색된 문서에서 법률 참조를 찾지 못함

**개선 방안**:
- `legal_references` 추출 로직 강화
- 문서 내용에서 법률 참조 패턴 추출 개선
- `statute_article` 타입 문서에서 법률 참조 추출 보장

### 11. Related Questions 생성 개선
**문제점**:
- 테스트 결과에서 `related_questions_count: 0`

**원인 분석**:
- `related_questions` 생성 로직이 실행되지 않음
- `phase_info`에 `suggested_questions`가 없음

**개선 방안**:
- `related_questions` 생성 로직 강화
- `phase_info`에서 `suggested_questions` 추출 로직 추가
- Fallback `related_questions` 생성 로직 추가

### 12. 답변 길이 개선
**문제점**:
- 테스트 결과에서 `answer_length: 3` (매우 짧은 답변)

**원인 분석**:
- 답변이 3자로 생성되어 거의 비어있음
- 검색 품질이 0.00이어서 답변 생성이 실패했을 가능성

**개선 방안**:
- 검색 품질이 낮을 때 fallback 답변 생성 로직 추가
- 최소 답변 길이 보장 로직 추가
- 답변 복구 로직 강화

## 우선순위별 개선 사항

### HIGH 우선순위
1. **검색 품질 점수 0.00 문제 해결** (#2)
2. **답변 길이 개선** (#12)
3. **Sources 변환률 개선** (#9)

### MEDIUM 우선순위
4. **retrieved_docs 복구 로직 개선** (#1)
5. **컨텍스트 사용률(Coverage) 개선** (#3)
6. **인용 수 부족 문제 해결** (#4)
7. **Legal References 생성률 개선** (#10)

### LOW 우선순위
8. **판례/결정례 문서 복원 실패 개선** (#5)
9. **query_type 복원 실패 개선** (#6)
10. **답변 시작 검증 실패 개선** (#7)
11. **성능 최적화** (#8)
12. **Related Questions 생성 개선** (#11)

