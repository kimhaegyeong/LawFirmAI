# legal_workflow_enhanced.py 리팩토링 계획

## 현재 상태 분석

### 파일 통계
- **총 라인 수**: 8,278줄
- **메서드 수**: 약 208개
- **워크플로우 노드**: 17개 (@observe 데코레이터)
- **래퍼 메서드**: 약 117개

### 이미 완료된 리팩토링
- ✅ Phase 13: 검색 결과 처리 로직 분리 (SearchResultProcessor)
- ✅ Phase 14: 문서 처리 로직 분리 (WorkflowDocumentProcessor)
- ✅ Phase 15: 프롬프트 빌딩 로직 통합 (WorkflowPromptBuilder)
- ✅ Phase 16: 검증 로직 통합 (WorkflowValidator)
- ✅ Phase 17: 통계 및 유틸리티 메서드 분리 (WorkflowUtils)

### 남은 리팩토링 영역

## 리팩토링 계획 (우선순위 순)

### Phase 18: 래퍼 메서드 제거 및 직접 호출
**목표**: 불필요한 래퍼 메서드 제거, 직접 호출로 변경

**영향 범위**:
- 약 117개의 래퍼 메서드
- 예상 감소: ~500-800줄

**작업 내용**:
1. 래퍼 메서드 식별 및 분류
   - ContextBuilder 래퍼 (~10개)
   - AnswerGenerator 래퍼 (~15개)
   - AnswerFormatterHandler 래퍼 (~10개)
   - SearchHandler 래퍼 (~20개)
   - WorkflowUtils 래퍼 (~15개)
   - ClassificationHandler 래퍼 (~10개)
   - WorkflowRoutes 래퍼 (~5개)
   - 기타 래퍼 (~32개)

2. 직접 호출로 변경
   - `self._method()` → `self.component.method()` 직접 호출
   - 호출 위치 확인 및 수정

3. 테스트 및 검증
   - 각 변경 후 테스트 실행
   - 기능 동작 확인

**예상 소요 시간**: 2-3일

---

### Phase 19: 문서 분석 로직 분리
**목표**: `analyze_document` 노드 및 관련 로직을 별도 클래스로 분리

**영향 범위**:
- `analyze_document` 노드 (~200줄)
- `_analyze_legal_document_with_chain` (~500줄)
- `_detect_document_type` (~50줄)
- `_analyze_legal_document` (~100줄)
- `_extract_contract_clauses` (래퍼)
- `_identify_contract_issues` (~100줄)
- `_extract_complaint_elements` (래퍼)
- `_identify_complaint_issues` (~50줄)
- `_generate_document_summary` (~50줄)
- `_generate_document_summary_fallback` (~50줄)
- 파서 메서드들 (~200줄)

**새로운 클래스**:
```
core/workflow/processors/document_analysis_processor.py
- DocumentAnalysisProcessor
  - analyze_document()
  - analyze_legal_document_with_chain()
  - detect_document_type()
  - analyze_legal_document()
  - identify_contract_issues()
  - identify_complaint_issues()
  - generate_document_summary()
  - parse_*_response() 메서드들
```

**예상 감소**: ~1,300줄

**예상 소요 시간**: 3-4일

---

### Phase 20: 검색 실행 로직 분리
**목표**: 검색 실행 관련 내부 메서드를 별도 클래스로 분리

**영향 범위**:
- `execute_searches_parallel` 노드 (~300줄)
- `_execute_semantic_search_internal` (~150줄)
- `_execute_keyword_search_internal` (~100줄)
- `_get_search_params_batch` (~100줄)
- `_fallback_sequential_search` (~50줄)

**새로운 클래스**:
```
core/search/executors/search_executor.py
- SearchExecutor
  - execute_searches_parallel()
  - execute_semantic_search()
  - execute_keyword_search()
  - get_search_params()
  - fallback_sequential_search()
```

**예상 감소**: ~700줄

**예상 소요 시간**: 2-3일

---

### Phase 21: 컨텍스트 확장 로직 분리
**목표**: 컨텍스트 확장 관련 로직을 별도 클래스로 분리

**영향 범위**:
- `_adaptive_context_expansion` (~100줄)
- `_should_expand_context` (~60줄)
- `_build_expanded_query` (~50줄)
- `_validate_context_quality` (일부)
- `_validate_context_quality_original` (~60줄)

**새로운 클래스**:
```
core/workflow/processors/context_expansion_processor.py
- ContextExpansionProcessor
  - expand_context()
  - should_expand()
  - build_expanded_query()
  - validate_context_quality()
```

**예상 감소**: ~300줄

**예상 소요 시간**: 1-2일

---

### Phase 22: 답변 검증 로직 분리
**목표**: 답변 검증 관련 로직을 별도 클래스로 분리

**영향 범위**:
- `_validate_answer_quality_internal` (~300줄)
- `_validate_with_llm` (~200줄)
- `_detect_format_errors` (~40줄)
- `_detect_specific_case_copy` (~100줄)
- `_validate_answer_uses_context` (래퍼)

**새로운 클래스**:
```
core/workflow/validators/answer_quality_validator.py
- AnswerQualityValidator
  - validate_answer_quality()
  - validate_with_llm()
  - detect_format_errors()
  - detect_specific_case_copy()
  - validate_answer_uses_context()
```

**예상 감소**: ~650줄

**예상 소요 시간**: 2-3일

---

### Phase 23: 그래프 빌드 로직 분리
**목표**: 워크플로우 그래프 구축 로직을 별도 클래스로 분리

**영향 범위**:
- `_build_graph` (~200줄)
- 그래프 엣지 정의 로직
- 라우팅 조건 정의

**새로운 클래스**:
```
core/workflow/builders/workflow_graph_builder.py
- WorkflowGraphBuilder
  - build_graph()
  - add_nodes()
  - add_edges()
  - setup_routing()
```

**예상 감소**: ~250줄

**예상 소요 시간**: 1-2일

---

### Phase 24: LLM 초기화 로직 분리
**목표**: LLM 초기화 관련 로직을 별도 클래스로 분리

**영향 범위**:
- `_initialize_llm` (~20줄)
- `_initialize_gemini` (~50줄)
- `_initialize_ollama` (~30줄)
- `_create_mock_llm` (~20줄)
- `_initialize_llm_fast` (~50줄)
- `_initialize_validator_llm` (~50줄)

**새로운 클래스**:
```
core/workflow/initializers/llm_initializer.py
- LLMInitializer
  - initialize_llm()
  - initialize_gemini()
  - initialize_ollama()
  - initialize_llm_fast()
  - initialize_validator_llm()
  - create_mock_llm()
```

**예상 감소**: ~220줄

**예상 소요 시간**: 1일

---

### Phase 25: 통계 관리 로직 분리
**목표**: 통계 관리 관련 로직을 별도 클래스로 분리

**영향 범위**:
- `update_statistics` (~50줄)
- `get_statistics` (~20줄)
- 통계 초기화 로직

**새로운 클래스**:
```
core/workflow/utils/workflow_statistics.py
- WorkflowStatistics
  - update_statistics()
  - get_statistics()
  - reset_statistics()
```

**예상 감소**: ~100줄

**예상 소요 시간**: 0.5일

---

### Phase 26: 긴급도 및 멀티턴 처리 로직 분리
**목표**: 긴급도 평가 및 멀티턴 처리 로직을 별도 클래스로 분리

**영향 범위**:
- `assess_urgency` 노드 (~50줄)
- `_assess_urgency_internal` (~30줄)
- `_assess_urgency_fallback` (~20줄)
- `resolve_multi_turn` 노드 (~70줄)
- `_resolve_multi_turn_internal` (~30줄)

**새로운 클래스**:
```
core/workflow/processors/conversation_processor.py
- ConversationProcessor
  - assess_urgency()
  - resolve_multi_turn()
  - assess_urgency_fallback()
```

**예상 감소**: ~200줄

**예상 소요 시간**: 1일

---

### Phase 27: 답변 생성 로직 정리
**목표**: 답변 생성 관련 메서드 정리 및 최적화

**영향 범위**:
- `generate_answer_enhanced` 노드 (~900줄) - 일부 로직 분리 가능
- `continue_answer_generation` 노드 (~100줄)
- `_prepare_answer_generation` (~50줄)
- `_restore_query_type` (~30줄)
- `_restore_retrieved_docs` (~30줄)
- `_build_context_dict` (~200줄)

**작업 내용**:
- 중복 로직 제거
- 메서드 분리 및 재구성
- 에러 처리 개선

**예상 감소**: ~300줄 (중복 제거)

**예상 소요 시간**: 2-3일

---

### Phase 28: 검색 결과 처리 로직 정리
**목표**: 검색 결과 처리 관련 메서드 정리 및 최적화

**영향 범위**:
- `process_search_results_combined` 노드 (~700줄)
- `filter_and_validate_results` 노드 (~200줄)
- `merge_and_rerank_with_keyword_weights` 노드 (~200줄)
- `evaluate_search_quality` 노드 (~60줄)
- `conditional_retry_search` 노드 (~100줄)
- `update_search_metadata` 노드 (~50줄)

**작업 내용**:
- 중복 로직 제거
- 메서드 분리 및 재구성
- 성능 최적화

**예상 감소**: ~400줄 (중복 제거)

**예상 소요 시간**: 2-3일

---

## 전체 리팩토링 요약

### 예상 감소량
- Phase 18: ~600줄
- Phase 19: ~1,300줄
- Phase 20: ~700줄
- Phase 21: ~300줄
- Phase 22: ~650줄
- Phase 23: ~250줄
- Phase 24: ~220줄
- Phase 25: ~100줄
- Phase 26: ~200줄
- Phase 27: ~300줄
- Phase 28: ~400줄

**총 예상 감소**: ~5,020줄
**최종 예상 크기**: ~3,258줄 (현재 8,278줄의 약 39%)

### 예상 소요 시간
**총 예상 시간**: 약 18-25일

### 우선순위별 진행 계획

#### 높은 우선순위 (즉시 진행)
1. **Phase 18**: 래퍼 메서드 제거 (가장 많은 줄 수 감소)
2. **Phase 19**: 문서 분석 로직 분리 (복잡도 높음)
3. **Phase 20**: 검색 실행 로직 분리 (복잡도 높음)

#### 중간 우선순위 (단기 개선)
4. **Phase 22**: 답변 검증 로직 분리
5. **Phase 21**: 컨텍스트 확장 로직 분리
6. **Phase 27**: 답변 생성 로직 정리

#### 낮은 우선순위 (장기 개선)
7. **Phase 23**: 그래프 빌드 로직 분리
8. **Phase 24**: LLM 초기화 로직 분리
9. **Phase 25**: 통계 관리 로직 분리
10. **Phase 26**: 긴급도 및 멀티턴 처리 로직 분리
11. **Phase 28**: 검색 결과 처리 로직 정리

---

## 리팩토링 원칙

### 1. 기존 코드 보존
- 원본 코드의 구조와 스타일 최대한 유지
- 기능 변경 없이 구조만 개선

### 2. 점진적 리팩토링
- 한 번에 하나의 Phase씩 진행
- 각 Phase 완료 후 테스트 및 검증

### 3. 테스트 우선
- 리팩토링 전 기존 테스트 실행
- 리팩토링 후 동일한 테스트로 검증

### 4. 문서화
- 각 Phase별 변경 사항 문서화
- 새로운 클래스 및 메서드 문서화

---

## 리팩토링 체크리스트

각 Phase 완료 시 확인:
- [ ] 기능 동작 확인 (테스트 통과)
- [ ] 성능 영향 없음 확인
- [ ] 코드 가독성 향상
- [ ] 중복 코드 제거
- [ ] 문서 업데이트
- [ ] Linter 오류 없음

---

## 참고 사항

### 주의사항
- 래퍼 메서드 제거 시 호출 위치를 정확히 파악해야 함
- 문서 분석 로직은 복잡한 체인 로직이 포함되어 있어 신중하게 진행
- 검색 실행 로직은 성능에 직접적인 영향을 미치므로 최적화 고려

### 개선 효과
- 코드 가독성 향상
- 유지보수성 향상
- 테스트 용이성 향상
- 재사용성 향상
- 성능 최적화 기회 확대

---

## 진행 상황 추적

### 완료된 Phase
- [x] Phase 13: 검색 결과 처리 로직 분리
- [x] Phase 14: 문서 처리 로직 분리
- [x] Phase 15: 프롬프트 빌딩 로직 통합
- [x] Phase 16: 검증 로직 통합
- [x] Phase 17: 통계 및 유틸리티 메서드 분리

### 진행 중인 Phase
- [ ] Phase 18: 래퍼 메서드 제거 및 직접 호출

### 대기 중인 Phase
- [ ] Phase 19: 문서 분석 로직 분리
- [ ] Phase 20: 검색 실행 로직 분리
- [ ] Phase 21: 컨텍스트 확장 로직 분리
- [ ] Phase 22: 답변 검증 로직 분리
- [ ] Phase 23: 그래프 빌드 로직 분리
- [ ] Phase 24: LLM 초기화 로직 분리
- [ ] Phase 25: 통계 관리 로직 분리
- [ ] Phase 26: 긴급도 및 멀티턴 처리 로직 분리
- [ ] Phase 27: 답변 생성 로직 정리
- [ ] Phase 28: 검색 결과 처리 로직 정리

