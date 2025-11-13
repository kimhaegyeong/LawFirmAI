# 리팩토링된 메서드 문서화

## 개요

이 문서는 `legal_workflow_enhanced.py`에서 리팩토링된 메서드들의 역할과 책임을 명확히 설명합니다.

## 리팩토링된 메인 메서드

### 1. `generate_answer_enhanced`

**역할**: 개선된 답변 생성 - UnifiedPromptManager 활용

**리팩토링 전**: 약 1,100줄  
**리팩토링 후**: 약 103줄 (메인 메서드)  
**분리된 헬퍼 메서드**: 23개

**주요 흐름**:
1. `_recover_retrieved_docs_at_start`: retrieved_docs 복원
2. `_validate_and_generate_prompt_context`: 프롬프트 컨텍스트 검증 및 생성
3. `_build_and_validate_context_dict`: 컨텍스트 딕셔너리 구축 및 검증
4. `_validate_context_quality_and_expand`: 컨텍스트 품질 검증 및 확장
5. `_inject_search_results_into_context`: 검색 결과를 컨텍스트에 주입
6. `_generate_and_validate_prompt`: 프롬프트 생성 및 검증
7. `_generate_answer_with_cache`: 캐시를 활용한 답변 생성
8. `_validate_and_enhance_answer`: 답변 검증 및 보강

### 2. `process_search_results_combined`

**역할**: 검색 결과 처리 통합 노드 (6개 노드를 1개로 병합)

**리팩토링 전**: 약 600줄  
**리팩토링 후**: 약 111줄 (메인 메서드)  
**분리된 헬퍼 메서드**: 7개

**주요 흐름**:
1. `_prepare_search_inputs`: 검색 결과 입력 데이터 준비
2. `_perform_conditional_retry_search`: 조건부 재검색 수행
3. `_merge_and_rerank_results`: 병합 및 재순위
4. `_apply_keyword_weights_and_rerank`: 키워드 가중치 적용 및 재정렬
5. `_filter_and_validate_documents`: 필터링 및 검증
6. `_ensure_diversity_and_limit`: 다양성 보장 및 최종 문서 제한
7. `_save_final_results_to_state`: 최종 결과를 State에 저장

### 3. `generate_answer_final`

**역할**: 최종 검증 및 포맷팅 노드 - 검증과 포맷팅만 수행

**리팩토링 전**: 약 150줄  
**리팩토링 후**: 약 25줄 (메인 메서드)  
**분리된 헬퍼 메서드**: 5개

**주요 흐름**:
1. `_restore_state_data_for_final`: State 데이터 복원
2. `_validate_and_handle_regeneration`: 품질 검증 및 재생성 처리
3. `_handle_format_errors`: 형식 오류 처리
4. `_format_and_finalize`: 포맷팅 및 최종 준비
5. `_handle_final_node_error`: 오류 처리

## 헬퍼 메서드 상세 설명

### `generate_answer_enhanced` 헬퍼 메서드들

#### `_recover_retrieved_docs_at_start`
- **역할**: `generate_answer_enhanced` 시작 시 `retrieved_docs` 복원
- **입력**: `state: LegalWorkflowState`
- **출력**: `None` (state 수정)
- **복원 경로**: state → search 그룹 → global cache

#### `_validate_and_generate_prompt_context`
- **역할**: 프롬프트 컨텍스트 검증 및 생성
- **입력**: `state`, `retrieved_docs`, `query`, `extracted_keywords`, `query_type`
- **출력**: `Dict[str, Any]` (prompt_optimized_context)
- **기능**: `prompt_optimized_context`가 없거나 유효하지 않으면 자동 생성

#### `_build_and_validate_context_dict`
- **역할**: 컨텍스트 딕셔너리 구축 및 검증
- **입력**: `state`, `query_type`, `retrieved_docs`, `prompt_optimized_context`
- **출력**: `Dict[str, Any]` (context_dict)
- **기능**: 타입 검사 및 변환, 내용 유효성 검증

#### `_validate_context_quality_and_expand`
- **역할**: 컨텍스트 품질 검증 및 확장
- **입력**: `state`, `context_dict`, `query`, `query_type`, `extracted_keywords`, `retrieved_docs`
- **출력**: `Tuple[Dict[str, Any], Dict[str, Any], List[Dict[str, Any]]]` (context_dict, validation_results, retrieved_docs)
- **기능**: 검색 품질 모니터링, 컨텍스트 확장, 재검증

#### `_monitor_search_quality`
- **역할**: 검색 품질 모니터링 및 `retrieved_docs` 복구
- **입력**: `validation_results`, `overall_score`, `state`, `retrieved_docs`
- **출력**: `List[Dict[str, Any]]` (복구된 retrieved_docs)
- **기능**: 품질 점수가 낮을 경우 `retrieved_docs` 복구 시도

#### `_generate_and_validate_prompt`
- **역할**: 최적화된 프롬프트 생성 및 검증
- **입력**: `state`, `context_dict`, `query`, `question_type`, `domain`, `model_type`, `base_prompt_type`, `retrieved_docs`
- **출력**: `Tuple[str, Optional[Path], int, int]` (optimized_prompt, prompt_file, prompt_length, structured_docs_count)
- **기능**: UnifiedPromptManager를 사용한 프롬프트 생성, 파일 저장, 길이 검증

#### `_generate_answer_with_cache`
- **역할**: 캐시를 활용한 답변 생성
- **입력**: `state`, `optimized_prompt`, `query`, `query_type`, `context_dict`, `retrieved_docs`, `quality_feedback`, `is_retry`
- **출력**: `str` (normalized_response)
- **기능**: 캐시 확인, 답변 생성, 품질 검증, 캐시 저장

#### `_validate_and_enhance_answer`
- **역할**: 답변 검증 및 보강
- **입력**: `state`, `normalized_response`, `query`, `context_dict`, `retrieved_docs`, `prompt_length`, `prompt_file`, `structured_docs_count`
- **출력**: `str` (검증 및 보강된 답변)
- **기능**: 답변 시작 부분 검증, 인용 검증 및 보강, 검색 결과 사용 추적, 품질 모니터링

### `process_search_results_combined` 헬퍼 메서드들

#### `_prepare_search_inputs`
- **역할**: 검색 결과 입력 데이터 준비
- **입력**: `state: LegalWorkflowState`
- **출력**: `Dict[str, Any]` (검색 입력 데이터)
- **반환 필드**: `semantic_results`, `keyword_results`, `semantic_count`, `keyword_count`, `query`, `query_type_str`, `search_params`, `extracted_keywords`

#### `_perform_conditional_retry_search`
- **역할**: 조건부 재검색 수행
- **입력**: `state`, `semantic_results`, `keyword_results`, `semantic_count`, `keyword_count`, `quality_evaluation`, `query`, `query_type_str`, `search_params`, `extracted_keywords`
- **출력**: `Tuple[List[Dict[str, Any]], List[Dict[str, Any]], int, int]` (업데이트된 검색 결과 및 카운트)
- **조건**: `needs_retry`가 True이고 `overall_quality < 0.6`이고 총 결과 수 < 10

#### `_merge_and_rerank_results`
- **역할**: 병합 및 재순위
- **입력**: `state`, `semantic_results`, `keyword_results`, `query`
- **출력**: `List[Dict[str, Any]]` (병합된 문서)
- **기능**: `search_handler.merge_and_rerank_search_results` 또는 `_merge_search_results_internal` 사용

#### `_apply_keyword_weights_and_rerank`
- **역할**: 키워드 가중치 적용 및 재정렬
- **입력**: `state`, `merged_docs`, `query`, `query_type_str`, `extracted_keywords`, `search_params`, `overall_quality`
- **출력**: `List[Dict[str, Any]]` (가중치가 적용된 문서)
- **기능**: 키워드 가중치 계산, 가중치 적용, 다단계 재정렬 또는 citation boost

#### `_filter_and_validate_documents`
- **역할**: 필터링 및 검증
- **입력**: `state`, `weighted_docs`, `query`, `extracted_keywords`, `merged_docs`
- **출력**: `List[Dict[str, Any]]` (필터링된 문서)
- **필터링 기준**:
  - 최소 content 길이 (판례/결정례/법령: 3자, 기타: 5자)
  - 관련성 키워드 포함 여부
  - 최소 점수 임계값 (판례/결정례/법령: 0.03, 기타: 0.05)

#### `_ensure_diversity_and_limit`
- **역할**: 다양성 보장 및 최종 문서 제한
- **입력**: `state`, `filtered_docs`, `weighted_docs`, `merged_docs`, `query`, `query_type_str`, `semantic_results`
- **출력**: `List[Dict[str, Any]]` (최종 문서)
- **기능**: 
  - 판례/결정례 복원 (필터링 후 누락된 경우)
  - 다양성 보장 (`_ensure_diverse_source_types`)
  - 최대 문서 수 제한
  - 폴백 전략 (semantic_results 변환)

#### `_save_final_results_to_state`
- **역할**: 최종 결과를 State에 저장
- **입력**: `state`, `final_docs`, `merged_docs`, `filtered_docs`, `overall_quality`, `semantic_count`, `keyword_count`, `needs_retry`, `start_time`
- **출력**: `None` (state 수정)
- **저장 위치**: 최상위, `search` 그룹, `common` 그룹, 전역 캐시

### `generate_answer_final` 헬퍼 메서드들

#### `_restore_state_data_for_final`
- **역할**: 최종 노드를 위한 State 데이터 복원
- **입력**: `state: LegalWorkflowState`
- **출력**: `None` (state 수정)
- **복원 항목**: `retrieved_docs`, `structured_documents`, `query_type`
- **복원 경로**: state → global cache

#### `_validate_and_handle_regeneration`
- **역할**: 품질 검증 및 재생성 처리
- **입력**: `state: LegalWorkflowState`
- **출력**: `bool` (quality_check_passed)
- **기능**: 
  - 품질 검증 수행
  - 재생성 필요 여부 확인
  - 재생성 가능 시 `generate_answer_enhanced` 재호출

#### `_handle_format_errors`
- **역할**: 형식 오류 처리
- **입력**: `state: LegalWorkflowState`, `quality_check_passed: bool`
- **출력**: `bool` (업데이트된 quality_check_passed)
- **기능**: 
  - 형식 오류 감지
  - 답변 정규화 시도
  - 정규화 후에도 오류가 있으면 재생성

#### `_format_and_finalize`
- **역할**: 포맷팅 및 최종 준비
- **입력**: `state: LegalWorkflowState`, `overall_start_time: float`
- **출력**: `None` (state 수정)
- **기능**: 
  - `_format_and_finalize_answer` 호출
  - 오류 발생 시 기본 포맷 사용
  - 처리 시간 및 신뢰도 로깅

#### `_handle_final_node_error`
- **역할**: 최종 노드 오류 처리
- **입력**: `state: LegalWorkflowState`, `error: Exception`
- **출력**: `None` (state 수정)
- **기능**: 
  - 'control' 오류 특별 처리
  - 기존 답변 보존 또는 최소 답변 생성
  - 품질 점수 및 검증 플래그 설정

## 성능 특성

### 실행 시간 (평균)
- `generate_answer_final`: ~50ms
- `process_search_results_combined`: ~690ms
- 헬퍼 메서드들: <1ms

### 메모리 사용량
- `generate_answer_final`: ~1.04 MB (Peak: 1.18 MB)
- `process_search_results_combined`: ~0.12 MB (Peak: 0.44 MB)

## 사용 가이드

### 메서드 호출 순서

#### `generate_answer_enhanced` 호출 시
```python
state = workflow.generate_answer_enhanced(state)
# 내부적으로 다음 순서로 실행:
# 1. _recover_retrieved_docs_at_start
# 2. _validate_and_generate_prompt_context
# 3. _build_and_validate_context_dict
# 4. _validate_context_quality_and_expand
# 5. _inject_search_results_into_context
# 6. _generate_and_validate_prompt
# 7. _generate_answer_with_cache
# 8. _validate_and_enhance_answer
```

#### `process_search_results_combined` 호출 시
```python
state = workflow.process_search_results_combined(state)
# 내부적으로 다음 순서로 실행:
# 1. _prepare_search_inputs
# 2. _perform_conditional_retry_search
# 3. _merge_and_rerank_results
# 4. _apply_keyword_weights_and_rerank
# 5. _filter_and_validate_documents
# 6. _ensure_diversity_and_limit
# 7. _save_final_results_to_state
```

#### `generate_answer_final` 호출 시
```python
state = workflow.generate_answer_final(state)
# 내부적으로 다음 순서로 실행:
# 1. _restore_state_data_for_final
# 2. _validate_and_handle_regeneration
# 3. _handle_format_errors
# 4. _format_and_finalize (품질 검증 통과 시)
```

## 주의사항

1. **State 수정**: 모든 헬퍼 메서드는 `state`를 직접 수정합니다. 원본 state를 보존하려면 복사본을 사용하세요.

2. **의존성**: 헬퍼 메서드들은 특정 순서로 호출되어야 합니다. 메인 메서드를 통해 호출하는 것을 권장합니다.

3. **오류 처리**: 각 헬퍼 메서드는 내부적으로 오류를 처리하지만, 메인 메서드에서도 최종 오류 처리를 수행합니다.

4. **캐싱**: `_generate_answer_with_cache`는 캐시를 활용하므로, 동일한 입력에 대해 빠른 응답을 제공할 수 있습니다.

5. **전역 캐시**: `_save_final_results_to_state`는 전역 캐시에도 저장하므로, State reduction 후에도 데이터를 복원할 수 있습니다.

