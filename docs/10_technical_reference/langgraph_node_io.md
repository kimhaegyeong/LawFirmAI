# LangGraph Nodes Input/Output Specification

본 문서는 `source/agents/node_input_output_spec.py`에 정의된 LangGraph 노드들의 입력/출력 사양을 정리한 레퍼런스입니다. 각 노드는 필요한 최소 입력만 요구하도록 설계되어 있으며, 상태(State) 그룹 의존성을 최소화하여 메모리와 결합도를 낮춥니다.

### 범례
- required_input: 노드 수행에 반드시 필요한 입력 필드
- optional_input: 있으면 활용하는 선택 입력 필드
- output: 노드가 생성/갱신하는 출력 필드
- required_state_groups: 노드가 접근하는 최소 상태 그룹
- output_state_groups: 노드가 출력으로 갱신하는 상태 그룹

---

### classify_query
- 설명: 질문 유형 분류 및 법률 분야 판단
- required_input: `query`
- optional_input: `legal_field`
- output: `query_type`, `confidence`, `legal_field`, `legal_domain`
- required_state_groups: `input`
- output_state_groups: `classification`

### assess_urgency
- 설명: 질문의 긴급도 평가
- required_input: `query`
- optional_input: `query_type`, `legal_field`
- output: `urgency_level`, `urgency_reasoning`, `emergency_type`
- required_state_groups: `input`
- output_state_groups: `classification`

### resolve_multi_turn
- 설명: 멀티턴 대화 처리
- required_input: `query`
- optional_input: (없음)
- output: `is_multi_turn`, `multi_turn_confidence`, `conversation_history`, `conversation_context`
- required_state_groups: `input`
- output_state_groups: `multi_turn`

### route_expert
- 설명: 전문가 라우팅 결정
- required_input: `query`, `query_type`
- optional_input: `legal_field`, `urgency_level`
- output: `complexity_level`, `requires_expert`, `expert_subgraph`
- required_state_groups: `input`, `classification`
- output_state_groups: `classification`

### analyze_document
- 설명: 업로드된 문서 분석
- required_input: `query`
- optional_input: `document_file`
- output: `document_type`, `document_analysis`, `key_clauses`, `potential_issues`
- required_state_groups: `input`
- output_state_groups: `document`

### expand_keywords_ai
- 설명: AI 기반 키워드 확장
- required_input: `query`, `query_type`
- optional_input: `legal_field`, `extracted_keywords`
- output: `search_query`, `extracted_keywords`, `ai_keyword_expansion`
- required_state_groups: `input`, `classification`
- output_state_groups: `search`

### prepare_search_query
- 설명: 검색 쿼리 준비 및 최적화
- required_input: `query`, `query_type`
- optional_input: `legal_field`, `extracted_keywords`, `search_query`
- output: `optimized_queries`, `search_params`, `search_cache_hit`
- required_state_groups: `input`, `classification`
- output_state_groups: `search`

### execute_searches_parallel
- 설명: 의미적 검색과 키워드 검색을 병렬로 실행
- required_input: `query`, `optimized_queries`, `search_params`
- optional_input: `query_type`, `legal_field`, `extracted_keywords`
- output: `semantic_results`, `keyword_results`, `semantic_count`, `keyword_count`
- required_state_groups: `input`, `search`
- output_state_groups: `search`

### evaluate_search_quality
- 설명: 검색 결과 품질 평가
- required_input: `semantic_results`, `keyword_results`
- optional_input: `query`, `query_type`, `search_params`
- output: `search_quality_evaluation`
- required_state_groups: `input`, `search`
- output_state_groups: `search`, `common`

### conditional_retry_search
- 설명: 검색 품질에 따른 조건부 재검색
- required_input: `search_quality_evaluation`, `semantic_results`, `keyword_results`
- optional_input: `query`, `optimized_queries`
- output: `semantic_results`, `keyword_results`
- required_state_groups: `input`, `search`
- output_state_groups: `search`

### merge_and_rerank_with_keyword_weights
- 설명: 키워드별 가중치를 적용한 결과 병합 및 Reranking
- required_input: `semantic_results`, `keyword_results`
- optional_input: `query`, `optimized_queries`, `search_params`, `extracted_keywords`, `legal_field`
- output: `merged_documents`, `keyword_weights`, `retrieved_docs`
- required_state_groups: `input`, `search`
- output_state_groups: `search`

### filter_and_validate_results
- 설명: 검색 결과 필터링 및 품질 검증
- required_input: `merged_documents`
- optional_input: `query`, `query_type`, `legal_field`, `search_params`, `retrieved_docs`
- output: `retrieved_docs`
- required_state_groups: `input`, `search`
- output_state_groups: `search`

### update_search_metadata
- 설명: 검색 메타데이터 업데이트
- required_input: `retrieved_docs`
- optional_input: `semantic_count`, `keyword_count`, `optimized_queries`
- output: `search_metadata`
- required_state_groups: `input`, `search`
- output_state_groups: `search`, `common`

### process_legal_terms
- 설명: 법률 용어 처리 및 통합
- required_input: `query`, `retrieved_docs`
- optional_input: `legal_field`
- output: `legal_references`, `legal_citations`, `analysis`
- required_state_groups: `input`, `search`
- output_state_groups: `analysis`

### prepare_document_context_for_prompt
- 설명: 프롬프트용 문서 컨텍스트 준비
- required_input: `query`, `retrieved_docs`
- optional_input: `query_type`, `extracted_keywords`, `legal_field`
- output: `prompt_optimized_context`
- required_state_groups: `input`, `search`
- output_state_groups: `search`, `common`

### generate_answer_enhanced
- 설명: 향상된 답변 생성 (LLM 활용)
- required_input: `query`, `retrieved_docs`
- optional_input: `query_type`, `legal_field`, `analysis`, `legal_references`, `prompt_optimized_context`
- output: `answer`, `confidence`, `legal_references`, `legal_citations`
- required_state_groups: `input`, `search`
- output_state_groups: `answer`, `analysis`, `common`

### validate_answer_quality
- 설명: 답변 품질 및 법령 검증
- required_input: `answer`, `query`
- optional_input: `retrieved_docs`, `sources`, `legal_references`
- output: `quality_check_passed`, `quality_score`, `legal_validity_check`, `legal_basis_validation`
- required_state_groups: `input`, `answer`
- output_state_groups: `validation`, `control`, `common`

### enhance_answer_structure
- 설명: 답변 구조화 및 법적 근거 강화
- required_input: `answer`, `query_type`
- optional_input: `legal_references`, `legal_citations`, `retrieved_docs`
- output: `answer`, `structure_confidence`
- required_state_groups: `answer`, `classification`
- output_state_groups: `answer`

### apply_visual_formatting
- 설명: 시각적 포맷팅 적용
- required_input: `answer`
- optional_input: `query_type`, `legal_references`
- output: `answer`
- required_state_groups: `answer`
- output_state_groups: `answer`

### prepare_final_response
- 설명: 최종 응답 준비
- required_input: `answer`
- optional_input: `sources`, `legal_references`, `confidence`, `legal_validity_check`
- output: `answer`, `sources`, `confidence`
- required_state_groups: `answer`
- output_state_groups: `answer`, `common`

---

## 노드 카테고리

각 노드는 다음 카테고리 중 하나에 속합니다:

- **INPUT**: 입력 처리 노드
- **CLASSIFICATION**: 분류 및 라우팅 노드 (classify_query, assess_urgency, resolve_multi_turn, route_expert, analyze_document)
- **SEARCH**: 검색 관련 노드 (expand_keywords_ai, prepare_search_query, execute_searches_parallel, evaluate_search_quality, conditional_retry_search, merge_and_rerank_with_keyword_weights, filter_and_validate_results, update_search_metadata)
- **GENERATION**: 답변 생성 노드 (generate_answer_enhanced, prepare_final_response)
- **VALIDATION**: 검증 노드 (validate_answer_quality)
- **ENHANCEMENT**: 향상 노드 (process_legal_terms, prepare_document_context_for_prompt, enhance_answer_structure, apply_visual_formatting)
- **CONTROL**: 제어 노드

## 설계 원칙

### 최소 입력 원칙
각 노드는 필요한 필드만 요구하여 상태 전달을 절약합니다.

### 그룹 의존 최소화
`required_state_groups`를 축소하여 불필요한 그룹 접근을 방지합니다.

### 추적 가능성
품질 점수·메타데이터는 `common` 그룹에 표준화하여 경로 경고를 제거합니다.

### 선택 입력 정리
사용 빈도가 낮거나 어댑터에서 관리되는 필드(예: `conversation_history`)는 optional_input에서 제거했습니다.

## State 그룹 구조

LangGraph 워크플로우는 다음과 같은 State 그룹을 사용합니다:

- **input**: 사용자 입력 (query, session_id)
- **classification**: 분류 결과 (query_type, legal_field, urgency_level 등)
- **multi_turn**: 멀티턴 처리 결과
- **search**: 검색 관련 데이터 (retrieved_docs, search_query 등)
- **analysis**: 분석 결과 (legal_references, analysis 등)
- **answer**: 답변 데이터 (answer, confidence 등)
- **validation**: 검증 결과 (quality_check_passed, legal_validity_check 등)
- **document**: 문서 분석 결과
- **control**: 제어 플래그 (retry_count, quality_check_passed 등)
- **common**: 공통 메타데이터 (processing_time, metadata 등)
