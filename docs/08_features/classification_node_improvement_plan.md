# Classification Node 개선 계획

## 📋 문제 분석

### 현재 상황

**워크플로우 실행 순서**:
1. `classify_query_and_complexity` (엔트리 포인트) - 질의 분류 및 복잡도 평가
2. `expand_keywords` - 키워드 확장
3. `multi_query_search_agent` - 멀티 질의 생성
4. `prepare_search_query` - 검색 쿼리 준비 (여기서 `complexity`를 사용하여 멀티 질의 개수 결정)

**문제점**:
1. `classify_query_and_complexity`가 키워드 확장 전에 실행되어 확장 결과를 반영하지 못함
2. 멀티 질의 생성 전에 복잡도가 결정되어 실제 질의 특성을 반영하지 못함
3. `prepare_search_query`에서 사용하는 `complexity`가 실제 검색 결과를 반영하지 못함

### 핵심 제약사항

**`query_type`이 검색 필터링에 직접 사용됨**:
- `multi_query_search_agent.py`의 `_get_source_types_from_query_type()` 메서드가 `query_type`에 따라 검색할 문서 타입을 결정
- `law_inquiry`: `["statute_article", "precedent_content"]`
- `precedent_search`: `["precedent_content"]`
- `general_question`: `None` (모든 타입)
- `legal_advice`: `None` (모든 타입)

**따라서**:
- `query_type`은 검색 전에 결정되어야 함 (검색 필터링에 필수)
- `complexity`는 멀티 질의 개수 결정에 사용되므로 키워드 확장 후 재평가하면 더 정확함

## 🎯 개선 목표

1. **`query_type` 빠른 결정**: 검색 필터링에 필요한 `query_type`을 빠르게 결정
2. **`complexity` 정확한 평가**: 키워드 확장 결과를 반영하여 복잡도를 재평가
3. **멀티 질의 개수 최적화**: 정확한 복잡도 평가를 통해 적절한 멀티 질의 개수 결정
4. **검색 품질 향상**: 올바른 문서 타입 필터링으로 검색 품질 향상

## 🔧 개선 방안

### 방안 1: 2단계 분류 (권장)

**구조**:
```
1단계: classify_query_simple (엔트리 포인트)
  → query_type만 빠르게 결정 (검색 필터링에 필수)
  
2단계: classify_complexity_after_keywords (키워드 확장 후)
  → complexity 재평가 (키워드 확장 결과 반영)
  → query_type 재평가 (선택적, 필요시)
```

**장점**:
- ✅ `query_type`은 빠르게 결정되어 검색 필터링에 사용 가능
- ✅ `complexity`는 키워드 확장 결과를 반영하여 더 정확하게 평가
- ✅ 기존 구조와의 호환성 유지
- ✅ 점진적 개선 가능

**단점**:
- ⚠️ 2단계 분류로 인한 약간의 오버헤드 (하지만 키워드 확장과 병렬 가능)

### 방안 2: 동적 복잡도 재평가

**구조**:
```
classify_query_and_complexity (엔트리 포인트)
  → query_type + 초기 complexity 결정
  
expand_keywords
  → 키워드 확장
  
prepare_search_query (개선)
  → 키워드 확장 결과를 반영하여 complexity 재평가
  → 멀티 질의 개수 동적 조정
```

**장점**:
- ✅ 기존 구조 변경 최소화
- ✅ 키워드 확장 결과 반영

**단점**:
- ⚠️ `prepare_search_query` 노드가 복잡해짐
- ⚠️ 복잡도 재평가 로직이 분산됨

### 방안 3: 하이브리드 접근

**구조**:
```
classify_query_simple (엔트리 포인트)
  → query_type만 빠르게 결정
  
expand_keywords
  → 키워드 확장
  
classify_complexity_after_keywords
  → complexity 재평가
  → query_type 재평가 (선택적, 신뢰도 낮을 때만)
```

**장점**:
- ✅ `query_type` 빠른 결정
- ✅ `complexity` 정확한 평가
- ✅ 필요시 `query_type` 재평가 가능

**단점**:
- ⚠️ 구현 복잡도 증가

## ✅ 권장 방안: 방안 1 (2단계 분류)

### 구현 계획

#### 1단계: `classify_query_simple` 노드 추가

**목적**: `query_type`만 빠르게 결정

**구현 위치**: `lawfirm_langgraph/core/workflow/nodes/classification_nodes.py`

**기능**:
- 질의 타입만 분류 (`law_inquiry`, `precedent_search`, `general_question`, `legal_advice`)
- 빠른 규칙 기반 분류 또는 경량 LLM 호출
- 초기 `complexity`는 기본값 설정 (나중에 재평가)

**시그니처**:
```python
def classify_query_simple(self, state: LegalWorkflowState) -> LegalWorkflowState:
    """질의 타입만 빠르게 분류 (검색 필터링에 필수)"""
    query = state.query
    
    # 빠른 분류 (규칙 기반 또는 경량 LLM)
    query_type, confidence = self._quick_classify_query_type(query)
    
    # 초기 complexity는 기본값 설정 (나중에 재평가)
    initial_complexity = QueryComplexity.MODERATE  # 기본값
    
    return state.update(
        query_type=query_type,
        query_type_confidence=confidence,
        complexity=initial_complexity,
        needs_search=True  # 대부분 검색 필요
    )
```

#### 2단계: `classify_complexity_after_keywords` 노드 추가

**목적**: 키워드 확장 결과를 반영하여 복잡도 재평가

**구현 위치**: `lawfirm_langgraph/core/workflow/nodes/classification_nodes.py`

**기능**:
- 키워드 확장 결과를 반영하여 복잡도 재평가
- 필요시 `query_type` 재평가 (신뢰도 낮을 때만)
- 멀티 질의 개수 결정에 사용될 정확한 복잡도 제공

**시그니처**:
```python
def classify_complexity_after_keywords(self, state: LegalWorkflowState) -> LegalWorkflowState:
    """키워드 확장 결과를 반영하여 복잡도 재평가"""
    query = state.query
    expanded_keywords = state.expanded_keywords or []
    query_type = state.query_type
    query_type_confidence = state.query_type_confidence or 0.0
    
    # 키워드 확장 결과를 반영하여 복잡도 평가
    complexity, complexity_confidence = self._classify_complexity_with_keywords(
        query, query_type, expanded_keywords
    )
    
    # query_type 재평가 (신뢰도 낮을 때만)
    if query_type_confidence < 0.7 and expanded_keywords:
        # 키워드 확장 결과를 반영하여 query_type 재평가
        new_query_type, new_confidence = self._reclassify_query_type_with_keywords(
            query, query_type, expanded_keywords
        )
        if new_confidence > query_type_confidence:
            query_type = new_query_type
            query_type_confidence = new_confidence
    
    return state.update(
        complexity=complexity,
        complexity_confidence=complexity_confidence,
        query_type=query_type,
        query_type_confidence=query_type_confidence
    )
```

#### 3단계: 워크플로우 엣지 구조 수정

**수정 위치**: `lawfirm_langgraph/core/workflow/builders/modular_graph_builder.py`

**변경사항**:
```python
# 엔트리 포인트 변경
workflow.set_entry_point("classify_query_simple")  # 기존: classify_query_and_complexity

# 엣지 추가
workflow.add_edge("classify_query_simple", "_route_by_complexity")
workflow.add_edge("expand_keywords", "classify_complexity_after_keywords")
workflow.add_edge("classify_complexity_after_keywords", "multi_query_search_agent")
```

**수정 위치**: `lawfirm_langgraph/core/workflow/edges/classification_edges.py`

**변경사항**:
```python
def add_classification_edges(self, workflow: StateGraph, use_agentic_mode: bool = False) -> None:
    # classify_query_simple → _route_by_complexity
    workflow.add_conditional_edges(
        "classify_query_simple",  # 기존: classify_query_and_complexity
        self.route_by_complexity_func,
        {
            "ethical_reject": "ethical_rejection",
            "simple": "direct_answer_node",
            "moderate": "classification_parallel",
            "complex": "classification_parallel",
        }
    )
    
    # expand_keywords → classify_complexity_after_keywords
    workflow.add_edge("expand_keywords", "classify_complexity_after_keywords")
    
    # classify_complexity_after_keywords → multi_query_search_agent
    workflow.add_edge("classify_complexity_after_keywords", "multi_query_search_agent")
```

#### 4단계: `_route_by_complexity` 함수 수정

**수정 위치**: `lawfirm_langgraph/core/workflow/legal_workflow_enhanced.py`

**변경사항**:
- `classify_query_simple`에서 초기 `complexity`를 설정하므로 기존 로직 유지
- 다만 초기 `complexity`는 기본값이므로, 실제 복잡도는 `classify_complexity_after_keywords`에서 결정됨

#### 5단계: `prepare_search_query` 수정

**수정 위치**: `lawfirm_langgraph/core/workflow/legal_workflow_enhanced.py`

**변경사항**:
- `classify_complexity_after_keywords`에서 이미 정확한 `complexity`를 설정하므로 기존 로직 유지
- 다만 `complexity`가 이미 재평가되었으므로 추가 재평가는 불필요

## 📊 예상 효과

### 성능 개선
- ✅ 멀티 질의 개수 최적화: 정확한 복잡도 평가로 적절한 멀티 질의 개수 결정
- ✅ 검색 품질 향상: 올바른 문서 타입 필터링으로 관련 문서 검색률 향상
- ✅ 불필요한 검색 감소: 잘못된 `query_type`으로 인한 불필요한 검색 감소

### 품질 개선
- ✅ 더 정확한 복잡도 평가: 키워드 확장 결과 반영
- ✅ 더 정확한 질의 타입 분류: 필요시 재평가로 정확도 향상
- ✅ 더 나은 멀티 질의 생성: 정확한 복잡도 기반 멀티 질의 개수 결정

## 🔄 마이그레이션 계획

### Phase 1: 노드 추가 (비파괴적)
1. `classify_query_simple` 노드 추가
2. `classify_complexity_after_keywords` 노드 추가
3. 기존 `classify_query_and_complexity` 노드 유지 (하위 호환성)

### Phase 2: 엣지 구조 수정
1. 엔트리 포인트 변경
2. 엣지 추가 및 수정
3. 테스트 및 검증

### Phase 3: 기존 노드 제거 (선택적)
1. 기존 `classify_query_and_complexity` 노드 제거 (사용하지 않을 경우)
2. 정리 및 문서화

## 🧪 테스트 계획

### 단위 테스트
- `classify_query_simple` 노드 테스트
- `classify_complexity_after_keywords` 노드 테스트
- 키워드 확장 결과 반영 테스트

### 통합 테스트
- 전체 워크플로우 실행 테스트
- 검색 필터링 정확도 테스트
- 멀티 질의 개수 최적화 테스트

### 성능 테스트
- 응답 시간 측정
- LLM 호출 횟수 측정
- 검색 품질 메트릭 측정

## 📝 참고사항

### 기존 코드 호환성
- 기존 `classify_query_and_complexity` 노드는 유지하여 하위 호환성 보장
- 점진적 마이그레이션 가능

### 환경 변수
- `USE_TWO_STAGE_CLASSIFICATION`: 2단계 분류 사용 여부 (기본값: `True`)
- `REQUERY_TYPE_RECLASSIFICATION`: `query_type` 재평가 사용 여부 (기본값: `False`)

### 로깅
- 각 단계별 분류 결과 로깅
- 키워드 확장 결과 반영 여부 로깅
- 재평가 발생 여부 로깅

