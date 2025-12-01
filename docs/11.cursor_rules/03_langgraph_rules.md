# LangGraph 개발 규칙 (CRITICAL)

**LangGraph의 node, edge, task를 올바르게 사용하여 워크플로우를 개발합니다.**

## 1. Node 정의 및 사용 규칙

### 1-1. Node 함수 시그니처
```python
from lawfirm_langgraph.core.workflow.state.state_definitions import LegalWorkflowState
from typing import Dict, Any

def my_node(state: LegalWorkflowState) -> LegalWorkflowState:
    """
    Node 함수는 반드시 다음 시그니처를 따라야 합니다:
    - 입력: LegalWorkflowState (또는 Dict[str, Any])
    - 출력: LegalWorkflowState (또는 Dict[str, Any])
    """
    # State에서 값 읽기
    query = state.get("query", "")
    
    # 작업 수행
    result = process_query(query)
    
    # State 업데이트
    state["result"] = result
    
    return state
```

### 1-2. Node 추가 패턴
```python
from langgraph.graph import StateGraph, END
from lawfirm_langgraph.core.workflow.state.state_definitions import LegalWorkflowState

def build_graph(self) -> StateGraph:
    """워크플로우 그래프 구축"""
    workflow = StateGraph(LegalWorkflowState)
    
    # Node 추가
    workflow.add_node("node_name", self.node_handler_function)
    
    # Entry point 설정
    workflow.set_entry_point("node_name")
    
    return workflow
```

### 1-3. Node 명명 규칙
- **snake_case** 사용 (예: `classify_query`, `generate_answer`)
- 동사로 시작 (예: `expand_keywords`, `process_results`)
- 명확하고 간결한 이름 사용
- 동일한 기능은 동일한 이름 사용 (일관성 유지)

### 1-4. Node 내부 처리 규칙
```python
def my_node(state: LegalWorkflowState) -> LegalWorkflowState:
    """Node 처리 예시"""
    try:
        # 1. State에서 필요한 값 추출
        query = state.get("query", "")
        context = state.get("context", [])
        
        # 2. 작업 수행
        result = perform_work(query, context)
        
        # 3. State 업데이트 (명시적으로)
        state["result"] = result
        state["metadata"] = {"processed_at": datetime.now()}
        
        # 4. 로깅
        self.logger.info(f"Node completed: {result}")
        
        return state
        
    except Exception as e:
        # 에러 처리 및 State 업데이트
        self.logger.error(f"Node error: {e}", exc_info=True)
        state["error"] = str(e)
        return state
```

## 2. Edge 정의 및 사용 규칙

### 2-1. 일반 Edge (Unconditional Edge)
```python
# 단순한 순차적 흐름
workflow.add_edge("node_a", "node_b")
workflow.add_edge("node_b", END)  # 종료점
```

### 2-2. 조건부 Edge (Conditional Edge)
```python
def route_function(state: LegalWorkflowState) -> str:
    """
    라우팅 함수는 반드시 다음을 반환해야 합니다:
    - 문자열: 다음 노드 이름
    - 또는 조건부 엣지 맵의 키
    """
    complexity = state.get("query_complexity", "moderate")
    
    if complexity == "simple":
        return "simple"  # 조건부 엣지 맵의 키
    elif complexity == "moderate":
        return "moderate"
    else:
        return "complex"

# 조건부 엣지 추가
workflow.add_conditional_edges(
    "source_node",
    route_function,  # 라우팅 함수
    {
        "simple": "direct_answer_node",
        "moderate": "classification_node",
        "complex": "expert_routing_node"
    }
)
```

### 2-3. 라우팅 함수 작성 규칙
```python
class WorkflowRoutes:
    """라우팅 함수들을 모아놓은 클래스"""
    
    def route_by_complexity(self, state: LegalWorkflowState) -> str:
        """복잡도에 따라 라우팅"""
        complexity = state.get("query_complexity", "moderate")
        
        # Enum 값 처리
        if hasattr(complexity, 'value'):
            complexity = complexity.value
        
        # 문자열 비교
        if complexity == "simple" or complexity == QueryComplexity.SIMPLE:
            return "simple"
        elif complexity == "moderate" or complexity == QueryComplexity.MODERATE:
            return "moderate"
        else:
            return "complex"
    
    def should_retry(self, state: LegalWorkflowState) -> str:
        """재시도 여부 판단"""
        retry_count = state.get("retry_count", 0)
        max_retries = state.get("max_retries", 3)
        
        if retry_count >= max_retries:
            return "accept"  # 재시도 포기
        else:
            return "retry"  # 재시도
```

### 2-4. 조건부 Edge 맵 작성 규칙
```python
# ✅ 좋은 예: 명확한 키와 값
workflow.add_conditional_edges(
    "generate_answer",
    self._should_retry_validation_func,
    {
        "accept": END,  # 종료
        "retry_generate": "generate_answer",  # 재생성
        "retry_search": "expand_keywords"  # 재검색
    }
)

# ❌ 나쁜 예: 모호한 키나 잘못된 노드 이름
workflow.add_conditional_edges(
    "generate_answer",
    self._should_retry_validation_func,
    {
        "ok": END,  # 모호한 키
        "retry": "generate_answer",  # 순환 참조 가능성
        "search": "expand_keywords"  # 불명확한 의미
    }
)
```

## 3. Task 정의 및 사용 규칙

### 3-1. Task 개념
- **Task**: Node 내부에서 병렬로 실행 가능한 작은 작업 단위
- **Node**: 하나 이상의 Task를 포함하는 워크플로우 단위
- **Subgraph**: 여러 Node를 포함하는 하위 그래프

### 3-2. Task를 Node로 분리하는 패턴
```python
# ❌ 나쁜 예: 하나의 큰 Node에 모든 작업 포함
def process_search_results_combined(state: LegalWorkflowState) -> LegalWorkflowState:
    # 600줄의 복잡한 로직
    # 1. 품질 평가
    # 2. 재검색
    # 3. 병합
    # 4. 재순위
    # 5. 필터링
    # 6. 메타데이터 업데이트
    pass

# ✅ 좋은 예: Task를 별도 Node로 분리
def evaluate_search_quality(state: LegalWorkflowState) -> LegalWorkflowState:
    """검색 결과 품질 평가"""
    quality_score = evaluate_quality(state.get("search_results", []))
    state["quality_score"] = quality_score
    return state

def merge_and_rerank(state: LegalWorkflowState) -> LegalWorkflowState:
    """결과 병합 및 재순위"""
    merged = merge_results(state.get("search_results", []))
    reranked = rerank_documents(merged)
    state["search_results"] = reranked
    return state

# 그래프에 추가
workflow.add_node("evaluate_search_quality", evaluate_search_quality)
workflow.add_node("merge_and_rerank", merge_and_rerank)
workflow.add_edge("evaluate_search_quality", "merge_and_rerank")
```

### 3-3. 병렬 Task 처리 패턴
```python
import asyncio
from typing import List, Dict, Any

def process_parallel_tasks(state: LegalWorkflowState) -> LegalWorkflowState:
    """병렬 Task 처리"""
    async def process_tasks():
        tasks = [
            evaluate_semantic_quality(state),
            evaluate_keyword_quality(state),
            calculate_relevance_scores(state)
        ]
        results = await asyncio.gather(*tasks, return_exceptions=True)
        return results
    
    # 동기 함수에서 비동기 실행
    results = asyncio.run(process_tasks())
    
    # 결과를 State에 반영
    state["semantic_quality"] = results[0]
    state["keyword_quality"] = results[1]
    state["relevance_scores"] = results[2]
    
    return state
```

## 4. StateGraph 구축 패턴

### 4-1. 그래프 빌더 패턴
```python
class WorkflowGraphBuilder:
    """워크플로우 그래프 빌더"""
    
    def __init__(self, config, logger, routing_functions):
        self.config = config
        self.logger = logger
        self.routing_functions = routing_functions
    
    def build_graph(self, node_handlers: Dict[str, Callable]) -> StateGraph:
        """그래프 구축"""
        workflow = StateGraph(LegalWorkflowState)
        
        # 1. Node 추가
        self.add_nodes(workflow, node_handlers)
        
        # 2. Entry point 설정
        self.setup_entry_point(workflow)
        
        # 3. 라우팅 설정 (조건부 엣지)
        self.setup_routing(workflow)
        
        # 4. 일반 엣지 추가
        self.add_edges(workflow)
        
        return workflow
    
    def add_nodes(self, workflow: StateGraph, node_handlers: Dict[str, Callable]):
        """Node 추가"""
        for node_name, handler in node_handlers.items():
            if handler:  # None 체크
                workflow.add_node(node_name, handler)
                self.logger.info(f"Node added: {node_name}")
    
    def setup_entry_point(self, workflow: StateGraph):
        """Entry point 설정"""
        workflow.set_entry_point("classify_query_and_complexity")
    
    def setup_routing(self, workflow: StateGraph):
        """조건부 엣지 설정"""
        workflow.add_conditional_edges(
            "classify_query_and_complexity",
            self.routing_functions["route_by_complexity"],
            {
                "simple": "direct_answer",
                "moderate": "classification_parallel",
                "complex": "classification_parallel"
            }
        )
    
    def add_edges(self, workflow: StateGraph):
        """일반 엣지 추가"""
        workflow.add_edge("direct_answer", END)
        workflow.add_edge("classification_parallel", "route_expert")
```

### 4-2. 그래프 구조 설계 원칙
1. **단일 책임 원칙**: 각 Node는 하나의 명확한 작업만 수행
2. **재사용성**: 공통 작업은 별도 Node로 분리하여 재사용
3. **테스트 가능성**: 각 Node는 독립적으로 테스트 가능해야 함
4. **명확한 흐름**: 그래프 구조가 비즈니스 로직을 명확히 표현

## 5. Subgraph 사용 규칙

### 5-1. Subgraph 생성 패턴
```python
from langgraph.graph import StateGraph

def create_search_results_subgraph() -> StateGraph:
    """검색 결과 처리 서브그래프"""
    subgraph = StateGraph(LegalWorkflowState)
    
    # Subgraph 내부 Node 추가
    subgraph.add_node("evaluate_quality", evaluate_search_quality)
    subgraph.add_node("merge_results", merge_and_rerank)
    subgraph.add_node("filter_results", filter_and_validate)
    
    # Subgraph 내부 엣지
    subgraph.set_entry_point("evaluate_quality")
    subgraph.add_edge("evaluate_quality", "merge_results")
    subgraph.add_edge("merge_results", "filter_results")
    subgraph.add_edge("filter_results", END)
    
    return subgraph.compile()

# 메인 그래프에 Subgraph 추가
main_workflow.add_node("search_results_processing", create_search_results_subgraph())
```

### 5-2. Subgraph 사용 시나리오
- **복잡한 로직 그룹화**: 관련된 여러 Node를 하나의 Subgraph로 묶기
- **재사용성**: 동일한 로직을 여러 곳에서 사용할 때
- **모듈화**: 큰 워크플로우를 작은 단위로 분리

## 6. State 관리 규칙

### 6-1. State 읽기/쓰기 패턴
```python
def my_node(state: LegalWorkflowState) -> LegalWorkflowState:
    """State 관리 예시"""
    # ✅ 좋은 예: 명시적인 State 접근
    query = state.get("query", "")
    context = state.get("context", [])
    
    # 작업 수행
    result = process(query, context)
    
    # ✅ 좋은 예: State 업데이트 (명시적)
    state["result"] = result
    state["metadata"] = {
        "processed_at": datetime.now().isoformat(),
        "node_name": "my_node"
    }
    
    return state
```

### 6-2. State 최적화 패턴
```python
from lawfirm_langgraph.core.shared.wrappers.node_wrappers import with_state_optimization

@with_state_optimization("node_name", enable_reduction=True)
def my_node(state: LegalWorkflowState) -> LegalWorkflowState:
    """State 최적화가 적용된 Node"""
    # State 최적화 데코레이터가 자동으로:
    # 1. 불필요한 State 필드 제거
    # 2. 메모리 사용량 최적화
    # 3. 성능 향상
    pass
```

## 7. 에러 처리 규칙

### 7-1. Node 내부 에러 처리
```python
def my_node(state: LegalWorkflowState) -> LegalWorkflowState:
    """에러 처리 예시"""
    try:
        # 작업 수행
        result = perform_work(state)
        state["result"] = result
        return state
        
    except ValueError as e:
        # 입력 오류 처리
        self.logger.error(f"Value error in my_node: {e}")
        state["error"] = f"입력 오류: {str(e)}"
        state["error_type"] = "value_error"
        return state
        
    except Exception as e:
        # 예상치 못한 오류 처리
        self.logger.error(f"Unexpected error in my_node: {e}", exc_info=True)
        state["error"] = f"처리 중 오류 발생: {str(e)}"
        state["error_type"] = "unexpected_error"
        return state
```

### 7-2. 라우팅 함수 에러 처리
```python
def route_function(state: LegalWorkflowState) -> str:
    """라우팅 함수 에러 처리"""
    try:
        complexity = state.get("query_complexity", "moderate")
        
        if complexity == "simple":
            return "simple"
        elif complexity == "moderate":
            return "moderate"
        else:
            return "complex"
            
    except Exception as e:
        # 기본값 반환 (안전한 폴백)
        self.logger.error(f"Routing error: {e}", exc_info=True)
        return "moderate"  # 기본 라우팅
```

## 8. 테스트 규칙

### 8-1. Node 단위 테스트
```python
import pytest
from lawfirm_langgraph.core.workflow.state.state_definitions import LegalWorkflowState

def test_my_node():
    """Node 테스트"""
    # 초기 State 생성
    initial_state: LegalWorkflowState = {
        "query": "테스트 질문",
        "context": []
    }
    
    # Node 실행
    result_state = my_node(initial_state)
    
    # 결과 검증
    assert "result" in result_state
    assert result_state["result"] is not None
```

### 8-2. 라우팅 함수 테스트
```python
def test_route_by_complexity():
    """라우팅 함수 테스트"""
    # 테스트 케이스 1: 간단한 질문
    state_simple: LegalWorkflowState = {
        "query_complexity": "simple"
    }
    assert route_by_complexity(state_simple) == "simple"
    
    # 테스트 케이스 2: 복잡한 질문
    state_complex: LegalWorkflowState = {
        "query_complexity": "complex"
    }
    assert route_by_complexity(state_complex) == "complex"
```

## 9. 성능 최적화 규칙

### 9-1. Node 최적화
- **State 최적화 데코레이터 사용**: `@with_state_optimization`
- **불필요한 State 필드 제거**: Node에서 사용하지 않는 필드는 제거
- **캐싱 활용**: 반복적인 작업은 캐싱

### 9-2. 병렬 처리 활용
- **병렬 실행 가능한 작업 분리**: 별도 Node로 분리하여 병렬 실행
- **비동기 처리**: I/O 작업은 비동기로 처리

## 10. 금지 사항

1. **Node 함수 시그니처 위반 금지**
   - State를 받지 않는 Node 함수 금지
   - State를 반환하지 않는 Node 함수 금지

2. **순환 참조 금지**
   - 무한 루프를 일으킬 수 있는 엣지 구조 금지
   - 재시도 로직은 명확한 종료 조건 필요

3. **State 직접 수정 금지**
   - State를 복사하지 않고 직접 수정하는 것 금지 (가능한 경우)
   - 불변성(immutability) 원칙 준수

4. **라우팅 함수에서 State 수정 금지**
   - 라우팅 함수는 라우팅 결정만 수행
   - State 수정은 Node에서만 수행

## 11. 체크리스트

LangGraph 개발 시 다음을 확인하세요:

- [ ] Node 함수가 올바른 시그니처를 가지고 있는가?
- [ ] 모든 Node가 그래프에 추가되었는가?
- [ ] Entry point가 설정되었는가?
- [ ] 조건부 엣지의 라우팅 함수가 올바르게 구현되었는가?
- [ ] 조건부 엣지 맵의 키가 라우팅 함수 반환값과 일치하는가?
- [ ] 모든 엣지가 올바른 노드 이름을 참조하는가?
- [ ] 에러 처리가 적절히 구현되었는가?
- [ ] State 최적화가 적용되었는가?
- [ ] 로깅이 적절히 구현되었는가?

