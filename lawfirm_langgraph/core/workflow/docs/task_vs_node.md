# Task vs Node 역할 명확화

## 개요

LangGraph 워크플로우에서 Task와 Node의 역할을 명확히 구분하여 코드의 가독성과 유지보수성을 향상시킵니다.

## Task

**Task**는 Node 내부에서 사용되는 재사용 가능한 작업 단위입니다.

### 특징
- 순수 함수/메서드로 구현
- State를 직접 다루지 않음
- 비즈니스 로직만 담당
- 여러 Node에서 재사용 가능
- 독립적으로 테스트 가능

### 예시
```python
# Task 예시: SearchExecutionTasks
class SearchExecutionTasks:
    @staticmethod
    def execute_searches_sync(
        optimized_queries: Dict[str, Any],
        search_params: Dict[str, Any],
        ...
    ) -> Tuple[List[Dict], int, List[Dict], int]:
        """검색 실행 Task - State를 직접 다루지 않음"""
        # 비즈니스 로직만 수행
        semantic_results, semantic_count = execute_semantic_search(...)
        keyword_results, keyword_count = execute_keyword_search(...)
        return semantic_results, semantic_count, keyword_results, keyword_count
```

## Node

**Node**는 LangGraph의 실행 단위로 State를 받아 State를 반환합니다.

### 특징
- LangGraph 워크플로우의 실행 단위
- State를 받아 State를 반환하는 시그니처 필수
- Node 내부에서 Task를 호출하여 비즈니스 로직 실행
- State 관리 및 업데이트 담당

### 예시
```python
# Node 예시: execute_searches_parallel
def execute_searches_parallel(state: LegalWorkflowState) -> LegalWorkflowState:
    """검색 실행 Node - State를 받아 State를 반환"""
    # State에서 필요한 값 추출
    optimized_queries = state.get("optimized_queries", {})
    search_params = state.get("search_params", {})
    
    # Task 실행 (비즈니스 로직)
    semantic_results, semantic_count, keyword_results, keyword_count = (
        SearchExecutionTasks.execute_searches_sync(
            optimized_queries=optimized_queries,
            search_params=search_params,
            ...
        )
    )
    
    # State 업데이트
    state["semantic_results"] = semantic_results
    state["keyword_results"] = keyword_results
    
    return state
```

## 원칙

1. **Task는 State를 직접 다루지 않음**: Task는 순수 함수로 구현하여 재사용성을 높입니다.
2. **Node는 State 관리 담당**: Node는 State를 받아 Task를 호출하고 결과를 State에 반영합니다.
3. **명확한 책임 분리**: Task는 비즈니스 로직, Node는 워크플로우 관리에 집중합니다.

## 파일 구조

```
core/workflow/
├── nodes/              # Node 클래스들
│   ├── classification_nodes.py
│   ├── search_nodes.py
│   └── ...
├── tasks/              # Task 클래스들 (기존)
│   ├── search_execution_tasks.py
│   └── search_result_tasks.py
└── ...
```

## 사용 가이드

### Task 작성 시
- State를 파라미터로 받지 않음
- 필요한 값만 파라미터로 받음
- 순수 함수로 구현 (부작용 최소화)

### Node 작성 시
- State를 받아 State를 반환하는 시그니처 필수
- State에서 필요한 값 추출
- Task 호출하여 비즈니스 로직 실행
- 결과를 State에 반영

