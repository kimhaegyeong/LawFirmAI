# -*- coding: utf-8 -*-
"""
LangGraph State Reduction 마이그레이션 예제
기존 코드에 최소한의 수정으로 State Reduction을 적용하는 방법

사용 방법:
1. 기존 노드 함수에 데코레이터 추가
2. 또는 state_adapter를 사용하여 자동 변환
"""

from typing import Any, Dict

from .node_input_output_spec import get_node_spec
from .state_adapter import adapt_state, validate_state_for_node

# State Reduction 및 Adapter import
from .state_reduction import (
    reduce_state_for_node,
    reduce_state_size,
    with_state_reduction,
)

# ============================================
# 방법 1: 데코레이터 사용 (권장)
# ============================================

@with_state_reduction("classify_query")
def classify_query_example(state: Dict[str, Any]) -> Dict[str, Any]:
    """
    예제: classify_query 노드
    데코레이터를 사용하면 자동으로 필요한 데이터만 전달됨
    """
    # 필요한 데이터만 포함된 state 사용
    query = state.get("query") or state.get("input", {}).get("query")

    # 기존 로직...
    query_type = "general_question"
    confidence = 0.85

    # 결과 반환
    return {
        "query_type": query_type,
        "confidence": confidence
    }


# ============================================
# 방법 2: 수동 축소 (더 많은 제어)
# ============================================

def retrieve_documents_example(state: Dict[str, Any]) -> Dict[str, Any]:
    """
    예제: retrieve_documents 노드
    필요한 만큼만 수동으로 축소
    """
    # 노드별 필요한 데이터만 추출
    reduced = reduce_state_for_node(state, "retrieve_documents")

    # 기존 로직 계속 사용 (호환성 유지)
    query = reduced.get("query") or reduced.get("input", {}).get("query")
    search_query = reduced.get("search_query") or query

    # 검색 로직...
    retrieved_docs = []  # 실제 검색 결과

    # 결과 반환
    return {
        "retrieved_docs": retrieved_docs
    }


# ============================================
# 방법 3: State 검증 및 자동 변환
# ============================================

def generate_answer_example(state: Dict[str, Any]) -> Dict[str, Any]:
    """
    예제: generate_answer_enhanced 노드
    검증 및 자동 변환 사용
    """
    # State 검증 및 변환
    is_valid, error, converted = validate_state_for_node(
        state,
        "generate_answer_enhanced"
    )

    if not is_valid:
        # 에러 처리
        return {
            "answer": "",
            "errors": [error]
        }

    # 변환된 state 사용
    query = converted.get("query") or converted.get("input", {}).get("query")
    retrieved_docs = converted.get("retrieved_docs", [])

    # 답변 생성 로직...
    answer = "답변 내용"

    return {
        "answer": answer,
        "sources": ["source1", "source2"]
    }


# ============================================
# 방법 4: 기존 코드 수정 없이 적용 (최소 침투)
# ============================================

def apply_state_reduction_wrapper(node_func, node_name: str):
    """
    기존 노드 함수를 State Reduction으로 래핑

    Usage:
        original_node = EnhancedLegalQuestionWorkflow.classify_query
        wrapped_node = apply_state_reduction_wrapper(original_node, "classify_query")
    """
    def wrapper(state: Dict[str, Any], **kwargs) -> Dict[str, Any]:
        # State 축소
        reduced = reduce_state_for_node(state, node_name)

        # 원본 함수 호출 (축소된 state 전달)
        result = node_func(reduced, **kwargs)

        # 결과를 원본 state에 병합
        if isinstance(state, dict) and isinstance(result, dict):
            state.update(result)
            return state

        return result

    return wrapper


# ============================================
# 방법 5: 클래스 메서드에 적용
# ============================================

class EnhancedLegalWorkflowExample:
    """
    LangGraph 워크플로우 예제 클래스
    State Reduction을 적용하는 방법
    """

    @with_state_reduction("classify_query")
    def classify_query(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """질문 분류"""
        # 필요한 데이터만 자동으로 전달됨
        query = state.get("query") or state.get("input", {}).get("query")

        # 분류 로직...
        return {
            "query_type": "general_question",
            "confidence": 0.85
        }

    @with_state_reduction("retrieve_documents")
    def retrieve_documents(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """문서 검색"""
        # 축소된 state 사용
        query = state.get("query") or state.get("input", {}).get("query")

        # 검색 로직...
        return {
            "retrieved_docs": []
        }

    @with_state_reduction("generate_answer_enhanced")
    def generate_answer_enhanced(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """답변 생성"""
        # 필요한 데이터만 포함된 state
        query = state.get("query") or state.get("input", {}).get("query")
        retrieved_docs = state.get("retrieved_docs", [])

        # 답변 생성 로직...
        return {
            "answer": "답변 내용"
        }


# ============================================
# 마이그레이션 가이드
# ============================================

def get_migration_guide():
    """
    마이그레이션 가이드 반환
    """
    return """
# LangGraph State Reduction 마이그레이션 가이드

## 단계별 마이그레이션

### 1단계: 테스트 (변경 없음)
- 현재 코드 그대로 작동
- State Reduction 적용 안 함

### 2단계: 점진적 적용 (권장)
```python
from core.agents.state_reduction import with_state_reduction

# 기존 코드에 데코레이터만 추가
@with_state_reduction("retrieve_documents")
def retrieve_documents(state):
    # 기존 로직 유지
    query = state.get("query")
    ...
```

### 3단계: 최적화
```python
# 노드별 필요한 그룹 확인
from core.agents.node_input_output_spec import get_required_state_groups

groups = get_required_state_groups("retrieve_documents")
# → {"input", "search", "classification"}

# 실제로 그룹만 사용하도록 코드 수정
def retrieve_documents(state):
    query = state["input"]["query"]  # 명시적 접근
    docs = state["search"]["retrieved_docs"]
    ...
```

## 주의사항

1. **기존 코드와 호환**: Flat 구조 지원
2. **점진적 마이그레이션**: 한 번에 다 바꿀 필요 없음
3. **테스트 강화**: 각 노드별 테스트 작성
4. **성능 모니터링**: 실제 개선 효과 측정

## 성능 측정

```python
from core.agents.state_reduction import StateReducer

reducer = StateReducer()
reduced = reducer.reduce_state_for_node(full_state, "retrieve_documents")

# 메모리 사용량 비교
before_size = estimate_memory_usage(full_state)
after_size = estimate_memory_usage(reduced)
print(f"메모리 감소: {before_size - after_size} bytes")
```
"""


# ============================================
# 마이그레이션 체크리스트
# ============================================

MIGRATION_CHECKLIST = {
    "preparation": [
        "State Reduction 시스템 구현 확인",
        "테스트 코드 작성",
        "성능 벤치마크 준비"
    ],
    "implementation": [
        "각 노드에 데코레이터 추가",
        "테스트 및 검증",
        "성능 모니터링"
    ],
    "optimization": [
        "불필요한 데이터 접근 제거",
        "State 그룹 명시적 사용",
        "최종 성능 테스트"
    ]
}


def print_migration_checklist():
    """마이그레이션 체크리스트 출력"""
    for phase, items in MIGRATION_CHECKLIST.items():
        print(f"\n{phase.upper()}:")
        for item in items:
            print(f"  - {item}")


if __name__ == "__main__":
    print("LangGraph State Reduction 마이그레이션 예제")
    print("=" * 80)
    print(get_migration_guide())
    print_migration_checklist()
