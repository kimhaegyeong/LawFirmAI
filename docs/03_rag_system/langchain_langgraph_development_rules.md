# LangChain & LangGraph 개발 규칙

## 개요

LawFirmAI 프로젝트에서 LangGraph 워크플로우 개발을 위한 핵심 규칙과 가이드라인을 정의합니다.

## 핵심 원칙

### 1. 아키텍처 원칙
- **상태 기반 워크플로우**: LangGraph를 통한 복잡한 법률 질문 처리
- **State 최적화**: 메모리 효율을 위한 자동 요약 및 정제
- **모듈화된 컴포넌트**: 재사용 가능한 서비스 설계
- **에러 처리**: 포괄적인 예외 처리 및 로깅

### 2. 개발 원칙
- **타입 안전성**: TypedDict를 활용한 상태 정의
- **성능 최적화**: State 크기 제한 및 Pruning
- **테스트 가능성**: 단위 테스트 및 통합 테스트 지원
- **확장성**: 새로운 노드와 기능 쉽게 추가 가능

## State 정의 규칙

### LegalWorkflowState

```python
from typing import TypedDict, Annotated, List, Dict, Any, Optional

class LegalWorkflowState(TypedDict):
    """법률 워크플로우 상태 정의"""
    
    # 입력 데이터
    query: str
    session_id: str
    
    # 질문 분류
    query_type: str
    confidence: float
    legal_field: str
    legal_domain: str
    
    # 긴급도 평가
    urgency_level: str
    urgency_reasoning: str
    emergency_type: Optional[str]
    
    # 멀티턴 처리
    is_multi_turn: bool
    conversation_history: List[Dict[str, Any]]
    
    # 검색 결과
    retrieved_docs: List[Dict[str, Any]]
    
    # 최종 답변
    answer: str
    sources: List[str]
    legal_references: List[str]
    
    # 처리 과정
    processing_steps: Annotated[List[str], add]
    errors: Annotated[List[str], add]
    metadata: Dict[str, Any]
    
    # 성능
    processing_time: float
```

### State 최적화 설정

```python
# state_utils.py
MAX_RETRIEVED_DOCS = 10              # 최대 검색 결과 수
MAX_DOCUMENT_CONTENT_LENGTH = 500    # 문서 content 최대 길이
MAX_CONVERSATION_HISTORY = 5        # 대화 이력 최대 개수
MAX_PROCESSING_STEPS = 20            # 처리 단계 최대 개수
```

## 워크플로우 구현 규칙

### 노드 함수 구현

```python
@observe(name="node_name")
def node_function(self, state: LegalWorkflowState) -> LegalWorkflowState:
    """
    노드 함수 구현
    
    Args:
        state: 현재 워크플로우 상태
        
    Returns:
        LegalWorkflowState: 업데이트된 상태
    """
    try:
        # 1. 로깅 시작
        self.logger.info(f"{state['query'][:50]}...")
        
        # 2. 비즈니스 로직
        result = self._perform_logic(state)
        state["result"] = result
        
        # 3. State 최적화 (필요시)
        state["processing_steps"] = prune_processing_steps(
            state["processing_steps"],
            max_items=MAX_PROCESSING_STEPS
        )
        
        # 4. 처리 단계 기록
        self._add_step(state, "노드 완료", result)
        
        return state
        
    except Exception as e:
        self._handle_error(state, str(e), "노드 오류")
        return state
```

### 그래프 구성

```python
def _build_graph(self):
    """워크플로우 그래프 구성"""
    
    # 노드 추가
    self.graph.add_node("classify_query", self.classify_query)
    self.graph.add_node("assess_urgency", self.assess_urgency)
    self.graph.add_node("retrieve_documents", self.retrieve_documents)
    
    # 엣지 정의
    self.graph.set_entry_point("classify_query")
    self.graph.add_edge("classify_query", "assess_urgency")
    self.graph.add_edge("assess_urgency", "retrieve_documents")
    self.graph.add_edge("retrieve_documents", END)
```

## State 최적화 규칙

### 문서 요약 (Document Pruning)

```python
from .state_utils import summarize_document, prune_retrieved_docs

# 문서 요약
summarized_doc = summarize_document(
    doc,
    max_content_length=MAX_DOCUMENT_CONTENT_LENGTH
)

# 검색 결과 정제
state["retrieved_docs"] = prune_retrieved_docs(
    state["retrieved_docs"],
    max_items=MAX_RETRIEVED_DOCS,
    max_content_per_doc=MAX_DOCUMENT_CONTENT_LENGTH
)
```

### 처리 단계 정제

```python
from .state_utils import prune_processing_steps

# 처리 단계 제한
state["processing_steps"] = prune_processing_steps(
    state["processing_steps"],
    max_items=MAX_PROCESSING_STEPS
)
```

## 에러 처리 규칙

### 에러 처리 패턴

```python
def node_function(self, state: LegalWorkflowState) -> LegalWorkflowState:
    try:
        # 비즈니스 로직
        result = self._perform_logic(state)
        
        # 성공 응답
        state["result"] = result
        self._add_step(state, "성공", "처리 완료")
        
    except ValueError as e:
        # 비즈니스 로직 오류
        self.logger.warning(f"비즈니스 로직 오류: {e}")
        self._handle_error(state, str(e), "처리 오류")
        state["result"] = None
        
    except Exception as e:
        # 시스템 오류
        self.logger.error(f"시스템 오류: {e}")
        self._handle_error(state, str(e), "시스템 오류")
        state["result"] = None
    
    return state
```

### 에러 로깅

```python
def _handle_error(
    self, 
    state: LegalWorkflowState, 
    error: str, 
    context: str
):
    """에러 처리 및 로깅"""
    error_msg = f"{context}: {error}"
    state["errors"].append(error_msg)
    self.logger.error(error_msg)
```

## 설정 관리

### LangGraphConfig

```python
@dataclass
class LangGraphConfig:
    """LangGraph 설정"""
    
    # LLM 설정
    llm_provider: str = "google"
    google_api_key: str = ""
    google_model: str = "gemini-2.5-flash-lite"
    
    # 워크플로우 설정
    recursion_limit: int = 100
    enable_statistics: bool = True
    
    # State 최적화
    max_retrieved_docs: int = 10
    max_document_content_length: int = 500
    max_conversation_history: int = 5
    max_processing_steps: int = 20
    enable_state_pruning: bool = True
```

### 환경 변수 설정

```bash
# .env
LLM_PROVIDER=google
GOOGLE_API_KEY=your_api_key
GOOGLE_MODEL=gemini-2.5-flash-lite

RECURSION_LIMIT=100

MAX_RETRIEVED_DOCS=10
MAX_DOCUMENT_CONTENT_LENGTH=500
MAX_CONVERSATION_HISTORY=5
MAX_PROCESSING_STEPS=20
```

## 로깅 규칙

### 로깅 설정

```python
import logging

logger = logging.getLogger(__name__)

class WorkflowNode:
    def __init__(self):
        self.logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}")
    
    def execute(self, state):
        # 성공 로깅
        self.logger.info(f"노드 실행 시작: {state['query'][:50]}...")
        
        # 오류 로깅
        try:
            result = self._process(state)
            self.logger.info(f"노드 실행 완료")
            return result
        except Exception as e:
            self.logger.error(f"노드 실행 실패: {e}")
            raise
```

## 테스트 규칙

### 단위 테스트

```python
import pytest
from source.services.langgraph.workflow_service import LangGraphWorkflowService

@pytest.mark.asyncio
async def test_workflow_success():
    """워크플로우 성공 테스트"""
    service = LangGraphWorkflowService()
    result = await service.process_query("테스트 질문")
    
    assert "answer" in result
    assert result["confidence"] > 0
    assert len(result["processing_steps"]) > 0
```

### 통합 테스트

```python
@pytest.mark.asyncio
async def test_multi_turn_conversation():
    """멀티턴 대화 테스트"""
    service = LangGraphWorkflowService()
    session_id = "test-session"
    
    # 첫 번째 질문
    result1 = await service.process_query(
        "이혼 절차는?",
        session_id=session_id
    )
    
    # 두 번째 질문 (대명사 포함)
    result2 = await service.process_query(
        "그것에 대해 더 자세히",
        session_id=session_id
    )
    
    assert result1["session_id"] == session_id
    assert result2["session_id"] == session_id
    assert result2.get("is_multi_turn", False)
```

## 성능 최적화 규칙

### 메모리 관리

```python
# State 크기 제한
def prune_state(state: LegalWorkflowState) -> LegalWorkflowState:
    """State 크기 제한"""
    
    # 검색 결과 정제
    state["retrieved_docs"] = prune_retrieved_docs(
        state["retrieved_docs"],
        max_items=MAX_RETRIEVED_DOCS
    )
    
    # 처리 단계 정제
    state["processing_steps"] = prune_processing_steps(
        state["processing_steps"],
        max_items=MAX_PROCESSING_STEPS
    )
    
    return state
```

### 캐싱 전략

```python
# LLM 응답 캐싱
from functools import lru_cache

@lru_cache(maxsize=128)
def cached_llm_invoke(prompt: str) -> str:
    """LLM 응답 캐싱"""
    return llm.invoke(prompt)
```

## 체크리스트

### 워크플로우 노드 체크리스트
- [ ] State 최적화가 적용되었는가?
- [ ] 에러 처리가 완전한가?
- [ ] 로깅이 적절한가?
- [ ] 테스트 코드가 작성되었는가?

### State 관리 체크리스트
- [ ] retrieved_docs가 최대값을 초과하지 않는가?
- [ ] document content가 최대 길이를 초과하지 않는가?
- [ ] processing_steps가 최대값을 초과하지 않는가?

### 성능 체크리스트
- [ ] State 크기가 적절한가?
- [ ] Pruning이 제대로 작동하는가?
- [ ] 메모리 사용량이 적절한가?

이 규칙들을 통해 LawFirmAI의 LangGraph 워크플로우를 일관성 있고 효율적으로 개발할 수 있습니다.
