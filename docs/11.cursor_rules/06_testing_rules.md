# 테스트 규칙

## 1. 단위 테스트
```python
import pytest
from unittest.mock import Mock, patch
from lawfirm_langgraph.core.workflow.state.state_definitions import LegalWorkflowState

class TestWorkflowNode:
    """워크플로우 노드 테스트 클래스"""
    
    def setup_method(self):
        """테스트 설정"""
        self.workflow = EnhancedLegalQuestionWorkflow(config)
    
    def test_node_execution(self):
        """노드 실행 테스트"""
        state: LegalWorkflowState = {"query": "테스트 질문"}
        result_state = self.workflow.classify_query_and_complexity(state)
        assert "query_type" in result_state
    
    def test_empty_input(self):
        """빈 입력 처리 테스트"""
        state: LegalWorkflowState = {"query": ""}
        with pytest.raises(ValueError):
            self.workflow.classify_query_and_complexity(state)
```

## 2. 통합 테스트
```python
# lawfirm_langgraph/tests/integration/test_api.py
import pytest
from fastapi.testclient import TestClient
from api.main import app

client = TestClient(app)

def test_chat_endpoint():
    """채팅 엔드포인트 테스트"""
    response = client.post(
        "/api/chat",
        json={"message": "계약서 검토 요청"}
    )
    assert response.status_code == 200
    assert "response" in response.json()
```

