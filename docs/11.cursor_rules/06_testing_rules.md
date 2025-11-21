# 테스트 규칙

## 0. pytest 실행 규칙 (CRITICAL)

**Windows 환경에서 pytest 실행 시 반드시 다음 옵션을 사용합니다:**

```bash
pytest -s --capture=tee-sys
```

또는 특정 테스트 파일/클래스/메서드 실행:

```bash
pytest -s --capture=tee-sys tests/unit/services/test_unified_prompt_manager.py
pytest -s --capture=tee-sys tests/unit/services/test_unified_prompt_manager.py::TestCleanContent
pytest -s --capture=tee-sys tests/unit/services/test_unified_prompt_manager.py::TestCleanContent::test_clean_content_removes_json_metadata
```

### 옵션 설명
- `-s`: 출력 캡처를 비활성화하여 print 문과 로그가 즉시 표시됩니다
- `--capture=tee-sys`: 출력을 캡처하면서 동시에 터미널에도 표시합니다 (Windows 버퍼 문제 해결)

### Windows 환경에서의 문제
Windows 환경에서 pytest를 실행할 때 `ValueError: underlying buffer has been detached` 오류가 발생할 수 있습니다. 이는 pytest의 출력 캡처 메커니즘과 Windows의 버퍼 처리 방식 간의 호환성 문제입니다.

### 해결 방법
1. **권장 방법**: `-s --capture=tee-sys` 옵션 사용
2. **대안**: `--capture=no` 옵션 사용 (출력 캡처 완전 비활성화)

### 예시
```bash
# 전체 테스트 실행
cd lawfirm_langgraph
pytest -s --capture=tee-sys

# 특정 디렉토리 테스트 실행
pytest -s --capture=tee-sys tests/unit/services/

# 특정 테스트 파일 실행
pytest -s --capture=tee-sys tests/unit/services/test_unified_prompt_manager.py -v

# 특정 테스트 클래스 실행
pytest -s --capture=tee-sys tests/unit/services/test_unified_prompt_manager.py::TestCleanContent -v
```

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

