# LawFirm LangGraph 테스트

LangGraph 워크플로우 시스템을 위한 테스트 모듈

## 테스트 구조

```
tests/
├── __init__.py              # 패키지 초기화
├── conftest.py              # pytest 설정 및 공통 픽스처
├── test_config.py           # LangGraphConfig 테스트
├── test_workflow_service.py  # LangGraphWorkflowService 테스트
├── test_workflow_nodes.py   # 워크플로우 노드 및 핸들러 테스트
├── test_integration.py      # 전체 워크플로우 통합 테스트
├── run_all_tests.py         # pytest 기반 전체 테스트 실행 스크립트
├── run_tests_manual.py      # 수동 테스트 실행 스크립트 (권장)
├── README.md                 # 테스트 가이드 문서
└── TEST_RESULTS.md          # 테스트 결과 문서
```

## 테스트 파일 설명

### test_config.py
- `LangGraphConfig` 클래스 테스트
- 설정 값 검증
- 환경 변수 로딩 테스트
- 설정 유효성 검사 테스트

### test_workflow_service.py
- `LangGraphWorkflowService` 클래스 테스트
- 서비스 초기화 테스트
- 비동기 쿼리 처리 테스트
- 에러 핸들링 테스트
- 워크플로우 테스트 메서드 테스트

### test_workflow_nodes.py
- 워크플로우 노드 테스트
- 핸들러 테스트 (Classification, Search, ContextBuilder, AnswerGenerator 등)
- 상태 관리 테스트
- 워크플로우 라우팅 테스트
- 에러 핸들링 테스트

### test_integration.py
- 전체 워크플로우 통합 테스트
- 간단한 쿼리 워크플로우 테스트
- 복잡한 쿼리 워크플로우 테스트
- 멀티턴 대화 테스트
- 에러 복구 워크플로우 테스트
- Agentic 모드 테스트
- 성능 테스트

## 실행 방법

### 전체 테스트 실행

```bash
# 수동 테스트 실행 스크립트 (권장 - pytest 버퍼 문제 우회)
python lawfirm_langgraph/tests/run_tests_manual.py

# pytest 직접 사용 (버퍼 문제가 해결된 경우)
pytest lawfirm_langgraph/tests/ -v

# run_all_tests.py 사용 (pytest 기반, 버퍼 문제 발생 가능)
python lawfirm_langgraph/tests/run_all_tests.py
```

### 특정 테스트 실행

```bash
# 특정 테스트 파일
pytest lawfirm_langgraph/tests/test_config.py -v

# 특정 테스트 클래스
pytest lawfirm_langgraph/tests/test_config.py::TestLangGraphConfig -v

# 특정 테스트 메서드
pytest lawfirm_langgraph/tests/test_config.py::TestLangGraphConfig::test_config_default_values -v
```

### 커버리지 리포트

```bash
# 커버리지 리포트 생성
pytest lawfirm_langgraph/tests/ --cov=lawfirm_langgraph --cov-report=html

# HTML 리포트 확인
# htmlcov/index.html 파일을 브라우저에서 열기
```

## 테스트 픽스처

`conftest.py`에 정의된 주요 픽스처:

- `mock_config`: Mock LangGraphConfig
- `mock_workflow_state`: Mock 워크플로우 상태
- `mock_llm_response`: Mock LLM 응답
- `mock_search_results`: Mock 검색 결과
- `mock_workflow_service`: Mock LangGraphWorkflowService
- `mock_legal_data_connector`: Mock LegalDataConnector
- `mock_answer_generator`: Mock AnswerGenerator
- `cleanup_test_files`: 테스트 파일 정리
- `setup_test_environment`: 테스트 환경 설정

## 테스트 작성 가이드

### 새로운 테스트 추가

1. 적절한 테스트 파일 선택 또는 새 파일 생성
2. 테스트 클래스 작성 (프로젝트 규칙에 따라)
3. `conftest.py`의 픽스처 활용
4. Mock 사용하여 외부 의존성 격리

### 예시

```python
import pytest
from unittest.mock import Mock, MagicMock, patch

class TestNewFeature:
    """새 기능 테스트"""
    
    @pytest.fixture
    def mock_component(self):
        """Mock 컴포넌트"""
        return MagicMock()
    
    def test_feature_basic(self, mock_component):
        """기본 기능 테스트"""
        mock_component.method.return_value = "result"
        
        result = mock_component.method()
        
        assert result == "result"
        mock_component.method.assert_called_once()
```

## 주의사항

1. **외부 의존성**: 모든 외부 의존성(LLM, 데이터베이스 등)은 Mock으로 처리
2. **비동기 테스트**: `@pytest.mark.asyncio` 데코레이터 사용
3. **환경 변수**: 테스트 환경은 자동으로 설정됨 (`conftest.py`의 `setup_test_environment`)
4. **파일 정리**: 테스트 생성 파일은 `cleanup_test_files` 픽스처로 정리

## 문제 해결

### Import 오류

```bash
# 프로젝트 루트에서 실행
cd /path/to/LawFirmAI
export PYTHONPATH="${PYTHONPATH}:$(pwd)"
pytest lawfirm_langgraph/tests/
```

### 비동기 테스트 오류

```bash
# pytest-asyncio 설치 확인
pip install pytest-asyncio
```

### Mock 관련 오류

- Mock 패치 경로 확인
- 실제 import 경로와 Mock 경로 일치 확인

## 참고

- [pytest 문서](https://docs.pytest.org/)
- [pytest-asyncio 문서](https://pytest-asyncio.readthedocs.io/)
- [unittest.mock 문서](https://docs.python.org/3/library/unittest.mock.html)

