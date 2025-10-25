# LawFirmAI 테스트 가이드

## 개요

LawFirmAI 프로젝트의 테스트 시스템은 기능별로 체계적으로 구성되어 있으며, 다양한 테스트 카테고리로 분류되어 있습니다.

## 테스트 폴더 구조

```
tests/
├── conftest.py                    # pytest 설정 및 공통 fixtures
├── run_tests.py                   # 테스트 실행 스크립트
├── fixtures/                      # 테스트 데이터 및 공통 fixtures
│   ├── test_data.json
│   └── ...
├── unit/                          # 단위 테스트
│   ├── services/                  # 서비스 단위 테스트
│   ├── models/                    # 모델 단위 테스트
│   ├── data/                      # 데이터 처리 단위 테스트
│   └── utils/                     # 유틸리티 단위 테스트
├── integration/                   # 통합 테스트
├── performance/                   # 성능 테스트
├── quality/                       # 품질 테스트
├── memory/                        # 메모리 관련 테스트
├── classification/                # 분류 시스템 테스트
├── legal_systems/                # 법률 시스템 테스트
├── contracts/                    # 계약 관련 테스트
├── external_integrations/        # 외부 시스템 통합 테스트
│   ├── akls/                     # AKLS 관련 테스트
│   ├── langfuse/                 # Langfuse 통합 테스트
│   └── gradio/                   # Gradio 인터페이스 테스트
├── conversational/               # 대화 관련 테스트
├── database/                     # 데이터베이스 테스트
├── demos/                        # 데모 및 예제 테스트
└── regression/                   # 회귀 테스트
```

## 테스트 카테고리 설명

### 1. 단위 테스트 (unit/)
- **목적**: 개별 컴포넌트의 독립적인 기능 테스트
- **범위**: 서비스, 모델, 데이터 처리, 유틸리티 클래스
- **특징**: Mock 객체 사용, 빠른 실행, 격리된 테스트

### 2. 통합 테스트 (integration/)
- **목적**: 여러 컴포넌트 간의 상호작용 테스트
- **범위**: RAG 시스템, LangChain 통합, 전체 시스템 통합
- **특징**: 실제 데이터베이스 사용, 외부 의존성 포함

### 3. 성능 테스트 (performance/)
- **목적**: 시스템 성능 및 응답 시간 측정
- **범위**: 벤치마크, 스트레스 테스트, 메모리 관리
- **특징**: 성능 메트릭 수집, 부하 테스트

### 4. 품질 테스트 (quality/)
- **목적**: 답변 품질 및 대화 품질 검증
- **범위**: 답변 정확도, 품질 개선 워크플로우
- **특징**: 품질 메트릭 측정, 사용자 만족도 평가

### 5. 메모리 테스트 (memory/)
- **목적**: 대화 메모리 및 컨텍스트 관리 테스트
- **범위**: 대화 기록, 컨텍스트 압축, 메모리 품질
- **특징**: 장기 대화 시나리오, 메모리 효율성

### 6. 분류 시스템 테스트 (classification/)
- **목적**: 질의 분류 및 질문 유형 분류 테스트
- **범위**: 질의 분류기, 질문 유형 매핑
- **특징**: 분류 정확도 측정, 다양한 질의 유형

### 7. 법률 시스템 테스트 (legal_systems/)
- **목적**: 법률 관련 기능 테스트
- **범위**: 법적 근거 제시, 법률 검색, 법률 제한
- **특징**: 법률 정확성 검증, 판례 검색

### 8. 계약 관련 테스트 (contracts/)
- **목적**: 계약서 분석 및 검토 기능 테스트
- **범위**: 계약서 분석, 인터랙티브 계약 시스템
- **특징**: 계약서 템플릿, 법률 검토 프로세스

### 9. 외부 시스템 통합 테스트 (external_integrations/)
- **목적**: 외부 시스템과의 통합 테스트
- **범위**: AKLS, Langfuse, Gradio 인터페이스
- **특징**: API 통합, 외부 의존성 관리

### 10. 대화 관련 테스트 (conversational/)
- **목적**: 자연어 대화 및 컨텍스트 관리 테스트
- **범위**: 자연어 처리, 대화 흐름, 개인화
- **특징**: 대화 품질, 사용자 경험

### 11. 데이터베이스 테스트 (database/)
- **목적**: 데이터베이스 연산 및 템플릿 시스템 테스트
- **범위**: 데이터베이스 CRUD, 템플릿 시스템
- **특징**: 데이터 무결성, 트랜잭션 처리

### 12. 데모 및 예제 테스트 (demos/)
- **목적**: 시연용 테스트 및 예제 코드
- **범위**: 간단한 계약 테스트, 종합 데모
- **특징**: 사용자 시연, 기능 소개

### 13. 회귀 테스트 (regression/)
- **목적**: 시스템 안정성 및 회귀 테스트
- **범위**: 구조 수정, 시스템 안정성
- **특징**: 버그 재발 방지, 안정성 검증

## 테스트 실행 방법

### 1. 전체 테스트 실행
```bash
# 기본 실행
python tests/run_tests.py

# 상세 출력과 함께
python tests/run_tests.py -v

# 커버리지 측정과 함께
python tests/run_tests.py --coverage
```

### 2. 카테고리별 테스트 실행
```bash
# 단위 테스트만 실행
python tests/run_tests.py unit

# 통합 테스트만 실행
python tests/run_tests.py integration

# 성능 테스트만 실행
python tests/run_tests.py performance

# 품질 테스트만 실행
python tests/run_tests.py quality
```

### 3. 특정 테스트 파일 실행
```bash
# 특정 테스트 파일 실행
python tests/run_tests.py test_chat_service.py

# 특정 경로의 테스트 실행
python tests/run_tests.py unit/services/test_chat_service.py
```

### 4. 마커를 사용한 필터링
```bash
# 느린 테스트 제외
python tests/run_tests.py -m "not slow"

# 특정 마커만 실행
python tests/run_tests.py -m "unit"

# 여러 마커 조합
python tests/run_tests.py -m "unit and not slow"
```

### 5. 병렬 실행
```bash
# 병렬 실행 (pytest-xdist 필요)
python tests/run_tests.py --parallel
```

## 공통 Fixtures 사용

### 1. 기본 Fixtures
```python
def test_example(temp_db, sample_queries, mock_chat_service):
    """기본 fixtures 사용 예시"""
    # temp_db: 임시 데이터베이스 경로
    # sample_queries: 샘플 질의 데이터
    # mock_chat_service: Mock ChatService
    pass
```

### 2. Mock Fixtures
```python
def test_with_mocks(mock_rag_service, mock_database, mock_vector_store):
    """Mock fixtures 사용 예시"""
    # mock_rag_service: Mock RAG Service
    # mock_database: Mock Database
    # mock_vector_store: Mock Vector Store
    pass
```

### 3. 테스트 데이터 Fixtures
```python
def test_with_data(sample_legal_documents, test_config, performance_metrics):
    """테스트 데이터 fixtures 사용 예시"""
    # sample_legal_documents: 샘플 법률 문서
    # test_config: 테스트 설정
    # performance_metrics: 성능 메트릭
    pass
```

## 테스트 작성 가이드

### 1. 테스트 파일 명명 규칙
- 단위 테스트: `test_<component_name>.py`
- 통합 테스트: `test_<system_name>_integration.py`
- 성능 테스트: `test_<component_name>_performance.py`

### 2. 테스트 클래스 구조
```python
class TestComponentName(unittest.TestCase):
    """컴포넌트 테스트 클래스"""
    
    def setUp(self):
        """테스트 설정"""
        pass
    
    def tearDown(self):
        """테스트 정리"""
        pass
    
    def test_specific_functionality(self):
        """특정 기능 테스트"""
        pass
```

### 3. 테스트 함수 명명 규칙
- `test_<functionality>_<condition>_<expected_result>`
- 예: `test_process_message_valid_input_returns_response`

### 4. Assertion 사용
```python
# 기본 assertion
self.assertEqual(actual, expected)
self.assertNotEqual(actual, expected)
self.assertTrue(condition)
self.assertFalse(condition)

# 예외 테스트
with self.assertRaises(ValueError):
    function_that_raises_error()

# Mock 검증
mock_service.process_message.assert_called_once_with("test message")
```

## 성능 테스트 가이드

### 1. 응답 시간 측정
```python
import time

def test_response_time(self):
    """응답 시간 테스트"""
    start_time = time.time()
    result = service.process_request("test query")
    end_time = time.time()
    
    response_time = end_time - start_time
    self.assertLess(response_time, 2.0)  # 2초 이내
```

### 2. 메모리 사용량 측정
```python
import psutil
import os

def test_memory_usage(self):
    """메모리 사용량 테스트"""
    process = psutil.Process(os.getpid())
    initial_memory = process.memory_info().rss
    
    # 테스트 실행
    service.process_large_dataset()
    
    final_memory = process.memory_info().rss
    memory_increase = final_memory - initial_memory
    
    self.assertLess(memory_increase, 100 * 1024 * 1024)  # 100MB 이내
```

## 품질 테스트 가이드

### 1. 답변 품질 평가
```python
def test_answer_quality(self):
    """답변 품질 테스트"""
    query = "계약 해지에 대해 알려주세요"
    response = service.process_message(query)
    
    # 답변 길이 검증
    self.assertGreater(len(response), 50)
    
    # 키워드 포함 검증
    self.assertIn("계약", response)
    self.assertIn("해지", response)
    
    # 신뢰도 검증
    confidence = service.get_confidence_score(response)
    self.assertGreater(confidence, 0.8)
```

### 2. 대화 품질 평가
```python
def test_conversation_quality(self):
    """대화 품질 테스트"""
    conversation = [
        ("사용자", "계약 해지에 대해 알려주세요"),
        ("AI", "계약 해지는 민법 제543조에 따라..."),
        ("사용자", "손해배상 범위는 어떻게 되나요?"),
        ("AI", "손해배상 범위는 민법 제544조에 따라...")
    ]
    
    quality_score = service.evaluate_conversation_quality(conversation)
    self.assertGreater(quality_score, 0.85)
```

## 디버깅 및 문제 해결

### 1. 테스트 실패 디버깅
```bash
# 상세 출력으로 실행
python tests/run_tests.py -v

# 특정 테스트만 실행
python tests/run_tests.py tests/unit/services/test_chat_service.py::TestChatService::test_process_message

# 디버그 모드로 실행
python -m pytest tests/unit/services/test_chat_service.py -v -s
```

### 2. 로그 확인
```python
import logging

# 테스트에서 로그 확인
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

def test_with_logging(self):
    logger.debug("테스트 시작")
    # 테스트 로직
    logger.debug("테스트 완료")
```

### 3. 일반적인 문제 해결

#### Import 오류
```python
# 프로젝트 루트 경로 추가
import sys
from pathlib import Path
project_root = Path(__file__).parent.parent.parent
sys.path.append(str(project_root))
```

#### 데이터베이스 연결 오류
```python
# 임시 데이터베이스 사용
@pytest.fixture
def temp_db():
    temp_dir = tempfile.mkdtemp()
    db_path = os.path.join(temp_dir, "test.db")
    yield db_path
    if os.path.exists(db_path):
        os.remove(db_path)
```

#### Mock 설정 오류
```python
# 올바른 Mock 설정
from unittest.mock import Mock, patch

@patch('source.services.chat_service.ChatService')
def test_with_mock(self, mock_service):
    mock_service.return_value.process_message.return_value = "test response"
    # 테스트 로직
```

## CI/CD 통합

### 1. GitHub Actions 예시
```yaml
name: Tests
on: [push, pull_request]

jobs:
  test:
    runs-on: ubuntu-latest
    strategy:
      matrix:
        python-version: [3.8, 3.9, 3.10]
    
    steps:
    - uses: actions/checkout@v2
    - name: Set up Python ${{ matrix.python-version }}
      uses: actions/setup-python@v2
      with:
        python-version: ${{ matrix.python-version }}
    
    - name: Install dependencies
      run: |
        pip install -r requirements.txt
        pip install pytest pytest-cov pytest-xdist
    
    - name: Run tests
      run: |
        python tests/run_tests.py --coverage
```

### 2. 테스트 리포트 생성
```bash
# HTML 리포트 생성
python tests/run_tests.py --coverage
# 리포트는 htmlcov/ 폴더에 생성됨

# JUnit XML 리포트 생성
python -m pytest tests/ --junitxml=test-results.xml
```

## 모범 사례

### 1. 테스트 작성 원칙
- **AAA 패턴**: Arrange, Act, Assert
- **단일 책임**: 하나의 테스트는 하나의 기능만 테스트
- **명확한 이름**: 테스트 이름만으로도 무엇을 테스트하는지 알 수 있어야 함
- **독립성**: 테스트 간 의존성 없이 독립적으로 실행 가능

### 2. 성능 고려사항
- 느린 테스트는 `@pytest.mark.slow` 마커 사용
- 대용량 데이터 테스트는 별도 마커 사용
- 병렬 실행 가능한 테스트 작성

### 3. 유지보수성
- 공통 로직은 fixtures로 분리
- 테스트 데이터는 별도 파일로 관리
- Mock 객체는 적절히 사용하여 외부 의존성 제거

이 가이드를 따라 테스트를 작성하고 실행하면 LawFirmAI 프로젝트의 품질과 안정성을 보장할 수 있습니다.
