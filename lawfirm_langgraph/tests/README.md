# LawFirm LangGraph 테스트

LangGraph 워크플로우 시스템을 위한 테스트 모듈

## 테스트 구조

```
tests/
├── __init__.py                # 패키지 초기화
├── conftest.py                # pytest 설정 및 공통 픽스처
│
├── unit/                      # 단위 테스트
│   ├── utils/                 # core/utils 테스트
│   │   ├── test_config.py
│   │   ├── test_logger.py
│   │   └── test_performance_optimizer.py
│   ├── data/                  # core/data 테스트
│   │   ├── test_database.py
│   │   ├── test_vector_store.py
│   │   └── test_conversation_store.py
│   ├── services/              # core/services 테스트
│   │   ├── test_chat_service.py
│   │   └── test_search_service.py
│   ├── models/                # core/models 테스트
│   │   └── test_sentence_bert.py
│   ├── classification/        # core/classification 테스트
│   │   └── test_classifiers.py
│   ├── search/                # core/search 테스트
│   │   ├── test_connectors.py
│   │   ├── test_handlers.py
│   │   └── test_query_expansion.py
│   ├── generation/            # core/generation 테스트
│   │   └── test_generators.py
│   ├── processing/            # core/processing 테스트
│   │   ├── test_extractors.py
│   │   └── test_parsers.py
│   └── agents/                # core/agents 테스트
│       ├── test_handlers.py
│       ├── test_extractors.py
│       └── test_parsers.py
│
├── integration/               # 통합 테스트
│   ├── test_workflow_service.py
│   ├── test_workflow_nodes.py
│   ├── test_integration.py
│   └── test_context_expansion.py
│
├── e2e/                       # End-to-End 테스트
│   ├── test_source_validator.py
│   ├── test_source_validator_enhanced.py
│   ├── test_source_formatting.py
│   ├── test_sources_detail.py
│   └── test_unified_source_formatter.py
│
├── config/                    # 설정 관련 테스트
│   └── test_config.py         # LangGraphConfig 테스트
│
├── scripts/                   # 테스트 실행 스크립트
│   ├── run_all_tests.py
│   ├── run_tests_manual.py
│   ├── run_single_query_test.py      # 기존 단일 질의 테스트 (레거시)
│   ├── run_single_query_test_stream.py
│   └── run_query_test.py              # 개선된 단일 질의 테스트 (권장)
│
└── fixtures/                   # 테스트 픽스처 및 데이터
    └── data/                   # 테스트 데이터 디렉토리
```

## 테스트 파일 설명

### 단위 테스트 (unit/)

#### unit/utils/
- `test_config.py`: Config 클래스 테스트 (core/utils/config.py)
- `test_logger.py`: Logger 유틸리티 테스트
- `test_performance_optimizer.py`: PerformanceMonitor, MemoryOptimizer, CacheManager 테스트

#### unit/data/
- `test_database.py`: DatabaseManager 클래스 테스트
- `test_vector_store.py`: LegalVectorStore 클래스 테스트
- `test_conversation_store.py`: ConversationStore 클래스 테스트

#### unit/services/
- `test_chat_service.py`: ChatService 클래스 테스트
- `test_search_service.py`: MLEnhancedSearchService 및 SearchService 테스트

#### unit/models/
- `test_sentence_bert.py`: Sentence-BERT 모델 테스트

#### unit/classification/
- `test_classifiers.py`: QuestionClassifier, RuleBasedClassifier, HybridQuestionClassifier 등 테스트

#### unit/search/
- `test_connectors.py`: 검색 커넥터 테스트
- `test_handlers.py`: 검색 핸들러 테스트
- `test_query_expansion.py`: 쿼리 확장 검색 테스트

#### unit/generation/
- `test_generators.py`: 답변 생성기 테스트

#### unit/processing/
- `test_extractors.py`: DocumentExtractor, QueryExtractor, ResponseExtractor 테스트
- `test_parsers.py`: ResponseParser, ClassificationParser 등 테스트

#### unit/agents/
- `test_handlers.py`: ClassificationHandler, SearchHandler, ContextBuilder 등 테스트
- `test_extractors.py`: Extractors 테스트
- `test_parsers.py`: Parsers 테스트

### 통합 테스트 (integration/)

- `test_workflow_service.py`: LangGraphWorkflowService 클래스 테스트
- `test_workflow_nodes.py`: 워크플로우 노드 및 핸들러 테스트
- `test_integration.py`: 전체 워크플로우 통합 테스트
- `test_context_expansion.py`: 컨텍스트 확장 개선 테스트

### E2E 테스트 (e2e/)

- `test_source_validator.py`: 소스 검증 테스트
- `test_source_validator_enhanced.py`: 향상된 소스 검증 테스트
- `test_source_formatting.py`: 소스 포맷팅 테스트
- `test_sources_detail.py`: 소스 상세 정보 테스트
- `test_unified_source_formatter.py`: 통합 소스 포맷터 테스트

### 설정 테스트 (config/)

- `test_config.py`: LangGraphConfig 클래스 테스트, CheckpointStorageType Enum 테스트

## 실행 방법

### 전체 테스트 실행

```bash
# 수동 테스트 실행 스크립트 (권장 - pytest 버퍼 문제 우회)
python lawfirm_langgraph/tests/scripts/run_tests_manual.py

# pytest 직접 사용
pytest lawfirm_langgraph/tests/ -v

# run_all_tests.py 사용 (pytest 기반)
python lawfirm_langgraph/tests/scripts/run_all_tests.py
```

### 단위 테스트만 실행

```bash
# 모든 단위 테스트
pytest lawfirm_langgraph/tests/unit/ -v

# 특정 모듈 테스트
pytest lawfirm_langgraph/tests/unit/utils/ -v
pytest lawfirm_langgraph/tests/unit/data/ -v
pytest lawfirm_langgraph/tests/unit/services/ -v
```

### 통합 테스트만 실행

```bash
pytest lawfirm_langgraph/tests/integration/ -v
```

### E2E 테스트만 실행

```bash
pytest lawfirm_langgraph/tests/e2e/ -v
```

### 단일 질의 테스트 실행

#### 개선된 버전 (권장)

```bash
# 기본 질의로 테스트 실행 (인덱스 0, 1, 2 중 선택)
python lawfirm_langgraph/tests/scripts/run_query_test.py 0

# 커스텀 질의로 테스트 실행
python lawfirm_langgraph/tests/scripts/run_query_test.py "계약서 작성 시 주의할 사항은 무엇인가요?"

# 환경 변수로 질의 전달 (PowerShell 인코딩 문제 회피)
$env:TEST_QUERY='민법 제750조 손해배상에 대해 설명해주세요'; python lawfirm_langgraph/tests/scripts/run_query_test.py

# 파일에서 질의 읽기
python lawfirm_langgraph/tests/scripts/run_query_test.py -f query.txt
```

**특징:**
- 순환 import 문제 해결
- Windows PowerShell 호환 로깅 (SafeStreamHandler)
- 간소화된 코드 구조
- LangGraph 최신 로직 반영

#### 기존 버전 (레거시)

```bash
# 기본 질의로 테스트 실행
python lawfirm_langgraph/tests/scripts/run_single_query_test.py

# 커스텀 질의로 테스트 실행
python lawfirm_langgraph/tests/scripts/run_single_query_test.py "계약서 작성 시 주의할 사항은 무엇인가요?"
```

### 특정 테스트 실행

```bash
# 특정 테스트 파일
pytest lawfirm_langgraph/tests/config/test_config.py -v

# 특정 테스트 클래스
pytest lawfirm_langgraph/tests/config/test_config.py::TestLangGraphConfig -v

# 특정 테스트 메서드
pytest lawfirm_langgraph/tests/config/test_config.py::TestLangGraphConfig::test_config_default_values -v
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

- `mock_config`: Mock LangGraphConfig - 테스트용 설정 객체
- `mock_workflow_state`: Mock 워크플로우 상태 - 기본 워크플로우 상태 딕셔너리
- `mock_llm_response`: Mock LLM 응답 - LLM 응답 모델
- `mock_search_results`: Mock 검색 결과 - 검색 결과 리스트
- `mock_workflow_service`: Mock LangGraphWorkflowService - 워크플로우 서비스 모의 객체
- `mock_legal_data_connector`: Mock LegalDataConnector - 법률 데이터 커넥터 모의 객체
- `mock_answer_generator`: Mock AnswerGenerator - 답변 생성기 모의 객체
- `cleanup_test_files`: 테스트 파일 정리 - 테스트 후 생성된 파일 정리
- `setup_test_environment`: 테스트 환경 설정 (autouse) - 자동으로 테스트 환경 변수 설정

## 테스트 작성 가이드

### 새로운 테스트 추가

1. 적절한 테스트 디렉토리 선택 (unit/, integration/, e2e/)
2. 해당 모듈 디렉토리에 테스트 파일 생성
3. 테스트 클래스 작성 (프로젝트 규칙에 따라)
4. `conftest.py`의 픽스처 활용
5. Mock 사용하여 외부 의존성 격리

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
5. **Import 경로**: 모든 테스트는 `lawfirm_langgraph.core.*` 형식의 import 경로 사용

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

### pytest 버퍼 문제

- `run_tests_manual.py` 사용 (권장)
- 또는 pytest 실행 시 `-s` 옵션 사용: `pytest lawfirm_langgraph/tests/ -v -s`

## 참고

- [pytest 문서](https://docs.pytest.org/)
- [pytest-asyncio 문서](https://pytest-asyncio.readthedocs.io/)
- [unittest.mock 문서](https://docs.python.org/3/library/unittest.mock.html)
