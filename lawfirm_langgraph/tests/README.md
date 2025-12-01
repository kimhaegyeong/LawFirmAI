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
│   ├── agents/                # core/agents 테스트
│   │   ├── test_handlers.py
│   │   ├── test_extractors.py
│   │   └── test_parsers.py
│   └── workflow/              # core/workflow 테스트 (리팩토링된 구조)
│       ├── test_classification_nodes.py
│       ├── test_search_nodes.py
│       ├── test_document_nodes.py
│       ├── test_answer_nodes.py
│       ├── test_agentic_nodes.py
│       ├── test_ethical_rejection_node.py
│       ├── test_subgraphs.py
│       ├── test_edges.py
│       └── test_registry.py
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
│
│   ├── test_processing_extractors.py
│   ├── test_processing_quality_validators.py
│   ├── test_processing_reasoning_extractor.py
│   ├── test_processing_response_parsers.py
│   ├── test_prompt_builders.py
│   ├── test_state_definitions.py
│   ├── test_state_modular_states.py
│   ├── test_state_utils.py
│   ├── test_tools_legal_search_tools.py
│   ├── test_utils_workflow_constants.py
│   ├── test_utils_workflow_routes.py
│   ├── test_workflow_legal_workflow_enhanced.py
│   ├── test_workflow_service.py
│   └── test_workflow_utils.py
│
├── config/                    # 설정 관련 테스트
│   ├── test_config.py         # LangGraphConfig 테스트
│   └── test_app_config.py     # AppConfig 테스트
│
├── scripts/                   # 테스트 실행 스크립트
│   ├── run_query_test.py      # 메인 쿼리 테스트 스크립트 (권장)
│   └── ...                    # 기타 테스트 스크립트들
│
├── run_coverage.py            # 커버리지 측정 스크립트 (Windows 호환)
├── run_tests_manual.py        # 수동 테스트 실행 스크립트
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

#### unit/workflow/ (리팩토링된 구조)
- `test_classification_nodes.py`: ClassificationNodes 클래스 테스트
- `test_search_nodes.py`: SearchNodes 클래스 테스트
- `test_document_nodes.py`: DocumentNodes 클래스 테스트
- `test_answer_nodes.py`: AnswerNodes 클래스 테스트
- `test_agentic_nodes.py`: AgenticNodes 클래스 테스트
- `test_ethical_rejection_node.py`: EthicalRejectionNode 클래스 테스트
- `test_subgraphs.py`: 서브그래프 (ClassificationSubgraph, SearchSubgraph 등) 테스트
- `test_edges.py`: 엣지 빌더 (ClassificationEdges, SearchEdges 등) 테스트
- `test_registry.py`: 레지스트리 패턴 (NodeRegistry, SubgraphRegistry) 테스트

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
- `test_app_config.py`: AppConfig 클래스 테스트

## 실행 방법

### 전체 테스트 실행

```bash
# pytest 직접 사용 (권장)
pytest lawfirm_langgraph/tests/ -v -s --capture=no

# Windows 환경에서 버퍼 이슈가 있는 경우
$env:PYTHONUNBUFFERED=1; pytest lawfirm_langgraph/tests/ -v -s --capture=no

# 수동 테스트 실행 스크립트 (pytest 버퍼 문제 우회)
python lawfirm_langgraph/tests/run_tests_manual.py
```

**Windows 환경 개선 사항:**
- `-s` 옵션과 `--capture=no` 옵션으로 출력 버퍼링 비활성화
- `PYTHONUNBUFFERED=1` 환경 변수 설정으로 Python 출력 버퍼링 비활성화
- `run_coverage.py`는 Windows 버퍼 이슈를 해결하기 위해 개선되었습니다

### 단위 테스트만 실행

```bash
# 모든 단위 테스트
pytest lawfirm_langgraph/tests/unit/ -v

# 특정 모듈 테스트
pytest lawfirm_langgraph/tests/unit/utils/ -v
pytest lawfirm_langgraph/tests/unit/data/ -v
pytest lawfirm_langgraph/tests/unit/services/ -v

# 워크플로우 노드 테스트 (리팩토링된 구조)
pytest lawfirm_langgraph/tests/unit/workflow/ -v
pytest lawfirm_langgraph/tests/unit/workflow/test_classification_nodes.py -v
pytest lawfirm_langgraph/tests/unit/workflow/test_search_nodes.py -v
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
# run_coverage.py 사용 (권장 - Windows 호환)
python lawfirm_langgraph/tests/run_coverage.py

# pytest 직접 사용
pytest lawfirm_langgraph/tests/ --cov=lawfirm_langgraph --cov-report=html --cov-report=term-missing

# HTML 리포트 확인
# lawfirm_langgraph/htmlcov/index.html 파일을 브라우저에서 열기
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
- `mock_workflow_instance`: Mock 워크플로우 인스턴스 (노드 클래스용)
- `classification_nodes`: ClassificationNodes 인스턴스
- `search_nodes`: SearchNodes 인스턴스
- `document_nodes`: DocumentNodes 인스턴스
- `answer_nodes`: AnswerNodes 인스턴스
- `agentic_nodes`: AgenticNodes 인스턴스
- `ethical_rejection_node`: EthicalRejectionNode 인스턴스
- `node_registry`: NodeRegistry 인스턴스
- `subgraph_registry`: SubgraphRegistry 인스턴스

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
5. **Import 경로**: 
   - Core 모듈: `lawfirm_langgraph.core.*` 형식
   - LangGraph Core 모듈: `lawfirm_langgraph.langgraph_core.*` 형식
   - Generation 모듈: `lawfirm_langgraph.core.generation.validators.*`, `lawfirm_langgraph.core.generation.formatters.*` 형식
6. **Windows 환경**: pytest 직접 사용 시 `-s --capture=no` 옵션과 `PYTHONUNBUFFERED=1` 환경 변수 사용 권장 (버퍼 이슈 해결)

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

### pytest 버퍼 문제 (Windows)

- pytest 직접 사용 (권장) - `-s --capture=no` 옵션과 `PYTHONUNBUFFERED=1` 환경 변수로 Windows 버퍼 이슈 해결
- `run_coverage.py` 사용 - 커버리지 측정 시 Windows 호환
- 또는 pytest 실행 시 `-s --capture=no` 옵션 사용: `pytest lawfirm_langgraph/tests/ -v -s --capture=no`

### Import 경로 변경 사항

일부 모듈의 경로가 변경되었습니다:

- `core.services.source_validator` → `core.generation.validators.source_validator`
- `core.services.unified_source_formatter` → `core.generation.formatters.unified_source_formatter`
- `DocumentProcessor` → `LegalDocumentProcessor` (core/processing/processors/)

## 참고

- [pytest 문서](https://docs.pytest.org/)
- [pytest-asyncio 문서](https://pytest-asyncio.readthedocs.io/)
- [unittest.mock 문서](https://docs.python.org/3/library/unittest.mock.html)
