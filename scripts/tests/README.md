# Scripts Tests

LawFirmAI 프로젝트의 스크립트 테스트 디렉토리입니다.

## 디렉토리 구조

```
scripts/tests/
├── unit/                    # 단위 테스트
│   ├── test_utils_*.py      # 유틸리티 모듈 테스트
│   ├── test_faiss_version_manager.py
│   ├── test_migration_manager.py
│   └── test_multi_version_search.py
├── integration/             # 통합 테스트
│   ├── test_active_version_auto_load.py
│   ├── test_faiss_version_integration.py
│   ├── test_faiss_version_with_real_data.py
│   ├── test_v2_integration.py
│   └── ...
├── functional/             # 기능 테스트
│   ├── chunking/           # 청킹 전략 테스트
│   ├── extraction/         # 키워드/법령 추출 테스트
│   ├── search/             # 검색 엔진 테스트
│   ├── quality/            # 품질 검증 테스트
│   ├── test_keyword_coverage_improvements.py
│   ├── test_optimization.py
│   └── test_cache_effectiveness.py
├── scripts/                # 독립 실행 스크립트
│   └── run_single_query.py
├── utils/                   # 테스트 유틸리티
│   ├── __init__.py
│   ├── test_helpers.py     # 공통 헬퍼 함수
│   └── test_data.py       # 테스트 데이터 생성
├── conftest.py             # pytest 공통 fixtures (자동 로드)
├── pytest.ini              # pytest 설정
└── README.md               # 이 파일
```

## 테스트 실행

### 모든 테스트 실행

```bash
# 프로젝트 루트에서
pytest scripts/tests/ -v
```

### 단위 테스트만 실행

```bash
pytest scripts/tests/unit/ -v
```

### 통합 테스트만 실행

```bash
pytest scripts/tests/integration/ -v
```

### 기능 테스트만 실행

```bash
pytest scripts/tests/functional/ -v
```

### 특정 카테고리 테스트 실행

```bash
# 청킹 테스트만
pytest scripts/tests/functional/chunking/ -v

# 검색 테스트만
pytest scripts/tests/functional/search/ -v

# 품질 테스트만
pytest scripts/tests/functional/quality/ -v
```

### 독립 실행 스크립트

```bash
# 단일 쿼리 실행
python scripts/tests/scripts/run_single_query.py "계약 해지 사유"
```

### 마커를 사용한 테스트 필터링

```bash
# 단위 테스트만
pytest scripts/tests/ -m unit

# 통합 테스트만
pytest scripts/tests/ -m integration

# 기능 테스트만
pytest scripts/tests/ -m functional

# 느린 테스트 제외
pytest scripts/tests/ -m "not slow"
```

## 공통 Fixtures

`conftest.py`에 정의된 공통 fixtures (pytest가 자동으로 로드):

### 기본 Fixtures
- `temp_dir`: 임시 디렉토리
- `project_root`: 프로젝트 루트 경로
- `test_db_path`: 테스트용 데이터베이스 경로
- `test_vector_store_path`: 테스트용 벡터 스토어 경로
- `temp_db`: 임시 데이터베이스 (기본 스키마 포함)

### 버전 관리 Fixtures
- `version_manager`: FAISSVersionManager 인스턴스
- `embedding_version_manager`: EmbeddingVersionManager 인스턴스
- `migration_manager`: FAISSMigrationManager 인스턴스
- `multi_search`: MultiVersionSearch 인스턴스

### 환경 설정 Fixtures
- `load_env`: 환경 변수 로드 (.env 파일)
- `logger`: 로거 인스턴스

### 서비스 Fixtures
- `workflow_service`: LangGraphWorkflowService 인스턴스
- `search_engine`: SemanticSearchEngineV2 인스턴스

### 실제 데이터 Fixtures
- `real_db_path`: 실제 데이터베이스 경로 (data/lawfirm_v2.db)
- `real_vector_store_path`: 실제 벡터 스토어 경로 (data/vector_store)

## 테스트 유틸리티

### test_helpers.py

공통 헬퍼 함수:

```python
from scripts.tests.utils.test_helpers import (
    setup_test_path,
    create_temp_dir,
    temporary_env,
    run_async,
    measure_time,
    print_section,
    print_test_header,
    validate_search_results,
    assert_result_quality,
    get_cache_stats,
    compare_cache_stats,
    analyze_workflow_result
)

# 프로젝트 경로 설정
setup_test_path()

# 임시 디렉토리 생성 (컨텍스트 매니저)
with create_temp_dir() as temp_dir:
    # 테스트 코드
    pass

# 환경 변수 임시 설정
with temporary_env(USE_EXTERNAL_VECTOR_STORE='false'):
    # 테스트 코드
    pass

# 비동기 함수 실행
result = run_async(async_function())

# 실행 시간 측정
@measure_time
def test_function():
    # 테스트 코드
    pass

# 검색 결과 검증
validation = validate_search_results(results, min_count=5, min_similarity=0.7)
assert validation["valid"], validation["errors"]

# 워크플로우 결과 품질 검증
assert_result_quality(result, min_answer_length=100, min_sources=3, min_confidence=0.8)

# 캐시 통계 조회 및 비교
stats_before = get_cache_stats(service)
# ... 테스트 실행 ...
stats_after = get_cache_stats(service)
comparison = compare_cache_stats(stats_before, stats_after)

# 워크플로우 결과 분석
analysis = analyze_workflow_result(result)
print(f"평균 유사도: {analysis['avg_similarity']:.4f}")
```

### test_data.py

테스트 데이터 생성:

```python
from scripts.tests.utils.test_data import (
    TEST_QUERIES,
    create_version_info,
    create_chunk_data,
    create_test_query,
    create_workflow_result,
    create_search_result,
    get_test_queries
)

# 버전 정보 생성
version_info = create_version_info(
    version_name="v1.0.0-test",
    embedding_version_id=1
)

# 청크 데이터 생성
chunk_data = create_chunk_data(
    chunk_id=1,
    content="Test content"
)

# 테스트 쿼리 생성
query_data = create_test_query(domain="민사법")
# 또는
query_data = create_test_query(query="계약 해지 사유")

# 표준 테스트 쿼리 조회
queries = get_test_queries(domain="노동법")
all_queries = get_test_queries()  # 모든 쿼리

# 워크플로우 결과 생성
workflow_result = create_workflow_result(
    answer="테스트 답변",
    sources=[...],
    confidence=0.9
)

# 검색 결과 생성
search_result = create_search_result(
    content="검색 결과 내용",
    similarity=0.85,
    source_type="statute_article"
)
```

## 테스트 작성 가이드

### 1. Fixture 사용

공통 fixture를 사용하여 중복 코드를 제거합니다:

```python
# ❌ 나쁜 예
import sys
import tempfile
from pathlib import Path

project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from dotenv import load_dotenv
load_dotenv(project_root / "api" / ".env")

from lawfirm_langgraph.config.langgraph_config import LangGraphConfig
from lawfirm_langgraph.langgraph_core.workflow.workflow_service import LangGraphWorkflowService

config = LangGraphConfig.from_env()
service = LangGraphWorkflowService(config)

# ✅ 좋은 예
def test_something(workflow_service):
    # workflow_service fixture가 자동으로 제공됨
    result = await workflow_service.process_query("테스트 질문")
    assert result is not None
```

### 1-1. 환경 변수 임시 설정

```python
from scripts.tests.utils.test_helpers import temporary_env

def test_with_env(workflow_service):
    with temporary_env(USE_EXTERNAL_VECTOR_STORE='false'):
        # 환경 변수가 임시로 설정됨
        result = await workflow_service.process_query("테스트")
    # 자동으로 원래 값으로 복원됨
```

### 2. 테스트 데이터 생성

`test_data.py`의 팩토리 함수를 사용합니다:

```python
from scripts.tests.utils.test_data import (
    create_version_info,
    create_test_query,
    get_test_queries
)

def test_create_version(version_manager):
    version_info = create_version_info(version_name="v1.0.0-test")
    version_path = version_manager.create_version(**version_info)
    assert version_path.exists()

def test_workflow(workflow_service):
    # 표준 테스트 쿼리 사용
    query_data = create_test_query(domain="민사법")
    result = await workflow_service.process_query(query_data["query"])
    assert result is not None
```

### 2-1. 표준 테스트 쿼리 사용

```python
from scripts.tests.utils.test_data import get_test_queries

def test_multiple_queries(workflow_service):
    queries = get_test_queries(domain="노동법")
    for query_data in queries:
        result = await workflow_service.process_query(query_data["query"])
        assert result is not None
```

### 3. 임시 디렉토리 사용

`temp_dir` fixture를 사용합니다:

```python
def test_something(temp_dir):
    file_path = temp_dir / "test.txt"
    file_path.write_text("test")
    assert file_path.exists()
```

### 4. 비동기 테스트

`run_async` 헬퍼 함수를 사용합니다:

```python
from scripts.tests.utils.test_helpers import run_async

def test_async_function():
    result = run_async(async_function())
    assert result is not None
```

### 5. 검색 결과 검증

`validate_search_results` 함수를 사용합니다:

```python
from scripts.tests.utils.test_helpers import validate_search_results

def test_search(search_engine):
    results = search_engine.search("계약 해지", k=5)
    validation = validate_search_results(results, min_count=3, min_similarity=0.7)
    assert validation["valid"], validation["errors"]
    print(f"평균 유사도: {validation['avg_similarity']:.4f}")
```

### 6. 워크플로우 결과 분석

`analyze_workflow_result` 함수를 사용합니다:

```python
from scripts.tests.utils.test_helpers import analyze_workflow_result

def test_workflow_quality(workflow_service):
    result = await workflow_service.process_query("테스트 질문")
    analysis = analyze_workflow_result(result)
    
    assert analysis["answer_length"] > 100
    assert analysis["sources_count"] >= 3
    assert analysis["avg_similarity"] > 0.7
```

### 7. 성능 측정

`measure_time` 데코레이터를 사용합니다:

```python
from scripts.tests.utils.test_helpers import measure_time

@measure_time
def test_performance(workflow_service):
    result = await workflow_service.process_query("테스트 질문")
    return result

# 실행 후
print(f"실행 시간: {test_performance.execution_times[0]:.2f}초")
```

## 주의사항

1. **경로 설정**: `conftest.py`에서 자동으로 프로젝트 경로가 설정되므로, 테스트 파일에서 `sys.path` 설정이 필요 없습니다.

2. **Import 경로**: `scripts.utils` 모듈은 절대 경로로 import합니다:
   ```python
   from scripts.utils.faiss_version_manager import FAISSVersionManager
   ```

3. **Fixture 의존성**: Fixture는 자동으로 의존성을 해결하므로, 필요한 fixture만 파라미터로 선언하면 됩니다.

4. **테스트 격리**: 각 테스트는 독립적으로 실행되며, `temp_dir` fixture를 통해 격리된 환경을 제공합니다.

## 테스트 유형 설명

### 단위 테스트 (unit/)
- 개별 모듈/함수의 동작을 검증
- 빠른 실행 속도
- 격리된 환경에서 실행
- 예: 유틸리티 함수, 매니저 클래스

### 통합 테스트 (integration/)
- 여러 컴포넌트 간의 상호작용 검증
- 실제 데이터베이스/인덱스 사용
- 예: FAISS 버전 관리, 워크플로우 통합

### 기능 테스트 (functional/)
- 실제 사용 시나리오 기반 테스트
- 엔드투엔드 워크플로우 검증
- 카테고리별 분류:
  - **chunking/**: 청킹 전략 테스트
  - **extraction/**: 키워드/법령 추출 테스트
  - **search/**: 검색 엔진 품질 테스트
  - **quality/**: 콘텐츠 품질 검증 테스트

## 관련 디렉토리

- **유틸리티 스크립트**: `scripts/tools/`
  - `checks/`: 검증 스크립트
  - `builds/`: 빌드 스크립트
- **소스 코드**: `scripts/utils/`
- **독립 실행 스크립트**: `scripts/tests/scripts/`

## 문제 해결

### ImportError 발생 시

`conftest.py`가 제대로 로드되지 않았을 수 있습니다. 다음을 확인하세요:

1. `fixtures/conftest.py` 파일이 존재하는지 확인
2. pytest가 프로젝트 루트에서 실행되는지 확인
3. `PYTHONPATH` 환경 변수가 올바르게 설정되었는지 확인

### Fixture를 찾을 수 없을 때

Fixture 이름이 정확한지 확인하고, `conftest.py`에 정의되어 있는지 확인하세요.

### 테스트가 느릴 때

`@pytest.mark.slow` 마커를 추가하고, 필요할 때만 실행하세요:

```bash
pytest scripts/tests/ -m "not slow"
```

