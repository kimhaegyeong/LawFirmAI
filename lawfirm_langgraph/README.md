# LawFirm LangGraph Studio Project

LangGraph Studio에서 실행 가능한 독립 프로젝트입니다.

## 개요

이 프로젝트는 LawFirmAI의 LangGraph 워크플로우를 LangGraph Studio에서 실행 및 디버깅할 수 있도록 구성된 독립 프로젝트입니다.

## 프로젝트 구조

프로젝트 규칙에 따라 다음과 같이 구조화되었습니다:

```
lawfirm_langgraph/
├── langgraph_core/           # Core Modules (LangGraph 워크플로우 전용)
│   ├── __init__.py
│   ├── models/              # AI 모델 관련
│   │   ├── __init__.py
│   │   ├── chain_builders.py
│   │   ├── node_wrappers.py
│   │   ├── node_input_output_spec.py
│   │   └── prompt_builders.py
│   ├── services/           # 비즈니스 로직
│   │   ├── __init__.py
│   │   ├── workflow_service.py
│   │   ├── legal_workflow_enhanced.py
│   │   ├── workflow_routes.py
│   │   ├── answer_generator.py
│   │   ├── classification_handler.py
│   │   ├── search_handler.py
│   │   ├── query_enhancer.py
│   │   ├── context_builder.py
│   │   ├── direct_answer_handler.py
│   │   ├── answer_formatter.py
│   │   ├── expert_subgraphs.py
│   │   ├── legal_data_connector_v2.py
│   │   └── feedback_system.py
│   ├── utils/              # 유틸리티
│   │   ├── __init__.py
│   │   ├── state_definitions.py
│   │   ├── state_utils.py
│   │   ├── state_helpers.py
│   │   ├── state_reduction.py
│   │   ├── state_reducer_custom.py
│   │   ├── state_adapter.py
│   │   ├── modular_states.py
│   │   ├── workflow_utils.py
│   │   ├── workflow_constants.py
│   │   ├── workflow_logger.py
│   │   ├── performance_optimizer.py
│   │   ├── prompt_chain_executor.py
│   │   ├── search_performance_monitor.py
│   │   ├── checkpoint_manager.py
│   │   ├── keyword_mapper.py
│   │   ├── synonym_database.py
│   │   ├── synonym_quality_manager.py
│   │   ├── real_gemini_synonym_expander.py
│   │   ├── enhanced_semantic_relations.py
│   │   └── query_optimizer.py
│   └── data/               # 데이터 처리
│       ├── __init__.py
│       ├── extractors.py
│       ├── response_parsers.py
│       ├── reasoning_extractor.py
│       └── quality_validators.py
├── agents/                 # 레거시 호환성 (backward compatibility)
│   └── __init__.py         # source 서비스 재export
├── config/                 # LangGraph 설정
│   ├── __init__.py
│   └── langgraph_config.py
├── tests/                  # 테스트 코드
│   ├── unit/               # 단위 테스트
│   │   ├── test_execute.py
│   │   ├── test_manual.py
│   │   ├── test_quick.py
│   │   ├── test_setup.py
│   │   ├── test_simple.py
│   │   ├── test_with_output.py
│   │   └── test_workflow.py
│   └── integration/        # 통합 테스트
├── docs/                   # 문서
│   ├── TEST_ANALYSIS.md
│   ├── CHECK_TEST_LOG.md
│   ├── RUN_TEST.md
│   ├── TEST_GUIDE.md
│   ├── QUICKSTART.md
│   └── SETUP.md
├── graph.py                # Studio용 그래프 export 파일
├── langgraph.json          # LangGraph Studio 설정 파일
├── requirements.txt        # LangGraph v1.0 의존성
└── README.md               # 이 파일
```

## 빠른 시작

### 1. 의존성 설치

```bash
cd lawfirm_langgraph
pip install -r requirements.txt
```

또는 LangGraph CLI만 설치:

```bash
pip install "langgraph-cli[inmem]>=0.4.5"
```

### 2. 설정 확인

```bash
python tests/unit/test_setup.py
```

이 스크립트는 다음을 확인합니다:
- Python 버전 (3.10+)
- LangGraph 버전 (1.0+)
- 모듈 import
- 그래프 생성
- LangGraph CLI 설치

### 3. LangGraph Studio 실행

```bash
langgraph dev
```

브라우저에서 제공된 URL로 접속하여 LangGraph Studio를 사용할 수 있습니다.

## 환경 변수 설정

상위 프로젝트의 `.env` 파일을 참조하거나, `lawfirm_langgraph/` 디렉토리에 `.env` 파일을 생성합니다.

필수 환경 변수:
- `GOOGLE_API_KEY`: Google Gemini API 키
- `LANGGRAPH_ENABLED`: LangGraph 활성화 (기본값: true)

선택 환경 변수:
- `LANGFUSE_SECRET_KEY`, `LANGFUSE_PUBLIC_KEY`: Langfuse 추적
- `LANGCHAIN_API_KEY`, `LANGCHAIN_TRACING_V2`: LangSmith 추적

## 그래프

이 프로젝트는 다음 그래프를 제공합니다:

- **legal_workflow**: `EnhancedLegalQuestionWorkflow` 그래프
- **workflow_service**: `LangGraphWorkflowService` 컴파일된 앱

Studio에서 그래프를 선택하여 시각화 및 디버깅할 수 있습니다.

## 상위 프로젝트 의존성

이 프로젝트는 상위 프로젝트(LawFirmAI)의 다음 모듈을 참조합니다:

- `core/services/` - 검색, 분류 등 서비스
- `source/services/` - 레거시 서비스
- `infrastructure/` - 인프라 설정

상위 프로젝트 경로가 `sys.path`에 자동으로 추가됩니다.

## 모듈 구조

### langgraph_core/models/
체인 빌더, 프롬프트 빌더, 노드 래퍼 등 모델 관련 컴포넌트

### langgraph_core/services/
워크플로우 서비스, 핸들러, 비즈니스 로직 등 서비스 레이어

### langgraph_core/utils/
상태 관리, 워크플로우 유틸리티, 성능 최적화 등 유틸리티 함수

### langgraph_core/data/
데이터 추출, 파싱, 검증 등 데이터 처리 모듈

## 문제 해결

### Import 오류

상위 프로젝트의 모듈을 찾을 수 없는 경우, 상위 프로젝트의 경로가 올바른지 확인하세요.

### 환경 변수 오류

`.env` 파일이 상위 프로젝트에 있는 경우, `langgraph.json`의 `env` 경로를 확인하세요 (기본값: `../.env`).

### LangGraph CLI 오류

CLI가 설치되지 않은 경우:

```bash
pip install "langgraph-cli[inmem]>=0.4.5"
```

### 그래프 로드 오류

`tests/unit/test_setup.py`를 실행하여 문제를 진단하세요:

```bash
python tests/unit/test_setup.py
```

## 참고

- LangGraph v1.0 사용
- Python 3.10+ 필요
- 상위 프로젝트의 `.env` 파일 참조 (../.env)

## 테스트

### 빠른 테스트

기본 기능이 정상 동작하는지 빠르게 확인:

```bash
python tests/unit/test_quick.py
```

### 종합 테스트

모든 기능을 종합적으로 테스트:

```bash
python tests/unit/test_workflow.py
```

### 테스트 가이드

상세한 테스트 실행 방법과 문제 해결은 `docs/TEST_GUIDE.md`를 참조하세요.

## 추가 문서

- `docs/SETUP.md`: 상세 설정 가이드
- `docs/TEST_GUIDE.md`: 테스트 실행 가이드
- `docs/QUICKSTART.md`: 빠른 시작 가이드
- `tests/unit/test_setup.py`: 설정 테스트 스크립트
- `tests/unit/test_quick.py`: 빠른 기본 기능 테스트
- `tests/unit/test_workflow.py`: 종합 워크플로우 테스트

## 마이그레이션 가이드

기존 `agents/` 디렉토리를 사용하던 코드는 새로운 `source/` 구조를 사용하도록 업데이트되었습니다.

### 이전 (deprecated)
```python
from agents.workflow_service import LangGraphWorkflowService
from agents.state_definitions import LegalWorkflowState
from agents.workflow_utils import WorkflowUtils
```

### 현재 (권장)
```python
from source.services.workflow_service import LangGraphWorkflowService
from source.utils.state_definitions import LegalWorkflowState
from source.utils.workflow_utils import WorkflowUtils
```

`agents/__init__.py`는 하위 호환성을 위해 유지되지만, 새로운 코드에서는 `source/` 경로를 사용하는 것을 권장합니다.
