# 프로젝트 구조 개선 완료

이 문서는 lawfirm_langgraph 프로젝트의 구조 개선 결과를 설명합니다.

## 개선 사항

### 1. 모듈 구조 재구성

기존의 `agents/` 디렉토리에 모든 파일이 섞여있던 구조를 프로젝트 규칙에 맞는 `source/` 구조로 재구성했습니다.

**변경 전:**
```
agents/
├── workflow_service.py
├── legal_workflow_enhanced.py
├── state_definitions.py
├── extractors.py
├── prompt_builders.py
└── ... (40개 이상의 파일)
```

**변경 후:**
```
source/
├── models/        # AI 모델 관련
├── services/      # 비즈니스 로직
├── utils/         # 유틸리티
└── data/          # 데이터 처리
```

### 2. 파일 분류

#### source/models/
- `chain_builders.py` - 체인 빌더
- `prompt_builders.py` - 프롬프트 빌더
- `node_wrappers.py` - 노드 래퍼
- `node_input_output_spec.py` - 노드 입출력 사양

#### source/services/
- `workflow_service.py` - 워크플로우 서비스
- `legal_workflow_enhanced.py` - 메인 워크플로우
- `workflow_routes.py` - 라우팅
- `answer_generator.py` - 답변 생성
- `classification_handler.py` - 분류 핸들러
- `search_handler.py` - 검색 핸들러
- `query_enhancer.py` - 쿼리 향상
- `context_builder.py` - 컨텍스트 빌더
- `direct_answer_handler.py` - 직접 답변 핸들러
- `answer_formatter.py` - 답변 포맷터
- `expert_subgraphs.py` - 전문가 서브그래프
- `legal_data_connector_v2.py` - 데이터 커넥터
- `feedback_system.py` - 피드백 시스템

#### source/utils/
- `state_definitions.py` - 상태 정의
- `state_utils.py` - 상태 유틸리티
- `state_helpers.py` - 상태 헬퍼
- `state_reduction.py` - 상태 축소
- `state_reducer_custom.py` - 커스텀 상태 축소
- `state_adapter.py` - 상태 어댑터
- `modular_states.py` - 모듈러 상태
- `workflow_utils.py` - 워크플로우 유틸리티
- `workflow_constants.py` - 워크플로우 상수
- `workflow_logger.py` - 워크플로우 로거
- `performance_optimizer.py` - 성능 최적화
- `prompt_chain_executor.py` - 프롬프트 체인 실행
- `search_performance_monitor.py` - 검색 성능 모니터
- `checkpoint_manager.py` - 체크포인트 관리자
- `keyword_mapper.py` - 키워드 매퍼
- `synonym_database.py` - 동의어 데이터베이스
- `synonym_quality_manager.py` - 동의어 품질 관리자
- `real_gemini_synonym_expander.py` - Gemini 동의어 확장
- `enhanced_semantic_relations.py` - 향상된 의미 관계
- `query_optimizer.py` - 쿼리 최적화

#### source/data/
- `extractors.py` - 추출기
- `response_parsers.py` - 응답 파서
- `reasoning_extractor.py` - 추론 추출기
- `quality_validators.py` - 품질 검증자

### 3. 테스트 구조 정리

**변경 전:**
```
test_*.py (루트 디렉토리에 산재)
```

**변경 후:**
```
tests/
├── unit/           # 단위 테스트
│   ├── test_execute.py
│   ├── test_manual.py
│   ├── test_quick.py
│   ├── test_setup.py
│   ├── test_simple.py
│   ├── test_with_output.py
│   └── test_workflow.py
└── integration/   # 통합 테스트 (준비됨)
```

### 4. 문서 및 스크립트 정리

**문서 파일:**
```
docs/
├── CHECK_TEST_LOG.md
├── QUICKSTART.md
├── RUN_TEST.md
├── SETUP.md
├── TEST_ANALYSIS.md
├── TEST_GUIDE.md
└── examples/
    ├── README.md
    └── migration_example.py
```

**스크립트 파일:**
```
scripts/
├── execute_test.py
├── run_test_direct.py
├── run_test_with_log.py
└── run_tests.ps1
```

### 5. Import 경로 업데이트

모든 파일의 import 경로를 새로운 구조에 맞게 업데이트했습니다.

**변경 전:**
```python
from agents.workflow_service import LangGraphWorkflowService
from agents.state_definitions import LegalWorkflowState
from agents.workflow_utils import WorkflowUtils
```

**변경 후:**
```python
from source.services.workflow_service import LangGraphWorkflowService
from source.utils.state_definitions import LegalWorkflowState
from source.utils.workflow_utils import WorkflowUtils
```

### 6. 하위 호환성 유지

기존 코드와의 호환성을 위해 `agents/__init__.py`에서 새로운 모듈을 재export합니다.

```python
# agents/__init__.py
from source.services.workflow_service import LangGraphWorkflowService

__all__ = ["LangGraphWorkflowService"]
```

### 7. 모듈 Export 최적화

각 모듈의 `__init__.py`에서 주요 클래스와 함수를 명시적으로 export하여 사용하기 쉽도록 했습니다.

## 주요 변경 파일

### 업데이트된 파일
- `graph.py` - import 경로 업데이트
- 모든 `source/` 하위 파일들 - import 경로 업데이트
- 모든 `tests/unit/` 테스트 파일 - import 경로 업데이트
- `README.md` - 구조 설명 업데이트

### 새로 생성된 파일
- `source/__init__.py`
- `source/models/__init__.py`
- `source/services/__init__.py`
- `source/utils/__init__.py`
- `source/data/__init__.py`
- `tests/__init__.py`
- `tests/unit/__init__.py`
- `tests/integration/__init__.py`
- `docs/examples/README.md`
- `STRUCTURE.md` (이 파일)

### 이동된 파일
- 모든 `agents/*.py` → `source/*/` (기능별 분류)
- 모든 `test_*.py` → `tests/unit/`
- 모든 `*.md` (README.md 제외) → `docs/`
- `agents/migration_example.py` → `docs/examples/migration_example.py`
- 실행 스크립트들 → `scripts/`
- 로그 파일들 → `logs/`

## 사용 방법

### 새로운 구조로 import

```python
# 권장 방식
from source.services.workflow_service import LangGraphWorkflowService
from source.utils.state_definitions import LegalWorkflowState
from source.data.extractors import DocumentExtractor
from source.models.chain_builders import AnswerGenerationChainBuilder
```

### 하위 호환성 (레거시)

```python
# 여전히 작동하지만 권장하지 않음
from agents.workflow_service import LangGraphWorkflowService
```

## 테스트

모든 테스트 파일의 import 경로가 업데이트되었으며, 다음 명령으로 테스트할 수 있습니다:

```bash
# 설정 테스트
python tests/unit/test_setup.py

# 빠른 테스트
python tests/unit/test_quick.py

# 종합 테스트
python tests/unit/test_workflow.py
```

## 장점

1. **명확한 계층 구조**: models, services, utils, data로 역할이 명확히 분리됨
2. **유지보수성 향상**: 관련 파일들이 논리적으로 그룹화됨
3. **확장성**: 새로운 기능 추가 시 적절한 위치에 쉽게 추가 가능
4. **일관성**: 프로젝트 규칙에 따른 표준 구조 적용
5. **가독성**: 파일 구조만 봐도 프로젝트의 구조를 이해할 수 있음

## 참고

- 프로젝트 규칙: 상위 프로젝트의 규칙 문서 참조
- 마이그레이션 가이드: `README.md`의 "마이그레이션 가이드" 섹션 참조
