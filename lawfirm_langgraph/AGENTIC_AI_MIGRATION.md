# Agentic AI 시스템 마이그레이션 가이드

## 개요

Agentic AI 시스템(Tool Use/Function Calling)이 `lawfirm_langgraph` 구조에 맞게 구현되었습니다.

## 파일 구조

### lawfirm_langgraph 구조 (주요 사용 위치)

```
lawfirm_langgraph/
└── langgraph_core/
    └── tools/
        ├── __init__.py                    # Tool 시스템 등록
        └── legal_search_tools.py          # 검색 Tool 구현
```

### core 구조

```
core/
└── agents/
    └── legal_workflow_enhanced.py        # Agentic 노드 구현 (langgraph_core.tools 사용)
    # tools/ 폴더는 삭제됨 (lawfirm_langgraph 구조로 완전 전환)
```

## 주요 변경사항

### 1. Tool 시스템 위치
- **위치**: `lawfirm_langgraph/langgraph_core/tools/` (단일 위치)
- **core/agents/tools/**: 삭제됨 (완전 전환 완료)

### 2. Import 경로
- **Tool import**: `from langgraph_core.tools import LEGAL_TOOLS` (단일 경로)
- **core/agents/tools/**: 삭제됨 (더 이상 사용하지 않음)

### 3. 설정 파일
- **위치**: `infrastructure/utils/langgraph_config.py`
- **설정 항목**: `use_agentic_mode: bool = False`
- **환경 변수**: `USE_AGENTIC_MODE=true`

## 사용 방법

### Agentic 모드 활성화

```bash
# 환경 변수 설정
export USE_AGENTIC_MODE=true
```

또는 `.env` 파일에:
```
USE_AGENTIC_MODE=true
```

### 코드에서 사용

```python
from infrastructure.utils.langgraph_config import LangGraphConfig
from source.agents.legal_workflow_enhanced import EnhancedLegalQuestionWorkflow

config = LangGraphConfig.from_env()
config.use_agentic_mode = True  # 또는 환경 변수로 설정

workflow = EnhancedLegalQuestionWorkflow(config)
```

## 구현된 기능

### 1. Tool 시스템
- `search_precedent_tool`: 판례 검색
- `search_law_tool`: 법령 검색
- `search_legal_term_tool`: 법률 용어 검색
- `hybrid_search_tool`: 통합 검색

### 2. Agentic 노드
- `agentic_decision_node`: LLM이 Tool을 자동으로 선택하고 실행
- 복잡한 질문(`complex`)은 Agentic 노드로 라우팅
- Tool 실행 결과를 기존 state 구조로 변환

### 3. 워크플로우 통합
- Agentic 모드 활성화 시 `agentic_decision` 노드 추가
- 조건부 라우팅: 검색 결과 유무에 따라 다음 단계 결정

## 마이그레이션 계획

### 현재 상태
- ✅ `lawfirm_langgraph/langgraph_core/tools/`에 Tool 시스템 구현
- ✅ `core/agents/legal_workflow_enhanced.py`에서 양쪽 구조 지원 (우선순위: langgraph_core)
- ✅ 설정 파일에 `use_agentic_mode` 추가

### 추후 작업 (core 폴더 삭제 전)
1. `core/agents/legal_workflow_enhanced.py`를 `lawfirm_langgraph`로 이동
2. 모든 import 경로를 `langgraph_core`로 변경
3. `core/agents/tools/` 삭제
4. 테스트 코드 업데이트

## 테스트

테스트 파일 위치:
- `tests/langgraph/test_agentic_integration.py`: 통합 테스트
- `tests/langgraph/test_agentic_workflow.py`: 워크플로우 테스트

실행:
```bash
python tests/langgraph/test_agentic_integration.py
python tests/langgraph/test_agentic_workflow.py
```

## 주의사항

1. **core 폴더 의존성**: 현재 Tool이 `core.services.search`를 참조하고 있습니다. 추후 이 부분도 `lawfirm_langgraph`로 마이그레이션 필요합니다.

2. **임시 호환성**: `core/agents/tools/`는 임시로 유지되지만, 우선순위는 `langgraph_core.tools`입니다.

3. **설정 파일**: `infrastructure/utils/langgraph_config.py`에 설정이 있습니다. `lawfirm_langgraph/config/`로 이동 검토 필요할 수 있습니다.

