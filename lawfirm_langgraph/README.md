# LawFirm LangGraph

법률 AI 어시스턴트를 위한 LangGraph 기반 워크플로우 시스템

## 개요

이 프로젝트는 LangGraph를 활용하여 법률 질문 처리, 문서 분석, 판례 검색 등의 기능을 제공하는 AI 어시스턴트입니다.

## 주요 기능

- **LangGraph 워크플로우**: 상태 기반 워크플로우 관리
- **Agentic AI**: Tool Use/Function Calling을 통한 동적 도구 선택
- **하이브리드 검색**: 의미적 검색 + 키워드 검색
- **답변 생성**: Google Gemini를 활용한 고품질 답변 생성

## 프로젝트 구조

```
lawfirm_langgraph/
├── config/              # 설정 파일
│   ├── langgraph_config.py
│   └── app_config.py
├── core/                # 핵심 비즈니스 로직
│   ├── workflow/        # 워크플로우 관련
│   │   ├── nodes/       # 워크플로우 노드
│   │   ├── state/       # 상태 정의 및 관리
│   │   ├── tools/       # Agentic AI Tools
│   │   ├── utils/       # 워크플로우 유틸리티
│   │   ├── builders/    # 체인 빌더
│   │   └── mixins/      # 워크플로우 믹스인
│   ├── processing/      # 데이터 처리
│   ├── agents/          # 에이전트 관련
│   ├── services/        # 비즈니스 서비스
│   ├── data/            # 데이터 레이어
│   ├── models/          # AI 모델
│   └── utils/           # 유틸리티
├── tests/              # 테스트 파일
│   ├── __init__.py
│   ├── conftest.py              # pytest 설정 및 픽스처
│   ├── test_config.py           # LangGraphConfig 테스트
│   ├── test_workflow_service.py  # LangGraphWorkflowService 테스트
│   ├── test_workflow_nodes.py   # 워크플로우 노드 테스트
│   ├── test_integration.py      # 통합 테스트
│   └── run_all_tests.py         # 전체 테스트 실행 스크립트
└── data/               # 데이터 파일
    ├── checkpoints/    # 체크포인트 데이터
    └── prompts/        # 프롬프트 데이터
```

## 빠른 시작

### 1. 의존성 설치

```bash
pip install -r requirements.txt
```

### 2. 환경 변수 설정

`.env` 파일 생성:

```env
LANGRAPH_ENABLED=true
USE_AGENTIC_MODE=false
GOOGLE_API_KEY=your_api_key_here
```

### 3. 기본 사용

```python
import asyncio
from lawfirm_langgraph.config.langgraph_config import LangGraphConfig
from lawfirm_langgraph.core.workflow.workflow_service import LangGraphWorkflowService

async def main():
    config = LangGraphConfig.from_env()
    service = LangGraphWorkflowService(config)
    
    result = await service.process_query_async(
        "계약서 작성 시 주의할 사항은?",
        "session_id"
    )
    print(result)

asyncio.run(main())
```

## 테스트

### 테스트 환경 설정

```bash
# pytest 및 테스트 의존성 설치
pip install pytest pytest-asyncio pytest-cov pytest-mock
```

### 전체 테스트 실행

```bash
# 방법 1: run_all_tests.py 스크립트 사용
cd lawfirm_langgraph/tests
python run_all_tests.py

# 방법 2: pytest 직접 실행
pytest lawfirm_langgraph/tests/ -v
```

### 개별 테스트 실행

```bash
# 설정 테스트
pytest lawfirm_langgraph/tests/test_config.py -v

# 워크플로우 서비스 테스트
pytest lawfirm_langgraph/tests/test_workflow_service.py -v

# 워크플로우 노드 테스트
pytest lawfirm_langgraph/tests/test_workflow_nodes.py -v

# 통합 테스트
pytest lawfirm_langgraph/tests/test_integration.py -v

# 특정 테스트 파일 실행 (스크립트 사용)
python lawfirm_langgraph/tests/run_all_tests.py config
```

### 테스트 커버리지

```bash
# 커버리지 리포트 생성 (pytest-cov 필요)
pytest lawfirm_langgraph/tests/ --cov=lawfirm_langgraph --cov-report=html

# HTML 리포트 확인
# htmlcov/index.html 파일을 브라우저에서 열기
```

## Agentic 모드

Agentic 모드를 활성화하면 LLM이 동적으로 도구를 선택하여 사용할 수 있습니다.

### 설정

```env
USE_AGENTIC_MODE=true
```

### 사용

```python
config = LangGraphConfig.from_env()
config.use_agentic_mode = True

service = LangGraphWorkflowService(config)
# 복잡한 질문은 자동으로 Agentic 모드로 처리됩니다
```

## 문서

- [테스트 결과](docs/TEST_RESULTS.md)
- [다음 단계 가이드](docs/NEXT_STEPS.md)

## 문제 해결

### Import 오류

```bash
pip install langchain langchain-core langgraph
```

### Agentic 모드 활성화 안됨

1. 환경 변수 확인: `USE_AGENTIC_MODE=true`
2. Config 확인: `config.use_agentic_mode`
3. Tool 시스템 로드 확인