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
│   └── langgraph_config.py
├── langgraph_core/      # 핵심 모듈
│   ├── services/       # 워크플로우 서비스
│   ├── utils/          # 유틸리티 함수
│   ├── tools/          # Agentic AI Tools
│   └── models/         # 모델 관련
├── tests/              # 테스트 파일
│   ├── test_migration_complete.py
│   ├── test_basic_functionality.py
│   ├── test_full_workflow.py
│   └── run_all_tests.py
└── docs/               # 문서
    ├── TEST_RESULTS.md
    └── NEXT_STEPS.md
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
from lawfirm_langgraph.langgraph_core.services.workflow_service import LangGraphWorkflowService

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

### 전체 테스트 실행

```bash
cd lawfirm_langgraph/tests
python run_all_tests.py
```

### 개별 테스트 실행

```bash
# 마이그레이션 검증
python lawfirm_langgraph/tests/test_migration_complete.py

# 기본 기능 테스트
python lawfirm_langgraph/tests/test_basic_functionality.py

# 전체 워크플로우 테스트
python lawfirm_langgraph/tests/test_full_workflow.py
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

자세한 내용은 [다음 단계 가이드](docs/NEXT_STEPS.md)를 참고하세요.

## 라이선스

이 프로젝트는 내부 사용 목적으로 개발되었습니다.
