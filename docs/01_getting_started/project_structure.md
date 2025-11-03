# 📁 LawFirmAI 프로젝트 구조

## 개요

LawFirmAI는 명확한 계층 구조로 구성된 법률 AI 시스템입니다.

## 전체 구조

```
LawFirmAI/
├── source/                   # 핵심 비즈니스 로직 (agents, services, models, data)
├── apps/                     # 애플리케이션 레이어
├── infrastructure/           # 인프라 및 유틸리티
├── scripts/                  # 실행 스크립트
├── data/                     # 데이터 파일
├── tests/                    # 테스트 코드
├── docs/                     # 문서
└── monitoring/               # 모니터링 시스템
```

## 📦 Source 모듈

### source/agents/ - LangGraph 에이전트
**역할**: AI 워크플로우 관리

```
source/agents/
├── workflow_service.py              # 워크플로우 서비스 (메인)
├── legal_workflow_enhanced.py       # 법률 워크플로우
├── state_definitions.py             # 상태 정의
├── state_utils.py                   # 상태 유틸리티
├── state_helpers.py                 # 상태 헬퍼 함수
├── state_reduction.py                # 상태 최적화
├── keyword_mapper.py                # 키워드 매퍼
├── legal_data_connector_v2.py       # 데이터 커넥터 (v2)
├── performance_optimizer.py          # 성능 최적화
├── node_wrappers.py                 # 노드 래퍼
├── query_optimizer.py               # 쿼리 최적화
└── ...
```

**사용 예시**:
```python
from source.agents.workflow_service import LangGraphWorkflowService
from infrastructure.utils.langgraph_config import LangGraphConfig

config = LangGraphConfig.from_env()
workflow = LangGraphWorkflowService(config)
result = await workflow.process_query("질문", "session_id")
```

### source/services/ - 검색 서비스
**역할**: 법률 문서 검색

```
source/services/
├── hybrid_search_engine.py          # 하이브리드 검색
├── exact_search_engine.py           # 정확한 매칭
├── semantic_search_engine.py        # 의미적 검색
├── precedent_search_engine.py       # 판례 검색
├── question_classifier.py           # 질문 분류
└── result_merger.py                 # 결과 병합
```

**사용 예시**:
```python
from source.services.search import HybridSearchEngine

engine = HybridSearchEngine()
results = engine.search("계약 해지", question_type="law_inquiry")
```

### source/services/ - 답변 생성
**역할**: 답변 생성 및 포맷팅 (source/services에 통합됨)

```
source/services/
├── answer_generator.py             # 답변 생성
├── improved_answer_generator.py   # 개선된 답변 생성
├── context_builder.py              # 컨텍스트 구축
└── answer_formatter.py             # 답변 포맷팅
```

**사용 예시**:
```python
from source.services.generation import AnswerGenerator

generator = AnswerGenerator()
answer = generator.generate(query, context)
```

### source/services/ - 품질 개선
**역할**: 답변 품질 향상 (source/services에 통합됨)

```
source/services/
└── confidence_calculator.py       # 신뢰도 계산
```

### source/models/ - AI 모델
**역할**: AI 모델 관리

```
source/models/
├── model_manager.py                # 모델 관리자
├── sentence_bert.py                # Sentence BERT
└── gemini_client.py                # Gemini 클라이언트
```

### source/data/ - 데이터 레이어
**역할**: 데이터 관리

```
source/data/
├── database.py                     # SQLite 데이터베이스
├── vector_store.py                 # FAISS 벡터 스토어
├── data_processor.py               # 데이터 처리
├── conversation_store.py            # 대화 저장소
└── legal_term_normalizer.py        # 법률 용어 정규화
```

## 📱 Apps 모듈

### apps/streamlit/
**역할**: Streamlit 웹 인터페이스

```
apps/streamlit/
├── app.py                          # 메인 앱
└── ...
```

### apps/api/
**역할**: FastAPI 서버

```
apps/api/
├── routes/                         # API 라우트
└── ...
```

## 🔧 Infrastructure 모듈

### infrastructure/utils/
**역할**: 유틸리티 함수

```
infrastructure/utils/
├── langgraph_config.py             # LangGraph 설정
├── langchain_config.py             # LangChain 설정
├── logger.py                       # 로깅
├── config.py                       # 일반 설정
├── ollama_client.py                # Ollama 클라이언트
└── ...
```

## 📊 데이터 흐름

### 1. 쿼리 처리
```
User Input
    ↓
apps/streamlit/app.py 또는 apps/api/
    ↓
core/agents/workflow_service.py
    ↓
core/agents/legal_workflow_enhanced.py (LangGraph 워크플로우)
    ↓
core/services/search/ (검색)
    ↓
core/services/generation/ (생성)
    ↓
core/services/enhancement/ (품질)
    ↓
User Output
```

### 2. 검색 프로세스
```
Query
    ↓
core/services/search/question_classifier.py (분류)
    ↓
core/services/search/hybrid_search_engine.py (검색)
    ├── exact_search_engine.py
    └── semantic_search_engine.py
    ↓
Results
```

## 🔗 Import 체계

### 프로젝트 루트 설정
```python
import sys
from pathlib import Path

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))
```

### Core 모듈 Import
```python
from source.agents.workflow_service import LangGraphWorkflowService
from source.services.search import HybridSearchEngine
from source.services.generation import AnswerGenerator
from infrastructure.utils.langgraph_config import LangGraphConfig
```

## 📚 확장 가이드

### 새 검색 엔진 추가
1. `source/services/search/new_engine.py` 생성
2. `source/services/search/__init__.py` 업데이트
3. 테스트 작성

### 새 답변 생성기 추가
1. `source/services/generation/new_generator.py` 생성
2. `source/services/generation/__init__.py` 업데이트
3. 테스트 작성

### 새 애플리케이션 추가
1. `apps/new_app/` 디렉토리 생성
2. 메인 파일 작성
3. Dockerfile 추가
4. 문서 업데이트

## 🎯 모듈별 책임

| 모듈 | 책임 | 의존성 |
|------|------|--------|
| `source/agents/` | 워크플로우 관리 | services, models |
| `source/services/search/` | 검색 로직 | data |
| `source/services/generation/` | 답변 생성 | search, models |
| `source/services/enhancement/` | 품질 개선 | generation |
| `source/models/` | AI 모델 | - |
| `source/data/` | 데이터 관리 | - |
| `apps/streamlit/` | 웹 UI | source/agents |
| `apps/api/` | API 서버 | source/agents |
| `infrastructure/` | 인프라 | - |
| `source/` | 레거시 모듈 | (호환성 유지) |

## 🚀 개발 워크플로우

### 1. 기능 추가
```bash
# 새 서비스 추가
vim source/services/new_service.py

# __init__.py 업데이트
vim source/services/__init__.py

# 테스트 작성
vim tests/test_new_service.py

# 테스트 실행
python tests/test_new_service.py
```

### 2. 디버깅
```python
# 로깅 활성화
import logging
logging.basicConfig(level=logging.DEBUG)

# 에러 추적
import traceback
try:
    # 코드
except Exception as e:
    traceback.print_exc()
```

### 3. 테스트
```bash
# 전체 테스트
python tests/test_core_imports.py
python tests/test_core_workflow.py

# 특정 테스트
python tests/test_hybrid_search.py
```

## 📝 규칙 및 컨벤션

### 1. Naming
- 파일: `snake_case.py`
- 클래스: `PascalCase`
- 함수/변수: `snake_case`
- 상수: `UPPER_SNAKE_CASE`

### 2. Import 순서
```python
# 표준 라이브러리
import os
import sys

# 서드파티
import torch
from fastapi import FastAPI

# 프로젝트 모듈
from source.agents import LangGraphWorkflowService
```

### 3. Docstring
```python
def process_data(data: Dict[str, Any]) -> str:
    """
    데이터 처리 함수
    
    Args:
        data: 처리할 데이터
        
    Returns:
        처리된 결과
    """
    pass
```
