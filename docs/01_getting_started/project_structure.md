# 📁 LawFirmAI 프로젝트 구조

## 개요

LawFirmAI는 명확한 계층 구조로 구성된 법률 AI 시스템입니다.

## 전체 구조

```
LawFirmAI/
├── core/                    # 핵심 비즈니스 로직
├── apps/                     # 애플리케이션 레이어
├── infrastructure/           # 인프라 및 유틸리티
├── source/                   # 레거시 모듈 (호환성 유지)
├── scripts/                  # 실행 스크립트
├── data/                     # 데이터 파일
├── tests/                    # 테스트 코드
├── docs/                     # 문서
└── monitoring/               # 모니터링 시스템
```

## 📦 Core 모듈

### lawfirm_langgraph/ - LangGraph 워크플로우 (권장)
**역할**: AI 워크플로우 관리 (LangGraph Studio 프로젝트)

> **⚠️ 중요**: `core/agents/`는 레거시이며 삭제 예정입니다. 새로운 코드는 `lawfirm_langgraph/source/`를 사용하세요.

```
lawfirm_langgraph/
├── source/
│   ├── services/                     # 비즈니스 로직
│   │   ├── workflow_service.py       # 워크플로우 서비스 (메인)
│   │   ├── legal_workflow_enhanced.py # 법률 워크플로우
│   │   ├── search_handler.py         # 검색 핸들러
│   │   └── ...
│   ├── utils/                        # 유틸리티
│   │   ├── state_definitions.py      # 상태 정의
│   │   ├── state_utils.py            # 상태 유틸리티
│   │   ├── state_helpers.py          # 상태 헬퍼 함수
│   │   ├── state_reduction.py       # 상태 최적화
│   │   ├── keyword_mapper.py        # 키워드 매퍼
│   │   ├── performance_optimizer.py  # 성능 최적화
│   │   └── ...
│   ├── models/                       # AI 모델 관련
│   │   ├── node_wrappers.py          # 노드 래퍼
│   │   └── ...
│   └── data/                         # 데이터 처리
│       └── ...
```

**사용 예시**:
```python
import sys
from pathlib import Path

# lawfirm_langgraph 경로 추가
lawfirm_langgraph_path = Path(__file__).parent.parent / "lawfirm_langgraph"
sys.path.insert(0, str(lawfirm_langgraph_path))

from source.services.workflow_service import LangGraphWorkflowService
from infrastructure.utils.langgraph_config import LangGraphConfig

config = LangGraphConfig.from_env()
workflow = LangGraphWorkflowService(config)
result = await workflow.process_query("질문", "session_id")
```

### core/agents/ - LangGraph 에이전트 (레거시, 삭제 예정)
> **⚠️ Deprecated**: 이 디렉토리는 레거시이며 삭제 예정입니다. `lawfirm_langgraph/source/`를 사용하세요.

### core/services/search/ - 검색 서비스
**역할**: 법률 문서 검색

```
core/services/search/
├── hybrid_search_engine.py          # 하이브리드 검색
├── exact_search_engine.py           # 정확한 매칭
├── semantic_search_engine.py        # 의미적 검색
├── precedent_search_engine.py       # 판례 검색
├── question_classifier.py           # 질문 분류
└── result_merger.py                 # 결과 병합
```

**사용 예시**:
```python
from core.services.search import HybridSearchEngine

engine = HybridSearchEngine()
results = engine.search("계약 해지", question_type="law_inquiry")
```

### core/services/generation/ - 답변 생성
**역할**: 답변 생성 및 포맷팅

```
core/services/generation/
├── answer_generator.py             # 답변 생성
├── improved_answer_generator.py   # 개선된 답변 생성
├── context_builder.py              # 컨텍스트 구축
└── answer_formatter.py             # 답변 포맷팅
```

**사용 예시**:
```python
from core.services.generation import AnswerGenerator

generator = AnswerGenerator()
answer = generator.generate(query, context)
```

### core/services/enhancement/ - 품질 개선
**역할**: 답변 품질 향상

```
core/services/enhancement/
└── confidence_calculator.py       # 신뢰도 계산
```

### core/models/ - AI 모델
**역할**: AI 모델 관리

```
core/models/
├── model_manager.py                # 모델 관리자
├── sentence_bert.py                # Sentence BERT
└── gemini_client.py                # Gemini 클라이언트
```

### core/data/ - 데이터 레이어
**역할**: 데이터 관리

```
core/data/
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
streamlit/app.py 또는 apps/api/
    ↓
lawfirm_langgraph/source/services/workflow_service.py
    ↓
lawfirm_langgraph/source/services/legal_workflow_enhanced.py (LangGraph 워크플로우)
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
import sys
from pathlib import Path

# lawfirm_langgraph 경로 추가
lawfirm_langgraph_path = Path(__file__).parent.parent / "lawfirm_langgraph"
sys.path.insert(0, str(lawfirm_langgraph_path))

# LangGraph 워크플로우 서비스
from source.services.workflow_service import LangGraphWorkflowService
from infrastructure.utils.langgraph_config import LangGraphConfig

# 기타 서비스
from core.services.search import HybridSearchEngine
from core.services.generation import AnswerGenerator
```

## 📚 확장 가이드

### 새 검색 엔진 추가
1. `core/services/search/new_engine.py` 생성
2. `core/services/search/__init__.py` 업데이트
3. 테스트 작성

### 새 답변 생성기 추가
1. `core/services/generation/new_generator.py` 생성
2. `core/services/generation/__init__.py` 업데이트
3. 테스트 작성

### 새 애플리케이션 추가
1. `apps/new_app/` 디렉토리 생성
2. 메인 파일 작성
3. Dockerfile 추가
4. 문서 업데이트

## 🎯 모듈별 책임

| 모듈 | 책임 | 의존성 |
|------|------|--------|
| `lawfirm_langgraph/source/` | 워크플로우 관리 (메인) ⭐ | services, models |
| `core/services/search/` | 검색 로직 | data |
| `core/services/generation/` | 답변 생성 | search, models |
| `core/services/enhancement/` | 품질 개선 | generation |
| `core/models/` | AI 모델 | - |
| `core/data/` | 데이터 관리 | - |
| `streamlit/` | 웹 UI | lawfirm_langgraph/source |
| `apps/api/` | API 서버 | lawfirm_langgraph/source |
| `infrastructure/` | 인프라 | - |
| `core/agents/` | 레거시 (삭제 예정) | (호환성 유지) |
| `source/` | 레거시 모듈 | (호환성 유지) |

## 🚀 개발 워크플로우

### 1. 기능 추가
```bash
# 새 서비스 추가
vim core/services/{domain}/new_service.py

# __init__.py 업데이트
vim core/services/{domain}/__init__.py

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
from pathlib import Path

# 서드파티
import torch
from fastapi import FastAPI

# 프로젝트 모듈
# lawfirm_langgraph 경로 추가
lawfirm_langgraph_path = Path(__file__).parent.parent / "lawfirm_langgraph"
sys.path.insert(0, str(lawfirm_langgraph_path))

from source.services.workflow_service import LangGraphWorkflowService
from core.services.search import HybridSearchEngine
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
