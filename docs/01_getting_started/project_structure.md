# 📁 LawFirmAI 프로젝트 구조

## 개요

LawFirmAI는 **LangGraph 기반 법률 AI 어시스턴트**로, 명확한 계층 구조로 구성된 법률 AI 시스템입니다.

## 전체 구조

```
LawFirmAI/
├── lawfirm_langgraph/       # 핵심 LangGraph 워크플로우 시스템
│   ├── config/              # 설정 파일
│   ├── core/                # 핵심 비즈니스 로직
│   │   ├── workflow/        # LangGraph 워크플로우 (메인)
│   │   ├── agents/          # 레거시 에이전트 코드 (유틸리티)
│   │   ├── services/        # 비즈니스 서비스
│   │   ├── data/            # 데이터 레이어
│   │   ├── models/          # AI 모델
│   │   └── utils/           # 유틸리티
│   ├── tests/               # 테스트 코드
│   └── data/                # 데이터 파일
├── scripts/                 # 유틸리티 스크립트
│   ├── data_collection/     # 데이터 수집
│   ├── data_processing/     # 데이터 전처리
│   ├── database/            # 데이터베이스 관리
│   ├── ml_training/         # ML 모델 훈련
│   └── monitoring/          # 모니터링
├── data/                    # 데이터 파일
│   ├── raw/                 # 원본 데이터
│   ├── processed/           # 전처리된 데이터
│   ├── embeddings/          # 벡터 임베딩
│   └── database/            # 데이터베이스 파일
├── monitoring/              # 모니터링 시스템
│   ├── grafana/             # Grafana 설정
│   └── prometheus/          # Prometheus 설정
├── docs/                    # 프로젝트 문서
└── README.md                # 프로젝트 문서
```

## 📦 lawfirm_langgraph 모듈

### lawfirm_langgraph/core/workflow/ - LangGraph 워크플로우 (메인)
**역할**: AI 워크플로우 관리 및 실행

```
lawfirm_langgraph/core/workflow/
├── workflow_service.py              # 워크플로우 서비스 (메인)
├── legal_workflow_enhanced.py       # 법률 워크플로우 구현
├── nodes/                           # 워크플로우 노드
│   ├── answer_nodes.py              # 답변 생성 노드
│   ├── classification_nodes.py      # 분류 노드
│   ├── search_nodes.py              # 검색 노드
│   ├── routing_nodes.py             # 라우팅 노드
│   ├── node_wrappers.py            # 노드 래퍼
│   └── ...
├── state/                           # 상태 정의 및 관리
│   ├── state_definitions.py         # 상태 정의
│   ├── modular_states.py            # 모듈화된 상태 구조
│   ├── state_helpers.py             # 상태 헬퍼 함수
│   ├── state_utils.py               # 상태 유틸리티
│   ├── state_reduction.py           # 상태 최적화
│   └── ...
├── tools/                           # Agentic AI Tools
│   └── legal_search_tools.py        # 법률 검색 도구
├── builders/                        # 체인 빌더
│   ├── chain_builders.py            # 체인 빌더
│   ├── prompt_builders.py           # 프롬프트 빌더
│   └── ...
├── mixins/                          # 워크플로우 믹스인
│   ├── answer_generation_mixin.py   # 답변 생성 믹스인
│   ├── classification_mixin.py      # 분류 믹스인
│   ├── search_mixin.py              # 검색 믹스인
│   └── ...
└── utils/                           # 워크플로우 유틸리티
    ├── workflow_constants.py        # 워크플로우 상수
    ├── workflow_logger.py           # 워크플로우 로거
    └── ...
```

**사용 예시**:
```python
from lawfirm_langgraph.config.langgraph_config import LangGraphConfig
from lawfirm_langgraph.core.workflow.workflow_service import LangGraphWorkflowService

config = LangGraphConfig.from_env()
workflow = LangGraphWorkflowService(config)
result = await workflow.process_query_async("질문", "session_id")
```

### lawfirm_langgraph/core/agents/ - 레거시 에이전트 코드
**역할**: 레거시 코드 및 유틸리티 (하위 호환성 유지)

**참고**: 새로운 코드는 `core/workflow/`를 사용하세요. 이 디렉토리는 하위 호환성을 위해 유지됩니다.

```
lawfirm_langgraph/core/agents/
├── handlers/                        # 핸들러 모듈 (레거시)
│   ├── answer_formatter.py
│   ├── answer_generator.py
│   ├── classification_handler.py
│   ├── context_builder.py
│   ├── direct_answer_handler.py
│   └── search_handler.py
├── keyword_mapper.py                # 키워드 매퍼
├── legal_data_connector_v2.py       # 데이터 커넥터 (v2)
├── optimizers/                      # 최적화 모듈
│   ├── performance_optimizer.py     # 성능 최적화
│   └── query_optimizer.py           # 쿼리 최적화
└── validators/                      # 검증 모듈
    └── quality_validators.py
```

### lawfirm_langgraph/core/services/ - 비즈니스 서비스
**역할**: 검색, 답변 생성, 품질 개선 등

```
lawfirm_langgraph/core/services/
├── hybrid_search_engine.py          # 하이브리드 검색
├── semantic_search_engine.py         # 의미적 검색
├── exact_search_engine.py            # 정확한 매칭
├── precedent_search_engine.py         # 판례 검색
├── question_classifier.py            # 질문 분류
├── answer_generator.py                # 답변 생성
├── context_builder.py                 # 컨텍스트 빌더
├── confidence_calculator.py          # 신뢰도 계산
├── gemini_client.py                   # Gemini 클라이언트
├── unified_prompt_manager.py          # 통합 프롬프트 관리
└── ... (70+ 서비스 파일)
```

**사용 예시**:
```python
from lawfirm_langgraph.core.services.hybrid_search_engine import HybridSearchEngine

engine = HybridSearchEngine()
results = engine.search("계약 해지", question_type="law_inquiry")
```

### lawfirm_langgraph/core/data/ - 데이터 레이어
**역할**: 데이터베이스 및 벡터 스토어 관리

```
lawfirm_langgraph/core/data/
├── database.py                     # SQLite 데이터베이스
├── vector_store.py                  # FAISS 벡터 스토어
├── data_processor.py                # 데이터 처리
├── conversation_store.py            # 대화 저장소
├── legal_term_normalizer.py         # 법률 용어 정규화
├── assembly_playwright_client.py    # Assembly 데이터 수집
└── versioned_schema.py              # 버전 관리 스키마
```

### lawfirm_langgraph/core/models/ - AI 모델
**역할**: AI 모델 관리

```
lawfirm_langgraph/core/models/
└── sentence_bert.py                 # Sentence BERT 임베딩 모델
```

### lawfirm_langgraph/core/utils/ - 유틸리티
**역할**: 설정 및 유틸리티 함수

```
lawfirm_langgraph/core/utils/
├── langgraph_config.py              # LangGraph 설정
├── langchain_config.py              # LangChain 설정
├── logger.py                         # 로깅
├── config.py                        # 일반 설정
├── ollama_client.py                  # Ollama 클라이언트
└── ... (기타 유틸리티)
```

### lawfirm_langgraph/config/ - 설정
**역할**: 설정 관리

```
lawfirm_langgraph/config/
├── langgraph_config.py               # LangGraph 설정
└── app_config.py                     # 애플리케이션 설정
```

## 📊 데이터 흐름

### 1. 쿼리 처리
```
User Input
    ↓
lawfirm_langgraph/core/workflow/workflow_service.py
    ↓
lawfirm_langgraph/core/workflow/legal_workflow_enhanced.py (LangGraph 워크플로우)
    ├── nodes/classification_nodes.py (질문 분류)
    ├── nodes/search_nodes.py (문서 검색)
    └── nodes/answer_nodes.py (답변 생성)
    ↓
lawfirm_langgraph/core/services/ (검색, 생성, 품질 개선)
    ├── hybrid_search_engine.py
    ├── answer_generator.py
    └── confidence_calculator.py
    ↓
User Output
```

### 2. 검색 프로세스
```
Query
    ↓
lawfirm_langgraph/core/services/question_classifier.py (분류)
    ↓
lawfirm_langgraph/core/services/hybrid_search_engine.py (검색)
    ├── semantic_search_engine.py
    └── exact_search_engine.py
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
from lawfirm_langgraph.config.langgraph_config import LangGraphConfig
from lawfirm_langgraph.core.workflow.workflow_service import LangGraphWorkflowService
from lawfirm_langgraph.core.workflow.legal_workflow_enhanced import EnhancedLegalQuestionWorkflow
from lawfirm_langgraph.core.services.hybrid_search_engine import HybridSearchEngine
from lawfirm_langgraph.core.services.answer_generator import AnswerGenerator
```

## 📚 scripts 모듈

### scripts/data_collection/
**역할**: 법률 데이터 수집
- Assembly 데이터 수집
- 국가법령정보센터 API 연동

### scripts/data_processing/
**역할**: 데이터 전처리 및 임베딩 생성
- 텍스트 정리 및 청킹
- 벡터 임베딩 생성
- FAISS 인덱스 구축

### scripts/database/
**역할**: 데이터베이스 관리
- 스키마 생성 및 마이그레이션
- 데이터 검증

### scripts/ml_training/
**역할**: ML 모델 훈련
- 모델 파인튜닝
- 성능 평가

## 🎯 모듈별 책임

| 모듈 | 책임 | 의존성 |
|------|------|--------|
| `lawfirm_langgraph/core/workflow/` | 워크플로우 관리 및 실행 | services, models, data |
| `lawfirm_langgraph/core/agents/` | 레거시 코드 및 유틸리티 | services, models, data |
| `lawfirm_langgraph/core/services/` | 비즈니스 로직 | data, models |
| `lawfirm_langgraph/core/data/` | 데이터 관리 | - |
| `lawfirm_langgraph/core/models/` | AI 모델 | - |
| `lawfirm_langgraph/core/utils/` | 유틸리티 | - |
| `scripts/` | 스크립트 및 유틸리티 | core 모듈 |
| `data/` | 데이터 파일 | - |

## 🚀 개발 워크플로우

### 1. 기능 추가
```bash
# 새 서비스 추가
vim lawfirm_langgraph/core/services/new_service.py

# __init__.py 업데이트
vim lawfirm_langgraph/core/services/__init__.py

# 테스트 작성
vim lawfirm_langgraph/tests/test_new_service.py

# 테스트 실행
pytest lawfirm_langgraph/tests/test_new_service.py -v
```

### 2. 디버깅
```python
# 로깅 활성화
import logging
from lawfirm_langgraph.core.utils.logger import setup_logging

setup_logging(level=logging.DEBUG)

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
cd lawfirm_langgraph/tests
python run_all_tests.py

# 특정 테스트
pytest lawfirm_langgraph/tests/test_workflow_service.py -v
pytest lawfirm_langgraph/tests/test_integration.py -v
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
from lawfirm_langgraph.core.workflow import LangGraphWorkflowService
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

## 📖 추가 정보

- [프로젝트 개요](project_overview.md)
- [아키텍처](architecture.md)
- [LangGraph 워크플로우 가이드](../03_rag_system/langgraph_integration_guide.md)