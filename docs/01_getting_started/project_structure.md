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
│   │   ├── search/          # 검색 시스템
│   │   │   ├── engines/     # 검색 엔진
│   │   │   ├── handlers/    # 검색 핸들러
│   │   │   ├── processors/  # 검색 결과 처리
│   │   │   └── optimizers/  # 검색 최적화
│   │   ├── generation/      # 답변 생성 시스템
│   │   │   ├── generators/  # 답변 생성기
│   │   │   ├── formatters/  # 답변 포맷터
│   │   │   └── validators/  # 답변 검증기
│   │   ├── classification/  # 분류 시스템
│   │   │   ├── classifiers/ # 분류기
│   │   │   ├── handlers/   # 분류 핸들러
│   │   │   └── analyzers/   # 분석기
│   │   ├── processing/      # 데이터 처리
│   │   │   ├── extractors/  # 추출기
│   │   │   ├── processors/  # 프로세서
│   │   │   └── parsers/     # 파서
│   │   ├── conversation/    # 대화 관리
│   │   ├── agents/          # 에이전트 및 유틸리티
│   │   ├── services/        # 비즈니스 서비스
│   │   ├── data/            # 데이터 레이어
│   │   ├── shared/          # 공유 유틸리티
│   │   └── utils/           # 유틸리티
│   ├── tests/               # 테스트 코드
│   └── data/                # 데이터 파일
├── api/                     # FastAPI 애플리케이션
│   ├── main.py             # FastAPI 메인 애플리케이션
│   ├── routers/            # API 라우터
│   │   ├── chat.py         # 채팅 엔드포인트
│   │   ├── session.py     # 세션 관리
│   │   ├── history.py      # 히스토리
│   │   ├── feedback.py     # 피드백
│   │   ├── health.py       # 헬스체크
│   │   └── auth.py         # 인증 (OAuth2)
│   ├── services/           # API 서비스
│   ├── schemas/            # Pydantic 스키마
│   ├── middleware/         # 미들웨어
│   └── database/           # 데이터베이스 모델
├── frontend/                # React 프론트엔드
│   ├── src/                # React 소스 코드
│   ├── package.json         # npm 의존성
│   └── vite.config.ts      # Vite 설정
├── scripts/                 # 유틸리티 스크립트
│   ├── checks/              # 체크 스크립트
│   ├── ingest/              # 데이터 수집
│   ├── rag/                 # RAG 관련 스크립트
│   ├── tests/               # 테스트 스크립트
│   ├── tools/               # 도구 스크립트
│   ├── utils/               # 유틸리티 스크립트
│   ├── ml_training/         # ML 훈련 및 평가
│   ├── data_collection/     # 데이터 수집
│   ├── data_processing/     # 데이터 전처리
│   └── embedding/           # 임베딩 생성
├── data/                    # 데이터 파일
│   ├── raw/                 # 원본 데이터
│   ├── processed/          # 전처리된 데이터
│   ├── embeddings/          # 벡터 임베딩
│   └── vector_store/        # FAISS 벡터 스토어
├── docs/                    # 프로젝트 문서
├── deployment/               # 배포 스크립트
├── monitoring/               # 모니터링 설정
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


### lawfirm_langgraph/core/search/ - 검색 시스템
**역할**: 하이브리드 검색 엔진 및 검색 결과 처리

```
lawfirm_langgraph/core/search/
├── engines/                          # 검색 엔진
│   ├── hybrid_search_engine_v2.py    # 하이브리드 검색 (메인)
│   ├── semantic_search_engine_v2.py  # 의미적 검색 (FAISS)
│   ├── exact_search_engine_v2.py    # 정확 매칭 검색 (FTS5)
│   ├── keyword_search_engine.py      # 키워드 검색
│   └── precedent_search_engine.py    # 판례 검색
├── handlers/                         # 검색 핸들러
│   └── search_handler.py            # 검색 핸들러
├── processors/                       # 검색 결과 처리
│   ├── result_merger.py             # 결과 병합
│   ├── result_ranker.py             # 결과 순위 결정
│   └── search_result_processor.py    # 검색 결과 프로세서
└── optimizers/                       # 검색 최적화
    ├── legal_query_optimizer.py     # 쿼리 최적화
    ├── keyword_mapper.py            # 키워드 매핑
    └── query_enhancer.py            # 쿼리 강화
```

**사용 예시**:
```python
from lawfirm_langgraph.core.search.engines.hybrid_search_engine_v2 import HybridSearchEngineV2

engine = HybridSearchEngineV2()
results = engine.search("계약 해지", k=10)
```

### lawfirm_langgraph/core/generation/ - 답변 생성 시스템
**역할**: 답변 생성, 포맷팅, 검증

```
lawfirm_langgraph/core/generation/
├── generators/                       # 답변 생성기
│   ├── answer_generator.py           # 답변 생성기
│   ├── context_builder.py            # 컨텍스트 빌더
│   └── direct_answer_handler.py      # 직접 답변 핸들러
├── formatters/                       # 답변 포맷터
│   ├── answer_structure_enhancer.py # 답변 구조 강화
│   ├── legal_citation_enhancer.py   # 법률 인용 강화
│   └── unified_source_formatter.py  # 통합 소스 포맷터
└── validators/                       # 답변 검증기
    ├── answer_quality_enhancer.py   # 답변 품질 강화
    ├── confidence_calculator.py     # 신뢰도 계산
    ├── legal_basis_validator.py     # 법적 근거 검증
    └── quality_validators.py        # 품질 검증기
```

### lawfirm_langgraph/core/classification/ - 분류 시스템
**역할**: 질문 분류 및 분석

```
lawfirm_langgraph/core/classification/
├── classifiers/                      # 분류기
│   ├── question_classifier.py      # 질문 분류기
│   └── domain_classifier.py        # 도메인 분류기
├── handlers/                        # 분류 핸들러
│   └── classification_handler.py   # 분류 핸들러
└── analyzers/                       # 분석기
    └── query_analyzer.py           # 쿼리 분석기
```

### lawfirm_langgraph/core/processing/ - 데이터 처리
**역할**: 데이터 추출, 처리, 파싱

```
lawfirm_langgraph/core/processing/
├── extractors/                      # 추출기
│   ├── query_extractor.py          # 쿼리 추출기
│   ├── document_extractor.py       # 문서 추출기
│   ├── reasoning_extractor.py      # 추론 추출기
│   └── ... (기타 추출기)
├── processors/                     # 프로세서
│   └── data_processor.py           # 데이터 프로세서
├── parsers/                        # 파서
│   ├── query_parser.py             # 쿼리 파서
│   ├── answer_parser.py            # 답변 파서
│   └── response_parsers.py         # 응답 파서
└── integration/                    # 통합 시스템
    └── term_integration_system.py  # 용어 통합 시스템
```

### lawfirm_langgraph/core/conversation/ - 대화 관리
**역할**: 대화 이력 및 세션 관리

```
lawfirm_langgraph/core/conversation/
├── conversation_manager.py          # 대화 관리자
├── conversation_flow_tracker.py     # 대화 흐름 추적
├── multi_turn_handler.py           # 멀티턴 핸들러
└── contextual_memory_manager.py    # 컨텍스트 메모리 관리
```

### lawfirm_langgraph/core/services/ - 비즈니스 서비스
**역할**: 통합 서비스 및 유틸리티

```
lawfirm_langgraph/core/services/
├── gemini_client.py                 # Gemini 클라이언트
├── unified_prompt_manager.py         # 통합 프롬프트 관리
├── chat_service.py                   # 채팅 서비스
├── context_manager.py                 # 컨텍스트 관리
├── context_compressor.py              # 컨텍스트 압축
├── legal_basis_validator.py           # 법적 근거 검증
├── prompt_optimizer.py                # 프롬프트 최적화
├── prompt_templates.py                # 프롬프트 템플릿
└── prompts/                           # 프롬프트 파일
```

### lawfirm_langgraph/core/shared/ - 공유 유틸리티
**역할**: 공유 유틸리티 및 헬퍼

```
lawfirm_langgraph/core/shared/
├── cache/                           # 캐싱 시스템
├── clients/                         # 클라이언트
├── monitoring/                      # 모니터링
├── utils/                           # 유틸리티
├── wrappers/                        # 래퍼
├── feedback/                        # 피드백 시스템
└── profiles/                        # 프로파일 관리
```

**사용 예시**:
```python
from lawfirm_langgraph.core.search.engines.hybrid_search_engine_v2 import HybridSearchEngineV2

engine = HybridSearchEngineV2()
results = engine.search("계약 해지", k=10)
```

### lawfirm_langgraph/core/data/ - 데이터 레이어
**역할**: 데이터베이스 및 벡터 스토어 관리

```
lawfirm_langgraph/core/data/
├── database.py                     # SQLite 데이터베이스 (연결 풀링 지원)
├── vector_store.py                  # FAISS 벡터 스토어
├── data_processor.py                # 데이터 처리
├── conversation_store.py            # 대화 저장소
├── legal_term_normalizer.py         # 법률 용어 정규화
├── assembly_playwright_client.py    # Assembly 데이터 수집
├── versioned_schema.py              # 버전 관리 스키마
├── connection_pool.py               # 데이터베이스 연결 풀
├── db_adapter.py                    # 데이터베이스 어댑터
├── sql_adapter.py                    # SQL 어댑터
└── routers/                         # 데이터 라우터
```

### lawfirm_langgraph/core/models/ - AI 모델
**역할**: AI 모델 관리

```
lawfirm_langgraph/core/models/
└── sentence_bert.py                 # Sentence BERT 임베딩 모델
    (참고: 실제 모델 파일은 다른 위치에 있을 수 있음)
```

### lawfirm_langgraph/core/utils/ - 유틸리티
**역할**: 설정 및 유틸리티 함수

```
lawfirm_langgraph/core/utils/
├── langgraph_config.py              # LangGraph 설정
├── logger.py                         # 로깅
├── config.py                        # 일반 설정
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
User Input (API Request)
    ↓
api/routers/chat.py
    ↓
api/services/chat_service.py
    ↓
lawfirm_langgraph/core/workflow/workflow_service.py
    ↓
lawfirm_langgraph/core/workflow/legal_workflow_enhanced.py (LangGraph 워크플로우)
    ├── nodes/classification_nodes.py (질문 분류)
    │   └── core/classification/classifiers/question_classifier.py
    ├── nodes/search_nodes.py (문서 검색)
    │   └── core/search/engines/hybrid_search_engine_v2.py
    │       ├── semantic_search_engine_v2.py (의미적 검색)
    │       └── exact_search_engine_v2.py (정확 매칭 검색)
    ├── core/search/processors/result_merger.py (결과 병합)
    └── nodes/answer_nodes.py (답변 생성)
        └── core/generation/generators/answer_generator.py
    ↓
lawfirm_langgraph/core/generation/ (답변 생성 및 검증)
    ├── generators/answer_generator.py
    ├── formatters/answer_structure_enhancer.py
    └── validators/quality_validators.py
    ↓
API Response (User Output)
```

### 2. 검색 프로세스
```
Query
    ↓
lawfirm_langgraph/core/classification/classifiers/question_classifier.py (분류)
    ↓
lawfirm_langgraph/core/search/engines/hybrid_search_engine_v2.py (하이브리드 검색)
    ├── semantic_search_engine_v2.py (FAISS 벡터 검색)
    └── exact_search_engine_v2.py (FTS5 키워드 검색)
    ↓
lawfirm_langgraph/core/search/processors/result_merger.py (결과 병합)
    ↓
lawfirm_langgraph/core/search/processors/result_ranker.py (결과 순위 결정)
    ↓
lawfirm_langgraph/core/search/processors/search_result_processor.py (검색 결과 처리)
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
# 설정
from lawfirm_langgraph.config.langgraph_config import LangGraphConfig

# 워크플로우
from lawfirm_langgraph.core.workflow.workflow_service import LangGraphWorkflowService
from lawfirm_langgraph.core.workflow.legal_workflow_enhanced import EnhancedLegalQuestionWorkflow

# 검색
from lawfirm_langgraph.core.search.engines.hybrid_search_engine_v2 import HybridSearchEngineV2
from lawfirm_langgraph.core.search.engines.semantic_search_engine_v2 import SemanticSearchEngineV2
from lawfirm_langgraph.core.search.processors.result_merger import ResultMerger

# 답변 생성
from lawfirm_langgraph.core.generation.generators.answer_generator import AnswerGenerator
from lawfirm_langgraph.core.generation.validators.quality_validators import AnswerValidator

# 분류
from lawfirm_langgraph.core.classification.classifiers.question_classifier import QuestionClassifier

# 데이터
from lawfirm_langgraph.core.data.database import DatabaseManager
from lawfirm_langgraph.core.data.vector_store import VectorStore
```

## 📚 scripts 모듈

### scripts/checks/
**역할**: 상태 확인 스크립트
- `check_embedding_status.py`: 임베딩 상태 확인
- `check_re_embedding_progress.py`: 재임베딩 진행 상황 확인

### scripts/ingest/
**역할**: 데이터 수집 및 처리
- `ingest_cases.py`: 판례 데이터 수집
- `check_ingestion_progress.py`: 수집 진행 상황 확인
- `check_ip_data_format.py`: 데이터 형식 확인

### scripts/rag/
**역할**: RAG 관련 스크립트
- `build_index.py`: FAISS 인덱스 구축
- `build_ko_legal_sbert_index.py`: 한국어 법률 SBERT 인덱스 구축
- `mlflow_manager.py`: MLflow 모델 관리
- `analyze_search_source_type_mismatch.py`: 검색 소스 타입 불일치 분석
- `verify_source_type_consistency.py`: 소스 타입 일관성 검증

### scripts/tests/
**역할**: 테스트 스크립트
- `unit/`: 단위 테스트
- `integration/`: 통합 테스트
- `functional/`: 기능 테스트

### scripts/tools/
**역할**: 도구 스크립트
- `builds/`: 빌드 관련 스크립트
- `checks/`: 체크 관련 스크립트
- `wait_and_build_faiss_index.py`: FAISS 인덱스 대기 및 구축

### scripts/utils/
**역할**: 유틸리티 스크립트
- `embedding_version_manager.py`: 임베딩 버전 관리
- `embeddings.py`: 임베딩 유틸리티

### scripts/ml_training/
**역할**: ML 훈련 및 평가
- `evaluation/`: 검색 성능 평가 스크립트
- Ground Truth 생성 및 RAG 평가

### scripts/data_collection/
**역할**: 데이터 수집
- AI허브 데이터 수집 스크립트

### scripts/data_processing/
**역할**: 데이터 전처리
- 증분 전처리 파이프라인
- Q&A 데이터셋 생성

### scripts/embedding/
**역할**: 임베딩 생성
- 벡터 임베딩 생성 스크립트

## 🎯 모듈별 책임

| 모듈 | 책임 | 의존성 |
|------|------|--------|
| `lawfirm_langgraph/core/workflow/` | 워크플로우 관리 및 실행 | search, generation, classification, data |
| `lawfirm_langgraph/core/search/` | 검색 엔진 및 결과 처리 | data, models |
| `lawfirm_langgraph/core/generation/` | 답변 생성 및 검증 | search, services |
| `lawfirm_langgraph/core/classification/` | 질문 분류 및 분석 | - |
| `lawfirm_langgraph/core/processing/` | 데이터 처리 및 추출 | - |
| `lawfirm_langgraph/core/conversation/` | 대화 관리 및 세션 | data |
| `lawfirm_langgraph/core/services/` | 통합 서비스 | data, models |
| `lawfirm_langgraph/core/data/` | 데이터 관리 | - |
| `lawfirm_langgraph/core/shared/` | 공유 유틸리티 | - |
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