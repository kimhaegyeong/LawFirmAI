# LawFirmAI 서비스 아키텍처

## 개요

LawFirmAI의 서비스 아키텍처는 **lawfirm_langgraph 모듈 기반**의 모듈화된 서비스로 구성되어 있으며, LangGraph 워크플로우 기반의 지능형 대화 시스템을 지원합니다.

## 아키텍처 개요

### 전체 아키텍처

```
┌─────────────────────────────────────────────────────────────┐
│                    LangGraph 워크플로우                      │
│  ├── lawfirm_langgraph/core/workflow/workflow_service.py   │
│  ├── lawfirm_langgraph/core/workflow/legal_workflow_enhanced.py│
│  ├── lawfirm_langgraph/core/workflow/nodes/               │
│  │   ├── classification_nodes.py      (질문 분류 노드)     │
│  │   ├── search_nodes.py             (검색 노드)         │
│  │   └── answer_nodes.py              (답변 생성 노드)     │
│  └── lawfirm_langgraph/core/workflow/state/                │
│      ├── state_definitions.py         (상태 정의)         │
│      └── modular_states.py            (모듈화된 상태)      │
├─────────────────────────────────────────────────────────────┤
│                    검색 레이어                              │
│  ├── lawfirm_langgraph/core/search/engines/                │
│  │   ├── hybrid_search_engine_v2.py       (하이브리드 검색) │
│  │   ├── semantic_search_engine_v2.py    (의미적 검색)    │
│  │   ├── exact_search_engine_v2.py       (정확 매칭)      │
│  │   └── keyword_search_engine.py        (키워드 검색)    │
│  ├── lawfirm_langgraph/core/search/handlers/               │
│  │   └── search_handler.py               (검색 핸들러)    │
│  ├── lawfirm_langgraph/core/search/processors/             │
│  │   ├── result_merger.py                 (결과 병합)      │
│  │   ├── result_ranker.py                 (결과 순위 결정) │
│  │   └── search_result_processor.py       (검색 결과 처리) │
│  └── lawfirm_langgraph/core/search/optimizers/             │
│      ├── legal_query_optimizer.py          (쿼리 최적화)    │
│      └── keyword_mapper.py                (키워드 매핑)    │
├─────────────────────────────────────────────────────────────┤
│                    답변 생성 레이어                         │
│  ├── lawfirm_langgraph/core/generation/generators/          │
│  │   ├── answer_generator.py              (답변 생성)      │
│  │   └── context_builder.py               (컨텍스트 구축)   │
│  ├── lawfirm_langgraph/core/generation/formatters/          │
│  │   ├── answer_structure_enhancer.py     (답변 구조 강화) │
│  │   └── legal_citation_enhancer.py      (법률 인용 강화) │
│  └── lawfirm_langgraph/core/generation/validators/         │
│      ├── quality_validators.py            (품질 검증)      │
│      └── confidence_calculator.py         (신뢰도 계산)    │
├─────────────────────────────────────────────────────────────┤
│                    분류 레이어                              │
│  ├── lawfirm_langgraph/core/classification/classifiers/    │
│  │   ├── question_classifier.py           (질문 분류)      │
│  │   └── domain_classifier.py             (도메인 분류)    │
│  └── lawfirm_langgraph/core/classification/handlers/        │
│      └── classification_handler.py        (분류 핸들러)    │
├─────────────────────────────────────────────────────────────┤
│                    데이터 레이어                            │
│  ├── lawfirm_langgraph/core/data/database.py               │
│  ├── lawfirm_langgraph/core/data/vector_store.py          │
│  └── lawfirm_langgraph/core/data/conversation_store.py    │
├─────────────────────────────────────────────────────────────┤
│                    대화 관리 레이어                         │
│  ├── lawfirm_langgraph/core/conversation/                  │
│  │   ├── conversation_manager.py           (대화 관리)      │
│  │   └── multi_turn_handler.py            (멀티턴 처리)    │
├─────────────────────────────────────────────────────────────┤
│                    AI 모델 레이어                            │
│  ├── lawfirm_langgraph/core/services/gemini_client.py     │
│  └── lawfirm_langgraph/core/models/sentence_bert.py        │
└─────────────────────────────────────────────────────────────┘
```

### 모듈별 책임

| 모듈 | 책임 | 주요 컴포넌트 |
|------|------|-------------|
| **lawfirm_langgraph/core/workflow/** | LangGraph 워크플로우 관리 | workflow_service, legal_workflow_enhanced, nodes, state |
| **lawfirm_langgraph/core/search/** | 검색 엔진 및 처리 | hybrid_search_engine_v2, semantic_search_engine_v2, exact_search_engine_v2, result_merger, result_ranker |
| **lawfirm_langgraph/core/generation/** | 답변 생성 및 검증 | answer_generator, context_builder, quality_validators, answer_structure_enhancer |
| **lawfirm_langgraph/core/classification/** | 분류 시스템 | question_classifier, domain_classifier, classification_handler |
| **lawfirm_langgraph/core/processing/** | 데이터 처리 | query_extractor, document_extractor, response_parsers |
| **lawfirm_langgraph/core/conversation/** | 대화 관리 | conversation_manager, multi_turn_handler, conversation_flow_tracker |
| **lawfirm_langgraph/core/services/** | 통합 서비스 | gemini_client, unified_prompt_manager |
| **lawfirm_langgraph/core/data/** | 데이터 관리 | database, vector_store, conversation_store |
| **lawfirm_langgraph/core/shared/** | 공유 유틸리티 | cache, clients, monitoring, utils |
| **lawfirm_langgraph/config/** | 설정 관리 | langgraph_config, app_config |

## 핵심 서비스

### 1. LangGraph 워크플로우 서비스

**파일**: `lawfirm_langgraph/core/workflow/workflow_service.py`

**역할**: LangGraph 기반 법률 질문 처리 워크플로우 관리

**주요 기능**:
- 질문 처리 워크플로우 실행
- 상태 관리 및 최적화
- 세션 관리
- Agentic AI 모드 지원

**사용 예시**:
```python
from lawfirm_langgraph.config.langgraph_config import LangGraphConfig
from lawfirm_langgraph.core.workflow.workflow_service import LangGraphWorkflowService

config = LangGraphConfig.from_env()
workflow = LangGraphWorkflowService(config)
result = await workflow.process_query_async("질문", "session_id")
```

### 2. 하이브리드 검색 엔진

**파일**: `lawfirm_langgraph/core/search/engines/hybrid_search_engine_v2.py`

**역할**: 의미적 검색 + 키워드 검색 통합

**검색 방식**:
- 의미적 검색 (FAISS 벡터, SemanticSearchEngineV2)
- 키워드 검색 (FTS5, ExactSearchEngineV2)
- 하이브리드 병합
- Keyword Coverage 기반 동적 가중치 조정

**사용 예시**:
```python
from lawfirm_langgraph.core.search.engines.hybrid_search_engine_v2 import HybridSearchEngineV2

engine = HybridSearchEngineV2()
results = engine.search("계약 해지", k=10)
```

### 3. 의미적 검색 엔진

**파일**: `lawfirm_langgraph/core/search/engines/semantic_search_engine_v2.py`

**역할**: FAISS 벡터 기반 의미적 유사도 검색 (IndexIVFPQ 지원)

**기능**:
- 벡터 임베딩 생성
- 유사도 검색
- 결과 랭킹

### 4. 정확 매칭 검색 엔진

**파일**: `lawfirm_langgraph/core/search/engines/exact_search_engine_v2.py`

**역할**: FTS5 기반 정확한 매칭 검색

**기능**:
- FTS5 풀텍스트 검색
- 키워드 매칭
- 검색 결과 랭킹

### 5. 질문 분류기

**파일**: `lawfirm_langgraph/core/classification/classifiers/question_classifier.py`

**역할**: 의미적 도메인 분류 및 처리 전략 결정

**분류 유형**:
- 법령 조문 문의 (law_inquiry)
- 판례 검색 (precedent_search)
- 절차 문의 (procedure_inquiry)
- 일반 질문 (general_question)
- 법률 자문 (legal_advice)

### 6. 검색 결과 처리 및 순위 결정

**파일**: `lawfirm_langgraph/core/search/processors/result_merger.py`

**역할**: 검색 결과 병합 및 순위 결정

**기능**:
- 검색 결과 병합 (ResultMerger)
- 다단계 재순위화 (ResultRanker)
- Keyword Coverage 평가
- 의미 기반 키워드 매칭 (선택적 실행, 모델 캐싱)
- 배치 임베딩 생성 (성능 최적화)

**사용 예시**:
```python
from lawfirm_langgraph.core.search.processors.result_merger import ResultMerger, ResultRanker

merger = ResultMerger()
ranker = ResultRanker()

# 검색 결과 병합
merged = merger.merge_results(exact_results, semantic_results, weights, query)

# 순위 결정 및 Keyword Coverage 평가
ranked = ranker.rank_results(merged, top_k=20, query=query)
quality = ranker.evaluate_search_quality(query, ranked, query_type, extracted_keywords)
```

### 7. 답변 생성기

**파일**: `lawfirm_langgraph/core/generation/generators/answer_generator.py`

**역할**: LLM 기반 답변 생성

**기능**:
- 컨텍스트 기반 답변 생성
- 법률 도메인 특화 프롬프트
- 스트리밍 지원

### 8. 답변 품질 검증기

**파일**: `lawfirm_langgraph/core/generation/validators/quality_validators.py`

**역할**: 답변 품질 검증 및 신뢰도 계산

**기능**:
- 답변 품질 평가
- 법적 근거 검증
- 신뢰도 계산

## 데이터 레이어

### 1. 데이터베이스 관리자

**파일**: `lawfirm_langgraph/core/data/database.py`

**기능**:
- SQLite 데이터베이스 관리
- 쿼리 최적화
- 연결 풀 관리

**주요 테이블**:
- `assembly_laws`: 법률 문서
- `assembly_articles`: 법률 조문
- `precedent_cases`: 판례 사건
- `precedent_sections`: 판례 섹션

### 2. 벡터 스토어

**파일**: `lawfirm_langgraph/core/data/vector_store.py`

**기능**:
- FAISS 벡터 인덱스 관리
- 임베딩 생성
- 유사도 검색

### 3. 대화 저장소

**파일**: `lawfirm_langgraph/core/data/conversation_store.py`

**기능**:
- 대화 데이터 저장
- 세션 관리
- 메타데이터 관리

## AI 모델 레이어

### 1. Sentence BERT

**파일**: `lawfirm_langgraph/core/models/sentence_bert.py`

**기능**:
- 텍스트 임베딩 생성
- 유사도 계산

### 2. Gemini 클라이언트

**파일**: `lawfirm_langgraph/core/services/gemini_client.py`

**기능**:
- Google Gemini API 통신
- 답변 생성
- 토큰 관리

## 데이터 흐름

### 1. 쿼리 처리 흐름

```
User Input
    ↓
lawfirm_langgraph/core/workflow/workflow_service.py
    ↓
lawfirm_langgraph/core/workflow/legal_workflow_enhanced.py (LangGraph 워크플로우)
    ├── nodes/classification_nodes.py (질문 분류)
    │   └── core/classification/classifiers/question_classifier.py
    ├── nodes/search_nodes.py (문서 검색)
    │   ├── core/search/engines/hybrid_search_engine_v2.py
    │   ├── core/search/engines/semantic_search_engine_v2.py
    │   └── core/search/engines/exact_search_engine_v2.py
    ├── core/search/processors/result_merger.py (결과 병합)
    └── nodes/answer_nodes.py (답변 생성)
        ├── core/generation/generators/answer_generator.py
        ├── core/generation/generators/context_builder.py
        └── core/generation/validators/quality_validators.py
    ↓
User Output
```

### 2. 검색 프로세스

```
Query
    ↓
lawfirm_langgraph/core/classification/classifiers/question_classifier.py (질문 분류)
    ↓
lawfirm_langgraph/core/search/engines/hybrid_search_engine_v2.py (하이브리드 검색)
    ├── core/search/engines/semantic_search_engine_v2.py (의미적 검색)
    └── core/search/engines/exact_search_engine_v2.py (정확 매칭 검색)
    ↓
lawfirm_langgraph/core/search/processors/result_merger.py (결과 병합)
    ↓
lawfirm_langgraph/core/search/processors/result_ranker.py (결과 순위 결정)
    ↓
lawfirm_langgraph/core/search/processors/search_result_processor.py (검색 결과 처리)
    ↓
Results
```

## 서비스 간 통신

### 1. 동기 통신

```python
# 직접 호출
from lawfirm_langgraph.core.search.engines.hybrid_search_engine_v2 import HybridSearchEngineV2
from lawfirm_langgraph.core.services.answer_generator import AnswerGenerator

search_engine = HybridSearchEngineV2()
answer_generator = AnswerGenerator()

results = search_engine.search("계약 해지", k=10)
answer = answer_generator.generate("계약 해지", results)
```

### 2. 비동기 통신

```python
import asyncio
from lawfirm_langgraph.core.workflow.workflow_service import LangGraphWorkflowService
from lawfirm_langgraph.config.langgraph_config import LangGraphConfig

async def process_query_async(query: str, session_id: str):
    config = LangGraphConfig.from_env()
    workflow = LangGraphWorkflowService(config)
    result = await workflow.process_query_async(query, session_id)
    return result
```

## 확장성 및 유지보수성

### 1. 모듈화 설계

**장점**:
- 독립적 개발 가능
- 테스트 용이성
- 재사용성

**구현**:
```python
# 인터페이스 정의
from abc import ABC, abstractmethod

class ServiceInterface(ABC):
    @abstractmethod
    def process(self, data: Any) -> Any:
        pass

# 서비스 구현
class ConcreteService(ServiceInterface):
    def process(self, data: Any) -> Any:
        return self._process_data(data)
```

### 2. 의존성 주입

```python
class WorkflowService:
    def __init__(self, 
                 search_engine: HybridSearchEngine,
                 answer_generator: AnswerGenerator,
                 confidence_calculator: ConfidenceCalculator):
        self.search_engine = search_engine
        self.answer_generator = answer_generator
        self.confidence_calculator = confidence_calculator
```

### 3. 설정 관리

```python
from lawfirm_langgraph.config.langgraph_config import LangGraphConfig
from lawfirm_langgraph.core.workflow.workflow_service import LangGraphWorkflowService

config = LangGraphConfig.from_env()
workflow = LangGraphWorkflowService(config)
```

## 성능 최적화

### 1. 메모리 최적화

- **State 최적화**: State reduction으로 메모리 효율성 향상
- **모델 캐싱**: SentenceTransformer 모델 클래스 변수로 캐싱 (약 7.5초 절약)
- **선택적 의미 기반 매칭**: Keyword Coverage 70% 이상 시 의미 기반 매칭 생략
- **배치 임베딩 생성**: 개별 생성 대신 배치로 처리 (batch_size=8)

### 2. 캐싱 전략

```python
from functools import lru_cache

@lru_cache(maxsize=128)
def cached_search(query: str):
    return search_engine.search(query)
```

### 3. 비동기 처리

```python
import asyncio

async def parallel_search(queries: List[str]):
    tasks = [search_engine.search_async(q) for q in queries]
    results = await asyncio.gather(*tasks)
    return results
```

## 보안 고려사항

### 1. 입력 검증

```python
from pydantic import BaseModel, Field, field_validator
from typing import Optional

class ChatRequest(BaseModel):
    message: str = Field(..., min_length=1, max_length=10000)
    session_id: Optional[str] = None
    
    @field_validator('message')
    @classmethod
    def validate_message(cls, v):
        if len(v) > 10000:
            raise ValueError('메시지가 너무 깁니다')
        return v.strip()
```

### 2. 데이터 보호

```python
import hashlib

class DataProtector:
    def hash_user_id(self, user_id: str) -> str:
        """사용자 ID 해싱"""
        return hashlib.sha256(user_id.encode()).hexdigest()
    
    def sanitize_input(self, text: str) -> str:
        """입력 데이터 정제"""
        return text.replace('<script>', '').replace('</script>', '')
```

## 테스트 전략

### 1. 단위 테스트

```python
import pytest
from unittest.mock import Mock

class TestHybridSearchEngine:
    def test_search(self):
        """검색 기능 테스트"""
        from lawfirm_langgraph.core.search.engines.hybrid_search_engine_v2 import HybridSearchEngineV2
        engine = HybridSearchEngineV2()
        results = engine.search("계약 해지", k=10)
        assert results is not None
        assert len(results) > 0
```

### 2. 통합 테스트

```python
from lawfirm_langgraph.core.workflow.workflow_service import LangGraphWorkflowService

class TestWorkflowIntegration:
    async def test_end_to_end(self):
        """전체 워크플로우 테스트"""
        config = LangGraphConfig.from_env()
        workflow = LangGraphWorkflowService(config)
        result = await workflow.process_query_async("계약 해지 조건은?", "session_123")
        assert result is not None
        assert "answer" in result
```

### 3. 성능 테스트

```python
import time
from lawfirm_langgraph.core.workflow.workflow_service import LangGraphWorkflowService

class TestPerformance:
    async def test_response_time(self):
        """응답 시간 테스트"""
        config = LangGraphConfig.from_env()
        workflow = LangGraphWorkflowService(config)
        
        start_time = time.time()
        result = await workflow.process_query_async("테스트 질문", "session_123")
        end_time = time.time()
        
        response_time = end_time - start_time
        assert response_time < 5.0  # 5초 이내
```

## 배포 고려사항

### 1. 환경별 설정

```python
from lawfirm_langgraph.config.langgraph_config import LangGraphConfig

config = LangGraphConfig.from_env()
```

### 2. 모니터링

```python
class HealthChecker:
    def check_health(self) -> Dict[str, str]:
        """서비스 상태 확인"""
        return {
            "database": "healthy" if self._check_database() else "unhealthy",
            "vector_store": "healthy" if self._check_vector_store() else "unhealthy",
            "models": "healthy" if self._check_models() else "unhealthy"
        }
```

## 📖 관련 문서

- [프로젝트 구조](project_structure.md)
- [프로젝트 개요](project_overview.md)
- [LangGraph 통합 가이드](../03_rag_system/langgraph_integration_guide.md)