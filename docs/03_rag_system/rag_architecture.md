# LawFirmAI RAG 시스템 가이드

## 개요

LawFirmAI 프로젝트의 LangChain 기반 RAG(Retrieval-Augmented Generation) 시스템 사용법을 설명합니다. 이 시스템은 법률 문서를 효율적으로 검색하고 AI 모델을 통해 정확한 답변을 생성하는 것을 목표로 합니다.

## 주요 기능

### 검색 시스템
- **의미적 검색 엔진**: FAISS 벡터 인덱스와 SentenceTransformer 모델 통합
- **데이터베이스 검색**: SQLite 기반 정확한 매칭 검색
- **실제 소스 제공**: 법률/판례 데이터베이스에서 실제 근거 자료 검색
- **하이브리드 검색**: 정확한 매칭과 의미적 검색 결과 병합

## 시스템 아키텍처

### 전체 구조

```
┌─────────────────────────────────────────────────────────────┐
│                    LangGraph RAG System                     │
├─────────────────────────────────────────────────────────────┤
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐        │
│  │   Client    │  │   API        │  │  React       │        │
│  │  Request    │  │  (FastAPI)   │  │  Frontend   │        │
│  └─────────────┘  └─────────────┘  └─────────────┘        │
│           │               │               │                │
│           └───────────────┼───────────────┘                │
│                           │                                │
│  ┌─────────────────────────────────────────────────────────┤
│  │              LangGraph Workflow                         │
│  │  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐    │
│  │  │ Workflow    │  │ Search      │  │ Generation  │    │
│  │  │ Service     │  │ Engine      │  │ Service     │    │
│  │  └─────────────┘  └─────────────┘  └─────────────┘    │
│  └─────────────────────────────────────────────────────────┤
│                           │                                │
│  ┌─────────────────────────────────────────────────────────┤
│  │              Data Layer                                 │
│  │  ┌─────────────┐  ┌─────────────┐                     │
│  │  │ Vector      │  │ Database    │                     │
│  │  │ Store       │  │ (SQLite)    │                     │
│  │  │ (FAISS)     │  └─────────────┘                     │
│  │  └─────────────┘                                     │
│  └─────────────────────────────────────────────────────────┘
└─────────────────────────────────────────────────────────────┘
```

### 핵심 컴포넌트

#### 1. LangGraph Workflow Service
**파일**: `lawfirm_langgraph/core/workflow/workflow_service.py`

- **역할**: LangGraph 기반 법률 질문 처리 워크플로우 오케스트레이션
- **주요 기능**:
  - 질문 처리 워크플로우 실행
  - 상태 관리 및 최적화
  - 세션 관리
  - Keyword Coverage 최적화 (평균 0.806)
  - 성능 최적화 (선택적 의미 기반 매칭, 배치 임베딩 생성)

#### 2. 검색 엔진
**파일**: `lawfirm_langgraph/core/search/engines/`

- **HybridSearchEngineV2**: 하이브리드 검색 (의미적 + 정확 매칭) - 메인 검색 엔진
- **SemanticSearchEngineV2**: FAISS 벡터 기반 의미적 검색
- **ExactSearchEngineV2**: SQLite FTS5 기반 정확한 매칭 검색
- **KeywordSearchEngine**: 키워드 기반 검색
- **PrecedentSearchEngine**: 판례 전용 검색 엔진
- **Keyword Coverage 기반 동적 가중치**: 검색 결과의 키워드 커버리지에 따라 가중치 조정

#### 3. 검색 결과 처리 및 순위 결정
**파일**: `lawfirm_langgraph/core/search/processors/`

- **ResultMerger**: 검색 결과 병합
- **ResultRanker**: 다단계 재순위화 및 Keyword Coverage 평가
- **SearchResultProcessor**: 검색 결과 처리 및 키워드 매칭 점수 계산
- **의미 기반 키워드 매칭**: SentenceTransformer를 활용한 의미적 유사도 기반 키워드 매칭 (선택적 실행)

#### 4. 데이터 레이어
**파일**: `lawfirm_langgraph/core/data/`

- **VectorStore**: FAISS 벡터 스토어 관리
- **Database**: SQLite 데이터베이스 관리
- **ConversationStore**: 대화 이력 저장

## 설치 및 설정

### 1. 필요한 패키지 설치

```bash
pip install langchain langfuse faiss-cpu sentence-transformers
pip install fastapi uvicorn
```

### 2. 환경 변수 설정

```bash
# LangChain 설정
export VECTOR_STORE_TYPE=faiss
export VECTOR_STORE_PATH=./data/embeddings/faiss_index
export EMBEDDING_MODEL=sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2
export CHUNK_SIZE=1000
export CHUNK_OVERLAP=200
export MAX_CONTEXT_LENGTH=4000

# LLM 설정
export LLM_PROVIDER=openai
export LLM_MODEL=gpt-3.5-turbo
export LLM_TEMPERATURE=0.7
export LLM_MAX_TOKENS=1000

# Google AI 설정 (LLM_PROVIDER=google인 경우)
export GOOGLE_API_KEY=your-google-api-key

# Langfuse 설정
export LANGFUSE_ENABLED=true
export LANGFUSE_SECRET_KEY=your-secret-key
export LANGFUSE_PUBLIC_KEY=your-public-key
export LANGFUSE_HOST=https://cloud.langfuse.com
export LANGFUSE_DEBUG=false

# 성능 설정
export ENABLE_CACHING=true
export CACHE_TTL=3600
export ENABLE_ASYNC=true
```

### 3. 설정 클래스 구조

```python
@dataclass
class LangChainConfig:
    # 벡터 저장소 설정
    vector_store_type: VectorStoreType
    vector_store_path: str
    embedding_model: str
    
    # 문서 처리 설정
    chunk_size: int
    chunk_overlap: int
    max_context_length: int
    
    # 검색 설정
    search_k: int
    similarity_threshold: float
    
    # LLM 설정
    llm_provider: LLMProvider
    llm_model: str
    llm_temperature: float
    llm_max_tokens: int
    
    # Langfuse 설정
    langfuse_enabled: bool
    langfuse_secret_key: Optional[str]
    langfuse_public_key: Optional[str]
    langfuse_host: str
    langfuse_debug: bool
```

## 기본 사용법

### 1. React 프론트엔드 실행

```bash
# React 개발 서버 시작
cd frontend
npm install
npm run dev
```

### 2. API 서버 실행

```bash
# FastAPI 서버 시작
cd api
python main.py
```

## 데이터 플로우

### 1. 쿼리 처리 플로우

```
사용자 쿼리 (frontend 또는 api)
    ↓
lawfirm_langgraph/core/workflow/workflow_service.py
    ↓
lawfirm_langgraph/core/workflow/legal_workflow_enhanced.py (LangGraph 워크플로우)
    ├── classify_query (질문 분류)
    ├── expand_keywords (키워드 확장 - LLM 기반)
    ├── retrieve_documents (문서 검색)
    │   ├── core/search/engines/hybrid_search_engine_v2.py
    │   ├── core/search/engines/semantic_search_engine_v2.py
    │   └── core/search/engines/exact_search_engine_v2.py
    ├── process_search_results_combined (검색 결과 처리)
    │   ├── core/search/processors/result_merger.py
    │   ├── core/search/processors/result_ranker.py
    │   └── core/search/processors/search_result_processor.py
    └── generate_answer (답변 생성)
        ├── core/generation/generators/answer_generator.py
        ├── core/generation/generators/context_builder.py
        └── core/generation/validators/quality_validators.py
    ↓
최종 응답 반환
```

### 2. 문서 추가 플로우

```
문서 입력
    ↓
lawfirm_langgraph/core/data/data_processor.py (문서 전처리)
    ↓
lawfirm_langgraph/core/data/vector_store.py (벡터 임베딩 생성)
    ↓
FAISS 인덱스 업데이트
    ↓
lawfirm_langgraph/core/data/database.py (메타데이터 저장)
    ↓
성공/실패 응답
```

## 기술 스택

### 핵심 라이브러리
- **LangGraph**: State 기반 워크플로우 관리
- **LangChain**: RAG 파이프라인 구축 및 체인 관리
- **FAISS**: 벡터 검색 및 유사도 계산
- **Sentence-Transformers**: 한국어 임베딩 모델 (snunlp/KR-SBERT-V40K-klueNLI-augSTS)
- **SQLite**: 정확한 매칭 검색 및 메타데이터 저장 (Python 내장 모듈)

### LLM 지원
- **Google Gemini 2.5 Flash Lite**: 클라우드 LLM 모델 (현재 사용 중)

### 벡터 저장소 옵션
- **FAISS**: 고성능 벡터 검색
- **Chroma**: 문서 중심 벡터 저장소
- **Pinecone**: 클라우드 벡터 데이터베이스

## 성능 최적화

### 1. 벡터 검색 최적화
- **인덱스 최적화**: FAISS 인덱스 타입 선택 (IndexIVFPQ 지원)
- **배치 처리**: 여러 쿼리 동시 처리
- **모델 캐싱**: SentenceTransformer 모델 클래스 변수로 캐싱 (약 7.5초 절약)
- **선택적 의미 기반 매칭**: Keyword Coverage 70% 이상 시 의미 기반 매칭 생략
- **배치 임베딩 생성**: 개별 생성 대신 배치로 처리 (batch_size=8)

### 2. 컨텍스트 관리 최적화
- **동적 길이 조절**: 쿼리 복잡도에 따른 컨텍스트 길이 조절
- **관련성 필터링**: Keyword Coverage 기반 문서 필터링
- **세션 캐싱**: 사용자 세션별 컨텍스트 캐싱
- **문서 손실 방지**: 10개 이하 문서 필터링 건너뛰기

### 3. LLM 호출 최적화
- **프롬프트 최적화**: 효율적인 프롬프트 템플릿
- **토큰 관리**: 최적의 토큰 사용량
- **비동기 처리**: 여러 LLM 호출 병렬 처리
- **타임아웃 최적화**: LLM 호출 타임아웃 3초로 단축
- **조건부 AI 확장**: 키워드 5개 이상, 쿼리 10자 이상일 때만 AI 확장

## 모니터링 및 디버깅

### 로깅 시스템
- **파일 로깅**: `logs/test/run_query_test_YYYYMMDD_HHMMSS.log`에 자동 저장
- **성능 메트릭**: 노드별 실행 시간 추적
- **Keyword Coverage 추적**: 검색 품질 메트릭 모니터링
- **메타데이터 정규화 로그**: 오타 필드명 자동 수정 추적

### 메트릭 수집
- **RAG 메트릭**: 검색 정확도, 응답 품질, 신뢰도, Keyword Coverage
- **성능 메트릭**: 응답 시간, 처리량, 리소스 사용량
- **사용자 메트릭**: 쿼리 패턴, 세션 길이, 만족도

## 확장성 고려사항

### 수평적 확장
- **마이크로서비스 아키텍처**: 각 컴포넌트를 독립적인 서비스로 분리
- **로드 밸런싱**: 여러 RAG 서비스 인스턴스 간 부하 분산
- **데이터베이스 샤딩**: 대용량 벡터 데이터 분산 저장

### 수직적 확장
- **GPU 가속**: 임베딩 및 LLM 추론 가속화
- **메모리 최적화**: 대용량 모델 로딩 및 캐싱
- **CPU 최적화**: 병렬 처리 및 벡터화 연산

## 보안 고려사항

### 데이터 보안
- **암호화**: 민감한 법률 문서 암호화 저장
- **접근 제어**: 사용자별 권한 관리
- **감사 로그**: 모든 접근 및 수정 기록

### API 보안
- **인증**: JWT 토큰 기반 인증
- **인가**: 역할 기반 접근 제어
- **Rate Limiting**: API 호출 제한

## 배포 전략

### 개발 환경
- **Docker Compose**: 로컬 개발 환경 구성
- **Hot Reload**: 코드 변경 시 자동 재시작
- **디버깅 도구**: Langfuse 디버깅 대시보드

### 프로덕션 환경
- **Kubernetes**: 컨테이너 오케스트레이션
- **CI/CD**: 자동화된 배포 파이프라인
- **모니터링**: Prometheus + Grafana 스택

## 하이브리드 검색 시스템

### 검색 엔진 구성

#### 1. 의미적 검색 엔진
**파일**: `lawfirm_langgraph/core/search/engines/semantic_search_engine_v2.py`

```python
from lawfirm_langgraph.core.search.engines.semantic_search_engine_v2 import SemanticSearchEngineV2

engine = SemanticSearchEngineV2()
results = engine.search("계약 해지", k=5)
```

**주요 기능**:
- FAISS 벡터 인덱스 기반 검색
- Sentence-BERT 모델을 사용한 의미적 유사도 계산 (snunlp/KR-SBERT-V40K-klueNLI-augSTS)
- 메타데이터 정규화 (오타 필드명 자동 수정)

#### 2. 정확 매칭 검색 엔진
**파일**: `lawfirm_langgraph/core/search/engines/exact_search_engine_v2.py`

```python
from lawfirm_langgraph.core.search.engines.exact_search_engine_v2 import ExactSearchEngineV2

engine = ExactSearchEngineV2()
results = engine.search("민법 제543조", k=5)
```

**주요 기능**:
- SQLite FTS5 기반 키워드 검색
- 법령명, 조문번호 등 정확한 매칭
- 빠른 응답 시간 (< 100ms)

#### 3. 하이브리드 검색 엔진
**파일**: `lawfirm_langgraph/core/search/engines/hybrid_search_engine_v2.py`

```python
from lawfirm_langgraph.core.search.engines.hybrid_search_engine_v2 import HybridSearchEngineV2

engine = HybridSearchEngineV2()
results = engine.search("계약 해지", question_type="law_inquiry")
```

**주요 기능**:
- 의미적 검색과 정확 매칭 검색 결과 통합
- 가중 평균을 통한 결과 재순위화
- 질문 유형별 동적 가중치 조정
- Keyword Coverage 기반 동적 가중치 조정

## LangGraph 기반 RAG 시스템

### 워크플로우 서비스

```python
from lawfirm_langgraph.core.workflow.workflow_service import LangGraphWorkflowService
from lawfirm_langgraph.config.langgraph_config import LangGraphConfig

# 워크플로우 서비스 초기화
config = LangGraphConfig.from_env()
workflow = LangGraphWorkflowService(config)

# 쿼리 처리
result = await workflow.process_query_async("계약 해지 조건은?", "session_id")

# 결과 확인
print(result["answer"])
print(f"신뢰도: {result.get('confidence', 'N/A')}")
print(f"소스: {result.get('sources', [])}")
print(f"Keyword Coverage: {result.get('keyword_coverage', 'N/A')}")
```

**주요 기능**:
- LangGraph 기반 State 워크플로우
- 질문 분류 및 긴급도 평가
- LLM 기반 키워드 확장
- 하이브리드 검색 자동 실행
- Keyword Coverage 기반 검색 결과 평가
- 답변 생성 및 품질 검증

## 사용 예시

### 1. 기본 RAG 쿼리

```python
from lawfirm_langgraph.core.workflow.workflow_service import LangGraphWorkflowService
from lawfirm_langgraph.config.langgraph_config import LangGraphConfig

# 워크플로우 서비스 초기화
config = LangGraphConfig.from_env()
workflow = LangGraphWorkflowService(config)

# 쿼리 처리
result = await workflow.process_query_async("계약서에서 주의해야 할 조항은 무엇인가요?", "session_id")

print(f"답변: {result['answer']}")
print(f"신뢰도: {result.get('confidence', 'N/A')}")
print(f"참조 문서: {len(result.get('sources', []))}개")
print(f"Keyword Coverage: {result.get('keyword_coverage', 'N/A')}")
```

### 2. 하이브리드 검색

```python
from lawfirm_langgraph.core.search.engines.hybrid_search_engine_v2 import HybridSearchEngineV2

# 검색 엔진 초기화
search_engine = HybridSearchEngineV2()

# 하이브리드 검색
results = search_engine.search("민법 제543조", question_type="law_inquiry")

for result in results:
    print(f"문서: {result.get('document_id')}")
    print(f"점수: {result.get('score', 0)}")
    print(f"검색 유형: {result.get('search_type', 'hybrid')}")
```

### 3. 벡터 저장소 관리

```python
from lawfirm_langgraph.core.data.vector_store import VectorStore

# 벡터 저장소 초기화
vector_store = VectorStore("snunlp/KR-SBERT-V40K-klueNLI-augSTS")

# 문서 추가
vector_store.add_documents(documents)

# 검색
results = vector_store.similarity_search("계약 해지", k=5)
```

## 문제 해결

### 자주 발생하는 문제

1. **벡터 저장소 로딩 실패**
   - 해결: FAISS 인덱스 파일 경로 확인
   - 해결: 임베딩 모델 다운로드 상태 확인

2. **LLM 응답 지연**
   - 해결: 토큰 수 제한 조정
   - 해결: 캐싱 시스템 활성화

3. **메모리 사용량 과다**
   - 해결: 배치 크기 조정
   - 해결: 모델 양자화 적용

### 로그 확인

```bash
# RAG 시스템 로그 확인
tail -f logs/test/run_query_test_*.log

# 성능 메트릭 추출
Select-String -Path logs/test/run_query_test_*.log -Pattern "PERFORMANCE|process_search_results_combined|expand_keywords"

# Keyword Coverage 확인
Select-String -Path logs/test/run_query_test_*.log -Pattern "Keyword Coverage"
```

## 성능 지표

### 현재 달성된 성능

| 지표 | 값 | 설명 |
|------|-----|------|
| **평균 Keyword Coverage** | 0.806 | 목표 0.70 이상 초과 달성 |
| **process_search_results_combined** | 4-5초 | 목표 5초 이하 달성 (68-75% 감소) |
| **expand_keywords** | 3-4초 | 목표 5초 이하 달성 (51-63% 감소) |
| **평균 검색 시간** | < 1초 | 매우 빠른 검색 성능 |
| **성공률** | 99.9% | 높은 안정성 |
| **메모리 사용량** | 최적화됨 | 모델 캐싱으로 메모리 효율성 향상 |

## 결론

LangGraph 기반 RAG 시스템은 법률 AI 어시스턴트의 핵심 기능을 제공하며, Keyword Coverage 최적화와 성능 최적화를 통해 시스템의 안정성과 성능을 보장합니다. 모듈화된 아키텍처와 확장 가능한 설계를 통해 향후 요구사항 변화에 유연하게 대응할 수 있습니다.

**주요 개선 사항**:
- Keyword Coverage: 평균 0.806 (목표 0.70 이상 초과 달성)
- 성능 최적화: 선택적 의미 기반 매칭, 배치 임베딩 생성, 모델 캐싱
- 메타데이터 정규화: 오타 필드명 자동 수정 및 복원
- LLM 기반 키워드 확장: 동의어/유사어 확장으로 검색 정확도 향상

---

*이 문서는 LawFirmAI 프로젝트의 RAG 시스템 사용법을 설명합니다. 자세한 기술적 구현 내용은 소스 코드를 참조하세요.*
