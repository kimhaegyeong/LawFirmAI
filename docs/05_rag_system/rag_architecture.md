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
│  │   Client    │  │   API        │  │  Streamlit   │        │
│  │  Request    │  │  (FastAPI)   │  │  Interface   │        │
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
**파일**: `core/agents/workflow_service.py`

- **역할**: LangGraph 기반 법률 질문 처리 워크플로우 오케스트레이션
- **주요 기능**:
  - 질문 처리 워크플로우 실행
  - 상태 관리 및 최적화
  - 세션 관리

#### 2. 검색 엔진
**파일**: `core/services/search/`

- **HybridSearchEngine**: 하이브리드 검색 (의미적 + 정확 매칭)
- **SemanticSearchEngine**: FAISS 벡터 기반 의미적 검색
- **ExactSearchEngine**: SQLite FTS 기반 정확한 매칭 검색
- **QuestionClassifier**: 질문 유형 분류

#### 3. 답변 생성 서비스
**파일**: `core/services/generation/`

- **AnswerGenerator**: LLM 기반 답변 생성
- **ContextBuilder**: 검색 결과 기반 컨텍스트 구축
- **AnswerFormatter**: 답변 포맷팅

#### 4. 데이터 레이어
**파일**: `core/data/`

- **VectorStore**: FAISS 벡터 스토어 관리
- **Database**: SQLite 데이터베이스 관리
- **ConversationStore**: 대화 이력 저장

## 설치 및 설정

### 1. 필요한 패키지 설치

```bash
pip install langchain langfuse faiss-cpu sentence-transformers
pip install streamlit fastapi uvicorn
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

### 1. Streamlit 웹 인터페이스 실행

```bash
# Streamlit 서버 시작
cd apps/streamlit
streamlit run app.py
```

### 2. API 서버 실행

```bash
# FastAPI 서버 시작
cd apps/api
uvicorn main:app --host 0.0.0.0 --port 8000
```

## 데이터 플로우

### 1. 쿼리 처리 플로우

```
사용자 쿼리 (apps/streamlit 또는 apps/api)
    ↓
core/agents/workflow_service.py
    ↓
core/agents/legal_workflow_enhanced.py (LangGraph 워크플로우)
    ├── classify_query (질문 분류)
    ├── resolve_multi_turn (멀티턴 처리)
    ├── retrieve_documents (문서 검색)
    │   ├── core/services/search/hybrid_search_engine.py
    │   ├── core/services/search/semantic_search_engine.py
    │   └── core/services/search/exact_search_engine.py
    ├── generate_answer (답변 생성)
    │   ├── core/services/generation/answer_generator.py
    │   └── core/services/generation/context_builder.py
    └── calculate_confidence (신뢰도 계산)
        └── core/services/enhancement/confidence_calculator.py
    ↓
최종 응답 반환
```

### 2. 문서 추가 플로우

```
문서 입력
    ↓
core/data/data_processor.py (문서 전처리)
    ↓
core/data/vector_store.py (벡터 임베딩 생성)
    ↓
FAISS 인덱스 업데이트
    ↓
core/data/database.py (메타데이터 저장)
    ↓
성공/실패 응답
```

## 기술 스택

### 핵심 라이브러리
- **LangChain**: RAG 파이프라인 구축 및 체인 관리
- **Langfuse**: LLM 관찰성 및 디버깅 플랫폼
- **FAISS**: 벡터 검색 및 유사도 계산
- **Sentence-Transformers**: 한국어 임베딩 모델
- **SQLite**: 정확한 매칭 검색 및 메타데이터 저장 (Python 내장 모듈)

### LLM 지원
- **OpenAI**: GPT-3.5-turbo, GPT-4
- **Anthropic**: Claude 모델
- **Google**: Gemini Pro, Gemini Pro Vision
- **로컬 모델**: KoGPT-2, KoBART 등

### 벡터 저장소 옵션
- **FAISS**: 고성능 벡터 검색
- **Chroma**: 문서 중심 벡터 저장소
- **Pinecone**: 클라우드 벡터 데이터베이스

## 성능 최적화

### 1. 벡터 검색 최적화
- **인덱스 최적화**: FAISS 인덱스 타입 선택
- **배치 처리**: 여러 쿼리 동시 처리
- **캐싱**: 자주 사용되는 검색 결과 캐싱

### 2. 컨텍스트 관리 최적화
- **동적 길이 조절**: 쿼리 복잡도에 따른 컨텍스트 길이 조절
- **관련성 필터링**: 임계값 기반 문서 필터링
- **세션 캐싱**: 사용자 세션별 컨텍스트 캐싱

### 3. LLM 호출 최적화
- **프롬프트 최적화**: 효율적인 프롬프트 템플릿
- **토큰 관리**: 최적의 토큰 사용량
- **비동기 처리**: 여러 LLM 호출 병렬 처리

## 모니터링 및 디버깅

### Langfuse 대시보드
- **실시간 추적**: 모든 LLM 호출의 실시간 모니터링
- **성능 메트릭**: 응답 시간, 토큰 사용량, 비용 분석
- **오류 추적**: 실패한 요청의 상세 분석
- **A/B 테스트**: 다양한 프롬프트 및 모델 비교

### 로깅 시스템

```python
# 로깅 설정 예시
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('logs/langchain_rag.log')
    ]
)
```

### 메트릭 수집
- **RAG 메트릭**: 검색 정확도, 응답 품질, 신뢰도
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
**파일**: `core/services/search/semantic_search_engine.py`

```python
from core.services.search import SemanticSearchEngine

engine = SemanticSearchEngine()
results = engine.search("계약 해지", k=5)
```

**주요 기능**:
- FAISS 벡터 인덱스 기반 검색
- Sentence-BERT 모델을 사용한 의미적 유사도 계산

#### 2. 정확 매칭 검색 엔진
**파일**: `core/services/search/exact_search_engine.py`

```python
from core.services.search import ExactSearchEngine

engine = ExactSearchEngine()
results = engine.search("민법 제543조", k=5)
```

**주요 기능**:
- SQLite FTS5 기반 키워드 검색
- 법령명, 조문번호 등 정확한 매칭

#### 3. 하이브리드 검색 엔진
**파일**: `core/services/search/hybrid_search_engine.py`

```python
from core.services.search import HybridSearchEngine

engine = HybridSearchEngine()
results = engine.search("계약 해지", question_type="law_inquiry")
```

**주요 기능**:
- 의미적 검색과 정확 매칭 검색 결과 통합
- 가중 평균을 통한 결과 재순위화
- 질문 유형별 동적 가중치 조정

## LangGraph 기반 RAG 시스템

### 워크플로우 서비스

```python
from core.agents.workflow_service import LangGraphWorkflowService
from infrastructure.utils.langgraph_config import LangGraphConfig

# 워크플로우 서비스 초기화
config = LangGraphConfig.from_env()
workflow = LangGraphWorkflowService(config)

# 쿼리 처리
result = await workflow.process_query("계약 해지 조건은?", "session_id")

# 결과 확인
print(result["answer"])
print(f"신뢰도: {result.get('confidence', 'N/A')}")
print(f"소스: {result.get('sources', [])}")
```

**주요 기능**:
- LangGraph 기반 State 워크플로우
- 질문 분류 및 긴급도 평가
- 하이브리드 검색 자동 실행
- 답변 생성 및 품질 검증

## 사용 예시

### 1. 기본 RAG 쿼리

```python
from core.agents.workflow_service import LangGraphWorkflowService
from infrastructure.utils.langgraph_config import LangGraphConfig

# 워크플로우 서비스 초기화
config = LangGraphConfig.from_env()
workflow = LangGraphWorkflowService(config)

# 쿼리 처리
result = await workflow.process_query("계약서에서 주의해야 할 조항은 무엇인가요?", "session_id")

print(f"답변: {result['answer']}")
print(f"신뢰도: {result.get('confidence', 'N/A')}")
print(f"참조 문서: {len(result.get('sources', []))}개")
```

### 2. 하이브리드 검색

```python
from core.services.search import HybridSearchEngine

# 검색 엔진 초기화
search_engine = HybridSearchEngine()

# 하이브리드 검색
results = search_engine.search("민법 제543조", question_type="law_inquiry")

for result in results:
    print(f"문서: {result.get('document_id')}")
    print(f"점수: {result.get('score', 0)}")
    print(f"검색 유형: {result.get('search_type', 'hybrid')}")
```

### 3. 벡터 저장소 관리

```python
from core.data.vector_store import VectorStore

# 벡터 저장소 초기화
vector_store = VectorStore("ko-sroberta-multitask")

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
tail -f logs/langchain_rag.log

# Langfuse 대시보드 접속
# https://cloud.langfuse.com
```

## 성능 지표

### 현재 달성된 성능

| 지표 | 값 | 설명 |
|------|-----|------|
| **평균 검색 시간** | 0.015초 | 매우 빠른 검색 성능 |
| **처리 속도** | 5.77 법률/초 | 안정적인 처리 속도 |
| **성공률** | 99.9% | 높은 안정성 |
| **메모리 사용량** | 190MB | 최적화된 메모리 사용 |
| **벡터 인덱스 크기** | 456.5 MB | 효율적인 인덱스 크기 |
| **메타데이터 크기** | 326.7 MB | 상세한 메타데이터 |

## 결론

LangChain 기반 RAG 시스템은 법률 AI 어시스턴트의 핵심 기능을 제공하며, Langfuse를 통한 관찰성과 디버깅 기능으로 시스템의 안정성과 성능을 보장합니다. 모듈화된 아키텍처와 확장 가능한 설계를 통해 향후 요구사항 변화에 유연하게 대응할 수 있습니다.

---

*이 문서는 LawFirmAI 프로젝트의 RAG 시스템 사용법을 설명합니다. 자세한 기술적 구현 내용은 소스 코드를 참조하세요.*
