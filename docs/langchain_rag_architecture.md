# LangChain RAG 아키텍처 문서

## 개요

LawFirmAI 프로젝트의 TASK 3.3에서 구현된 LangChain 기반 RAG(Retrieval-Augmented Generation) 시스템의 아키텍처를 설명합니다. 이 시스템은 법률 문서를 효율적으로 검색하고 AI 모델을 통해 정확한 답변을 생성하는 것을 목표로 합니다.

## 시스템 아키텍처

### 전체 구조도

```
┌─────────────────────────────────────────────────────────────┐
│                    LangChain RAG System                     │
├─────────────────────────────────────────────────────────────┤
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐        │
│  │   Client    │  │   API       │  │   Gradio    │        │
│  │  Request    │  │  Gateway    │  │  Interface  │        │
│  └─────────────┘  └─────────────┘  └─────────────┘        │
│           │               │               │                │
│           └───────────────┼───────────────┘                │
│                           │                                │
│  ┌─────────────────────────────────────────────────────────┤
│  │              LangChainRAGService                        │
│  │  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐    │
│  │  │ Document    │  │ Context     │  │ Answer      │    │
│  │  │ Processor   │  │ Manager     │  │ Generator   │    │
│  │  └─────────────┘  └─────────────┘  └─────────────┘    │
│  └─────────────────────────────────────────────────────────┤
│                           │                                │
│  ┌─────────────────────────────────────────────────────────┤
│  │              Langfuse Observability                     │
│  │  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐    │
│  │  │   Tracing   │  │  Metrics    │  │  Debugging  │    │
│  │  │   System    │  │ Collection  │  │ Dashboard   │    │
│  │  └─────────────┘  └─────────────┘  └─────────────┘    │
│  └─────────────────────────────────────────────────────────┤
│                           │                                │
│  ┌─────────────────────────────────────────────────────────┤
│  │              Data Layer                                 │
│  │  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐    │
│  │  │ Vector      │  │ Database    │  │ File        │    │
│  │  │ Store       │  │ Manager     │  │ System      │    │
│  │  │ (FAISS)     │  │ (SQLite)   │  │ Storage     │    │
│  │  └─────────────┘  └─────────────┘  └─────────────┘    │
│  └─────────────────────────────────────────────────────────┘
└─────────────────────────────────────────────────────────────┘
```

### 핵심 컴포넌트

#### 1. LangChainRAGService (메인 서비스)
- **역할**: 전체 RAG 파이프라인의 오케스트레이션
- **주요 기능**:
  - 쿼리 처리 및 라우팅
  - 컴포넌트 간 데이터 흐름 관리
  - 성능 모니터링 및 통계 수집
  - 오류 처리 및 복구

#### 2. DocumentProcessor (문서 처리기)
- **역할**: 법률 문서의 전처리 및 청킹
- **주요 기능**:
  - 다양한 문서 형식 지원 (TXT, PDF, DOCX)
  - 법률 문서 특화 전처리
  - 지능적 텍스트 분할 (RecursiveCharacterTextSplitter)
  - 법률 패턴 인식 및 메타데이터 추출

#### 3. ContextManager (컨텍스트 관리자)
- **역할**: 검색된 문서의 컨텍스트 윈도우 관리
- **주요 기능**:
  - 세션 기반 컨텍스트 관리
  - 동적 컨텍스트 길이 조절
  - 관련성 기반 컨텍스트 필터링
  - 컨텍스트 캐싱 및 최적화

#### 4. AnswerGenerator (답변 생성기)
- **역할**: LLM을 통한 답변 생성
- **주요 기능**:
  - 다양한 프롬프트 템플릿 지원
  - LLM 체인 관리 (LangChain)
  - 답변 품질 검증
  - 소스 인용 및 참조 생성

#### 5. LangfuseClient (관찰성 클라이언트)
- **역할**: LLM 호출 추적 및 성능 모니터링
- **주요 기능**:
  - 실시간 LLM 호출 추적
  - 성능 메트릭 수집
  - 오류 추적 및 디버깅
  - A/B 테스트 지원

## 데이터 플로우

### 1. 쿼리 처리 플로우

```
사용자 쿼리
    ↓
LangChainRAGService.process_query()
    ↓
ContextManager.add_query_to_session()
    ↓
DocumentProcessor.retrieve_documents()
    ↓
ContextManager.build_context_window()
    ↓
AnswerGenerator.generate_answer()
    ↓
LangfuseClient.track_rag_query()
    ↓
RAGResult 반환
```

### 2. 문서 추가 플로우

```
문서 입력
    ↓
DocumentProcessor.load_documents()
    ↓
DocumentProcessor.preprocess_document()
    ↓
DocumentProcessor.split_document()
    ↓
Vector Store 업데이트 (FAISS/Chroma)
    ↓
기존 LegalVectorStore 업데이트
    ↓
성공/실패 응답
```

### 3. 관찰성 플로우

```
LLM 호출
    ↓
LangfuseClient.track_llm_call()
    ↓
MetricsCollector.collect_rag_metrics()
    ↓
Langfuse Dashboard 업데이트
    ↓
성능 분석 및 디버깅
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

## 설정 관리

### 환경 변수
```bash
# LangChain 설정
VECTOR_STORE_TYPE=faiss
VECTOR_STORE_PATH=./data/embeddings/faiss_index
EMBEDDING_MODEL=sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2
CHUNK_SIZE=1000
CHUNK_OVERLAP=200
MAX_CONTEXT_LENGTH=4000

# LLM 설정
LLM_PROVIDER=openai
LLM_MODEL=gpt-3.5-turbo
LLM_TEMPERATURE=0.7
LLM_MAX_TOKENS=1000

# Google AI 설정 (LLM_PROVIDER=google인 경우)
GOOGLE_API_KEY=your-google-api-key

# Langfuse 설정
LANGFUSE_ENABLED=true
LANGFUSE_SECRET_KEY=your-secret-key
LANGFUSE_PUBLIC_KEY=your-public-key
LANGFUSE_HOST=https://cloud.langfuse.com
LANGFUSE_DEBUG=false

# 성능 설정
ENABLE_CACHING=true
CACHE_TTL=3600
ENABLE_ASYNC=true
```

### 설정 클래스 구조
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

## 결론

LangChain 기반 RAG 시스템은 법률 AI 어시스턴트의 핵심 기능을 제공하며, Langfuse를 통한 관찰성과 디버깅 기능으로 시스템의 안정성과 성능을 보장합니다. 모듈화된 아키텍처와 확장 가능한 설계를 통해 향후 요구사항 변화에 유연하게 대응할 수 있습니다.
