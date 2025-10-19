# LawFirmAI RAG 시스템 가이드

## 개요

LawFirmAI 프로젝트의 LangChain 기반 RAG(Retrieval-Augmented Generation) 시스템 사용법을 설명합니다. 이 시스템은 법률 문서를 효율적으로 검색하고 AI 모델을 통해 정확한 답변을 생성하는 것을 목표로 합니다.

## 🆕 최신 업데이트 (2025-10-20)

### 소스 검색 시스템 대폭 개선
- **의미적 검색 엔진**: FAISS 벡터 인덱스와 SentenceTransformer 모델 통합
- **데이터베이스 폴백**: 벡터 메타데이터 없이도 데이터베이스에서 직접 검색
- **실제 소스 제공**: 법률/판례 데이터베이스에서 실제 근거 자료 검색
- **하이브리드 검색**: 정확한 매칭과 의미적 검색 결과 병합

## 시스템 아키텍처

### 전체 구조

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

## 설치 및 설정

### 1. 필요한 패키지 설치

```bash
pip install langchain langfuse faiss-cpu sentence-transformers
pip install gradio fastapi uvicorn
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

### 1. Gradio 웹 인터페이스 실행

```bash
# Gradio 서버 시작
cd gradio
python simple_langchain_app.py
```

### 2. 간단한 테스트 실행

```bash
# 질의-답변 테스트 스크립트 실행
cd gradio
python test_simple_query.py
```

### 3. API 서버 실행

```bash
# FastAPI 서버 시작
uvicorn source.api.endpoints:app --host 0.0.0.0 --port 8000
```

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
```python
class SemanticSearchEngine:
    """의미적 유사도 기반 검색"""
    
    def search(self, query: str, k: int = 5) -> List[SearchResult]:
        # 벡터 임베딩 기반 검색
        query_embedding = self.embedding_model.encode(query)
        scores, indices = self.vector_store.search(query_embedding, k)
        
        return [SearchResult(
            document_id=indices[i],
            score=scores[i],
            search_type="semantic"
        ) for i in range(len(indices))]
```

#### 2. 정확 매칭 검색 엔진
```python
class ExactSearchEngine:
    """키워드 기반 정확 매칭 검색"""
    
    def search(self, query: str, k: int = 5) -> List[SearchResult]:
        # SQLite FTS5 기반 검색
        results = self.db.execute(
            "SELECT * FROM documents_fts WHERE documents_fts MATCH ? LIMIT ?",
            (query, k)
        ).fetchall()
        
        return [SearchResult(
            document_id=row['id'],
            score=row['rank'],
            search_type="exact"
        ) for row in results]
```

#### 3. 하이브리드 검색 엔진
```python
class HybridSearchEngine:
    """의미적 + 정확 매칭 통합 검색"""
    
    def search(self, query: str, k: int = 5) -> List[SearchResult]:
        # 의미적 검색
        semantic_results = self.semantic_engine.search(query, k)
        
        # 정확 매칭 검색
        exact_results = self.exact_engine.search(query, k)
        
        # 결과 통합 및 재순위화
        combined_results = self._combine_results(
            semantic_results, exact_results, k
        )
        
        return combined_results
    
    def _combine_results(self, semantic: List, exact: List, k: int) -> List:
        """검색 결과 통합"""
        # 가중 평균으로 점수 계산
        combined_scores = {}
        
        for result in semantic:
            doc_id = result.document_id
            combined_scores[doc_id] = {
                'semantic_score': result.score,
                'exact_score': 0.0,
                'combined_score': result.score * 0.7  # 의미적 검색 가중치
            }
        
        for result in exact:
            doc_id = result.document_id
            if doc_id in combined_scores:
                combined_scores[doc_id]['exact_score'] = result.score
                combined_scores[doc_id]['combined_score'] += result.score * 0.3
            else:
                combined_scores[doc_id] = {
                    'semantic_score': 0.0,
                    'exact_score': result.score,
                    'combined_score': result.score * 0.3
                }
        
        # 점수 기준으로 정렬
        sorted_results = sorted(
            combined_scores.items(),
            key=lambda x: x[1]['combined_score'],
            reverse=True
        )
        
        return [SearchResult(
            document_id=doc_id,
            score=scores['combined_score'],
            search_type="hybrid"
        ) for doc_id, scores in sorted_results[:k]]
```

## ML 강화 RAG 시스템

### ML 강화 서비스

```python
class MLEnhancedRAGService:
    """ML 강화 RAG 서비스"""
    
    def __init__(self):
        self.quality_classifier = self._load_quality_classifier()
        self.hybrid_search_engine = HybridSearchEngine()
        self.answer_generator = AnswerGenerator()
    
    def process_query(self, query: str) -> RAGResult:
        """ML 강화 쿼리 처리"""
        # 1. 하이브리드 검색
        search_results = self.hybrid_search_engine.search(query, k=10)
        
        # 2. ML 기반 품질 필터링
        filtered_results = self._filter_by_quality(search_results)
        
        # 3. 컨텍스트 구축
        context = self._build_context(filtered_results)
        
        # 4. 답변 생성
        answer = self.answer_generator.generate(query, context)
        
        return RAGResult(
            query=query,
            answer=answer,
            sources=filtered_results,
            confidence=self._calculate_confidence(answer, filtered_results)
        )
    
    def _filter_by_quality(self, results: List[SearchResult]) -> List[SearchResult]:
        """ML 기반 품질 필터링"""
        quality_scores = self.quality_classifier.predict_proba([
            self._extract_features(result) for result in results
        ])
        
        # 품질 점수가 임계값 이상인 결과만 반환
        filtered_results = []
        for i, result in enumerate(results):
            if quality_scores[i][1] > 0.7:  # 고품질 임계값
                result.quality_score = quality_scores[i][1]
                filtered_results.append(result)
        
        return filtered_results
```

## 사용 예시

### 1. 기본 RAG 쿼리

```python
from source.services.rag_service import LangChainRAGService

# RAG 서비스 초기화
rag_service = LangChainRAGService()

# 쿼리 처리
result = rag_service.process_query("계약서에서 주의해야 할 조항은 무엇인가요?")

print(f"답변: {result.answer}")
print(f"신뢰도: {result.confidence}")
print(f"참조 문서: {len(result.sources)}개")
```

### 2. 하이브리드 검색

```python
from source.services.search_service import MLEnhancedSearchService

# 검색 서비스 초기화
search_service = MLEnhancedSearchService()

# 하이브리드 검색
results = search_service.hybrid_search("민법 제543조", k=5)

for result in results:
    print(f"문서: {result.document_id}")
    print(f"점수: {result.score}")
    print(f"검색 유형: {result.search_type}")
```

### 3. 벡터 저장소 관리

```python
from source.data.vector_store import LegalVectorStore

# 벡터 저장소 초기화
vector_store = LegalVectorStore("ko-sroberta-multitask")

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
