# LawFirmAI 서비스 아키텍처

## 개요

LawFirmAI의 서비스 아키텍처는 140+ 개의 모듈화된 서비스로 구성되어 있으며, Enhanced Chat Service와 통합 검색 엔진 시스템을 통해 완전한 법률 AI 어시스턴트를 제공합니다.

## 아키텍처 개요

### 전체 아키텍처 (2025-01-18 업데이트)

```
┌─────────────────────────────────────────────────────────────┐
│                    Streamlit UI (웹 인터페이스)                  │
├─────────────────────────────────────────────────────────────┤
│                Enhanced Chat Service (통합)                   │
│                    (2,497라인)                              │
├─────────────────────────────────────────────────────────────┤
│  Phase 1: 대화 맥락 강화    │  Phase 2: 개인화 분석    │  Phase 3: 장기 기억    │
│  ├── IntegratedSessionManager │  ├── UserProfileManager    │  ├── ContextualMemoryManager │
│  ├── MultiTurnHandler         │  ├── EmotionIntentAnalyzer │  ├── ConversationQualityMonitor │
│  └── ContextCompressor        │  └── ConversationFlowTracker│  └── PerformanceOptimizer    │
├─────────────────────────────────────────────────────────────┤
│                    통합 검색 엔진 레이어                      │
│  ├── UnifiedSearchEngine      │  ├── IntegratedLawSearchService │  ├── EnhancedLawSearchEngine │
│  ├── PerformanceMonitor       │  ├── CurrentLawSearchEngine │  └── VectorSearchOptimizer    │
├─────────────────────────────────────────────────────────────┤
│                    핵심 서비스 레이어 (기능별 분리)            │
│  ├── services/chat/           │  ├── services/search/      │  ├── services/analysis/        │
│  │   ├── ChatService          │  │   ├── SearchService     │  │   ├── QuestionClassifier     │
│  │   ├── EnhancedChatService  │  │   ├── RAGService        │  │   ├── EmotionIntentAnalyzer │
│  │   └── OptimizedChatService │  │   ├── HybridSearchEngine│  │   └── QualityMonitor         │
│  ├── services/memory/         │  ├── services/optimization/│  └── services/langgraph_workflow/│
│  │   ├── ContextualMemory     │  │   ├── PerformanceMonitor│  │   ├── LegalWorkflow         │
│  │   ├── SessionManager       │  │   ├── CacheSystem       │  │   ├── KeywordMapper         │
│  │   └── ConversationStore    │  │   └── MemoryOptimizer   │  │   └── SynonymExpander       │
├─────────────────────────────────────────────────────────────┤
│                    데이터 레이어                            │
│  ├── data/DatabaseManager     │  ├── data/VectorStore      │  └── data/DataProcessor       │
│  ├── models/LegalModelManager  │  ├── models/KoBARTModel    │  └── models/SentenceBERT      │
│  ├── api/Endpoints            │  ├── api/Middleware        │  └── api/Schemas              │
│  └── utils/                   │  │   ├── Config            │  │   ├── Logger                │
│      ├── validation/          │  │   ├── security/          │  │   └── monitoring/           │
└─────────────────────────────────────────────────────────────┘
```

### 서비스 분류

| 분류 | 서비스 수 | 주요 역할 |
|------|-----------|-----------|
| **Enhanced Chat Service** | 1개 | 통합 채팅 서비스 (2,574라인) |
| **통합 검색 엔진** | 4개 | UnifiedSearchEngine, IntegratedLawSearchService 등 |
| **핵심 서비스** | 15개 | 기본 RAG, 검색, 답변 생성 |
| **Phase 서비스** | 9개 | Phase 1-3 기능 구현 |
| **최적화 서비스** | 12개 | 성능 최적화, 캐싱, 모니터링 |
| **데이터 서비스** | 8개 | 데이터 관리, 벡터 스토어 |
| **API 서비스** | 6개 | REST API, 엔드포인트 |
| **기타 서비스** | 85개 | 유틸리티, 모니터링, LangGraph |

## 핵심 서비스 (2025-01-18 업데이트)

### 1. Enhanced Chat Service (통합 서비스)

**파일**: `source/services/chat/enhanced_chat_service.py` (2,497라인)

**역할**: 모든 Phase를 통합하여 완전한 지능형 채팅 제공

**주요 특징**:
- Phase 1-3 시스템 완전 통합
- 법률 제한 시스템 및 보안 강화
- 성능 최적화 및 메모리 관리
- 자연스러운 대화 개선 시스템

**주요 메서드**:
```python
class EnhancedChatService:
    def process_message(self, message: str, user_id: str = None, session_id: str = None) -> Dict[str, Any]:
        """메시지 처리 (모든 Phase 통합)"""
        
    def _initialize_legal_restriction_systems(self):
        """법률 제한 시스템 초기화"""
        
    def _initialize_performance_monitoring(self):
        """성능 모니터링 시스템 초기화"""
```

### 2. 통합 검색 엔진 시스템 (services/search/)

#### 2.1 Unified Search Engine
**파일**: `source/services/search/unified_search_engine.py` (460라인)
**역할**: 모든 검색 기능을 통합한 단일 검색 엔진

#### 2.2 Integrated Law Search Service
**파일**: `source/services/search/integrated_law_search_service.py` (578라인)
**역할**: 통합 조문 검색 서비스

#### 2.3 Enhanced Law Search Engine
**파일**: `source/services/search/enhanced_law_search_engine.py` (1,299라인)
**역할**: 향상된 법령 검색 엔진

#### 2.4 MLEnhancedSearchService
**파일**: `source/services/search/search_service.py`
**역할**: ML 강화된 검색 서비스 (의존성 주입 개선)

**개선된 구조**:
```python
class MLEnhancedSearchService:
    def __init__(self, config: Config = None, database: DatabaseManager = None,
                 vector_store: VectorStore = None, model_manager: LegalModelManager = None):
        # 기본값으로 인스턴스 생성
        if config is None:
            config = Config()
        # ... 기타 의존성 초기화
        
    def search(self, query: str, max_results: int = 10, search_type: str = "hybrid") -> Dict[str, Any]:
        """통합 검색 메서드 (간편한 인터페이스)"""
```

### 3. RAGService (검색 증강 생성)

**파일**: `source/services/search/rag_service.py`
**역할**: ML 강화된 RAG 시스템 구현

**주요 컴포넌트**:
- 문서 검색
- 컨텍스트 구성
- 답변 생성
- 신뢰도 계산

**개선된 메서드**:
```python
class MLEnhancedRAGService:
    def generate_response(self, query: str, max_length: int = 512) -> str:
        """RAG 기반 응답 생성"""
        # 관련 문서 검색
        relevant_docs = self.retrieve_relevant_documents(query, top_k=5)
        
        # 컨텍스트 구성
        context = self.generate_context(query, relevant_docs)
        
        # 모델을 사용한 응답 생성
        response = self.model_manager.generate_response(
            prompt=f"질문: {query}\n\n관련 법률 정보:\n{context}\n\n답변:",
            max_length=max_length
        )
        
        return response
```

### 4. HybridSearchEngine (하이브리드 검색)

**파일**: `source/services/search/hybrid_search_engine.py`
**역할**: 의미적 검색 + 정확 매칭 통합

**검색 방식**:
- 의미적 검색 (FAISS 벡터)
- 정확 매칭 (데이터베이스)
- 하이브리드 병합

### 5. QuestionClassifier (질문 분류)

**파일**: `source/services/analysis/question_classifier.py`
**역할**: 질문 유형 분류 및 처리 전략 결정

**분류 유형**:
- 법령 조문 문의
- 판례 검색
- 절차 문의
- 계약서 검토

## Phase별 서비스

### Phase 1: 대화 맥락 강화

#### 1. IntegratedSessionManager
**파일**: `source/services/integrated_session_manager.py`

**기능**:
- 메모리와 DB 이중 관리
- 자동 동기화
- 세션 복원
- 캐시 전략

#### 2. MultiTurnQuestionHandler
**파일**: `source/services/multi_turn_handler.py`

**기능**:
- 대명사 해결
- 질문 완성
- 맥락 기반 재구성

#### 3. ContextCompressor
**파일**: `source/services/context_compressor.py`

**기능**:
- 중요 정보 추출
- 토큰 압축
- 우선순위 기반 선택

### Phase 2: 개인화 및 지능형 분석

#### 1. UserProfileManager
**파일**: `source/services/user_profile_manager.py`

**기능**:
- 사용자 프로필 관리
- 전문성 수준 추적
- 선호도 학습

#### 2. EmotionIntentAnalyzer
**파일**: `source/services/emotion_intent_analyzer.py`

**기능**:
- 감정 분석
- 의도 분석
- 응답 톤 조정

#### 3. ConversationFlowTracker
**파일**: `source/services/conversation_flow_tracker.py`

**기능**:
- 대화 흐름 추적
- 다음 의도 예측
- 후속 질문 제안

### Phase 3: 장기 기억 및 품질 모니터링

#### 1. ContextualMemoryManager
**파일**: `source/services/contextual_memory_manager.py`

**기능**:
- 중요 사실 추출
- 메모리 중요도 점수화
- 관련 메모리 검색

#### 2. ConversationQualityMonitor
**파일**: `source/services/conversation_quality_monitor.py`

**기능**:
- 품질 평가
- 문제점 감지
- 개선 제안

#### 3. PerformanceOptimizer
**파일**: `source/utils/performance_optimizer.py`

**기능**:
- 성능 모니터링
- 메모리 최적화
- 캐시 관리

## 최적화 서비스

### 1. OptimizedChatService
**파일**: `source/services/optimized_chat_service.py`

**최적화 기능**:
- 비동기 처리
- 메모리 효율성
- 캐시 최적화

### 2. OptimizedModelManager
**파일**: `source/services/optimized_model_manager.py`

**최적화 기능**:
- 지연 로딩
- 모델 공유
- 메모리 관리

### 3. IntegratedCacheSystem
**파일**: `source/services/integrated_cache_system.py`

**캐시 전략**:
- LRU 캐시
- TTL 관리
- 메모리 제한

## 데이터 관리 서비스

### 1. DatabaseManager
**파일**: `source/data/database.py`

**기능**:
- SQLite 데이터베이스 관리
- 쿼리 최적화
- 연결 풀 관리

**테이블 구조**:
```sql
-- 법률 문서
CREATE TABLE laws (
    id INTEGER PRIMARY KEY,
    title TEXT,
    content TEXT,
    article_number TEXT,
    created_at TIMESTAMP
);

-- 판례
CREATE TABLE precedent_cases (
    id INTEGER PRIMARY KEY,
    case_number TEXT,
    title TEXT,
    content TEXT,
    court TEXT,
    created_at TIMESTAMP
);

-- 사용자 프로필
CREATE TABLE user_profiles (
    user_id TEXT PRIMARY KEY,
    expertise_level TEXT,
    detail_level TEXT,
    interest_areas TEXT,
    created_at TIMESTAMP
);
```

### 2. VectorStore
**파일**: `source/data/vector_store.py`

**기능**:
- FAISS 벡터 인덱스 관리
- 임베딩 생성
- 유사도 검색

### 3. ConversationStore
**파일**: `source/data/conversation_store.py`

**기능**:
- 대화 데이터 저장
- 세션 관리
- 메타데이터 관리

## LangGraph 워크플로우

### 1. LegalWorkflow
**파일**: `source/services/langgraph/legal_workflow.py`

**기능**:
- 법률 질문 처리 워크플로우
- 상태 기반 처리
- 체크포인트 관리

### 2. KeywordMapper
**파일**: `source/services/langgraph/keyword_mapper.py`

**기능**:
- 키워드 매핑
- 동의어 처리
- 검색 최적화

### 3. SynonymExpander
**파일**: `source/services/langgraph/real_gemini_synonym_expander.py`

**기능**:
- LLM 기반 동의어 확장
- 법률 용어 처리
- 품질 관리

## 서비스 간 통신

### 1. 동기 통신

```python
# 직접 호출
result = rag_service.search_documents(query)
response = answer_generator.generate_answer(result)
```

### 2. 비동기 통신

```python
# 비동기 처리
async def process_message_async(message):
    search_task = asyncio.create_task(search_service.search_async(message))
    classify_task = asyncio.create_task(classifier.classify_async(message))
    
    search_result, classification = await asyncio.gather(search_task, classify_task)
    return await answer_generator.generate_async(search_result, classification)
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
class ChatService:
    def __init__(self, 
                 rag_service: RAGService,
                 search_service: SearchService,
                 session_manager: IntegratedSessionManager):
        self.rag_service = rag_service
        self.search_service = search_service
        self.session_manager = session_manager
```

### 3. 설정 관리

```python
class Config:
    def __init__(self):
        self.database_url = os.getenv("DATABASE_URL", "sqlite:///./data/lawfirm.db")
        self.model_path = os.getenv("MODEL_PATH", "./models")
        self.cache_size = int(os.getenv("CACHE_SIZE", "1000"))
```

## 성능 최적화

### 1. 메모리 최적화

```python
class MemoryOptimizer:
    def optimize_memory(self):
        """메모리 최적화"""
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    
    def monitor_memory(self) -> float:
        """메모리 사용량 모니터링"""
        return psutil.virtual_memory().percent
```

### 2. 캐시 최적화

```python
class CacheManager:
    def __init__(self, max_size: int = 1000):
        self.cache = {}
        self.max_size = max_size
        self.access_times = {}
    
    def get(self, key: str) -> Any:
        """캐시에서 데이터 조회"""
        if key in self.cache:
            self.access_times[key] = time.time()
            return self.cache[key]
        return None
    
    def set(self, key: str, value: Any):
        """캐시에 데이터 저장"""
        if len(self.cache) >= self.max_size:
            self._evict_lru()
        self.cache[key] = value
        self.access_times[key] = time.time()
```

### 3. 병렬 처리

```python
import concurrent.futures

class ParallelProcessor:
    def process_parallel(self, tasks: List[Callable]) -> List[Any]:
        """병렬 처리"""
        with concurrent.futures.ThreadPoolExecutor() as executor:
            futures = [executor.submit(task) for task in tasks]
            return [future.result() for future in futures]
```

## 보안 고려사항

### 1. 입력 검증

```python
from pydantic import BaseModel, validator

class SecureRequest(BaseModel):
    message: str
    
    @validator('message')
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

class TestChatService:
    def test_process_message(self):
        """메시지 처리 테스트"""
        chat_service = ChatService(config)
        result = chat_service.process_message("테스트 메시지")
        assert result is not None
        assert "response" in result
```

### 2. 통합 테스트

```python
class TestPhaseIntegration:
    def test_phase1_phase2_integration(self):
        """Phase 1-2 통합 테스트"""
        session_manager = IntegratedSessionManager(":memory:")
        profile_manager = UserProfileManager()
        
        session_id = session_manager.create_session("test_user")
        profile_manager.create_profile("test_user", ExpertiseLevel.BEGINNER)
        
        assert session_id is not None
        assert profile_manager.get_profile("test_user") is not None
```

### 3. 성능 테스트

```python
import time

class TestPerformance:
    def test_response_time(self):
        """응답 시간 테스트"""
        chat_service = ChatService(config)
        
        start_time = time.time()
        result = chat_service.process_message("테스트 메시지")
        end_time = time.time()
        
        response_time = end_time - start_time
        assert response_time < 5.0  # 5초 이내
```

## 배포 고려사항

### 1. 컨테이너화

```dockerfile
FROM python:3.9-slim

WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt

COPY source/ ./source/
COPY gradio/ ./gradio/

CMD ["python", "gradio/simple_langchain_app.py"]
```

### 2. 환경별 설정

```python
class EnvironmentConfig:
    def __init__(self, env: str):
        if env == "development":
            self.debug = True
            self.log_level = "DEBUG"
        elif env == "production":
            self.debug = False
            self.log_level = "INFO"
```

### 3. 모니터링

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
