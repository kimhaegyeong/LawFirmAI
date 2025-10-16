# LawFirmAI 모듈별 인터페이스 명세서

## 1. Models 모듈

### 1.1 kobart_model.py
```python
class KoBARTModel:
    """KoBART 모델 래퍼 클래스"""
    
    def __init__(self, model_name: str = "skt/kobart-base-v1", device: str = "cpu"):
        """KoBART 모델 초기화"""
        pass
    
    def generate_response(self, prompt: str, max_length: int = 512, temperature: float = 0.7) -> str:
        """텍스트 생성"""
        pass
    
    def summarize_text(self, text: str, max_length: int = 256) -> str:
        """텍스트 요약"""
        pass
    
    def extract_keywords(self, text: str, top_k: int = 10) -> List[str]:
        """키워드 추출"""
        pass
    
    def get_model_info(self) -> Dict[str, Any]:
        """모델 정보 반환"""
        pass
```

### 1.2 sentence_bert.py
```python
class SentenceBERTModel:
    """Sentence-BERT 모델 래퍼 클래스"""
    
    def __init__(self, model_name: str = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"):
        """Sentence-BERT 모델 초기화"""
        pass
    
    def encode_text(self, text: str) -> np.ndarray:
        """텍스트를 벡터로 인코딩"""
        pass
    
    def encode_batch(self, texts: List[str]) -> np.ndarray:
        """배치 텍스트 인코딩"""
        pass
    
    def compute_similarity(self, text1: str, text2: str) -> float:
        """두 텍스트 간 유사도 계산"""
        pass
    
    def get_embedding_dimension(self) -> int:
        """임베딩 차원 반환"""
        pass
```

### 1.3 model_manager.py
```python
class ModelManager:
    """AI 모델 통합 관리 클래스"""
    
    def __init__(self, config: Config):
        """모델 매니저 초기화"""
        pass
    
    def load_models(self) -> None:
        """모든 모델 로딩"""
        pass
    
    def get_kobart_model(self) -> KoBARTModel:
        """KoBART 모델 반환"""
        pass
    
    def get_sentence_bert_model(self) -> SentenceBERTModel:
        """Sentence-BERT 모델 반환"""
        pass
    
    def unload_models(self) -> None:
        """모델 언로딩 (메모리 절약)"""
        pass
    
    def get_memory_usage(self) -> Dict[str, float]:
        """메모리 사용량 반환"""
        pass
```

## 2. Services 모듈

### 2.1 chat_service.py
```python
class ChatService:
    """채팅 서비스 클래스"""
    
    def __init__(self, model_manager: ModelManager, rag_service: RAGService):
        """채팅 서비스 초기화"""
        pass
    
    def process_message(self, message: str, context: Optional[str] = None) -> ChatResponse:
        """사용자 메시지 처리"""
        pass
    
    def generate_response(self, prompt: str, context_docs: List[Dict]) -> str:
        """컨텍스트 기반 응답 생성"""
        pass
    
    def validate_input(self, message: str) -> bool:
        """입력 검증"""
        pass
    
    def get_conversation_history(self, session_id: str) -> List[Dict]:
        """대화 기록 조회"""
        pass
    
    def clear_conversation_history(self, session_id: str) -> None:
        """대화 기록 삭제"""
        pass
```

### 2.2 rag_service.py
```python
class RAGService:
    """RAG (Retrieval-Augmented Generation) 서비스"""
    
    def __init__(self, vector_store: VectorStore, database: DatabaseManager):
        """RAG 서비스 초기화"""
        pass
    
    def search_relevant_documents(self, query: str, top_k: int = 5) -> List[Dict]:
        """관련 문서 검색"""
        pass
    
    def retrieve_and_rank(self, query: str, top_k: int = 5) -> List[Dict]:
        """문서 검색 및 랭킹"""
        pass
    
    def build_context(self, documents: List[Dict], max_length: int = 2000) -> str:
        """컨텍스트 구성"""
        pass
    
    def update_vector_store(self, documents: List[Dict]) -> None:
        """벡터 스토어 업데이트"""
        pass
    
    def get_search_stats(self) -> Dict[str, Any]:
        """검색 통계 반환"""
        pass
```

### 2.3 search_service.py
```python
class SearchService:
    """검색 서비스 클래스"""
    
    def __init__(self, database: DatabaseManager, vector_store: VectorStore):
        """검색 서비스 초기화"""
        pass
    
    def search_precedents(self, query: str, filters: Optional[Dict] = None) -> List[Dict]:
        """판례 검색"""
        pass
    
    def search_laws(self, query: str, filters: Optional[Dict] = None) -> List[Dict]:
        """법령 검색"""
        pass
    
    def search_qa_pairs(self, query: str, filters: Optional[Dict] = None) -> List[Dict]:
        """Q&A 검색"""
        pass
    
    def advanced_search(self, search_params: SearchParams) -> SearchResults:
        """고급 검색"""
        pass
    
    def get_search_suggestions(self, partial_query: str) -> List[str]:
        """검색 제안"""
        pass
```

### 2.4 analysis_service.py
```python
class AnalysisService:
    """문서 분석 서비스 클래스"""
    
    def __init__(self, model_manager: ModelManager):
        """분석 서비스 초기화"""
        pass
    
    def analyze_contract(self, contract_text: str) -> ContractAnalysis:
        """계약서 분석"""
        pass
    
    def analyze_legal_document(self, document: str, doc_type: str) -> DocumentAnalysis:
        """법률 문서 분석"""
        pass
    
    def extract_legal_entities(self, text: str) -> List[LegalEntity]:
        """법률 개체 추출"""
        pass
    
    def identify_legal_issues(self, text: str) -> List[LegalIssue]:
        """법률 이슈 식별"""
        pass
    
    def generate_legal_summary(self, document: str) -> str:
        """법률 문서 요약"""
        pass
```

## 3. Data 모듈

### 3.1 database.py
```python
class DatabaseManager:
    """데이터베이스 관리 클래스"""
    
    def __init__(self, db_path: str):
        """데이터베이스 매니저 초기화"""
        pass
    
    def create_tables(self) -> None:
        """테이블 생성"""
        pass
    
    def insert_precedent(self, precedent: PrecedentData) -> int:
        """판례 데이터 삽입"""
        pass
    
    def insert_law(self, law: LawData) -> int:
        """법령 데이터 삽입"""
        pass
    
    def insert_qa_pair(self, qa: QAData) -> int:
        """Q&A 데이터 삽입"""
        pass
    
    def get_precedent_by_id(self, precedent_id: int) -> Optional[Dict]:
        """ID로 판례 조회"""
        pass
    
    def search_precedents(self, query: str, limit: int = 10) -> List[Dict]:
        """판례 검색"""
        pass
    
    def get_database_stats(self) -> Dict[str, int]:
        """데이터베이스 통계"""
        pass
```

### 3.2 vector_store.py
```python
class VectorStore:
    """벡터 저장소 관리 클래스"""
    
    def __init__(self, index_path: str, embedding_dim: int = 768):
        """벡터 스토어 초기화"""
        pass
    
    def add_documents(self, documents: List[Dict], embeddings: np.ndarray) -> None:
        """문서 및 임베딩 추가"""
        pass
    
    def search_similar(self, query_embedding: np.ndarray, top_k: int = 5) -> List[Dict]:
        """유사 문서 검색"""
        pass
    
    def update_document(self, doc_id: str, document: Dict, embedding: np.ndarray) -> None:
        """문서 업데이트"""
        pass
    
    def delete_document(self, doc_id: str) -> None:
        """문서 삭제"""
        pass
    
    def get_index_info(self) -> Dict[str, Any]:
        """인덱스 정보 반환"""
        pass
    
    def save_index(self, path: str) -> None:
        """인덱스 저장"""
        pass
    
    def load_index(self, path: str) -> None:
        """인덱스 로딩"""
        pass
```

### 3.3 data_processor.py
```python
class DataProcessor:
    """데이터 전처리 클래스"""
    
    def __init__(self, config: Config):
        """데이터 프로세서 초기화"""
        pass
    
    def preprocess_text(self, text: str) -> str:
        """텍스트 전처리"""
        pass
    
    def clean_legal_document(self, document: str) -> str:
        """법률 문서 정리"""
        pass
    
    def extract_metadata(self, document: str, doc_type: str) -> Dict[str, Any]:
        """메타데이터 추출"""
        pass
    
    def validate_data_quality(self, data: Dict) -> bool:
        """데이터 품질 검증"""
        pass
    
    def batch_process(self, documents: List[Dict]) -> List[Dict]:
        """배치 처리"""
        pass
```

## 4. API 모듈

### 4.1 endpoints.py
```python
class APIEndpoints:
    """API 엔드포인트 클래스"""
    
    def __init__(self, chat_service: ChatService, search_service: SearchService, analysis_service: AnalysisService):
        """API 엔드포인트 초기화"""
        pass
    
    def setup_routes(self, app: FastAPI) -> None:
        """라우트 설정"""
        pass
    
    async def chat_endpoint(self, request: ChatRequest) -> ChatResponse:
        """채팅 엔드포인트"""
        pass
    
    async def search_precedent_endpoint(self, request: SearchRequest) -> SearchResponse:
        """판례 검색 엔드포인트"""
        pass
    
    async def analyze_contract_endpoint(self, request: AnalysisRequest) -> AnalysisResponse:
        """계약서 분석 엔드포인트"""
        pass
    
    async def health_check_endpoint(self) -> Dict[str, str]:
        """헬스체크 엔드포인트"""
        pass
```

### 4.2 middleware.py
```python
class APIMiddleware:
    """API 미들웨어 클래스"""
    
    def __init__(self, config: Config):
        """미들웨어 초기화"""
        pass
    
    def setup_cors(self, app: FastAPI) -> None:
        """CORS 설정"""
        pass
    
    def setup_logging(self, app: FastAPI) -> None:
        """로깅 설정"""
        pass
    
    def setup_rate_limiting(self, app: FastAPI) -> None:
        """속도 제한 설정"""
        pass
    
    def setup_error_handling(self, app: FastAPI) -> None:
        """에러 처리 설정"""
        pass
```

### 4.3 schemas.py
```python
# Pydantic 모델 정의
class ChatRequest(BaseModel):
    message: str
    context: Optional[str] = None
    session_id: Optional[str] = None

class ChatResponse(BaseModel):
    response: str
    confidence: float
    sources: List[str]
    session_id: str

class SearchRequest(BaseModel):
    query: str
    filters: Optional[Dict[str, Any]] = None
    limit: int = 10

class SearchResponse(BaseModel):
    results: List[Dict[str, Any]]
    total_count: int
    query_time: float

class AnalysisRequest(BaseModel):
    document: str
    document_type: str
    analysis_depth: int = 1

class AnalysisResponse(BaseModel):
    summary: str
    issues: List[str]
    recommendations: List[str]
    confidence: float
```

## 5. Utils 모듈

### 5.1 config.py
```python
class Config:
    """설정 관리 클래스"""
    
    def __init__(self, env_file: str = ".env"):
        """설정 초기화"""
        pass
    
    def get_database_url(self) -> str:
        """데이터베이스 URL 반환"""
        pass
    
    def get_model_path(self) -> str:
        """모델 경로 반환"""
        pass
    
    def get_api_key(self) -> str:
        """API 키 반환"""
        pass
    
    def get_log_level(self) -> str:
        """로그 레벨 반환"""
        pass
    
    def is_development(self) -> bool:
        """개발 환경 여부"""
        pass
```

### 5.2 logger.py
```python
class Logger:
    """로깅 관리 클래스"""
    
    def __init__(self, name: str, level: str = "INFO"):
        """로거 초기화"""
        pass
    
    def setup_logging(self) -> None:
        """로깅 설정"""
        pass
    
    def get_logger(self, name: str) -> logging.Logger:
        """로거 반환"""
        pass
    
    def log_request(self, request: Dict) -> None:
        """요청 로깅"""
        pass
    
    def log_response(self, response: Dict) -> None:
        """응답 로깅"""
        pass
    
    def log_error(self, error: Exception, context: str = "") -> None:
        """에러 로깅"""
        pass
```

### 5.3 helpers.py
```python
class Helpers:
    """유틸리티 헬퍼 클래스"""
    
    @staticmethod
    def validate_text(text: str, max_length: int = 10000) -> bool:
        """텍스트 검증"""
        pass
    
    @staticmethod
    def clean_html(html_text: str) -> str:
        """HTML 태그 제거"""
        pass
    
    @staticmethod
    def extract_keywords(text: str, top_k: int = 10) -> List[str]:
        """키워드 추출"""
        pass
    
    @staticmethod
    def format_confidence(confidence: float) -> str:
        """신뢰도 포맷팅"""
        pass
    
    @staticmethod
    def generate_session_id() -> str:
        """세션 ID 생성"""
        pass
```

## 인터페이스 설계 원칙

### 1. 일관성
- 모든 서비스는 동일한 네이밍 컨벤션 사용
- 공통된 에러 처리 방식 적용
- 표준화된 응답 형식 사용

### 2. 확장성
- 인터페이스는 구현 세부사항에 의존하지 않음
- 새로운 기능 추가 시 기존 인터페이스 변경 최소화
- 플러그인 아키텍처 지원

### 3. 테스트 가능성
- 모든 메서드는 단위 테스트 가능
- 의존성 주입을 통한 모킹 지원
- 명확한 입력/출력 정의

### 4. 성능
- 비동기 처리 지원
- 캐싱 인터페이스 제공
- 메모리 사용량 모니터링

### 5. 보안
- 입력 검증 인터페이스
- 인증/인가 체계
- 민감한 데이터 보호
