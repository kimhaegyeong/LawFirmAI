# LawFirmAI 시스템 아키텍처 (2025-10-16)

## 전체 시스템 아키텍처

```mermaid
graph TB
    subgraph "Frontend Layer"
        UI[Gradio Web Interface<br/>LangChain 기반]
        API_UI[FastAPI Documentation UI]
    end
    
    subgraph "API Gateway Layer"
        FASTAPI[FastAPI Server<br/>RESTful API]
        MW[Middleware<br/>CORS, Logging]
        AUTH[Authentication<br/>API Key]
    end
    
    subgraph "Service Layer"
        CHAT[Chat Service<br/>기본 채팅]
        ML_RAG[ML Enhanced RAG Service<br/>품질 기반 검색]
        ML_SEARCH[ML Enhanced Search Service<br/>하이브리드 검색]
        HYBRID[Hybrid Search Engine<br/>의미적 + 정확 매칭]
        ANALYSIS[Analysis Service<br/>문서 분석]
    end
    
    subgraph "AI Model Layer"
        KOBART[KoBART Model<br/>한국어 생성]
        SENTBERT[ko-sroberta-multitask<br/>768차원 임베딩]
        BGE_M3[BGE-M3-Korean<br/>1024차원 임베딩]
        MODEL_MGR[Model Manager<br/>모델 통합 관리]
    end
    
    subgraph "Data Layer"
        SQLITE[(SQLite Database<br/>7,680개 법률 문서)]
        FAISS_KO[(FAISS Vector Store<br/>ko-sroberta 155,819개)]
        FAISS_BGE[(FAISS Vector Store<br/>BGE-M3 155,819개)]
        CACHE[(Memory Cache<br/>응답 캐싱)]
    end
    
    subgraph "External Services"
        HF[HuggingFace Hub<br/>모델 다운로드]
        SPACES[HuggingFace Spaces<br/>배포 플랫폼]
        OLLAMA[Ollama Qwen2.5:7b<br/>로컬 LLM]
    end
    
    subgraph "Monitoring"
        PROM[Prometheus<br/>메트릭 수집]
        GRAF[Grafana<br/>대시보드]
    end
    
    UI --> FASTAPI
    API_UI --> FASTAPI
    FASTAPI --> MW
    MW --> AUTH
    AUTH --> CHAT
    AUTH --> ML_RAG
    AUTH --> ML_SEARCH
    AUTH --> HYBRID
    AUTH --> ANALYSIS
    
    CHAT --> MODEL_MGR
    ML_RAG --> ML_SEARCH
    ML_SEARCH --> HYBRID
    HYBRID --> SQLITE
    HYBRID --> FAISS_KO
    HYBRID --> FAISS_BGE
    ANALYSIS --> KOBART
    
    MODEL_MGR --> KOBART
    MODEL_MGR --> SENTBERT
    MODEL_MGR --> BGE_M3
    MODEL_MGR --> OLLAMA
    
    CHAT --> CACHE
    ML_RAG --> CACHE
    ML_SEARCH --> CACHE
    HYBRID --> CACHE
    
    SQLITE --> FAISS_KO
    SQLITE --> FAISS_BGE
    MODEL_MGR --> HF
    FASTAPI --> SPACES
    
    FASTAPI --> PROM
    PROM --> GRAF
```

## 데이터 흐름도 (ML 강화 RAG 시스템)

```mermaid
sequenceDiagram
    participant User
    participant Gradio
    participant FastAPI
    participant LawFirmAI
    participant ML_RAG
    participant ML_Search
    participant HybridEngine
    participant Database
    participant VectorStore_KO
    participant VectorStore_BGE
    participant ModelManager
    
    User->>Gradio: 질문 입력
    Gradio->>LawFirmAI: process_query()
    LawFirmAI->>ML_RAG: retrieve_relevant_documents()
    
    ML_RAG->>ML_Search: search_documents()
    ML_Search->>HybridEngine: hybrid_search()
    
    par 병렬 검색
        HybridEngine->>Database: exact_search()
        Database-->>HybridEngine: 정확 매칭 결과
    and
        HybridEngine->>VectorStore_KO: semantic_search()
        VectorStore_KO-->>HybridEngine: ko-sroberta 결과
    and
        HybridEngine->>VectorStore_BGE: semantic_search()
        VectorStore_BGE-->>HybridEngine: BGE-M3 결과
    end
    
    HybridEngine-->>ML_Search: 통합 검색 결과
    ML_Search-->>ML_RAG: 품질 필터링된 문서
    ML_RAG-->>LawFirmAI: 컨텍스트 문서
    
    LawFirmAI->>ModelManager: generate_response()
    ModelManager->>ModelManager: LangChain RAG 추론
    ModelManager-->>LawFirmAI: 생성된 응답
    
    LawFirmAI-->>Gradio: 응답 + 신뢰도 + 소스
    Gradio-->>User: 답변 표시
```

## 컴포넌트 상호작용도 (현재 구현)

```mermaid
graph LR
    subgraph "Core Services"
        CS[Chat Service<br/>기본 채팅]
        MLR[ML Enhanced RAG<br/>품질 기반 검색]
        MLS[ML Enhanced Search<br/>하이브리드 검색]
        HSE[Hybrid Search Engine<br/>의미적 + 정확 매칭]
        AS[Analysis Service<br/>문서 분석]
    end
    
    subgraph "Data Services"
        DB[Database Manager<br/>SQLite 관리]
        VS_KO[Vector Store KO<br/>ko-sroberta]
        VS_BGE[Vector Store BGE<br/>BGE-M3]
        DP[Data Processor<br/>데이터 전처리]
    end
    
    subgraph "AI Services"
        KM[KoBART Model<br/>한국어 생성]
        SM[ko-sroberta-multitask<br/>768차원]
        BM[BGE-M3-Korean<br/>1024차원]
        MM[Model Manager<br/>모델 통합]
    end
    
    subgraph "API Layer"
        EP[API Endpoints<br/>RESTful API]
        MW[Middleware<br/>CORS, Logging]
        SC[Schemas<br/>Pydantic 모델]
    end
    
    subgraph "Frontend"
        GR[Gradio App<br/>LangChain 기반]
        PM[Prompt Manager<br/>프롬프트 관리]
    end
    
    CS --> MLR
    CS --> MM
    MLR --> MLS
    MLS --> HSE
    HSE --> VS_KO
    HSE --> VS_BGE
    HSE --> DB
    AS --> KM
    
    MM --> KM
    MM --> SM
    MM --> BM
    VS_KO --> SM
    VS_BGE --> BM
    
    EP --> CS
    EP --> MLR
    EP --> MLS
    EP --> HSE
    EP --> AS
    
    MW --> EP
    SC --> EP
    
    GR --> EP
    PM --> GR
```

## 모듈별 의존성 관계 (현재 구조)

```mermaid
graph TD
    subgraph "source/models"
        A[kobart_model.py<br/>KoBART 생성 모델]
        B[sentence_bert.py<br/>임베딩 모델]
        C[model_manager.py<br/>모델 통합 관리]
    end
    
    subgraph "source/services"
        D[chat_service.py<br/>기본 채팅]
        E[rag_service.py<br/>ML 강화 RAG]
        F[search_service.py<br/>ML 강화 검색]
        G[hybrid_search_engine.py<br/>하이브리드 검색]
        H[semantic_search_engine.py<br/>의미적 검색]
        I[exact_search_engine.py<br/>정확 매칭]
        J[analysis_service.py<br/>문서 분석]
    end
    
    subgraph "source/data"
        K[database.py<br/>SQLite 관리]
        L[vector_store.py<br/>벡터 저장소]
        M[data_processor.py<br/>데이터 전처리]
    end
    
    subgraph "source/api"
        N[endpoints.py<br/>API 엔드포인트]
        O[middleware.py<br/>미들웨어]
        P[schemas.py<br/>데이터 스키마]
    end
    
    subgraph "source/utils"
        Q[config.py<br/>설정 관리]
        R[logger.py<br/>로깅 설정]
        S[helpers.py<br/>유틸리티]
    end
    
    subgraph "gradio"
        T[simple_langchain_app.py<br/>메인 앱]
        U[prompt_manager.py<br/>프롬프트 관리]
    end
    
    D --> C
    D --> E
    E --> F
    F --> G
    G --> H
    G --> I
    G --> L
    G --> K
    J --> A
    
    C --> A
    C --> B
    
    N --> D
    N --> E
    N --> F
    N --> G
    N --> J
    N --> P
    
    O --> N
    P --> N
    
    T --> N
    T --> U
    
    A --> Q
    B --> Q
    C --> Q
    D --> Q
    E --> Q
    F --> Q
    G --> Q
    H --> Q
    I --> Q
    J --> Q
    K --> Q
    L --> Q
    M --> Q
```

## 성능 최적화 포인트 (현재 구현)

```mermaid
graph TB
    subgraph "캐싱 계층"
        MC[Memory Cache<br/>응답 캐싱]
        QC[Query Cache<br/>질의 캐싱]
        RC[Response Cache<br/>결과 캐싱]
    end
    
    subgraph "비동기 처리"
        AP[Async Processing<br/>비동기 처리]
        BP[Batch Processing<br/>배치 처리]
        QP[Queue Processing<br/>큐 처리]
    end
    
    subgraph "모델 최적화"
        LO[Lazy Loading<br/>지연 로딩]
        QZ[Quantization<br/>모델 양자화]
        ON[ONNX Runtime<br/>최적화 추론]
    end
    
    subgraph "데이터 최적화"
        ID[Index Design<br/>인덱스 설계]
        PQ[Query Optimization<br/>쿼리 최적화]
        CD[Data Compression<br/>데이터 압축]
    end
    
    subgraph "현재 구현된 최적화"
        FAISS_OPT[FAISS 인덱스<br/>456.5MB 최적화]
        VECTOR_OPT[벡터 압축<br/>768차원 → 효율성]
        PARALLEL[병렬 검색<br/>의미적 + 정확 매칭]
        CHECKPOINT[체크포인트<br/>중단점 복구]
    end
    
    MC --> AP
    QC --> BP
    RC --> QP
    
    LO --> QZ
    QZ --> ON
    
    ID --> PQ
    PQ --> CD
    
    FAISS_OPT --> ID
    VECTOR_OPT --> CD
    PARALLEL --> AP
    CHECKPOINT --> BP
```

## 기술 스택 상세

### AI/ML 모델
- **KoBART**: 한국어 생성 모델 (법률 특화)
- **ko-sroberta-multitask**: 768차원 임베딩 모델
- **BGE-M3-Korean**: 1024차원 다국어 임베딩 모델
- **Ollama Qwen2.5:7b**: 로컬 LLM 모델

### 백엔드 기술
- **FastAPI**: RESTful API 서버
- **LangChain**: RAG 프레임워크
- **SQLite**: 관계형 데이터베이스 (7,680개 법률 문서)
- **FAISS**: 벡터 검색 엔진 (155,819개 벡터)

### 프론트엔드 기술
- **Gradio 4.0.0**: 웹 인터페이스
- **LangChain Integration**: RAG 시스템 통합

### 모니터링 및 배포
- **Prometheus**: 메트릭 수집
- **Grafana**: 대시보드
- **Docker**: 컨테이너화
- **HuggingFace Spaces**: 배포 플랫폼

## 데이터 현황

### 데이터베이스
- **총 법률 문서**: 7,680개
- **벡터 임베딩**: 155,819개 문서
- **FAISS 인덱스 크기**: 456.5 MB
- **메타데이터 크기**: 326.7 MB

### 성능 지표
- **평균 검색 시간**: 0.015초
- **처리 속도**: 5.77 법률/초
- **성공률**: 99.9%
- **메모리 사용량**: 최적화됨 (190MB)
