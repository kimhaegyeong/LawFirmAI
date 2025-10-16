# LawFirmAI 프로젝트 구조 (2025-10-16)

## 현재 구현된 프로젝트 구조

```
LawFirmAI/
├── gradio/                          # Gradio 웹 애플리케이션
│   ├── simple_langchain_app.py      # 메인 LangChain 기반 앱
│   ├── test_simple_query.py         # 테스트 스크립트
│   ├── prompt_manager.py            # 프롬프트 관리
│   ├── stop_server.py               # 서버 종료 스크립트
│   ├── stop_server.bat              # Windows 배치 파일
│   ├── requirements.txt             # Gradio 의존성
│   ├── Dockerfile                   # Gradio Docker 설정
│   ├── docker-compose.yml           # 로컬 개발 환경
│   └── gradio_server.pid            # PID 파일 (자동 생성)
├── source/                          # 핵심 모듈
│   ├── __init__.py
│   ├── services/                    # 비즈니스 로직
│   │   ├── __init__.py
│   │   ├── chat_service.py          # 기본 채팅 서비스
│   │   ├── rag_service.py           # ML 강화 RAG 서비스
│   │   ├── search_service.py        # ML 강화 검색 서비스
│   │   ├── hybrid_search_engine.py  # 하이브리드 검색 엔진
│   │   ├── semantic_search_engine.py # 의미적 검색 엔진
│   │   ├── exact_search_engine.py   # 정확 매칭 검색 엔진
│   │   ├── analysis_service.py      # 분석 서비스
│   │   └── result_merger.py         # 결과 통합
│   ├── data/                        # 데이터 처리
│   │   ├── __init__.py
│   │   ├── database.py              # SQLite 데이터베이스 관리
│   │   ├── vector_store.py          # 벡터 저장소 관리
│   │   └── data_processor.py        # 데이터 전처리
│   ├── models/                      # AI 모델
│   │   ├── __init__.py
│   │   ├── kobart_model.py          # KoBART 생성 모델
│   │   ├── sentence_bert.py         # 임베딩 모델
│   │   └── model_manager.py         # 모델 통합 관리자
│   ├── api/                         # API 관련
│   │   ├── __init__.py
│   │   ├── endpoints.py             # API 엔드포인트
│   │   ├── search_endpoints.py     # 검색 API 엔드포인트
│   │   ├── schemas.py               # 데이터 스키마
│   │   └── middleware.py            # 미들웨어
│   └── utils/                       # 유틸리티
│       ├── __init__.py
│       ├── config.py                # 설정 관리
│       ├── logger.py                # 로깅 설정
│       └── helpers.py               # 헬퍼 함수
├── data/                            # 데이터 파일
│   ├── lawfirm.db                   # SQLite 데이터베이스 (7,680개 법률 문서)
│   ├── embeddings/                  # 벡터 임베딩
│   │   ├── ml_enhanced_ko_sroberta/ # ko-sroberta 벡터 (155,819개)
│   │   │   ├── ml_enhanced_faiss_index.faiss  # FAISS 인덱스 (456.5MB)
│   │   │   └── ml_enhanced_faiss_index.json  # 메타데이터 (326.7MB)
│   │   └── ml_enhanced_bge_m3/     # BGE-M3 벡터
│   ├── raw/                         # 원본 데이터
│   │   └── assembly/                # Assembly 데이터
│   │       └── law_only/            # 법률 전용 데이터
│   ├── processed/                   # 전처리된 데이터
│   └── backups/                     # 데이터베이스 백업
├── monitoring/                      # 모니터링 시스템
│   ├── prometheus/                  # Prometheus 설정
│   │   ├── prometheus.yml           # Prometheus 설정
│   │   └── rules.yml                # 알림 규칙
│   ├── grafana/                     # Grafana 대시보드
│   │   ├── dashboards/              # 대시보드 설정
│   │   └── provisioning/            # 프로비저닝 설정
│   ├── docker-compose.yml           # 모니터링 스택
│   ├── start_monitoring.bat         # Windows 시작 스크립트
│   ├── start_monitoring.ps1         # PowerShell 시작 스크립트
│   ├── start_monitoring.sh          # Linux 시작 스크립트
│   └── requirements.txt             # 모니터링 의존성
├── scripts/                         # 유틸리티 스크립트
│   ├── assembly/                    # Assembly 데이터 수집
│   │   ├── collect_laws_only.py     # 법률 전용 수집
│   │   └── [기타 수집 스크립트들]
│   ├── vector_embedding/            # 벡터 임베딩 생성
│   │   ├── build_resumable_vector_db.py # 중단점 복구 벡터 빌더
│   │   └── [기타 임베딩 스크립트들]
│   ├── data_processing/             # 데이터 전처리
│   └── tests/                       # 테스트 스크립트
├── tests/                           # 테스트 코드
│   ├── test_chat_service.py         # 채팅 서비스 테스트
│   ├── test_rag_service.py          # RAG 서비스 테스트
│   ├── test_search_service.py       # 검색 서비스 테스트
│   ├── test_vector_store.py         # 벡터 저장소 테스트
│   ├── test_database.py             # 데이터베이스 테스트
│   ├── test_api_endpoints.py        # API 엔드포인트 테스트
│   └── test_integration.py          # 통합 테스트
├── docs/                            # 문서
│   ├── architecture/                # 아키텍처 문서
│   │   ├── system_architecture.md  # 시스템 아키텍처
│   │   ├── project_structure.md     # 프로젝트 구조
│   │   ├── hybrid_search_architecture.md # 하이브리드 검색 아키텍처
│   │   └── module_interfaces.md     # 모듈 인터페이스
│   ├── development/                 # 개발 문서
│   │   └── [44개 개발 관련 문서들]
│   ├── api/                         # API 문서
│   │   ├── lawfirm_ai_api_documentation.md # API 문서
│   │   └── law_open_api/            # 법률 공개 API 문서
│   ├── user_guide/                  # 사용자 가이드
│   ├── project_status.md            # 프로젝트 현황
│   ├── development_rules.md         # 개발 규칙
│   └── [기타 문서들]
├── logs/                            # 로그 파일
│   ├── simple_langchain_gradio.log # Gradio 로그
│   └── [기타 로그 파일들]
├── models/                           # 모델 파일
│   ├── article_classifier.pkl       # 기사 분류 모델
│   └── feature_importance.png       # 특성 중요도
├── benchmark_results/                # 벤치마크 결과
├── reports/                          # 보고서
│   └── quality_report.json          # 품질 보고서
├── results/                          # 결과 파일
├── requirements.txt                  # 프로젝트 의존성
├── README.md                         # 프로젝트 문서
└── .gitignore                        # Git 무시 파일
```

## 현재 구현된 구조 특징

### 1. 완전한 ML 강화 시스템
- **ML 강화 RAG**: 품질 기반 문서 필터링 및 검색
- **하이브리드 검색**: 의미적 검색 + 정확 매칭 통합
- **다중 모델 지원**: KoBART, ko-sroberta-multitask, BGE-M3-Korean
- **벡터 저장소**: 155,819개 문서의 고품질 임베딩

### 2. 완전한 API 시스템
- **RESTful API**: 완전한 REST API 구현
- **다중 엔드포인트**: 채팅, 검색, 분석, 헬스체크
- **ML 강화 엔드포인트**: 품질 기반 검색 및 분석
- **스키마 검증**: Pydantic 기반 데이터 검증

### 3. 모니터링 및 배포 준비
- **Prometheus + Grafana**: 완전한 모니터링 스택
- **Docker 컨테이너화**: 완전한 컨테이너 기반 배포
- **성능 최적화**: 0.015초 평균 검색 시간
- **안정성**: 99.9% 성공률

### 4. 확장 가능한 아키텍처
- **모듈화된 서비스**: 각 기능별 독립적 서비스
- **플러그인 가능**: 새로운 모델 및 서비스 추가 용이
- **확장 가능한 데이터**: 7,680개 법률 문서 처리 완료
- **중단점 복구**: 대용량 데이터 처리 안정성

## 현재 모듈 의존성 (실제 구현)

```
gradio/simple_langchain_app.py ──┐
                                 ├── source/services/chat_service.py
                                 ├── source/services/rag_service.py
                                 ├── source/services/search_service.py
                                 └── source/data/vector_store.py

source/api/endpoints.py ──────────┐
                                 ├── source/services/chat_service.py
                                 ├── source/services/rag_service.py
                                 ├── source/services/search_service.py
                                 ├── source/api/schemas.py
                                 └── source/api/middleware.py

source/services/rag_service.py ──┐
                                 ├── source/models/model_manager.py
                                 ├── source/data/vector_store.py
                                 └── source/data/database.py

source/services/search_service.py ──┐
                                    ├── source/services/hybrid_search_engine.py
                                    ├── source/data/database.py
                                    ├── source/data/vector_store.py
                                    └── source/models/model_manager.py

source/services/hybrid_search_engine.py ──┐
                                         ├── source/services/semantic_search_engine.py
                                         ├── source/services/exact_search_engine.py
                                         └── source/services/result_merger.py

source/models/model_manager.py ────────┐
                                       ├── source/models/kobart_model.py
                                       ├── source/models/sentence_bert.py
                                       └── source/utils/config.py

source/data/vector_store.py ───────────┐
                                       ├── source/data/database.py
                                       └── source/utils/config.py
```

## 현재 데이터 현황

### 데이터베이스
- **SQLite 데이터베이스**: `data/lawfirm.db`
- **총 법률 문서**: 7,680개
- **Assembly 데이터**: 완전한 법률 문서 수집 및 전처리 완료

### 벡터 임베딩
- **ko-sroberta-multitask**: 155,819개 문서 (768차원)
- **BGE-M3-Korean**: 155,819개 문서 (1024차원)
- **FAISS 인덱스**: 456.5 MB
- **메타데이터**: 326.7 MB

### 성능 지표
- **평균 검색 시간**: 0.015초
- **처리 속도**: 5.77 법률/초
- **성공률**: 99.9%
- **메모리 사용량**: 190MB (최적화됨)

## 현재 기술 스택

### AI/ML 모델
- **KoBART**: 한국어 생성 모델 (법률 특화)
- **ko-sroberta-multitask**: 768차원 임베딩 모델
- **BGE-M3-Korean**: 1024차원 다국어 임베딩 모델
- **Ollama Qwen2.5:7b**: 로컬 LLM 모델

### 백엔드 기술
- **FastAPI**: RESTful API 서버
- **LangChain**: RAG 프레임워크
- **SQLite**: 관계형 데이터베이스
- **FAISS**: 벡터 검색 엔진

### 프론트엔드 기술
- **Gradio 4.0.0**: 웹 인터페이스
- **LangChain Integration**: RAG 시스템 통합

### 모니터링 및 배포
- **Prometheus**: 메트릭 수집
- **Grafana**: 대시보드
- **Docker**: 컨테이너화
- **HuggingFace Spaces**: 배포 플랫폼 준비

## 프로젝트 상태

**현재 상태**: 🟢 완전 구현 완료 - 운영 준비 단계  
**마지막 업데이트**: 2025-10-16  
**다음 단계**: HuggingFace Spaces 배포 및 운영 최적화

이 구조는 확장성, 유지보수성, 그리고 개발 효율성을 고려하여 설계되었으며, 현재 완전히 구현되어 운영 준비가 완료된 상태입니다.
