# ⚖️ LawFirmAI - 법률 AI 어시스턴트

법률 관련 질문에 답변해드리는 AI 어시스턴트입니다. 판례, 법령, Q&A 데이터베이스를 기반으로 정확한 법률 정보를 제공합니다.

## 📖 프로젝트 소개

LawFirmAI는 LangGraph 기반의 State 기반 워크플로우를 통해 법률 질문을 처리하고, 하이브리드 검색 시스템(FAISS 벡터 검색 + 키워드 검색)을 활용하여 정확한 법률 정보를 제공하는 AI 어시스턴트입니다.

### 핵심 기능

- ✅ **LangGraph 워크플로우**: State 기반 법률 질문 처리 시스템
- ✅ **실시간 스트리밍**: LLM 응답을 실시간으로 스트리밍하여 사용자 경험 향상
- ✅ **하이브리드 검색**: FAISS 벡터 검색 + 키워드 검색 결합
- ✅ **성능 최적화**: 응답 시간 최소화, 메모리 효율 관리
- ✅ **통합 프롬프트 관리**: 법률 도메인별 최적화된 프롬프트 시스템

### 데이터 시스템

- ✅ **Assembly 데이터 수집**: 국회 법률정보시스템 기반 데이터 수집
- ✅ **벡터 임베딩**: FAISS 기반 초고속 검색
- ✅ **증분 전처리**: 자동화된 데이터 파이프라인
- ✅ **Q&A 데이터셋**: 법률 Q&A 쌍 생성 및 관리
- ✅ **메모리 최적화**: Float16 양자화, 지연 로딩, 자동 메모리 정리
- ✅ **ML 훈련 및 평가**: Ground Truth 생성, RAG 검색 평가, 파라미터 튜닝

## 🛠️ 기술 스택

### AI/ML
- **LangGraph**: State 기반 워크플로우 관리
- **Google Gemini 2.5 Flash Lite**: 클라우드 LLM 모델
- **Sentence-BERT**: 텍스트 임베딩 모델 (jhgan/ko-sroberta-multitask)
- **FAISS**: 벡터 검색 엔진
- **Ollama Qwen2.5:7b**: 로컬 LLM 모델 (Q&A 생성, 답변 생성)

### Backend
- **FastAPI**: RESTful API 서버
- **SQLite**: 관계형 데이터베이스 (정확한 매칭 검색)
- **FAISS**: 벡터 데이터베이스 (의미적 검색)
- **Pydantic**: 데이터 검증
- **LangChain**: LLM 통합 프레임워크

### Frontend
- **React 18+ with TypeScript**: 모던 웹 인터페이스
- **Vite**: 빠른 빌드 도구
- **Tailwind CSS**: 유틸리티 기반 스타일링

## 📁 프로젝트 구조

```
LawFirmAI/
├── api/                    # FastAPI 서버
├── frontend/              # React 프론트엔드
├── lawfirm_langgraph/     # LangGraph 워크플로우 코어
│   ├── config/            # 설정 파일
│   ├── core/              # 핵심 비즈니스 로직
│   │   ├── workflow/      # LangGraph 워크플로우
│   │   ├── agents/        # 레거시 에이전트 (하위 호환성)
│   │   ├── services/      # 비즈니스 서비스
│   │   ├── data/          # 데이터 레이어
│   │   ├── models/        # AI 모델
│   │   └── utils/         # 유틸리티
│   └── tests/             # 테스트 코드
├── scripts/               # 유틸리티 스크립트
│   ├── data_collection/   # 데이터 수집
│   ├── data_processing/   # 데이터 전처리
│   ├── database/          # 데이터베이스 관리
│   └── monitoring/       # 모니터링
├── data/                  # 데이터 파일
│   ├── raw/               # 원본 데이터
│   ├── processed/         # 전처리된 데이터
│   ├── embeddings/        # 벡터 임베딩
│   └── database/          # 데이터베이스 파일
├── monitoring/            # 모니터링 시스템
│   ├── grafana/           # Grafana 설정
│   └── prometheus/        # Prometheus 설정
├── docs/                  # 프로젝트 문서
└── tests/                 # 테스트 코드
```

자세한 프로젝트 구조는 [프로젝트 구조 문서](docs/01_getting_started/project_structure.md)를 참조하세요.

## 📊 모니터링 시스템

### Grafana + Prometheus 기반 실시간 모니터링

LawFirmAI는 법률 수집 성능을 실시간으로 모니터링하는 시스템을 제공합니다.

#### 주요 기능
- **실시간 메트릭 수집**: 페이지 처리, 법률 수집, 에러율 등
- **지속적 메트릭 누적**: 여러 실행에 걸쳐 메트릭 값 누적
- **Grafana 대시보드**: 시각적 모니터링 및 알림
- **성능 분석**: 처리량, 메모리 사용량, CPU 사용률 추적

#### 빠른 시작

```bash
# 1. 모니터링 스택 시작
cd monitoring
docker-compose up -d

# 2. 메트릭 서버 독립 실행
python scripts/monitoring/metrics_collector.py --port 8000

# 3. 법률 수집 실행 (메트릭 포함)
python scripts/assembly/collect_laws_optimized.py --sample 50 --enable-metrics
```

#### 접근 URL
- **Grafana**: http://localhost:3000 (admin/admin123)
- **Prometheus**: http://localhost:9090
- **메트릭 엔드포인트**: http://localhost:8000/metrics

#### 수집되는 메트릭
- `law_collection_pages_processed_total`: 처리된 총 페이지 수
- `law_collection_laws_collected_total`: 수집된 총 법률 수
- `law_collection_page_processing_seconds`: 페이지 처리 시간
- `law_collection_memory_usage_bytes`: 메모리 사용량
- `law_collection_cpu_usage_percent`: CPU 사용률

자세한 내용은 [모니터링 가이드](docs/monitoring/monitoring_guide.md)를 참조하세요.

## 🚀 빠른 시작

### 1. 저장소 클론

```bash
git clone https://github.com/kimhaegyeong/LawFirmAI.git
cd LawFirmAI
```

### 2. 가상환경 설정

```bash
# 가상환경 생성
python -m venv venv

# 가상환경 활성화 (Windows)
venv\Scripts\activate

# 가상환경 활성화 (Linux/Mac)
source venv/bin/activate
```

### 3. 환경 변수 설정

프로젝트 루트에 `.env` 파일을 생성하고 필요한 환경 변수를 설정하세요:

```bash
# Google AI API 키 설정 (필수)
GOOGLE_API_KEY="your_google_key"
```

### 4. 의존성 설치 및 실행

```bash
# API 서버 실행
cd api
pip install -r requirements.txt
python main.py

# React 프론트엔드 실행 (새 터미널)
cd frontend
npm install
npm run dev
```

### 5. 접속

- **React 프론트엔드**: http://localhost:3000
- **FastAPI 서버**: http://localhost:8000
- **API 문서**: http://localhost:8000/docs

자세한 설치 및 실행 가이드는 [빠른 시작 가이드](docs/06_deployment/quick_start.md)를 참조하세요.

## 📚 문서

### 시작하기
- [프로젝트 개요](docs/01_getting_started/project_overview.md)
- [아키텍처](docs/01_getting_started/architecture.md)
- [프로젝트 구조](docs/01_getting_started/project_structure.md)

### 데이터
- [데이터 수집 가이드](docs/02_data/collection/data_collection_guide.md)
- [데이터 전처리 가이드](docs/02_data/processing/preprocessing_guide.md)
- [자동 완료 스크립트 가이드](docs/02_data/processing/auto_complete_script_guide.md)
- [임베딩 가이드](docs/02_data/embedding/embedding_guide.md)
- [FAISS 버전 관리 가이드](docs/02_data/embedding/faiss_version_management_guide.md)
- [FAISS 빠른 시작](docs/02_data/embedding/faiss_version_quick_start.md)
- [ML 훈련 및 평가 시스템](docs/02_data/ml_training/README.md)

### 개발
- [개발 규칙](docs/10_technical_reference/development_rules.md)
- [인코딩 개발 규칙](docs/10_technical_reference/encoding_development_rules.md)
- [성능 최적화 가이드](docs/04_models/performance/performance_optimization_guide.md)
- [LangGraph 통합 가이드](docs/03_rag_system/langgraph_integration_guide.md)

### API
- [API 문서](docs/07_api/API_Documentation.md)
- [API 사용 예제](docs/07_api/usage_examples.md)
- [스트리밍 기능 가이드](docs/07_api/streaming_guide.md)

### 배포
- [배포 가이드](docs/06_deployment/Deployment_Guide.md)
- [빠른 시작 가이드](docs/06_deployment/quick_start.md)
- [AWS 배포 가이드](docs/06_deployment/aws_deployment_quickstart.md)
- [배포 체크리스트](docs/06_deployment/DEPLOYMENT_CHECKLIST.md)

### 모니터링
- [모니터링 가이드](docs/monitoring/monitoring_guide.md)

전체 문서 목록은 [문서 인덱스](docs/README.md)를 참조하세요.

## 🔍 주요 기능

### 하이브리드 검색 시스템

LawFirmAI는 관계형 데이터베이스(SQLite)와 벡터 데이터베이스(FAISS)를 결합한 하이브리드 검색 시스템을 사용합니다.

- **정확한 매칭 검색**: 법령명, 조문번호, 사건번호 등 정확한 검색
- **의미적 검색**: 자연어 쿼리를 통한 맥락적 검색
- **하이브리드 검색**: 두 검색 방식의 결과를 통합하여 최적의 결과 제공

자세한 내용은 [하이브리드 검색 아키텍처](docs/03_rag_system/rag_architecture.md)를 참조하세요.

### LangGraph 워크플로우

LawFirmAI는 LangGraph를 활용한 State 기반 워크플로우 시스템을 사용합니다.

- **상태 기반 처리**: 질문 분류 → 검색 → 답변 생성의 단계별 처리
- **Agentic AI**: Tool Use/Function Calling을 통한 동적 도구 선택
- **실시간 스트리밍**: LLM 응답을 토큰 단위로 실시간 전달

자세한 내용은 [LangGraph 통합 가이드](docs/03_rag_system/langgraph_integration_guide.md)를 참조하세요.

## 📊 데이터 현황

| 데이터 타입 | 수량 | 상태 | 비고 |
|------------|------|------|------|
| 법령 (Assembly) | 7,680개 | ✅ 완료 | 전체 Raw 데이터 전처리 완료 |
| 판례 (Assembly) | 민사: 397개, 형사: 8개, 조세: 472개 | ✅ 완료 | 섹션별 임베딩 완료 |
| 헌재결정례 | 수집 중 | ⏳ 진행 | 데이터 수집 필요 |
| 법령해석례 | 수집 중 | ⏳ 진행 | 데이터 수집 필요 |
| 행정규칙 | 수집 중 | ⏳ 진행 | 데이터 수집 필요 |
| 자치법규 | 수집 중 | ⏳ 진행 | 데이터 수집 필요 |

## 🎯 주요 성과

### 성능 지표
- **워크플로우 실행 시간**: 16.79초
- **API 응답 시간**: 16.59초
- **검색 응답 시간**: < 1초
- **신뢰도 점수**: 0.87 (87%)

### 벡터 스토어
- **총 문서 수**: 60,223개
- **임베딩 차원**: 768차원 (Sentence-BERT)
- **문서 타입 분포**:
  - 판례: 49,796개 (82.7%)
  - 결정례: 7,246개 (12.0%)
  - 법령 조문: 2,093개 (3.5%)
  - 해석례: 1,088개 (1.8%)

### 개선 사항
- **프롬프트 최적화**: 길이 30% 감소 (7,620자 → 5,334자)
- **검색 쿼리 다변화**: 문서 타입별 최적화된 쿼리 생성 (13개 쿼리 타입)
- **검색 결과 다양성**: 각 문서 타입에서 최소 1개씩 보장하는 로직 구현
- **Sources 생성률**: 100%
- **Sources Detail 생성률**: 100%
- **Legal References 생성률**: 93.3%
- **평균 답변 길이**: 2,851자

## 📄 라이선스

이 프로젝트는 MIT 라이선스 하에 배포됩니다. 자세한 내용은 [LICENSE](LICENSE) 파일을 참조하세요.

## 🙏 감사의 말

- [HuggingFace](https://huggingface.co/) - AI 모델 제공
- [FastAPI](https://fastapi.tiangolo.com/) - 웹 프레임워크
- [React](https://react.dev/) - 프론트엔드 프레임워크
- [LangGraph](https://langchain-ai.github.io/langgraph/) - 워크플로우 관리
- [FAISS](https://github.com/facebookresearch/faiss) - 벡터 검색 엔진
- [Sentence-BERT](https://www.sbert.net/) - 텍스트 임베딩 모델

---

*LawFirmAI는 법률 전문가의 도구로 사용되며, 법률 자문을 대체하지 않습니다. 중요한 법률 문제는 반드시 자격을 갖춘 법률 전문가와 상담하시기 바랍니다.*
