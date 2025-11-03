# ⚖️ LawFirmAI - 법률 AI 어시스턴트

법률 관련 질문에 답변해드리는 AI 어시스턴트입니다. 판례, 법령, Q&A 데이터베이스를 기반으로 정확한 법률 정보를 제공합니다.

## 📋 목차

1. [개요](#-개요)
2. [주요 기능](#-주요-기능)
3. [기술 스택](#️-기술-스택)
4. [프로젝트 구조](#-프로젝트-구조)
5. [빠른 시작](#-빠른-시작)
6. [문서 가이드](#-문서-가이드)
7. [개발 규칙](#-개발-규칙)
8. [API 문서](#-api-문서)
9. [데이터 수집](#-데이터-수집)

## 🎯 개요

LawFirmAI는 LangGraph 기반 워크플로우를 사용하는 법률 AI 시스템입니다. 하이브리드 검색(의미적 검색 + 정확한 매칭)을 통해 법률 문서를 검색하고, Google Gemini 2.5 Flash Lite를 사용하여 정확한 답변을 생성합니다.

**자세한 내용**: 
- [프로젝트 개요](docs/01_getting_started/project_overview.md)
- [아키텍처](docs/01_getting_started/architecture.md)

## 🔧 주요 기능

### 핵심 기능
- ✅ **LangGraph 워크플로우**: State 기반 법률 질문 처리 시스템
- ✅ **하이브리드 검색**: FAISS 벡터 검색 + 키워드 검색 결합
- ✅ **성능 최적화**: 응답 시간 최소화, 메모리 효율 관리
- ✅ **통합 프롬프트 관리**: 법률 도메인별 최적화된 프롬프트 시스템

### 데이터 시스템
- ✅ **Assembly 데이터 수집**: 국회 법률정보시스템 기반 데이터 수집
- ✅ **벡터 임베딩**: FAISS 기반 초고속 검색
- ✅ **증분 전처리**: 자동화된 데이터 파이프라인
- ✅ **Q&A 데이터셋**: 법률 Q&A 쌍 생성 및 관리
- ✅ **메모리 최적화**: Float16 양자화, 지연 로딩, 자동 메모리 정리

**자세한 내용**: 
- [RAG 시스템 아키텍처](docs/05_rag_system/rag_architecture.md)
- [LangGraph 통합 가이드](docs/05_rag_system/langgraph_integration_guide.md)
- [성능 최적화 보고서](docs/04_models/performance/performance_optimization_report.md)

## 🛠️ 기술 스택

### AI/ML
- **LangGraph**: State 기반 워크플로우 관리
- **Google Gemini 2.5 Flash Lite**: 클라우드 LLM 모델
- **Sentence-BERT**: 텍스트 임베딩 모델 (jhgan/ko-sroberta-multitask)
- **FAISS**: 벡터 검색 엔진
- **Ollama Qwen2.5:7b**: 로컬 LLM 모델 (Q&A 생성, 답변 생성)
- **UnifiedPromptManager**: 법률 도메인별 프롬프트 통합 관리

### Backend
- **FastAPI**: RESTful API 서버
- **SQLite**: 관계형 데이터베이스 (정확한 매칭 검색)
- **FAISS**: 벡터 데이터베이스 (의미적 검색)
- **Pydantic**: 데이터 검증
- **LangChain**: LLM 통합 프레임워크
- **psutil**: 메모리 모니터링 및 시스템 리소스 관리

### Frontend
- **Streamlit**: 웹 인터페이스
- **HuggingFace Spaces**: 배포 플랫폼

## 📁 프로젝트 구조

```
LawFirmAI/
├── lawfirm_langgraph/          # LangGraph 워크플로우 (메인) ⭐
│   ├── source/                  # 워크플로우 소스 코드
│   │   ├── services/            # 비즈니스 로직
│   │   ├── utils/               # 유틸리티
│   │   └── models/              # AI 모델 래퍼
│   ├── graph.py                 # LangGraph 그래프 정의
│   └── streamlit/               # Streamlit 통합
├── core/                        # 핵심 비즈니스 로직
│   ├── services/                # 검색, 생성, 향상 서비스
│   │   ├── search/               # 검색 엔진들
│   │   ├── generation/          # 답변 생성
│   │   └── enhancement/         # 품질 개선
│   ├── data/                    # 데이터 레이어
│   │   ├── database.py          # SQLite 데이터베이스
│   │   └── vector_store.py      # FAISS 벡터 스토어
│   └── models/                  # AI 모델
├── streamlit/                   # Streamlit 웹 인터페이스
│   └── app.py                   # 메인 애플리케이션
├── infrastructure/              # 인프라 및 유틸리티
│   └── utils/                   # 설정, 로깅 등
├── scripts/                     # 실행 스크립트
│   ├── data_collection/         # 데이터 수집
│   ├── data_processing/         # 데이터 전처리
│   └── ...
├── data/                        # 데이터 파일
│   ├── raw/                     # 원본 데이터
│   ├── processed/               # 전처리된 데이터
│   └── embeddings/              # 벡터 임베딩
├── tests/                       # 테스트 코드
└── docs/                        # 문서
```

> ⚠️ **참고**: `core/agents/`는 레거시이며 삭제 예정입니다. 새로운 코드는 `lawfirm_langgraph/`를 사용하세요.

**자세한 내용**: [프로젝트 구조 상세 가이드](docs/01_getting_started/project_structure.md)

## 🚀 빠른 시작

### 1. 저장소 클론

```bash
git clone https://github.com/your-username/LawFirmAI.git
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

**자세한 내용**: [가상환경 가이드](docs/VENV_GUIDE.md)

### 3. 의존성 설치

```bash
# 기본 의존성
pip install -r requirements.txt

# Streamlit 실행 시
cd streamlit
pip install -r requirements.txt
```

### 4. 환경 변수 설정

```bash
# Google AI API 키 설정 (필수)
export GOOGLE_API_KEY="your_google_key"

# 디버그 모드 (선택사항)
export DEBUG="true"
```

### 5. Streamlit 애플리케이션 실행

```bash
cd streamlit
streamlit run app.py
```

접속: **http://localhost:8501**

**자세한 내용**: 
- [시작하기 가이드](docs/01_getting_started/README.md)
- [배포 가이드](docs/06_deployment/Deployment_Guide.md)

## 📚 문서 가이드

LawFirmAI의 모든 문서는 `docs/` 폴더에 체계적으로 정리되어 있습니다.

### 📖 문서 인덱스

- **[전체 문서 인덱스](docs/README.md)**: 모든 문서의 구조화된 목차

### 📁 주요 문서 카테고리

#### 01. 시작하기 (`docs/01_getting_started/`)
- [프로젝트 개요](docs/01_getting_started/project_overview.md)
- [프로젝트 구조](docs/01_getting_started/project_structure.md)
- [아키텍처](docs/01_getting_started/architecture.md)

#### 02. 데이터 (`docs/02_data/`)
- [데이터 수집 가이드](docs/02_data/collection/README.md)
- [데이터 전처리 가이드](docs/02_data/processing/README.md)
- [벡터 임베딩 가이드](docs/02_data/embedding/README.md)

#### 03. RAG 시스템 (`docs/05_rag_system/`)
- [RAG 아키텍처](docs/05_rag_system/rag_architecture.md)
- [LangGraph 통합 가이드](docs/05_rag_system/langgraph_integration_guide.md)
- [개발 규칙](docs/05_rag_system/langchain_langgraph_development_rules.md)

#### 04. 모델 및 성능 (`docs/04_models/`)
- [성능 최적화 보고서](docs/04_models/performance/performance_optimization_report.md)
- [성능 최적화 가이드](docs/04_models/performance/performance_optimization_guide.md)
- [메모리 최적화 가이드](docs/04_models/performance/memory_optimization_guide.md)

#### 05. 품질 관리 (`docs/05_quality/`)
- [품질 개선 시스템](docs/05_quality/quality_improvement_system.md)
- [프롬프트 시스템 강화](docs/05_quality/prompt_system_enhancement.md)

#### 06. 배포 (`docs/06_deployment/`)
- [배포 가이드](docs/06_deployment/Deployment_Guide.md)
- [AWS 배포 가이드](docs/06_deployment/aws_deployment_quickstart.md)
- [HuggingFace Spaces 최적화](docs/06_deployment/huggingface_spaces_optimization_plan.md)

#### 07. API (`docs/07_api/`)
- [API 문서](docs/07_api/API_Documentation.md)
- [API 엔드포인트](docs/07_api/api_endpoints.md)
- [국가법령정보 Open API 가이드](docs/07_api/open_law/README.md)

#### 10. 기술 참고 (`docs/10_technical_reference/`)
- [개발 규칙](docs/10_technical_reference/development_rules.md)
- [인코딩 개발 규칙](docs/10_technical_reference/encoding_development_rules.md)
- [Core 모듈 가이드](docs/10_technical_reference/core_modules_guide.md)
- [문제 해결 가이드](docs/10_technical_reference/Troubleshooting_Guide.md)

**전체 문서 목차**: [docs/README.md](docs/README.md)

## 🔧 개발 규칙

### ⚠️ 중요: Streamlit 서버 관리 규칙

**절대 사용하지 말 것**:
```bash
# 모든 Python 프로세스 종료 (위험!)
taskkill /f /im python.exe
```

**올바른 서버 종료 방법**:
```bash
# Streamlit 서버 종료
# Ctrl+C로 안전하게 종료하거나
# 프로세스 매니저에서 streamlit 프로세스 종료
```

**자세한 내용**: 
- [개발 규칙](docs/10_technical_reference/development_rules.md)
- [인코딩 개발 규칙](docs/10_technical_reference/encoding_development_rules.md)

## 📊 데이터 수집

LawFirmAI는 국가법령정보센터 LAW OPEN API와 국회 법률정보시스템을 통해 법률 데이터를 수집합니다.

### 빠른 시작

```bash
# 전체 데이터 수집 및 벡터DB 구축
python scripts/run_data_pipeline.py --mode full --oc your_email_id

# 특정 데이터 타입만 수집
python scripts/run_data_pipeline.py --mode laws --oc your_email_id --query "민법"

# 벡터DB 구축만 실행
python scripts/run_data_pipeline.py --mode build
```

### 지원 데이터 유형

- **법령**: 주요 법령 (민법, 상법, 형법 등)
- **판례**: 판례 (최근 5년간)
- **헌재결정례**: 헌법재판소 결정례
- **법령해석례**: 법령 해석례
- **행정규칙**: 행정규칙 및 자치법규

**자세한 내용**: 
- [데이터 수집 가이드](docs/02_data/collection/README.md)
- [데이터 전처리 가이드](docs/02_data/processing/README.md)
- [벡터 임베딩 가이드](docs/02_data/embedding/README.md)

## 🔍 하이브리드 검색 시스템

LawFirmAI는 관계형 데이터베이스(SQLite)와 벡터 데이터베이스(FAISS)를 결합한 하이브리드 검색 시스템을 사용합니다.

### 검색 타입

1. **정확한 매칭 검색**: 법령명, 조문번호, 사건번호 등 정확한 검색
2. **의미적 검색**: 자연어 쿼리를 통한 맥락적 검색
3. **하이브리드 검색**: 두 검색 방식의 결과를 통합하여 최적의 결과 제공

**자세한 내용**: [RAG 아키텍처](docs/05_rag_system/rag_architecture.md)

## 🔧 개발 가이드

### 개발 환경 설정

```bash
# 프로젝트 의존성 설치
pip install -r requirements.txt

# 개발 의존성 설치
pip install -e .[dev]

# 코드 포맷팅
black core/ apps/
isort core/ apps/

# 테스트 실행
pytest tests/
```

### 코드 스타일

- **Python**: PEP 8 준수
- **타입 힌트**: 모든 함수에 타입 힌트 사용
- **문서화**: 모든 클래스와 함수에 docstring 작성

**자세한 내용**: 
- [개발 규칙](docs/10_technical_reference/development_rules.md)
- [인코딩 개발 규칙](docs/10_technical_reference/encoding_development_rules.md)
- [Core 모듈 가이드](docs/10_technical_reference/core_modules_guide.md)

## 📚 API 문서

### 주요 엔드포인트

- `POST /api/v1/chat` - 채팅 메시지 처리 (LangGraph 워크플로우)
- `POST /api/v1/search/hybrid` - 하이브리드 검색
- `POST /api/v1/search/exact` - 정확한 매칭 검색
- `POST /api/v1/search/semantic` - 의미적 검색
- `GET /api/v1/health` - 헬스체크

### 빠른 사용 예제

```python
import requests

# 채팅 요청
response = requests.post(
    "http://localhost:8000/api/v1/chat",
    json={
        "message": "계약 해제 조건이 무엇인가요?",
        "session_id": "user_session_123"
    }
)
result = response.json()
print(f"답변: {result['answer']}")
```

**자세한 내용**: 
- [API 문서](docs/07_api/API_Documentation.md)
- [API 엔드포인트 상세](docs/07_api/api_endpoints.md)

## 📊 데이터 현황

| 데이터 타입 | 수량 | 상태 | 비고 |
|------------|------|------|------|
| 법령 (Assembly) | 7,680개 | ✅ 완료 | 전체 Raw 데이터 전처리 완료 |
| 판례 (Assembly) | 민사: 397개, 형사: 8개, 조세: 472개 | ✅ 완료 | 섹션별 임베딩 완료 |
| 헌재결정례 | 수집 중 | ⏳ 진행 | 데이터 수집 필요 |
| 법령해석례 | 수집 중 | ⏳ 진행 | 데이터 수집 필요 |

## 🤝 기여하기

1. Fork the Project
2. Create your Feature Branch (`git checkout -b feature/AmazingFeature`)
3. Commit your Changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the Branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## 📄 라이선스

이 프로젝트는 MIT 라이선스 하에 배포됩니다. 자세한 내용은 [LICENSE](LICENSE) 파일을 참조하세요.


## 🙏 감사의 말

- [HuggingFace](https://huggingface.co/) - AI 모델 제공
- [FastAPI](https://fastapi.tiangolo.com/) - 웹 프레임워크
- [Streamlit](https://streamlit.io/) - UI 프레임워크
- [LangGraph](https://langchain-ai.github.io/langgraph/) - 워크플로우 관리
- [FAISS](https://github.com/facebookresearch/faiss) - 벡터 검색 엔진
- [Sentence-BERT](https://www.sbert.net/) - 텍스트 임베딩 모델

---



*LawFirmAI는 법률 전문가의 도구로 사용되며, 법률 자문을 대체하지 않습니다. 중요한 법률 문제는 반드시 자격을 갖춘 법률 전문가와 상담하시기 바랍니다.*
