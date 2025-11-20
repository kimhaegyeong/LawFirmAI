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

- ✅ **Aihub 데이터 수집**: AI허브 기반 법률 데이터 수집
- ✅ **벡터 임베딩**: FAISS 기반 초고속 검색
- ✅ **자동화된 데이터 파이프라인**: 증분 전처리 및 Q&A 데이터셋 생성

자세한 내용은 [데이터 수집 가이드](docs/02_data/collection/data_collection_guide.md)와 [임베딩 가이드](docs/02_data/embedding/embedding_guide.md)를 참조하세요.

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
├── scripts/               # 유틸리티 스크립트
├── data/                  # 데이터 파일
├── docs/                  # 프로젝트 문서
└── tests/                 # 테스트 코드
```

자세한 프로젝트 구조는 [프로젝트 구조 문서](docs/01_getting_started/project_structure.md)를 참조하세요.

## 📊 모니터링

Grafana + Prometheus 기반 실시간 모니터링 시스템을 제공합니다. 자세한 내용은 [모니터링 가이드](docs/monitoring/monitoring_guide.md)를 참조하세요.

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

전체 문서는 [문서 인덱스](docs/README.md)를 참조하세요.

### 주요 문서
- [프로젝트 개요](docs/01_getting_started/project_overview.md) - 프로젝트 소개 및 아키텍처
- [빠른 시작 가이드](docs/06_deployment/quick_start.md) - 설치 및 실행 가이드
- [API 문서](docs/07_api/API_Documentation.md) - API 사용법
- [성능 최적화 가이드](docs/04_models/performance/performance_optimization_guide.md) - 성능 최적화 방법

## 🔍 주요 기능

- **하이브리드 검색**: SQLite + FAISS 기반 정확한 매칭 및 의미적 검색
- **LangGraph 워크플로우**: State 기반 질문 처리 및 실시간 스트리밍
- **Agentic AI**: Tool Use/Function Calling을 통한 동적 도구 선택

자세한 내용은 [RAG 아키텍처](docs/03_rag_system/rag_architecture.md)와 [LangGraph 통합 가이드](docs/03_rag_system/langgraph_integration_guide.md)를 참조하세요.

## 🎯 주요 성과

### 성능 지표
- **워크플로우 실행 시간**: 10-15초 (기존 대비 40-50% 개선)
- **검색 응답 시간**: < 1초
- **신뢰도 점수**: 87%

### 벡터 스토어
- **총 문서 수**: 60,223개
- **임베딩 차원**: 768차원 (Sentence-BERT)
- **문서 타입**: 판례, 결정례, 법령 조문, 해석례

자세한 성능 최적화 내용은 [성능 최적화 가이드](docs/04_models/performance/performance_optimization_guide.md)를 참조하세요.

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
