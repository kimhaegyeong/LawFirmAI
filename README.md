# ⚖️ LawFirmAI - 법률 AI 어시스턴트
# 🔍 Vector Search Comparison - pgvector vs FAISS

PostgreSQL 기반 법률 데이터에 대해 **pgvector**와 **FAISS** 두 가지 벡터 검색 엔진의 성능, 정확도, 운영 편의성을 비교 테스트하는 프로젝트입니다.

## 📖 프로젝트 소개

이 프로젝트는 PostgreSQL에 저장된 법률 데이터(법령 조문, 판례 청크)를 대상으로 pgvector와 FAISS 벡터 검색 시스템을 구현하고 체계적으로 비교 분석합니다.

### 핵심 목표

- ✅ **pgvector 기반 벡터 검색 시스템 구현**: PostgreSQL 네이티브 벡터 검색
- ✅ **FAISS 기반 벡터 검색 시스템 구현**: 고성능 벡터 검색 라이브러리
- ✅ **성능 벤치마크**: 검색 속도, 메모리 사용량, 처리량 측정
- ✅ **정확도 비교**: 검색 결과 일치도, 순위 상관관계 분석
- ✅ **운영 편의성 평가**: 실시간 업데이트, 인덱스 관리, 배포 복잡도 비교

### 테스트 데이터

- **법령 데이터**: `statutes_articles` 테이블 (법령 조문)
- **판례 데이터**: `precedent_chunks` 테이블 (판례 청크)
- **임베딩 모델**: `woong0322/ko-legal-sbert-finetuned` (768차원)
- **데이터 규모**: 소규모(1K), 중규모(10K), 대규모(100K) 테스트 지원

자세한 내용은 [비교 테스트 계획](docs/07_api/open_law/ingestion/EMBEDDING_COMPARISON_PLAN.md)을 참조하세요.

## 🛠️ 기술 스택

### 벡터 검색 엔진
- **pgvector**: PostgreSQL 확장 프로그램 (IVFFlat, HNSW 인덱스 지원)
- **FAISS**: Facebook AI Similarity Search (IndexIVFPQ, IndexFlatL2 등 지원)

### 데이터베이스
- **PostgreSQL 12+**: 관계형 데이터베이스
- **pgvector 확장**: 벡터 유사도 검색
- **pg_trgm**: 한국어 텍스트 검색 (trigram 기반)

### AI/ML
- **Sentence-BERT**: 텍스트 임베딩 모델 (snunlp/KR-SBERT-V40K-klueNLI-augSTS)
- **LangGraph**: State 기반 워크플로우 관리 (테스트 자동화)
- **Google Gemini 2.5 Flash Lite**: LLM (검색 결과 평가용)

### Backend
- **FastAPI**: RESTful API 서버
- **Pydantic V2**: 데이터 검증
- **SQLAlchemy**: ORM 및 연결 풀 관리

### Frontend
- **React 18+ with TypeScript**: 비교 결과 시각화
- **Vite**: 빠른 빌드 도구
- **Tailwind CSS**: 유틸리티 기반 스타일링