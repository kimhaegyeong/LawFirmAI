# LawFirmAI 프로젝트 개요

## 🎯 프로젝트 소개

**프로젝트명**: LawFirmAI - 지능형 법률 AI 어시스턴트  
**목표**: LangGraph 기반 법률 질문 처리 시스템 개발  
**현재 상태**: LangGraph 워크플로우 시스템 통합 완료

## 🚀 핵심 기능

### 1. LangGraph 워크플로우 시스템
- **상태 기반 워크플로우**: State 기반 법률 질문 처리 시스템
- **Agentic AI**: Tool Use/Function Calling을 통한 동적 도구 선택
- **워크플로우 서비스**: `lawfirm_langgraph/core/workflow/workflow_service.py`
- **법률 워크플로우**: `lawfirm_langgraph/core/workflow/legal_workflow_enhanced.py`

### 2. 하이브리드 검색 시스템
- **벡터 검색**: FAISS 기반 의미적 유사도 검색 (SemanticSearchEngineV2)
- **키워드 검색**: FTS5 기반 키워드 검색 (KeywordSearchEngine)
- **하이브리드 통합**: 가중 평균으로 검색 결과 결합
- **Keyword Coverage 기반 동적 가중치**: 검색 결과의 키워드 커버리지에 따라 가중치 조정
- **의미 기반 키워드 매칭**: SentenceTransformer를 활용한 의미적 유사도 기반 키워드 매칭
- **검색 엔진**: `lawfirm_langgraph/core/search/engines/semantic_search_engine_v2.py`
- **하이브리드 엔진**: `lawfirm_langgraph/core/search/engines/hybrid_search_engine_v2.py`

### 3. 데이터베이스 시스템
- **SQLite 데이터베이스**: 법률 및 판례 문서 저장
- **Assembly 데이터**: 국회 법률정보시스템 데이터 수집 완료
- **판례 데이터**: 민사/형사/가사/조세/행정/특허 판례 분류 처리
- **메타데이터 관리**: 구조화된 법령/판례 정보 저장
- **데이터베이스 관리**: `lawfirm_langgraph/core/data/database.py`

### 4. 벡터 임베딩 시스템
- **FAISS 벡터 인덱스**: 법률 및 판례 문서 벡터 임베딩
- **임베딩 모델**: snunlp/KR-SBERT-V40K-klueNLI-augSTS (768차원)
- **IndexIVFPQ 지원**: 메모리 효율적인 근사 검색
- **버전 관리**: EmbeddingVersionManager, FAISSVersionManager
- **검색 성능**: 평균 응답 시간 < 1초
- **증분 업데이트**: 새로운 데이터 자동 처리
- **벡터 스토어**: `lawfirm_langgraph/core/data/vector_store.py`

### 5. 지능형 답변 생성
- **질문 분류**: 의미적 도메인 분류 시스템
- **동적 검색**: 질문 유형별 검색 가중치 조정
- **구조화된 답변**: 질문 유형별 맞춤형 답변
- **신뢰도 시스템**: 답변 신뢰성 수치화
- **Keyword Coverage 최적화**: 평균 0.806 (목표 0.70 이상 초과 달성)
- **성능 최적화**: 선택적 의미 기반 매칭, 배치 임베딩 생성, 모델 캐싱

### 6. AI 모델 시스템
- **Google Gemini 2.5 Flash Lite**: 클라우드 LLM 모델
- **LangChain 기반**: 안정적인 LLM 관리
- **LangGraph 워크플로우**: State 기반 법률 질문 처리 시스템
- **UnifiedPromptManager**: 법률 도메인별 프롬프트 통합 관리
- **Gemini 클라이언트**: `lawfirm_langgraph/core/services/gemini_client.py`
- **State 최적화**: State reduction으로 메모리 효율성 향상
- **성능 최적화**: 
  - 선택적 의미 기반 매칭 (Keyword Coverage 70% 이상 시 생략)
  - 배치 임베딩 생성 (batch_size=8)
  - SentenceTransformer 모델 캐싱 (약 7.5초 절약)
  - LLM 호출 타임아웃 최적화 (3초)
- **메타데이터 캐싱**: TTL 기반 캐싱으로 검색 성능 향상
- **메타데이터 정규화**: 오타 필드명 자동 수정 및 복원

### 7. ML 훈련 및 평가 시스템
- **Ground Truth 생성**: 의사 쿼리 및 클러스터링 기반 Ground Truth 생성
- **RAG 검색 평가**: Recall@K, Precision@K, MRR 등 검색 성능 평가
- **검색 파라미터 튜닝**: 최적의 검색 파라미터 탐색
- **평가 결과 분석**: Test/Val/Train 데이터셋 비교 분석
- **체크포인트 지원**: 대규모 평가 작업 중단 후 재개 가능
- **비용 최적화**: Gemini API 호출 최소화 및 배치 처리
- **메모리 최적화**: 효율적인 배치 처리 및 체크포인트 관리
- **평가 스크립트**: `scripts/ml_training/evaluation/`

## 🔧 기술 스택

### AI/ML
- **LangGraph**: State 기반 워크플로우 관리
- **LangChain**: LLM 통합 프레임워크
- **Google Gemini 2.5 Flash Lite**: 클라우드 LLM 모델
- **Sentence-BERT**: 텍스트 임베딩 모델 (snunlp/KR-SBERT-V40K-klueNLI-augSTS)
- **FAISS**: 벡터 검색 엔진 (외부 인덱스 지원, 버전 관리)
- **SemanticSearchEngineV2**: 최적화된 의미적 검색 엔진

### Backend
- **FastAPI**: RESTful API 서버
- **SQLite**: 관계형 데이터베이스 (FTS5 기반 정확한 매칭 검색, 연결 풀링 지원)
- **FAISS**: 벡터 데이터베이스 (의미적 검색, IndexIVFPQ 지원)
- **Pydantic V2**: 데이터 검증
- **OAuth2**: Google OAuth2 인증 지원
- **psutil**: 메모리 모니터링 및 시스템 리소스 관리

### Frontend
- **React 18+ with TypeScript**: 모던 웹 인터페이스
- **Vite**: 빠른 빌드 도구
- **Tailwind CSS**: 유틸리티 기반 스타일링

### 모니터링
- **Grafana**: 메트릭 시각화
- **Prometheus**: 메트릭 수집

## 📊 데이터 현황

- **법률 문서**: Assembly 데이터 수집 완료
- **벡터 임베딩**: ko-sroberta-multitask 모델 사용
- **검색 성능**: 평균 응답 시간 < 1초

## 📁 프로젝트 구조

자세한 프로젝트 구조는 [프로젝트 구조 문서](project_structure.md)를 참조하세요.

주요 디렉토리:
- `lawfirm_langgraph/`: 핵심 LangGraph 워크플로우 시스템
- `scripts/`: 데이터 수집, 전처리, ML 훈련 스크립트
- `data/`: 원본 데이터, 전처리된 데이터, 벡터 임베딩
- `docs/`: 프로젝트 문서

## 🚀 빠른 시작

### 1. 환경 설정

```bash
# 의존성 설치
pip install -r requirements.txt

# 환경 변수 설정
export GOOGLE_API_KEY="your_api_key_here"
export LANGRAPH_ENABLED=true
```

### 2. 기본 사용

```python
import asyncio
from lawfirm_langgraph.config.langgraph_config import LangGraphConfig
from lawfirm_langgraph.core.workflow.workflow_service import LangGraphWorkflowService

async def main():
    config = LangGraphConfig.from_env()
    service = LangGraphWorkflowService(config)
    
    result = await service.process_query_async(
        "계약서 작성 시 주의할 사항은?",
        "session_id"
    )
    print(result)

asyncio.run(main())
```

### 3. 테스트 실행

```bash
# 전체 테스트
cd lawfirm_langgraph/tests
python run_all_tests.py

# 특정 테스트
pytest lawfirm_langgraph/tests/test_workflow_service.py -v
```

## 📖 관련 문서

- [프로젝트 구조](project_structure.md)
- [아키텍처](architecture.md)
- [LangGraph 통합 가이드](../03_rag_system/langgraph_integration_guide.md)
- [ML 훈련 및 평가 시스템](../02_data/ml_training/README.md)
- [개발 규칙](../10_technical_reference/development_rules.md)