# LawFirmAI 프로젝트 개요

## 🎯 프로젝트 소개

**프로젝트명**: LawFirmAI - 지능형 법률 AI 어시스턴트  
**목표**: LangGraph 기반 법률 질문 처리 시스템 개발  
**현재 상태**: LangGraph 워크플로우 시스템 통합 완료

## 🚀 핵심 기능

### 1. LangGraph 워크플로우 시스템
- **상태 기반 워크플로우**: State 기반 법률 질문 처리 시스템
- **Agentic AI**: Tool Use/Function Calling을 통한 동적 도구 선택
- **워크플로우 서비스**: `lawfirm_langgraph/core/agents/workflow_service.py`
- **법률 워크플로우**: `lawfirm_langgraph/core/agents/legal_workflow_enhanced.py`

### 2. 하이브리드 검색 시스템
- **벡터 검색**: FAISS 기반 의미적 유사도 검색
- **정확 매칭**: 키워드 기반 정확한 매칭
- **결과 통합**: 가중 평균으로 검색 결과 결합
- **검색 엔진**: `lawfirm_langgraph/core/services/hybrid_search_engine.py`

### 3. 데이터베이스 시스템
- **SQLite 데이터베이스**: 법률 및 판례 문서 저장
- **Assembly 데이터**: 국회 법률정보시스템 데이터 수집 완료
- **판례 데이터**: 민사/형사/가사/조세/행정/특허 판례 분류 처리
- **메타데이터 관리**: 구조화된 법령/판례 정보 저장
- **데이터베이스 관리**: `lawfirm_langgraph/core/data/database.py`

### 4. 벡터 임베딩 시스템
- **FAISS 벡터 인덱스**: 법률 및 판례 문서 벡터 임베딩
- **임베딩 모델**: ko-sroberta-multitask (768차원)
- **검색 성능**: 평균 응답 시간 < 1초
- **증분 업데이트**: 새로운 데이터 자동 처리
- **벡터 스토어**: `lawfirm_langgraph/core/data/vector_store.py`

### 5. 지능형 답변 생성
- **질문 분류**: 6가지 질문 유형 자동 분류
- **동적 검색**: 질문 유형별 검색 가중치 조정
- **구조화된 답변**: 질문 유형별 맞춤형 답변
- **신뢰도 시스템**: 답변 신뢰성 수치화
- **답변 생성기**: `lawfirm_langgraph/core/services/answer_generator.py`

### 6. AI 모델 시스템
- **Google Gemini 2.5 Flash Lite**: 클라우드 LLM 모델
- **LangChain 기반**: 안정적인 LLM 관리
- **LangGraph 워크플로우**: State 기반 법률 질문 처리 시스템
- **UnifiedPromptManager**: 법률 도메인별 프롬프트 통합 관리
- **Gemini 클라이언트**: `lawfirm_langgraph/core/services/gemini_client.py`

## 🔧 기술 스택

### AI/ML
- **LangGraph**: State 기반 워크플로우 관리
- **LangChain**: LLM 통합 프레임워크
- **Google Gemini 2.5 Flash Lite**: 클라우드 LLM 모델
- **Sentence-BERT**: 텍스트 임베딩 모델 (jhgan/ko-sroberta-multitask)
- **FAISS**: 벡터 검색 엔진
- **Ollama Qwen2.5:7b**: 로컬 LLM 모델 (선택적 사용)

### Backend
- **FastAPI**: RESTful API 서버 (선택적)
- **SQLite**: 관계형 데이터베이스 (정확한 매칭 검색)
- **FAISS**: 벡터 데이터베이스 (의미적 검색)
- **Pydantic**: 데이터 검증
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

```
LawFirmAI/
├── lawfirm_langgraph/      # 핵심 LangGraph 워크플로우 시스템
│   ├── config/             # 설정 파일
│   ├── core/               # 핵심 비즈니스 로직
│   │   ├── agents/         # LangGraph 워크플로우 에이전트
│   │   ├── services/       # 비즈니스 서비스
│   │   ├── data/           # 데이터 레이어
│   │   ├── models/         # AI 모델
│   │   └── utils/          # 유틸리티
│   ├── langgraph_core/     # LangGraph 핵심 모듈
│   ├── tests/              # 테스트 코드
│   └── docs/               # 문서
├── scripts/                # 유틸리티 스크립트
│   ├── data_collection/    # 데이터 수집
│   ├── data_processing/    # 데이터 전처리
│   ├── database/           # 데이터베이스 관리
│   └── ml_training/        # ML 모델 훈련
├── data/                   # 데이터 파일
│   ├── raw/                # 원본 데이터
│   ├── processed/          # 전처리된 데이터
│   ├── embeddings/         # 벡터 임베딩
│   └── database/           # 데이터베이스 파일
├── monitoring/             # 모니터링 시스템
├── docs/                   # 문서
└── README.md               # 프로젝트 문서
```

## 🎉 주요 성과

### 시스템 완성도
- ✅ **LangGraph 워크플로우**: State 기반 법률 질문 처리 시스템
- ✅ **하이브리드 검색**: 의미적 검색 + 정확 매칭 통합
- ✅ **지능형 답변 생성**: 질문 유형별 최적화된 답변 시스템
- ✅ **통합 프롬프트 관리**: 법률 도메인별 최적화된 프롬프트 시스템
- ✅ **완전한 테스트 시스템**: 단위 테스트 및 통합 테스트 완료

### 기술적 혁신
- ✅ **규칙 기반 파서**: 안정적인 법률 문서 구조 분석
- ✅ **하이브리드 아키텍처**: 다중 검색 방식 통합
- ✅ **확장 가능한 설계**: 모듈화된 서비스 아키텍처
- ✅ **지능형 질문 분류**: 질문 유형 자동 분류 및 최적화
- ✅ **컨텍스트 최적화**: 토큰 제한 내에서 관련성 높은 정보 선별

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
from lawfirm_langgraph.core.agents.workflow_service import LangGraphWorkflowService

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

## 📚 다음 단계

### 1. 데이터 확장 (우선순위: 높음)
- 추가 판례 데이터 수집 및 처리
- 헌재결정례 데이터 수집 및 임베딩
- 법령해석례 데이터 수집 및 임베딩

### 2. 시스템 고도화 (우선순위: 중간)
- API 성능 최적화
- 법률 용어 사전 확장 및 업데이트
- 질문 유형 분류 정확도 향상

### 3. 기능 확장 (우선순위: 중간)
- 계약서 분석 기능 고도화
- 다국어 지원 (영어, 일본어)
- 개인화된 답변 시스템

## 📖 관련 문서

- [프로젝트 구조](project_structure.md)
- [아키텍처](architecture.md)
- [LangGraph 통합 가이드](../03_rag_system/langgraph_integration_guide.md)
- [개발 규칙](../10_technical_reference/development_rules.md)