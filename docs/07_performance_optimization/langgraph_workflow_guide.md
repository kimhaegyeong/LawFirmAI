# LangGraph 워크플로우 시스템 가이드

## 📋 개요

LawFirmAI의 LangGraph 워크플로우 시스템은 상태 기반 AI 워크플로우 프레임워크를 활용하여 고도화된 법률 질문 처리를 제공합니다.

## 🚀 주요 기능

### 1. 상태 기반 워크플로우
- **LangGraph 프레임워크**: 상태 기반 AI 워크플로우 관리
- **Enhanced Legal Question Workflow**: 향상된 법률 질문 처리 파이프라인
- **통합 워크플로우 서비스**: 모든 워크플로우 컴포넌트 통합 관리

### 2. 벡터 검색 통합
- **벡터 스토어 통합**: LangGraph 워크플로우에 벡터 스토어 직접 통합
- **하이브리드 검색**: 벡터 검색 우선, 데이터베이스 검색 보완
- **다중 인덱스 지원**: 여러 벡터 인덱스 경로에서 자동 로딩

### 3. 동적 신뢰도 계산
- **응답 길이 기반**: 긴 응답일수록 높은 신뢰도
- **문서 수 기반**: 검색된 문서 수에 따른 신뢰도 조정
- **질문 유형별 가중치**: 법률 자문, 절차 안내 등 유형별 차등 적용
- **법률 키워드 분석**: 법률 용어 포함 여부에 따른 품질 점수

### 4. 메모리 최적화
- **벡터 스토어 메모리 임계값**: 1500MB로 조정
- **적극적 메모리 정리**: 3초 이상 처리 시 자동 메모리 정리
- **가비지 컬렉션**: 매번 실행으로 메모리 효율성 향상

## 🏗️ 아키텍처

### 워크플로우 구조
```
EnhancedLegalQuestionWorkflow
├── classify_query()           # 질문 분류
├── retrieve_documents()       # 문서 검색 (벡터+DB)
├── generate_answer_enhanced() # 답변 생성
├── format_response()          # 응답 포맷팅
└── _calculate_dynamic_confidence() # 동적 신뢰도 계산
```

### 컴포넌트 구성
- **LegalPromptTemplates**: 법률 프롬프트 템플릿 관리
- **LegalKeywordMapper**: 법률 키워드 매핑
- **LegalDataConnector**: 데이터베이스 연결
- **PerformanceOptimizer**: 성능 최적화
- **LegalVectorStore**: 벡터 스토어 관리

## 📊 성능 지표

### 현재 성능 (2025-01-18)
| 지표 | 값 | 설명 |
|------|-----|------|
| **평균 처리 시간** | 21.810초 | LangGraph 워크플로우 기반 질문 처리 |
| **성공률** | 100% | 모든 테스트 케이스 통과 |
| **평균 신뢰도** | 0.30 | 동적 신뢰도 계산 시스템 |
| **메모리 사용량** | 1500MB 임계값 | 최적화된 메모리 관리 |
| **벡터 검색 통합** | 활성화 | LangGraph 워크플로우에 벡터 스토어 통합 |
| **하이브리드 검색** | 벡터+DB | 벡터 검색 우선, 데이터베이스 보완 |

## 🔧 사용법

### 1. 기본 사용법
```python
from source.services.langgraph_workflow.integrated_workflow_service import IntegratedWorkflowService
from source.utils.langgraph_config import langgraph_config

# 워크플로우 서비스 초기화
workflow_service = IntegratedWorkflowService(langgraph_config)

# 질문 처리
result = await workflow_service.process_query(
    query="퇴직금 계산 방법과 지급 시기를 알려주세요",
    context="노동법 관련 질문",
    session_id="user_session_123",
    user_id="user_001"
)

print(f"답변: {result['response']}")
print(f"신뢰도: {result['confidence']}")
print(f"처리 시간: {result['processing_time']}초")
```

### 2. API 엔드포인트 사용
```python
import requests

# LangGraph 기반 고도화된 질문 처리
response = requests.post(
    "http://localhost:8000/api/v1/chat/langgraph-enhanced",
    json={
        "message": "퇴직금 계산 방법과 지급 시기를 알려주세요",
        "session_id": "user_session_123",
        "context": "노동법 관련 질문",
        "user_id": "user_001"
    }
)

result = response.json()
print(f"답변: {result['response']}")
print(f"신뢰도: {result['confidence']}")
print(f"처리 시간: {result['processing_time']}초")
print(f"검색 결과 수: {len(result['sources'])}")
```

## 🛠️ 설정

### 환경 변수
```bash
# LangGraph 설정
export LANGGRAPH_ENABLED="true"
export LANGGRAPH_CHECKPOINT_DB="./data/checkpoints/langgraph.db"

# 벡터 스토어 설정
export VECTOR_STORE_PATH="./data/embeddings/faiss_index"
export EMBEDDING_MODEL="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"

# Google AI 설정 (필수)
export GOOGLE_API_KEY="your_google_api_key"

# Langfuse 모니터링 설정 (선택사항)
export LANGFUSE_PUBLIC_KEY="pk-lf-your-public-key"
export LANGFUSE_SECRET_KEY="sk-lf-your-secret-key"
export LANGFUSE_HOST="https://cloud.langfuse.com"
```

### 벡터 인덱스 경로
시스템은 다음 순서로 벡터 인덱스를 로드합니다:
1. `data/embeddings/ml_enhanced_ko_sroberta/ml_enhanced_faiss_index` (최대 데이터셋)
2. `data/embeddings/ml_enhanced_ko_sroberta_precedents/ml_enhanced_faiss_index` (판례 데이터)
3. `data/embeddings/legal_vector_index` (기본 데이터)

## 🧪 테스트

### 종합 테스트 실행
```bash
# LangGraph 워크플로우 기반 종합 테스트
python tests/demos/final_comprehensive_test_simple.py
```

### 테스트 결과 예시
```
📊 종합 답변 품질 테스트 결과
==================================================
총 테스트: 5
성공한 테스트: 5
실패한 테스트: 0
제한된 테스트: 0
총 실행 시간: 125.20초
평균 신뢰도: 0.30
평균 처리 시간: 21.810초

🔧 생성 방법별 분석
------------------------------
langgraph_workflow: 5개, 평균 신뢰도: 0.30, 평균 시간: 21.810초
```

## 🔍 문제 해결

### 벡터 검색 문제
**문제**: "Search failed: Model or index not initialized"
**해결책**:
1. 벡터 인덱스 파일 존재 확인
2. 벡터 스토어 초기화 로그 확인
3. 메모리 사용량 확인 (1500MB 임계값)

### 메모리 사용량 문제
**문제**: 높은 메모리 사용량 (4000MB+)
**해결책**:
1. 벡터 스토어 메모리 임계값 조정
2. 적극적 메모리 정리 활성화
3. 가비지 컬렉션 실행

### 신뢰도 계산 문제
**문제**: 모든 답변의 신뢰도가 동일
**해결책**:
1. 동적 신뢰도 계산 로직 확인
2. 응답 길이, 문서 수, 질문 유형 분석
3. 법률 키워드 포함 여부 확인

## 📈 향후 개선 계획

### 1. 벡터 검색 최적화
- 벡터 인덱스 로딩 안정성 개선
- 검색 결과 품질 향상
- 메모리 사용량 추가 최적화

### 2. 신뢰도 계산 개선
- 더 정교한 신뢰도 계산 알고리즘
- 사용자 피드백 기반 신뢰도 조정
- 도메인별 신뢰도 가중치 적용

### 3. 성능 최적화
- 워크플로우 처리 시간 단축
- 병렬 처리 최적화
- 캐싱 시스템 개선

## 📚 관련 문서

- [Enhanced Chat Service 가이드](enhanced_chat_service_guide.md)
- [벡터 검색 시스템 가이드](vector_search_guide.md)
- [성능 최적화 보고서](performance_optimization_report.md)
- [API 문서](../api/api_documentation.md)

---

*이 문서는 LawFirmAI 프로젝트의 LangGraph 워크플로우 시스템에 대한 종합 가이드입니다. 최신 업데이트: 2025-01-18*
