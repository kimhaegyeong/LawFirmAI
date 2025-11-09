# Core Modules 가이드

## 개요

`lawfirm_langgraph/core/` 모듈은 LawFirmAI 프로젝트의 핵심 비즈니스 로직을 담당합니다. 이 문서는 `lawfirm_langgraph/core/` 모듈의 각 컴포넌트의 역할과 사용법을 설명합니다.

## 디렉토리 구조

```
lawfirm_langgraph/core/
├── agents/          # LangGraph 워크플로우 에이전트
├── services/        # 비즈니스 서비스
├── data/            # 데이터 레이어
├── models/          # AI 모델
└── utils/           # 유틸리티
```

## 1. Agents 모듈 (`lawfirm_langgraph/core/agents/`)

### 1.1 LangGraph 워크플로우

#### workflow_service.py
**역할**: LangGraph 워크플로우 서비스 메인 진입점

**주요 클래스**:
- `LangGraphWorkflowService`: 워크플로우 실행 서비스

**사용 예시**:
```python
from lawfirm_langgraph.config.langgraph_config import LangGraphConfig
from lawfirm_langgraph.core.agents.workflow_service import LangGraphWorkflowService

# 설정 초기화
config = LangGraphConfig.from_env()

# 워크플로우 서비스 초기화
workflow = LangGraphWorkflowService(config)

# 질문 처리
result = await workflow.process_query_async("계약 해지 조건은?", "session_id")
```

#### legal_workflow_enhanced.py
**역할**: 향상된 법률 질문 처리 워크플로우 구현

**주요 클래스**:
- `EnhancedLegalQuestionWorkflow`: 향상된 워크플로우 클래스
- `RetryCounterManager`: 재시도 카운터 관리

**주요 노드**:
- `classify_query`: 질문 분류
- `assess_urgency`: 긴급도 평가
- `resolve_multi_turn`: 멀티턴 처리
- `retrieve_documents`: 문서 검색
- `generate_answer_enhanced`: 답변 생성

### 1.2 State 관리

#### state_definitions.py
**역할**: LangGraph State 정의

**주요 타입**:
- `LegalWorkflowState`: 워크플로우 State (flat 구조)
- `AgentWorkflowState`: 에이전트 State
- `StreamingWorkflowState`: 스트리밍 State

#### modular_states.py
**역할**: 모듈화된 State 구조 (11개 그룹)

**State 그룹**:
- `InputState`: 입력 데이터
- `ClassificationState`: 분류 결과
- `SearchState`: 검색 결과
- `AnalysisState`: 분석 결과
- `AnswerState`: 답변 데이터
- `ValidationState`: 검증 결과
- `DocumentState`: 문서 분석
- `MultiTurnState`: 멀티턴 처리
- `ControlState`: 제어 플래그
- `CommonState`: 공통 메타데이터

#### state_helpers.py
**역할**: State 접근 헬퍼 함수

**주요 함수**:
- `get_input(state)`: Input State 추출
- `get_query(state)`: 쿼리 추출
- `get_classification(state)`: 분류 결과 추출

#### state_utils.py
**역할**: State 최적화 유틸리티

**주요 함수**:
- `prune_retrieved_docs()`: 검색 문서 정리
- `prune_processing_steps()`: 처리 단계 정리
- `summarize_document()`: 문서 요약

**설정 상수**:
- `MAX_RETRIEVED_DOCS`: 최대 검색 문서 수 (10)
- `MAX_DOCUMENT_CONTENT_LENGTH`: 최대 문서 길이 (500)
- `MAX_CONVERSATION_HISTORY`: 최대 대화 이력 (5)
- `MAX_PROCESSING_STEPS`: 최대 처리 단계 (20)

#### state_reduction.py
**역할**: State 크기 최적화

**주요 기능**:
- State 필드 자동 제거
- 불필요한 데이터 정리
- 메모리 사용량 감소

#### state_adapter.py
**역할**: State 구조 변환 (flat ↔ nested)

**주요 함수**:
- `to_nested_state()`: flat → nested 변환
- `to_flat_state()`: nested → flat 변환

### 1.3 노드 관리

#### node_wrappers.py
**역할**: 노드 래퍼 및 최적화 데코레이터

**주요 데코레이터**:
- `with_state_optimization()`: State 최적화 적용

**사용 예시**:
```python
from lawfirm_langgraph.core.agents.node_wrappers import with_state_optimization

@with_state_optimization("classify_query")
def classify_query(state: LegalWorkflowState) -> LegalWorkflowState:
    # 노드 로직
    return state
```

#### node_input_output_spec.py
**역할**: 노드별 입출력 사양 정의

**주요 클래스**:
- `NodeIOSpec`: 노드 입출력 사양
- `NodeCategory`: 노드 카테고리

**주요 함수**:
- `get_node_spec()`: 노드 사양 조회
- `validate_node_input()`: 입력 검증
- `get_required_state_groups()`: 필요한 State 그룹 반환

### 1.4 데이터 커넥터

#### legal_data_connector_v2.py
**역할**: 법률 데이터 커넥터 (벡터 스토어 + 데이터베이스)

**주요 클래스**:
- `LegalDataConnectorV2`: 통합 데이터 커넥터

**주요 메서드**:
- `retrieve_documents()`: 문서 검색
- `search_by_similarity()`: 유사도 검색
- `search_by_keywords()`: 키워드 검색

#### keyword_mapper.py
**역할**: 법률 키워드 매핑

**주요 클래스**:
- `LegalKeywordMapper`: 법률 키워드 매퍼

### 1.5 최적화 및 모니터링

#### performance_optimizer.py
**역할**: 성능 최적화

**주요 기능**:
- State 최적화
- 메모리 관리
- 캐시 관리

#### query_optimizer.py
**역할**: 쿼리 최적화

**주요 기능**:
- 검색 쿼리 개선
- 키워드 확장
- 동의어 처리

#### search_performance_monitor.py
**역할**: 검색 성능 모니터링

**주요 기능**:
- 검색 시간 측정
- 결과 품질 평가
- 성능 메트릭 수집

### 1.6 기타 유틸리티

#### workflow_logger.py
**역할**: 워크플로우 로깅

**주요 기능**:
- 노드 실행 로깅
- State 변화 추적
- 오류 로깅

#### checkpoint_manager.py
**역할**: 체크포인트 관리

**주요 기능**:
- State 저장
- 워크플로우 재개
- 이력 관리

## 2. Services 모듈 (`lawfirm_langgraph/core/services/`)

### 2.1 검색 서비스

#### hybrid_search_engine.py
**역할**: 하이브리드 검색 엔진 (의미적 + 정확한 매칭)

**주요 클래스**:
- `HybridSearchEngine`: 통합 검색 엔진

**사용 예시**:
```python
from lawfirm_langgraph.core.services.hybrid_search_engine import HybridSearchEngine

engine = HybridSearchEngine()
results = engine.search("민법 제543조", question_type="law_inquiry")
```

#### semantic_search_engine.py
**역할**: 의미적 검색 엔진 (FAISS 벡터)

**주요 클래스**:
- `SemanticSearchEngine`: 의미적 검색

**사용 예시**:
```python
from lawfirm_langgraph.core.services.semantic_search_engine import SemanticSearchEngine

engine = SemanticSearchEngine()
results = engine.search("계약 해지", k=5)
```

#### exact_search_engine.py
**역할**: 정확한 매칭 검색 (SQLite FTS)

**주요 클래스**:
- `ExactSearchEngine`: 정확한 매칭 검색

**사용 예시**:
```python
from lawfirm_langgraph.core.services.exact_search_engine import ExactSearchEngine

engine = ExactSearchEngine()
results = engine.search("민법 제543조", k=5)
```

#### question_classifier.py
**역할**: 질문 유형 분류

**주요 클래스**:
- `QuestionClassifier`: 질문 분류기
- `QuestionType`: 질문 유형 enum

**질문 유형**:
- `law_inquiry`: 법령 조회
- `precedent_search`: 판례 검색
- `document_analysis`: 문서 분석
- `general_question`: 일반 질문

#### result_merger.py
**역할**: 검색 결과 병합 및 재순위화

**주요 클래스**:
- `ResultMerger`: 결과 병합기
- `ResultRanker`: 결과 재순위화

### 2.2 답변 생성 서비스

#### answer_generator.py
**역할**: LLM 기반 답변 생성

**주요 클래스**:
- `AnswerGenerator`: 답변 생성기

**사용 예시**:
```python
from lawfirm_langgraph.core.services.answer_generator import AnswerGenerator

generator = AnswerGenerator()
answer = generator.generate(query, context)
```

#### context_builder.py
**역할**: 검색 결과 기반 컨텍스트 구축

**주요 클래스**:
- `ContextBuilder`: 컨텍스트 빌더

#### answer_formatter.py
**역할**: 답변 포맷팅

**주요 클래스**:
- `AnswerFormatter`: 답변 포맷터

**주요 기능**:
- Markdown 포맷팅
- 법령 인용 형식화
- 목록/표 구조화

### 2.3 품질 개선 서비스

#### confidence_calculator.py
**역할**: 답변 신뢰도 계산

**주요 클래스**:
- `ConfidenceCalculator`: 신뢰도 계산기

## 3. Data 모듈 (`lawfirm_langgraph/core/data/`)

### 3.1 데이터베이스

#### database.py
**역할**: SQLite 데이터베이스 관리

**주요 클래스**:
- `DatabaseManager`: 데이터베이스 관리자

**주요 메서드**:
- `execute_query()`: 쿼리 실행
- `get_connection()`: 연결 가져오기
- `mark_file_as_processed()`: 파일 처리 표시

**사용 예시**:
```python
from lawfirm_langgraph.core.data.database import DatabaseManager

db = DatabaseManager()
results = db.execute_query("SELECT * FROM assembly_laws LIMIT 10")
```

### 3.2 벡터 스토어

#### vector_store.py
**역할**: FAISS 벡터 스토어 관리

**주요 클래스**:
- `VectorStore`: 벡터 스토어

**주요 메서드**:
- `similarity_search()`: 유사도 검색
- `add_documents()`: 문서 추가
- `load_index()`: 인덱스 로드

**사용 예시**:
```python
from lawfirm_langgraph.core.data.vector_store import VectorStore

vector_store = VectorStore("ko-sroberta-multitask")
results = vector_store.similarity_search("계약 해지", k=5)
```

### 3.3 대화 저장소

#### conversation_store.py
**역할**: 대화 이력 저장

**주요 클래스**:
- `ConversationStore`: 대화 저장소

**주요 메서드**:
- `save_turn()`: 대화 턴 저장
- `get_conversation()`: 대화 조회
- `get_session()`: 세션 조회

### 3.4 데이터 처리

#### data_processor.py
**역할**: 데이터 전처리

**주요 기능**:
- 문서 청킹
- 텍스트 정규화
- 법률 용어 추출

#### legal_term_normalizer.py
**역할**: 법률 용어 정규화

**주요 기능**:
- 법령명 정규화
- 조문번호 표준화
- 약어 확장

## 4. Models 모듈 (`lawfirm_langgraph/core/models/`)

### 4.1 AI 모델

#### sentence_bert.py
**역할**: Sentence-BERT 모델 관리

**주요 클래스**:
- `SentenceBERT`: Sentence-BERT 모델

**주요 메서드**:
- `encode()`: 텍스트 임베딩
- `encode_batch()`: 배치 임베딩

**사용 예시**:
```python
from lawfirm_langgraph.core.models.sentence_bert import SentenceBERT

model = SentenceBERT("ko-sroberta-multitask")
embedding = model.encode("계약 해지 조건")
```

#### gemini_client.py
**역할**: Google Gemini API 클라이언트

**위치**: `lawfirm_langgraph/core/services/gemini_client.py`

**주요 클래스**:
- `GeminiClient`: Gemini 클라이언트

## 사용 예시

### 전체 워크플로우 실행

```python
from lawfirm_langgraph.config.langgraph_config import LangGraphConfig
from lawfirm_langgraph.core.agents.workflow_service import LangGraphWorkflowService

# 1. 설정 초기화
config = LangGraphConfig.from_env()

# 2. 워크플로우 서비스 초기화
workflow = LangGraphWorkflowService(config)

# 3. 질문 처리
result = await workflow.process_query_async(
    query="계약 해지 조건은?",
    session_id="session_123"
)

# 4. 결과 확인
print(f"답변: {result['answer']}")
print(f"신뢰도: {result.get('confidence', 0)}")
print(f"소스: {result.get('sources', [])}")
```

### 검색 엔진 직접 사용

```python
from lawfirm_langgraph.core.services.hybrid_search_engine import HybridSearchEngine

# 하이브리드 검색 엔진 초기화
search_engine = HybridSearchEngine()

# 검색 실행
results = search_engine.search(
    query="계약 해지",
    question_type="law_inquiry"
)

# 결과 확인
for result in results:
    print(f"문서 ID: {result.get('document_id')}")
    print(f"점수: {result.get('score', 0)}")
    print(f"내용: {result.get('content', '')[:100]}...")
```

### 데이터베이스 직접 사용

```python
from lawfirm_langgraph.core.data.database import DatabaseManager

# 데이터베이스 관리자 초기화
db = DatabaseManager()

# 법률 데이터 조회
laws = db.execute_query(
    "SELECT law_name FROM assembly_laws LIMIT 10"
)

for law in laws:
    print(law['law_name'])
```

## 모듈 간 의존성

```
workflow_service
    ↓
legal_workflow_enhanced
    ↓
    ├── services/search (검색)
    ├── services/generation (답변 생성)
    ├── services/enhancement (품질 개선)
    ├── data/database (데이터베이스)
    ├── data/vector_store (벡터 스토어)
    └── models (AI 모델)
```

## 설계 원칙

### 1. 모듈화
각 기능은 독립적인 모듈로 분리되어 재사용성이 높습니다.

### 2. 의존성 최소화
모듈 간 의존성을 최소화하여 테스트와 유지보수가 용이합니다.

### 3. 인터페이스 표준화
동일한 인터페이스를 사용하여 모듈 교체가 쉽습니다.

### 4. 성능 최적화
State 최적화, 캐싱 등을 통해 성능을 향상시킵니다.

## 참고 자료

- [LangGraph 통합 가이드](../03_rag_system/langgraph_integration_guide.md)
- [RAG 아키텍처](../03_rag_system/rag_architecture.md)
- [LangGraph Node I/O](langgraph_node_io.md)
- [데이터베이스 스키마](database_schema.md)
