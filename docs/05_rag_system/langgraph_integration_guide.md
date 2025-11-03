# LangGraph 워크플로우 통합 가이드

## 개요

LawFirmAI는 LangGraph를 활용하여 복잡한 법률 질문 처리 워크플로우를 구현한 상태 기반 시스템입니다. 이 문서는 현재 구현된 워크플로우의 구조, 기능, 사용법을 상세히 설명합니다.

## 주요 기능

### 핵심 기능

1. **긴급도 평가 (Urgency Assessment)**
   - 질문의 긴급도를 자동 판단 (critical, high, medium, low)
   - 긴급 유형 분류 (법적 마감, 소송 진행, 일반 문의)
   - 키워드 기반 감정 및 의도 분석

2. **법률 분야 분류 (Legal Field Classification)**
   - 민사법, 형사법, 가족법, 상사법, 노동법 등 자동 분류
   - LegalDomain enum 기반 분류 시스템
   - 법률 용어 추출 및 확장

3. **법령 검증 (Legal Basis Validation)**
   - 법령 인용 검증
   - 구 법령 자동 감지
   - 법적 근거 유효성 확인

4. **문서 분석 (Document Analysis)**
   - 계약서, 고소장 등 업로드 문서 자동 분석
   - 핵심 조항 추출
   - 잠재적 문제점 식별

5. **전문가 라우팅 (Expert Router)**
   - 질문 복잡도 평가
   - 전문 분야별 서브그래프 라우팅
   - 가족법/기업법/지적재산권 전문가 라우팅

6. **멀티턴 대화 처리 (Multi-turn Conversation)**
   - 대화 맥락 유지
   - 대명사 해결 및 질문 확장
   - 세션 기반 대화 이력 관리

### 최적화 기능

- **State 최적화**: 메모리 효율을 위한 상태 관리
- **문서 요약**: 대용량 문서 자동 요약
- **검색 결과 선별**: 관련성 높은 문서만 선택
- **처리 단계 제한**: 불필요한 단계 축소

## 아키텍처

### 워크플로우 그래프

```
Entry Point
    ↓
classify_query (질문 분류)
    ↓
assess_urgency (긴급도 평가)
    ↓
analyze_document? (문서 분석 - 조건부)
    ↓
resolve_multi_turn (멀티턴 처리)
    ↓
extract_keywords (키워드 추출)
    ↓
retrieve_documents (문서 검색)
    ↓
process_legal_terms (법률 용어 처리)
    ↓
generate_answer_enhanced (답변 생성)
    ↓
validate_answer_quality (답변 검증)
    ↓
enhance_answer_structure (답변 구조화)
    ↓
apply_visual_formatting (시각적 포맷팅)
    ↓
prepare_final_response (최종 응답 준비)
    ↓
END
```

### 주요 노드 설명

#### 1. classify_query
- **기능**: 질문 유형 분류 (질문형, 절차형, 비교형 등)
- **결과**: query_type, confidence, legal_field, legal_domain
- **폴백**: 키워드 기반 분류

#### 2. assess_urgency
- **기능**: 긴급도 평가
- **결과**: urgency_level (critical/high/medium/low), urgency_reasoning, emergency_type
- **도구**: EmotionIntentAnalyzer

#### 3. analyze_document
- **기능**: 업로드된 문서 분석
- **결과**: document_type, document_analysis, key_clauses, potential_issues
- **도구**: LegalDocumentProcessor
- **조건**: uploaded_document 존재 시 실행

#### 4. route_expert
- **기능**: 전문가 서브그래프로 라우팅
- **결과**: complexity_level, requires_expert, expert_subgraph
- **조건**: 복잡도가 높고 특정 법률 분야인 경우

#### 5. resolve_multi_turn
- **기능**: 멀티턴 대화 처리 및 대명사 해결
- **결과**: search_query (해결된 쿼리), is_multi_turn, conversation_history
- **도구**: MultiTurnQuestionHandler, ConversationManager

#### 6. extract_keywords
- **기능**: 키워드 추출 및 AI 기반 확장
- **결과**: extracted_keywords, search_query, ai_keyword_expansion
- **도구**: LegalKeywordMapper, AIKeywordGenerator

#### 7. retrieve_documents
- **기능**: 하이브리드 검색 (벡터 + 키워드)
- **결과**: retrieved_docs (최대 10개, 요약됨)
- **도구**: SemanticSearchEngine, HybridSearchEngine

#### 8. process_legal_terms
- **기능**: 법률 용어 추출 및 통합
- **결과**: legal_references
- **도구**: TermIntegrator

#### 9. generate_answer_enhanced
- **기능**: LLM 기반 답변 생성
- **결과**: answer
- **도구**: Google Gemini (우선), Ollama (백업)

#### 10. validate_answer_quality
- **기능**: 답변 품질 검증
- **결과**: quality_check_passed
- **도구**: LegalBasisValidator

#### 11. enhance_answer_structure
- **기능**: 답변 구조화 및 법적 근거 강화
- **결과**: answer, legal_citations, structure_confidence
- **도구**: AnswerStructureEnhancer

#### 12. apply_visual_formatting
- **기능**: 시각적 포맷팅 적용
- **결과**: answer (포맷팅됨)
- **도구**: AnswerFormatter

#### 13. prepare_final_response
- **기능**: 최종 응답 준비 및 State 최적화
- **결과**: sources, metadata
- **최적화**: processing_steps, errors, retrieved_docs pruning

## 상태 (State) 정의

### LegalWorkflowState

```python
class LegalWorkflowState(TypedDict):
    """법률 워크플로우 상태 정의 - Flat 구조"""
    
    # 입력
    query: str
    session_id: str
    
    # 질문 분류
    query_type: str
    confidence: float
    legal_field: str
    legal_domain: str
    
    # 긴급도 평가
    urgency_level: str
    urgency_reasoning: str
    emergency_type: Optional[str]
    
    # 법령 검증
    legal_validity_check: bool
    legal_basis_validation: Optional[Dict[str, Any]]
    outdated_laws: List[str]
    
    # 문서 분석
    document_type: Optional[str]
    document_analysis: Optional[Dict[str, Any]]
    key_clauses: List[Dict[str, Any]]
    potential_issues: List[Dict[str, Any]]
    
    # 전문가 라우팅
    complexity_level: str
    requires_expert: bool
    expert_subgraph: Optional[str]
    
    # 멀티턴 처리
    is_multi_turn: bool
    multi_turn_confidence: float
    conversation_history: List[Dict[str, Any]]
    conversation_context: Optional[Dict[str, Any]]
    
    # 키워드 & 검색
    extracted_keywords: List[str]
    search_query: str
    ai_keyword_expansion: Optional[Dict[str, Any]]
    retrieved_docs: List[Dict[str, Any]]
    
    # 분석 & 답변
    analysis: Optional[str]
    legal_references: List[str]
    structure_confidence: float
    legal_citations: Optional[List[Dict[str, Any]]]
    answer: str
    sources: List[str]
    
    # 처리 과정
    processing_steps: Annotated[List[str], add]
    errors: Annotated[List[str], add]
    metadata: Dict[str, Any]
    
    # 성능
    processing_time: float
    tokens_used: int
    
    # 재시도 제어
    retry_count: int
    quality_check_passed: bool
    needs_enhancement: bool
```

### State 최적화

State 크기를 줄이기 위해 다음과 같은 최적화가 적용됩니다:

```python
# Configuration (state_utils.py)
MAX_RETRIEVED_DOCS = 10              # 검색된 문서 수 최대값
MAX_DOCUMENT_CONTENT_LENGTH = 500    # 문서 content 최대 길이
MAX_CONVERSATION_HISTORY = 5        # 대화 이력 최대 개수
MAX_PROCESSING_STEPS = 20            # 처리 단계 최대 개수
```

#### 문서 요약 (Pruning)

- `retrieved_docs`: 최대 10개만 유지, 각 문서의 content는 500자로 제한
- `conversation_history`: 최근 5개만 유지
- `processing_steps`: 최근 20개만 유지

#### 요약 로직

```python
def summarize_document(doc: Dict[str, Any], max_content_length: int = 500):
    """문서 content를 앞뒤 보존하며 요약"""
    # 앞부분 + "... (truncated) ..." + 뒷부분
    # 최종 길이가 max_content_length를 초과하지 않도록 조정
```

## 설치 및 설정

### 의존성 설치

```bash
pip install langgraph>=0.2.0
pip install langchain-core>=0.2.0
pip install langchain-google-genai
```

### 환경 변수 설정

`.env` 파일에 다음 설정 추가:

```bash
# LLM 설정
LLM_PROVIDER=google              # google 또는 ollama
GOOGLE_API_KEY=your_api_key      # Google Gemini API 키
GOOGLE_MODEL=gemini-2.5-flash-lite

# Ollama 설정 (백업)
OLLAMA_BASE_URL=http://localhost:11434
OLLAMA_MODEL=qwen2.5:7b

# LangGraph 설정
LANGGRAPH_ENABLED=true
RECURSION_LIMIT=100              # 워크플로우 복잡도에 맞게 증가

# State 최적화
MAX_RETRIEVED_DOCS=10
MAX_DOCUMENT_CONTENT_LENGTH=500
MAX_CONVERSATION_HISTORY=5
MAX_PROCESSING_STEPS=20
```

## 사용 방법

### 기본 사용법

```python
from core.agents.workflow_service import LangGraphWorkflowService
from infrastructure.utils.langgraph_config import LangGraphConfig

# 설정 및 서비스 초기화
config = LangGraphConfig.from_env()
service = LangGraphWorkflowService(config)

# 질문 처리
result = await service.process_query("계약서 작성 시 주의사항은?", "session_id")

# 결과 확인
print(result["answer"])
print(f"신뢰도: {result['confidence']}")
print(f"법률 참조: {result.get('legal_references', [])}")
```

### 세션 기반 대화

```python
session_id = "user-session-123"

# 첫 번째 질문
result1 = await service.process_query(
    "이혼 절차는 어떻게 되나요?",
    session_id=session_id
)

# 멀티턴 질문 (대명사 자동 해결)
result2 = await service.process_query(
    "그것에 대해 더 자세히 알려주세요",
    session_id=session_id
)
```

### 문서 분석 포함

```python
# 문서 업로드 및 분석
result = await service.process_query(
    "이 계약서를 검토해주세요",
    session_id=session_id,
    document={
        "type": "contract",
        "content": "계약서 내용..."
    }
)

# 분석 결과 확인
print(result.get("document_analysis"))
print(result.get("key_clauses"))
print(result.get("potential_issues"))
```

### 전문가 라우팅

```python
# 복잡한 질문은 자동으로 전문가 서브그래프로 라우팅
result = await service.process_query(
    "복잡한 이혼 소송 절차 및 재산 분할에 대해 알려주세요",
    session_id=session_id
)

# 라우팅 정보 확인
print(f"복잡도: {result.get('complexity_level')}")
print(f"전문가 필요: {result.get('requires_expert')}")
print(f"전문가 서브그래프: {result.get('expert_subgraph')}")
```

## 워크플로우 상세

### 긴급도 평가 로직

```python
def assess_urgency(query: str) -> Dict[str, Any]:
    """긴급도 평가"""
    
    # 1. 키워드 기반 긴급도 평가
    if "긴급" in query or "즉시" in query:
        urgency_level = "critical"
    elif "오늘" in query or "내일" in query:
        urgency_level = "high"
    else:
        urgency_level = "medium"
    
    # 2. 긴급 유형 판별
    if "마감" in query or "기한" in query:
        emergency_type = "legal_deadline"
    elif "소송" in query or "재판" in query:
        emergency_type = "case_progress"
    else:
        emergency_type = None
    
    return {
        "urgency_level": urgency_level,
        "emergency_type": emergency_type,
        "urgency_reasoning": f"키워드 분석 결과: {urgency_level}"
    }
```

### 문서 분석 로직

```python
def analyze_document(doc_text: str) -> Dict[str, Any]:
    """문서 분석"""
    
    # 1. 문서 유형 감지
    if "계약" in doc_text:
        doc_type = "contract"
    elif "고소" in doc_text or "소송" in doc_text:
        doc_type = "complaint"
    else:
        doc_type = "general"
    
    # 2. 핵심 조항 추출
    key_clauses = extract_key_clauses(doc_text)
    
    # 3. 잠재적 문제점 식별
    potential_issues = identify_issues(doc_text)
    
    return {
        "document_type": doc_type,
        "key_clauses": key_clauses,
        "potential_issues": potential_issues,
        "summary": create_summary(doc_text)
    }
```

### 복잡도 평가 로직

```python
def assess_complexity(state: Dict[str, Any]) -> str:
    """질문 복잡도 평가"""
    
    complexity_score = 0
    
    # 복잡도 지표
    indicators = {
        "query_length": len(state["query"]),
        "num_keywords": len(state.get("extracted_keywords", [])),
        "has_document": bool(state.get("document_analysis")),
        "high_urgency": state.get("urgency_level") in ["high", "critical"],
        "multiple_issues": len(state.get("potential_issues", [])) > 2
    }
    
    # 점수 계산
    if indicators["query_length"] > 200:
        complexity_score += 2
    if indicators["num_keywords"] > 10:
        complexity_score += 2
    if indicators["has_document"]:
        complexity_score += 3
    if indicators["high_urgency"]:
        complexity_score += 1
    if indicators["multiple_issues"]:
        complexity_score += 2
    
    # 복잡도 판정
    if complexity_score >= 7:
        return "complex"
    elif complexity_score >= 4:
        return "medium"
    else:
        return "simple"
```

## 성능 최적화

### State 크기 제한

워크플로우 실행 중 State 크기를 다음과 같이 제한합니다:

| 필드 | 최대값 | 설명 |
|------|--------|------|
| retrieved_docs | 10개 | 검색된 문서 수 |
| document content | 500자 | 각 문서의 content 길이 |
| conversation_history | 5개 | 대화 이력 수 |
| processing_steps | 20개 | 처리 단계 수 |

### Pruning 전략

각 노드에서 State를 자동으로 정제합니다:

```python
# retrieve_documents 노드
state["retrieved_docs"] = prune_retrieved_docs(
    state["retrieved_docs"],
    max_items=MAX_RETRIEVED_DOCS,
    max_content_per_doc=MAX_DOCUMENT_CONTENT_LENGTH
)

# _add_step 메서드
state["processing_steps"] = prune_processing_steps(
    state["processing_steps"],
    max_items=MAX_PROCESSING_STEPS
)
```

## 트러블슈팅

### 일반적인 문제

#### 1. Recursion Limit 도달

**증상**: `Recursion limit of 100 reached without hitting a stop condition`

**원인**: 워크플로우가 너무 많은 단계를 실행

**해결**:
```python
# workflow_service.py에서 recursion_limit 증가
self.app = self.legal_workflow.graph.compile(
    checkpointer=None,
    recursion_limit=100  # 이미 설정됨
)
```

#### 2. LangSmith 데이터 크기 제한

**증상**: `field size exceeds maximum allowed size of 209715200 bytes`

**원인**: State가 너무 큼 (약 1.6GB)

**해결**: LangSmith 트레이싱이 이미 비활성화되어 있음

```python
# workflow_service.py
os.environ["LANGCHAIN_TRACING_V2"] = "false"
```

#### 3. 문서 분석 실패

**증상**: `No document provided for analysis`

**원인**: document_analysis 필드에 문서가 없음

**해결**: 문서를 metadata나 session context에 전달

#### 4. LLM 초기화 실패

**증상**: `Failed to initialize Google Gemini LLM`

**원인**: API 키가 없거나 잘못됨

**해결**:
```bash
# .env 파일 설정
GOOGLE_API_KEY=your_api_key_here
```

### 디버깅

#### 로그 레벨 설정

```python
import logging

# DEBUG 레벨로 설정
logging.basicConfig(level=logging.DEBUG)

# LangGraph 로거만 설정
logger = logging.getLogger("source.services.langgraph")
logger.setLevel(logging.DEBUG)
```

#### State 확인

```python
# 결과에서 State 확인
result = await service.process_query("test query")

print(f"Processing steps: {result.get('processing_steps', [])}")
print(f"Retrieved docs: {len(result.get('retrieved_docs', []))}개")
print(f"Errors: {result.get('errors', [])}")
```

## 예제 코드

### 완전한 예제

```python
import asyncio
from core.agents.workflow_service import LangGraphWorkflowService
from infrastructure.utils.langgraph_config import LangGraphConfig

async def main():
    # 설정 및 서비스 초기화
    config = LangGraphConfig.from_env()
    service = LangGraphWorkflowService(config)
    
    # 질문 목록
    questions = [
        "계약서 작성 시 주의사항은?",
        "이혼 절차는 어떻게 되나요?",
        "손해배상 관련 판례를 찾아주세요"
    ]
    
    session_id = "demo-session"
    
    for i, question in enumerate(questions, 1):
        print(f"\n=== 질문 {i}: {question} ===")
        
        result = await service.process_query(
            question,
            session_id=session_id
        )
        
        # 결과 출력
        print(f"답변: {result['answer']}")
        print(f"신뢰도: {result['confidence']}")
        print(f"처리 시간: {result.get('processing_time', 0):.2f}초")
        print(f"긴급도: {result.get('urgency_level', 'unknown')}")
        print(f"법률 분야: {result.get('legal_field', 'unknown')}")
        
        if result.get('legal_references'):
            print(f"법률 참조: {result['legal_references']}")
        
        if result.get('sources'):
            print(f"소스 수: {len(result['sources'])}개")
        
        if result.get('processing_steps'):
            print(f"처리 단계: {len(result['processing_steps'])}개")

if __name__ == "__main__":
    asyncio.run(main())
```

### 서비스 상태 확인

```python
# 서비스 상태 확인
status = service.get_service_status()

print(f"서비스 상태: {status['status']}")
print(f"워크플로우 초기화: {status.get('workflow_initialized', False)}")
print(f"LLM 제공자: {status.get('llm_provider', 'unknown')}")
```

## 향후 확장 계획

### Phase 1: 전문가 서브그래프 구현 ✅
- [x] 가족법 전문가 서브그래프
- [x] 기업법 전문가 서브그래프
- [x] 지적재산권 전문가 서브그래프

### Phase 2: 실시간 스트리밍
- [ ] 스트리밍 워크플로우 구현
- [ ] 점진적 답변 제공
- [ ] WebSocket 통합

### Phase 3: 에이전트 시스템
- [ ] 멀티 에이전트 아키텍처
- [ ] 에이전트 간 협업
- [ ] 에이전트 메모리 관리

### Phase 4: 고급 기능
- [ ] PostgreSQL 체크포인트 저장소
- [ ] LangSmith 재활성화 (최적화 후)
- [ ] 동적 워크플로우 생성

## API 참조

### LangGraphWorkflowService

#### 주요 메서드

- `process_query(query, session_id=None)`: 질문 처리
- `get_service_status()`: 서비스 상태 확인
- `get_service_statistics()`: 통계 정보 조회

#### 응답 구조

```python
{
    "answer": str,                      # 최종 답변
    "confidence": float,                # 신뢰도 (0.0 ~ 1.0)
    "sources": List[str],               # 소스 목록
    "legal_references": List[str],      # 법률 참조
    "urgency_level": str,               # 긴급도
    "legal_field": str,                  # 법률 분야
    "processing_steps": List[str],      # 처리 단계
    "processing_time": float,           # 처리 시간
    "session_id": str,                  # 세션 ID
    "errors": List[str]                 # 오류 목록
}
```

## 관련 파일

- `lawfirm_langgraph/source/services/legal_workflow_enhanced.py` - 워크플로우 구현
- `lawfirm_langgraph/source/utils/state_definitions.py` - State 정의
- `lawfirm_langgraph/source/services/workflow_service.py` - 워크플로우 서비스
- `lawfirm_langgraph/source/utils/state_utils.py` - State 최적화 유틸리티
- `infrastructure/utils/langgraph_config.py` - 설정 관리

> ⚠️ **참고**: `core/agents/` 경로는 레거시입니다. 위의 `lawfirm_langgraph/source/` 경로를 사용하세요.

이 가이드를 통해 LawFirmAI의 LangGraph 워크플로우를 효과적으로 활용할 수 있습니다. 추가 질문이나 문제가 있으시면 프로젝트 이슈를 통해 문의해주세요.
