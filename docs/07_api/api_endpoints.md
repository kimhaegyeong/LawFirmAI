# LawFirmAI API 문서

## 📋 개요

LawFirmAI는 지능형 법률 AI 어시스턴트를 위한 RESTful API를 제공합니다. Phase 3 완료로 성능 모니터링, 사용자 피드백 수집, API v2 통합, HuggingFace Spaces 최적화 등이 추가되었습니다.

## 🚀 주요 기능

### Phase 2 신규 기능
- **질문 유형 분류**: 6가지 질문 유형 자동 분류
- **동적 검색 가중치**: 질문 유형별 법률/판례 검색 비중 자동 조정
- **구조화된 답변**: 질문 유형별 맞춤형 답변 포맷
- **신뢰도 시스템**: 답변의 신뢰성을 수치화하여 제공
- **컨텍스트 최적화**: 토큰 제한 내에서 가장 관련성 높은 정보만 선별
- **법률 용어 확장**: 동의어 및 관련 용어를 통한 검색 정확도 향상

## 🔗 API 엔드포인트

### 기본 정보
- **Base URL**: `http://localhost:8000/api/v1`
- **Content-Type**: `application/json`
- **API Version**: v1.0.0

### 실제 엔드포인트 목록

#### 채팅 엔드포인트
- `POST /chat` - 기본 채팅 메시지 처리
- `POST /chat/stream` - 스트리밍 채팅 응답
- `GET /chat/{session_id}/sources` - 세션별 소스 조회

#### 세션 관리 엔드포인트
- `GET /sessions` - 세션 목록 조회
- `GET /sessions/by-date` - 날짜별 세션 목록 조회
- `POST /sessions` - 새 세션 생성
- `GET /sessions/{session_id}` - 세션 상세 조회
- `PUT /sessions/{session_id}` - 세션 업데이트
- `DELETE /sessions/{session_id}` - 세션 삭제
- `POST /sessions/{session_id}/generate-title` - 세션 제목 생성

#### 히스토리 엔드포인트
- `GET /history` - 대화 히스토리 조회
- `POST /history/export` - 히스토리 내보내기

#### 피드백 엔드포인트
- `POST /feedback` - 피드백 제출

#### 헬스체크 엔드포인트
- `GET /health` - 서비스 상태 확인

### 1. 기본 채팅 엔드포인트

#### `POST /chat`
기본 채팅 기능

**Request Body:**
```json
{
  "message": "손해배상 청구 방법",
  "context": "민법 관련 질문",
  "session_id": "optional_session_id"
}
```

**Response:**
```json
{
  "response": "손해배상 청구 방법은 다음과 같습니다...",
  "confidence": 0.85,
  "sources": ["민법 제750조", "민법 제751조"]
}
```

### 2. 스트리밍 채팅 엔드포인트

#### `POST /chat/stream`
스트리밍 채팅 응답 (Server-Sent Events)

**Request Body:**
```json
{
  "message": "계약 해제 조건",
  "session_id": "session_123"
}
```

**Response:** Server-Sent Events (SSE) 형식으로 스트리밍

**스트리밍 이벤트 타입:**
- `progress`: 진행 상황 알림
- `stream`: 실시간 답변 청크 (토큰 단위)
- `final`: 최종 답변 및 메타데이터
- `error`: 오류 발생 시

**예시 응답:**
```
data: {"type":"progress","content":"답변 생성 중...","timestamp":"2025-11-12T09:30:00"}

data: {"type":"stream","content":"계약","timestamp":"2025-11-12T09:30:01"}

data: {"type":"stream","content":" 해제","timestamp":"2025-11-12T09:30:01"}

data: {"type":"stream","content":" 조건은","timestamp":"2025-11-12T09:30:01"}

...

data: {"type":"final","content":"전체 답변...","sources":[...],"legal_references":[...]}
```

**구현 특징:**
- LangGraph의 `astream_events()`를 사용하여 실시간 이벤트 스트리밍
- `StreamingCallbackHandler`를 통한 LLM 토큰 단위 스트리밍
- `generate_answer_stream` 노드 사용 (환경 변수 `USE_STREAMING_MODE=true`일 때)
- 비동기 큐를 통한 콜백 이벤트 처리

### 3. 소스 조회 엔드포인트

#### `GET /chat/{session_id}/sources`
세션별 소스 조회

**Response:**
```json
{
  "sources": [
    {
      "type": "law",
      "law_name": "민법",
      "article_number": "제543조",
      "content": "계약 해제에 관한 규정",
      "similarity": 0.95
    }
  ]
}
```

### 4. 세션 관리 엔드포인트

#### `GET /sessions`
세션 목록 조회

#### `POST /sessions`
새 세션 생성

#### `GET /sessions/{session_id}`
세션 상세 조회

#### `PUT /sessions/{session_id}`
세션 업데이트

#### `DELETE /sessions/{session_id}`
세션 삭제

#### `POST /sessions/{session_id}/generate-title`
세션 제목 생성

### 5. 히스토리 엔드포인트

#### `GET /history`
대화 히스토리 조회

#### `POST /history/export`
히스토리 내보내기

### 6. 피드백 엔드포인트

#### `POST /feedback`
피드백 제출

### 7. 헬스체크 엔드포인트

#### `GET /health`
서비스 상태 확인

**Response:**
```json
{
  "status": "healthy",
  "timestamp": "2024-01-01T00:00:00Z"
}
```

### 스트리밍 구현 상세

#### 아키텍처
```
Client Request
    ↓
ChatService.stream_final_answer()
    ↓
StreamingCallbackHandler 생성 (asyncio.Queue)
    ↓
LangGraphWorkflowService.app.astream_events()
    ↓
generate_answer_stream 노드 실행
    ↓
LLM.stream() → on_llm_stream 이벤트
    ↓
StreamingCallbackHandler.on_llm_stream()
    ↓
asyncio.Queue에 청크 저장
    ↓
ChatService에서 큐 모니터링 및 SSE 전송
    ↓
Client (실시간 스트리밍 수신)
```

#### 환경 변수 설정
```bash
# 스트리밍 모드 활성화 (기본값: true)
USE_STREAMING_MODE=true   # API용: 스트리밍 노드 사용
USE_STREAMING_MODE=false  # 테스트용: 최종 검증 노드 사용
```

**Request Body:**
```json
{
  "message": "손해배상 관련 판례를 찾아주세요",
  "session_id": "session_123",
  "max_results": 10,
  "include_law_sources": true,
  "include_precedent_sources": true
}
```

**Response:**
```json
{
  "answer": "손해배상 관련 판례를 찾았습니다...",
  "formatted_answer": {
    "formatted_content": "## 관련 판례 분석\n\n### 🔍 판례 분석\n...",
    "sections": {
      "analysis": "판례 분석 내용",
      "precedents": "참고 판례 목록",
      "laws": "적용 법률 목록",
      "confidence": "신뢰도 정보"
    },
    "metadata": {
      "question_type": "precedent_search",
      "confidence_level": "HIGH",
      "confidence_score": 0.89,
      "source_count": {
        "laws": 2,
        "precedents": 5
      },
      "sections_count": 4
    }
  },
  "question_type": "precedent_search",
  "confidence": {
    "confidence": 0.89,
    "reliability_level": "HIGH",
    "similarity_score": 0.92,
    "matching_score": 0.85,
    "answer_quality": 0.90
  },
  "law_sources": [
    {
      "law_name": "민법",
      "article_number": "제750조",
      "content": "불법행위로 인한 손해배상",
      "similarity": 0.90
    }
  ],
  "precedent_sources": [
    {
      "case_name": "손해배상청구 사건",
      "case_number": "2023다12345",
      "court": "서울중앙지방법원",
      "decision_date": "2023.05.15",
      "summary": "불법행위로 인한 손해배상 청구권 인정",
      "similarity": 0.88
    }
  ],
  "search_stats": {
    "total_results": 7,
    "law_results_count": 2,
    "precedent_results_count": 5,
    "search_time": 0.15,
    "question_classification_time": 0.05
  },
  "processing_time": 2.3,
  "warnings": [],
  "recommendations": ["전문가 상담 권장"]
}
```

### 4. 지능형 채팅 엔드포인트 v2 (Phase 2 최신)

#### `POST /chat/intelligent-v2`
모든 개선사항이 통합된 최신 엔드포인트

**Request Body:**
```json
{
  "message": "이혼 절차는 어떻게 진행하나요?",
  "session_id": "session_123",
  "max_results": 10,
  "include_law_sources": true,
  "include_precedent_sources": true,
  "include_conversation_history": true,
  "context_optimization": true,
  "answer_formatting": true
}
```

**Response:**
```json
{
  "answer": "이혼 절차에 대해 설명드리겠습니다...",
  "formatted_answer": {
    "formatted_content": "## 절차 안내\n\n### 📊 절차 개요\n...",
    "sections": {
      "overview": "이혼 절차 개요",
      "steps": "단계별 절차",
      "documents": "필요 서류",
      "timeline": "처리 기간",
      "confidence": "신뢰도 정보"
    },
    "metadata": {
      "question_type": "procedure_guide",
      "confidence_level": "HIGH",
      "confidence_score": 0.87,
      "source_count": {
        "laws": 3,
        "precedents": 2
      },
      "sections_count": 5
    }
  },
  "question_type": "procedure_guide",
  "confidence": {
    "confidence": 0.87,
    "reliability_level": "HIGH",
    "similarity_score": 0.89,
    "matching_score": 0.84,
    "answer_quality": 0.88
  },
  "law_sources": [
    {
      "law_name": "민법",
      "article_number": "제836조",
      "content": "이혼에 관한 규정",
      "similarity": 0.91
    }
  ],
  "precedent_sources": [
    {
      "case_name": "이혼 사건",
      "case_number": "2023가합12345",
      "court": "서울가정법원",
      "decision_date": "2023.03.20",
      "summary": "이혼 절차 관련 판결",
      "similarity": 0.86
    }
  ],
  "search_stats": {
    "total_results": 5,
    "law_results_count": 3,
    "precedent_results_count": 2,
    "search_time": 0.12,
    "question_classification_time": 0.03
  },
  "context_stats": {
    "total_items": 8,
    "total_tokens": 3200,
    "utilization_rate": 0.80,
    "priority_distribution": {
      "high": 3,
      "medium": 4,
      "low": 1
    }
  },
  "processing_time": 1.8,
  "warnings": [],
  "recommendations": ["가정법원 상담 권장"]
}
```

### 5. 검색 엔드포인트

#### `POST /search`
하이브리드 검색 기능

**Request Body:**
```json
{
  "query": "계약 해제",
  "search_type": "hybrid",
  "max_results": 10,
  "ml_enhanced": true
}
```

**Response:**
```json
{
  "results": [
    {
      "type": "law",
      "law_name": "민법",
      "article_number": "제543조",
      "content": "계약 해제에 관한 규정",
      "similarity": 0.95,
      "score": 0.92
    }
  ],
  "total_count": 15,
  "search_type": "hybrid",
  "ml_enhanced": true,
  "processing_time": 0.25
}
```

### 6. 시스템 상태 확인

#### `GET /system/status`
시스템 전체 상태 점검

**Response:**
```json
{
  "timestamp": 1697123456.789,
  "overall_status": "healthy",
  "components": {
    "database": {
      "status": "healthy",
      "total_articles": 180684,
      "connection": "active"
    },
    "vector_store": {
      "status": "healthy",
      "stats": {
        "total_vectors": 196251,
        "index_size_mb": 456.5
      }
    },
    "ai_models": {
      "status": "healthy",
      "question_classifier": "active",
      "test_classification": "general_question"
    },
    "search_engines": {
      "status": "healthy",
      "hybrid_search": "active",
      "test_results_count": 5
    },
    "answer_generator": {
      "status": "healthy",
      "ollama_client": "active",
      "answer_formatter": "active",
      "context_builder": "active",
      "test_answer_length": 245
    }
  },
  "version": "2.0.0"
}
```

## 📊 질문 유형 분류

### 지원하는 질문 유형

| 질문 유형 | 설명 | 법률 가중치 | 판례 가중치 |
|----------|------|------------|------------|
| `precedent_search` | 판례 검색 | 0.2 | 0.8 |
| `law_inquiry` | 법률 문의 | 0.8 | 0.2 |
| `legal_advice` | 법적 조언 | 0.5 | 0.5 |
| `procedure_guide` | 절차 안내 | 0.6 | 0.4 |
| `term_explanation` | 용어 해설 | 0.7 | 0.3 |
| `general_question` | 일반 질문 | 0.4 | 0.4 |

### 질문 유형별 키워드 예시

#### 판례 검색 (`precedent_search`)
- 키워드: "판례", "사건", "법원", "판결", "대법원", "참고판례"
- 패턴: "판례를 찾아주세요", "유사한 사건이 있나요"

#### 법률 문의 (`law_inquiry`)
- 키워드: "법률", "조문", "법령", "규정", "법적근거"
- 패턴: "법률이 무엇인가요", "조문 내용을 알려주세요"

#### 법적 조언 (`legal_advice`)
- 키워드: "조언", "상담", "해결방법", "어떻게", "해야"
- 패턴: "어떻게 해야 하나요", "조언해주세요"

#### 절차 안내 (`procedure_guide`)
- 키워드: "절차", "신청", "제출", "서류", "기간"
- 패턴: "절차는 어떻게", "신청 방법을 알려주세요"

#### 용어 해설 (`term_explanation`)
- 키워드: "의미", "정의", "뜻", "개념", "용어"
- 패턴: "의미가 무엇인가요", "정의를 알려주세요"

## 🔧 신뢰도 시스템

### 신뢰도 계산 요소

1. **검색 결과 유사도** (40%)
   - 벡터 검색 유사도 점수
   - 정확 매칭 점수

2. **법률/판례 매칭 정확도** (30%)
   - 질문 유형과 검색 결과의 일치도
   - 관련성 점수

3. **답변 품질** (30%)
   - 답변의 완성도
   - 구조화 정도

### 신뢰도 수준

| 수준 | 점수 범위 | 설명 |
|------|----------|------|
| `HIGH` | 0.8 이상 | 높은 신뢰도, 전문가 수준 답변 |
| `MEDIUM` | 0.6-0.8 | 보통 신뢰도, 참고용 답변 |
| `LOW` | 0.4-0.6 | 낮은 신뢰도, 추가 확인 필요 |
| `VERY_LOW` | 0.4 미만 | 매우 낮은 신뢰도, 전문가 상담 권장 |

## 📝 에러 처리

### HTTP 상태 코드

| 코드 | 설명 |
|------|------|
| 200 | 성공 |
| 400 | 잘못된 요청 |
| 404 | 리소스 없음 |
| 500 | 서버 오류 |

### 에러 응답 형식

```json
{
  "detail": "에러 메시지",
  "error_code": "ERROR_CODE",
  "timestamp": "2025-10-16T10:30:00Z"
}
```

## 🚀 사용 예시

### Python 클라이언트 예시

```python
import requests
import json

# 기본 설정
BASE_URL = "http://localhost:8000/api/v1"
headers = {"Content-Type": "application/json"}

# 지능형 채팅 요청
def intelligent_chat(message, session_id=None):
    url = f"{BASE_URL}/chat/intelligent-v2"
    data = {
        "message": message,
        "session_id": session_id,
        "max_results": 10,
        "include_law_sources": True,
        "include_precedent_sources": True,
        "include_conversation_history": True,
        "context_optimization": True,
        "answer_formatting": True
    }
    
    response = requests.post(url, headers=headers, json=data)
    return response.json()

# 사용 예시
result = intelligent_chat("손해배상 청구 방법")
print(f"질문 유형: {result['question_type']}")
print(f"신뢰도: {result['confidence']['confidence']:.2%}")
print(f"답변: {result['answer']}")
```

### JavaScript 클라이언트 예시

```javascript
// 지능형 채팅 함수
async function intelligentChat(message, sessionId = null) {
    const url = 'http://localhost:8000/api/v1/chat/intelligent-v2';
    const data = {
        message: message,
        session_id: sessionId,
        max_results: 10,
        include_law_sources: true,
        include_precedent_sources: true,
        include_conversation_history: true,
        context_optimization: true,
        answer_formatting: true
    };
    
    const response = await fetch(url, {
        method: 'POST',
        headers: {
            'Content-Type': 'application/json'
        },
        body: JSON.stringify(data)
    });
    
    return await response.json();
}

// 사용 예시
intelligentChat("계약 해제 조건")
    .then(result => {
        console.log(`질문 유형: ${result.question_type}`);
        console.log(`신뢰도: ${(result.confidence.confidence * 100).toFixed(1)}%`);
        console.log(`답변: ${result.answer}`);
    })
    .catch(error => console.error('Error:', error));
```

## 📚 추가 리소스

- [프로젝트 개요](../01_getting_started/project_overview.md)
- [데이터베이스 스키마](../10_technical_reference/database_schema.md)
- [RAG 시스템 아키텍처](../03_rag_system/rag_architecture.md)
- [데이터 처리 가이드](../02_data/processing/preprocessing_guide.md)

---

*이 문서는 LawFirmAI API v2.0.0을 기준으로 작성되었습니다. 최신 업데이트는 프로젝트 저장소를 확인하세요.*
