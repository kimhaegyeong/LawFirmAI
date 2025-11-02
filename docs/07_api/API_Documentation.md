# LawFirmAI API 문서

## 개요

LawFirmAI는 법률 관련 질문에 답변하는 지능형 AI 어시스턴트입니다. 이 문서는 LawFirmAI의 API와 사용법을 설명합니다.

## 주요 기능

### LangGraph 워크플로우 기반 처리
- **질문 분류**: 자동 질문 유형 분류 및 처리 전략 결정
- **하이브리드 검색**: 의미적 검색과 정확한 매칭 통합
- **답변 생성**: LLM 기반 법률 답변 생성
- **신뢰도 계산**: 답변의 신뢰도 자동 계산

### 세션 관리
- **세션 기반 대화**: 대화 맥락 유지
- **멀티턴 처리**: 대명사 해결 및 질문 확장

### 품질 개선
- **답변 품질 검증**: 자동 품질 평가
- **법적 근거 제공**: 관련 법령 및 판례 인용

## API 엔드포인트

### 기본 정보
- **Base URL**: `http://localhost:8000/api/v1`
- **Content-Type**: `application/json`
- **인증**: 현재 인증 불필요 (향후 추가 예정)

### 1. 채팅 API

#### `POST /api/v1/chat`

채팅 메시지 처리 엔드포인트 (LangGraph 워크플로우)

**요청 파라미터:**
```json
{
  "message": "민법 제750조에 대해 설명해주세요",
  "session_id": "session_id_123"
}
```

**응답:**
```json
{
  "answer": "민법 제750조는 불법행위로 인한 손해배상 책임을 규정하는 중요한 조문입니다...",
  "confidence": 0.85,
  "sources": ["민법 제750조", "대법원 판례 2020다12345"],
  "session_id": "session_id_123",
  "question_type": "law_inquiry",
  "legal_references": ["민법 제750조"],
  "processing_time": 1.2
}
```

### 2. 검색 API

#### `POST /api/v1/search/hybrid`

하이브리드 검색 엔드포인트

**요청 파라미터:**
```json
{
  "query": "손해배상 관련 판례",
  "search_type": "hybrid",
  "limit": 10,
  "filters": {
    "document_type": "precedent",
    "court_name": "대법원"
  }
}
```

**응답:**
```json
{
  "results": [
    {
      "id": "case_001",
      "title": "손해배상 관련 대법원 판례",
      "content": "판례 내용...",
      "source": "대법원 판례 2020다12345",
      "similarity_score": 0.95,
      "search_type": "hybrid"
    }
  ],
  "total_count": 1,
  "search_type": "hybrid",
  "processing_time": 0.5
}
```

#### `POST /api/v1/search/semantic`

의미적 검색 엔드포인트

**요청 파라미터:**
```json
{
  "query": "계약 해지 조건",
  "limit": 5
}
```

**응답:**
```json
{
  "results": [
    {
      "id": "doc_001",
      "title": "계약 해지 관련 법령",
      "similarity_score": 0.92
    }
  ],
  "total_count": 1,
  "search_type": "semantic"
}
```

#### `POST /api/v1/search/exact`

정확한 매칭 검색 엔드포인트

**요청 파라미터:**
```json
{
  "query": "민법 제543조",
  "limit": 5
}
```

**응답:**
```json
{
  "results": [
    {
      "id": "law_001",
      "title": "민법 제543조",
      "content": "조문 내용..."
    }
  ],
  "total_count": 1,
  "search_type": "exact"
}
```

#### `GET /api/v1/search/suggestions`

검색 제안 엔드포인트

**요청 파라미터:**
- `query`: 검색 쿼리 (필수)
- `limit`: 제안 개수 (기본값: 5, 최대: 10)

**응답:**
```json
{
  "suggestions": [
    "손해배상 청구 절차",
    "손해배상 범위",
    "손해배상 계산 방법"
  ],
  "query": "손해배상",
  "total_count": 3
}
```

### 3. 법률 엔티티 API

#### `POST /api/v1/legal-entities`

법률 엔티티 추출 엔드포인트

**요청 파라미터:**
```json
{
  "query": "민법 제750조 불법행위 손해배상"
}
```

**응답:**
```json
{
  "laws": ["민법"],
  "articles": ["제750조"],
  "cases": ["대법원 판례 2020다12345"],
  "supplementary": ["불법행위", "손해배상"]
}
```

### 4. 성능 모니터링 API

#### `GET /api/v1/performance/metrics`

성능 지표 조회 엔드포인트

**응답:**
```json
{
  "metrics": {
    "average_response_time": 1.2,
    "cache_hit_rate": 0.75,
    "memory_usage_mb": 256.5,
    "active_sessions": 15,
    "total_requests": 1250
  },
  "timestamp": "2025-01-01T00:00:00Z"
}
```

#### `GET /api/v1/performance/health`

시스템 상태 확인 엔드포인트

**응답:**
```json
{
  "status": "healthy",
  "services": {
    "database": "healthy",
    "vector_store": "healthy",
    "ml_models": "healthy",
    "cache": "healthy"
  },
  "timestamp": "2025-01-01T00:00:00Z"
}
```

### 5. 피드백 API

#### `POST /api/v1/feedback`

사용자 피드백 수집 엔드포인트

**요청 파라미터:**
```json
{
  "session_id": "session_id_example",
  "user_id": "user123",
  "feedback_type": "rating",
  "rating": 5,
  "comment": "매우 도움이 되었습니다."
}
```

**응답:**
```json
{
  "feedback_id": "feedback_001",
  "status": "received",
  "timestamp": "2025-01-01T00:00:00Z"
}
```

#### `GET /api/v1/feedback/analysis`

피드백 분석 결과 조회 엔드포인트

**응답:**
```json
{
  "analysis": {
    "average_rating": 4.2,
    "total_feedback": 150,
    "positive_feedback": 120,
    "negative_feedback": 30,
    "common_issues": ["응답 속도", "정확성"],
    "improvement_suggestions": ["캐시 최적화", "모델 업데이트"]
  },
  "timestamp": "2025-01-01T00:00:00Z"
}
```

```

## 에러 코드

### HTTP 상태 코드

| 코드 | 설명 | 해결 방법 |
|------|------|----------|
| 200 | 성공 | - |
| 400 | 잘못된 요청 | 요청 파라미터 확인 |
| 401 | 인증 실패 | API 키 확인 (향후 적용) |
| 404 | 리소스 없음 | 엔드포인트 URL 확인 |
| 500 | 서버 오류 | 서버 로그 확인 |

### 에러 응답 형식

```json
{
  "error": {
    "code": "INVALID_REQUEST",
    "message": "요청 파라미터가 올바르지 않습니다.",
    "details": {
      "field": "message",
      "issue": "필수 필드가 누락되었습니다."
    }
  },
  "timestamp": "2025-01-01T00:00:00Z"
}
```

## 사용 예제

### Python 예제

```python
import requests
import json

# 기본 채팅 API 사용
def chat_with_lawfirm_ai(message, session_id=None):
    url = "http://localhost:8000/api/v1/chat"
    data = {
        "message": message,
        "session_id": session_id
    }
    
    response = requests.post(url, json=data)
    return response.json()

# 사용 예제
result = chat_with_lawfirm_ai("민법 제750조에 대해 설명해주세요", "session_id_123")
print(result["answer"])
```

### JavaScript 예제

```javascript
// 기본 채팅 API 사용
async function chatWithLawFirmAI(message, sessionId = null) {
    const response = await fetch('http://localhost:8000/api/v1/chat', {
        method: 'POST',
        headers: {
            'Content-Type': 'application/json',
        },
        body: JSON.stringify({
            message: message,
            session_id: sessionId
        })
    });
    
    return await response.json();
}

// 사용 예제
chatWithLawFirmAI("민법 제750조에 대해 설명해주세요", "session_id_123")
    .then(result => console.log(result.answer));
```

## 성능 최적화 팁

### 1. 세션 관리
- `session_id`를 사용하여 대화 맥락을 유지하세요
- 세션을 재사용하면 더 나은 응답을 받을 수 있습니다

### 2. 검색 옵션 활용
- 하이브리드 검색을 통해 의미적 검색과 정확한 매칭을 통합하세요
- 질문 유형에 따라 적절한 검색 방식을 선택하세요

### 3. 에러 처리
- 적절한 에러 처리를 구현하여 사용자 경험을 향상시키세요
- 재시도 로직을 구현하여 일시적인 오류를 처리하세요

---


#### `POST /api/user/profile/{user_id}`

사용자 프로필을 생성하거나 업데이트합니다.

**요청 파라미터:**
```json
{
  "expertise_level": "advanced",
  "preferred_detail_level": "detailed",
  "interest_areas": ["민법", "형법", "상법"],
  "preferences": {
    "response_style": "formal",
    "include_precedents": true
  }
}
```

### 3. 세션 관리

#### `GET /api/sessions/{user_id}`

사용자의 세션 목록을 조회합니다.

**응답:**
```json
{
  "sessions": [
    {
      "session_id": "session_id_example",
      "created_at": "2025-01-01T00:00:00Z",
      "last_updated": "2025-01-01T00:00:00Z",
      "turn_count": 5,
      "topics": ["민법", "손해배상"],
      "summary": "민법 제750조 관련 질문들"
    }
  ]
}
```

#### `GET /api/sessions/{session_id}/details`

특정 세션의 상세 정보를 조회합니다.

**응답:**
```json
{
  "session_id": "session_id_example",
  "user_id": "user123",
  "turns": [
    {
      "user_query": "민법 제750조에 대해 설명해주세요",
      "bot_response": "민법 제750조는 불법행위로 인한 손해배상 책임을 규정합니다...",
      "timestamp": "2025-01-01T00:00:00Z",
      "question_type": "law_inquiry",
      "entities": {"laws": ["민법"], "articles": ["제750조"]}
    }
  ],
  "entities": {...},
  "topic_stack": ["민법", "손해배상"],
  "created_at": "2025-01-01T00:00:00Z",
  "last_updated": "2025-01-01T00:00:00Z"
}
```

### 4. 메모리 관리

#### `GET /api/memory/search`

관련 메모리를 검색합니다.

**요청 파라미터:**
```json
{
  "query": "손해배상 관련 질문",
  "session_id": "session_id_example",
  "user_id": "user123"
}
```

**응답:**
```json
{
  "results": [
    {
      "memory_id": "mem_user123_session_id_example_abc123",
      "content": "민법 제750조는 불법행위로 인한 손해배상 책임을 규정합니다",
      "memory_type": "legal_knowledge",
      "importance_score": 0.85,
      "relevance_score": 0.92,
      "match_reason": "공통 키워드: 손해배상, 민법"
    }
  ]
}
```

#### `POST /api/memory/consolidate`

사용자의 메모리를 통합합니다.

**응답:**
```json
{
  "status": "success",
  "consolidated_count": 3,
  "message": "3개의 중복 메모리가 통합되었습니다"
}
```

### 5. 품질 모니터링

#### `GET /api/quality/assessment/{session_id}`

세션의 품질 평가를 조회합니다.

**응답:**
```json
{
  "overall_score": 0.82,
  "completeness_score": 0.85,
  "satisfaction_score": 0.80,
  "accuracy_score": 0.81,
  "issues": [],
  "suggestions": [
    "답변에 더 구체적인 예시를 포함하세요",
    "관련 법령이나 판례를 추가로 언급하세요"
  ],
  "assessment_timestamp": "2025-01-01T00:00:00Z"
}
```

#### `GET /api/quality/trends/{user_id}`

사용자의 품질 트렌드를 분석합니다.

**응답:**
```json
{
  "periods": [
    {
      "period": "2025-01-01",
      "avg_completeness": 0.85,
      "avg_satisfaction": 0.80,
      "avg_accuracy": 0.81,
      "total_sessions": 5,
      "improvement_suggestions": [...]
    }
  ],
  "overall_trend": "improving",
  "key_insights": ["품질이 지속적으로 향상되고 있습니다"],
  "recommendations": ["현재 접근 방식을 유지하세요"]
}
```

### 6. 성능 모니터링

#### `GET /api/performance/metrics`

성능 메트릭을 조회합니다.

**응답:**
```json
{
  "timestamp": "2025-01-01T00:00:00Z",
  "performance_monitor": {
    "summary": {
      "period_hours": 24,
      "total_requests": 150,
      "averages": {
        "cpu_usage": 45.2,
        "memory_usage": 62.8,
        "response_time": 1.2,
        "cache_hit_rate": 0.75
      },
      "status": "healthy"
    },
    "system_health": {
      "status": "healthy",
      "system": {
        "cpu_usage": 45.2,
        "memory_usage": 62.8,
        "memory_available_gb": 8.5
      }
    }
  },
  "memory_optimizer": {
    "memory_usage": {
      "total_mb": 16384,
      "used_mb": 10240,
      "available_mb": 6144,
      "percentage": 62.5,
      "process_mb": 256.5
    },
    "memory_trend": {
      "trend": "stable",
      "trend_rate_percent": 2.1,
      "memory_leak_warning": false
    }
  },
  "cache_manager": {
    "cache_size": 150,
    "max_size": 1000,
    "hits": 1200,
    "misses": 400,
    "hit_rate": 0.75
  }
}
```

#### `POST /api/performance/optimize`

성능 최적화를 수행합니다.

**응답:**
```json
{
  "timestamp": "2025-01-01T00:00:00Z",
  "actions_taken": [
    "Garbage collection: 45 objects collected",
    "Cache cleanup: 12 entries removed"
  ],
  "memory_optimization": {
    "memory_freed_mb": 15.2,
    "actions_taken": ["Garbage collection: 45 objects collected"]
  },
  "cache_optimization": {
    "cache_cleared": 12,
    "stats_before": {...}
  }
}
```

## 에러 코드

| 코드 | 설명 | 해결 방법 |
|------|------|-----------|
| 400 | 잘못된 요청 | 요청 파라미터를 확인하세요 |
| 401 | 인증 실패 | 유효한 API 키를 사용하세요 |
| 404 | 리소스를 찾을 수 없음 | 요청한 리소스가 존재하는지 확인하세요 |
| 429 | 요청 한도 초과 | 요청 빈도를 줄이세요 |
| 500 | 서버 내부 오류 | 잠시 후 다시 시도하세요 |

## 사용 예제

### Python 예제

```python
import requests
import json

# 기본 설정
BASE_URL = "https://api.lawfirmai.com"
API_KEY = "your_api_key_here"
headers = {
    "Authorization": f"Bearer {API_KEY}",
    "Content-Type": "application/json"
}

# 메시지 전송
def send_message(message, session_id=None):
    url = f"{BASE_URL}/api/v1/chat"
    data = {
        "message": message,
        "session_id": session_id
    }
    
    response = requests.post(url, headers=headers, json=data)
    return response.json()

# 사용 예제
if __name__ == "__main__":
    # 메시지 전송
    result = send_message(
        "민법 제750조에 대해 설명해주세요",
        session_id="session_id_example"
    )
    print(f"답변: {result['answer']}")
    print(f"신뢰도: {result.get('confidence', 'N/A')}")
```

### JavaScript 예제

```javascript
// 기본 설정
const BASE_URL = 'https://api.lawfirmai.com';
const API_KEY = 'your_api_key_here';

const headers = {
    'Authorization': `Bearer ${API_KEY}`,
    'Content-Type': 'application/json'
};

// 메시지 전송
async function sendMessage(message, sessionId = null) {
    const url = `${BASE_URL}/api/v1/chat`;
    const data = {
        message: message,
        session_id: sessionId
    };
    
    const response = await fetch(url, {
        method: 'POST',
        headers: headers,
        body: JSON.stringify(data)
    });
    
    return await response.json();
}

// 사용 예제
async function example() {
    try {
        // 메시지 전송
        const result = await sendMessage(
            '민법 제750조에 대해 설명해주세요',
            'session_id_example'
        );
        console.log('답변:', result.answer);
        console.log('신뢰도:', result.confidence);
        
    } catch (error) {
        console.error('오류 발생:', error);
    }
}

example();
```

## 제한사항

- **요청 한도**: 분당 60회, 시간당 1000회
- **메시지 길이**: 최대 10,000자
- **세션 수**: 사용자당 최대 100개 활성 세션
- **메모리 보관**: 30일 후 자동 삭제
- **캐시 크기**: 최대 1000개 항목
