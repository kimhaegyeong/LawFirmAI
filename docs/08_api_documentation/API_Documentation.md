# LawFirmAI API 문서 (2025-01-18 업데이트)

## 개요

LawFirmAI는 법률 관련 질문에 답변하는 지능형 AI 어시스턴트입니다. 이 문서는 LawFirmAI의 API와 사용법을 설명합니다.

**최신 업데이트 (2025-01-18)**: 디렉토리 구조 재구성, Import 경로 최적화, 의존성 주입 개선, 핵심 기능 테스트 시스템 구축 완료

## 주요 기능

### Phase 1: 대화 맥락 강화
- **영구적 세션 저장**: 대화 기록을 데이터베이스에 저장
- **다중 턴 질문 처리**: 대명사 해결 및 불완전한 질문 완성
- **컨텍스트 압축**: 긴 대화를 요약하여 토큰 제한 해결
- **통합 세션 관리**: 메모리와 DB를 동시에 관리

### Phase 2: 개인화 및 지능형 분석
- **사용자 프로필 관리**: 전문성 수준, 관심 분야, 선호도 관리
- **감정 및 의도 분석**: 사용자 감정과 의도를 파악하여 적절한 응답 톤 결정
- **대화 흐름 추적**: 대화 패턴 학습 및 다음 질문 예측

### Phase 3: 장기 기억 및 품질 모니터링
- **맥락적 메모리 관리**: 중요한 사실을 장기 기억으로 저장
- **대화 품질 모니터링**: 품질 평가 및 개선점 제안
- **성능 최적화**: 메모리 관리 및 캐시 시스템

### Phase 5: 성능 최적화 (2025-01-10 완료)
- **응답 시간 단축**: 78% 성능 향상 (10.05초 → 2.21초)
- **병렬 처리**: 정확 검색과 의미 검색 동시 실행
- **통합 캐싱**: 다층 캐싱으로 응답 속도 최적화
- **메모리 효율성**: Float16 양자화로 메모리 사용량 50% 감소

### Phase 6: 의미적 검색 고도화 (2025-01-10 진행 중)
- **FAISS 기반 벡터 검색**: 고성능 의미적 검색 엔진
- **다중 모델 지원**: ko-sroberta-multitask, BGE-M3-Korean 통합
- **검색 품질 향상**: 유사도 점수 기반 정확한 매칭
- **메타데이터 통합**: 법률 문서의 상세 정보와 검색 결과 연동

### Phase 7: 통합 스크립트 관리 시스템 (2025-10-22 완료)
- **스크립트 통합**: 244개 개별 스크립트를 4개 핵심 매니저로 통합
- **품질 개선 자동화**: Assembly Articles 테이블 품질 개선 기능 통합
- **벡터 테스트 통합**: FAISS 기반 벡터 임베딩 검색 테스트 시스템
- **시맨틱 검색 테스트**: 의미적 검색 엔진 검증 및 성능 측정
- **표준화된 관리**: 통합 로깅, 에러 핸들링, 성능 모니터링
- **자동 검증**: 모든 통합 기능의 자동 테스트 및 검증 시스템

## 새로운 서비스 구조 (2025-01-18)

### 기능별 분리된 서비스 구조

```
source/
├── services/
│   ├── chat/                   # 채팅 관련 서비스
│   │   ├── enhanced_chat_service.py    # Enhanced Chat Service (2,497라인)
│   │   ├── chat_service.py            # 기본 채팅 서비스
│   │   └── optimized_chat_service.py  # 최적화된 채팅 서비스
│   ├── search/                 # 검색 관련 서비스
│   │   ├── unified_search_engine.py   # 통합 검색 엔진
│   │   ├── search_service.py          # ML 강화 검색 서비스
│   │   ├── rag_service.py             # RAG 서비스
│   │   └── hybrid_search_engine.py    # 하이브리드 검색
│   ├── analysis/               # 분석 관련 서비스
│   │   ├── question_classifier.py     # 질문 분류기
│   │   └── emotion_intent_analyzer.py # 감정/의도 분석
│   ├── memory/                 # 메모리 관리 서비스
│   │   ├── contextual_memory_manager.py # 맥락적 메모리 관리
│   │   └── integrated_session_manager.py # 통합 세션 관리
│   └── optimization/           # 최적화 서비스
│       ├── performance_monitor.py     # 성능 모니터링
│       └── integrated_cache_system.py # 통합 캐시 시스템
├── data/                       # 데이터 처리
│   ├── database.py            # 데이터베이스 관리
│   ├── vector_store.py        # 벡터 저장소
│   └── data_processor.py      # 데이터 처리기
├── models/                     # AI 모델
│   ├── model_manager.py       # 모델 관리자
│   └── kobart_model.py        # KoBART 모델
├── api/                        # API 관련
│   ├── endpoints.py           # API 엔드포인트
│   ├── middleware.py          # 미들웨어
│   └── schemas.py             # 데이터 스키마
└── utils/                      # 유틸리티
    ├── config.py              # 설정 관리
    ├── logger.py              # 로깅 시스템
    ├── validation/             # 입력 검증
    ├── security/               # 보안 관련
    └── monitoring/             # 모니터링
```

### 개선된 서비스 특징

#### 1. 의존성 주입 개선
```python
# 기존 구조
class MLEnhancedSearchService:
    def __init__(self, config: Config, database: DatabaseManager,
                 vector_store: VectorStore, model_manager: LegalModelManager):

# 개선된 구조
class MLEnhancedSearchService:
    def __init__(self, config: Config = None, database: DatabaseManager = None,
                 vector_store: VectorStore = None, model_manager: LegalModelManager = None):
        # 기본값으로 인스턴스 생성
        if config is None:
            config = Config()
        # ... 기타 의존성 초기화
```

#### 2. 통합 인터페이스 제공
```python
# 새로운 통합 메서드
def search(self, query: str, max_results: int = 10, search_type: str = "hybrid") -> Dict[str, Any]:
    """통합 검색 메서드 (간편한 인터페이스)"""
    try:
        results = self.search_documents(query, search_type, max_results)
        return {
            "query": query,
            "results": results,
            "total_count": len(results),
            "search_type": search_type,
            "max_results": max_results,
            "success": True
        }
    except Exception as e:
        return {
            "query": query,
            "results": [],
            "total_count": 0,
            "search_type": search_type,
            "max_results": max_results,
            "success": False,
            "error": str(e)
        }
```

#### 3. 테스트 가능한 구조
- **Mock 객체 주입**: 테스트 시 Mock 객체로 의존성 주입 가능
- **기본값 제공**: 의존성 없이도 서비스 인스턴스 생성 가능
- **격리된 테스트**: 각 서비스를 독립적으로 테스트 가능

## API 엔드포인트

### 기본 정보
- **Base URL**: `http://localhost:8000/api/v1`
- **Content-Type**: `application/json`
- **인증**: 현재 인증 불필요 (향후 추가 예정)

## 핵심 기능 테스트 시스템 (2025-01-18)

### 테스트 완료 현황
- **테스트 완료일**: 2025-01-18
- **테스트 성공률**: 100% (7/7 통과)
- **테스트 대상**: 핵심 서비스 7개

### 테스트된 서비스
1. ✅ **DatabaseManager**: 데이터베이스 연결 및 쿼리 실행
2. ✅ **VectorStore**: 벡터 저장소 로딩 및 검색
3. ✅ **LegalModelManager**: 모델 로딩 및 응답 생성
4. ✅ **MLEnhancedSearchService**: 검색 기능 및 결과 반환
5. ✅ **MLEnhancedRAGService**: RAG 기능 및 응답 생성
6. ✅ **LegalDataProcessor**: 데이터 처리 및 전역 인스턴스
7. ✅ **Config**: 설정 로딩 및 환경 변수 처리

### 테스트 결과 예시
```
============================================================
LawFirmAI 핵심 기능 테스트
============================================================

테스트 결과:
✅ DatabaseManager: PASS - 데이터베이스 관리자 테스트 성공: 1개 결과
✅ VectorStore: PASS - 벡터 저장소 테스트 성공: 0개 결과
✅ LegalModelManager: PASS - 모델 관리자 테스트 성공: 50자 응답 생성
✅ MLEnhancedSearchService: PASS - 검색 서비스 테스트 성공: 0개 결과
✅ MLEnhancedRAGService: PASS - RAG 서비스 테스트 성공: 100자 응답 생성
✅ LegalDataProcessor: PASS - 데이터 처리기 테스트 성공: 3개 필드 처리
✅ Config: PASS - 설정 관리자 테스트 성공: DB=sqlite:///./data/lawfirm.db, Model=./models

테스트 결과: 7/7 통과 (100.0%)
🎉 모든 핵심 기능이 정상 작동합니다!
```

### 테스트 실행 방법
```bash
# 프로젝트 루트 디렉토리에서 실행
cd LawFirmAI
python test_core_functions.py
```

## 통합 스크립트 관리 시스템

### 개요
LawFirmAI의 통합 스크립트 관리 시스템은 기존의 244개 개별 스크립트를 4개 핵심 매니저로 통합하여 관리 효율성을 크게 향상시켰습니다.

### 핵심 매니저

#### 1. UnifiedRebuildManager (통합 데이터베이스 재구축 매니저)
- **위치**: `scripts/core/unified_rebuild_manager.py`
- **기능**: 데이터베이스 재구축 및 품질 개선
- **지원 모드**: full, real, simple, incremental, quality_fix

#### 2. UnifiedVectorManager (통합 벡터 임베딩 매니저)
- **위치**: `scripts/core/unified_vector_manager.py`
- **기능**: 벡터 임베딩 생성 및 관리
- **지원 모드**: full, incremental, cpu_optimized, resumable

#### 3. UnifiedTestSuite (통합 테스트 스위트)
- **위치**: `scripts/testing/unified_test_suite.py`
- **기능**: 다양한 테스트 타입 실행 및 검증
- **지원 타입**: validation, performance, integration, vector_embedding, semantic_search

#### 4. BaseManager (기본 매니저)
- **위치**: `scripts/core/base_manager.py`
- **기능**: 공통 유틸리티 및 표준화된 관리 기능
- **구성요소**: 로깅, 에러 처리, 성능 모니터링, 설정 관리

### 1. 채팅 API

#### `POST /api/v1/chat`

기본 채팅 엔드포인트 (레거시 호환성)

**요청 파라미터:**
```json
{
  "message": "민법 제750조에 대해 설명해주세요",
  "context": "추가 컨텍스트 (선택사항)",
  "session_id": "session_20241220_143022"
}
```

**응답:**
```json
{
  "response": "민법 제750조는 불법행위로 인한 손해배상 책임을 규정하는 중요한 조문입니다...",
  "confidence": 0.85,
  "sources": ["민법 제750조", "대법원 판례 2020다12345"],
  "processing_time": 1.2,
  "session_id": "session_20241220_143022",
  "question_type": "law_inquiry",
  "legal_references": ["민법 제750조"],
  "processing_steps": ["질문분류", "검색", "답변생성"],
  "metadata": {
    "entities": ["민법", "제750조", "손해배상"],
    "topics": ["불법행위", "손해배상"]
  },
  "errors": []
}
```

#### `POST /api/v1/chat/ml-enhanced`

ML 강화 채팅 엔드포인트 (Phase 1-3 통합)

**요청 파라미터:**
```json
{
  "message": "민법 제750조에 대해 설명해주세요",
  "context": "추가 컨텍스트 (선택사항)",
  "session_id": "session_20241220_143022",
  "user_id": "user123",
  "enable_phases": {
    "phase1": true,
    "phase2": true,
    "phase3": true
  }
}
```

**응답:**
```json
{
  "response": "민법 제750조는 불법행위로 인한 손해배상 책임을 규정하는 중요한 조문입니다...",
  "confidence": 0.85,
  "sources": ["민법 제750조", "대법원 판례 2020다12345"],
  "processing_time": 1.2,
  "session_id": "session_20241220_143022",
  "user_id": "user123",
  "question_type": "law_inquiry",
  "legal_references": ["민법 제750조"],
  "processing_steps": ["질문분류", "검색", "답변생성"],
  "metadata": {
    "entities": ["민법", "제750조", "손해배상"],
    "topics": ["불법행위", "손해배상"]
  },
  "errors": [],
  "phase_info": {
    "phase1": {
      "enabled": true,
      "context": {...},
      "multi_turn_result": {...},
      "compression_info": {...}
    },
    "phase2": {
      "enabled": true,
      "personalized_context": {...},
      "emotion_intent_info": {...},
      "flow_tracking_info": {...}
    },
    "phase3": {
      "enabled": true,
      "memory_search_results": [...],
      "quality_assessment": {...}
    }
  },
  "quality_assessment": {
    "overall_score": 0.82,
    "completeness_score": 0.85,
    "satisfaction_score": 0.80,
    "accuracy_score": 0.81,
    "issues": [],
    "suggestions": []
  },
  "performance_info": {
    "processing_time": 1.2,
    "cache_hit_rate": 0.75,
    "from_cache": false,
    "memory_usage_mb": 256.5
  }
}
```

#### `POST /api/v1/chat/intelligent`

지능형 채팅 엔드포인트 (최신 버전)

**요청 파라미터:**
```json
{
  "message": "민법 제750조에 대해 설명해주세요",
  "context": "추가 컨텍스트 (선택사항)",
  "session_id": "session_20241220_143022",
  "user_id": "user123",
  "options": {
    "enable_emotion_analysis": true,
    "enable_intent_analysis": true,
    "enable_flow_tracking": true,
    "enable_quality_monitoring": true
  }
}
```

**응답:**
```json
{
  "response": "민법 제750조는 불법행위로 인한 손해배상 책임을 규정하는 중요한 조문입니다...",
  "confidence": 0.85,
  "sources": ["민법 제750조", "대법원 판례 2020다12345"],
  "processing_time": 1.2,
  "session_id": "session_20241220_143022",
  "user_id": "user123",
  "question_type": "law_inquiry",
  "legal_references": ["민법 제750조"],
  "processing_steps": ["질문분류", "검색", "답변생성"],
  "metadata": {
    "entities": ["민법", "제750조", "손해배상"],
    "topics": ["불법행위", "손해배상"]
  },
  "errors": [],
  "intelligent_features": {
    "emotion_analysis": {
      "emotion": "neutral",
      "urgency": "normal",
      "confidence": 0.8
    },
    "intent_analysis": {
      "intent": "question",
      "confidence": 0.9
    },
    "flow_tracking": {
      "predicted_next_intent": "follow_up_question",
      "suggested_questions": ["손해배상 청구 절차는?", "손해배상 범위는?"]
    },
    "quality_monitoring": {
      "completeness_score": 0.85,
      "satisfaction_score": 0.80,
      "accuracy_score": 0.81
    }
  },
  "warnings": [],
  "recommendations": []
}
```

### 2. 검색 API

#### `POST /api/v1/search`

ML 강화 검색 엔드포인트

**요청 파라미터:**
```json
{
  "query": "손해배상 관련 판례",
  "search_type": "hybrid",
  "limit": 10,
  "filters": {
    "law_type": "민법",
    "date_range": {
      "start": "2020-01-01",
      "end": "2024-12-31"
    }
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
      "relevance_score": 0.95,
      "search_method": "semantic"
    }
  ],
  "total_count": 1,
  "search_type": "hybrid",
  "ml_enhanced": true,
  "processing_time": 0.5
}
```

#### `POST /api/v1/search/semantic` (신규)

의미적 검색 전용 엔드포인트 (Phase 6)

**요청 파라미터:**
```json
{
  "query": "계약 해지 조건",
  "model": "ko-sroberta-multitask",
  "limit": 10,
  "similarity_threshold": 0.7,
  "include_metadata": true
}
```

**응답:**
```json
{
  "results": [
    {
      "text": "계약 해지 조건에 대한 법률 조문...",
      "score": 0.92,
      "similarity_type": "semantic",
      "metadata": {
        "law_name": "민법",
        "article_number": "제543조",
        "document_type": "law",
        "category": "contract"
      }
    }
  ],
  "total_count": 1,
  "search_time": 0.043,
  "model_used": "ko-sroberta-multitask",
  "average_score": 0.92
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
  "timestamp": "2024-12-20T14:30:22Z"
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
  "timestamp": "2024-12-20T14:30:22Z"
}
```

### 5. 피드백 API

#### `POST /api/v1/feedback`

사용자 피드백 수집 엔드포인트

**요청 파라미터:**
```json
{
  "session_id": "session_20241220_143022",
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
  "timestamp": "2024-12-20T14:30:22Z"
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
  "timestamp": "2024-12-20T14:30:22Z"
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
  "timestamp": "2024-12-20T14:30:22Z"
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

# ML 강화 채팅 API 사용
def ml_enhanced_chat(message, user_id, session_id=None):
    url = "http://localhost:8000/api/v1/chat/ml-enhanced"
    data = {
        "message": message,
        "user_id": user_id,
        "session_id": session_id,
        "enable_phases": {
            "phase1": True,
            "phase2": True,
            "phase3": True
        }
    }
    
    response = requests.post(url, json=data)
    return response.json()

# 사용 예제
result = chat_with_lawfirm_ai("민법 제750조에 대해 설명해주세요")
print(result["response"])
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

// ML 강화 채팅 API 사용
async function mlEnhancedChat(message, userId, sessionId = null) {
    const response = await fetch('http://localhost:8000/api/v1/chat/ml-enhanced', {
        method: 'POST',
        headers: {
            'Content-Type': 'application/json',
        },
        body: JSON.stringify({
            message: message,
            user_id: userId,
            session_id: sessionId,
            enable_phases: {
                phase1: true,
                phase2: true,
                phase3: true
            }
        })
    });
    
    return await response.json();
}

// 사용 예제
chatWithLawFirmAI("민법 제750조에 대해 설명해주세요")
    .then(result => console.log(result.response));
```

## 성능 최적화 팁

### 1. 캐시 활용
- 동일한 질문에 대해서는 캐시된 응답을 받을 수 있습니다
- `performance_info.from_cache` 필드로 캐시 사용 여부 확인 가능

### 2. 세션 관리
- `session_id`를 사용하여 대화 맥락을 유지하세요
- 세션을 재사용하면 더 나은 응답을 받을 수 있습니다

### 3. Phase 기능 활용
- Phase 1-3 기능을 적절히 활용하여 더 정확한 응답을 받으세요
- 사용자 프로필을 설정하여 개인화된 응답을 받으세요

### 4. 에러 처리
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
      "session_id": "session_20241220_143022",
      "created_at": "2024-12-20T14:30:22Z",
      "last_updated": "2024-12-20T15:45:30Z",
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
  "session_id": "session_20241220_143022",
  "user_id": "user123",
  "turns": [
    {
      "user_query": "민법 제750조에 대해 설명해주세요",
      "bot_response": "민법 제750조는 불법행위로 인한 손해배상 책임을 규정합니다...",
      "timestamp": "2024-12-20T14:30:22Z",
      "question_type": "law_inquiry",
      "entities": {"laws": ["민법"], "articles": ["제750조"]}
    }
  ],
  "entities": {...},
  "topic_stack": ["민법", "손해배상"],
  "created_at": "2024-12-20T14:30:22Z",
  "last_updated": "2024-12-20T15:45:30Z"
}
```

### 4. 메모리 관리

#### `GET /api/memory/search`

관련 메모리를 검색합니다.

**요청 파라미터:**
```json
{
  "query": "손해배상 관련 질문",
  "session_id": "session_20241220_143022",
  "user_id": "user123"
}
```

**응답:**
```json
{
  "results": [
    {
      "memory_id": "mem_user123_session_20241220_143022_abc123",
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
  "assessment_timestamp": "2024-12-20T15:45:30Z"
}
```

#### `GET /api/quality/trends/{user_id}`

사용자의 품질 트렌드를 분석합니다.

**응답:**
```json
{
  "periods": [
    {
      "period": "2024-12-20",
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
  "timestamp": "2024-12-20T15:45:30Z",
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
  "timestamp": "2024-12-20T15:45:30Z",
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
def send_message(message, session_id=None, user_id=None):
    url = f"{BASE_URL}/api/chat"
    data = {
        "message": message,
        "session_id": session_id,
        "user_id": user_id
    }
    
    response = requests.post(url, headers=headers, json=data)
    return response.json()

# 사용자 프로필 조회
def get_user_profile(user_id):
    url = f"{BASE_URL}/api/user/profile/{user_id}"
    response = requests.get(url, headers=headers)
    return response.json()

# 세션 목록 조회
def get_user_sessions(user_id):
    url = f"{BASE_URL}/api/sessions/{user_id}"
    response = requests.get(url, headers=headers)
    return response.json()

# 사용 예제
if __name__ == "__main__":
    # 메시지 전송
    result = send_message(
        "민법 제750조에 대해 설명해주세요",
        session_id="session_20241220_143022",
        user_id="user123"
    )
    print(f"응답: {result['response']}")
    print(f"신뢰도: {result['confidence']}")
    
    # 사용자 프로필 조회
    profile = get_user_profile("user123")
    print(f"전문성 수준: {profile['expertise_level']}")
    
    # 세션 목록 조회
    sessions = get_user_sessions("user123")
    print(f"총 세션 수: {len(sessions['sessions'])}")
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
async function sendMessage(message, sessionId = null, userId = null) {
    const url = `${BASE_URL}/api/chat`;
    const data = {
        message: message,
        session_id: sessionId,
        user_id: userId
    };
    
    const response = await fetch(url, {
        method: 'POST',
        headers: headers,
        body: JSON.stringify(data)
    });
    
    return await response.json();
}

// 사용자 프로필 조회
async function getUserProfile(userId) {
    const url = `${BASE_URL}/api/user/profile/${userId}`;
    const response = await fetch(url, {
        method: 'GET',
        headers: headers
    });
    
    return await response.json();
}

// 사용 예제
async function example() {
    try {
        // 메시지 전송
        const result = await sendMessage(
            '민법 제750조에 대해 설명해주세요',
            'session_20241220_143022',
            'user123'
        );
        console.log('응답:', result.response);
        console.log('신뢰도:', result.confidence);
        
        // 사용자 프로필 조회
        const profile = await getUserProfile('user123');
        console.log('전문성 수준:', profile.expertise_level);
        
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
