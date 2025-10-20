# LawFirmAI API 문서 (2025-10-16)

## 📋 개요

LawFirmAI는 법률 AI 어시스턴트를 위한 RESTful API입니다. ML 강화 RAG 시스템과 하이브리드 검색을 통해 법률 문서 검색, 분석, 질의응답 기능을 제공합니다.

## 🚀 기본 정보

- **Base URL**: `http://localhost:8000`
- **API Version**: `v1`
- **Content-Type**: `application/json`
- **인코딩**: `UTF-8`

## 🔧 인증

현재 버전에서는 API 키 인증을 사용합니다.

```http
Authorization: Bearer YOUR_API_KEY
```

## 📚 API 엔드포인트

### 1. 채팅 API

#### 기본 채팅
```http
POST /api/v1/chat
```

**요청 본문:**
```json
{
  "message": "계약서 검토 요청",
  "context": "추가 컨텍스트 (선택사항)",
  "session_id": "세션 ID (선택사항)"
}
```

**응답:**
```json
{
  "response": "계약서 검토 결과...",
  "confidence": 0.85,
  "sources": [
    {
      "title": "민법 제543조",
      "content": "계약 내용...",
      "similarity": 0.92
    }
  ],
  "processing_time": 1.23,
  "model": "KoBART",
  "retrieved_docs_count": 5
}
```

#### ML 강화 채팅
```http
POST /api/v1/chat/ml-enhanced
```

**요청 본문:**
```json
{
  "message": "계약서 검토 요청",
  "context": "추가 컨텍스트",
  "session_id": "세션 ID",
  "use_ml_enhanced": true,
  "quality_threshold": 0.7
}
```

**응답:**
```json
{
  "response": "ML 강화된 계약서 검토 결과...",
  "confidence": 0.92,
  "sources": [
    {
      "title": "민법 제543조",
      "content": "계약 내용...",
      "similarity": 0.95,
      "ml_quality_score": 0.88
    }
  ],
  "processing_time": 1.45,
  "model": "KoBART + ML Enhanced",
  "retrieved_docs_count": 8,
  "ml_enhanced": true,
  "quality_stats": {
    "high_quality_docs": 6,
    "medium_quality_docs": 2,
    "low_quality_docs": 0
  }
}
```

### 2. 검색 API

#### 기본 검색
```http
POST /api/v1/search
```

**요청 본문:**
```json
{
  "query": "계약 해지 조건",
  "search_type": "hybrid",
  "limit": 10,
  "filters": {
    "document_type": "law",
    "date_from": "2020-01-01"
  }
}
```

**응답:**
```json
{
  "results": [
    {
      "document_id": 123,
      "title": "민법 제543조",
      "content": "계약 해지 조건...",
      "similarity": 0.89,
      "search_type": "hybrid",
      "matched_keywords": ["계약", "해지", "조건"],
      "chunk_index": 0
    }
  ],
  "total_count": 15,
  "query": "계약 해지 조건",
  "search_type": "hybrid",
  "processing_time": 0.15,
  "ml_enhanced": true
}
```

#### 하이브리드 검색
```http
POST /api/search/
```

**요청 본문:**
```json
{
  "query": "계약 해지 조건",
  "search_types": ["semantic", "keyword"],
  "max_results": 20,
  "include_exact": true,
  "include_semantic": true
}
```

**응답:**
```json
{
  "query": "계약 해지 조건",
  "results": [
    {
      "document_id": 123,
      "title": "민법 제543조",
      "content": "계약 해지 조건...",
      "similarity": 0.89,
      "search_type": "hybrid",
      "matched_keywords": ["계약", "해지", "조건"],
      "chunk_index": 0
    }
  ],
  "total_results": 15,
  "search_stats": {
    "exact_matches": 5,
    "semantic_matches": 10,
    "processing_time": 0.15
  },
  "success": true
}
```

#### 법률 전용 검색
```http
GET /api/search/laws?query=계약&max_results=20
```

**응답:**
```json
{
  "laws": [
    {
      "law_id": "민법",
      "article": "제543조",
      "title": "계약의 해지",
      "content": "계약 해지 조건...",
      "similarity": 0.89
    }
  ],
  "total_count": 8,
  "query": "계약",
  "processing_time": 0.12
}
```

### 3. 법률 엔티티 추출 API

```http
POST /api/v1/legal-entities
```

**요청 본문:**
```json
{
  "query": "민법 제543조에 따른 계약 해지"
}
```

**응답:**
```json
{
  "laws": [
    {
      "name": "민법",
      "article": "제543조",
      "context": "계약 해지"
    }
  ],
  "articles": [
    {
      "law": "민법",
      "number": "543",
      "title": "계약의 해지"
    }
  ],
  "cases": [
    {
      "case_number": "2020다12345",
      "court": "대법원",
      "context": "계약 해지 관련"
    }
  ],
  "supplementary": [
    {
      "type": "부칙",
      "content": "시행령 관련 내용"
    }
  ]
}
```

### 4. 검색 제안 API

```http
GET /api/v1/search/suggestions?query=계약&limit=5
```

**응답:**
```json
{
  "suggestions": [
    "계약 해지",
    "계약 위반",
    "계약 조건",
    "계약 갱신",
    "계약 해제"
  ],
  "query": "계약",
  "total_suggestions": 5
}
```

### 5. 품질 통계 API

```http
GET /api/v1/quality/stats
```

**응답:**
```json
{
  "total_documents": 7680,
  "quality_distribution": {
    "high_quality": 5120,
    "medium_quality": 2048,
    "low_quality": 512
  },
  "average_quality_score": 0.85,
  "ml_enhanced_documents": 7680,
  "last_updated": "2025-10-16T10:30:00Z"
}
```

### 6. 헬스체크 API

```http
GET /api/v1/health
```

**응답:**
```json
{
  "status": "healthy",
  "service": "LawFirmAI",
  "version": "1.0.0",
  "timestamp": "2025-10-16T10:30:00Z",
  "models": {
    "kobart": "loaded",
    "ko_sroberta": "loaded",
    "bge_m3": "loaded"
  },
  "database_status": {
    "sqlite": "connected",
    "faiss": "loaded",
    "total_documents": 7680
  }
}
```

## 📊 데이터 모델

### 요청 모델

#### ChatRequest
```json
{
  "message": "string (1-10000자)",
  "context": "string (선택사항, 최대 5000자)",
  "session_id": "string (선택사항, 최대 100자)"
}
```

#### SearchRequest
```json
{
  "query": "string (1-1000자)",
  "search_type": "semantic|keyword|hybrid",
  "limit": "integer (1-100)",
  "filters": {
    "document_type": "contract|case|law|general",
    "date_from": "datetime",
    "date_to": "datetime",
    "keyword": "string (최대 100자)"
  }
}
```

### 응답 모델

#### ChatResponse
```json
{
  "response": "string",
  "confidence": "float (0.0-1.0)",
  "sources": [
    {
      "title": "string",
      "content": "string",
      "similarity": "float",
      "ml_quality_score": "float (ML 강화 시)"
    }
  ],
  "processing_time": "float (초)",
  "model": "string",
  "retrieved_docs_count": "integer"
}
```

#### SearchResponse
```json
{
  "results": [
    {
      "document_id": "integer",
      "title": "string",
      "content": "string",
      "similarity": "float",
      "search_type": "string",
      "matched_keywords": ["string"],
      "chunk_index": "integer"
    }
  ],
  "total_count": "integer",
  "query": "string",
  "search_type": "string",
  "processing_time": "float (초)"
}
```

## 🔍 검색 유형

### 1. Semantic Search (의미적 검색)
- **모델**: ko-sroberta-multitask (768차원)
- **특징**: 의미적 유사도 기반 검색
- **장점**: 의도 파악, 동의어 처리
- **사용 사례**: 일반적인 법률 질문

### 2. Keyword Search (키워드 검색)
- **방식**: 정확한 매칭
- **특징**: 키워드 기반 정확한 검색
- **장점**: 정확한 조문 검색
- **사용 사례**: 특정 법령 조문 검색

### 3. Hybrid Search (하이브리드 검색)
- **방식**: 의미적 + 키워드 통합
- **가중치**: 의미적 60% + 키워드 40%
- **특징**: 두 방식의 장점 결합
- **사용 사례**: 복합적인 법률 질문

## ⚡ 성능 지표

### 현재 성능
- **평균 검색 시간**: 0.015초
- **처리 속도**: 5.77 법률/초
- **성공률**: 99.9%
- **메모리 사용량**: 190MB (최적화됨)

### 데이터 현황
- **총 법률 문서**: 7,680개
- **벡터 임베딩**: 155,819개 문서
- **FAISS 인덱스 크기**: 456.5 MB
- **메타데이터 크기**: 326.7 MB

## 🚨 에러 처리

### HTTP 상태 코드
- `200`: 성공
- `400`: 잘못된 요청
- `401`: 인증 실패
- `404`: 리소스 없음
- `500`: 서버 내부 오류

### 에러 응답 형식
```json
{
  "error": "에러 메시지",
  "detail": "상세 에러 정보",
  "timestamp": "2025-10-16T10:30:00Z"
}
```

### 일반적인 에러
- `INVALID_REQUEST`: 잘못된 요청 형식
- `MODEL_NOT_LOADED`: 모델 로딩 실패
- `DATABASE_ERROR`: 데이터베이스 오류
- `VECTOR_STORE_ERROR`: 벡터 저장소 오류

## 📝 사용 예시

### Python 클라이언트 예시
```python
import requests

# 기본 채팅
response = requests.post(
    "http://localhost:8000/api/v1/chat",
    json={
        "message": "계약서 검토 요청",
        "context": "부동산 매매 계약서"
    }
)
print(response.json())

# ML 강화 검색
response = requests.post(
    "http://localhost:8000/api/v1/search",
    json={
        "query": "계약 해지 조건",
        "search_type": "hybrid",
        "limit": 10
    }
)
print(response.json())
```

### cURL 예시
```bash
# 기본 채팅
curl -X POST "http://localhost:8000/api/v1/chat" \
  -H "Content-Type: application/json" \
  -d '{
    "message": "계약서 검토 요청",
    "context": "부동산 매매 계약서"
  }'

# 하이브리드 검색
curl -X POST "http://localhost:8000/api/search/" \
  -H "Content-Type: application/json" \
  -d '{
    "query": "계약 해지 조건",
    "search_types": ["semantic", "keyword"],
    "max_results": 20
  }'
```

## 🔄 버전 관리

### 현재 버전: v1.0.0
- ML 강화 RAG 시스템
- 하이브리드 검색 엔진
- 다중 모델 지원 (KoBART, ko-sroberta, BGE-M3)
- 완전한 API 엔드포인트

### 향후 계획
- v1.1.0: 계약서 분석 기능 확장
- v1.2.0: 다국어 지원 (영어, 일본어)
- v2.0.0: 실시간 협업 기능

## 📞 지원 및 문의

- **문서 버전**: 1.0
- **마지막 업데이트**: 2025-10-16
- **상태**: 🟢 완전 구현 완료 - 운영 준비 단계

API 사용에 대한 문의사항이나 개선 제안이 있으시면 프로젝트 관리자에게 연락해주세요.

