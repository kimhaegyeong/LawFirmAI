# API 사용 예제

LawFirmAI API를 사용하는 예제입니다.

## 주요 엔드포인트

- `POST /api/v1/chat` - 채팅 메시지 처리 (LangGraph 워크플로우)
- `POST /api/v1/search/hybrid` - 하이브리드 검색 (정확한 매칭 + 의미적 검색)
- `POST /api/v1/search/exact` - 정확한 매칭 검색
- `POST /api/v1/search/semantic` - 의미적 검색
- `GET /api/v1/health` - 헬스체크
- `GET /docs` - API 문서 (Swagger UI)

## 사용 예제

### 채팅 API

```python
import requests

# 채팅 요청 (LangGraph 워크플로우)
response = requests.post(
    "http://localhost:8000/api/v1/chat",
    json={
        "message": "계약 해제 조건이 무엇인가요?",
        "session_id": "user_session_123"
    }
)

result = response.json()
print(f"답변: {result['answer']}")
print(f"신뢰도: {result.get('confidence', 'N/A')}")
```

### 하이브리드 검색 API

```python
import requests

# 하이브리드 검색 요청
response = requests.post(
    "http://localhost:8000/api/v1/search/hybrid",
    json={
        "query": "계약 해지 손해배상",
        "search_type": "hybrid",
        "filters": {
            "document_type": "precedent",
            "court_name": "대법원"
        },
        "limit": 10
    }
)

result = response.json()
print(f"총 {result['total_count']}건의 결과")
for doc in result['results']:
    print(f"제목: {doc['title']}")
    print(f"유사도 점수: {doc['similarity_score']:.3f}")
```

## API 문서 구조

- **[API 설계 명세서](docs/07_api/API_Documentation.md)** - LawFirmAI API 전체 명세
- **[국가법령정보 OPEN API 가이드](docs/07_api/open_law/README.md)** - 외부 API 연동 가이드
- **[API별 상세 가이드](docs/07_api/open_law/README.md)** - 각 API별 상세 문서


