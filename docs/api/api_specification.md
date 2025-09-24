# LawFirmAI API 설계 명세서

## 1. API 개요

### 1.1 기본 정보
- **API 이름**: LawFirmAI REST API
- **버전**: v1.0.0
- **기본 URL**: `http://localhost:8000/api/v1` (로컬 개발)
- **인증**: API Key (필수)
- **응답 형식**: JSON
- **문자 인코딩**: UTF-8

### 1.2 API 특징
- RESTful 설계 원칙 준수
- OpenAPI 3.0 스펙 준수
- 비동기 처리 지원
- 자동 문서화 (Swagger UI)
- 속도 제한 (Rate Limiting)
- CORS 지원

## 2. 엔드포인트 목록

### 2.1 채팅 관련 엔드포인트
| 메서드 | 엔드포인트 | 설명 | 인증 |
|--------|------------|------|------|
| POST | `/api/v1/chat` | 채팅 메시지 처리 | 필수 |
| GET | `/api/v1/chat/history/{session_id}` | 대화 기록 조회 | 필수 |
| DELETE | `/api/v1/chat/history/{session_id}` | 대화 기록 삭제 | 필수 |

### 2.2 검색 관련 엔드포인트
| 메서드 | 엔드포인트 | 설명 | 인증 |
|--------|------------|------|------|
| POST | `/api/v1/search/precedents` | 판례 검색 | 필수 |
| POST | `/api/v1/search/laws` | 법령 검색 | 필수 |
| POST | `/api/v1/search/qa` | Q&A 검색 | 필수 |
| POST | `/api/v1/search/advanced` | 고급 검색 | 필수 |

### 2.3 분석 관련 엔드포인트
| 메서드 | 엔드포인트 | 설명 | 인증 |
|--------|------------|------|------|
| POST | `/api/v1/analyze/contract` | 계약서 분석 | 필수 |
| POST | `/api/v1/analyze/document` | 법률 문서 분석 | 필수 |
| POST | `/api/v1/analyze/entities` | 법률 개체 추출 | 필수 |

### 2.4 시스템 관련 엔드포인트
| 메서드 | 엔드포인트 | 설명 | 인증 |
|--------|------------|------|------|
| GET | `/api/v1/health` | 헬스체크 | 없음 |
| GET | `/api/v1/status` | 시스템 상태 | 없음 |
| GET | `/api/v1/metrics` | 성능 메트릭 | 관리자 |

## 3. 요청/응답 스키마

### 3.1 채팅 API

#### POST /api/v1/chat
**요청 스키마:**
```json
{
  "message": "string",
  "context": "string (optional)",
  "session_id": "string (optional)",
  "max_length": "integer (optional, default: 512)",
  "temperature": "number (optional, default: 0.7)"
}
```

**응답 스키마:**
```json
{
  "response": "string",
  "confidence": "number",
  "sources": [
    {
      "type": "string",
      "id": "string",
      "title": "string",
      "url": "string (optional)"
    }
  ],
  "session_id": "string",
  "processing_time": "number",
  "timestamp": "string"
}
```

**에러 응답:**
```json
{
  "error": {
    "code": "string",
    "message": "string",
    "details": "object (optional)"
  },
  "timestamp": "string"
}
```

### 3.2 판례 검색 API

#### POST /api/v1/search/precedents
**요청 스키마:**
```json
{
  "query": "string",
  "filters": {
    "court_name": "string (optional)",
    "case_type": "string (optional)",
    "date_from": "string (optional, ISO 8601)",
    "date_to": "string (optional, ISO 8601)",
    "keywords": ["string"] (optional)
  },
  "limit": "integer (optional, default: 10)",
  "offset": "integer (optional, default: 0)",
  "sort_by": "string (optional, default: 'relevance')"
}
```

**응답 스키마:**
```json
{
  "results": [
    {
      "id": "string",
      "case_number": "string",
      "court_name": "string",
      "case_type": "string",
      "judgment_date": "string",
      "summary": "string",
      "keywords": ["string"],
      "relevance_score": "number",
      "url": "string (optional)"
    }
  ],
  "total_count": "integer",
  "query_time": "number",
  "pagination": {
    "limit": "integer",
    "offset": "integer",
    "has_more": "boolean"
  }
}
```

### 3.3 계약서 분석 API

#### POST /api/v1/analyze/contract
**요청 스키마:**
```json
{
  "document": "string",
  "analysis_depth": "integer (optional, default: 1)",
  "include_recommendations": "boolean (optional, default: true)",
  "language": "string (optional, default: 'ko')"
}
```

**응답 스키마:**
```json
{
  "summary": "string",
  "issues": [
    {
      "type": "string",
      "severity": "string",
      "description": "string",
      "location": "string (optional)",
      "suggestion": "string (optional)"
    }
  ],
  "recommendations": [
    {
      "category": "string",
      "description": "string",
      "priority": "string"
    }
  ],
  "entities": [
    {
      "type": "string",
      "value": "string",
      "confidence": "number"
    }
  ],
  "confidence": "number",
  "processing_time": "number"
}
```

## 4. 에러 코드 정의

### 4.1 HTTP 상태 코드
| 코드 | 의미 | 설명 |
|------|------|------|
| 200 | OK | 요청 성공 |
| 201 | Created | 리소스 생성 성공 |
| 400 | Bad Request | 잘못된 요청 |
| 401 | Unauthorized | 인증 실패 |
| 403 | Forbidden | 권한 없음 |
| 404 | Not Found | 리소스 없음 |
| 429 | Too Many Requests | 요청 한도 초과 |
| 500 | Internal Server Error | 서버 내부 오류 |
| 503 | Service Unavailable | 서비스 이용 불가 |

### 4.2 비즈니스 에러 코드
| 코드 | 의미 | 설명 |
|------|------|------|
| VALIDATION_ERROR | 검증 오류 | 입력 데이터 검증 실패 |
| MODEL_ERROR | 모델 오류 | AI 모델 처리 실패 |
| DATABASE_ERROR | 데이터베이스 오류 | DB 연결 또는 쿼리 실패 |
| RATE_LIMIT_EXCEEDED | 속도 제한 초과 | 요청 한도 초과 |
| INSUFFICIENT_CONTEXT | 컨텍스트 부족 | 응답 생성에 필요한 컨텍스트 부족 |
| TIMEOUT | 시간 초과 | 요청 처리 시간 초과 |
| API_KEY_MISSING | API 키 누락 | Authorization 헤더에 API 키가 없음 |
| API_KEY_INVALID | API 키 무효 | 제공된 API 키가 유효하지 않음 |
| API_KEY_EXPIRED | API 키 만료 | API 키가 만료됨 |

## 5. 인증 및 보안

### 5.1 API 키 인증
```http
Authorization: Bearer your-api-key-here
```

**로컬 개발 환경 설정:**
```bash
# 환경 변수 설정
export LAW_FIRM_AI_API_KEY="your-development-api-key"

# 또는 .env 파일에 추가
echo "LAW_FIRM_AI_API_KEY=your-development-api-key" >> .env
```

### 5.2 요청 제한 (로컬 개발)
- **개발 환경**: 100 requests/minute (과도한 호출 방지)
- **API 키 미제공시**: 401 Unauthorized 응답
- **잘못된 API 키**: 403 Forbidden 응답

### 5.3 입력 검증
- **메시지 길이**: 최대 10,000자
- **문서 크기**: 최대 1MB
- **파일 형식**: 텍스트, PDF, DOCX
- **특수 문자**: HTML 태그 자동 제거

### 5.4 로컬 개발 보안 고려사항
- **API 키 보안**: `.env` 파일을 `.gitignore`에 추가
- **CORS 설정**: 로컬 개발 시에만 `localhost` 허용
- **로깅**: API 키는 로그에 기록하지 않음
- **테스트 데이터**: 실제 민감한 데이터 사용 금지
- **포트 보안**: 개발 서버는 외부 접근 제한

## 6. OpenAPI 3.0 스펙

### 6.1 기본 정보
```yaml
openapi: 3.0.3
info:
  title: LawFirmAI API
  description: 법률 AI 어시스턴트 API
  version: 1.0.0
  contact:
    name: LawFirmAI Team
    email: schema9@gmail.com
  license:
    name: MIT
    url: https://opensource.org/licenses/MIT

servers:
  - url: http://localhost:8000/api/v1
    description: Local development server (기본)
  - url: https://api.lawfirmai.com/v1
    description: Production server
  - url: https://staging-api.lawfirmai.com/v1
    description: Staging server
```

### 6.2 컴포넌트 정의
```yaml
components:
  securitySchemes:
    ApiKeyAuth:
      type: apiKey
      in: header
      name: X-API-Key
    BearerAuth:
      type: http
      scheme: bearer
      bearerFormat: JWT

  schemas:
    ChatRequest:
      type: object
      required:
        - message
      properties:
        message:
          type: string
          maxLength: 10000
          description: 사용자 질문
        context:
          type: string
          description: 추가 컨텍스트
        session_id:
          type: string
          description: 세션 ID
        max_length:
          type: integer
          default: 512
          minimum: 100
          maximum: 2048
        temperature:
          type: number
          default: 0.7
          minimum: 0.1
          maximum: 2.0

    ChatResponse:
      type: object
      properties:
        response:
          type: string
          description: AI 응답
        confidence:
          type: number
          minimum: 0
          maximum: 1
          description: 신뢰도 점수
        sources:
          type: array
          items:
            $ref: '#/components/schemas/Source'
        session_id:
          type: string
        processing_time:
          type: number
          description: 처리 시간 (초)
        timestamp:
          type: string
          format: date-time

    Source:
      type: object
      properties:
        type:
          type: string
          enum: [precedent, law, qa]
        id:
          type: string
        title:
          type: string
        url:
          type: string
```

### 6.3 경로 정의
```yaml
paths:
  /chat:
    post:
      summary: 채팅 메시지 처리
      description: 사용자 질문을 처리하고 AI 응답을 반환합니다.
      tags:
        - Chat
      security:
        - ApiKeyAuth: []
        - BearerAuth: []
      requestBody:
        required: true
        content:
          application/json:
            schema:
              $ref: '#/components/schemas/ChatRequest'
      responses:
        '200':
          description: 성공
          content:
            application/json:
              schema:
                $ref: '#/components/schemas/ChatResponse'
        '400':
          description: 잘못된 요청
          content:
            application/json:
              schema:
                $ref: '#/components/schemas/ErrorResponse'
        '429':
          description: 요청 한도 초과
        '500':
          description: 서버 오류

  /search/precedents:
    post:
      summary: 판례 검색
      description: 판례 데이터를 검색합니다.
      tags:
        - Search
      security:
        - ApiKeyAuth: []
        - BearerAuth: []
      requestBody:
        required: true
        content:
          application/json:
            schema:
              $ref: '#/components/schemas/PrecedentSearchRequest'
      responses:
        '200':
          description: 성공
          content:
            application/json:
              schema:
                $ref: '#/components/schemas/PrecedentSearchResponse'
```

## 7. 사용 예제

### 7.1 채팅 API 사용 예제 (로컬 개발)
```python
import requests
import os

# 로컬 개발 서버 URL
url = "http://localhost:8000/api/v1/chat"

# API 키는 환경변수에서 가져오기 (필수)
api_key = os.getenv("LAW_FIRM_AI_API_KEY")
if not api_key:
    raise ValueError("LAW_FIRM_AI_API_KEY 환경변수가 설정되지 않았습니다")

headers = {
    "Authorization": f"Bearer {api_key}",
    "Content-Type": "application/json"
}

data = {
    "message": "계약서에서 주의해야 할 조항은 무엇인가요?",
    "session_id": "user123_session456",
    "max_length": 512,
    "temperature": 0.7
}

try:
    response = requests.post(url, headers=headers, json=data)
    response.raise_for_status()  # HTTP 에러 체크
    result = response.json()
    
    print(f"응답: {result['response']}")
    print(f"신뢰도: {result['confidence']}")
    print(f"처리 시간: {result['processing_time']}초")
except requests.exceptions.HTTPError as e:
    if e.response.status_code == 401:
        print("API 키가 유효하지 않습니다")
    elif e.response.status_code == 403:
        print("API 키가 잘못되었습니다")
    else:
        print(f"HTTP 에러: {e}")
except Exception as e:
    print(f"요청 실패: {e}")
```

### 7.2 판례 검색 API 사용 예제 (로컬 개발)
```python
import requests
import os

# 로컬 개발 서버 URL
url = "http://localhost:8000/api/v1/search/precedents"

# API 키는 환경변수에서 가져오기 (필수)
api_key = os.getenv("LAW_FIRM_AI_API_KEY")
if not api_key:
    raise ValueError("LAW_FIRM_AI_API_KEY 환경변수가 설정되지 않았습니다")

headers = {
    "Authorization": f"Bearer {api_key}",
    "Content-Type": "application/json"
}

data = {
    "query": "계약 해지 손해배상",
    "filters": {
        "court_name": "대법원",
        "case_type": "민사",
        "date_from": "2020-01-01",
        "date_to": "2023-12-31"
    },
    "limit": 10,
    "sort_by": "relevance"
}

try:
    response = requests.post(url, headers=headers, json=data)
    response.raise_for_status()  # HTTP 에러 체크
    result = response.json()
    
    for precedent in result['results']:
        print(f"사건번호: {precedent['case_number']}")
        print(f"법원: {precedent['court_name']}")
        print(f"요약: {precedent['summary']}")
        print(f"관련도: {precedent['relevance_score']}")
        print("---")
except requests.exceptions.HTTPError as e:
    if e.response.status_code == 401:
        print("API 키가 유효하지 않습니다")
    elif e.response.status_code == 403:
        print("API 키가 잘못되었습니다")
    else:
        print(f"HTTP 에러: {e}")
except Exception as e:
    print(f"요청 실패: {e}")
```

## 8. SDK 및 클라이언트 라이브러리

### 8.1 Python SDK (로컬 개발)
```python
import os
from lawfirmai import LawFirmAIClient

# API 키는 환경변수에서 가져오기 (필수)
api_key = os.getenv("LAW_FIRM_AI_API_KEY")
if not api_key:
    raise ValueError("LAW_FIRM_AI_API_KEY 환경변수가 설정되지 않았습니다")

# 로컬 개발 서버 사용
client = LawFirmAIClient(
    api_key=api_key,
    base_url="http://localhost:8000/api/v1"
)

# 채팅
response = client.chat("계약서 검토 요청")
print(response.text)

# 판례 검색
results = client.search_precedents("계약 해지", limit=5)
for result in results:
    print(result.title)
```

### 8.2 JavaScript SDK (로컬 개발)
```javascript
import LawFirmAI from 'lawfirmai-js';

// API 키는 환경변수에서 가져오기 (필수)
const apiKey = process.env.LAW_FIRM_AI_API_KEY;
if (!apiKey) {
  throw new Error('LAW_FIRM_AI_API_KEY 환경변수가 설정되지 않았습니다');
}

const client = new LawFirmAI({
  apiKey: apiKey,
  baseURL: 'http://localhost:8000/api/v1'
});

// 채팅
const response = await client.chat({
  message: '계약서 검토 요청',
  sessionId: 'user123'
});

console.log(response.response);
```

## 9. 모니터링 및 로깅

### 9.1 메트릭 수집
- **응답 시간**: 평균, 95th percentile
- **처리량**: requests per second
- **에러율**: 4xx, 5xx 에러 비율
- **사용량**: API 호출 수, 데이터 전송량

### 9.2 로그 형식
```json
{
  "timestamp": "2024-01-15T10:30:00Z",
  "level": "INFO",
  "service": "lawfirmai-api",
  "request_id": "req_123456789",
  "method": "POST",
  "endpoint": "/api/v1/chat",
  "status_code": 200,
  "response_time": 1.234,
  "user_id": "user123",
  "session_id": "session456"
}
```

## 10. 로컬 개발 환경 설정

### 10.1 개발 환경 요구사항
- **Python**: 3.9 이상
- **FastAPI**: 0.100.0 이상
- **Uvicorn**: 0.23.0 이상
- **포트**: 8000 (기본)

### 10.2 환경 변수 설정
```bash
# .env 파일 생성
cat > .env << EOF
# API 키 (필수)
LAW_FIRM_AI_API_KEY=dev-api-key-12345

# 데이터베이스 설정
DATABASE_URL=sqlite:///./data/lawfirm.db

# 모델 설정
MODEL_PATH=./models
DEVICE=cpu

# 로깅 설정
LOG_LEVEL=DEBUG
LOG_FILE=./logs/lawfirm_ai.log

# 서버 설정
HOST=0.0.0.0
PORT=8000
DEBUG=true
EOF
```

### 10.3 개발 서버 실행
```bash
# 의존성 설치
pip install -r requirements.txt

# 환경 변수 로드
source .env

# 개발 서버 실행
uvicorn main:app --host 0.0.0.0 --port 8000 --reload

# 또는 Python으로 직접 실행
python main.py
```

### 10.4 API 키 검증
```python
# API 키 검증 예제
import requests

def verify_api_key(api_key: str) -> bool:
    """API 키 유효성 검증"""
    url = "http://localhost:8000/api/v1/health"
    headers = {"Authorization": f"Bearer {api_key}"}
    
    try:
        response = requests.get(url, headers=headers)
        return response.status_code == 200
    except:
        return False

# 사용 예제
api_key = "dev-api-key-12345"
if verify_api_key(api_key):
    print("API 키가 유효합니다")
else:
    print("API 키가 유효하지 않습니다")
```

### 10.5 개발용 API 키 생성
```python
# 개발용 API 키 생성 스크립트
import secrets
import string

def generate_dev_api_key(length: int = 32) -> str:
    """개발용 API 키 생성"""
    alphabet = string.ascii_letters + string.digits
    return ''.join(secrets.choice(alphabet) for _ in range(length))

# 사용 예제
dev_key = generate_dev_api_key()
print(f"개발용 API 키: {dev_key}")
```

## 11. 버전 관리 및 호환성

### 11.1 버전 정책
- **메이저 버전**: 호환되지 않는 변경사항
- **마이너 버전**: 새로운 기능 추가 (하위 호환)
- **패치 버전**: 버그 수정 (하위 호환)

### 11.2 지원 기간
- **현재 버전**: 12개월 지원
- **이전 버전**: 6개월 지원
- **더 이상 지원하지 않는 버전**: 3개월 경고 후 중단

이 API 설계 명세서는 LawFirmAI 시스템의 안정적이고 확장 가능한 API 서비스를 위한 기반을 제공합니다.
