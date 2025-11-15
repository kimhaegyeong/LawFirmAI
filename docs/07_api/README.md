# LawFirmAI API Server

FastAPI 기반 법률 AI 어시스턴트 백엔드 서버

## 설치 및 실행

### 1. 의존성 설치

```bash
pip install -r requirements.txt
```

### 2. 환경 변수 설정

`.env` 파일을 생성하고 필요한 환경 변수를 설정하세요:

```bash
cp .env.example .env
```

### 3. 서버 실행

```bash
python -m api.main
```

또는 uvicorn 직접 실행:

```bash
uvicorn api.main:app --host 0.0.0.0 --port 8000 --reload
```

## API 엔드포인트

### 채팅
- `POST /api/v1/chat` - 채팅 메시지 처리
- `POST /api/v1/chat/stream` - 스트리밍 채팅 응답
- `GET /api/v1/chat/{session_id}/sources` - 세션별 소스 조회

### 세션 관리
- `GET /api/v1/sessions` - 세션 목록 조회
- `GET /api/v1/sessions/by-date` - 날짜별 세션 목록 조회
- `POST /api/v1/sessions` - 새 세션 생성
- `GET /api/v1/sessions/{session_id}` - 세션 상세 조회
- `PUT /api/v1/sessions/{session_id}` - 세션 업데이트
- `DELETE /api/v1/sessions/{session_id}` - 세션 삭제
- `POST /api/v1/sessions/{session_id}/generate-title` - 세션 제목 생성

### 히스토리
- `GET /api/v1/history` - 히스토리 조회
- `POST /api/v1/history/export` - 히스토리 내보내기

### 피드백
- `POST /api/v1/feedback` - 피드백 제출

### 헬스체크
- `GET /api/v1/health` - 헬스체크

## Docker 실행

```bash
docker-compose up
```

## API 문서

서버 실행 후 다음 URL에서 API 문서를 확인할 수 있습니다:

- Swagger UI: http://localhost:8000/docs
- ReDoc: http://localhost:8000/redoc

## 관련 문서

- [시작 가이드](START_GUIDE.md) - API 서버 시작 가이드
- [보안 감사 가이드](SECURITY_AUDIT.md) - 보안 감사 가이드
- [보안 점검 계획서](SECURITY_CHECKLIST.md) - 보안 점검 계획서
- [API 문서](API_Documentation.md) - 상세 API 문서
- [API 엔드포인트](api_endpoints.md) - API 엔드포인트 목록
