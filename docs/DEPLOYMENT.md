# 배포 가이드

## 개요

LawFirmAI는 API 서버와 React 프론트엔드로 분리되어 독립적으로 배포할 수 있습니다.

## 아키텍처

```
┌─────────────────┐         ┌─────────────────┐
│  React          │  HTTP   │  FastAPI        │
│  Frontend       │ ──────> │  API Server    │
│  (Port 3000)    │         │  (Port 8000)    │
└─────────────────┘         └─────────────────┘
                                      │
                                      │
                              ┌───────▼────────┐
                              │  lawfirm_      │
                              │  langgraph     │
                              │  (Backend)     │
                              └────────────────┘
```

## 개발 환경 실행

### 1. API 서버 실행

```bash
cd api
pip install -r requirements.txt
python -m api.main
```

API 서버는 `http://localhost:8000`에서 실행됩니다.

### 2. React 프론트엔드 실행

```bash
cd frontend
npm install

# 환경 변수 설정
# .env 파일에 API_BASE_URL=http://localhost:8000 설정

npm run dev
```

React 프론트엔드는 `http://localhost:3000`에서 실행됩니다.

## Docker를 사용한 배포

### API 서버

```bash
cd api
docker build -t lawfirm-api .
docker run -p 8000:8000 lawfirm-api
```

### React 프론트엔드

```bash
cd frontend
docker build -t lawfirm-frontend .
docker run -p 3000:3000 -e API_BASE_URL=http://api-server:8000 lawfirm-frontend
```

### Docker Compose (통합)

프로젝트 루트에서:

```bash
docker-compose -f api/docker-compose.yml up
docker-compose -f frontend/docker-compose.yml up
```

## 프로덕션 배포

### API 서버

1. 환경 변수 설정:
   - `API_HOST`: 0.0.0.0
   - `API_PORT`: 8000
   - `CORS_ORIGINS`: React 프론트엔드 URL
   - `DATABASE_URL`: 프로덕션 데이터베이스 URL

2. 서버 실행:
   ```bash
   uvicorn api.main:app --host 0.0.0.0 --port 8000 --workers 4
   ```

### React 프론트엔드

1. 환경 변수 설정:
   - `API_BASE_URL`: API 서버 URL

2. 빌드 및 실행:
   ```bash
   npm run build
   npm run preview
   ```

또는 프로덕션 서버에서:
```bash
npm run build
npx serve -s dist -l 3000
```

## 환경 변수

### API 서버

| 변수 | 설명 | 기본값 |
|------|------|--------|
| `API_HOST` | 서버 호스트 | 0.0.0.0 |
| `API_PORT` | 서버 포트 | 8000 |
| `CORS_ORIGINS` | CORS 허용 오리진 | * |
| `DATABASE_URL` | 데이터베이스 URL | sqlite:///./data/api_sessions.db |
| `LANGGRAPH_ENABLED` | LangGraph 활성화 | true |

### React 프론트엔드

| 변수 | 설명 | 기본값 |
|------|------|--------|
| `VITE_API_BASE_URL` | API 서버 URL | http://localhost:8000 |

## 모니터링

### 헬스체크

```bash
curl http://localhost:8000/api/v1/health
```

### API 문서

- Swagger UI: http://localhost:8000/docs
- ReDoc: http://localhost:8000/redoc

## 문제 해결

### API 서버 연결 실패

1. `API_BASE_URL` 환경 변수 확인
2. API 서버가 실행 중인지 확인
3. CORS 설정 확인

### 세션 저장 실패

1. 데이터베이스 파일 권한 확인
2. 데이터 디렉토리 존재 확인

