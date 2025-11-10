# API 서버 시작 가이드

## Windows

### 시작
```bash
start_server.bat
```

### 종료
```bash
stop_server.bat
```

또는 터미널에서 `Ctrl+C`

## Linux/Mac

### 시작
```bash
chmod +x start_server.sh
./start_server.sh
```

### 종료
```bash
chmod +x stop_server.sh
./stop_server.sh
```

또는 터미널에서 `Ctrl+C`

## 수동 실행

### 1. 가상 환경 생성
```bash
python -m venv venv
```

### 2. 가상 환경 활성화

**Windows:**
```bash
venv\Scripts\activate
```

**Linux/Mac:**
```bash
source venv/bin/activate
```

### 3. 의존성 설치
```bash
pip install -r requirements.txt
```

### 4. 서버 실행
```bash
python -m api.main
```

## 환경 변수

`.env` 파일을 생성하거나 환경 변수로 설정:

```env
API_HOST=0.0.0.0
API_PORT=8000
CORS_ORIGINS=*
DATABASE_URL=sqlite:///./data/api_sessions.db
LANGGRAPH_ENABLED=true
```

## 접속 확인

- API 서버: http://localhost:8000
- API 문서: http://localhost:8000/docs

