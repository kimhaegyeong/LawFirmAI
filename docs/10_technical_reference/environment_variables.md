# 환경 변수 관리 가이드

## 개요

LawFirmAI 프로젝트는 **중앙 집중식 .env 로더**를 사용하여 환경 변수를 관리합니다.

## .env 파일 구조

프로젝트는 다음 순서로 .env 파일을 로드합니다 (우선순위 낮음 → 높음):

1. **프로젝트 루트 `.env`** - 공통 설정 (최저 우선순위)
2. **`lawfirm_langgraph/.env`** - LangGraph 설정
3. **`api/.env`** - API 서버 전용 설정 (최고 우선순위)

## 사용 방법

### 1. 환경 변수 로드

프로젝트의 모든 모듈에서 공통 로더를 사용합니다:

```python
from utils.env_loader import load_all_env_files, ensure_env_loaded
from pathlib import Path

# 프로젝트 루트 경로
project_root = Path(__file__).parent.parent

# 모든 .env 파일 로드
load_all_env_files(project_root)

# 또는 이미 로드되었는지 확인 후 로드 (안전)
ensure_env_loaded(project_root)
```

### 2. .env 파일 생성

프로젝트 루트에 `.env.example` 파일을 참고하여 다음 파일들을 생성하세요:

- `.env` (프로젝트 루트) - 공통 설정
- `api/.env` - API 서버 전용 설정
- `lawfirm_langgraph/.env` - LangGraph 설정

### 3. 주요 환경 변수

#### 공통 설정
```bash
ENVIRONMENT=development
LOG_LEVEL=info
```

#### API 서버 설정 (api/.env)
```bash
API_HOST=0.0.0.0
API_PORT=8000
DEBUG=false
CORS_ORIGINS=http://localhost:3000,http://127.0.0.1:3000
DATABASE_URL=sqlite:///./data/api_sessions.db
```

#### LangGraph 설정 (lawfirm_langgraph/.env)
```bash
LANGGRAPH_ENABLED=true
GOOGLE_API_KEY=your_google_api_key_here
GOOGLE_MODEL=gemini-2.5-flash-lite
LLM_PROVIDER=google
```

## 구현 세부사항

### utils/env_loader.py

중앙 집중식 환경 변수 로더는 다음 기능을 제공합니다:

- `load_all_env_files(project_root)`: 모든 .env 파일을 우선순위에 따라 로드
- `ensure_env_loaded(project_root)`: 이미 로드되었는지 확인 후 로드 (안전)

### 로딩 순서

1. 프로젝트 루트 `.env` 로드 (override=False)
2. `lawfirm_langgraph/.env` 로드 (override=False)
3. `api/.env` 로드 (override=True) ← 최우선

이 순서로 인해 `api/.env`의 설정이 다른 파일의 설정을 덮어씁니다.

## 주의사항

1. **보안**: `.env` 파일은 Git에 커밋하지 마세요. `.gitignore`에 추가되어 있습니다.
2. **템플릿**: `.env.example` 파일을 참고하여 실제 `.env` 파일을 생성하세요.
3. **우선순위**: `api/.env`의 설정이 가장 높은 우선순위를 가집니다.

## 문제 해결

### 환경 변수가 로드되지 않는 경우

1. `utils/env_loader.py`가 프로젝트 루트에 있는지 확인
2. `python-dotenv` 패키지가 설치되어 있는지 확인: `pip install python-dotenv`
3. `.env` 파일이 올바른 위치에 있는지 확인

### 중복 로딩 문제

`ensure_env_loaded()` 함수를 사용하면 이미 로드된 환경 변수를 안전하게 처리합니다.

