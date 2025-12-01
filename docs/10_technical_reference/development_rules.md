# LawFirmAI 개발 규칙 및 가이드라인

## 📋 문서 개요

본 문서는 LawFirmAI 프로젝트의 개발 규칙, 코딩 스타일, 운영 가이드라인을 정의합니다.

## 🚀 프로세스 관리 규칙

### API 서버 관리

#### 서버 시작
```bash
# API 서버 시작
cd api
python main.py
```

#### 서버 종료

**⚠️ 중요**: `taskkill /f /im python.exe` 사용 금지

**올바른 종료 방법**:

1. **Ctrl+C로 종료**: 터미널에서 `Ctrl+C` 입력

2. **포트 기반 종료**:
```bash
# 8000 포트 사용 프로세스 확인
netstat -ano | findstr :8000

# 특정 PID 종료
taskkill /PID [PID번호] /F
```

#### 금지 사항

**❌ 절대 사용하지 말 것**:
```bash
# 모든 Python 프로세스 종료 (위험!)
taskkill /f /im python.exe
```

**✅ 올바른 방법**:
```bash
# 특정 PID만 종료
taskkill /PID 12345 /F
```

### React 프론트엔드 관리

#### 개발 서버 시작
```bash
# React 개발 서버 시작
cd frontend
npm install
npm run dev
```

#### 프로덕션 빌드
```bash
# 프로덕션 빌드
cd frontend
npm run build
npm run preview
```

## 🔧 개발 환경 규칙

### 디렉토리 구조 준수
```
LawFirmAI/
├── lawfirm_langgraph/               # 핵심 LangGraph 워크플로우 시스템
│   ├── config/                      # 설정 파일
│   ├── core/                        # 핵심 비즈니스 로직
│   │   ├── workflow/                # LangGraph 워크플로우 (메인)
│   │   ├── agents/                  # 레거시 에이전트 (하위 호환성)
│   │   ├── services/                # 비즈니스 서비스
│   │   ├── data/                    # 데이터 레이어
│   │   ├── models/                  # AI 모델
│   │   └── utils/                   # 유틸리티
│   └── tests/                       # 테스트 코드
├── frontend/                        # React 프론트엔드
│   ├── src/                         # 소스 코드
│   ├── package.json                 # 의존성
│   └── vite.config.ts               # Vite 설정
├── api/                              # FastAPI 서버
│   ├── main.py                      # 메인 앱
│   └── requirements.txt             # 의존성
├── scripts/                         # 유틸리티 스크립트
│   ├── data_collection/             # 데이터 수집
│   ├── data_processing/             # 데이터 전처리
│   ├── database/                    # 데이터베이스 관리
│   └── monitoring/                  # 모니터링
├── data/                            # 데이터 파일
│   ├── lawfirm_v2.db                # SQLite 데이터베이스
│   └── embeddings/                  # 벡터 임베딩
└── docs/                            # 문서
```

### Import 규칙

**프로젝트 모듈 Import**:
```python
# 프로젝트 루트 설정
import sys
from pathlib import Path

project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

# Core 모듈 Import
from lawfirm_langgraph.core.workflow.workflow_service import LangGraphWorkflowService
from lawfirm_langgraph.core.services.hybrid_search_engine import HybridSearchEngine
from lawfirm_langgraph.core.services.answer_generator import AnswerGenerator
from lawfirm_langgraph.config.langgraph_config import LangGraphConfig
```

**Import 순서**:
```python
# 1. 표준 라이브러리
import os
import sys
from typing import Dict, List, Optional

# 2. 서드파티 라이브러리
import torch
from fastapi import FastAPI
from langchain_google_genai import ChatGoogleGenerativeAI

# 3. 프로젝트 모듈
from lawfirm_langgraph.core.workflow.workflow_service import LangGraphWorkflowService
from lawfirm_langgraph.core.services.hybrid_search_engine import HybridSearchEngine
```

## 📝 로깅 규칙

### Windows 환경 로깅 주의사항

**이모지 사용 금지** (Windows cp949 인코딩 문제):
```python
# ❌ 잘못된 예시
logger.info("🚀 Starting process...")
logger.info("✅ Process completed")

# ✅ 올바른 예시  
logger.info("Starting process...")
logger.info("Process completed")
logger.info("[OK] Process completed")
logger.info("[ERROR] Process failed")
```

### 한국어 인코딩 처리 규칙

**⚠️ 중요**: Windows 환경에서 한국어 콘솔 출력 문제 해결을 위한 규칙

#### 환경 변수 설정 (필수)
```python
# 모든 Python 파일 상단에 추가
import os
import sys

# 인코딩 설정 (최우선)
os.environ['PYTHONIOENCODING'] = 'utf-8'
if sys.platform == 'win32':
    os.environ['PYTHONLEGACYWINDOWSSTDIO'] = 'utf-8'
```

#### 안전한 콘솔 출력
```python
def safe_print(message: str):
    """안전한 콘솔 출력 (인코딩 처리)"""
    try:
        print(message)
    except UnicodeEncodeError:
        print(message.encode('utf-8', errors='replace').decode('utf-8'))

# 사용 예시
safe_print("법률 문서 분석을 시작합니다.")
safe_print("벡터 저장소 로딩 완료")
```

### 현재 구현된 로깅 시스템
```python
# infrastructure/utils/logger.py에서 사용 중
import logging

# 로깅 설정
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('logs/lawfirm_ai.log')
    ]
)
logger = logging.getLogger(__name__)

# 사용 예시
logger.info("LawFirmAI service initialized")
logger.info("Vector store loaded successfully")
logger.warning("Configuration issue detected")
logger.error("Critical error occurred")
```

## 🛡️ 보안 규칙

### 환경 변수 관리
```python
import os
from pathlib import Path

# 환경 변수 파일 로드
env_file = Path(".env")
if env_file.exists():
    from dotenv import load_dotenv
    load_dotenv()

# API 키 관리
api_key = os.getenv("OPENAI_API_KEY")
if not api_key:
    logger.warning("OpenAI API key not found, using fallback")

# 현재 사용 중인 환경 변수
required_env_vars = [
    "OPENAI_API_KEY",      # OpenAI API 키
    "GOOGLE_API_KEY",      # Google API 키 (선택사항)
    "DATABASE_URL",        # 데이터베이스 URL
    "MODEL_PATH"           # 모델 경로
]
```

### .env.example 동기화 규칙 (필수)

다음 규칙을 반드시 준수하세요. 누락 시 리뷰에서 변경 요청됩니다.

- 신규/변경된 환경변수는 실제 `.env`에 추가하는 동시에 반드시 `.env.example`에도 동일 키를 추가합니다.
- 민감한 값은 `.env.example`에는 비워두거나 예시값으로 대체합니다. 예: `OPENAI_API_KEY=your-api-key-here`
- 불필요해진 환경변수 제거 시 `.env.example`에서도 함께 제거합니다.
- PR에 환경변수 변경이 포함되면, 변경사항을 `docs/07_api`의 관련 문서 또는 README의 설정 섹션에도 간단히 반영합니다.
- 로컬 실행 또는 배포 스크립트가 참조하는 키 목록은 `.env.example`와 불일치가 없도록 검증합니다.

권장 템플릿 예시:
```env
# API Keys
OPENAI_API_KEY=
GOOGLE_API_KEY=

# Application
DATABASE_URL=sqlite:///./data/lawfirm.db
MODEL_PATH=./models
```

## 🧪 테스트 규칙

### 현재 구현된 테스트 시스템
```python
# tests/ 디렉토리에서 구현
import sys
from pathlib import Path

# 프로젝트 루트를 Python 경로에 추가
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

def test_vector_store_loading():
    """벡터 저장소 로딩 테스트"""
    from lawfirm_langgraph.core.data.vector_store import VectorStore
    
    vector_store = VectorStore("test-model")
    assert vector_store is not None

def test_workflow_service():
    """워크플로우 서비스 테스트"""
    from lawfirm_langgraph.core.workflow.workflow_service import LangGraphWorkflowService
    from lawfirm_langgraph.config.langgraph_config import LangGraphConfig
    
    config = LangGraphConfig.from_env()
    workflow = LangGraphWorkflowService(config)
    assert workflow is not None
```

## 📊 성능 모니터링 규칙

### 메모리 사용량 모니터링
```python
import psutil
import time

def monitor_memory():
    """메모리 사용량 모니터링"""
    process = psutil.Process()
    memory_info = process.memory_info()
    
    logger.info(f"Memory usage: {memory_info.rss / 1024 / 1024:.1f} MB")
    
    # 메모리 사용량이 임계값을 초과하면 경고
    if memory_info.rss > 1024 * 1024 * 1024:  # 1GB
        logger.warning("High memory usage detected")
```

### 응답 시간 측정
```python
import time
from functools import wraps

def measure_time(func):
    """실행 시간 측정 데코레이터"""
    @wraps(func)
    def wrapper(*args, **kwargs):
        start_time = time.time()
        result = func(*args, **kwargs)
        end_time = time.time()
        
        logger.info(f"{func.__name__} executed in {end_time - start_time:.3f}s")
        return result
    return wrapper

# 사용 예시
@measure_time
def search_documents(query):
    # 검색 로직
    pass
```

## 🔄 배포 규칙

### 현재 구현된 Docker 설정
```dockerfile
# apps/streamlit/Dockerfile
FROM python:3.9-slim

WORKDIR /app

# 의존성 설치
COPY apps/streamlit/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# 애플리케이션 복사
COPY apps/streamlit/ ./apps/streamlit/
COPY lawfirm_langgraph/ ./lawfirm_langgraph/
COPY infrastructure/ ./infrastructure/

# 비root 사용자로 실행
RUN useradd --create-home --shell /bin/bash app
USER app

# 포트 노출
EXPOSE 8501

# 헬스체크 추가
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:8501/ || exit 1

CMD ["streamlit", "run", "apps/streamlit/app.py"]
```
