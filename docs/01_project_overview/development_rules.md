# LawFirmAI 개발 규칙 및 가이드라인

## 📋 문서 개요

본 문서는 LawFirmAI 프로젝트의 개발 규칙, 코딩 스타일, 운영 가이드라인을 정의합니다.

## 🚀 프로세스 관리 규칙

### Gradio 서버 관리

#### 서버 시작
```bash
# Gradio 서버 시작 (LangChain 기반)
cd gradio
python simple_langchain_app.py
```

#### 서버 종료 (PID 기준)
**⚠️ 중요**: `taskkill /f /im python.exe` 사용 금지

**올바른 종료 방법**:

1. **PID 파일 기반 종료** (권장):
```bash
# Windows
python gradio/stop_server.py

# 또는 배치 파일 사용
gradio/stop_server.bat
```

2. **포트 기반 종료**:
```bash
# 7860 포트 사용 프로세스 확인
netstat -ano | findstr :7860

# 특정 PID 종료
taskkill /PID [PID번호] /F
```

#### PID 관리 구현 규칙

**모든 Gradio 애플리케이션은 다음 규칙을 따라야 합니다**:

1. **PID 파일 생성**:
```python
import os
import signal
import atexit
from pathlib import Path

def save_pid():
    """현재 프로세스 PID를 파일에 저장"""
    pid = os.getpid()
    pid_file = Path("gradio_server.pid")
    
    try:
        with open(pid_file, 'w') as f:
            f.write(str(pid))
        print(f"PID {pid} saved to {pid_file}")
    except Exception as e:
        print(f"Failed to save PID: {e}")

def cleanup_pid():
    """PID 파일 정리"""
    pid_file = Path("gradio_server.pid")
    if pid_file.exists():
        try:
            pid_file.unlink()
            print("PID file removed")
        except Exception as e:
            print(f"Failed to remove PID file: {e}")

# 앱 시작 시
save_pid()

# 앱 종료 시 정리
atexit.register(cleanup_pid)
signal.signal(signal.SIGINT, lambda s, f: cleanup_pid() or exit(0))
signal.signal(signal.SIGTERM, lambda s, f: cleanup_pid() or exit(0))
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

# 또는 제공된 스크립트 사용
python gradio/stop_server.py
```

## 🔧 개발 환경 규칙

### 디렉토리 구조 준수
```
LawFirmAI/
├── gradio/                          # Gradio 웹 애플리케이션
│   ├── simple_langchain_app.py      # 메인 LangChain 기반 앱
│   ├── app.py                       # 기본 Gradio 앱
│   ├── stop_server.py               # 서버 종료 스크립트
│   ├── requirements.txt             # Gradio 의존성
│   └── Dockerfile                   # Gradio Docker 설정
├── source/                          # 핵심 모듈
│   ├── services/                    # 비즈니스 로직 (80+ 서비스)
│   ├── data/                        # 데이터 처리
│   ├── models/                      # AI 모델
│   └── utils/                       # 유틸리티
├── data/                            # 데이터 파일
│   ├── lawfirm.db                   # SQLite 데이터베이스
│   └── embeddings/                  # 벡터 임베딩
└── docs/                            # 문서
```

### 벡터 저장소 경로 규칙
```python
# 현재 사용 중인 벡터 저장소
vector_store_paths = [
    "data/embeddings/ml_enhanced_ko_sroberta",  # ko-sroberta 벡터
    "data/embeddings/ml_enhanced_bge_m3",       # BGE-M3 벡터
]
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
# gradio/simple_langchain_app.py에서 사용 중
import logging

# 로깅 설정
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('logs/simple_langchain_gradio.log')
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

## 🧪 테스트 규칙

### 현재 구현된 테스트 시스템
```python
# gradio/test_simple_query.py에서 구현
import sys
from pathlib import Path

# 프로젝트 루트를 Python 경로에 추가
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

def test_vector_store_loading():
    """벡터 저장소 로딩 테스트"""
    from source.data.vector_store import LegalVectorStore
    
    vector_store = LegalVectorStore("test-model")
    assert vector_store is not None

def test_gradio_app_startup():
    """Gradio 앱 시작 테스트"""
    import subprocess
    import time
    
    # 앱 시작
    process = subprocess.Popen(['python', 'gradio/simple_langchain_app.py'])
    
    # 잠시 대기
    time.sleep(5)
    
    # 프로세스 상태 확인
    assert process.poll() is None, "App should be running"
    
    # 정리
    process.terminate()
    process.wait()
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
# gradio/Dockerfile (현재 구현)
FROM python:3.9-slim

WORKDIR /app

# 의존성 설치
COPY gradio/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# 애플리케이션 복사
COPY gradio/ ./gradio/
COPY source/ ./source/

# 비root 사용자로 실행
RUN useradd --create-home --shell /bin/bash app
USER app

# 포트 노출
EXPOSE 7860

# 헬스체크 추가
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:7860/ || exit 1

CMD ["python", "gradio/simple_langchain_app.py"]
```

