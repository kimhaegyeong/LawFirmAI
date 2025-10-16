# LawFirmAI 개발 규칙 및 가이드라인

## 📋 문서 개요

본 문서는 LawFirmAI 프로젝트의 개발 규칙, 코딩 스타일, 운영 가이드라인을 정의합니다.

## 🚀 프로세스 관리 규칙

### Gradio 서버 관리

#### 서버 시작
```bash
# Gradio 서버 시작 (리팩토링된 버전)
cd gradio
python simple_langchain_app.py
```

#### 간단한 테스트 실행
```bash
# 질의-답변 테스트 스크립트 실행
cd gradio
python test_simple_query.py
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

3. **프로그램 내 종료**:
```python
# Gradio 앱 내에서 graceful shutdown
import signal
import os

def signal_handler(signum, frame):
    print("Graceful shutdown initiated...")
    # 정리 작업 수행
    if os.path.exists("gradio_server.pid"):
        os.remove("gradio_server.pid")
    exit(0)

signal.signal(signal.SIGINT, signal_handler)
signal.signal(signal.SIGTERM, signal_handler)
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

2. **종료 스크립트 구현 (Windows 호환)**:
```python
# gradio/stop_server.py
import os
import subprocess
import sys
import json
from pathlib import Path

def stop_by_pid():
    """PID 파일을 이용한 서버 종료 (Windows 호환)"""
    pid_file = Path("gradio_server.pid")
    
    if not pid_file.exists():
        print("PID file not found")
        return False
    
    try:
        # JSON 형식의 PID 파일 읽기
        with open(pid_file, 'r', encoding='utf-8') as f:
            pid_data = json.load(f)
        
        pid = pid_data.get('pid')
        if not pid:
            print("Invalid PID data")
            return False
        
        # 프로세스 존재 확인 (Windows 호환)
        check_result = subprocess.run(['tasklist', '/FI', f'PID eq {pid}'], 
                                    capture_output=True, text=True, encoding='cp949')
        
        if str(pid) not in check_result.stdout:
            print(f"Process {pid} not found")
            pid_file.unlink()
            return False
        
        # Windows에서 프로세스 종료
        result = subprocess.run(['taskkill', '/PID', str(pid), '/F'], 
                              capture_output=True, text=True, encoding='cp949')
        
        if result.returncode == 0:
            print(f"Server with PID {pid} stopped successfully")
            pid_file.unlink()  # PID 파일 삭제
            return True
        else:
            print(f"Failed to stop server: {result.stderr}")
            return False
            
    except Exception as e:
        print(f"Error stopping server: {e}")
        return False

def stop_by_port():
    """포트 기반 서버 종료"""
    try:
        # netstat으로 포트 사용 프로세스 찾기
        result = subprocess.run(['netstat', '-ano'], 
                              capture_output=True, text=True)
        
        for line in result.stdout.split('\n'):
            if ':7860' in line and 'LISTENING' in line:
                parts = line.split()
                if len(parts) >= 5:
                    pid = parts[-1]
                    # Python 프로세스인지 확인
                    tasklist_result = subprocess.run(['tasklist', '/FI', f'PID eq {pid}'], 
                                                   capture_output=True, text=True)
                    if 'python.exe' in tasklist_result.stdout:
                        subprocess.run(['taskkill', '/PID', pid, '/F'])
                        print(f"Stopped Python process with PID {pid}")
                        return True
        
        print("No Python process found using port 7860")
        return False
        
    except Exception as e:
        print(f"Error stopping by port: {e}")
        return False

if __name__ == "__main__":
    print("Stopping Gradio server...")
    
    # PID 파일 기반 종료 시도
    if stop_by_pid():
        sys.exit(0)
    
    # 포트 기반 종료 시도
    if stop_by_port():
        sys.exit(0)
    
    print("No server found to stop")
    sys.exit(1)
```

#### 금지 사항

**❌ 절대 사용하지 말 것**:
```bash
# 모든 Python 프로세스 종료 (위험!)
taskkill /f /im python.exe

# 다른 개발 중인 프로세스까지 종료될 수 있음
```

**❌ Windows에서 사용하지 말 것**:
```python
# Unix 계열 시스템용 (Windows에서 오류 발생)
os.kill(pid, signal.SIGTERM)
os.kill(pid, signal.SIGKILL)
```

**✅ 올바른 방법**:
```bash
# 특정 PID만 종료
taskkill /PID 12345 /F

# 또는 제공된 스크립트 사용
python gradio/stop_server.py
```

**✅ Windows 호환 코드**:
```python
# Windows에서 프로세스 종료
subprocess.run(['taskkill', '/PID', str(pid), '/F'], 
              capture_output=True, text=True, encoding='cp949')

# 프로세스 존재 확인
subprocess.run(['tasklist', '/FI', f'PID eq {pid}'], 
              capture_output=True, text=True, encoding='cp949')
```

## 🔧 개발 환경 규칙

### 디렉토리 구조 준수
```
LawFirmAI/
├── gradio/                  # Gradio 애플리케이션
│   ├── app.py              # 메인 애플리케이션
│   ├── simple_langchain_app.py  # LangChain 기반 앱
│   ├── stop_server.py      # 서버 종료 스크립트
│   ├── stop_server.bat     # Windows 배치 파일
│   └── gradio_server.pid   # PID 파일 (자동 생성)
├── source/                 # 핵심 모듈
├── data/                   # 데이터 파일
└── docs/                   # 문서
```

### 벡터 저장소 경로 규칙

**상대 경로 사용 시 주의사항**:
```python
# Gradio 앱에서 실행 시 (gradio/ 디렉토리)
vector_store_paths = [
    "../data/embeddings/ml_enhanced_ko_sroberta",  # 상위 디렉토리로 이동
    "../data/embeddings/ml_enhanced_bge_m3",
    "../data/embeddings/faiss_index"
]

# 프로젝트 루트에서 실행 시
vector_store_paths = [
    "./data/embeddings/ml_enhanced_ko_sroberta",
    "./data/embeddings/ml_enhanced_bge_m3", 
    "./data/embeddings/faiss_index"
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

### 로깅 레벨 규칙
```python
import logging

# 로깅 설정
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('logs/app.log')
    ]
)

logger = logging.getLogger(__name__)

# 사용 예시
logger.info("Application started")
logger.warning("Configuration issue detected")
logger.error("Critical error occurred")
logger.debug("Debug information")  # 개발 시에만 사용
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
```

### 파일 권한 관리
```python
# 파일 생성 시 권한 설정
def create_secure_file(file_path, content):
    """보안이 적용된 파일 생성"""
    try:
        with open(file_path, 'w', encoding='utf-8') as f:
            f.write(content)
        
        # Windows에서 파일 권한 설정 (필요시)
        # os.chmod(file_path, 0o600)  # 소유자만 읽기/쓰기
        
        return True
    except Exception as e:
        logger.error(f"Failed to create file {file_path}: {e}")
        return False
```

## 🧪 테스트 규칙

### 단위 테스트
```python
import pytest
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
```

### 통합 테스트
```python
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

### Docker 컨테이너 관리
```dockerfile
# Dockerfile 예시
FROM python:3.9-slim

WORKDIR /app

# 의존성 설치
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# 애플리케이션 복사
COPY . .

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

### 환경별 설정
```python
# config.py
import os
from enum import Enum

class Environment(Enum):
    DEVELOPMENT = "development"
    STAGING = "staging" 
    PRODUCTION = "production"

class Config:
    def __init__(self):
        self.env = Environment(os.getenv("ENVIRONMENT", "development"))
        self.debug = self.env == Environment.DEVELOPMENT
        self.log_level = "DEBUG" if self.debug else "INFO"
        
        # 환경별 설정
        if self.env == Environment.PRODUCTION:
            self.host = "0.0.0.0"
            self.port = 7860
        else:
            self.host = "127.0.0.1"
            self.port = 7860
```

## 📋 체크리스트

### 개발 시작 전 체크리스트
- [ ] 프로젝트 구조 규칙 준수
- [ ] 환경 변수 설정 완료
- [ ] 의존성 설치 완료
- [ ] 로깅 설정 확인

### 코드 커밋 전 체크리스트
- [ ] 이모지 제거 (Windows 호환성)
- [ ] 상대 경로 올바른 설정
- [ ] PID 관리 코드 포함
- [ ] 에러 처리 구현
- [ ] 로깅 메시지 추가

### 배포 전 체크리스트
- [ ] 모든 테스트 통과
- [ ] 성능 테스트 완료
- [ ] 보안 검토 완료
- [ ] 문서 업데이트 완료

---

## 📞 문의 및 지원

개발 규칙에 대한 문의사항이나 개선 제안이 있으시면 프로젝트 관리자에게 연락해주세요.

**마지막 업데이트**: 2025-10-16
**버전**: 1.0
