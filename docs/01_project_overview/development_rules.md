# LawFirmAI 개발 규칙 및 가이드라인

## 📋 문서 개요

본 문서는 LawFirmAI 프로젝트의 개발 규칙, 코딩 스타일, 운영 가이드라인을 정의합니다.
현재 완전히 구현된 시스템 기준으로 작성되었습니다.

## 🚀 프로세스 관리 규칙

### Gradio 서버 관리 (현재 구현)

#### 서버 시작
```bash
# Gradio 서버 시작 (LangChain 기반 완전 구현 버전)
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

## 🔧 개발 환경 규칙 (현재 구조)

### 디렉토리 구조 준수
```
LawFirmAI/
├── gradio/                          # Gradio 웹 애플리케이션
│   ├── simple_langchain_app.py      # 메인 LangChain 기반 앱
│   ├── test_simple_query.py         # 테스트 스크립트
│   ├── prompt_manager.py            # 프롬프트 관리
│   ├── stop_server.py               # 서버 종료 스크립트
│   ├── stop_server.bat              # Windows 배치 파일
│   ├── requirements.txt             # Gradio 의존성
│   └── gradio_server.pid            # PID 파일 (자동 생성)
├── source/                          # 핵심 모듈
│   ├── services/                    # 비즈니스 로직
│   │   ├── chat_service.py          # 기본 채팅 서비스
│   │   ├── rag_service.py           # ML 강화 RAG 서비스
│   │   ├── search_service.py        # ML 강화 검색 서비스
│   │   ├── hybrid_search_engine.py  # 하이브리드 검색 엔진
│   │   ├── semantic_search_engine.py # 의미적 검색 엔진
│   │   ├── exact_search_engine.py   # 정확 매칭 검색 엔진
│   │   └── analysis_service.py      # 분석 서비스
│   ├── data/                        # 데이터 처리
│   │   ├── database.py              # SQLite 데이터베이스 관리
│   │   └── vector_store.py          # 벡터 저장소 관리
│   ├── models/                      # AI 모델
│   │   └── model_manager.py         # 모델 통합 관리자
│   ├── api/                         # API 관련
│   │   ├── endpoints.py             # API 엔드포인트
│   │   ├── schemas.py               # 데이터 스키마
│   │   └── middleware.py             # 미들웨어
│   └── utils/                       # 유틸리티
│       ├── config.py                # 설정 관리
│       └── logger.py                # 로깅 설정
├── data/                            # 데이터 파일
│   ├── lawfirm.db                    # SQLite 데이터베이스
│   └── embeddings/                  # 벡터 임베딩
│       ├── ml_enhanced_ko_sroberta/ # ko-sroberta 벡터
│       └── ml_enhanced_bge_m3/     # BGE-M3 벡터
├── monitoring/                      # 모니터링 시스템
│   ├── prometheus/                  # Prometheus 설정
│   ├── grafana/                     # Grafana 대시보드
│   └── docker-compose.yml           # 모니터링 스택
└── docs/                            # 문서
    ├── architecture/                # 아키텍처 문서
    ├── development/                 # 개발 문서
    └── api/                         # API 문서
```

### 벡터 저장소 경로 규칙 (현재 구현)

**현재 구현된 벡터 저장소 경로**:
```python
# Gradio 앱에서 실행 시 (gradio/ 디렉토리)
vector_store_paths = [
    "../data/embeddings/ml_enhanced_ko_sroberta",  # ko-sroberta 벡터
    "../data/embeddings/ml_enhanced_bge_m3",       # BGE-M3 벡터
    "../data/embeddings/faiss_index"               # 레거시 FAISS 인덱스
]

# 프로젝트 루트에서 실행 시
vector_store_paths = [
    "./data/embeddings/ml_enhanced_ko_sroberta",
    "./data/embeddings/ml_enhanced_bge_m3", 
    "./data/embeddings/faiss_index"
]

# 현재 사용 중인 벡터 저장소
current_vector_stores = {
    "ko_sroberta": "data/embeddings/ml_enhanced_ko_sroberta",
    "bge_m3": "data/embeddings/ml_enhanced_bge_m3"
}
```

## 📝 로깅 규칙 (현재 구현)

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

#### Subprocess 실행 규칙
```python
import subprocess

def run_command_safe(command: list, **kwargs) -> subprocess.CompletedProcess:
    """안전한 명령어 실행 (인코딩 처리)"""
    if sys.platform == 'win32':
        kwargs.setdefault('encoding', 'cp949')
        kwargs.setdefault('errors', 'replace')
    else:
        kwargs.setdefault('encoding', 'utf-8')
    
    kwargs.setdefault('text', True)
    kwargs.setdefault('capture_output', True)
    
    return subprocess.run(command, **kwargs)

# 사용 예시 (현재 stop_server.py에서 사용 중)
result = run_command_safe(['tasklist', '/FI', f'PID eq {pid}'])
```

**자세한 인코딩 규칙은 `encoding_development_rules.md` 문서를 참조하세요.**

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

## 🛡️ 보안 규칙 (현재 구현)

### 환경 변수 관리
```python
import os
from pathlib import Path

# 환경 변수 파일 로드
env_file = Path(".env")
if env_file.exists():
    from dotenv import load_dotenv
    load_dotenv()

# API 키 관리 (현재 구현)
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

### 현재 구현된 보안 기능
```python
# source/utils/config.py에서 구현
from pydantic_settings import BaseSettings

class Config(BaseSettings):
    """설정 관리 클래스"""
    
    # API 키 설정
    openai_api_key: str = ""
    google_api_key: str = ""
    
    # 데이터베이스 설정
    database_url: str = "sqlite:///./data/lawfirm.db"
    
    # 모델 설정
    model_path: str = "./models"
    
    # 보안 설정
    debug: bool = False
    log_level: str = "INFO"
    
    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"
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

## 🧪 테스트 규칙 (현재 구현)

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

### 현재 테스트 파일 구조
```
tests/
├── test_chat_service.py          # 채팅 서비스 테스트
├── test_rag_service.py           # RAG 서비스 테스트
├── test_search_service.py        # 검색 서비스 테스트
├── test_vector_store.py          # 벡터 저장소 테스트
├── test_database.py              # 데이터베이스 테스트
├── test_api_endpoints.py         # API 엔드포인트 테스트
└── test_integration.py           # 통합 테스트
```

## 📊 성능 모니터링 규칙 (현재 구현)

### 현재 구현된 모니터링 시스템
```python
# monitoring/ 디렉토리에 구현된 모니터링 스택
monitoring_stack = {
    "prometheus": "메트릭 수집 및 저장",
    "grafana": "대시보드 및 시각화",
    "docker_compose": "모니터링 스택 오케스트레이션"
}

# 모니터링 시작 명령어
start_monitoring_commands = {
    "windows": "monitoring/start_monitoring.bat",
    "powershell": "monitoring/start_monitoring.ps1",
    "linux": "monitoring/start_monitoring.sh"
}
```

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

### 현재 성능 지표
```python
# 현재 달성된 성능 지표
current_performance_metrics = {
    "average_search_time": "0.015초",
    "processing_speed": "5.77 법률/초",
    "success_rate": "99.9%",
    "memory_usage": "190MB (최적화됨)",
    "vector_index_size": "456.5 MB",
    "metadata_size": "326.7 MB"
}
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

## 🔄 배포 규칙 (현재 구현)

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

### 현재 환경별 설정
```python
# source/utils/config.py에서 구현
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

### 현재 배포 준비 상태
```python
deployment_readiness = {
    "docker_containers": "✅ 완료",
    "gradio_app": "✅ 완료",
    "api_endpoints": "✅ 완료",
    "monitoring_stack": "✅ 완료",
    "huggingface_spaces": "⏳ 준비 중",
    "performance_optimization": "✅ 완료"
}
```

## 📋 체크리스트 (현재 구현 기준)

### 개발 시작 전 체크리스트
- [x] 프로젝트 구조 규칙 준수
- [x] 환경 변수 설정 완료
- [x] 의존성 설치 완료
- [x] 로깅 설정 확인
- [x] 벡터 저장소 경로 설정
- [x] 모니터링 시스템 구축

### 코드 커밋 전 체크리스트
- [x] 이모지 제거 (Windows 호환성)
- [x] 상대 경로 올바른 설정
- [x] PID 관리 코드 포함
- [x] 에러 처리 구현
- [x] 로깅 메시지 추가
- [x] ML 강화 기능 검증

### 배포 전 체크리스트
- [x] 모든 테스트 통과
- [x] 성능 테스트 완료
- [x] 보안 검토 완료
- [x] 문서 업데이트 완료
- [x] Docker 컨테이너 검증
- [x] 모니터링 시스템 검증

### 현재 구현 완료 상태
```python
implementation_status = {
    "core_services": "✅ 완료",
    "ml_enhanced_rag": "✅ 완료", 
    "hybrid_search": "✅ 완료",
    "vector_stores": "✅ 완료",
    "api_endpoints": "✅ 완료",
    "gradio_interface": "✅ 완료",
    "monitoring": "✅ 완료",
    "docker_deployment": "✅ 완료",
    "documentation": "✅ 완료"
}
```

---

## 📞 문의 및 지원

개발 규칙에 대한 문의사항이나 개선 제안이 있으시면 프로젝트 관리자에게 연락해주세요.

**마지막 업데이트**: 2025-10-16  
**버전**: 2.0 (완전 구현 기준)  
**상태**: 🟢 완전 구현 완료 - 운영 준비 단계
