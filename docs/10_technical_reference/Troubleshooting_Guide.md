# LawFirmAI 문제 해결 가이드

## 개요

이 가이드는 LawFirmAI 사용 중 발생할 수 있는 문제들을 진단하고 해결하는 방법을 제공합니다.

## 목차

1. [일반적인 문제](#일반적인-문제)
2. [설치 및 설정 문제](#설치-및-설정-문제)
3. [성능 문제](#성능-문제)
4. [데이터베이스 문제](#데이터베이스-문제)
5. [API 문제](#api-문제)
6. [UI 문제](#ui-문제)
7. [로그 분석](#로그-분석)
8. [고급 문제 해결](#고급-문제-해결)

## 일반적인 문제

### 문제: 애플리케이션이 시작되지 않음

#### 증상
- 터미널에서 오류 메시지 표시
- 브라우저에서 연결할 수 없음
- 포트가 사용 중이라는 메시지

#### 진단 단계

1. **포트 확인**
```bash
# 포트 7860 (Gradio) 사용 확인
netstat -tulpn | grep :7860
lsof -i :7860

# 포트 8000 (FastAPI) 사용 확인
netstat -tulpn | grep :8000
lsof -i :8000
```

2. **Python 버전 확인**
```bash
python --version
python3 --version
```

3. **의존성 확인**
```bash
pip list | grep gradio
pip list | grep fastapi
```

#### 해결 방법

1. **포트 충돌 해결**
```bash
# 사용 중인 프로세스 종료
sudo kill -9 <PID>

# 또는 다른 포트 사용
python gradio/app.py --server-port 7861
```

2. **Python 버전 문제**
```bash
# Python 3.9+ 설치 (Ubuntu/Debian)
sudo apt update
sudo apt install python3.9 python3.9-pip python3.9-venv

# 가상환경 재생성
rm -rf venv
python3.9 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

3. **의존성 재설치**
```bash
pip uninstall -y gradio fastapi
pip install -r requirements.txt
```

### 문제: 메모리 부족 오류

#### 증상
- "Out of memory" 오류 메시지
- 애플리케이션이 갑자기 종료됨
- 응답이 매우 느려짐

#### 진단 단계

1. **메모리 사용량 확인**
```bash
# 시스템 메모리 확인
free -h

# 프로세스별 메모리 사용량
ps aux --sort=-%mem | head -10

# Python 프로세스 메모리 사용량
ps aux | grep python
```

2. **메모리 사용량 모니터링**
```bash
# 실시간 메모리 모니터링
watch -n 1 'free -h'

# 메모리 사용량 로그
vmstat 1 10
```

#### 해결 방법

1. **메모리 최적화 설정**
```python
# 환경 변수 설정
export MEMORY_LIMIT_MB=2048
export MAX_CACHE_SIZE=500
export CACHE_TTL=1800

# 또는 .env 파일에 추가
MEMORY_LIMIT_MB=2048
MAX_CACHE_SIZE=500
CACHE_TTL=1800
```

2. **캐시 크기 조정**
```python
# ChatService에서 캐시 크기 줄이기
self.cache_manager = CacheManager(max_size=500, ttl=1800)
```

3. **시스템 메모리 증설**
```bash
# 더 큰 인스턴스 사용 (클라우드 환경)
# 또는 스왑 메모리 추가
sudo fallocate -l 2G /swapfile
sudo chmod 600 /swapfile
sudo mkswap /swapfile
sudo swapon /swapfile
```

### 문제: 느린 응답 시간

#### 증상
- 질문 후 응답까지 10초 이상 소요
- UI가 멈춘 것처럼 보임
- 타임아웃 오류 발생

#### 진단 단계

1. **응답 시간 측정**
```bash
# API 응답 시간 테스트
curl -w "@curl-format.txt" -o /dev/null -s "http://localhost:8000/api/chat"

# curl-format.txt 내용:
#      time_namelookup:  %{time_namelookup}\n
#         time_connect:  %{time_connect}\n
#      time_appconnect:  %{time_appconnect}\n
#     time_pretransfer:  %{time_pretransfer}\n
#        time_redirect:  %{time_redirect}\n
#   time_starttransfer:  %{time_starttransfer}\n
#                      ----------\n
#           time_total:  %{time_total}\n
```

2. **시스템 리소스 확인**
```bash
# CPU 사용률
top -p $(pgrep -f "python.*app.py")

# 디스크 I/O
iostat -x 1 5

# 네트워크 상태
netstat -i
```

#### 해결 방법

1. **캐시 활용**
```python
# 캐시 히트율 확인
cache_stats = self.cache_manager.get_stats()
print(f"Cache hit rate: {cache_stats['hit_rate']}")

# 캐시 크기 증가
self.cache_manager = CacheManager(max_size=2000, ttl=3600)
```

2. **데이터베이스 최적화**
```sql
-- 인덱스 생성
CREATE INDEX IF NOT EXISTS idx_sessions_user_id ON conversation_sessions(user_id);
CREATE INDEX IF NOT EXISTS idx_turns_session_id ON conversation_turns(session_id);

-- 쿼리 최적화
EXPLAIN QUERY PLAN SELECT * FROM conversation_turns WHERE session_id = ?;
```

3. **모델 최적화**
```python
# 모델 로딩 최적화
torch.set_num_threads(4)  # CPU 스레드 수 제한
torch.set_num_interop_threads(2)  # 인터럽트 스레드 수 제한
```

## 설치 및 설정 문제

### 문제: 의존성 설치 실패

#### 증상
- `pip install` 명령 실행 시 오류
- 패키지 버전 충돌
- 컴파일 오류

#### 해결 방법

1. **시스템 패키지 설치**
```bash
# Ubuntu/Debian
sudo apt update
sudo apt install build-essential python3-dev libffi-dev libssl-dev

# CentOS/RHEL
sudo yum groupinstall "Development Tools"
sudo yum install python3-devel libffi-devel openssl-devel
```

2. **가상환경 재생성**
```bash
# 기존 가상환경 삭제
rm -rf venv

# 새 가상환경 생성
python3 -m venv venv
source venv/bin/activate

# pip 업그레이드
pip install --upgrade pip setuptools wheel

# 의존성 설치
pip install -r requirements.txt
```

3. **개별 패키지 설치**
```bash
# 문제가 되는 패키지 개별 설치
pip install torch --index-url https://download.pytorch.org/whl/cpu
pip install transformers
pip install gradio
```

### 문제: 환경 변수 설정 오류

#### 증상
- 설정 파일을 찾을 수 없음
- 데이터베이스 연결 실패
- API 키 인증 실패

#### 해결 방법

1. **환경 변수 확인**
```bash
# 현재 환경 변수 확인
env | grep -E "(DATABASE|API|LOG)"

# .env 파일 확인
cat .env
```

2. **환경 변수 설정**
```bash
# .env 파일 생성
cat > .env << EOF
DATABASE_URL=sqlite:///./data/lawfirm.db
API_KEY=your_api_key_here
SECRET_KEY=your_secret_key_here
LOG_LEVEL=INFO
EOF
```

3. **환경 변수 로드 확인**
```python
# Python에서 환경 변수 확인
import os
from dotenv import load_dotenv

load_dotenv()
print(f"DATABASE_URL: {os.getenv('DATABASE_URL')}")
print(f"API_KEY: {os.getenv('API_KEY')}")
```

## 성능 문제

### 문제: 높은 CPU 사용률

#### 증상
- CPU 사용률이 90% 이상
- 시스템이 느려짐
- 팬 소음 증가

#### 진단 단계

1. **CPU 사용률 확인**
```bash
# 실시간 CPU 사용률
top -p $(pgrep -f "python.*app.py")

# CPU 사용률 히스토리
sar -u 1 10
```

2. **프로세스별 CPU 사용률**
```bash
# Python 프로세스 CPU 사용률
ps aux | grep python | sort -k3 -nr
```

#### 해결 방법

1. **CPU 스레드 수 제한**
```python
# torch 스레드 수 제한
torch.set_num_threads(2)
torch.set_num_interop_threads(1)

# 환경 변수로 설정
export OMP_NUM_THREADS=2
export MKL_NUM_THREADS=2
```

2. **배치 처리 최적화**
```python
# 배치 크기 조정
BATCH_SIZE = 1  # CPU에서는 작은 배치 크기 사용

# 모델 최적화
model.eval()
torch.no_grad()
```

3. **캐시 활용**
```python
# 자주 사용되는 결과 캐시
@lru_cache(maxsize=1000)
def process_query(query: str):
    # 쿼리 처리 로직
    pass
```

### 문제: 디스크 I/O 병목

#### 증상
- 디스크 사용률이 100%
- 파일 읽기/쓰기 속도 저하
- 시스템 응답 지연

#### 진단 단계

1. **디스크 사용률 확인**
```bash
# 디스크 사용률
iostat -x 1 5

# 디스크 공간 확인
df -h

# I/O 대기 프로세스
iotop
```

2. **파일 시스템 상태**
```bash
# 파일 시스템 체크
fsck /dev/sda1

# 디스크 성능 테스트
dd if=/dev/zero of=testfile bs=1M count=1000
```

#### 해결 방법

1. **SSD 사용**
```bash
# 디스크 타입 확인
lsblk -d -o name,rota

# SSD로 마이그레이션 (클라우드 환경)
# 또는 로컬 SSD 추가
```

2. **I/O 최적화**
```python
# 데이터베이스 최적화
PRAGMA journal_mode=WAL;
PRAGMA synchronous=NORMAL;
PRAGMA cache_size=10000;
```

3. **로그 로테이션**
```bash
# logrotate 설정
sudo nano /etc/logrotate.d/lawfirmai

# 내용:
/var/log/lawfirmai/*.log {
    daily
    missingok
    rotate 7
    compress
    delaycompress
    notifempty
    create 644 www-data www-data
}
```

## 데이터베이스 문제

### 문제: 데이터베이스 연결 실패

#### 증상
- "Database connection failed" 오류
- 데이터베이스 파일을 찾을 수 없음
- 권한 오류

#### 진단 단계

1. **데이터베이스 파일 확인**
```bash
# 데이터베이스 파일 존재 확인
ls -la data/lawfirm.db

# 파일 권한 확인
stat data/lawfirm.db
```

2. **데이터베이스 연결 테스트**
```bash
# SQLite 연결 테스트
sqlite3 data/lawfirm.db "SELECT name FROM sqlite_master WHERE type='table';"
```

#### 해결 방법

1. **권한 문제 해결**
```bash
# 데이터 디렉토리 권한 설정
chmod 755 data/
chmod 664 data/lawfirm.db
chown -R $USER:$USER data/
```

2. **데이터베이스 재생성**
```bash
# 기존 데이터베이스 백업
cp data/lawfirm.db data/lawfirm.db.backup

# 새 데이터베이스 생성
python -c "
from source.data.conversation_store import ConversationStore
store = ConversationStore('data/lawfirm.db')
print('Database created successfully')
"
```

3. **데이터베이스 무결성 확인**
```bash
# 무결성 체크
sqlite3 data/lawfirm.db "PRAGMA integrity_check;"

# 복구 시도
sqlite3 data/lawfirm.db "PRAGMA quick_check;"
```

### 문제: 데이터베이스 성능 저하

#### 증상
- 쿼리 실행 시간이 길어짐
- 데이터베이스 파일 크기 증가
- 메모리 사용량 증가

#### 진단 단계

1. **데이터베이스 크기 확인**
```bash
# 데이터베이스 파일 크기
ls -lh data/lawfirm.db

# 테이블별 크기 확인
sqlite3 data/lawfirm.db "
SELECT name, 
       (SELECT COUNT(*) FROM sqlite_master WHERE type='table' AND name=name) as row_count
FROM sqlite_master 
WHERE type='table';
"
```

2. **쿼리 성능 분석**
```bash
# 쿼리 실행 계획 확인
sqlite3 data/lawfirm.db "EXPLAIN QUERY PLAN SELECT * FROM conversation_turns WHERE session_id = 'test';"
```

#### 해결 방법

1. **인덱스 생성**
```sql
-- 자주 사용되는 컬럼에 인덱스 생성
CREATE INDEX IF NOT EXISTS idx_sessions_user_id ON conversation_sessions(user_id);
CREATE INDEX IF NOT EXISTS idx_turns_session_id ON conversation_turns(session_id);
CREATE INDEX IF NOT EXISTS idx_turns_timestamp ON conversation_turns(timestamp);
CREATE INDEX IF NOT EXISTS idx_memories_user_id ON contextual_memories(user_id);
```

2. **데이터베이스 최적화**
```sql
-- VACUUM 실행 (공간 정리)
VACUUM;

-- ANALYZE 실행 (통계 업데이트)
ANALYZE;

-- WAL 모드 활성화
PRAGMA journal_mode=WAL;
```

3. **오래된 데이터 정리**
```python
# 오래된 세션 정리
def cleanup_old_sessions(days=30):
    cutoff_date = datetime.now() - timedelta(days=days)
    # 오래된 세션 삭제 로직
    pass
```

## API 문제

### 문제: API 인증 실패

#### 증상
- 401 Unauthorized 오류
- API 키가 유효하지 않음
- 인증 헤더 누락

#### 진단 단계

1. **API 키 확인**
```bash
# 환경 변수에서 API 키 확인
echo $API_KEY

# 요청 헤더 확인
curl -H "Authorization: Bearer $API_KEY" http://localhost:8000/api/health
```

2. **인증 로직 확인**
```python
# API 키 검증 로직 테스트
def test_api_key():
    api_key = os.getenv('API_KEY')
    if not api_key:
        print("API_KEY not set")
        return False
    
    # API 키 형식 확인
    if len(api_key) < 32:
        print("API_KEY too short")
        return False
    
    return True
```

#### 해결 방법

1. **API 키 재생성**
```python
import secrets

# 새 API 키 생성
new_api_key = secrets.token_urlsafe(32)
print(f"New API key: {new_api_key}")

# .env 파일 업데이트
with open('.env', 'a') as f:
    f.write(f"\nAPI_KEY={new_api_key}\n")
```

2. **인증 로직 수정**
```python
# API 키 검증 로직 개선
def verify_api_key(request):
    auth_header = request.headers.get('Authorization')
    if not auth_header:
        raise HTTPException(status_code=401, detail="Authorization header missing")
    
    if not auth_header.startswith('Bearer '):
        raise HTTPException(status_code=401, detail="Invalid authorization format")
    
    token = auth_header[7:]  # "Bearer " 제거
    if token != os.getenv('API_KEY'):
        raise HTTPException(status_code=401, detail="Invalid API key")
```

### 문제: API 응답 오류

#### 증상
- 500 Internal Server Error
- JSON 파싱 오류
- 타임아웃 오류

#### 진단 단계

1. **API 응답 확인**
```bash
# API 엔드포인트 테스트
curl -X POST http://localhost:8000/api/chat \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer $API_KEY" \
  -d '{"message": "test"}'
```

2. **서버 로그 확인**
```bash
# 실시간 로그 모니터링
tail -f logs/lawfirm.log | grep ERROR
```

#### 해결 방법

1. **오류 처리 개선**
```python
# 예외 처리 강화
@app.post("/api/chat")
async def chat(request: ChatRequest):
    try:
        result = await chat_service.process_message(
            request.message,
            session_id=request.session_id,
            user_id=request.user_id
        )
        return result
    except Exception as e:
        logger.error(f"Chat API error: {e}")
        return {
            "error": "Internal server error",
            "message": str(e),
            "status": "error"
        }
```

2. **입력 검증 강화**
```python
# Pydantic 모델로 입력 검증
class ChatRequest(BaseModel):
    message: str
    session_id: Optional[str] = None
    user_id: Optional[str] = None
    
    @validator('message')
    def validate_message(cls, v):
        if not v or len(v.strip()) == 0:
            raise ValueError('Message cannot be empty')
        if len(v) > 10000:
            raise ValueError('Message too long')
        return v.strip()
```

## UI 문제

### 문제: Gradio 인터페이스가 로드되지 않음

#### 증상
- 브라우저에서 페이지가 로드되지 않음
- JavaScript 오류
- CSS 스타일이 적용되지 않음

#### 진단 단계

1. **브라우저 콘솔 확인**
```javascript
// 브라우저 개발자 도구에서 확인
console.log("Gradio interface loaded");
```

2. **네트워크 요청 확인**
```bash
# 서버 응답 확인
curl -I http://localhost:7860
```

#### 해결 방법

1. **Gradio 버전 확인**
```bash
# Gradio 버전 확인
pip show gradio

# 최신 버전으로 업그레이드
pip install --upgrade gradio
```

2. **포트 충돌 해결**
```python
# 다른 포트 사용
iface.launch(server_port=7861, server_name="0.0.0.0")
```

3. **정적 파일 문제 해결**
```python
# 정적 파일 경로 확인
import os
static_path = os.path.join(os.path.dirname(__file__), 'static')
if os.path.exists(static_path):
    print(f"Static files found at: {static_path}")
else:
    print("Static files not found")
```

### 문제: UI 응답성 문제

#### 증상
- 버튼 클릭이 반응하지 않음
- 입력 필드가 비활성화됨
- 로딩 상태가 계속 표시됨

#### 해결 방법

1. **JavaScript 오류 확인**
```javascript
// 브라우저 콘솔에서 오류 확인
window.addEventListener('error', function(e) {
    console.error('JavaScript error:', e.error);
});
```

2. **Gradio 컴포넌트 상태 확인**
```python
# 컴포넌트 상태 확인
def check_component_state():
    try:
        # 컴포넌트가 정상적으로 초기화되었는지 확인
        if chatbot is None:
            print("Chatbot component not initialized")
        if msg is None:
            print("Message input not initialized")
    except Exception as e:
        print(f"Component error: {e}")
```

3. **이벤트 핸들러 확인**
```python
# 이벤트 핸들러가 정상적으로 등록되었는지 확인
def test_event_handlers():
    try:
        # 이벤트 핸들러 테스트
        test_message = "test message"
        result = respond(test_message, [], None)
        print(f"Event handler test result: {result}")
    except Exception as e:
        print(f"Event handler error: {e}")
```

## 로그 분석

### 로그 파일 위치

```bash
# 로그 파일 확인
ls -la logs/

# 주요 로그 파일
# - lawfirm.log: 메인 애플리케이션 로그
# - error.log: 오류 로그
# - access.log: 접근 로그
# - performance.log: 성능 로그
```

### 로그 분석 도구

1. **실시간 로그 모니터링**
```bash
# 실시간 로그 확인
tail -f logs/lawfirm.log

# 특정 키워드 필터링
tail -f logs/lawfirm.log | grep ERROR
tail -f logs/lawfirm.log | grep WARNING
```

2. **로그 통계 분석**
```bash
# 오류 발생 빈도
grep ERROR logs/lawfirm.log | wc -l

# 경고 발생 빈도
grep WARNING logs/lawfirm.log | wc -l

# 시간대별 로그 분석
grep "2024-12-20 14:" logs/lawfirm.log
```

3. **로그 패턴 분석**
```bash
# 가장 자주 발생하는 오류
grep ERROR logs/lawfirm.log | sort | uniq -c | sort -nr

# 특정 사용자의 활동
grep "user123" logs/lawfirm.log

# 성능 관련 로그
grep "processing_time" logs/lawfirm.log
```

### 로그 레벨 설정

```python
# 로깅 설정
import logging

# 로그 레벨 설정
logging.basicConfig(
    level=logging.DEBUG,  # DEBUG, INFO, WARNING, ERROR, CRITICAL
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('logs/lawfirm.log'),
        logging.StreamHandler()
    ]
)

# 특정 모듈의 로그 레벨 설정
logging.getLogger('transformers').setLevel(logging.WARNING)
logging.getLogger('torch').setLevel(logging.WARNING)
```

## 고급 문제 해결

### 성능 프로파일링

1. **Python 프로파일링**
```python
import cProfile
import pstats

# 프로파일링 실행
profiler = cProfile.Profile()
profiler.enable()

# 코드 실행
result = await chat_service.process_message("test message")

profiler.disable()

# 결과 분석
stats = pstats.Stats(profiler)
stats.sort_stats('cumulative')
stats.print_stats(10)  # 상위 10개 함수
```

2. **메모리 프로파일링**
```python
from memory_profiler import profile

@profile
def process_message(self, message: str):
    # 메시지 처리 로직
    pass
```

### 디버깅 도구

1. **디버거 사용**
```python
import pdb

def debug_function():
    # 디버그 포인트 설정
    pdb.set_trace()
    
    # 코드 실행
    result = some_function()
    return result
```

2. **로깅 강화**
```python
import logging

logger = logging.getLogger(__name__)

def detailed_logging():
    logger.debug("Function started")
    logger.info("Processing message")
    logger.warning("Potential issue detected")
    logger.error("Error occurred")
    logger.critical("Critical error")
```

### 시스템 모니터링

1. **시스템 리소스 모니터링**
```bash
# 시스템 모니터링 도구 설치
sudo apt install htop iotop nethogs

# 실시간 모니터링
htop          # CPU, 메모리 사용량
iotop         # 디스크 I/O
nethogs       # 네트워크 사용량
```

2. **애플리케이션 모니터링**
```python
# 애플리케이션 메트릭 수집
import psutil
import time

def collect_metrics():
    metrics = {
        'cpu_percent': psutil.cpu_percent(),
        'memory_percent': psutil.virtual_memory().percent,
        'disk_usage': psutil.disk_usage('/').percent,
        'timestamp': time.time()
    }
    return metrics
```

---

**문제 해결을 통해 LawFirmAI를 안정적으로 운영하세요!** 🔧
