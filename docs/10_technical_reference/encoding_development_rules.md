# LawFirmAI - 한국어 인코딩 개발 규칙

## 📋 문서 개요

본 문서는 LawFirmAI 프로젝트에서 Windows 환경의 CP949 인코딩으로 인한 한국어 콘솔 출력 문제를 해결하기 위한 개발 규칙을 정의합니다.

## 🚨 문제 상황

### Windows 환경에서 발생하는 인코딩 문제
- **기본 인코딩**: Windows 콘솔은 기본적으로 CP949 (EUC-KR) 인코딩 사용
- **문제점**: UTF-8로 작성된 한국어 텍스트가 콘솔에서 깨져서 표시됨
- **영향 범위**: 로깅, 콘솔 출력, subprocess 실행 결과, 파일 입출력

### 현재 프로젝트에서 확인된 문제 사례
```python
# ❌ 문제가 되는 코드 예시
print("법률 문서 분석 중...")  # 콘솔에서 깨짐
logger.info("벡터 저장소 로딩 완료")  # 로그에서 깨짐
subprocess.run(['tasklist'], text=True)  # 결과에서 한국어 깨짐
```

## ✅ 해결 방안

### 1. 환경 변수 설정 규칙

#### 시스템 레벨 설정 (권장)
```bash
# Windows 환경 변수 설정
set PYTHONIOENCODING=utf-8
set PYTHONLEGACYWINDOWSSTDIO=utf-8

# PowerShell에서 영구 설정
[Environment]::SetEnvironmentVariable("PYTHONIOENCODING", "utf-8", "User")
[Environment]::SetEnvironmentVariable("PYTHONLEGACYWINDOWSSTDIO", "utf-8", "User")
```

#### 프로젝트 레벨 설정
```python
# 모든 Python 파일 상단에 추가
import os
import sys

# 인코딩 설정 (최우선)
os.environ['PYTHONIOENCODING'] = 'utf-8'
os.environ['PYTHONLEGACYWINDOWSSTDIO'] = 'utf-8'

# Windows 콘솔 인코딩 설정
if sys.platform == 'win32':
    import codecs
    sys.stdout = codecs.getwriter('utf-8')(sys.stdout.detach())
    sys.stderr = codecs.getwriter('utf-8')(sys.stderr.detach())
```

### 2. 파일 인코딩 규칙

#### 모든 Python 파일 헤더 규칙
```python
# -*- coding: utf-8 -*-
"""
파일 설명
"""
import os
import sys

# 인코딩 설정 (Windows 호환성)
os.environ['PYTHONIOENCODING'] = 'utf-8'
if sys.platform == 'win32':
    os.environ['PYTHONLEGACYWINDOWSSTDIO'] = 'utf-8'
```

#### 파일 입출력 규칙
```python
# ✅ 올바른 파일 읽기/쓰기
def read_file_safe(file_path: str) -> str:
    """안전한 파일 읽기 (인코딩 처리)"""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            return f.read()
    except UnicodeDecodeError:
        # UTF-8 실패 시 CP949로 재시도
        try:
            with open(file_path, 'r', encoding='cp949') as f:
                return f.read()
        except UnicodeDecodeError:
            # 마지막으로 latin-1로 시도
            with open(file_path, 'r', encoding='latin-1') as f:
                return f.read()

def write_file_safe(file_path: str, content: str) -> bool:
    """안전한 파일 쓰기 (인코딩 처리)"""
    try:
        with open(file_path, 'w', encoding='utf-8') as f:
            f.write(content)
        return True
    except Exception as e:
        logger.error(f"파일 쓰기 실패 {file_path}: {e}")
        return False
```

### 3. 콘솔 출력 규칙

#### print 문 사용 규칙
```python
# ✅ 안전한 콘솔 출력
def safe_print(message: str):
    """안전한 콘솔 출력 (인코딩 처리)"""
    try:
        print(message)
    except UnicodeEncodeError:
        # 인코딩 오류 시 대체 출력
        print(message.encode('utf-8', errors='replace').decode('utf-8'))

# 사용 예시
safe_print("법률 문서 분석을 시작합니다.")
safe_print("벡터 저장소 로딩 완료")
```

#### 로깅 규칙
```python
import logging
import sys

# 로깅 설정 (인코딩 처리)
def setup_logging():
    """인코딩이 안전한 로깅 설정"""
    
    # 콘솔 핸들러 설정
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(logging.INFO)
    
    # 포맷터 설정 (이모지 제거)
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    console_handler.setFormatter(formatter)
    
    # 로거 설정
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    logger.addHandler(console_handler)
    
    return logger

# 사용 예시
logger = setup_logging()
logger.info("시스템 초기화 완료")
logger.warning("설정 파일을 찾을 수 없습니다")
logger.error("데이터베이스 연결 실패")
```

### 4. Subprocess 실행 규칙

#### Windows 호환 subprocess 규칙
```python
import subprocess
import sys

def run_command_safe(command: list, **kwargs) -> subprocess.CompletedProcess:
    """안전한 명령어 실행 (인코딩 처리)"""
    
    # Windows 환경에서 인코딩 설정
    if sys.platform == 'win32':
        kwargs.setdefault('encoding', 'cp949')
        kwargs.setdefault('errors', 'replace')
    else:
        kwargs.setdefault('encoding', 'utf-8')
    
    kwargs.setdefault('text', True)
    kwargs.setdefault('capture_output', True)
    
    try:
        result = subprocess.run(command, **kwargs)
        return result
    except Exception as e:
        logger.error(f"명령어 실행 실패 {command}: {e}")
        raise

# 사용 예시
def check_process_status(pid: int) -> bool:
    """프로세스 상태 확인 (Windows 호환)"""
    try:
        result = run_command_safe(['tasklist', '/FI', f'PID eq {pid}'])
        return 'python.exe' in result.stdout
    except Exception as e:
        logger.error(f"프로세스 상태 확인 실패: {e}")
        return False

def stop_process(pid: int) -> bool:
    """프로세스 종료 (Windows 호환)"""
    try:
        result = run_command_safe(['taskkill', '/PID', str(pid), '/F'])
        return result.returncode == 0
    except Exception as e:
        logger.error(f"프로세스 종료 실패: {e}")
        return False
```

### 5. 데이터베이스 인코딩 규칙

#### SQLite 데이터베이스 규칙
```python
import sqlite3
from contextlib import contextmanager

class SafeDatabaseManager:
    """인코딩이 안전한 데이터베이스 관리자"""
    
    def __init__(self, db_path: str):
        self.db_path = db_path
        self._setup_database()
    
    def _setup_database(self):
        """데이터베이스 설정 (인코딩 처리)"""
        with self.get_connection() as conn:
            # SQLite UTF-8 설정
            conn.execute("PRAGMA encoding = 'UTF-8'")
            conn.execute("PRAGMA foreign_keys = ON")
    
    @contextmanager
    def get_connection(self):
        """데이터베이스 연결 컨텍스트 매니저"""
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        try:
            yield conn
        finally:
            conn.close()
    
    def execute_query_safe(self, query: str, params: tuple = ()) -> list:
        """안전한 쿼리 실행 (인코딩 처리)"""
        try:
            with self.get_connection() as conn:
                cursor = conn.cursor()
                cursor.execute(query, params)
                return [dict(row) for row in cursor.fetchall()]
        except Exception as e:
            logger.error(f"쿼리 실행 실패: {e}")
            return []
```

### 6. JSON 파일 처리 규칙

#### JSON 파일 읽기/쓰기 규칙
```python
import json
from pathlib import Path

def load_json_safe(file_path: str) -> dict:
    """안전한 JSON 파일 로드 (인코딩 처리)"""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            return json.load(f)
    except UnicodeDecodeError:
        try:
            with open(file_path, 'r', encoding='cp949') as f:
                return json.load(f)
        except Exception as e:
            logger.error(f"JSON 파일 로드 실패 {file_path}: {e}")
            return {}

def save_json_safe(file_path: str, data: dict) -> bool:
    """안전한 JSON 파일 저장 (인코딩 처리)"""
    try:
        # 디렉토리 생성
        Path(file_path).parent.mkdir(parents=True, exist_ok=True)
        
        with open(file_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        return True
    except Exception as e:
        logger.error(f"JSON 파일 저장 실패 {file_path}: {e}")
        return False
```

## 🔧 개발 환경 설정

### 1. IDE 설정 (VS Code)
```json
// .vscode/settings.json
{
    "files.encoding": "utf8",
    "files.autoGuessEncoding": true,
    "terminal.integrated.shellArgs.windows": [
        "-NoExit",
        "-Command",
        "$env:PYTHONIOENCODING='utf-8'; $env:PYTHONLEGACYWINDOWSSTDIO='utf-8'"
    ],
    "python.defaultInterpreterPath": "python",
    "python.terminal.activateEnvironment": true
}
```

### 2. Git 설정
```bash
# Git 인코딩 설정
git config --global core.quotepath false
git config --global core.autocrlf true
git config --global i18n.commitencoding utf-8
git config --global i18n.logoutputencoding utf-8
```

### 3. PowerShell 프로필 설정
```powershell
# PowerShell 프로필에 추가
$env:PYTHONIOENCODING = "utf-8"
$env:PYTHONLEGACYWINDOWSSTDIO = "utf-8"

# 콘솔 인코딩 설정
[Console]::OutputEncoding = [System.Text.Encoding]::UTF8
```

## 📝 코드 템플릿

### 1. 기본 Python 파일 템플릿
```python
# -*- coding: utf-8 -*-
"""
파일 설명
"""
import os
import sys
import logging
from pathlib import Path

# 인코딩 설정 (Windows 호환성)
os.environ['PYTHONIOENCODING'] = 'utf-8'
if sys.platform == 'win32':
    os.environ['PYTHONLEGACYWINDOWSSTDIO'] = 'utf-8'

# 로깅 설정
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler('logs/app.log', encoding='utf-8')
    ]
)
logger = logging.getLogger(__name__)

def safe_print(message: str):
    """안전한 콘솔 출력"""
    try:
        print(message)
    except UnicodeEncodeError:
        print(message.encode('utf-8', errors='replace').decode('utf-8'))

def main():
    """메인 함수"""
    safe_print("애플리케이션 시작")
    logger.info("시스템 초기화 완료")

if __name__ == "__main__":
    main()
```

### 2. 서비스 클래스 템플릿
```python
# -*- coding: utf-8 -*-
"""
서비스 클래스 설명
"""
import os
import sys
import logging
from typing import Optional, Dict, Any

# 인코딩 설정
os.environ['PYTHONIOENCODING'] = 'utf-8'
if sys.platform == 'win32':
    os.environ['PYTHONLEGACYWINDOWSSTDIO'] = 'utf-8'

logger = logging.getLogger(__name__)

class ServiceClass:
    """서비스 클래스 설명"""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """초기화"""
        self.config = config or {}
        logger.info("서비스 초기화 완료")
    
    def process_data(self, data: str) -> str:
        """데이터 처리"""
        try:
            logger.info("데이터 처리 시작")
            # 처리 로직
            result = f"처리된 데이터: {data}"
            logger.info("데이터 처리 완료")
            return result
        except Exception as e:
            logger.error(f"데이터 처리 실패: {e}")
            raise
```

## 🚫 금지 사항

### 1. 인코딩 관련 금지 사항
```python
# ❌ 절대 사용하지 말 것
print("한글 텍스트")  # 인코딩 설정 없이 직접 출력
logger.info("한글 로그")  # 인코딩 설정 없이 로깅
subprocess.run(['command'], text=True)  # 인코딩 지정 없이 실행
open('file.txt', 'r')  # 인코딩 지정 없이 파일 열기

# ❌ 이모지 사용 금지 (Windows 콘솔에서 깨짐)
print("🚀 시작")
logger.info("✅ 완료")
print("❌ 오류")

# ❌ 하드코딩된 인코딩 변환
text.encode('cp949').decode('utf-8')  # 위험한 변환
```

### 2. Windows 특화 금지 사항
```python
# ❌ Unix 전용 코드
os.kill(pid, signal.SIGTERM)  # Windows에서 작동하지 않음
subprocess.run(['ps', 'aux'])  # Windows에서 사용할 수 없음

# ❌ 인코딩 무시
subprocess.run(['tasklist'], encoding=None)  # 기본 인코딩 사용
```

## ✅ 권장 사항

### 1. 개발 시작 전 체크리스트
- [ ] 환경 변수 설정 완료 (`PYTHONIOENCODING`, `PYTHONLEGACYWINDOWSSTDIO`)
- [ ] IDE 인코딩 설정 완료 (UTF-8)
- [ ] Git 인코딩 설정 완료
- [ ] PowerShell 프로필 설정 완료

### 2. 코드 작성 시 체크리스트
- [ ] 파일 상단에 `# -*- coding: utf-8 -*-` 추가
- [ ] 인코딩 설정 코드 추가
- [ ] 안전한 출력 함수 사용
- [ ] subprocess 실행 시 인코딩 지정
- [ ] 파일 입출력 시 인코딩 지정
- [ ] 이모지 사용 금지

### 3. 테스트 시 체크리스트
- [ ] 콘솔에서 한국어 출력 확인
- [ ] 로그 파일에서 한국어 확인
- [ ] subprocess 결과에서 한국어 확인
- [ ] 파일 저장/로드에서 한국어 확인

## 🔍 문제 해결 가이드

### 1. 콘솔에서 한국어가 깨질 때
```python
# 문제 진단
import sys
print(f"기본 인코딩: {sys.getdefaultencoding()}")
print(f"stdout 인코딩: {sys.stdout.encoding}")
print(f"stderr 인코딩: {sys.stderr.encoding}")

# 해결 방법
os.environ['PYTHONIOENCODING'] = 'utf-8'
if sys.platform == 'win32':
    os.environ['PYTHONLEGACYWINDOWSSTDIO'] = 'utf-8'
```

### 2. 로그에서 한국어가 깨질 때
```python
# 로깅 핸들러 재설정
import logging
import sys

# 기존 핸들러 제거
for handler in logging.root.handlers[:]:
    logging.root.removeHandler(handler)

# 새 핸들러 추가
handler = logging.StreamHandler(sys.stdout)
handler.setLevel(logging.INFO)
formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
handler.setFormatter(formatter)
logging.root.addHandler(handler)
```

### 3. subprocess 결과에서 한국어가 깨질 때
```python
# Windows에서 안전한 subprocess 실행
import subprocess
import sys

def safe_subprocess(command, **kwargs):
    if sys.platform == 'win32':
        kwargs.setdefault('encoding', 'cp949')
        kwargs.setdefault('errors', 'replace')
    else:
        kwargs.setdefault('encoding', 'utf-8')
    
    kwargs.setdefault('text', True)
    kwargs.setdefault('capture_output', True)
    
    return subprocess.run(command, **kwargs)
```

## 📊 성능 고려사항

### 1. 인코딩 변환 오버헤드 최소화
```python
# ✅ 효율적인 방법
text = "한국어 텍스트"
safe_print(text)  # 한 번만 변환

# ❌ 비효율적인 방법
print(text.encode('utf-8').decode('utf-8'))  # 불필요한 변환
```

### 2. 메모리 사용량 최적화
```python
# ✅ 스트림 처리
def process_large_file(file_path: str):
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            process_line(line)

# ❌ 전체 파일 로드
def process_large_file_bad(file_path: str):
    with open(file_path, 'r', encoding='utf-8') as f:
        content = f.read()  # 메모리 사용량 증가
        for line in content.split('\n'):
            process_line(line)
```

## 📞 지원 및 문의

인코딩 관련 문제가 발생하면 다음 순서로 해결하세요:

1. **환경 변수 확인**: `echo $env:PYTHONIOENCODING`
2. **IDE 설정 확인**: VS Code 인코딩 설정
3. **코드 검토**: 인코딩 규칙 준수 여부
4. **테스트 실행**: 간단한 한국어 출력 테스트
5. **문서 참조**: 이 문서의 문제 해결 가이드

---

**마지막 업데이트**: 2025-01-18  
**버전**: 1.0  
**상태**: 🟢 완전 구현 완료
