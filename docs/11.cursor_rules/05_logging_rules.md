# 로깅 규칙 (CRITICAL)

## 1. 로거 생성 방법

### 기본 패턴 (권장)
```python
import logging

# 모듈 레벨에서 로거 생성
logger = logging.getLogger(__name__)
```

### 유틸리티 함수 사용 (선택)
```python
from lawfirm_langgraph.core.utils.logger import get_logger

# 환경 변수 LOG_LEVEL을 읽어서 자동 설정
logger = get_logger(__name__)
```

### 클래스 내부에서 로거 사용
```python
class MyService:
    def __init__(self, logger=None):
        # 로거를 파라미터로 받거나 새로 생성
        self.logger = logger or logging.getLogger(__name__)
    
    def process(self):
        self.logger.info("Processing started")
        try:
            # 작업 수행
            self.logger.debug("Detailed processing info")
        except Exception as e:
            self.logger.error(f"Error occurred: {e}", exc_info=True)
```

## 2. 환경 변수로 로그 레벨 제어

환경 변수 `LOG_LEVEL`로 로그 레벨을 설정합니다:

```bash
# Windows PowerShell
$env:LOG_LEVEL="DEBUG"

# Linux/Mac
export LOG_LEVEL=DEBUG
```

**지원 레벨:**
- `CRITICAL` - 가장 심각한 오류만
- `ERROR` - 오류 메시지
- `WARNING` - 경고 메시지
- `INFO` - 일반 정보 (기본값)
- `DEBUG` - 상세 디버깅 정보

**기본값**: `INFO`

## 3. 로그 레벨 사용법

```python
logger.debug("상세 디버깅 정보")
logger.info("일반 정보 메시지")
logger.warning("경고 메시지")
logger.error("오류 메시지")
logger.critical("심각한 오류 메시지")
```

## 4. Windows 환경 주의사항 (CRITICAL)

### 이모지 사용 금지
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

**환경 변수 설정 (필요 시):**
```python
# 모든 Python 파일 상단에 추가
import os
import sys

# 인코딩 설정 (최우선)
os.environ['PYTHONIOENCODING'] = 'utf-8'
if sys.platform == 'win32':
    os.environ['PYTHONLEGACYWINDOWSSTDIO'] = 'utf-8'
```

**안전한 콘솔 출력:**
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

## 5. 예외 로깅

```python
try:
    # 작업 수행
    pass
except Exception as e:
    # 스택 트레이스 포함
    logger.error(f"Error occurred: {e}", exc_info=True)
    
    # 또는
    logger.exception("Error occurred")  # 자동으로 exc_info=True
```

## 6. 로그 포맷

**기본 포맷:**
```
%(asctime)s - %(name)s - %(levelname)s - %(message)s
```

**예시:**
```
2024-01-15 10:30:45 - lawfirm_langgraph.core.workflow - INFO - Workflow initialized
```

## 7. 실제 사용 예시

```python
# lawfirm_langgraph/core/workflow/legal_workflow_enhanced.py
import logging

logger = logging.getLogger(__name__)

class LegalWorkflowEnhanced:
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.logger.info("LegalWorkflowEnhanced initialized")
    
    def process(self):
        self.logger.debug("Processing started")
        self.logger.info("Processing completed")
```

## 8. 외부 라이브러리 로깅 제어

프로젝트는 외부 라이브러리 로깅을 자동으로 비활성화합니다:
- `faiss`, `sentence_transformers`, `transformers`, `torch`
- `numpy`, `scipy`, `sklearn`
- `requests`, `urllib3`, `httpx`
- 기타 ML 라이브러리

이들은 `CRITICAL` 레벨로 설정되어 출력되지 않습니다.

## 9. 로깅 규칙 요약

1. **로거 생성**: `logger = logging.getLogger(__name__)`
2. **로그 레벨**: 환경 변수 `LOG_LEVEL`로 제어 (기본값: INFO)
3. **Windows**: 이모지 사용 금지, 인코딩 설정 필요 시 추가
4. **예외 처리**: `logger.error(..., exc_info=True)` 또는 `logger.exception()`
5. **클래스**: `self.logger = logger or logging.getLogger(__name__)` 패턴 사용

