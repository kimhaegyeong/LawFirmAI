# run_query_test.py 로그 파일 저장 규칙

## 개요

`run_query_test.py` 스크립트를 실행할 때 로그를 파일로 저장하여 분석할 수 있도록 하는 규칙입니다.

## 로그 파일 저장 방법

### 1. 자동 로그 파일 생성 (기본)

기본적으로 로그 파일은 자동으로 생성됩니다.

```bash
python lawfirm_langgraph/tests/scripts/run_query_test.py "계약 해지 사유에 대해 알려주세요"
```

**로그 파일 위치**: `logs/test/run_query_test_YYYYMMDD_HHMMSS.log`

예시:
- `logs/test/run_query_test_20251117_193000.log`

### 2. 환경 변수로 로그 파일 경로 지정

환경 변수 `TEST_LOG_FILE`을 사용하여 로그 파일 경로를 직접 지정할 수 있습니다.

**Windows PowerShell**:
```powershell
$env:TEST_LOG_FILE="logs/test/my_test.log"
python lawfirm_langgraph/tests/scripts/run_query_test.py "계약 해지 사유에 대해 알려주세요"
```

**Windows CMD**:
```cmd
set TEST_LOG_FILE=logs/test/my_test.log
python lawfirm_langgraph/tests/scripts/run_query_test.py "계약 해지 사유에 대해 알려주세요"
```

**Linux/Mac**:
```bash
export TEST_LOG_FILE="logs/test/my_test.log"
python lawfirm_langgraph/tests/scripts/run_query_test.py "계약 해지 사유에 대해 알려주세요"
```

### 3. 로그 디렉토리 변경

환경 변수 `TEST_LOG_DIR`을 사용하여 로그 디렉토리를 변경할 수 있습니다.

**Windows PowerShell**:
```powershell
$env:TEST_LOG_DIR="logs/custom_test"
python lawfirm_langgraph/tests/scripts/run_query_test.py "계약 해지 사유에 대해 알려주세요"
```

**기본값**: `logs/test`

### 4. 로그 레벨 설정

환경 변수 `TEST_LOG_LEVEL`을 사용하여 로그 레벨을 설정할 수 있습니다.

**사용 가능한 레벨**: `DEBUG`, `INFO`, `WARNING`, `ERROR`

**Windows PowerShell**:
```powershell
$env:TEST_LOG_LEVEL="DEBUG"
python lawfirm_langgraph/tests/scripts/run_query_test.py "계약 해지 사유에 대해 알려주세요"
```

**기본값**: `INFO`

### 5. 환경 변수로 테스트 쿼리 지정

환경 변수 `TEST_QUERY`를 사용하여 테스트 쿼리를 지정할 수 있습니다.

**Windows PowerShell**:
```powershell
$env:TEST_QUERY="계약 해지 사유에 대해 알려주세요"
python lawfirm_langgraph/tests/scripts/run_query_test.py
```

**Linux/Mac**:
```bash
export TEST_QUERY="계약 해지 사유에 대해 알려주세요"
python lawfirm_langgraph/tests/scripts/run_query_test.py
```

**우선순위**: 환경 변수 `TEST_QUERY` > 명령줄 인자

## 로그 파일 형식

로그 파일은 다음 형식으로 저장됩니다:

```
2025-11-17 19:30:00 - lawfirm_langgraph.tests - INFO - 📝 로그 파일: logs/test/run_query_test_20251117_193000.log
2025-11-17 19:30:00 - lawfirm_langgraph.tests - INFO - ================================================================================
2025-11-17 19:30:00 - lawfirm_langgraph.tests - INFO - LangGraph 질의 테스트
2025-11-17 19:30:00 - lawfirm_langgraph.tests - INFO - ================================================================================
2025-11-17 19:30:00 - lawfirm_langgraph.tests - INFO - 
2025-11-17 19:30:00 - lawfirm_langgraph.tests - INFO - 📋 질의: 계약 해지 사유에 대해 알려주세요
...
```

**로그 파일 모드**: `mode='w'` (덮어쓰기 모드) - 같은 파일 경로를 지정하면 이전 로그가 덮어씌워집니다.

## 로그 분석 예시

### 1. 성능 메트릭 추출

```bash
# Windows PowerShell
Select-String -Pattern "PERFORMANCE|process_search_results_combined|expand_keywords" logs/test/run_query_test_*.log

# Linux/Mac
grep -E "PERFORMANCE|process_search_results_combined|expand_keywords" logs/test/run_query_test_*.log
```

### 2. Keyword Coverage 추출

```bash
# Windows PowerShell
Select-String -Pattern "Keyword Coverage" logs/test/run_query_test_*.log

# Linux/Mac
grep "Keyword Coverage" logs/test/run_query_test_*.log
```

### 3. 에러 로그 추출

```bash
# Windows PowerShell
Select-String -Pattern "ERROR|❌|⚠️" logs/test/run_query_test_*.log

# Linux/Mac
grep -E "ERROR|❌|⚠️" logs/test/run_query_test_*.log
```

### 4. 메타데이터 오타 확인

```bash
# Windows PowerShell
Select-String -Pattern "interpretation_id|interpretatiion_id|interpretattion_id|Normalized typo|Fixed typo" logs/test/run_query_test_*.log

# Linux/Mac
grep -E "interpretation_id|interpretatiion_id|interpretattion_id|Normalized typo|Fixed typo" logs/test/run_query_test_*.log
```

## 로그 파일 관리

### 로그 파일 자동 정리

오래된 로그 파일을 자동으로 정리하려면 다음 스크립트를 사용할 수 있습니다:

```python
# scripts/cleanup_test_logs.py
import os
from pathlib import Path
from datetime import datetime, timedelta

log_dir = Path("logs/test")
max_age_days = 7  # 7일 이상 된 로그 파일 삭제

if log_dir.exists():
    cutoff_date = datetime.now() - timedelta(days=max_age_days)
    for log_file in log_dir.glob("run_query_test_*.log"):
        file_time = datetime.fromtimestamp(log_file.stat().st_mtime)
        if file_time < cutoff_date:
            log_file.unlink()
            print(f"Deleted: {log_file}")
```

## 주의사항

1. **로그 파일 크기**: 로그 파일은 시간이 지날수록 커질 수 있으므로 정기적으로 정리하는 것이 좋습니다.

2. **디스크 공간**: 로그 파일이 많이 쌓이면 디스크 공간을 많이 사용할 수 있습니다.

3. **인코딩**: 로그 파일은 UTF-8 인코딩으로 저장되므로 한글도 정상적으로 표시됩니다.

4. **동시 실행**: 여러 테스트를 동시에 실행하면 각각 다른 타임스탬프를 가진 로그 파일이 생성됩니다.

5. **로그 파일 모드**: 로그 파일은 `mode='w'` (덮어쓰기 모드)로 열리므로, 같은 파일 경로를 지정하면 이전 로그가 덮어씌워집니다. 자동 생성 모드에서는 타임스탬프가 포함되어 덮어쓰기가 발생하지 않습니다.

6. **Windows PowerShell 호환성**: `SafeStreamHandler`를 사용하여 Windows PowerShell에서 발생할 수 있는 버퍼 분리 오류를 방지합니다.

## 예시: 전체 테스트 실행 및 로그 분석

```powershell
# 1. 테스트 실행 (로그 파일 자동 생성)
python lawfirm_langgraph/tests/scripts/run_query_test.py "계약 해지 사유에 대해 알려주세요"

# 2. 로그 파일 확인
Get-ChildItem logs/test/run_query_test_*.log | Sort-Object LastWriteTime -Descending | Select-Object -First 1

# 3. 성능 메트릭 추출
$latestLog = Get-ChildItem logs/test/run_query_test_*.log | Sort-Object LastWriteTime -Descending | Select-Object -First 1
Select-String -Pattern "PERFORMANCE|Keyword Coverage" $latestLog.FullName

# 4. 에러 확인
Select-String -Pattern "ERROR|❌" $latestLog.FullName
```

## 환경 변수 요약

| 환경 변수 | 설명 | 기본값 | 우선순위 |
|---------|------|--------|---------|
| `TEST_LOG_FILE` | 로그 파일 경로 (전체 경로) | 자동 생성 | 1순위 |
| `TEST_LOG_DIR` | 로그 디렉토리 (TEST_LOG_FILE이 없을 때 사용) | `logs/test` | 2순위 |
| `TEST_LOG_LEVEL` | 로그 레벨 | `INFO` | - |
| `TEST_QUERY` | 테스트 쿼리 | 명령줄 인자 사용 | 1순위 (명령줄 인자보다 우선) |

## 참고

- 로그 파일은 콘솔 출력과 동일한 내용을 포함합니다.
- 로그 파일은 UTF-8 인코딩으로 저장됩니다.
- 로그 파일은 테스트 실행 시작 시 생성되고, 테스트 완료 시까지 계속 기록됩니다.
- Windows PowerShell 호환성을 위해 `SafeStreamHandler`가 사용되어 버퍼 분리 오류를 방지합니다.
- 로그 파일 생성 실패 시 경고 메시지가 출력되지만, 테스트는 계속 진행됩니다 (콘솔 로그만 사용).

## 구현 세부사항

### 로깅 설정 함수

`setup_logging` 함수는 다음 기능을 제공합니다:

- **자동 로그 파일 생성**: 타임스탬프 기반 파일명 생성
- **환경 변수 지원**: `TEST_LOG_FILE`, `TEST_LOG_DIR`, `TEST_LOG_LEVEL`
- **SafeStreamHandler**: Windows PowerShell 호환성 보장
- **에러 처리**: 로그 파일 생성 실패 시에도 테스트 계속 진행

### 로그 파일 경로 우선순위

1. `TEST_LOG_FILE` 환경 변수 (전체 경로 지정)
2. `TEST_LOG_DIR` 환경 변수 + 자동 파일명 생성
3. 기본 디렉토리 (`logs/test`) + 자동 파일명 생성

