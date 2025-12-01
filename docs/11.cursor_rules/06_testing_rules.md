# 테스트 규칙

## 0. pytest 실행 규칙 (CRITICAL)

**Windows 환경에서 pytest 실행 시 반드시 다음 옵션을 사용합니다:**

```bash
pytest -s --capture=tee-sys
```

또는 특정 테스트 파일/클래스/메서드 실행:

```bash
pytest -s --capture=tee-sys tests/unit/services/test_unified_prompt_manager.py
pytest -s --capture=tee-sys tests/unit/services/test_unified_prompt_manager.py::TestCleanContent
pytest -s --capture=tee-sys tests/unit/services/test_unified_prompt_manager.py::TestCleanContent::test_clean_content_removes_json_metadata
```

### 옵션 설명
- `-s`: 출력 캡처를 비활성화하여 print 문과 로그가 즉시 표시됩니다
- `--capture=tee-sys`: 출력을 캡처하면서 동시에 터미널에도 표시합니다 (Windows 버퍼 문제 해결)

### Windows 환경에서의 문제
Windows 환경에서 pytest를 실행할 때 `ValueError: underlying buffer has been detached` 오류가 발생할 수 있습니다. 이는 pytest의 출력 캡처 메커니즘과 Windows의 버퍼 처리 방식 간의 호환성 문제입니다.

### 해결 방법
1. **권장 방법**: `-s --capture=tee-sys` 옵션 사용
2. **대안**: `--capture=no` 옵션 사용 (출력 캡처 완전 비활성화)

### 예시
```bash
# 전체 테스트 실행
cd lawfirm_langgraph
pytest -s --capture=tee-sys

# 특정 디렉토리 테스트 실행
pytest -s --capture=tee-sys tests/unit/services/

# 특정 테스트 파일 실행
pytest -s --capture=tee-sys tests/unit/services/test_unified_prompt_manager.py -v

# 특정 테스트 클래스 실행
pytest -s --capture=tee-sys tests/unit/services/test_unified_prompt_manager.py::TestCleanContent -v
```

## 1. 단위 테스트
```python
import pytest
from unittest.mock import Mock, patch
from lawfirm_langgraph.core.workflow.state.state_definitions import LegalWorkflowState

class TestWorkflowNode:
    """워크플로우 노드 테스트 클래스"""
    
    def setup_method(self):
        """테스트 설정"""
        self.workflow = EnhancedLegalQuestionWorkflow(config)
    
    def test_node_execution(self):
        """노드 실행 테스트"""
        state: LegalWorkflowState = {"query": "테스트 질문"}
        result_state = self.workflow.classify_query_and_complexity(state)
        assert "query_type" in result_state
    
    def test_empty_input(self):
        """빈 입력 처리 테스트"""
        state: LegalWorkflowState = {"query": ""}
        with pytest.raises(ValueError):
            self.workflow.classify_query_and_complexity(state)
```

## 2. 통합 테스트
```python
# lawfirm_langgraph/tests/integration/test_api.py
import pytest
from fastapi.testclient import TestClient
from api.main import app

client = TestClient(app)

def test_chat_endpoint():
    """채팅 엔드포인트 테스트"""
    response = client.post(
        "/api/chat",
        json={"message": "계약서 검토 요청"}
    )
    assert response.status_code == 200
    assert "response" in response.json()
```

## 3. LangGraph 질의 테스트 실행 규칙 (CRITICAL)

**`run_query_test.py` 실행 시 반드시 가상환경을 활성화하고 실행합니다.**

### Windows PowerShell 실행 방법

```powershell
# 1. 프로젝트 루트로 이동
cd D:\project\LawFirmAI\LawFirmAI

# 2. 가상환경 활성화 (api\venv 사용)
.\api\venv\Scripts\Activate.ps1

# 3. 테스트 실행
python lawfirm_langgraph/tests/runners/run_query_test.py "질의 내용"

# 예시
python lawfirm_langgraph/tests/runners/run_query_test.py "민법 제543조"
```

### Windows CMD 실행 방법

```cmd
# 1. 프로젝트 루트로 이동
cd D:\project\LawFirmAI\LawFirmAI

# 2. 가상환경 활성화
api\venv\Scripts\activate.bat

# 3. 테스트 실행
python lawfirm_langgraph/tests/runners/run_query_test.py "질의 내용"
```

### Linux/Mac 실행 방법

```bash
# 1. 프로젝트 루트로 이동
cd /path/to/LawFirmAI

# 2. 가상환경 활성화
source api/venv/bin/activate

# 3. 테스트 실행
python lawfirm_langgraph/tests/runners/run_query_test.py "질의 내용"
```

### 환경 변수 사용

```powershell
# 환경 변수로 질의 지정
$env:TEST_QUERY="민법 제543조"
python lawfirm_langgraph/tests/runners/run_query_test.py

# 로그 레벨 설정
$env:LOG_LEVEL="DEBUG"
python lawfirm_langgraph/tests/runners/run_query_test.py "질의 내용"
```

### 중요 사항

1. **가상환경 활성화 필수**: `run_query_test.py` 실행 전 반드시 가상환경을 활성화해야 합니다.
2. **가상환경 위치**: 프로젝트의 가상환경은 `api\venv` (Windows) 또는 `api/venv` (Linux/Mac)에 위치합니다.
3. **PostgreSQL 연결 확인**: 테스트 실행 전 PostgreSQL 데이터베이스 연결이 설정되어 있어야 합니다.
4. **로그 파일**: 테스트 실행 시 `logs/langgraph/test_langgraph_query_YYYYMMDD_HHMMSS.log` 파일에 로그가 저장됩니다.

### 실행 예시

```powershell
# 기본 질의로 테스트
.\api\venv\Scripts\Activate.ps1
python lawfirm_langgraph/tests/runners/run_query_test.py

# 특정 질의로 테스트
.\api\venv\Scripts\Activate.ps1
python lawfirm_langgraph/tests/runners/run_query_test.py "계약 해지 사유에 대해 알려주세요"

# 디버그 모드로 테스트
.\api\venv\Scripts\Activate.ps1
$env:LOG_LEVEL="DEBUG"
python lawfirm_langgraph/tests/runners/run_query_test.py "민법 제543조"
```

## 4. 테스트 로그 파일 검증 규칙 (CRITICAL)

**`run_query_test.py` 실행 시 생성되는 로그 파일을 사용하여 테스트 결과를 검증합니다.**

### 로그 파일 위치

- **기본 위치**: `logs/langgraph/test_langgraph_query_YYYYMMDD_HHMMSS.log`
- **환경 변수로 변경 가능**:
  - `TEST_LOG_DIR`: 로그 디렉토리 경로
  - `TEST_LOG_FILE`: 로그 파일 전체 경로

### 로그 파일 자동 생성

`run_query_test.py` 실행 시 자동으로 로그 파일이 생성됩니다:

```powershell
# 로그 파일은 자동으로 생성됨
.\api\venv\Scripts\Activate.ps1
python lawfirm_langgraph/tests/runners/run_query_test.py "계약 해지 사유에 대해 알려주세요"

# 실행 후 콘솔에 로그 파일 경로가 표시됨
# 예: "로그 파일: D:\project\LawFirmAI\LawFirmAI\logs\langgraph\test_langgraph_query_20251127_093000.log"
```

### 로그 파일 검증 방법

#### Windows PowerShell

```powershell
# 1. 최신 로그 파일 찾기
$latestLog = Get-ChildItem -Path "logs\langgraph\test_langgraph_query_*.log" | Sort-Object LastWriteTime -Descending | Select-Object -First 1

# 2. MERGE EXPANDED 메시지 확인
Select-String -Path $latestLog.FullName -Pattern "MERGE EXPANDED|Found.*query sources|Consolidation" -Context 2,2

# 3. MULTI-QUERY 메시지 확인
Select-String -Path $latestLog.FullName -Pattern "MULTI-QUERY|Direct search completed" -Context 1,1

# 4. process_search_results_combined 실행 확인
Select-String -Path $latestLog.FullName -Pattern "process_search_results_combined|Process Search Results Combined" -Context 1,1

# 5. 검색 결과 확인
Select-String -Path $latestLog.FullName -Pattern "📥.*SEARCH RESULTS|semantic_results|keyword_results" -Context 1,1

# 6. 에러 확인
Select-String -Path $latestLog.FullName -Pattern "ERROR|❌|⚠️.*EARLY EXIT" -Context 1,1
```

#### Linux/Mac

```bash
# 1. 최신 로그 파일 찾기
LATEST_LOG=$(ls -t logs/langgraph/test_langgraph_query_*.log | head -1)

# 2. MERGE EXPANDED 메시지 확인
grep -E "MERGE EXPANDED|Found.*query sources|Consolidation" "$LATEST_LOG" -A 2 -B 2

# 3. MULTI-QUERY 메시지 확인
grep -E "MULTI-QUERY|Direct search completed" "$LATEST_LOG" -A 1 -B 1

# 4. process_search_results_combined 실행 확인
grep -E "process_search_results_combined|Process Search Results Combined" "$LATEST_LOG" -A 1 -B 1

# 5. 검색 결과 확인
grep -E "📥.*SEARCH RESULTS|semantic_results|keyword_results" "$LATEST_LOG" -A 1 -B 1

# 6. 에러 확인
grep -E "ERROR|❌|⚠️.*EARLY EXIT" "$LATEST_LOG" -A 1 -B 1
```

### 주요 검증 항목

#### 1. 확장된 쿼리 결과 병합 (MERGE EXPANDED)

```powershell
# MERGE EXPANDED 메시지 확인
Select-String -Path $latestLog.FullName -Pattern "MERGE EXPANDED" -Context 3,3

# 예상 출력:
# 🔄 [MERGE EXPANDED] Found 3 query sources: {'original': 1, 'sub_query_1': 2, 'sub_query_2': 1}
# 🔄 [MERGE EXPANDED] Consolidation: 4 → 2 (removed 2 duplicates, sources: 3)
```

#### 2. Multi-Query 검색 실행 확인

```powershell
# MULTI-QUERY 메시지 확인
Select-String -Path $latestLog.FullName -Pattern "MULTI-QUERY.*Direct search completed" -Context 2,2

# 예상 출력:
# ✅ [MULTI-QUERY] Generated 3 queries (original + 2 variations)
# ✅ [MULTI-QUERY] Direct search completed, 15 docs
```

#### 3. process_search_results_combined 실행 확인

```powershell
# process_search_results_combined 실행 확인
Select-String -Path $latestLog.FullName -Pattern "process_search_results_combined|Process Search Results Combined" -Context 1,1

# 예상 출력:
# [10] 🔄 실행 중: process_search_results_combined
#       → Process Search Results Combined
```

#### 4. 검색 결과 입력 데이터 확인

```powershell
# 검색 결과 입력 데이터 확인
Select-String -Path $latestLog.FullName -Pattern "📥.*SEARCH RESULTS.*최종 입력 데이터" -Context 0,0

# 예상 출력:
# 📥 [SEARCH RESULTS] 최종 입력 데이터 - semantic: 15, keyword: 0, semantic_count: 15, keyword_count: 0
```

#### 5. sub_query 필드 확인

```powershell
# sub_query 필드가 있는지 확인
Select-String -Path $latestLog.FullName -Pattern "sub_query|multi_query_source|source_query" -Context 0,0

# 또는 디버그 로그에서 확인
Select-String -Path $latestLog.FullName -Pattern "Found expanded query results|Multi-query results found" -Context 0,0
```

### 통합 검증 스크립트 예시

```powershell
# test_validation.ps1
$latestLog = Get-ChildItem -Path "logs\langgraph\test_langgraph_query_*.log" | Sort-Object LastWriteTime -Descending | Select-Object -First 1

Write-Host "=== 테스트 로그 검증 ===" -ForegroundColor Cyan
Write-Host "로그 파일: $($latestLog.FullName)" -ForegroundColor Yellow

# 1. MERGE EXPANDED 확인
Write-Host "`n1. MERGE EXPANDED 확인:" -ForegroundColor Green
$mergeExpanded = Select-String -Path $latestLog.FullName -Pattern "MERGE EXPANDED" -Context 1,1
if ($mergeExpanded) {
    Write-Host "   ✅ MERGE EXPANDED 메시지 발견" -ForegroundColor Green
    $mergeExpanded | ForEach-Object { Write-Host "   $_" }
} else {
    Write-Host "   ❌ MERGE EXPANDED 메시지 없음" -ForegroundColor Red
}

# 2. MULTI-QUERY 확인
Write-Host "`n2. MULTI-QUERY 확인:" -ForegroundColor Green
$multiQuery = Select-String -Path $latestLog.FullName -Pattern "MULTI-QUERY.*Direct search completed" -Context 1,1
if ($multiQuery) {
    Write-Host "   ✅ MULTI-QUERY 실행 확인" -ForegroundColor Green
    $multiQuery | ForEach-Object { Write-Host "   $_" }
} else {
    Write-Host "   ❌ MULTI-QUERY 실행 없음" -ForegroundColor Red
}

# 3. process_search_results_combined 확인
Write-Host "`n3. process_search_results_combined 실행 확인:" -ForegroundColor Green
$processResults = Select-String -Path $latestLog.FullName -Pattern "process_search_results_combined" -Context 1,1
if ($processResults) {
    Write-Host "   ✅ process_search_results_combined 실행 확인" -ForegroundColor Green
    $processResults | ForEach-Object { Write-Host "   $_" }
} else {
    Write-Host "   ❌ process_search_results_combined 실행 없음" -ForegroundColor Red
}

# 4. 에러 확인
Write-Host "`n4. 에러 확인:" -ForegroundColor Green
$errors = Select-String -Path $latestLog.FullName -Pattern "ERROR|❌" -Context 1,1
if ($errors) {
    Write-Host "   ⚠️  에러 발견:" -ForegroundColor Yellow
    $errors | ForEach-Object { Write-Host "   $_" }
} else {
    Write-Host "   ✅ 에러 없음" -ForegroundColor Green
}
```

### 환경 변수 설정

```powershell
# 로그 디렉토리 지정
$env:TEST_LOG_DIR="logs/custom_test"
python lawfirm_langgraph/tests/runners/run_query_test.py "질의 내용"

# 로그 파일 경로 직접 지정
$env:TEST_LOG_FILE="logs/custom_test/my_test.log"
python lawfirm_langgraph/tests/runners/run_query_test.py "질의 내용"

# 로그 레벨 설정
$env:LOG_LEVEL="DEBUG"
python lawfirm_langgraph/tests/runners/run_query_test.py "질의 내용"
```

### 중요 사항

1. **로그 파일 자동 생성**: `run_query_test.py` 실행 시 자동으로 로그 파일이 생성됩니다.
2. **로그 파일 경로 확인**: 테스트 실행 시 콘솔에 로그 파일 경로가 표시됩니다.
3. **검증 우선순위**: 
   - MERGE EXPANDED 메시지 확인 (확장된 쿼리 결과 병합)
   - MULTI-QUERY 실행 확인 (멀티 쿼리 검색)
   - process_search_results_combined 실행 확인 (결과 처리)
   - 에러 메시지 확인
4. **로그 레벨**: DEBUG 레벨로 실행하면 더 상세한 정보를 확인할 수 있습니다.

