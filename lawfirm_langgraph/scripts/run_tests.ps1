# LangGraph 테스트 실행 스크립트 (PowerShell)
# 사용법: .\run_tests.ps1

$ErrorActionPreference = "Stop"

Write-Host "========================================"
Write-Host "LangGraph 테스트 실행"
Write-Host "========================================"
Write-Host ""

# 현재 디렉토리 확인
$scriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
Set-Location $scriptDir

Write-Host "작업 디렉토리: $(Get-Location)"
Write-Host ""

# 가상환경 확인
if (-not (Test-Path ".venv\Scripts\python.exe")) {
    Write-Host "오류: 가상환경이 없습니다."
    Write-Host "먼저 가상환경을 설정하세요:"
    Write-Host "  python -m venv .venv"
    Write-Host "  .venv\Scripts\Activate.ps1"
    Write-Host "  pip install -r requirements.txt"
    exit 1
}

Write-Host "가상환경 Python 경로: .venv\Scripts\python.exe"
Write-Host ""

# 테스트 실행
Write-Host "기본 설정 테스트 실행 중..."
Write-Host "----------------------------------------"
& .venv\Scripts\python.exe test_setup.py
$setupResult = $LASTEXITCODE

Write-Host ""
Write-Host "----------------------------------------"
Write-Host ""

Write-Host "워크플로우 종합 테스트 실행 중..."
Write-Host "----------------------------------------"
& .venv\Scripts\python.exe test_workflow.py
$workflowResult = $LASTEXITCODE

Write-Host ""
Write-Host "========================================"
Write-Host "테스트 결과 요약"
Write-Host "========================================"
Write-Host ""

if ($setupResult -eq 0) {
    Write-Host "✓ 기본 설정 테스트: 통과" -ForegroundColor Green
} else {
    Write-Host "✗ 기본 설정 테스트: 실패 (종료 코드: $setupResult)" -ForegroundColor Red
}

if ($workflowResult -eq 0) {
    Write-Host "✓ 워크플로우 테스트: 통과" -ForegroundColor Green
} else {
    Write-Host "✗ 워크플로우 테스트: 실패 (종료 코드: $workflowResult)" -ForegroundColor Red
}

Write-Host ""

if ($setupResult -eq 0 -and $workflowResult -eq 0) {
    Write-Host "모든 테스트 통과!" -ForegroundColor Green
    Write-Host ""
    Write-Host "다음 단계:"
    Write-Host "  langgraph dev  # LangGraph Studio 실행"
    exit 0
} else {
    Write-Host "일부 테스트 실패. 위의 오류를 확인하세요." -ForegroundColor Yellow
    exit 1
}
