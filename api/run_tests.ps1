# API 테스트 실행 스크립트 (PowerShell)
# 사용법: .\run_tests.ps1 [테스트 경로]

Set-Location $PSScriptRoot

# 가상환경 활성화
if (Test-Path "venv\Scripts\Activate.ps1") {
    .\venv\Scripts\Activate.ps1
} else {
    Write-Host "Error: venv not found. Please create virtual environment first." -ForegroundColor Red
    Write-Host "Run: python -m venv venv" -ForegroundColor Yellow
    exit 1
}

# pytest 실행
if ($args.Count -eq 0) {
    pytest tests/
} else {
    pytest $args[0]
}

