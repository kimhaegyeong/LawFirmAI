# 간단한 테스트 실행 스크립트 (PowerShell)
# Windows 버퍼 문제 해결 포함

$env:PYTHONUNBUFFERED = "1"
Set-Location $PSScriptRoot

Write-Host "========================================" -ForegroundColor Cyan
Write-Host "API 테스트 실행" -ForegroundColor Cyan
Write-Host "========================================" -ForegroundColor Cyan
Write-Host ""

# api/tests/pytest.ini를 사용하여 실행
pytest -v

$exitCode = $LASTEXITCODE

if ($exitCode -eq 0) {
    Write-Host ""
    Write-Host "========================================" -ForegroundColor Green
    Write-Host "테스트 성공!" -ForegroundColor Green
    Write-Host "========================================" -ForegroundColor Green
    exit 0
} else {
    Write-Host ""
    Write-Host "========================================" -ForegroundColor Red
    Write-Host "테스트 실패 (오류 코드: $exitCode)" -ForegroundColor Red
    Write-Host "========================================" -ForegroundColor Red
    exit $exitCode
}

