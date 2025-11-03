# Scripts 가상환경 활성화 스크립트 (PowerShell)

$scriptPath = Split-Path -Parent $MyInvocation.MyCommand.Path
$venvPath = Join-Path $scriptPath ".venv"

if (-not (Test-Path $venvPath)) {
    Write-Host "가상환경이 없습니다. 생성 중..." -ForegroundColor Yellow
    python -m venv $venvPath

    if (-not (Test-Path $venvPath)) {
        Write-Host "가상환경 생성 실패!" -ForegroundColor Red
        exit 1
    }

    Write-Host "가상환경 생성 완료. 의존성 설치 중..." -ForegroundColor Yellow
    & "$venvPath\Scripts\Activate.ps1"
    pip install --upgrade pip
    pip install -r "$scriptPath\requirements.txt"
    Write-Host "의존성 설치 완료!" -ForegroundColor Green
} else {
    Write-Host "가상환경 활성화 중..." -ForegroundColor Green
    & "$venvPath\Scripts\Activate.ps1"
    Write-Host "Scripts 가상환경이 활성화되었습니다." -ForegroundColor Green
}
