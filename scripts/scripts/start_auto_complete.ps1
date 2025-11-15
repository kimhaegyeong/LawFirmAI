# 자동 완료 스크립트 시작 (Windows PowerShell)

$ScriptsRoot = Split-Path $PSScriptRoot -Parent
$ScriptPath = Join-Path $ScriptsRoot "automation/auto_complete_re_embedding.py"
$LogDir = Join-Path (Split-Path $ScriptsRoot -Parent) "logs"

# 로그 디렉토리 생성
if (-not (Test-Path $LogDir)) {
    New-Item -ItemType Directory -Path $LogDir -Force | Out-Null
}

$LogFile = Join-Path $LogDir "auto_complete_dynamic_chunking.log"
$ErrorLog = Join-Path $LogDir "auto_complete_dynamic_chunking_error.log"

Write-Host "============================================================" -ForegroundColor Cyan
Write-Host "자동 완료 스크립트 시작" -ForegroundColor Cyan
Write-Host "============================================================" -ForegroundColor Cyan
Write-Host ""
Write-Host "스크립트: $ScriptPath" -ForegroundColor Yellow
Write-Host "로그 파일: $LogFile" -ForegroundColor Yellow
Write-Host "에러 로그: $ErrorLog" -ForegroundColor Yellow
Write-Host ""
Write-Host "재임베딩 완료를 주기적으로 확인하고," -ForegroundColor Green
Write-Host "완료되면 자동으로 FAISS 인덱스를 빌드합니다." -ForegroundColor Green
Write-Host ""
Write-Host "진행 상황 확인:" -ForegroundColor Cyan
Write-Host "  python scripts/monitoring/monitor_re_embedding_progress.py --db data/lawfirm_v2.db --version-id 5" -ForegroundColor Gray
Write-Host ""
Write-Host "로그 확인:" -ForegroundColor Cyan
Write-Host "  Get-Content logs/auto_complete_dynamic_chunking.log -Tail 20" -ForegroundColor Gray
Write-Host ""
Write-Host "============================================================" -ForegroundColor Cyan
Write-Host ""

# Python 스크립트 실행
$pythonArgs = @(
    $ScriptPath,
    "--db", "data/lawfirm_v2.db",
    "--version-id", "5",
    "--check-interval", "300",
    "--timeout", "14400"
)

Start-Process python -ArgumentList $pythonArgs -NoNewWindow -RedirectStandardOutput $LogFile -RedirectStandardError $ErrorLog

Write-Host "✓ 자동 완료 스크립트가 시작되었습니다." -ForegroundColor Green
Write-Host ""
Write-Host "프로세스 확인:" -ForegroundColor Cyan
Start-Sleep -Seconds 2
Get-Process python -ErrorAction SilentlyContinue | Where-Object {
    $_.CommandLine -like "*auto_complete*"
} | Select-Object Id, ProcessName, StartTime

