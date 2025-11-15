# 자동 완료 스크립트 모니터링 (Windows PowerShell용)

$LogFile = "logs/auto_complete_dynamic_chunking.log"
$ErrorLog = "logs/auto_complete_dynamic_chunking_error.log"

Write-Host "============================================================" -ForegroundColor Cyan
Write-Host "자동 완료 스크립트 모니터링" -ForegroundColor Cyan
Write-Host "============================================================" -ForegroundColor Cyan
Write-Host ""

if (Test-Path $LogFile) {
    Write-Host "✓ 로그 파일: $LogFile" -ForegroundColor Green
    $logInfo = Get-Item $LogFile
    Write-Host "  크기: $([math]::Round($logInfo.Length / 1KB, 2)) KB"
    Write-Host "  수정 시간: $($logInfo.LastWriteTime)"
    Write-Host ""
    Write-Host "최근 로그 (마지막 20줄):" -ForegroundColor Yellow
    Write-Host "------------------------------------------------------------"
    Get-Content $LogFile -Tail 20 -ErrorAction SilentlyContinue
    Write-Host "------------------------------------------------------------"
} else {
    Write-Host "✗ 로그 파일 없음: $LogFile" -ForegroundColor Red
}

Write-Host ""

if (Test-Path $ErrorLog) {
    $errorInfo = Get-Item $ErrorLog
    if ($errorInfo.Length -gt 0) {
        Write-Host "⚠️  에러 로그에 내용이 있습니다:" -ForegroundColor Yellow
        Get-Content $ErrorLog -Tail 10 -ErrorAction SilentlyContinue
    } else {
        Write-Host "✓ 에러 로그 없음 (정상)" -ForegroundColor Green
    }
}

Write-Host ""
Write-Host "프로세스 확인:" -ForegroundColor Cyan
Get-Process python -ErrorAction SilentlyContinue | Where-Object {
    $_.CommandLine -like "*auto_complete*"
} | Select-Object Id, ProcessName, StartTime, @{Name="Runtime";Expression={(Get-Date) - $_.StartTime}}

Write-Host ""
Write-Host "재임베딩 진행 상황:" -ForegroundColor Cyan
python scripts/monitoring/monitor_re_embedding_progress.py --db data/lawfirm_v2.db --version-id 5

