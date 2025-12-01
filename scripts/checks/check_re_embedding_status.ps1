# 재임베딩 상태 확인 스크립트
# Usage: .\scripts\check_re_embedding_status.ps1

$dbPath = "data/lawfirm_v2.db"
$versionId = 5

Write-Host "================================================================================`n재임베딩 상태 확인`n================================================================================`n" -ForegroundColor Cyan

# 진행 상황 확인
python scripts/monitoring/monitor_re_embedding_progress.py --db $dbPath --version-id $versionId

Write-Host "`n================================================================================`n성능 확인`n================================================================================`n" -ForegroundColor Cyan

# 성능 확인
python scripts/monitoring/check_re_embedding_performance.py --db $dbPath --version-id $versionId

Write-Host "`n================================================================================`n로그 확인 (마지막 20줄)`n================================================================================`n" -ForegroundColor Cyan

if (Test-Path "logs/re_embedding_optimized.log") {
    Get-Content "logs/re_embedding_optimized.log" -Tail 20
} else {
    Write-Host "로그 파일이 없습니다." -ForegroundColor Yellow
}

Write-Host "`n================================================================================`n프로세스 확인`n================================================================================`n" -ForegroundColor Cyan

$processes = Get-Process python -ErrorAction SilentlyContinue | Where-Object {
    $_.CommandLine -like "*re_embed_existing_data_optimized*"
}

if ($processes) {
    Write-Host "재임베딩 프로세스가 실행 중입니다:" -ForegroundColor Green
    $processes | Format-Table Id, CPU, WorkingSet -AutoSize
} else {
    Write-Host "재임베딩 프로세스가 실행 중이지 않습니다." -ForegroundColor Yellow
}

