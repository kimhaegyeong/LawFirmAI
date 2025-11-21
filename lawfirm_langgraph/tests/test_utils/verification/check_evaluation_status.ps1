# 평가 진행 상황 확인 스크립트

$logFile = "logs/evaluation_progress.log"
$resultFile = "logs/search_quality_evaluation_with_improvements.json"

Write-Host "`n============================================================" -ForegroundColor Cyan
Write-Host "검색 품질 평가 진행 상황" -ForegroundColor Cyan
Write-Host "============================================================`n" -ForegroundColor Cyan

# 로그 파일 확인
if (Test-Path $logFile) {
    Write-Host "📝 최근 로그 (마지막 20줄):" -ForegroundColor Yellow
    Write-Host "------------------------------------------------------------" -ForegroundColor Gray
    Get-Content $logFile -Tail 20 -ErrorAction SilentlyContinue
    Write-Host "------------------------------------------------------------`n" -ForegroundColor Gray
} else {
    Write-Host "⚠️  로그 파일이 아직 생성되지 않았습니다." -ForegroundColor Yellow
    Write-Host "   평가가 시작 중이거나 아직 시작되지 않았습니다.`n" -ForegroundColor Gray
}

# 결과 파일 확인
if (Test-Path $resultFile) {
    Write-Host "✅ 결과 파일이 생성되었습니다!" -ForegroundColor Green
    Write-Host "   파일: $resultFile`n" -ForegroundColor Gray
    
    try {
        $result = Get-Content $resultFile -Raw -Encoding UTF8 | ConvertFrom-Json
        
        Write-Host "📊 평가 결과 요약:" -ForegroundColor Cyan
        Write-Host "   - 총 쿼리 수: $($result.total_queries)" -ForegroundColor White
        Write-Host "   - 성공한 쿼리: $($result.successful_queries)" -ForegroundColor Green
        Write-Host "   - 실패한 쿼리: $($result.failed_queries)" -ForegroundColor $(if ($result.failed_queries -gt 0) { "Red" } else { "Green" })
        
        if ($result.average_metrics) {
            Write-Host "`n📈 평균 메트릭:" -ForegroundColor Cyan
            $result.average_metrics.PSObject.Properties | ForEach-Object {
                if ($_.Value -is [double] -or $_.Value -is [int]) {
                    $value = if ($_.Value -is [double]) { "{0:F4}" -f $_.Value } else { $_.Value }
                    Write-Host "   - $($_.Name): $value" -ForegroundColor White
                }
            }
        }
    } catch {
        Write-Host "⚠️  결과 파일을 읽는 중 오류 발생: $_" -ForegroundColor Yellow
    }
} else {
    Write-Host "⏳ 평가가 아직 진행 중입니다..." -ForegroundColor Yellow
    Write-Host "   결과 파일이 생성되면 자동으로 표시됩니다.`n" -ForegroundColor Gray
}

Write-Host "`n💡 팁: 이 스크립트를 주기적으로 실행하여 진행 상황을 확인하세요." -ForegroundColor Cyan
Write-Host "   예: while (`$true) { .\check_evaluation_status.ps1; Start-Sleep -Seconds 30 }`n" -ForegroundColor Gray

