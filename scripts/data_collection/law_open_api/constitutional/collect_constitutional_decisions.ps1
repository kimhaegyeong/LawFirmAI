# 헌재결정례 수집 PowerShell 스크립트

Write-Host "헌재결정례 수집 시작..." -ForegroundColor Green
Write-Host ""

# 환경변수 확인
if (-not $env:LAW_OPEN_API_OC -or $env:LAW_OPEN_API_OC -eq "{OC}") {
    Write-Host "❌ LAW_OPEN_API_OC 환경변수가 설정되지 않았습니다." -ForegroundColor Red
    Write-Host "다음과 같이 설정해주세요:" -ForegroundColor Yellow
    Write-Host '$env:LAW_OPEN_API_OC="your_email@example.com"' -ForegroundColor Cyan
    Read-Host "Press Enter to exit"
    exit 1
}

Write-Host "✅ OC 파라미터: $env:LAW_OPEN_API_OC" -ForegroundColor Green
Write-Host ""

# 헌재결정례 수집 실행
try {
    python scripts/data_collection/constitutional/collect_constitutional_decisions.py `
        --keyword "" `
        --max-count 1000 `
        --batch-size 100 `
        --sort-order dasc `
        --no-details false `
        --no-database false `
        --no-vectors false
    
    Write-Host ""
    Write-Host "✅ 헌재결정례 수집 완료" -ForegroundColor Green
} catch {
    Write-Host ""
    Write-Host "❌ 헌재결정례 수집 실패: $($_.Exception.Message)" -ForegroundColor Red
}

Read-Host "Press Enter to exit"
