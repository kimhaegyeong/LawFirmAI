@echo off
echo 헌재결정례 수집 시작...
echo.

REM 환경변수 확인
if "%LAW_OPEN_API_OC%"=="" (
    echo ❌ LAW_OPEN_API_OC 환경변수가 설정되지 않았습니다.
    echo 다음과 같이 설정해주세요:
    echo set LAW_OPEN_API_OC=your_email@example.com
    pause
    exit /b 1
)

echo ✅ OC 파라미터: %LAW_OPEN_API_OC%
echo.

REM 헌재결정례 수집 실행
python scripts/data_collection/constitutional/collect_constitutional_decisions.py ^
    --keyword "" ^
    --max-count 1000 ^
    --batch-size 100 ^
    --sort-order dasc ^
    --no-details false ^
    --no-database false ^
    --no-vectors false

echo.
echo 헌재결정례 수집 완료
pause
