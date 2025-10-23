@echo off
REM 법령용어 스케줄러 시작 배치 파일
REM LawFirmAI 프로젝트의 법령용어 주기적 수집 스케줄러를 시작합니다.

echo ============================================================
echo 법령용어 스케줄러 시작
echo ============================================================
echo 시작 시간: %date% %time%
echo.

REM 프로젝트 루트 디렉토리로 이동
cd /d %~dp0\..\..\..\..
echo 프로젝트 루트: %cd%

REM 환경변수 확인
if "%LAW_OPEN_API_OC%"=="" (
    echo ❌ LAW_OPEN_API_OC 환경변수가 설정되지 않았습니다.
    echo 다음과 같이 설정해주세요:
    echo set LAW_OPEN_API_OC=your_email@example.com
    echo.
    pause
    exit /b 1
)

echo ✅ API 키 확인: %LAW_OPEN_API_OC%
echo.

REM Python 스크립트 실행
echo 스케줄러 시작 중...
python scripts/data_collection/law_open_api/scripts/start_legal_term_scheduler.py

REM 실행 결과 확인
if %errorlevel% neq 0 (
    echo.
    echo ❌ 스케줄러 실행 실패 (오류 코드: %errorlevel%)
) else (
    echo.
    echo ✅ 스케줄러 정상 종료
)

echo.
echo ============================================================
echo 스케줄러 종료
echo ============================================================
pause




