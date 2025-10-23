@echo off
REM 법령용어 수집 상태 모니터링 배치 파일
REM LawFirmAI 프로젝트의 법령용어 수집 상태를 모니터링합니다.

echo ============================================================
echo 법령용어 수집 상태 모니터링
echo ============================================================
echo 모니터링 시간: %date% %time%
echo.

REM 프로젝트 루트 디렉토리로 이동
cd /d %~dp0\..\..\..\..
echo 프로젝트 루트: %cd%

REM Python 스크립트 실행
echo 상태 모니터링 중...
python scripts/data_collection/law_open_api/scripts/monitor_collection_status.py

REM 실행 결과 확인
if %errorlevel% neq 0 (
    echo.
    echo ❌ 모니터링 실행 실패 (오류 코드: %errorlevel%)
) else (
    echo.
    echo ✅ 모니터링 정상 완료
)

echo.
echo ============================================================
echo 모니터링 완료
echo ============================================================
pause




