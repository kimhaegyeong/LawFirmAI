@echo off
REM 간단한 테스트 실행 배치 파일
REM Windows 버퍼 문제 해결 포함

set PYTHONUNBUFFERED=1
cd /d %~dp0

echo ========================================
echo API 테스트 실행
echo ========================================
echo.

REM api/tests/pytest.ini를 사용하여 실행
pytest -v

if %ERRORLEVEL% EQU 0 (
    echo.
    echo ========================================
    echo 테스트 성공!
    echo ========================================
    exit /b 0
) else (
    echo.
    echo ========================================
    echo 테스트 실패 (오류 코드: %ERRORLEVEL%)
    echo ========================================
    exit /b %ERRORLEVEL%
)

