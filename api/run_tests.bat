@echo off
REM API 테스트 실행 스크립트
REM 사용법: run_tests.bat [테스트 경로]

cd /d %~dp0

REM 가상환경 활성화
if exist "venv\Scripts\activate.bat" (
    call venv\Scripts\activate.bat
) else (
    echo Error: venv not found. Please create virtual environment first.
    echo Run: python -m venv venv
    exit /b 1
)

REM pytest 실행
if "%1"=="" (
    pytest tests/
) else (
    pytest %1
)

