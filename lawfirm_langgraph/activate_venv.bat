@echo off
REM LangGraph 가상환경 활성화 스크립트 (CMD)

set SCRIPT_DIR=%~dp0
set VENV_PATH=%SCRIPT_DIR%.venv

if not exist "%VENV_PATH%" (
    echo 가상환경이 없습니다. 생성 중...
    python -m venv "%VENV_PATH%"

    if not exist "%VENV_PATH%" (
        echo 가상환경 생성 실패!
        exit /b 1
    )

    echo 가상환경 생성 완료. 의존성 설치 중...
    call "%VENV_PATH%\Scripts\activate.bat"
    python -m pip install --upgrade pip
    pip install -r "%SCRIPT_DIR%requirements.txt"
    echo 의존성 설치 완료!
) else (
    echo 가상환경 활성화 중...
    call "%VENV_PATH%\Scripts\activate.bat"
    echo LangGraph 가상환경이 활성화되었습니다.
)
