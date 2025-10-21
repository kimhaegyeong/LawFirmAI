@echo off
REM LawFirmAI Production Startup Script for Windows

echo Starting LawFirmAI Production Application...

REM 환경 변수 설정
set PYTHONPATH=%PYTHONPATH%;%CD%
set GRADIO_SERVER_NAME=0.0.0.0
set GRADIO_SERVER_PORT=7860
set HF_HUB_DISABLE_SYMLINKS_WARNING=1

REM 로그 디렉토리 생성
if not exist logs mkdir logs

REM 프로덕션 앱 실행
echo Launching production interface...
python gradio/app_final_production.py

echo LawFirmAI Production Application started successfully!
echo Access the application at: http://localhost:7860
pause
