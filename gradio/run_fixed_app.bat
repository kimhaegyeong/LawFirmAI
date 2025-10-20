@echo off
echo Starting LawFirmAI - Fixed Gradio App...
echo.
echo Fixed issues:
echo - share-modal.js error resolved
echo - manifest.json 404 error resolved
echo - Static file serving improved
echo - Stable launch configuration
echo.

cd /d "%~dp0"
python run_fixed_app.py

pause
