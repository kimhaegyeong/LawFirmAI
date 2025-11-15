@echo off
REM Run tests with coverage report
REM This script runs pytest with coverage and generates HTML report

echo [INFO] Running tests with coverage...
cd /d "%~dp0\.."

REM Run pytest with coverage
python -m pytest test/ --cov=api --cov-report=html --cov-report=term-missing --cov-report=xml

echo.
echo [INFO] Coverage report generated in htmlcov/index.html
echo [INFO] Open htmlcov/index.html in your browser to view the report

