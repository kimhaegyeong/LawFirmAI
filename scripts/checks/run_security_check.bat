@echo off
REM Security Check Script (Batch version)
REM GitHub Actions security-check.yml과 동일한 검사를 로컬에서 실행합니다.
REM
REM 사용법:
REM   run_security_check.bat [옵션]
REM
REM 옵션:
REM   --install-deps    npm 의존성을 설치한 후 검사를 실행합니다.
REM                     기본적으로 의존성 설치를 건너뜁니다.
REM   --help            사용법을 표시합니다.
REM
REM 예시:
REM   run_security_check.bat                    기본 실행 (의존성 설치 건너뜀)
REM   run_security_check.bat --install-deps     의존성 설치 포함 실행
REM   run_security_check.bat --help              도움말 표시
REM
REM 참고:
REM   - 의존성이 이미 설치되어 있는 경우 기본 실행으로 충분합니다.
REM   - node_modules가 없는 경우 --install-deps 플래그를 사용하세요.

setlocal enabledelayedexpansion

REM 명령줄 인자 확인
set "INSTALL_DEPS=0"
:parse_args
if "%~1"=="" goto args_done
if /i "%~1"=="--install-deps" (
    set "INSTALL_DEPS=1"
    shift
    goto parse_args
)
if /i "%~1"=="--help" (
    echo Usage: %~nx0 [--install-deps]
    echo.
    echo Options:
    echo   --install-deps    Install npm dependencies before running checks
    echo   --help            Show this help message
    echo.
    echo By default, dependency installation is skipped.
    echo.
    exit /b 0
)
echo Unknown option: %~1
echo Use --help for usage information.
exit /b 1
:args_done

echo ========================================
echo Security Check (Local)
echo ========================================
echo.

REM scripts 가상환경 경로 확인 및 활성화
pushd "%~dp0\.."
set "SCRIPTS_DIR=%CD%"
popd
set "VENV_PATH=%SCRIPTS_DIR%\venv"
set "VENV_ACTIVATE=%VENV_PATH%\Scripts\activate.bat"

if exist "%VENV_ACTIVATE%" (
    echo [Pre-check] Activating scripts virtual environment...
    call "%VENV_ACTIVATE%"
    if errorlevel 1 (
        echo Warning: Failed to activate virtual environment (continuing...)
    ) else (
        echo Virtual environment activated
    )
) else (
    echo Warning: scripts virtual environment not found at %VENV_ACTIVATE%
    echo Continuing without virtual environment...
)

echo.

REM Node.js 버전 확인
echo [1/3] Checking Node.js version...
node --version >nul 2>&1
if errorlevel 1 (
    echo Error: Node.js is not installed
    exit /b 1
)
node --version

REM frontend 디렉토리로 이동
cd /d "%~dp0\..\..\frontend"
if not exist "package.json" (
    echo Error: frontend directory not found
    exit /b 1
)

REM 의존성 설치 (옵션)
if %INSTALL_DEPS%==1 (
    echo.
    echo [Optional] Installing dependencies...
    echo Attempting npm ci...
    npm ci 2>&1
    set NPM_CI_RESULT=!errorlevel!
    if !NPM_CI_RESULT! neq 0 (
        echo.
        echo Warning: npm ci failed (exit code: !NPM_CI_RESULT!)
        echo This might be due to file permission issues or locked files.
        echo Attempting fallback: npm install...
        echo.
        npm install 2>&1
        set NPM_INSTALL_RESULT=!errorlevel!
        if !NPM_INSTALL_RESULT! neq 0 (
            echo.
            echo Error: npm install also failed (exit code: !NPM_INSTALL_RESULT!)
            echo.
            echo Troubleshooting tips:
            echo 1. Close any programs that might be using node_modules files
            echo 2. Try running as Administrator
            echo 3. Delete node_modules and try again: rmdir /s /q node_modules
            echo 4. Or use PowerShell script with -CleanNodeModules option
            exit /b 1
        )
        echo.
        echo Dependencies installed successfully using npm install
    ) else (
        echo Dependencies installed successfully using npm ci
    )
) else (
    echo.
    echo [Skipped] Dependency installation (use --install-deps to install)
    if not exist "node_modules" (
        echo Warning: node_modules directory not found. Some checks may fail.
        echo Run with --install-deps flag to install dependencies.
    )
)

REM ESLint 실행
echo.
echo [2/3] Running ESLint...
REM node_modules\.bin이 PATH에 있는지 확인
if not exist "node_modules\.bin\eslint.cmd" (
    echo Error: eslint not found in node_modules\.bin
    echo Dependencies may not have been installed.
    echo Please run with --install-deps flag to install dependencies.
    exit /b 1
)
npm run lint
if errorlevel 1 (
    echo Error: ESLint check failed
    exit /b 1
)
echo ESLint check passed

REM TypeScript 타입 체크
echo.
echo [3/3] Running TypeScript type check...
npm run type-check
if errorlevel 1 (
    echo Error: TypeScript type check failed
    exit /b 1
)
echo TypeScript type check passed

REM npm audit 실행
echo.
echo [Optional] Running npm audit...
npm audit --audit-level=moderate
if errorlevel 1 (
    echo Warning: npm audit found issues (non-blocking)
) else (
    echo npm audit passed
)

echo.
echo ========================================
echo Security checks completed successfully!
echo ========================================

