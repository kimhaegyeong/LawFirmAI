# Security Check Script
# GitHub Actions security-check.yml과 동일한 검사를 로컬에서 실행합니다.
#
# 사용법:
#   .\run_security_check.ps1 [옵션]
#
# 옵션:
#   -InstallDeps        npm 의존성을 설치한 후 검사를 실행합니다.
#                       기본적으로 의존성 설치를 건너뜁니다.
#   -CleanNodeModules   node_modules를 삭제한 후 의존성을 재설치합니다.
#   -SkipAudit          npm audit 검사를 건너뜁니다.
#   -Help               사용법을 표시합니다.
#
# 예시:
#   .\run_security_check.ps1                    기본 실행 (의존성 설치 건너뜀)
#   .\run_security_check.ps1 -InstallDeps       의존성 설치 포함 실행
#   .\run_security_check.ps1 -CleanNodeModules  node_modules 정리 후 실행
#   .\run_security_check.ps1 -Help              도움말 표시
#
# 참고:
#   - 의존성이 이미 설치되어 있는 경우 기본 실행으로 충분합니다.
#   - node_modules가 없는 경우 -InstallDeps 플래그를 사용하세요.

param(
    [switch]$InstallDeps = $false,
    [switch]$CleanNodeModules = $false,
    [switch]$SkipAudit = $false,
    [switch]$Help = $false
)

$ErrorActionPreference = "Stop"

# 도움말 표시
if ($Help) {
    Write-Host "Usage: .\run_security_check.ps1 [Options]" -ForegroundColor Cyan
    Write-Host ""
    Write-Host "Options:" -ForegroundColor Yellow
    Write-Host "  -InstallDeps        Install npm dependencies before running checks" -ForegroundColor White
    Write-Host "  -CleanNodeModules   Clean node_modules before installing dependencies" -ForegroundColor White
    Write-Host "  -SkipAudit          Skip npm audit check" -ForegroundColor White
    Write-Host "  -Help               Show this help message" -ForegroundColor White
    Write-Host ""
    Write-Host "By default, dependency installation is skipped." -ForegroundColor Gray
    Write-Host ""
    exit 0
}

Write-Host "========================================" -ForegroundColor Cyan
Write-Host "Security Check (Local)" -ForegroundColor Cyan
Write-Host "========================================" -ForegroundColor Cyan
Write-Host ""

# scripts 가상환경 경로 확인 및 활성화
$scriptsPath = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
$venvPath = Join-Path $scriptsPath "venv"
$venvActivateScript = Join-Path $venvPath "Scripts" "Activate.ps1"

if (Test-Path $venvActivateScript) {
    Write-Host "[Pre-check] Activating scripts virtual environment..." -ForegroundColor Yellow
    & $venvActivateScript
    if ($LASTEXITCODE -eq 0) {
        Write-Host "Virtual environment activated" -ForegroundColor Green
    } else {
        Write-Host "Warning: Failed to activate virtual environment (continuing...)" -ForegroundColor Yellow
    }
} else {
    Write-Host "Warning: scripts virtual environment not found at $venvActivateScript" -ForegroundColor Yellow
    Write-Host "Continuing without virtual environment..." -ForegroundColor Yellow
}

Write-Host ""

# Node.js 버전 확인
Write-Host "[1/3] Checking Node.js version..." -ForegroundColor Yellow
$nodeVersion = node --version
Write-Host "Node.js version: $nodeVersion" -ForegroundColor Green

if (-not $nodeVersion) {
    Write-Host "Error: Node.js is not installed" -ForegroundColor Red
    exit 1
}

# frontend 디렉토리로 이동
$frontendPath = Join-Path $PSScriptRoot ".." ".." "frontend"
if (-not (Test-Path $frontendPath)) {
    Write-Host "Error: frontend directory not found at $frontendPath" -ForegroundColor Red
    exit 1
}

Push-Location $frontendPath

try {
    # 의존성 설치 (옵션)
    if ($InstallDeps) {
        # node_modules 정리 (옵션)
        if ($CleanNodeModules) {
            Write-Host ""
            Write-Host "[Pre-install] Cleaning node_modules..." -ForegroundColor Yellow
            $nodeModulesPath = Join-Path $frontendPath "node_modules"
            if (Test-Path $nodeModulesPath) {
                Remove-Item -Recurse -Force $nodeModulesPath -ErrorAction SilentlyContinue
                Write-Host "node_modules cleaned" -ForegroundColor Green
            }
        }
        
        Write-Host ""
        Write-Host "[Optional] Installing dependencies..." -ForegroundColor Yellow
        Write-Host "Attempting npm ci..." -ForegroundColor Gray
        
        & npm ci
        $npmCiResult = $LASTEXITCODE
        
        if ($npmCiResult -ne 0) {
            Write-Host ""
            Write-Host "Warning: npm ci failed (exit code: $npmCiResult)" -ForegroundColor Yellow
            Write-Host "This might be due to file permission issues or locked files." -ForegroundColor Yellow
            Write-Host "Attempting fallback: npm install..." -ForegroundColor Yellow
            Write-Host ""
            
            & npm install
            $npmInstallResult = $LASTEXITCODE
            
            if ($npmInstallResult -ne 0) {
                Write-Host ""
                Write-Host "Error: npm install also failed (exit code: $npmInstallResult)" -ForegroundColor Red
                Write-Host ""
                Write-Host "Troubleshooting tips:" -ForegroundColor Yellow
                Write-Host "1. Close any programs that might be using node_modules files" -ForegroundColor White
                Write-Host "2. Try running as Administrator" -ForegroundColor White
                Write-Host "3. Delete node_modules and try again: Remove-Item -Recurse -Force node_modules" -ForegroundColor White
                Write-Host "4. Or use -CleanNodeModules option: .\scripts\checks\run_security_check.ps1 -InstallDeps -CleanNodeModules" -ForegroundColor White
                exit 1
            }
            Write-Host ""
            Write-Host "Dependencies installed successfully using npm install" -ForegroundColor Green
        } else {
            Write-Host "Dependencies installed successfully using npm ci" -ForegroundColor Green
        }
    } else {
        Write-Host ""
        Write-Host "[Skipped] Dependency installation (use -InstallDeps to install)" -ForegroundColor Yellow
        $nodeModulesPath = Join-Path $frontendPath "node_modules"
        if (-not (Test-Path $nodeModulesPath)) {
            Write-Host "Warning: node_modules directory not found. Some checks may fail." -ForegroundColor Yellow
            Write-Host "Run with -InstallDeps flag to install dependencies." -ForegroundColor Yellow
        }
    }

    # ESLint 실행
    Write-Host ""
    Write-Host "[2/3] Running ESLint..." -ForegroundColor Yellow
    
    # node_modules\.bin이 존재하는지 확인
    $eslintPath = Join-Path $frontendPath "node_modules" ".bin" "eslint.cmd"
    if (-not (Test-Path $eslintPath)) {
        Write-Host "Error: eslint not found in node_modules\.bin" -ForegroundColor Red
        Write-Host "Dependencies may not have been installed." -ForegroundColor Red
        Write-Host "Please run with -InstallDeps flag to install dependencies." -ForegroundColor Red
        exit 1
    }
    
    npm run lint
    if ($LASTEXITCODE -ne 0) {
        Write-Host "Error: ESLint check failed" -ForegroundColor Red
        exit 1
    }
    Write-Host "ESLint check passed" -ForegroundColor Green

    # TypeScript 타입 체크
    Write-Host ""
    Write-Host "[3/3] Running TypeScript type check..." -ForegroundColor Yellow
    npm run type-check
    if ($LASTEXITCODE -ne 0) {
        Write-Host "Error: TypeScript type check failed" -ForegroundColor Red
        exit 1
    }
    Write-Host "TypeScript type check passed" -ForegroundColor Green

    # npm audit 실행 (선택적)
    if (-not $SkipAudit) {
        Write-Host ""
        Write-Host "[Optional] Running npm audit..." -ForegroundColor Yellow
        npm audit --audit-level=moderate
        if ($LASTEXITCODE -ne 0) {
            Write-Host "Warning: npm audit found issues (non-blocking)" -ForegroundColor Yellow
        } else {
            Write-Host "npm audit passed" -ForegroundColor Green
        }
    }

    Write-Host ""
    Write-Host "========================================" -ForegroundColor Cyan
    Write-Host "Security checks completed successfully!" -ForegroundColor Green
    Write-Host "========================================" -ForegroundColor Cyan
}
catch {
    Write-Host ""
    Write-Host "Error: $($_.Exception.Message)" -ForegroundColor Red
    exit 1
}
finally {
    Pop-Location
}

