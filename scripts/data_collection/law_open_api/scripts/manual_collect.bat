@echo off
REM 법령용어 수동 수집 배치 파일
REM LawFirmAI 프로젝트의 법령용어 수동 수집을 실행합니다.

echo ============================================================
echo 법령용어 수동 수집
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

REM 수집 모드 선택
echo 수집 모드를 선택하세요:
echo 1. 증분 수집 (incremental) - 기본값
echo 2. 전체 수집 (full)
echo.
set /p mode="모드 선택 (1 또는 2, Enter=1): "

if "%mode%"=="" set mode=1
if "%mode%"=="1" set collection_mode=incremental
if "%mode%"=="2" set collection_mode=full
if not defined collection_mode set collection_mode=incremental

echo 선택된 모드: %collection_mode%
echo.

REM Python 스크립트 실행
echo 수집 시작 중...
python scripts/data_collection/law_open_api/scripts/manual_collect_legal_terms.py --mode %collection_mode%

REM 실행 결과 확인
if %errorlevel% neq 0 (
    echo.
    echo ❌ 수집 실행 실패 (오류 코드: %errorlevel%)
) else (
    echo.
    echo ✅ 수집 정상 완료
)

echo.
echo ============================================================
echo 수집 완료
echo ============================================================
pause




