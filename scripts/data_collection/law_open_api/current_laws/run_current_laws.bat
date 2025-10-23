@echo off
REM 현행법령 데이터 수집 배치 파일

echo 현행법령 데이터 수집 시스템
echo ================================

REM 환경변수 확인
if "%LAW_OPEN_API_OC%"=="" (
    echo ❌ LAW_OPEN_API_OC 환경변수가 설정되지 않았습니다.
    echo 다음과 같이 설정해주세요:
    echo set LAW_OPEN_API_OC=your_email@example.com
    pause
    exit /b 1
)

echo ✅ OC 파라미터: %LAW_OPEN_API_OC%

REM 메뉴 선택
echo.
echo 실행할 작업을 선택하세요:
echo 1. 전체 파이프라인 실행 (수집 → 데이터베이스 → 벡터)
echo 2. 데이터 수집만 실행
echo 3. 데이터베이스 업데이트만 실행
echo 4. 벡터 저장소 업데이트만 실행
echo 5. 연결 테스트
echo 6. 샘플 수집 (10개)
echo 0. 종료
echo.

set /p choice="선택 (0-6): "

if "%choice%"=="1" (
    echo 전체 파이프라인 실행 중...
    python scripts/data_collection/law_open_api/current_laws/run_pipeline.py --all
) else if "%choice%"=="2" (
    echo 데이터 수집 실행 중...
    python scripts/data_collection/law_open_api/current_laws/collect_current_laws.py
) else if "%choice%"=="3" (
    echo 데이터베이스 업데이트 실행 중...
    python scripts/data_collection/law_open_api/current_laws/update_database.py
) else if "%choice%"=="4" (
    echo 벡터 저장소 업데이트 실행 중...
    python scripts/data_collection/law_open_api/current_laws/update_vectors.py
) else if "%choice%"=="5" (
    echo 연결 테스트 실행 중...
    python scripts/data_collection/law_open_api/current_laws/run_pipeline.py --test
) else if "%choice%"=="6" (
    echo 샘플 수집 실행 중...
    python scripts/data_collection/law_open_api/current_laws/collect_current_laws.py --sample 10
) else if "%choice%"=="0" (
    echo 종료합니다.
    exit /b 0
) else (
    echo 잘못된 선택입니다.
)

echo.
echo 작업이 완료되었습니다.
pause
