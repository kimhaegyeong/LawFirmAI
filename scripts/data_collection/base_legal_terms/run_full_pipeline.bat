@echo off
REM 법령정보지식베이스 법령용어 통합 파이프라인 실행 스크립트 (Windows)

echo ========================================
echo 법령정보지식베이스 법령용어 통합 파이프라인 시작
echo ========================================

REM 환경 변수 설정 (필요시 수정)
set BASE_LEGAL_API_OC_ID=test
set BASE_LOG_LEVEL=INFO

REM 통합 파이프라인 실행
echo 통합 파이프라인 실행...
python scripts/data_collection/base_legal_terms/run_pipeline.py --collect-alternating --start-page 1 --end-page 5 --batch-size 20 --detail-batch-size 50 --verbose

echo ========================================
echo 법령정보지식베이스 법령용어 통합 파이프라인 완료
echo ========================================

pause
