#!/bin/bash
# 현행법령 데이터 수집 스크립트

echo "현행법령 데이터 수집 시스템"
echo "================================"

# 환경변수 확인
if [ -z "$LAW_OPEN_API_OC" ]; then
    echo "❌ LAW_OPEN_API_OC 환경변수가 설정되지 않았습니다."
    echo "다음과 같이 설정해주세요:"
    echo "export LAW_OPEN_API_OC='your_email@example.com'"
    exit 1
fi

echo "✅ OC 파라미터: $LAW_OPEN_API_OC"

# 메뉴 선택
echo ""
echo "실행할 작업을 선택하세요:"
echo "1. 전체 파이프라인 실행 (수집 → 데이터베이스 → 벡터)"
echo "2. 데이터 수집만 실행"
echo "3. 데이터베이스 업데이트만 실행"
echo "4. 벡터 저장소 업데이트만 실행"
echo "5. 연결 테스트"
echo "6. 샘플 수집 (10개)"
echo "0. 종료"
echo ""

read -p "선택 (0-6): " choice

case $choice in
    1)
        echo "전체 파이프라인 실행 중..."
        python scripts/data_collection/law_open_api/current_laws/run_pipeline.py --all
        ;;
    2)
        echo "데이터 수집 실행 중..."
        python scripts/data_collection/law_open_api/current_laws/collect_current_laws.py
        ;;
    3)
        echo "데이터베이스 업데이트 실행 중..."
        python scripts/data_collection/law_open_api/current_laws/update_database.py
        ;;
    4)
        echo "벡터 저장소 업데이트 실행 중..."
        python scripts/data_collection/law_open_api/current_laws/update_vectors.py
        ;;
    5)
        echo "연결 테스트 실행 중..."
        python scripts/data_collection/law_open_api/current_laws/run_pipeline.py --test
        ;;
    6)
        echo "샘플 수집 실행 중..."
        python scripts/data_collection/law_open_api/current_laws/collect_current_laws.py --sample 10
        ;;
    0)
        echo "종료합니다."
        exit 0
        ;;
    *)
        echo "잘못된 선택입니다."
        ;;
esac

echo ""
echo "작업이 완료되었습니다."
