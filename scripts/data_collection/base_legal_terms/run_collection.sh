#!/bin/bash
# 법령정보지식베이스 법령용어 수집 실행 스크립트 (Linux/Mac)

echo "========================================"
echo "법령정보지식베이스 법령용어 수집 시작"
echo "========================================"

# 환경 변수 설정 (필요시 수정)
export BASE_LEGAL_API_OC_ID=test
export BASE_LOG_LEVEL=INFO

# 기본 수집 실행 (목록만)
echo "기본 목록 수집 실행..."
python scripts/data_collection/base_legal_terms/base_legal_term_collector.py --collect-lists --start-page 1 --end-page 10 --batch-size 20 --verbose

# 상세 정보 수집 실행
echo "상세 정보 수집 실행..."
python scripts/data_collection/base_legal_terms/base_legal_term_collector.py --collect-details --detail-batch-size 50 --verbose

# 데이터 처리 실행
echo "데이터 처리 실행..."
python scripts/data_processing/base_legal_terms/process_terms.py

# 벡터 임베딩 생성 실행
echo "벡터 임베딩 생성 실행..."
python scripts/data_processing/base_legal_terms/generate_embeddings.py

echo "========================================"
echo "법령정보지식베이스 법령용어 수집 완료"
echo "========================================"
