#!/bin/bash
# 전체 데이터 재임베딩 스크립트

DB_PATH="data/lawfirm_v2.db"
CHUNKING_STRATEGY="standard"
BATCH_SIZE=128

echo "=========================================="
echo "전체 데이터 재임베딩 시작"
echo "=========================================="
echo "데이터베이스: $DB_PATH"
echo "청킹 전략: $CHUNKING_STRATEGY"
echo "배치 크기: $BATCH_SIZE"
echo ""

# 백업 확인
if [ ! -f "${DB_PATH}.backup" ]; then
    echo "⚠️  백업 파일이 없습니다. 백업을 생성합니다..."
    cp "$DB_PATH" "${DB_PATH}.backup"
    echo "✅ 백업 완료: ${DB_PATH}.backup"
    echo ""
fi

# 재임베딩 시작
echo "재임베딩 시작..."
python scripts/migrations/re_embed_existing_data.py \
    --db "$DB_PATH" \
    --chunking-strategy "$CHUNKING_STRATEGY" \
    --batch-size "$BATCH_SIZE"

echo ""
echo "=========================================="
echo "재임베딩 완료"
echo "=========================================="

