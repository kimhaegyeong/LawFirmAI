#!/bin/bash
# PostgreSQL 데이터베이스 백업 스크립트

set -e

BACKUP_DIR="/mnt/backups"
DATE=$(date +%Y%m%d_%H%M%S)
S3_BUCKET="${S3_BUCKET:-lawfirmai-backups}"

# 환경 변수 확인
if [ -z "$POSTGRES_DB" ] || [ -z "$POSTGRES_USER" ]; then
    echo "Error: POSTGRES_DB and POSTGRES_USER environment variables are required"
    exit 1
fi

# 백업 디렉토리 생성
mkdir -p $BACKUP_DIR

# PostgreSQL 백업
echo "Backing up PostgreSQL database..."
PGPASSWORD="${POSTGRES_PASSWORD}" pg_dump -h "${POSTGRES_HOST:-postgres}" -U "${POSTGRES_USER}" -d "${POSTGRES_DB}" -F c -f "$BACKUP_DIR/lawfirmai_$DATE.dump"

echo "Database backup completed: $BACKUP_DIR/lawfirmai_$DATE.dump"

# S3에 업로드 (S3_BUCKET이 설정된 경우)
if [ ! -z "$S3_BUCKET" ] && command -v aws &> /dev/null; then
    echo "Uploading to S3..."
    aws s3 cp "$BACKUP_DIR/lawfirmai_$DATE.dump" "s3://$S3_BUCKET/database/" || echo "S3 upload failed"
fi

# 로컬 백업 정리 (7일 이상 된 파일 삭제)
find $BACKUP_DIR -name "*.dump" -mtime +7 -delete

echo "Backup completed successfully"

