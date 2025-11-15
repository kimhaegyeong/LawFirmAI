#!/bin/bash
# PostgreSQL 데이터베이스 복구 스크립트

set -e

BACKUP_FILE="${1:-/mnt/backups/latest.dump}"

# 환경 변수 확인
if [ -z "$POSTGRES_DB" ] || [ -z "$POSTGRES_USER" ]; then
    echo "Error: POSTGRES_DB and POSTGRES_USER environment variables are required"
    exit 1
fi

# 백업 파일 확인
if [ ! -f "$BACKUP_FILE" ]; then
    echo "Error: Backup file not found: $BACKUP_FILE"
    exit 1
fi

# 복구 확인
read -p "Are you sure you want to restore from $BACKUP_FILE? This will overwrite existing data. (yes/no): " confirm
if [ "$confirm" != "yes" ]; then
    echo "Restore cancelled"
    exit 0
fi

# PostgreSQL 복구
echo "Restoring PostgreSQL database from $BACKUP_FILE..."
PGPASSWORD="${POSTGRES_PASSWORD}" pg_restore -h "${POSTGRES_HOST:-postgres}" -U "${POSTGRES_USER}" -d "${POSTGRES_DB}" -c -F c "$BACKUP_FILE"

echo "Database restore completed successfully"

