#!/bin/bash
set -e

# 한국어 locale 설정 스크립트
echo "Setting up Korean locale for PostgreSQL..."

# 데이터베이스에 한국어 locale 적용
psql -v ON_ERROR_STOP=1 --username "$POSTGRES_USER" --dbname "$POSTGRES_DB" <<-EOSQL
    -- 한국어 locale 설정 확인
    DO \$\$
    BEGIN
        RAISE NOTICE 'Korean locale settings:';
        RAISE NOTICE 'lc_messages: %', current_setting('lc_messages');
        RAISE NOTICE 'lc_monetary: %', current_setting('lc_monetary');
        RAISE NOTICE 'lc_numeric: %', current_setting('lc_numeric');
        RAISE NOTICE 'lc_time: %', current_setting('lc_time');
    END
    \$\$;
EOSQL

echo "Korean locale setup completed."

