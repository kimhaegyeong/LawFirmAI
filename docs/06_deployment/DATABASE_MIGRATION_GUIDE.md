# 데이터베이스 마이그레이션 가이드

## 개요

이 가이드는 SQLite에서 PostgreSQL로 마이그레이션하는 방법을 설명합니다.

## 환경별 데이터베이스 선택

### 로컬 개발
- **데이터베이스**: SQLite
- **설정**: `DATABASE_URL=sqlite:///./data/lawfirm.db`
- **이유**: 간편함, 추가 설정 불필요

### 개발 서버
- **데이터베이스**: PostgreSQL
- **설정**: `DATABASE_URL=postgresql://lawfirmai:dev_password@postgres:5432/lawfirmai_dev`
- **이유**: 개발 환경과 운영 환경 일치

### 운영 서버
- **데이터베이스**: PostgreSQL
- **설정**: `DATABASE_URL=postgresql://lawfirmai:secure_password@postgres:5432/lawfirmai_prod`
- **이유**: 성능, 확장성, 동시성

## 단계별 마이그레이션 절차

### Step 1: PostgreSQL 서비스 시작

#### 개발 환경

```bash
# PostgreSQL 포함하여 실행
docker-compose -f deployment/docker-compose.dev.yml up -d postgres

# PostgreSQL 상태 확인
docker-compose -f deployment/docker-compose.dev.yml exec postgres pg_isready -U lawfirmai
```

#### 운영 환경

```bash
# PostgreSQL 포함하여 실행
docker-compose -f deployment/docker-compose.prod.yml up -d postgres

# PostgreSQL 상태 확인
docker-compose -f deployment/docker-compose.prod.yml exec postgres pg_isready -U lawfirmai
```

### Step 2: 데이터베이스 초기화

```bash
# 환경 변수 설정
export DATABASE_URL=postgresql://lawfirmai:password@postgres:5432/lawfirmai_dev

# 데이터베이스 초기화
python scripts/database/init_postgresql.py
```

또는 Docker Compose 사용:

```bash
# 개발 환경
docker-compose -f deployment/docker-compose.dev.yml exec api python scripts/database/init_postgresql.py

# 운영 환경
docker-compose -f deployment/docker-compose.prod.yml exec api python scripts/database/init_postgresql.py
```

### Step 3: 데이터 마이그레이션

#### SQLite → PostgreSQL

```bash
# 환경 변수 설정
export SQLITE_PATH=./data/api_sessions.db
export POSTGRES_URL=postgresql://lawfirmai:password@postgres:5432/lawfirmai_dev

# 마이그레이션 실행
python scripts/database/migrate_to_postgresql.py
```

또는 Docker Compose 사용:

```bash
# 개발 환경
docker-compose -f deployment/docker-compose.dev.yml exec api python scripts/database/migrate_to_postgresql.py

# 운영 환경
docker-compose -f deployment/docker-compose.prod.yml exec api python scripts/database/migrate_to_postgresql.py
```

### Step 4: 애플리케이션 재시작

```bash
# 개발 환경
docker-compose -f deployment/docker-compose.dev.yml restart api

# 운영 환경
docker-compose -f deployment/docker-compose.prod.yml restart api
```

### Step 5: 검증

```bash
# 데이터베이스 연결 확인
docker-compose exec api python -c "from api.database.connection import get_session; session = get_session(); print('✅ Database connection successful')"

# 데이터 확인
docker-compose exec postgres psql -U lawfirmai -d lawfirmai_dev -c "SELECT COUNT(*) FROM sessions;"
```

## 환경 변수 설정

### 개발 서버

`.env.development` 파일 생성:

```env
# PostgreSQL 설정
DATABASE_URL=postgresql://lawfirmai:dev_password@postgres:5432/lawfirmai_dev
POSTGRES_HOST=postgres
POSTGRES_PORT=5432
POSTGRES_DB=lawfirmai_dev
POSTGRES_USER=lawfirmai
POSTGRES_PASSWORD=dev_password
```

### 운영 서버

`.env.production` 파일 생성:

```env
# PostgreSQL 설정
DATABASE_URL=postgresql://lawfirmai:secure_password@postgres:5432/lawfirmai_prod
POSTGRES_HOST=postgres
POSTGRES_PORT=5432
POSTGRES_DB=lawfirmai_prod
POSTGRES_USER=lawfirmai
POSTGRES_PASSWORD=secure_password
```

## 데이터베이스 백업 및 복구

### 백업

```bash
# PostgreSQL 백업
./scripts/database/backup_postgresql.sh

# 또는 직접 실행
PGPASSWORD=password pg_dump -h postgres -U lawfirmai -d lawfirmai_dev -F c -f backup.dump
```

### 복구

```bash
# PostgreSQL 복구
./scripts/database/restore_postgresql.sh /path/to/backup.dump

# 또는 직접 실행
PGPASSWORD=password pg_restore -h postgres -U lawfirmai -d lawfirmai_dev -c -F c backup.dump
```

## 문제 해결

### 연결 실패

```bash
# PostgreSQL 상태 확인
docker-compose -f deployment/docker-compose.prod.yml exec postgres pg_isready -U lawfirmai

# 로그 확인
docker-compose -f deployment/docker-compose.prod.yml logs postgres

# 네트워크 확인
docker-compose -f deployment/docker-compose.prod.yml exec api ping postgres
```

### 권한 문제

```bash
# PostgreSQL 사용자 권한 확인
docker-compose -f deployment/docker-compose.prod.yml exec postgres psql -U lawfirmai -d lawfirmai_dev -c "\du"

# 데이터베이스 권한 확인
docker-compose -f deployment/docker-compose.prod.yml exec postgres psql -U lawfirmai -d lawfirmai_dev -c "\l"
```

### 데이터 마이그레이션 실패

```bash
# SQLite 데이터 확인
sqlite3 ./data/api_sessions.db "SELECT COUNT(*) FROM sessions;"

# PostgreSQL 데이터 확인
docker-compose exec postgres psql -U lawfirmai -d lawfirmai_dev -c "SELECT COUNT(*) FROM sessions;"

# 마이그레이션 재시도
python scripts/database/migrate_to_postgresql.py
```

## 롤백 절차

PostgreSQL에서 SQLite로 롤백:

```bash
# 환경 변수 변경
export DATABASE_URL=sqlite:///./data/lawfirm.db

# 애플리케이션 재시작
docker-compose restart api
```

## 성능 최적화

### PostgreSQL 설정

```sql
-- 연결 수 증가
ALTER SYSTEM SET max_connections = 200;

-- 공유 버퍼 증가
ALTER SYSTEM SET shared_buffers = '256MB';

-- 효과적인 캐시 크기
ALTER SYSTEM SET effective_cache_size = '1GB';

-- 설정 적용
SELECT pg_reload_conf();
```

### 인덱스 최적화

```sql
-- 인덱스 생성
CREATE INDEX IF NOT EXISTS idx_sessions_user_id ON sessions(user_id);
CREATE INDEX IF NOT EXISTS idx_sessions_updated_at ON sessions(updated_at);
CREATE INDEX IF NOT EXISTS idx_messages_session_id ON messages(session_id);
CREATE INDEX IF NOT EXISTS idx_messages_timestamp ON messages(timestamp);

-- 인덱스 통계 업데이트
ANALYZE sessions;
ANALYZE messages;
```

## 모니터링

### 데이터베이스 크기 확인

```sql
-- 데이터베이스 크기
SELECT pg_size_pretty(pg_database_size('lawfirmai_dev'));

-- 테이블 크기
SELECT 
    schemaname,
    tablename,
    pg_size_pretty(pg_total_relation_size(schemaname||'.'||tablename)) AS size
FROM pg_tables
WHERE schemaname = 'public'
ORDER BY pg_total_relation_size(schemaname||'.'||tablename) DESC;
```

### 연결 수 확인

```sql
-- 현재 연결 수
SELECT count(*) FROM pg_stat_activity;

-- 연결 상세 정보
SELECT 
    pid,
    usename,
    application_name,
    client_addr,
    state,
    query
FROM pg_stat_activity
WHERE datname = 'lawfirmai_dev';
```

## 참고 자료

- [PostgreSQL 공식 문서](https://www.postgresql.org/docs/)
- [SQLAlchemy 문서](https://docs.sqlalchemy.org/)
- [Alembic 마이그레이션](https://alembic.sqlalchemy.org/)
- [PostgreSQL 마이그레이션 계획](POSTGRESQL_MIGRATION_PLAN.md)

---

**자세한 마이그레이션 계획은 [POSTGRESQL_MIGRATION_PLAN.md](POSTGRESQL_MIGRATION_PLAN.md)를 참조하세요.**

