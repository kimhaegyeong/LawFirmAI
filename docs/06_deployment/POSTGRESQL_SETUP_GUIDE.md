# PostgreSQL 설정 가이드

## 개요

이 가이드는 LawFirmAI에서 PostgreSQL을 사용하는 방법을 설명합니다.

## 환경별 데이터베이스 선택

### 로컬 개발
- **데이터베이스**: SQLite (기본값)
- **이유**: 간편함, 추가 설정 불필요
- **설정**: `DATABASE_URL=sqlite:///./data/lawfirm.db`

### 개발 서버
- **데이터베이스**: PostgreSQL
- **이유**: 개발 환경과 운영 환경 일치
- **설정**: `DATABASE_URL=postgresql://user:password@postgres:5432/lawfirmai_dev`

### 운영 서버
- **데이터베이스**: PostgreSQL
- **이유**: 성능, 확장성, 동시성
- **설정**: `DATABASE_URL=postgresql://user:password@postgres:5432/lawfirmai_prod`

## Docker Compose 설정

### 개발 환경

```bash
# 개발 환경 실행
docker-compose -f deployment/docker-compose.dev.yml up -d

# 데이터베이스 초기화
docker-compose -f deployment/docker-compose.dev.yml exec api python scripts/database/init_postgresql.py
```

### 운영 환경

```bash
# 운영 환경 실행
docker-compose -f deployment/docker-compose.prod.yml up -d

# 데이터베이스 초기화
docker-compose -f deployment/docker-compose.prod.yml exec api python scripts/database/init_postgresql.py
```

## 환경 변수 설정

### 개발 서버 (.env.development)

```env
DATABASE_URL=postgresql://lawfirmai:dev_password@postgres:5432/lawfirmai_dev
POSTGRES_HOST=postgres
POSTGRES_PORT=5432
POSTGRES_DB=lawfirmai_dev
POSTGRES_USER=lawfirmai
POSTGRES_PASSWORD=dev_password
```

### 운영 서버 (.env.production)

```env
DATABASE_URL=postgresql://lawfirmai:secure_password@postgres:5432/lawfirmai_prod
POSTGRES_HOST=postgres
POSTGRES_PORT=5432
POSTGRES_DB=lawfirmai_prod
POSTGRES_USER=lawfirmai
POSTGRES_PASSWORD=secure_password
```

## 데이터 마이그레이션

### SQLite → PostgreSQL

```bash
# 환경 변수 설정
export SQLITE_PATH=./data/api_sessions.db
export POSTGRES_URL=postgresql://lawfirmai:password@postgres:5432/lawfirmai_dev

# 마이그레이션 실행
python scripts/database/migrate_to_postgresql.py
```

## 데이터베이스 초기화

### PostgreSQL 초기화

```bash
# 환경 변수 설정
export DATABASE_URL=postgresql://lawfirmai:password@postgres:5432/lawfirmai_dev

# 초기화 실행
python scripts/database/init_postgresql.py
```

## 데이터베이스 백업

### PostgreSQL 백업

```bash
# 백업
docker-compose exec postgres pg_dump -U lawfirmai lawfirmai_dev > backup.sql

# 복구
docker-compose exec -T postgres psql -U lawfirmai lawfirmai_dev < backup.sql
```

## 문제 해결

### 연결 실패

```bash
# PostgreSQL 상태 확인
docker-compose -f deployment/docker-compose.prod.yml exec postgres pg_isready -U lawfirmai

# 로그 확인
docker-compose -f deployment/docker-compose.prod.yml logs postgres
```

### 권한 문제

```bash
# PostgreSQL 사용자 권한 확인
docker-compose -f deployment/docker-compose.prod.yml exec postgres psql -U lawfirmai -d lawfirmai_dev -c "\du"
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
```

## 참고 자료

- [PostgreSQL 공식 문서](https://www.postgresql.org/docs/)
- [SQLAlchemy 문서](https://docs.sqlalchemy.org/)
- [Alembic 마이그레이션](https://alembic.sqlalchemy.org/)

---

**자세한 마이그레이션 계획은 [POSTGRESQL_MIGRATION_PLAN.md](POSTGRESQL_MIGRATION_PLAN.md)를 참조하세요.**

