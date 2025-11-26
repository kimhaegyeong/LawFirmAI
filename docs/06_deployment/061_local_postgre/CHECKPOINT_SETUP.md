# 로컬 개발 환경에서 PostgreSQL 체크포인트 설정 가이드

## 개요

로컬 개발 환경에서 PostgreSQL을 사용하여 LangGraph 체크포인트를 저장하는 방법을 설명합니다.

## 빠른 시작

### 1. PostgreSQL Docker 컨테이너 실행

```bash
# 로컬 PostgreSQL + pgAdmin 실행
cd docs/06_deployment/061_local_postgre
docker-compose -f docker-compose.local.yml up -d
```

또는 Windows 배치 파일 사용:
```batch
start-local-docker.bat
```

### 2. 환경 변수 설정

`.env` 파일 또는 환경 변수에 다음을 설정:

```env
# 메인 데이터베이스 (애플리케이션 데이터용)
DATABASE_URL=postgresql://lawfirmai:local_password@localhost:5432/lawfirmai_local

# 체크포인트 설정
CHECKPOINT_STORAGE=postgres
# CHECKPOINT_DATABASE_URL=postgresql://lawfirmai:local_password@localhost:5432/lawfirmai_local
```

**참고**: `CHECKPOINT_DATABASE_URL`이 설정되지 않으면 `DATABASE_URL`을 자동으로 사용합니다.

### 3. 애플리케이션 실행

체크포인트는 자동으로 PostgreSQL에 저장됩니다.

## 상세 설정

### 체크포인트 데이터베이스 옵션

#### 옵션 1: 메인 데이터베이스와 동일한 데이터베이스 사용 (권장)

```env
DATABASE_URL=postgresql://lawfirmai:local_password@localhost:5432/lawfirmai_local
CHECKPOINT_STORAGE=postgres
# CHECKPOINT_DATABASE_URL 설정 안 함 (자동으로 DATABASE_URL 사용)
```

#### 옵션 2: 별도 데이터베이스 사용

```env
# 메인 데이터베이스
DATABASE_URL=postgresql://lawfirmai:local_password@localhost:5432/lawfirmai_main

# 체크포인트 전용 데이터베이스
CHECKPOINT_DATABASE_URL=postgresql://lawfirmai:local_password@localhost:5432/lawfirmai_checkpoints
CHECKPOINT_STORAGE=postgres
```

### 체크포인트 저장소 타입

- **`memory`**: 메모리 기반 (재시작 시 데이터 손실, 개발/테스트용)
- **`postgres`**: PostgreSQL 기반 (영구 저장, 프로덕션용)
- **`disabled`**: 체크포인트 비활성화

### 환경별 권장 설정

#### 로컬 개발
```env
CHECKPOINT_STORAGE=postgres
DATABASE_URL=postgresql://lawfirmai:local_password@localhost:5432/lawfirmai_local
```

#### 개발 서버
```env
CHECKPOINT_STORAGE=postgres
DATABASE_URL=postgresql://lawfirmai:dev_password@postgres:5432/lawfirmai_dev
```

#### 프로덕션
```env
CHECKPOINT_STORAGE=postgres
DATABASE_URL=postgresql://lawfirmai:secure_password@postgres:5432/lawfirmai_prod
```

## PostgreSQL 연결 정보

### 로컬 Docker 컨테이너

- **호스트**: `localhost`
- **포트**: `5432`
- **데이터베이스**: `lawfirmai_local` (기본값)
- **사용자**: `lawfirmai`
- **비밀번호**: `local_password`

### 연결 문자열 형식

```
postgresql://[user]:[password]@[host]:[port]/[database]
```

예시:
```
postgresql://lawfirmai:local_password@localhost:5432/lawfirmai_local
```

## 체크포인트 테이블 자동 생성

LangGraph의 `PostgresSaver`가 자동으로 다음 테이블을 생성합니다:

- `checkpoints`: 체크포인트 데이터 저장
- `checkpoint_blobs`: 대용량 체크포인트 데이터 저장

별도의 마이그레이션 스크립트가 필요하지 않습니다.

## 문제 해결

### 체크포인트가 저장되지 않는 경우

1. **PostgreSQL 연결 확인**
   ```bash
   docker ps | grep postgres
   ```

2. **환경 변수 확인**
   ```bash
   echo $DATABASE_URL
   echo $CHECKPOINT_STORAGE
   ```

3. **로그 확인**
   - 체크포인트 초기화 로그 확인
   - PostgreSQL 연결 오류 확인

### MemorySaver로 폴백되는 경우

- `CHECKPOINT_STORAGE=postgres`로 설정되어 있는지 확인
- `DATABASE_URL` 또는 `CHECKPOINT_DATABASE_URL`이 PostgreSQL URL인지 확인
- `langgraph-checkpoint-postgres` 패키지가 설치되어 있는지 확인:
  ```bash
  pip install langgraph-checkpoint-postgres
  ```

### SQLite URL이 감지된 경우

SQLite는 더 이상 체크포인트 저장소로 지원되지 않습니다. PostgreSQL URL을 사용하세요:

```env
# ❌ 잘못된 예
DATABASE_URL=sqlite:///./data/lawfirm.db

# ✅ 올바른 예
DATABASE_URL=postgresql://lawfirmai:local_password@localhost:5432/lawfirmai_local
```

## pgAdmin으로 체크포인트 확인

1. http://localhost:5050 접속
2. 로그인 (admin@lawfirmai.local / admin)
3. PostgreSQL 서버 연결
4. `lawfirmai_local` 데이터베이스 선택
5. `checkpoints` 테이블 확인

## 추가 리소스

- [로컬 PostgreSQL Docker 설정](./README.local-docker.md)
- [PostgreSQL 마이그레이션 가이드](../POSTGRESQL_MIGRATION_PLAN.md)
- [LangGraph 체크포인트 문서](https://langchain-ai.github.io/langgraph/how-tos/persistence/)

