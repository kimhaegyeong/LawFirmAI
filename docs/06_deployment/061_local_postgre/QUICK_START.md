# 로컬 PostgreSQL 체크포인트 빠른 시작 가이드

## 1분 안에 시작하기

### 1단계: PostgreSQL 실행

```bash
cd docs/06_deployment/061_local_postgre
docker-compose -f docker-compose.local.yml up -d
```

또는 Windows:
```batch
start-local-docker.bat
```

### 2단계: 환경 변수 설정

프로젝트 루트의 `.env` 파일에 추가:

```env
DATABASE_URL=postgresql://lawfirmai:local_password@localhost:5432/lawfirmai_local
CHECKPOINT_STORAGE=postgres
```

### 3단계: 완료!

이제 애플리케이션을 실행하면 체크포인트가 자동으로 PostgreSQL에 저장됩니다.

## 확인 방법

### pgAdmin으로 확인

1. http://localhost:5050 접속
2. 로그인: `admin@lawfirmai.local` / `admin`
3. PostgreSQL 서버 → `lawfirmai_local` 데이터베이스 → `checkpoints` 테이블 확인

### 명령줄으로 확인

```bash
docker exec -it lawfirmai-postgres-local psql -U lawfirmai -d lawfirmai_local -c "SELECT COUNT(*) FROM checkpoints;"
```

## 문제 해결

### PostgreSQL이 실행되지 않는 경우

```bash
docker ps | grep postgres
```

### 연결 오류가 발생하는 경우

1. PostgreSQL이 실행 중인지 확인
2. `.env` 파일의 `DATABASE_URL` 확인
3. 비밀번호가 `local_password`인지 확인

### 더 자세한 정보

- [상세 설정 가이드](./CHECKPOINT_SETUP.md)
- [Docker 설정 가이드](./README.local-docker.md)

