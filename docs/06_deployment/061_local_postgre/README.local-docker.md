# 로컬 개발용 PostgreSQL + pgAdmin Docker 설정

로컬 개발 환경에서 PostgreSQL과 pgAdmin을 사용하기 위한 Docker Compose 설정입니다.

## 시작하기

### 1. Docker Compose 실행

**Windows 사용자 (권장):**
```batch
# 배치 파일 사용 (가장 간단)
# 이 폴더(docs\06_deployment\061.local_postgre)에서 실행
start-local-docker.bat
```

**또는 직접 실행:**
```bash
# 이 폴더(docs/06_deployment/061.local_postgre)에서 실행
docker-compose -f docker-compose.local.yml up -d
```

### 2. 서비스 접속 정보

#### PostgreSQL
- **호스트**: `localhost`
- **포트**: `5432`
- **데이터베이스**: `lawfirmai_local`
- **사용자**: `lawfirmai`
- **비밀번호**: `local_password`

#### pgAdmin
- **URL**: http://localhost:5050
- **이메일**: `admin@lawfirmai.local`
- **비밀번호**: `admin`

### 3. 연결 문자열

```
postgresql://lawfirmai:local_password@localhost:5432/lawfirmai_local
```

## 한국어 Locale 설정

이 설정은 PostgreSQL과 pgAdmin에서 한국어 locale을 사용하도록 구성되어 있습니다:

- **LC_ALL**: `ko_KR.UTF-8`
- **LANG**: `ko_KR.UTF-8`
- **데이터베이스 locale**: `ko_KR.UTF-8`
- **메시지 locale**: `ko_KR.UTF-8`
- **날짜/시간 locale**: `ko_KR.UTF-8`
- **숫자/통화 locale**: `ko_KR.UTF-8`

## pgAdmin에서 서버 연결

1. http://localhost:5050 접속
2. 로그인 (admin@lawfirmai.local / admin)
3. 서버가 자동으로 등록되어 있음:
   - **이름**: LawFirmAI Local PostgreSQL
   - **호스트**: postgres
   - **포트**: 5432
   - **데이터베이스**: lawfirmai_local
   - **사용자**: lawfirmai
   - **비밀번호**: local_password

## 유용한 명령어

### 서비스 시작
**Windows:**
```batch
# 이 폴더에서 실행
start-local-docker.bat
```

**또는:**
```bash
# 이 폴더에서 실행
docker-compose -f docker-compose.local.yml up -d
```

### 서비스 중지
**Windows:**
```batch
# 이 폴더에서 실행
stop-local-docker.bat
```

**또는:**
```bash
# 이 폴더에서 실행
docker-compose -f docker-compose.local.yml stop
```

### 서비스 재시작
**Windows:**
```batch
# 이 폴더에서 실행
restart-local-docker.bat
```

**또는:**
```bash
# 이 폴더에서 실행
docker-compose -f docker-compose.local.yml restart
```

### 서비스 종료 및 볼륨 삭제
```bash
docker-compose -f docker-compose.local.yml down -v
```

### 로그 확인
```bash
# PostgreSQL 로그
docker-compose -f docker-compose.local.yml logs postgres

# pgAdmin 로그
docker-compose -f docker-compose.local.yml logs pgadmin

# 모든 로그
docker-compose -f docker-compose.local.yml logs -f
```

### PostgreSQL에 직접 접속
```bash
docker exec -it lawfirmai-postgres-local psql -U lawfirmai -d lawfirmai_local
```

### 데이터베이스 백업
```bash
docker exec lawfirmai-postgres-local pg_dump -U lawfirmai lawfirmai_local > backup.sql
```

### 데이터베이스 복원
```bash
docker exec -i lawfirmai-postgres-local psql -U lawfirmai lawfirmai_local < backup.sql
```

## 환경 변수 커스터마이징

프로젝트 루트에 `.env` 파일을 생성하여 환경 변수를 커스터마이징할 수 있습니다:

```env
# PostgreSQL 설정
POSTGRES_DB=lawfirmai_local
POSTGRES_USER=lawfirmai
POSTGRES_PASSWORD=local_password

# pgAdmin 설정
PGADMIN_DEFAULT_EMAIL=admin@lawfirmai.local
PGADMIN_DEFAULT_PASSWORD=admin
```

docker-compose는 자동으로 `.env` 파일을 읽습니다. 환경 변수가 설정되지 않은 경우 기본값이 사용됩니다.

**PowerShell에서 환경 변수 설정 예시:**
```powershell
$env:POSTGRES_PASSWORD="my_secure_password"
$env:PGADMIN_DEFAULT_PASSWORD="my_admin_password"
docker-compose -f docker-compose.local.yml up -d
```

## 문제 해결

### 포트 충돌
포트 5432나 5050이 이미 사용 중인 경우, `docker-compose.local.yml`에서 포트 매핑을 변경하세요:

```yaml
ports:
  - "5433:5432"  # PostgreSQL 포트 변경
  - "5051:80"    # pgAdmin 포트 변경
```

### 한국어 locale 오류
한국어 locale이 제대로 작동하지 않는 경우:

1. 호스트 시스템에 한국어 locale이 설치되어 있는지 확인
2. Docker 이미지가 한국어 locale을 지원하는지 확인
3. `init-locale.sh` 스크립트가 실행되었는지 확인

### 볼륨 권한 문제
볼륨 권한 문제가 발생하는 경우:

```bash
# 볼륨 삭제 후 재생성
docker-compose -f docker-compose.local.yml down -v
docker-compose -f docker-compose.local.yml up -d
```

## 참고 사항

- 이 설정은 **로컬 개발 전용**입니다. 프로덕션 환경에서는 사용하지 마세요.
- 기본 비밀번호는 보안이 취약하므로, 프로덕션 환경에서는 반드시 변경하세요.
- 데이터는 Docker 볼륨에 저장되므로, `docker-compose down -v` 실행 시 모든 데이터가 삭제됩니다.

