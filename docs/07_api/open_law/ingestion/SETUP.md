# Open Law API 데이터 수집 환경 설정 가이드

## 1. 환경 변수 설정

### 방법 1: .env 파일 사용 (권장)

프로젝트 루트에 `.env` 파일을 생성하고 다음 내용을 추가하세요:

```bash
# PostgreSQL 데이터베이스 URL
DATABASE_URL=postgresql://username:password@localhost:5432/lawfirmai

# Open Law API 인증 정보
LAW_OPEN_API_OC=your_email_id  # g4c@korea.kr일 경우 oc=g4c
```

### 방법 2: 환경 변수 직접 설정

#### Windows (PowerShell)
```powershell
$env:DATABASE_URL="postgresql://username:password@localhost:5432/lawfirmai"
$env:LAW_OPEN_API_OC="your_email_id"
```

#### Windows (CMD)
```cmd
set DATABASE_URL=postgresql://username:password@localhost:5432/lawfirmai
set LAW_OPEN_API_OC=your_email_id
```

#### Linux/macOS
```bash
export DATABASE_URL="postgresql://username:password@localhost:5432/lawfirmai"
export LAW_OPEN_API_OC="your_email_id"
```

## 2. PostgreSQL 데이터베이스 준비

### 데이터베이스 생성

```sql
-- PostgreSQL에 접속
psql -U postgres

-- 데이터베이스 생성
CREATE DATABASE lawfirmai;

-- 사용자 생성 (선택사항)
CREATE USER lawfirmai_user WITH PASSWORD 'your_password';
GRANT ALL PRIVILEGES ON DATABASE lawfirmai TO lawfirmai_user;
```

### 연결 문자열 형식

```
postgresql://[사용자명]:[비밀번호]@[호스트]:[포트]/[데이터베이스명]
```

예시:
```
postgresql://lawfirmai_user:password123@localhost:5432/lawfirmai
```

## 3. Open Law API 인증 정보

### OC 값 확인 방법

Open Law API를 사용하려면 국가법령정보센터에 회원가입이 필요합니다.

1. [국가법령정보센터](https://www.law.go.kr) 접속
2. 회원가입 및 로그인
3. OPEN API 활용가이드 페이지에서 OC 값 확인
   - 이메일이 `g4c@korea.kr`인 경우 OC 값은 `g4c`
   - 이메일이 `test@example.com`인 경우 OC 값은 `test`

## 4. 실행 전 체크

환경 설정이 완료되었는지 확인:

```bash
# 실행 전 체크 (API 테스트 제외)
python scripts/ingest/open_law/scripts/preflight_check.py --skip-api-test

# 전체 체크 (API 테스트 포함)
python scripts/ingest/open_law/scripts/preflight_check.py
```

## 5. 스키마 초기화

데이터베이스에 스키마를 생성:

```bash
# 방법 1: Python 스크립트 사용
python scripts/migrations/init_open_law_schema.py --db $DATABASE_URL

# 방법 2: SQL 파일 직접 실행
psql -U username -d lawfirmai -f scripts/migrations/create_open_law_schema.sql
```

## 6. 필요한 Python 패키지

다음 패키지들이 설치되어 있어야 합니다:

```bash
pip install sqlalchemy psycopg2-binary requests
```

## 문제 해결

### 데이터베이스 연결 실패

- PostgreSQL이 실행 중인지 확인
- 연결 문자열이 올바른지 확인
- 방화벽 설정 확인
- 사용자 권한 확인

### API 접근 실패

- OC 값이 올바른지 확인
- 인터넷 연결 확인
- API 서버 상태 확인

### 패키지 설치 오류

- Python 버전 확인 (3.8 이상 권장)
- pip 업그레이드: `python -m pip install --upgrade pip`
- 가상환경 사용 권장

