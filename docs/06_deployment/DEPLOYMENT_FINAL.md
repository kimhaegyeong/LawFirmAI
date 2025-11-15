# LawFirmAI 최종 배포 가이드

## 📋 목차

1. [개요](#개요)
2. [배포 아키텍처](#배포-아키텍처)
3. [사전 준비사항](#사전-준비사항)
4. [AWS 인프라 구성](#aws-인프라-구성)
5. [데이터베이스 설정](#데이터베이스-설정)
6. [Docker 및 배포 설정](#docker-및-배포-설정)
7. [CI/CD 설정](#cicd-설정)
8. [환경 변수 및 보안](#환경-변수-및-보안)
9. [단계별 배포 절차](#단계별-배포-절차)
10. [모니터링 및 백업](#모니터링-및-백업)
11. [문제 해결](#문제-해결)
12. [프리 티어 최적화](#프리-티어-최적화)

---

## 개요

이 문서는 LawFirmAI를 AWS에 프로덕션 배포하는 전체 과정을 설명합니다.

### 배포 옵션

| 옵션 | 설명 | 비용 | 권장 용도 |
|------|------|------|----------|
| **프리 티어** | t2.micro/t3.micro | $0/월 (12개월) | 테스트, 학습 |
| **프로덕션** | t3.large+ | $70-150/월 | 실제 서비스 |
| **고가용성** | 다중 인스턴스 | $200-300/월 | 대규모 서비스 |

### 환경별 데이터베이스

| 환경 | 데이터베이스 | 설정 |
|------|------------|------|
| **로컬 개발** | SQLite | `DATABASE_URL=sqlite:///./data/lawfirm.db` |
| **개발 서버** | PostgreSQL | `DATABASE_URL=postgresql://user:pass@postgres:5432/db` |
| **운영 서버** | PostgreSQL | `DATABASE_URL=postgresql://user:pass@postgres:5432/db` |

---

## 배포 아키텍처

### 선택된 아키텍처: EC2 + Docker Compose

```
┌─────────────────────────────────────────┐
│           AWS EC2 Instance                │
│  ┌─────────────────────────────────────┐ │
│  │         Nginx (Port 80/443)          │ │
│  │         (Reverse Proxy)              │ │
│  └──────────────┬──────────────────────┘ │
│                 │                         │
│  ┌──────────────▼──────────────────────┐ │
│  │    Frontend Container (React)        │ │
│  └──────────────┬──────────────────────┘ │
│                 │                         │
│  ┌───────────────▼───────────────────────┐ │
│  │     API Container (FastAPI)           │ │
│  └──────────────┬───────────────────────┘ │
│                 │                          │
│  ┌──────────────▼───────────────────────┐ │
│  │   PostgreSQL Container (선택)        │ │
│  └──────────────────────────────────────┘ │
└─────────────────────────────────────────┘
```

### 인프라 구성

- **EC2**: Ubuntu 22.04 LTS
- **ECR**: Docker 이미지 저장소
- **Docker Compose**: 멀티 컨테이너 오케스트레이션
- **Nginx**: 리버스 프록시 및 정적 파일 서빙
- **PostgreSQL**: 데이터베이스 (개발/운영)
- **Let's Encrypt**: SSL/TLS 인증서

---

## 사전 준비사항

### 1. AWS 계정 설정

```bash
# AWS CLI 설치 및 설정
aws configure

# 필요한 정보:
# - AWS Access Key ID
# - AWS Secret Access Key
# - Default region: ap-northeast-2
# - Default output format: json
```

### 2. IAM 사용자 생성

**필요한 권한:**
- EC2 (인스턴스 생성, 관리)
- ECR (컨테이너 레지스트리)
- CloudWatch (로깅)
- S3 (백업, 선택사항)
- Systems Manager Parameter Store (환경 변수, 선택사항)

### 3. GitHub 설정

**GitHub Secrets 설정:**
- `AWS_ACCESS_KEY_ID`
- `AWS_SECRET_ACCESS_KEY`
- `AWS_REGION` (예: `ap-northeast-2`)
- `EC2_SSH_KEY` (EC2 SSH 프라이빗 키)
- `EC2_HOST` (EC2 퍼블릭 IP 또는 도메인)
- `GOOGLE_API_KEY` (Google AI API 키)
- `ECR_REGISTRY` (ECR 레지스트리 URL)

### 4. 도메인 설정 (선택사항)

- 도메인 구매 또는 기존 도메인 확인
- Route 53 호스팅 영역 생성
- DNS 레코드 설정

---

## AWS 인프라 구성

### 1. ECR 저장소 생성

```bash
# API 이미지 저장소
aws ecr create-repository \
  --repository-name lawfirmai-api \
  --region ap-northeast-2

# Frontend 이미지 저장소
aws ecr create-repository \
  --repository-name lawfirmai-frontend \
  --region ap-northeast-2
```

### 2. EC2 인스턴스 생성

#### 프리 티어 구성 (신규 AWS 계정)

**AWS Console 설정:**
1. **EC2** → **Launch Instance**
2. **AMI**: Ubuntu 22.04 LTS (프리 티어 자격)
3. **Instance type**: `t2.micro` 또는 `t3.micro`
4. **Key pair**: 새로 생성 또는 기존 사용
5. **Network settings**:
   - 퍼블릭 IP 자동 할당 활성화
   - 보안 그룹 생성:
     - SSH (22): 내 IP만
     - HTTP (80): 0.0.0.0/0
     - HTTPS (443): 0.0.0.0/0
     - Custom TCP (8000): 내 IP만 (API)
6. **Storage**: 30GB GP2 SSD
7. **Launch Instance**

**프리 티어 제한사항:**
- 인스턴스 타입: `t2.micro` 또는 `t3.micro`만 무료
- 스토리지: 30GB 이하
- 사용 시간: 750시간/월 무료
- 기간: 신규 계정 12개월간

#### 프로덕션 구성

**권장 사양:**
- **Instance type**: `t3.large` 또는 `t3.xlarge`
- **Storage**: 50GB+ GP3 SSD
- **Memory**: 8GB+ (권장 16GB)
- **CPU**: 2 vCPU+ (권장 4 vCPU)

### 3. EC2 초기 설정

```bash
# EC2 인스턴스에 SSH 접속
ssh -i your-key.pem ubuntu@your-ec2-ip

# 초기 설정 스크립트 실행
cd /opt/lawfirmai
sudo bash deployment/setup_ec2.sh
```

**초기 설정 스크립트가 수행하는 작업:**
- Docker 및 Docker Compose 설치
- AWS CLI 설정
- Swap 메모리 설정 (프리 티어 필수)
- 불필요한 서비스 비활성화
- 디렉토리 생성

---

## 데이터베이스 설정

### 환경별 데이터베이스 선택

#### 로컬 개발 (SQLite)

```env
# .env
DATABASE_URL=sqlite:///./data/lawfirm.db
DATABASE_TYPE=sqlite
```

**장점:**
- 간편함, 추가 설정 불필요
- 파일 기반, 백업 용이

#### 개발/운영 서버 (PostgreSQL)

```env
# .env.development 또는 .env.production
DATABASE_URL=postgresql://lawfirmai:password@postgres:5432/lawfirmai_dev
POSTGRES_HOST=postgres
POSTGRES_PORT=5432
POSTGRES_DB=lawfirmai_dev
POSTGRES_USER=lawfirmai
POSTGRES_PASSWORD=password
```

**장점:**
- 높은 성능 및 확장성
- 동시성 지원
- 트랜잭션 및 ACID 보장

### PostgreSQL 초기화

```bash
# Docker Compose로 PostgreSQL 시작
docker-compose -f deployment/docker-compose.prod.yml up -d postgres

# 데이터베이스 초기화
docker-compose -f deployment/docker-compose.prod.yml exec api python scripts/database/init_postgresql.py
```

### 데이터 마이그레이션 (SQLite → PostgreSQL)

```bash
# 환경 변수 설정
export SQLITE_PATH=./data/api_sessions.db
export POSTGRES_URL=postgresql://lawfirmai:password@postgres:5432/lawfirmai_prod

# 마이그레이션 실행
python scripts/database/migrate_to_postgresql.py
```

**자세한 내용:**
- [PostgreSQL 마이그레이션 계획](POSTGRESQL_MIGRATION_PLAN.md)
- [데이터베이스 마이그레이션 가이드](DATABASE_MIGRATION_GUIDE.md)

---

## Docker 및 배포 설정

### Docker Compose 파일

#### 개발 환경 (`deployment/docker-compose.dev.yml`)

```yaml
version: '3.8'

services:
  postgres:
    image: postgres:15-alpine
    environment:
      POSTGRES_DB: lawfirmai_dev
      POSTGRES_USER: lawfirmai
      POSTGRES_PASSWORD: dev_password
    ports:
      - "5432:5432"
    volumes:
      - postgres_dev_data:/var/lib/postgresql/data

  api:
    build:
      context: ..
      dockerfile: api/Dockerfile
    ports:
      - "8000:8000"
    environment:
      - DATABASE_URL=postgresql://lawfirmai:dev_password@postgres:5432/lawfirmai_dev
    depends_on:
      - postgres

  frontend:
    build:
      context: ..
      dockerfile: frontend/Dockerfile
    ports:
      - "3000:80"
    depends_on:
      - api
```

#### 운영 환경 (`deployment/docker-compose.prod.yml`)

```yaml
version: '3.8'

services:
  postgres:
    image: postgres:15-alpine
    environment:
      POSTGRES_DB: ${POSTGRES_DB:-lawfirmai}
      POSTGRES_USER: ${POSTGRES_USER:-lawfirmai}
      POSTGRES_PASSWORD: ${POSTGRES_PASSWORD:-changeme}
    volumes:
      - postgres_data:/var/lib/postgresql/data
    healthcheck:
      test: ["CMD-SHELL", "pg_isready -U ${POSTGRES_USER:-lawfirmai}"]
      interval: 10s
      timeout: 5s
      retries: 5

  api:
    image: ${ECR_REGISTRY}/lawfirmai-api:latest
    environment:
      - DATABASE_URL=${DATABASE_URL:-postgresql://${POSTGRES_USER:-lawfirmai}:${POSTGRES_PASSWORD:-changeme}@postgres:5432/${POSTGRES_DB:-lawfirmai}}
    depends_on:
      postgres:
        condition: service_healthy

  frontend:
    image: ${ECR_REGISTRY}/lawfirmai-frontend:latest
    ports:
      - "80:80"
    depends_on:
      - api
```

### 배포 스크립트

**`deployment/deploy.sh`** - 자동 배포 스크립트

```bash
#!/bin/bash
# ECR 로그인
aws ecr get-login-password --region ap-northeast-2 | \
  docker login --username AWS --password-stdin $ECR_REGISTRY

# 최신 이미지 pull
docker pull $ECR_REGISTRY/lawfirmai-api:latest
docker pull $ECR_REGISTRY/lawfirmai-frontend:latest

# 기존 컨테이너 중지
docker-compose -f deployment/docker-compose.prod.yml down

# 새 컨테이너 시작
docker-compose -f deployment/docker-compose.prod.yml up -d

# PostgreSQL 초기화 (필요 시)
if [ -n "$POSTGRES_DB" ]; then
  docker-compose -f deployment/docker-compose.prod.yml exec api python scripts/database/init_postgresql.py
fi

# Health check
curl -f http://localhost:8000/health
```

---

## CI/CD 설정

### GitHub Actions 워크플로우

**`.github/workflows/deploy.yml`**

```yaml
name: Deploy to AWS

on:
  push:
    branches: [main]
  workflow_dispatch:

jobs:
  build-and-deploy:
    runs-on: ubuntu-latest
    
    steps:
      - uses: actions/checkout@v3
      
      - name: Configure AWS credentials
        uses: aws-actions/configure-aws-credentials@v2
        with:
          aws-access-key-id: ${{ secrets.AWS_ACCESS_KEY_ID }}
          aws-secret-access-key: ${{ secrets.AWS_SECRET_ACCESS_KEY }}
          aws-region: ${{ secrets.AWS_REGION }}
      
      - name: Login to Amazon ECR
        id: login-ecr
        uses: aws-actions/amazon-ecr-login@v1
      
      - name: Build and push API image
        env:
          ECR_REGISTRY: ${{ steps.login-ecr.outputs.registry }}
          ECR_REPOSITORY: lawfirmai-api
        run: |
          docker build -t $ECR_REGISTRY/$ECR_REPOSITORY:latest -f api/Dockerfile .
          docker push $ECR_REGISTRY/$ECR_REPOSITORY:latest
      
      - name: Build and push Frontend image
        env:
          ECR_REGISTRY: ${{ steps.login-ecr.outputs.registry }}
          ECR_REPOSITORY: lawfirmai-frontend
        run: |
          docker build -t $ECR_REGISTRY/$ECR_REPOSITORY:latest -f frontend/Dockerfile .
          docker push $ECR_REGISTRY/$ECR_REPOSITORY:latest
      
      - name: Deploy to EC2
        uses: appleboy/ssh-action@master
        with:
          host: ${{ secrets.EC2_HOST }}
          username: ubuntu
          key: ${{ secrets.EC2_SSH_KEY }}
          script: |
            cd /opt/lawfirmai
            ./deployment/deploy.sh
```

---

## 환경 변수 및 보안

### 환경 변수 설정

#### 로컬 개발 (`.env`)

```env
# API 설정
GOOGLE_API_KEY=your_google_api_key_here
LOG_LEVEL=DEBUG
DEBUG=true

# 데이터베이스 (SQLite)
DATABASE_URL=sqlite:///./data/lawfirm.db

# CORS 설정
CORS_ORIGINS=http://localhost:3000,http://127.0.0.1:3000
```

#### 개발 서버 (`.env.development`)

```env
# API 설정
GOOGLE_API_KEY=your_google_api_key_here
LOG_LEVEL=DEBUG
DEBUG=true

# 데이터베이스 (PostgreSQL)
DATABASE_URL=postgresql://lawfirmai:dev_password@postgres:5432/lawfirmai_dev
POSTGRES_HOST=postgres
POSTGRES_PORT=5432
POSTGRES_DB=lawfirmai_dev
POSTGRES_USER=lawfirmai
POSTGRES_PASSWORD=dev_password

# CORS 설정
CORS_ORIGINS=http://localhost:3000,http://127.0.0.1:3000
```

#### 운영 서버 (`.env.production`)

```env
# API 설정
GOOGLE_API_KEY=your_google_api_key_here
LOG_LEVEL=INFO
DEBUG=false

# 데이터베이스 (PostgreSQL)
DATABASE_URL=postgresql://lawfirmai:secure_password@postgres:5432/lawfirmai_prod
POSTGRES_HOST=postgres
POSTGRES_PORT=5432
POSTGRES_DB=lawfirmai_prod
POSTGRES_USER=lawfirmai
POSTGRES_PASSWORD=secure_password

# CORS 설정
CORS_ORIGINS=https://your-domain.com

# ECR 설정
ECR_REGISTRY=your_account_id.dkr.ecr.ap-northeast-2.amazonaws.com
```

### 보안 설정

#### 1. AWS Systems Manager Parameter Store 사용 (권장)

```bash
# 민감한 정보 저장
aws ssm put-parameter \
  --name "/lawfirmai/prod/GOOGLE_API_KEY" \
  --value "your_api_key" \
  --type "SecureString"

# 환경 변수에서 읽기
GOOGLE_API_KEY=$(aws ssm get-parameter \
  --name "/lawfirmai/prod/GOOGLE_API_KEY" \
  --with-decryption \
  --query 'Parameter.Value' \
  --output text)
```

#### 2. Nginx 보안 설정

**보안 헤더 추가:**
- `X-Frame-Options: DENY`
- `X-Content-Type-Options: nosniff`
- `X-XSS-Protection: 1; mode=block`
- `Referrer-Policy: strict-origin-when-cross-origin`
- `Permissions-Policy`
- `Strict-Transport-Security` (HTTPS 사용 시)

**자세한 내용:**
- [Nginx 보안 가이드](NGINX_SECURITY.md)

#### 3. SSL/TLS 인증서 설정

```bash
# Certbot 설치
sudo apt-get update
sudo apt-get install certbot python3-certbot-nginx

# SSL 인증서 발급
sudo certbot --nginx -d your-domain.com

# 자동 갱신 테스트
sudo certbot renew --dry-run
```

---

## 단계별 배포 절차

### Phase 1: 사전 준비 (1일)

1. **AWS 계정 및 IAM 설정**
   - AWS 계정 생성
   - IAM 사용자 생성 및 권한 부여
   - AWS CLI 설정

2. **GitHub 설정**
   - GitHub Secrets 설정
   - GitHub Actions 활성화

3. **도메인 설정** (선택사항)
   - 도메인 구매 또는 기존 도메인 확인
   - Route 53 호스팅 영역 생성

### Phase 2: AWS 인프라 생성 (1일)

1. **ECR 저장소 생성**
   ```bash
   aws ecr create-repository --repository-name lawfirmai-api --region ap-northeast-2
   aws ecr create-repository --repository-name lawfirmai-frontend --region ap-northeast-2
   ```

2. **EC2 인스턴스 생성**
   - AWS Console에서 인스턴스 생성
   - 보안 그룹 설정
   - 키 페어 생성

3. **EC2 초기 설정**
   ```bash
   ssh -i your-key.pem ubuntu@your-ec2-ip
   cd /opt/lawfirmai
   sudo bash deployment/setup_ec2.sh
   ```

### Phase 3: 첫 배포 (1일)

1. **환경 변수 설정**
   ```bash
   nano /opt/lawfirmai/.env
   # 환경 변수 입력
   ```

2. **Docker Compose 파일 복사**
   ```bash
   # GitHub에서 클론 또는 직접 생성
   git clone https://github.com/your-username/LawFirmAI.git /opt/lawfirmai
   ```

3. **PostgreSQL 초기화** (PostgreSQL 사용 시)
   ```bash
   docker-compose -f docker-compose.prod.yml up -d postgres
   docker-compose -f docker-compose.prod.yml exec api python scripts/database/init_postgresql.py
   ```

4. **첫 배포 실행**
   ```bash
   # 방법 1: GitHub Actions 사용
   git push origin main
   
   # 방법 2: 수동 배포
   ./deployment/deploy.sh
   ```

### Phase 4: 검증 및 최적화 (1일)

1. **Health Check 확인**
   ```bash
   curl http://localhost:8000/health
   curl http://your-ec2-ip
   ```

2. **SSL 인증서 설정** (도메인 사용 시)
   ```bash
   sudo certbot --nginx -d your-domain.com
   ```

3. **모니터링 설정**
   - CloudWatch 로그 그룹 생성
   - 알람 설정

4. **백업 설정**
   ```bash
   # Crontab 설정
   crontab -e
   # 매일 새벽 2시에 백업
   0 2 * * * /opt/lawfirmai/deployment/backup.sh
   ```

---

## 모니터링 및 백업

### 모니터링

#### CloudWatch 설정

```bash
# 로그 그룹 생성
aws logs create-log-group --log-group-name /lawfirmai/api
aws logs create-log-group --log-group-name /lawfirmai/frontend

# 메트릭 확인
aws cloudwatch get-metric-statistics \
  --namespace AWS/EC2 \
  --metric-name CPUUtilization \
  --dimensions Name=InstanceId,Value=i-xxxxx \
  --start-time 2024-01-01T00:00:00Z \
  --end-time 2024-01-02T00:00:00Z \
  --period 3600 \
  --statistics Average
```

#### 로컬 모니터링

```bash
# Docker 컨테이너 상태
docker-compose ps

# 리소스 사용량
docker stats

# 로그 확인
docker-compose logs -f --tail=100

# PostgreSQL 상태 (PostgreSQL 사용 시)
docker-compose exec postgres psql -U lawfirmai -d lawfirmai_prod -c "SELECT version();"
```

### 백업

#### 자동 백업 설정

```bash
# 백업 스크립트 실행 권한 부여
chmod +x /opt/lawfirmai/deployment/backup.sh
chmod +x /opt/lawfirmai/scripts/database/backup_postgresql.sh

# Crontab 설정
crontab -e

# 매일 새벽 2시에 백업
0 2 * * * /opt/lawfirmai/deployment/backup.sh

# PostgreSQL 사용 시
0 2 * * * /opt/lawfirmai/scripts/database/backup_postgresql.sh
```

#### 수동 백업

```bash
# SQLite 백업
sqlite3 /opt/lawfirmai/data/lawfirm.db ".backup /mnt/backups/lawfirm_$(date +%Y%m%d).db"

# PostgreSQL 백업
PGPASSWORD=password pg_dump -h postgres -U lawfirmai -d lawfirmai_prod -F c -f /mnt/backups/lawfirmai_$(date +%Y%m%d).dump
```

#### 백업 복구

```bash
# SQLite 복구
sqlite3 /opt/lawfirmai/data/lawfirm.db < /mnt/backups/lawfirm_20240101.db

# PostgreSQL 복구
PGPASSWORD=password pg_restore -h postgres -U lawfirmai -d lawfirmai_prod -c /mnt/backups/lawfirmai_20240101.dump
```

---

## 문제 해결

### 일반적인 문제

#### 1. Docker 이미지 Pull 실패

```bash
# ECR 로그인 확인
aws ecr get-login-password --region ap-northeast-2 | \
  docker login --username AWS --password-stdin \
  your_account_id.dkr.ecr.ap-northeast-2.amazonaws.com

# 권한 확인
aws ecr describe-repositories
```

#### 2. 컨테이너 시작 실패

```bash
# 로그 확인
docker-compose -f deployment/docker-compose.prod.yml logs api
docker-compose -f deployment/docker-compose.prod.yml logs postgres
docker-compose -f deployment/docker-compose.prod.yml logs frontend

# 환경 변수 확인
docker-compose -f deployment/docker-compose.prod.yml config

# PostgreSQL 상태 확인 (PostgreSQL 사용 시)
docker-compose -f deployment/docker-compose.prod.yml exec postgres pg_isready -U lawfirmai

# 컨테이너 재시작
docker-compose -f deployment/docker-compose.prod.yml restart
```

#### 3. 포트 충돌

```bash
# 포트 사용 확인
sudo netstat -tlnp | grep :8000
sudo netstat -tlnp | grep :80

# 프로세스 종료
sudo kill -9 <PID>
```

#### 4. 메모리 부족

```bash
# 메모리 사용량 확인
free -h
docker stats

# Docker 시스템 정리
docker system prune -a

# Swap 메모리 확인 (프리 티어)
swapon --show
```

#### 5. 데이터베이스 연결 문제

```bash
# PostgreSQL 연결 확인
docker-compose -f deployment/docker-compose.prod.yml exec postgres psql -U lawfirmai -d lawfirmai_prod -c "SELECT version();"

# 데이터베이스 목록 확인
docker-compose -f deployment/docker-compose.prod.yml exec postgres psql -U lawfirmai -c "\l"

# 테이블 목록 확인
docker-compose -f deployment/docker-compose.prod.yml exec postgres psql -U lawfirmai -d lawfirmai_prod -c "\dt"

# 연결 수 확인
docker-compose -f deployment/docker-compose.prod.yml exec postgres psql -U lawfirmai -d lawfirmai_prod -c "SELECT count(*) FROM pg_stat_activity;"
```

### 로그 확인

```bash
# CloudWatch 로그
aws logs tail /lawfirmai/api --follow

# Docker 로그
docker-compose logs -f --tail=100

# PostgreSQL 로그 (PostgreSQL 사용 시)
docker-compose logs postgres

# 시스템 로그
sudo journalctl -u docker -f
```

---

## 프리 티어 최적화

### 프리 티어 제한사항

- **인스턴스 타입**: t2.micro 또는 t3.micro만 무료
- **스토리지**: 30GB 이하
- **메모리**: 1GB
- **CPU**: 1 vCPU
- **사용 시간**: 750시간/월 무료
- **기간**: 신규 계정 12개월간

### 최적화 방법

#### 1. Swap 메모리 설정

```bash
# Swap 파일 생성 (2GB)
sudo fallocate -l 2G /swapfile
sudo chmod 600 /swapfile
sudo mkswap /swapfile
sudo swapon /swapfile

# 영구 설정
echo '/swapfile none swap sw 0 0' | sudo tee -a /etc/fstab
```

#### 2. 불필요한 서비스 비활성화

```bash
# snapd 비활성화
sudo systemctl disable snapd
sudo systemctl stop snapd

# unattended-upgrades 비활성화
sudo systemctl disable unattended-upgrades
```

#### 3. 리소스 제한 설정

**`deployment/docker-compose.prod.free-tier.yml`**

```yaml
services:
  api:
    deploy:
      resources:
        limits:
          memory: 512M
          cpus: '0.5'
        reservations:
          memory: 256M
          cpus: '0.25'
  
  postgres:
    deploy:
      resources:
        limits:
          memory: 256M
          cpus: '0.25'
        reservations:
          memory: 128M
          cpus: '0.1'
```

#### 4. 디스크 공간 최적화

```bash
# Docker 이미지 정리
docker system prune -a --volumes

# 로그 파일 정리
sudo journalctl --vacuum-time=7d

# 불필요한 패키지 제거
sudo apt-get autoremove -y
sudo apt-get autoclean
```

**자세한 내용:**
- [프리 티어 최적화 가이드](FREE_TIER_OPTIMIZATION.md)

---

## 비용 예상

### 프리 티어 구성

- **EC2**: $0/월 (12개월간)
- **ECR**: $0/월 (500MB 이하)
- **EBS**: $0/월 (30GB 이하)
- **데이터 전송**: $0/월 (15GB 이하)
- **총 비용**: $0-5/월 (데이터 전송 초과 시)

### 프로덕션 구성

- **EC2 (t3.large)**: $60-80/월
- **ECR**: $1-2/월
- **EBS (50GB)**: $4/월
- **데이터 전송**: $5-10/월
- **CloudWatch**: $5-10/월
- **총 비용**: $75-106/월

### 고가용성 구성

- **EC2 (다중 인스턴스)**: $120-160/월
- **로드 밸런서**: $20-30/월
- **기타 서비스**: $20-30/월
- **총 비용**: $160-220/월

---

## 참고 문서

### 배포 관련
- [배포 체크리스트](DEPLOYMENT_CHECKLIST.md) - 배포 전 확인사항
- [AWS 빠른 시작](QUICK_START_AWS.md) - 빠른 배포 가이드

### 데이터베이스 관련
- [PostgreSQL 마이그레이션 계획](POSTGRESQL_MIGRATION_PLAN.md) - PostgreSQL 마이그레이션 계획
- [PostgreSQL 설정 가이드](POSTGRESQL_SETUP_GUIDE.md) - PostgreSQL 설정 방법
- [데이터베이스 마이그레이션 가이드](DATABASE_MIGRATION_GUIDE.md) - 데이터 마이그레이션 방법

### 보안 및 최적화
- [Nginx 보안 가이드](NGINX_SECURITY.md) - Nginx 보안 설정
- [프리 티어 최적화 가이드](FREE_TIER_OPTIMIZATION.md) - 프리 티어 최적화 방법

---

## 다음 단계

1. **배포 체크리스트 확인**
   - [배포 체크리스트](DEPLOYMENT_CHECKLIST.md)를 따라 모든 항목 확인

2. **빠른 시작 가이드 따라하기**
   - [AWS 빠른 시작](QUICK_START_AWS.md)을 따라 첫 배포 시작

3. **모니터링 및 백업 설정**
   - CloudWatch 설정
   - 자동 백업 설정

4. **SSL 인증서 설정** (도메인 사용 시)
   - Let's Encrypt 인증서 발급

5. **성능 최적화**
   - 리소스 모니터링
   - 필요 시 인스턴스 타입 업그레이드

---

**배포 준비가 완료되었습니다!** 🚀

이 문서를 따라 단계별로 배포를 진행하세요. 문제가 발생하면 [문제 해결](#문제-해결) 섹션을 참조하거나 관련 문서를 확인하세요.

