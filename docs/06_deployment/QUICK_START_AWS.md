# AWS 배포 빠른 시작 가이드

> **⚠️ 참고**: 이 문서는 빠른 시작 가이드입니다. 전체 배포 과정은 **[최종 배포 가이드](DEPLOYMENT_FINAL.md)**를 참조하세요.

이 가이드는 LawFirmAI를 AWS에 빠르게 배포하는 방법을 단계별로 설명합니다.

## 사전 준비사항

1. **AWS 계정** 및 IAM 사용자 (프로그래밍 방식 액세스)
2. **GitHub 저장소** 및 GitHub Actions 활성화
3. **도메인** (선택사항, IP로도 접속 가능)

## 1단계: AWS 인프라 생성

### ECR 저장소 생성

```bash
# AWS CLI 설정
aws configure

# ECR 저장소 생성
aws ecr create-repository --repository-name lawfirmai-api --region ap-northeast-2
aws ecr create-repository --repository-name lawfirmai-frontend --region ap-northeast-2
```

### EC2 인스턴스 생성

#### 프리 티어 구성 (신규 AWS 계정 - 무료)

1. **AWS Console** 접속 → **EC2** → **Launch Instance**
2. 설정:
   - **AMI**: Ubuntu 22.04 LTS (프리 티어 자격)
   - **Instance type**: `t2.micro` 또는 `t3.micro` (프리 티어)
   - **Key pair**: 새로 생성 또는 기존 사용
   - **Network settings**: 
     - 퍼블릭 IP 자동 할당 활성화
     - 보안 그룹 생성:
       - SSH (22): 내 IP만
       - HTTP (80): 0.0.0.0/0
       - HTTPS (443): 0.0.0.0/0
   - **Storage**: 30GB GP2 SSD (프리 티어 제한)
3. **Launch Instance** 클릭

**프리 티어 주의사항:**
- 인스턴스 타입은 반드시 `t2.micro` 또는 `t3.micro`만 선택
- 스토리지는 30GB 이하로 설정
- 월 750시간 무료 (1개 인스턴스 기준 24시간 운영 가능)

#### 프로덕션 구성 (유료)

1. **AWS Console** 접속 → **EC2** → **Launch Instance**
2. 설정:
   - **AMI**: Ubuntu 22.04 LTS
   - **Instance type**: `t3.large`
   - **Key pair**: 새로 생성 또는 기존 사용
   - **Network settings**: 
     - 퍼블릭 IP 자동 할당 활성화
     - 보안 그룹 생성:
       - SSH (22): 내 IP만
       - HTTP (80): 0.0.0.0/0
       - HTTPS (443): 0.0.0.0/0
   - **Storage**: 50GB GP3 SSD
3. **Launch Instance** 클릭

## 2단계: GitHub Secrets 설정

GitHub 저장소 → **Settings** → **Secrets and variables** → **Actions**에서 다음 Secrets 추가:

```
AWS_ACCESS_KEY_ID          # AWS IAM 사용자 액세스 키
AWS_SECRET_ACCESS_KEY      # AWS IAM 사용자 시크릿 키
AWS_REGION                 # ap-northeast-2
EC2_SSH_KEY                # EC2 SSH 프라이빗 키 (.pem 파일 내용)
EC2_HOST                   # EC2 퍼블릭 IP 또는 도메인
GOOGLE_API_KEY             # Google AI API 키
```

## 3단계: EC2 인스턴스 초기 설정

### SSH 접속

```bash
# SSH 키 권한 설정
chmod 400 your-key.pem

# EC2 인스턴스 접속
ssh -i your-key.pem ubuntu@your-ec2-ip
```

### 초기 설정 스크립트 실행

```bash
# 스크립트 다운로드 (GitHub에서)
wget https://raw.githubusercontent.com/your-username/LawFirmAI/main/deployment/setup_ec2.sh

# 또는 직접 생성
nano setup_ec2.sh
# (deployment/setup_ec2.sh 내용 복사)

# 실행 권한 부여 및 실행
chmod +x setup_ec2.sh
./setup_ec2.sh
```

### AWS CLI 설정

```bash
aws configure
# AWS Access Key ID 입력
# AWS Secret Access Key 입력
# Default region: ap-northeast-2
# Default output format: json
```

## 4단계: 환경 변수 설정

```bash
# 애플리케이션 디렉토리로 이동
cd /opt/lawfirmai

# .env 파일 생성
nano .env
```

`.env` 파일 내용:

```env
GOOGLE_API_KEY=your_google_api_key_here
LOG_LEVEL=INFO

# PostgreSQL 설정 (개발/운영 서버)
DATABASE_URL=postgresql://lawfirmai:secure_password@postgres:5432/lawfirmai_prod
POSTGRES_HOST=postgres
POSTGRES_PORT=5432
POSTGRES_DB=lawfirmai_prod
POSTGRES_USER=lawfirmai
POSTGRES_PASSWORD=secure_password

# 또는 SQLite 사용 (로컬 개발)
# DATABASE_URL=sqlite:///./data/lawfirm.db

ECR_REGISTRY=your_account_id.dkr.ecr.ap-northeast-2.amazonaws.com
```

**ECR_REGISTRY 확인 방법:**
```bash
aws ecr describe-repositories --region ap-northeast-2
```

## 5단계: Docker Compose 파일 설정

```bash
# docker-compose.prod.yml 파일 생성
nano /opt/lawfirmai/deployment/docker-compose.prod.yml
```

프로젝트의 `deployment/docker-compose.prod.yml` 파일 내용을 복사하거나, GitHub에서 다운로드:

```bash
mkdir -p /opt/lawfirmai/deployment
wget https://raw.githubusercontent.com/your-username/LawFirmAI/main/deployment/docker-compose.prod.yml -O /opt/lawfirmai/deployment/docker-compose.prod.yml
```

## 6단계: 첫 배포

### 방법 1: GitHub Actions 사용 (권장)

1. **코드를 main 브랜치에 푸시:**
   ```bash
   git add .
   git commit -m "Initial deployment setup"
   git push origin main
   ```

2. **GitHub Actions에서 자동 배포 확인:**
   - GitHub 저장소 → **Actions** 탭
   - 워크플로우 실행 상태 확인

### 방법 2: 수동 배포

```bash
# EC2 인스턴스에 SSH 접속
ssh -i your-key.pem ubuntu@your-ec2-ip

# ECR 로그인
aws ecr get-login-password --region ap-northeast-2 | \
  docker login --username AWS --password-stdin \
  your_account_id.dkr.ecr.ap-northeast-2.amazonaws.com

# 배포 스크립트 실행
cd /opt/lawfirmai
./deployment/deploy.sh
```

## 7단계: Nginx 설정

```bash
# Nginx 설정 파일 생성
sudo nano /etc/nginx/sites-available/lawfirmai
```

설정 내용:

```nginx
server {
    listen 80;
    server_name your-ec2-ip;  # 또는 your-domain.com

    location / {
        proxy_pass http://localhost:80;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto $scheme;
    }

    location /api {
        proxy_pass http://localhost:8000;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto $scheme;
    }
}
```

활성화:

```bash
# 심볼릭 링크 생성
sudo ln -s /etc/nginx/sites-available/lawfirmai /etc/nginx/sites-enabled/

# 기본 설정 제거
sudo rm /etc/nginx/sites-enabled/default

# Nginx 설정 테스트
sudo nginx -t

# Nginx 재시작
sudo systemctl restart nginx
```

## 8단계: SSL 인증서 설정 (선택사항)

도메인이 있는 경우:

```bash
# Certbot으로 SSL 인증서 발급
sudo certbot --nginx -d your-domain.com

# 자동 갱신 테스트
sudo certbot renew --dry-run
```

## 9단계: 배포 확인

### Health Check

```bash
# API Health Check
curl http://localhost:8000/health

# 또는 EC2 퍼블릭 IP 사용
curl http://your-ec2-ip:8000/health
```

### 브라우저에서 확인

- **프론트엔드**: `http://your-ec2-ip` 또는 `https://your-domain.com`
- **API 문서**: `http://your-ec2-ip:8000/docs` 또는 `https://your-domain.com/api/docs`

## 문제 해결

### 컨테이너가 시작되지 않는 경우

```bash
# 로그 확인
cd /opt/lawfirmai
docker-compose -f deployment/docker-compose.prod.yml logs

# 컨테이너 상태 확인
docker-compose -f deployment/docker-compose.prod.yml ps

# 컨테이너 재시작
docker-compose -f deployment/docker-compose.prod.yml restart
```

### 이미지 Pull 실패

```bash
# ECR 로그인 확인
aws ecr get-login-password --region ap-northeast-2 | \
  docker login --username AWS --password-stdin \
  your_account_id.dkr.ecr.ap-northeast-2.amazonaws.com

# 권한 확인
aws ecr describe-repositories
```

### 포트 충돌

```bash
# 포트 사용 확인
sudo netstat -tlnp | grep :8000
sudo netstat -tlnp | grep :80

# 프로세스 종료
sudo kill -9 <PID>
```

## 다음 단계

1. **모니터링 설정**: CloudWatch 로그 및 알람 설정
2. **백업 설정**: 자동 백업 스크립트 설정
3. **고가용성**: 여러 AZ에 인스턴스 배포
4. **로드 밸런서**: Application Load Balancer 추가
5. **Auto Scaling**: 트래픽에 따른 자동 스케일링

## 참고 문서

- [최종 배포 가이드](DEPLOYMENT_FINAL.md)
- [배포 체크리스트](DEPLOYMENT_CHECKLIST.md)
- [문제 해결 가이드](DEPLOYMENT_FINAL.md#문제-해결)

---

**도움이 필요하신가요?** GitHub Issues에 문의하세요.

