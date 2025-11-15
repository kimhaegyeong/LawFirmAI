# 배포 체크리스트

> **⚠️ 참고**: 이 문서는 배포 체크리스트입니다. 전체 배포 과정은 **[최종 배포 가이드](DEPLOYMENT_FINAL.md)**를 참조하세요.

## 사전 준비사항

### AWS 계정 설정
- [ ] AWS 계정 생성 및 활성화
- [ ] IAM 사용자 생성 (프로그래밍 방식 액세스)
- [ ] IAM 사용자에 필요한 권한 부여:
  - [ ] EC2 (인스턴스 생성, 관리)
  - [ ] ECR (컨테이너 레지스트리)
  - [ ] CloudWatch (로깅)
  - [ ] S3 (백업, 선택사항)
  - [ ] Systems Manager Parameter Store (환경 변수, 선택사항)

### GitHub 설정
- [ ] GitHub 저장소 생성 또는 기존 저장소 확인
- [ ] GitHub Actions 활성화
- [ ] GitHub Secrets 설정:
  - [ ] `AWS_ACCESS_KEY_ID`
  - [ ] `AWS_SECRET_ACCESS_KEY`
  - [ ] `AWS_REGION` (예: `ap-northeast-2`)
  - [ ] `EC2_SSH_KEY` (EC2 SSH 프라이빗 키)
  - [ ] `EC2_HOST` (EC2 퍼블릭 IP 또는 도메인)
  - [ ] `GOOGLE_API_KEY` (Google AI API 키)

### 도메인 설정 (선택사항)
- [ ] 도메인 구매 또는 기존 도메인 확인
- [ ] Route 53 호스팅 영역 생성
- [ ] DNS 레코드 설정

## AWS 인프라 구성

### PostgreSQL (선택사항 - 개발/운영 서버)

- [ ] PostgreSQL 사용 여부 결정
- [ ] Docker Compose에 PostgreSQL 서비스 추가 확인
- [ ] PostgreSQL 환경 변수 설정:
  ```env
  POSTGRES_DB=lawfirmai_prod
  POSTGRES_USER=lawfirmai
  POSTGRES_PASSWORD=secure_password
  ```
- [ ] PostgreSQL 초기화 스크립트 실행
- [ ] 데이터 마이그레이션 (SQLite → PostgreSQL, 필요 시)

### ECR (Elastic Container Registry)
- [ ] API 이미지 저장소 생성:
  ```bash
  aws ecr create-repository --repository-name lawfirmai-api --region ap-northeast-2
  ```
- [ ] Frontend 이미지 저장소 생성:
  ```bash
  aws ecr create-repository --repository-name lawfirmai-frontend --region ap-northeast-2
  ```

### EC2 인스턴스

#### 프리 티어 구성 (신규 AWS 계정)
- [ ] EC2 인스턴스 생성:
  - [ ] AMI: Ubuntu 22.04 LTS (프리 티어 자격)
  - [ ] 인스턴스 타입: `t2.micro` 또는 `t3.micro` (프리 티어)
  - [ ] 키 페어 생성 또는 기존 사용
  - [ ] 보안 그룹 설정:
    - [ ] SSH (22): 내 IP만
    - [ ] HTTP (80): 0.0.0.0/0
    - [ ] HTTPS (443): 0.0.0.0/0
    - [ ] Custom TCP (8000): 내 IP만 (API)
  - [ ] 스토리지: 30GB GP2 SSD (프리 티어 제한)
  - [ ] User Data 스크립트 추가 (선택사항)
- [ ] 프리 티어 제한사항 확인:
  - [ ] 인스턴스 타입이 t2.micro 또는 t3.micro인지 확인
  - [ ] 스토리지가 30GB 이하인지 확인
  - [ ] 프리 티어 기간 확인 (12개월)
- [ ] Swap 메모리 설정 (프리 티어 필수)

#### 프로덕션 구성
- [ ] EC2 인스턴스 생성:
  - [ ] AMI: Ubuntu 22.04 LTS
  - [ ] 인스턴스 타입: `t3.large` 또는 `t3.xlarge`
  - [ ] 키 페어 생성 또는 기존 사용
  - [ ] 보안 그룹 설정:
    - [ ] SSH (22): 내 IP만
    - [ ] HTTP (80): 0.0.0.0/0
    - [ ] HTTPS (443): 0.0.0.0/0
    - [ ] Custom TCP (8000): 내 IP만 (API)
  - [ ] 스토리지: 50GB+ GP3 SSD
  - [ ] User Data 스크립트 추가 (선택사항)

### IAM 역할 (선택사항)
- [ ] EC2 인스턴스용 IAM 역할 생성
- [ ] ECR 접근 권한 부여
- [ ] Systems Manager Parameter Store 접근 권한 부여
- [ ] EC2 인스턴스에 IAM 역할 연결

## EC2 인스턴스 초기 설정

### SSH 접속
- [ ] SSH 키 권한 설정:
  ```bash
  chmod 400 your-key.pem
  ```
- [ ] EC2 인스턴스 SSH 접속:
  ```bash
  ssh -i your-key.pem ubuntu@your-ec2-ip
  ```

### 초기 설정 스크립트 실행
- [ ] `deployment/setup_ec2.sh` 다운로드 또는 복사
- [ ] 스크립트 실행 권한 부여:
  ```bash
  chmod +x deployment/setup_ec2.sh
  ```
- [ ] 스크립트 실행:
  ```bash
  ./deployment/setup_ec2.sh
  ```

### AWS CLI 설정
- [ ] AWS CLI 설치 확인:
  ```bash
  aws --version
  ```
- [ ] AWS 자격 증명 설정:
  ```bash
  aws configure
  ```

### 애플리케이션 디렉토리 설정
- [ ] 애플리케이션 디렉토리 생성:
  ```bash
  sudo mkdir -p /opt/lawfirmai
  sudo chown ubuntu:ubuntu /opt/lawfirmai
  ```
- [ ] 필요한 디렉토리 생성:
  ```bash
  mkdir -p /opt/lawfirmai/data
  mkdir -p /opt/lawfirmai/logs
  ```

### 환경 변수 설정
- [ ] `.env` 파일 생성:
  ```bash
  nano /opt/lawfirmai/.env
  ```
- [ ] 필수 환경 변수 설정:
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

### Docker Compose 파일 설정
- [ ] `deployment/docker-compose.prod.yml` 파일 복사 또는 생성
- [ ] PostgreSQL 서비스 포함 확인
- [ ] 환경 변수 확인
- [ ] PostgreSQL 초기화 스크립트 실행:
  ```bash
  docker-compose -f deployment/docker-compose.prod.yml exec api python scripts/database/init_postgresql.py
  ```

## 첫 배포

### GitHub Actions 트리거
- [ ] `main` 브랜치에 코드 푸시
- [ ] GitHub Actions 워크플로우 실행 확인
- [ ] 빌드 및 ECR 푸시 성공 확인

### EC2에서 배포
- [ ] ECR 로그인:
  ```bash
  aws ecr get-login-password --region ap-northeast-2 | \
    docker login --username AWS --password-stdin \
    your_account_id.dkr.ecr.ap-northeast-2.amazonaws.com
  ```
- [ ] 배포 스크립트 실행:
  ```bash
  cd /opt/lawfirmai
  ./deployment/deploy.sh
  ```
- [ ] 컨테이너 상태 확인:
  ```bash
  docker-compose -f deployment/docker-compose.prod.yml ps
  ```
- [ ] 로그 확인:
  ```bash
  docker-compose -f deployment/docker-compose.prod.yml logs -f
  ```

## Nginx 설정 (프론트엔드)

### Nginx 설정 파일 생성
- [ ] Nginx 설정 파일 생성:
  ```bash
  sudo nano /etc/nginx/sites-available/lawfirmai
  ```
- [ ] 설정 내용 추가 (도메인 또는 IP 기반)
- [ ] 심볼릭 링크 생성:
  ```bash
  sudo ln -s /etc/nginx/sites-available/lawfirmai /etc/nginx/sites-enabled/
  ```
- [ ] 기본 설정 제거:
  ```bash
  sudo rm /etc/nginx/sites-enabled/default
  ```
- [ ] Nginx 설정 테스트:
  ```bash
  sudo nginx -t
  ```
- [ ] Nginx 재시작:
  ```bash
  sudo systemctl restart nginx
  ```

## SSL 인증서 설정 (Let's Encrypt)

### Certbot 설치 및 설정
- [ ] Certbot으로 SSL 인증서 발급:
  ```bash
  sudo certbot --nginx -d your-domain.com
  ```
- [ ] 자동 갱신 테스트:
  ```bash
  sudo certbot renew --dry-run
  ```

## 배포 검증

### Health Check
- [ ] API Health Check:
  ```bash
  curl http://localhost:8000/health
  ```
- [ ] 프론트엔드 접속 확인:
  - [ ] 브라우저에서 `http://your-ec2-ip` 접속
  - [ ] 또는 `https://your-domain.com` 접속 (SSL 설정 시)

### API 문서 확인
- [ ] API 문서 접속:
  - [ ] `http://your-ec2-ip:8000/docs`
  - [ ] 또는 `https://your-domain.com/api/docs`

### 기능 테스트
- [ ] 프론트엔드 UI 정상 작동 확인
- [ ] 채팅 기능 테스트
- [ ] API 엔드포인트 테스트
- [ ] 데이터베이스 연결 확인:
  ```bash
  # PostgreSQL 사용 시
  docker-compose -f deployment/docker-compose.prod.yml exec postgres psql -U lawfirmai -d lawfirmai_prod -c "SELECT COUNT(*) FROM sessions;"
  
  # SQLite 사용 시
  sqlite3 /opt/lawfirmai/data/lawfirm.db "SELECT COUNT(*) FROM sessions;"
  ```

## 모니터링 설정

### CloudWatch 로그
- [ ] CloudWatch 로그 그룹 생성:
  ```bash
  aws logs create-log-group --log-group-name /lawfirmai/api
  aws logs create-log-group --log-group-name /lawfirmai/frontend
  ```
- [ ] Docker Compose에 CloudWatch 로그 드라이버 추가 (선택사항)

### CloudWatch 알람
- [ ] CPU 사용률 알람 설정
- [ ] 메모리 사용률 알람 설정
- [ ] 디스크 사용률 알람 설정

## 백업 설정

### 자동 백업
- [ ] 백업 스크립트 확인:
  ```bash
  chmod +x /opt/lawfirmai/deployment/backup.sh
  ```
- [ ] Crontab 설정 확인:
  ```bash
  crontab -l
  ```
- [ ] 수동 백업 테스트:
  ```bash
  /opt/lawfirmai/deployment/backup.sh
  ```

### S3 백업 (선택사항)
- [ ] S3 버킷 생성:
  ```bash
  aws s3 mb s3://lawfirmai-backups
  ```
- [ ] 백업 스크립트에 S3 업로드 추가

## 보안 체크리스트

### 네트워크 보안
- [ ] 보안 그룹 최소 권한 원칙 적용
- [ ] SSH 접근 제한 (내 IP만)
- [ ] API 포트 접근 제한 (필요 시)

### 애플리케이션 보안
- [ ] 환경 변수 안전하게 관리
- [ ] API 키 GitHub에 커밋되지 않았는지 확인
- [ ] SSL/TLS 인증서 설정
- [ ] CORS 설정 확인

### 시스템 보안
- [ ] 정기적인 시스템 업데이트
- [ ] 방화벽 설정 확인
- [ ] 불필요한 포트 닫기

## 성능 최적화

### 리소스 모니터링
- [ ] CPU 사용률 모니터링
- [ ] 메모리 사용률 모니터링
- [ ] 디스크 사용률 모니터링
- [ ] 네트워크 트래픽 모니터링

### 최적화 조치
- [ ] 인스턴스 타입 조정 (필요 시)
- [ ] 스토리지 타입 최적화
- [ ] 캐싱 전략 적용
- [ ] CDN 설정 (선택사항)

## 문제 해결

### 일반적인 문제
- [ ] Docker 이미지 Pull 실패 해결
- [ ] 컨테이너 시작 실패 해결
- [ ] 포트 충돌 해결
- [ ] 메모리 부족 해결

### 로그 확인
- [ ] Docker 로그 확인
- [ ] Nginx 로그 확인
- [ ] 시스템 로그 확인
- [ ] CloudWatch 로그 확인

## 문서화

### 배포 문서
- [ ] 배포 가이드 작성 완료
- [ ] 문제 해결 가이드 작성
- [ ] 운영 매뉴얼 작성

### 코드 문서
- [ ] README 업데이트
- [ ] API 문서 업데이트
- [ ] 환경 변수 문서 업데이트

## 완료 확인

- [ ] 모든 체크리스트 항목 완료
- [ ] 프로덕션 환경 정상 작동 확인
- [ ] 모니터링 시스템 정상 작동 확인
- [ ] 백업 시스템 정상 작동 확인
- [ ] 팀원에게 배포 완료 알림

---

**배포 완료 후:**
- 정기적인 모니터링 및 로그 확인
- 정기적인 백업 확인
- 정기적인 보안 업데이트
- 정기적인 성능 최적화 검토

