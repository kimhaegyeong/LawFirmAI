# LawFirmAI 배포 가이드

## 개요

이 가이드는 LawFirmAI를 다양한 환경에 배포하는 방법을 설명합니다. Phase 1-6이 완료된 지능형 대화 시스템과 성능 최적화된 의미적 검색 시스템의 실제 배포 방법을 다룹니다.

## 목차

1. [시스템 요구사항](#시스템-요구사항)
2. [로컬 개발 환경 설정](#로컬-개발-환경-설정)
3. [HuggingFace Spaces 배포](#huggingface-spaces-배포)
4. [Docker 배포](#docker-배포)
5. [환경 변수 설정](#환경-변수-설정)
6. [모니터링 및 로깅](#모니터링-및-로깅)
7. [문제 해결](#문제-해결)

## 시스템 요구사항

### 소프트웨어 요구사항
- **Python**: 3.9+
- **Docker**: 20.10+ (Docker 배포 시)
- **Git**: 2.30+
- **메모리**: 최소 4GB RAM (권장 8GB+)
- **저장공간**: 최소 10GB 여유 공간

### 하드웨어 요구사항
- **CPU**: 4코어 이상 (권장 8코어+)
- **GPU**: 선택사항 (CUDA 지원 시 성능 향상)
- **네트워크**: 안정적인 인터넷 연결

### Phase별 요구사항

#### Phase 1-3: 지능형 대화 시스템
- **메모리**: 최소 4GB RAM
- **저장공간**: 5GB (데이터베이스 + 로그)
- **네트워크**: 안정적인 연결 (세션 동기화)

#### Phase 5: 성능 최적화 시스템
- **메모리**: 최소 6GB RAM (캐싱 시스템)
- **CPU**: 4코어 이상 (병렬 처리)
- **저장공간**: 8GB (벡터 인덱스 + 캐시)

#### Phase 6: 의미적 검색 시스템
- **메모리**: 최소 8GB RAM (FAISS 인덱스)
- **저장공간**: 15GB (벡터 임베딩 + 메타데이터)
- **CPU**: 6코어 이상 (벡터 검색)

## 로컬 개발 환경 설정

### 1. 저장소 클론

```bash
git clone https://github.com/lawfirmai/lawfirmai.git
cd lawfirmai
```

### 2. 가상환경 설정

```bash
# Python 가상환경 생성
python -m venv venv

# 가상환경 활성화
# Windows
venv\Scripts\activate
# macOS/Linux
source venv/bin/activate
```

### 3. 의존성 설치

```bash
# Streamlit 앱 의존성 설치
pip install streamlit
pip install -r requirements.txt
```

### 4. 환경 변수 설정

`.env` 파일을 생성하고 다음 내용을 추가하세요:

```env
# LangGraph 설정
USE_LANGGRAPH=true

# 데이터베이스 설정
DATABASE_URL=sqlite:///./data/lawfirm.db

# 모델 설정
MODEL_PATH=./models
USE_GPU=false

# API 설정 (선택사항)
OPENAI_API_KEY=your_openai_api_key_here
GOOGLE_API_KEY=your_google_api_key_here

# 로깅 설정
LOG_LEVEL=INFO
LOG_FILE=./logs/lawfirm.log

# 성능 설정 (Phase 5)
MAX_CACHE_SIZE=1000
CACHE_TTL=3600
MEMORY_LIMIT_MB=2048

# 의미적 검색 설정 (Phase 6)
SEMANTIC_SEARCH_ENABLED=true
FAISS_INDEX_PATH=./data/embeddings/ml_enhanced_ko_sroberta/ml_enhanced_faiss_index.faiss
EMBEDDING_MODEL=ko-sroberta-multitask

# HuggingFace 설정 (선택사항)
HUGGINGFACE_TOKEN=your_hf_token_here
```

### 5. 데이터베이스 초기화

```bash
# 데이터베이스 디렉토리 생성
mkdir -p data logs runtime

# 데이터베이스 초기화 (자동으로 생성됨)
python -c "from source.data.database import DatabaseManager; db = DatabaseManager(); print('Database initialized')"
```

### 6. 애플리케이션 실행

```bash
# Streamlit 앱 실행
streamlit run streamlit_app.py
```

웹 브라우저에서 `http://localhost:8501`에 접속하여 LawFirmAI를 사용할 수 있습니다.

## HuggingFace Spaces 배포 (권장)

### 1. HuggingFace 계정 및 Space 생성

1. [HuggingFace](https://huggingface.co/) 계정 생성
2. 새로운 Space 생성:
   - **Name**: `lawfirm-ai`
   - **License**: `MIT`
   - **SDK**: `Docker`
   - **Hardware**: `CPU basic` (무료) 또는 `CPU upgrade` (유료)

### 2. Docker 설정

`Dockerfile` 파일을 생성하세요:

```dockerfile
FROM python:3.9-slim

WORKDIR /app

# 시스템 패키지 설치
RUN apt-get update && apt-get install -y \
    git \
    && rm -rf /var/lib/apt/lists/*

# Python 의존성 설치
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
RUN pip install streamlit

# 애플리케이션 코드 복사
COPY streamlit_app.py .
COPY source/ ./source/

# 필요한 디렉토리 생성
RUN mkdir -p data logs runtime

# 환경 변수 설정
ENV PYTHONPATH=/app

# 포트 설정
EXPOSE 8501

# 애플리케이션 실행
CMD ["streamlit", "run", "streamlit_app.py", "--server.port=8501", "--server.address=0.0.0.0"]
```

### 3. Space 설정

HuggingFace Space의 `README.md` 파일에 다음 내용을 추가:

```markdown
---
title: LawFirmAI - 법률 AI 어시스턴트
emoji: ⚖️
colorFrom: blue
colorTo: purple
sdk: docker
pinned: false
license: mit
app_port: 8501
---

# LawFirmAI - 법률 AI 어시스턴트

지능형 법률 AI 어시스턴트입니다.

## 주요 기능

- 법령 조문 해석
- 판례 검색 및 분석
- 계약서 검토
- 법률 절차 안내
- 개인화된 답변
```

### 4. 배포 및 테스트

1. 코드를 HuggingFace Space에 푸시
2. 자동 빌드 및 배포 확인
3. `https://huggingface.co/spaces/your-username/lawfirm-ai`에서 접속 테스트

## Docker 배포

### 1. 로컬 Docker 배포

```bash
# Streamlit Docker 이미지 빌드
docker build -t lawfirm-ai-streamlit .

# 컨테이너 실행
docker run -p 8501:8501 lawfirm-ai-streamlit
```

### 2. Docker Compose 배포

`docker-compose.yml` 파일을 생성하세요:

```yaml
version: '3.8'

services:
  lawfirm-ai:
    build: .
    ports:
      - "8501:8501"
    environment:
      - LOG_LEVEL=INFO
    volumes:
      - ./data:/app/data
      - ./logs:/app/logs
    restart: unless-stopped
```

실행:
```bash
docker-compose up -d
```

### 3. 프로덕션 Docker 배포

```bash
# 프로덕션용 이미지 빌드
docker build -t lawfirm-ai:latest .

# 프로덕션 컨테이너 실행
docker run -d \
  --name lawfirm-ai \
  -p 8501:8501 \
  -e LOG_LEVEL=INFO \
  -v /path/to/data:/app/data \
  -v /path/to/logs:/app/logs \
  --restart unless-stopped \
  lawfirm-ai:latest
```

## 환경 변수 설정

### 필수 환경 변수

```env
# LangGraph 활성화 (필수)
USE_LANGGRAPH=true

# 데이터베이스 설정
DATABASE_URL=sqlite:///./data/lawfirm.db

# 로깅 설정
LOG_LEVEL=INFO
LOG_FILE=./logs/lawfirm.log
```

### 선택적 환경 변수

```env
# API 키 (선택사항)
OPENAI_API_KEY=your_openai_api_key_here
GOOGLE_API_KEY=your_google_api_key_here

# 성능 설정
MAX_CACHE_SIZE=1000
CACHE_TTL=3600
MEMORY_LIMIT_MB=2048

# HuggingFace 설정
HUGGINGFACE_TOKEN=your_hf_token_here
```

### 환경별 설정

#### 개발 환경
```env
USE_LANGGRAPH=true
LOG_LEVEL=DEBUG
DEBUG=true
```

#### 프로덕션 환경
```env
USE_LANGGRAPH=true
LOG_LEVEL=INFO
DEBUG=false
MEMORY_LIMIT_MB=4096
```

## 모니터링 및 로깅

### 1. 로그 모니터링

```bash
# 실시간 로그 확인
tail -f logs/lawfirm.log

# 에러 로그만 확인
grep "ERROR" logs/lawfirm.log

# 특정 시간대 로그 확인
grep "2024-12-20" logs/lawfirm.log
```

### 2. 성능 모니터링

Gradio UI의 **⚙️ 고급 설정** 탭에서 실시간 모니터링:

- **메모리 사용량**: 실시간 메모리 모니터링
- **CPU 사용률**: 시스템 리소스 사용량
- **응답 시간**: 평균/최대 응답 시간
- **캐시 성능**: 캐시 히트율 및 효율성

### 3. 시스템 상태 확인

```bash
# 프로세스 확인
ps aux | grep python

# 메모리 사용량 확인
free -h

# 디스크 사용량 확인
df -h

# 네트워크 포트 확인
netstat -tlnp | grep 7861
```

### 4. Prometheus + Grafana 모니터링

```bash
# 모니터링 스택 실행
cd monitoring
docker-compose up -d

# Grafana 대시보드 접속
# http://localhost:3000 (admin/admin)
```

## 문제 해결

### 일반적인 문제

#### 1. 포트 충돌
```bash
# 포트 사용 중인 프로세스 확인
lsof -i :7861

# 프로세스 종료
kill -9 <PID>
```

#### 2. 메모리 부족
```bash
# 메모리 사용량 확인
free -h

# 메모리 정리
sudo sync && sudo echo 3 > /proc/sys/vm/drop_caches
```

#### 3. 데이터베이스 오류
```bash
# 데이터베이스 파일 권한 확인
ls -la data/lawfirm.db

# 데이터베이스 재생성
rm data/lawfirm.db
python -c "from source.data.database import DatabaseManager; db = DatabaseManager()"
```

#### 4. 모델 로딩 실패
```bash
# 모델 디렉토리 확인
ls -la models/

# 모델 재다운로드
python -c "from source.models.model_manager import LegalModelManager; mm = LegalModelManager()"
```

### 성능 최적화

#### 1. 메모리 최적화
- **⚙️ 고급 설정** 탭에서 **메모리 사용량** 모니터링
- 메모리 사용량이 높으면 브라우저 새로고침
- 캐시 크기 조정

#### 2. 응답 속도 최적화
- **⚙️ 고급 설정** 탭에서 **응답 모드**를 **빠른** 모드로 변경
- 캐시 히트율 확인 및 최적화
- 불필요한 Phase 기능 비활성화

#### 3. 시스템 리소스 최적화
- CPU 사용률 모니터링
- 디스크 I/O 최적화
- 네트워크 대역폭 관리

### 로그 분석

#### 1. 에러 로그 분석
```bash
# 에러 로그 확인
grep "ERROR" logs/lawfirm.log | tail -20

# 경고 로그 확인
grep "WARNING" logs/lawfirm.log | tail -20
```

#### 2. 성능 로그 분석
```bash
# 응답 시간 분석
grep "processing_time" logs/lawfirm.log | awk '{print $NF}' | sort -n

# 메모리 사용량 분석
grep "memory_usage" logs/lawfirm.log | tail -10
```

### 백업 및 복구

#### 1. 데이터 백업
```bash
# 데이터베이스 백업
cp data/lawfirm.db data/backup/lawfirm_$(date +%Y%m%d_%H%M%S).db

# 설정 파일 백업
cp .env data/backup/env_$(date +%Y%m%d_%H%M%S).env
```

#### 2. 복구 절차
```bash
# 데이터베이스 복구
cp data/backup/lawfirm_20241220_143022.db data/lawfirm.db

# 설정 복구
cp data/backup/env_20241220_143022.env .env
```

## 배포 체크리스트

### 배포 전 확인사항

- [ ] 모든 의존성이 설치되었는가?
- [ ] 환경 변수가 올바르게 설정되었는가?
- [ ] 데이터베이스가 초기화되었는가?
- [ ] 로그 디렉토리가 생성되었는가?
- [ ] 포트가 사용 가능한가?

### 배포 후 확인사항

- [ ] 애플리케이션이 정상적으로 시작되었는가?
- [ ] 웹 인터페이스에 접속할 수 있는가?
- [ ] 모든 탭이 정상적으로 작동하는가?
- [ ] Phase 1-3 기능이 활성화되었는가?
- [ ] 로그가 정상적으로 생성되는가?

### 성능 확인사항

- [ ] 응답 시간이 적절한가? (< 5초)
- [ ] 메모리 사용량이 정상적인가? (< 4GB)
- [ ] 캐시가 정상적으로 작동하는가?
- [ ] 에러가 발생하지 않는가?

---


## Docker 배포

### 1. Dockerfile 생성

**gradio/Dockerfile**:
```dockerfile
FROM python:3.9-slim

WORKDIR /app

# 시스템 패키지 설치
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    && rm -rf /var/lib/apt/lists/*

# Python 의존성 설치
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# 애플리케이션 코드 복사
COPY . .

# 데이터 디렉토리 생성
RUN mkdir -p data logs

# 포트 노출
EXPOSE 7860

# 애플리케이션 실행
CMD ["python", "gradio/app.py"]
```

**api/Dockerfile**:
```dockerfile
FROM python:3.9-slim

WORKDIR /app

# 시스템 패키지 설치
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    && rm -rf /var/lib/apt/lists/*

# Python 의존성 설치
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# 애플리케이션 코드 복사
COPY . .

# 데이터 디렉토리 생성
RUN mkdir -p data logs

# 포트 노출
EXPOSE 8000

# 애플리케이션 실행
CMD ["python", "api/main.py"]
```

### 2. Docker Compose 설정

**docker-compose.yml**:
```yaml
version: '3.8'

services:
  gradio-app:
    build: ./gradio
    ports:
      - "7860:7860"
    environment:
      - DATABASE_URL=sqlite:///./data/lawfirm.db
      - LOG_LEVEL=INFO
    volumes:
      - ./data:/app/data
      - ./logs:/app/logs
    restart: unless-stopped

  api-server:
    build: ./api
    ports:
      - "8000:8000"
    environment:
      - DATABASE_URL=sqlite:///./data/lawfirm.db
      - LOG_LEVEL=INFO
    volumes:
      - ./data:/app/data
      - ./logs:/app/logs
    restart: unless-stopped

  nginx:
    image: nginx:alpine
    ports:
      - "80:80"
      - "443:443"
    volumes:
      - ./nginx.conf:/etc/nginx/nginx.conf
      - ./ssl:/etc/nginx/ssl
    depends_on:
      - gradio-app
      - api-server
    restart: unless-stopped
```

### 3. Docker 배포 실행

```bash
# Docker 이미지 빌드
docker-compose build

# 서비스 시작
docker-compose up -d

# 로그 확인
docker-compose logs -f

# 서비스 중지
docker-compose down
```

## HuggingFace Spaces 배포

### 1. Space 생성

1. [HuggingFace Spaces](https://huggingface.co/spaces)에 로그인
2. **Create new Space** 클릭
3. 다음 정보 입력:
   - **Space name**: lawfirmai
   - **License**: MIT
   - **SDK**: Gradio
   - **Hardware**: CPU Basic (또는 GPU)

### 2. 파일 업로드

다음 파일들을 Space에 업로드하세요:

```
lawfirmai/
├── app.py                 # Gradio 앱 메인 파일
├── requirements.txt       # Python 의존성
├── README.md             # Space 설명
├── source/               # 소스 코드
├── data/                 # 데이터 파일
└── .env                  # 환경 변수 (선택사항)
```

### 3. requirements.txt

```txt
gradio>=4.0.0
fastapi>=0.100.0
uvicorn>=0.20.0
sqlite3
psutil>=5.9.0
numpy>=1.24.0
pandas>=2.0.0
scikit-learn>=1.3.0
transformers>=4.30.0
torch>=2.0.0
```

### 4. README.md

```markdown
---
title: LawFirmAI
emoji: ⚖️
colorFrom: blue
colorTo: purple
sdk: gradio
sdk_version: 4.0.0
app_file: app.py
pinned: false
license: mit
---

# LawFirmAI - 법률 AI 어시스턴트

LawFirmAI는 법률 관련 질문에 답변하는 지능형 AI 어시스턴트입니다.

## 기능

- 법령 조문 해석
- 판례 검색 및 분석
- 계약서 검토
- 법률 절차 안내
- 개인화된 답변

## 사용법

1. 질문을 입력하세요
2. AI가 답변을 제공합니다
3. 추가 질문을 통해 더 자세한 정보를 얻으세요

## 지원

- 이메일: support@lawfirmai.com
- GitHub: https://github.com/lawfirmai/lawfirmai
```

### 5. 배포 확인

1. Space가 성공적으로 빌드되었는지 확인
2. 앱이 정상적으로 실행되는지 테스트
3. 로그에서 오류가 없는지 확인

## 클라우드 배포

### AWS 배포

#### 1. EC2 인스턴스 설정

```bash
# Ubuntu 20.04 LTS 인스턴스 생성
# t3.medium 이상 권장

# 시스템 업데이트
sudo apt update && sudo apt upgrade -y

# Python 3.9 설치
sudo apt install python3.9 python3.9-pip python3.9-venv -y

# Git 설치
sudo apt install git -y

# 애플리케이션 클론
git clone https://github.com/lawfirmai/lawfirmai.git
cd lawfirmai

# 가상환경 설정
python3.9 -m venv venv
source venv/bin/activate

# 의존성 설치
pip install -r requirements.txt
```

#### 2. Nginx 설정

```bash
# Nginx 설치
sudo apt install nginx -y

# 설정 파일 생성
sudo nano /etc/nginx/sites-available/lawfirmai
```

**/etc/nginx/sites-available/lawfirmai**:
```nginx
server {
    listen 80;
    server_name your-domain.com;

    location / {
        proxy_pass http://127.0.0.1:7860;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto $scheme;
    }

    location /api {
        proxy_pass http://127.0.0.1:8000;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto $scheme;
    }
}
```

#### 3. Systemd 서비스 설정

```bash
# 서비스 파일 생성
sudo nano /etc/systemd/system/lawfirmai.service
```

**/etc/systemd/system/lawfirmai.service**:
```ini
[Unit]
Description=LawFirmAI Application
After=network.target

[Service]
Type=simple
User=ubuntu
WorkingDirectory=/home/ubuntu/lawfirmai
Environment=PATH=/home/ubuntu/lawfirmai/venv/bin
ExecStart=/home/ubuntu/lawfirmai/venv/bin/python gradio/app.py
Restart=always

[Install]
WantedBy=multi-user.target
```

#### 4. 서비스 시작

```bash
# Nginx 설정 활성화
sudo ln -s /etc/nginx/sites-available/lawfirmai /etc/nginx/sites-enabled/
sudo nginx -t
sudo systemctl restart nginx

# LawFirmAI 서비스 시작
sudo systemctl daemon-reload
sudo systemctl enable lawfirmai
sudo systemctl start lawfirmai

# 상태 확인
sudo systemctl status lawfirmai
```

### Google Cloud Platform 배포

#### 1. Cloud Run 배포

```bash
# Google Cloud SDK 설치
curl https://sdk.cloud.google.com | bash
exec -l $SHELL

# 프로젝트 설정
gcloud config set project your-project-id

# Docker 이미지 빌드
gcloud builds submit --tag gcr.io/your-project-id/lawfirmai

# Cloud Run 배포
gcloud run deploy lawfirmai \
  --image gcr.io/your-project-id/lawfirmai \
  --platform managed \
  --region asia-northeast1 \
  --allow-unauthenticated \
  --memory 2Gi \
  --cpu 2
```

### Azure 배포

#### 1. Container Instances 배포

```bash
# Azure CLI 설치
curl -sL https://aka.ms/InstallAzureCLIDeb | sudo bash

# 로그인
az login

# 리소스 그룹 생성
az group create --name lawfirmai-rg --location eastus

# Container Instance 배포
az container create \
  --resource-group lawfirmai-rg \
  --name lawfirmai \
  --image your-registry/lawfirmai:latest \
  --cpu 2 \
  --memory 4 \
  --ports 7860 \
  --environment-variables \
    DATABASE_URL=sqlite:///./data/lawfirm.db \
    LOG_LEVEL=INFO
```

## 환경 변수 설정

### 필수 환경 변수

```env
# 데이터베이스
DATABASE_URL=sqlite:///./data/lawfirm.db

# 보안
SECRET_KEY=your-secret-key-here
API_KEY=your-api-key-here

# 로깅
LOG_LEVEL=INFO
LOG_FILE=./logs/lawfirm.log
```

### 선택적 환경 변수

```env
# 모델 설정
MODEL_PATH=./models
USE_GPU=false
MODEL_CACHE_DIR=./cache

# 성능 설정
MAX_CACHE_SIZE=1000
CACHE_TTL=3600
MEMORY_LIMIT_MB=2048
MAX_CONCURRENT_REQUESTS=10

# 외부 서비스
HUGGINGFACE_TOKEN=your-hf-token
OPENAI_API_KEY=your-openai-key

# 모니터링
SENTRY_DSN=your-sentry-dsn
PROMETHEUS_ENABLED=true
```

## 모니터링 및 로깅

### 로깅 설정

**logging.conf**:
```ini
[loggers]
keys=root,lawfirmai

[handlers]
keys=consoleHandler,fileHandler

[formatters]
keys=simpleFormatter,detailedFormatter

[logger_root]
level=INFO
handlers=consoleHandler

[logger_lawfirmai]
level=DEBUG
handlers=consoleHandler,fileHandler
qualname=lawfirmai
propagate=0

[handler_consoleHandler]
class=StreamHandler
level=INFO
formatter=simpleFormatter
args=(sys.stdout,)

[handler_fileHandler]
class=FileHandler
level=DEBUG
formatter=detailedFormatter
args=('./logs/lawfirm.log',)

[formatter_simpleFormatter]
format=%(asctime)s - %(name)s - %(levelname)s - %(message)s

[formatter_detailedFormatter]
format=%(asctime)s - %(name)s - %(levelname)s - %(module)s - %(funcName)s - %(lineno)d - %(message)s
```

### 모니터링 도구

#### Prometheus + Grafana

**prometheus.yml**:
```yaml
global:
  scrape_interval: 15s

scrape_configs:
  - job_name: 'lawfirmai'
    static_configs:
      - targets: ['localhost:8000']
    metrics_path: '/metrics'
```

#### Health Check 엔드포인트

```python
@app.get("/health")
async def health_check():
    return {
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "version": "1.0.0",
        "uptime": get_uptime()
    }
```

## 문제 해결

### 일반적인 문제

#### 1. 메모리 부족 오류

```bash
# 메모리 사용량 확인
free -h
ps aux --sort=-%mem | head

# 해결 방법
# - 더 큰 인스턴스 사용
# - 메모리 최적화 설정 조정
# - 캐시 크기 줄이기
```

#### 2. 데이터베이스 연결 오류

```bash
# 데이터베이스 파일 권한 확인
ls -la data/
chmod 664 data/lawfirm.db

# 데이터베이스 무결성 확인
sqlite3 data/lawfirm.db "PRAGMA integrity_check;"
```

#### 3. 포트 충돌

```bash
# 포트 사용 확인
netstat -tulpn | grep :7860
netstat -tulpn | grep :8000

# 프로세스 종료
sudo kill -9 <PID>
```

### 로그 분석

```bash
# 실시간 로그 모니터링
tail -f logs/lawfirm.log

# 에러 로그만 확인
grep "ERROR" logs/lawfirm.log

# 특정 시간대 로그 확인
grep "2024-12-20 14:" logs/lawfirm.log
```

### 성능 최적화

#### 1. 캐시 설정 조정

```python
# 캐시 크기 증가
CACHE_SIZE = 2000
CACHE_TTL = 7200  # 2시간

# 메모리 사용량 모니터링
MEMORY_LIMIT_MB = 4096
```

#### 2. 데이터베이스 최적화

```sql
-- 인덱스 생성
CREATE INDEX idx_sessions_user_id ON conversation_sessions(user_id);
CREATE INDEX idx_turns_session_id ON conversation_turns(session_id);
CREATE INDEX idx_memories_user_id ON contextual_memories(user_id);

-- 정기적인 VACUUM 실행
VACUUM;
```

## 보안 고려사항

### 1. API 키 보안

```python
# 환경 변수에서 API 키 로드
import os
API_KEY = os.getenv('API_KEY')

# 요청 검증
def verify_api_key(request):
    if request.headers.get('Authorization') != f'Bearer {API_KEY}':
        raise HTTPException(status_code=401, detail="Invalid API key")
```

### 2. 입력 검증

```python
from pydantic import BaseModel, validator

class ChatRequest(BaseModel):
    message: str
    
    @validator('message')
    def validate_message(cls, v):
        if len(v) > 10000:
            raise ValueError('Message too long')
        if not v.strip():
            raise ValueError('Message cannot be empty')
        return v.strip()
```

### 3. Rate Limiting

```python
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.util import get_remote_address

limiter = Limiter(key_func=get_remote_address)

@app.post("/api/chat")
@limiter.limit("10/minute")
async def chat(request: Request, ...):
    pass
```

## 백업 및 복구

### 1. 데이터베이스 백업

```bash
#!/bin/bash
# backup.sh

BACKUP_DIR="/backup/lawfirmai"
DATE=$(date +%Y%m%d_%H%M%S)

# 백업 디렉토리 생성
mkdir -p $BACKUP_DIR

# 데이터베이스 백업
sqlite3 data/lawfirm.db ".backup $BACKUP_DIR/lawfirm_$DATE.db"

# 로그 백업
cp -r logs $BACKUP_DIR/logs_$DATE

# 오래된 백업 삭제 (30일 이상)
find $BACKUP_DIR -name "*.db" -mtime +30 -delete
```

### 2. 자동 백업 설정

```bash
# Crontab 설정
crontab -e

# 매일 새벽 2시에 백업 실행
0 2 * * * /path/to/backup.sh
```

---
