# AWS 프로덕션 아키텍처 설계 문서
## LawFirmAI - Streamlit 기반

**작성일**: 2025-10-31  
**버전**: 1.0  
**대상**: Streamlit + FastAPI 하이브리드 아키텍처

---

## 📋 목차

1. [아키텍처 개요](#아키텍처-개요)
2. [세부 구성 요소](#세부-구성-요소)
3. [Streamlit 특화 구성](#streamlit-특화-구성)
4. [데이터 흐름](#데이터-흐름)
5. [배포 전략](#배포-전략)
6. [비용 최적화](#비용-최적화)
7. [모니터링 및 로깅](#모니터링-및-로깅)
8. [보안 구성](#보안-구성)
9. [재해 복구](#재해-복구)
10. [마이그레이션 계획](#마이그레이션-계획)

---

## 🎯 아키텍처 개요

### 전체 구조 다이어그램

#### 완전 구성 (프로덕션)

```
┌─────────────────────────────────────────────────────────────────┐
│                        CloudFront (CDN)                         │
│                      + WAF (보안)                               │
│                  (정적 파일 캐싱 + DDoS 방어)                    │
└────────────────────┬────────────────────────────────────────────┘
                     │
         ┌───────────┴───────────┐
         │   Application Load     │
         │      Balancer (ALB)     │
         │    (SSL 종료 + 라우팅)   │
         └───────────┬───────────┘
                     │
    ┌────────────────┼────────────────┐
    │                │                │
┌───▼───┐      ┌────▼────┐     ┌────▼────┐
│Streamlit│     │ FastAPI │     │ Admin   │
│  ECS   │      │  ECS    │     │  ECS    │
│ Tasks  │      │ Tasks   │     │ Tasks   │
│ :8501  │      │ :8000   │     │         │
└───┬───┘      └────┬────┘     └────┬────┘
    │               │                │
    └───────────────┼────────────────┘
                    │
        ┌───────────▼───────────┐
        │   ECS Service         │
        │   (Fargate)           │
        │   Multi-AZ 배포        │
        └───────────┬───────────┘
                    │
    ┌───────────────┼───────────────┐
    │               │               │
┌───▼────┐    ┌─────▼─────┐  ┌─────▼─────┐
│ RDS    │    │  ElastiCache │  │  S3       │
│ Aurora │    │  Redis      │  │ (Models)  │
│ MySQL  │    │  (Cache)    │  │           │
└────────┘    └────────────┘  └─────┬─────┘
                                    │
                        ┌───────────┴───────────┐
                        │                       │
                ┌───────▼──────┐       ┌────────▼────────┐
                │ OpenSearch   │       │  EFS (Vector     │
                │ (Vector      │       │   Store +        │
                │  Search)     │       │   Embeddings)    │
                └──────────────┘       └──────────────────┘
```

#### 최소 스펙 구성 (프로덕션 연습용) ✅

```
                    사용자
                      ↓
         ┌───────────────────────┐
         │   Application Load    │
         │      Balancer (ALB)    │
         │   (HTTP/HTTPS, SSL)    │
         └───────────┬────────────┘
                     │
            ┌────────▼────────┐
            │  Streamlit ECS │
            │  (1-2 tasks)   │
            │  Fargate        │
            └────────┬────────┘
                     │
          ┌──────────┴──────────┐
          │                      │
    ┌─────▼─────┐         ┌─────▼─────┐
    │  RDS      │         │    EFS    │
    │  Aurora   │         │  (Vector  │
    │ Serverless│         │   Store)  │
    │  v2       │         │           │
    │ (0.5-2ACU)│         │ FAISS/    │
    └─────┬─────┘         │ ChromaDB  │
          │               └───────────┘
          │
    ┌─────▼─────┐
    │    S3     │
    │  (Models) │
    └───────────┘

제외된 서비스 (나중에 추가 가능):
  ❌ CloudFront + WAF
  ❌ FastAPI Service
  ❌ ElastiCache Redis
  ❌ OpenSearch
  ❌ Multi-AZ (단일 AZ로 시작)
```

**비용 비교**:
- 완전 구성: ~$500-800/월
- 최소 스펙: ~$110-160/월 (약 70% 절감)

### 핵심 특징

- **Streamlit 기반 프론트엔드**: 사용자 친화적 웹 인터페이스
- **FastAPI 백엔드**: RESTful API 제공 (외부 통합용)
- **LangGraph 워크플로우**: Streamlit에서 직접 호출 또는 FastAPI 경유
- **서버리스 아키텍처**: ECS Fargate 기반 자동 스케일링
- **하이브리드 검색**: OpenSearch (벡터) + RDS (키워드)
- **캐싱 계층**: ElastiCache Redis로 응답 속도 향상

---

## 🏗️ 세부 구성 요소

### 1. 프론트엔드 레이어

#### 1.1 CloudFront CDN

**역할**: 전역 콘텐츠 전송 및 DDoS 방어

```yaml
설정:
  - Origin: Application Load Balancer
  - SSL/TLS: ACM 인증서 (자동 갱신)
  - WAF 연동:
      - SQL Injection 방어
      - XSS 방어
      - Rate Limiting: IP당 1000 req/min
      - Geo-blocking (필요시)
  - 캐싱 정책:
      - 정적 파일 (JS, CSS, 이미지): 1년
      - API 응답: 캐싱 비활성화 (동적 콘텐츠)
      - Streamlit 응답: 1분 (세션 상태 고려)
  - Price Class: Use only North America and Europe (비용 절감)
```

**비용**: ~$30-150/월 (트래픽 규모에 따라)

#### 1.2 Application Load Balancer (ALB)

**역할**: 트래픽 분산 및 SSL 종료

```yaml
설정:
  - Listeners:
      - HTTPS (443): SSL 인증서 (ACM)
      - HTTP (80): HTTPS 리다이렉트
  - Target Groups:
      - /streamlit/* 또는 / → Streamlit ECS Tasks (포트 8501)
      - /api/* → FastAPI ECS Tasks (포트 8000)
      - /health → 두 서비스 모두 Health Check
  - Health Checks:
      - Streamlit: GET /_stcore/health
      - FastAPI: GET /api/v1/health
      - Interval: 30초
      - Timeout: 5초
      - Healthy Threshold: 2
      - Unhealthy Threshold: 3
  - Idle Timeout: 60초
  - Connection Draining: 300초
```

**비용**: ~$25-50/월

### 2. 애플리케이션 레이어 (ECS Fargate)

#### 2.1 Streamlit Service

**역할**: 사용자 인터페이스 제공, LangGraph 워크플로우 직접 호출

```yaml
Task Definition:
  Family: lawfirm-streamlit
  CPU: 1 vCPU
  Memory: 2 GB
  Container:
    Image: ECR의 Streamlit 이미지
    Port: 8501
    Environment Variables:
      - STREAMLIT_SERVER_PORT: 8501
      - STREAMLIT_SERVER_ADDRESS: 0.0.0.0
      - STREAMLIT_SERVER_HEADLESS: true
      - DATABASE_URL: RDS Aurora 엔드포인트
      - REDIS_URL: ElastiCache 엔드포인트
      - MODEL_PATH: s3://lawfirm-models/koGPT-2/
      - EMBEDDING_MODEL: jhgan/ko-sroberta-multitask
      - USE_LANGGRAPH: true
      - LANGGRAPH_CONFIG: Parameter Store 경로
    Volumes:
      - EFS Mount: /app/data/embeddings
      - EFS Mount: /app/model_cache
      - EFS Mount: /app/data/chroma_db
    Health Check:
      Command: ["CMD-SHELL", "curl -f http://localhost:8501/_stcore/health || exit 1"]
      Interval: 30
      Timeout: 10
      Retries: 3
      Start Period: 60  # Streamlit 초기화 시간 고려

Auto Scaling:
  Min Capacity: 2 tasks (고가용성)
  Max Capacity: 5 tasks
  Target Metrics:
    - CPUUtilization: 60%
    - MemoryUtilization: 70%
  Scale-out Cooldown: 60초
  Scale-in Cooldown: 300초

Network:
  - Security Group: Allow 8501 from ALB only
  - Subnet: Private Subnets (인터넷 접근 불가)
```

**특징**:
- Streamlit은 세션 상태를 메모리에 저장 (Redis 연동 권장)
- LangGraph 워크플로우 직접 호출로 FastAPI 우회 가능
- 채팅 히스토리는 RDS에 저장

#### 2.2 FastAPI Service

**역할**: RESTful API 제공, 외부 시스템 통합

```yaml
Task Definition:
  Family: lawfirm-api
  CPU: 2 vCPU
  Memory: 4 GB
  Container:
    Image: ECR의 FastAPI 이미지
    Port: 8000
    Environment Variables:
      - DATABASE_URL: RDS Aurora 엔드포인트
      - REDIS_URL: ElastiCache 엔드포인트
      - MODEL_PATH: s3://lawfirm-models/koGPT-2/
      - EMBEDDING_MODEL: jhgan/ko-sroberta-multitask
      - API_HOST: 0.0.0.0
      - API_PORT: 8000
    Volumes:
      - EFS Mount: /app/data/embeddings
      - EFS Mount: /app/model_cache
    Health Check:
      Command: ["CMD-SHELL", "curl -f http://localhost:8000/api/v1/health || exit 1"]
      Interval: 30
      Timeout: 5
      Retries: 3

Auto Scaling:
  Min Capacity: 2 tasks
  Max Capacity: 10 tasks
  Target Metrics:
    - CPUUtilization: 70%
    - MemoryUtilization: 80%
  Scale-out Cooldown: 60초
  Scale-in Cooldown: 300초

Network:
  - Security Group: Allow 8000 from ALB and Streamlit Tasks
  - Subnet: Private Subnets
```

**특징**:
- Streamlit과 동일한 데이터 소스 사용
- 외부 API 클라이언트를 위한 표준 RESTful 인터페이스
- API 키 기반 인증 지원

#### 2.3 Batch Processing Service (선택)

**역할**: 벡터 임베딩 생성, 모델 재학습, 대량 데이터 처리

```yaml
Task Definition:
  Family: lawfirm-batch
  CPU: 4 vCPU
  Memory: 8 GB
  Trigger:
    - EventBridge: 스케줄 기반 (예: 매일 새벽)
    - SQS: 큐 기반 (실시간 처리 필요 시)
  
용도:
  - 법령/판례 데이터 수집 및 전처리
  - 벡터 임베딩 생성 및 OpenSearch 인덱싱
  - 모델 재학습 및 평가
  - 데이터 백업
```

---

## 🎨 Streamlit 특화 구성

### 3.0 Python 버전 요구사항

**프로덕션 환경**: Python 3.11 (개발 환경과 동일)

```yaml
Python 버전:
  - 개발 환경: Python 3.11
  - 프로덕션: Python 3.11
  - Docker Base Image: python:3.11-slim

호환성:
  - LangGraph: Python 3.11 지원 ✅
  - Streamlit: Python 3.11 지원 ✅
  - KoGPT-2 모델: Python 3.11 지원 ✅
  - FAISS/ChromaDB: Python 3.11 지원 ✅

주의사항:
  - Python 3.11은 성능 개선 및 타입 힌트 향상
  - 일부 오래된 라이브러리는 호환성 확인 필요
  - 개발/프로덕션 환경 일치로 버그 예방
```

### 3.1 세션 관리 전략

**문제점**: Streamlit은 기본적으로 메모리 기반 세션 관리

**해결책**: Redis 기반 세션 저장소

```python
# streamlit/app.py 수정 예시
import redis
import json
from streamlit.web.server.websocket_headers import _get_websocket_headers

# Redis 연결
redis_client = redis.from_url(os.getenv("REDIS_URL"))

def get_session_id():
    """세션 ID 추출 (쿠키 또는 헤더에서)"""
    headers = _get_websocket_headers()
    session_id = headers.get("X-Session-Id") or st.session_state.get("session_id")
    if not session_id:
        session_id = f"session_{uuid.uuid4()}"
        st.session_state.session_id = session_id
    return session_id

def load_chat_history(session_id: str):
    """Redis에서 채팅 히스토리 로드"""
    history_json = redis_client.get(f"chat_history:{session_id}")
    if history_json:
        return json.loads(history_json)
    return []

def save_chat_history(session_id: str, history: list):
    """채팅 히스토리를 Redis에 저장"""
    redis_client.setex(
        f"chat_history:{session_id}",
        86400,  # 24시간 TTL
        json.dumps(history)
    )
```

### 3.2 LangGraph 통합

**현재 구현**: Streamlit에서 LangGraph 워크플로우 직접 호출

```python
# streamlit/app.py (현재 구조)
from source.agents.workflow_service import LangGraphWorkflowService

app = StreamlitApp()
result = app.process_query(query, session_id)
```

**AWS 배포 시 고려사항**:
- LangGraph 체크포인트를 RDS에 저장 (SQLite 대신)
- 세션 상태를 Redis에 저장
- 워크플로우 실행 결과를 캐싱

### 3.3 스케일링 고려사항

**Streamlit의 제한사항**:
- 각 세션은 독립적인 Python 인터프리터
- 메모리 사용량이 상대적으로 높음
- 세션 간 상태 공유 어려움

**해결책**:
- 세션 상태는 Redis에 저장
- 채팅 히스토리는 RDS에 저장
- 모델은 EFS 공유 스토리지에서 로딩
- 태스크당 최대 세션 수 제한 (메모리 기반)

```yaml
Resource Limits:
  - CPU: 1 vCPU per task
  - Memory: 2 GB per task
  - Estimated Sessions per Task: 10-20 (메모리 사용량에 따라)
  - Max Tasks: 5
  - Total Concurrent Sessions: 50-100
```

---

## 📊 데이터 흐름

### 4.1 사용자 질문 처리 흐름

```
1. 사용자 → CloudFront
   └─> WAF 검증
       └─> ALB
           └─> Streamlit ECS Task

2. Streamlit Task
   ├─> Redis: 세션 상태 확인
   ├─> LangGraph Workflow 실행
   │   ├─> RDS: 채팅 히스토리 조회
   │   ├─> OpenSearch: 벡터 검색
   │   ├─> RDS: 키워드 검색
   │   └─> KoGPT-2: 답변 생성
   ├─> Redis: 응답 캐싱 (선택)
   ├─> RDS: 채팅 히스토리 저장
   └─> 사용자에게 응답 반환

3. 캐시 히트 시
   └─> Redis에서 즉시 응답 반환 (7.96초 → 0.1초)
```

### 4.2 FastAPI 경유 흐름 (선택)

```
Streamlit → FastAPI → (동일한 백엔드 서비스)
```

**장점**: 
- Streamlit과 FastAPI가 동일한 로직 공유
- 외부 API 클라이언트도 동일한 기능 사용

**단점**: 
- 추가 네트워크 지연
- 복잡도 증가

**권장**: Streamlit은 직접 LangGraph 호출, FastAPI는 외부 통합용으로만 사용

---

## 🚀 배포 전략

### 5.1 컨테이너 이미지 빌드

#### Streamlit Dockerfile

```dockerfile
# streamlit/Dockerfile (프로덕션 최적화)

FROM python:3.11-slim as builder

WORKDIR /app

# 시스템 의존성 설치
RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    git \
    && rm -rf /var/lib/apt/lists/*

# Python 의존성 설치
COPY streamlit/requirements.txt .
RUN pip install --no-cache-dir --user -r requirements.txt

FROM python:3.11-slim

# Non-root 사용자
RUN useradd --create-home --shell /bin/bash app
USER app

WORKDIR /app

# 패키지 복사
COPY --from=builder /root/.local /home/app/.local
ENV PATH=/home/app/.local/bin:$PATH

# 애플리케이션 코드 복사
COPY --chown=app:app streamlit/ ./streamlit/
COPY --chown=app:app source/ ./source/
COPY --chown=app:app core/ ./source/
COPY --chown=app:app infrastructure/ ./infrastructure/

# 디렉토리 생성
RUN mkdir -p data logs model_cache

# 환경 변수
ENV PYTHONPATH=/app
ENV PYTHONUNBUFFERED=1
ENV STREAMLIT_SERVER_PORT=8501
ENV STREAMLIT_SERVER_ADDRESS=0.0.0.0
ENV STREAMLIT_SERVER_HEADLESS=true

EXPOSE 8501

HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD curl -f http://localhost:8501/_stcore/health || exit 1

CMD ["streamlit", "run", "streamlit/app.py", \
     "--server.port=8501", \
     "--server.address=0.0.0.0", \
     "--server.headless=true"]
```

#### ECR 저장소 생성

```bash
# ECR 저장소 생성
aws ecr create-repository --repository-name lawfirm-streamlit
aws ecr create-repository --repository-name lawfirm-api

# 이미지 푸시
docker build -t lawfirm-streamlit -f streamlit/Dockerfile .
docker tag lawfirm-streamlit:latest <account>.dkr.ecr.<region>.amazonaws.com/lawfirm-streamlit:latest
aws ecr get-login-password --region <region> | docker login --username AWS --password-stdin <account>.dkr.ecr.<region>.amazonaws.com
docker push <account>.dkr.ecr.<region>.amazonaws.com/lawfirm-streamlit:latest
```

### 5.2 Infrastructure as Code (Terraform/CDK)

**권장**: AWS CDK (Python) 사용

```python
# infrastructure/cdk/app.py (예시)

from aws_cdk import (
    core,
    aws_ecs as ecs,
    aws_ecs_patterns as ecs_patterns,
    aws_ecr as ecr,
)

class LawFirmStack(core.Stack):
    def __init__(self, scope, id, **kwargs):
        super().__init__(scope, id, **kwargs)

        # ECR 저장소
        streamlit_repo = ecr.Repository(self, "StreamlitRepo")
        api_repo = ecr.Repository(self, "ApiRepo")

        # ECS 클러스터
        cluster = ecs.Cluster(self, "LawFirmCluster")

        # Streamlit 서비스
        streamlit_service = ecs_patterns.ApplicationLoadBalancedFargateService(
            self, "StreamlitService",
            cluster=cluster,
            task_image_options=ecs_patterns.ApplicationLoadBalancedTaskImageOptions(
                image=ecs.ContainerImage.from_ecr_repository(streamlit_repo),
                container_port=8501,
                environment={
                    "DATABASE_URL": "...",
                    "REDIS_URL": "...",
                }
            ),
            desired_count=2,
            memory_limit_mib=2048,
            cpu=1024,
        )

        # Auto Scaling
        streamlit_service.service.auto_scale_task_count(
            min_capacity=2,
            max_capacity=5
        ).scale_on_cpu_utilization(
            "CpuScaling",
            target_utilization_percent=60
        ).scale_on_memory_utilization(
            "MemoryScaling",
            target_utilization_percent=70
        )
```

### 5.3 CI/CD 파이프라인

**GitHub Actions 예시**:

```yaml
# .github/workflows/deploy.yml

name: Deploy to AWS

on:
  push:
    branches: [main]
    paths:
      - 'streamlit/**'
      - 'source/**'

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
          aws-region: ap-northeast-2

      - name: Login to Amazon ECR
        id: login-ecr
        uses: aws-actions/amazon-ecr-login@v1

      - name: Build and push Streamlit image
        env:
          ECR_REGISTRY: ${{ steps.login-ecr.outputs.registry }}
          ECR_REPOSITORY: lawfirm-streamlit
          IMAGE_TAG: ${{ github.sha }}
        run: |
          docker build -t $ECR_REGISTRY/$ECR_REPOSITORY:$IMAGE_TAG -f streamlit/Dockerfile .
          docker push $ECR_REGISTRY/$ECR_REPOSITORY:$IMAGE_TAG
          docker tag $ECR_REGISTRY/$ECR_REPOSITORY:$IMAGE_TAG $ECR_REGISTRY/$ECR_REPOSITORY:latest
          docker push $ECR_REGISTRY/$ECR_REPOSITORY:latest

      - name: Deploy to ECS
        run: |
          aws ecs update-service \
            --cluster lawfirm-cluster \
            --service streamlit-service \
            --force-new-deployment
```

### 5.4 Blue-Green 배포

**ECS Blue-Green 배포 설정**:

```yaml
Deployment Configuration:
  Type: Blue/Green
  Minimum Healthy Percent: 100
  Maximum Percent: 200
  Task Definition: lawfirm-streamlit:latest
  
Health Check:
  - ALB Health Check 통과 후 트래픽 전환
  - 실패 시 자동 롤백
```

---

## 💰 비용 최적화

### 6.0 최소 스펙 구성 (프로덕션 연습용)

**목적**: 실제 프로덕션 환경을 최소 비용으로 학습 및 연습

#### 최소 필수 구성

```yaml
필수 서비스:
  - ECS Fargate: Streamlit 서비스만 (FastAPI는 선택)
  - RDS Aurora Serverless v2: 최소 용량 (0.5 ACU)
  - S3: 모델 파일 저장
  - EFS: 벡터 스토어 공유 (FAISS/ChromaDB 사용)
  - ALB: 기본 로드 밸런서 (CloudFront 제외 가능)

제외 가능 (나중에 추가):
  - CloudFront + WAF: 직접 ALB 접근으로 대체
  - OpenSearch: FAISS/ChromaDB를 EFS에서 직접 사용
  - ElastiCache Redis: 초기에는 없어도 동작 (캐싱 없이)
  - FastAPI 서비스: Streamlit만으로 시작 가능
  - Multi-AZ: 단일 AZ로 시작 (비용 절감)
  - Batch Processing: 나중에 추가

최소 스펙:
  Streamlit:
    - Tasks: 1 (Min) - 2 (Max)
    - CPU: 1 vCPU
    - Memory: 2 GB
    - 비용: ~$30-40/월

  RDS Aurora Serverless v2:
    - Min Capacity: 0.5 ACU
    - Max Capacity: 2 ACU
    - 비용: ~$50-70/월 (실제 사용량에 따라)

  S3:
    - 모델 파일 저장: ~5 GB
    - 비용: ~$0.12/월

  EFS:
    - 벡터 스토어: ~10 GB
    - 비용: ~$3/월

  ALB:
    - 기본 로드 밸런서
    - 비용: ~$16-20/월

  ECR:
    - 이미지 저장: ~5 GB
    - 비용: ~$0.50/월

  기타 (VPC, CloudWatch 등):
    - 비용: ~$10-15/월

총 예상 비용: ~$110-160/월
```

#### 최소 스펙 아키텍처

```
사용자
  ↓
ALB (HTTP/HTTPS)
  ↓
┌─────────────────┐
│  Streamlit ECS  │
│  (1-2 tasks)    │
│  Fargate        │
└────────┬────────┘
         │
    ┌────┴────┐
    │         │
┌───▼──┐  ┌──▼─────┐
│ RDS  │  │  EFS   │
│Aurora│  │(Vector)│
│Server│  └────────┘
└──┬───┘
   │
┌──▼────┐
│  S3   │
│(Models)│
└───────┘
```

#### 최소 스펙 설정 예시

```yaml
Streamlit Service (최소):
  Task Definition:
    CPU: 1 vCPU
    Memory: 2 GB
    Min Tasks: 1
    Max Tasks: 2
  
  Environment Variables:
    - DATABASE_URL: RDS Aurora 엔드포인트
    - MODEL_PATH: s3://lawfirm-models/
    - EMBEDDING_MODEL: jhgan/ko-sroberta-multitask
    # Redis 제외: REDIS_URL 없음
    # OpenSearch 제외: FAISS/ChromaDB 직접 사용

RDS Aurora (최소):
  Engine: Aurora MySQL Serverless v2
  Min Capacity: 0.5 ACU (~1 GB RAM)
  Max Capacity: 2 ACU (~4 GB RAM)
  Single-AZ: 활성화 (비용 절감)
  
  백업: 7일 자동 백업 (기본)

EFS (벡터 스토어):
  Performance Mode: General Purpose
  Throughput Mode: Bursting
  Size: ~10 GB (초기)

S3 (모델 저장):
  버킷: lawfirm-models
  저장소 클래스: Standard
  크기: ~5 GB (KoGPT-2 + Sentence-BERT)
```

#### 최소 스펙에서 제거된 기능

```yaml
제거 기능:
  1. Redis 캐싱
     - 영향: 응답 속도 약간 느려짐 (7.96초 → 8-10초)
     - 대안: Python 내부 캐싱 (메모리 기반, 제한적)
     - 추가 시기: 트래픽 증가 시

  2. OpenSearch
     - 영향: FAISS/ChromaDB를 EFS에서 직접 사용
     - 성능: 검색 속도는 유사, 하지만 실시간 인덱싱 어려움
     - 추가 시기: 데이터 크기 증가 시 (10,000+ 문서)

  3. CloudFront CDN
     - 영향: 정적 파일 캐싱 없음, 지리적 분산 없음
     - 대안: ALB에서 직접 제공 (한국 리전만 지원)
     - 추가 시기: 글로벌 서비스 시

  4. WAF
     - 영향: 기본 보안만 (Security Groups)
     - 대안: ALB Security Groups, Rate Limiting (나중에 추가)
     - 추가 시기: 공격 우려 증가 시

  5. FastAPI 서비스
     - 영향: 외부 API 통합 어려움
     - 대안: Streamlit에서 직접 LangGraph 사용
     - 추가 시기: 외부 시스템 연동 필요 시
```

#### 단계적 확장 계획

```yaml
Phase 1: 최소 스펙 (연습용) - $110-160/월
  ✅ ECS Fargate (Streamlit)
  ✅ RDS Aurora Serverless v2
  ✅ S3 (모델 저장)
  ✅ EFS (벡터 스토어)
  ✅ ALB

Phase 2: 기본 최적화 - +$50-80/월
  ➕ ElastiCache Redis (캐싱)
  ➕ CloudWatch 모니터링 강화
  ➕ Multi-AZ (고가용성)

Phase 3: 성능 향상 - +$100-200/월
  ➕ OpenSearch (벡터 검색)
  ➕ FastAPI 서비스
  ➕ CloudFront + WAF

Phase 4: 대규모 운영 - +$300-500/월
  ➕ Auto Scaling 강화
  ➕ Batch Processing
  ➕ Cross-Region 복제
```

### 6.1 리소스 사이징

#### 최소 스펙 (프로덕션 연습용)

```yaml
Streamlit:
  - Tasks: 1 (Min) - 2 (Max)
  - CPU: 1 vCPU per task
  - Memory: 2 GB per task
  - 비용: ~$30-40/월

RDS Aurora:
  - Serverless v2: 0.5-2 ACU
  - Single-AZ
  - 비용: ~$50-70/월

S3 + EFS:
  - 비용: ~$3-5/월

ALB:
  - 비용: ~$16-20/월

기타:
  - 비용: ~$10-15/월

Total: ~$110-160/월
```

#### 소규모 운영 (일평균 1,000명 사용자)

```yaml
Streamlit:
  - Tasks: 2 (Min) - 3 (Max)
  - CPU: 1 vCPU per task
  - Memory: 2 GB per task
  - 비용: ~$60/월

FastAPI:
  - Tasks: 2 (Min) - 5 (Max)
  - CPU: 2 vCPU per task
  - Memory: 4 GB per task
  - 비용: ~$120/월

Total Compute: ~$180/월
```

#### 중규모 운영 (일평균 10,000명 사용자)

```yaml
Streamlit:
  - Tasks: 3 (Min) - 5 (Max)
  - 비용: ~$150/월

FastAPI:
  - Tasks: 3 (Min) - 10 (Max)
  - 비용: ~$400/월

Total Compute: ~$550/월
```

### 6.2 캐싱 전략

**ElastiCache Redis 활용**:

```python
캐싱 대상:
  - 사용자 질문 + 답변: 1시간 TTL
  - 벡터 검색 결과: 30분 TTL
  - 세션 상태: 24시간 TTL
  - 모델 응답: 1시간 TTL

예상 효과:
  - 캐시 히트율: 30-40%
  - DB 부하 감소: 30-40%
  - 응답 시간 단축: 98% (7.96초 → 0.1초)
```

### 6.3 스토리지 최적화

**S3 Intelligent-Tiering**:
- 자주 접근: Standard
- 드물게 접근: Standard-IA (자동 전환)
- 아카이브: Glacier (90일 후)

**EFS 스토리지 클래스**:
- General Purpose: 일반적인 워크로드
- Provisioned Throughput: 필요시만 활성화

---

## 📈 모니터링 및 로깅

### 7.1 CloudWatch 메트릭

#### Streamlit 특화 메트릭

```yaml
Custom Metrics:
  - ActiveSessions: 활성 세션 수
  - SessionDuration: 세션 평균 지속 시간
  - QueryProcessingTime: 질문 처리 시간
  - LangGraphExecutionTime: LangGraph 워크플로우 실행 시간
  - CacheHitRate: 캐시 히트율

Standard Metrics:
  - CPUUtilization: CPU 사용률
  - MemoryUtilization: 메모리 사용률
  - TaskCount: 실행 중인 태스크 수
  - TargetResponseTime: ALB 응답 시간
```

#### CloudWatch Dashboard 구성

```yaml
Dashboard: LawFirmAI-Production
  Panels:
    1. Streamlit Service Health
       - Active Tasks
       - CPU/Memory Usage
       - Active Sessions
    2. API Performance
       - Request Count
       - Response Time (P50/P95/P99)
       - Error Rate
    3. Backend Services
       - RDS Connections
       - Redis Cache Hit Rate
       - OpenSearch Query Latency
    4. Business Metrics
       - Daily Active Users
       - Queries per Hour
       - Average Response Time
```

### 7.2 로깅 전략

#### Streamlit 로그

```python
# streamlit/app.py
import logging
import json
from datetime import datetime

# JSON 포맷 로거 설정
logger = logging.getLogger(__name__)
handler = logging.StreamHandler()
formatter = logging.Formatter(
    '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
handler.setFormatter(formatter)
logger.addHandler(handler)
logger.setLevel(logging.INFO)

def log_query(session_id: str, query: str, result: dict, processing_time: float):
    """구조화된 로그 기록"""
    log_entry = {
        "timestamp": datetime.now().isoformat(),
        "session_id": session_id,
        "query": query[:100],  # 개인정보 보호
        "processing_time": processing_time,
        "confidence": result.get("confidence", {}).get("confidence", 0),
        "error": result.get("error")
    }
    logger.info(json.dumps(log_entry))
```

#### CloudWatch Logs Insights 쿼리

```sql
-- 평균 응답 시간 분석
fields @timestamp, processing_time
| stats avg(processing_time) as avg_time, 
        percentile(processing_time, 95) as p95_time
| filter @message like /query/

-- 에러율 분석
fields @timestamp, error
| stats count(*) as total, 
        count_if(error is not null) as errors
| filter @message like /query/
| stats (errors / total * 100) as error_rate
```

### 7.3 알람 설정

```yaml
Critical Alarms:
  - StreamlitServiceUnhealthy:
      Metric: UnhealthyHostCount
      Threshold: > 0
      Action: SNS → PagerDuty/Slack
      
  - HighErrorRate:
      Metric: 5xxErrorRate
      Threshold: > 5%
      Action: SNS → Team

  - HighResponseTime:
      Metric: TargetResponseTime
      Threshold: > 10초 (P95)
      Action: SNS → Team

Warning Alarms:
  - HighCPUUsage:
      Metric: CPUUtilization
      Threshold: > 80%
      
  - HighMemoryUsage:
      Metric: MemoryUtilization
      Threshold: > 85%
      
  - LowCacheHitRate:
      Metric: CacheHitRate
      Threshold: < 20%
```

---

## 🔒 보안 구성

### 8.1 네트워크 보안

```yaml
VPC 구조:
  CIDR: 10.0.0.0/16
  Subnets:
    - Public (10.0.1.0/24, 10.0.2.0/24):
        - ALB
        - NAT Gateway
    - Private (10.0.10.0/24, 10.0.11.0/24):
        - ECS Tasks (Streamlit, FastAPI)
    - Database (10.0.20.0/24, 10.0.21.0/24):
        - RDS Aurora
        - ElastiCache
        - OpenSearch

Security Groups:
  - ALB-SG:
      - Inbound: HTTPS (443) from 0.0.0.0/0
      - Outbound: All traffic
      
  - Streamlit-SG:
      - Inbound: Port 8501 from ALB-SG only
      - Outbound: 
          - RDS (3306)
          - Redis (6379)
          - OpenSearch (443)
          - S3
          
  - FastAPI-SG:
      - Inbound: Port 8000 from ALB-SG and Streamlit-SG
      - Outbound: Same as Streamlit-SG
      
  - RDS-SG:
      - Inbound: MySQL (3306) from ECS-SG only
      
  - Redis-SG:
      - Inbound: Redis (6379) from ECS-SG only
```

### 8.2 암호화

```yaml
In-Transit:
  - 모든 통신: TLS 1.2+
  - ALB: ACM 인증서
  - RDS: SSL 연결 강제
  - ElastiCache: In-transit encryption 활성화
  - OpenSearch: HTTPS only

At-Rest:
  - RDS: KMS 암호화
  - S3: SSE-S3 또는 SSE-KMS
  - EFS: 암호화 활성화
  - EBS: 기본 암호화
```

### 8.3 접근 제어

```yaml
IAM Roles:
  - ECS Task Role (Streamlit):
      Policies:
        - s3:GetObject (모델 파일)
        - secretsmanager:GetSecretValue
        - rds-db:connect (RDS Proxy)
        - elasticache:Connect
        - logs:CreateLogGroup, logs:CreateLogStream, logs:PutLogEvents
        
  - ECS Task Role (FastAPI):
      Policies: Same as Streamlit

Secrets Management:
  - Database Credentials: Secrets Manager
  - API Keys: Parameter Store (encrypted)
  - External API Keys: Secrets Manager
```

### 8.4 데이터 보호

```yaml
개인정보 보호:
  - 채팅 로그: 90일 후 자동 삭제
  - 세션 데이터: 24시간 TTL
  - 로그 마스킹: 개인정보 자동 마스킹

규정 준수:
  - GDPR 준수: 데이터 삭제 요청 처리
  - 로그 보관: CloudWatch Logs → S3 (7년)
```

---

## 🛡️ 재해 복구

### 9.1 백업 전략

```yaml
RDS Aurora:
  - 자동 백업: 7일 보관
  - 수동 스냅샷: 주 1회
  - Cross-Region 복제: 필요시 활성화

S3:
  - Versioning: 활성화
  - Cross-Region Replication: 활성화
  - Lifecycle Policy: 90일 후 Glacier

EFS:
  - AWS Backup: 일일 백업
  - Retention: 30일

OpenSearch:
  - 자동 스냅샷: S3에 저장
  - Retention: 7일
```

### 9.2 복구 목표

```yaml
RTO (Recovery Time Objective): 4시간
  - Multi-AZ 배포로 대부분의 장애 자동 복구
  - 수동 개입 필요 시 4시간 내 복구

RPO (Recovery Point Objective): 1시간
  - RDS 자동 백업: 5분 간격
  - S3 실시간 복제
```

### 9.3 장애 시나리오

#### 시나리오 1: Streamlit 서비스 장애

```
1. ALB Health Check 실패 감지
2. 장애 태스크 자동 제거
3. Auto Scaling이 새 태스크 시작
4. 새 태스크 Health Check 통과 시 트래픽 복구
예상 복구 시간: 2-5분
```

#### 시나리오 2: RDS 장애

```
1. Multi-AZ 배포로 자동 장애 조치
2. Standby 인스턴스로 자동 전환
3. DNS 업데이트 (자동)
예상 복구 시간: 1-2분
```

#### 시나리오 3: 리전 전체 장애

```
1. Cross-Region 백업에서 복구
2. 새 리전에 인프라 재구성
3. DNS 라우팅 변경 (Route 53)
예상 복구 시간: 4시간
```

---

## 🔄 마이그레이션 계획

### 10.0 최소 스펙 마이그레이션 (연습용)

**목적**: 프로덕션 환경 학습을 위한 최소 구성 시작

#### Phase 1: 최소 인프라 구축 (1주)

```yaml
Day 1-2: 기본 인프라
  - VPC 및 기본 네트워크 구성 (단일 AZ)
  - S3 버킷 생성 및 모델 파일 업로드
  - ECR 저장소 생성
  
Day 3-4: 데이터베이스 및 스토리지
  - RDS Aurora Serverless v2 생성 (0.5-2 ACU)
  - SQLite → Aurora 데이터 마이그레이션
  - EFS 생성 및 벡터 스토어 마이그레이션 (FAISS/ChromaDB)
  
Day 5-7: 애플리케이션 배포
  - Streamlit Docker 이미지 빌드 및 ECR 푸시
  - ECS 클러스터 및 서비스 생성 (1-2 tasks)
  - ALB 구성 및 라우팅 설정
  - SSL 인증서 설정 (ACM)
  - 기본 테스트 및 검증
```

**예상 비용**: ~$110-160/월  
**예상 시간**: 1주

#### Phase 2: 기본 최적화 (선택, 1주)

```yaml
추가할 서비스:
  - ElastiCache Redis (캐싱)
  - CloudWatch 모니터링 강화
  - Multi-AZ 배포 (고가용성)

예상 추가 비용: +$50-80/월
```

### 10.1 단계별 마이그레이션 (완전 구성)

#### Phase 1: 인프라 구축 (2주)

```yaml
Week 1:
  - VPC 및 네트워크 구성
  - RDS Aurora 생성 및 데이터 마이그레이션
  - S3 버킷 생성 및 모델 파일 업로드
  - EFS 생성 및 벡터 스토어 마이그레이션
  
Week 2:
  - ElastiCache Redis 구성
  - OpenSearch 클러스터 생성 (FAISS 대체)
  - ECR 저장소 생성
  - 기본 보안 그룹 구성
```

#### Phase 2: 애플리케이션 배포 (2주)

```yaml
Week 3:
  - Streamlit Docker 이미지 빌드 및 ECR 푸시
  - FastAPI Docker 이미지 빌드 및 ECR 푸시
  - ECS 클러스터 및 서비스 생성
  - ALB 구성 및 라우팅 설정
  
Week 4:
  - CloudFront 구성
  - SSL 인증서 설정
  - Health Check 및 모니터링 설정
  - 기본 테스트 및 검증
```

#### Phase 3: 최적화 및 안정화 (2주)

```yaml
Week 5:
  - Auto Scaling 튜닝
  - 캐싱 전략 적용 및 최적화
  - 성능 테스트 및 병목 지점 개선
  - 보안 감사 및 강화
  
Week 6:
  - 모니터링 대시보드 구성
  - 알람 설정
  - 문서화 완료
  - 운영 매뉴얼 작성
```

### 10.2 데이터 마이그레이션

#### SQLite → RDS Aurora

```bash
# 1. SQLite 데이터 덤프
sqlite3 data/lawfirm_v2.db .dump > lawfirm_dump.sql

# 2. MySQL 호환성 수정
sed -i 's/INTEGER PRIMARY KEY AUTOINCREMENT/INT AUTO_INCREMENT PRIMARY KEY/g' lawfirm_dump.sql
sed -i 's/TEXT NOT NULL/VARCHAR(255) NOT NULL/g' lawfirm_dump.sql

# 3. RDS Aurora로 임포트
mysql -h <aurora-endpoint> -u admin -p lawfirm < lawfirm_dump.sql
```

#### FAISS → OpenSearch

```python
# 마이그레이션 스크립트
from source.data.vector_store import LegalVectorStore
from opensearchpy import OpenSearch

# 1. FAISS에서 벡터 읽기
vector_store = LegalVectorStore(...)
vectors = vector_store.get_all_vectors()

# 2. OpenSearch에 인덱싱
opensearch_client = OpenSearch(...)
for vector_id, vector, metadata in vectors:
    opensearch_client.index(
        index="legal_documents",
        id=vector_id,
        body={
            "vector": vector.tolist(),
            "text": metadata["text"],
            "metadata": metadata
        }
    )
```

### 10.3 트래픽 전환 계획

```yaml
Stage 1: Canary 배포 (10% 트래픽)
  - 새 Streamlit 서비스에 10% 트래픽 라우팅
  - 모니터링 및 검증 (24시간)
  
Stage 2: 점진적 전환 (50% 트래픽)
  - 문제 없으면 50%로 증가
  - 모니터링 지속 (24시간)
  
Stage 3: 전체 전환 (100% 트래픽)
  - 모든 트래픽을 새 서비스로 라우팅
  - 기존 인프라 유지 (롤백 대비, 7일)
  
Stage 4: 정리
  - 기존 인프라 종료
  - 최종 문서화
```

---

## 📝 체크리스트

### 배포 전 확인사항

#### 최소 스펙 (연습용) 체크리스트 ✅

**필수 항목**:
- [ ] VPC 및 기본 네트워크 구성 완료 (단일 AZ OK)
- [ ] RDS Aurora Serverless v2 생성 (0.5-2 ACU)
- [ ] SQLite → Aurora 데이터 마이그레이션 완료
- [ ] S3 버킷 생성 및 모델 파일 업로드 완료
- [ ] EFS 생성 및 벡터 스토어 마이그레이션 완료 (FAISS/ChromaDB)
- [ ] ECR 저장소 생성 완료
- [ ] Streamlit Docker 이미지 빌드 및 푸시 완료
- [ ] ECS 클러스터 및 서비스 생성 완료 (1-2 tasks)
- [ ] ALB 구성 및 라우팅 설정 완료
- [ ] SSL 인증서 설정 완료 (ACM)
- [ ] Health Check 통과 확인
- [ ] 기본 보안 그룹 설정 완료
- [ ] CloudWatch 로그 그룹 생성 완료

**선택 항목** (나중에 추가):
- [ ] ElastiCache Redis 구성
- [ ] OpenSearch 클러스터 생성
- [ ] FastAPI 서비스 배포
- [ ] CloudFront 구성
- [ ] WAF 설정
- [ ] Multi-AZ 배포
- [ ] 상세 모니터링 대시보드
- [ ] 알람 설정

#### 완전 구성 (프로덕션) 체크리스트

- [ ] VPC 및 네트워크 구성 완료 (Multi-AZ)
- [ ] RDS Aurora 데이터 마이그레이션 완료
- [ ] S3 모델 파일 업로드 완료
- [ ] EFS 벡터 스토어 마이그레이션 완료
- [ ] ElastiCache Redis 구성 완료
- [ ] OpenSearch 클러스터 생성 완료
- [ ] ECR 이미지 빌드 및 푸시 완료
- [ ] ECS 서비스 배포 및 Health Check 통과
- [ ] ALB 라우팅 설정 완료
- [ ] CloudFront 구성 완료
- [ ] SSL 인증서 설정 완료
- [ ] 모니터링 대시보드 구성 완료
- [ ] 알람 설정 완료
- [ ] 보안 그룹 및 IAM 역할 설정 완료
- [ ] 백업 정책 설정 완료
- [ ] 문서화 완료

### 운영 중 모니터링 항목

- [ ] CPU/Memory 사용률
- [ ] 응답 시간 (P50/P95/P99)
- [ ] 에러율 (4xx/5xx)
- [ ] 캐시 히트율
- [ ] 활성 세션 수
- [ ] 데이터베이스 연결 수
- [ ] OpenSearch 쿼리 지연 시간
- [ ] 비용 사용량

---

## 📚 참고 자료

- [AWS ECS Fargate 가이드](https://docs.aws.amazon.com/AmazonECS/latest/developerguide/AWS_Fargate.html)
- [Streamlit 배포 가이드](https://docs.streamlit.io/knowledge-base/deploy)
- [Amazon OpenSearch Service 가이드](https://docs.aws.amazon.com/opensearch-service/)
- [RDS Aurora 가이드](https://docs.aws.amazon.com/AmazonRDS/latest/AuroraUserGuide/)
- [ElastiCache Redis 가이드](https://docs.aws.amazon.com/AmazonElastiCache/latest/red-ug/)

---

**문서 버전**: 1.0  
**최종 업데이트**: 2025-10-31
