# AWS 배포 빠른 시작 가이드
## LawFirmAI - Streamlit 기반

> 상세 아키텍처는 [aws_production_architecture_streamlit.md](./aws_production_architecture_streamlit.md) 참고

## 🚀 빠른 시작 (5분 요약)

### 1. 필수 AWS 서비스

```
✅ VPC (네트워크)
✅ ECS Fargate (컨테이너 실행)
✅ RDS Aurora MySQL (데이터베이스)
✅ ElastiCache Redis (캐싱)
✅ S3 (모델 파일 저장)
✅ EFS (벡터 스토어 공유)
✅ Application Load Balancer (로드 밸런서)
✅ CloudFront (CDN)
✅ OpenSearch (벡터 검색, 선택사항)
```

### 2. 최소 구성 (소규모 운영)

```yaml
Streamlit:
  - Tasks: 2개
  - CPU: 1 vCPU, Memory: 2 GB
  - 포트: 8501

FastAPI:
  - Tasks: 2개  
  - CPU: 2 vCPU, Memory: 4 GB
  - 포트: 8000

RDS Aurora:
  - Serverless v2: 0.5-4 ACU
  - MySQL 8.0

ElastiCache:
  - Redis 7.0: cache.t3.micro
```

### 3. 배포 순서

```bash
# 1. 인프라 구축
terraform apply  # 또는 AWS CDK

# 2. 데이터베이스 마이그레이션
python scripts/migrate_to_aurora.py

# 3. 모델 파일 업로드
aws s3 sync ./models s3://lawfirm-models/

# 4. Docker 이미지 빌드 및 푸시
docker build -t lawfirm-streamlit -f streamlit/Dockerfile .
aws ecr get-login-password | docker login --username AWS --password-stdin <account>.dkr.ecr.<region>.amazonaws.com
docker tag lawfirm-streamlit:latest <account>.dkr.ecr.<region>.amazonaws.com/lawfirm-streamlit:latest
docker push <account>.dkr.ecr.<region>.amazonaws.com/lawfirm-streamlit:latest

# 5. ECS 서비스 배포
aws ecs create-service --cluster lawfirm-cluster --service-name streamlit-service ...
```

### 4. 환경 변수 설정

```bash
# Systems Manager Parameter Store 또는 Secrets Manager
DATABASE_URL=mysql://user:pass@aurora-endpoint:3306/lawfirm
REDIS_URL=redis://elasticache-endpoint:6379
MODEL_PATH=s3://lawfirm-models/koGPT-2/
EMBEDDING_MODEL=jhgan/ko-sroberta-multitask
USE_LANGGRAPH=true
```

### 5. 접속 URL

```
프로덕션 URL: https://lawfirm.yourdomain.com
Streamlit: https://lawfirm.yourdomain.com/
FastAPI: https://lawfirm.yourdomain.com/api/v1/health
```

## 💰 예상 비용

### 소규모 (일평균 1,000명)
```
ECS Fargate: ~$180/월
RDS Aurora: ~$100/월
ElastiCache: ~$80/월
S3: ~$50/월
ALB + CloudFront: ~$55/월
기타: ~$35/월
─────────────────────
총계: ~$500/월
```

### 중규모 (일평균 10,000명)
```
총계: ~$2,400/월
```

## 🔍 핵심 체크포인트

- [ ] RDS에 데이터 마이그레이션 완료
- [ ] S3에 모델 파일 업로드 완료
- [ ] ECS 서비스 Health Check 통과
- [ ] ALB에서 SSL 인증서 설정 완료
- [ ] CloudWatch 모니터링 설정 완료
- [ ] 백업 정책 설정 완료

## 📞 문제 해결

### Health Check 실패
```bash
# ECS Task 로그 확인
aws logs tail /ecs/lawfirm-streamlit --follow

# 컨테이너 직접 접속
aws ecs execute-command --cluster lawfirm-cluster --task <task-id> --container streamlit --command "/bin/bash"
```

### 메모리 부족
- Streamlit Task: 2 GB → 3 GB로 증가
- FastAPI Task: 4 GB → 6 GB로 증가

### 응답 시간 느림
- ElastiCache 캐싱 활성화 확인
- OpenSearch 쿼리 최적화
- 모델 양자화 확인 (Float16)

---

**더 자세한 내용은 [aws_production_architecture_streamlit.md](./aws_production_architecture_streamlit.md) 참고**
