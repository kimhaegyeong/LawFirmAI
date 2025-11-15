#!/bin/bash
# AWS EC2 배포 스크립트

set -e

# 색상 정의
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# 환경 변수 확인
if [ -z "$ECR_REGISTRY" ]; then
    echo -e "${RED}Error: ECR_REGISTRY environment variable is not set${NC}"
    exit 1
fi

if [ -z "$GOOGLE_API_KEY" ]; then
    echo -e "${YELLOW}Warning: GOOGLE_API_KEY environment variable is not set${NC}"
fi

echo -e "${GREEN}Starting deployment...${NC}"

# ECR 로그인
echo -e "${GREEN}Logging in to ECR...${NC}"
aws ecr get-login-password --region ap-northeast-2 | \
    docker login --username AWS --password-stdin $ECR_REGISTRY

# 최신 이미지 pull
echo -e "${GREEN}Pulling latest images...${NC}"
docker pull $ECR_REGISTRY/lawfirmai-api:latest
docker pull $ECR_REGISTRY/lawfirmai-frontend:latest

# 기존 컨테이너 중지 및 제거
echo -e "${GREEN}Stopping existing containers...${NC}"
docker-compose -f docker-compose.prod.yml down

# 새 컨테이너 시작
echo -e "${GREEN}Starting new containers...${NC}"
docker-compose -f docker-compose.prod.yml up -d

# PostgreSQL 초기화 (PostgreSQL 사용 시)
if [ -n "$POSTGRES_DB" ] && [ -n "$POSTGRES_USER" ]; then
    echo -e "${GREEN}Initializing PostgreSQL database...${NC}"
    sleep 5  # PostgreSQL 시작 대기
    docker-compose -f docker-compose.prod.yml exec -T api python scripts/database/init_postgresql.py || echo -e "${YELLOW}Warning: PostgreSQL initialization failed or already initialized${NC}"
fi

# 컨테이너 상태 확인
echo -e "${GREEN}Checking container status...${NC}"
sleep 5
docker-compose -f docker-compose.prod.yml ps

# Health check
echo -e "${GREEN}Performing health check...${NC}"
sleep 10
if curl -f http://localhost:8000/health > /dev/null 2>&1; then
    echo -e "${GREEN}Health check passed!${NC}"
else
    echo -e "${RED}Health check failed!${NC}"
    docker-compose -f docker-compose.prod.yml logs --tail=50
    exit 1
fi

echo -e "${GREEN}Deployment completed successfully!${NC}"

