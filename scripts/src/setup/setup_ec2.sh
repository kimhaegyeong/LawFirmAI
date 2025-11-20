#!/bin/bash
# EC2 인스턴스 초기 설정 스크립트

set -e

echo "Setting up EC2 instance for LawFirmAI..."

# 시스템 업데이트
echo "Updating system packages..."
sudo apt-get update
sudo apt-get upgrade -y

# 필수 패키지 설치
echo "Installing required packages..."
sudo apt-get install -y \
    docker.io \
    docker-compose \
    awscli \
    git \
    nginx \
    certbot \
    python3-certbot-nginx \
    curl \
    sqlite3

# Docker 서비스 시작 및 자동 시작 설정
echo "Configuring Docker..."
sudo systemctl start docker
sudo systemctl enable docker
sudo usermod -aG docker ubuntu

# 애플리케이션 디렉토리 생성
echo "Creating application directories..."
sudo mkdir -p /opt/lawfirmai
sudo mkdir -p /opt/lawfirmai/data
sudo mkdir -p /opt/lawfirmai/logs
sudo mkdir -p /mnt/backups
sudo chown -R ubuntu:ubuntu /opt/lawfirmai
sudo chown -R ubuntu:ubuntu /mnt/backups

# Nginx 설정 디렉토리 준비
echo "Preparing Nginx configuration..."
sudo mkdir -p /etc/nginx/sites-available
sudo mkdir -p /etc/nginx/sites-enabled

# 배포 스크립트 권한 설정
if [ -f "/opt/lawfirmai/scripts/deploy.sh" ]; then
    chmod +x /opt/lawfirmai/scripts/deploy.sh
fi

if [ -f "/opt/lawfirmai/scripts/backup.sh" ]; then
    chmod +x /opt/lawfirmai/scripts/backup.sh
fi

# Swap 메모리 설정 (프리 티어 필수)
echo "Setting up swap memory..."
if [ ! -f /swapfile ]; then
    sudo fallocate -l 2G /swapfile
    sudo chmod 600 /swapfile
    sudo mkswap /swapfile
    sudo swapon /swapfile
    echo '/swapfile none swap sw 0 0' | sudo tee -a /etc/fstab
    echo "Swap memory (2GB) created and activated"
else
    echo "Swap file already exists"
fi

# 불필요한 서비스 비활성화 (프리 티어 최적화)
echo "Disabling unnecessary services..."
sudo systemctl disable snapd 2>/dev/null || true
sudo systemctl stop snapd 2>/dev/null || true

# Crontab 설정 (백업 자동화)
echo "Setting up cron jobs..."
(crontab -l 2>/dev/null; echo "0 2 * * * /opt/lawfirmai/scripts/backup.sh >> /opt/lawfirmai/logs/backup.log 2>&1") | crontab -

echo "EC2 setup completed successfully!"
echo ""
echo "Next steps:"
echo "1. Configure AWS credentials: aws configure"
echo "2. Create .env file in /opt/lawfirmai"
echo "3. Copy deployment/docker-compose.prod.yml (or deployment/docker-compose.prod.free-tier.yml for free tier) to /opt/lawfirmai"
echo "4. Run deployment script: /opt/lawfirmai/scripts/deploy.sh"
echo ""
echo "For free tier optimization, see: docs/06_deployment/FREE_TIER_OPTIMIZATION.md"

