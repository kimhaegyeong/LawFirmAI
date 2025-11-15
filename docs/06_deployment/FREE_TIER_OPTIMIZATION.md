# 프리 티어 최적화 가이드

## 개요

이 가이드는 AWS 프리 티어를 활용하여 LawFirmAI를 무료로 운영하는 방법을 설명합니다.

## 프리 티어 제한사항

### EC2 프리 티어
- **인스턴스 타입**: t2.micro 또는 t3.micro만 무료
- **사용 시간**: 750시간/월 (1개 인스턴스 기준 24시간 운영 가능)
- **리소스**: 1 vCPU, 1GB RAM
- **기간**: 신규 계정 12개월간

### EBS 프리 티어
- **스토리지**: 30GB 범용 SSD (gp2) 무료
- **IOPS**: 기본 IOPS (3,000 IOPS)

### 데이터 전송 프리 티어
- **아웃바운드**: 15GB/월 무료
- **인바운드**: 항상 무료

### ECR 프리 티어
- **스토리지**: 500MB 무료

## 성능 최적화

### 1. 메모리 최적화

프리 티어는 1GB RAM만 제공하므로 Swap 메모리 설정이 필수입니다:

```bash
# Swap 파일 생성 (2GB)
sudo fallocate -l 2G /swapfile
sudo chmod 600 /swapfile
sudo mkswap /swapfile
sudo swapon /swapfile

# 재부팅 시 자동 활성화
echo '/swapfile none swap sw 0 0' | sudo tee -a /etc/fstab

# Swap 사용량 확인
free -h
```

**Docker 메모리 제한 설정:**

`deployment/docker-compose.prod.free-tier.yml`에 메모리 제한 추가:

```yaml
services:
  api:
    deploy:
      resources:
        limits:
          memory: 512M
        reservations:
          memory: 256M
  frontend:
    deploy:
      resources:
        limits:
          memory: 256M
        reservations:
          memory: 128M
```

### 2. CPU 최적화

**불필요한 서비스 비활성화:**

```bash
# Snap 서비스 비활성화
sudo systemctl disable snapd
sudo systemctl stop snapd

# 자동 업데이트 비활성화 (선택사항)
sudo systemctl disable unattended-upgrades

# 불필요한 서비스 확인
sudo systemctl list-units --type=service --state=running
```

**Docker CPU 제한 설정:**

```yaml
services:
  api:
    deploy:
      resources:
        limits:
          cpus: '0.5'
        reservations:
          cpus: '0.25'
```

### 3. 디스크 공간 최적화

**Docker 이미지 최적화:**

```bash
# 멀티스테이지 빌드 사용
# Alpine Linux 기반 이미지 사용
# 불필요한 레이어 제거

# Docker 이미지 정리
docker system prune -a --volumes

# 사용하지 않는 이미지 제거
docker image prune -a
```

**로그 파일 정리:**

```bash
# Journal 로그 보관 기간 설정 (7일)
sudo journalctl --vacuum-time=7d

# Docker 로그 크기 제한
# /etc/docker/daemon.json에 추가:
{
  "log-driver": "json-file",
  "log-opts": {
    "max-size": "10m",
    "max-file": "3"
  }
}
```

**불필요한 패키지 제거:**

```bash
# 사용하지 않는 패키지 제거
sudo apt-get autoremove -y
sudo apt-get autoclean

# 패키지 캐시 정리
sudo apt-get clean
```

### 4. 애플리케이션 최적화

**메모리 사용량 감소:**

```python
# Python 메모리 최적화
# - 불필요한 변수 즉시 삭제
# - 지연 로딩 사용
# - 캐시 크기 제한

# 환경 변수 설정
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=1
```

**모델 로딩 최적화:**

```python
# Float16 양자화 사용
# 지연 로딩 사용
# 모델 캐싱 최소화
```

## 프리 티어 제한 대응

### 1. 메모리 부족 문제

**증상:**
- 컨테이너가 자주 재시작됨
- OOM (Out of Memory) 에러 발생

**해결책:**
- Swap 메모리 설정 (필수)
- Docker 메모리 제한 설정
- 불필요한 프로세스 종료
- 모델 로딩 최적화

### 2. CPU 부족 문제

**증상:**
- 응답 시간이 느림
- CPU 사용률 100% 지속

**해결책:**
- CPU 제한 설정
- 불필요한 서비스 비활성화
- 비동기 처리 최적화
- 요청 처리량 제한

### 3. 디스크 공간 부족

**증상:**
- 디스크 사용률 100%
- 파일 쓰기 실패

**해결책:**
- Docker 이미지 정리
- 로그 파일 정리
- 불필요한 패키지 제거
- 데이터베이스 최적화

### 4. 데이터 전송 초과

**증상:**
- 데이터 전송 비용 발생

**해결책:**
- CDN 사용 (CloudFront)
- 정적 파일 최적화
- Gzip 압축 활성화
- 캐싱 전략 적용

## 모니터링

### 리소스 사용량 모니터링

```bash
# 메모리 사용량 확인
free -h

# CPU 사용량 확인
top
htop

# 디스크 사용량 확인
df -h
du -sh /opt/lawfirmai/*

# Docker 리소스 사용량
docker stats
```

### CloudWatch 메트릭 확인

```bash
# CPU 사용률 확인
aws cloudwatch get-metric-statistics \
  --namespace AWS/EC2 \
  --metric-name CPUUtilization \
  --dimensions Name=InstanceId,Value=i-xxxxx \
  --start-time 2024-01-01T00:00:00Z \
  --end-time 2024-01-02T00:00:00Z \
  --period 3600 \
  --statistics Average
```

## 프리 티어 체크리스트

### 배포 전
- [ ] 인스턴스 타입이 t2.micro 또는 t3.micro인지 확인
- [ ] 스토리지가 30GB 이하인지 확인
- [ ] Swap 메모리 설정 완료
- [ ] Docker 메모리 제한 설정 완료
- [ ] 불필요한 서비스 비활성화 완료

### 배포 후
- [ ] 메모리 사용량 모니터링
- [ ] CPU 사용률 모니터링
- [ ] 디스크 사용량 모니터링
- [ ] 데이터 전송량 모니터링
- [ ] 애플리케이션 성능 확인

## 프리 티어 기간 종료 후

프리 티어 기간(12개월)이 종료되면:

1. **비용 발생 시작**
   - t2.micro: 시간당 약 $0.0116 USD
   - 월 약 $8.50 USD (24시간 운영 시)

2. **업그레이드 고려**
   - t3.small: 시간당 약 $0.0208 USD
   - t3.medium: 시간당 약 $0.0416 USD

3. **최적화 계속**
   - 리소스 사용량 모니터링
   - 불필요한 비용 제거
   - 예약 인스턴스 고려

## 주의사항

1. **프리 티어 제한 초과 시**
   - 자동으로 유료 요금이 부과됩니다
   - AWS Billing 알람 설정 권장

2. **프로덕션 환경**
   - 프리 티어는 학습/테스트용입니다
   - 프로덕션 환경에는 적합하지 않습니다

3. **성능 제약**
   - 제한된 리소스로 인해
   - 느린 응답 시간
   - 동시 사용자 수 제한
   - 복잡한 작업 처리 어려움

## 참고 자료

- [AWS 프리 티어](https://aws.amazon.com/ko/free/)
- [EC2 프리 티어](https://aws.amazon.com/ko/ec2/pricing/free-tier/)
- [프리 티어 사용량 확인](https://console.aws.amazon.com/billing/home#/freetier)

---

**프리 티어로 LawFirmAI를 무료로 운영할 수 있지만, 성능 제약이 있습니다. 학습 및 테스트 목적으로 사용하시기 바랍니다.**

