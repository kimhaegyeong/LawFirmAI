# Grafana + Prometheus 모니터링 설정 가이드

## 개요

LawFirmAI 프로젝트의 법률 수집 성능을 모니터링하기 위한 Grafana + Prometheus 기반 모니터링 시스템입니다.

## 빠른 시작

### 1. 모니터링 스택 시작

```bash
# 모니터링 디렉토리로 이동
cd monitoring

# Docker 스택 시작
./start_monitoring.sh

# 또는 직접 실행
docker-compose up -d
```

### 2. 서비스 접근

- **Grafana**: http://localhost:3000 (admin/admin123)
- **Prometheus**: http://localhost:9090
- **Node Exporter**: http://localhost:9100
- **메트릭 엔드포인트**: http://localhost:8000/metrics

### 3. 법률 수집 실행 (메트릭 포함)

```bash
# 메트릭 수집 활성화 (기본값)
python scripts/assembly/collect_laws_optimized.py --sample 50 --enable-metrics

# 메트릭 수집 비활성화
python scripts/assembly/collect_laws_optimized.py --sample 50 --disable-metrics
```

## 구성 요소

### Docker 서비스

1. **Prometheus** (포트 9090)
   - 메트릭 수집 및 저장
   - 알림 규칙 관리
   - 쿼리 엔진

2. **Grafana** (포트 3000)
   - 대시보드 시각화
   - 알림 관리
   - 데이터 소스 관리

3. **Node Exporter** (포트 9100)
   - 시스템 메트릭 수집
   - CPU, 메모리, 디스크 사용률

### Python 메트릭 수집기

- **LawCollectionMetrics**: 법률 수집 성능 메트릭 수집
- **포트 8000**: Prometheus가 스크래핑하는 메트릭 엔드포인트

## 수집되는 메트릭

### 법률 수집 메트릭

- `law_collection_pages_processed_total`: 처리된 총 페이지 수
- `law_collection_laws_collected_total`: 수집된 총 법률 수
- `law_collection_errors_total`: 에러 수 (유형별)
- `law_collection_page_processing_seconds`: 페이지 처리 시간 (히스토그램)
- `law_collection_memory_usage_bytes`: 메모리 사용량
- `law_collection_cpu_usage_percent`: CPU 사용률
- `law_collection_throughput_laws_per_minute`: 처리량 (분당 법률 수)
- `law_collection_status`: 수집 상태 (0=중지, 1=실행, 2=일시정지)

### 시스템 메트릭

- `node_cpu_seconds_total`: CPU 사용 시간
- `node_memory_MemTotal_bytes`: 총 메모리
- `node_memory_MemAvailable_bytes`: 사용 가능한 메모리
- `node_filesystem_size_bytes`: 파일시스템 크기
- `node_network_receive_bytes_total`: 네트워크 수신 바이트

## 대시보드

### 1. 법률 수집 성능 모니터링

- 처리된 페이지 수
- 수집된 법률 수
- 현재 페이지
- 수집 상태
- 페이지 처리 시간 (50th, 95th percentile)
- 처리량 (분당 법률 수)
- 메모리 사용량
- CPU 사용률
- 에러율 (유형별)

### 2. 시스템 개요

- 시스템 메모리 사용률
- CPU 사용률
- 디스크 사용률
- 네트워크 트래픽
- 서비스 상태

## 알림 규칙

### 법률 수집 알림

- **HighErrorRate**: 에러율이 0.1/초 이상 2분간 지속
- **HighMemoryUsage**: 메모리 사용량이 1GB 이상 5분간 지속
- **SlowProcessing**: 95th percentile 처리 시간이 60초 이상 3분간 지속
- **CollectionStalled**: 10분간 페이지 처리 없음
- **LowThroughput**: 처리량이 1 법률/분 미만 5분간 지속

### 시스템 알림

- **HighSystemMemoryUsage**: 시스템 메모리 사용률 90% 이상
- **HighCPUUsage**: CPU 사용률 80% 이상

## 로그 분석

### 분석 도구 사용법

```bash
# 특정 날짜 분석
python scripts/monitoring/analyze_logs.py --date 20250112

# 리포트 파일로 저장
python scripts/monitoring/analyze_logs.py --date 20250112 --output report.md

# 두 실행 결과 비교
python scripts/monitoring/analyze_logs.py --compare 20250111 20250112

# 다른 로그 디렉토리 지정
python scripts/monitoring/analyze_logs.py --date 20250112 --log-dir /path/to/logs
```

### 분석 리포트 내용

- 요약 정보 (총 페이지 수, 법률 수, 처리 시간)
- 성능 분석 (처리량, 효율성 점수)
- 콘텐츠 분석 (법률 유형별 분포, 중복률)
- 타임라인 분석 (시작/종료 시간, 소요 시간)

## 고급 설정

### Prometheus 설정 수정

`monitoring/prometheus/prometheus.yml` 파일을 수정하여:
- 스크래핑 간격 조정
- 새로운 타겟 추가
- 보존 기간 변경

### Grafana 대시보드 커스터마이징

1. Grafana 웹 인터페이스에서 대시보드 편집
2. JSON 파일 직접 수정 후 재시작
3. 새로운 패널 추가

### 알림 규칙 추가

`monitoring/prometheus/rules/alerts.yml`에 새로운 규칙 추가:

```yaml
- alert: CustomAlert
  expr: your_metric_expression
  for: 5m
  labels:
    severity: warning
  annotations:
    summary: "Custom alert triggered"
    description: "Alert details here"
```

## 문제 해결

### 일반적인 문제

1. **메트릭 서버 시작 실패**
   - 포트 8000이 이미 사용 중인지 확인
   - 방화벽 설정 확인

2. **Prometheus가 메트릭을 수집하지 못함**
   - `host.docker.internal` 접근 가능한지 확인
   - 메트릭 엔드포인트가 응답하는지 확인

3. **Grafana 대시보드가 비어있음**
   - Prometheus 데이터 소스 연결 확인
   - 시간 범위 설정 확인

### 디버깅 명령어

```bash
# Docker 컨테이너 상태 확인
docker-compose ps

# 로그 확인
docker-compose logs prometheus
docker-compose logs grafana

# 메트릭 엔드포인트 테스트
curl http://localhost:8000/metrics

# Prometheus 타겟 상태 확인
curl http://localhost:9090/api/v1/targets
```

## 성능 최적화

### 메트릭 수집 최적화

- 불필요한 메트릭 비활성화
- 스크래핑 간격 조정
- 보존 기간 단축

### 리소스 사용량 최적화

- Docker 컨테이너 리소스 제한 설정
- Prometheus 보존 정책 최적화
- Grafana 대시보드 패널 수 제한

## 보안 고려사항

- 기본 비밀번호 변경
- 네트워크 접근 제한
- SSL/TLS 인증서 설정 (프로덕션 환경)

## 확장 계획

1. **추가 서비스 모니터링**
   - FastAPI 서비스 메트릭
   - Gradio 서비스 메트릭
   - 데이터베이스 성능 메트릭

2. **고급 알림**
   - 이메일 알림
   - Slack 통합
   - 웹훅 알림

3. **자동화**
   - 성능 회귀 테스트
   - 자동 리포트 생성
   - 성능 임계값 자동 조정
