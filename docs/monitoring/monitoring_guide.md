# 모니터링 시스템 가이드

LawFirmAI의 모니터링 시스템에 대한 상세 가이드입니다.

## Grafana + Prometheus 기반 실시간 모니터링

LawFirmAI는 법률 수집 성능을 실시간으로 모니터링하는 시스템을 제공합니다.

### 주요 기능

- **실시간 메트릭 수집**: 페이지 처리, 법률 수집, 에러율 등
- **지속적 메트릭 누적**: 여러 실행에 걸쳐 메트릭 값 누적
- **Grafana 대시보드**: 시각적 모니터링 및 알림
- **성능 분석**: 처리량, 메모리 사용량, CPU 사용률 추적

### 빠른 시작

```bash
# 1. 모니터링 스택 시작
cd monitoring
docker-compose up -d

# 2. 메트릭 서버 독립 실행
python scripts/monitoring/metrics_collector.py --port 8000

# 3. 법률 수집 실행 (메트릭 포함)
python scripts/data_collection/assembly/collect_laws_optimized.py --sample 50 --enable-metrics
```

### 접근 URL

- **Grafana**: http://localhost:3000 (admin/admin123)
- **Prometheus**: http://localhost:9090
- **메트릭 엔드포인트**: http://localhost:8000/metrics

### 수집되는 메트릭

- `law_collection_pages_processed_total`: 처리된 총 페이지 수
- `law_collection_laws_collected_total`: 수집된 총 법률 수
- `law_collection_page_processing_seconds`: 페이지 처리 시간
- `law_collection_memory_usage_bytes`: 메모리 사용량
- `law_collection_cpu_usage_percent`: CPU 사용률

## 로그 확인

### React 프론트엔드 로그

```bash
# React 개발 서버 로그는 터미널에 직접 출력됩니다
# 프로덕션 빌드 시 브라우저 콘솔에서 확인 가능
```

### API 서버 로그

```bash
# Windows PowerShell - 실시간 로그 모니터링
Get-Content logs\api_server.log -Wait -Tail 50

# Windows CMD - 전체 로그 확인
type logs\api_server.log

# Linux/Mac - 실시간 로그 모니터링
tail -f logs/api_server.log

# Linux/Mac - 최근 50줄 확인
tail -n 50 logs/api_server.log
```

### LangGraph 워크플로우 로그

```bash
# LangGraph 워크플로우 실행 로그 확인
# 로그는 logs/ 디렉토리에 자동 저장됩니다
```

### 로그 레벨 설정

```bash
# DEBUG 레벨로 실행 (더 자세한 로그)
# Windows
set LOG_LEVEL=DEBUG
cd api
python main.py

# PowerShell
$env:LOG_LEVEL="DEBUG"
cd api
python main.py

# Linux/Mac
export LOG_LEVEL=DEBUG
cd api
python main.py
```

### 로그 파일 위치

- **API 서버 로그**: `logs/api_server.log`
- **데이터 처리 로그**: `logs/` 디렉토리의 각종 `.log` 파일들

자세한 내용은 [Windows 모니터링 가이드](docs/development/windows_monitoring_guide.md)를 참조하세요.

