# Windows 환경에서 Grafana + Prometheus 모니터링 사용법

## 🪟 Windows 환경 설정

### 1. Docker Desktop 설치 및 실행

1. **Docker Desktop 다운로드 및 설치**
   - https://www.docker.com/products/docker-desktop/ 에서 다운로드
   - Windows용 Docker Desktop 설치

2. **Docker Desktop 실행**
   - 시작 메뉴에서 "Docker Desktop" 실행
   - 시스템 트레이에서 Docker 아이콘이 녹색이 될 때까지 대기

### 2. 모니터링 스택 시작

#### 방법 1: PowerShell 스크립트 사용 (권장)

```powershell
# 모니터링 디렉토리로 이동
cd D:\project\LawFirmAI\LawFirmAI\monitoring

# PowerShell 스크립트 실행
.\start_monitoring.ps1
```

#### 방법 2: 배치 파일 사용

```cmd
# 모니터링 디렉토리로 이동
cd D:\project\LawFirmAI\LawFirmAI\monitoring

# 배치 파일 실행
start_monitoring.bat
```

#### 방법 3: 직접 Docker Compose 실행

```cmd
# 모니터링 디렉토리로 이동
cd D:\project\LawFirmAI\LawFirmAI\monitoring

# Docker Compose 실행
docker-compose up -d
```

### 3. 서비스 접근

모니터링 스택이 시작되면 다음 URL로 접근할 수 있습니다:

- **Grafana**: http://localhost:3000 (admin/admin123)
- **Prometheus**: http://localhost:9090
- **Node Exporter**: http://localhost:9100
- **메트릭 엔드포인트**: http://localhost:8000/metrics

## 📊 법률 수집 실행 (메트릭 포함)

### 메트릭 수집 활성화

```cmd
# 메트릭 수집 활성화 (기본값)
python scripts\assembly\collect_laws_optimized.py --sample 50 --enable-metrics

# 메트릭 수집 비활성화
python scripts\assembly\collect_laws_optimized.py --sample 50 --disable-metrics
```

### 메트릭 서버 독립 실행 (권장)

**방법 1: 독립 메트릭 서버 + 스크립트 연결 (권장)**

```cmd
# 1단계: 메트릭 서버 독립 실행 (백그라운드)
python scripts\monitoring\metrics_collector.py --port 8000

# 2단계: 법률 수집 스크립트 실행 (메트릭 서버에 연결)
python scripts\assembly\collect_laws_optimized.py --sample 50 --enable-metrics
```

**방법 2: 스크립트가 직접 메트릭 서버 시작**

```cmd
# 스크립트가 메트릭 서버를 직접 시작하고 실행
python scripts\assembly\collect_laws_optimized.py --sample 50 --enable-metrics
```

### 메트릭 지속성 및 누적

메트릭 서버는 다음 기능을 제공합니다:

- **상태 지속성**: `data/metrics_state.json` 파일에 메트릭 상태 저장
- **누적 메트릭**: 여러 실행에 걸쳐 메트릭 값 누적
- **실시간 업데이트**: 페이지 처리 및 법률 수집 시마다 메트릭 업데이트
- **백그라운드 저장**: 30초마다 메트릭 상태 자동 저장

## 🔧 문제 해결

### Docker Desktop 관련 문제

1. **Docker Desktop이 시작되지 않는 경우**
   ```
   Error: unable to get image 'prom/node-exporter:latest': error during connect: Get "http://%2F%2F.%2Fpipe%2FdockerDesktopLinuxEngine/v1.48/images/prom/node-exporter:latest/json": open //./pipe/dockerDesktopLinuxEngine: The system cannot find the file specified.
   ```
   
   **해결방법:**
   - Docker Desktop이 실행 중인지 확인
   - Docker Desktop 재시작
   - Windows 재부팅 후 Docker Desktop 실행

2. **PowerShell 실행 정책 오류**
   ```
   .\start_monitoring.ps1 : 이 시스템에서 스크립트를 실행할 수 없으므로...
   ```
   
   **해결방법:**
   ```powershell
   # PowerShell을 관리자 권한으로 실행 후
   Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser
   ```

### 메트릭 서버 관련 문제

1. **UnicodeEncodeError 오류**
   ```
   UnicodeEncodeError: 'cp949' codec can't encode character '\U0001f4ca'
   ```
   
   **해결방법:** 이미 수정됨 (이모지 제거)

2. **포트 충돌 오류**
   ```
   Failed to start metrics server: [Errno 10048] Only one usage of each socket address
   ```
   
   **해결방법:**
   ```cmd
   # 다른 포트 사용
   python scripts\monitoring\metrics_collector.py --port 8001
   ```

## 📈 메트릭 확인 방법

### 1. 웹 브라우저에서 확인

- **Prometheus**: http://localhost:9090
  - "Status" → "Targets"에서 메트릭 수집 상태 확인
  - "Graph"에서 메트릭 쿼리 실행

- **Grafana**: http://localhost:3000
  - 대시보드에서 실시간 메트릭 시각화
  - 알림 설정 및 관리

### 2. PowerShell에서 확인

```powershell
# 메트릭 엔드포인트 확인
Invoke-WebRequest -Uri "http://localhost:8000/metrics" -UseBasicParsing

# 특정 메트릭만 확인
Invoke-WebRequest -Uri "http://localhost:8000/metrics" -UseBasicParsing | Select-Object -ExpandProperty Content | Select-String "law_collection"

# 주요 메트릭 값 확인
Invoke-WebRequest -Uri "http://localhost:8000/metrics" -UseBasicParsing | Select-Object -ExpandProperty Content | Select-String "law_collection_pages_processed_total|law_collection_laws_collected_total"
```

### 3. 메트릭 상태 파일 확인

```cmd
# 메트릭 상태 파일 내용 확인
type data\metrics_state.json

# 또는 PowerShell에서
Get-Content data\metrics_state.json
```

### 4. 명령 프롬프트에서 확인

```cmd
# curl 사용 (Windows 10/11)
curl http://localhost:8000/metrics

# 또는 PowerShell 명령 사용
powershell -Command "Invoke-WebRequest -Uri 'http://localhost:8000/metrics' -UseBasicParsing"
```

## 🛑 모니터링 스택 중지

### PowerShell 스크립트 사용

```powershell
.\stop_monitoring.ps1
```

### 배치 파일 사용

```cmd
stop_monitoring.bat
```

### 직접 Docker Compose 실행

```cmd
docker-compose down
```

## 📋 로그 분석

### 수집 완료 후 성능 분석

```cmd
# 특정 날짜 분석
python scripts\monitoring\analyze_logs.py --date 20250112

# 두 실행 결과 비교
python scripts\monitoring\analyze_logs.py --compare 20250111 20250112

# 리포트 생성
python scripts\monitoring\analyze_logs.py --date 20250112 --output report.md
```

## 🔍 성능 모니터링 체크리스트

### 수집 전 확인사항

- [ ] Docker Desktop이 실행 중인가?
- [ ] 모니터링 스택이 정상 시작되었는가?
- [ ] 메트릭 서버가 실행 중인가? (`http://localhost:8000/metrics`)
- [ ] 메트릭 상태 파일이 존재하는가? (`data/metrics_state.json`)
- [ ] Grafana 대시보드에 접근할 수 있는가?

### 수집 중 모니터링

- [ ] Grafana에서 실시간 메트릭 확인
- [ ] 메모리 사용량 모니터링
- [ ] 처리량 및 에러율 확인
- [ ] 알림 발생 여부 확인

### 수집 후 분석

- [ ] 로그 분석 도구로 성능 리포트 생성
- [ ] 이전 실행과 성능 비교
- [ ] 병목 지점 식별 및 개선 방안 도출

## 🎯 메트릭 수집 성공 사례

### 실제 테스트 결과

다음 명령어로 테스트한 결과:

```cmd
# 메트릭 서버 독립 실행
python scripts\monitoring\metrics_collector.py --port 8000

# 법률 수집 실행 (10개 샘플)
python scripts\assembly\collect_laws_optimized.py --sample 10 --start-page 595 --no-resume --enable-metrics
```

**결과:**
- 처리된 페이지: 36페이지
- 수집된 법률: 360개
- 메트릭 지속성: ✅ 성공
- Grafana 연동: ✅ 성공

### 메트릭 값 확인

```powershell
# 메트릭 값 확인
Invoke-WebRequest -Uri "http://localhost:8000/metrics" -UseBasicParsing | Select-Object -ExpandProperty Content | Select-String "law_collection_pages_processed_total|law_collection_laws_collected_total"
```

**출력 예시:**
```
law_collection_pages_processed_total 36.0
law_collection_laws_collected_total 360.0
```

## 💡 추가 팁

### 1. 방화벽 설정

Windows 방화벽에서 다음 포트가 허용되어야 합니다:
- 3000 (Grafana)
- 9090 (Prometheus)
- 9100 (Node Exporter)
- 8000 (메트릭 서버)

### 2. 리소스 최적화

- Docker Desktop 메모리 할당량 조정 (설정 → Resources → Memory)
- 불필요한 서비스 중지로 리소스 확보

### 3. 백업 및 복구

- Grafana 대시보드 설정 백업
- Prometheus 데이터 보존 정책 설정
- 모니터링 설정 파일 버전 관리

## 📝 업데이트 이력

### 2025-10-12: 메트릭 수집 기능 구현 완료

- ✅ **메트릭 서버 독립 실행**: 백그라운드에서 지속적으로 실행되는 메트릭 서버
- ✅ **메트릭 지속성**: 파일 기반 메트릭 상태 저장/복원 (`data/metrics_state.json`)
- ✅ **실시간 메트릭 누적**: 페이지 처리 및 법률 수집 시마다 메트릭 업데이트
- ✅ **Grafana 연동**: 법률 수집 성능 모니터링 대시보드에서 실시간 데이터 확인 가능
- ✅ **문제 해결**: 메트릭이 0으로 표시되던 문제 해결

### 테스트 완료 항목

- 메트릭 서버 독립 실행 및 백그라운드 유지
- 스크립트와 메트릭 서버 연결
- 메트릭 값 지속성 및 누적
- Grafana 대시보드 데이터 표시
- PowerShell 명령어를 통한 메트릭 확인

이제 Windows 환경에서도 완전한 모니터링 시스템을 사용할 수 있습니다! 🎉
