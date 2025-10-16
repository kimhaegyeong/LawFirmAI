# LawFirmAI 로깅 가이드

## 개요

LawFirmAI 프로젝트의 로깅 시스템과 로그 확인 방법에 대한 종합 가이드입니다.

## 로깅 시스템 구조

### 로그 레벨
- **DEBUG**: 상세한 디버깅 정보
- **INFO**: 일반적인 정보 메시지
- **WARNING**: 경고 메시지
- **ERROR**: 오류 메시지
- **CRITICAL**: 심각한 오류 메시지

### 로그 출력 위치
1. **콘솔 출력**: 실시간 로그 확인
2. **파일 출력**: `logs/` 디렉토리에 저장

## Gradio 애플리케이션 로그

### 로그 파일 위치
- **메인 로그**: `logs/gradio_app.log`
- **백업 로그**: 자동 로테이션 (필요시)

### 실시간 로그 확인

#### Windows 환경
```powershell
# PowerShell - 실시간 로그 모니터링
Get-Content logs\gradio_app.log -Wait -Tail 50

# PowerShell - 최근 100줄 확인
Get-Content logs\gradio_app.log -Tail 100

# CMD - 전체 로그 확인
type logs\gradio_app.log

# CMD - 최근 50줄 확인
powershell "Get-Content logs\gradio_app.log -Tail 50"
```

#### Linux/Mac 환경
```bash
# 실시간 로그 모니터링
tail -f logs/gradio_app.log

# 최근 50줄 확인
tail -n 50 logs/gradio_app.log

# 최근 100줄 확인
tail -n 100 logs/gradio_app.log

# 전체 로그 확인
cat logs/gradio_app.log

# 로그 검색 (특정 키워드)
grep "ERROR" logs/gradio_app.log
grep "WARNING" logs/gradio_app.log
grep "DEBUG" logs/gradio_app.log
```

### 로그 레벨 설정

#### 환경변수로 설정
```bash
# Windows CMD
set LOG_LEVEL=DEBUG
python gradio/app.py

# Windows PowerShell
$env:LOG_LEVEL="DEBUG"
python gradio/app.py

# Linux/Mac
export LOG_LEVEL=DEBUG
python gradio/app.py
```

#### 코드에서 설정
```python
# gradio/app.py에서 직접 수정
log_level = os.getenv("LOG_LEVEL", "DEBUG").upper()  # 기본값을 DEBUG로 변경
```

### 로그 필터링

#### 특정 레벨만 확인
```bash
# ERROR 레벨만 확인
grep "ERROR" logs/gradio_app.log

# WARNING 이상 레벨 확인
grep -E "(WARNING|ERROR|CRITICAL)" logs/gradio_app.log

# 특정 모듈 로그만 확인
grep "ChatService" logs/gradio_app.log
grep "RAGService" logs/gradio_app.log
```

#### 시간 범위로 필터링
```bash
# 특정 시간대 로그 확인 (예: 14:00-15:00)
grep "14:" logs/gradio_app.log | grep "15:"

# 오늘 로그만 확인
grep "$(date +%Y-%m-%d)" logs/gradio_app.log
```

## 데이터 처리 로그

### 로그 파일 종류
- `batch_update_law_content.log`: 법률 내용 배치 업데이트
- `build_faiss_from_sqlite.log`: FAISS 인덱스 빌드
- `database_analysis.log`: 데이터베이스 분석
- `preprocessing.log`: 데이터 전처리
- `validation.log`: 데이터 검증

### 로그 확인 명령어
```bash
# 특정 처리 단계 로그 확인
cat logs/preprocessing.log
cat logs/validation.log

# 에러가 발생한 로그만 확인
grep -i "error\|exception\|failed" logs/*.log

# 성공한 처리만 확인
grep -i "success\|completed\|finished" logs/*.log
```

## 로그 분석 도구

### 기본 분석 명령어
```bash
# 로그 파일 크기 확인
ls -lh logs/*.log

# 로그 파일 개수 확인
ls logs/*.log | wc -l

# 가장 큰 로그 파일 확인
ls -lh logs/*.log | sort -k5 -hr | head -5

# 최근 수정된 로그 파일 확인
ls -lt logs/*.log | head -10
```

### 고급 분석 명령어
```bash
# 로그 레벨별 통계
grep -o "\[DEBUG\]\|\[INFO\]\|\[WARNING\]\|\[ERROR\]\|\[CRITICAL\]" logs/gradio_app.log | sort | uniq -c

# 시간대별 로그 분포
grep -o "[0-9][0-9]:[0-9][0-9]" logs/gradio_app.log | sort | uniq -c

# 에러 메시지 요약
grep "ERROR" logs/gradio_app.log | cut -d' ' -f4- | sort | uniq -c | sort -nr

# 특정 키워드 빈도 확인
grep -o "ChatService\|RAGService\|DatabaseManager" logs/gradio_app.log | sort | uniq -c
```

## 로그 로테이션

### 자동 로테이션 설정 (권장)
```bash
# logrotate 설정 파일 생성
sudo nano /etc/logrotate.d/lawfirmai

# 설정 내용
/path/to/LawFirmAI/logs/*.log {
    daily
    missingok
    rotate 7
    compress
    delaycompress
    notifempty
    create 644 user group
}
```

### 수동 로그 정리
```bash
# 7일 이상 된 로그 파일 압축
find logs/ -name "*.log" -mtime +7 -exec gzip {} \;

# 30일 이상 된 로그 파일 삭제
find logs/ -name "*.log.gz" -mtime +30 -delete

# 빈 로그 파일 삭제
find logs/ -name "*.log" -size 0 -delete
```

## 디버깅을 위한 로그 설정

### 개발 환경 로그 설정
```python
# gradio/app.py에서 개발용 로그 설정
import logging

# 개발 환경에서는 DEBUG 레벨 사용
if os.getenv("ENVIRONMENT") == "development":
    logging.basicConfig(
        level=logging.DEBUG,
        format='%(asctime)s - %(name)s - %(levelname)s - %(funcName)s:%(lineno)d - %(message)s',
        handlers=[
            logging.StreamHandler(sys.stdout),
            logging.FileHandler('logs/gradio_app_debug.log', encoding='utf-8')
        ]
    )
```

### 프로덕션 환경 로그 설정
```python
# 프로덕션 환경에서는 INFO 레벨 사용
if os.getenv("ENVIRONMENT") == "production":
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(sys.stdout),
            logging.FileHandler('logs/gradio_app.log', encoding='utf-8')
        ]
    )
```

## 로그 모니터링 스크립트

### 실시간 에러 모니터링
```bash
#!/bin/bash
# monitor_errors.sh

echo "LawFirmAI 에러 모니터링 시작..."
tail -f logs/gradio_app.log | grep --line-buffered "ERROR\|CRITICAL" | while read line; do
    echo "[$(date)] ERROR DETECTED: $line"
    # 필요시 알림 전송 (이메일, 슬랙 등)
done
```

### 로그 통계 생성
```bash
#!/bin/bash
# log_stats.sh

echo "=== LawFirmAI 로그 통계 ==="
echo "생성일: $(date)"
echo ""

echo "=== 로그 파일 크기 ==="
ls -lh logs/*.log

echo ""
echo "=== 로그 레벨별 통계 ==="
grep -o "\[DEBUG\]\|\[INFO\]\|\[WARNING\]\|\[ERROR\]\|\[CRITICAL\]" logs/gradio_app.log | sort | uniq -c

echo ""
echo "=== 최근 에러 (최근 10개) ==="
grep "ERROR" logs/gradio_app.log | tail -10
```

## 문제 해결

### 일반적인 문제

1. **로그 파일이 생성되지 않음**
   - `logs/` 디렉토리 권한 확인
   - 디스크 공간 확인
   - 로그 설정 확인

2. **로그가 너무 많음**
   - 로그 레벨을 INFO로 변경
   - 로그 로테이션 설정
   - 불필요한 로그 제거

3. **로그 파일이 너무 큼**
   - 로그 로테이션 설정
   - 오래된 로그 파일 삭제
   - 로그 압축

### 디버깅 팁

1. **특정 기능 디버깅**
   ```bash
   # 특정 서비스 로그만 확인
   grep "ChatService" logs/gradio_app.log | tail -20
   ```

2. **성능 문제 디버깅**
   ```bash
   # 처리 시간 관련 로그 확인
   grep "executed in" logs/gradio_app.log
   ```

3. **메모리 문제 디버깅**
   ```bash
   # 메모리 관련 로그 확인
   grep -i "memory\|oom\|out of memory" logs/gradio_app.log
   ```

## 로그 보안

### 민감한 정보 제거
- API 키, 비밀번호 등은 로그에 기록하지 않음
- 개인정보는 마스킹 처리
- 디버그 로그는 프로덕션에서 비활성화

### 로그 접근 제어
- 로그 파일 권한 설정 (644)
- 로그 디렉토리 권한 설정 (755)
- 필요시 로그 암호화

## 모니터링 통합

### Grafana 대시보드 연동
- 로그 메트릭을 Prometheus로 전송
- Grafana에서 로그 시각화
- 알림 규칙 설정

### 외부 로그 수집 시스템
- ELK Stack (Elasticsearch, Logstash, Kibana)
- Fluentd
- Splunk

## 참고 자료

- [Python Logging 공식 문서](https://docs.python.org/3/library/logging.html)
- [Gradio 로깅 가이드](https://gradio.app/docs/#logging)
- [Docker 로깅 가이드](https://docs.docker.com/config/containers/logging/)

