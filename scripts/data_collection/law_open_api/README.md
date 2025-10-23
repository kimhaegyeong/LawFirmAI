# Law Open API 데이터 수집 시스템

국가법령정보센터 OPEN API를 활용한 법령용어 주기적 수집 시스템입니다.

## 📋 개요

이 시스템은 Python schedule 라이브러리를 사용하여 법령용어 데이터를 주기적으로 수집하고 관리합니다.

### 주요 기능

- **증분 수집**: 변경된 데이터만 수집하여 효율성 향상
- **전체 수집**: 모든 데이터를 처음부터 수집
- **상세 정보 수집**: 법령용어 본문 조회 API를 활용한 상세 정보 수집
- **등록일자 순 정렬**: 등록일자 오름차순(rasc)으로 체계적 수집
- **체크포인트 시스템**: 중단 후 재시작을 위한 체크포인트 관리
- **배치 저장 시스템**: 일정 크기마다 파일로 저장하여 메모리 효율성 향상
- **자동 스케줄링**: 매일 자동으로 데이터 수집
- **수동 실행**: 필요시 수동으로 수집 실행
- **상태 모니터링**: 수집 상태 및 로그 모니터링
- **파일 기반 로깅**: 모든 작업 이력을 파일로 저장
- **진행 상황 표시**: 실시간 진행률 및 상태 표시

## 🏗️ 시스템 구조

```
scripts/data_collection/law_open_api/
├── collectors/          # 수집기 모듈
│   ├── incremental_legal_term_collector.py
│   └── __init__.py
├── schedulers/         # 스케줄러 모듈
│   ├── daily_scheduler.py
│   └── __init__.py
├── utils/              # 유틸리티 모듈
│   ├── timestamp_manager.py
│   ├── change_detector.py
│   ├── logging_utils.py
│   ├── checkpoint_manager.py  # 체크포인트 관리
│   └── __init__.py
├── scripts/            # 실행 스크립트
│   ├── start_legal_term_scheduler.py
│   ├── manual_collect_legal_terms.py
│   ├── monitor_collection_status.py
│   ├── manage_checkpoints.py  # 체크포인트 관리 스크립트
│   ├── start_scheduler.bat
│   ├── manual_collect.bat
│   ├── monitor_status.bat
│   └── __init__.py
└── __init__.py
```

## 📁 데이터 저장 구조

```
data/raw/law_open_api/
├── legal_terms/
│   ├── incremental/     # 증분 수집 데이터
│   │   └── daily/
│   │       └── YYYY-MM-DD/
│   │           ├── new_records.json
│   │           ├── updated_records.json
│   │           ├── deleted_records.json
│   │           ├── detailed_terms.json  # 상세 정보
│   │           └── summary.json
│   ├── batches/         # 배치 저장 데이터
│   │   ├── batch_YYYYMMDD_HHMMSS_001.json
│   │   ├── batch_YYYYMMDD_HHMMSS_002.json
│   │   ├── batch_summary_YYYYMMDD_HHMMSS.json
│   │   └── detailed_batches/  # 상세 정보 배치
│   │       ├── detailed_batch_YYYYMMDD_HHMMSS_001.json
│   │       └── detailed_batch_summary_YYYYMMDD_HHMMSS.json
│   └── full/           # 전체 수집 데이터
│       └── legal_terms_full.json
├── checkpoints/        # 체크포인트 데이터
│   ├── legal_terms_page_checkpoint.json
│   └── legal_terms_collection_checkpoint.json
└── metadata/
    ├── collection_timestamps.json
    └── change_log.json

logs/legal_term_collection/
├── collection_YYYYMMDD.log
├── scheduler_YYYYMMDD.log
└── errors_YYYYMMDD.log

reports/
└── legal_term_status_YYYYMMDD_HHMMSS.json
```

## ⚙️ 설정

### 환경변수 설정

```bash
# Windows
set LAW_OPEN_API_OC=your_email@example.com

# Linux/Mac
export LAW_OPEN_API_OC=your_email@example.com
```

### 설정 파일

`config/legal_term_collection_config.yaml` 파일에서 상세 설정을 관리합니다.

## 🚀 사용법

### 1. 자동 스케줄링 실행

#### Python 스크립트로 실행
```bash
python scripts/data_collection/law_open_api/scripts/start_legal_term_scheduler.py
```

#### Windows 배치 파일로 실행
```cmd
scripts\data_collection\law_open_api\scripts\start_scheduler.bat
```

### 2. 수동 수집 실행

#### 증분 수집 (기본)
```bash
python scripts/data_collection/law_open_api/scripts/manual_collect_legal_terms.py --mode incremental
```

#### 상세 정보 포함 증분 수집 (권장)
```bash
python scripts/data_collection/law_open_api/scripts/manual_collect_legal_terms.py --mode incremental --include-details
```

#### 전체 수집
```bash
python scripts/data_collection/law_open_api/scripts/manual_collect_legal_terms.py --mode full
```

#### 체크포인트에서 재시작하지 않고 처음부터 시작
```bash
python scripts/data_collection/law_open_api/scripts/manual_collect_legal_terms.py --mode incremental --no-resume
```

#### 배치 크기 설정 (메모리 효율성 향상)
```bash
# 작은 배치 크기 (500개씩 저장)
python scripts/data_collection/law_open_api/scripts/manual_collect_legal_terms.py --mode incremental --include-details --batch-size 500

# 큰 배치 크기 (2000개씩 저장)
python scripts/data_collection/law_open_api/scripts/manual_collect_legal_terms.py --mode incremental --include-details --batch-size 2000
```

#### Windows 배치 파일로 실행
```cmd
scripts\data_collection\law_open_api\scripts\manual_collect.bat
```

### 4. 체크포인트 관리

#### 체크포인트 목록 조회
```bash
python scripts/data_collection/law_open_api/scripts/manage_checkpoints.py list
```

#### 체크포인트 상세 정보 조회
```bash
python scripts/data_collection/law_open_api/scripts/manage_checkpoints.py show --data-type legal_terms
```

#### 체크포인트 삭제
```bash
python scripts/data_collection/law_open_api/scripts/manage_checkpoints.py delete --data-type legal_terms
```

#### 오래된 체크포인트 정리
```bash
python scripts/data_collection/law_open_api/scripts/manage_checkpoints.py cleanup --days 7
```

### 5. 상태 모니터링

#### Python 스크립트로 실행
```bash
python scripts/data_collection/law_open_api/scripts/monitor_collection_status.py
```

#### Windows 배치 파일로 실행
```cmd
scripts\data_collection\law_open_api\scripts\monitor_status.bat
```

## 📦 배치 저장 시스템

### 개요

배치 저장 시스템은 대용량 데이터 수집 시 메모리 효율성을 높이고 중간 결과를 보존하기 위해 일정 크기마다 파일로 저장하는 기능입니다.

### 주요 기능

- **자동 배치 저장**: 설정된 크기마다 자동으로 파일 저장
- **메모리 효율성**: 대용량 데이터 수집 시 메모리 사용량 최적화
- **중간 결과 보존**: 수집 중단 시에도 이미 저장된 배치 데이터 보존
- **배치 요약 정보**: 전체 배치 정보를 담은 요약 파일 생성
- **상세 정보 배치**: 상세 정보 수집 시 별도 배치 디렉토리 사용

### 배치 파일 구조

#### 일반 배치 파일
```json
{
  "batch_number": 1,
  "batch_size": 1000,
  "start_page": 1,
  "end_page": 10,
  "timestamp": "2025-10-23T16:08:21.549334",
  "terms": [
    {
      "id": "1",
      "법령용어ID": "13411",
      "법령용어명": "가격협상",
      "법령종류코드": "010101",
      "사전구분코드": "011403"
    }
  ]
}
```

#### 상세 정보 배치 파일
```json
{
  "batch_number": 1,
  "batch_size": 1000,
  "start_index": 1,
  "end_index": 1000,
  "timestamp": "2025-10-23T16:08:21.549334",
  "terms": [
    {
      "id": "1",
      "법령용어ID": "13411",
      "법령용어명": "가격협상",
      "detailed_info": {
        "법령용어명_한글": "가격협상",
        "법령용어명_한자": "價格協商",
        "법령용어정의": "Price negotiation"
      }
    }
  ]
}
```

#### 배치 요약 파일
```json
{
  "total_batches": 73,
  "total_terms": 72808,
  "batch_size": 1000,
  "timestamp": "20251023_160820",
  "start_time": "2025-10-23T16:08:23.648012",
  "end_time": "2025-10-23T16:08:23.648043",
  "query": "",
  "sort": "rasc",
  "max_pages": null
}
```

### 사용법

```bash
# 기본 배치 크기 (1000개)
python scripts/data_collection/law_open_api/scripts/manual_collect_legal_terms.py --mode incremental --include-details

# 작은 배치 크기 (500개)
python scripts/data_collection/law_open_api/scripts/manual_collect_legal_terms.py --mode incremental --include-details --batch-size 500

# 큰 배치 크기 (2000개)
python scripts/data_collection/law_open_api/scripts/manual_collect_legal_terms.py --mode incremental --include-details --batch-size 2000
```

## 🔄 체크포인트 시스템

### 개요

체크포인트 시스템은 대용량 데이터 수집 중 중단이 발생해도 마지막 위치부터 재시작할 수 있도록 지원합니다.

### 주요 기능

- **자동 체크포인트 저장**: 매 10페이지마다 자동으로 체크포인트 저장
- **중단 후 재시작**: 마지막 수집 페이지부터 자동 재시작
- **진행 상황 추적**: 현재 페이지, 전체 페이지, 수집된 항목 수 추적
- **체크포인트 관리**: 체크포인트 조회, 삭제, 정리 기능

### 체크포인트 데이터 구조

#### 페이지 체크포인트
```json
{
  "data_type": "legal_terms",
  "current_page": 10,
  "total_pages": 729,
  "collected_count": 1000,
  "last_term_id": "13411",
  "timestamp": "2025-10-23T15:52:59.905431",
  "status": "in_progress"
}
```

#### 수집 체크포인트
```json
{
  "data_type": "legal_terms",
  "collection_info": {
    "include_details": true,
    "last_collection": "2025-10-23T15:24:07.719018"
  },
  "timestamp": "2025-10-23T15:52:59.905431",
  "status": "collection_in_progress"
}
```

## 📚 상세 정보 수집

### 개요

법령용어 본문 조회 API를 활용하여 각 법령용어의 상세 정보를 수집합니다.

### 수집되는 상세 정보

- **법령용어명_한글**: 한글 법령용어명
- **법령용어명_한자**: 한자 법령용어명
- **법령용어정의**: 영어 번역 포함 정의
- **법령용어코드명**: 법령용어 코드명
- **출처 정보**: 관련 법령 정보

### 사용법

```bash
# 상세 정보 포함 수집
python scripts/data_collection/law_open_api/scripts/manual_collect_legal_terms.py --mode incremental --include-details
```

## 📊 모니터링 및 로깅

### 로그 파일

- **collection_YYYYMMDD.log**: 수집 작업 로그
- **scheduler_YYYYMMDD.log**: 스케줄러 실행 로그
- **errors_YYYYMMDD.log**: 에러 로그

### 상태 보고서

모니터링 스크립트를 실행하면 `reports/` 디렉토리에 JSON 형식의 상태 보고서가 생성됩니다.

## 🔧 고급 사용법

### 테스트 모드

```bash
# 스케줄러 설정만 확인
python scripts/data_collection/law_open_api/scripts/start_legal_term_scheduler.py --test

# API 연결 테스트만 실행
python scripts/data_collection/law_open_api/scripts/manual_collect_legal_terms.py --test
```

### 상세 로깅

```bash
# 상세 로깅 활성화
python scripts/data_collection/law_open_api/scripts/manual_collect_legal_terms.py --verbose
```

### 체크포인트 활용

```bash
# 체크포인트에서 재시작 (기본값)
python scripts/data_collection/law_open_api/scripts/manual_collect_legal_terms.py --mode incremental

# 처음부터 시작 (체크포인트 무시)
python scripts/data_collection/law_open_api/scripts/manual_collect_legal_terms.py --mode incremental --no-resume

# 체크포인트 상태 확인
python scripts/data_collection/law_open_api/scripts/manage_checkpoints.py show --data-type legal_terms
```

### 보고서 생성

```bash
# 상태 보고서 생성
python scripts/data_collection/law_open_api/scripts/monitor_collection_status.py --output reports/my_report.json
```

## 📦 필요한 패키지

```bash
pip install schedule pyyaml requests
```

## 🛠️ 문제 해결

### 일반적인 문제

1. **API 연결 실패**
   - `LAW_OPEN_API_OC` 환경변수 확인
   - 인터넷 연결 상태 확인
   - API 서비스 상태 확인

2. **권한 오류**
   - 데이터 디렉토리 쓰기 권한 확인
   - 로그 디렉토리 쓰기 권한 확인

3. **메모리 부족**
   - 설정 파일에서 `max_memory_mb` 값 조정
   - 배치 크기 조정

### 로그 확인

문제 발생 시 다음 로그 파일들을 확인하세요:

```bash
# 최근 로그 파일 확인
ls -la logs/legal_term_collection/

# 에러 로그 확인
tail -f logs/legal_term_collection/errors_$(date +%Y%m%d).log
```

## 📈 성능 최적화

### 설정 조정

`config/legal_term_collection_config.yaml`에서 다음 설정을 조정할 수 있습니다:

- `performance.memory.max_memory_mb`: 최대 메모리 사용량
- `performance.batch.size`: 배치 크기
- `api.min_request_interval`: API 요청 간격 (기본값: 1.0초)
- `api.page_size`: 페이지당 항목 수 (기본값: 100)

### 배치 저장 최적화

- **배치 크기 조정**: 메모리 상황에 따라 배치 크기 조정
  - 작은 배치 (500개): 메모리가 제한적인 환경
  - 기본 배치 (1000개): 일반적인 환경
  - 큰 배치 (2000개): 메모리가 충분한 환경
- **배치 파일 관리**: 정기적으로 오래된 배치 파일 정리
- **디스크 공간 관리**: 배치 파일들이 차지하는 디스크 공간 모니터링

### 체크포인트 최적화

- **체크포인트 저장 주기**: 매 10페이지마다 자동 저장
- **체크포인트 정리**: 정기적으로 오래된 체크포인트 정리
- **메모리 관리**: 체크포인트 데이터는 JSON 파일로 저장하여 메모리 효율성 확보

### 진행 상황 표시

- **실시간 진행률**: 페이지별 진행률 표시
- **수집 통계**: 수집된 항목 수, 새로운 레코드, 업데이트된 레코드 표시
- **상세 정보**: 체크포인트 상태, 마지막 수집 시간 등 표시

### 모니터링

정기적으로 상태 모니터링을 실행하여 시스템 상태를 확인하세요:

```bash
python scripts/data_collection/law_open_api/scripts/monitor_collection_status.py
```

## 🔄 유지보수

### 로그 정리

```bash
# 30일 이상 된 로그 파일 정리
python scripts/data_collection/law_open_api/scripts/monitor_collection_status.py --cleanup --days 30
```

### 체크포인트 정리

```bash
# 7일 이상 된 체크포인트 정리
python scripts/data_collection/law_open_api/scripts/manage_checkpoints.py cleanup --days 7
```

### 데이터 백업

정기적으로 다음 디렉토리를 백업하세요:

- `data/raw/law_open_api/legal_terms/`: 수집된 법령용어 데이터
- `data/raw/law_open_api/checkpoints/`: 체크포인트 데이터
- `data/raw/law_open_api/metadata/`: 메타데이터

### 시스템 상태 점검

```bash
# 전체 시스템 상태 점검
python scripts/data_collection/law_open_api/scripts/monitor_collection_status.py

# 체크포인트 상태 확인
python scripts/data_collection/law_open_api/scripts/manage_checkpoints.py list
```

## 📞 지원

문제가 발생하거나 질문이 있으시면:

1. 로그 파일 확인
2. 상태 모니터링 실행
3. GitHub Issues에 문제 보고

---

**참고**: 이 시스템은 국가법령정보센터 OPEN API의 이용약관을 준수하여 사용하시기 바랍니다.




