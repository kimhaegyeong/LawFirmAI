# 헌재결정례 날짜 기반 수집 전략

## 개요

헌재결정례 데이터를 체계적으로 수집하기 위한 날짜 기반 수집 전략을 구현했습니다. 이 전략은 연도별, 분기별, 월별로 헌재결정례를 수집하며, 배치 단위 저장과 체크포인트 복구 기능을 제공합니다.

## 주요 기능

### 1. 날짜 기반 수집 전략

- **연도별 수집**: 특정 연도의 모든 헌재결정례 수집
- **분기별 수집**: 특정 분기(1-4분기)의 헌재결정례 수집
- **월별 수집**: 특정 월의 헌재결정례 수집
- **종국일자/선고일자 기준**: 두 가지 날짜 기준으로 수집 가능

### 2. 배치 단위 저장

- **10건 배치 저장**: 10건마다 자동으로 파일 저장
- **100건 안전장치**: 100건마다 추가 안전장치
- **갑작스런 종료 방지**: 프로그램 중단 시에도 수집된 데이터 보존

### 3. 체크포인트 복구

- **진행 상황 기록**: `checkpoint.json`에 수집 진행 상황 저장
- **중단 복구**: 중단된 지점부터 수집 재개 가능
- **자동 복구**: 수집 시작 시 체크포인트 자동 감지

## 데이터 구조

### ConstitutionalDecisionData 클래스

```python
@dataclass
class ConstitutionalDecisionData:
    """헌재결정례 데이터 클래스 - 목록 데이터 내부에 본문 데이터 포함"""
    # 목록 조회 API 응답 (기본 정보)
    id: str  # 검색결과번호
    사건번호: str
    종국일자: str
    헌재결정례일련번호: str
    사건명: str
    헌재결정례상세링크: str
    
    # 상세 조회 API 응답 (본문 데이터)
    사건종류명: Optional[str] = None
    판시사항: Optional[str] = None
    결정요지: Optional[str] = None
    전문: Optional[str] = None
    참조조문: Optional[str] = None
    참조판례: Optional[str] = None
    심판대상조문: Optional[str] = None
    
    # 메타데이터
    document_type: str = "constitutional_decision"
    collected_at: str = field(default_factory=lambda: datetime.now().isoformat())
```

### 저장 파일 구조

```
data/raw/constitutional_decisions/
├── yearly_2025_20250926_100644/
│   ├── page_001_2025년_200461-200447_10건_20250926_100549.json
│   ├── page_002_2025년_200847-200785_10건_20250926_100837.json
│   ├── checkpoint.json
│   └── yearly_collection_summary_20250926_100644.json
```

## API 매개변수

### 목록 조회 API (`/lawSearch.do`)

- `target`: `detc` (헌재결정례)
- `type`: `JSON`
- `search`: `1` (헌재결정례명 검색)
- `sort`: `ddes` (선고일자 내림차순) 또는 `efdes` (종국일자 내림차순)
- `date`: 종국일자 (YYYYMMDD 형식)
- `edYd`: 종국일자 기간 검색 (YYYYMMDD-YYYYMMDD 형식)

### 상세 조회 API (`/lawService.do`)

- `target`: `detc`
- `type`: `JSON`
- `ID`: 헌재결정례 일련번호

## 사용법

### 기본 사용법

```bash
# 2025년 헌재결정례 수집 (종국일자 기준)
python scripts/constitutional_decision/collect_by_date.py --strategy yearly --year 2025 --final-date

# 2024년 헌재결정례 수집 (선고일자 기준)
python scripts/constitutional_decision/collect_by_date.py --strategy yearly --year 2024

# 특정 건수만 수집
python scripts/constitutional_decision/collect_by_date.py --strategy yearly --year 2025 --target 100 --final-date

# API 요청 간격 설정 (네트워크 안정성 향상)
python scripts/constitutional_decision/collect_by_date.py --strategy yearly --year 2024 --interval 3.0 --interval-range 2.0

# 체크포인트부터 재시작
python scripts/constitutional_decision/collect_by_date.py --strategy yearly --year 2024 --resume --interval 2.5 --interval-range 1.5
```

### 분기별 수집

```bash
# 2025년 1분기 수집
python scripts/constitutional_decision/collect_by_date.py --strategy quarterly --year 2025 --quarter 1

# 2025년 2분기 수집
python scripts/constitutional_decision/collect_by_date.py --strategy quarterly --year 2025 --quarter 2

# 분기별 수집 + API 간격 설정
python scripts/constitutional_decision/collect_by_date.py --strategy quarterly --year 2025 --quarter 1 --interval 2.0 --interval-range 1.0
```

### 월별 수집

```bash
# 2025년 8월 수집
python scripts/constitutional_decision/collect_by_date.py --strategy monthly --year 2025 --month 8

# 2025년 12월 수집
python scripts/constitutional_decision/collect_by_date.py --strategy monthly --year 2025 --month 12

# 월별 수집 + API 간격 설정
python scripts/constitutional_decision/collect_by_date.py --strategy monthly --year 2025 --month 8 --interval 2.5 --interval-range 1.5
```

## 배치 저장 메커니즘

### 1. 자동 배치 저장

```python
# 10건마다 자동 저장
if len(batch_decisions) >= 10:
    self._save_batch(batch_decisions, output_dir, page, category)
    batch_decisions = []  # 배치 초기화
```

### 2. 체크포인트 저장

```python
def _save_checkpoint(self, output_dir: Path, page_num: int, collected_count: int):
    """체크포인트 저장 (진행 상황 기록)"""
    checkpoint_data = {
        "checkpoint_info": {
            "last_page": page_num,
            "collected_count": collected_count,
            "timestamp": datetime.now().isoformat(),
            "status": "in_progress"
        }
    }
```

### 3. 체크포인트 복구

```python
def _load_checkpoint(self, output_dir: Path) -> dict:
    """체크포인트 로드 (중단된 수집 재개)"""
    checkpoint_file = output_dir / "checkpoint.json"
    if checkpoint_file.exists():
        # 체크포인트에서 중단된 지점 확인
        # 해당 지점부터 수집 재개
```

## 데이터 품질 관리

### 1. 중복 방지

- 수집된 헌재결정례 ID를 메모리에 저장
- 중복 ID 발견 시 건너뛰기
- 중복 통계 제공

### 2. 오류 처리

- API 호출 실패 시 재시도
- 상세 정보 조회 실패 시 기본 정보만 저장
- 오류 통계 제공

### 3. 데이터 검증

- 필수 필드 존재 확인
- 결정요지가 비어있으면 전문 내용 사용
- 날짜 형식 검증

## 성능 최적화

### 1. 배치 처리

- 10건 단위로 파일 저장하여 메모리 효율성 향상
- 대용량 데이터 수집 시 안정성 보장

### 2. API 호출 최적화

- **사용자 설정 가능한 지연 시간**: `--interval`과 `--interval-range` 옵션으로 API 요청 간격 조정
- **기본 지연 시간**: 2-4초 (1-3초에서 증가)
- **개선된 재시도 메커니즘**: 지수 백오프 방식으로 재시도 간격 점진적 증가
- **타임아웃 설정 개선**: 연결 타임아웃(30초)과 읽기 타임아웃(120초) 분리
- **재시도 횟수 증가**: 5회 → 10회로 증가
- **원격 호스트 연결 끊김 처리**: `RemoteDisconnected`, `ConnectionResetError` 특별 처리
- 요청 제한 준수

### 3. 메모리 관리

- 수집된 데이터 즉시 파일로 저장
- **실시간 메모리 모니터링**: `psutil`을 사용한 메모리 사용량 추적
- **자동 메모리 정리**: 매 10페이지마다 가비지 컬렉션 실행
- **메모리 임계값 관리**: 800MB 이상 사용 시 자동 정리
- **대용량 데이터 구조 제한**: 수집된 결정례 ID를 10,000개로 제한
- 불필요한 메모리 사용 방지
- 가비지 컬렉션 최적화

## 모니터링 및 로깅

### 1. 실시간 진행 상황

```
2025-09-26 10:05:40 - INFO - ✅ 새로운 헌재결정례 수집: 교도소 내 부당처우행위 위헌확인 등 (ID: 200461)
2025-09-26 10:05:49 - INFO - ✅ 배치 저장 완료: page_001_2025년_200461-200447_10건_20250926_100549.json (10건)
2025-09-26 10:05:49 - INFO - 📋 체크포인트 저장: 페이지 1, 수집된 건수 10
2025-09-26 10:06:00 - INFO - 🧠 메모리 사용량: 245.3MB
2025-09-26 10:06:00 - DEBUG - 가비지 컬렉션 완료: 15개 객체 정리
```

### 2. 메모리 모니터링

**실시간 메모리 추적**:
- 매 10페이지마다 메모리 사용량 체크
- 800MB 이상 사용 시 자동 메모리 정리
- 1GB 이상 사용 시 경고 메시지 출력

**메모리 정리 과정**:
```
2025-09-26 10:06:00 - INFO - 🧠 메모리 사용량: 245.3MB
2025-09-26 10:06:00 - DEBUG - 가비지 컬렉션 완료: 15개 객체 정리
2025-09-26 10:06:00 - INFO - 메모리 정리: 수집된 결정례 5,000개 제거
2025-09-26 10:06:00 - INFO - 메모리 정리: 대기 중인 결정례 데이터 정리
```

### 3. 수집 통계

```json
{
  "statistics": {
    "total_collected": 50,
    "total_duplicates": 0,
    "total_errors": 0,
    "api_requests_made": 1,
    "api_errors": 0,
    "success_rate": 100.0
  }
}
```

## 문제 해결

### 1. 참조조문, 참조판례, 심판대상조문이 비어있는 경우

**원인**: LAW OPEN API에서 해당 필드가 실제로 빈 문자열로 반환됨

**해결책**: 
- 이는 시스템 오류가 아닌 API 데이터의 특성
- 일부 헌재결정례에서는 참조 정보가 제공되지 않음
- 수집된 데이터를 그대로 저장하는 것이 정확함

### 2. 중단된 수집 재개

```bash
# 동일한 명령어로 재실행하면 자동으로 체크포인트에서 재개
python scripts/constitutional_decision/collect_by_date.py --strategy yearly --year 2025 --final-date
```

### 3. 네트워크 연결 문제

**DNS 해결 실패 오류**:
```
Failed to resolve 'www.law.go.kr' ([Errno 11001] getaddrinfo failed)
```

**해결책**:
- 인터넷 연결 상태 확인
- 방화벽이나 프록시 설정 확인
- DNS 서버 설정 확인
- 잠시 후 다시 시도

**타임아웃 오류**:
```
Read timed out. (read timeout=60)
```

**해결책**:
- 연결 타임아웃이 30초, 읽기 타임아웃이 120초로 설정됨
- 네트워크 상태가 불안정한 경우 자동으로 재시도됨
- 재시도 간격이 점진적으로 증가 (최대 60초)

### 4. 메모리 관련 문제

**PyTorch 크래시**:
```
Unhandled exception caught in c10/util/AbortHandler.h
```

**해결책**:
- **실시간 메모리 모니터링**: 매 10페이지마다 메모리 사용량 체크
- **자동 메모리 정리**: 800MB 이상 사용 시 자동 가비지 컬렉션
- **메모리 사용량 제한**: 수집된 결정례 ID를 10,000개로 제한
- 다른 프로그램 종료하여 메모리 확보
- 목표 건수를 줄여서 다시 시도

### 5. 디렉토리 생성 오류

**오류**:
```
[Errno 2] No such file or directory: 'data\\raw\\constitutional_decisions\\...'
```

**해결책**:
- 출력 디렉토리 자동 생성 강화
- 디렉토리가 존재하지 않으면 자동으로 재생성
- 권한 문제가 있는 경우 관리자 권한으로 실행

### 6. 원격 호스트 연결 끊김 문제

**오류 메시지**:
```
연결 오류 (시도 1/10): ('Connection aborted.', RemoteDisconnected('Remote end closed connection without response'))
연결 오류 (시도 1/10): ('Connection aborted.', ConnectionResetError(10054, '현재 연결은 원격 호스트에 의해 강제로 끊겼습니다', None, 10054, None))
```

**가능한 원인**:
- **서버 측 요청 제한 (Rate Limiting)**: LAW OPEN API 일일 1000회 제한
- **방화벽/보안 정책**: 국가법령정보센터 서버의 보안 정책에 의한 차단
- **서버 과부하**: 서버가 일시적으로 과부하 상태이거나 점검 중
- **네트워크 불안정**: 인터넷 연결 상태 불안정

**해결책**:
- **API 요청 간격 설정**: `--interval`과 `--interval-range` 옵션으로 간격 조정
  ```bash
  # 더 긴 간격으로 설정 (3-5초)
  python scripts/constitutional_decision/collect_by_date.py --strategy yearly --year 2024 --interval 4.0 --interval-range 2.0
  
  # 보수적 설정 (5-7초)
  python scripts/constitutional_decision/collect_by_date.py --strategy yearly --year 2024 --interval 6.0 --interval-range 2.0
  ```
- **자동 재시도**: 지수 백오프 방식으로 재시도 간격 점진적 증가
- **상세한 오류 분석**: 원격 호스트 연결 끊김 원인별 구체적인 해결 방법 제시

### 7. 에러 핸들링 개선

**새로운 기능**:
- **상세한 오류 메시지**: 네트워크, 메모리 관련 오류에 대한 구체적인 해결 방법 제시
- **사용자 친화적 메시지**: 이모지와 함께 명확한 오류 설명 및 해결책 제공
- **오류 분류**: DNS, 연결, 타임아웃, 메모리, 원격 호스트 오류를 각각 다르게 처리

## 향후 개선 사항

### 1. 병렬 처리

- 여러 연도 동시 수집
- 멀티프로세싱 활용

### 2. 데이터 검증 강화

- 수집된 데이터 품질 검증
- 자동 데이터 정제

### 3. 웹 인터페이스

- 수집 진행 상황 실시간 모니터링
- 수집 설정 웹 UI 제공

## 관련 파일

- `scripts/constitutional_decision/date_based_collector.py`: 메인 수집 클래스
- `scripts/constitutional_decision/collect_by_date.py`: CLI 인터페이스
- `source/data/law_open_api_client.py`: API 클라이언트
- `data/raw/constitutional_decisions/`: 수집된 데이터 저장 위치

## 의존성

### 필수 라이브러리

```bash
# 메모리 모니터링
pip install psutil>=5.9.0

# 기본 라이브러리
pip install requests>=2.31.0
pip install beautifulsoup4>=4.12.0
pip install lxml>=4.9.0

# 데이터 처리
pip install pandas>=2.0.0
pip install numpy>=1.24.0

# 로깅
pip install colorlog>=6.7.0
```

### 전체 의존성 설치

```bash
# 프로젝트 루트에서 실행
pip install -r requirements.txt
```

## 최신 업데이트 (2025-09-26)

### 주요 개선사항

1. **네트워크 안정성 향상**
   - DNS 해결 실패 감지 및 처리
   - 타임아웃 설정 개선 (연결 30초, 읽기 120초)
   - 재시도 횟수 증가 (5회 → 10회)
   - 지수 백오프 방식 재시도 로직
   - **원격 호스트 연결 끊김 처리**: `RemoteDisconnected`, `ConnectionResetError` 특별 처리

2. **메모리 관리 강화**
   - 실시간 메모리 모니터링 (`psutil` 사용)
   - 자동 가비지 컬렉션 (매 10페이지마다)
   - 메모리 임계값 관리 (800MB 이상 시 정리)
   - 대용량 데이터 구조 제한

3. **에러 핸들링 개선**
   - 상세한 오류 메시지 및 해결 방법 제시
   - 사용자 친화적 오류 메시지 (이모지 포함)
   - 오류 유형별 분류 처리

4. **디렉토리 생성 강화**
   - 출력 디렉토리 자동 생성 및 재생성
   - 권한 문제 감지 및 해결 방법 제시

5. **API 요청 간격 설정 기능 (NEW)**
   - `--interval`: 기본 API 요청 간격 설정 (기본값: 2.0초)
   - `--interval-range`: 간격 범위 설정 (기본값: 2.0초)
   - 실제 간격: `interval ± interval-range` 범위에서 랜덤 선택
   - 네트워크 상태에 따른 유연한 간격 조정 가능