# 판례 수집 모듈

이 디렉토리는 LawFirmAI 프로젝트의 판례 수집 관련 모듈들을 포함합니다.

## 📁 디렉토리 구조

```
scripts/precedent/
├── README.md                           # 이 파일
├── collect_by_date.py                  # 날짜 기반 판례 수집 실행 스크립트 (NEW)
├── collect_precedents.py               # 기존 판례 수집 실행 스크립트
├── date_based_collector.py             # 날짜 기반 판례 수집 클래스 (NEW)
├── precedent_collector.py              # 기존 판례 수집 클래스
├── precedent_models.py                 # 판례 데이터 모델 정의
├── precedent_logger.py                 # 로깅 설정
└── run_precedent_collection.py         # 판례 수집 실행 스크립트
```

## 🚀 사용법

### 1. 날짜 기반 판례 수집 (NEW)

날짜별로 체계적인 판례 수집을 수행합니다. 각 수집 실행마다 별도의 폴더에 raw 데이터를 저장하며, **API 요청마다 즉시 파일을 생성**하여 데이터 손실을 방지합니다.

#### 주요 개선사항 (NEW)
- ✅ **페이지별 즉시 저장**: API 요청마다 즉시 파일 생성
- ✅ **판례일련번호 기준 파일명**: 파일명에 판례일련번호 범위 포함
- ✅ **실시간 진행상황**: 페이지별 상세 로그와 통계 정보
- ✅ **오류 복구**: 중간 오류 발생 시에도 데이터 자동 저장

#### 파일명 예시 (NEW)
```
page_001_민사-계약손해_123456-123789_150건_20250125_143045.json
page_002_형사_123790-124200_200건_20250125_143100.json
page_003_행정_124201-124500_180건_20250125_143200.json
```

#### 기본 사용법

##### **특정 연도 수집 (NEW)**
```bash
# 2024년 판례만 수집
python scripts/precedent/collect_by_date.py --strategy yearly --year 2024 --unlimited

# 2023년 판례만 수집
python scripts/precedent/collect_by_date.py --strategy yearly --year 2023 --unlimited

# 2022년 판례만 수집
python scripts/precedent/collect_by_date.py --strategy yearly --year 2022 --unlimited
```

##### **기간별 수집**
```bash
# 연도별 수집 (최근 5년, 연간 2000건)
python scripts/precedent/collect_by_date.py --strategy yearly --target 10000

# 분기별 수집 (최근 2년, 분기당 500건)
python scripts/precedent/collect_by_date.py --strategy quarterly --target 4000

# 월별 수집 (최근 1년, 월간 200건)
python scripts/precedent/collect_by_date.py --strategy monthly --target 2400

# 주별 수집 (최근 3개월, 주간 100건)
python scripts/precedent/collect_by_date.py --strategy weekly --target 1200

# 모든 전략 순차 실행 (총 17,600건)
python scripts/precedent/collect_by_date.py --strategy all --target 20000
```

#### 고급 옵션

##### **특정 연도 고급 옵션**
```bash
# 특정 연도 + 건수 제한
python scripts/precedent/collect_by_date.py --strategy yearly --year 2024 --target 5000

# 특정 연도 + 출력 디렉토리 지정
python scripts/precedent/collect_by_date.py --strategy yearly --year 2024 --unlimited --output data/custom/precedents

# 특정 연도 + 드라이런 모드
python scripts/precedent/collect_by_date.py --strategy yearly --year 2024 --unlimited --dry-run
```

##### **기간별 고급 옵션**
```bash
# 특정 기간 수 수집
python scripts/precedent/collect_by_date.py --strategy yearly --target 5000 --count 3

# 출력 디렉토리 지정
python scripts/precedent/collect_by_date.py --strategy monthly --target 1000 --output data/custom/precedents

# 드라이런 모드 (실제 수집 없이 계획만 출력)
python scripts/precedent/collect_by_date.py --strategy all --target 10000 --dry-run

# 재시작 모드 (중단된 지점부터 재시작)
python scripts/precedent/collect_by_date.py --strategy yearly --target 5000 --resume
```

### 2. 기존 키워드 기반 판례 수집

키워드 기반으로 판례를 수집합니다.

```bash
# 기본 판례 수집 (목표 5000건)
python scripts/precedent/collect_precedents.py

# 목표 건수 지정
python scripts/precedent/collect_precedents.py --target 10000

# 출력 디렉토리 지정
python scripts/precedent/collect_precedents.py --output data/custom/precedents
```

## 📊 수집 전략 비교

| 전략 | 기간 | 목표/기간 | 총 목표 | 예상 실제 | 폴더 수 | 특징 |
|------|------|-----------|---------|-----------|---------|------|
| **연도별** | 5년 | 2,000건/년 | 10,000건 | 8,000-12,000건 | 5개 | 최신 판례 우선 |
| **분기별** | 2년 | 500건/분기 | 4,000건 | 3,000-5,000건 | 8개 | 분기별 집중 수집 |
| **월별** | 1년 | 200건/월 | 2,400건 | 2,000-3,000건 | 12개 | 월별 세분화 |
| **주별** | 3개월 | 100건/주 | 1,200건 | 800-1,500건 | 12개 | 최신 주간 판례 |
| **키워드** | - | - | 5,000건 | 7,699건 | 1개 | 키워드 기반 검색 |

## 📁 폴더 구조

### 날짜 기반 수집 폴더 구조

```
data/raw/precedents/
├── yearly_2025_20250125_143022/          # 연도별 수집 (2025년)
│   ├── batch_민사_계약손해_150건_20250125_143045.json
│   ├── batch_형사_200건_20250125_143102.json
│   └── batch_행정_100건_20250125_143115.json
├── quarterly_2024Q4_20250125_144500/     # 분기별 수집 (2024년 4분기)
│   ├── batch_민사_계약손해_80건_20250125_144520.json
│   └── batch_형사_120건_20250125_144535.json
├── monthly_2024년12월_20250125_145000/   # 월별 수집 (2024년 12월)
│   └── batch_민사_계약손해_50건_20250125_145015.json
├── weekly_20250120주_20250125_145500/    # 주별 수집 (2025년 1월 20일 주)
│   └── batch_형사_30건_20250125_145520.json
├── yearly_collection_summary_20250125_143022.json
├── quarterly_collection_summary_20250125_144500.json
├── monthly_collection_summary_20250125_145000.json
└── weekly_collection_summary_20250125_145500.json
```

### 기존 키워드 기반 수집 폴더 구조

```
data/raw/precedents/
├── batch_민사_계약손해_200건_20250125_140000.json
├── batch_형사_300건_20250125_140500.json
├── batch_행정_150건_20250125_141000.json
├── collection_checkpoint_20250125_140000.json
└── collection_summary_20250125_141500.json
```

## 🔧 주요 기능

### 날짜 기반 수집 (NEW)

- **체계적 수집**: 연도별, 분기별, 월별, 주별 수집 전략
- **폴더별 저장**: 각 수집 실행마다 별도 폴더로 데이터 구분
- **선고일자 내림차순**: 최신 판례 우선 수집 (`sort: "ddes"`)
- **중복 방지**: 기존 수집 데이터와 중복 제거
- **배치 저장**: 100건 단위로 효율적인 저장
- **진행 상황 추적**: 실시간 진행 상황 모니터링

### 기존 키워드 기반 수집

- **키워드 기반**: 법률 분야별 키워드로 검색
- **다중 전략**: 6가지 검색 전략 동시 적용
- **우선순위**: 중요도에 따른 차등 수집
- **체크포인트**: 중단 시 진행 상황 저장 및 복구

## 📈 성능 최적화

### API 요청 최적화

- **선고일자 내림차순 정렬**: `sort: "ddes"` 파라미터 사용
- **배치 크기 최적화**: 한 번에 100건씩 조회
- **랜덤 지연**: API 요청 간 1-3초 랜덤 지연
- **재시도 메커니즘**: 최대 3회 재시도

### 메모리 최적화

- **배치 저장**: 100건마다 파일로 저장
- **중복 방지**: Set 자료구조로 중복 ID 관리
- **지연 로딩**: 필요할 때만 데이터 로드

## 📋 데이터 형식

### 배치 파일 구조

```json
{
  "metadata": {
    "category": "민사_계약손해",
    "count": 150,
    "saved_at": "2025-01-25T14:30:45.123456",
    "batch_id": "20250125_143045"
  },
  "precedents": [
    {
      "판례일련번호": "123456",
      "사건명": "계약금 반환 청구 사건",
      "사건번호": "2024다12345",
      "법원코드": "01",
      "사건유형코드": "01",
      "판결일자": "20241215",
      "선고일자": "20241215",
      "법원명": "대법원",
      "사건유형": "민사",
      "판결요지": "계약금의 성질과 반환 사유에 관한 판단...",
      "판결주문": "원고의 청구를 기각한다.",
      "참조조문": "민법 제565조",
      "참조판례": "대법원 2023다12345 판결",
      "판례내용": "상세한 판례 내용..."
    }
  ]
}
```

### 수집 요약 파일 구조

```json
{
  "strategy": "yearly",
  "years": [2025, 2024, 2023, 2022, 2021],
  "target_per_year": 2000,
  "collected_by_year": {
    "2025": 1850,
    "2024": 1950,
    "2023": 1800,
    "2022": 1750,
    "2021": 1650
  },
  "total_collected": 9000,
  "start_time": "2025-01-25T14:30:22.123456",
  "end_time": "2025-01-25T15:45:30.654321"
}
```

## 🚨 주의사항

### API 제한

- **일일 1000회 제한**: API 요청 수 모니터링 필요
- **요청 간 지연**: 1-3초 랜덤 지연 필수
- **재시도 제한**: 최대 3회 재시도

### 저장 공간

- **폴더 수 증가**: 수집 실행마다 새 폴더 생성
- **디스크 공간**: 대용량 데이터 저장 고려
- **정리 작업**: 오래된 폴더 정리 필요

## 📚 관련 문서

- [날짜 기반 판례 수집 전략 상세 문서](../docs/development/date_based_collection_strategy.md)
- [TASK별 상세 개발 계획](../docs/development/TASK/TASK별 상세 개발 계획_v1.0.md)
- [프로젝트 README](../../README.md)

## 🤝 기여하기

새로운 수집 전략이나 개선사항이 있으시면 이슈를 생성하거나 풀 리퀘스트를 보내주세요.

## 📄 라이선스

이 프로젝트는 MIT 라이선스 하에 배포됩니다.
