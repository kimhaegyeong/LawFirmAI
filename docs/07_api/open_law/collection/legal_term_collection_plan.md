# 법령용어 수집 계획서

## 📋 목차

1. [개요](#개요)
2. [API 분석](#api-분석)
3. [수집 전략](#수집-전략)
4. [구현 계획](#구현-계획)
5. [데이터 구조](#데이터-구조)
6. [실행 방법](#실행-방법)
7. [주의사항](#주의사항)
8. [참고 자료](#참고-자료)

---

## 개요

### 목표

국가법령정보 공동활용 LAW OPEN DATA의 두 엔드포인트를 활용하여 법령용어를 체계적으로 수집하고 관리합니다.

- **lsTrmListGuide**: 법령용어 목록 조회 (검색/페이징 지원)
- **lsTrmInfoGuide**: 법령용어 상세 조회 (정의, 출처 등 상세 정보)

### 수집 범위

- 법령용어명 (한글/한자)
- 법령용어 정의
- 법령용어 코드 및 분류
- 출처 정보
- 관련 법령 정보

### 활용 목적

1. **동의어 품질 관리 시스템 개선**
   - 하드코딩된 동의어를 동적으로 보강
   - 법률 도메인별 용어 자동 수집

2. **키워드 확장 정확도 향상**
   - 법률 전문 용어 사전 구축
   - 검색 쿼리 품질 개선

3. **법률 용어 표준화**
   - 국가법령정보센터 기준 용어 표준화
   - 용어 정의 및 출처 추적

---

## API 분석

### 1. lsTrmListGuide - 법령용어 목록 조회

#### 요청 정보

- **Base URL**: `http://www.law.go.kr/DRF/lawSearch.do`
- **Method**: GET
- **target**: `lstrm` (필수)

#### 요청 파라미터

| 파라미터 | 타입 | 필수 | 설명 | 기본값/허용값 |
|---------|------|------|------|--------------|
| OC | string | 필수 | 사용자 이메일 ID (g4c@korea.kr일 경우 OC=g4c) | - |
| target | string | 필수 | 서비스 대상 (lstrm) | lstrm |
| type | char | 필수 | 출력 형태 | HTML/XML/JSON |
| query | string | 선택 | 법령용어명에서 검색할 질의 | - |
| display | int | 선택 | 검색된 결과 개수 | 20 (max=100) |
| page | int | 선택 | 검색 결과 페이지 | 1 |
| sort | string | 선택 | 정렬 옵션 | lasc (법령용어명 오름차순) |
| regDt | string | 선택 | 등록일자 범위 검색 | YYYYMMDD~YYYYMMDD |
| gana | string | 선택 | 사전식 검색 | ga, na, da, ra, ma, ba, sa, a, ja, cha, ka, ta, pa, ha |
| popYn | string | 선택 | 상세화면 팝업창 여부 | Y/N |
| dicKndCd | int | 선택 | 법령 종류 코드 | 010101 (법령), 010102 (행정규칙) |

#### 정렬 옵션 (sort)

- `lasc`: 법령용어명 오름차순 (기본값)
- `ldes`: 법령용어명 내림차순
- `rasc`: 등록일자 오름차순
- `rdes`: 등록일자 내림차순

#### 샘플 URL

```bash
# JSON 형식으로 검색
http://www.law.go.kr/DRF/lawSearch.do?OC=test&target=lstrm&type=JSON&query=계약

# 사전식 검색 (가나다순)
http://www.law.go.kr/DRF/lawSearch.do?OC=test&target=lstrm&type=JSON&gana=ga

# 페이징 처리
http://www.law.go.kr/DRF/lawSearch.do?OC=test&target=lstrm&type=JSON&query=계약&page=1&display=100
```

#### 응답 필드

| 필드 | 타입 | 설명 |
|------|------|------|
| target | string | 검색서비스 대상 |
| 키워드 | string | 검색어 |
| section | string | 검색범위 |
| totalCnt | int | 검색건수 |
| page | int | 결과페이지번호 |
| lstrm id | int | 결과 번호 |
| 법령용어ID | string | 법령용어ID |
| 법령용어명 | string | 법령용어명 |
| 법령용어상세검색 | string | 법령용어상세검색 |
| 사전구분코드 | string | 사전구분코드 (011401: 법령용어사전, 011402: 법령정의사전, 011403: 법령한영사전) |
| 법령용어상세링크 | string | 법령용어상세링크 |
| 법령종류코드 | int | 법령 종류 코드 (010101: 법령, 010102: 행정규칙) |

### 2. lsTrmInfoGuide - 법령용어 상세 조회

#### 요청 정보

- **Base URL**: `http://www.law.go.kr/DRF/lawService.do`
- **Method**: GET
- **target**: `lstrm` (필수)

#### 요청 파라미터

| 파라미터 | 타입 | 필수 | 설명 |
|---------|------|------|------|
| OC | string | 필수 | 사용자 이메일 ID |
| target | string | 필수 | 서비스 대상 (lstrm) |
| type | char | 필수 | 출력 형태 (HTML/XML/JSON) |
| query | string | 필수 | 상세조회하고자 하는 법령용어명 |

#### 샘플 URL

```bash
# JSON 형식으로 상세 조회
http://www.law.go.kr/DRF/lawService.do?OC=test&target=lstrm&query=선박&type=JSON
```

#### 응답 필드

| 필드 | 타입 | 설명 |
|------|------|------|
| 법령용어 일련번호 | int | 법령용어 일련번호 |
| 법령용어명_한글 | string | 법령용어명 한글 |
| 법령용어명_한자 | string | 법령용어명 한자 |
| 법령용어코드 | int | 법령용어코드 |
| 법령용어코드명 | string | 법령용어코드명 |
| 출처 | string | 출처 |
| 법령용어정의 | string | 법령용어정의 |

---

## 수집 전략

### 수집 방식

#### 방식 A: 키워드 기반 수집 (권장) ⭐

**장점**:
- 법률 도메인별 체계적 수집
- 관련 용어 그룹화 용이
- 수집 범위 제어 가능

**키워드 목록**:

```python
# 민사법
["계약", "해지", "손해배상", "위약금", "채무", "채권", "계약해제", "계약불이행"]

# 형사법
["범죄", "형벌", "벌금", "징역", "구속", "기소", "공소", "재판"]

# 노동법
["근로", "해고", "임금", "근로시간", "휴가", "부당해고", "근로계약", "임금체불"]

# 가족법
["이혼", "양육권", "위자료", "재산분할", "친권", "면접교섭권", "부양"]

# 부동산법
["아파트", "매매", "임대", "등기", "전세", "월세", "부동산", "소유권"]

# 상법
["회사", "주식", "이사회", "주주", "합병", "분할", "상법", "법인"]
```

#### 방식 B: 사전식 검색 기반 수집

**장점**:
- 체계적인 전체 수집
- 누락 방지
- 알파벳 순서로 정리

**사전식 검색 목록**:

```python
gana_list = [
    'ga', 'na', 'da', 'ra', 'ma', 'ba', 'sa', 
    'a', 'ja', 'cha', 'ka', 'ta', 'pa', 'ha'
]
```

#### 방식 C: 전체 수집

**장점**:
- 모든 용어 수집
- 완전성 보장

**단점**:
- 시간 소요 큼
- API 제한 고려 필요

### 수집 워크플로우

```
┌─────────────────────────────────────────┐
│  1단계: 목록 수집 (lsTrmListGuide)      │
│  - 키워드/사전식 검색으로 용어 목록 조회 │
│  - 페이징 처리로 전체 목록 수집          │
└──────────────┬──────────────────────────┘
               │
               ▼
┌─────────────────────────────────────────┐
│  2단계: 상세 정보 수집 (lsTrmInfoGuide) │
│  - 각 용어의 상세 정보 조회             │
│  - 정의, 출처, 코드 정보 수집           │
└──────────────┬──────────────────────────┘
               │
               ▼
┌─────────────────────────────────────────┐
│  3단계: 데이터 정제 및 저장             │
│  - 중복 제거                             │
│  - 데이터 검증                           │
│  - JSON 형식으로 저장                    │
└──────────────┬──────────────────────────┘
               │
               ▼
┌─────────────────────────────────────────┐
│  4단계: 통계 및 모니터링                 │
│  - 수집 통계 생성                         │
│  - 수집 로그 저장                         │
└─────────────────────────────────────────┘
```

---

## 구현 계획

### 파일 구조

```
scripts/data_collection/legal_term/
├── open_law_term_collector.py      # 메인 수집 스크립트
├── open_law_term_api_client.py     # API 클라이언트
├── term_data_processor.py            # 데이터 처리 및 저장
└── term_collection_config.py       # 설정 관리
```

### 주요 클래스 설계

#### 1. OpenLawTermAPIClient

```python
class OpenLawTermAPIClient:
    """국가법령정보 법령용어 API 클라이언트"""
    
    def __init__(self, oc: str, base_url: str = "http://www.law.go.kr/DRF"):
        """
        Args:
            oc: 사용자 이메일 ID
            base_url: API 기본 URL
        """
        pass
    
    def search_terms(
        self, 
        query: str = "", 
        page: int = 1, 
        display: int = 100,
        sort: str = "lasc",
        gana: str = None,
        reg_dt: str = None
    ) -> Dict[str, Any]:
        """법령용어 목록 조회 (lsTrmListGuide)"""
        pass
    
    def get_term_detail(self, term_name: str) -> Dict[str, Any]:
        """법령용어 상세 조회 (lsTrmInfoGuide)"""
        pass
    
    def _make_request(self, url: str, params: Dict) -> Dict[str, Any]:
        """API 요청 실행 (재시도 로직 포함)"""
        pass
```

#### 2. OpenLawTermCollector

```python
class OpenLawTermCollector:
    """법령용어 수집기"""
    
    def __init__(self, api_client: OpenLawTermAPIClient):
        """수집기 초기화"""
        pass
    
    def collect_by_keywords(
        self, 
        keywords: List[str], 
        max_terms_per_keyword: int = None
    ) -> List[Dict]:
        """키워드 기반 수집"""
        pass
    
    def collect_by_gana(
        self, 
        gana_list: List[str], 
        max_terms: int = None
    ) -> List[Dict]:
        """사전식 검색 기반 수집"""
        pass
    
    def collect_all(self, max_terms: int = None) -> List[Dict]:
        """전체 수집"""
        pass
    
    def _collect_term_list(
        self, 
        query: str = "", 
        gana: str = None
    ) -> List[Dict]:
        """용어 목록 수집 (페이징 처리)"""
        pass
    
    def _enrich_with_detail(self, term: Dict) -> Dict:
        """상세 정보로 보강"""
        pass
```

#### 3. TermDataProcessor

```python
class TermDataProcessor:
    """수집된 용어 데이터 처리"""
    
    def process_term_list(self, list_data: Dict) -> List[Dict]:
        """목록 데이터 처리"""
        pass
    
    def enrich_with_detail(self, term: Dict) -> Dict:
        """상세 정보로 보강"""
        pass
    
    def deduplicate_terms(self, terms: List[Dict]) -> List[Dict]:
        """중복 제거 (일련번호 기준)"""
        pass
    
    def validate_term_data(self, term: Dict) -> bool:
        """용어 데이터 검증"""
        pass
    
    def save_terms(
        self, 
        terms: List[Dict], 
        filepath: str,
        format: str = "json"
    ) -> bool:
        """용어 데이터 저장"""
        pass
```

#### 4. TermCollectionConfig

```python
@dataclass
class TermCollectionConfig:
    """수집 설정"""
    oc: str
    base_url: str = "http://www.law.go.kr/DRF"
    output_dir: str = "data/raw/legal_terms"
    max_retries: int = 3
    retry_delay: float = 2.0
    rate_limit_delay: float = 0.5  # 요청 간 최소 지연 (초)
    batch_size: int = 50
    checkpoint_interval: int = 100  # 체크포인트 저장 간격
```

### 구현 단계

#### Phase 1: 기본 API 클라이언트 구현

- [ ] `OpenLawTermAPIClient` 클래스 구현
- [ ] 목록 조회 메서드 (`search_terms`)
- [ ] 상세 조회 메서드 (`get_term_detail`)
- [ ] 에러 처리 및 재시도 로직
- [ ] 요청 간 지연 처리 (Rate Limiting)

#### Phase 2: 수집기 구현

- [ ] `OpenLawTermCollector` 클래스 구현
- [ ] 키워드 기반 수집 로직
- [ ] 사전식 검색 기반 수집 로직
- [ ] 페이징 처리 로직
- [ ] 상세 정보 보강 로직

#### Phase 3: 데이터 처리 구현

- [ ] `TermDataProcessor` 클래스 구현
- [ ] 데이터 정제 로직
- [ ] 중복 제거 로직
- [ ] 데이터 검증 로직
- [ ] 저장 로직 (JSON 형식)

#### Phase 4: 고급 기능 구현

- [ ] 체크포인트 및 재개 기능
- [ ] 통계 및 모니터링
- [ ] 배치 처리 최적화
- [ ] 로깅 시스템

---

## 데이터 구조

### 저장 형식

#### JSON 형식

```json
{
  "법령용어_일련번호": 12345,
  "법령용어명_한글": "계약",
  "법령용어명_한자": "契約",
  "법령용어코드": 101,
  "법령용어코드명": "계약법",
  "출처": "민법 제105조",
  "법령용어정의": "당사자 간의 의사표시의 합치로 성립하는 법률행위",
  "사전구분코드": "011401",
  "사전구분명": "법령용어사전",
  "법령종류코드": 10101,
  "법령종류명": "법령",
  "수집일시": "2024-01-01T00:00:00",
  "수집방법": "keyword:계약",
  "수집키워드": "계약"
}
```

### 저장 위치

```
data/raw/legal_terms/
├── open_law_terms.json              # 전체 용어 데이터 (통합)
├── open_law_terms_by_keyword/       # 키워드별 분류
│   ├── 계약.json
│   ├── 손해배상.json
│   ├── 소송.json
│   └── ...
├── open_law_terms_by_gana/          # 사전식 검색 결과
│   ├── ga.json
│   ├── na.json
│   └── ...
└── metadata/
    ├── collection_log.json          # 수집 로그
    ├── statistics.json              # 통계 정보
    └── checkpoint.json              # 체크포인트 데이터
```

### 통계 정보 구조

```json
{
  "collection_session": {
    "session_id": "20240101_120000",
    "start_time": "2024-01-01T12:00:00",
    "end_time": "2024-01-01T13:30:00",
    "duration_seconds": 5400
  },
  "collection_method": "keyword",
  "statistics": {
    "total_collected": 1250,
    "total_keywords": 50,
    "total_pages": 125,
    "total_api_requests": 1375,
    "duplicates_removed": 25,
    "errors": 5
  },
  "by_keyword": {
    "계약": {
      "collected": 45,
      "pages": 5
    },
    "손해배상": {
      "collected": 32,
      "pages": 4
    }
  }
}
```

---

## 실행 방법

### 환경 설정

#### 1. 환경 변수 설정

```bash
# .env 파일 또는 환경 변수
export LAW_OPEN_API_OC="your_email_id"
```

#### 2. 의존성 설치

```bash
pip install requests python-dotenv
```

### 실행 명령어

#### 키워드 기반 수집 (권장)

```bash
# 기본 키워드로 수집
python scripts/data_collection/legal_term/open_law_term_collector.py \
    --method keyword \
    --keywords "계약,손해배상,소송" \
    --max-terms 1000

# 키워드 파일로 수집
python scripts/data_collection/legal_term/open_law_term_collector.py \
    --method keyword \
    --keyword-file data/keywords/civil_law_keywords.txt \
    --max-terms-per-keyword 50
```

#### 사전식 검색 기반 수집

```bash
# 특정 사전식 검색으로 수집
python scripts/data_collection/legal_term/open_law_term_collector.py \
    --method gana \
    --gana "ga,na,da" \
    --max-terms 5000

# 모든 사전식 검색으로 수집
python scripts/data_collection/legal_term/open_law_term_collector.py \
    --method gana \
    --gana all \
    --max-terms 10000
```

#### 전체 수집

```bash
# 전체 용어 수집 (시간 소요 큼)
python scripts/data_collection/legal_term/open_law_term_collector.py \
    --method all \
    --max-terms 20000
```

#### 추가 옵션

```bash
# 체크포인트에서 재개
python scripts/data_collection/legal_term/open_law_term_collector.py \
    --method keyword \
    --keywords "계약" \
    --resume

# 출력 디렉토리 지정
python scripts/data_collection/legal_term/open_law_term_collector.py \
    --method keyword \
    --keywords "계약" \
    --output-dir data/custom/legal_terms

# 상세 로그 출력
python scripts/data_collection/legal_term/open_law_term_collector.py \
    --method keyword \
    --keywords "계약" \
    --verbose
```

### 명령어 옵션

| 옵션 | 설명 | 기본값 |
|------|------|--------|
| `--method` | 수집 방식 (keyword/gana/all) | keyword |
| `--keywords` | 키워드 목록 (쉼표 구분) | - |
| `--keyword-file` | 키워드 파일 경로 | - |
| `--gana` | 사전식 검색 목록 (쉼표 구분 또는 all) | - |
| `--max-terms` | 최대 수집 용어 수 | 무제한 |
| `--max-terms-per-keyword` | 키워드당 최대 수집 용어 수 | 100 |
| `--output-dir` | 출력 디렉토리 | data/raw/legal_terms |
| `--resume` | 체크포인트에서 재개 | False |
| `--verbose` | 상세 로그 출력 | False |
| `--rate-limit` | 요청 간 지연 시간 (초) | 0.5 |

---

## 주의사항

### API 제한

1. **요청 간 지연**
   - API 서버 부하 방지를 위해 요청 간 최소 0.5초 지연 권장
   - 대량 수집 시 1초 이상 지연 고려

2. **일일 요청 한도**
   - 국가법령정보센터의 일일 요청 한도 확인 필요
   - 필요 시 여러 세션으로 분산 수집

3. **동시 요청 제한**
   - 동시 요청 수 제한 (권장: 1개씩 순차 처리)

### 데이터 품질

1. **중복 제거**
   - 법령용어 일련번호 기준으로 중복 제거
   - 한글/한자 매칭 검증

2. **데이터 검증**
   - 필수 필드 존재 여부 확인
   - 정의 데이터 유효성 검증
   - 출처 정보 검증

3. **데이터 정제**
   - HTML 태그 제거 (필요 시)
   - 특수문자 정규화
   - 공백 정리

### 메모리 관리

1. **배치 처리**
   - 대량 수집 시 배치 단위로 처리
   - 배치 처리 후 메모리 정리

2. **주기적 저장**
   - 일정 간격으로 중간 저장
   - 체크포인트 저장으로 중단 시 재개 가능

3. **메모리 모니터링**
   - 메모리 사용량 추적
   - 필요 시 가비지 컬렉션 실행

### 에러 처리

1. **재시도 로직**
   - 네트워크 오류 시 자동 재시도 (최대 3회)
   - 지수 백오프(exponential backoff) 적용

2. **에러 로깅**
   - 모든 에러를 로그 파일에 기록
   - 에러 통계 수집

3. **부분 실패 처리**
   - 일부 용어 수집 실패 시에도 계속 진행
   - 실패한 용어 목록 별도 저장

### 체크포인트 및 재개

1. **체크포인트 저장**
   - 일정 간격으로 진행 상황 저장
   - 수집된 키워드, 페이지, 용어 수 등 저장

2. **재개 기능**
   - 중단된 지점부터 재개
   - 중복 수집 방지

---

## 참고 자료

### 관련 문서

- [lsTrmListGuide 가이드](guides/lsTrmListGuide.md) - 법령용어 목록 조회 API 가이드
- [lsTrmInfoGuide 가이드](guides/lsTrmInfoGuide.md) - 법령용어 상세 조회 API 가이드
- [데이터 수집 가이드](../../02_data/collection/data_collection_guide.md) - 전체 데이터 수집 가이드
- [Open Law API 가이드 맵](guide_id_map.md) - 전체 API 가이드 맵

### 외부 링크

- [국가법령정보센터 Open API](https://open.law.go.kr/LSO/openApi/guideList.do) - 공식 API 가이드
- [LAW OPEN DATA](http://www.law.go.kr/DRF/lawService.do) - API 엔드포인트

### 관련 코드

- `scripts/data_collection/legal_term/term_collector.py` - 기존 용어 수집기
- `scripts/data_collection/legal_term/collect_legal_terms.py` - 용어 수집 실행 스크립트

---

## 구현 일정

### Phase 1: 기본 구현 (1주)

- [ ] API 클라이언트 구현
- [ ] 기본 수집 로직 구현
- [ ] 데이터 저장 로직 구현

### Phase 2: 고급 기능 (1주)

- [ ] 체크포인트 및 재개 기능
- [ ] 통계 및 모니터링
- [ ] 에러 처리 강화

### Phase 3: 최적화 (1주)

- [ ] 성능 최적화
- [ ] 메모리 관리 개선
- [ ] 배치 처리 최적화

### Phase 4: 테스트 및 문서화 (1주)

- [ ] 단위 테스트 작성
- [ ] 통합 테스트 작성
- [ ] 사용자 가이드 작성

---

**작성일**: 2024-01-01  
**최종 수정일**: 2024-01-01  
**작성자**: LawFirmAI Development Team

