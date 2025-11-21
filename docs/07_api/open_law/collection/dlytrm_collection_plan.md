# dlytrm (일상용어) API 데이터 수집 계획서

## 📋 목차

1. [API 개요](#api-개요)
2. [데이터베이스 스키마 설계](#데이터베이스-스키마-설계)
3. [파일 구조](#파일-구조)
4. [구현 계획](#구현-계획)
5. [수집 전략](#수집-전략)
6. [주요 특징](#주요-특징)
7. [실행 예시](#실행-예시)
8. [데이터 확인](#데이터-확인)
9. [주의사항](#주의사항)

---

## API 개요

### API 정보

**요청 URL**: `https://www.law.go.kr/DRF/lawSearch.do?target=dlytrm`  
**Method**: GET  
**출력 형식**: JSON  
**용도**: 일상용어 검색 (일상용어-법령용어 연계 정보)

### 요청 파라미터

| 파라미터 | 타입 | 필수 | 설명 | 기본값/허용값 |
|---------|------|------|------|--------------|
| OC | string | 필수 | 사용자 이메일 ID (g4c@korea.kr일 경우 OC=g4c) | - |
| target | string | 필수 | 서비스 대상 | dlytrm |
| type | char | 필수 | 출력 형태 | JSON |
| query | string | 선택 | 일상용어명에서 검색을 원하는 질의 | - |
| display | int | 선택 | 검색된 결과 개수 | 20 (max=100) |
| page | int | 선택 | 검색 결과 페이지 | 1 |

### 응답 필드

| 필드 | 타입 | 설명 |
|------|------|------|
| target | string | 검색서비스 대상 |
| 키워드 | string | 검색 단어 |
| 검색결과개수 | int | 검색 건수 |
| section | string | 검색범위 |
| page | int | 현재 페이지번호 |
| numOfRows | int | 페이지 당 출력 결과 수 |
| 일상용어 id | string | 일상용어 순번 |
| 일상용어명 | string | 일상용어명 |
| 출처 | string | 일상용어 출처 |
| 용어간관계링크 | string | 일상용어-법령용어 연계 정보 상세링크 |

### 샘플 URL

```bash
# JSON 형식으로 검색
https://www.law.go.kr/DRF/lawSearch.do?OC=test&target=dlytrm&type=JSON&query=민원

# 페이징 처리
https://www.law.go.kr/DRF/lawSearch.do?OC=test&target=dlytrm&type=JSON&query=민원&page=1&display=100
```

---

## 데이터베이스 스키마 설계

원본 JSON을 그대로 저장할 테이블 설계:

```sql
CREATE TABLE IF NOT EXISTS open_law_dlytrm_data (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    
    -- 검색 메타데이터
    search_keyword TEXT,
    search_page INTEGER,
    search_display INTEGER,
    
    -- API 응답 원본 데이터 (JSON)
    raw_response_json TEXT NOT NULL,  -- 전체 응답 JSON 원본 저장
    
    -- 개별 결과 항목 (배열의 각 항목)
    term_id TEXT,                    -- 일상용어 id
    term_name TEXT,                  -- 일상용어명
    source TEXT,                     -- 출처
    term_relation_link TEXT,         -- 용어간관계링크 (일상용어-법령용어 연계)
    
    -- 수집 메타데이터
    collected_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    collection_method TEXT,          -- 'keyword', 'pagination', 'all'
    api_request_url TEXT,            -- 실제 요청 URL
    
    -- 통계 정보
    total_count INTEGER,              -- 검색결과개수
    page_number INTEGER,              -- page
    num_of_rows INTEGER,             -- numOfRows
    
    -- 인덱스
    UNIQUE(term_id, search_keyword, search_page)
);

CREATE INDEX IF NOT EXISTS idx_dlytrm_term_id ON open_law_dlytrm_data(term_id);
CREATE INDEX IF NOT EXISTS idx_dlytrm_keyword ON open_law_dlytrm_data(search_keyword);
CREATE INDEX IF NOT EXISTS idx_dlytrm_collected_at ON open_law_dlytrm_data(collected_at);
```

### 테이블 설계 특징

1. **원본 JSON 보존**: `raw_response_json` 필드에 전체 API 응답을 JSON 문자열로 저장
2. **개별 항목 저장**: 각 검색 결과 항목을 개별 레코드로 저장하여 검색 및 분석 용이
3. **중복 방지**: `term_id + search_keyword + search_page` 조합으로 UNIQUE 제약
4. **메타데이터 보존**: 수집 시점, 방법, 요청 URL 등 추적 가능

---

## 파일 구조

```
scripts/ingest/
├── ingest_dlytrm.py          # 메인 수집 스크립트
├── dlytrm_client.py           # API 클라이언트
└── dlytrm_collector.py        # 데이터 수집기

lawfirm_langgraph/core/data/
└── connection_pool.py          # 연결 풀 (기존)
```

---

## 구현 계획

### 4.1 API 클라이언트 클래스

`scripts/ingest/dlytrm_client.py`에 구현

### 4.2 데이터 수집기

`scripts/ingest/dlytrm_collector.py`에 구현

### 4.3 메인 수집 스크립트

`scripts/ingest/ingest_dlytrm.py`에 구현

---

## 수집 전략

### 5.1 키워드 기반 수집 (권장)

일상생활에서 자주 사용되는 법률 관련 용어를 키워드로 사용:

```python
# 일반 법률 용어
keywords = ["민원", "신고", "신청", "처분", "행정", "법률", "규정", "조례"]

# 일상 생활 관련
keywords = ["계약", "임대", "매매", "소유", "상속", "이혼", "양육", "부양"]

# 행정 관련
keywords = ["허가", "인가", "등록", "신고", "납세", "세금", "부담금"]
```

### 5.2 전체 수집

- `query` 없이 전체 페이지 순회
- 페이징으로 전체 데이터 수집
- 시간 소요가 크므로 주의 필요

---

## 주요 특징

### 1. 원본 JSON 보존
- `raw_response_json` 필드에 전체 API 응답을 JSON 문자열로 저장
- 나중에 원본 데이터 분석 및 재처리 가능

### 2. 연결 풀 사용
- `get_connection_pool()` 사용하여 스레드 안전성 보장
- 연결 재사용으로 성능 향상

### 3. 중복 방지
- `term_id + search_keyword + search_page` 조합으로 UNIQUE 제약
- 동일한 데이터 중복 저장 방지

### 4. Rate Limiting
- 요청 간 지연 시간 설정 (기본 0.5초)
- API 서버 부하 방지

### 5. 재시도 로직
- 네트워크 오류 시 자동 재시도 (최대 3회)
- 지수 백오프(exponential backoff) 적용

### 6. 로깅
- 수집 과정 상세 로깅
- 에러 발생 시 로그 기록

---

## 실행 예시

### 키워드 기반 수집

```bash
# 기본 키워드로 수집
python scripts/ingest/ingest_dlytrm.py \
    --oc schema9 \
    --keywords "민원,신고,신청" \
    --max-pages 10 \
    --display 100

# 키워드 파일로 수집
python scripts/ingest/ingest_dlytrm.py \
    --oc schema9 \
    --keyword-file data/keywords/daily_keywords.txt \
    --max-pages 5

# 요청 간 지연 시간 조정
python scripts/ingest/ingest_dlytrm.py \
    --oc schema9 \
    --keywords "민원" \
    --rate-limit 1.0
```

### 전체 수집

```bash
# query 없이 전체 수집
python scripts/ingest/ingest_dlytrm.py \
    --oc schema9 \
    --query "" \
    --max-pages 100

# 특정 질의로 수집
python scripts/ingest/ingest_dlytrm.py \
    --oc schema9 \
    --query "민원" \
    --max-pages 50

# 특정 페이지부터 수집
python scripts/ingest/ingest_dlytrm.py \
    --oc schema9 \
    --query "" \
    --start-page 101 \
    --max-pages 500
```

### 환경 변수 사용

```bash
# .env 파일 또는 환경 변수
export LAW_OPEN_API_OC="your_email_id"

python scripts/ingest/ingest_dlytrm.py \
    --oc $LAW_OPEN_API_OC \
    --keywords "민원"
```

---

## 데이터 확인

### 원본 JSON 확인

```sql
-- 원본 JSON 확인
SELECT raw_response_json FROM open_law_dlytrm_data LIMIT 1;

-- JSON 파싱하여 확인
SELECT 
    json_extract(raw_response_json, '$.검색결과개수') as total_count,
    json_extract(raw_response_json, '$.page') as page
FROM open_law_dlytrm_data 
LIMIT 1;
```

### 통계 확인

```sql
-- 키워드별 통계
SELECT 
    search_keyword,
    COUNT(*) as count,
    COUNT(DISTINCT term_id) as unique_terms,
    MIN(collected_at) as first_collected,
    MAX(collected_at) as last_collected
FROM open_law_dlytrm_data
GROUP BY search_keyword
ORDER BY count DESC;

-- 수집 일자별 통계
SELECT 
    DATE(collected_at) as collection_date,
    COUNT(*) as count,
    COUNT(DISTINCT term_id) as unique_terms
FROM open_law_dlytrm_data
GROUP BY DATE(collected_at)
ORDER BY collection_date DESC;
```

### 특정 용어 검색

```sql
-- 용어명으로 검색
SELECT * FROM open_law_dlytrm_data 
WHERE term_name LIKE '%민원%'
ORDER BY collected_at DESC;

-- 용어 ID로 검색
SELECT * FROM open_law_dlytrm_data 
WHERE term_id = '12345';
```

### 원본 데이터 추출

```python
import sqlite3
import json

conn = sqlite3.connect('data/lawfirm_v2.db')
cursor = conn.cursor()

# 원본 JSON 추출
cursor.execute("SELECT raw_response_json FROM open_law_dlytrm_data LIMIT 1")
row = cursor.fetchone()
if row:
    original_data = json.loads(row[0])
    print(json.dumps(original_data, ensure_ascii=False, indent=2))
```

---

## 주의사항

### 1. API 제한

- **요청 간 지연**: API 서버 부하 방지를 위해 요청 간 최소 0.5초 지연 권장
- **일일 요청 한도**: 국가법령정보센터의 일일 요청 한도 확인 필요
- **동시 요청 제한**: 동시 요청 수 제한 (권장: 1개씩 순차 처리)

### 2. 데이터 품질

- **중복 제거**: `term_id + search_keyword + search_page` 조합으로 자동 중복 방지
- **데이터 검증**: 필수 필드 존재 여부 확인
- **원본 보존**: `raw_response_json`에 원본 데이터 저장

### 3. 메모리 관리

- **배치 처리**: 대량 수집 시 배치 단위로 처리
- **주기적 저장**: 일정 간격으로 중간 저장
- **메모리 모니터링**: 메모리 사용량 추적

### 4. 에러 처리

- **재시도 로직**: 네트워크 오류 시 자동 재시도 (최대 3회)
- **에러 로깅**: 모든 에러를 로그 파일에 기록
- **부분 실패 처리**: 일부 용어 수집 실패 시에도 계속 진행

### 5. 데이터베이스

- **연결 풀 사용**: 반드시 `get_connection_pool()` 사용 (CRITICAL)
- **트랜잭션 관리**: 에러 발생 시 롤백 처리
- **인덱스 활용**: 검색 성능 향상을 위한 인덱스 활용

### 6. 응답 구조 확인

- **응답 구조 변동 가능성**: 실제 API 응답 구조가 가이드와 다를 수 있음
- **디버깅 로깅**: 응답 구조 확인을 위한 상세 로깅 포함
- **유연한 파싱**: `items`, `일상용어` 등 다양한 키 이름 지원

---

## 구현 단계

### Phase 1: 기본 구현
- [x] 계획서 작성
- [ ] API 클라이언트 구현 (`DlytrmClient`)
- [ ] 데이터 수집기 구현 (`DlytrmCollector`)
- [ ] 데이터베이스 테이블 생성
- [ ] 메인 수집 스크립트 구현

### Phase 2: 고급 기능
- [ ] 재시도 로직 강화
- [ ] 로깅 시스템 개선
- [ ] 통계 및 모니터링
- [ ] 체크포인트 기능 (선택)

### Phase 3: 테스트
- [ ] 단위 테스트 작성
- [ ] 통합 테스트 작성
- [ ] 실제 API 테스트

### Phase 4: 문서화
- [ ] 사용자 가이드 작성
- [ ] API 문서 업데이트

---

## 참고 자료

### 관련 문서

- [dlytrmGuide 가이드](../guides/dlytrmGuide.md) - dlytrm API 가이드
- [lstrmAI 수집 계획](lstrm_ai_collection_plan.md) - 법령용어 수집 계획 (참고)
- [법령용어 수집 계획](legal_term_collection_plan.md) - 법령용어 수집 계획
- [Open Law API 가이드 맵](../guide_id_map.md) - 전체 API 가이드 맵

### 외부 링크

- [국가법령정보센터 Open API](https://open.law.go.kr/LSO/openApi/guideList.do) - 공식 API 가이드
- [LAW OPEN DATA](http://www.law.go.kr/DRF/lawService.do) - API 엔드포인트

---

**작성일**: 2024-01-01  
**최종 수정일**: 2024-01-01  
**작성자**: LawFirmAI Development Team

