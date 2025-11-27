# Open Law API 빠른 수집 계획 (민사법 → 형법)

## 개요

PostgreSQL 마이그레이션을 위한 현행법령 및 판례 데이터 수집 계획입니다.
민사법을 우선으로 수집하고, 이어서 형법을 수집합니다.

## 수집 목표

### 1단계: 현행법령 수집 (민사법 → 형법)

**우선순위 API**:
- `lsEfYdListGuide`: 현행법령(시행일) 목록 조회
- `lsEfYdJoListGuide`: 현행법령(시행일) 본문 조항호목 조회
- `lsEfYdInfoGuide`: 현행법령(시행일) 본문 조회 (메타데이터)

**수집 범위**:
- 민사법 관련 법령 (민법, 민사소송법, 가족법 등)
- 형법 관련 법령 (형법, 형사소송법 등)

### 2단계: 판례 수집 (민사법 → 형법)

**우선순위 API**:
- `precListGuide`: 판례 목록 조회
- `precInfoGuide`: 판례 본문 조회

**수집 범위**:
- 민사법 관련 판례
- 형법 관련 판례

## 수집 전략

### 법령 분류 기준

#### 민사법 (Civil Law)
- **법령명 키워드**: 민법, 민사소송법, 가족법, 가사소송법, 부동산등기법, 상속법
- **법령종류 코드 (knd)**: 
  - 법률: `L` (일반 법률)
  - 민법 관련: 검색 쿼리로 필터링
- **검색 쿼리**: `query=민법 OR query=민사소송법 OR query=가족법`

#### 형법 (Criminal Law)
- **법령명 키워드**: 형법, 형사소송법, 특별형법
- **법령종류 코드 (knd)**: 
  - 법률: `L` (일반 법률)
  - 형법 관련: 검색 쿼리로 필터링
- **검색 쿼리**: `query=형법 OR query=형사소송법`

### 판례 분류 기준

#### 민사법 판례
- **사건종류**: 민사, 가사
- **참조법령 (JO)**: 민법, 민사소송법, 가족법
- **검색 쿼리**: `query=민사 OR query=가사`

#### 형법 판례
- **사건종류**: 형사
- **참조법령 (JO)**: 형법, 형사소송법
- **검색 쿼리**: `query=형사`

## 수집 프로세스

### Phase 1: 현행법령 목록 수집

**API**: `lsEfYdListGuide`
- **Base URL**: `http://www.law.go.kr/DRF/lawSearch.do?target=eflaw`
- **Method**: GET
- **Parameters**:
  - `OC`: 사용자 이메일 ID (필수)
  - `target`: `eflaw` (필수)
  - `type`: `JSON` (권장)
  - `query`: 법령명 검색어
  - `nw`: `3` (현행법령만)
  - `display`: `100` (최대)
  - `page`: 페이지 번호

**수집 순서**:
1. 민사법 법령 목록 수집
   - `query=민법` → 민법 관련 법령 목록
   - `query=민사소송법` → 민사소송법 관련 법령 목록
   - `query=가족법` → 가족법 관련 법령 목록
2. 형법 법령 목록 수집
   - `query=형법` → 형법 관련 법령 목록
   - `query=형사소송법` → 형사소송법 관련 법령 목록

**예상 수집량**:
- 민사법: 약 50-100개 법령
- 형법: 약 20-50개 법령

### Phase 2: 법령 본문 및 조문 수집

**API**: `lsEfYdInfoGuide` + `lsEfYdJoListGuide`

**수집 프로세스**:
1. Phase 1에서 수집한 법령 ID 목록을 순회
2. 각 법령에 대해:
   - `lsEfYdInfoGuide`: 법령 메타데이터 및 전체 본문 조회
   - `lsEfYdJoListGuide`: 각 조문별 상세 조회 (조, 항, 호, 목)

**API 엔드포인트**:
- **법령 본문**: `http://www.law.go.kr/DRF/lawService.do?target=eflaw`
  - Parameters: `ID` (법령ID) 또는 `MST` (법령 마스터번호) + `efYd` (시행일자)
- **조문 상세**: `http://www.law.go.kr/DRF/lawService.do?target=eflawjosub`
  - Parameters: `ID` 또는 `MST` + `efYd` + `JO` (조번호, 6자리)

**예상 수집량**:
- 민사법: 약 5,000-10,000개 조문
- 형법: 약 2,000-5,000개 조문

### Phase 3: 판례 목록 수집

**API**: `precListGuide`
- **Base URL**: `http://www.law.go.kr/DRF/lawSearch.do?target=prec`
- **Method**: GET
- **Parameters**:
  - `OC`: 사용자 이메일 ID (필수)
  - `target`: `prec` (필수)
  - `type`: `JSON` (권장)
  - `query`: 검색어
  - `JO`: 참조법령명 (민법, 형법 등)
  - `display`: `100` (최대)
  - `page`: 페이지 번호

**수집 순서**:
1. 민사법 판례 목록 수집
   - `query=민사` + `JO=민법`
   - `query=가사` + `JO=가족법`
2. 형법 판례 목록 수집
   - `query=형사` + `JO=형법`

**예상 수집량**:
- 민사법 판례: 약 10,000-20,000건
- 형법 판례: 약 5,000-10,000건

### Phase 4: 판례 본문 수집

**API**: `precInfoGuide`
- **Base URL**: `http://www.law.go.kr/DRF/lawService.do?target=prec`
- **Method**: GET
- **Parameters**:
  - `OC`: 사용자 이메일 ID (필수)
  - `target`: `prec` (필수)
  - `type`: `JSON` (권장)
  - `ID`: 판례 일련번호 (필수)

**수집 프로세스**:
1. Phase 3에서 수집한 판례 일련번호 목록을 순회
2. 각 판례에 대해 `precInfoGuide`로 본문 조회

**예상 수집량**:
- 민사법 판례 본문: 약 10,000-20,000건
- 형법 판례 본문: 약 5,000-10,000건

## PostgreSQL 스키마 설계

### 법령 데이터 테이블

```sql
-- 법령 메타데이터 테이블
CREATE TABLE statutes (
    id SERIAL PRIMARY KEY,
    law_id INTEGER NOT NULL UNIQUE,              -- 법령ID
    law_name_kr TEXT NOT NULL,                    -- 법령명(한글)
    law_name_hanja TEXT,                          -- 법령명(한자)
    law_name_en TEXT,                             -- 법령명(영어)
    law_abbrv TEXT,                               -- 법령약칭
    law_type TEXT,                                -- 법령종류
    law_type_code TEXT,                           -- 법종구분코드
    proclamation_date DATE,                       -- 공포일자
    proclamation_number INTEGER,                  -- 공포번호
    effective_date DATE,                          -- 시행일자
    ministry_code INTEGER,                        -- 소관부처코드
    ministry_name TEXT,                           -- 소관부처명
    amendment_type TEXT,                          -- 제개정구분
    domain TEXT,                                  -- 분야 (civil_law, criminal_law)
    raw_response_json JSONB,                      -- 원본 JSON 응답
    collected_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- 조문 테이블
CREATE TABLE statutes_articles (
    id SERIAL PRIMARY KEY,
    statute_id INTEGER NOT NULL REFERENCES statutes(id) ON DELETE CASCADE,
    article_no TEXT NOT NULL,                     -- 조문번호 (예: "000200" = 제2조)
    article_title TEXT,                           -- 조문제목
    article_content TEXT NOT NULL,                -- 조문내용
    clause_no TEXT,                               -- 항번호
    clause_content TEXT,                          -- 항내용
    item_no TEXT,                                 -- 호번호
    item_content TEXT,                            -- 호내용
    sub_item_no TEXT,                             -- 목번호
    sub_item_content TEXT,                        -- 목내용
    effective_date DATE,                          -- 조문시행일자
    raw_response_json JSONB,                      -- 원본 JSON 응답
    collected_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    UNIQUE(statute_id, article_no, clause_no, item_no, sub_item_no)
);

-- 판례 메타데이터 테이블
CREATE TABLE precedents (
    id SERIAL PRIMARY KEY,
    precedent_id INTEGER NOT NULL UNIQUE,         -- 판례정보일련번호
    case_name TEXT NOT NULL,                      -- 사건명
    case_number TEXT,                             -- 사건번호
    decision_date DATE,                           -- 선고일자
    court_name TEXT,                              -- 법원명
    court_type_code INTEGER,                      -- 법원종류코드
    case_type_name TEXT,                          -- 사건종류명
    case_type_code INTEGER,                       -- 사건종류코드
    decision_type TEXT,                           -- 판결유형
    decision_result TEXT,                          -- 선고
    domain TEXT,                                  -- 분야 (civil_law, criminal_law)
    raw_response_json JSONB,                      -- 원본 JSON 응답
    collected_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- 판례 본문 테이블
CREATE TABLE precedent_contents (
    id SERIAL PRIMARY KEY,
    precedent_id INTEGER NOT NULL REFERENCES precedents(id) ON DELETE CASCADE,
    section_type TEXT NOT NULL,                   -- 섹션 유형 (판시사항, 판결요지, 판례내용)
    section_content TEXT NOT NULL,                -- 섹션 내용
    referenced_articles TEXT,                     -- 참조조문
    referenced_precedents TEXT,                   -- 참조판례
    raw_response_json JSONB,                      -- 원본 JSON 응답
    collected_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- 인덱스
CREATE INDEX idx_statutes_domain ON statutes(domain);
CREATE INDEX idx_statutes_law_id ON statutes(law_id);
CREATE INDEX idx_statutes_law_name ON statutes(law_name_kr);
CREATE INDEX idx_articles_statute_id ON statutes_articles(statute_id);
CREATE INDEX idx_articles_article_no ON statutes_articles(article_no);
CREATE INDEX idx_precedents_domain ON precedents(domain);
CREATE INDEX idx_precedents_precedent_id ON precedents(precedent_id);
CREATE INDEX idx_precedents_case_name ON precedents(case_name);
CREATE INDEX idx_precedents_decision_date ON precedents(decision_date);
CREATE INDEX idx_precedent_contents_precedent_id ON precedent_contents(precedent_id);
CREATE INDEX idx_precedent_contents_section_type ON precedent_contents(section_type);

-- Full-Text Search 인덱스 (PostgreSQL)
CREATE INDEX idx_statutes_fts ON statutes USING gin(to_tsvector('korean', law_name_kr || ' ' || COALESCE(law_abbrv, '')));
CREATE INDEX idx_articles_fts ON statutes_articles USING gin(to_tsvector('korean', article_content));
CREATE INDEX idx_precedents_fts ON precedents USING gin(to_tsvector('korean', case_name));
CREATE INDEX idx_precedent_contents_fts ON precedent_contents USING gin(to_tsvector('korean', section_content));
```

## 수집 스크립트 구조

### 디렉토리 구조

```
scripts/ingest/open_law/
├── __init__.py
├── client.py                    # Open Law API 클라이언트
├── collectors/
│   ├── __init__.py
│   ├── statute_collector.py    # 법령 수집기
│   └── precedent_collector.py  # 판례 수집기
├── collectors/
│   ├── __init__.py
│   ├── civil_law_collector.py  # 민사법 수집기
│   └── criminal_law_collector.py # 형법 수집기
└── scripts/
    ├── collect_civil_statutes.py
    ├── collect_criminal_statutes.py
    ├── collect_civil_precedents.py
    └── collect_criminal_precedents.py
```

## 실행 계획

### 1주차: 민사법 현행법령 수집

**Day 1-2**: 법령 목록 수집
```bash
python scripts/ingest/open_law/scripts/collect_civil_statutes.py \
    --oc YOUR_OC \
    --phase list \
    --output data/raw/open_law/civil_statutes_list.json
```

**Day 3-5**: 법령 본문 및 조문 수집
```bash
python scripts/ingest/open_law/scripts/collect_civil_statutes.py \
    --oc YOUR_OC \
    --phase content \
    --input data/raw/open_law/civil_statutes_list.json \
    --db postgresql://user:pass@host:5432/lawfirmai
```

### 2주차: 형법 현행법령 수집

**Day 1-2**: 법령 목록 수집
```bash
python scripts/ingest/open_law/scripts/collect_criminal_statutes.py \
    --oc YOUR_OC \
    --phase list \
    --output data/raw/open_law/criminal_statutes_list.json
```

**Day 3-5**: 법령 본문 및 조문 수집
```bash
python scripts/ingest/open_law/scripts/collect_criminal_statutes.py \
    --oc YOUR_OC \
    --phase content \
    --input data/raw/open_law/criminal_statutes_list.json \
    --db postgresql://user:pass@host:5432/lawfirmai
```

### 3주차: 민사법 판례 수집

**Day 1-2**: 판례 목록 수집
```bash
python scripts/ingest/open_law/scripts/collect_civil_precedents.py \
    --oc YOUR_OC \
    --phase list \
    --max-pages 200 \
    --output data/raw/open_law/civil_precedents_list.json
```

**Day 3-5**: 판례 본문 수집
```bash
python scripts/ingest/open_law/scripts/collect_civil_precedents.py \
    --oc YOUR_OC \
    --phase content \
    --input data/raw/open_law/civil_precedents_list.json \
    --db postgresql://user:pass@host:5432/lawfirmai
```

### 4주차: 형법 판례 수집

**Day 1-2**: 판례 목록 수집
```bash
python scripts/ingest/open_law/scripts/collect_criminal_precedents.py \
    --oc YOUR_OC \
    --phase list \
    --max-pages 100 \
    --output data/raw/open_law/criminal_precedents_list.json
```

**Day 3-5**: 판례 본문 수집
```bash
python scripts/ingest/open_law/scripts/collect_criminal_precedents.py \
    --oc YOUR_OC \
    --phase content \
    --input data/raw/open_law/criminal_precedents_list.json \
    --db postgresql://user:pass@host:5432/lawfirmai
```

## 성능 최적화

### Rate Limiting
- API 요청 간 지연: 0.5-1.0초
- 배치 처리: 100개 단위로 묶어서 처리
- 재시도 로직: 실패 시 최대 3회 재시도

### 병렬 처리
- 법령별 조문 수집: 비동기 처리 (최대 5개 동시)
- 판례 본문 수집: 비동기 처리 (최대 10개 동시)

### 데이터베이스 최적화
- 배치 INSERT: 1000개 단위로 묶어서 INSERT
- 트랜잭션 관리: 배치 단위로 COMMIT
- 인덱스 생성: 수집 완료 후 인덱스 생성

## 모니터링 및 로깅

### 수집 진행 상황 추적
- 수집된 법령 수 / 전체 법령 수
- 수집된 조문 수 / 전체 조문 수
- 수집된 판례 수 / 전체 판례 수
- 수집 속도 (건/시간)
- 오류 발생 건수

### 로그 파일
- `logs/open_law/civil_statutes_collection.log`
- `logs/open_law/criminal_statutes_collection.log`
- `logs/open_law/civil_precedents_collection.log`
- `logs/open_law/criminal_precedents_collection.log`

## 예상 소요 시간

### 현행법령 수집
- 민사법: 약 5-7일
- 형법: 약 3-5일
- **총계**: 약 8-12일

### 판례 수집
- 민사법: 약 5-7일
- 형법: 약 3-5일
- **총계**: 약 8-12일

### 전체 수집 기간
- **총 예상 기간**: 약 16-24일 (약 3-4주)

## 다음 단계

1. **데이터 검증**: 수집된 데이터의 품질 검증
2. **임베딩 생성**: 조문 및 판례 본문에 대한 벡터 임베딩 생성
3. **검색 인덱스 구축**: PostgreSQL Full-Text Search 인덱스 최적화
4. **통합 테스트**: 검색 기능 테스트

## 참고 문서

- [Open Law API 가이드](./guide_id_map.md)
- [수집된 엔드포인트 문서](./collected_endpoints.md)
- [PostgreSQL 마이그레이션 계획](../../../06_deployment/POSTGRESQL_MIGRATION_PLAN.md)

