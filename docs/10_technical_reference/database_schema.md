# LawFirmAI 데이터베이스 스키마

## 개요

LawFirmAI 프로젝트는 SQLite 데이터베이스를 사용하여 법률 데이터, 벡터 임베딩, 처리 이력 등을 관리합니다. 이 문서는 데이터베이스의 구조와 각 테이블의 상세 정보를 제공합니다.

## 📊 데이터베이스 구조

### 주요 테이블
- `assembly_laws`: 법률 데이터 저장
- `assembly_articles`: 법률 조문 데이터 저장
- `precedent_cases`: 판례 사건 데이터 저장
- `precedent_sections`: 판례 섹션 데이터 저장 (판시사항, 판결요지 등)
- `precedent_parties`: 판례 당사자 데이터 저장
- `processed_files`: 파일 처리 이력 추적
- `fts_laws`: 법률 전체 텍스트 검색 인덱스
- `fts_articles`: 조문 전체 텍스트 검색 인덱스
- `fts_precedent_cases`: 판례 사건 전체 텍스트 검색 인덱스
- `fts_precedent_sections`: 판례 섹션 전체 텍스트 검색 인덱스

## 🗃️ 테이블 상세 정보

### assembly_laws 테이블
법률의 기본 정보와 메타데이터를 저장합니다.

```sql
CREATE TABLE assembly_laws (
    law_id TEXT PRIMARY KEY,                    -- 법률 고유 ID
    source TEXT NOT NULL,                       -- 데이터 소스 (assembly)
    law_name TEXT NOT NULL,                     -- 법률명
    law_type TEXT,                              -- 법률 유형
    category TEXT,                              -- 카테고리
    row_number TEXT,                            -- 행 번호
    promulgation_number TEXT,                   -- 공포번호
    promulgation_date TEXT,                     -- 공포일
    enforcement_date TEXT,                      -- 시행일
    amendment_type TEXT,                        -- 개정 유형
    ministry TEXT,                              -- 소관부처
    parent_law TEXT,                            -- 상위법
    related_laws TEXT,                          -- 관련법 (JSON)
    full_text TEXT,                             -- 전체 텍스트
    searchable_text TEXT,                       -- 검색용 텍스트
    keywords TEXT,                              -- 키워드 (JSON)
    summary TEXT,                               -- 요약
    html_clean_text TEXT,                       -- HTML 정리된 텍스트
    main_article_count INTEGER DEFAULT 0,      -- 본칙 조문 수
    supplementary_article_count INTEGER DEFAULT 0, -- 부칙 조문 수
    ml_enhanced BOOLEAN DEFAULT FALSE,          -- ML 강화 여부
    parsing_quality_score REAL DEFAULT 0.0,     -- 파싱 품질 점수
    processing_version TEXT DEFAULT '1.0',      -- 처리 버전
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);
```

### assembly_articles 테이블
법률의 개별 조문 정보를 저장합니다.

```sql
CREATE TABLE assembly_articles (
    article_id TEXT PRIMARY KEY,                -- 조문 고유 ID
    law_id TEXT NOT NULL,                      -- 법률 ID (외래키)
    article_number INTEGER NOT NULL,           -- 조문 번호
    article_title TEXT,                        -- 조문 제목
    article_content TEXT NOT NULL,             -- 조문 내용
    is_supplementary BOOLEAN DEFAULT FALSE,     -- 부칙 여부
    ml_confidence_score REAL DEFAULT 0.0,      -- ML 신뢰도 점수
    parsing_method TEXT DEFAULT 'rule_based',   -- 파싱 방법
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (law_id) REFERENCES assembly_laws(law_id)
);
```

### precedent_cases 테이블 (신규)
판례 사건의 기본 정보를 저장합니다.

```sql
CREATE TABLE precedent_cases (
    case_id TEXT PRIMARY KEY,                    -- 판례 고유 ID
    category TEXT NOT NULL,                      -- 카테고리 (civil, criminal, family)
    case_name TEXT NOT NULL,                     -- 사건명
    case_number TEXT NOT NULL,                   -- 사건번호
    decision_date TEXT,                          -- 판결일
    field TEXT,                                  -- 분야 (민사, 형사, 가사)
    court TEXT,                                  -- 법원
    detail_url TEXT,                             -- 상세 URL
    full_text TEXT,                              -- 전체 텍스트
    searchable_text TEXT,                        -- 검색용 텍스트
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);
```

### precedent_sections 테이블 (신규)
판례의 각 섹션 정보를 저장합니다 (판시사항, 판결요지 등).

```sql
CREATE TABLE precedent_sections (
    section_id TEXT PRIMARY KEY,                 -- 섹션 고유 ID
    case_id TEXT NOT NULL,                       -- 판례 ID (외래키)
    section_type TEXT NOT NULL,                  -- 섹션 유형 (판시사항, 판결요지 등)
    section_type_korean TEXT,                    -- 섹션 유형 한글명
    section_content TEXT NOT NULL,               -- 섹션 내용
    section_length INTEGER DEFAULT 0,            -- 섹션 길이
    has_content BOOLEAN DEFAULT FALSE,           -- 내용 존재 여부
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (case_id) REFERENCES precedent_cases(case_id)
);
```

### processed_files 테이블 (신규)
파일 처리 이력을 추적하여 증분 처리를 지원합니다.

```sql
CREATE TABLE processed_files (
    id INTEGER PRIMARY KEY AUTOINCREMENT,      -- 자동 증가 ID
    file_path TEXT UNIQUE NOT NULL,            -- 파일 경로
    file_hash TEXT NOT NULL,                   -- 파일 해시 (SHA256)
    data_type TEXT NOT NULL,                   -- 데이터 유형
    processed_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP, -- 처리 완료 시간
    processing_status TEXT DEFAULT 'completed', -- 처리 상태
    record_count INTEGER DEFAULT 0,            -- 처리된 레코드 수
    processing_version TEXT DEFAULT '1.0',     -- 처리 버전
    error_message TEXT,                        -- 오류 메시지
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);
```

### 인덱스
성능 최적화를 위한 인덱스들:

```sql
-- assembly_laws 테이블 인덱스
CREATE INDEX idx_assembly_laws_source ON assembly_laws(source);
CREATE INDEX idx_assembly_laws_category ON assembly_laws(category);
CREATE INDEX idx_assembly_laws_ministry ON assembly_laws(ministry);
CREATE INDEX idx_assembly_laws_created_at ON assembly_laws(created_at);

-- assembly_articles 테이블 인덱스
CREATE INDEX idx_assembly_articles_law_id ON assembly_articles(law_id);
CREATE INDEX idx_assembly_articles_number ON assembly_articles(article_number);
CREATE INDEX idx_assembly_articles_supplementary ON assembly_articles(is_supplementary);

-- processed_files 테이블 인덱스
CREATE INDEX idx_processed_files_path ON processed_files(file_path);
CREATE INDEX idx_processed_files_type ON processed_files(data_type);
CREATE INDEX idx_processed_files_status ON processed_files(processing_status);

-- precedent_cases 테이블 인덱스
CREATE INDEX idx_precedent_cases_category ON precedent_cases(category);
CREATE INDEX idx_precedent_cases_date ON precedent_cases(decision_date);
CREATE INDEX idx_precedent_cases_court ON precedent_cases(court);

-- precedent_sections 테이블 인덱스
CREATE INDEX idx_precedent_sections_case_id ON precedent_sections(case_id);
CREATE INDEX idx_precedent_sections_type ON precedent_sections(section_type);

-- precedent_parties 테이블 인덱스
CREATE INDEX idx_precedent_parties_case_id ON precedent_parties(case_id);
CREATE INDEX idx_precedent_parties_type ON precedent_parties(party_type);
```

### 전체 텍스트 검색 (FTS5) 테이블
SQLite의 FTS5 확장을 사용한 전체 텍스트 검색:

```sql
-- 법률 전체 텍스트 검색
CREATE VIRTUAL TABLE fts_laws USING fts5(
    law_id,
    law_name,
    full_text,
    searchable_text,
    keywords,
    content='assembly_laws',
    content_rowid='rowid'
);

-- 조문 전체 텍스트 검색
CREATE VIRTUAL TABLE fts_articles USING fts5(
    article_id,
    law_id,
    article_title,
    article_content,
    content='assembly_articles',
    content_rowid='rowid'
);

-- 판례 사건 전체 텍스트 검색
CREATE VIRTUAL TABLE fts_precedent_cases USING fts5(
    case_id,
    case_name,
    case_number,
    full_text,
    searchable_text,
    content='precedent_cases',
    content_rowid='rowid'
);

-- 판례 섹션 전체 텍스트 검색
CREATE VIRTUAL TABLE fts_precedent_sections USING fts5(
    section_id,
    case_id,
    section_content,
    content='precedent_sections',
    content_rowid='rowid'
);
```

## 🔄 데이터베이스 관리

### 테이블 생성
```python
from source.data.database import DatabaseManager

# 데이터베이스 관리자 초기화 (테이블 자동 생성)
db_manager = DatabaseManager()
```

### 처리 상태 추적
```python
# 파일 처리 완료 표시
db_manager.mark_file_as_processed(
    file_path="data/raw/assembly/law_only/20251016/file.json",
    file_hash="sha256_hash",
    data_type="law_only",
    record_count=5,
    processing_version="1.0"
)

# 파일 처리 상태 확인
is_processed = db_manager.is_file_processed("data/raw/assembly/law_only/20251016/file.json")

# 처리 통계 조회
stats = db_manager.get_processing_statistics()
```

### 증분 처리 지원
```python
# 특정 데이터 유형의 처리된 파일 조회
processed_files = db_manager.get_processed_files_by_type("law_only", status="completed")

# 처리 상태 업데이트
db_manager.update_file_processing_status(
    file_path="data/raw/assembly/law_only/20251016/file.json",
    status="embedded"
)
```

## 📈 성능 최적화

### 쿼리 최적화
- **인덱스 활용**: 자주 사용되는 컬럼에 인덱스 생성
- **외래키 제약**: 데이터 무결성 보장
- **배치 처리**: 대량 데이터 처리 시 트랜잭션 활용

### 메모리 관리
- **연결 풀링**: 데이터베이스 연결 재사용
- **컨텍스트 매니저**: 자동 연결 해제
- **배치 크기 조정**: 메모리 사용량에 따른 배치 크기 조정

## 🔍 모니터링 및 디버깅

### 데이터베이스 상태 확인
```python
# 테이블별 레코드 수 확인
laws_count = db_manager.execute_query("SELECT COUNT(*) FROM assembly_laws")
articles_count = db_manager.execute_query("SELECT COUNT(*) FROM assembly_articles")
processed_count = db_manager.execute_query("SELECT COUNT(*) FROM processed_files")

print(f"법률 수: {laws_count[0]['COUNT(*)']}")
print(f"조문 수: {articles_count[0]['COUNT(*)']}")
print(f"처리된 파일 수: {processed_count[0]['COUNT(*)']}")
```

### 처리 통계 조회
```python
# 처리 상태별 통계
stats = db_manager.get_processing_statistics()
print(f"전체 파일: {stats['total_files']}")
print(f"완료: {stats['completed']}")
print(f"실패: {stats['failed']}")
print(f"임베딩 완료: {stats['embedded']}")
```

### 오류 파일 조회
```python
# 실패한 파일 목록
failed_files = db_manager.get_processed_files_by_type("law_only", status="failed")
for file_info in failed_files:
    print(f"실패 파일: {file_info['file_path']}")
    print(f"오류: {file_info['error_message']}")
```

## 🚨 백업 및 복구

### 데이터베이스 백업
```bash
# SQLite 데이터베이스 백업
sqlite3 data/lawfirm.db ".backup data/lawfirm_backup.db"
```

### 데이터 복구
```bash
# 백업에서 복구
cp data/lawfirm_backup.db data/lawfirm.db
```

## 🔮 향후 개선 계획

### 단기 계획
- [ ] 파티셔닝: 날짜별 테이블 분할
- [ ] 압축: 대용량 텍스트 필드 압축
- [ ] 캐싱: 자주 조회되는 데이터 캐싱

### 중기 계획
- [ ] 샤딩: 데이터베이스 수평 분할
- [ ] 복제: 읽기 전용 복제본 구축
- [ ] 모니터링: 실시간 성능 모니터링

### 장기 계획
- [ ] NoSQL 연동: MongoDB/Elasticsearch 연동
- [ ] 클라우드 DB: PostgreSQL/MySQL 마이그레이션
- [ ] 분산 처리: 여러 노드 분산 처리

---

**LawFirmAI 개발팀**  
*최종 업데이트: 2025-10-16*
