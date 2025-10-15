# LawFirmAI 데이터베이스 스키마 설계

## 📋 개요

**데이터베이스**: SQLite 3  
**버전**: 1.0.0  
**설계 일시**: 2025-09-24  
**목적**: 법률 AI 어시스턴트를 위한 효율적인 데이터 저장 및 검색

---

## 🏛️ 1. 판례 테이블 (precedents)

### 1.1 기본 정보
- **테이블명**: `precedents`
- **목적**: 법원 판례 데이터 저장
- **예상 데이터량**: 10,000 ~ 50,000건
- **주요 검색 필드**: case_number, court_name, summary, keywords

### 1.2 스키마 정의

```sql
CREATE TABLE precedents (
    -- 기본 식별자
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    case_number TEXT UNIQUE NOT NULL,
    
    -- 법원 정보
    court_name TEXT NOT NULL,
    court_type TEXT NOT NULL, -- '대법원', '고등법원', '지방법원', '특허법원' 등
    case_type TEXT NOT NULL, -- '민사', '형사', '행정', '특허' 등
    
    -- 판결 정보
    judgment_date DATE NOT NULL,
    judgment_type TEXT NOT NULL, -- '선고', '기각', '각하', '취하' 등
    judgment_result TEXT, -- '원고승소', '피고승소', '일부승소' 등
    
    -- 사건 내용
    summary TEXT NOT NULL,
    full_text TEXT NOT NULL,
    legal_issues TEXT, -- JSON 형태로 저장
    keywords TEXT, -- 쉼표로 구분된 키워드
    
    -- 벡터 검색 연동
    embedding_id INTEGER,
    embedding_vector BLOB, -- FAISS/ChromaDB 연동용
    
    -- 메타데이터
    source_url TEXT,
    file_path TEXT,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    
    -- 품질 관리
    quality_score REAL DEFAULT 0.0, -- 0.0 ~ 1.0
    review_status TEXT DEFAULT 'pending', -- 'pending', 'reviewed', 'approved'
    reviewer_id TEXT,
    review_comment TEXT
);
```

### 1.3 인덱스 설계

```sql
-- 기본 검색 인덱스
CREATE INDEX idx_precedents_court ON precedents(court_name);
CREATE INDEX idx_precedents_date ON precedents(judgment_date);
CREATE INDEX idx_precedents_type ON precedents(case_type);
CREATE INDEX idx_precedents_judgment_type ON precedents(judgment_type);

-- 복합 검색 인덱스
CREATE INDEX idx_precedents_search ON precedents(court_name, case_type, judgment_date);
CREATE INDEX idx_precedents_text ON precedents(summary, keywords);

-- 품질 관리 인덱스
CREATE INDEX idx_precedents_quality ON precedents(quality_score, review_status);
CREATE INDEX idx_precedents_embedding ON precedents(embedding_id);

-- 풀텍스트 검색 인덱스 (SQLite FTS5)
CREATE VIRTUAL TABLE precedents_fts USING fts5(
    case_number, 
    court_name, 
    summary, 
    keywords, 
    content='precedents', 
    content_rowid='id'
);
```

---

## 📄 2. 통합 문서 테이블 (documents) - 하이브리드 검색용

### 2.1 기본 정보
- **테이블명**: `documents`
- **목적**: 모든 법률 문서를 통합하여 하이브리드 검색 지원
- **현재 데이터량**: 24개 문서 (laws 13개, precedents 11개)
- **주요 검색 필드**: document_type, title, content

### 2.2 스키마 정의

```sql
CREATE TABLE documents (
    -- 기본 식별자
    id TEXT PRIMARY KEY, -- 'law_1', 'precedent_1' 등
    document_type TEXT NOT NULL, -- 'law', 'precedent', 'constitutional_decision' 등
    title TEXT NOT NULL,
    content TEXT NOT NULL,
    source_url TEXT,
    
    -- 메타데이터
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);
```

### 2.3 인덱스 설계

```sql
-- 기본 검색 인덱스
CREATE INDEX idx_documents_type ON documents(document_type);
CREATE INDEX idx_documents_title ON documents(title);

-- 풀텍스트 검색 인덱스
CREATE VIRTUAL TABLE documents_fts USING fts5(
    title, 
    content, 
    content='documents', 
    content_rowid='id'
);
```

### 2.4 메타데이터 테이블들

#### 법령 메타데이터 (law_metadata)
```sql
CREATE TABLE law_metadata (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    document_id TEXT NOT NULL,
    law_name TEXT,
    article_number INTEGER,
    promulgation_date TEXT,
    enforcement_date TEXT,
    department TEXT,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (document_id) REFERENCES documents (id)
);
```

#### 판례 메타데이터 (precedent_metadata)
```sql
CREATE TABLE precedent_metadata (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    document_id TEXT NOT NULL,
    case_number TEXT,
    court_name TEXT,
    decision_date TEXT,
    case_type TEXT,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (document_id) REFERENCES documents (id)
);
```

---

## ⚖️ 3. 법령 테이블 (laws) - 레거시

### 2.1 기본 정보
- **테이블명**: `laws`
- **목적**: 법령 조문 데이터 저장
- **예상 데이터량**: 1,000 ~ 5,000건
- **주요 검색 필드**: law_name, law_code, article_number, content

### 2.2 스키마 정의

```sql
CREATE TABLE laws (
    -- 기본 식별자
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    law_name TEXT NOT NULL,
    law_code TEXT UNIQUE NOT NULL, -- '법률 제12345호' 형태
    article_number TEXT NOT NULL, -- '제1조', '제2조의2' 등
    
    -- 조문 정보
    article_title TEXT,
    content TEXT NOT NULL,
    paragraph_number INTEGER, -- 1, 2, 3... (항 번호)
    subparagraph_number INTEGER, -- 1, 2, 3... (호 번호)
    
    -- 분류 정보
    category TEXT NOT NULL, -- '민법', '형법', '상법' 등
    subcategory TEXT, -- '총칙', '각칙' 등
    chapter TEXT, -- '제1장', '제2장' 등
    section TEXT, -- '제1절', '제2절' 등
    
    -- 시행 정보
    effective_date DATE NOT NULL,
    amendment_date DATE,
    status TEXT DEFAULT 'active', -- 'active', 'amended', 'repealed'
    
    -- 벡터 검색 연동
    embedding_id INTEGER,
    embedding_vector BLOB,
    
    -- 메타데이터
    source_url TEXT,
    file_path TEXT,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    
    -- 품질 관리
    quality_score REAL DEFAULT 0.0,
    review_status TEXT DEFAULT 'pending',
    reviewer_id TEXT,
    review_comment TEXT
);
```

### 2.3 인덱스 설계

```sql
-- 기본 검색 인덱스
CREATE INDEX idx_laws_name ON laws(law_name);
CREATE INDEX idx_laws_code ON laws(law_code);
CREATE INDEX idx_laws_category ON laws(category);
CREATE INDEX idx_laws_article ON laws(article_number);
CREATE INDEX idx_laws_status ON laws(status);

-- 복합 검색 인덱스
CREATE INDEX idx_laws_search ON laws(law_name, category, effective_date);
CREATE INDEX idx_laws_content ON laws(content);

-- 계층 구조 검색 인덱스
CREATE INDEX idx_laws_hierarchy ON laws(category, chapter, section);
CREATE INDEX idx_laws_paragraph ON laws(article_number, paragraph_number, subparagraph_number);

-- 품질 관리 인덱스
CREATE INDEX idx_laws_quality ON laws(quality_score, review_status);
CREATE INDEX idx_laws_embedding ON laws(embedding_id);

-- 풀텍스트 검색 인덱스
CREATE VIRTUAL TABLE laws_fts USING fts5(
    law_name, 
    law_code, 
    article_title, 
    content, 
    content='laws', 
    content_rowid='id'
);
```

---

## ❓ 3. Q&A 테이블 (qa_pairs)

### 3.1 기본 정보
- **테이블명**: `qa_pairs`
- **목적**: 질문-답변 쌍 데이터 저장
- **예상 데이터량**: 5,000 ~ 20,000건
- **주요 검색 필드**: question, answer, category, confidence_score

### 3.2 스키마 정의

```sql
CREATE TABLE qa_pairs (
    -- 기본 식별자
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    question TEXT NOT NULL,
    answer TEXT NOT NULL,
    
    -- 분류 정보
    category TEXT NOT NULL, -- 'contract', 'civil_law', 'criminal_law' 등
    subcategory TEXT, -- 'damages', 'liability', 'property' 등
    difficulty_level INTEGER DEFAULT 1, -- 1-5 (1: 초급, 5: 고급)
    
    -- 품질 정보
    confidence_score REAL DEFAULT 0.0, -- 0.0 ~ 1.0
    quality_score REAL DEFAULT 0.0, -- 0.0 ~ 1.0
    source_type TEXT NOT NULL, -- 'precedent', 'law', 'generated', 'manual'
    source_id INTEGER, -- 참조하는 판례 또는 법령 ID
    
    -- 태그 및 메타데이터
    tags TEXT, -- JSON 형태로 저장
    keywords TEXT, -- 쉼표로 구분된 키워드
    language TEXT DEFAULT 'ko', -- 'ko', 'en' 등
    
    -- 벡터 검색 연동
    embedding_id INTEGER,
    embedding_vector BLOB,
    
    -- 사용 통계
    usage_count INTEGER DEFAULT 0,
    last_used_at TIMESTAMP,
    user_feedback_score REAL, -- 사용자 피드백 점수
    
    -- 메타데이터
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    created_by TEXT, -- 생성자 ID
    updated_by TEXT, -- 수정자 ID
    
    -- 품질 관리
    review_status TEXT DEFAULT 'pending',
    reviewer_id TEXT,
    review_comment TEXT,
    is_verified BOOLEAN DEFAULT FALSE
);
```

### 3.3 인덱스 설계

```sql
-- 기본 검색 인덱스
CREATE INDEX idx_qa_category ON qa_pairs(category);
CREATE INDEX idx_qa_subcategory ON qa_pairs(subcategory);
CREATE INDEX idx_qa_confidence ON qa_pairs(confidence_score);
CREATE INDEX idx_qa_quality ON qa_pairs(quality_score);
CREATE INDEX idx_qa_source ON qa_pairs(source_type, source_id);

-- 복합 검색 인덱스
CREATE INDEX idx_qa_search ON qa_pairs(category, confidence_score, usage_count);
CREATE INDEX idx_qa_difficulty ON qa_pairs(difficulty_level, category);
CREATE INDEX idx_qa_usage ON qa_pairs(usage_count, last_used_at);

-- 품질 관리 인덱스
CREATE INDEX idx_qa_review ON qa_pairs(review_status, is_verified);
CREATE INDEX idx_qa_embedding ON qa_pairs(embedding_id);

-- 풀텍스트 검색 인덱스
CREATE VIRTUAL TABLE qa_pairs_fts USING fts5(
    question, 
    answer, 
    keywords, 
    content='qa_pairs', 
    content_rowid='id'
);
```

---

## 🔗 4. 관계 테이블

### 4.1 판례 메타데이터 테이블

```sql
CREATE TABLE precedent_metadata (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    precedent_id INTEGER REFERENCES precedents(id) ON DELETE CASCADE,
    metadata_key TEXT NOT NULL,
    metadata_value TEXT NOT NULL,
    metadata_type TEXT DEFAULT 'text', -- 'text', 'number', 'date', 'json'
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX idx_precedent_metadata_precedent ON precedent_metadata(precedent_id);
CREATE INDEX idx_precedent_metadata_key ON precedent_metadata(metadata_key);
```

### 4.2 법령 계층 구조 테이블

```sql
CREATE TABLE law_hierarchy (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    parent_law_id INTEGER REFERENCES laws(id) ON DELETE CASCADE,
    child_law_id INTEGER REFERENCES laws(id) ON DELETE CASCADE,
    relationship_type TEXT NOT NULL, -- 'amendment', 'reference', 'related'
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX idx_law_hierarchy_parent ON law_hierarchy(parent_law_id);
CREATE INDEX idx_law_hierarchy_child ON law_hierarchy(child_law_id);
CREATE INDEX idx_law_hierarchy_type ON law_hierarchy(relationship_type);
```

### 4.3 Q&A 품질 관리 테이블

```sql
CREATE TABLE qa_quality (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    qa_id INTEGER REFERENCES qa_pairs(id) ON DELETE CASCADE,
    quality_score REAL NOT NULL,
    reviewer_id TEXT NOT NULL,
    review_comment TEXT,
    review_criteria TEXT, -- JSON 형태로 저장
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX idx_qa_quality_qa ON qa_quality(qa_id);
CREATE INDEX idx_qa_quality_reviewer ON qa_quality(reviewer_id);
CREATE INDEX idx_qa_quality_score ON qa_quality(quality_score);
```

---

## 📊 5. 통계 및 모니터링 테이블

### 5.1 검색 통계 테이블

```sql
CREATE TABLE search_statistics (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    search_query TEXT NOT NULL,
    search_type TEXT NOT NULL, -- 'precedent', 'law', 'qa', 'combined'
    results_count INTEGER NOT NULL,
    response_time_ms INTEGER NOT NULL,
    user_id TEXT,
    session_id TEXT,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX idx_search_stats_query ON search_statistics(search_query);
CREATE INDEX idx_search_stats_type ON search_statistics(search_type);
CREATE INDEX idx_search_stats_time ON search_statistics(created_at);
```

### 5.2 사용자 피드백 테이블

```sql
CREATE TABLE user_feedback (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    qa_id INTEGER REFERENCES qa_pairs(id) ON DELETE CASCADE,
    user_id TEXT NOT NULL,
    feedback_type TEXT NOT NULL, -- 'helpful', 'not_helpful', 'incorrect', 'incomplete'
    feedback_score INTEGER NOT NULL, -- 1-5
    feedback_comment TEXT,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX idx_user_feedback_qa ON user_feedback(qa_id);
CREATE INDEX idx_user_feedback_user ON user_feedback(user_id);
CREATE INDEX idx_user_feedback_type ON user_feedback(feedback_type);
```

---

## 🚀 6. 데이터베이스 초기화 스크립트

### 6.1 스키마 생성 스크립트

```sql
-- 데이터베이스 초기화
PRAGMA foreign_keys = ON;
PRAGMA journal_mode = WAL;
PRAGMA synchronous = NORMAL;
PRAGMA cache_size = 10000;
PRAGMA temp_store = MEMORY;

-- 테이블 생성 (위의 스키마들)
-- ... (생략)

-- 초기 데이터 삽입
INSERT INTO precedents (case_number, court_name, court_type, case_type, judgment_date, judgment_type, summary, full_text, keywords) VALUES
('2020다12345', '대법원', '대법원', '민사', '2020-01-15', '선고', '계약 해지와 관련된 손해배상 청구권에 대한 판례', '전문 내용...', '계약, 해지, 손해배상, 청구권');

-- 인덱스 생성
-- ... (생략)

-- 통계 업데이트
ANALYZE;
```

### 6.2 데이터 마이그레이션 스크립트

```sql
-- 버전 1.0.0에서 1.1.0으로 마이그레이션
ALTER TABLE precedents ADD COLUMN quality_score REAL DEFAULT 0.0;
ALTER TABLE laws ADD COLUMN quality_score REAL DEFAULT 0.0;
ALTER TABLE qa_pairs ADD COLUMN quality_score REAL DEFAULT 0.0;

-- 인덱스 재생성
DROP INDEX IF EXISTS idx_precedents_quality;
CREATE INDEX idx_precedents_quality ON precedents(quality_score, review_status);
```

---

## 🔍 7. 쿼리 최적화 전략

### 7.1 자주 사용되는 쿼리 패턴

```sql
-- 1. 판례 검색 (법원, 사건유형, 날짜 범위)
SELECT * FROM precedents 
WHERE court_name = ? 
  AND case_type = ? 
  AND judgment_date BETWEEN ? AND ?
ORDER BY judgment_date DESC 
LIMIT 20;

-- 2. 법령 검색 (법률명, 조문)
SELECT * FROM laws 
WHERE law_name LIKE ? 
  AND article_number = ?
ORDER BY effective_date DESC;

-- 3. Q&A 검색 (카테고리, 신뢰도)
SELECT * FROM qa_pairs 
WHERE category = ? 
  AND confidence_score >= ?
ORDER BY quality_score DESC, usage_count DESC 
LIMIT 10;

-- 4. 풀텍스트 검색
SELECT p.*, rank 
FROM precedents p 
JOIN precedents_fts fts ON p.id = fts.rowid 
WHERE precedents_fts MATCH ? 
ORDER BY rank 
LIMIT 20;
```

### 7.2 성능 최적화 팁

1. **인덱스 활용**: 복합 인덱스를 활용한 검색 최적화
2. **쿼리 캐싱**: 자주 사용되는 쿼리 결과 캐싱
3. **배치 처리**: 대량 데이터 처리 시 배치 크기 조정
4. **파티셔닝**: 날짜별 파티셔닝 고려 (향후 확장)

---

## 📈 8. 모니터링 및 유지보수

### 8.1 성능 모니터링

```sql
-- 테이블 크기 확인
SELECT 
    name,
    COUNT(*) as row_count,
    SUM(pgsize) as size_bytes
FROM dbstat 
GROUP BY name;

-- 인덱스 사용률 확인
SELECT 
    name,
    COUNT(*) as usage_count
FROM sqlite_stat1 
GROUP BY name;

-- 느린 쿼리 모니터링
EXPLAIN QUERY PLAN SELECT * FROM precedents WHERE court_name = '대법원';
```

### 8.2 정기 유지보수

```sql
-- 1. 통계 업데이트 (주간)
ANALYZE;

-- 2. 인덱스 재구성 (월간)
REINDEX;

-- 3. 데이터베이스 최적화 (월간)
VACUUM;

-- 4. 오래된 데이터 정리 (분기)
DELETE FROM search_statistics 
WHERE created_at < datetime('now', '-3 months');
```

---

## 🎯 9. 향후 확장 계획

### 9.1 단기 (3개월)
- [ ] 벡터 임베딩 테이블 추가
- [ ] 사용자 세션 테이블 추가
- [ ] API 로그 테이블 추가

### 9.2 중기 (6개월)
- [ ] 다국어 지원 테이블 추가
- [ ] 문서 버전 관리 테이블 추가
- [ ] 실시간 통계 테이블 추가

### 9.3 장기 (1년)
- [ ] PostgreSQL 마이그레이션 검토
- [ ] 분산 데이터베이스 아키텍처 설계
- [ ] 실시간 데이터 동기화 구현

---

*이 스키마 설계는 LawFirmAI 프로젝트의 데이터 저장 및 검색 요구사항을 충족하기 위해 설계되었습니다.*
