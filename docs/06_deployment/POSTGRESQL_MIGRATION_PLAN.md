# SQLite → PostgreSQL 마이그레이션 계획서

## 📋 목차

1. [개요](#개요)
2. [현재 상태 분석](#현재-상태-분석)
3. [벡터 검색 전략 선택](#벡터-검색-전략-선택)
4. [영향받는 시스템 구성요소](#영향받는-시스템-구성요소)
5. [데이터 구조 변경 사항](#데이터-구조-변경-사항)
6. [단계별 마이그레이션 계획](#단계별-마이그레이션-계획)
7. [기술적 변경 사항](#기술적-변경-사항)
8. [데이터 마이그레이션 전략](#데이터-마이그레이션-전략)
9. [리스크 및 대응 방안](#리스크-및-대응-방안)
10. [테스트 계획](#테스트-계획)
11. [롤백 계획](#롤백-계획)

---

## 개요

### 목적
LawFirmAI 프로젝트의 데이터베이스를 SQLite에서 PostgreSQL로 마이그레이션하여 다음과 같은 이점을 얻습니다:

- **확장성**: 대용량 데이터 처리 능력 향상
- **동시성**: 다중 사용자/프로세스 동시 접근 지원
- **벡터 검색**: pgvector 또는 FAISS 선택 가능한 유연한 벡터 검색 시스템
- **전체 텍스트 검색**: PostgreSQL Full-Text Search 활용
- **프로덕션 준비**: 클라우드 배포 및 운영 환경 대응

### 범위
- `lawfirm_langgraph/` 폴더 내 모든 SQLite 사용 코드
- 데이터 적재 스크립트 (`scripts/ingest/`)
- 벡터 임베딩 생성 및 검색 시스템
- 전체 텍스트 검색 (FTS5 → PostgreSQL FTS)
- 버전 관리 시스템
- 테스트 코드

---

## 현재 상태 분석

### SQLite 사용 현황

#### 1. 데이터베이스 연결
- **연결 풀**: `core/data/connection_pool.py` - ThreadLocalConnectionPool (SQLite 전용)
- **직접 연결**: 약 20개 이상의 파일에서 `sqlite3.connect()` 직접 사용

#### 2. 주요 사용 파일 (우선순위별)

**High Priority (핵심 검색 엔진)**
- `core/search/engines/semantic_search_engine_v2.py` - 벡터 검색 엔진
- `core/search/connectors/legal_data_connector_v2.py` - 데이터 커넥터
- `core/search/engines/precedent_search_engine.py` - 판례 검색 엔진

**Medium Priority (서비스 레이어)**
- `core/services/database_keyword_manager.py` - 키워드 관리
- `core/search/optimizers/synonym_database.py` - 동의어 데이터베이스
- `core/shared/feedback/feedback_system.py` - 피드백 시스템
- `core/workflow/checkpoint_manager.py` - 체크포인트 관리
- `core/agents/optimizers/performance_optimizer.py` - 성능 최적화

**Low Priority (유틸리티)**
- `core/data/versioned_schema.py` - 버전 관리 스키마
- 테스트 파일들

#### 3. 데이터 구조

**SQLite 스키마 (현재)**
- `embeddings` 테이블: `vector BLOB` (numpy 배열을 BLOB으로 저장)
- FTS5 가상 테이블: `fts_assembly_laws`, `fts_assembly_articles` 등
- `INTEGER PRIMARY KEY AUTOINCREMENT`
- `sqlite_master` 시스템 테이블 사용

**PostgreSQL 스키마 (목표)**
- `statute_embeddings` 테이블: `embedding_vector VECTOR(768)` (pgvector)
- `precedent_chunks` 테이블: `embedding_vector VECTOR(768)`
- `SERIAL PRIMARY KEY` 또는 `BIGSERIAL PRIMARY KEY`
- `pg_tables` / `information_schema.tables` 사용

---

## 벡터 검색 전략 선택

### 개요

PostgreSQL 마이그레이션 후 벡터 검색을 위해 **pgvector**와 **FAISS** 중 선택하거나, 둘을 함께 사용할 수 있습니다. 각 방법의 특징과 선택 기준을 제공합니다.

### 방법 비교

| 항목 | pgvector | FAISS | 하이브리드 |
|------|----------|-------|-----------|
| **검색 속도** | 중간 (100만개 이하 최적) | 매우 빠름 (대규모 최적) | 상황별 최적 |
| **설치 복잡도** | 낮음 (PostgreSQL 확장) | 중간 (별도 인덱스 관리) | 높음 |
| **운영 복잡도** | 낮음 (단일 시스템) | 높음 (동기화 필요) | 중간 |
| **하이브리드 검색** | 용이 (SQL 통합) | 어려움 (별도 구현) | 용이 |
| **트랜잭션 지원** | 지원 | 미지원 | 부분 지원 |
| **데이터 일관성** | 높음 | 중간 (동기화 필요) | 중간 |
| **확장성** | 중간 | 높음 | 높음 |
| **메모리 사용** | 중간 | 낮음 | 중간 |

### 선택 기준

#### pgvector 권장 상황
- ✅ 데이터 규모: 100만개 이하
- ✅ 하이브리드 검색 필요 (벡터 + 키워드)
- ✅ 운영 단순화 중요
- ✅ 트랜잭션 일관성 중요
- ✅ SQL 쿼리와 통합 필요

#### FAISS 권장 상황
- ✅ 데이터 규모: 100만개 이상
- ✅ 최고 성능 필요
- ✅ 단순 벡터 검색만 필요
- ✅ 메모리 효율 중요
- ✅ 검증된 성능 필요

#### 하이브리드 권장 상황
- ✅ 법령 데이터: pgvector (규모 적당, SQL 통합)
- ✅ 판례 데이터: FAISS (대규모, 빠른 검색)
- ✅ 환경별 선택: 개발(pgvector), 프로덕션(FAISS)
- ✅ 점진적 마이그레이션 필요

### 구현 전략

#### 전략 1: 단일 방법 선택
```bash
# 환경 변수로 선택
VECTOR_SEARCH_METHOD=pgvector  # 또는 faiss
```

#### 전략 2: 데이터 타입별 선택
```bash
# 법령은 pgvector, 판례는 FAISS
STATUTE_VECTOR_METHOD=pgvector
PRECEDENT_VECTOR_METHOD=faiss
```

#### 전략 3: 동적 전환
```python
# 런타임에 전환 가능
search_engine.set_vector_method('pgvector')  # 또는 'faiss'
```

### 성능 비교 가이드

마이그레이션 전 실제 데이터로 성능 테스트를 권장합니다:

```python
# 벤치마크 스크립트 예시
def benchmark_vector_search(query_vectors, limit=10):
    # pgvector 테스트
    pg_times = []
    for qv in query_vectors:
        start = time.time()
        pg_results = pgvector_adapter.search(qv, limit)
        pg_times.append(time.time() - start)
    
    # FAISS 테스트
    faiss_times = []
    for qv in query_vectors:
        start = time.time()
        faiss_results = faiss_adapter.search(qv, limit)
        faiss_times.append(time.time() - start)
    
    print(f"pgvector 평균: {np.mean(pg_times):.4f}초")
    print(f"FAISS 평균: {np.mean(faiss_times):.4f}초")
```

---

## 영향받는 시스템 구성요소

### 1. 데이터베이스 연결 레이어

#### 1.1 연결 풀 시스템
**파일**: `lawfirm_langgraph/core/data/connection_pool.py`

**현재 구조**:
```python
class ThreadLocalConnectionPool:
    def __init__(self, db_path: str):
        self.db_path = db_path
        self._local = threading.local()
    
    def get_connection(self) -> sqlite3.Connection:
        # SQLite 연결 생성
```

**변경 필요**:
- PostgreSQL 연결 풀 지원 (`psycopg2.pool.ThreadedConnectionPool`)
- 데이터베이스 타입 자동 감지 (SQLite vs PostgreSQL)
- 연결 URL 기반 초기화

#### 1.2 직접 연결 사용
**영향받는 파일**: 약 20개 파일
- 모든 `sqlite3.connect()` 호출 제거
- 어댑터 레이어를 통한 연결 사용

### 2. 벡터 임베딩 시스템

#### 2.1 벡터 검색 전략 선택

PostgreSQL 마이그레이션 후 벡터 검색을 위해 **pgvector**와 **FAISS** 중 선택할 수 있습니다. 각 방법의 장단점은 다음과 같습니다:

**pgvector (PostgreSQL 네이티브)**
- ✅ 장점:
  - 데이터베이스 내 벡터 저장 및 검색 (단일 시스템)
  - SQL 쿼리와 통합된 벡터 검색
  - 트랜잭션 지원 및 데이터 일관성
  - 하이브리드 검색 (벡터 + 키워드) 용이
  - 운영 및 관리 단순화
- ❌ 단점:
  - 대규모 데이터셋에서 FAISS보다 느릴 수 있음
  - 인덱스 튜닝 필요
  - PostgreSQL 확장 설치 필요

**FAISS (외부 인덱스)**
- ✅ 장점:
  - 매우 빠른 검색 성능 (대규모 데이터셋)
  - 다양한 인덱스 타입 지원 (IVF, HNSW 등)
  - 메모리 효율적인 인덱싱
  - 검증된 성능
- ❌ 단점:
  - 별도 인덱스 파일 관리 필요
  - 데이터베이스와 인덱스 동기화 필요
  - 하이브리드 검색 구현 복잡
  - 운영 복잡도 증가

**하이브리드 접근 (권장)**
- 법령 데이터: pgvector (데이터 규모가 적당하고 SQL 통합 중요)
- 판례 데이터: FAISS (대규모 데이터, 빠른 검색 필요)
- 또는 환경별 선택 (개발: pgvector, 프로덕션: FAISS)

#### 2.2 벡터 검색 추상화 레이어

벡터 검색 방법을 선택 가능하도록 추상화 레이어를 구현합니다:

**파일**: `lawfirm_langgraph/core/search/engines/vector_search_adapter.py` (신규)

```python
from abc import ABC, abstractmethod
from typing import List, Tuple, Optional
import numpy as np

class VectorSearchAdapter(ABC):
    """벡터 검색 어댑터 인터페이스"""
    
    @abstractmethod
    def search(
        self, 
        query_vector: np.ndarray, 
        limit: int,
        filters: Optional[dict] = None
    ) -> List[Tuple[int, float]]:
        """
        벡터 유사도 검색
        
        Args:
            query_vector: 쿼리 벡터
            limit: 반환할 결과 수
            filters: 필터 조건 (예: {'article_id': [1, 2, 3]})
        
        Returns:
            [(id, distance), ...] 리스트
        """
        pass
    
    @abstractmethod
    def add_vectors(
        self,
        vectors: np.ndarray,
        ids: List[int],
        metadata: Optional[List[dict]] = None
    ):
        """벡터 추가"""
        pass

class PgVectorAdapter(VectorSearchAdapter):
    """pgvector 기반 벡터 검색"""
    
    def __init__(self, connection, table_name: str):
        self.conn = connection
        self.table_name = table_name
    
    def search(
        self, 
        query_vector: np.ndarray, 
        limit: int,
        filters: Optional[dict] = None
    ) -> List[Tuple[int, float]]:
        cursor = self.conn.cursor()
        
        # 필터 조건 추가
        where_clause = ""
        params = [query_vector, limit]
        
        if filters:
            conditions = []
            for key, values in filters.items():
                conditions.append(f"{key} = ANY(%s)")
                params.insert(-1, values)
            where_clause = "WHERE " + " AND ".join(conditions)
        
        query = f"""
            SELECT article_id, embedding_vector <=> %s::vector AS distance
            FROM {self.table_name}
            {where_clause}
            ORDER BY distance
            LIMIT %s
        """
        
        cursor.execute(query, params)
        return [(row[0], row[1]) for row in cursor.fetchall()]

class FaissAdapter(VectorSearchAdapter):
    """FAISS 기반 벡터 검색"""
    
    def __init__(self, index_path: str, vector_loader):
        import faiss
        self.index = faiss.read_index(index_path)
        self.vector_loader = vector_loader  # 벡터 로더 함수
    
    def search(
        self, 
        query_vector: np.ndarray, 
        limit: int,
        filters: Optional[dict] = None
    ) -> List[Tuple[int, float]]:
        query_vector = query_vector.reshape(1, -1).astype('float32')
        distances, indices = self.index.search(query_vector, limit)
        
        results = []
        for idx, dist in zip(indices[0], distances[0]):
            if idx != -1:  # FAISS의 -1은 유효하지 않은 결과
                results.append((int(idx), float(dist)))
        
        # 필터 적용 (필요시)
        if filters:
            results = self._apply_filters(results, filters)
        
        return results

class VectorSearchFactory:
    """벡터 검색 어댑터 팩토리"""
    
    @staticmethod
    def create(
        method: str,
        connection=None,
        table_name: str = None,
        index_path: str = None,
        vector_loader=None
    ) -> VectorSearchAdapter:
        """
        벡터 검색 어댑터 생성
        
        Args:
            method: 'pgvector' 또는 'faiss'
            connection: PostgreSQL 연결 (pgvector용)
            table_name: 테이블명 (pgvector용)
            index_path: FAISS 인덱스 경로 (faiss용)
            vector_loader: 벡터 로더 함수 (faiss용)
        """
        if method == 'pgvector':
            return PgVectorAdapter(connection, table_name)
        elif method == 'faiss':
            return FaissAdapter(index_path, vector_loader)
        else:
            raise ValueError(f"Unknown vector search method: {method}")
```

#### 2.3 임베딩 저장 방식

**옵션 1: pgvector 사용**
```python
# pgvector로 저장
from pgvector.psycopg2 import register_vector
cursor.execute(
    "INSERT INTO statute_embeddings (article_id, embedding_vector) VALUES (%s, %s)",
    (article_id, vector)
)

# pgvector에서 검색
cursor.execute(
    "SELECT article_id, embedding_vector <=> %s::vector AS distance "
    "FROM statute_embeddings ORDER BY distance LIMIT %s",
    (query_vector, limit)
)
```

**옵션 2: FAISS 사용**
```python
# PostgreSQL에 메타데이터만 저장, 벡터는 FAISS 인덱스에
cursor.execute(
    "INSERT INTO statute_embeddings (article_id, embedding_version, metadata) "
    "VALUES (%s, %s, %s)",
    (article_id, version, metadata_json)
)

# FAISS 인덱스에 벡터 추가
import faiss
index.add(vectors.astype('float32'))
faiss.write_index(index, 'statute_embeddings.index')
```

**옵션 3: 하이브리드 (pgvector + FAISS)**
```python
# pgvector에 저장 (주 인덱스)
cursor.execute(
    "INSERT INTO statute_embeddings (article_id, embedding_vector) VALUES (%s, %s)",
    (article_id, vector)
)

# FAISS 인덱스에도 저장 (고성능 검색용)
faiss_index.add(vector.reshape(1, -1).astype('float32'))
```

#### 2.4 임베딩 테이블 구조

**pgvector 사용 시**:
```sql
CREATE TABLE statute_embeddings (
    id SERIAL PRIMARY KEY,
    article_id INTEGER NOT NULL REFERENCES statutes_articles(id),
    embedding_vector VECTOR(768),
    embedding_version INTEGER NOT NULL DEFAULT 1,
    metadata JSONB,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX ON statute_embeddings 
USING ivfflat (embedding_vector vector_cosine_ops) WITH (lists = 100);
```

**FAISS 사용 시**:
```sql
CREATE TABLE statute_embeddings (
    id SERIAL PRIMARY KEY,
    article_id INTEGER NOT NULL REFERENCES statutes_articles(id),
    embedding_version INTEGER NOT NULL DEFAULT 1,
    metadata JSONB,
    faiss_index_path TEXT,  -- FAISS 인덱스 파일 경로
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);
-- 벡터는 FAISS 인덱스 파일에 저장
```

**하이브리드 사용 시**:
```sql
CREATE TABLE statute_embeddings (
    id SERIAL PRIMARY KEY,
    article_id INTEGER NOT NULL REFERENCES statutes_articles(id),
    embedding_vector VECTOR(768),  -- pgvector용
    embedding_version INTEGER NOT NULL DEFAULT 1,
    metadata JSONB,
    faiss_index_path TEXT,  -- FAISS 인덱스 경로 (선택적)
    search_method VARCHAR(20) DEFAULT 'pgvector',  -- 'pgvector' 또는 'faiss'
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- pgvector 인덱스 (선택적)
CREATE INDEX ON statute_embeddings 
USING ivfflat (embedding_vector vector_cosine_ops) WITH (lists = 100);
```

**영향받는 파일**:
- `core/search/engines/semantic_search_engine_v2.py`
- `core/services/semantic_search_engine_v2.py`
- `core/search/connectors/legal_data_connector_v2.py`

### 3. 전체 텍스트 검색 (FTS)

#### 3.1 FTS5 → PostgreSQL Full-Text Search
**현재 (SQLite FTS5)**:
```sql
CREATE VIRTUAL TABLE fts_assembly_articles USING fts5(
    article_id,
    article_content,
    content='assembly_articles',
    content_rowid='rowid'
);

SELECT * FROM fts_assembly_articles WHERE fts_assembly_articles MATCH ?;
```

**변경 후 (PostgreSQL FTS)**:
```sql
-- GIN 인덱스 생성
CREATE INDEX idx_statute_articles_fts ON statute_articles 
USING gin(to_tsvector('korean', text));

-- 검색 쿼리
SELECT * FROM statute_articles 
WHERE to_tsvector('korean', text) @@ to_tsquery('korean', ?)
ORDER BY ts_rank(to_tsvector('korean', text), to_tsquery('korean', ?)) DESC;
```

**영향받는 파일**:
- `core/search/engines/precedent_search_engine.py`
- `core/search/handlers/search_service.py`
- `core/search/connectors/legal_data_connector_v2.py`
- `core/services/precedent_search_engine.py`

#### 3.2 FTS 쿼리 변환
**FTS5 특수 문자 처리**:
- `"`, `*`, `^`, `(`, `)` → PostgreSQL FTS 토큰화 규칙으로 변환
- `AND`, `OR`, `NOT` → PostgreSQL FTS 연산자로 변환

### 4. 데이터 적재 시스템

#### 4.1 적재 스크립트
**영향받는 스크립트**:
- `scripts/ingest/open_law/embedding/generate_statute_embeddings.py` ✅ (이미 PostgreSQL 사용)
- `scripts/ingest/ingest_statutes.py` - SQLite 사용 가능
- `scripts/ingest/ingest_cases.py` - SQLite 사용 가능
- `scripts/ingest/ingest_interpretations.py` - SQLite 사용 가능

**변경 필요**:
- 모든 적재 스크립트가 PostgreSQL 연결 사용
- 테이블 스키마 확인 및 자동 생성
- 배치 삽입 최적화 (PostgreSQL COPY 명령 활용)

#### 4.2 버전 관리 시스템
**파일**: `core/data/versioned_schema.py`

**현재**:
```python
def ensure_versioned_schema(db_path: Path) -> None:
    with connect(db_path) as conn:  # SQLite 연결
        cur = conn.cursor()
        cur.execute("CREATE TABLE IF NOT EXISTS laws (...)")
```

**변경 필요**:
- PostgreSQL 스키마 생성 지원
- 버전별 스키마 분리 (스키마 네임스페이스 활용)
- 마이그레이션 스크립트 실행

### 5. 검색 쿼리 시스템

#### 5.1 SQL 문법 차이

**파라미터 바인딩**:
- SQLite: `?` → PostgreSQL: `%s`

**시스템 테이블**:
- SQLite: `sqlite_master` → PostgreSQL: `pg_tables` / `information_schema.tables`

**데이터 타입**:
- `INTEGER PRIMARY KEY AUTOINCREMENT` → `SERIAL PRIMARY KEY`
- `TEXT` → `TEXT` (동일)
- `BLOB` → `BYTEA` (일반 데이터) 또는 `VECTOR` (임베딩)

**집계 함수**:
- `GROUP_CONCAT(text, '\n\n')` → `STRING_AGG(text, E'\n\n')`

**NULL 처리**:
- SQLite: `NULLS LAST` 미지원 (CASE 문 사용)
- PostgreSQL: `NULLS LAST` 네이티브 지원

#### 5.2 쿼리 변환 예시

**예시 1: 테이블 존재 확인**
```python
# SQLite
cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name=?", (table_name,))

# PostgreSQL
cursor.execute(
    "SELECT tablename FROM pg_tables WHERE schemaname='public' AND tablename=%s",
    (table_name,)
)
```

**예시 2: 텍스트 연결**
```python
# SQLite
cursor.execute("SELECT GROUP_CONCAT(cp.text, '\n\n') FROM case_paragraphs cp WHERE cp.case_id=?", (case_id,))

# PostgreSQL
cursor.execute(
    "SELECT STRING_AGG(cp.text, E'\\n\\n') FROM case_paragraphs cp WHERE cp.case_id=%s",
    (case_id,)
)
```

**예시 3: 벡터 검색**
```python
# SQLite (FAISS 사용)
vectors = load_vectors_from_blob()
index = faiss.IndexFlatL2(dim)
index.add(vectors)
distances, indices = index.search(query_vector, k)

# PostgreSQL (pgvector)
cursor.execute(
    "SELECT article_id, embedding_vector <=> %s::vector AS distance "
    "FROM statute_embeddings ORDER BY distance LIMIT %s",
    (query_vector, k)
)
```

---

## 데이터 구조 변경 사항

### 1. 테이블 스키마 변경

#### 1.1 법령 데이터
**SQLite**:
```sql
CREATE TABLE assembly_articles (
    article_id TEXT PRIMARY KEY,
    law_id TEXT NOT NULL,
    article_content TEXT NOT NULL
);
```

**PostgreSQL**:
```sql
CREATE TABLE statutes_articles (
    id SERIAL PRIMARY KEY,
    statute_id INTEGER NOT NULL REFERENCES statutes(id),
    article_no VARCHAR(50) NOT NULL,
    text TEXT NOT NULL
);
```

#### 1.2 임베딩 테이블
**SQLite**:
```sql
CREATE TABLE embeddings (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    chunk_id INTEGER,
    vector BLOB NOT NULL,
    dim INTEGER NOT NULL
);
```

**PostgreSQL**:
```sql
CREATE TABLE statute_embeddings (
    id SERIAL PRIMARY KEY,
    article_id INTEGER NOT NULL REFERENCES statutes_articles(id),
    embedding_vector VECTOR(768),
    embedding_version INTEGER NOT NULL DEFAULT 1,
    metadata JSONB,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);
```

### 2. 인덱스 변경

#### 2.1 벡터 인덱스
**SQLite**: FAISS 인덱스 (외부 파일)
**PostgreSQL**: IVFFlat 인덱스 (pgvector)

```sql
CREATE INDEX ON statute_embeddings 
USING ivfflat (embedding_vector vector_cosine_ops) WITH (lists = 100);
```

#### 2.2 전체 텍스트 검색 인덱스
**SQLite**: FTS5 가상 테이블
**PostgreSQL**: GIN 인덱스

```sql
CREATE INDEX idx_statute_articles_fts ON statutes_articles 
USING gin(to_tsvector('korean', text));
```

### 3. 메타데이터 구조

#### 3.1 JSON 지원
**SQLite**: `TEXT` (JSON 문자열)
**PostgreSQL**: `JSONB` (네이티브 JSON 타입)

```sql
-- PostgreSQL
metadata JSONB  -- 인덱싱 및 쿼리 최적화 가능
```

---

## 단계별 마이그레이션 계획

### Phase 1: 인프라 및 추상화 레이어 구축 (3-4일)

#### 1.1 데이터베이스 어댑터 생성
**파일**: `lawfirm_langgraph/core/data/db_adapter.py` (신규)

**기능**:
- 데이터베이스 타입 자동 감지
- 통합 연결 인터페이스
- SQL 문법 변환 유틸리티
- Row 객체 변환

**구현 내용**:
```python
class DatabaseAdapter:
    """데이터베이스 타입에 독립적인 어댑터"""
    
    def __init__(self, database_url: str):
        self.db_type = self._detect_db_type(database_url)
        self.connection_pool = self._create_connection_pool(database_url)
    
    def get_connection(self):
        """연결 가져오기"""
        pass
    
    def execute_query(self, query: str, params: tuple):
        """쿼리 실행 (자동 변환)"""
        pass
    
    def convert_sql(self, sqlite_sql: str) -> str:
        """SQLite SQL을 PostgreSQL SQL로 변환"""
        pass
```

#### 1.2 벡터 검색 추상화 레이어 생성
**파일**: `lawfirm_langgraph/core/search/engines/vector_search_adapter.py` (신규)

**기능**:
- pgvector와 FAISS 통합 인터페이스
- 환경 변수 기반 방법 선택
- 동적 전환 지원

**구현 내용**: [2.2 벡터 검색 추상화 레이어](#22-벡터-검색-추상화-레이어) 참조

#### 1.3 연결 풀 확장
**파일**: `lawfirm_langgraph/core/data/connection_pool.py` (수정)

**변경 내용**:
- PostgreSQL 연결 풀 지원 추가
- 데이터베이스 타입별 연결 풀 생성
- 기존 SQLite 연결 풀 유지 (하위 호환성)

#### 1.4 SQL 변환 유틸리티
**파일**: `lawfirm_langgraph/core/data/sql_adapter.py` (신규)

**기능**:
- `?` → `%s` 변환
- `sqlite_master` → `pg_tables` 변환
- `GROUP_CONCAT` → `STRING_AGG` 변환
- `INTEGER PRIMARY KEY AUTOINCREMENT` → `SERIAL PRIMARY KEY` 변환

### Phase 2: 설정 시스템 업데이트 (1일)

#### 2.1 설정 파일 수정
**파일**: `lawfirm_langgraph/config/app_config.py`

**변경 내용**:
- `database_url` 기본값을 PostgreSQL 형식 지원
- 데이터베이스 타입 자동 감지 로직

#### 2.2 환경 변수 처리
**파일**: `lawfirm_langgraph/core/shared/utils/config.py`

**변경 내용**:
- PostgreSQL URL 파싱 로직 추가
- SQLite와 PostgreSQL 모두 지원

### Phase 3: 핵심 검색 엔진 마이그레이션 (4-5일)

#### 3.1 벡터 검색 엔진
**파일**: `core/search/engines/semantic_search_engine_v2.py`

**변경 내용**:
- 벡터 검색 추상화 레이어 통합
- pgvector 또는 FAISS 선택 가능
- 환경 변수로 검색 방법 제어

**주요 변경점**:
```python
# 기존
def _load_chunk_vectors(self):
    vector_blob = row['vector']
    vector = np.frombuffer(vector_blob, dtype=np.float32)

# 변경 후 - 추상화 레이어 사용
from core.search.engines.vector_search_adapter import VectorSearchFactory

class SemanticSearchEngineV2:
    def __init__(self, config):
        # 환경 변수에서 벡터 검색 방법 선택
        vector_method = os.getenv('VECTOR_SEARCH_METHOD', 'pgvector')
        
        if vector_method == 'pgvector':
            self.vector_adapter = VectorSearchFactory.create(
                method='pgvector',
                connection=self.connection,
                table_name='statute_embeddings'
            )
        elif vector_method == 'faiss':
            self.vector_adapter = VectorSearchFactory.create(
                method='faiss',
                index_path=config.faiss_index_path,
                vector_loader=self._load_vectors_from_db
            )
        else:
            raise ValueError(f"Unknown vector search method: {vector_method}")
    
    def _search_vectors(self, query_vector, limit, filters=None):
        """벡터 검색 (어댑터를 통한 통합 인터페이스)"""
        return self.vector_adapter.search(query_vector, limit, filters)
```

#### 3.2 데이터 커넥터
**파일**: `core/search/connectors/legal_data_connector_v2.py`

**변경 내용**:
- 모든 SQLite 쿼리를 어댑터를 통해 실행
- FTS5 쿼리 → PostgreSQL FTS 변환
- 벡터 검색 쿼리 추가

#### 3.3 판례 검색 엔진
**파일**: `core/search/engines/precedent_search_engine.py`

**변경 내용**:
- FTS5 검색 → PostgreSQL FTS 변환
- 쿼리 최적화 (한국어 토크나이저 활용)

### Phase 4: 서비스 레이어 마이그레이션 (3-4일)

#### 4.1 키워드 관리
**파일**: `core/services/database_keyword_manager.py`

**변경 내용**:
- 모든 `sqlite3.connect()` 제거
- 어댑터 사용

#### 4.2 동의어 데이터베이스
**파일**: `core/search/optimizers/synonym_database.py`

**변경 내용**:
- SQLite 연결 → 어댑터 사용

#### 4.3 피드백 시스템
**파일**: `core/shared/feedback/feedback_system.py`

**변경 내용**:
- SQLite 연결 → 어댑터 사용

#### 4.4 체크포인트 관리
**파일**: `core/workflow/checkpoint_manager.py`

**변경 내용**:
- SQLite 연결 → 어댑터 사용
- LangGraph 체크포인트 저장소 지원 (PostgreSQL)

### Phase 5: 데이터 적재 시스템 업데이트 (2-3일)

#### 5.1 적재 스크립트 검토
**파일**: `scripts/ingest/` 내 모든 스크립트

**작업**:
- SQLite 사용 스크립트 식별
- PostgreSQL 연결로 변경
- 배치 삽입 최적화 (COPY 명령 활용)

#### 5.2 버전 관리 시스템
**파일**: `core/data/versioned_schema.py`

**변경 내용**:
- PostgreSQL 스키마 생성 지원
- 마이그레이션 스크립트 실행

### Phase 6: 테스트 및 검증 (3-4일)

#### 6.1 단위 테스트 업데이트
**파일**: `tests/` 내 모든 테스트 파일

**작업**:
- 테스트 데이터베이스 설정 (PostgreSQL)
- SQLite 의존성 제거
- 테스트 쿼리 업데이트

#### 6.2 통합 테스트
- 전체 워크플로우 테스트
- 성능 비교 테스트
- 데이터 무결성 검증

---

## 기술적 변경 사항

### 1. 의존성 추가

#### 1.1 Python 패키지
```txt
# requirements.txt 또는 pyproject.toml에 추가
psycopg2-binary>=2.9.0  # PostgreSQL 드라이버
pgvector>=0.2.0          # pgvector 확장 (pgvector 사용 시)
faiss-cpu>=1.7.4         # FAISS (FAISS 사용 시, 또는 faiss-gpu)
sqlalchemy>=2.0.0        # ORM (선택적)
```

**선택적 의존성**:
- pgvector 사용 시: `pgvector>=0.2.0` 필수
- FAISS 사용 시: `faiss-cpu>=1.7.4` 또는 `faiss-gpu` 필수
- 둘 다 사용 시: 두 패키지 모두 설치

#### 1.2 PostgreSQL 확장
```sql
-- PostgreSQL에서 실행 필요 (pgvector 사용 시)
CREATE EXTENSION IF NOT EXISTS vector;  -- pgvector
CREATE EXTENSION IF NOT EXISTS pg_trgm;  -- 트라이그램 (선택적)
```

**주의**: FAISS만 사용하는 경우 pgvector 확장은 불필요합니다.

### 2. 연결 문자열 형식

#### 2.1 SQLite
```
sqlite:///./data/lawfirm_v2.db
```

#### 2.2 PostgreSQL
```
postgresql://user:password@host:port/database
```

### 3. 벡터 검색 성능 최적화

#### 3.1 pgvector 인덱스 튜닝
```sql
-- IVFFlat 인덱스 (리스트 수 조정)
CREATE INDEX ON statute_embeddings 
USING ivfflat (embedding_vector vector_cosine_ops) 
WITH (lists = 100);  -- 데이터가 많으면 증가 (예: 1000만개 이상 시 1000+)

-- HNSW 인덱스 (PostgreSQL 15+, pgvector 0.5+, 더 빠른 검색)
CREATE INDEX ON statute_embeddings 
USING hnsw (embedding_vector vector_cosine_ops)
WITH (m = 16, ef_construction = 64);
```

#### 3.2 FAISS 인덱스 튜닝
```python
import faiss

# IVF 인덱스 (대규모 데이터)
dim = 768
nlist = 100  # 클러스터 수
quantizer = faiss.IndexFlatL2(dim)
index = faiss.IndexIVFFlat(quantizer, dim, nlist)
index.train(vectors)
index.add(vectors)

# HNSW 인덱스 (최고 성능)
index = faiss.IndexHNSWFlat(dim, 32)  # 32는 연결 수
index.add(vectors)

# 인덱스 저장
faiss.write_index(index, 'statute_embeddings.index')
```

#### 3.3 선택 기준

**pgvector 권장 상황**:
- 데이터 규모: 100만개 이하
- 하이브리드 검색 필요 (벡터 + 키워드)
- 운영 단순화 중요
- 트랜잭션 일관성 중요

**FAISS 권장 상황**:
- 데이터 규모: 100만개 이상
- 최고 성능 필요
- 단순 벡터 검색만 필요
- 메모리 효율 중요

**하이브리드 권장 상황**:
- 법령 데이터: pgvector (규모 적당, SQL 통합)
- 판례 데이터: FAISS (대규모, 빠른 검색)
- 또는 환경별: 개발(pgvector), 프로덕션(FAISS)

### 4. 전체 텍스트 검색 최적화

#### 4.1 한국어 토크나이저
```sql
-- 한국어 텍스트 검색 최적화
CREATE INDEX idx_statute_articles_fts ON statutes_articles 
USING gin(to_tsvector('korean', text));

-- 검색 쿼리
SELECT * FROM statutes_articles 
WHERE to_tsvector('korean', text) @@ to_tsquery('korean', '계약')
ORDER BY ts_rank(to_tsvector('korean', text), to_tsquery('korean', '계약')) DESC;
```

---

## 데이터 마이그레이션 전략

### 1. 데이터 내보내기 (SQLite)

#### 1.1 스키마 내보내기
```bash
sqlite3 lawfirm_v2.db .schema > schema_export.sql
```

#### 1.2 데이터 내보내기
```python
# CSV로 내보내기
import sqlite3
import csv

conn = sqlite3.connect('lawfirm_v2.db')
cursor = conn.cursor()

cursor.execute("SELECT * FROM assembly_articles")
with open('assembly_articles.csv', 'w', encoding='utf-8') as f:
    writer = csv.writer(f)
    writer.writerow([description[0] for description in cursor.description])
    writer.writerows(cursor.fetchall())
```

### 2. 데이터 가져오기 (PostgreSQL)

#### 2.1 스키마 생성
```bash
psql -d lawfirm_ai -f scripts/migrations/002_migrate_sqlite_to_postgresql.sql
```

#### 2.2 데이터 가져오기
```python
# COPY 명령 사용 (고성능)
import psycopg2

conn = psycopg2.connect("postgresql://...")
cursor = conn.cursor()

with open('assembly_articles.csv', 'r', encoding='utf-8') as f:
    cursor.copy_expert(
        "COPY statutes_articles FROM STDIN WITH CSV HEADER",
        f
    )
conn.commit()
```

### 3. 벡터 임베딩 마이그레이션

#### 3.1 pgvector로 마이그레이션
```python
# SQLite에서 벡터 로드
sqlite_conn = sqlite3.connect('lawfirm_v2.db')
cursor = sqlite_conn.cursor()
cursor.execute("SELECT chunk_id, vector, dim FROM embeddings")

# PostgreSQL에 저장 (pgvector)
pg_conn = psycopg2.connect("postgresql://...")
pg_cursor = pg_conn.cursor()

for row in cursor.fetchall():
    chunk_id, vector_blob, dim = row
    vector = np.frombuffer(vector_blob, dtype=np.float32)
    
    pg_cursor.execute(
        "INSERT INTO statute_embeddings (article_id, embedding_vector) VALUES (%s, %s)",
        (chunk_id, vector)
    )

pg_conn.commit()
```

#### 3.2 FAISS로 마이그레이션
```python
# SQLite에서 벡터 로드
sqlite_conn = sqlite3.connect('lawfirm_v2.db')
cursor = sqlite_conn.cursor()
cursor.execute("SELECT chunk_id, vector, dim FROM embeddings")

# FAISS 인덱스 생성
import faiss
import numpy as np

dim = 768
vectors = []
ids = []

for row in cursor.fetchall():
    chunk_id, vector_blob, dim = row
    vector = np.frombuffer(vector_blob, dtype=np.float32)
    vectors.append(vector)
    ids.append(chunk_id)

# 벡터 배열 생성
vectors_array = np.array(vectors).astype('float32')

# FAISS 인덱스 생성 및 저장
index = faiss.IndexFlatL2(dim)  # 또는 IndexIVFFlat, IndexHNSWFlat
index.add(vectors_array)
faiss.write_index(index, 'statute_embeddings.index')

# PostgreSQL에 메타데이터만 저장
pg_conn = psycopg2.connect("postgresql://...")
pg_cursor = pg_conn.cursor()

for chunk_id in ids:
    pg_cursor.execute(
        "INSERT INTO statute_embeddings (article_id, faiss_index_path) VALUES (%s, %s)",
        (chunk_id, 'statute_embeddings.index')
    )

pg_conn.commit()
```

#### 3.3 하이브리드 마이그레이션
```python
# SQLite에서 벡터 로드
sqlite_conn = sqlite3.connect('lawfirm_v2.db')
cursor = sqlite_conn.cursor()
cursor.execute("SELECT chunk_id, vector, dim FROM embeddings")

# PostgreSQL 연결
pg_conn = psycopg2.connect("postgresql://...")
pg_cursor = pg_conn.cursor()

# FAISS 인덱스 생성
import faiss
vectors = []
ids = []

for row in cursor.fetchall():
    chunk_id, vector_blob, dim = row
    vector = np.frombuffer(vector_blob, dtype=np.float32)
    vectors.append(vector)
    ids.append(chunk_id)

vectors_array = np.array(vectors).astype('float32')
index = faiss.IndexFlatL2(dim)
index.add(vectors_array)
faiss.write_index(index, 'statute_embeddings.index')

# PostgreSQL에 pgvector와 FAISS 경로 모두 저장
for chunk_id, vector in zip(ids, vectors):
    pg_cursor.execute(
        "INSERT INTO statute_embeddings "
        "(article_id, embedding_vector, faiss_index_path, search_method) "
        "VALUES (%s, %s, %s, %s)",
        (chunk_id, vector, 'statute_embeddings.index', 'faiss')
    )

pg_conn.commit()
```

### 4. 검증

#### 4.1 데이터 개수 확인
```sql
-- SQLite
SELECT COUNT(*) FROM assembly_articles;

-- PostgreSQL
SELECT COUNT(*) FROM statutes_articles;
```

#### 4.2 샘플 데이터 비교
```python
# SQLite에서 샘플 로드
sqlite_sample = sqlite_cursor.execute("SELECT * FROM assembly_articles LIMIT 10").fetchall()

# PostgreSQL에서 샘플 로드
pg_sample = pg_cursor.execute("SELECT * FROM statutes_articles LIMIT 10").fetchall()

# 비교
assert len(sqlite_sample) == len(pg_sample)
```

---

## 리스크 및 대응 방안

### 1. 기술적 리스크

#### 1.1 SQL 문법 차이
**리스크**: SQLite와 PostgreSQL의 SQL 문법 차이로 인한 버그

**대응**:
- SQL 어댑터 레이어로 자동 변환
- 철저한 단위 테스트
- 쿼리 로깅 및 모니터링

#### 1.2 성능 저하
**리스크**: PostgreSQL로 변경 시 성능 저하 가능성

**대응**:
- 인덱스 최적화
- 연결 풀 튜닝
- 쿼리 실행 계획 분석 및 최적화
- 벤치마크 테스트

#### 1.3 벡터 검색 정확도
**리스크**: pgvector와 FAISS 간 검색 결과 차이

**대응**:
- 검색 결과 비교 테스트 (동일 쿼리로 두 방법 테스트)
- 정확도 메트릭 수집 (Recall@K, Precision@K)
- 필요 시 하이브리드 접근 (pgvector + FAISS)
- 환경 변수로 검색 방법 전환 가능하도록 구현

### 2. 운영 리스크

#### 2.1 데이터 손실
**리스크**: 마이그레이션 중 데이터 손실

**대응**:
- 마이그레이션 전 전체 백업
- 단계별 검증
- 롤백 계획 수립

#### 2.2 다운타임
**리스크**: 마이그레이션 중 서비스 중단

**대응**:
- 단계적 마이그레이션
- 읽기 전용 모드 지원
- 트래픽 분산 (로드 밸런서)

### 3. 데이터 일관성

#### 3.1 외래키 제약
**리스크**: PostgreSQL의 엄격한 외래키 제약

**대응**:
- 데이터 정제 (마이그레이션 전)
- 외래키 제약 조건 검증
- 순차적 데이터 로딩

---

## 테스트 계획

### 1. 단위 테스트

#### 1.1 데이터베이스 어댑터
- SQLite와 PostgreSQL 모두 지원 확인
- SQL 변환 정확도 검증
- 연결 풀 동작 확인

#### 1.2 벡터 검색
- pgvector 검색 정확도 및 성능
- FAISS 검색 정확도 및 성능
- 두 방법 간 결과 비교 (Recall@K, Precision@K)
- 추상화 레이어 동작 확인
- 환경 변수 전환 테스트

#### 1.3 전체 텍스트 검색
- PostgreSQL FTS 검색 결과
- FTS5와 결과 비교
- 한국어 토크나이저 동작 확인

### 2. 통합 테스트

#### 2.1 검색 워크플로우
- 전체 검색 파이프라인 테스트
- 하이브리드 검색 (벡터 + 키워드)
- 결과 정확도 검증

#### 2.2 데이터 적재
- 대량 데이터 적재 테스트
- 배치 처리 성능
- 트랜잭션 무결성

### 3. 성능 테스트

#### 3.1 벤치마크
- 검색 속도 비교
- 동시 접속 처리 능력
- 메모리 사용량

#### 3.2 부하 테스트
- 동시 쿼리 처리
- 대용량 데이터 검색
- 연결 풀 한계 테스트

---

## 롤백 계획

### 1. 롤백 조건
- 데이터 손실 발생
- 성능 저하가 허용 범위 초과
- 치명적 버그 발견

### 2. 롤백 절차

#### 2.1 즉시 롤백
1. PostgreSQL 연결 차단
2. SQLite 데이터베이스 복원
3. 설정 파일 원복
4. 서비스 재시작

#### 2.2 단계별 롤백
1. 문제가 발생한 단계 이전으로 롤백
2. 해당 단계 재검토 및 수정
3. 재마이그레이션

### 3. 롤백 검증
- 데이터 무결성 확인
- 서비스 정상 동작 확인
- 성능 지표 확인

---

## 예상 작업량 및 일정

### 작업량 추정

| Phase | 작업 내용 | 예상 기간 | 담당자 |
|------|----------|----------|--------|
| Phase 1 | 인프라 구축 | 3-4일 | 백엔드 개발자 |
| Phase 2 | 설정 시스템 | 1일 | 백엔드 개발자 |
| Phase 3 | 핵심 엔진 마이그레이션 | 4-5일 | 백엔드 개발자 |
| Phase 4 | 서비스 레이어 | 3-4일 | 백엔드 개발자 |
| Phase 5 | 데이터 적재 시스템 | 2-3일 | 데이터 엔지니어 |
| Phase 6 | 테스트 및 검증 | 3-4일 | QA + 개발자 |

**총 예상 기간**: 16-21일 (약 3-4주)

### 마일스톤

- **Week 1**: Phase 1-2 완료 (인프라 구축)
- **Week 2**: Phase 3 완료 (핵심 엔진 마이그레이션)
- **Week 3**: Phase 4-5 완료 (서비스 레이어 및 데이터 적재)
- **Week 4**: Phase 6 완료 (테스트 및 검증)

---

## 체크리스트

### 마이그레이션 전
- [ ] PostgreSQL 서버 설정 및 접근 확인
- [ ] 벡터 검색 방법 결정 (pgvector / FAISS / 하이브리드)
- [ ] pgvector 확장 설치 확인 (pgvector 사용 시)
- [ ] FAISS 설치 확인 (FAISS 사용 시)
- [ ] 데이터베이스 백업 완료
- [ ] 테스트 환경 구축
- [ ] 환경 변수 설정 (`VECTOR_SEARCH_METHOD`)

### 마이그레이션 중
- [ ] 각 Phase별 코드 리뷰
- [ ] 단위 테스트 통과
- [ ] 통합 테스트 통과
- [ ] 성능 테스트 통과

### 마이그레이션 후
- [ ] 프로덕션 데이터 검증
- [ ] 모니터링 설정
- [ ] 문서 업데이트
- [ ] 팀 교육 및 공유

---

## 참고 자료

### 문서
- [PostgreSQL 공식 문서](https://www.postgresql.org/docs/)
- [pgvector 문서](https://github.com/pgvector/pgvector)
- [PostgreSQL Full-Text Search](https://www.postgresql.org/docs/current/textsearch.html)

### 마이그레이션 스크립트
- `scripts/migrations/002_migrate_sqlite_to_postgresql.sql`
- `scripts/migrations/005_add_embedding_version_management_postgresql.sql`

### 관련 파일
- `scripts/database/init_postgresql.py`
- `scripts/ingest/open_law/embedding/pgvector/pgvector_embedder.py`
- `scripts/ingest/open_law/embedding/faiss/faiss_embedder.py`

### 환경 변수 설정

#### 벡터 검색 방법 선택
```bash
# pgvector 사용
VECTOR_SEARCH_METHOD=pgvector

# FAISS 사용
VECTOR_SEARCH_METHOD=faiss
FAISS_INDEX_PATH=./data/embeddings/statute_embeddings.index

# 하이브리드 (법령: pgvector, 판례: FAISS)
VECTOR_SEARCH_METHOD=hybrid
STATUTE_VECTOR_METHOD=pgvector
PRECEDENT_VECTOR_METHOD=faiss
PRECEDENT_FAISS_INDEX_PATH=./data/embeddings/precedent_embeddings.index
```

---

**작성일**: 2025-01-XX  
**최종 수정일**: 2025-01-XX  
**작성자**: LawFirmAI 개발팀
