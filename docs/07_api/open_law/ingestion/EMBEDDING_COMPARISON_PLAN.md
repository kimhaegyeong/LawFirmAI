# FAISS vs pgvector 비교 테스트 개발 계획

## 📋 목차

1. [프로젝트 개요](#프로젝트-개요)
2. [목표](#목표)
3. [시스템 구조](#시스템-구조)
4. [구현 단계](#구현-단계)
5. [파일 구조](#파일-구조)
6. [API 설계](#api-설계)
7. [테스트 계획](#테스트-계획)
8. [성능 측정 지표](#성능-측정-지표)
9. [예상 작업 시간](#예상-작업-시간)
10. [다음 단계](#다음-단계)

---

## 프로젝트 개요

PostgreSQL에 수집된 Open Law API 데이터(법령 조문, 판례 청크)에 대해 FAISS와 pgvector 두 가지 벡터 검색 시스템을 구현하고 성능을 비교 테스트합니다.

### 배경

- **현재 상황**: PostgreSQL에 법령 및 판례 데이터 수집 완료
- **기존 시스템**: FAISS 기반 벡터 검색 (SQLite 데이터용)
- **새 요구사항**: PostgreSQL 데이터에 대한 벡터 검색 시스템 필요
- **목적**: FAISS와 pgvector의 성능, 정확도, 운영 편의성 비교

### 데이터 소스

- **법령 데이터**: `statutes_articles` 테이블
- **판례 데이터**: `precedent_chunks` 테이블 (이미 청킹됨)
- **임베딩 모델**: `jhgan/ko-sroberta-multitask` (768차원)

---

## 목표

### 핵심 원칙

**법령 벡터 인덱스와 판례 벡터 인덱스는 반드시 분리해야 합니다.**

- 법령과 판례는 검색 목적, 텍스트 길이, 의미적 분포가 완전히 다르기 때문
- 각각 최적화된 인덱스 구조와 검색 파라미터 적용 가능
- 독립적인 업데이트 및 관리 가능
- 검색 성능 및 정확도 향상

### 주요 목표

1. **pgvector 기반 벡터 검색 시스템 구현**
   - PostgreSQL에 직접 임베딩 저장
   - pgvector 확장 활용
   - 실시간 업데이트 지원
   - **법령과 판례 인덱스 분리 구현**

2. **FAISS 기반 벡터 검색 시스템 구현** (PostgreSQL 데이터용)
   - PostgreSQL에서 데이터 추출
   - FAISS 인덱스 생성
   - 기존 시스템과 통합
   - **법령과 판례 인덱스 분리 구현**

3. **성능 및 정확도 비교**
   - 검색 속도 비교
   - 메모리 사용량 비교
   - 검색 결과 정확도 비교
   - 운영 편의성 비교
   - **법령/판례별 성능 비교**

4. **선택 기준 제시**
   - 데이터 규모별 권장사항
   - 사용 사례별 권장사항
   - 하이브리드 접근 방법 제안
   - **법령/판례별 최적화 전략**

---

## 시스템 구조

### 전체 아키텍처

**중요**: 법령 벡터 인덱스와 판례 벡터 인덱스는 반드시 분리되어야 합니다.
- 법령과 판례는 검색 목적, 텍스트 길이, 의미적 분포가 완전히 다르기 때문
- 분리된 인덱스를 사용하여 각각 최적화된 검색 성능 확보

```
PostgreSQL Database
├── statutes_articles          # 법령 조문
└── statute_embeddings         # 법령 임베딩 테이블 (생성 예정)
    └── embedding_vector       # pgvector 컬럼 (VECTOR(768))

PostgreSQL Database
├── precedent_chunks           # 판례 청크 (청킹 완료)
    └── embedding_vector       # pgvector 컬럼 (VECTOR(768))

                    ↓ 임베딩 생성 (법령/판례 분리)
                    
┌─────────────────────────────────────────────────┐
│         Embedding Generation Layer              │
│  - Base Embedder (SentenceTransformer)          │
│  - Data Loader (PostgreSQL → Text)              │
│  - 법령/판례 분리 처리                          │
└─────────────────────────────────────────────────┘
                    ↓
        ┌───────────┴───────────┐
        ↓                       ↓
┌───────────────┐      ┌───────────────┐
│   pgvector    │      │     FAISS     │
│   System      │      │    System     │
├───────────────┤      ├───────────────┤
│ - Embedder    │      │ - Embedder    │
│ - Indexer     │      │ - Indexer     │
│ - Searcher    │      │ - Searcher    │
│               │      │               │
│ 법령/판례     │      │ 법령/판례     │
│ 인덱스 분리   │      │ 인덱스 분리   │
└───────────────┘      └───────────────┘
        ↓                       ↓
┌─────────────────────────────────────────────────┐
│         Comparison & Benchmark Layer            │
│  - Performance Benchmark                       │
│  - Search Result Comparison                      │
│  - Report Generator                             │
│  - 법령/판례별 비교 리포트                      │
└─────────────────────────────────────────────────┘
```

### 데이터 흐름

1. **임베딩 생성 단계**
   ```
   PostgreSQL → Data Loader → SentenceTransformer → Embeddings
                                                          ↓
                                    ┌─────────────────────┴─────────────────────┐
                                    ↓                                           ↓
                            pgvector (DB 저장)                          FAISS (파일 저장)
   ```

2. **검색 단계**
   ```
   Query → Embedding → Search Engine → Results
                              ↓
                    ┌─────────┴─────────┐
                    ↓                   ↓
              pgvector Search      FAISS Search
                    ↓                   ↓
              PostgreSQL Query    FAISS Index Search
                    ↓                   ↓
              Results + Metadata  Results + Metadata (DB 조회)
   ```

---

## 구현 단계

### Phase 1: 공통 인프라 구축 (우선순위: 높음)

#### 1.1 공통 데이터 로더
**파일**: `scripts/ingest/open_law/embedding/data_loader.py`

**기능**:
- PostgreSQL에서 법령 조문 로드 (`statutes_articles`)
- PostgreSQL에서 판례 청크 로드 (`precedent_chunks`)
- 메타데이터 포함
- 필터링 지원 (도메인, 날짜 등)
- 배치 로딩 지원

**인터페이스**:
```python
class PostgreSQLDataLoader:
    def load_statute_articles(
        self, 
        domain: Optional[str] = None,
        limit: Optional[int] = None,
        offset: int = 0
    ) -> List[Dict[str, Any]]
    
    def load_precedent_chunks(
        self,
        domain: Optional[str] = None,
        section_type: Optional[str] = None,
        limit: Optional[int] = None,
        offset: int = 0
    ) -> List[Dict[str, Any]]
```

#### 1.2 공통 임베딩 생성기
**파일**: `scripts/ingest/open_law/embedding/base_embedder.py`

**기능**:
- SentenceTransformer 모델 로드
- 배치 임베딩 생성
- 진행 상황 모니터링
- 에러 처리 및 재시도

**인터페이스**:
```python
class BaseEmbedder:
    def __init__(self, model_name: str = "jhgan/ko-sroberta-multitask")
    def encode(
        self, 
        texts: List[str], 
        batch_size: int = 100,
        show_progress: bool = True
    ) -> np.ndarray
```

**예상 작업 시간**: 4-6시간

---

### Phase 2: pgvector 구현 (우선순위: 높음)

#### 2.1 pgvector 임베딩 생성
**파일**: `scripts/ingest/open_law/embedding/pgvector/pgvector_embedder.py`

**기능**:
- PostgreSQL `precedent_chunks` 테이블에 임베딩 저장
- `statute_embeddings` 테이블 생성 및 임베딩 저장
- 배치 처리 (100개씩)
- 중복 방지 (이미 임베딩된 데이터 건너뛰기)
- 진행 상황 로깅
- 트랜잭션 관리

**주요 메서드**:
```python
class PgVectorEmbedder:
    def generate_precedent_embeddings(
        self,
        batch_size: int = 100,
        limit: Optional[int] = None
    ) -> Dict[str, Any]
    
    def generate_statute_embeddings(
        self,
        batch_size: int = 100,
        limit: Optional[int] = None
    ) -> Dict[str, Any]
    
    def _save_embeddings(
        self,
        chunk_id: int,
        embedding: np.ndarray,
        table_name: str
    ) -> bool
```

**실행 방법**:
```bash
python scripts/ingest/open_law/embedding/pgvector/pgvector_embedder.py \
    --db $DATABASE_URL \
    --data-type precedents \
    --batch-size 100
```

#### 2.2 pgvector 인덱스 생성
**파일**: `scripts/ingest/open_law/embedding/pgvector/pgvector_indexer.py`

**기능**:
- ivfflat 인덱스 생성 (기본)
- HNSW 인덱스 생성 (선택, 성능 테스트용)
- 인덱스 파라미터 튜닝
- 인덱스 통계 수집

**인덱스 타입**:
- **ivfflat**: 빠른 검색, 적은 메모리
- **hnsw**: 매우 빠른 검색, 더 많은 메모리

**SQL 예시** (법령/판례 분리):
```sql
-- 법령 ivfflat 인덱스
CREATE INDEX idx_statute_embeddings_vector_ivfflat 
ON statute_embeddings 
USING ivfflat (embedding_vector vector_cosine_ops)
WITH (lists = 100);

-- 판례 ivfflat 인덱스
CREATE INDEX idx_precedent_chunks_vector_ivfflat 
ON precedent_chunks 
USING ivfflat (embedding_vector vector_cosine_ops)
WITH (lists = 100);

-- 법령 HNSW 인덱스 (선택)
CREATE INDEX idx_statute_embeddings_vector_hnsw 
ON statute_embeddings 
USING hnsw (embedding_vector vector_cosine_ops)
WITH (m = 16, ef_construction = 64);

-- 판례 HNSW 인덱스 (선택)
CREATE INDEX idx_precedent_chunks_vector_hnsw 
ON precedent_chunks 
USING hnsw (embedding_vector vector_cosine_ops)
WITH (m = 16, ef_construction = 64);
```

#### 2.3 pgvector 검색 엔진
**파일**: `scripts/ingest/open_law/embedding/pgvector/pgvector_search.py`

**기능**:
- 벡터 유사도 검색 (`<->` 연산자)
- 메타데이터 필터링과 결합
- 하이브리드 검색 (FTS + Vector)
- 결과 정렬 및 스코어링

**검색 쿼리 예시** (법령/판례 분리):

**법령 검색**:
```sql
SELECT 
    se.id,
    sa.article_content,
    se.embedding_vector <-> query_vector AS distance,
    s.law_name_kr,
    s.domain
FROM statute_embeddings se
JOIN statutes_articles sa ON se.article_id = sa.id
JOIN statutes s ON sa.statute_id = s.id
WHERE s.domain = 'civil_law'
  AND se.embedding_vector <-> query_vector < 0.5
ORDER BY se.embedding_vector <-> query_vector
LIMIT 10;
```

**판례 검색**:
```sql
SELECT 
    pc.id,
    pc.chunk_content,
    pc.embedding_vector <-> query_vector AS distance,
    p.case_name,
    p.decision_date
FROM precedent_chunks pc
JOIN precedent_contents pcon ON pc.precedent_content_id = pcon.id
JOIN precedents p ON pcon.precedent_id = p.id
WHERE p.domain = 'civil_law'
  AND pc.embedding_vector <-> query_vector < 0.5
ORDER BY pc.embedding_vector <-> query_vector
LIMIT 10;
```

**인터페이스**:
```python
class PgVectorSearcher:
    def __init__(
        self,
        db_url: str,
        data_type: str  # 'statutes' or 'precedents'
    )
    
    def search(
        self,
        query: str,
        top_k: int = 10,
        domain: Optional[str] = None,
        section_type: Optional[str] = None,  # 판례 전용
        similarity_threshold: float = 0.5
    ) -> List[Dict[str, Any]]
    
    def hybrid_search(
        self,
        query: str,
        top_k: int = 10,
        vector_weight: float = 0.7,
        fts_weight: float = 0.3
    ) -> List[Dict[str, Any]]
```

**예상 작업 시간**: 6-8시간

---

### Phase 3: FAISS 구현 (PostgreSQL 데이터용) (우선순위: 높음)

#### 3.1 FAISS 임베딩 생성
**파일**: `scripts/ingest/open_law/embedding/faiss/faiss_embedder.py`

**기능**:
- PostgreSQL에서 데이터 읽기
- 임베딩 생성
- FAISS 인덱스에 추가
- 메타데이터 JSON 저장
- chunk_id 매핑 저장

**주요 메서드**:
```python
class FaissEmbedder:
    def generate_embeddings(
        self,
        data_type: str,  # 'precedents' or 'statutes'
        batch_size: int = 100,
        limit: Optional[int] = None
    ) -> Dict[str, Any]
    
    def _add_to_index(
        self,
        embeddings: np.ndarray,
        chunk_ids: List[int],
        metadata: List[Dict[str, Any]]
    ) -> bool
```

**출력 파일** (법령/판례 분리):
```
data/embeddings/open_law_postgresql/
├── statutes/
│   ├── statutes_faiss_index.faiss      # 법령 FAISS 인덱스
│   ├── statutes_chunk_ids.json          # 법령 chunk_id 매핑
│   ├── statutes_metadata.json           # 법령 메타데이터
│   └── statutes_stats.json             # 법령 통계 정보
└── precedents/
    ├── precedents_faiss_index.faiss    # 판례 FAISS 인덱스
    ├── precedents_chunk_ids.json        # 판례 chunk_id 매핑
    ├── precedents_metadata.json          # 판례 메타데이터
    └── precedents_stats.json            # 판례 통계 정보
```

**주의**: 법령과 판례는 각각 별도의 인덱스로 생성되며, 같은 디렉토리에 혼합되지 않습니다.

#### 3.2 FAISS 인덱스 생성
**파일**: `scripts/ingest/open_law/embedding/faiss/faiss_indexer.py`

**기능**:
- IndexIVFFlat 인덱스 생성 (기본)
- IndexIVFPQ 인덱스 생성 (선택, 메모리 최적화)
- 인덱스 파라미터 최적화
- 인덱스 저장 및 버전 관리

**인덱스 타입**:
- **IndexIVFFlat**: 빠른 검색, 정확도 높음
- **IndexIVFPQ**: 메모리 효율적, 약간의 정확도 손실

**인터페이스**:
```python
class FaissIndexer:
    def build_index(
        self,
        embeddings: np.ndarray,
        index_type: str = "ivfflat",  # "ivfflat" or "ivfpq"
        nlist: Optional[int] = None
    ) -> faiss.Index
    
    def save_index(
        self,
        index: faiss.Index,
        output_path: Path,
        chunk_ids: List[int],
        metadata: List[Dict[str, Any]]
    ) -> bool
```

#### 3.3 FAISS 검색 엔진
**파일**: `scripts/ingest/open_law/embedding/faiss/faiss_search.py`

**기능**:
- FAISS 인덱스 로드
- 벡터 유사도 검색
- 메타데이터 필터링 (PostgreSQL 조회)
- 결과 정렬 및 스코어링

**인터페이스**:
```python
class FaissSearcher:
    def __init__(
        self,
        index_path: Path,
        db_url: str,
        data_type: str  # 'statutes' or 'precedents'
    )
    
    def search(
        self,
        query: str,
        top_k: int = 10,
        domain: Optional[str] = None,
        similarity_threshold: float = 0.5
    ) -> List[Dict[str, Any]]
    
    def search_by_vector(
        self,
        query_vector: np.ndarray,
        top_k: int = 10
    ) -> List[Dict[str, Any]]
```

**예상 작업 시간**: 4-6시간

---

### Phase 4: 비교 테스트 시스템 (우선순위: 중간)

#### 4.1 성능 벤치마크
**파일**: `scripts/ingest/open_law/embedding/comparison/benchmark.py`

**측정 항목**:
- 임베딩 생성 시간
- 인덱스 빌드 시간
- 검색 속도 (평균, P50, P95, P99)
- 메모리 사용량
- 디스크 사용량
- 동시 검색 처리량 (QPS)

**벤치마크 쿼리**:
```python
TEST_QUERIES = [
    "계약 해지 사유",
    "손해배상 청구 요건",
    "이혼 재산분할",
    "교통사고 과실",
    "상속 분쟁",
    "형사 처벌 요건",
    "계약 위반 손해배상",
    "부동산 매매 계약",
    "근로계약 해지",
    "지적재산권 침해"
]
```

**인터페이스**:
```python
class EmbeddingBenchmark:
    def benchmark_embedding_generation(
        self,
        data_type: str,
        sample_size: int = 1000
    ) -> Dict[str, Any]
    
    def benchmark_index_building(
        self,
        data_type: str
    ) -> Dict[str, Any]
    
    def benchmark_search(
        self,
        queries: List[str],
        top_k: int = 10,
        iterations: int = 10
    ) -> Dict[str, Any]
    
    def run_full_benchmark(
        self
    ) -> Dict[str, Any]
```

#### 4.2 검색 결과 비교
**파일**: `scripts/ingest/open_law/embedding/comparison/search_comparison.py`

**비교 항목**:
- 검색 결과 일치도 (Top-K overlap)
- 검색 순위 차이 (Kendall's tau)
- 스코어 분포
- 정확도 (Ground truth 기반, 선택)
- 재현율 (선택)

**인터페이스**:
```python
class SearchComparison:
    def compare_results(
        self,
        query: str,
        pgvector_results: List[Dict],
        faiss_results: List[Dict],
        top_k: int = 10
    ) -> Dict[str, Any]
    
    def calculate_overlap(
        self,
        results1: List[Dict],
        results2: List[Dict],
        top_k: int = 10
    ) -> float
    
    def calculate_rank_correlation(
        self,
        results1: List[Dict],
        results2: List[Dict]
    ) -> float
    
    def compare_all_queries(
        self,
        queries: List[str]
    ) -> Dict[str, Any]
```

#### 4.3 리포트 생성
**파일**: `scripts/ingest/open_law/embedding/comparison/report_generator.py`

**생성 리포트**:
- 성능 비교 리포트 (HTML/JSON)
- 검색 결과 비교 리포트
- 시각화 (그래프, 차트)
- 권장사항

**인터페이스**:
```python
class ReportGenerator:
    def generate_performance_report(
        self,
        benchmark_results: Dict[str, Any],
        output_path: Path
    ) -> bool
    
    def generate_comparison_report(
        self,
        comparison_results: Dict[str, Any],
        output_path: Path
    ) -> bool
    
    def generate_summary_report(
        self,
        all_results: Dict[str, Any],
        output_path: Path
    ) -> bool
```

**예상 작업 시간**: 6-8시간

---

### Phase 5: 통합 스크립트 (우선순위: 낮음)

#### 5.1 통합 임베딩 생성
**파일**: `scripts/ingest/open_law/embedding/generate_embeddings.py`

**기능**:
- `--method` 옵션 (pgvector, faiss, both)
- 두 시스템 동시 생성
- 진행 상황 모니터링
- 에러 처리 및 재시도

**사용 예시** (법령/판례 분리 처리):
```bash
# 법령 임베딩 생성 (pgvector)
python scripts/ingest/open_law/embedding/generate_embeddings.py \
    --db $DATABASE_URL \
    --method pgvector \
    --data-type statutes

# 판례 임베딩 생성 (pgvector)
python scripts/ingest/open_law/embedding/generate_embeddings.py \
    --db $DATABASE_URL \
    --method pgvector \
    --data-type precedents

# 법령 임베딩 생성 (FAISS)
python scripts/ingest/open_law/embedding/generate_embeddings.py \
    --db $DATABASE_URL \
    --method faiss \
    --data-type statutes

# 판례 임베딩 생성 (FAISS)
python scripts/ingest/open_law/embedding/generate_embeddings.py \
    --db $DATABASE_URL \
    --method faiss \
    --data-type precedents

# 주의: --data-type both 옵션은 사용하지 않습니다.
# 법령과 판례는 각각 별도의 인덱스로 생성해야 합니다.
# 두 시스템 모두 생성하려면 각각 별도로 실행하세요.
```

#### 5.2 비교 테스트 실행
**파일**: `scripts/ingest/open_law/embedding/run_comparison.py`

**기능**:
- 벤치마크 실행
- 검색 결과 비교
- 리포트 생성
- 결과 저장

**사용 예시** (법령/판례 분리 비교):
```bash
# 법령 인덱스 비교 테스트
python scripts/ingest/open_law/embedding/run_comparison.py \
    --db $DATABASE_URL \
    --data-type statutes \
    --faiss-index data/embeddings/open_law_postgresql/statutes/statutes_faiss_index.faiss \
    --output-dir reports/comparison/statutes

# 판례 인덱스 비교 테스트
python scripts/ingest/open_law/embedding/run_comparison.py \
    --db $DATABASE_URL \
    --data-type precedents \
    --faiss-index data/embeddings/open_law_postgresql/precedents/precedents_faiss_index.faiss \
    --output-dir reports/comparison/precedents
```

**예상 작업 시간**: 2-4시간

---

## 파일 구조

```
scripts/ingest/open_law/embedding/
├── __init__.py
├── generate_embeddings.py          # 통합 임베딩 생성 스크립트
├── generate_statute_embeddings.py  # 법령 전용 임베딩 생성 스크립트
├── run_comparison.py                # 비교 테스트 실행 스크립트
│
├── data_loader.py                   # 공통 데이터 로더
├── base_embedder.py                 # 공통 임베딩 생성기
│
├── pgvector/
│   ├── __init__.py
│   ├── pgvector_embedder.py         # pgvector 임베딩 생성 (법령/판례 분리)
│   ├── pgvector_indexer.py          # pgvector 인덱스 생성 (법령/판례 분리)
│   └── pgvector_search.py           # pgvector 검색 엔진 (법령/판례 분리)
│
├── faiss/
│   ├── __init__.py
│   ├── faiss_embedder.py            # FAISS 임베딩 생성 (법령/판례 분리)
│   ├── faiss_indexer.py             # FAISS 인덱스 생성 (법령/판례 분리)
│   └── faiss_search.py              # FAISS 검색 엔진 (법령/판례 분리)
│
├── comparison/
│   ├── __init__.py
│   ├── benchmark.py                 # 성능 벤치마크 (법령/판례 분리)
│   ├── search_comparison.py         # 검색 결과 비교 (법령/판례 분리)
│   ├── report_generator.py          # 비교 리포트 생성 (법령/판례 분리)
│   └── test_queries.py              # 테스트 쿼리 세트
│
└── tests/
    ├── __init__.py
    ├── test_pgvector.py             # pgvector 테스트 (법령/판례 분리)
    ├── test_faiss.py                # FAISS 테스트 (법령/판례 분리)
    └── test_comparison.py           # 비교 테스트 (법령/판례 분리)
```

**출력 디렉토리 구조**:
```
data/embeddings/open_law_postgresql/
├── statutes/                        # 법령 인덱스 (분리)
│   ├── statutes_faiss_index.faiss
│   ├── statutes_chunk_ids.json
│   ├── statutes_metadata.json
│   └── statutes_stats.json
└── precedents/                      # 판례 인덱스 (분리)
    ├── precedents_faiss_index.faiss
    ├── precedents_chunk_ids.json
    ├── precedents_metadata.json
    └── precedents_stats.json
```

---

## API 설계

### 공통 인터페이스

#### Embedder 인터페이스
```python
class BaseEmbedder(ABC):
    @abstractmethod
    def generate_embeddings(
        self,
        data_type: str,
        batch_size: int = 100,
        limit: Optional[int] = None
    ) -> Dict[str, Any]:
        """임베딩 생성"""
        pass
```

#### Searcher 인터페이스
```python
class BaseSearcher(ABC):
    @abstractmethod
    def search(
        self,
        query: str,
        top_k: int = 10,
        **filters
    ) -> List[Dict[str, Any]]:
        """검색 실행"""
        pass
```

### pgvector API

```python
# 법령 임베딩 생성
statute_embedder = PgVectorEmbedder(db_url)
statute_results = statute_embedder.generate_statute_embeddings(batch_size=100)

# 판례 임베딩 생성
precedent_embedder = PgVectorEmbedder(db_url)
precedent_results = precedent_embedder.generate_precedent_embeddings(batch_size=100)

# 법령 인덱스 생성
statute_indexer = PgVectorIndexer(db_url)
statute_indexer.create_ivfflat_index('statute_embeddings', lists=100)

# 판례 인덱스 생성
precedent_indexer = PgVectorIndexer(db_url)
precedent_indexer.create_ivfflat_index('precedent_chunks', lists=100)

# 법령 검색
statute_searcher = PgVectorSearcher(db_url, data_type='statutes')
statute_results = statute_searcher.search(
    query="계약 해지 사유",
    top_k=10,
    domain="civil_law"
)

# 판례 검색
precedent_searcher = PgVectorSearcher(db_url, data_type='precedents')
precedent_results = precedent_searcher.search(
    query="계약 해지 사유",
    top_k=10,
    domain="civil_law"
)
```

### FAISS API

```python
# 법령 임베딩 생성
statute_embedder = FaissEmbedder(
    db_url, 
    output_path / 'statutes',
    model_name='jhgan/ko-sroberta-multitask'
)
statute_results = statute_embedder.generate_embeddings('statutes', batch_size=100)
statute_embedder.save_embeddings('statutes')

# 판례 임베딩 생성
precedent_embedder = FaissEmbedder(
    db_url,
    output_path / 'precedents',
    model_name='jhgan/ko-sroberta-multitask'
)
precedent_results = precedent_embedder.generate_embeddings('precedents', batch_size=100)
precedent_embedder.save_embeddings('precedents')

# 법령 인덱스 생성
statute_indexer = FaissIndexer()
statute_index = statute_indexer.build_index(statute_embeddings, index_type="ivfflat")
statute_indexer.save_index(
    statute_index,
    output_path / 'statutes',
    statute_chunk_ids,
    statute_metadata
)

# 판례 인덱스 생성
precedent_indexer = FaissIndexer()
precedent_index = precedent_indexer.build_index(precedent_embeddings, index_type="ivfflat")
precedent_indexer.save_index(
    precedent_index,
    output_path / 'precedents',
    precedent_chunk_ids,
    precedent_metadata
)

# 법령 검색
statute_searcher = FaissSearcher(
    index_path='data/embeddings/open_law_postgresql/statutes/statutes_faiss_index.faiss',
    db_url=db_url,
    data_type='statutes'
)
statute_results = statute_searcher.search(
    query="계약 해지 사유",
    top_k=10,
    domain="civil_law"
)

# 판례 검색
precedent_searcher = FaissSearcher(
    index_path='data/embeddings/open_law_postgresql/precedents/precedents_faiss_index.faiss',
    db_url=db_url,
    data_type='precedents'
)
precedent_results = precedent_searcher.search(
    query="계약 해지 사유",
    top_k=10,
    domain="civil_law"
)
```

---

## 테스트 계획

### 단위 테스트

#### pgvector 테스트
- 임베딩 생성 테스트
- 인덱스 생성 테스트
- 검색 기능 테스트
- 메타데이터 필터링 테스트

#### FAISS 테스트
- 임베딩 생성 테스트
- 인덱스 생성 테스트
- 검색 기능 테스트
- 메타데이터 조회 테스트

### 통합 테스트

#### 비교 테스트
- 동일 쿼리로 두 시스템 검색
- 결과 일치도 확인
- 성능 측정

### 성능 테스트

#### 벤치마크 시나리오
1. **소규모 데이터** (1,000개 청크)
2. **중규모 데이터** (10,000개 청크)
3. **대규모 데이터** (100,000개 청크)

#### 측정 항목
- 임베딩 생성 시간
- 인덱스 빌드 시간
- 검색 속도 (단일 쿼리, 배치 쿼리)
- 메모리 사용량
- 디스크 사용량

---

## 성능 측정 지표

### 1. 임베딩 생성 성능

| 지표 | pgvector | FAISS |
|------|----------|-------|
| 생성 속도 (청크/초) | 측정 | 측정 |
| 배치 처리 효율 | 측정 | 측정 |
| 메모리 사용량 | 측정 | 측정 |

### 2. 인덱스 빌드 성능

| 지표 | pgvector | FAISS |
|------|----------|-------|
| 빌드 시간 | 측정 | 측정 |
| 인덱스 크기 | 측정 | 측정 |
| 메모리 사용량 | 측정 | 측정 |

### 3. 검색 성능

| 지표 | pgvector | FAISS |
|------|----------|-------|
| 평균 검색 시간 | 측정 | 측정 |
| P50 검색 시간 | 측정 | 측정 |
| P95 검색 시간 | 측정 | 측정 |
| P99 검색 시간 | 측정 | 측정 |
| 동시 검색 처리량 (QPS) | 측정 | 측정 |

### 4. 검색 정확도

| 지표 | 측정 방법 |
|------|----------|
| Top-K Overlap | 두 시스템의 Top-K 결과 일치도 |
| 순위 상관관계 | Kendall's tau |
| 스코어 분포 | 스코어 히스토그램 비교 |

### 5. 운영 편의성

| 항목 | pgvector | FAISS |
|------|----------|-------|
| 실시간 업데이트 | ✅ 가능 | ❌ 인덱스 재빌드 필요 |
| 백업/복구 | ✅ DB 백업 포함 | ⚠️ 별도 파일 관리 |
| 버전 관리 | ✅ DB 마이그레이션 | ⚠️ 파일 기반 |
| 메타데이터 필터링 | ✅ SQL WHERE | ⚠️ 별도 조회 필요 |

---

## 예상 작업 시간

### Phase별 작업 시간

| Phase | 작업 내용 | 예상 시간 |
|-------|----------|----------|
| Phase 1 | 공통 인프라 구축 | 4-6시간 |
| Phase 2 | pgvector 구현 | 6-8시간 |
| Phase 3 | FAISS 구현 | 4-6시간 |
| Phase 4 | 비교 테스트 시스템 | 6-8시간 |
| Phase 5 | 통합 스크립트 | 2-4시간 |
| **총계** | | **22-32시간** |

### 일정 계획

- **1주차**: Phase 1-2 (공통 인프라 + pgvector)
- **2주차**: Phase 3-4 (FAISS + 비교 테스트)
- **3주차**: Phase 5 + 최적화 + 문서화

---

## 다음 단계

### 즉시 시작 가능한 작업

1. **Phase 1 시작**
   - `data_loader.py` 구현
   - `base_embedder.py` 구현

2. **환경 준비**
   - pgvector 확장 설치 확인
   - 테스트 데이터 준비
   - 개발 환경 설정

### 우선순위 결정

**옵션 A: pgvector 우선**
- PostgreSQL 통합이 목표인 경우
- 실시간 업데이트가 중요한 경우

**옵션 B: FAISS 우선**
- 기존 시스템과의 통합이 목표인 경우
- 검증된 시스템이 필요한 경우

**옵션 C: 병렬 개발**
- 두 시스템 동시 개발
- 빠른 비교 테스트 가능

---

## 참고 문서

- [판례 데이터 청킹 전략](./CHUNKING.md)
- [Open Law API 데이터 수집](./README.md)
- [벡터 임베딩 시스템 가이드](../../../02_data/embedding/embedding_guide.md)
- [FAISS 최적화 가이드](../../../04_models/performance/faiss_search_optimization_proposals.md)
