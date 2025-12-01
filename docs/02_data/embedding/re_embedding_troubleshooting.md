# 재임베딩 문제 해결 가이드

## 개요

본 문서는 재임베딩 작업 중 발생한 문제와 해결 방법을 정리한 가이드입니다. 향후 유사한 문제 발생 시 참고 자료로 활용할 수 있습니다.

## 발생한 문제 목록

1. [get_unique_documents 함수 문제](#1-get_unique_documents-함수-문제)
2. [statute_article 청킹 실패](#2-statute_article-청킹-실패)
3. [UNIQUE 제약 오류](#3-unique-제약-오류)
4. [성능 문제](#4-성능-문제)
5. [건너뛴 문서 문제](#5-건너뛴-문서-문제)

## 1. get_unique_documents 함수 문제

### 문제 상황
- 재임베딩 시 모든 문서가 건너뛰어짐
- 이미 임베딩된 문서만 포함되어 새로 처리할 문서가 없음

### 원인 분석
**파일**: `scripts/migrations/re_embed_existing_data_optimized.py`

**문제 코드**:
```python
def get_unique_documents(conn, source_type=None):
    query = """
        SELECT DISTINCT source_type, source_id
        FROM text_chunks
        WHERE embedding_version_id = ?
    """
    # text_chunks 테이블에서만 조회 → 이미 임베딩된 문서만 포함
```

**문제점**:
- `text_chunks` 테이블에서만 문서를 가져옴
- 이미 임베딩된 문서만 포함되어 새로 처리할 문서가 없음
- 원본 테이블의 문서를 고려하지 않음

### 해결 방법

**수정된 코드**:
```python
def get_unique_documents(conn, source_type=None):
    documents = []
    
    # 원본 테이블에서 문서 조회
    if source_type is None or source_type == 'statute_article':
        cursor = conn.execute("SELECT id FROM statute_articles")
        for row in cursor.fetchall():
            documents.append({
                'source_type': 'statute_article',
                'source_id': row[0]
            })
    
    if source_type is None or source_type == 'case_paragraph':
        cursor = conn.execute("SELECT DISTINCT case_id FROM case_paragraphs")
        for row in cursor.fetchall():
            documents.append({
                'source_type': 'case_paragraph',
                'source_id': row[0]
            })
    
    # decision_paragraph, interpretation_paragraph도 동일하게 처리
    # ...
    
    return documents
```

### 효과
- 모든 원본 문서를 재임베딩 대상으로 포함
- 재임베딩이 정상적으로 진행됨

## 2. statute_article 청킹 실패

### 문제 상황
- `statute_article` 문서 2,104개가 모두 건너뛰어짐
- 청크가 생성되지 않음

### 원인 분석
**파일**: `scripts/utils/text_chunker.py`

**문제 코드**:
```python
def chunk_statute(sentences, min_chars=200, max_chars=1200, overlap_ratio=0.2):
    articles = split_statute_sentences_into_articles(sentences)
    
    # articles가 비어있으면 빈 리스트 반환
    if not articles:
        return []
    
    # ...
```

**문제점**:
1. `split_statute_sentences_into_articles` 함수가 "제 X조" 헤더를 찾음
2. 실제 `statute_article` 텍스트에는 헤더가 없음 (본문만 저장됨)
3. 결과적으로 article이 생성되지 않음 (0개 반환)
4. `chunk_statute`가 빈 리스트를 반환
5. `collect_chunks_for_batch`에서 `if not chunk_results: continue`로 건너뜀

### 해결 방법

**수정된 코드**:
```python
def chunk_statute(sentences, min_chars=200, max_chars=1200, overlap_ratio=0.2):
    chunks = []
    articles = split_statute_sentences_into_articles(sentences)
    
    # Fallback: article이 생성되지 않으면 원본 텍스트를 직접 청킹
    if not articles:
        full_text = "\n".join(sentences)
        if not full_text or not full_text.strip():
            return chunks
        
        # min_chars가 300이어도, 50자 이상이면 최소한 1개 청크는 생성
        text_stripped = full_text.strip()
        if len(text_stripped) >= 50:
            # max_chars를 초과하면 분할
            if len(full_text) > max_chars:
                start = 0
                i = 0
                while start < len(full_text):
                    end = min(len(full_text), start + max_chars)
                    seg = full_text[start:end]
                    chunks.append({
                        "level": "article",
                        "article_no": None,
                        "clause_no": None,
                        "item_no": None,
                        "chunk_index": i,
                        "text": seg,
                    })
                    if end >= len(full_text):
                        break
                    overlap = int(max_chars * overlap_ratio)
                    start = end - overlap
                    i += 1
            else:
                chunks.append({
                    "level": "article",
                    "article_no": None,
                    "clause_no": None,
                    "item_no": None,
                    "chunk_index": 0,
                    "text": full_text.strip(),
                })
        return chunks
    
    # 기존 로직 (articles가 있는 경우)
    # ...
```

### 효과
- 1,639개 `statute_article` 처리 완료
- 50자 이상인 모든 문서가 정상적으로 청킹됨

## 3. UNIQUE 제약 오류

### 문제 상황
- 재임베딩 중 `sqlite3.IntegrityError: UNIQUE constraint failed` 오류 발생
- `(source_type, source_id, chunk_index)` 제약 위반

### 원인 분석
**파일**: `scripts/migrations/re_embed_existing_data_optimized.py`

**문제 코드**:
```python
# embedding_version_id = 5인 것만 삭제
DELETE FROM text_chunks 
WHERE source_type = ? AND source_id = ? AND embedding_version_id = ?
```

**문제점**:
- 특정 버전의 청크만 삭제
- 다른 버전의 청크가 같은 `chunk_index`를 가지면 UNIQUE 제약 충돌
- 재임베딩은 완전 교체 방식이므로 모든 버전의 청크를 삭제해야 함

### 해결 방법

**수정된 코드**:
```python
# 모든 버전의 청크 삭제 (UNIQUE 제약 충돌 방지)
DELETE FROM text_chunks 
WHERE source_type = ? AND source_id = ?

# 삭제 후 즉시 커밋
conn.commit()
```

### 효과
- UNIQUE 제약 오류 완전 해결
- 재임베딩이 정상적으로 진행됨

## 4. 성능 문제

### 문제 상황
- 문서당 처리 시간: 37.91초
- 처리 속도: 약 95 문서/시간
- 예상 완료 시간: 약 15일

### 원인 분석
1. **N+1 쿼리 문제**: 삽입된 청크 ID를 각 청크마다 개별 쿼리로 조회
2. **배치 처리 부족**: 각 문서마다 개별 쿼리로 처리
3. **청킹 전략 재생성**: 각 배치마다 청킹 전략을 새로 생성
4. **데이터베이스 I/O 병목**: 각 문서마다 개별 커밋

### 해결 방법

**1. N+1 쿼리 문제 해결**
```python
# 이전: N번의 쿼리
for chunk in chunks:
    cursor.execute("INSERT INTO ...")
    chunk_id = cursor.execute("SELECT id FROM ...").fetchone()[0]

# 개선: lastrowid 사용
for chunk in chunks:
    cursor.execute("INSERT INTO ...")
    chunk_id = cursor.lastrowid
```

**2. 배치 필터링**
```python
def filter_existing_documents_batch(documents, version_id, chunking_strategy):
    # 배치 단위로 일괄 조회
    # ...
```

**3. 청킹 전략 재사용**
```python
# 전략을 한 번만 생성하고 재사용
strategy = ChunkingFactory.create_strategy(strategy_name=chunking_strategy)
for batch in batches:
    collect_chunks_for_batch(batch, strategy)
```

**4. 데이터베이스 커밋 최적화**
```python
# 여러 배치마다 커밋
if batch_num % commit_interval == 0:
    conn.commit()
```

### 효과
- 문서당 처리 시간: 37.91초 → 3.74초 (-90.1%)
- 처리 속도: 95 문서/시간 → 962.6 문서/시간 (+913.3%)
- 예상 완료 시간: 15일 → 14시간 (-96.1%)

## 5. 건너뛴 문서 문제

### 문제 상황
- 건너뛴 문서 465개 발생
- `statute_article` 문서가 처리되지 않음

### 원인 분석
1. **텍스트 없음**: 171개 (원본 테이블에 텍스트가 없음)
2. **텍스트 너무 짧음**: 294개 (50자 미만)

### 해결 방법

**분석 결과**:
- 모든 건너뛴 문서는 처리 불가능한 문서로 정상 동작
- 텍스트 없음: 원본 데이터 문제
- 텍스트 너무 짧음: 50자 미만은 처리하지 않는 것이 정상

**결론**:
- 건너뛴 문서는 정상적으로 건너뛰어진 문서
- 추가 조치 불필요

## 예방 가이드

### 1. 재임베딩 전 체크리스트
- [ ] 원본 테이블에 데이터가 있는지 확인
- [ ] 시스템 사양 확인 (CPU, 메모리)
- [ ] 디스크 공간 확인
- [ ] 데이터베이스 백업

### 2. 실행 중 모니터링
- [ ] 진행 상황 모니터링
- [ ] 성능 확인
- [ ] 메모리 사용량 확인
- [ ] 오류 로그 확인

### 3. 문제 발생 시 대응
1. **성능 저하**: 배치 크기 조정, 시스템 리소스 확인
2. **메모리 부족**: 배치 크기 감소, 가비지 컬렉션 확인
3. **데이터베이스 오류**: 백업 확인, 트랜잭션 롤백
4. **청킹 실패**: 원본 텍스트 확인, 청킹 전략 확인

## 관련 문서

- [완료 보고서](./re_embedding_complete_report.md): 전체 작업 결과
- [최적화 가이드](./re_embedding_optimization_guide.md): 성능 최적화 방법
- [건너뛴 문서 분석](./skipped_documents_analysis.md): 건너뛴 문서 상세 분석

