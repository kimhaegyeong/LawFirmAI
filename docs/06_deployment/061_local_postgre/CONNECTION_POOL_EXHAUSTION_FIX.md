# 커넥션 풀 고갈 문제 분석 및 해결 방안

## 문제 개요

데이터베이스 커넥션 풀 고갈 오류가 자주 발생하고 있습니다:
- `Connection timeout after 30s: connection pool exhausted`
- `Connection held for 30.05s without queries (longer than expected)`

커넥션 풀 개수는 충분하지만, 커밋/롤백 기능이 정상적으로 동작하지 않아 커넥션 풀 반환이 제대로 이루어지지 않는 것으로 확인되었습니다.

## 원인 분석

### 1. 직접 `get_connection()` 호출 후 반환 누락

**문제점:**
- `get_connection()`을 직접 호출한 후 예외 발생 시 연결이 반환되지 않음
- `try-finally` 블록이 없거나 불완전함
- 중간에 `return` 문이 있어 연결 반환 코드가 실행되지 않음

**영향받는 파일:**
- `semantic_search_engine_v2.py`: 5개 위치
- `legal_data_connector_v2.py`: 12개 위치
- `answer_formatter.py`: 2개 위치

### 2. `conn.close()` 잘못된 사용

**문제점:**
- PostgreSQL 연결 풀에서는 `conn.close()`가 연결을 닫아버려 풀에 반환하지 않음
- `putconn()` 또는 `_safe_close_connection()`을 사용해야 함

**영향받는 파일:**
- `legal_data_connector_v2.py`: 모든 `conn.close()` 호출 (약 20개)
- `answer_formatter.py`: 2개 위치

### 3. 트랜잭션 상태 확인 없이 연결 반환

**문제점:**
- 커밋/롤백 실패 시 트랜잭션이 열린 상태로 남아있음
- 트랜잭션이 열린 상태로 연결이 풀에 반환되면 다음 사용 시 문제 발생

**해결:**
- `PostgreSQLConnection.ensure_transaction_closed()` 메서드 추가 완료
- `get_connection_context()`의 `finally` 블록에서 트랜잭션 상태 확인 후 반환

### 4. 예외 발생 시 연결 반환 누락

**문제점:**
- `try-except` 블록에서 예외 발생 시 `finally` 블록이 없어 연결이 반환되지 않음
- 중첩된 예외 처리에서 연결 반환이 누락됨

## 해결 방안

### 1. `get_connection_context()` 사용 (권장)

**가장 안전한 방법:**
```python
# ❌ 나쁜 예
conn = self._get_connection()
try:
    cursor = conn.cursor()
    cursor.execute("SELECT ...")
finally:
    self._safe_close_connection(conn)

# ✅ 좋은 예
with self._get_connection_context() as conn:
    cursor = conn.cursor()
    cursor.execute("SELECT ...")
    # 자동으로 commit/rollback 및 연결 반환
```

**장점:**
- 자동 커밋/롤백 처리
- 예외 발생 시에도 연결 반환 보장
- 트랜잭션 상태 확인 후 안전하게 반환

### 2. `conn.close()` 대신 `putconn()` 또는 `_safe_close_connection()` 사용

**PostgreSQL 연결 풀 사용 시:**
```python
# ❌ 나쁜 예
conn.close()  # 연결을 닫아버려 풀에 반환되지 않음

# ✅ 좋은 예
if hasattr(conn, 'conn'):
    self._db_adapter.connection_pool.putconn(conn.conn)
else:
    self._safe_close_connection(conn)
```

### 3. `try-finally` 블록 보장

**예외 발생 시에도 연결 반환:**
```python
conn = None
try:
    conn = self._get_connection()
    cursor = conn.cursor()
    cursor.execute("SELECT ...")
except Exception as e:
    logger.error(f"Error: {e}")
    raise
finally:
    if conn:
        self._safe_close_connection(conn)
```

### 4. 트랜잭션 상태 확인

**연결 반환 전 트랜잭션 상태 확인:**
```python
if conn_wrapper:
    try:
        if hasattr(conn_wrapper, 'ensure_transaction_closed'):
            conn_wrapper.ensure_transaction_closed()
    except Exception as tx_error:
        logger.debug(f"Transaction cleanup failed: {tx_error}")
    
    self._safe_return_connection(conn_wrapper)
```

## 수정 우선순위

### 높음 (즉시 수정 필요)
1. `legal_data_connector_v2.py`: 모든 `conn.close()` → `putconn()` 또는 컨텍스트 매니저 사용
2. `semantic_search_engine_v2.py`: 직접 호출 → 컨텍스트 매니저 사용
3. `answer_formatter.py`: 직접 호출 → 컨텍스트 매니저 사용

### 중간 (점진적 수정)
1. `keyword_search_engine.py`: SQLite 연결 풀 사용 (우선순위 낮음)
2. `feedback_system.py`: SQLite 연결 풀 사용 (우선순위 낮음)
3. `performance_optimizer.py`: SQLite 연결 풀 사용 (우선순위 낮음)

## 수정 진행 상황

- [x] `PostgreSQLConnection`에 트랜잭션 상태 확인 메서드 추가
- [x] `get_connection_context()`에서 commit 실패 시 rollback 보장
- [x] `finally` 블록에서 트랜잭션 상태 확인 후 안전하게 연결 반환
- [x] `legal_data_connector_v2.py` 수정 완료
  - 모든 직접 `get_connection()` 호출을 컨텍스트 매니저로 변경
  - 모든 `conn.close()` 호출 제거
- [x] `semantic_search_engine_v2.py` 주요 부분 수정 완료
  - 데이터베이스 상태 확인 부분 수정
  - 메타데이터 로드 부분 수정
- [x] `answer_formatter.py` 수정 완료
  - 모든 직접 `get_connection()` 호출을 컨텍스트 매니저로 변경
  - 모든 `conn.close()` 호출 제거

## 테스트 방법

1. **연결 풀 상태 모니터링:**
```python
pool_status = db_adapter.get_pool_status()
logger.info(f"Pool status: {pool_status}")
```

2. **연결 통계 확인:**
```python
db_adapter.log_connection_stats()
```

3. **타임아웃 테스트:**
- 동시 요청 다수 발생 시 연결 풀 고갈 여부 확인
- 각 요청이 완료 후 연결이 정상적으로 반환되는지 확인

## 참고 사항

- PostgreSQL 연결 풀은 `psycopg2.pool.ThreadedConnectionPool` 사용
- SQLite 연결 풀은 `ThreadLocalConnectionPool` 사용 (스레드별 독립 연결)
- 모든 PostgreSQL 연결은 반드시 `putconn()` 또는 `_safe_close_connection()`으로 반환
- 컨텍스트 매니저 사용을 최우선으로 권장

