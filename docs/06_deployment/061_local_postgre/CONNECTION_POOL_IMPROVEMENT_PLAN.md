# 연결 풀 자동화 개선 계획

## 📋 현재 문제점

1. **연결 풀 고갈**: `connection pool exhausted` 오류 발생
2. **연결 반환 불완전**: 예외 상황에서 연결이 제대로 반환되지 않을 수 있음
3. **상태 모니터링 부족**: 연결 풀 상태를 실시간으로 확인하기 어려움
4. **타임아웃 미설정**: 무한 대기 가능성
5. **재시도 로직 부재**: 연결 실패 시 자동 재시도 없음

## 🎯 개선 목표

1. **Context Manager 완전 자동화**: 모든 연결이 자동으로 반환되도록 보장
2. **연결 풀 상태 모니터링**: 실시간 상태 확인 및 경고
3. **자동 재연결**: 연결 실패 시 자동 재시도
4. **타임아웃 설정**: 무한 대기 방지
5. **통계 및 로깅**: 연결 풀 사용 통계 수집

## 📝 개선 계획

### 1단계: Context Manager 강화

#### 1.1 안전한 연결 반환 보장
```python
@contextmanager
def get_connection_context(self, timeout: Optional[int] = None):
    """
    개선된 컨텍스트 매니저
    
    특징:
    - 항상 연결 반환 보장 (예외 발생 시에도)
    - 타임아웃 지원
    - 자동 재연결
    - 연결 상태 검증
    """
    conn = None
    conn_wrapper = None
    start_time = time.time()
    
    try:
        # 타임아웃이 설정된 경우 연결 대기 시간 제한
        conn_wrapper = self._get_connection_with_timeout(timeout)
        conn = conn_wrapper
        
        # 연결 상태 검증
        self._validate_connection(conn)
        
        yield conn
        
        # 정상 종료 시 commit
        if hasattr(conn, 'commit'):
            try:
                conn.commit()
            except Exception:
                pass
                
    except psycopg2.pool.PoolError as e:
        # 연결 풀 고갈 시 재시도
        logger.warning(f"Connection pool exhausted, retrying...")
        conn_wrapper = self._retry_get_connection(timeout)
        conn = conn_wrapper
        yield conn
        
    except Exception as e:
        # 예외 발생 시 rollback
        if conn and hasattr(conn, 'rollback'):
            try:
                conn.rollback()
            except Exception:
                pass
        raise
        
    finally:
        # 항상 연결 반환 (예외 발생 여부와 무관)
        if conn_wrapper:
            self._safe_return_connection(conn_wrapper)
        
        # 실행 시간 로깅
        elapsed = time.time() - start_time
        if elapsed > 1.0:  # 1초 이상 걸린 경우 경고
            logger.warning(f"Connection held for {elapsed:.2f}s (longer than expected)")
```

#### 1.2 연결 풀 상태 모니터링
```python
def get_pool_status(self) -> Dict[str, Any]:
    """연결 풀 상태 조회"""
    if not self.connection_pool:
        return {"status": "not_initialized"}
    
    try:
        # ThreadedConnectionPool의 내부 상태 접근
        # (psycopg2.pool.ThreadedConnectionPool는 직접적인 상태 API가 없으므로
        #  추적을 위해 래퍼 클래스 필요)
        return {
            "minconn": self.connection_pool.minconn,
            "maxconn": self.connection_pool.maxconn,
            "status": "active"
        }
    except Exception as e:
        logger.error(f"Failed to get pool status: {e}")
        return {"status": "error", "error": str(e)}
```

### 2단계: 연결 풀 래퍼 클래스 생성

#### 2.1 상태 추적 가능한 연결 풀
```python
class TrackedThreadedConnectionPool(ThreadedConnectionPool):
    """상태 추적이 가능한 연결 풀"""
    
    def __init__(self, minconn, maxconn, *args, **kwargs):
        super().__init__(minconn, maxconn, *args, **kwargs)
        self._active_connections = 0
        self._total_connections = 0
        self._failed_connections = 0
        self._lock = threading.Lock()
    
    def getconn(self, key=None):
        """연결 가져오기 (상태 추적)"""
        try:
            conn = super().getconn(key)
            with self._lock:
                self._active_connections += 1
                self._total_connections += 1
            return conn
        except Exception as e:
            with self._lock:
                self._failed_connections += 1
            raise
    
    def putconn(self, conn, key=None, close=False):
        """연결 반환 (상태 추적)"""
        try:
            super().putconn(conn, key, close)
            with self._lock:
                if not close:
                    self._active_connections = max(0, self._active_connections - 1)
        except Exception as e:
            logger.warning(f"Error returning connection: {e}")
    
    def get_stats(self) -> Dict[str, Any]:
        """연결 풀 통계"""
        with self._lock:
            return {
                "minconn": self.minconn,
                "maxconn": self.maxconn,
                "active_connections": self._active_connections,
                "available_connections": self.maxconn - self._active_connections,
                "total_connections": self._total_connections,
                "failed_connections": self._failed_connections,
                "utilization": self._active_connections / self.maxconn if self.maxconn > 0 else 0
            }
```

### 3단계: 자동 재연결 및 타임아웃

#### 3.1 타임아웃 지원 연결 가져오기
```python
def _get_connection_with_timeout(self, timeout: Optional[int] = None) -> DatabaseConnection:
    """타임아웃이 있는 연결 가져오기"""
    if timeout is None:
        timeout = int(os.getenv("DB_CONNECTION_TIMEOUT", "30"))  # 기본 30초
    
    start_time = time.time()
    retry_count = 0
    max_retries = 3
    
    while True:
        try:
            if self.db_type == 'postgresql':
                if not self.connection_pool:
                    raise RuntimeError("PostgreSQL connection pool not initialized")
                
                # 연결 풀 상태 확인
                stats = self.get_pool_status()
                if stats.get("available_connections", 0) <= 0:
                    logger.warning("Connection pool exhausted, waiting...")
                    time.sleep(0.1)  # 짧은 대기 후 재시도
                
                conn = self.connection_pool.getconn()
                return PostgreSQLConnection(conn)
            else:
                raise ValueError(f"Unsupported database type: {self.db_type}")
                
        except psycopg2.pool.PoolError as e:
            elapsed = time.time() - start_time
            if elapsed >= timeout:
                raise TimeoutError(f"Connection timeout after {timeout}s: {e}")
            
            retry_count += 1
            if retry_count >= max_retries:
                raise RuntimeError(f"Failed to get connection after {max_retries} retries: {e}")
            
            wait_time = min(0.5 * retry_count, 2.0)  # 지수 백오프 (최대 2초)
            logger.debug(f"Retrying connection ({retry_count}/{max_retries}) after {wait_time}s...")
            time.sleep(wait_time)
```

#### 3.2 안전한 연결 반환
```python
def _safe_return_connection(self, conn_wrapper: DatabaseConnection):
    """안전하게 연결 반환"""
    if not conn_wrapper:
        return
    
    if self.db_type == 'postgresql' and self.connection_pool:
        if hasattr(conn_wrapper, 'conn'):
            try:
                # 연결 상태 확인
                if hasattr(conn_wrapper, '_is_closed') and not conn_wrapper._is_closed():
                    # 연결이 유효한 경우에만 풀에 반환
                    self.connection_pool.putconn(conn_wrapper.conn)
                    logger.debug("Connection returned to pool successfully")
                else:
                    # 연결이 이미 닫혀있으면 풀에서 제거됨
                    logger.debug("Connection already closed, not returning to pool")
            except Exception as e:
                logger.warning(f"Error returning connection to pool: {e}")
                # 연결이 손상된 경우 닫기
                try:
                    if hasattr(conn_wrapper, 'conn'):
                        conn_wrapper.conn.close()
                except Exception:
                    pass
```

### 4단계: 연결 풀 상태 모니터링 및 경고

#### 4.1 주기적 상태 체크
```python
def _monitor_pool_health(self):
    """연결 풀 상태 모니터링 (백그라운드 스레드)"""
    while True:
        try:
            stats = self.get_pool_status()
            utilization = stats.get("utilization", 0)
            
            # 경고 임계값 (80% 이상 사용 시)
            if utilization > 0.8:
                logger.warning(
                    f"Connection pool utilization high: {utilization:.1%} "
                    f"({stats.get('active_connections')}/{stats.get('maxconn')})"
                )
            
            # 연결 풀 고갈 경고 (95% 이상)
            if utilization > 0.95:
                logger.error(
                    f"Connection pool nearly exhausted: {utilization:.1%} "
                    f"({stats.get('active_connections')}/{stats.get('maxconn')})"
                )
            
            time.sleep(10)  # 10초마다 체크
            
        except Exception as e:
            logger.error(f"Error monitoring pool health: {e}")
            time.sleep(30)  # 오류 시 30초 대기
```

### 5단계: 사용 가이드 및 모범 사례

#### 5.1 권장 사용 패턴
```python
# ✅ 권장: Context Manager 사용
with db_adapter.get_connection_context() as conn:
    cursor = conn.cursor()
    cursor.execute("SELECT ...")
    results = cursor.fetchall()
    # 자동으로 연결 반환됨

# ❌ 비권장: 직접 연결 가져오기
conn = db_adapter.get_connection()
try:
    cursor = conn.cursor()
    cursor.execute("SELECT ...")
finally:
    # 수동으로 반환해야 함 (잊어버리기 쉬움)
    pass
```

#### 5.2 타임아웃 설정
```python
# 환경 변수로 타임아웃 설정
DB_CONNECTION_TIMEOUT=30  # 30초

# 또는 코드에서 직접 설정
with db_adapter.get_connection_context(timeout=60) as conn:
    # 60초 타임아웃으로 연결 가져오기
    pass
```

## 🔧 구현 우선순위

1. **높음 (즉시 구현)**
   - Context Manager 개선 (안전한 연결 반환)
   - 타임아웃 지원
   - 연결 상태 검증

2. **중간 (단기)**
   - 연결 풀 상태 모니터링
   - 자동 재연결 로직
   - 통계 수집

3. **낮음 (중기)**
   - 백그라운드 모니터링 스레드
   - 상세한 메트릭 수집
   - 대시보드 연동

## 📊 예상 효과

1. **연결 풀 고갈 방지**: 타임아웃 및 재시도로 안정성 향상
2. **자동 리소스 관리**: Context Manager로 누수 방지
3. **문제 조기 발견**: 모니터링으로 경고 발생
4. **성능 최적화**: 통계 기반 튜닝 가능

## 🚀 다음 단계

1. Context Manager 개선 코드 구현
2. 테스트 코드 작성
3. 기존 코드 마이그레이션
4. 모니터링 도구 통합

