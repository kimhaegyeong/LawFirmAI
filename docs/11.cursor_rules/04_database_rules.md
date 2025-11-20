# 데이터베이스 규칙

## 1. 연결 풀링 시스템 사용 규칙 (CRITICAL)

**SQLite 데이터베이스 연결은 반드시 연결 풀링 시스템을 사용해야 합니다.**

### 원칙
1. **연결 풀링 필수 사용**
   - 모든 SQLite 데이터베이스 연결은 `lawfirm_langgraph.core.data.connection_pool`의 `ThreadLocalConnectionPool` 사용
   - 매번 `sqlite3.connect()`를 직접 호출하는 방식 금지
   - Thread-local storage를 사용하여 스레드 안전성 보장

2. **연결 풀 초기화 패턴**
   ```python
   # ✅ 좋은 예: 연결 풀 사용
   from lawfirm_langgraph.core.data.connection_pool import get_connection_pool
   
   class DatabaseService:
       def __init__(self, db_path: str):
           self.db_path = db_path
           self._connection_pool = get_connection_pool(self.db_path)
       
       def _get_connection(self):
           """연결 풀에서 연결 가져오기"""
           return self._connection_pool.get_connection()
       
       def execute_query(self, query: str, params: tuple = ()):
           conn = self._get_connection()
           cursor = conn.cursor()
           cursor.execute(query, params)
           return cursor.fetchall()
   ```

3. **기존 코드 마이그레이션 패턴**
   ```python
   # ❌ 나쁜 예: 매번 새 연결 생성
   def get_data(self):
       conn = sqlite3.connect(self.db_path)
       cursor = conn.cursor()
       cursor.execute("SELECT * FROM table")
       result = cursor.fetchall()
       conn.close()
       return result
   
   # ✅ 좋은 예: 연결 풀 사용
   def __init__(self, db_path: str):
       self.db_path = db_path
       try:
           from lawfirm_langgraph.core.data.connection_pool import get_connection_pool
           self._connection_pool = get_connection_pool(self.db_path)
       except ImportError:
           self._connection_pool = None
   
   def get_data(self):
       if self._connection_pool:
           conn = self._connection_pool.get_connection()
       else:
           conn = sqlite3.connect(self.db_path)
       cursor = conn.cursor()
       cursor.execute("SELECT * FROM table")
       result = cursor.fetchall()
       if not self._connection_pool:
           conn.close()
       return result
   ```

4. **연결 닫기 처리**
   ```python
   # ✅ 좋은 예: 연결 풀 사용 시 close() 호출 생략 가능
   def process_data(self):
       conn = self._connection_pool.get_connection()
       cursor = conn.cursor()
       cursor.execute("SELECT ...")
       result = cursor.fetchall()
       # 연결 풀 사용 시 conn.close() 호출 불필요 (자동 재사용)
       return result
   
   # ✅ 좋은 예: 폴백 지원 (연결 풀 없을 때)
   def process_data(self):
       if self._connection_pool:
           conn = self._connection_pool.get_connection()
       else:
           conn = sqlite3.connect(self.db_path)
       
       try:
           cursor = conn.cursor()
           cursor.execute("SELECT ...")
           return cursor.fetchall()
       finally:
           if not self._connection_pool:
               conn.close()
   ```

5. **금지 사항**
   - `sqlite3.connect()` 직접 호출 금지 (연결 풀 사용 필수)
   - 연결 풀 없이 매번 새 연결 생성하는 방식 금지
   - 스레드 간 연결 공유 금지 (Thread-local storage 사용 필수)

6. **적용 대상**
   - `LegalDataConnectorV2`: `_get_connection()` 메서드
   - `SemanticSearchEngineV2`: `_get_connection()` 메서드
   - `PerformanceCache`: 모든 데이터베이스 접근 메서드
   - 기타 SQLite 데이터베이스를 사용하는 모든 클래스

7. **성능 이점**
   - 연결 생성 오버헤드 감소 (20-30% 성능 향상)
   - 스레드 안전성 보장
   - 메모리 사용량 최적화

## 2. 기본 데이터베이스 규칙
```python
import sqlite3
from contextlib import contextmanager
from typing import List, Dict, Any

class DatabaseManager:
    """데이터베이스 관리 클래스"""
    
    def __init__(self, db_path: str):
        self.db_path = db_path
        # 연결 풀 초기화
        try:
            from lawfirm_langgraph.core.data.connection_pool import get_connection_pool
            self._connection_pool = get_connection_pool(self.db_path)
        except ImportError:
            self._connection_pool = None
        self._create_tables()
    
    def _get_connection(self):
        """연결 풀에서 연결 가져오기"""
        if self._connection_pool:
            return self._connection_pool.get_connection()
        else:
            conn = sqlite3.connect(self.db_path)
            conn.row_factory = sqlite3.Row
            return conn
    
    @contextmanager
    def get_connection(self):
        """데이터베이스 연결 컨텍스트 매니저"""
        conn = self._get_connection()
        try:
            yield conn
        finally:
            if not self._connection_pool:
                conn.close()
    
    def execute_query(self, query: str, params: tuple = ()) -> List[Dict[str, Any]]:
        """쿼리 실행"""
        conn = self._get_connection()
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()
        cursor.execute(query, params)
        result = [dict(row) for row in cursor.fetchall()]
        if not self._connection_pool:
            conn.close()
        return result
```

