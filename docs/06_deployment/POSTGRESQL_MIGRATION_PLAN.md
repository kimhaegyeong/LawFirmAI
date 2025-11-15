# PostgreSQL 마이그레이션 계획

## 개요

현재 LawFirmAI는 SQLite를 사용하고 있으나, 개발 서버와 운영 서버에서는 PostgreSQL을 사용하도록 마이그레이션합니다.

## 목표

- **로컬 개발**: SQLite 유지 (간편함)
- **개발 서버**: PostgreSQL 사용
- **운영 서버**: PostgreSQL 사용

## 현재 상황

### SQLite 사용 위치

1. **API 서비스**
   - `api/services/session_service.py` - 세션 관리
   - SQLite 직접 사용 (`sqlite3` 모듈)

2. **법률 데이터 검색**
   - `lawfirm_langgraph/core/search/connectors/legal_data_connector.py` - 법률 데이터 연결
   - `lawfirm_langgraph/core/services/semantic_search_engine_v2.py` - 의미 검색
   - `lawfirm_langgraph/core/search/engines/` - 여러 검색 엔진들
   - SQLite FTS5 사용

3. **캐시 및 최적화**
   - `lawfirm_langgraph/core/agents/optimizers/performance_optimizer.py` - 성능 최적화 캐시

## 마이그레이션 전략

### Phase 1: 데이터베이스 추상화 레이어 구축

**목표**: SQLite와 PostgreSQL을 모두 지원하는 추상화 레이어 생성

**방법**:
- SQLAlchemy ORM 사용
- 데이터베이스 URL 기반 자동 감지
- 환경별 설정 분리

### Phase 2: API 서비스 마이그레이션

**목표**: API 서비스의 세션 관리를 SQLAlchemy로 전환

**대상**:
- `api/services/session_service.py`

### Phase 3: 법률 데이터 검색 마이그레이션

**목표**: 법률 데이터 검색을 PostgreSQL로 전환

**고려사항**:
- SQLite FTS5 → PostgreSQL Full-Text Search
- 성능 최적화
- 점진적 마이그레이션

### Phase 4: 캐시 및 최적화 마이그레이션

**목표**: 캐시 데이터베이스를 PostgreSQL로 전환

## 기술 스택

### 데이터베이스 추상화
- **SQLAlchemy**: ORM 및 데이터베이스 추상화
- **Alembic**: 마이그레이션 관리
- **psycopg2**: PostgreSQL 드라이버

### 환경별 설정
- **로컬**: SQLite (기본값)
- **개발**: PostgreSQL
- **운영**: PostgreSQL

## 데이터베이스 스키마

### 세션 관리 테이블

```sql
-- sessions 테이블
CREATE TABLE sessions (
    session_id VARCHAR(255) PRIMARY KEY,
    title TEXT,
    category TEXT,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    message_count INTEGER DEFAULT 0,
    user_id VARCHAR(255),
    ip_address VARCHAR(45),
    metadata JSONB
);

-- messages 테이블
CREATE TABLE messages (
    message_id VARCHAR(255) PRIMARY KEY,
    session_id VARCHAR(255) NOT NULL,
    role VARCHAR(50) NOT NULL,
    content TEXT NOT NULL,
    timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    metadata JSONB,
    FOREIGN KEY (session_id) REFERENCES sessions(session_id) ON DELETE CASCADE
);

-- 인덱스
CREATE INDEX idx_messages_session_id ON messages(session_id);
CREATE INDEX idx_sessions_updated_at ON sessions(updated_at);
CREATE INDEX idx_sessions_user_id ON sessions(user_id);
```

### 법률 데이터 테이블 (기존 SQLite 구조 유지)

PostgreSQL로 마이그레이션 시:
- FTS5 → PostgreSQL Full-Text Search
- 벡터 검색은 FAISS 유지 (별도 시스템)

## 환경별 설정

### 로컬 개발 (.env)

```env
# 로컬: SQLite 사용
DATABASE_URL=sqlite:///./data/lawfirm.db
DATABASE_TYPE=sqlite
```

### 개발 서버 (.env.development)

```env
# 개발: PostgreSQL 사용
DATABASE_URL=postgresql://lawfirmai:password@postgres:5432/lawfirmai_dev
DATABASE_TYPE=postgresql
POSTGRES_HOST=postgres
POSTGRES_PORT=5432
POSTGRES_DB=lawfirmai_dev
POSTGRES_USER=lawfirmai
POSTGRES_PASSWORD=password
```

### 운영 서버 (.env.production)

```env
# 운영: PostgreSQL 사용
DATABASE_URL=postgresql://lawfirmai:secure_password@postgres:5432/lawfirmai_prod
DATABASE_TYPE=postgresql
POSTGRES_HOST=postgres
POSTGRES_PORT=5432
POSTGRES_DB=lawfirmai_prod
POSTGRES_USER=lawfirmai
POSTGRES_PASSWORD=secure_password
```

## Docker Compose 구성

### 개발 환경

```yaml
services:
  postgres:
    image: postgres:15-alpine
    environment:
      POSTGRES_DB: lawfirmai_dev
      POSTGRES_USER: lawfirmai
      POSTGRES_PASSWORD: password
    volumes:
      - postgres_data:/var/lib/postgresql/data
    ports:
      - "5432:5432"
    healthcheck:
      test: ["CMD-SHELL", "pg_isready -U lawfirmai"]
      interval: 10s
      timeout: 5s
      retries: 5
```

### 운영 환경

```yaml
services:
  postgres:
    image: postgres:15-alpine
    environment:
      POSTGRES_DB: ${POSTGRES_DB}
      POSTGRES_USER: ${POSTGRES_USER}
      POSTGRES_PASSWORD: ${POSTGRES_PASSWORD}
    volumes:
      - postgres_data:/var/lib/postgresql/data
    restart: unless-stopped
    healthcheck:
      test: ["CMD-SHELL", "pg_isready -U ${POSTGRES_USER}"]
      interval: 10s
      timeout: 5s
      retries: 5
```

## 마이그레이션 절차

### 1. 데이터베이스 추상화 레이어 생성

**파일**: `api/database/connection.py`

```python
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.ext.declarative import declarative_base
import os

Base = declarative_base()

def get_database_url():
    """환경 변수에서 데이터베이스 URL 가져오기"""
    db_url = os.getenv("DATABASE_URL", "sqlite:///./data/lawfirm.db")
    return db_url

def get_engine():
    """데이터베이스 엔진 생성"""
    db_url = get_database_url()
    
    # SQLite 설정
    if db_url.startswith("sqlite"):
        engine = create_engine(
            db_url,
            connect_args={"check_same_thread": False},
            echo=False
        )
    # PostgreSQL 설정
    elif db_url.startswith("postgresql"):
        engine = create_engine(
            db_url,
            pool_size=10,
            max_overflow=20,
            pool_pre_ping=True,
            echo=False
        )
    else:
        raise ValueError(f"Unsupported database URL: {db_url}")
    
    return engine

def get_session():
    """데이터베이스 세션 생성"""
    engine = get_engine()
    SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
    return SessionLocal()
```

### 2. 모델 정의

**파일**: `api/database/models.py`

```python
from sqlalchemy import Column, String, Integer, DateTime, Text, ForeignKey, JSON
from sqlalchemy.orm import relationship
from sqlalchemy.sql import func
from api.database.connection import Base

class Session(Base):
    __tablename__ = "sessions"
    
    session_id = Column(String(255), primary_key=True)
    title = Column(Text)
    category = Column(Text)
    created_at = Column(DateTime, server_default=func.now())
    updated_at = Column(DateTime, server_default=func.now(), onupdate=func.now())
    message_count = Column(Integer, default=0)
    user_id = Column(String(255))
    ip_address = Column(String(45))
    metadata = Column(JSON)
    
    messages = relationship("Message", back_populates="session", cascade="all, delete-orphan")

class Message(Base):
    __tablename__ = "messages"
    
    message_id = Column(String(255), primary_key=True)
    session_id = Column(String(255), ForeignKey("sessions.session_id", ondelete="CASCADE"), nullable=False)
    role = Column(String(50), nullable=False)
    content = Column(Text, nullable=False)
    timestamp = Column(DateTime, server_default=func.now())
    metadata = Column(JSON)
    
    session = relationship("Session", back_populates="messages")
```

### 3. 마이그레이션 스크립트

**파일**: `scripts/database/migrate_to_postgresql.py`

```python
"""
SQLite에서 PostgreSQL로 데이터 마이그레이션
"""
import sqlite3
import psycopg2
from psycopg2.extras import execute_values
import os
from typing import List, Dict, Any

def migrate_sessions(sqlite_path: str, postgres_url: str):
    """세션 데이터 마이그레이션"""
    # SQLite 연결
    sqlite_conn = sqlite3.connect(sqlite_path)
    sqlite_conn.row_factory = sqlite3.Row
    sqlite_cursor = sqlite_conn.cursor()
    
    # PostgreSQL 연결
    postgres_conn = psycopg2.connect(postgres_url)
    postgres_cursor = postgres_conn.cursor()
    
    try:
        # 세션 데이터 읽기
        sqlite_cursor.execute("SELECT * FROM sessions")
        sessions = sqlite_cursor.fetchall()
        
        # PostgreSQL에 삽입
        for session in sessions:
            postgres_cursor.execute("""
                INSERT INTO sessions (session_id, title, category, created_at, updated_at, 
                                    message_count, user_id, ip_address, metadata)
                VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s)
                ON CONFLICT (session_id) DO UPDATE SET
                    title = EXCLUDED.title,
                    category = EXCLUDED.category,
                    updated_at = EXCLUDED.updated_at,
                    message_count = EXCLUDED.message_count,
                    user_id = EXCLUDED.user_id,
                    ip_address = EXCLUDED.ip_address,
                    metadata = EXCLUDED.metadata
            """, (
                session['session_id'],
                session.get('title'),
                session.get('category'),
                session.get('created_at'),
                session.get('updated_at'),
                session.get('message_count', 0),
                session.get('user_id'),
                session.get('ip_address'),
                session.get('metadata')
            ))
        
        # 메시지 데이터 읽기
        sqlite_cursor.execute("SELECT * FROM messages")
        messages = sqlite_cursor.fetchall()
        
        # PostgreSQL에 삽입
        for message in messages:
            postgres_cursor.execute("""
                INSERT INTO messages (message_id, session_id, role, content, timestamp, metadata)
                VALUES (%s, %s, %s, %s, %s, %s)
                ON CONFLICT (message_id) DO UPDATE SET
                    session_id = EXCLUDED.session_id,
                    role = EXCLUDED.role,
                    content = EXCLUDED.content,
                    timestamp = EXCLUDED.timestamp,
                    metadata = EXCLUDED.metadata
            """, (
                message['message_id'],
                message['session_id'],
                message['role'],
                message['content'],
                message.get('timestamp'),
                message.get('metadata')
            ))
        
        postgres_conn.commit()
        print(f"✅ Migrated {len(sessions)} sessions and {len(messages)} messages")
        
    except Exception as e:
        postgres_conn.rollback()
        print(f"❌ Migration failed: {e}")
        raise
    finally:
        sqlite_conn.close()
        postgres_conn.close()
```

## 단계별 마이그레이션 계획

### Step 1: 인프라 준비 (1주)

1. **Docker Compose에 PostgreSQL 추가**
   - 개발 환경 Docker Compose
   - 운영 환경 Docker Compose

2. **환경 변수 설정**
   - `.env.development` 생성
   - `.env.production` 생성

3. **의존성 추가**
   - `requirements.txt`에 SQLAlchemy, psycopg2 추가

### Step 2: 추상화 레이어 구축 (1주)

1. **데이터베이스 연결 모듈 생성**
   - `api/database/connection.py`
   - `api/database/models.py`

2. **기존 코드 리팩토링**
   - `api/services/session_service.py`를 SQLAlchemy로 전환

### Step 3: 마이그레이션 스크립트 작성 (3일)

1. **Alembic 설정**
   - Alembic 초기화
   - 마이그레이션 스크립트 작성

2. **데이터 마이그레이션 스크립트**
   - SQLite → PostgreSQL 데이터 이전

### Step 4: 테스트 및 검증 (1주)

1. **단위 테스트**
   - SQLite 환경 테스트
   - PostgreSQL 환경 테스트

2. **통합 테스트**
   - 개발 서버 배포 및 테스트

### Step 5: 운영 배포 (1주)

1. **운영 서버 배포**
   - 데이터 마이그레이션
   - 검증 및 모니터링

## 주의사항

### 1. SQLite FTS5 → PostgreSQL Full-Text Search

**차이점**:
- SQLite FTS5: `MATCH` 연산자 사용
- PostgreSQL: `to_tsvector`, `to_tsquery` 사용

**마이그레이션 전략**:
- 점진적 마이그레이션
- 검색 엔진별로 개별 마이그레이션
- 성능 비교 및 최적화

### 2. 데이터 타입 차이

**SQLite**:
- TEXT, INTEGER, REAL, BLOB
- JSON은 TEXT로 저장

**PostgreSQL**:
- VARCHAR, TEXT, INTEGER, BIGINT, NUMERIC, JSONB
- JSONB 사용 권장 (인덱싱 및 쿼리 성능)

### 3. 트랜잭션 및 동시성

**SQLite**:
- 파일 기반 락
- 동시성 제한

**PostgreSQL**:
- MVCC (Multi-Version Concurrency Control)
- 높은 동시성 지원

### 4. 백업 및 복구

**SQLite**:
- 파일 복사로 백업

**PostgreSQL**:
- `pg_dump` / `pg_restore` 사용
- WAL (Write-Ahead Logging) 기반 백업

## 비용 고려사항

### 개발 서버
- **로컬 PostgreSQL**: 무료
- **Docker 컨테이너**: 무료
- **클라우드 PostgreSQL**: 월 $10-50 (사용량에 따라)

### 운영 서버
- **AWS RDS PostgreSQL**: 월 $15-100 (인스턴스 타입에 따라)
- **자체 호스팅**: EC2 인스턴스 비용 + 관리 비용

## 다음 단계

1. **데이터베이스 추상화 레이어 구현**
2. **Docker Compose에 PostgreSQL 추가**
3. **마이그레이션 스크립트 작성**
4. **테스트 및 검증**
5. **운영 배포**

---

**참고**: 법률 데이터 검색의 SQLite FTS5는 별도로 마이그레이션 계획을 수립해야 합니다.

