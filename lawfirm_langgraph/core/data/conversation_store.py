# -*- coding: utf-8 -*-
"""
대화 저장소
대화 맥락의 영구 저장 및 관리
"""

import logging
try:
    from lawfirm_langgraph.core.utils.logger import get_logger
except ImportError:
    from core.utils.logger import get_logger
import sqlite3
import json
from typing import Dict, List, Any, Optional
from datetime import datetime
from contextlib import contextmanager
from pathlib import Path

logger = get_logger(__name__)


class ConversationStore:
    """대화 저장소"""
    
    def __init__(self, db_path: str = "data/conversations.db"):
        """대화 저장소 초기화"""
        self.logger = get_logger(__name__)
        self.db_path = db_path
        
        # 연결 풀 초기화 (필수)
        try:
            from core.data.connection_pool import get_connection_pool
            self._connection_pool = get_connection_pool(self.db_path)
            self.logger.debug("Using connection pool for conversation database")
        except ImportError:
            try:
                from lawfirm_langgraph.core.data.connection_pool import get_connection_pool
                self._connection_pool = get_connection_pool(self.db_path)
                self.logger.debug("Using connection pool for conversation database")
            except ImportError:
                raise RuntimeError(
                    "Connection pool is required. "
                    "Please ensure connection_pool module is available. "
                    "Direct SQLite connections are not allowed per project rules."
                )
        if not self._connection_pool:
            raise RuntimeError(
                "Connection pool initialization failed. "
                "Direct SQLite connections are not allowed per project rules."
            )
        
        # 데이터베이스 초기화
        self._create_tables()
    
    def _create_tables(self):
        """테이블 생성"""
        try:
            with self.get_connection() as conn:
                cursor = conn.cursor()
                
                # 대화 세션 테이블
                cursor.execute("""
                CREATE TABLE IF NOT EXISTS conversation_sessions (
                    session_id TEXT PRIMARY KEY,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    last_updated TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    topic_stack TEXT,  -- JSON 형태로 저장
                    metadata TEXT      -- JSON 형태로 저장
                )
                """)
                
                # 대화 턴 테이블
                cursor.execute("""
                CREATE TABLE IF NOT EXISTS conversation_turns (
                    turn_id INTEGER PRIMARY KEY AUTOINCREMENT,
                    session_id TEXT NOT NULL,
                    user_query TEXT NOT NULL,
                    bot_response TEXT NOT NULL,
                    timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    question_type TEXT,
                    entities TEXT,     -- JSON 형태로 저장
                    FOREIGN KEY (session_id) REFERENCES conversation_sessions(session_id)
                )
                """)
                
                # 법률 엔티티 테이블
                cursor.execute("""
                CREATE TABLE IF NOT EXISTS legal_entities (
                    entity_id INTEGER PRIMARY KEY AUTOINCREMENT,
                    session_id TEXT NOT NULL,
                    entity_type TEXT NOT NULL,  -- laws, articles, precedents, legal_terms
                    entity_value TEXT NOT NULL,
                    first_mentioned TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    last_mentioned TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    mention_count INTEGER DEFAULT 1,
                    FOREIGN KEY (session_id) REFERENCES conversation_sessions(session_id)
                )
                """)
                
                # 사용자 프로필 테이블
                cursor.execute("""
                CREATE TABLE IF NOT EXISTS user_profiles (
                    user_id TEXT PRIMARY KEY,
                    expertise_level TEXT DEFAULT 'beginner',
                    preferred_detail_level TEXT DEFAULT 'medium',
                    preferred_language TEXT DEFAULT 'ko',
                    interest_areas TEXT,  -- JSON
                    device_info TEXT,     -- JSON
                    location_info TEXT,   -- JSON
                    preferences TEXT,     -- JSON
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    last_updated TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
                """)
                
                # 맥락적 메모리 테이블
                cursor.execute("""
                CREATE TABLE IF NOT EXISTS contextual_memories (
                    memory_id INTEGER PRIMARY KEY AUTOINCREMENT,
                    session_id TEXT,
                    user_id TEXT,
                    memory_type TEXT,  -- fact/preference/case_detail
                    memory_content TEXT,
                    content TEXT,      -- 별칭으로 추가
                    importance_score REAL DEFAULT 0.5,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    last_accessed TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    access_count INTEGER DEFAULT 0,
                    related_entities TEXT,  -- JSON
                    tags TEXT,         -- JSON 형태로 저장
                    confidence REAL DEFAULT 0.5,
                    FOREIGN KEY (session_id) REFERENCES conversation_sessions(session_id),
                    FOREIGN KEY (user_id) REFERENCES user_profiles(user_id)
                )
                """)
                
                # 품질 메트릭 테이블
                cursor.execute("""
                CREATE TABLE IF NOT EXISTS quality_metrics (
                    metric_id INTEGER PRIMARY KEY AUTOINCREMENT,
                    session_id TEXT,
                    turn_id INTEGER,
                    completeness_score REAL,
                    satisfaction_score REAL,
                    accuracy_score REAL,
                    response_time REAL,
                    issues_detected TEXT,  -- JSON
                    timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    FOREIGN KEY (session_id) REFERENCES conversation_sessions(session_id)
                )
                """)
                
                # 세션 메타데이터 확장 (컬럼이 이미 존재하는 경우 무시)
                try:
                    cursor.execute("ALTER TABLE conversation_sessions ADD COLUMN user_id TEXT")
                except sqlite3.OperationalError:
                    pass  # 컬럼이 이미 존재하는 경우 무시
                
                try:
                    cursor.execute("ALTER TABLE conversation_sessions ADD COLUMN device_info TEXT")
                except sqlite3.OperationalError:
                    pass
                
                try:
                    cursor.execute("ALTER TABLE conversation_sessions ADD COLUMN location_info TEXT")
                except sqlite3.OperationalError:
                    pass
                
                try:
                    cursor.execute("ALTER TABLE conversation_sessions ADD COLUMN session_type TEXT DEFAULT 'normal'")
                except sqlite3.OperationalError:
                    pass
                
                # contextual_memories 테이블에 누락된 컬럼들 추가
                try:
                    cursor.execute("ALTER TABLE contextual_memories ADD COLUMN content TEXT")
                except sqlite3.OperationalError:
                    pass
                
                try:
                    cursor.execute("ALTER TABLE contextual_memories ADD COLUMN tags TEXT")
                except sqlite3.OperationalError:
                    pass
                
                try:
                    cursor.execute("ALTER TABLE contextual_memories ADD COLUMN confidence REAL DEFAULT 0.5")
                except sqlite3.OperationalError:
                    pass
                
                # 인덱스 생성
                cursor.execute("CREATE INDEX IF NOT EXISTS idx_turns_session_id ON conversation_turns(session_id)")
                cursor.execute("CREATE INDEX IF NOT EXISTS idx_turns_timestamp ON conversation_turns(timestamp)")
                cursor.execute("CREATE INDEX IF NOT EXISTS idx_entities_session_id ON legal_entities(session_id)")
                cursor.execute("CREATE INDEX IF NOT EXISTS idx_entities_type ON legal_entities(entity_type)")
                cursor.execute("CREATE INDEX IF NOT EXISTS idx_sessions_user_id ON conversation_sessions(user_id)")
                cursor.execute("CREATE INDEX IF NOT EXISTS idx_memories_user_id ON contextual_memories(user_id)")
                cursor.execute("CREATE INDEX IF NOT EXISTS idx_memories_session_id ON contextual_memories(session_id)")
                cursor.execute("CREATE INDEX IF NOT EXISTS idx_quality_session_id ON quality_metrics(session_id)")
                
                conn.commit()
                self.logger.info("Conversation store tables created successfully")
                
        except Exception as e:
            self.logger.error(f"Error creating tables: {e}")
    
    @contextmanager
    def get_connection(self):
        """데이터베이스 연결 컨텍스트 매니저"""
        # 연결 풀 필수 사용
        if not self._connection_pool:
            raise RuntimeError("Connection pool is required")
        conn = self._connection_pool.get_connection()
        conn.row_factory = sqlite3.Row
        try:
            yield conn
        finally:
            # 연결 풀을 사용하므로 수동으로 닫지 않음
            pass
    
    def save_session(self, session_data: Dict[str, Any]) -> bool:
        """세션 저장"""
        try:
            with self.get_connection() as conn:
                cursor = conn.cursor()
                
                # 세션 정보 저장
                metadata = session_data.get("metadata", {})
                user_id = metadata.get("user_id") if metadata else None
                
                cursor.execute("""
                INSERT OR REPLACE INTO conversation_sessions 
                (session_id, created_at, last_updated, topic_stack, metadata, user_id)
                VALUES (?, ?, ?, ?, ?, ?)
                """, (
                    session_data["session_id"],
                    session_data["created_at"],
                    session_data["last_updated"],
                    json.dumps(session_data.get("topic_stack", [])),
                    json.dumps(metadata),
                    user_id
                ))
                
                # 기존 턴들 삭제
                cursor.execute("DELETE FROM conversation_turns WHERE session_id = ?", 
                              (session_data["session_id"],))
                
                # 턴들 저장
                for turn in session_data.get("turns", []):
                    cursor.execute("""
                    INSERT INTO conversation_turns 
                    (session_id, user_query, bot_response, timestamp, question_type, entities)
                    VALUES (?, ?, ?, ?, ?, ?)
                    """, (
                        session_data["session_id"],
                        turn["user_query"],
                        turn["bot_response"],
                        turn["timestamp"],
                        turn.get("question_type"),
                        json.dumps(turn.get("entities", {}))
                    ))
                
                # 엔티티들 저장
                cursor.execute("DELETE FROM legal_entities WHERE session_id = ?", 
                              (session_data["session_id"],))
                
                for entity_type, entity_list in session_data.get("entities", {}).items():
                    for entity_value in entity_list:
                        cursor.execute("""
                        INSERT INTO legal_entities 
                        (session_id, entity_type, entity_value, first_mentioned, last_mentioned, mention_count)
                        VALUES (?, ?, ?, ?, ?, ?)
                        """, (
                            session_data["session_id"],
                            entity_type,
                            entity_value,
                            datetime.now().isoformat(),
                            datetime.now().isoformat(),
                            1
                        ))
                
                conn.commit()
                self.logger.info(f"Session {session_data['session_id']} saved successfully")
                return True
                
        except Exception as e:
            self.logger.error(f"Error saving session: {e}")
            return False
    
    def load_session(self, session_id: str) -> Optional[Dict[str, Any]]:
        """세션 로드"""
        try:
            with self.get_connection() as conn:
                cursor = conn.cursor()
                
                # 세션 정보 조회
                cursor.execute("""
                SELECT * FROM conversation_sessions WHERE session_id = ?
                """, (session_id,))
                
                session_row = cursor.fetchone()
                if not session_row:
                    return None
                
                # 턴들 조회
                cursor.execute("""
                SELECT * FROM conversation_turns 
                WHERE session_id = ? 
                ORDER BY timestamp
                """, (session_id,))
                
                turns = []
                for turn_row in cursor.fetchall():
                    turns.append({
                        "user_query": turn_row["user_query"],
                        "bot_response": turn_row["bot_response"],
                        "timestamp": turn_row["timestamp"],
                        "question_type": turn_row["question_type"],
                        "entities": json.loads(turn_row["entities"]) if turn_row["entities"] else {}
                    })
                
                # 엔티티들 조회
                cursor.execute("""
                SELECT entity_type, entity_value FROM legal_entities 
                WHERE session_id = ?
                """, (session_id,))
                
                entities = {"laws": [], "articles": [], "precedents": [], "legal_terms": []}
                for entity_row in cursor.fetchall():
                    entity_type = entity_row["entity_type"]
                    entity_value = entity_row["entity_value"]
                    if entity_type in entities:
                        entities[entity_type].append(entity_value)
                
                return {
                    "session_id": session_row["session_id"],
                    "created_at": session_row["created_at"],
                    "last_updated": session_row["last_updated"],
                    "topic_stack": json.loads(session_row["topic_stack"]) if session_row["topic_stack"] else [],
                    "metadata": json.loads(session_row["metadata"]) if session_row["metadata"] else {},
                    "turns": turns,
                    "entities": entities
                }
                
        except Exception as e:
            self.logger.error(f"Error loading session: {e}")
            return None
    
    def add_turn(self, session_id: str, turn_data: Dict[str, Any]) -> bool:
        """턴 추가"""
        try:
            with self.get_connection() as conn:
                cursor = conn.cursor()
                
                # 턴 저장
                cursor.execute("""
                INSERT INTO conversation_turns 
                (session_id, user_query, bot_response, timestamp, question_type, entities)
                VALUES (?, ?, ?, ?, ?, ?)
                """, (
                    session_id,
                    turn_data["user_query"],
                    turn_data["bot_response"],
                    turn_data["timestamp"],
                    turn_data.get("question_type"),
                    json.dumps(turn_data.get("entities", {}))
                ))
                
                # 엔티티들 업데이트
                entities = turn_data.get("entities", {})
                for entity_type, entity_list in entities.items():
                    for entity_value in entity_list:
                        # 기존 엔티티 확인
                        cursor.execute("""
                        SELECT mention_count FROM legal_entities 
                        WHERE session_id = ? AND entity_type = ? AND entity_value = ?
                        """, (session_id, entity_type, entity_value))
                        
                        existing = cursor.fetchone()
                        if existing:
                            # 기존 엔티티 업데이트
                            cursor.execute("""
                            UPDATE legal_entities 
                            SET last_mentioned = ?, mention_count = mention_count + 1
                            WHERE session_id = ? AND entity_type = ? AND entity_value = ?
                            """, (datetime.now().isoformat(), session_id, entity_type, entity_value))
                        else:
                            # 새 엔티티 추가
                            cursor.execute("""
                            INSERT INTO legal_entities 
                            (session_id, entity_type, entity_value, first_mentioned, last_mentioned, mention_count)
                            VALUES (?, ?, ?, ?, ?, ?)
                            """, (
                                session_id, entity_type, entity_value,
                                datetime.now().isoformat(),
                                datetime.now().isoformat(),
                                1
                            ))
                
                # 세션 업데이트 시간 갱신
                cursor.execute("""
                UPDATE conversation_sessions 
                SET last_updated = ?
                WHERE session_id = ?
                """, (datetime.now().isoformat(), session_id))
                
                conn.commit()
                self.logger.info(f"Turn added to session {session_id}")
                return True
                
        except Exception as e:
            self.logger.error(f"Error adding turn: {e}")
            return False
    
    def get_session_list(self, limit: int = 100) -> List[Dict[str, Any]]:
        """세션 목록 조회"""
        try:
            with self.get_connection() as conn:
                cursor = conn.cursor()
                
                cursor.execute("""
                SELECT s.session_id, s.created_at, s.last_updated, 
                       COUNT(t.turn_id) as turn_count,
                       COUNT(e.entity_id) as entity_count
                FROM conversation_sessions s
                LEFT JOIN conversation_turns t ON s.session_id = t.session_id
                LEFT JOIN legal_entities e ON s.session_id = e.session_id
                GROUP BY s.session_id
                ORDER BY s.last_updated DESC
                LIMIT ?
                """, (limit,))
                
                sessions = []
                for row in cursor.fetchall():
                    sessions.append({
                        "session_id": row["session_id"],
                        "created_at": row["created_at"],
                        "last_updated": row["last_updated"],
                        "turn_count": row["turn_count"],
                        "entity_count": row["entity_count"]
                    })
                
                return sessions
                
        except Exception as e:
            self.logger.error(f"Error getting session list: {e}")
            return []
    
    def delete_session(self, session_id: str) -> bool:
        """세션 삭제"""
        try:
            with self.get_connection() as conn:
                cursor = conn.cursor()
                
                # 관련 데이터 삭제
                cursor.execute("DELETE FROM conversation_turns WHERE session_id = ?", (session_id,))
                cursor.execute("DELETE FROM legal_entities WHERE session_id = ?", (session_id,))
                cursor.execute("DELETE FROM conversation_sessions WHERE session_id = ?", (session_id,))
                
                conn.commit()
                self.logger.info(f"Session {session_id} deleted")
                return True
                
        except Exception as e:
            self.logger.error(f"Error deleting session: {e}")
            return False
    
    def cleanup_old_sessions(self, days: int = 30) -> int:
        """오래된 세션 정리"""
        try:
            with self.get_connection() as conn:
                cursor = conn.cursor()
                
                cutoff_date = datetime.now().replace(hour=0, minute=0, second=0, microsecond=0)
                cutoff_date = cutoff_date.replace(day=cutoff_date.day - days)
                
                # 삭제할 세션 ID 조회
                cursor.execute("""
                SELECT session_id FROM conversation_sessions 
                WHERE last_updated < ?
                """, (cutoff_date.isoformat(),))
                
                session_ids = [row["session_id"] for row in cursor.fetchall()]
                
                # 세션들 삭제
                deleted_count = 0
                for session_id in session_ids:
                    if self.delete_session(session_id):
                        deleted_count += 1
                
                self.logger.info(f"Cleaned up {deleted_count} old sessions")
                return deleted_count
                
        except Exception as e:
            self.logger.error(f"Error cleaning up old sessions: {e}")
            return 0
    
    def get_user_sessions(self, user_id: str, limit: int = 10) -> List[Dict[str, Any]]:
        """사용자별 세션 목록 조회"""
        try:
            with self.get_connection() as conn:
                cursor = conn.cursor()
                
                cursor.execute("""
                SELECT s.session_id, s.created_at, s.last_updated, 
                       COUNT(t.turn_id) as turn_count,
                       COUNT(e.entity_id) as entity_count
                FROM conversation_sessions s
                LEFT JOIN conversation_turns t ON s.session_id = t.session_id
                LEFT JOIN legal_entities e ON s.session_id = e.session_id
                WHERE s.user_id = ?
                GROUP BY s.session_id
                ORDER BY s.last_updated DESC
                LIMIT ?
                """, (user_id, limit))
                
                sessions = []
                for row in cursor.fetchall():
                    sessions.append({
                        "session_id": row["session_id"],
                        "created_at": row["created_at"],
                        "last_updated": row["last_updated"],
                        "turn_count": row["turn_count"],
                        "entity_count": row["entity_count"]
                    })
                
                return sessions
                
        except Exception as e:
            self.logger.error(f"Error getting user sessions: {e}")
            return []
    
    def search_sessions(self, query: str, filters: Dict[str, Any]) -> List[Dict[str, Any]]:
        """세션 검색 (키워드, 날짜, 엔티티 기반)"""
        try:
            with self.get_connection() as conn:
                cursor = conn.cursor()
                
                # 기본 쿼리 구성
                base_query = """
                SELECT DISTINCT s.session_id, s.created_at, s.last_updated, 
                       COUNT(t.turn_id) as turn_count,
                       COUNT(e.entity_id) as entity_count
                FROM conversation_sessions s
                LEFT JOIN conversation_turns t ON s.session_id = t.session_id
                LEFT JOIN legal_entities e ON s.session_id = e.session_id
                WHERE 1=1
                """
                
                params = []
                
                # 키워드 검색
                if query:
                    base_query += """
                    AND (t.user_query LIKE ? OR t.bot_response LIKE ? 
                         OR e.entity_value LIKE ?)
                    """
                    search_term = f"%{query}%"
                    params.extend([search_term, search_term, search_term])
                
                # 필터 적용
                if filters.get("user_id"):
                    base_query += " AND s.user_id = ?"
                    params.append(filters["user_id"])
                
                if filters.get("start_date"):
                    base_query += " AND s.created_at >= ?"
                    params.append(filters["start_date"])
                
                if filters.get("end_date"):
                    base_query += " AND s.created_at <= ?"
                    params.append(filters["end_date"])
                
                if filters.get("entity_type"):
                    base_query += " AND e.entity_type = ?"
                    params.append(filters["entity_type"])
                
                base_query += """
                GROUP BY s.session_id
                ORDER BY s.last_updated DESC
                LIMIT ?
                """
                params.append(filters.get("limit", 50))
                
                cursor.execute(base_query, params)
                
                sessions = []
                for row in cursor.fetchall():
                    sessions.append({
                        "session_id": row["session_id"],
                        "created_at": row["created_at"],
                        "last_updated": row["last_updated"],
                        "turn_count": row["turn_count"],
                        "entity_count": row["entity_count"]
                    })
                
                return sessions
                
        except Exception as e:
            self.logger.error(f"Error searching sessions: {e}")
            return []
    
    def backup_session(self, session_id: str, backup_path: str) -> bool:
        """세션 백업"""
        try:
            import shutil
            from pathlib import Path
            
            # 세션 데이터 조회
            session_data = self.load_session(session_id)
            if not session_data:
                return False
            
            # 백업 디렉토리 생성
            backup_dir = Path(backup_path)
            backup_dir.mkdir(parents=True, exist_ok=True)
            
            # 백업 파일 경로
            backup_file = backup_dir / f"{session_id}_backup.json"
            
            # 백업 데이터 저장
            with open(backup_file, 'w', encoding='utf-8') as f:
                json.dump(session_data, f, ensure_ascii=False, indent=2)
            
            self.logger.info(f"Session {session_id} backed up to {backup_file}")
            return True
            
        except Exception as e:
            self.logger.error(f"Error backing up session: {e}")
            return False
    
    def restore_session(self, backup_path: str) -> Optional[str]:
        """세션 복원"""
        try:
            from pathlib import Path
            
            backup_file = Path(backup_path)
            if not backup_file.exists():
                return None
            
            # 백업 데이터 로드
            with open(backup_file, 'r', encoding='utf-8') as f:
                session_data = json.load(f)
            
            # 세션 ID 생성 (기존과 충돌 방지)
            original_session_id = session_data["session_id"]
            restored_session_id = f"{original_session_id}_restored_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            session_data["session_id"] = restored_session_id
            
            # 세션 저장
            if self.save_session(session_data):
                self.logger.info(f"Session restored from {backup_path} as {restored_session_id}")
                return restored_session_id
            
            return None
            
        except Exception as e:
            self.logger.error(f"Error restoring session: {e}")
            return None
    
    def get_statistics(self) -> Dict[str, Any]:
        """통계 정보 조회"""
        try:
            with self.get_connection() as conn:
                cursor = conn.cursor()
                
                # 기본 통계
                cursor.execute("SELECT COUNT(*) as session_count FROM conversation_sessions")
                session_count = cursor.fetchone()["session_count"]
                
                cursor.execute("SELECT COUNT(*) as turn_count FROM conversation_turns")
                turn_count = cursor.fetchone()["turn_count"]
                
                cursor.execute("SELECT COUNT(*) as entity_count FROM legal_entities")
                entity_count = cursor.fetchone()["entity_count"]
                
                cursor.execute("SELECT COUNT(*) as user_count FROM user_profiles")
                user_count = cursor.fetchone()["user_count"]
                
                cursor.execute("SELECT COUNT(*) as memory_count FROM contextual_memories")
                memory_count = cursor.fetchone()["memory_count"]
                
                # 엔티티 타입별 통계
                cursor.execute("""
                SELECT entity_type, COUNT(*) as count 
                FROM legal_entities 
                GROUP BY entity_type
                """)
                
                entity_stats = {}
                for row in cursor.fetchall():
                    entity_stats[row["entity_type"]] = row["count"]
                
                # 질문 유형별 통계
                cursor.execute("""
                SELECT question_type, COUNT(*) as count 
                FROM conversation_turns 
                WHERE question_type IS NOT NULL
                GROUP BY question_type
                """)
                
                question_type_stats = {}
                for row in cursor.fetchall():
                    question_type_stats[row["question_type"]] = row["count"]
                
                # 사용자 전문성 수준별 통계
                cursor.execute("""
                SELECT expertise_level, COUNT(*) as count 
                FROM user_profiles 
                GROUP BY expertise_level
                """)
                
                expertise_stats = {}
                for row in cursor.fetchall():
                    expertise_stats[row["expertise_level"]] = row["count"]
                
                return {
                    "session_count": session_count,
                    "turn_count": turn_count,
                    "entity_count": entity_count,
                    "user_count": user_count,
                    "memory_count": memory_count,
                    "avg_turns_per_session": turn_count / session_count if session_count > 0 else 0,
                    "entity_stats": entity_stats,
                    "question_type_stats": question_type_stats,
                    "expertise_stats": expertise_stats
                }
                
        except Exception as e:
            self.logger.error(f"Error getting statistics: {e}")
            return {}


# 테스트 함수
def test_conversation_store():
    """대화 저장소 테스트"""
    store = ConversationStore("test_conversations.db")
    
    print("=== 대화 저장소 테스트 ===")
    
    # 테스트 세션 데이터
    test_session = {
        "session_id": "test_session_001",
        "created_at": datetime.now().isoformat(),
        "last_updated": datetime.now().isoformat(),
        "topic_stack": ["손해배상", "계약"],
        "metadata": {"user_id": "test_user"},
        "turns": [
            {
                "user_query": "손해배상 청구 방법을 알려주세요",
                "bot_response": "민법 제750조에 따른 손해배상 청구 방법을 설명드리겠습니다...",
                "timestamp": datetime.now().isoformat(),
                "question_type": "legal_advice",
                "entities": {"laws": ["민법"], "articles": ["제750조"]}
            }
        ],
        "entities": {
            "laws": ["민법"],
            "articles": ["제750조"],
            "precedents": [],
            "legal_terms": ["손해배상"]
        }
    }
    
    # 세션 저장
    print("1. 세션 저장 테스트")
    success = store.save_session(test_session)
    print(f"저장 결과: {success}")
    
    # 세션 로드
    print("\n2. 세션 로드 테스트")
    loaded_session = store.load_session("test_session_001")
    if loaded_session:
        print(f"로드된 세션: {loaded_session['session_id']}")
        print(f"턴 수: {len(loaded_session['turns'])}")
        print(f"엔티티 수: {sum(len(entities) for entities in loaded_session['entities'].values())}")
    
    # 턴 추가
    print("\n3. 턴 추가 테스트")
    new_turn = {
        "user_query": "계약 해지 절차는 어떻게 되나요?",
        "bot_response": "계약 해지 절차는 다음과 같습니다...",
        "timestamp": datetime.now().isoformat(),
        "question_type": "procedure_guide",
        "entities": {"legal_terms": ["계약", "해지"]}
    }
    
    success = store.add_turn("test_session_001", new_turn)
    print(f"턴 추가 결과: {success}")
    
    # 세션 목록 조회
    print("\n4. 세션 목록 조회 테스트")
    sessions = store.get_session_list()
    print(f"총 세션 수: {len(sessions)}")
    for session in sessions:
        print(f"- {session['session_id']}: {session['turn_count']}턴, {session['entity_count']}엔티티")
    
    # 통계 조회
    print("\n5. 통계 조회 테스트")
    stats = store.get_statistics()
    print(f"통계: {stats}")
    
    # 테스트 데이터 정리
    store.delete_session("test_session_001")
    print("\n테스트 데이터 정리 완료")


if __name__ == "__main__":
    test_conversation_store()
