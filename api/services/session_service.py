"""
세션 관리 서비스
"""
import os
import uuid
import sqlite3
from datetime import datetime, timezone, timedelta
from typing import List, Optional, Dict, Any
from pathlib import Path
import logging

from api.config import api_config

logger = logging.getLogger(__name__)

# KST 시간대 (UTC+9)
KST = timezone(timedelta(hours=9))


def get_kst_now() -> datetime:
    """KST 기준 현재 시간 반환"""
    return datetime.now(KST)


class SessionService:
    """세션 관리 서비스"""
    
    def __init__(self):
        """초기화"""
        self.db_path = Path(api_config.database_url.replace("sqlite:///", ""))
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._init_database()
    
    def _init_database(self):
        """데이터베이스 초기화"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            # 세션 테이블 생성
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS sessions (
                    session_id TEXT PRIMARY KEY,
                    title TEXT,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    message_count INTEGER DEFAULT 0,
                    user_id TEXT,
                    ip_address TEXT
                )
            """)
            
            # 기존 테이블에 컬럼 추가 (마이그레이션)
            cursor.execute("PRAGMA table_info(sessions)")
            existing_columns = [row[1] for row in cursor.fetchall()]
            
            if "user_id" not in existing_columns:
                try:
                    cursor.execute("ALTER TABLE sessions ADD COLUMN user_id TEXT")
                    logger.info("Added user_id column to sessions table")
                except sqlite3.OperationalError as e:
                    logger.warning(f"Failed to add user_id column: {e}")
            
            if "ip_address" not in existing_columns:
                try:
                    cursor.execute("ALTER TABLE sessions ADD COLUMN ip_address TEXT")
                    logger.info("Added ip_address column to sessions table")
                except sqlite3.OperationalError as e:
                    logger.warning(f"Failed to add ip_address column: {e}")
            
            if "metadata" in existing_columns:
                try:
                    sqlite_version = sqlite3.sqlite_version_info
                    if sqlite_version >= (3, 35, 0):
                        cursor.execute("ALTER TABLE sessions DROP COLUMN metadata")
                        logger.info("Removed metadata column from sessions table")
                    else:
                        logger.info("SQLite version < 3.35.0, metadata column will be ignored (not removed)")
                except sqlite3.OperationalError as e:
                    logger.warning(f"Failed to remove metadata column: {e}")
            
            # 메시지 테이블 생성
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS messages (
                    message_id TEXT PRIMARY KEY,
                    session_id TEXT NOT NULL,
                    role TEXT NOT NULL,
                    content TEXT NOT NULL,
                    timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    metadata TEXT,
                    FOREIGN KEY (session_id) REFERENCES sessions(session_id)
                )
            """)
            
            # 인덱스 생성
            cursor.execute("""
                CREATE INDEX IF NOT EXISTS idx_messages_session_id 
                ON messages(session_id)
            """)
            
            cursor.execute("""
                CREATE INDEX IF NOT EXISTS idx_sessions_updated_at 
                ON sessions(updated_at)
            """)
            
            conn.commit()
            conn.close()
            logger.info("Database initialized successfully")
            self._set_database_permissions()
        except Exception as e:
            logger.error(f"Failed to initialize database: {e}")
    
    def _set_database_permissions(self):
        """데이터베이스 파일 권한 설정"""
        try:
            if self.db_path.exists():
                if os.name == 'posix':
                    os.chmod(self.db_path, 0o600)
                    logger.info(f"Database file permissions set to 600: {self.db_path}")
                elif os.name == 'nt':
                    import stat
                    os.chmod(self.db_path, stat.S_IREAD | stat.S_IWRITE)
                    logger.info(f"Database file permissions set (Windows): {self.db_path}")
        except Exception as e:
            logger.warning(f"Failed to set database permissions: {e}")
    
    def _ensure_columns_exist(self, conn: sqlite3.Connection):
        """필요한 컬럼이 존재하는지 확인하고 없으면 추가"""
        try:
            cursor = conn.cursor()
            cursor.execute("PRAGMA table_info(sessions)")
            existing_columns = [row[1] for row in cursor.fetchall()]
            
            if "user_id" not in existing_columns:
                try:
                    cursor.execute("ALTER TABLE sessions ADD COLUMN user_id TEXT")
                    logger.info("Added user_id column to sessions table")
                except sqlite3.OperationalError as e:
                    logger.warning(f"Failed to add user_id column: {e}")
            
            if "ip_address" not in existing_columns:
                try:
                    cursor.execute("ALTER TABLE sessions ADD COLUMN ip_address TEXT")
                    logger.info("Added ip_address column to sessions table")
                except sqlite3.OperationalError as e:
                    logger.warning(f"Failed to add ip_address column: {e}")
            
            if "category" in existing_columns:
                try:
                    sqlite_version = sqlite3.sqlite_version_info
                    if sqlite_version >= (3, 35, 0):
                        cursor.execute("ALTER TABLE sessions DROP COLUMN category")
                        logger.info("Removed category column from sessions table")
                    else:
                        logger.info("SQLite version < 3.35.0, category column will be ignored (not removed)")
                except sqlite3.OperationalError as e:
                    logger.warning(f"Failed to remove category column: {e}")
            
            conn.commit()
        except Exception as e:
            logger.error(f"Failed to ensure columns exist: {e}")
    
    def create_session(
        self,
        title: Optional[str] = None,
        user_id: Optional[str] = None,
        ip_address: Optional[str] = None
    ) -> str:
        """세션 생성"""
        session_id = str(uuid.uuid4())
        
        try:
            conn = sqlite3.connect(self.db_path)
            self._ensure_columns_exist(conn)
            cursor = conn.cursor()
            
            now_kst = get_kst_now()
            now_kst_str = now_kst.isoformat()
            cursor.execute("""
                INSERT INTO sessions (session_id, title, created_at, updated_at, user_id, ip_address)
                VALUES (?, ?, ?, ?, ?, ?)
            """, (session_id, title, now_kst_str, now_kst_str, user_id, ip_address))
            
            conn.commit()
            conn.close()
            logger.info(f"Session created: {session_id}")
            return session_id
        except Exception as e:
            logger.error(f"Failed to create session: {e}")
            raise
    
    def get_session(self, session_id: str, check_expiry: bool = True) -> Optional[Dict[str, Any]]:
        """세션 조회"""
        try:
            conn = sqlite3.connect(self.db_path)
            conn.row_factory = sqlite3.Row
            cursor = conn.cursor()
            
            cursor.execute("""
                SELECT * FROM sessions WHERE session_id = ?
            """, (session_id,))
            
            row = cursor.fetchone()
            
            if row:
                session_dict = dict(row)
                # datetime 객체를 ISO 형식 문자열로 변환
                if isinstance(session_dict.get("created_at"), datetime):
                    session_dict["created_at"] = session_dict["created_at"].isoformat()
                if isinstance(session_dict.get("updated_at"), datetime):
                    session_dict["updated_at"] = session_dict["updated_at"].isoformat()
                
                # 세션 만료 시간 확인
                if check_expiry:
                    updated_at = session_dict.get("updated_at")
                    if updated_at:
                        if isinstance(updated_at, str):
                            updated_at = datetime.fromisoformat(updated_at.replace('Z', '+00:00'))
                        if isinstance(updated_at, datetime):
                            expiry_hours = api_config.session_ttl_hours
                            expiry_time = updated_at + timedelta(hours=expiry_hours)
                            if get_kst_now() > expiry_time:
                                logger.warning(f"Session expired: {session_id}")
                                conn.close()
                                return None
                
                conn.close()
                return session_dict
            
            conn.close()
            return None
        except Exception as e:
            logger.error(f"Failed to get session: {e}")
            return None
    
    def update_session(
        self,
        session_id: str,
        title: Optional[str] = None
    ) -> bool:
        """세션 업데이트"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            updates = []
            params = []
            
            if title is not None:
                updates.append("title = ?")
                params.append(title)
            
            if not updates:
                conn.close()
                return False
            
            updates.append("updated_at = ?")
            params.append(get_kst_now().isoformat())
            params.append(session_id)
            
            query = "UPDATE sessions SET " + ", ".join(updates) + " WHERE session_id = ?"
            cursor.execute(query, params)
            
            conn.commit()
            conn.close()
            logger.info(f"Session updated: {session_id}")
            return True
        except Exception as e:
            logger.error(f"Failed to update session: {e}")
            return False
    
    def delete_session(self, session_id: str, user_id: Optional[str] = None) -> bool:
        """세션 삭제"""
        try:
            conn = sqlite3.connect(self.db_path)
            conn.row_factory = sqlite3.Row
            cursor = conn.cursor()
            
            # 세션 소유자 확인
            if user_id:
                cursor.execute("PRAGMA table_info(sessions)")
                existing_columns = [row[1] for row in cursor.fetchall()]
                if "user_id" in existing_columns:
                    cursor.execute("""
                        SELECT user_id FROM sessions WHERE session_id = ?
                    """, (session_id,))
                    row = cursor.fetchone()
                    if row:
                        session_user_id = row["user_id"]
                        if session_user_id and session_user_id != user_id:
                            conn.close()
                            logger.warning(f"Session ownership mismatch: {session_id}, user: {user_id}")
                            return False
            
            # 메시지도 함께 삭제
            cursor.execute("DELETE FROM messages WHERE session_id = ?", (session_id,))
            cursor.execute("DELETE FROM sessions WHERE session_id = ?", (session_id,))
            
            conn.commit()
            conn.close()
            logger.info(f"Session deleted: {session_id}")
            return True
        except Exception as e:
            logger.error(f"Failed to delete session: {e}")
            return False
    
    def delete_user_sessions(self, user_id: str) -> int:
        """사용자의 모든 세션 및 메시지 일괄 삭제"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            # 컬럼 존재 여부 확인
            cursor.execute("PRAGMA table_info(sessions)")
            existing_columns = [row[1] for row in cursor.fetchall()]
            has_user_id = "user_id" in existing_columns
            
            if has_user_id:
                # 사용자의 모든 세션 ID 조회
                cursor.execute("""
                    SELECT session_id FROM sessions WHERE user_id = ?
                """, (user_id,))
                session_rows = cursor.fetchall()
                session_ids = [row[0] for row in session_rows]
                
                if not session_ids:
                    conn.close()
                    logger.info(f"No sessions found for user: {user_id}")
                    return 0
                
                # 모든 메시지 삭제
                placeholders = ','.join(['?'] * len(session_ids))
                cursor.execute(f"""
                    DELETE FROM messages WHERE session_id IN ({placeholders})
                """, session_ids)
                
                # 모든 세션 삭제
                cursor.execute("""
                    DELETE FROM sessions WHERE user_id = ?
                """, (user_id,))
            else:
                conn.close()
                logger.warning("user_id column does not exist, cannot delete user sessions")
                return 0
            
            deleted_count = len(session_ids)
            
            conn.commit()
            conn.close()
            logger.info(f"Deleted {deleted_count} sessions for user: {user_id}")
            return deleted_count
        except Exception as e:
            logger.error(f"Failed to delete user sessions: {e}")
            return 0
    
    def list_sessions(
        self,
        search: Optional[str] = None,
        page: int = 1,
        page_size: int = 10,
        sort_by: str = "updated_at",
        sort_order: str = "desc",
        date_from: Optional[str] = None,
        date_to: Optional[str] = None,
        user_id: Optional[str] = None,
        ip_address: Optional[str] = None
    ) -> tuple[List[Dict[str, Any]], int]:
        """세션 목록 조회"""
        try:
            conn = sqlite3.connect(self.db_path)
            conn.row_factory = sqlite3.Row
            cursor = conn.cursor()
            
            # 컬럼 존재 여부 확인
            cursor.execute("PRAGMA table_info(sessions)")
            existing_columns = [row[1] for row in cursor.fetchall()]
            has_user_id = "user_id" in existing_columns
            has_ip_address = "ip_address" in existing_columns
            
            # WHERE 절 구성
            where_clauses = []
            params = []
            
            # 사용자별 필터링
            if user_id:
                # 익명 세션 ID는 anonymous_ prefix를 사용하므로 별도 처리
                if user_id.startswith("anonymous_"):
                    if has_user_id:
                        where_clauses.append("user_id = ?")
                        params.append(user_id)
                else:
                    if has_user_id:
                        where_clauses.append("user_id = ?")
                        params.append(user_id)
            elif ip_address:
                if has_ip_address:
                    if has_user_id:
                        where_clauses.append("ip_address = ? AND (user_id IS NULL OR user_id = '')")
                    else:
                        where_clauses.append("ip_address = ?")
                    params.append(ip_address)
            
            if search:
                where_clauses.append("(title LIKE ? OR session_id LIKE ?)")
                params.append(f"%{search}%")
                params.append(f"%{search}%")
            
            # 날짜 필터 추가 (KST 기준)
            # SQLite strftime() 함수를 사용하여 날짜 비교 (더 안정적)
            if date_from:
                try:
                    # 날짜 형식 검증
                    datetime.strptime(date_from, "%Y-%m-%d")
                    # strftime을 사용하여 날짜 부분만 추출하여 비교
                    where_clauses.append("strftime('%Y-%m-%d', updated_at) >= ?")
                    params.append(date_from)
                except ValueError:
                    logger.warning(f"Invalid date_from format: {date_from}, using strftime() function")
                    where_clauses.append("strftime('%Y-%m-%d', updated_at) >= ?")
                    params.append(date_from)
            
            if date_to:
                try:
                    # 날짜 형식 검증
                    datetime.strptime(date_to, "%Y-%m-%d")
                    # strftime을 사용하여 날짜 부분만 추출하여 비교
                    where_clauses.append("strftime('%Y-%m-%d', updated_at) <= ?")
                    params.append(date_to)
                except ValueError:
                    logger.warning(f"Invalid date_to format: {date_to}, using strftime() function")
                    where_clauses.append("strftime('%Y-%m-%d', updated_at) <= ?")
                    params.append(date_to)
            
            where_sql = " AND ".join(where_clauses) if where_clauses else "1=1"
            
            # 정렬 필드 검증 (SQL Injection 방지)
            valid_sort_fields = ["created_at", "updated_at", "title", "message_count"]
            sort_field = sort_by if sort_by in valid_sort_fields else "updated_at"
            sort_dir = "DESC" if sort_order.lower() == "desc" else "ASC"
            
            # 디버깅을 위한 로깅
            logger.debug(f"list_sessions query: WHERE {where_sql}, params: {params}, user_id: {user_id}, date_from: {date_from}, date_to: {date_to}")
            
            # 전체 개수 조회 (파라미터화된 쿼리 사용)
            count_query = "SELECT COUNT(*) FROM sessions WHERE " + where_sql
            cursor.execute(count_query, params)
            total = cursor.fetchone()[0]
            
            logger.debug(f"list_sessions total count: {total}")
            
            # 페이지네이션
            offset = (page - 1) * page_size
            # 파라미터화된 쿼리 사용 (정렬 필드는 화이트리스트로 검증됨)
            # 정렬 필드는 화이트리스트로 검증되었으므로 안전하게 사용 가능
            query = "SELECT * FROM sessions WHERE " + where_sql + " ORDER BY " + sort_field + " " + sort_dir + " LIMIT ? OFFSET ?"
            query_params = params + [page_size, offset]
            
            logger.debug(f"list_sessions full query: {query}, params: {query_params}")
            
            cursor.execute(query, query_params)
            rows = cursor.fetchall()
            
            logger.debug(f"list_sessions returned {len(rows)} rows")
            
            conn.close()
            
            sessions = []
            for row in rows:
                session_dict = dict(row)
                # datetime 객체를 ISO 형식 문자열로 변환
                if isinstance(session_dict.get("created_at"), datetime):
                    session_dict["created_at"] = session_dict["created_at"].isoformat()
                if isinstance(session_dict.get("updated_at"), datetime):
                    session_dict["updated_at"] = session_dict["updated_at"].isoformat()
                
                sessions.append(session_dict)
            
            return sessions, total
        except Exception as e:
            logger.error(f"Failed to list sessions: {e}")
            return [], 0
    
    def add_message(
        self,
        session_id: str,
        role: str,
        content: str,
        metadata: Optional[Dict[str, Any]] = None,
        message_id: Optional[str] = None
    ) -> str:
        """메시지 추가"""
        if message_id is None:
            message_id = str(uuid.uuid4())
        
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            import json
            metadata_str = json.dumps(metadata) if metadata else None
            
            now_kst = get_kst_now()
            now_kst_str = now_kst.isoformat()
            cursor.execute("""
                INSERT INTO messages (message_id, session_id, role, content, timestamp, metadata)
                VALUES (?, ?, ?, ?, ?, ?)
            """, (message_id, session_id, role, content, now_kst_str, metadata_str))
            
            # 메시지 개수 업데이트
            cursor.execute("""
                UPDATE sessions 
                SET message_count = message_count + 1, updated_at = ?
                WHERE session_id = ?
            """, (now_kst_str, session_id))
            
            conn.commit()
            conn.close()
            return message_id
        except Exception as e:
            logger.error(f"Failed to add message: {e}")
            raise
    
    def get_messages(
        self,
        session_id: str,
        limit: Optional[int] = None,
        user_id: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """메시지 조회"""
        try:
            # 세션 소유권 확인 (user_id가 제공된 경우)
            if user_id:
                session = self.get_session(session_id)
                if not session:
                    return []
                
                session_user_id = session.get("user_id")
                if session_user_id != user_id:
                    # 소유권이 없으면 빈 결과 반환
                    return []
            
            conn = sqlite3.connect(self.db_path)
            conn.row_factory = sqlite3.Row
            cursor = conn.cursor()
            
            query = "SELECT * FROM messages WHERE session_id = ? ORDER BY timestamp ASC"
            params = [session_id]
            
            if limit:
                query += " LIMIT ?"
                params.append(limit)
            
            cursor.execute(query, params)
            rows = cursor.fetchall()
            
            conn.close()
            
            messages = []
            for row in rows:
                msg = dict(row)
                
                # timestamp 처리: SQLite에서 문자열로 반환될 수 있으므로 그대로 유지
                # (API 레이어에서 datetime으로 변환)
                if msg.get("timestamp"):
                    # datetime 객체인 경우 ISO 형식 문자열로 변환
                    if isinstance(msg["timestamp"], datetime):
                        msg["timestamp"] = msg["timestamp"].isoformat()
                    # 이미 문자열이면 그대로 유지
                    elif isinstance(msg["timestamp"], str):
                        pass  # 그대로 유지
                    else:
                        # 다른 타입이면 문자열로 변환
                        msg["timestamp"] = str(msg["timestamp"])
                
                # metadata 파싱
                if msg.get("metadata"):
                    import json
                    try:
                        if isinstance(msg["metadata"], str):
                            msg["metadata"] = json.loads(msg["metadata"])
                        # 이미 dict이면 그대로 유지
                    except Exception as e:
                        logger.warning(f"Failed to parse metadata: {e}")
                        msg["metadata"] = {}
                else:
                    msg["metadata"] = {}
                
                # 필수 필드 확인
                if not msg.get("message_id"):
                    logger.warning(f"Message missing message_id: {msg}")
                    continue
                if not msg.get("session_id"):
                    logger.warning(f"Message missing session_id: {msg}")
                    continue
                if not msg.get("role"):
                    logger.warning(f"Message missing role: {msg}")
                    continue
                if not msg.get("content"):
                    msg["content"] = ""  # 빈 내용은 허용
                
                messages.append(msg)
            
            return messages
        except Exception as e:
            logger.error(f"Failed to get messages: {e}", exc_info=True)
            return []
    
    def generate_session_title(self, session_id: str) -> Optional[str]:
        """
        첫 번째 질문과 답변을 기반으로 Gemini를 사용하여 세션 제목 생성
        
        Args:
            session_id: 세션 ID
            
        Returns:
            생성된 제목 또는 None (실패 시)
        """
        try:
            # 세션 존재 확인
            session = self.get_session(session_id)
            if not session:
                logger.warning(f"Session not found: {session_id}")
                return None
            
            # 첫 번째 메시지 2개 가져오기 (질문 + 답변)
            messages = self.get_messages(session_id, limit=2)
            
            if len(messages) < 2:
                logger.warning(f"Not enough messages for title generation in session {session_id}")
                return None
            
            user_message = None
            assistant_message = None
            
            for msg in messages:
                if msg["role"] == "user" and not user_message:
                    user_message = msg["content"]
                elif msg["role"] == "assistant" and not assistant_message:
                    assistant_message = msg["content"]
            
            if not user_message or not assistant_message:
                logger.warning(f"Missing user or assistant message for title generation")
                return None
            
            # Gemini 클라이언트 생성 및 사용
            import sys
            from pathlib import Path
            
            # lawfirm_langgraph 경로 추가
            project_root = Path(__file__).parent.parent.parent
            lawfirm_langgraph_path = project_root / "lawfirm_langgraph"
            if lawfirm_langgraph_path.exists():
                sys.path.insert(0, str(lawfirm_langgraph_path))
            
            try:
                from core.shared.clients.gemini_client import GeminiClient
                gemini_client = GeminiClient()
            except ImportError:
                try:
                    from core.services.gemini_client import GeminiClient
                    gemini_client = GeminiClient()
                except Exception as e:
                    logger.error(f"Failed to initialize Gemini client: {e}")
                    gemini_client = None
            except Exception as e:
                logger.error(f"Failed to initialize Gemini client: {e}")
                gemini_client = None
                # 대체: 첫 질문의 일부를 제목으로 사용
                fallback_title = user_message[:20] + "..." if len(user_message) > 20 else user_message
                self.update_session(session_id, title=fallback_title)
                return fallback_title
            
            # 시스템 프롬프트 구성
            system_prompt = """당신은 법률 대화의 제목을 생성하는 전문가입니다. 
사용자의 질문과 AI의 답변을 간결하게 요약하여 10-20자 이내의 제목을 생성해주세요.
제목은 대화의 핵심 내용을 잘 반영해야 하며, 법률 용어를 정확하게 사용해야 합니다.
제목만 출력하고 다른 설명이나 문장은 하지 마세요."""
            
            # 프롬프트 구성
            prompt = f"""다음 대화를 기반으로 간결한 제목을 생성해주세요.

질문: {user_message[:300]}

답변: {assistant_message[:500]}

위 대화의 핵심을 담은 10-20자 이내의 제목만 생성해주세요. 제목만 출력하고 다른 설명은 하지 마세요."""
            
            # Gemini를 사용하여 제목 생성
            gemini_response = gemini_client.generate(
                prompt=prompt,
                system_prompt=system_prompt,
                temperature=0.7,
                max_tokens=50
            )
            
            # 응답 안전하게 처리
            if not gemini_response or not hasattr(gemini_response, 'response'):
                logger.warning("Invalid Gemini response structure")
                title = user_message[:20] + "..." if len(user_message) > 20 else user_message
            else:
                title = gemini_response.response.strip() if gemini_response.response else ""
            
            # 제목 정제 (따옴표, 불필요한 문자 제거)
            title = title.strip().strip('"').strip("'").strip()
            
            # 제목이 비어있거나 너무 길면 처리
            if not title:
                # 대체: 첫 질문의 일부를 제목으로 사용
                title = user_message[:20] + "..." if len(user_message) > 20 else user_message
            elif len(title) > 30:
                # 30자 초과 시 앞부분만 사용
                title = title[:30].rstrip() + "..."
            
            # 제목 업데이트
            if title:
                self.update_session(session_id, title=title)
                logger.info(f"Generated title for session {session_id}: {title}")
            
            return title
        except Exception as e:
            logger.error(f"Error generating session title: {e}", exc_info=True)
            # 대체: 첫 질문의 일부를 제목으로 사용
            try:
                messages = self.get_messages(session_id, limit=1)
                if messages and messages[0].get("role") == "user":
                    fallback_title = messages[0]["content"][:20] + "..." if len(messages[0]["content"]) > 20 else messages[0]["content"]
                    self.update_session(session_id, title=fallback_title)
                    return fallback_title
            except:
                pass
            return None
    
    def save_full_answer(
        self,
        session_id: str,
        message_id: str,
        full_answer: str,
        metadata: Optional[Dict[str, Any]] = None
    ) -> bool:
        """전체 답변을 메시지에 저장 (잘린 부분 포함)"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            import json
            metadata_str = json.dumps(metadata) if metadata else None
            
            cursor.execute("""
                UPDATE messages 
                SET content = ?, metadata = ?
                WHERE message_id = ? AND session_id = ?
            """, (full_answer, metadata_str, message_id, session_id))
            
            conn.commit()
            conn.close()
            return True
        except Exception as e:
            logger.error(f"Failed to save full answer: {e}")
            return False
    
    def get_answer_chunk(
        self,
        session_id: str,
        message_id: str,
        chunk_index: int
    ) -> Optional[Dict[str, Any]]:
        """특정 청크의 답변 가져오기"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            cursor.execute("""
                SELECT content, metadata
                FROM messages
                WHERE message_id = ? AND session_id = ? AND role = 'assistant'
            """, (message_id, session_id))
            
            row = cursor.fetchone()
            conn.close()
            
            if not row:
                return None
            
            full_answer = row[0]
            metadata_str = row[1]
            
            import json
            metadata = json.loads(metadata_str) if metadata_str else {}
            
            from api.services.answer_splitter import AnswerSplitter
            chunk_size = metadata.get("chunk_size", 2000)
            splitter = AnswerSplitter(chunk_size=chunk_size)
            chunks = splitter.split_answer(full_answer)
            
            if 0 <= chunk_index < len(chunks):
                chunk = chunks[chunk_index]
                return {
                    "content": chunk.content,
                    "chunk_index": chunk.chunk_index,
                    "total_chunks": chunk.total_chunks,
                    "has_more": chunk.has_more
                }
            
            return None
        except Exception as e:
            logger.error(f"Failed to get answer chunk: {e}")
            return None


# 전역 서비스 인스턴스
session_service = SessionService()

