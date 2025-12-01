"""
세션 관리 서비스
"""
import os
import uuid
import json
from datetime import datetime, timezone, timedelta
from typing import List, Optional, Dict, Any
from pathlib import Path
import logging
from sqlalchemy import inspect, text, func, and_, or_
from sqlalchemy.exc import OperationalError, IntegrityError
from sqlalchemy.orm import Session

from api.config import api_config
from api.database.connection import get_session, init_database, get_database_type
from api.database.models import Session as SessionModel, Message as MessageModel

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
        self._init_database()
    
    def _init_database(self):
        """데이터베이스 초기화"""
        try:
            # SQLAlchemy를 사용하여 테이블 생성
            init_database()
            
            # 컬럼 마이그레이션 (PostgreSQL만 지원)
            self._migrate_columns_postgresql()
            
            logger.info("Database initialized successfully")
        except Exception as e:
            logger.error(f"Failed to initialize database: {e}")
    
    def _migrate_columns_postgresql(self):
        """PostgreSQL 컬럼 마이그레이션"""
        db = get_session()
        try:
            inspector = inspect(db.bind)
            existing_columns = [col['name'] for col in inspector.get_columns('sessions')]
            
            # user_id 컬럼 추가
            if "user_id" not in existing_columns:
                try:
                    db.execute(text("ALTER TABLE sessions ADD COLUMN user_id VARCHAR(255)"))
                    db.commit()
                    logger.info("Added user_id column to sessions table")
                except OperationalError as e:
                    db.rollback()
                    logger.warning(f"Failed to add user_id column: {e}")
            
            # ip_address 컬럼 추가
            if "ip_address" not in existing_columns:
                try:
                    db.execute(text("ALTER TABLE sessions ADD COLUMN ip_address VARCHAR(45)"))
                    db.commit()
                    logger.info("Added ip_address column to sessions table")
                except OperationalError as e:
                    db.rollback()
                    logger.warning(f"Failed to add ip_address column: {e}")
            
            # metadata 컬럼 제거 (있는 경우)
            if "metadata" in existing_columns:
                try:
                    db.execute(text("ALTER TABLE sessions DROP COLUMN metadata"))
                    db.commit()
                    logger.info("Removed metadata column from sessions table")
                except OperationalError as e:
                    db.rollback()
                    logger.warning(f"Failed to remove metadata column: {e}")
            
            # category 컬럼 제거 (있는 경우)
            if "category" in existing_columns:
                try:
                    db.execute(text("ALTER TABLE sessions DROP COLUMN category"))
                    db.commit()
                    logger.info("Removed category column from sessions table")
                except OperationalError as e:
                    db.rollback()
                    logger.warning(f"Failed to remove category column: {e}")
        finally:
            db.close()
    
    
    def create_session(
        self,
        title: Optional[str] = None,
        user_id: Optional[str] = None,
        ip_address: Optional[str] = None
    ) -> str:
        """세션 생성"""
        session_id = str(uuid.uuid4())
        
        logger.debug(f"[create_session] Creating session: session_id={session_id}, title={title}, user_id={user_id}")
        
        db = get_session()
        try:
            now_kst = get_kst_now()
            session = SessionModel(
                session_id=session_id,
                title=title,
                created_at=now_kst,
                updated_at=now_kst,
                message_count=0,
                user_id=user_id,
                ip_address=ip_address
            )
            db.add(session)
            db.commit()
            logger.info(f"[create_session] Session created successfully: session_id={session_id}, title={title}")
            
            # 생성 확인
            verify_session = db.query(SessionModel).filter(SessionModel.session_id == session_id).first()
            if not verify_session:
                logger.error(f"[create_session] Session created but not found in database: session_id={session_id}")
            else:
                logger.debug(f"[create_session] Session verified in database: session_id={session_id}")
            
            return session_id
        except Exception as e:
            db.rollback()
            logger.error(f"[create_session] Failed to create session: {e}", exc_info=True)
            raise
        finally:
            db.close()
    
    def get_session(self, session_id: str, check_expiry: bool = True) -> Optional[Dict[str, Any]]:
        """세션 조회"""
        db = get_session()
        try:
            session = db.query(SessionModel).filter(SessionModel.session_id == session_id).first()
            
            if not session:
                logger.debug(f"[get_session] Session not found in database: {session_id}")
                return None
            
            session_dict = session.to_dict()
            
            # 세션 만료 시간 확인
            if check_expiry:
                updated_at = session_dict.get("updated_at")
                if updated_at:
                    try:
                        if isinstance(updated_at, str):
                            updated_at = datetime.fromisoformat(updated_at.replace('Z', '+00:00'))
                        if isinstance(updated_at, datetime):
                            # timezone-naive인 경우 KST로 변환
                            if updated_at.tzinfo is None:
                                # 데이터베이스에서 가져온 시간은 UTC로 가정하고 KST로 변환
                                updated_at = updated_at.replace(tzinfo=timezone.utc).astimezone(KST)
                            elif updated_at.tzinfo != KST:
                                # 다른 timezone인 경우 KST로 변환
                                updated_at = updated_at.astimezone(KST)
                            
                            expiry_hours = api_config.session_ttl_hours
                            expiry_time = updated_at + timedelta(hours=expiry_hours)
                            now_kst = get_kst_now()
                            if now_kst > expiry_time:
                                logger.warning(f"[get_session] Session expired: {session_id}, expiry_time={expiry_time}, now={now_kst}")
                                return None
                    except Exception as expiry_error:
                        logger.error(f"[get_session] Error checking expiry for session {session_id}: {expiry_error}", exc_info=True)
                        # 만료 확인 오류 시에도 세션 반환 (안전한 기본값)
            
            logger.debug(f"[get_session] Session found: {session_id}, title={session_dict.get('title')}")
            return session_dict
        except Exception as e:
            logger.error(f"[get_session] Failed to get session {session_id}: {e}", exc_info=True)
            return None
        finally:
            db.close()
    
    def update_session(
        self,
        session_id: str,
        title: Optional[str] = None,
        user_id: Optional[str] = None
    ) -> bool:
        """세션 업데이트"""
        db = get_session()
        try:
            session = db.query(SessionModel).filter(SessionModel.session_id == session_id).first()
            
            if not session:
                return False
            
            if title is not None:
                session.title = title
            if user_id is not None:
                session.user_id = user_id
            
            session.updated_at = get_kst_now()
            db.commit()
            logger.info(f"Session updated: {session_id}, title={title is not None}, user_id={user_id is not None}")
            return True
        except Exception as e:
            db.rollback()
            logger.error(f"Failed to update session: {e}")
            return False
        finally:
            db.close()
    
    def delete_session(self, session_id: str, user_id: Optional[str] = None) -> bool:
        """세션 삭제"""
        db = get_session()
        try:
            session = db.query(SessionModel).filter(SessionModel.session_id == session_id).first()
            
            if not session:
                return False
            
            # 세션 소유자 확인
            if user_id and session.user_id and session.user_id != user_id:
                logger.warning(f"Session ownership mismatch: {session_id}, user: {user_id}")
                return False
            
            # 메시지도 함께 삭제 (cascade로 자동 삭제됨)
            db.delete(session)
            db.commit()
            logger.info(f"Session deleted: {session_id}")
            return True
        except Exception as e:
            db.rollback()
            logger.error(f"Failed to delete session: {e}")
            return False
        finally:
            db.close()
    
    def delete_user_sessions(self, user_id: str) -> int:
        """사용자의 모든 세션 및 메시지 일괄 삭제"""
        db = get_session()
        try:
            sessions = db.query(SessionModel).filter(SessionModel.user_id == user_id).all()
            
            if not sessions:
                logger.info(f"No sessions found for user: {user_id}")
                return 0
            
            deleted_count = len(sessions)
            for session in sessions:
                db.delete(session)
            
            db.commit()
            logger.info(f"Deleted {deleted_count} sessions for user: {user_id}")
            return deleted_count
        except Exception as e:
            db.rollback()
            logger.error(f"Failed to delete user sessions: {e}")
            return 0
        finally:
            db.close()
    
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
        db = get_session()
        try:
            query = db.query(SessionModel)
            
            # WHERE 절 구성
            conditions = []
            
            # 사용자별 필터링
            if user_id:
                if user_id.startswith("anonymous_"):
                    conditions.append(SessionModel.user_id == user_id)
                else:
                    conditions.append(SessionModel.user_id == user_id)
            elif ip_address:
                conditions.append(SessionModel.ip_address == ip_address)
                conditions.append(or_(SessionModel.user_id.is_(None), SessionModel.user_id == ""))
            
            # 검색 필터
            if search:
                conditions.append(
                    or_(
                        SessionModel.title.like(f"%{search}%"),
                        SessionModel.session_id.like(f"%{search}%")
                    )
                )
            
            # 날짜 필터 (PostgreSQL)
            if date_from:
                try:
                    datetime.strptime(date_from, "%Y-%m-%d")
                    conditions.append(func.date(SessionModel.updated_at) >= date_from)
                except ValueError:
                    logger.warning(f"Invalid date_from format: {date_from}")
            
            if date_to:
                try:
                    datetime.strptime(date_to, "%Y-%m-%d")
                    conditions.append(func.date(SessionModel.updated_at) <= date_to)
                except ValueError:
                    logger.warning(f"Invalid date_to format: {date_to}")
            
            if conditions:
                query = query.filter(and_(*conditions))
            
            # 전체 개수 조회
            total = query.count()
            
            # 정렬 필드 검증 (SQL Injection 방지)
            valid_sort_fields = ["created_at", "updated_at", "title", "message_count"]
            sort_field = sort_by if sort_by in valid_sort_fields else "updated_at"
            sort_attr = getattr(SessionModel, sort_field)
            sort_dir = sort_attr.desc() if sort_order.lower() == "desc" else sort_attr.asc()
            
            # 정렬 및 페이지네이션
            offset = (page - 1) * page_size
            sessions = query.order_by(sort_dir).offset(offset).limit(page_size).all()
            
            return [session.to_dict() for session in sessions], total
        except Exception as e:
            logger.error(f"Failed to list sessions: {e}")
            return [], 0
        finally:
            db.close()
    
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
        
        db = get_session()
        try:
            now_kst = get_kst_now()
            message = MessageModel(
                message_id=message_id,
                session_id=session_id,
                role=role,
                content=content,
                timestamp=now_kst,
                metadata=metadata
            )
            db.add(message)
            
            # 메시지 개수 업데이트
            session = db.query(SessionModel).filter(SessionModel.session_id == session_id).first()
            if session:
                session.message_count = session.message_count + 1
                session.updated_at = now_kst
            
            db.commit()
            return message_id
        except Exception as e:
            db.rollback()
            logger.error(f"Failed to add message: {e}")
            raise
        finally:
            db.close()
    
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
                if session_user_id is None or session_user_id != user_id:
                    logger.debug(f"Session ownership mismatch: session_user_id={session_user_id}, user_id={user_id}")
                    return []
            
            db = get_session()
            try:
                query = db.query(MessageModel).filter(MessageModel.session_id == session_id).order_by(MessageModel.timestamp.asc())
                
                if limit:
                    query = query.limit(limit)
                
                messages = query.all()
                
                result = []
                for msg in messages:
                    msg_dict = msg.to_dict()
                    
                    # metadata가 이미 dict이면 그대로 사용
                    if msg_dict.get("metadata") is None:
                        msg_dict["metadata"] = {}
                    
                    # 필수 필드 확인
                    if not msg_dict.get("message_id"):
                        logger.warning(f"Message missing message_id: {msg_dict}")
                        continue
                    if not msg_dict.get("session_id"):
                        logger.warning(f"Message missing session_id: {msg_dict}")
                        continue
                    if not msg_dict.get("role"):
                        logger.warning(f"Message missing role: {msg_dict}")
                        continue
                    if not msg_dict.get("content"):
                        msg_dict["content"] = ""
                    
                    result.append(msg_dict)
                
                return result
            finally:
                db.close()
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
        db = get_session()
        try:
            message = db.query(MessageModel).filter(
                MessageModel.message_id == message_id,
                MessageModel.session_id == session_id
            ).first()
            
            if not message:
                return False
            
            message.content = full_answer
            message.message_metadata = metadata
            db.commit()
            return True
        except Exception as e:
            db.rollback()
            logger.error(f"Failed to save full answer: {e}")
            return False
        finally:
            db.close()
    
    def get_answer_chunk(
        self,
        session_id: str,
        message_id: str,
        chunk_index: int
    ) -> Optional[Dict[str, Any]]:
        """특정 청크의 답변 가져오기"""
        db = get_session()
        try:
            message = db.query(MessageModel).filter(
                MessageModel.message_id == message_id,
                MessageModel.session_id == session_id,
                MessageModel.role == 'assistant'
            ).first()
            
            if not message:
                return None
            
            full_answer = message.content
            metadata = message.message_metadata or {}
            
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
        finally:
            db.close()


# 전역 서비스 인스턴스
session_service = SessionService()
