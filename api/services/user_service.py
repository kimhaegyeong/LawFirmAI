"""
사용자 관리 서비스
"""
import os
from datetime import datetime
from typing import Optional, Dict, Any
from pathlib import Path
import logging
from sqlalchemy import Index, text, inspect
from sqlalchemy.exc import OperationalError, IntegrityError, ProgrammingError, DatabaseError

from api.config import api_config
from api.database.connection import get_session, init_database
from api.database.models import User

logger = logging.getLogger(__name__)


class UserService:
    """사용자 관리 서비스"""
    
    def __init__(self):
        """초기화"""
        self._init_database()
    
    def _init_database(self):
        """데이터베이스 초기화"""
        try:
            # SQLAlchemy를 사용하여 테이블 생성
            init_database()
            
            # 테이블 존재 여부 확인
            db = get_session()
            try:
                inspector = inspect(db.bind)
                tables = inspector.get_table_names()
                
                if 'users' not in tables:
                    logger.error("Users table was not created after init_database() call")
                    raise RuntimeError("Users table does not exist in database")
                
                logger.info(f"Users table exists. Total tables: {len(tables)}")
                
                # 인덱스 생성 (이미 존재하면 무시됨)
                # 인덱스는 모델에서 정의하거나 별도로 생성
                # PostgreSQL만 지원
                db.execute(text("CREATE INDEX IF NOT EXISTS idx_users_email ON users(email)"))
                db.execute(text("CREATE INDEX IF NOT EXISTS idx_users_provider ON users(provider)"))
                db.commit()
            except OperationalError as e:
                # 인덱스가 이미 존재하는 경우 무시
                db.rollback()
                logger.debug(f"Index creation skipped (may already exist): {e}")
            except Exception as e:
                db.rollback()
                logger.error(f"Failed to verify or create indexes: {e}", exc_info=True)
                raise
            finally:
                db.close()
            
            logger.info("Users table initialized successfully")
        except Exception as e:
            logger.error(f"Failed to initialize users table: {e}", exc_info=True)
            raise
    
    def create_or_update_user(
        self,
        user_id: str,
        email: Optional[str] = None,
        name: Optional[str] = None,
        picture: Optional[str] = None,
        provider: Optional[str] = None,
        google_access_token: Optional[str] = None,
        google_refresh_token: Optional[str] = None
    ) -> bool:
        """사용자 생성 또는 업데이트"""
        if not user_id:
            logger.error("create_or_update_user: user_id is required")
            return False
        
        db = get_session()
        try:
            # 데이터 유효성 검사
            if user_id and len(user_id) > 255:
                logger.error(f"create_or_update_user: user_id too long (max 255): {len(user_id)}")
                return False
            
            user = db.query(User).filter(User.user_id == user_id).first()
            
            if user:
                # 업데이트
                logger.debug(f"Updating existing user: user_id={user_id}")
                if email is not None:
                    user.email = email
                if name is not None:
                    user.name = name
                if picture is not None:
                    user.picture = picture
                if provider is not None:
                    user.provider = provider
                if google_access_token is not None:
                    user.google_access_token = google_access_token
                if google_refresh_token is not None:
                    user.google_refresh_token = google_refresh_token
                user.updated_at = datetime.now()
            else:
                # 생성
                logger.debug(f"Creating new user: user_id={user_id}, email={email}")
                user = User(
                    user_id=user_id,
                    email=email,
                    name=name,
                    picture=picture,
                    provider=provider,
                    google_access_token=google_access_token,
                    google_refresh_token=google_refresh_token,
                    created_at=datetime.now(),
                    updated_at=datetime.now()
                )
                db.add(user)
            
            # 트랜잭션 커밋
            db.commit()
            logger.info(f"User created or updated successfully: user_id={user_id}, email={email}")
            return True
            
        except IntegrityError as e:
            db.rollback()
            error_msg = str(e.orig) if hasattr(e, 'orig') else str(e)
            logger.error(
                f"Failed to create or update user (IntegrityError): user_id={user_id}, "
                f"error={error_msg}",
                exc_info=True
            )
            # UNIQUE 제약조건 위반 등
            if "unique" in error_msg.lower() or "duplicate" in error_msg.lower():
                logger.warning(f"Duplicate user detected: user_id={user_id}")
            return False
            
        except OperationalError as e:
            db.rollback()
            error_msg = str(e.orig) if hasattr(e, 'orig') else str(e)
            logger.error(
                f"Failed to create or update user (OperationalError): user_id={user_id}, "
                f"error={error_msg}",
                exc_info=True
            )
            # 연결 오류, 테이블 없음 등
            if "does not exist" in error_msg.lower() or "relation" in error_msg.lower():
                logger.error(f"Database table may not exist: {error_msg}")
            elif "connection" in error_msg.lower() or "timeout" in error_msg.lower():
                logger.error(f"Database connection error: {error_msg}")
            return False
            
        except ProgrammingError as e:
            db.rollback()
            error_msg = str(e.orig) if hasattr(e, 'orig') else str(e)
            logger.error(
                f"Failed to create or update user (ProgrammingError): user_id={user_id}, "
                f"error={error_msg}",
                exc_info=True
            )
            # SQL 구문 오류 등
            logger.error(f"SQL syntax or database programming error: {error_msg}")
            return False
            
        except DatabaseError as e:
            db.rollback()
            error_msg = str(e.orig) if hasattr(e, 'orig') else str(e)
            logger.error(
                f"Failed to create or update user (DatabaseError): user_id={user_id}, "
                f"error={error_msg}",
                exc_info=True
            )
            return False
            
        except Exception as e:
            db.rollback()
            logger.error(
                f"Failed to create or update user (Unexpected error): user_id={user_id}, "
                f"error={type(e).__name__}: {str(e)}",
                exc_info=True
            )
            return False
        finally:
            db.close()
    
    def get_user(self, user_id: str) -> Optional[Dict[str, Any]]:
        """사용자 정보 조회"""
        if not user_id:
            logger.warning("get_user: user_id is required")
            return None
        
        db = get_session()
        try:
            user = db.query(User).filter(User.user_id == user_id).first()
            
            if user:
                return user.to_dict()
            return None
        except OperationalError as e:
            error_msg = str(e.orig) if hasattr(e, 'orig') else str(e)
            logger.error(
                f"Failed to get user (OperationalError): user_id={user_id}, "
                f"error={error_msg}",
                exc_info=True
            )
            return None
        except Exception as e:
            logger.error(
                f"Failed to get user: user_id={user_id}, error={type(e).__name__}: {str(e)}",
                exc_info=True
            )
            return None
        finally:
            db.close()
    
    def get_google_tokens(self, user_id: str) -> Optional[Dict[str, str]]:
        """구글 토큰 조회"""
        if not user_id:
            logger.warning("get_google_tokens: user_id is required")
            return None
        
        db = get_session()
        try:
            user = db.query(User).filter(
                User.user_id == user_id,
                User.provider == 'google'
            ).first()
            
            if user:
                return {
                    "access_token": user.google_access_token,
                    "refresh_token": user.google_refresh_token
                }
            return None
        except Exception as e:
            logger.error(
                f"Failed to get Google tokens: user_id={user_id}, "
                f"error={type(e).__name__}: {str(e)}",
                exc_info=True
            )
            return None
        finally:
            db.close()
    
    def delete_user(self, user_id: str) -> bool:
        """사용자 삭제"""
        if not user_id:
            logger.warning("delete_user: user_id is required")
            return False
        
        db = get_session()
        try:
            user = db.query(User).filter(User.user_id == user_id).first()
            
            if not user:
                logger.debug(f"User not found for deletion: user_id={user_id}")
                return False
            
            db.delete(user)
            db.commit()
            logger.info(f"User deleted successfully: user_id={user_id}")
            return True
        except IntegrityError as e:
            db.rollback()
            error_msg = str(e.orig) if hasattr(e, 'orig') else str(e)
            logger.error(
                f"Failed to delete user (IntegrityError): user_id={user_id}, "
                f"error={error_msg}",
                exc_info=True
            )
            return False
        except OperationalError as e:
            db.rollback()
            error_msg = str(e.orig) if hasattr(e, 'orig') else str(e)
            logger.error(
                f"Failed to delete user (OperationalError): user_id={user_id}, "
                f"error={error_msg}",
                exc_info=True
            )
            return False
        except Exception as e:
            db.rollback()
            logger.error(
                f"Failed to delete user: user_id={user_id}, "
                f"error={type(e).__name__}: {str(e)}",
                exc_info=True
            )
            return False
        finally:
            db.close()


user_service = UserService()
