"""
사용자 관리 서비스
"""
import os
import sqlite3
import json
from datetime import datetime
from typing import Optional, Dict, Any
from pathlib import Path
import logging

from api.config import api_config

logger = logging.getLogger(__name__)


class UserService:
    """사용자 관리 서비스"""
    
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
            
            # 사용자 테이블 생성
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS users (
                    user_id TEXT PRIMARY KEY,
                    email TEXT,
                    name TEXT,
                    picture TEXT,
                    provider TEXT,
                    google_access_token TEXT,
                    google_refresh_token TEXT,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """)
            
            # 인덱스 생성
            cursor.execute("""
                CREATE INDEX IF NOT EXISTS idx_users_email 
                ON users(email)
            """)
            
            cursor.execute("""
                CREATE INDEX IF NOT EXISTS idx_users_provider 
                ON users(provider)
            """)
            
            conn.commit()
            conn.close()
            logger.info("Users table initialized successfully")
        except Exception as e:
            logger.error(f"Failed to initialize users table: {e}")
    
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
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            # 기존 사용자 확인
            cursor.execute("SELECT user_id FROM users WHERE user_id = ?", (user_id,))
            existing = cursor.fetchone()
            
            if existing:
                # 업데이트
                updates = []
                params = []
                
                if email is not None:
                    updates.append("email = ?")
                    params.append(email)
                
                if name is not None:
                    updates.append("name = ?")
                    params.append(name)
                
                if picture is not None:
                    updates.append("picture = ?")
                    params.append(picture)
                
                if provider is not None:
                    updates.append("provider = ?")
                    params.append(provider)
                
                if google_access_token is not None:
                    updates.append("google_access_token = ?")
                    params.append(google_access_token)
                
                if google_refresh_token is not None:
                    updates.append("google_refresh_token = ?")
                    params.append(google_refresh_token)
                
                if updates:
                    updates.append("updated_at = ?")
                    params.append(datetime.now())
                    params.append(user_id)
                    
                    query = "UPDATE users SET " + ", ".join(updates) + " WHERE user_id = ?"
                    cursor.execute(query, params)
            else:
                # 생성
                cursor.execute("""
                    INSERT INTO users 
                    (user_id, email, name, picture, provider, google_access_token, google_refresh_token, created_at, updated_at)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    user_id,
                    email,
                    name,
                    picture,
                    provider,
                    google_access_token,
                    google_refresh_token,
                    datetime.now(),
                    datetime.now()
                ))
            
            conn.commit()
            conn.close()
            logger.info(f"User created or updated: user_id={user_id}")
            return True
        except Exception as e:
            logger.error(f"Failed to create or update user: {e}")
            return False
    
    def get_user(self, user_id: str) -> Optional[Dict[str, Any]]:
        """사용자 정보 조회"""
        try:
            conn = sqlite3.connect(self.db_path)
            conn.row_factory = sqlite3.Row
            cursor = conn.cursor()
            
            cursor.execute("SELECT * FROM users WHERE user_id = ?", (user_id,))
            row = cursor.fetchone()
            
            conn.close()
            
            if row:
                return dict(row)
            return None
        except Exception as e:
            logger.error(f"Failed to get user: {e}")
            return None
    
    def get_google_tokens(self, user_id: str) -> Optional[Dict[str, str]]:
        """구글 토큰 조회"""
        try:
            conn = sqlite3.connect(self.db_path)
            conn.row_factory = sqlite3.Row
            cursor = conn.cursor()
            
            cursor.execute("""
                SELECT google_access_token, google_refresh_token 
                FROM users 
                WHERE user_id = ? AND provider = 'google'
            """, (user_id,))
            row = cursor.fetchone()
            
            conn.close()
            
            if row:
                return {
                    "access_token": row["google_access_token"],
                    "refresh_token": row["google_refresh_token"]
                }
            return None
        except Exception as e:
            logger.error(f"Failed to get Google tokens: {e}")
            return None
    
    def delete_user(self, user_id: str) -> bool:
        """사용자 삭제"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            cursor.execute("DELETE FROM users WHERE user_id = ?", (user_id,))
            
            conn.commit()
            conn.close()
            logger.info(f"User deleted: user_id={user_id}")
            return True
        except Exception as e:
            logger.error(f"Failed to delete user: {e}")
            return False


user_service = UserService()

