# -*- coding: utf-8 -*-
"""
Open Law 수집 유틸리티
공통 함수들
"""

import os
from urllib.parse import quote_plus
from typing import Optional


def build_database_url() -> Optional[str]:
    """
    개별 PostgreSQL 환경 변수로부터 DATABASE_URL 구성
    
    우선순위:
    1. DATABASE_URL 환경 변수가 직접 설정되어 있고 postgresql://로 시작하면 사용
    2. 개별 POSTGRES_* 환경 변수로부터 구성 (SQLite URL 무시)
    
    Returns:
        Optional[str]: PostgreSQL 데이터베이스 URL 또는 None
    """
    # DATABASE_URL이 직접 설정되어 있고 PostgreSQL인 경우 사용
    db_url = os.getenv('DATABASE_URL')
    if db_url and db_url.startswith('postgresql'):
        return db_url
    
    # 개별 변수로부터 구성 (SQLite URL은 무시)
    host = os.getenv('POSTGRES_HOST', 'localhost')
    port = os.getenv('POSTGRES_PORT', '5432')
    db = os.getenv('POSTGRES_DB')
    user = os.getenv('POSTGRES_USER')
    password = os.getenv('POSTGRES_PASSWORD')
    
    if db and user and password:
        # 비밀번호에 특수문자가 있을 수 있으므로 URL 인코딩
        encoded_password = quote_plus(password)
        return f"postgresql://{user}:{encoded_password}@{host}:{port}/{db}"
    
    # PostgreSQL 환경변수가 없으면 None 반환 (SQLite 사용 안 함)
    return None

