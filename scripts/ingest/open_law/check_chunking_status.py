#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""청킹 진행 상황 확인 스크립트"""

import sys
from pathlib import Path

_CURRENT_FILE = Path(__file__).resolve()
# scripts/ingest/open_law/check_chunking_status.py -> parents[3] = 프로젝트 루트
_PROJECT_ROOT = _CURRENT_FILE.parents[3]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

# .env 파일 로드
try:
    from utils.env_loader import ensure_env_loaded
    ensure_env_loaded(_PROJECT_ROOT)
except ImportError:
    try:
        from dotenv import load_dotenv
        scripts_env = _PROJECT_ROOT / "scripts" / ".env"
        if scripts_env.exists():
            load_dotenv(dotenv_path=str(scripts_env), override=True)
        root_env = _PROJECT_ROOT / ".env"
        if root_env.exists():
            load_dotenv(dotenv_path=str(root_env), override=False)
    except ImportError:
        pass

# scripts 디렉토리를 sys.path에 추가
_SCRIPTS_DIR = _PROJECT_ROOT / "scripts"
if str(_SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS_DIR))

try:
    from scripts.ingest.open_law.utils import build_database_url
except ImportError:
    from urllib.parse import quote_plus
    import os
    def build_database_url():
        db_url = os.getenv('DATABASE_URL')
        if db_url and db_url.startswith('postgresql'):
            return db_url
        host = os.getenv('POSTGRES_HOST', 'localhost')
        port = os.getenv('POSTGRES_PORT', '5432')
        db = os.getenv('POSTGRES_DB')
        user = os.getenv('POSTGRES_USER')
        password = os.getenv('POSTGRES_PASSWORD')
        if db and user and password:
            encoded_password = quote_plus(password)
            return f"postgresql://{user}:{encoded_password}@{host}:{port}/{db}"
        return None

from sqlalchemy import create_engine, text

db_url = build_database_url()
if not db_url:
    print("ERROR: DATABASE_URL not found")
    sys.exit(1)

engine = create_engine(db_url, pool_pre_ping=True)

with engine.connect() as conn:
    # 형법 판례 통계
    print("=" * 60)
    print("형법 판례 청킹 진행 상황")
    print("=" * 60)
    
    # 전체 섹션 수
    result = conn.execute(text("""
        SELECT COUNT(*) 
        FROM precedent_contents pc 
        JOIN precedents p ON pc.precedent_id = p.id 
        WHERE p.domain = 'criminal_law'
    """))
    total_sections = result.scalar()
    
    # 청킹된 청크 수
    result = conn.execute(text("""
        SELECT COUNT(*) 
        FROM precedent_chunks pch 
        JOIN precedent_contents pc ON pch.precedent_content_id = pc.id 
        JOIN precedents p ON pc.precedent_id = p.id 
        WHERE p.domain = 'criminal_law' AND pch.embedding_version = 1
    """))
    chunked_count = result.scalar()
    
    # 청킹 대기 중인 섹션 수
    result = conn.execute(text("""
        SELECT COUNT(DISTINCT pc.id) 
        FROM precedent_contents pc 
        JOIN precedents p ON pc.precedent_id = p.id 
        WHERE p.domain = 'criminal_law' 
        AND NOT EXISTS (
            SELECT 1 FROM precedent_chunks pch 
            WHERE pch.precedent_content_id = pc.id
        )
    """))
    remaining_sections = result.scalar()
    
    # 청킹 완료된 섹션 수
    processed_sections = total_sections - remaining_sections
    
    print(f"전체 섹션: {total_sections:,}개")
    print(f"청킹 완료: {processed_sections:,}개 섹션 ({processed_sections*100//total_sections if total_sections > 0 else 0}%)")
    print(f"청킹 대기: {remaining_sections:,}개 섹션")
    print(f"생성된 청크: {chunked_count:,}개")
    if processed_sections > 0:
        avg_chunks = chunked_count / processed_sections
        print(f"평균 청크/섹션: {avg_chunks:.2f}개")
    
    print()
    
    # 민사법 판례 통계 (비교용)
    print("=" * 60)
    print("민사법 판례 청킹 진행 상황")
    print("=" * 60)
    
    result = conn.execute(text("""
        SELECT COUNT(*) 
        FROM precedent_contents pc 
        JOIN precedents p ON pc.precedent_id = p.id 
        WHERE p.domain = 'civil_law'
    """))
    civil_total = result.scalar()
    
    result = conn.execute(text("""
        SELECT COUNT(*) 
        FROM precedent_chunks pch 
        JOIN precedent_contents pc ON pch.precedent_content_id = pc.id 
        JOIN precedents p ON pc.precedent_id = p.id 
        WHERE p.domain = 'civil_law' AND pch.embedding_version = 1
    """))
    civil_chunked = result.scalar()
    
    result = conn.execute(text("""
        SELECT COUNT(DISTINCT pc.id) 
        FROM precedent_contents pc 
        JOIN precedents p ON pc.precedent_id = p.id 
        WHERE p.domain = 'civil_law' 
        AND NOT EXISTS (
            SELECT 1 FROM precedent_chunks pch 
            WHERE pch.precedent_content_id = pc.id
        )
    """))
    civil_remaining = result.scalar()
    civil_processed = civil_total - civil_remaining
    
    print(f"전체 섹션: {civil_total:,}개")
    print(f"청킹 완료: {civil_processed:,}개 섹션 ({civil_processed*100//civil_total if civil_total > 0 else 0}%)")
    print(f"청킹 대기: {civil_remaining:,}개 섹션")
    print(f"생성된 청크: {civil_chunked:,}개")
    if civil_processed > 0:
        avg_chunks = civil_chunked / civil_processed
        print(f"평균 청크/섹션: {avg_chunks:.2f}개")

