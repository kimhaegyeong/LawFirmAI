#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""형법 판례 임베딩 상태 확인 스크립트"""

import sys
from pathlib import Path

_CURRENT_FILE = Path(__file__).resolve()
_PROJECT_ROOT = _CURRENT_FILE.parents[3]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

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
    print("=" * 80)
    print("📊 형법 판례 임베딩 상태 확인")
    print("=" * 80)
    print()
    
    # 1. 전체 형법 판례 수
    result = conn.execute(text("""
        SELECT COUNT(DISTINCT p.id)
        FROM precedents p
        WHERE p.domain = 'criminal_law'
    """))
    total_precedents = result.scalar()
    print(f"✅ 전체 형법 판례 수: {total_precedents:,}개")
    
    # 2. 전체 청크 수
    result = conn.execute(text("""
        SELECT COUNT(*) 
        FROM precedent_chunks pc 
        JOIN precedent_contents pcon ON pc.precedent_content_id = pcon.id 
        JOIN precedents p ON pcon.precedent_id = p.id 
        WHERE p.domain = 'criminal_law'
    """))
    total_chunks = result.scalar()
    print(f"✅ 전체 청크 수: {total_chunks:,}개")
    
    # 3. 임베딩 완료 청크 수
    result = conn.execute(text("""
        SELECT COUNT(*) 
        FROM precedent_chunks pc 
        JOIN precedent_contents pcon ON pc.precedent_content_id = pcon.id 
        JOIN precedents p ON pcon.precedent_id = p.id 
        WHERE p.domain = 'criminal_law' 
          AND pc.embedding_vector IS NOT NULL
    """))
    embedded_chunks = result.scalar()
    
    # 4. 임베딩 완료 판례 수
    result = conn.execute(text("""
        SELECT COUNT(DISTINCT p.id)
        FROM precedents p
        JOIN precedent_contents pcon ON p.id = pcon.precedent_id
        JOIN precedent_chunks pc ON pcon.id = pc.precedent_content_id
        WHERE p.domain = 'criminal_law' 
          AND pc.embedding_vector IS NOT NULL
    """))
    embedded_precedents = result.scalar()
    
    print()
    print("=" * 80)
    print("📈 임베딩 통계")
    print("=" * 80)
    
    if total_chunks > 0:
        embedding_rate = (embedded_chunks / total_chunks) * 100
        print(f"✅ 임베딩 완료 청크: {embedded_chunks:,}개 / {total_chunks:,}개 ({embedding_rate:.2f}%)")
        print(f"⏳ 임베딩 대기 청크: {total_chunks - embedded_chunks:,}개")
    else:
        print("⚠️  청크가 없습니다. 먼저 청킹을 진행해주세요.")
    
    if total_precedents > 0:
        precedent_rate = (embedded_precedents / total_precedents) * 100
        print(f"✅ 임베딩 완료 판례: {embedded_precedents:,}개 / {total_precedents:,}개 ({precedent_rate:.2f}%)")
        print(f"⏳ 임베딩 대기 판례: {total_precedents - embedded_precedents:,}개")
    
    print()
    
    # 5. 버전별 임베딩 통계
    result = conn.execute(text("""
        SELECT 
            pc.embedding_version,
            COUNT(*) as embedded_count
        FROM precedent_chunks pc 
        JOIN precedent_contents pcon ON pc.precedent_content_id = pcon.id 
        JOIN precedents p ON pcon.precedent_id = p.id 
        WHERE p.domain = 'criminal_law'
          AND pc.embedding_vector IS NOT NULL
        GROUP BY pc.embedding_version
        ORDER BY pc.embedding_version DESC
    """))
    version_stats = result.fetchall()
    
    if version_stats:
        print("=" * 80)
        print("📊 버전별 임베딩 통계")
        print("=" * 80)
        for version, count in version_stats:
            print(f"  Version {version}: {count:,}개")
        print()
    
    # 6. 최근 임베딩 시간
    result = conn.execute(text("""
        SELECT MAX(pc.created_at)
        FROM precedent_chunks pc 
        JOIN precedent_contents pcon ON pc.precedent_content_id = pcon.id 
        JOIN precedents p ON pcon.precedent_id = p.id 
        WHERE p.domain = 'criminal_law' 
          AND pc.embedding_vector IS NOT NULL
    """))
    last_embedding_time = result.scalar()
    
    if last_embedding_time:
        print(f"📅 최근 임베딩 시간: {last_embedding_time}")
        print()
    
    # 7. 상태 요약
    print("=" * 80)
    print("📋 상태 요약")
    print("=" * 80)
    
    if total_chunks == 0:
        print("❌ 형법 판례가 청킹되지 않았습니다.")
        print("   → 청킹을 먼저 진행해주세요: python scripts/ingest/open_law/chunk_precedents.py --domain criminal_law")
    elif embedded_chunks == 0:
        print("⚠️  청킹은 완료되었지만 임베딩이 전혀 진행되지 않았습니다.")
        print("   → 임베딩을 진행해주세요: python scripts/ingest/open_law/embedding/pgvector/pgvector_embedder.py --domain criminal_law")
    elif embedded_chunks == total_chunks:
        print("✅ 모든 형법 판례 임베딩이 완료되었습니다!")
    elif embedded_chunks / total_chunks >= 0.9:
        print("✅ 임베딩이 거의 완료되었습니다 (90% 이상)")
    elif embedded_chunks / total_chunks >= 0.5:
        print("⚠️  임베딩이 진행 중입니다 (50% 이상)")
    else:
        print("⚠️  임베딩이 초기 단계입니다 (50% 미만)")
    
    print("=" * 80)




