#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""민법 법령 임베딩 진행 상황 확인 스크립트"""

import sys
from pathlib import Path
from datetime import datetime

sys.path.insert(0, str(Path.cwd()))

from utils.env_loader import ensure_env_loaded
ensure_env_loaded()

from scripts.ingest.open_law.utils import build_database_url
from urllib.parse import quote_plus
import os
from sqlalchemy import create_engine, text

# 데이터베이스 연결
db_url = build_database_url()
if not db_url or not db_url.startswith('postgresql'):
    host = os.getenv('POSTGRES_HOST', 'localhost')
    port = os.getenv('POSTGRES_PORT', '5432')
    db = os.getenv('POSTGRES_DB')
    user = os.getenv('POSTGRES_USER')
    password = os.getenv('POSTGRES_PASSWORD')
    if db and user and password:
        encoded_password = quote_plus(password)
        db_url = f"postgresql://{user}:{encoded_password}@{host}:{port}/{db}"

if not db_url:
    print("❌ 데이터베이스 연결 정보를 찾을 수 없습니다.")
    print(f"POSTGRES_HOST: {os.getenv('POSTGRES_HOST')}")
    print(f"POSTGRES_DB: {os.getenv('POSTGRES_DB')}")
    sys.exit(1)

print(f"🔗 데이터베이스 연결 중... (URL: {db_url[:50]}...)")
engine = create_engine(db_url)
conn = engine.connect()
print("✅ 데이터베이스 연결 성공")

# 전체 민법 조문 수
result1 = conn.execute(text("""
    SELECT COUNT(*) 
    FROM statutes_articles sa
    JOIN statutes s ON sa.statute_id = s.id
    WHERE s.domain = :domain
"""), {'domain': 'civil_law'})
total = result1.scalar()

# 임베딩 완료 수 (모든 버전)
result2 = conn.execute(text("""
    SELECT COUNT(DISTINCT se.article_id)
    FROM statute_embeddings se
    JOIN statutes_articles sa ON se.article_id = sa.id
    JOIN statutes s ON sa.statute_id = s.id
    WHERE s.domain = :domain
      AND se.embedding_vector IS NOT NULL
"""), {'domain': 'civil_law'})
embedded = result2.scalar()

# 최신 버전 확인
result3 = conn.execute(text("""
    SELECT MAX(version) as max_version
    FROM embedding_versions
    WHERE data_type = 'statutes'
"""))
max_version = result3.scalar()

# 최신 버전의 임베딩 완료 수
if max_version:
    result4 = conn.execute(text("""
        SELECT COUNT(DISTINCT se.article_id)
        FROM statute_embeddings se
        JOIN statutes_articles sa ON se.article_id = sa.id
        JOIN statutes s ON sa.statute_id = s.id
        WHERE s.domain = :domain
          AND se.embedding_version = :version
          AND se.embedding_vector IS NOT NULL
    """), {'domain': 'civil_law', 'version': max_version})
    recent_embedded = result4.scalar()
else:
    recent_embedded = 0
    max_version = None

# 최근 임베딩 시간
result5 = conn.execute(text("""
    SELECT MAX(se.created_at)
    FROM statute_embeddings se
    JOIN statutes_articles sa ON se.article_id = sa.id
    JOIN statutes s ON sa.statute_id = s.id
    WHERE s.domain = :domain
      AND se.embedding_vector IS NOT NULL
"""), {'domain': 'civil_law'})
last_embedding_time = result5.scalar()

# 버전별 통계
result6 = conn.execute(text("""
    SELECT 
        se.embedding_version,
        COUNT(DISTINCT se.article_id) as embedded_count
    FROM statute_embeddings se
    JOIN statutes_articles sa ON se.article_id = sa.id
    JOIN statutes s ON sa.statute_id = s.id
    WHERE s.domain = :domain
      AND se.embedding_vector IS NOT NULL
    GROUP BY se.embedding_version
    ORDER BY se.embedding_version DESC
"""), {'domain': 'civil_law'})
version_stats = result6.fetchall()

print("=" * 80)
print("📊 민법 법령 조문 임베딩 진행 상황")
print("=" * 80)
print(f"✅ 전체 조문 수: {total:,}개")
print(f"✅ 임베딩 완료: {embedded:,}개 ({embedded/total*100:.2f}%)" if total > 0 else "✅ 임베딩 완료: 0개")
print(f"⏳ 남은 작업: {total-embedded:,}개" if total > 0 else "⏳ 남은 작업: 0개")
print()

if max_version:
    print(f"📌 최신 버전: {max_version}")
    print(f"📊 최신 버전 임베딩 완료: {recent_embedded:,}개")
    print()

if version_stats:
    print("📊 버전별 임베딩 통계:")
    for version, count in version_stats:
        print(f"  Version {version}: {count:,}개")
    print()

if last_embedding_time:
    print(f"📅 최근 임베딩 시간: {last_embedding_time}")

print("=" * 80)

conn.close()

