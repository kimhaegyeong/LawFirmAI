#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""형법 판례 임베딩 진행 상황 확인 스크립트 (상세 버전)"""

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

try:
    engine = create_engine(db_url)
    conn = engine.connect()
    
    domain = 'criminal_law'  # 형법 판례
    
    # 전체 형법 판례 청크 수
    result1 = conn.execute(text("""
        SELECT COUNT(*) 
        FROM precedent_chunks pc 
        JOIN precedent_contents pcon ON pc.precedent_content_id = pcon.id 
        JOIN precedents p ON pcon.precedent_id = p.id 
        WHERE p.domain = :domain
    """), {'domain': domain})
    total = result1.scalar()
    
    # 임베딩 완료 수 (모든 버전)
    result2 = conn.execute(text("""
        SELECT COUNT(*) 
        FROM precedent_chunks pc 
        JOIN precedent_contents pcon ON pc.precedent_content_id = pcon.id 
        JOIN precedents p ON pcon.precedent_id = p.id 
        WHERE p.domain = :domain AND pc.embedding_vector IS NOT NULL
    """), {'domain': domain})
    embedded = result2.scalar()
    
    # 최신 버전 확인
    result3 = conn.execute(text("""
        SELECT MAX(version) as max_version
        FROM embedding_versions
        WHERE data_type = 'precedents'
    """))
    max_version = result3.scalar()
    
    # 최신 버전의 임베딩 완료 수
    if max_version:
        result4 = conn.execute(text("""
            SELECT COUNT(*) 
            FROM precedent_chunks pc 
            JOIN precedent_contents pcon ON pc.precedent_content_id = pcon.id 
            JOIN precedents p ON pcon.precedent_id = p.id 
            WHERE p.domain = :domain 
              AND pc.embedding_vector IS NOT NULL 
              AND pc.embedding_version = :version
        """), {'domain': domain, 'version': max_version})
        recent_embedded = result4.scalar()
    else:
        recent_embedded = 0
        max_version = None
    
    # 최근 임베딩 시간
    result5 = conn.execute(text("""
        SELECT MAX(pc.created_at)
        FROM precedent_chunks pc 
        JOIN precedent_contents pcon ON pc.precedent_content_id = pcon.id 
        JOIN precedents p ON pcon.precedent_id = p.id 
        WHERE p.domain = :domain AND pc.embedding_vector IS NOT NULL
    """), {'domain': domain})
    last_embedding_time = result5.scalar()
    
    # 버전별 통계
    result6 = conn.execute(text("""
        SELECT 
            pc.embedding_version,
            COUNT(*) as embedded_count
        FROM precedent_chunks pc 
        JOIN precedent_contents pcon ON pc.precedent_content_id = pcon.id 
        JOIN precedents p ON pcon.precedent_id = p.id 
        WHERE p.domain = :domain
          AND pc.embedding_vector IS NOT NULL
        GROUP BY pc.embedding_version
        ORDER BY pc.embedding_version DESC
    """), {'domain': domain})
    version_stats = result6.fetchall()
    
    # 판례 수 통계
    result7 = conn.execute(text("""
        SELECT COUNT(DISTINCT p.id)
        FROM precedents p
        WHERE p.domain = :domain
    """), {'domain': domain})
    total_precedents = result7.scalar()
    
    result8 = conn.execute(text("""
        SELECT COUNT(DISTINCT p.id)
        FROM precedents p
        JOIN precedent_contents pcon ON p.id = pcon.precedent_id
        JOIN precedent_chunks pc ON pcon.id = pc.precedent_content_id
        WHERE p.domain = :domain 
          AND pc.embedding_vector IS NOT NULL
    """), {'domain': domain})
    embedded_precedents = result8.scalar()
    
    # 출력을 파일로도 저장
    output_lines = []
    output_lines.append("=" * 80)
    output_lines.append("📊 형법 판례 임베딩 진행 상황")
    output_lines.append("=" * 80)
    output_lines.append(f"✅ 전체 판례 수: {total_precedents:,}개")
    output_lines.append(f"✅ 임베딩 완료 판례: {embedded_precedents:,}개")
    output_lines.append("")
    output_lines.append(f"✅ 전체 청크 수: {total:,}개")
    output_lines.append(f"✅ 임베딩 완료 청크: {embedded:,}개 ({embedded/total*100:.2f}%)" if total > 0 else "✅ 임베딩 완료 청크: 0개")
    output_lines.append(f"⏳ 남은 작업: {total-embedded:,}개" if total > 0 else "⏳ 남은 작업: 0개")
    output_lines.append("")
    
    if max_version:
        output_lines.append(f"📌 최신 버전: {max_version}")
        output_lines.append(f"📊 최신 버전 임베딩 완료: {recent_embedded:,}개")
        output_lines.append("")
    
    if version_stats:
        output_lines.append("📊 버전별 임베딩 통계:")
        for version, count in version_stats:
            output_lines.append(f"  Version {version}: {count:,}개")
        output_lines.append("")
    
    if last_embedding_time:
        output_lines.append(f"📅 최근 임베딩 시간: {last_embedding_time}")
    
    output_lines.append("=" * 80)
    
    # 출력
    output_text = "\n".join(output_lines)
    print(output_text)
    sys.stdout.flush()
    
    # 파일로도 저장
    output_file = Path("logs/criminal_precedent_embedding_status.txt")
    output_file.parent.mkdir(parents=True, exist_ok=True)
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write(output_text)
    print(f"\n💾 결과가 파일로 저장되었습니다: {output_file}")
    
    conn.close()
    
except Exception as e:
    print(f"❌ 오류 발생: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)
