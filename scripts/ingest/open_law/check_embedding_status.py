#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""임베딩 진행 상황 확인 스크립트"""

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

engine = create_engine(db_url)
conn = engine.connect()

# 전체 청크 수
result1 = conn.execute(text("""
    SELECT COUNT(*) 
    FROM precedent_chunks pc 
    JOIN precedent_contents pcon ON pc.precedent_content_id = pcon.id 
    JOIN precedents p ON pcon.precedent_id = p.id 
    WHERE p.domain = :domain
"""), {'domain': 'civil_law'})
total = result1.scalar()

# 임베딩 완료 수
result2 = conn.execute(text("""
    SELECT COUNT(*) 
    FROM precedent_chunks pc 
    JOIN precedent_contents pcon ON pc.precedent_content_id = pcon.id 
    JOIN precedents p ON pcon.precedent_id = p.id 
    WHERE p.domain = :domain AND pc.embedding_vector IS NOT NULL
"""), {'domain': 'civil_law'})
embedded = result2.scalar()

# 최근 임베딩 시간 확인 (더 정확한 방법: 실제 임베딩된 청크의 최근 시간)
# 1. 실제 임베딩된 청크의 최근 시간 (가장 정확)
result3a = conn.execute(text("""
    SELECT MAX(pc.created_at)
    FROM precedent_chunks pc 
    JOIN precedent_contents pcon ON pc.precedent_content_id = pcon.id 
    JOIN precedents p ON pcon.precedent_id = p.id 
    WHERE p.domain = :domain AND pc.embedding_vector IS NOT NULL
"""), {'domain': 'civil_law'})
last_embedding_time = result3a.scalar()

# 2. 임베딩 버전이 생성된 시간 (참고용)
result3b = conn.execute(text("""
    SELECT MAX(ev.created_at)
    FROM embedding_versions ev
    WHERE ev.data_type = 'precedents'
"""))
last_version_created = result3b.scalar()

# 최근 임베딩된 청크 확인 (임베딩 버전 업데이트 시간 또는 최근 처리된 청크 ID 확인)
# 임베딩이 진행 중인지 확인하기 위해 최근 처리된 청크의 ID를 확인
result4 = conn.execute(text("""
    SELECT MAX(pc.id)
    FROM precedent_chunks pc 
    JOIN precedent_contents pcon ON pc.precedent_content_id = pcon.id 
    JOIN precedents p ON pcon.precedent_id = p.id 
    WHERE p.domain = :domain AND pc.embedding_vector IS NOT NULL
"""), {'domain': 'civil_law'})
last_embedded_id = result4.scalar()

# 최근 임베딩된 청크의 ID를 기반으로 처리 시간 추정
# (실제로는 임베딩 버전 테이블의 updated_at을 확인하는 것이 더 정확)
result5 = conn.execute(text("""
    SELECT COUNT(*) 
    FROM precedent_chunks pc 
    JOIN precedent_contents pcon ON pc.precedent_content_id = pcon.id 
    JOIN precedents p ON pcon.precedent_id = p.id 
    WHERE p.domain = :domain 
      AND pc.embedding_vector IS NOT NULL 
      AND pc.embedding_version = (
          SELECT MAX(version) FROM embedding_versions WHERE data_type = 'precedents'
      )
"""), {'domain': 'civil_law'})
recent_embedded_count = result5.scalar()

# 출력 내용 수집
output_lines = []
output_lines.append("=" * 80)
output_lines.append("📊 민사법 판례 청크 임베딩 진행 상황")
output_lines.append("=" * 80)
output_lines.append(f"✅ 임베딩 완료: {embedded:,}개 / {total:,}개 ({embedded/total*100:.2f}%)")
output_lines.append(f"⏳ 남은 작업: {total-embedded:,}개")
output_lines.append("")

if last_embedded_id:
    output_lines.append(f"📌 최근 임베딩된 청크 ID: {last_embedded_id}")
    output_lines.append(f"📊 현재 버전 임베딩 수: {recent_embedded_count:,}개")
    
    # 이전 확인과 비교하여 진행 여부 확인
    # (실제로는 파일이나 DB에 이전 상태를 저장해야 하지만, 간단히 ID로 판단)
    output_lines.append("")
    output_lines.append("💡 진행 상황 판단:")
    if recent_embedded_count > embedded - 1000:  # 최근 1000개 이내면 진행 중으로 간주
        output_lines.append("✅ 임베딩이 진행 중인 것으로 보입니다.")
    else:
        output_lines.append("⚠️  임베딩이 중지되었을 수 있습니다. 프로세스를 확인하세요.")

if last_embedding_time:
    output_lines.append(f"📅 최근 임베딩 시간: {last_embedding_time}")
if last_version_created:
    output_lines.append(f"📅 버전 생성 시간: {last_version_created} (참고용)")

output_lines.append("=" * 80)

# 출력
output_text = "\n".join(output_lines)
print(output_text)
sys.stdout.flush()

# 파일로도 저장
from pathlib import Path
output_file = Path("logs/precedent_embedding_status.txt")
output_file.parent.mkdir(parents=True, exist_ok=True)
with open(output_file, 'w', encoding='utf-8') as f:
    f.write(output_text)
print(f"\n💾 결과가 파일로 저장되었습니다: {output_file}")

conn.close()

