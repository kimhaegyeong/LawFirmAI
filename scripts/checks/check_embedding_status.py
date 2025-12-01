#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""임베딩 상태 확인 스크립트"""

import sqlite3
from pathlib import Path

# 프로젝트 루트 기준으로 데이터베이스 경로 설정
script_dir = Path(__file__).parent
project_root = script_dir.parent.parent
db_path = project_root / "data" / "lawfirm_v2.db"

conn = sqlite3.connect(str(db_path))
conn.row_factory = sqlite3.Row

print("=" * 60)
print("case_paragraph 임베딩 상태 확인")
print("=" * 60)

# case_paragraph 상태
cursor = conn.execute("""
    SELECT 
        COUNT(DISTINCT tc.id) as total,
        COUNT(DISTINCT CASE WHEN e.model = 'woong0322/ko-legal-sbert-finetuned' THEN e.chunk_id END) as ko_legal,
        COUNT(DISTINCT CASE WHEN e.model != 'woong0322/ko-legal-sbert-finetuned' AND e.model IS NOT NULL THEN e.chunk_id END) as other_model
    FROM text_chunks tc
    LEFT JOIN embeddings e ON tc.id = e.chunk_id
    WHERE tc.source_type = 'case_paragraph'
""")
row = cursor.fetchone()
total = row['total']
ko_legal = row['ko_legal']
other_model = row['other_model']
no_embedding = total - ko_legal - other_model

print(f"\n1. case_paragraph 청크 상태:")
print(f"   총 청크: {total:,}개")
print(f"   Ko-Legal-SBERT: {ko_legal:,}개")
print(f"   다른 모델: {other_model:,}개")
print(f"   임베딩 없음: {no_embedding:,}개")

# 버전 정보
print(f"\n2. Ko-Legal-SBERT 버전 정보:")
cursor = conn.execute("""
    SELECT id, version_name, model_name, chunking_strategy
    FROM embedding_versions
    WHERE model_name LIKE '%ko-legal%' OR model_name LIKE '%Legal%'
    ORDER BY created_at DESC
    LIMIT 3
""")
for row in cursor.fetchall():
    print(f"   ID {row['id']}: {row['version_name']}")
    print(f"      모델: {row['model_name']}")
    print(f"      청킹: {row['chunking_strategy']}")

# 활성 버전 확인
print(f"\n3. 활성 버전 확인 (standard 청킹):")
cursor = conn.execute("""
    SELECT id, version_name, model_name, chunking_strategy
    FROM embedding_versions
    WHERE chunking_strategy = 'standard'
    ORDER BY created_at DESC
    LIMIT 3
""")
for row in cursor.fetchall():
    print(f"   ID {row['id']}: {row['version_name']} ({row['model_name']})")

conn.close()

