#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""활성 embedding version 확인 스크립트"""

import sqlite3
import sys
from pathlib import Path

db_path = Path(__file__).parent.parent.parent.parent / "data" / "lawfirm_v2.db"

if not db_path.exists():
    print(f"❌ 데이터베이스 파일을 찾을 수 없습니다: {db_path}")
    sys.exit(1)

conn = sqlite3.connect(str(db_path))
conn.row_factory = sqlite3.Row

print("="*80)
print("최근 embedding versions:")
print("="*80)

cursor = conn.execute("""
    SELECT id, version_name, chunking_strategy, model_name, is_active, created_at
    FROM embedding_versions
    ORDER BY created_at DESC
    LIMIT 5
""")
rows = cursor.fetchall()

for row in rows:
    print(f"  ID: {row['id']}, Name: {row['version_name']}, "
          f"Strategy: {row['chunking_strategy']}, "
          f"Active: {row['is_active']}, Created: {row['created_at']}")

print("\n" + "="*80)
print("활성 embedding version:")
print("="*80)

cursor = conn.execute("""
    SELECT id, version_name, chunking_strategy, model_name, created_at
    FROM embedding_versions
    WHERE is_active = 1
    ORDER BY created_at DESC
    LIMIT 1
""")
row = cursor.fetchone()

if row:
    print(f"  ✅ 활성 버전 ID: {row['id']}")
    print(f"     버전 이름: {row['version_name']}")
    print(f"     청킹 전략: {row['chunking_strategy']}")
    print(f"     모델: {row['model_name']}")
    print(f"     생성일: {row['created_at']}")
    
    # 해당 버전의 청크 수 확인
    cursor2 = conn.execute("""
        SELECT COUNT(*) as count
        FROM text_chunks
        WHERE embedding_version_id = ?
    """, (row['id'],))
    chunk_count = cursor2.fetchone()['count']
    print(f"     청크 수: {chunk_count:,}개")
    
    # 해당 버전의 임베딩 수 확인
    cursor3 = conn.execute("""
        SELECT COUNT(*) as count
        FROM embeddings e
        JOIN text_chunks tc ON e.chunk_id = tc.id
        WHERE tc.embedding_version_id = ?
    """, (row['id'],))
    embedding_count = cursor3.fetchone()['count']
    print(f"     임베딩 수: {embedding_count:,}개")
    
    print(f"\n  💡 FAISS 인덱스 빌드 명령:")
    print(f"     python scripts/tools/wait_and_build_faiss_index.py --db data/lawfirm_v2.db --version-id {row['id']} --skip-wait")
else:
    print("  ❌ 활성 버전이 없습니다.")

conn.close()

