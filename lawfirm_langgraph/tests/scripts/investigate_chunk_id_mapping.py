# -*- coding: utf-8 -*-
"""
chunk_id 매핑 문제 조사 스크립트
"""

import sys
import os
from pathlib import Path
import sqlite3

# 프로젝트 경로 설정
script_dir = Path(__file__).parent
tests_dir = script_dir.parent
lawfirm_langgraph_dir = tests_dir.parent
project_root = lawfirm_langgraph_dir.parent

if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

db_path = project_root / "data" / "lawfirm_v2.db"

if not db_path.exists():
    print(f"데이터베이스 파일을 찾을 수 없습니다: {db_path}")
    sys.exit(1)

conn = sqlite3.connect(str(db_path))
conn.row_factory = sqlite3.Row

print("="*80)
print("1. embeddings 테이블에서 chunk_id 조회 (ORDER BY chunk_id)")
print("="*80)
cursor = conn.execute("""
    SELECT chunk_id FROM embeddings 
    WHERE model = 'snunlp/KR-SBERT-V40K-klueNLI-augSTS' 
    ORDER BY chunk_id 
    LIMIT 20
""")
rows = cursor.fetchall()
print(f"  총 {len(rows)}개 (샘플 20개):")
for i, row in enumerate(rows[:20], 1):
    print(f"    인덱스 {i-1}: chunk_id={row['chunk_id']}")

print("\n" + "="*80)
print("2. embeddings 테이블에서 chunk_id 조회 (ORDER BY id - 삽입 순서)")
print("="*80)
cursor = conn.execute("""
    SELECT chunk_id FROM embeddings 
    WHERE model = 'snunlp/KR-SBERT-V40K-klueNLI-augSTS' 
    ORDER BY id 
    LIMIT 20
""")
rows = cursor.fetchall()
print(f"  총 {len(rows)}개 (샘플 20개):")
for i, row in enumerate(rows[:20], 1):
    print(f"    인덱스 {i-1}: chunk_id={row['chunk_id']}")

print("\n" + "="*80)
print("3. text_chunks 테이블에서 실제 존재하는 chunk_id 확인")
print("="*80)
test_chunk_ids = [1775, 734, 1776, 196, 1740, 1428, 539, 1837, 324, 537]
for chunk_id in test_chunk_ids:
    cursor = conn.execute("SELECT id, source_type, source_id, embedding_version_id FROM text_chunks WHERE id = ?", (chunk_id,))
    row = cursor.fetchone()
    if row:
        print(f"  ✅ chunk_id={chunk_id}: source_type={row['source_type']}, source_id={row['source_id']}, embedding_version_id={row['embedding_version_id']}")
    else:
        print(f"  ❌ chunk_id={chunk_id}: 존재하지 않음")

print("\n" + "="*80)
print("4. embeddings 테이블에서 해당 chunk_id 확인")
print("="*80)
for chunk_id in test_chunk_ids:
    cursor = conn.execute("SELECT id, chunk_id, model FROM embeddings WHERE chunk_id = ?", (chunk_id,))
    row = cursor.fetchone()
    if row:
        print(f"  ✅ chunk_id={chunk_id}: embeddings.id={row['id']}, model={row['model']}")
    else:
        print(f"  ❌ chunk_id={chunk_id}: embeddings 테이블에 없음")

print("\n" + "="*80)
print("5. FAISS 인덱스에 포함된 벡터 수 vs embeddings 테이블의 벡터 수")
print("="*80)
cursor = conn.execute("""
    SELECT COUNT(*) as count FROM embeddings 
    WHERE model = 'snunlp/KR-SBERT-V40K-klueNLI-augSTS'
""")
row = cursor.fetchone()
embeddings_count = row['count'] if row else 0
print(f"  embeddings 테이블 벡터 수: {embeddings_count}")

cursor = conn.execute("SELECT COUNT(*) as count FROM text_chunks")
row = cursor.fetchone()
chunks_count = row['count'] if row else 0
print(f"  text_chunks 테이블 청크 수: {chunks_count}")

print("\n" + "="*80)
print("6. embeddings와 text_chunks 간의 불일치 확인")
print("="*80)
cursor = conn.execute("""
    SELECT e.chunk_id 
    FROM embeddings e
    LEFT JOIN text_chunks tc ON e.chunk_id = tc.id
    WHERE e.model = 'snunlp/KR-SBERT-V40K-klueNLI-augSTS' AND tc.id IS NULL
    LIMIT 10
""")
rows = cursor.fetchall()
if rows:
    print(f"  ⚠️  embeddings에 있지만 text_chunks에 없는 chunk_id: {len(rows)}개")
    for row in rows[:10]:
        print(f"    chunk_id={row['chunk_id']}")
else:
    print("  ✅ 모든 embeddings의 chunk_id가 text_chunks에 존재")

conn.close()

