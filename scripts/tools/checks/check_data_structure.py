# -*- coding: utf-8 -*-
"""
데이터 구조 확인 스크립트
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
print("1. text_chunks 테이블 샘플 (case_paragraph)")
print("="*80)
cursor = conn.execute("""
    SELECT id, source_type, source_id, embedding_version_id
    FROM text_chunks
    WHERE source_type = 'case_paragraph'
    LIMIT 5
""")
rows = cursor.fetchall()
for row in rows:
    print(f"  chunk_id={row['id']}, source_type={row['source_type']}, "
          f"source_id={row['source_id']} (type: {type(row['source_id']).__name__}), "
          f"embedding_version_id={row['embedding_version_id']}")

print("\n" + "="*80)
print("2. cases 테이블 샘플")
print("="*80)
cursor = conn.execute("""
    SELECT id, doc_id, casenames, court
    FROM cases
    LIMIT 5
""")
rows = cursor.fetchall()
for row in rows:
    print(f"  case_id={row['id']}, doc_id={row['doc_id']}, "
          f"casenames={row['casenames']}, court={row['court']}")

print("\n" + "="*80)
print("3. text_chunks와 cases JOIN 샘플")
print("="*80)
cursor = conn.execute("""
    SELECT tc.id as chunk_id, tc.source_id, c.id as case_id, c.doc_id, c.casenames, c.court
    FROM text_chunks tc
    JOIN cases c ON tc.source_id = c.id
    WHERE tc.source_type = 'case_paragraph'
    LIMIT 5
""")
rows = cursor.fetchall()
for row in rows:
    print(f"  chunk_id={row['chunk_id']}, text_chunks.source_id={row['source_id']}, "
          f"cases.id={row['case_id']}, doc_id={row['doc_id']}, "
          f"casenames={row['casenames']}, court={row['court']}")

print("\n" + "="*80)
print("4. embedding_version_id가 NULL인 chunk 개수")
print("="*80)
cursor = conn.execute("""
    SELECT COUNT(*) as count
    FROM text_chunks
    WHERE embedding_version_id IS NULL
""")
row = cursor.fetchone()
print(f"  NULL인 chunk 개수: {row['count']}")

cursor = conn.execute("""
    SELECT COUNT(*) as count
    FROM text_chunks
    WHERE embedding_version_id IS NOT NULL
""")
row = cursor.fetchone()
print(f"  NULL이 아닌 chunk 개수: {row['count']}")

print("\n" + "="*80)
print("5. 활성 embedding_version 확인")
print("="*80)
cursor = conn.execute("""
    SELECT id, model_name, is_active, created_at
    FROM embedding_versions
    WHERE is_active = 1
    ORDER BY created_at DESC
    LIMIT 1
""")
row = cursor.fetchone()
if row:
    print(f"  활성 버전 ID: {row['id']}, 모델: {row['model_name']}")
else:
    print("  활성 버전 없음")

conn.close()

