# -*- coding: utf-8 -*-
"""
chunk_metadata 확인 스크립트
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
print("chunk_id별 embedding_version_id 확인")
print("="*80)
chunk_ids = [2446, 2462, 1573]
for chunk_id in chunk_ids:
    cursor = conn.execute(
        "SELECT id, embedding_version_id, source_type, source_id FROM text_chunks WHERE id = ?",
        (chunk_id,)
    )
    row = cursor.fetchone()
    if row:
        print(f"  chunk_id={row['id']}, embedding_version_id={row['embedding_version_id']}, "
              f"source_type={row['source_type']}, source_id={row['source_id']}")

conn.close()

