#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Version 5를 활성화하는 스크립트"""

import sys
import sqlite3
from pathlib import Path

project_root = Path(__file__).parent.parent.parent.parent
db_path = project_root / "data" / "lawfirm_v2.db"

if not db_path.exists():
    print(f"❌ 데이터베이스 파일을 찾을 수 없습니다: {db_path}")
    sys.exit(1)

conn = sqlite3.connect(str(db_path))
conn.row_factory = sqlite3.Row

print("="*80)
print("Version 5 활성화")
print("="*80)

# 현재 활성 버전 확인
cursor = conn.execute("""
    SELECT id, version_name, chunking_strategy, is_active
    FROM embedding_versions
    WHERE is_active = 1
""")
active_versions = cursor.fetchall()

if active_versions:
    print("\n현재 활성 버전:")
    for row in active_versions:
        print(f"  ID: {row['id']}, Name: {row['version_name']}, Strategy: {row['chunking_strategy']}")

# 모든 버전 비활성화
print("\n모든 버전 비활성화 중...")
conn.execute("UPDATE embedding_versions SET is_active = 0")
conn.commit()

# Version 5 활성화
print("Version 5 활성화 중...")
cursor = conn.execute("""
    UPDATE embedding_versions
    SET is_active = 1
    WHERE id = 5
""")
conn.commit()

if cursor.rowcount > 0:
    print("✅ Version 5가 활성화되었습니다")
    
    # 확인
    cursor = conn.execute("""
        SELECT id, version_name, chunking_strategy, is_active
        FROM embedding_versions
        WHERE id = 5
    """)
    row = cursor.fetchone()
    if row:
        print(f"\n활성화된 버전:")
        print(f"  ID: {row['id']}")
        print(f"  Name: {row['version_name']}")
        print(f"  Strategy: {row['chunking_strategy']}")
        print(f"  Active: {row['is_active']}")
else:
    print("❌ Version 5를 찾을 수 없습니다")

conn.close()

