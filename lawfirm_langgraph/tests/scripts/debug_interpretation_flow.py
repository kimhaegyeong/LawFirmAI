# -*- coding: utf-8 -*-
"""
interpretation_paragraph 데이터 흐름 디버깅
"""

import sys
import os
import sqlite3
from pathlib import Path

# 프로젝트 루트 경로 추가
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..'))
sys.path.insert(0, project_root)

def check_interpretation_flow():
    """interpretation_paragraph 데이터 흐름 확인"""
    print("\n" + "=" * 80)
    print("interpretation_paragraph 데이터 흐름 디버깅")
    print("=" * 80)
    
    # 1. DB에서 데이터 확인
    db_path = "data/lawfirm_v2.db"
    if not os.path.exists(db_path):
        print(f"\n❌ DB 파일 없음: {db_path}")
        return
    
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    cursor = conn.execute(
        "SELECT COUNT(*) as count FROM text_chunks WHERE source_type = ? AND text IS NOT NULL AND LENGTH(text) > 50",
        ("interpretation_paragraph",)
    )
    count = cursor.fetchone()['count']
    print(f"\n1️⃣ DB에 interpretation_paragraph 문서 수: {count}개")
    
    if count == 0:
        print("   ❌ DB에 데이터가 없습니다!")
        return
    
    # 2. 샘플링 쿼리 테스트
    print("\n2️⃣ 샘플링 쿼리 테스트:")
    cursor = conn.execute(
        """
        SELECT id, text, source_id, source_type
        FROM text_chunks
        WHERE source_type = ? AND text IS NOT NULL AND LENGTH(text) > 50
        ORDER BY RANDOM()
        LIMIT 2
        """,
        ("interpretation_paragraph",)
    )
    rows = cursor.fetchall()
    print(f"   샘플링 결과: {len(rows)}개")
    for idx, row in enumerate(rows, 1):
        print(f"   - 샘플 {idx}: chunk_id={row['id']}, source_id={row['source_id']}, text_length={len(row['text'])}")
    
    # 3. 메타데이터 조회 테스트
    if rows:
        print("\n3️⃣ 메타데이터 조회 테스트:")
        source_id = rows[0]['source_id']
        try:
            cursor = conn.execute("""
                SELECT ip.*, i.org, i.doc_id, i.title, i.response_date
                FROM interpretation_paragraphs ip
                JOIN interpretations i ON ip.interpretation_id = i.id
                WHERE ip.id = ?
            """, (source_id,))
            meta_row = cursor.fetchone()
            if meta_row:
                print(f"   ✅ 메타데이터 조회 성공:")
                print(f"      - org: {meta_row.get('org', 'N/A')}")
                print(f"      - title: {meta_row.get('title', 'N/A')}")
                print(f"      - doc_id: {meta_row.get('doc_id', 'N/A')}")
            else:
                print(f"   ❌ 메타데이터 조회 실패: source_id={source_id}")
        except Exception as e:
            print(f"   ❌ 메타데이터 조회 오류: {e}")
    
    conn.close()
    
    print("\n" + "=" * 80)
    print("디버깅 완료")
    print("=" * 80)

if __name__ == "__main__":
    check_interpretation_flow()

