# -*- coding: utf-8 -*-
"""
임베딩 상태 확인 스크립트
"""

import sqlite3
import sys
from pathlib import Path

def check_embedding_status(db_path: str):
    """임베딩 상태 확인"""
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    
    print("=" * 60)
    print("임베딩 상태 확인")
    print("=" * 60)
    
    # 테이블 존재 확인
    cursor = conn.execute("SELECT name FROM sqlite_master WHERE type='table'")
    tables = [row[0] for row in cursor.fetchall()]
    print(f"\n총 테이블 수: {len(tables)}")
    
    if 'embeddings' in tables:
        # 모델별 임베딩 수
        cursor = conn.execute("""
            SELECT model, COUNT(*) as count 
            FROM embeddings 
            GROUP BY model 
            ORDER BY count DESC
        """)
        print("\n현재 사용 중인 임베딩 모델:")
        for row in cursor.fetchall():
            print(f"  {row['model']}: {row['count']:,}개")
    else:
        print("\n⚠️ embeddings 테이블이 없습니다.")
    
    if 'text_chunks' in tables:
        # 총 청크 수
        cursor = conn.execute("SELECT COUNT(*) FROM text_chunks")
        total_chunks = cursor.fetchone()[0]
        print(f"\n총 청크 수: {total_chunks:,}개")
        
        # 소스 타입별 청크 수
        cursor = conn.execute("""
            SELECT source_type, COUNT(*) as count 
            FROM text_chunks 
            GROUP BY source_type 
            ORDER BY count DESC
        """)
        print("\n소스 타입별 청크 수:")
        for row in cursor.fetchall():
            print(f"  {row['source_type']}: {row['count']:,}개")
    else:
        print("\n⚠️ text_chunks 테이블이 없습니다.")
    
    if 'embedding_versions' in tables:
        # 임베딩 버전 정보
        cursor = conn.execute("""
            SELECT version_name, chunking_strategy, model_name, is_active, 
                   (SELECT COUNT(*) FROM text_chunks WHERE embedding_version_id = ev.id) as chunk_count
            FROM embedding_versions ev
            ORDER BY ev.id DESC
            LIMIT 10
        """)
        print("\n임베딩 버전 정보 (최근 10개):")
        for row in cursor.fetchall():
            active = "✅ 활성" if row['is_active'] else "❌ 비활성"
            print(f"  {row['version_name']} ({row['chunking_strategy']}) - {row['model_name']} - {active} - {row['chunk_count']:,}개")
    
    conn.close()

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="임베딩 상태 확인")
    parser.add_argument("--db", default="data/lawfirm_v2.db", help="데이터베이스 경로")
    args = parser.parse_args()
    
    check_embedding_status(args.db)

