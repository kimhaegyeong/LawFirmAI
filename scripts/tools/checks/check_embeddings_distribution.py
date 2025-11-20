#!/usr/bin/env python3
"""
embeddings 테이블의 source_type 분포 확인 스크립트
"""

import sys
import sqlite3
from pathlib import Path

project_root = Path(__file__).parent.parent.parent.parent
lawfirm_langgraph_path = project_root / "lawfirm_langgraph"
sys.path.insert(0, str(lawfirm_langgraph_path))

from core.utils.config import Config

print("="*80)
print("embeddings 테이블 source_type 분포 확인")
print("="*80)

config = Config()
db_path = config.database_path

# 여러 경로 시도
possible_paths = [
    Path(db_path).resolve() if db_path else None,
    project_root / "data" / "lawfirm_v2.db",
    project_root.parent / "data" / "lawfirm_v2.db",
    Path(r"D:\project\LawFirmAI\LawFirmAI\data\lawfirm_v2.db"),
]

actual_db_path = None
for path in possible_paths:
    if path and path.exists():
        actual_db_path = path
        break

if not actual_db_path:
    print(f"❌ 데이터베이스 파일을 찾을 수 없습니다!")
    sys.exit(1)

print(f"\n[데이터베이스]")
print(f"  {actual_db_path}")

try:
    conn = sqlite3.connect(str(actual_db_path))
    conn.row_factory = sqlite3.Row
    cursor = conn.cursor()
    
    # embeddings 테이블의 source_type 분포 확인
    print(f"\n[embeddings 테이블 - source_type 분포]")
    try:
        cursor.execute("""
            SELECT 
                tc.source_type,
                COUNT(*) as count,
                COUNT(DISTINCT e.model) as model_count
            FROM embeddings e
            JOIN text_chunks tc ON e.chunk_id = tc.id
            GROUP BY tc.source_type
            ORDER BY count DESC
        """)
        
        type_distribution = {}
        for row in cursor.fetchall():
            source_type = row['source_type']
            count = row['count']
            model_count = row['model_count']
            type_distribution[source_type] = count
            print(f"  {source_type}: {count:,}개 (모델: {model_count}개)")
        
        # 전체 embeddings 개수
        cursor.execute("SELECT COUNT(*) as count FROM embeddings")
        total_embeddings = cursor.fetchone()['count']
        print(f"\n  총 embeddings: {total_embeddings:,}개")
        
        # 모델별 분포
        print(f"\n[모델별 embeddings 분포]")
        cursor.execute("""
            SELECT 
                model,
                COUNT(*) as count
            FROM embeddings
            GROUP BY model
            ORDER BY count DESC
        """)
        
        for row in cursor.fetchall():
            model = row['model']
            count = row['count']
            print(f"  {model}: {count:,}개")
        
        # text_chunks와 embeddings 매칭 확인
        print(f"\n[text_chunks vs embeddings 매칭 확인]")
        cursor.execute("""
            SELECT 
                tc.source_type,
                COUNT(DISTINCT tc.id) as chunks_count,
                COUNT(DISTINCT e.chunk_id) as embeddings_count,
                COUNT(DISTINCT tc.id) - COUNT(DISTINCT e.chunk_id) as missing_count
            FROM text_chunks tc
            LEFT JOIN embeddings e ON tc.id = e.chunk_id
            GROUP BY tc.source_type
            ORDER BY chunks_count DESC
        """)
        
        for row in cursor.fetchall():
            source_type = row['source_type']
            chunks_count = row['chunks_count']
            embeddings_count = row['embeddings_count'] or 0
            missing_count = row['missing_count'] or 0
            coverage = (embeddings_count / chunks_count * 100) if chunks_count > 0 else 0
            print(f"  {source_type}:")
            print(f"    text_chunks: {chunks_count:,}개")
            print(f"    embeddings: {embeddings_count:,}개")
            print(f"    누락: {missing_count:,}개")
            print(f"    커버리지: {coverage:.1f}%")
        
        # FAISS 인덱스에 포함될 벡터 확인
        print(f"\n[FAISS 인덱스 포함 가능성 확인]")
        print(f"  embeddings 테이블에 모든 타입의 벡터가 있는지 확인")
        
        # 각 타입별로 샘플 확인
        print(f"\n[각 타입별 embeddings 샘플]")
        for source_type in ["statute_article", "case_paragraph", "decision_paragraph", "interpretation_paragraph"]:
            cursor.execute("""
                SELECT 
                    e.chunk_id,
                    tc.source_type,
                    LENGTH(e.vector) as vector_size,
                    e.dim
                FROM embeddings e
                JOIN text_chunks tc ON e.chunk_id = tc.id
                WHERE tc.source_type = ?
                LIMIT 3
            """, (source_type,))
            
            rows = cursor.fetchall()
            if rows:
                print(f"  {source_type}: {len(rows)}개 샘플 발견")
                for row in rows:
                    print(f"    - chunk_id={row['chunk_id']}, vector_size={row['vector_size']}, dim={row['dim']}")
            else:
                print(f"  {source_type}: 샘플 없음 (임베딩이 생성되지 않았을 수 있음)")
        
        conn.close()
        
        # 결과 요약
        print(f"\n[결과 요약]")
        total_chunks = sum(type_distribution.values())
        if total_chunks > 0:
            print(f"  총 embeddings: {total_chunks:,}개")
            for source_type, count in sorted(type_distribution.items(), key=lambda x: x[1], reverse=True):
                percentage = count / total_chunks * 100
                print(f"  {source_type}: {count:,}개 ({percentage:.1f}%)")
        else:
            print(f"  ❌ embeddings가 없습니다!")
    
    except sqlite3.OperationalError as e:
        print(f"  ❌ 쿼리 실행 오류: {e}")
        print(f"  embeddings 테이블이 없거나 구조가 다를 수 있습니다.")
        
        # 테이블 구조 확인
        print(f"\n[embeddings 테이블 구조 확인]")
        try:
            cursor.execute("PRAGMA table_info(embeddings)")
            columns = cursor.fetchall()
            if columns:
                print(f"  컬럼:")
                for col in columns:
                    print(f"    - {col['name']}: {col['type']}")
            else:
                print(f"  embeddings 테이블이 존재하지 않습니다.")
        except Exception as e2:
            print(f"  테이블 구조 확인 실패: {e2}")
    
except Exception as e:
    print(f"\n❌ 데이터베이스 확인 중 오류 발생: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

