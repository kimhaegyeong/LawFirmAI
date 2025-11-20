#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""재임베딩 진행 상황 확인 스크립트"""

import sqlite3
from pathlib import Path
from datetime import datetime
import os

# 프로젝트 루트 기준으로 데이터베이스 경로 설정
script_dir = Path(__file__).parent
project_root = script_dir.parent.parent
db_path = project_root / "data" / "lawfirm_v2.db"

if not db_path.exists():
    print(f"❌ 데이터베이스 파일을 찾을 수 없습니다: {db_path}")
    exit(1)

conn = sqlite3.connect(str(db_path))
conn.row_factory = sqlite3.Row

print("=" * 60)
print(f"재임베딩 진행 상황 확인 - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
print("=" * 60)

# 전체 재임베딩 상태
print("\n1. 전체 재임베딩 상태:")
cursor = conn.execute("""
    SELECT 
        COUNT(DISTINCT tc.id) as total_chunks,
        COUNT(DISTINCT e.chunk_id) as embedded_chunks
    FROM text_chunks tc
    LEFT JOIN embeddings e ON tc.id = e.chunk_id 
        AND e.model = 'woong0322/ko-legal-sbert-finetuned'
""")
row = cursor.fetchone()
total = row['total_chunks']
embedded = row['embedded_chunks']
percentage = (embedded / total * 100) if total > 0 else 0
print(f"   전체: {embedded:,}/{total:,} 청크 ({percentage:.1f}%)")

# 소스 타입별 재임베딩 상태
print("\n2. 소스 타입별 재임베딩 상태:")
cursor = conn.execute("""
    SELECT 
        tc.source_type,
        COUNT(DISTINCT tc.id) as total_chunks,
        COUNT(DISTINCT e.chunk_id) as embedded_chunks
    FROM text_chunks tc
    LEFT JOIN embeddings e ON tc.id = e.chunk_id 
        AND e.model = 'woong0322/ko-legal-sbert-finetuned'
    GROUP BY tc.source_type
    ORDER BY tc.source_type
""")
for row in cursor.fetchall():
    total = row['total_chunks']
    embedded = row['embedded_chunks']
    if total > 0:
        percentage = (embedded / total * 100)
        status = "✅ 완료" if percentage >= 99.9 else "⏳ 진행 중"
        print(f"   {row['source_type']}: {embedded:,}/{total:,} 청크 ({percentage:.1f}%) {status}")
    else:
        print(f"   {row['source_type']}: {embedded:,}/0 청크")

# case_paragraph 상세 정보
print("\n3. case_paragraph 상세 정보:")
cursor = conn.execute("""
    SELECT 
        COUNT(DISTINCT tc.id) as total_chunks,
        COUNT(DISTINCT e.chunk_id) as embedded_chunks
    FROM text_chunks tc
    LEFT JOIN embeddings e ON tc.id = e.chunk_id 
        AND e.model = 'woong0322/ko-legal-sbert-finetuned'
    WHERE tc.source_type = 'case_paragraph'
""")
row = cursor.fetchone()
total_chunks = row['total_chunks']
embedded_chunks = row['embedded_chunks']
remaining_chunks = total_chunks - embedded_chunks

if total_chunks > 0:
    chunk_percentage = (embedded_chunks / total_chunks * 100)
    print(f"   총 청크: {total_chunks:,}개")
    print(f"   재임베딩 완료: {embedded_chunks:,}개 ({chunk_percentage:.1f}%)")
    print(f"   남은 청크: {remaining_chunks:,}개")
    
    if remaining_chunks > 0:
        # 예상 소요 시간 계산 (대략적인 추정)
        # CPU: 약 0.1-0.2초/청크, GPU: 약 0.05-0.1초/청크
        estimated_hours_cpu = (remaining_chunks * 0.15) / 3600
        estimated_hours_gpu = (remaining_chunks * 0.075) / 3600
        print(f"\n   예상 소요 시간:")
        print(f"   - CPU 사용 시: 약 {estimated_hours_cpu:.1f}시간")
        print(f"   - GPU 사용 시: 약 {estimated_hours_gpu:.1f}시간")

conn.close()
print("\n" + "=" * 60)

