# -*- coding: utf-8 -*-
"""
외부 인덱스와 text_chunks 매핑 조사 스크립트
"""

import sys
import os
from pathlib import Path
import sqlite3
import json

# 프로젝트 경로 설정
script_dir = Path(__file__).parent
tests_dir = script_dir.parent
lawfirm_langgraph_dir = tests_dir.parent
project_root = lawfirm_langgraph_dir.parent

if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

db_path = project_root / "data" / "lawfirm_v2.db"
metadata_path = project_root / "data" / "embeddings" / "ml_enhanced_ko_sroberta_precedents" / "ml_enhanced_faiss_index.json"

if not db_path.exists():
    print(f"데이터베이스 파일을 찾을 수 없습니다: {db_path}")
    sys.exit(1)

if not metadata_path.exists():
    print(f"메타데이터 파일을 찾을 수 없습니다: {metadata_path}")
    sys.exit(1)

conn = sqlite3.connect(str(db_path))
conn.row_factory = sqlite3.Row

# 외부 인덱스 메타데이터 로드
with open(metadata_path, 'r', encoding='utf-8') as f:
    metadata_content = json.load(f)

metadata_list = []
if 'document_metadata' in metadata_content and 'document_texts' in metadata_content:
    metadata_list_data = metadata_content['document_metadata']
    texts_list = metadata_content['document_texts']
    for meta, text in zip(metadata_list_data, texts_list):
        combined = meta.copy()
        combined['content'] = text
        combined['text'] = text
        metadata_list.append(combined)

print("="*80)
print("외부 인덱스와 text_chunks 매핑 조사")
print("="*80)

# 샘플 몇 개 확인
sample_indices = [0, 100, 500, 1000, 2462, 2446, 1573]
for idx in sample_indices:
    if idx >= len(metadata_list):
        continue
    
    meta = metadata_list[idx]
    case_id = meta.get('case_id', '')
    
    print(f"\n인덱스 {idx}:")
    print(f"  case_id: {case_id}")
    
    # case_id로 cases 테이블에서 찾기
    if case_id:
        # case_id가 "case_2023다240299" 형식인 경우 doc_id로 조회
        cursor = conn.execute("""
            SELECT id, doc_id, casenames, court FROM cases WHERE doc_id = ?
        """, (case_id,))
        row = cursor.fetchone()
        
        if row:
            case_db_id = row['id']
            print(f"  ✅ cases 테이블에서 찾음: case.id={case_db_id}, doc_id={row['doc_id']}")
            
            # text_chunks에서 해당 case의 chunk 찾기
            cursor2 = conn.execute("""
                SELECT id, chunk_index, text FROM text_chunks 
                WHERE source_type='case_paragraph' AND source_id=?
                ORDER BY chunk_index
                LIMIT 5
            """, (case_db_id,))
            chunks = cursor2.fetchall()
            
            if chunks:
                print(f"  ✅ text_chunks에서 {len(chunks)}개 청크 찾음 (샘플):")
                for chunk in chunks[:3]:
                    print(f"    chunk_id={chunk['id']}, chunk_index={chunk['chunk_index']}, text_length={len(chunk['text']) if chunk['text'] else 0}")
            else:
                print(f"  ❌ text_chunks에 해당 case의 청크가 없음")
        else:
            print(f"  ❌ cases 테이블에서 찾을 수 없음")

print("\n" + "="*80)
print("외부 인덱스 인덱스 번호와 실제 chunk_id 매핑 확인")
print("="*80)

# 외부 인덱스의 인덱스 번호가 실제 chunk_id와 어떻게 매핑되는지 확인
# 외부 인덱스는 전체 문서를 하나의 벡터로 임베딩했을 가능성이 높음
# 따라서 외부 인덱스의 인덱스 번호는 실제 chunk_id가 아니라 문서 인덱스입니다.

print("외부 인덱스는 전체 문서를 하나의 벡터로 임베딩한 것으로 보입니다.")
print("따라서 외부 인덱스의 인덱스 번호는 실제 chunk_id가 아닙니다.")
print("\n해결 방안:")
print("1. 외부 인덱스 사용 시 case_id를 기반으로 text_chunks에서 청크를 찾아야 합니다.")
print("2. 또는 외부 인덱스의 메타데이터에 실제 chunk_id를 추가해야 합니다.")

conn.close()

